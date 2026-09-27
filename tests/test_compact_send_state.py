"""/compact send state: one provider method, chosen statically before any provider code runs.

compact_conversation decides with inspect.getattr_static on the provider instance whether to
call chat_async or chat (a statically declared, non-None chat_async selects the async method;
a missing one, None, or one reachable only through __getattr__ selects chat), then calls exactly
that method once. No exception, malformed reply, timeout, SDK or parser error is ever followed by
the other method, and a failed provider-backed compaction leaves the conversation unchanged: the
command reports "Compact failed: ..." and never substitutes a local summary.

No network: synthetic providers never touch a transport, and the OfflineCase classes (real SDK
clients, disk-loaded plugin) scrub credential, base-URL and proxy environment variables, refuse
sockets and DNS, record sleeps, and count every SDK HTTP send at
``httpx.HTTPTransport.handle_request`` (and ``AsyncHTTPTransport.handle_async_request``). The one
true-async test runs a real event loop; only asyncio's self-pipe ``socket.socketpair`` may then
connect, and only to loopback.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import re
import socket
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anthropic
import httpx
import openai

from src.agent import Conversation
from src.command_system import builtins as command_builtins
from src.command_system.engine import create_command_context
from src.compact_service import service
from src.compact_service.service import (
    CompactMethodUndeterminedError,
    CompactSummaryError,
    NotEnoughMessagesToCompactError,
    _declares_chat_async,
    compact_conversation,
)
from src.cost_tracker import CostTracker
from src.history import HistoryLog
from src.plugins.extensions import load_active_plugin_extensions, register_plugin_provider_extensions
from src.plugins.runtime import reconcile_python_plugins
from src.providers import clear_plugin_providers, get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import BaseProvider, ChatResponse, FinishStatusError, IncompleteResponseError
from src.providers.openai_provider import OpenAIProvider

SRC = Path(__file__).resolve().parents[1] / "src"
SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
OPENAI_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
UNDETERMINED = "Cannot safely determine the provider's compaction method."
NO_SUMMARY = "Provider returned no usable compact summary."


def reply(text: Any = "ASYNC SUMMARY", usage: Any = None) -> ChatResponse:
    return ChatResponse(content=text, model="plugin-model",
                        usage={"input_tokens": 11, "output_tokens": 5} if usage is None else usage,
                        finish_reason="stop")


class SubNIE(NotImplementedError):
    """A plugin's own NotImplementedError subclass."""


class PluginError(Exception):
    """A plugin's own exception type."""


class BrokenStr(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("this exception cannot be printed")


class BrokenStrNIE(NotImplementedError):
    def __str__(self) -> str:
        raise ValueError("this exception cannot be printed")


class ExplodingStr(str):
    def strip(self, *args: Any) -> str:  # type: ignore[override]
        raise ValueError("strip exploded")


class OddStr(RuntimeError):
    """__str__ returns a str subclass whose methods raise."""

    def __str__(self) -> str:
        return ExplodingStr("odd text")


# --- synthetic plugin providers (their own ledger of entered methods) ------------------------------

class SyncPlugin(BaseProvider):
    """A direct BaseProvider plugin without chat_async: compaction must choose chat()."""

    def __init__(self, api_key: str = "k", base_url: str | None = None, model: str | None = None,
                 *, result: Any = None, error: BaseException | None = None):
        super().__init__(api_key, base_url, model or "plugin-model")
        self.entered: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.result = reply("SYNC SUMMARY", {"input_tokens": 7, "output_tokens": 3}) if result is None else result
        self.error = error

    def chat(self, messages, tools=None, **kwargs):
        self.entered.append("chat")
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result

    def chat_stream(self, messages, tools=None, **kwargs):
        self.entered.append("chat_stream")
        yield "never"

    def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
        self.entered.append("chat_stream_response")
        raise AssertionError("compaction must not stream")

    def get_available_models(self):
        return ["plugin-model"]


class AsyncPlugin(SyncPlugin):
    """Declares async def chat_async: raises before or after its 'send', or returns ``result``."""

    def __init__(self, *, before_send: BaseException | None = None, after_send: BaseException | None = None,
                 result: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.before_send = before_send
        self.after_send = after_send
        self.async_result = reply() if result is None else result
        self.sends = 0

    async def chat_async(self, messages, tools=None, **kwargs):
        self.entered.append("chat_async")
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        if self.before_send is not None:
            raise self.before_send
        self.sends += 1  # the request went out
        if self.after_send is not None:
            raise self.after_send
        return self.async_result


def conversation() -> Conversation:
    conv = Conversation()
    conv.add_user_message("first question")
    conv.add_assistant_message("first answer")
    return conv


def snapshot(conv: Conversation) -> list[tuple[int, str, Any]]:
    return [(id(m), m.role, m.content) for m in conv.messages]


def message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(getattr(block, "text", "") for block in content)


def _run_without_event_loop(coroutine: Any) -> Any:
    """Drive a coroutine that never suspends (asyncio's Windows loop needs a local socket
    pair, which the offline guard refuses)."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("the coroutine suspended; it was expected to finish without awaiting")


def compact(provider: Any, conv: Conversation | None = None) -> tuple[Any, BaseException | None, Conversation, list]:
    conv = conversation() if conv is None else conv
    before = snapshot(conv)
    try:
        return _run_without_event_loop(compact_conversation(conv, provider, "plugin-model")), None, conv, before
    except BaseException as exc:  # noqa: BLE001 - recorded and asserted by the caller
        return None, exc, conv, before


# --- shared offline harness ------------------------------------------------------------------------

_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each SDK client costs an SSL context)."""
    if name not in _SHARED:
        _SHARED[name] = get_provider_class(name)(api_key="test-dummy-key", base_url=None, model=None)
    return _SHARED[name]


def oa_reply(text: Any = "SDK SUMMARY", status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": {"message": "server error", "type": "server_error"}})
    return httpx.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


def an_reply(text: str = "SDK SUMMARY") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2}})


class OfflineCase(unittest.TestCase):
    """No network, no real keys, no real sleeps; every SDK HTTP send is counted."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        clean_env = {k: v for k, v in os.environ.items() if not SCRUBBED_ENV.search(k)}
        clean_env["CLAWD_ACTIVITY_LEDGER"] = str(self.tmp / "activity.jsonl")
        clean_env["CLAWD_CHANGE_LEDGER"] = str(self.tmp / "changes.jsonl")
        self._start(patch.dict(os.environ, clean_env, clear=True))
        self.refused: list[str] = []
        for target in ("socket.create_connection", "socket.getaddrinfo", "socket.socket.connect"):
            self._start(patch(target, self._refuse(target)))
        self.sleeps: list[float] = []
        self._start(patch("time.sleep", side_effect=self.sleeps.append))
        self._start(patch("src.token_estimation._load_tiktoken", return_value=None))
        self.sent: list[httpx.Request] = []
        self.script: list[httpx.Response] = []

        def next_response(request: httpx.Request) -> httpx.Response:
            self.sent.append(request)
            if not self.script:
                raise AssertionError("unexpected extra provider request")
            response = self.script.pop(0)
            response.request = request
            return response

        def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            request.read()
            return next_response(request)

        async def handle_async_request(transport: httpx.AsyncHTTPTransport, request: httpx.Request):
            await request.aread()
            return next_response(request)

        self._start(patch.object(httpx.HTTPTransport, "handle_request", handle_request))
        self._start(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request))

    def _start(self, patcher: Any) -> None:
        patcher.start()
        self.addCleanup(patcher.stop)

    def _refuse(self, name: str) -> Callable[..., Any]:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.refused.append(name)
            raise AssertionError(f"{name} used in an offline test")
        return refuse

    def tearDown(self) -> None:
        self.assertEqual(self.refused, [], "a test reached the network")
        self.assertEqual(self.sleeps, [], "a test slept (an SDK retry or backoff)")

    def play(self, *responses: httpx.Response) -> None:
        self.sent.clear()
        self.script = list(responses)

    @contextlib.contextmanager
    def loopback_socketpair_only(self):
        """asyncio.run needs a self-pipe; on Windows socket.socketpair connects to loopback."""
        import _socket

        real_connect = _socket.socket.connect
        real_socketpair = socket.socketpair
        inside = {"socketpair": False}

        def guarded_connect(sock, address, *rest):
            host = address[0] if isinstance(address, tuple) else address
            if inside["socketpair"] and host in ("127.0.0.1", "::1"):
                return real_connect(sock, address)
            self.refused.append(f"socket.socket.connect{address!r}")
            raise AssertionError(f"connect outside the loopback socketpair allowance: {address!r}")

        def scoped_socketpair(*args: Any, **kwargs: Any):
            inside["socketpair"] = True
            try:
                return real_socketpair(*args, **kwargs)
            finally:
                inside["socketpair"] = False

        with patch("socket.socket.connect", guarded_connect), patch("socket.socketpair", scoped_socketpair):
            yield


# --- the static decision ------------------------------------------------------------------------------

TRAPS: list[str] = []


class AsyncDef(SyncPlugin):
    async def chat_async(self, **kwargs):
        TRAPS.append("AsyncDef.chat_async")
        return reply()


class InheritedAsync(AsyncDef):
    pass


class InstanceBound(SyncPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.chat_async = self._implementation

    async def _implementation(self, **kwargs):
        TRAPS.append("InstanceBound._implementation")
        return reply()


class PropertyAsync(SyncPlugin):
    @property
    def chat_async(self):  # type: ignore[override]
        TRAPS.append("property getter")
        raise AssertionError("a property was executed")


class Descriptor:
    def __get__(self, instance: Any, owner: Any) -> Any:
        TRAPS.append("descriptor __get__")
        raise AssertionError("a descriptor was executed")


class DescriptorAsync(SyncPlugin):
    chat_async = Descriptor()


class StaticAsync(SyncPlugin):
    @staticmethod
    async def chat_async(**kwargs):
        TRAPS.append("staticmethod")
        return reply()


class ClassAsync(SyncPlugin):
    @classmethod
    async def chat_async(cls, **kwargs):
        TRAPS.append("classmethod")
        return reply()


class SyncReturnsCoroutine(SyncPlugin):
    def chat_async(self, **kwargs):
        TRAPS.append("sync def returning a coroutine")
        return self._coroutine()

    async def _coroutine(self):
        return reply()


class CallableObject:
    def __call__(self, **kwargs: Any) -> Any:
        TRAPS.append("callable instance")
        raise AssertionError("a callable instance was executed")


class CallableAsync(SyncPlugin):
    chat_async = CallableObject()


class NonCallableAsync(SyncPlugin):
    chat_async = "not a method"  # type: ignore[assignment]


class TrapValue:
    """A declared value whose truthiness, length and comparisons must never be consulted."""

    def __bool__(self) -> bool:
        TRAPS.append("__bool__")
        return False

    def __len__(self) -> int:
        TRAPS.append("__len__")
        return 0

    def __eq__(self, other: object) -> bool:
        TRAPS.append("__eq__")
        return other is None

    def __ne__(self, other: object) -> bool:
        TRAPS.append("__ne__")
        return other is not None

    __hash__ = object.__hash__


class FalseAsync(SyncPlugin):
    chat_async = False  # type: ignore[assignment]


class ZeroAsync(SyncPlugin):
    chat_async = 0  # type: ignore[assignment]


class EmptyStringAsync(SyncPlugin):
    chat_async = ""  # type: ignore[assignment]


class TrapAsync(SyncPlugin):
    chat_async = TrapValue()  # type: ignore[assignment]


class GetattributeOverride(AsyncDef):
    def __getattribute__(self, name: str) -> Any:
        if name == "chat_async":
            TRAPS.append("__getattribute__ chat_async")
        return object.__getattribute__(self, name)


class OptOut(AsyncDef):
    chat_async = None  # type: ignore[assignment]


class GetattrOnly(SyncPlugin):
    def __getattr__(self, name: str) -> Any:
        TRAPS.append(f"__getattr__ {name}")
        if name == "chat_async":
            return AsyncDef().chat_async
        raise AttributeError(name)


class ForwardingProxy:
    """A runtime-only provider proxy (not a BaseProvider)."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        TRAPS.append(f"proxy __getattr__ {name}")
        return getattr(object.__getattribute__(self, "_target"), name)


class MetaWithAsync(type(BaseProvider)):
    async def chat_async(cls, **kwargs):  # visible on the class object only, never on instances
        TRAPS.append("metaclass chat_async")
        return reply()


class MetaclassOnly(SyncPlugin, metaclass=MetaWithAsync):
    pass


def declared_shapes() -> list[tuple[str, Any]]:
    instance_mock = MagicMock()
    instance_mock.chat_async = AsyncMock(return_value=reply())
    return [
        ("async def", AsyncDef()), ("inherited async def", InheritedAsync()),
        ("instance-bound method", InstanceBound()), ("MagicMock with an instance AsyncMock", instance_mock),
        ("property", PropertyAsync()), ("descriptor", DescriptorAsync()), ("staticmethod", StaticAsync()),
        ("classmethod", ClassAsync()), ("sync def returning a coroutine", SyncReturnsCoroutine()),
        ("callable instance", CallableAsync()), ("non-callable non-None value", NonCallableAsync()),
        ("__getattribute__ override", GetattributeOverride()), ("False", FalseAsync()), ("0", ZeroAsync()),
        ("empty string", EmptyStringAsync()), ("value with trapped truthiness and comparisons", TrapAsync()),
    ]


def undeclared_shapes() -> list[tuple[str, Any]]:
    return [
        ("missing", SyncPlugin()), ("None opt-out", OptOut()), ("__getattr__ only", GetattrOnly()),
        ("forwarding proxy", ForwardingProxy(AsyncDef())), ("metaclass only", MetaclassOnly()),
        ("bare Mock", Mock()), ("bare MagicMock", MagicMock()), ("AsyncMock provider", AsyncMock()),
    ]


class TestCompactMethodDecision(unittest.TestCase):
    """The decision is static: nothing on the provider is executed to make it."""

    def setUp(self) -> None:
        TRAPS.clear()

    def test_statically_declared_chat_async_selects_the_async_method(self) -> None:
        for label, provider in declared_shapes():
            with self.subTest(label):
                self.assertIs(_declares_chat_async(provider), True)
        self.assertEqual(TRAPS, [])

    def test_missing_none_and_dynamic_only_select_chat(self) -> None:
        for label, provider in undeclared_shapes():
            with self.subTest(label):
                self.assertIs(_declares_chat_async(provider), False)
                if isinstance(provider, (Mock, MagicMock, AsyncMock)):
                    self.assertNotIn("chat_async", provider._mock_children)  # no child was created
        self.assertEqual(TRAPS, [])

    def test_uninspectable_provider_is_refused_before_any_provider_call(self) -> None:
        class FreshAsync(AsyncPlugin):
            pass

        provider = FreshAsync()
        conv = conversation()
        before = snapshot(conv)
        with patch("src.compact_service.service.inspect.getattr_static", side_effect=RuntimeError("uninspectable")):
            with self.assertRaises(CompactMethodUndeterminedError) as caught:
                _declares_chat_async(provider)
            self.assertEqual(str(caught.exception), UNDETERMINED)
            _, error, conv, _ = compact(provider, conv)
        self.assertIsInstance(error, CompactMethodUndeterminedError)
        self.assertEqual(provider.entered, [])  # neither chat_async nor chat
        self.assertEqual(snapshot(conv), before)


    def test_decision_is_per_instance_not_per_class(self) -> None:
        class Fresh(SyncPlugin):
            pass

        with_async, without = Fresh(), Fresh()
        with_async.chat_async = AsyncMock(return_value=reply())
        self.assertIs(_declares_chat_async(with_async), True)
        self.assertIs(_declares_chat_async(without), False)
        opted_out, declared = AsyncPlugin(), AsyncPlugin()
        opted_out.chat_async = None  # type: ignore[assignment]
        self.assertIs(_declares_chat_async(opted_out), False)
        self.assertIs(_declares_chat_async(declared), True)
        result, error, _, _ = compact(without)
        self.assertIsNone(error)
        self.assertEqual(without.entered, ["chat"])
        result, error, _, _ = compact(opted_out)
        self.assertIsNone(error)
        self.assertEqual(opted_out.entered, ["chat"])

    def test_falsy_declaration_is_called_and_fails_without_chat(self) -> None:
        provider = FalseAsync()
        _, error, conv, before = compact(provider)
        self.assertEqual(provider.entered, [])  # chat() never entered
        self.assertEqual(snapshot(conv), before)
        self.assertIsInstance(error, TypeError)


# --- async selected: any failure ends the attempt ------------------------------------------------------

def async_failures() -> list[tuple[str, dict[str, Any], type[BaseException]]]:
    return [
        ("RuntimeError after send", {"after_send": RuntimeError("after send")}, RuntimeError),
        ("RuntimeError before send", {"before_send": RuntimeError("before send")}, RuntimeError),
        ("AttributeError", {"after_send": AttributeError("inside")}, AttributeError),
        ("TypeError", {"after_send": TypeError("inside")}, TypeError),
        ("ValueError", {"after_send": ValueError("inside")}, ValueError),
        ("KeyError", {"after_send": KeyError("inside")}, KeyError),
        ("TimeoutError", {"after_send": TimeoutError("timed out")}, TimeoutError),
        ("asyncio.TimeoutError", {"after_send": asyncio.TimeoutError()}, asyncio.TimeoutError),
        ("ConnectionResetError", {"after_send": ConnectionResetError("reset")}, ConnectionResetError),
        ("httpx.ReadTimeout", {"after_send": httpx.ReadTimeout("read timeout", request=OPENAI_REQUEST)},
         httpx.ReadTimeout),
        ("httpx.RemoteProtocolError",
         {"after_send": httpx.RemoteProtocolError("peer closed", request=OPENAI_REQUEST)}, httpx.RemoteProtocolError),
        ("openai.APITimeoutError", {"after_send": openai.APITimeoutError(request=OPENAI_REQUEST)},
         openai.APITimeoutError),
        ("openai.APIConnectionError", {"after_send": openai.APIConnectionError(request=OPENAI_REQUEST)},
         openai.APIConnectionError),
        ("openai.InternalServerError", {"after_send": openai.InternalServerError(
            "Error code: 500", response=httpx.Response(500, request=OPENAI_REQUEST), body=None)},
         openai.InternalServerError),
        ("anthropic.APIStatusError 529", {"after_send": anthropic.APIStatusError(
            "overloaded", response=httpx.Response(529, request=ANTHROPIC_REQUEST), body=None)},
         anthropic.APIStatusError),
        ("json.JSONDecodeError", {"after_send": json.JSONDecodeError("Expecting value", "", 0)},
         json.JSONDecodeError),
        ("IncompleteResponseError", {"after_send": IncompleteResponseError(
            "output_limit", partial_text="PARTIAL", partial_usage={"input_tokens": 9, "output_tokens": 4096})},
         IncompleteResponseError),
        ("FinishStatusError", {"after_send": FinishStatusError(
            "blocked", finish_value="content_filter", partial_text="PARTIAL")}, FinishStatusError),
        ("NotImplementedError", {"after_send": NotImplementedError("async unsupported after send")},
         NotImplementedError),
        ("NotImplementedError before send", {"before_send": NotImplementedError("no async")}, NotImplementedError),
        ("NotImplementedError subclass", {"after_send": SubNIE("plugin")}, SubNIE),
        ("plugin Exception subclass", {"after_send": PluginError("plugin")}, PluginError),
        ("RecursionError", {"after_send": RecursionError("deep")}, RecursionError),
    ]


def malformed_results() -> list[tuple[str, Any]]:
    return [
        ("content None", reply(None)), ("no content attribute", object()), ("a dict", {"content": "text"}),
        ("int content", reply(5)), ("list content", reply(["text"])), ("empty content", reply("")),
        ("whitespace content", reply("  \n ")),
    ]


class UsageRaises:
    content = "GOOD SUMMARY"

    @property
    def usage(self) -> Any:
        raise RuntimeError("usage metadata unavailable")


class TestAsyncSelectedFailures(unittest.TestCase):
    def assert_single_async_attempt(self, provider: AsyncPlugin, conv: Conversation, before: list) -> None:
        self.assertEqual(provider.entered, ["chat_async"])  # chat() never entered
        self.assertLessEqual(provider.sends, 1)
        self.assertEqual(snapshot(conv), before)  # history untouched

    def test_every_exception_ends_the_attempt_without_chat(self) -> None:
        for label, kwargs, expected in async_failures():
            with self.subTest(label):
                provider = AsyncPlugin(**kwargs)
                _, error, conv, before = compact(provider)
                self.assert_single_async_attempt(provider, conv, before)
                self.assertIsInstance(error, expected)

    def test_malformed_or_empty_summary_fails_without_chat(self) -> None:
        for label, result in malformed_results():
            with self.subTest(label):
                provider = AsyncPlugin(result=result)
                _, error, conv, before = compact(provider)
                self.assert_single_async_attempt(provider, conv, before)
                self.assertEqual(provider.sends, 1)
                self.assertIsInstance(error, CompactSummaryError)
                self.assertEqual(str(error), NO_SUMMARY)

    def test_cancellation_and_interrupt_propagate_unchanged(self) -> None:
        for error in (asyncio.CancelledError(), KeyboardInterrupt()):
            with self.subTest(type(error).__name__):
                provider = AsyncPlugin(after_send=error)
                _, raised, conv, before = compact(provider)
                self.assertIs(type(raised), type(error))
                self.assert_single_async_attempt(provider, conv, before)

    def test_a_good_summary_survives_unreadable_usage(self) -> None:
        for label, result in (("usage getter raises", UsageRaises()), ("usage is a list", reply(usage=[1, 2]))):
            with self.subTest(label):
                provider = AsyncPlugin(result=result)
                result_obj, error, conv, _ = compact(provider)
                self.assertIsNone(error)
                self.assertEqual(provider.entered, ["chat_async"])
                self.assertIsNone(result_obj.usage)
                self.assertEqual([m.role for m in conv.messages], ["system", "user"])  # boundary + summary
                self.assertIn("GOOD SUMMARY" if label == "usage getter raises" else "ASYNC SUMMARY",
                              message_text(conv.messages[1]))

    def test_success_uses_one_async_request(self) -> None:
        provider = AsyncPlugin()
        result, error, conv, _ = compact(provider)
        self.assertIsNone(error)
        self.assertEqual(provider.entered, ["chat_async"])
        self.assertEqual(result.summary_text, "ASYNC SUMMARY")
        self.assertEqual(result.usage, {"input_tokens": 11, "output_tokens": 5})
        self.assertEqual(provider.requests[0]["tools"], None)
        self.assertEqual(provider.requests[0]["max_tokens"], service.COMPACT_MAX_OUTPUT_TOKENS)


class TestPayloadAndState(unittest.TestCase):
    def test_both_methods_receive_the_same_single_payload(self) -> None:
        sync, asynchronous = SyncPlugin(), AsyncPlugin()
        compact(sync)
        compact(asynchronous)
        self.assertEqual(len(sync.requests), 1)
        self.assertEqual(len(asynchronous.requests), 1)
        for request in (sync.requests[0], asynchronous.requests[0]):
            self.assertEqual(set(request), {"messages", "tools", "model", "max_tokens"})
            self.assertIsNone(request["tools"])
            self.assertEqual(request["model"], "plugin-model")
            self.assertEqual(request["max_tokens"], service.COMPACT_MAX_OUTPUT_TOKENS)
        self.assertEqual(sync.requests[0], asynchronous.requests[0])

    def test_long_history_is_untouched_by_a_failed_request(self) -> None:
        for label, provider in (("async raises", AsyncPlugin(after_send=RuntimeError("after send"))),
                                ("sync empty summary", SyncPlugin(result=reply("  ")))):
            with self.subTest(label):
                conv = Conversation()
                for index in range(15):
                    conv.add_user_message(f"question {index}")
                    conv.add_assistant_message(f"answer {index}")
                _, error, conv, before = compact(provider, conv)
                self.assertIsNotNone(error)
                self.assertEqual(len(conv.messages), 30)
                self.assertEqual(snapshot(conv), before)

    def test_usage_is_copied(self) -> None:
        usage = {"input_tokens": 11, "output_tokens": 5}
        result, error, _, _ = compact(AsyncPlugin(result=reply(usage=usage)))
        self.assertIsNone(error)
        self.assertEqual(result.usage, usage)
        self.assertIsNot(result.usage, usage)


class TestSyncSelected(unittest.TestCase):
    def test_success_uses_one_chat_request(self) -> None:
        provider = SyncPlugin()
        result, error, conv, _ = compact(provider)
        self.assertIsNone(error)
        self.assertEqual(provider.entered, ["chat"])
        self.assertEqual(result.summary_text, "SYNC SUMMARY")
        self.assertEqual(result.usage, {"input_tokens": 7, "output_tokens": 3})

    def test_failure_surfaces_and_keeps_history(self) -> None:
        cases = [("RuntimeError", RuntimeError("chat failed"), RuntimeError),
                 ("NotImplementedError", NotImplementedError("no sync"), NotImplementedError),
                 ("IncompleteResponseError", IncompleteResponseError("output_limit", partial_text="PART"),
                  IncompleteResponseError)]
        for label, raised, expected in cases:
            with self.subTest(label):
                provider = SyncPlugin(error=raised)
                _, error, conv, before = compact(provider)
                self.assertEqual(snapshot(conv), before, "history replaced after a failed request")
                self.assertEqual(provider.entered, ["chat"])
                self.assertIsInstance(error, expected)

    def test_empty_summary_fails_and_keeps_history(self) -> None:
        provider = SyncPlugin(result=reply("   "))
        _, error, conv, before = compact(provider)
        self.assertEqual(snapshot(conv), before, "history replaced by a local stub")
        self.assertEqual(provider.entered, ["chat"])
        self.assertIsInstance(error, CompactSummaryError)


# --- attribute hooks and unusual shapes -----------------------------------------------------------------

class GetterSendsThenWorks(SyncPlugin):
    getter_runs = 0

    @property
    def chat_async(self):  # type: ignore[override]
        type(self).getter_runs += 1
        self.entered.append("getter-send")

        async def implementation(**kwargs):
            self.entered.append("chat_async")
            return reply()
        return implementation


class GetterSendsThenFails(SyncPlugin):
    @property
    def chat_async(self):  # type: ignore[override]
        self.entered.append("getter-send")
        raise AttributeError("chat_async")


class DynamicAsync(SyncPlugin):
    asked: list[str] = []

    def __getattr__(self, name: str) -> Any:
        type(self).asked.append(name)
        if name == "chat_async":
            raise AssertionError("dynamic chat_async was looked up")
        raise AttributeError(name)


class NarrowSignature(SyncPlugin):
    async def chat_async(self, messages):
        self.entered.append("chat_async body")
        return reply()


class AsyncGenerator(SyncPlugin):
    async def chat_async(self, **kwargs):
        self.entered.append("chat_async body")
        yield reply()


class TestAttributeHooksAndShapes(unittest.TestCase):
    def test_property_getter_runs_once_only_when_called(self) -> None:
        provider = GetterSendsThenWorks()
        GetterSendsThenWorks.getter_runs = 0
        self.assertTrue(_declares_chat_async(provider))
        self.assertEqual(GetterSendsThenWorks.getter_runs, 0)
        result, error, _, _ = compact(provider)
        self.assertIsNone(error)
        self.assertEqual(GetterSendsThenWorks.getter_runs, 1)
        self.assertEqual(provider.entered, ["getter-send", "chat_async"])

    def test_failing_getter_is_not_followed_by_chat(self) -> None:
        provider = GetterSendsThenFails()
        _, error, conv, before = compact(provider)
        self.assertEqual(provider.entered, ["getter-send"])
        self.assertIsInstance(error, AttributeError)
        self.assertEqual(snapshot(conv), before)

    def test_dynamic_only_chat_async_is_never_asked_for(self) -> None:
        provider = DynamicAsync()
        DynamicAsync.asked = []
        result, error, _, _ = compact(provider)
        self.assertIsNone(error)
        self.assertEqual(provider.entered, ["chat"])
        self.assertNotIn("chat_async", DynamicAsync.asked)

    def test_sync_function_returning_a_coroutine_works(self) -> None:
        TRAPS.clear()
        result, error, _, _ = compact(SyncReturnsCoroutine())
        self.assertIsNone(error)
        self.assertEqual(result.summary_text, "ASYNC SUMMARY")

    def test_unusable_declarations_fail_visibly_without_chat(self) -> None:
        for label, provider in (("non-callable value", NonCallableAsync()), ("narrow signature", NarrowSignature()),
                                ("async generator", AsyncGenerator())):
            with self.subTest(label):
                _, error, conv, before = compact(provider)
                self.assertNotIn("chat", provider.entered)
                self.assertEqual(snapshot(conv), before)
                self.assertIsInstance(error, TypeError)


# --- real SDK requests counted at the transport ------------------------------------------------------

class OpenAISendThenFail(OpenAIProvider):
    mode = "raise"

    async def chat_async(self, messages, tools=None, **kwargs):
        response = self.chat(messages, tools=tools, **kwargs)  # a real SDK request (counted)
        if self.mode == "raise":
            raise RuntimeError("post-processing failed")
        return ChatResponse(content=None, model=response.model, usage=response.usage,  # type: ignore[arg-type]
                            finish_reason="stop")


class AnthropicSendThenFail(AnthropicProvider):
    async def chat_async(self, messages, tools=None, **kwargs):
        self.chat(messages, tools=tools, **kwargs)  # a real SDK request (counted)
        raise RuntimeError("post-processing failed")


class OpenAITrueAsync(OpenAIProvider):
    async_client: Any = None

    async def chat_async(self, messages, tools=None, **kwargs):
        await self.async_client.chat.completions.create(
            model=self._get_model(**kwargs), messages=self._prepare_messages(messages),
            max_tokens=kwargs.get("max_tokens"))
        raise RuntimeError("post-processing failed")


class TestRealTransport(OfflineCase):
    def _subclass(self, cls: type, name: str) -> Any:
        provider = cls(api_key="test-dummy-key")
        shared = shared_provider(name)
        if name == "anthropic":
            provider.client = shared._ensure_client()
        else:
            provider._client = shared.client
        return provider

    def test_async_send_then_failure_is_one_physical_request(self) -> None:
        cases = [("openai raises", self._subclass(OpenAISendThenFail, "openai"), oa_reply, RuntimeError),
                 ("anthropic raises", self._subclass(AnthropicSendThenFail, "anthropic"), an_reply, RuntimeError)]
        malformed = self._subclass(OpenAISendThenFail, "openai")
        malformed.mode = "content None"
        cases.append(("openai malformed reply", malformed, oa_reply, CompactSummaryError))
        for label, provider, wire, expected in cases:
            with self.subTest(label):
                self.play(wire("FIRST"), wire("TRAP-RESEND"))
                _, error, conv, before = compact(provider)
                self.assertEqual(len(self.sent), 1, "physical sends")
                self.assertEqual(len(self.script), 1, "the trap reply was consumed (a second request)")
                self.assertEqual(snapshot(conv), before)
                self.assertIsInstance(error, expected)

    def test_unmodified_builtins_make_one_request(self) -> None:
        for name, wire in (("openai", oa_reply), ("anthropic", an_reply)):
            with self.subTest(name):
                self.play(wire("BUILTIN SUMMARY"), wire("TRAP-RESEND"))
                result, error, _, _ = compact(shared_provider(name))
                self.assertIsNone(error)
                self.assertEqual(result.summary_text, "BUILTIN SUMMARY")
                self.assertEqual(len(self.sent), 1)
        self.play(oa_reply(status=500), oa_reply("TRAP-RESEND"))
        _, error, conv, before = compact(shared_provider("openai"))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(snapshot(conv), before, "history replaced by a local stub after a failed request")
        self.assertIsInstance(error, openai.InternalServerError)

    def test_true_async_sdk_send_then_failure(self) -> None:
        provider = OpenAITrueAsync(api_key="test-dummy-key")
        provider.async_client = openai.AsyncOpenAI(
            api_key="test-dummy-key", max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(follow_redirects=False))
        conv = conversation()
        before = snapshot(conv)
        self.play(oa_reply("FIRST"), oa_reply("TRAP-RESEND"))

        async def compact_then_close() -> None:
            try:
                await compact_conversation(conv, provider, "gpt-test")
            finally:
                await provider.async_client.close()

        with self.loopback_socketpair_only():
            try:
                asyncio.run(compact_then_close())
                error = None
            except RuntimeError as exc:
                error = exc
        self.assertEqual(len(self.sent), 1, "physical sends (one async)")
        self.assertEqual(len(self.script), 1)
        self.assertEqual(snapshot(conv), before)
        self.assertIsInstance(error, RuntimeError)


class TestDiskLoadedPlugin(OfflineCase):
    SOURCE = '''
from src.providers.base import BaseProvider, ChatResponse

class AsyncSendThenRaise(BaseProvider):
    LEDGER = []
    def __init__(self, api_key, base_url=None, model=None):
        super().__init__(api_key, base_url, model or "m")
    async def chat_async(self, **kwargs):
        type(self).LEDGER.append("chat_async")
        raise RuntimeError("after send")
    def chat(self, messages, tools=None, **kwargs):
        type(self).LEDGER.append("chat")
        return ChatResponse(content="resend", model="m", usage={}, finish_reason="stop")
    def chat_stream(self, messages, tools=None, **kwargs):
        yield "never"
    def get_available_models(self):
        return ["m"]

PROVIDERS = [{"name": "disk-async-send-raise", "label": "disk", "provider_class": AsyncSendThenRaise,
              "default_base_url": "http://127.0.0.1:11434/v1", "default_model": "m",
              "available_models": ["m"], "requires_api_key": False, "local_only": True}]
'''

    def test_disk_loaded_async_plugin_is_never_followed_by_chat(self) -> None:
        self.addCleanup(clear_plugin_providers)
        plugins = self.tmp / ".clawd" / "plugins"
        plugin_dir = plugins / "compactsend"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.json").write_text(json.dumps({
            "schema_version": 1, "name": "compactsend", "version": "1.0.0",
            "entrypoint": "plugin.py", "extensions": ["providers"]}), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(self.SOURCE, encoding="utf-8")
        policy = self.tmp / ".clawd" / "python_plugins.json"
        report = reconcile_python_plugins(plugin_root=plugins, operator_manifest=policy)
        digest = report["records"]["compactsend"]["artifact_sha256"]
        policy.write_text(json.dumps({"schema_version": 1, "plugins": {
            "compactsend": {"enabled": True, "artifact_sha256": digest}}}), encoding="utf-8")
        with patch("src.plugins.runtime.default_plugin_root", return_value=plugins), \
             patch("src.plugins.runtime.default_operator_manifest_path", return_value=policy):
            loaded = load_active_plugin_extensions()
        self.assertEqual(loaded.issues, [])
        self.assertEqual(register_plugin_provider_extensions(loaded), [])
        cls = get_provider_class("disk-async-send-raise")
        _, error, conv, before = compact(cls(api_key="", base_url=None, model=None))
        self.assertEqual(cls.LEDGER, ["chat_async"])
        self.assertEqual(snapshot(conv), before)
        self.assertIsInstance(error, RuntimeError)


# --- command layer ----------------------------------------------------------------------------------

class TestCommandLayer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.conversation = conversation()
        self.cost_tracker = CostTracker()
        self.context = create_command_context(
            workspace_root=Path(self._tmpdir.name).resolve(), conversation=self.conversation,
            cost_tracker=self.cost_tracker, history=HistoryLog())
        self.fallback = MagicMock(wraps=command_builtins._sync_compact_fallback)
        patcher = patch.object(command_builtins, "_sync_compact_fallback", self.fallback)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_command(self, provider: Any) -> Any:
        self.context.config["provider"] = provider
        self.context.config["model"] = "plugin-model"
        with patch.object(command_builtins.asyncio, "run", side_effect=_run_without_event_loop):
            return command_builtins.compact_command_call("", self.context)

    def assert_failed_and_preserved(self, result: Any, before: list, text: str) -> None:
        self.assertEqual(self.fallback.call_count, 0, "the local _sync_compact_fallback ran")
        self.assertEqual(snapshot(self.conversation), before, "history was replaced")
        self.assertEqual((result.type, result.value), ("text", text))
        self.assertEqual(self.cost_tracker.provider_usage, {})

    def test_provider_failure_reports_compact_failed(self) -> None:
        before = snapshot(self.conversation)
        provider = AsyncPlugin(after_send=RuntimeError("after send"))
        with patch("traceback.print_exc"):
            result = self.run_command(provider)
        self.assertEqual(provider.entered, ["chat_async"])
        self.assert_failed_and_preserved(result, before, "Compact failed: after send")

    def test_unprintable_exception_gets_safe_text_and_no_local_fallback(self) -> None:
        for error in (BrokenStr(), BrokenStrNIE()):
            with self.subTest(type(error).__name__):
                self.conversation.messages[:] = conversation().messages
                before = snapshot(self.conversation)
                provider = AsyncPlugin(after_send=error)
                with patch("traceback.print_exc"):
                    result = self.run_command(provider)
                self.assert_failed_and_preserved(result, before, f"Compact failed: {type(error).__name__}")
                self.assertEqual(provider.entered, ["chat_async"])

    def test_provider_value_errors_are_compact_failures(self) -> None:
        for error in (ValueError("bad value"), json.JSONDecodeError("Expecting value", "", 0)):
            with self.subTest(type(error).__name__):
                self.conversation.messages[:] = conversation().messages
                before = snapshot(self.conversation)
                provider = AsyncPlugin(after_send=error)
                with patch("traceback.print_exc"):
                    result = self.run_command(provider)
                self.assertEqual(provider.entered, ["chat_async"])
                self.assert_failed_and_preserved(result, before, f"Compact failed: {error}")

    def test_service_failures_report_compact_failed_after_one_attempt(self) -> None:
        cases = [
            ("unusable async summary", AsyncPlugin(result=reply("   ")), ["chat_async"], f"Compact failed: {NO_SUMMARY}"),
            ("unusable sync summary", SyncPlugin(result=reply(None)), ["chat"], f"Compact failed: {NO_SUMMARY}"),
            ("message-less timeout", AsyncPlugin(after_send=TimeoutError()), ["chat_async"],
             "Compact failed: TimeoutError"),
            ("blank message", AsyncPlugin(after_send=RuntimeError("  ")), ["chat_async"], "Compact failed: RuntimeError"),
            ("str subclass with raising methods", AsyncPlugin(after_send=OddStr()), ["chat_async"],
             "Compact failed: odd text"),  # normalised to an exact str; no subclass method runs
            ("partial usage is not recorded", AsyncPlugin(after_send=IncompleteResponseError(
                "output_limit", partial_text="P", partial_usage={"input_tokens": 9, "output_tokens": 4096})),
             ["chat_async"], "Compact failed: The provider stopped at its output limit; the response is incomplete."),
        ]
        for label, provider, entered, text in cases:
            with self.subTest(label):
                self.conversation.messages[:] = conversation().messages
                before = snapshot(self.conversation)
                with patch("traceback.print_exc"):
                    result = self.run_command(provider)
                self.assertEqual(provider.entered, entered)  # one attempt, one method
                self.assert_failed_and_preserved(result, before, text)

    def test_undetermined_method_reports_compact_failed_without_a_call(self) -> None:
        class FreshAsync(AsyncPlugin):
            pass

        before = snapshot(self.conversation)
        provider = FreshAsync()
        with patch("src.compact_service.service.inspect.getattr_static", side_effect=RuntimeError("uninspectable")), \
             patch("traceback.print_exc"):
            result = self.run_command(provider)
        self.assertEqual(provider.entered, [])
        self.assert_failed_and_preserved(result, before, f"Compact failed: {UNDETERMINED}")

    def test_interrupts_propagate_through_the_command(self) -> None:
        for error in (KeyboardInterrupt(), asyncio.CancelledError()):
            with self.subTest(type(error).__name__):
                self.conversation.messages[:] = conversation().messages
                before = snapshot(self.conversation)
                provider = AsyncPlugin(after_send=error)
                with self.assertRaises(type(error)):
                    self.run_command(provider)
                self.assertEqual(provider.entered, ["chat_async"])
                self.assertEqual(self.fallback.call_count, 0)
                self.assertEqual(snapshot(self.conversation), before)

    def test_unprintable_exception_escaping_asyncio_run_gets_safe_text(self) -> None:
        before = snapshot(self.conversation)
        self.context.config["provider"] = AsyncPlugin()

        def loop_fails(coroutine: Any) -> Any:
            coroutine.close()
            raise BrokenStr()

        with patch.object(command_builtins.asyncio, "run", side_effect=loop_fails), patch("traceback.print_exc"):
            result = command_builtins.compact_command_call("", self.context)
        self.assert_failed_and_preserved(result, before, "Compact failed: BrokenStr")

    def test_exception_escaping_asyncio_run_never_runs_the_local_fallback(self) -> None:
        before = snapshot(self.conversation)
        provider = AsyncPlugin()
        self.context.config["provider"] = provider

        def loop_fails(coroutine: Any) -> Any:
            coroutine.close()
            raise RuntimeError("event loop could not start")

        with patch.object(command_builtins.asyncio, "run", side_effect=loop_fails), patch("traceback.print_exc"):
            result = command_builtins.compact_command_call("", self.context)
        self.assert_failed_and_preserved(result, before, "Compact failed: event loop could not start")
        self.assertEqual(provider.entered, [])

    def test_running_event_loop_keeps_the_local_only_path(self) -> None:
        provider = AsyncPlugin()
        self.context.config["provider"] = provider
        with patch.object(command_builtins.asyncio, "get_running_loop", return_value=object()), \
             patch.object(command_builtins.asyncio, "run", side_effect=AssertionError("asyncio.run must not run")):
            result = command_builtins.compact_command_call("", self.context)
        self.fallback.assert_called_once_with(self.context)
        self.assertEqual(result.type, "compact")
        self.assertEqual(provider.entered, [])

    def test_not_enough_messages_keeps_the_friendly_text(self) -> None:
        from src.compact_service.messages import create_compact_boundary_message

        self.conversation.messages[:] = [conversation().messages[0],
                                         create_compact_boundary_message(trigger="manual", pre_compact_token_count=1)]
        before = snapshot(self.conversation)
        provider = AsyncPlugin()
        result = self.run_command(provider)
        self.assertEqual(result.type, "text")
        self.assertEqual(result.value, "Not enough messages to compact.")
        self.assertEqual(provider.entered, [])
        self.assertEqual(snapshot(self.conversation), before)
        self.assertTrue(issubclass(NotEnoughMessagesToCompactError, ValueError))  # direct callers keep working
        with self.assertRaises(NotEnoughMessagesToCompactError):
            _run_without_event_loop(compact_conversation(self.conversation, provider, "m"))

    def test_success_records_usage_and_compacts(self) -> None:
        provider = SyncPlugin()
        result = self.run_command(provider)
        self.assertEqual(result.type, "compact")
        self.assertEqual(provider.entered, ["chat"])
        self.assertEqual([m.role for m in self.conversation.messages], ["system", "user"])
        self.assertEqual(sum(v["input_tokens"] for v in self.cost_tracker.provider_usage.values()), 7)
        self.fallback.assert_not_called()


# --- whitelist source guard -------------------------------------------------------------------------

PROVIDER_METHODS = {"chat", "chat_async", "chat_stream", "chat_stream_response"}
FORBIDDEN_ANCESTORS = (ast.ExceptHandler, ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp,
                       ast.DictComp, ast.GeneratorExp, ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef,
                       ast.Try)


def _function(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    found = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    assert len(found) == 1, name
    return found[0]


def _parents(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST], stop: ast.AST) -> list[ast.AST]:
    chain = []
    while node in parents and parents[node] is not stop:
        node = parents[node]
        chain.append(node)
    return chain


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


class TestCompactSourceGuard(unittest.TestCase):
    """Defence in depth: the behaviour tests above catch the resends; this pins the shape."""

    def setUp(self) -> None:
        self.service = ast.parse((SRC / "compact_service" / "service.py").read_text(encoding="utf-8-sig"))
        self.builtins = ast.parse((SRC / "command_system" / "builtins.py").read_text(encoding="utf-8-sig"))

    def test_compact_conversation_calls_exactly_one_selected_method(self) -> None:
        function = _function(self.service, "compact_conversation")
        parents = _parents(function)
        provider_uses = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == "provider"]
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        method_calls = [c for c in calls if isinstance(c.func, ast.Attribute)
                        and isinstance(c.func.value, ast.Name) and c.func.value.id == "provider"]
        decisions = [c for c in calls if _call_name(c) == "_declares_chat_async"]
        self.assertEqual(sorted(c.func.attr for c in method_calls), ["chat", "chat_async"])
        self.assertEqual(len(decisions), 1)
        self.assertEqual([type(a).__name__ for a in decisions[0].args], ["Name"])
        # provider is used only as the decision argument and as the receiver of the two calls
        allowed = {id(decisions[0].args[0])} | {id(c.func.value) for c in method_calls}
        self.assertEqual({id(n) for n in provider_uses}, allowed)
        selection = parents[decisions[0]]
        self.assertIsInstance(selection, ast.If)
        branches = {}
        for call in method_calls:
            chain = _ancestors(call, parents, function)
            self.assertFalse([type(a).__name__ for a in chain if isinstance(a, FORBIDDEN_ANCESTORS)],
                             f"provider.{call.func.attr} is inside a handler, loop, nested scope or try")
            self.assertIn(selection, chain)
            branch = next(a for a in [call, *chain] if parents.get(a) is selection)
            branches[call.func.attr] = "body" if branch in selection.body else "orelse"
        self.assertEqual(branches, {"chat_async": "body", "chat": "orelse"})
        self.assertNotIn("compact_conversation", {_call_name(c) for c in calls})  # no recursion
        self.assertNotIn("_fallback_summary", {_call_name(c) for c in calls})

    def test_no_other_provider_method_use_in_the_service(self) -> None:
        compact_fn = _function(self.service, "compact_conversation")
        inside = {id(n) for n in ast.walk(compact_fn)}
        for node in ast.walk(self.service):
            if id(node) in inside:
                continue
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, PROVIDER_METHODS, f"line {node.lineno}")
            if isinstance(node, ast.Call) and _call_name(node) in {"getattr", "hasattr", "setattr"}:
                self.fail(f"dynamic attribute access at line {node.lineno}")
        decide = _function(self.service, "_declares_chat_async")
        static = [c for c in ast.walk(decide) if isinstance(c, ast.Call) and _call_name(c) == "getattr_static"]
        self.assertEqual(len(static), 1)
        self.assertIsInstance(static[0].args[0], ast.Name)  # the instance, not type(provider)
        self.assertEqual(static[0].args[0].id, "provider")
        self.assertEqual(static[0].args[1].value, "chat_async")  # type: ignore[attr-defined]
        for handler in (n for n in ast.walk(self.service) if isinstance(n, ast.ExceptHandler)):
            names = {_call_name(c) for c in ast.walk(handler) if isinstance(c, ast.Call)}
            self.assertLessEqual(names, {"CompactMethodUndeterminedError"}, f"handler at line {handler.lineno}")

    def test_command_layer_never_calls_a_provider_or_falls_back_after_asyncio_run(self) -> None:
        compact_async = _function(self.builtins, "_compact_async")
        command_call = _function(self.builtins, "compact_command_call")
        sync_fallback = _function(self.builtins, "_sync_compact_fallback")
        for function in (compact_async, command_call, sync_fallback):
            for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
                self.assertNotIn(_call_name(call), PROVIDER_METHODS, f"{function.name} line {call.lineno}")
                self.assertNotIn(_call_name(call), {"compact_command_call"}, f"{function.name} recursion")
        self.assertFalse([n for n in ast.walk(sync_fallback) if isinstance(n, ast.Constant) and n.value == "provider"])
        compact_calls = [c for c in ast.walk(compact_async)
                         if isinstance(c, ast.Call) and _call_name(c) == "compact_conversation"]
        self.assertEqual(len(compact_calls), 1)
        chain = _ancestors(compact_calls[0], _parents(compact_async), compact_async)
        self.assertFalse([type(a).__name__ for a in chain if isinstance(a, FORBIDDEN_ANCESTORS)
                          and not isinstance(a, ast.Try)], "compact_conversation is looped, handled or nested")
        self.assertNotIn("_sync_compact_fallback",
                         {_call_name(c) for c in ast.walk(compact_async) if isinstance(c, ast.Call)})
        parents = _parents(command_call)
        runs = [c for c in ast.walk(command_call) if isinstance(c, ast.Call) and _call_name(c) == "run"]
        self.assertEqual(len(runs), 1)
        fallbacks = [c for c in ast.walk(command_call)
                     if isinstance(c, ast.Call) and _call_name(c) == "_sync_compact_fallback"]
        self.assertEqual(len(fallbacks), 1)
        for call in runs + fallbacks:
            chain = _ancestors(call, parents, command_call)
            self.assertFalse([a for a in chain if isinstance(a, (ast.ExceptHandler, ast.For, ast.AsyncFor,
                                                                   ast.While))], f"line {call.lineno}")
        fallback_chain = _ancestors(fallbacks[0], parents, command_call)
        self.assertFalse([a for a in fallback_chain if isinstance(a, ast.Try)],
                         "the local fallback must not sit inside the asyncio.run try")
        self.assertLess(fallbacks[0].lineno, runs[0].lineno)  # decided before asyncio.run is reached


if __name__ == "__main__":
    unittest.main()
