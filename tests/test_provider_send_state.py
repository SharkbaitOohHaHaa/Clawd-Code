"""Provider method choice and send state: an exception never triggers another provider request.

Clawd decides before calling anything whether a provider offers chat_stream_response
(supports_structured_streaming: a static SUPPORTS_STRUCTURED_STREAMING bool, else "does the
class override BaseProvider's default"). Once it has chosen a method, any exception from that
method (NotImplementedError included) surfaces; no chat() / chat_stream() retry follows,
because an exception can never prove that nothing was sent.

Offline only: credential, base-URL and proxy environment variables are scrubbed, sockets and
DNS are refused, sleeps are recorded, and every SDK HTTP send is counted at
``httpx.HTTPTransport.handle_request``. Synthetic plugin providers keep their own send ledger.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, patch

import httpx
from prompt_toolkit.output import DummyOutput

from src.agent import Conversation
from src.compact_service.service import compact_conversation
from src.plugins.extensions import load_active_plugin_extensions, register_plugin_provider_extensions
from src.plugins.runtime import reconcile_python_plugins
from src.providers import (
    _BUILTIN_PROVIDER_NAMES,
    clear_plugin_providers,
    get_provider_class,
)
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import BaseProvider, ChatResponse, supports_structured_streaming
from src.providers.openai_provider import OpenAIProvider
from src.repl import ClawdREPL
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.permissions import ToolPermissionContext

SRC = Path(__file__).resolve().parents[1] / "src"
SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
ANTHROPIC_FAMILY = ("anthropic", "minimax")
NIE_HEADLINE = "Provider error: the active provider reported an unimplemented operation."
NO_RETRY = "The request failed. Clawd did not issue a fallback retry."
EMPTY_NOTICE = "The provider returned an empty response. Clawd did not issue a fallback retry."


class SubNIE(NotImplementedError):
    """A plugin's own NotImplementedError subclass."""


# --- synthetic plugin providers (their own send ledger) -----------------------------------

def reply(text: str = "done") -> ChatResponse:
    return ChatResponse(content=text, model="plugin-model", usage={}, finish_reason="stop")


class MinimalPlugin(BaseProvider):
    """A direct BaseProvider plugin with only the required surface (base chat_stream_response)."""

    def __init__(self, api_key: str = "k", base_url: str | None = None, model: str | None = None):
        super().__init__(api_key, base_url, model or "plugin-model")
        self.ledger: list[str] = []
        self.legacy_chunks = ["legacy ", "text"]
        self.chat_text = "done"

    def chat(self, messages, tools=None, **kwargs):
        self.ledger.append("chat")
        return reply(self.chat_text)

    def chat_stream(self, messages, tools=None, **kwargs):
        self.ledger.append("chat_stream")
        yield from self.legacy_chunks

    def get_available_models(self):
        return ["plugin-model"]


class StreamingPlugin(MinimalPlugin):
    """Overrides chat_stream_response: 'sends', runs a scripted action, then raises."""

    def __init__(self, *args: Any, before_raise: Callable[..., None] | None = None,
                 error: BaseException | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.before_raise = before_raise
        self.error = error if error is not None else NotImplementedError("unsupported delta")

    def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
        self.ledger.append("chat_stream_response")  # the request went out
        if self.before_raise is not None:
            self.before_raise(messages, tools, on_text_chunk, kwargs)
        raise self.error


class DeclaredUnsupported(StreamingPlugin):
    SUPPORTS_STRUCTURED_STREAMING = False


class AsyncPlugin(MinimalPlugin):
    def __init__(self, *args: Any, error: BaseException, sends_first: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.error = error
        self.sends_first = sends_first

    async def chat_async(self, **kwargs):
        if self.sends_first:
            self.ledger.append("chat_async")
        raise self.error


def emit(text: str) -> Callable[..., None]:
    return lambda messages, tools, on_text_chunk, kwargs: on_text_chunk(text)


def mutate(messages, tools, on_text_chunk, kwargs) -> None:
    messages.append({"role": "user", "content": "INJECTED"})
    if isinstance(tools, list):
        tools.append({"name": "InjectedTool", "description": "", "input_schema": {"type": "object"}})


# --- shared offline harness -----------------------------------------------------------------

_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each SDK client costs an SSL context)."""
    if name not in _SHARED:
        _SHARED[name] = get_provider_class(name)(
            api_key=FAKE_KEYS.get(name, "test-dummy-key"), base_url=None, model=None
        )
    return _SHARED[name]


def _sse(events: list[Any]) -> bytes:
    lines: list[str] = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        lines.append("data: " + (data if isinstance(data, str) else json.dumps(data)))
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


OA_CHUNK = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}


def oa_chunk(delta: dict[str, Any], finish: Any = None) -> tuple[None, dict[str, Any]]:
    return (None, {**OA_CHUNK, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


OA_TOOL_DELTA = oa_chunk({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                          "function": {"name": "Read", "arguments": '{"file_path": '}}]})
AN_START = ("message_start", {"type": "message_start", "message": {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
    "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}})
AN_TOOL_START = ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
    "type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}})
AN_THINKING_START = ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
    "type": "thinking", "thinking": "", "signature": ""}})


def stream_response(events: list[Any]) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


class RaisingStream(httpx.SyncByteStream):
    """A response body that delivers a first SSE part, then fails while being read (post-send)."""

    def __init__(self, first: bytes) -> None:
        self.first = first

    def __iter__(self):
        yield self.first
        raise NotImplementedError("stream decoder not implemented")

    def close(self) -> None:
        pass


def failing_stream(events: list[Any]) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=RaisingStream(_sse(events)))


def text_stream(name: str, text: str) -> httpx.Response:
    if name in ANTHROPIC_FAMILY:
        return stream_response([
            AN_START,
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 7}}),
            ("message_stop", {"type": "message_stop"}),
        ])
    return stream_response([oa_chunk({"role": "assistant", "content": text}), oa_chunk({}, "stop"), (None, "[DONE]")])


def plain_reply(name: str, text: str = "done") -> httpx.Response:
    if name in ANTHROPIC_FAMILY:
        return httpx.Response(200, json={
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2}})
    return httpx.Response(200, json={
        "id": "c2", "object": "chat.completion", "created": 0, "model": "gpt-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


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

        def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            request.read()
            self.sent.append(request)
            if not self.script:
                raise AssertionError("unexpected extra provider request")
            response = self.script.pop(0)
            response.request = request
            return response

        self._start(patch.object(httpx.HTTPTransport, "handle_request", handle_request))
        self.registry = build_default_registry()

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

    def play(self, *responses: httpx.Response) -> None:
        self.sent.clear()
        self.sleeps.clear()
        self.script = list(responses)

    def run_loop(self, provider: Any, *, stream: bool = True) -> Any:
        conversation = Conversation()
        conversation.add_user_message("use the tools")
        return run_agent_loop(
            conversation=conversation, provider=provider, tool_registry=self.registry,
            tool_context=ToolContext(workspace_root=self.tmp), stream=stream, verbose=False,
        )

    def spy_tools(self) -> list[MagicMock]:
        spies = []
        for name in ("Read", "TaskList"):
            tool = self.registry.get(name)
            run = MagicMock(side_effect=AssertionError(f"{name} must not run"))
            spies.append(run)
            self._start(patch.object(tool, "run", run))
        return spies


def _run_without_event_loop(coroutine: Any) -> Any:
    """Drive a coroutine that never suspends (asyncio's Windows loop needs a local socket
    pair, which the offline guard refuses)."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("the coroutine suspended; it was expected to finish without awaiting")


class TestCapabilityDecision(unittest.TestCase):
    """The choice is static: nothing on the provider is executed to make it."""

    def test_base_default_is_unsupported_and_builtins_are_supported(self) -> None:
        self.assertFalse(supports_structured_streaming(MinimalPlugin()))
        self.assertEqual(set(_BUILTIN_PROVIDER_NAMES),
                         {"anthropic", "minimax", "openai", "deepseek", "qwen", "glm"})
        for name in _BUILTIN_PROVIDER_NAMES:
            with self.subTest(provider=name):
                cls = get_provider_class(name)
                self.assertIsNone(cls.SUPPORTS_STRUCTURED_STREAMING)
                self.assertTrue(supports_structured_streaming(object.__new__(cls)))
        self.assertIsNone(BaseProvider.SUPPORTS_STRUCTURED_STREAMING)

    def test_override_derives_supported(self) -> None:
        self.assertTrue(supports_structured_streaming(StreamingPlugin()))

        class InheritsOverride(StreamingPlugin):
            pass

        self.assertTrue(supports_structured_streaming(InheritsOverride()))

    def test_explicit_class_and_instance_bools_are_honored(self) -> None:
        class ClassTrue(MinimalPlugin):
            SUPPORTS_STRUCTURED_STREAMING = True

        class ClassNone(StreamingPlugin):
            SUPPORTS_STRUCTURED_STREAMING = None

        self.assertFalse(supports_structured_streaming(DeclaredUnsupported()))  # False despite override
        self.assertTrue(supports_structured_streaming(ClassTrue()))  # True without override
        self.assertTrue(supports_structured_streaming(ClassNone()))  # None derives
        per_model = StreamingPlugin()
        per_model.SUPPORTS_STRUCTURED_STREAMING = False  # e.g. set in __init__ for one model
        self.assertFalse(supports_structured_streaming(per_model))
        enabled = MinimalPlugin()
        enabled.SUPPORTS_STRUCTURED_STREAMING = True
        self.assertTrue(supports_structured_streaming(enabled))

    def test_non_bool_declarations_are_ignored(self) -> None:
        for value in (1, 0, "false", "yes", [], None):
            with self.subTest(value=value):
                minimal, overriding = MinimalPlugin(), StreamingPlugin()
                minimal.SUPPORTS_STRUCTURED_STREAMING = value
                overriding.SUPPORTS_STRUCTURED_STREAMING = value
                self.assertFalse(supports_structured_streaming(minimal))
                self.assertTrue(supports_structured_streaming(overriding))

    def test_property_getattr_and_descriptor_declarations_are_never_executed(self) -> None:
        calls: list[str] = []

        class PropertyDeclaration(MinimalPlugin):
            @property
            def SUPPORTS_STRUCTURED_STREAMING(self):  # type: ignore[override]
                calls.append("property")
                raise AssertionError("a declaration property was executed")

        class DynamicAttributes(StreamingPlugin):
            def __getattr__(self, name: str) -> Any:
                calls.append(name)
                raise AssertionError("__getattr__ was executed")

        class Descriptor:
            def __get__(self, instance, owner):
                calls.append("descriptor")
                raise AssertionError("a descriptor was executed")

        class DescriptorDeclaration(StreamingPlugin):
            SUPPORTS_STRUCTURED_STREAMING = Descriptor()

        self.assertFalse(supports_structured_streaming(PropertyDeclaration()))  # derives: base default
        self.assertTrue(supports_structured_streaming(DynamicAttributes()))
        self.assertTrue(supports_structured_streaming(DescriptorDeclaration()))
        self.assertEqual(calls, [])

    def test_unknown_or_uninspectable_shapes_are_supported(self) -> None:
        self.assertTrue(supports_structured_streaming(Mock()))
        self.assertTrue(supports_structured_streaming(MagicMock()))
        self.assertTrue(supports_structured_streaming(object()))
        declared = Mock()
        declared.SUPPORTS_STRUCTURED_STREAMING = False
        self.assertFalse(supports_structured_streaming(declared))
        with patch("src.providers.base.inspect.getattr_static", side_effect=RuntimeError("uninspectable")):
            self.assertTrue(supports_structured_streaming(MinimalPlugin()))


class TestNoProviderHandlerRetries(unittest.TestCase):
    """Source guard: no production NotImplementedError handler can lead to another provider call."""

    @staticmethod
    def _names_nie(node: ast.expr | None) -> bool:
        if node is None:
            return False
        items = node.elts if isinstance(node, ast.Tuple) else [node]
        return any(isinstance(item, ast.Name) and item.id == "NotImplementedError" for item in items)

    def test_only_compaction_has_one_and_it_re_raises_before_the_generic_handler(self) -> None:
        found: list[tuple[str, int]] = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                for index, handler in enumerate(node.handlers):
                    if not self._names_nie(handler.type):
                        continue
                    rel = path.relative_to(SRC).as_posix()
                    found.append((rel, index))
                    with self.subTest(file=rel, line=handler.lineno):
                        self.assertEqual(len(handler.body), 1)
                        self.assertIsInstance(handler.body[0], ast.Raise)
                        self.assertIsNone(handler.body[0].exc)
                        broad = [i for i, h in enumerate(node.handlers)
                                 if isinstance(h.type, ast.Name) and h.type.id in {"Exception", "BaseException"}]
                        self.assertTrue(all(index < i for i in broad))
        self.assertEqual([rel for rel, _ in found], ["compact_service/service.py"])


class TestAgentLoopSendState(OfflineCase):
    def test_unsupported_providers_call_chat_directly(self) -> None:
        for provider in (MinimalPlugin(), DeclaredUnsupported()):
            with self.subTest(provider=type(provider).__name__):
                result = self.run_loop(provider)
                self.assertEqual(provider.ledger, ["chat"])
                self.assertEqual(result.response_text, "done")

    def test_supported_provider_errors_never_fall_back(self) -> None:
        cases = [
            ("send then NotImplementedError", None, NotImplementedError("unsupported delta")),
            ("send then NotImplementedError subclass", None, SubNIE("plugin subclass")),
            ("empty chunk then NotImplementedError", emit(""), NotImplementedError("late")),
            ("tool/reasoning/usage-only state then NotImplementedError", None, NotImplementedError("tool delta")),
            ("text then NotImplementedError", emit("partial"), NotImplementedError("late")),
            ("payload mutated then NotImplementedError", mutate, NotImplementedError("mutated")),
        ]
        spies = self.spy_tools()
        for label, before, error in cases:
            with self.subTest(label):
                provider = StreamingPlugin(before_raise=before, error=error)
                with self.assertRaises(type(error)):
                    self.run_loop(provider)
                self.assertEqual(provider.ledger, ["chat_stream_response"])
        for spy in spies:
            spy.assert_not_called()

    def test_non_stream_never_calls_chat_stream_response(self) -> None:
        provider = StreamingPlugin()
        self.run_loop(provider, stream=False)
        self.assertEqual(provider.ledger, ["chat"])

    def test_builtin_structured_streaming_still_used_once(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            with self.subTest(provider=name):
                self.play(text_stream(name, "done"))
                result = self.run_loop(shared_provider(name))
                self.assertEqual(result.response_text, "done")
                self.assertEqual(len(self.sent), 1)
                self.assertTrue(json.loads(self.sent[0].content).get("stream"))
                self.assertEqual(self.sleeps, [])


class TestRealSdkSendState(OfflineCase):
    """Real SDK requests counted at the transport: after a send, NotImplementedError never resends."""

    def test_builtin_subclasses_raising_after_a_real_send(self) -> None:
        class OpenAISubclass(OpenAIProvider):
            def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
                super().chat_stream_response(messages, tools=tools, on_text_chunk=on_text_chunk, **kwargs)
                raise NotImplementedError("post-processing not supported")

        class AnthropicSubclass(AnthropicProvider):
            def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
                super().chat_stream_response(messages, tools=tools, on_text_chunk=on_text_chunk, **kwargs)
                raise NotImplementedError("post-processing not supported")

        openai_like = OpenAISubclass(api_key="test-dummy-key")
        openai_like._client = shared_provider("openai").client
        anthropic_like = AnthropicSubclass(api_key="test-dummy-key")
        anthropic_like.client = shared_provider("anthropic")._ensure_client()
        cases = [
            ("openai subclass", openai_like, stream_response([OA_TOOL_DELTA, oa_chunk({}, "tool_calls"), (None, "[DONE]")]),
             plain_reply("openai")),
            ("anthropic subclass", anthropic_like, stream_response([
                AN_START, AN_TOOL_START,
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                                   "usage": {"output_tokens": 7}}),
                ("message_stop", {"type": "message_stop"}),
            ]), plain_reply("anthropic")),
        ]
        spies = self.spy_tools()
        for label, provider, first, fallback in cases:
            with self.subTest(label):
                self.play(first, fallback)
                with self.assertRaises(NotImplementedError):
                    self.run_loop(provider)
                self.assertEqual(len(self.sent), 1)
                self.assertTrue(json.loads(self.sent[0].content).get("stream"))
        for spy in spies:
            spy.assert_not_called()

    def test_unmodified_builtin_with_a_post_send_not_implemented_error(self) -> None:
        """RT-builtin-inject: even an SDK/parser NotImplementedError after the send never resends."""
        firsts = {
            "openai": [OA_TOOL_DELTA],
            "glm": [OA_TOOL_DELTA],
            "anthropic": [AN_START, AN_TOOL_START],
            "minimax": [AN_START, AN_THINKING_START],
        }
        for name, events in firsts.items():
            with self.subTest(provider=name):
                self.play(failing_stream(events), plain_reply(name))
                with self.assertRaises(NotImplementedError):
                    self.run_loop(shared_provider(name))
                self.assertEqual(len(self.sent), 1)


def _make_repl(provider: Any, *, stream: bool, root: Path) -> ClawdREPL:
    """A REPL around the given provider, set up like tests/test_repl.py (TestREPL)."""
    config_dir = root / ".clawd"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps({
        "default_provider": "glm",
        "providers": {"glm": {"api_key": "test_api_key_12345678",
                              "base_url": "https://open.bigmodel.cn/api/paas/v4",
                              "default_model": "glm-4.5"}},
    }), encoding="utf-8")
    with patch("src.config.get_config_path", return_value=config_dir / "config.json"), \
         patch("src.repl.core.Session.create") as session_factory, \
         patch("src.repl.core.get_provider_class") as provider_class:
        session = Mock()
        session.conversation = Conversation()
        session_factory.return_value = session
        provider_class.return_value = Mock(return_value=provider)
        repl = ClawdREPL(provider_name="glm", stream=stream)
    repl.console.print = Mock()
    return repl


class TestReplDirectRouteSendState(OfflineCase):
    def setUp(self) -> None:
        super().setUp()
        self._start(patch("prompt_toolkit.output.defaults.create_output", return_value=DummyOutput()))
        self._start(patch("src.repl.core.load_permission_context",
                          side_effect=lambda workspace_root: ToolPermissionContext(
                              workspace_root=Path(workspace_root).resolve())))

    def _chat(self, repl: ClawdREPL, prompt: str = "你好呀") -> tuple[str, MagicMock, MagicMock, MagicMock]:
        with patch("src.repl.core.append_provider_usage") as ledger, \
             patch("src.repl.core.run_agent_loop") as agent_loop, \
             patch("rich.prompt.Prompt.ask") as ask, \
             patch("traceback.print_exc"):
            repl.chat(prompt)
        agent_loop.assert_not_called()
        ask.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in repl.console.print.call_args_list if c.args)
        return printed, ledger, agent_loop, ask

    @staticmethod
    def _history(repl: ClawdREPL) -> list[tuple[str, Any]]:
        return [(m.role, m.content) for m in repl.session.conversation.messages]

    def test_unsupported_provider_uses_one_chat_request_in_stream_mode(self) -> None:
        provider = MinimalPlugin()
        repl = _make_repl(provider, stream=True, root=self.tmp)
        printed, *_ = self._chat(repl)
        self.assertEqual(provider.ledger, ["chat"])
        self.assertEqual(self._history(repl), [("user", "你好呀"), ("assistant", "done")])
        self.assertEqual(printed.count("done"), 1)
        self.assertIn("did not return token counts", printed)

    def test_declared_unsupported_empty_chat_keeps_the_e7_outcome(self) -> None:
        provider = DeclaredUnsupported()
        provider.chat_text = ""
        repl = _make_repl(provider, stream=True, root=self.tmp)
        printed, ledger, *_ = self._chat(repl)
        self.assertEqual(provider.ledger, ["chat"])
        self.assertEqual(printed.count(EMPTY_NOTICE), 1)
        self.assertEqual(self._history(repl), [])  # E7 rollback of the unanswered turn
        ledger.assert_not_called()

    def test_supported_provider_errors_surface_without_legacy_retry(self) -> None:
        cases = [
            ("send then NotImplementedError", None, NotImplementedError("unsupported delta")),
            ("empty chunk then NotImplementedError", emit(""), NotImplementedError("late")),
            ("subclass with auth-looking text", None, SubNIE("HTTP 401 Unauthorized: invalid api key")),
        ]
        for label, before, error in cases:
            with self.subTest(label):
                provider = StreamingPlugin(before_raise=before, error=error)
                repl = _make_repl(provider, stream=True, root=self.tmp)
                printed, *_ = self._chat(repl)
                self.assertEqual(provider.ledger, ["chat_stream_response"])
                self.assertIn(NIE_HEADLINE, printed)
                self.assertIn(NO_RETRY, printed)
                for absent in (EMPTY_NOTICE, "Authentication Error", "401"):
                    self.assertNotIn(absent, printed)
                self.assertEqual(self._history(repl), [("user", "你好呀")])  # generic semantics: turn kept

    def test_unmodified_builtin_post_send_error_is_one_real_request(self) -> None:
        """Direct route, real SDK: reasoning/thinking arrives, then the stream fails; no legacy resend."""
        firsts = {"openai": [oa_chunk({"reasoning_content": "thinking"})],
                  "anthropic": [AN_START, AN_THINKING_START]}
        for name, events in firsts.items():
            with self.subTest(provider=name):
                repl = _make_repl(shared_provider(name), stream=True, root=self.tmp)
                self.play(failing_stream(events), text_stream(name, "legacy resend"))
                printed, *_ = self._chat(repl)
                self.assertEqual(len(self.sent), 1)
                self.assertIn(NIE_HEADLINE, printed)
                self.assertNotIn("legacy resend", printed)


class TestCompactionSendState(OfflineCase):
    def _compact(self, provider: Any) -> Any:
        conversation = Conversation()
        conversation.add_user_message("first question")
        conversation.add_assistant_message("first answer")
        return _run_without_event_loop(compact_conversation(conversation, provider, "plugin-model"))

    def test_missing_chat_async_uses_one_sync_chat(self) -> None:
        provider = MinimalPlugin()
        self._compact(provider)
        self.assertEqual(provider.ledger, ["chat"])

    def test_not_implemented_from_chat_async_never_falls_back_to_chat(self) -> None:
        cases = [
            ("sends then NotImplementedError", True, NotImplementedError("async unsupported after send")),
            ("NotImplementedError before sending", False, NotImplementedError("no async support")),
            ("NotImplementedError subclass", True, SubNIE("plugin subclass")),
        ]
        for label, sends_first, error in cases:
            with self.subTest(label):
                provider = AsyncPlugin(error=error, sends_first=sends_first)
                with self.assertRaises(NotImplementedError):
                    self._compact(provider)
                self.assertEqual(provider.ledger, ["chat_async"] if sends_first else [])

    def test_other_chat_async_errors_keep_the_out_of_scope_sync_fallback(self) -> None:
        """Characterization of a known, separately queued gap (not a statement that it is safe)."""
        provider = AsyncPlugin(error=RuntimeError("timeout after send"))
        self._compact(provider)
        self.assertEqual(provider.ledger, ["chat_async", "chat"])


class TestDiskLoadedPluginProviders(OfflineCase):
    """Real discovery, exact-hash pin, exec and registration, then the agent loop."""

    SOURCE = '''
from src.providers.base import BaseProvider, ChatResponse

def _reply():
    return ChatResponse(content="done", model="m", usage={}, finish_reason="stop")

class Minimal(BaseProvider):
    LEDGER = []
    def __init__(self, api_key, base_url=None, model=None):
        super().__init__(api_key, base_url, model or "m")
    def chat(self, messages, tools=None, **kwargs):
        type(self).LEDGER.append("chat")
        return _reply()
    def chat_stream(self, messages, tools=None, **kwargs):
        type(self).LEDGER.append("chat_stream")
        yield "text"
    def get_available_models(self):
        return ["m"]

class SendThenNie(Minimal):
    LEDGER = []
    def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
        type(self).LEDGER.append("chat_stream_response")
        raise NotImplementedError("unsupported delta")

class Declared(SendThenNie):
    LEDGER = []
    SUPPORTS_STRUCTURED_STREAMING = False

def _entry(name, cls):
    return {"name": name, "label": name, "provider_class": cls,
            "default_base_url": "http://127.0.0.1:11434/v1", "default_model": "m",
            "available_models": ["m"], "requires_api_key": False, "local_only": True}

PROVIDERS = [_entry("disk-minimal", Minimal), _entry("disk-send-nie", SendThenNie),
             _entry("disk-declared", Declared)]
'''

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(clear_plugin_providers)
        plugins = self.tmp / ".clawd" / "plugins"
        plugin_dir = plugins / "sendstate"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.json").write_text(json.dumps({
            "schema_version": 1, "name": "sendstate", "version": "1.0.0",
            "entrypoint": "plugin.py", "extensions": ["providers"]}), encoding="utf-8")
        (plugin_dir / "plugin.py").write_text(self.SOURCE, encoding="utf-8")
        policy = self.tmp / ".clawd" / "python_plugins.json"
        report = reconcile_python_plugins(plugin_root=plugins, operator_manifest=policy)
        digest = report["records"]["sendstate"]["artifact_sha256"]
        policy.write_text(json.dumps({"schema_version": 1, "plugins": {
            "sendstate": {"enabled": True, "artifact_sha256": digest}}}), encoding="utf-8")
        with patch("src.plugins.runtime.default_plugin_root", return_value=plugins), \
             patch("src.plugins.runtime.default_operator_manifest_path", return_value=policy):
            loaded = load_active_plugin_extensions()
        self.assertEqual(loaded.issues, [])
        self.assertEqual(register_plugin_provider_extensions(loaded), [])

    def test_disk_loaded_providers(self) -> None:
        cases = [("disk-minimal", None, ["chat"]),
                 ("disk-send-nie", NotImplementedError, ["chat_stream_response"]),
                 ("disk-declared", None, ["chat"])]
        for name, raises, expected in cases:
            with self.subTest(provider=name):
                cls = get_provider_class(name)
                provider = cls(api_key="", base_url=None, model=None)
                if raises is None:
                    self.run_loop(provider)
                else:
                    with self.assertRaises(raises):
                        self.run_loop(provider)
                self.assertEqual(cls.LEDGER, expected)


if __name__ == "__main__":
    unittest.main()
