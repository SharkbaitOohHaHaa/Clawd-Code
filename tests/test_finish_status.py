"""Provider finish/stop status: a response the provider marks as not completed normally
is never returned as a success.

Built-in providers raise FinishStatusError for the abnormal finish values their own
documentation defines (Anthropic refusal / pause_turn / model_context_window_exceeded,
OpenAI and DeepSeek content_filter, DeepSeek insufficient_system_resource / aborted, GLM
sensitive / network_error / model_context_window_exceeded), and for any unrecognized value
on a response that carries tool calls. Unrecognized values on text-only responses, normal
values, the frozen output-limit values (max_tokens / length) and missing values are
unchanged.

Offline only: credential, base-URL and proxy environment variables are scrubbed, sockets
and DNS are refused, sleeps are recorded, and every HTTP send is counted at
``httpx.HTTPTransport.handle_request``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, Mock, patch

import anthropic
import httpx
import openai
from prompt_toolkit.output import DummyOutput

from src.agent import Conversation
from src.compact_service.service import compact_conversation
from src.providers import _BUILTIN_PROVIDER_NAMES, get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import (
    ChatResponse,
    FinishStatusError,
    IncompleteResponseError,
    InvalidToolInputError,
    _FINISH_PROFILES,
    classify_finish_status,
)
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.glm_provider import GLMProvider
from src.providers.minimax_provider import MinimaxProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.qwen_provider import QwenProvider
from src.repl import ClawdREPL
from src.repl.core import _is_provider_authentication_error
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.permission_handler import PermissionResult
from src.tool_system.permissions import ToolPermissionContext
from src.tool_system.protocol import ToolResult
from src.tool_system.tools.qwen_media import QwenMediaAnalyzeTool

SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
ANTHROPIC_FAMILY = ("anthropic", "minimax")
OPENAI_FAMILY = ("openai", "deepseek", "qwen", "glm")
MESSAGES = [{"role": "user", "content": "offline probe"}]
READ = {"name": "Read", "description": "read", "input_schema": {"type": "object", "properties": {}}}
MISSING = object()  # the finish / stop field is absent (non-stream) or never sent (stream)

# What each built-in provider documents (exact, case-sensitive). Deliberately repeated here
# rather than read from the production map, so a change to either shows up.
NORMAL = {
    "anthropic": ("end_turn", "stop_sequence", "tool_use"),
    "minimax": ("end_turn", "tool_use"),
    "openai": ("stop", "tool_calls", "function_call"),
    "deepseek": ("stop", "tool_calls"),
    "qwen": ("stop", "tool_calls"),
    "glm": ("stop", "tool_calls"),
}
OUTPUT_LIMIT = {"anthropic": "max_tokens", "minimax": "max_tokens", "openai": "length",
                "deepseek": "length", "qwen": "length", "glm": "length"}
ABNORMAL = {
    "anthropic": {"refusal": "blocked", "pause_turn": "paused",
                  "model_context_window_exceeded": "context_window"},
    "minimax": {},
    "openai": {"content_filter": "blocked"},
    "deepseek": {"content_filter": "blocked", "insufficient_system_resource": "interrupted",
                 "aborted": "interrupted"},
    "qwen": {},
    "glm": {"sensitive": "blocked", "network_error": "interrupted",
            "model_context_window_exceeded": "context_window"},
}
# Unrecognized for that provider: custom, case variants, other providers' values, and
# third-party values that are deliberately NOT mapped in this first cut.
UNKNOWN = {
    "anthropic": ("custom_x", "compaction", "REFUSAL", "stop", "content_filter", "error"),
    "minimax": ("custom_x", "refusal", "stop_sequence", "pause_turn", "MAX_TOKENS"),
    "openai": ("custom_x", "sensitive", "error", "model_length", "eos_token", "STOP", "LENGTH"),
    "deepseek": ("custom_x", "function_call", "network_error", "end_turn"),
    "qwen": ("custom_x", "content_filter", "sensitive", "Stop"),
    "glm": ("custom_x", "content_filter", "aborted", "refusal", "length "),
}
CATEGORY_MESSAGES = {
    "blocked": "The provider blocked this response with a safety or content-filter status; "
               "the response was not used.",
    "interrupted": "The provider reported that generation was interrupted; the response is "
                   "incomplete and was not used.",
    "context_window": "The provider stopped at the model context window; the response is "
                      "incomplete and was not used.",
    "paused": "The provider paused this turn; the response is incomplete and was not used.",
    "unrecognized": "The provider ended a tool-call response with an unrecognized finish "
                    "status; the response was not used.",
}

_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each SDK client costs an SSL context); built only
    inside OfflineCase tests, so under the scrubbed environment."""
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


def _stream_response(events: list[Any]) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


# --- Anthropic wire (anthropic, minimax) ------------------------------------------------
# tools: (name, input). A dict input is streamed as input_json_delta; any other input is put
# on the block start as-is (the only way to deliver a non-object input). server=True adds a
# server_tool_use block; open_tool=(name, partial_json) leaves one tool block unfinished.

def an_reply(stream: bool, *, text: str = "", tools: tuple = (), stop: Any = "end_turn",
             server: bool = False, open_tool: tuple | None = None,
             stop_details: dict | None = None) -> httpx.Response:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    if server:
        blocks.append({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                       "input": {"query": "x"}})
    for i, (tool_name, tool_input) in enumerate(tools):
        blocks.append({"type": "tool_use", "id": f"toolu_{i}", "name": tool_name, "input": tool_input})
    if not stream:
        body: dict[str, Any] = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": blocks, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
        if stop is not MISSING:
            body["stop_reason"] = stop
        if stop_details is not None:
            body["stop_details"] = stop_details
        return httpx.Response(200, json=body)
    events: list[Any] = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}})]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start: dict[str, Any] = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"][:4]},
                      {"type": "text_delta", "text": block["text"][4:]}]
        elif isinstance(block["input"], dict):
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        else:
            start, deltas = block, []
        events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                               "content_block": start}))
        for delta in deltas:
            events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": delta}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    if open_tool is not None:
        index = len(blocks)
        events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                               "content_block": {"type": "tool_use", "id": "toolu_open",
                                                                 "name": open_tool[0], "input": {}}}))
        events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                               "delta": {"type": "input_json_delta",
                                                         "partial_json": open_tool[1]}}))
    if stop is not MISSING:
        delta: dict[str, Any] = {"stop_reason": stop, "stop_sequence": None}
        if stop_details is not None:
            delta["stop_details"] = stop_details
        events.append(("message_delta", {"type": "message_delta", "delta": delta,
                                         "usage": {"output_tokens": 7}}))
        events.append(("message_stop", {"type": "message_stop"}))
    return _stream_response(events)


# --- OpenAI wire (openai, deepseek, qwen, glm) --------------------------------------------
# tools: (name, arguments); a non-string is JSON-encoded. finish: one value, or for a stream
# a list of values sent on successive chunks (MISSING: no finish value at all).

def oa_reply(stream: bool, *, text: str = "", tools: tuple = (), finish: Any = "stop",
             refusal: str | None = None) -> httpx.Response:
    calls = [(tool_name, args if isinstance(args, str) else json.dumps(args))
             for tool_name, args in tools]
    usage = {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}
    if not stream:
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if refusal is not None:
            message["refusal"] = refusal
        if calls:
            message["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                for i, (n, a) in enumerate(calls)
            ]
        choice: dict[str, Any] = {"index": 0, "message": message}
        if finish is not MISSING:
            choice["finish_reason"] = finish
        return httpx.Response(200, json={"id": "c1", "object": "chat.completion", "created": 0,
                                         "model": "gpt-test", "choices": [choice], "usage": usage})
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}

    def chunk(delta: dict[str, Any], finish_value: Any = None) -> tuple[None, dict[str, Any]]:
        return (None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_value}]})

    events: list[Any] = []
    if text:
        events += [chunk({"role": "assistant", "content": text[:4]}), chunk({"content": text[4:]})]
    if refusal is not None:
        events.append(chunk({"refusal": refusal}))
    for i, (tool_name, args) in enumerate(calls):
        events.append(chunk({"tool_calls": [{"index": i, "id": f"call_{i}", "type": "function",
                                             "function": {"name": tool_name, "arguments": args[:3]}}]}))
        events.append(chunk({"tool_calls": [{"index": i, "function": {"arguments": args[3:]}}]}))
    finishes = [] if finish is MISSING else (finish if isinstance(finish, list) else [finish])
    for value in finishes:
        events.append(chunk({}, value))
    events.append((None, {**base, "choices": [], "usage": usage}))
    events.append((None, "[DONE]"))
    return _stream_response(events)


def reply(name: str, stream: bool, *, text: str = "", tools: tuple = (), finish: Any = None,
          **extra: Any) -> httpx.Response:
    """A response in the provider's own wire format; finish=None means its normal stop."""
    if name in ANTHROPIC_FAMILY:
        stop = ("tool_use" if tools else "end_turn") if finish is None else finish
        return an_reply(stream, text=text, tools=tools, stop=stop, **extra)
    value = ("tool_calls" if tools else "stop") if finish is None else finish
    return oa_reply(stream, text=text, tools=tools, finish=value, **extra)


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
        # The anthropic SDK's stream snapshot warns when a (synthetic) tool input is a list.
        self._start(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

    def _start(self, patcher: Any) -> None:
        if isinstance(patcher, warnings.catch_warnings):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
            return
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

    def call(self, name: str, stream: bool, chunks: list[str] | None = None,
             provider: Any = None) -> Any:
        provider = provider or shared_provider(name)
        if stream:
            sink = chunks if chunks is not None else []
            return provider.chat_stream_response(MESSAGES, tools=[READ], on_text_chunk=sink.append)
        return provider.chat(MESSAGES, tools=[READ])


class TestClassifier(unittest.TestCase):
    """T1: the single source of truth, value by value (pure, no provider)."""

    def test_the_map_is_exactly_the_documented_built_in_contracts(self) -> None:
        self.assertEqual(set(_FINISH_PROFILES), set(NORMAL))
        for profile, spec in _FINISH_PROFILES.items():
            with self.subTest(profile=profile):
                self.assertEqual(spec.normal, frozenset(NORMAL[profile]))
                self.assertEqual(spec.output_limit, OUTPUT_LIMIT[profile])
                self.assertEqual(dict(spec.abnormal), ABNORMAL[profile])
                self.assertEqual(spec.context_reset,
                                 frozenset({"refusal"}) if profile == "anthropic" else frozenset())

    def test_each_builtin_provider_uses_its_own_profile(self) -> None:
        self.assertEqual(set(_BUILTIN_PROVIDER_NAMES), set(NORMAL))
        classes = {"anthropic": AnthropicProvider, "minimax": MinimaxProvider,
                   "openai": OpenAIProvider, "deepseek": DeepSeekProvider,
                   "qwen": QwenProvider, "glm": GLMProvider}
        for name in _BUILTIN_PROVIDER_NAMES:
            with self.subTest(provider=name):
                self.assertIs(get_provider_class(name), classes[name])
                self.assertEqual(get_provider_class(name).FINISH_STATUS_PROFILE, name)
        # The generic OpenAI-compatible base implements the OpenAI Chat Completions contract.
        self.assertEqual(OpenAICompatibleProvider.FINISH_STATUS_PROFILE, "openai")

    def test_normal_output_limit_and_missing_values_pass_with_or_without_tools(self) -> None:
        for profile in NORMAL:
            for value in (*NORMAL[profile], OUTPUT_LIMIT[profile], None, ""):
                for tools in (False, True):
                    with self.subTest(profile=profile, value=value, tools=tools):
                        self.assertIsNone(classify_finish_status(profile, [value], has_tool_calls=tools))
            with self.subTest(profile=profile, value="no values"):
                self.assertIsNone(classify_finish_status(profile, [], has_tool_calls=True))

    def test_documented_abnormal_values_are_classified_with_or_without_tools(self) -> None:
        for profile, mapping in ABNORMAL.items():
            for value, category in mapping.items():
                for tools in (False, True):
                    with self.subTest(profile=profile, value=value, tools=tools):
                        status = classify_finish_status(profile, [value], has_tool_calls=tools)
                        self.assertEqual(status.category, category)
                        self.assertEqual(status.finish_value, value)
                        self.assertIs(status.context_reset, profile == "anthropic" and value == "refusal")

    def test_unrecognized_values_gain_no_tool_authority_but_text_passes(self) -> None:
        odd_values: tuple[Any, ...] = (123, 1.5, {"k": 1}, ["stop"], True)
        for profile, values in UNKNOWN.items():
            for value in (*values, *odd_values):
                with self.subTest(profile=profile, value=value):
                    self.assertIsNone(classify_finish_status(profile, [value], has_tool_calls=False))
                    status = classify_finish_status(profile, [value], has_tool_calls=True)
                    self.assertEqual(status.category, "unrecognized")
                    self.assertEqual(status.finish_value, value)
                    self.assertFalse(status.context_reset)

    def test_stream_value_order(self) -> None:
        cases = [
            ("openai", ["content_filter", "stop"], False, "blocked"),   # abnormal anywhere wins
            ("glm", ["stop", "sensitive"], False, "blocked"),
            ("glm", ["sensitive", "tool_calls"], True, "blocked"),
            ("deepseek", ["aborted", "stop"], False, "interrupted"),
            ("deepseek", ["custom_x", "content_filter"], True, "blocked"),
            ("openai", ["custom_x", "tool_calls"], True, None),         # unknown judged on the final value
            ("qwen", ["null", "tool_calls"], True, None),
            ("openai", ["tool_calls", "custom_x"], True, "unrecognized"),
            ("openai", ["tool_calls", "custom_x"], False, None),
            ("openai", ["length", "stop"], True, None),  # frozen output-limit gap (characterization)
        ]
        for profile, values, tools, expected in cases:
            with self.subTest(profile=profile, values=values, tools=tools):
                status = classify_finish_status(profile, values, has_tool_calls=tools)
                self.assertEqual(status.category if status else None, expected)


class TestFinishStatusErrorContract(unittest.TestCase):
    """T2: nothing can fall back, retry, rewrap or auth-classify it; no provider text in str()."""

    def test_fixed_message_and_safe_class(self) -> None:
        not_bases = (NotImplementedError, IncompleteResponseError, InvalidToolInputError,
                     anthropic.APIError, openai.APIError, ValueError, TypeError, AttributeError,
                     KeyError, IndexError)
        for category, message in CATEGORY_MESSAGES.items():
            with self.subTest(category=category):
                error = FinishStatusError(
                    category, finish_value="HTTP 401 Unauthorized",
                    partial_text="401 Unauthorized: invalid api key",
                    partial_usage={"input_tokens": 401}, tool_call_dropped=True,
                )
                self.assertIsInstance(error, RuntimeError)
                for base in not_bases:
                    self.assertNotIsInstance(error, base)
                self.assertEqual(str(error), message)
                self.assertNotRegex(str(error), r"\d")
                self.assertFalse(hasattr(error, "status_code"))
                self.assertFalse(hasattr(error, "response"))
                self.assertFalse(hasattr(error, "tool_uses"))
                self.assertFalse(_is_provider_authentication_error(error))
                self.assertEqual(error.category, category)
                self.assertEqual(error.finish_value, "HTTP 401 Unauthorized")
                self.assertEqual(error.partial_usage, {"input_tokens": 401})
                self.assertTrue(error.tool_call_dropped)
                self.assertFalse(error.context_reset)

    def test_display_value_is_sanitized(self) -> None:
        cases = [
            ("content_filter", "content_filter"),
            ("model_context_window_exceeded", "model_context_window_exceeded"),
            ("HTTP 401 Unauthorized", "unprintable value"),
            ("[/] [red]x[/red]", "unprintable value"),
            ("esc\x1b[31m", "unprintable value"),
            ("x" * 41, "unprintable value"),
            ("", "unprintable value"),
            (123, "unprintable value"),
            (None, "unprintable value"),
        ]
        for value, shown in cases:
            with self.subTest(value=value):
                error = FinishStatusError("unrecognized", finish_value=value)
                self.assertEqual(error.safe_finish_value, shown)
                self.assertEqual(error.finish_value, value)


class TestProviderFinishStatus(OfflineCase):
    """T3: all six built-ins, streamed and not."""

    SHAPES = [  # label, text, tools
        ("text", "Partial answer: the first", ()),
        ("tool", "", (("Read", {"file_path": "a.txt"}),)),
        ("text + tool", "Let me check.", (("Read", {"file_path": "a.txt"}),)),
        ("two tools", "", (("Read", {"file_path": "a.txt"}), ("Glob", {"pattern": "*"}))),
        ("empty", "", ()),
    ]

    def _expect_status(self, name: str, stream: bool, response: httpx.Response,
                       category: str, value: Any, text: str, dropped: bool) -> FinishStatusError:
        self.play(response)
        chunks: list[str] = []
        with self.assertRaises(FinishStatusError) as caught:
            self.call(name, stream, chunks)
        error = caught.exception
        self.assertEqual(error.category, category)
        self.assertEqual(error.finish_value, value)
        self.assertEqual(error.partial_text, text)
        self.assertIs(error.tool_call_dropped, dropped)
        self.assertEqual(int(error.partial_usage.get("input_tokens", 0) or 0),
                         11 if name in ANTHROPIC_FAMILY else 9)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sleeps, [])
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        if stream:
            self.assertEqual("".join(chunks), text)  # already delivered live before the raise
        return error

    def test_documented_abnormal_values_raise_for_every_shape_and_mode(self) -> None:
        count = 0
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for value, category in ABNORMAL[name].items():
                for stream in (False, True):
                    for label, text, tools in self.SHAPES:
                        with self.subTest(provider=name, value=value, stream=stream, shape=label):
                            error = self._expect_status(
                                name, stream, reply(name, stream, text=text, tools=tools, finish=value),
                                category, value, text, bool(tools))
                            self.assertIs(error.context_reset, name == "anthropic" and value == "refusal")
                            count += 1
        self.assertEqual(count, 10 * 2 * len(self.SHAPES))

    def test_unrecognized_value_blocks_tool_responses_only(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for value in UNKNOWN[name]:
                for stream in (False, True):
                    with self.subTest(provider=name, value=value, stream=stream, shape="text"):
                        self.play(reply(name, stream, text="Done here.", finish=value))
                        response = self.call(name, stream)
                        self.assertIsInstance(response, ChatResponse)
                        self.assertEqual(response.content, "Done here.")
                        self.assertEqual(response.finish_reason, value)
                        self.assertEqual(len(self.sent), 1)
                    with self.subTest(provider=name, value=value, stream=stream, shape="text + tool"):
                        self._expect_status(
                            name, stream,
                            reply(name, stream, text="Let me check.",
                                  tools=(("Read", {"file_path": "a.txt"}),), finish=value),
                            "unrecognized", value, "Let me check.", True)

    def test_normal_values_are_unchanged(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for value in NORMAL[name]:
                for stream in (False, True):
                    for label, text, tools in self.SHAPES:
                        with self.subTest(provider=name, value=value, stream=stream, shape=label):
                            self.play(reply(name, stream, text=text, tools=tools, finish=value))
                            response = self.call(name, stream)
                            self.assertIsInstance(response, ChatResponse)
                            self.assertEqual(response.content, text)
                            self.assertEqual(response.finish_reason, value)
                            expected = [dict(tool_input) for _n, tool_input in tools] or None
                            got = [tu["input"] for tu in response.tool_uses] if response.tool_uses else None
                            self.assertEqual(got, expected)
                            self.assertEqual(len(self.sent), 1)

    def test_output_limit_still_wins_as_incomplete_response(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for stream in (False, True):
                for label, text, tools in self.SHAPES:
                    with self.subTest(provider=name, stream=stream, shape=label):
                        self.play(reply(name, stream, text=text, tools=tools, finish=OUTPUT_LIMIT[name]))
                        with self.assertRaises(IncompleteResponseError) as caught:
                            self.call(name, stream)
                        self.assertIs(caught.exception.tool_call_dropped, bool(tools))
                        self.assertEqual(len(self.sent), 1)

    def test_missing_finish_values_are_unchanged_even_with_tool_calls(self) -> None:
        """Null / missing / never-sent finish values stay with Phase B; tool calls still returned."""
        tool = (("Read", {"file_path": "a.txt"}),)
        cases = [(name, stream, finish)
                 for name in sorted(_BUILTIN_PROVIDER_NAMES)
                 for stream in (False, True)
                 for finish in (MISSING, None)
                 if not (stream and finish is None)]
        cases += [(name, True, "") for name in OPENAI_FAMILY]
        for name, stream, finish in cases:
            label = "missing" if finish is MISSING else repr(finish)
            with self.subTest(provider=name, stream=stream, finish=label):
                self.play(reply(name, stream, text="Reading.", tools=tool, finish=finish))
                response = self.call(name, stream)
                self.assertIsInstance(response, ChatResponse)
                self.assertEqual(response.tool_uses[0]["input"], {"file_path": "a.txt"})
                self.assertEqual(len(self.sent), 1)

    def test_openai_compatible_subclass_inherits_the_gate(self) -> None:
        class PluginProvider(OpenAICompatibleProvider):
            def _create_client(self) -> Any:
                return shared_provider("openai").client

            def get_available_models(self) -> list[str]:
                return ["plugin-model"]

        plugin = PluginProvider(api_key="test-dummy-key", model="plugin-model")
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.play(oa_reply(stream, text="Partial", finish="content_filter"))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call("openai", stream, provider=plugin)
                self.assertEqual(caught.exception.category, "blocked")
                self.assertEqual(len(self.sent), 1)


class TestPrecedence(OfflineCase):
    """T4: the sealed output-limit and tool-input checks still come first."""

    def test_anthropic_family_non_object_input_still_raises_invalid_tool_input(self) -> None:
        for name in ANTHROPIC_FAMILY:
            for stop in ("refusal", "pause_turn", "custom_x", "tool_use"):
                for stream in (False, True):
                    with self.subTest(provider=name, stop=stop, stream=stream):
                        self.play(an_reply(stream, tools=(("Read", [["file_path", "a.txt"]]),), stop=stop))
                        with self.assertRaises(InvalidToolInputError):
                            self.call(name, stream)
                        self.assertEqual(len(self.sent), 1)
            for stream in (False, True):
                with self.subTest(provider=name, stop="max_tokens", stream=stream):
                    self.play(an_reply(stream, tools=(("Read", [["file_path", "a.txt"]]),), stop="max_tokens"))
                    with self.assertRaises(IncompleteResponseError):
                        self.call(name, stream)

    def test_openai_family_malformed_arguments_with_abnormal_finish(self) -> None:
        cases = [("openai", "content_filter", "blocked"), ("deepseek", "aborted", "interrupted"),
                 ("glm", "sensitive", "blocked"), ("qwen", "custom_x", "unrecognized")]
        for name, finish, category in cases:
            for stream in (False, True):
                with self.subTest(provider=name, finish=finish, stream=stream):
                    self.play(oa_reply(stream, tools=(("Read", '{"file_path":'),), finish=finish))
                    with self.assertRaises(FinishStatusError) as caught:
                        self.call(name, stream)
                    self.assertEqual(caught.exception.category, category)
                    self.assertTrue(caught.exception.tool_call_dropped)
                with self.subTest(provider=name, finish="length", stream=stream):
                    self.play(oa_reply(stream, tools=(("Read", '{"file_path":'),), finish="length"))
                    with self.assertRaises(IncompleteResponseError):
                        self.call(name, stream)


class TestOpenAIStreamOrdering(OfflineCase):
    """T5: every finish value a stream reports counts, not only the last one."""

    def test_finish_value_sequences(self) -> None:
        tool = (("Read", {"file_path": "a.txt"}),)
        cases = [  # provider, finishes, tools, expected category (None: returned unchanged)
            ("openai", ["content_filter", "stop"], (), "blocked"),
            ("glm", ["stop", "sensitive"], (), "blocked"),
            ("glm", ["sensitive", ""], (), "blocked"),
            ("deepseek", ["aborted", "stop"], (), "interrupted"),
            ("glm", ["network_error", "tool_calls"], tool, "interrupted"),
            ("openai", ["custom_x", "tool_calls"], tool, None),
            ("qwen", ["null", "tool_calls"], tool, None),
            ("openai", ["tool_calls", "custom_x"], tool, "unrecognized"),
            ("glm", ["tool_calls", "custom_x"], (), None),
        ]
        for name, finishes, tools, expected in cases:
            with self.subTest(provider=name, finishes=finishes, tools=bool(tools)):
                self.play(oa_reply(True, text="Partial", tools=tools, finish=finishes))
                if expected is None:
                    response = self.call(name, True)
                    self.assertIsInstance(response, ChatResponse)
                    self.assertEqual(response.finish_reason, finishes[-1] or finishes[0])
                    self.assertEqual(bool(response.tool_uses), bool(tools))
                else:
                    with self.assertRaises(FinishStatusError) as caught:
                        self.call(name, True)
                    self.assertEqual(caught.exception.category, expected)
                self.assertEqual(len(self.sent), 1)

    def test_output_limit_still_wins_after_an_abnormal_value(self) -> None:
        for name, value in (("openai", "content_filter"), ("glm", "sensitive"), ("deepseek", "aborted")):
            with self.subTest(provider=name, value=value):
                self.play(oa_reply(True, text="Hello wor", tools=(("Read", {"file_path": "a.txt"}),),
                                   finish=[value, "length"]))
                with self.assertRaises(IncompleteResponseError) as caught:
                    self.call(name, True)
                self.assertTrue(caught.exception.tool_call_dropped)
                self.assertEqual(len(self.sent), 1)

    def test_falsy_non_string_values_are_unrecognized_in_both_modes(self) -> None:
        tool = (("Read", {"file_path": "a.txt"}),)
        for value in (0, False):
            for stream in (False, True):
                with self.subTest(value=value, stream=stream, shape="tool"):
                    finish = ["tool_calls", value] if stream else value
                    self.play(oa_reply(stream, text="Checking", tools=tool, finish=finish))
                    with self.assertRaises(FinishStatusError) as caught:
                        self.call("openai", stream)
                    self.assertEqual(caught.exception.category, "unrecognized")
                    self.assertEqual(caught.exception.finish_value, value)
                with self.subTest(value=value, stream=stream, shape="text"):
                    self.play(oa_reply(stream, text="Done", finish=[value] if stream else value))
                    self.assertEqual(self.call("openai", stream).content, "Done")

    def test_nameless_tool_fragment_counts_as_a_dropped_tool_call(self) -> None:
        """As for the output limit: tool-call deltas that never got a name are still reported."""
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}
        events = [
            (None, {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1}'}}]}}]}),
            (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}),
            (None, "[DONE]"),
        ]
        for name in ("openai", "deepseek"):
            with self.subTest(provider=name):
                self.play(_stream_response(events))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call(name, True)
                self.assertEqual(caught.exception.category, "blocked")
                self.assertTrue(caught.exception.tool_call_dropped)

    def test_output_limit_followed_by_stop_is_a_frozen_known_gap(self) -> None:
        """Characterization only (Phase A is frozen): the last value wins for length."""
        for name in ("openai", "glm"):
            with self.subTest(provider=name):
                self.play(oa_reply(True, text="Hello wor", finish=["length", "stop"]))
                response = self.call(name, True)
                self.assertIsInstance(response, ChatResponse)
                self.assertEqual(response.finish_reason, "stop")


class TestAnthropicShapes(OfflineCase):
    """T6: the documented refusal / pause shapes."""

    def test_streamed_refusal_with_an_open_tool_block_never_returns_the_tool(self) -> None:
        for name, category in (("anthropic", "blocked"), ("minimax", "unrecognized")):
            with self.subTest(provider=name):
                self.play(an_reply(True, text="I'll read it.", stop="refusal",
                                   open_tool=("Read", '{"file_path": "a.txt", "limit": 5'),
                                   stop_details={"type": "refusal", "category": None, "explanation": None}))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call(name, True)
                self.assertEqual(caught.exception.category, category)
                self.assertTrue(caught.exception.tool_call_dropped)
                self.assertEqual(len(self.sent), 1)

    def test_refusal_explanation_is_not_exposed(self) -> None:
        details = {"type": "refusal", "category": "cyber", "explanation": "EXPLANATION-SENTINEL [red]"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.play(an_reply(stream, text="Hello..", stop="refusal", stop_details=details))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call("anthropic", stream)
                error = caught.exception
                self.assertFalse(hasattr(error, "explanation"))
                self.assertNotIn("EXPLANATION-SENTINEL", str(error) + repr(vars(error)))

    def test_pause_turn_with_a_server_tool_block_is_paused_not_resumed(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = None
                self.play(an_reply(stream, text="Searching", server=True, stop="pause_turn"))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call("anthropic", stream)
                error = caught.exception
                self.assertEqual(error.category, "paused")
                self.assertFalse(error.tool_call_dropped)  # a server tool is never a client call
                self.assertEqual(len(self.sent), 1)

    def test_compaction_is_only_an_unrecognized_value(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream, shape="text"):
                self.play(an_reply(stream, text="Summary", stop="compaction"))
                self.assertEqual(self.call("anthropic", stream).finish_reason, "compaction")
            with self.subTest(stream=stream, shape="tool"):
                self.play(an_reply(stream, tools=(("Read", {"file_path": "a.txt"}),), stop="compaction"))
                with self.assertRaises(FinishStatusError) as caught:
                    self.call("anthropic", stream)
                self.assertEqual(caught.exception.category, "unrecognized")


class TestOpenAIRefusalFieldCharacterization(OfflineCase):
    """Characterization only (deferred): message.refusal / delta.refusal is not read today."""

    def test_refusal_field_is_ignored(self) -> None:
        for name in ("openai", "deepseek"):
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream, finish="stop"):
                    self.play(oa_reply(stream, refusal="I can't help with that.", finish="stop"))
                    response = self.call(name, stream)
                    self.assertEqual(response.content, "")
                    self.assertEqual(response.finish_reason, "stop")
                with self.subTest(provider=name, stream=stream, finish="content_filter"):
                    self.play(oa_reply(stream, refusal="I can't help with that.", finish="content_filter"))
                    with self.assertRaises(FinishStatusError) as caught:
                        self.call(name, stream)
                    self.assertNotIn("help", caught.exception.partial_text)


class TestAgentLoopNeverRunsToolsOnFinishStatus(OfflineCase):
    """T7: real provider + real default registry + real agent loop."""

    CASES = [  # provider, finish value
        ("anthropic", "refusal"), ("anthropic", "pause_turn"),
        ("anthropic", "model_context_window_exceeded"), ("anthropic", "custom_x"),
        ("minimax", "refusal"), ("openai", "content_filter"), ("openai", "custom_x"),
        ("deepseek", "aborted"), ("deepseek", "insufficient_system_resource"),
        ("qwen", "content_filter"), ("glm", "sensitive"), ("glm", "network_error"),
        ("glm", "model_context_window_exceeded"),
    ]

    def setUp(self) -> None:
        super().setUp()
        self.registry = build_default_registry()
        (self.tmp / "a.txt").write_text("hello\n", encoding="utf-8")

    def run_loop(self, name: str, stream: bool) -> Any:
        conversation = Conversation()
        conversation.add_user_message("use the tools")
        return run_agent_loop(
            conversation=conversation, provider=shared_provider(name), tool_registry=self.registry,
            tool_context=ToolContext(workspace_root=self.tmp), stream=stream, verbose=False,
        )

    def _spies(self, *tool_names: str) -> tuple[list[MagicMock], list[Any]]:
        mocks: list[MagicMock] = []
        patches: list[Any] = []
        for tool_name in tool_names:
            tool = self.registry.get(tool_name)
            run = MagicMock(return_value=ToolResult(name=tool_name, output={"ok": True}))
            mocks.append(run)
            patches.append(patch.object(tool, "run", run))
            if hasattr(tool, "check_permissions"):
                permissions = MagicMock(return_value=PermissionResult.allow())
                mocks.append(permissions)
                patches.append(patch.object(tool, "check_permissions", permissions))
        for patcher in patches:
            patcher.start()
        return mocks, patches

    def test_no_permission_check_no_run_no_continuation(self) -> None:
        tools = (("Read", {"file_path": "a.txt"}), ("TaskList", {}))
        for name, value in self.CASES:
            for stream in (False, True):
                with self.subTest(provider=name, value=value, stream=stream):
                    mocks, patches = self._spies("Read", "TaskList")
                    try:
                        self.play(reply(name, stream, text="Checking.", tools=tools, finish=value),
                                  reply(name, stream, text="done"))
                        with self.assertRaises(FinishStatusError) as caught:
                            self.run_loop(name, stream)
                    finally:
                        for patcher in patches:
                            patcher.stop()
                    for mock in mocks:
                        mock.assert_not_called()
                    self.assertTrue(caught.exception.tool_call_dropped)
                    self.assertEqual(len(self.sent), 1)  # the follow-up is never requested
                    self.assertEqual(self.sleeps, [])

    def test_control_normal_tool_turn_runs_and_continues(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    mocks, patches = self._spies("Read")
                    try:
                        self.play(reply(name, stream, tools=(("Read", {"file_path": "a.txt"}),)),
                                  reply(name, stream, text="done"))
                        result = self.run_loop(name, stream)
                    finally:
                        for patcher in patches:
                            patcher.stop()
                    mocks[0].assert_called_once()
                    self.assertEqual(result.response_text, "done")
                    self.assertEqual(len(self.sent), 2)

    def test_unrecognized_text_only_answer_is_unchanged(self) -> None:
        for name in ("anthropic", "openai", "glm"):
            with self.subTest(provider=name):
                self.play(reply(name, False, text="All done.", finish="custom_x"))
                result = self.run_loop(name, False)
                self.assertEqual(result.response_text, "All done.")
                self.assertEqual(len(self.sent), 1)


def _make_repl(provider: Any, *, stream: bool, root: Path) -> ClawdREPL:
    """A REPL around a real provider, set up like tests/test_repl.py (TestREPL)."""
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


class TestReplWithRealProviders(OfflineCase):
    """T8 (real adapters): what the user sees and what the conversation keeps."""

    def setUp(self) -> None:
        super().setUp()
        self._start(patch("prompt_toolkit.output.defaults.create_output", return_value=DummyOutput()))
        self._start(patch("src.repl.core.load_permission_context",
                          side_effect=lambda workspace_root: ToolPermissionContext(
                              workspace_root=Path(workspace_root).resolve())))

    def _chat(self, repl: ClawdREPL, prompt: str) -> tuple[MagicMock, MagicMock]:
        with patch("src.repl.core.append_provider_usage") as ledger, \
             patch.object(ClawdREPL, "_record_and_print_task_usage") as record_usage, \
             patch.object(ClawdREPL, "_confirm_high_token_agent_request", return_value=True), \
             patch.object(repl.tool_registry, "dispatch") as dispatch, \
             patch("rich.prompt.Prompt.ask") as ask, \
             patch("traceback.print_exc") as print_exc:
            repl.chat(prompt)
        ledger.assert_not_called()
        record_usage.assert_not_called()
        ask.assert_not_called()
        print_exc.assert_not_called()
        return dispatch, record_usage

    @staticmethod
    def _printed(repl: ClawdREPL) -> str:
        return " ".join(str(call.args[0]) for call in repl.console.print.call_args_list if call.args)

    @staticmethod
    def _history(repl: ClawdREPL) -> list[tuple[str, Any]]:
        return [(m.role, m.content) for m in repl.session.conversation.messages]

    def test_anthropic_refusal_resets_a_clean_turn(self) -> None:
        for stream in (False, True):
            for prompt in ("你好呀", "Fix this file"):  # direct route, then agent route
                with self.subTest(stream=stream, prompt=prompt):
                    repl = _make_repl(shared_provider("anthropic"), stream=stream, root=self.tmp)
                    self.play(an_reply(stream, text="Hello..", tools=(("Read", {"file_path": "a.txt"}),),
                                       stop="refusal"))
                    dispatch, _ = self._chat(repl, prompt)
                    dispatch.assert_not_called()
                    self.assertEqual(len(self.sent), 1)
                    self.assertEqual(self._history(repl), [])
                    printed = self._printed(repl)
                    self.assertIn("Response blocked: the provider stopped this response with a safety or "
                                  "content-filter status (refusal).", printed)
                    self.assertIn("The refused message was removed from the conversation", printed)
                    self.assertIn("A tool call in this response was not run.", printed)
                    # Only a stream already showed it live (in two chunks); nothing prints it after.
                    self.assertEqual(printed.count("Hell"), 1 if stream else 0)
                    self.assertNotIn("Error:", printed)

    def test_glm_sensitive_keeps_the_turn_with_a_marker_only(self) -> None:
        marker = ("[Response blocked: the provider stopped this response with a safety or "
                  "content-filter status. No partial output was kept.]")
        for stream in (False, True):
            with self.subTest(stream=stream):
                repl = _make_repl(shared_provider("glm"), stream=stream, root=self.tmp)
                self.play(oa_reply(stream, text="BLOCKED-PARTIAL text", finish="sensitive"))
                self._chat(repl, "你好呀")
                self.assertEqual(self._history(repl), [("user", "你好呀"), ("assistant", marker)])
                printed = self._printed(repl)
                self.assertEqual(printed.count("BLOC"), 1 if stream else 0)  # streamed live only
                self.assertEqual("The streamed text above was blocked" in printed, stream)
                self.assertIn("(sensitive).", printed)
                self.assertEqual(len(self.sent), 1)

    def test_deepseek_interrupted_agent_turn_keeps_marked_partial(self) -> None:
        marker = ("[Incomplete response: the provider reported that generation was interrupted. "
                  "A tool call was not run. The text below is partial and is not a complete answer.]"
                  "\n\nChecking the file")
        for stream in (False, True):
            with self.subTest(stream=stream):
                repl = _make_repl(shared_provider("deepseek"), stream=stream, root=self.tmp)
                self.play(oa_reply(stream, text="Checking the file", tools=(("Read", {"file_path": "a.txt"}),),
                                   finish="aborted"),
                          oa_reply(stream, text="done"))
                dispatch, _ = self._chat(repl, "Fix this file")
                dispatch.assert_not_called()
                self.assertEqual(len(self.sent), 1)
                self.assertEqual(self._history(repl), [("user", "Fix this file"), ("assistant", marker)])
                self.assertIn("Clawd did not issue a fallback retry.", self._printed(repl))

    def test_minimax_refusal_is_unrecognized_and_never_resets_the_turn(self) -> None:
        """MiniMax does not document refusal: it is an unknown value, not Anthropic's reset rule."""
        marker = ("[Unverified response: the provider ended a tool-call response with a finish status "
                  "Clawd does not recognize. A tool call was not run. The text below was not confirmed "
                  "as complete.]\n\nLet me look")
        for stream in (False, True):
            with self.subTest(stream=stream):
                repl = _make_repl(shared_provider("minimax"), stream=stream, root=self.tmp)
                self.play(an_reply(stream, text="Let me look", tools=(("Read", {"file_path": "a.txt"}),),
                                   stop="refusal"))
                dispatch, _ = self._chat(repl, "Fix this file")
                dispatch.assert_not_called()
                self.assertEqual(self._history(repl), [("user", "Fix this file"), ("assistant", marker)])
                printed = self._printed(repl)
                self.assertNotIn("The refused message was removed", printed)
                self.assertNotIn("The streamed text above was blocked", printed)
                self.assertEqual(len(self.sent), 1)

    def test_openai_unrecognized_tool_response_is_unverified(self) -> None:
        repl = _make_repl(shared_provider("openai"), stream=False, root=self.tmp)
        self.play(oa_reply(False, text="Let me look", tools=(("Read", {"file_path": "a.txt"}),),
                           finish="eos_token"))
        dispatch, _ = self._chat(repl, "Fix this file")
        dispatch.assert_not_called()
        printed = self._printed(repl)
        self.assertIn("Unverified response:", printed)
        self.assertIn("(eos_token).", printed)
        self.assertIn("marked as unverified", printed)
        self.assertEqual(self._history(repl)[-1][1].split("\n\n")[1], "Let me look")


def _run_without_event_loop(coroutine: Any) -> Any:
    """Drive a coroutine that never suspends (asyncio's Windows loop needs a local socket
    pair, which the offline guard refuses)."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    coroutine.close()
    raise AssertionError("the coroutine suspended; it was expected to finish without awaiting")


class TestNestedCallers(OfflineCase):
    """T9: QwenMediaAnalyze and /compact see the error once, with no retry and no extra send."""

    def _qwen_env(self) -> Any:
        return patch.dict(os.environ, {"QWEN_MEDIA_LIVE_ENABLED": "1", "DASHSCOPE_API_KEY": "dummy-key"})

    def test_qwen_media_real_provider_one_send_no_retry(self) -> None:
        image = self.tmp / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"safe-test-image")
        tool = QwenMediaAnalyzeTool()
        with self._qwen_env(), patch("src.tool_system.tools.qwen_media.append_provider_event") as event:
            self.play(oa_reply(False, text="partial", tools=(("Read", {"file_path": "a"}),),
                               finish="custom_x"))
            with self.assertRaises(FinishStatusError) as caught:
                tool.run({"source": str(image)}, ToolContext(workspace_root=self.tmp))
        self.assertEqual(caught.exception.category, "unrecognized")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sleeps, [])
        attempts = [c.kwargs for c in event.call_args_list if c.kwargs.get("stage") == "provider_attempt"]
        self.assertEqual([(a["attempt"], a["status"]) for a in attempts], [(1, "started"), (1, "failed")])
        self.assertFalse(attempts[1]["retryable"])

    def test_qwen_media_error_becomes_a_normal_tool_error_in_the_agent_loop(self) -> None:
        image = self.tmp / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"safe-test-image")
        registry = build_default_registry()
        qwen_tool = registry.get("QwenMediaAnalyze")
        error = FinishStatusError("blocked", finish_value="content_filter", partial_text="x")
        conversation = Conversation()
        conversation.add_user_message("describe image.png")
        with self._qwen_env(), \
             patch("src.tool_system.tools.qwen_media.append_provider_event"), \
             patch.object(QwenProvider, "chat", side_effect=error) as qwen_chat, \
             patch.object(qwen_tool, "check_permissions", return_value=PermissionResult.allow()):
            self.play(oa_reply(False, tools=(("QwenMediaAnalyze", {"source": str(image)}),)),
                      oa_reply(False, text="done"))
            result = run_agent_loop(conversation=conversation, provider=shared_provider("openai"),
                                    tool_registry=registry, tool_context=ToolContext(workspace_root=self.tmp),
                                    stream=False, verbose=False)
        self.assertEqual(qwen_chat.call_count, 1)
        self.assertEqual(self.sleeps, [])
        self.assertEqual(result.response_text, "done")
        tool_messages = [m for m in json.loads(self.sent[1].content)["messages"] if m.get("role") == "tool"]
        self.assertEqual(tool_messages[0]["content"], f"Error: {CATEGORY_MESSAGES['blocked']}")

    def test_compact_uses_the_local_fallback_summary_after_one_send(self) -> None:
        cases = [("openai", "content_filter"), ("glm", "sensitive"), ("anthropic", "refusal"),
                 ("deepseek", "aborted")]
        for name, value in cases:
            with self.subTest(provider=name, value=value):
                provider = shared_provider(name)
                self.assertFalse(hasattr(provider, "chat_async"))  # today: no async send at all
                conversation = Conversation()
                conversation.add_user_message("first question")
                conversation.add_assistant_message("first answer")
                self.play(reply(name, False, text="FILTERED-SUMMARY", finish=value))
                result = _run_without_event_loop(compact_conversation(conversation, provider, "model-x"))
                self.assertEqual(len(self.sent), 1)
                self.assertNotIn("FILTERED-SUMMARY", result.summary_text)
                self.assertIn("first question", result.summary_text)


if __name__ == "__main__":
    unittest.main()
