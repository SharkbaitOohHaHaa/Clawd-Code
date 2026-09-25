"""Tool-call arguments are never replaced by invented input.

OpenAI-compatible providers (openai, deepseek, qwen, glm) keep invalid serialized arguments
(malformed, empty, whitespace-only, NaN/Infinity, non-string) as the raw text instead of {};
every default tool schema expects an object, so validation rejects it before permission
checks or run() and the model gets a normal tool error. Anthropic-family providers reject a
non-object tool input instead of coercing it, after the sealed output-limit check.

Offline only: credential, base-URL and proxy environment variables are scrubbed, sockets and
DNS are refused, sleeps are recorded, and every HTTP send is counted at
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
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai

from src.agent.conversation import Conversation
from src.providers import get_provider_class
from src.providers.base import IncompleteResponseError, InvalidToolInputError
from src.repl.core import _is_provider_authentication_error
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.permission_handler import PermissionResult
from src.tool_system.protocol import ToolResult

SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
OPENAI_FAMILY = ("openai", "deepseek", "qwen", "glm")
ANTHROPIC_FAMILY = ("anthropic", "minimax")
MESSAGES = [{"role": "user", "content": "offline probe"}]
TOOLS = [{"name": "Read", "description": "read", "input_schema": {"type": "object", "properties": {}}}]

# Invalid serialized arguments: kept as the raw text (never {}).
INVALID_ARGUMENTS = [
    ("malformed", '{"path":'),
    ("truncated but finish=tool_calls", '{"file_path": "a.txt", "limit": 1'),
    ("trailing garbage", '{"a": 1} junk'),
    ("two concatenated objects", '{"a": 1}{"b": 2}'),
    ("invalid escape (Windows path)", r'{"plan": "C:\Users\x"}'),
    ("empty string", ""),
    ("whitespace only", "   "),
    ("NaN", "NaN"),
    ("Infinity", "Infinity"),
    ("-Infinity", "-Infinity"),
    ("NaN inside an object", '{"n": NaN}'),
    ("overflow to infinity", '{"n": 1e999}'),
]
# Valid JSON: parsed exactly as today.
VALID_ARGUMENTS = [
    ("empty object", "{}", {}),
    ("object", '{"file_path": "a.txt"}', {"file_path": "a.txt"}),
    ("float", '{"t": 1.5}', {"t": 1.5}),
    ("duplicate keys (last wins)", '{"a": 1, "a": 2}', {"a": 2}),
    ("array", "[1, 2]", [1, 2]),
    ("string", '"text"', "text"),
    ("null", "null", None),
    ("number", "3", 3),
    ("boolean", "true", True),
]

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


def oa_tool_response(stream: bool, calls: list[tuple[str, Any]], finish: str = "tool_calls") -> httpx.Response:
    """OpenAI-format response whose tool calls carry exactly the given wire arguments."""
    if not stream:
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ]}
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
        })
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}
    events: list[Any] = []
    for i, (name, args) in enumerate(calls):
        head, tail = (args[: len(args) // 2], args[len(args) // 2:]) if isinstance(args, str) else (args, None)
        events.append((None, {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": i, "id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": head}}]}}]}))
        if tail:
            events.append((None, {**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
                {"index": i, "function": {"arguments": tail}}]}}]}))
    events.append((None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    events.append((None, "[DONE]"))
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


def oa_text_response(stream: bool, text: str) -> httpx.Response:
    if not stream:
        return httpx.Response(200, json={
            "id": "c2", "object": "chat.completion", "created": 0, "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        })
    base = {"id": "c2", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse([
        (None, {**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}),
        (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        (None, "[DONE]"),
    ]))


def an_tool_response(stream: bool, tool_input: Any, stop_reason: str = "tool_use") -> httpx.Response:
    """Anthropic-format response with one tool_use block whose input is exactly tool_input."""
    block = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": tool_input}
    if not stream:
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [block], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        })
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse([
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    ]))


class OfflineCase(unittest.TestCase):
    """No network, no real keys, no real sleeps; every SDK HTTP send is counted."""

    def setUp(self) -> None:
        clean_env = {k: v for k, v in os.environ.items() if not SCRUBBED_ENV.search(k)}
        self._start(patch.dict(os.environ, clean_env, clear=True))
        self.refused: list[str] = []
        for target in ("socket.create_connection", "socket.getaddrinfo", "socket.socket.connect"):
            self._start(patch(target, self._refuse(target)))
        self.sleeps: list[float] = []
        self._start(patch("time.sleep", side_effect=self.sleeps.append))
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
        # The anthropic SDK's stream snapshot warns when a (synthetic) tool input is a list,
        # which is exactly the shape these tests inject.
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

    def call(self, name: str, stream: bool) -> Any:
        provider = shared_provider(name)
        if stream:
            return provider.chat_stream_response(MESSAGES, tools=TOOLS, on_text_chunk=lambda _t: None)
        return provider.chat(MESSAGES, tools=TOOLS)


class TestOpenAICompatibleArgumentParsing(OfflineCase):
    """openai / deepseek / qwen / glm, chat() and chat_stream_response()."""

    def test_invalid_arguments_stay_raw_and_are_never_invented(self) -> None:
        for name in OPENAI_FAMILY:
            for stream in (False, True):
                for label, raw in INVALID_ARGUMENTS:
                    with self.subTest(provider=name, stream=stream, case=label):
                        self.play(oa_tool_response(stream, [("Read", raw)]))
                        tool_input = self.call(name, stream).tool_uses[0]["input"]
                        self.assertEqual(tool_input, raw)
                        self.assertIs(type(tool_input), str)
                        self.assertEqual(len(self.sent), 1)

    def test_valid_json_is_parsed_exactly_as_today(self) -> None:
        for name in OPENAI_FAMILY:
            for stream in (False, True):
                for label, raw, expected in VALID_ARGUMENTS:
                    with self.subTest(provider=name, stream=stream, case=label):
                        self.play(oa_tool_response(stream, [("Read", raw)]))
                        tool_input = self.call(name, stream).tool_uses[0]["input"]
                        self.assertEqual(tool_input, expected)
                        self.assertIs(type(tool_input), type(expected))
                        self.assertEqual(len(self.sent), 1)

    def test_non_string_wire_arguments_are_never_parsed_into_input(self) -> None:
        for name in OPENAI_FAMILY:
            for stream in (False, True):
                for label, wire in (("object", {"file_path": "a.txt"}), ("empty object", {}),
                                    ("number", 7), ("null", None)):
                    with self.subTest(provider=name, stream=stream, case=label):
                        self.play(oa_tool_response(stream, [("Read", wire)]))
                        tool_input = self.call(name, stream).tool_uses[0]["input"]
                        self.assertIs(type(tool_input), str)
                        self.assertEqual(len(self.sent), 1)

    def test_mixed_batch_keeps_each_call_honest(self) -> None:
        for name in OPENAI_FAMILY:
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    self.play(oa_tool_response(stream, [("Read", '{"file_path": "a.txt"}'), ("TaskList", '{"x":')]))
                    inputs = [use["input"] for use in self.call(name, stream).tool_uses]
                    self.assertEqual(inputs, [{"file_path": "a.txt"}, '{"x":'])

    def test_output_limit_still_wins_before_argument_parsing(self) -> None:
        for name in OPENAI_FAMILY:
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    self.play(oa_tool_response(stream, [("Read", '{"x":')], finish="length"))
                    with self.assertRaises(IncompleteResponseError) as caught:
                        self.call(name, stream)
                    self.assertEqual(caught.exception.reason, "tool_input_truncated")


class TestAnthropicFamilyToolInput(OfflineCase):
    """anthropic / minimax: a non-object tool input is rejected, never coerced."""

    NON_OBJECT_INPUTS = [("empty list", []), ("list of pairs", [["file_path", "a.txt"]]),
                         ("empty string", ""), ("string", "text"), ("null", None), ("number", 3)]

    def test_dict_input_is_unchanged(self) -> None:
        for name in ANTHROPIC_FAMILY:
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    self.play(an_tool_response(stream, {"file_path": "a.txt"}))
                    response = self.call(name, stream)
                    self.assertEqual(response.tool_uses[0]["input"], {"file_path": "a.txt"})
                    self.assertEqual(len(self.sent), 1)

    def test_non_object_input_fails_closed(self) -> None:
        for name in ANTHROPIC_FAMILY:
            for stream in (False, True):
                for label, value in self.NON_OBJECT_INPUTS:
                    if stream and label not in ("empty list", "list of pairs"):
                        continue  # a streamed block start only carries non-object input as a list here
                    with self.subTest(provider=name, stream=stream, case=label):
                        self.play(an_tool_response(stream, value))
                        with self.assertRaises(InvalidToolInputError) as caught:
                            self.call(name, stream)
                        self.assertEqual(len(self.sent), 1)
                        self.assertIsNone(caught.exception.__cause__)
                        self.assertIsNone(caught.exception.__context__)

    def test_max_tokens_with_non_object_input_stays_incomplete_response(self) -> None:
        """Phase A is frozen: the output-limit check wins over the new input check."""
        for name in ANTHROPIC_FAMILY:
            for stream in (False, True):
                for label, value in self.NON_OBJECT_INPUTS:
                    if stream and label not in ("empty list", "list of pairs"):
                        continue
                    with self.subTest(provider=name, stream=stream, case=label):
                        self.play(an_tool_response(stream, value, stop_reason="max_tokens"))
                        with self.assertRaises(IncompleteResponseError) as caught:
                            self.call(name, stream)
                        self.assertEqual(caught.exception.reason, "tool_input_truncated")
                        self.assertIs(caught.exception.tool_call_dropped, True)


class TestInvalidToolInputErrorContract(unittest.TestCase):
    def test_fixed_message_and_safe_class(self) -> None:
        error = InvalidToolInputError()
        self.assertIsInstance(error, RuntimeError)
        for forbidden in (NotImplementedError, ValueError, TypeError, anthropic.APIError, openai.APIError):
            self.assertNotIsInstance(error, forbidden)
        self.assertNotRegex(str(error), r"\d")
        self.assertFalse(_is_provider_authentication_error(error))
        self.assertEqual(vars(error), {})  # carries no tool input


class TestAgentLoopNeverRunsInvalidArguments(OfflineCase):
    """Real provider + real default registry + real agent loop: invalid arguments never run."""

    TOOLS_UNDER_TEST = {  # tool -> valid explicit input for the control
        "StructuredOutput": {},
        "ExitPlanMode": {},
        "ListMcpToolsTool": {},
        "ListMcpResourcesTool": {},
        "TaskList": {},
        "Read": {"file_path": "a.txt"},
    }
    BAD_ARGUMENTS = ['{"x":', "", "   ", "NaN"]

    def setUp(self) -> None:
        super().setUp()
        self._start(patch("src.token_estimation._load_tiktoken", return_value=None))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = build_default_registry()

    def run_loop(self, name: str, stream: bool) -> Any:
        conversation = Conversation()
        conversation.add_user_message("use the tool")
        return run_agent_loop(
            conversation=conversation, provider=shared_provider(name), tool_registry=self.registry,
            tool_context=ToolContext(workspace_root=Path(self._tmp.name)), stream=stream, verbose=False,
        )

    def _spy(self, tool_name: str) -> tuple[MagicMock, MagicMock, Any]:
        tool = self.registry.get(tool_name)
        run = MagicMock(return_value=ToolResult(name=tool_name, output={"ok": True}))
        permissions = MagicMock(return_value=PermissionResult.allow())
        patches = [patch.object(tool, "run", run)]
        if hasattr(tool, "check_permissions"):  # "allow"-policy tools have no permission check
            patches.append(patch.object(tool, "check_permissions", permissions))
        for patcher in patches:
            patcher.start()
        return run, permissions, patches

    def _second_request_messages(self) -> list[dict[str, Any]]:
        return json.loads(self.sent[1].content)["messages"]

    def test_invalid_arguments_never_execute_and_the_model_gets_a_tool_error(self) -> None:
        for name in ("openai", "glm"):
            for stream in (False, True):
                for tool_name in self.TOOLS_UNDER_TEST:
                    for raw in self.BAD_ARGUMENTS:
                        with self.subTest(provider=name, stream=stream, tool=tool_name, arguments=raw):
                            run, permissions, patches = self._spy(tool_name)
                            try:
                                self.play(oa_tool_response(stream, [(tool_name, raw)]),
                                          oa_text_response(stream, "done"))
                                result = self.run_loop(name, stream)
                            finally:
                                for patcher in patches:
                                    patcher.stop()
                            run.assert_not_called()
                            permissions.assert_not_called()  # validation rejected it first
                            self.assertEqual(result.response_text, "done")
                            self.assertEqual(len(self.sent), 2)  # the call, then a normal follow-up
                            messages = self._second_request_messages()
                            tool_messages = [m for m in messages if m.get("role") == "tool"]
                            self.assertIn("expected object, got string", tool_messages[0]["content"])
                            replayed = [m for m in messages if m.get("tool_calls")][0]["tool_calls"][0]
                            self.assertNotEqual(replayed["function"]["arguments"], "{}")

    def test_control_valid_arguments_run_once(self) -> None:
        for name in ("openai", "glm"):
            for stream in (False, True):
                for tool_name, valid in self.TOOLS_UNDER_TEST.items():
                    with self.subTest(provider=name, stream=stream, tool=tool_name):
                        run, _permissions, patches = self._spy(tool_name)
                        try:
                            self.play(oa_tool_response(stream, [(tool_name, json.dumps(valid))]),
                                      oa_text_response(stream, "done"))
                            self.run_loop(name, stream)
                        finally:
                            for patcher in patches:
                                patcher.stop()
                        run.assert_called_once()
                        self.assertEqual(run.call_args.args[0], valid)
                        self.assertEqual(len(self.sent), 2)

    def test_anthropic_family_non_object_input_never_runs(self) -> None:
        for name in ANTHROPIC_FAMILY:
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    run, permissions, patches = self._spy("Read")
                    try:
                        self.play(an_tool_response(stream, [["file_path", "a.txt"]]))
                        with self.assertRaises(InvalidToolInputError):
                            self.run_loop(name, stream)
                    finally:
                        for patcher in patches:
                            patcher.stop()
                    run.assert_not_called()
                    permissions.assert_not_called()
                    self.assertEqual(len(self.sent), 1)


if __name__ == "__main__":
    unittest.main()
