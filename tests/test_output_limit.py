"""Provider-declared output-limit truncation (Phase A).

When a provider says generation stopped at its output limit (Anthropic ``stop_reason ==
"max_tokens"``, OpenAI-compatible ``finish_reason == "length"``), the built-in providers
raise IncompleteResponseError, streamed or not: the response is never complete and a
truncated tool call never runs. Missing-terminal detection is deliberately NOT enforced.

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
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import httpx

from src.agent.conversation import Conversation
from src.providers import _BUILTIN_PROVIDER_NAMES, get_provider_class
from src.providers.base import ChatResponse, IncompleteResponseError
from src.repl.core import _is_provider_authentication_error
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec

SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
ANTHROPIC_FAMILY = {"anthropic", "minimax"}
MESSAGES = [{"role": "user", "content": "offline probe"}]
READ_SCHEMA = {
    "name": "Read",
    "description": "read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}

_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each SDK client costs an SSL context). Only built
    inside OfflineCase tests, so construction happens under the scrubbed environment."""
    if name not in _SHARED:
        _SHARED[name] = get_provider_class(name)(
            api_key=FAKE_KEYS.get(name, "test-dummy-key"), base_url=None, model=None
        )
    return _SHARED[name]


def _sse(events: list[tuple[str | None, Any]]) -> bytes:
    lines: list[str] = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        lines.append("data: " + (data if isinstance(data, str) else json.dumps(data)))
        lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- Anthropic wire format (anthropic + minimax) ---------------------------------------

def _an_block(kind: str, value: str) -> dict[str, Any]:
    if kind == "text":
        return {"type": "text", "text": value}
    if kind == "thinking":
        return {"type": "thinking", "thinking": value, "signature": "sig"}
    return {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": json.loads(value)}


def an_json(blocks: list[tuple[str, str]], stop_reason: str) -> httpx.Response:
    body = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [_an_block(kind, value) for kind, value in blocks],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    return httpx.Response(200, json=body)


def an_stream(blocks: list[tuple[str, str]], stop_reason: str | None) -> httpx.Response:
    events: list[tuple[str | None, Any]] = [("message_start", {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1}},
    })]
    for index, (kind, value) in enumerate(blocks):
        if kind == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": value}
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": value}
        else:
            start = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}
            delta = {"type": "input_json_delta", "partial_json": value}
        events += [
            ("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
    if stop_reason is not None:  # None = the stream simply ends (missing terminal; Phase B)
        events += [
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": {"output_tokens": 7}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


# --- OpenAI wire format (openai, deepseek, qwen, glm) ----------------------------------

def oa_json(blocks: list[tuple[str, str]], finish: str) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": "".join(v for k, v in blocks if k == "text")}
    reasoning = "".join(v for k, v in blocks if k == "thinking")
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_args = [v for k, v in blocks if k == "tool"]
    if tool_args:
        message["tool_calls"] = [{"id": "call_1", "type": "function",
                                  "function": {"name": "Read", "arguments": tool_args[0]}}]
    body = {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
    }
    return httpx.Response(200, json=body)


def oa_stream(blocks: list[tuple[str, str]], finish: str | None) -> httpx.Response:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}
    events: list[tuple[str | None, Any]] = []
    for kind, value in blocks:
        if kind == "text":
            delta: dict[str, Any] = {"content": value}
        elif kind == "thinking":
            delta = {"reasoning_content": value}
        else:
            delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "Read", "arguments": value}}]}
        events.append((None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}))
    if finish is not None:  # None = no finish_reason ever (missing terminal; Phase B)
        events.append((None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    events.append((None, "[DONE]"))
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


def response_for(name: str, stream: bool, blocks: list[tuple[str, str]], anthropic_stop: str | None,
                 openai_finish: str | None) -> httpx.Response:
    if name in ANTHROPIC_FAMILY:
        return an_stream(blocks, anthropic_stop) if stream else an_json(blocks, anthropic_stop or "end_turn")
    return oa_stream(blocks, openai_finish) if stream else oa_json(blocks, openai_finish or "stop")


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

    def call(self, name: str, stream: bool, chunks: list[str]) -> Any:
        provider = shared_provider(name)
        if stream:
            return provider.chat_stream_response(MESSAGES, tools=[READ_SCHEMA], on_text_chunk=chunks.append)
        return provider.chat(MESSAGES, tools=[READ_SCHEMA])


class TestProviderDeclaredOutputLimit(OfflineCase):
    """All six built-ins, streamed and not: max_tokens / length is never a complete answer."""

    def test_output_limit_raises_incomplete_for_every_builtin_and_mode(self) -> None:
        cases = [
            ("text at limit", [("text", "Hello wor")], "output_limit", "Hello wor", False),
            ("tool call at limit", [("text", "Reading."), ("tool", '{"file_path": "a.t')],
             "tool_input_truncated", "Reading.", True),
            ("reasoning only at limit", [("thinking", "E7-REASONING-SENTINEL")], "output_limit", "", False),
        ]
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            for stream in (False, True):
                for label, blocks, reason, partial, dropped in cases:
                    with self.subTest(provider=name, stream=stream, case=label):
                        if not stream and name in ANTHROPIC_FAMILY and blocks[-1][0] == "tool":
                            blocks = [blocks[0], ("tool", "{}")]  # non-stream tool input arrives parsed
                        self.play(response_for(name, stream, blocks, "max_tokens", "length"))
                        chunks: list[str] = []
                        with self.assertRaises(IncompleteResponseError) as caught:
                            self.call(name, stream, chunks)
                        error = caught.exception
                        self.assertEqual(error.reason, reason)
                        self.assertEqual(error.partial_text, partial)
                        self.assertIs(error.tool_call_dropped, dropped)
                        self.assertFalse(hasattr(error, "tool_uses"))
                        self.assertNotIn("E7-REASONING-SENTINEL", error.partial_text + str(error))
                        self.assertEqual(len(self.sent), 1)
                        self.assertEqual(self.sleeps, [])
                        if stream:
                            # Partial text is never suppressed: it was delivered before the raise.
                            self.assertEqual("".join(chunks), partial)
                        expected_input = 11 if name in ANTHROPIC_FAMILY else (0 if stream else 9)
                        self.assertEqual(int(error.partial_usage.get("input_tokens", 0) or 0), expected_input)
                        self.assertIsNone(error.__cause__)
                        self.assertIsNone(error.__context__)

    def test_complete_responses_are_unchanged(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            family_stop = ("end_turn", "tool_use") if name in ANTHROPIC_FAMILY else ("stop", "tool_calls")
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream, case="text"):
                    self.play(response_for(name, stream, [("text", "Hello world.")], family_stop[0], family_stop[0]))
                    response = self.call(name, stream, [])
                    self.assertIsInstance(response, ChatResponse)
                    self.assertEqual(response.content, "Hello world.")
                    self.assertEqual(response.finish_reason, family_stop[0])
                    self.assertEqual(len(self.sent), 1)
                with self.subTest(provider=name, stream=stream, case="tool call"):
                    self.play(response_for(name, stream, [("tool", '{"file_path": "a.txt"}')],
                                           family_stop[1], family_stop[1]))
                    response = self.call(name, stream, [])
                    self.assertEqual(response.tool_uses[0]["name"], "Read")
                    self.assertEqual(response.tool_uses[0]["input"], {"file_path": "a.txt"})
                    self.assertEqual(len(self.sent), 1)

    def test_missing_terminal_signal_is_not_enforced_in_phase_a(self) -> None:
        """Deferred (Phase B): a stream without any finish / stop reason still returns."""
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            with self.subTest(provider=name):
                self.play(response_for(name, True, [("text", "Hello wor")], None, None))
                chunks: list[str] = []
                response = self.call(name, True, chunks)
                self.assertIsInstance(response, ChatResponse)
                self.assertEqual(response.content, "Hello wor")
                self.assertEqual(len(self.sent), 1)


class TestIncompleteResponseErrorContract(unittest.TestCase):
    def test_message_is_fixed_and_cannot_look_like_an_auth_failure(self) -> None:
        for reason in ("output_limit", "tool_input_truncated"):
            with self.subTest(reason=reason):
                error = IncompleteResponseError(
                    reason, partial_text="HTTP 401 Unauthorized: invalid api key",
                    partial_usage={"input_tokens": 401}, tool_call_dropped=True,
                )
                self.assertNotIsInstance(error, NotImplementedError)
                self.assertIsInstance(error, RuntimeError)
                self.assertNotRegex(str(error), r"\d")
                self.assertNotIn("401", str(error))
                self.assertFalse(hasattr(error, "status_code"))
                self.assertFalse(hasattr(error, "response"))
                self.assertFalse(_is_provider_authentication_error(error))


class TestAgentLoopNeverRunsTruncatedToolCall(OfflineCase):
    """Real provider + real agent loop: a truncated tool call never executes; one send."""

    def setUp(self) -> None:
        super().setUp()
        self._start(patch("src.token_estimation._load_tiktoken", return_value=None))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.calls: list[dict[str, Any]] = []
        calls = self.calls

        class SpyRead:
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="Read", description="read a file", input_schema=READ_SCHEMA["input_schema"],
                    permission_policy="allow",
                )

            def run(self, tool_input: dict[str, Any], context: Any) -> ToolResult:
                calls.append(dict(tool_input))
                return ToolResult(name="Read", output={"content": "ok"})

        self.registry = ToolRegistry([SpyRead()])

    def run_loop(self, name: str, stream: bool) -> Any:
        conversation = Conversation()
        conversation.add_user_message("read a.txt")
        return run_agent_loop(
            conversation=conversation, provider=shared_provider(name), tool_registry=self.registry,
            tool_context=ToolContext(workspace_root=Path(self._tmp.name)), stream=stream, verbose=False,
        )

    def test_truncated_tool_call_never_runs(self) -> None:
        for name in ("anthropic", "openai"):
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    self.calls.clear()
                    args = '{"file_path": "a.t' if stream or name != "anthropic" else "{}"
                    self.play(response_for(name, stream, [("tool", args)], "max_tokens", "length"))
                    with self.assertRaises(IncompleteResponseError) as caught:
                        self.run_loop(name, stream)
                    self.assertEqual(caught.exception.reason, "tool_input_truncated")
                    self.assertEqual(self.calls, [])
                    self.assertEqual(len(self.sent), 1)

    def test_control_complete_tool_call_runs_once(self) -> None:
        for name in ("anthropic", "openai"):
            for stream in (False, True):
                with self.subTest(provider=name, stream=stream):
                    self.calls.clear()
                    tool_stop = ("tool_use", "tool_calls")
                    self.play(
                        response_for(name, stream, [("tool", '{"file_path": "a.txt"}')], *tool_stop),
                        response_for(name, stream, [("text", "done")], "end_turn", "stop"),
                    )
                    result = self.run_loop(name, stream)
                    self.assertEqual(self.calls, [{"file_path": "a.txt"}])
                    self.assertEqual(result.response_text, "done")
                    self.assertEqual(len(self.sent), 2)


if __name__ == "__main__":
    unittest.main()
