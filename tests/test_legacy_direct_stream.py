"""Stream-mode direct replies from providers without structured streaming.

A provider that does not support chat_stream_response (a direct BaseProvider plugin, or a
built-in subclass / instance declaring SUPPORTS_STRUCTURED_STREAMING = False) gets exactly one
chat() request on the direct REPL route, even with the stream setting on; the method is chosen
before sending. The legacy text-only chat_stream() is never called: it reports no finish
status and no usage, so the output-limit and finish-status checks could not run on it. The
complete reply is shown once, after it is stored.

The report helpers print a partial response unless response text was actually shown live
(stream_started), never on the strength of the stream setting alone.

Offline only: credential, base-URL and proxy environment variables are scrubbed, sockets and
DNS are refused, sleeps are recorded, and every SDK HTTP send is counted at
``httpx.HTTPTransport.handle_request``. Each scripted reply is served in the provider's own
wire format for the mode that was actually requested (stream or not).
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import httpx
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown

from src.agent import Conversation
from src.providers import get_provider_class
from src.providers.base import (
    BaseProvider,
    ChatResponse,
    FinishStatusError,
    IncompleteResponseError,
    supports_structured_streaming,
)
from src.repl import ClawdREPL
from src.tool_system.agent_loop import build_agent_preflight, run_agent_loop
from src.tool_system.permissions import ToolPermissionContext

SRC = Path(__file__).resolve().parents[1] / "src"
SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
ANTHROPIC_FAMILY = ("anthropic", "minimax")
DIRECT_PROMPT = "你好呀"
AGENT_PROMPT = "Fix this file"
PARTIAL = "Partial answer"
FULL = "Complete answer"
TRAP = "TRAP-RESEND"
EMPTY_NOTICE = "The provider returned an empty response. Clawd did not issue a fallback retry."
NO_RETRY = "Clawd did not issue a fallback retry."
STREAMED_BLOCKED = "The streamed text above was blocked"
TEXT_ABOVE = "The text above is partial."
LIMIT = "Incomplete response: the provider stopped at its output limit."
NOTES = {
    "blocked": "Response blocked: the provider stopped this response with a safety or "
               "content-filter status",
    "interrupted": "Incomplete response: the provider reported that generation was interrupted",
    "context_window": "Incomplete response: the provider stopped at the model's context window",
    "paused": "Incomplete response: the provider paused this turn",
    "unrecognized": "Unverified response: the provider ended a tool-call response with a finish "
                    "status Clawd does not recognize",
}
# (input, output, total) as each wire below reports it.
USAGE = {name: (11, 7, 18) if name in ANTHROPIC_FAMILY else (9, 5, 14)
         for name in ("anthropic", "minimax", "openai", "deepseek", "glm", "qwen")}
TOOL = (("Read", {"file_path": "a.txt"}),)

_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each SDK client costs an SSL context)."""
    if name not in _SHARED:
        _SHARED[name] = get_provider_class(name)(
            api_key=FAKE_KEYS.get(name, "test-dummy-key"), base_url=None, model=None
        )
    return _SHARED[name]


def declared_unsupported(name: str, *, per_instance: bool = False) -> Any:
    """A real built-in provider declaring no structured streaming (class or instance level).

    The class-level variant records which provider method Clawd entered.
    """
    base = get_provider_class(name)
    if per_instance:
        provider = base(api_key=FAKE_KEYS.get(name, "test-dummy-key"), base_url=None, model=None)
        provider.SUPPORTS_STRUCTURED_STREAMING = False
    else:

        class Declared(base):  # type: ignore[valid-type, misc]
            SUPPORTS_STRUCTURED_STREAMING = False

            def chat(self, *args: Any, **kwargs: Any) -> Any:
                self.ledger.append("chat")
                return super().chat(*args, **kwargs)

            def chat_stream(self, *args: Any, **kwargs: Any) -> Any:
                self.ledger.append("chat_stream")
                return super().chat_stream(*args, **kwargs)

            def chat_stream_response(self, *args: Any, **kwargs: Any) -> Any:
                self.ledger.append("chat_stream_response")
                return super().chat_stream_response(*args, **kwargs)

        provider = Declared(api_key=FAKE_KEYS.get(name, "test-dummy-key"), base_url=None, model=None)
        provider.ledger = []
    shared = shared_provider(name)
    if name in ANTHROPIC_FAMILY:
        provider.client = shared._ensure_client()
    else:
        provider._client = shared.client
    return provider


# --- wire formats -----------------------------------------------------------------------------

def _sse(events: list[Any]) -> httpx.Response:
    lines: list[str] = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        lines.append("data: " + (data if isinstance(data, str) else json.dumps(data)))
        lines.append("")
    body = ("\n".join(lines) + "\n").encode("utf-8")
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def oa_wire(stream: bool, *, text: str, tools: tuple, finish: str) -> httpx.Response:
    usage = {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}
    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": tool_name, "arguments": json.dumps(tool_input)}}
             for i, (tool_name, tool_input) in enumerate(tools)]
    if not stream:
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = calls
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test"}

    def chunk(delta: dict[str, Any], finish_value: Any = None, **extra: Any) -> tuple[None, dict]:
        return (None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_value}],
                       **extra})

    events: list[Any] = []
    if text:
        events += [chunk({"role": "assistant", "content": text[:4]}), chunk({"content": text[4:]})]
    events += [chunk({"tool_calls": [{"index": i, **call}]}) for i, call in enumerate(calls)]
    # Usage rides on the final choice chunk (Clawd never asks for a usage-only chunk).
    events += [chunk({}, finish, usage=usage), (None, "[DONE]")]
    return _sse(events)


def an_wire(stream: bool, *, text: str, tools: tuple, finish: str) -> httpx.Response:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks += [{"type": "tool_use", "id": f"toolu_{i}", "name": tool_name, "input": tool_input}
               for i, (tool_name, tool_input) in enumerate(tools)]
    if not stream:
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": blocks, "stop_reason": finish, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7}})
    events: list[Any] = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}})]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start: dict[str, Any] = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"][:4]},
                      {"type": "text_delta", "text": block["text"][4:]}]
        else:
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                               "content_block": start}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
                   for delta in deltas]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events += [("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": finish, "stop_sequence": None},
                                  "usage": {"output_tokens": 7}}),
               ("message_stop", {"type": "message_stop"})]
    return _sse(events)


Responder = Callable[[httpx.Request], httpx.Response]


def wire(name: str, *, text: str = "", tools: tuple = (), finish: str | None = None) -> Responder:
    """The provider's own wire format, for whichever mode (stream or not) was requested."""

    def respond(request: httpx.Request) -> httpx.Response:
        stream = bool(json.loads(request.content).get("stream"))
        if name in ANTHROPIC_FAMILY:
            return an_wire(stream, text=text, tools=tools,
                           finish=finish or ("tool_use" if tools else "end_turn"))
        return oa_wire(stream, text=text, tools=tools, finish=finish or ("tool_calls" if tools else "stop"))

    return respond


# --- synthetic plugin providers (their own send ledger) ---------------------------------------

def plugin_reply(text: str = "Plugin reply", **overrides: Any) -> ChatResponse:
    fields: dict[str, Any] = dict(content=text, model="plugin-model",
                                  usage={"input_tokens": 4, "output_tokens": 3}, finish_reason="stop")
    fields.update(overrides)
    return ChatResponse(**fields)


def raising(error: BaseException) -> Callable[..., Any]:
    def action(*args: Any) -> Any:
        raise error
    return action


class ChatPlugin(BaseProvider):
    """A direct BaseProvider plugin without structured streaming; chat() runs a scripted action."""

    def __init__(self, api_key: str = "k", base_url: str | None = None, model: str | None = None,
                 *, action: Callable[[], Any] | None = None):
        super().__init__(api_key, base_url, model or "plugin-model")
        self.ledger: list[str] = []
        self.action = action or plugin_reply

    def chat(self, messages, tools=None, **kwargs):
        self.ledger.append("chat")
        return self.action()

    def chat_stream(self, messages, tools=None, **kwargs):
        self.ledger.append("chat_stream")
        yield "LEGACY-TEXT"

    def get_available_models(self):
        return ["plugin-model"]


class StructuredPlugin(ChatPlugin):
    """Supports structured streaming; the scripted action receives the chunk callback."""

    def __init__(self, *, stream_action: Callable[[Any], Any]):
        super().__init__()
        self.stream_action = stream_action

    def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):
        self.ledger.append("chat_stream_response")
        return self.stream_action(on_text_chunk)


def emit_then(error: BaseException) -> Callable[[Any], Any]:
    def action(on_text_chunk: Any) -> Any:
        on_text_chunk(PARTIAL)
        raise error
    return action


def limit_error() -> IncompleteResponseError:
    return IncompleteResponseError("output_limit", partial_text=PARTIAL,
                                   partial_usage={"input_tokens": 11, "output_tokens": 7})


def finish_error(category: str, value: str) -> FinishStatusError:
    return FinishStatusError(category, finish_value=value, partial_text=PARTIAL,
                             partial_usage={"input_tokens": 11, "output_tokens": 7})


# --- shared offline harness -------------------------------------------------------------------

@dataclass
class Turn:
    printed: str
    lines: list[str]
    markdown: bool
    history: list[tuple[str, Any]]
    usage: list[tuple[int, int, int]]
    traced: int
    agent_route: bool
    stream_requests: list[bool]  # one entry per SDK HTTP send: was it a stream request?
    unconsumed: int  # scripted responses left (the trap must remain)


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


class OfflineReplCase(unittest.TestCase):
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
        self.script: list[Any] = []

        def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            request.read()
            self.sent.append(request)
            if not self.script:
                raise AssertionError("unexpected extra provider request")
            scripted = self.script.pop(0)
            response = scripted(request) if callable(scripted) else scripted
            response.request = request
            return response

        self._start(patch.object(httpx.HTTPTransport, "handle_request", handle_request))
        self._start(patch("prompt_toolkit.output.defaults.create_output", return_value=DummyOutput()))
        self._start(patch("src.repl.core.load_permission_context",
                          side_effect=lambda workspace_root: ToolPermissionContext(
                              workspace_root=Path(workspace_root).resolve())))

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

    def play(self, *responses: Any) -> None:
        self.sent.clear()
        self.script = list(responses)

    def turn(self, provider: Any, *, stream: bool, prompt: str = DIRECT_PROMPT,
             console_error: Callable[..., Any] | None = None) -> Turn:
        repl = _make_repl(provider, stream=stream, root=self.tmp)
        if console_error is not None:
            repl.console.print.side_effect = console_error
        with patch("src.repl.core.append_provider_usage") as ledger, \
             patch("src.repl.core.build_agent_preflight", wraps=build_agent_preflight) as preflight, \
             patch("src.repl.core.run_agent_loop", wraps=run_agent_loop) as agent_loop, \
             patch.object(repl.tool_registry, "dispatch") as dispatch, \
             patch.object(ClawdREPL, "_confirm_high_token_agent_request", return_value=True), \
             patch("rich.prompt.Prompt.ask") as ask, \
             patch("traceback.print_exc") as print_exc:
            repl.chat(prompt)
        dispatch.assert_not_called()
        ask.assert_not_called()
        self.assertEqual(preflight.called, agent_loop.called)
        calls = repl.console.print.call_args_list
        lines = [str(call.args[0]) if call.args else "" for call in calls]
        return Turn(
            printed=" ".join(line for call, line in zip(calls, lines) if call.args),
            lines=lines,
            markdown=any(call.args and isinstance(call.args[0], Markdown) for call in calls),
            history=[(m.role, m.content) for m in repl.session.conversation.messages],
            usage=[(c.args[0]["input_tokens"], c.args[0]["output_tokens"], c.args[0]["total_tokens"])
                   for c in ledger.call_args_list],
            traced=print_exc.call_count,
            agent_route=agent_loop.called,
            stream_requests=[bool(json.loads(request.content).get("stream")) for request in self.sent],
            unconsumed=len(self.script),
        )

    def assert_one_non_stream_send(self, out: Turn) -> None:
        self.assertEqual(out.stream_requests, [False], "not exactly one non-stream request")
        self.assertEqual(out.unconsumed, 1, "the trap response was consumed (a second request)")


# --- N1: declared-unsupported built-ins, every relevant terminal condition ---------------------

# (label, wire arguments, expected outcome); the category names are FinishStatusError's.
ROWS: dict[str, list[tuple[str, dict[str, Any], str]]] = {
    "anthropic": [
        ("max_tokens", dict(text=PARTIAL, finish="max_tokens"), "limit"),
        ("refusal", dict(text=PARTIAL, finish="refusal"), "reset"),
        ("pause_turn", dict(text=PARTIAL, finish="pause_turn"), "paused"),
        ("context window", dict(text=PARTIAL, finish="model_context_window_exceeded"), "context_window"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
    "minimax": [
        ("max_tokens", dict(text=PARTIAL, finish="max_tokens"), "limit"),
        ("refusal with a tool call (undocumented)", dict(text=PARTIAL, tools=TOOL, finish="refusal"),
         "unrecognized"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
    "openai": [
        ("length", dict(text=PARTIAL, finish="length"), "limit"),
        ("length with a tool call", dict(text=PARTIAL, tools=TOOL, finish="length"), "limit"),
        ("content_filter", dict(text=PARTIAL, finish="content_filter"), "blocked"),
        ("content_filter without text", dict(finish="content_filter"), "blocked"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
    "deepseek": [
        ("length", dict(text=PARTIAL, finish="length"), "limit"),
        ("content_filter", dict(text=PARTIAL, finish="content_filter"), "blocked"),
        ("aborted", dict(text=PARTIAL, finish="aborted"), "interrupted"),
        ("insufficient_system_resource", dict(text=PARTIAL, finish="insufficient_system_resource"),
         "interrupted"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
    "glm": [
        ("length", dict(text=PARTIAL, finish="length"), "limit"),
        ("sensitive", dict(text=PARTIAL, finish="sensitive"), "blocked"),
        ("sensitive without text", dict(finish="sensitive"), "blocked"),
        ("network_error", dict(text=PARTIAL, finish="network_error"), "interrupted"),
        ("context window", dict(text=PARTIAL, finish="model_context_window_exceeded"), "context_window"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
    "qwen": [
        ("length", dict(text=PARTIAL, finish="length"), "limit"),
        ("unknown finish with a tool call", dict(text=PARTIAL, tools=TOOL, finish="custom_x"), "unrecognized"),
        ("empty", dict(), "empty"),
        ("normal", dict(text=FULL), "normal"),
    ],
}


class TestDeclaredUnsupportedBuiltins(OfflineReplCase):
    """N1: stream mode = one non-stream chat() request with the provider's own checks."""

    def _run(self, name: str, wire_args: dict[str, Any], *, stream: bool,
             per_instance: bool = False) -> tuple[Turn, Any]:
        provider = declared_unsupported(name, per_instance=per_instance)
        self.assertFalse(supports_structured_streaming(provider))
        self.play(wire(name, **wire_args), wire(name, text=TRAP))
        return self.turn(provider, stream=stream), provider

    def _assert_route(self, out: Turn, provider: Any) -> None:
        """How it was sent: one non-stream chat() request, no agent route, no resend."""
        self.assert_one_non_stream_send(out)
        self.assertFalse(out.agent_route)
        self.assertEqual(out.traced, 0)
        self.assertNotIn(TRAP, out.printed)
        if hasattr(provider, "ledger"):  # class-level declaration: the entered methods
            self.assertEqual(provider.ledger, ["chat"], "not exactly one chat() call")

    def _assert_outcome(self, name: str, wire_args: dict[str, Any], kind: str, out: Turn) -> None:
        """What the user sees and what the conversation keeps."""
        text = wire_args.get("text", "")
        tools = wire_args.get("tools", ())
        value = wire_args.get("finish")
        user = ("user", DIRECT_PROMPT)
        self.assertNotIn("Error:", out.printed)
        self.assertNotIn(STREAMED_BLOCKED, out.printed, "claims streamed text that was not shown")
        if kind == "normal":
            self.assertEqual(out.history, [user, ("assistant", text)])
            self.assertEqual(out.usage, [USAGE[name]], "usage not recorded exactly once")
            self.assertNotIn(EMPTY_NOTICE, out.printed)
            self.assertNotIn("did not return token counts", out.printed)
            return
        if kind == "empty":
            self.assertEqual(out.history, [])  # E7: the unanswered turn is rolled back
            self.assertEqual(out.printed.count(EMPTY_NOTICE), 1)
            self.assertEqual(out.usage, [USAGE[name]], "reported usage not recorded exactly once")
            return
        # Abnormal endings: never stored, shown or recorded as a completed reply.
        self.assertNotIn(EMPTY_NOTICE, out.printed, "abnormal ending misreported as an empty response")
        dropped = " A tool call was not run." if tools else ""
        category = {"reset": "blocked"}.get(kind, kind)
        if kind == "limit":
            self.assertEqual(len(out.history), 2, "stored outcome")
            self.assertTrue(out.history[1][1].startswith(f"[{LIMIT}"),
                            f"output-limit ending stored without the incomplete marker: {out.history}")
            self.assertTrue(out.history[1][1].endswith(
                "The text below is partial and is not a complete answer.]\n\n" + text))
            self.assertIn(LIMIT, out.printed)
            self.assertIn(TEXT_ABOVE, out.printed)
            self.assertEqual(out.printed.count(text), 1)  # shown once, by the report helper
            self.assertEqual(
                "A tool call in this response was cut off and was not run." in out.printed, bool(tools))
        elif kind == "reset":
            self.assertEqual(out.history, [], "refused turn was not reset")
            self.assertIn("The refused message was removed from the conversation", out.printed)
            self.assertEqual(out.printed.count(text), 0)
        elif kind == "blocked":
            self.assertEqual(out.history, [user, ("assistant", f"[{NOTES['blocked']}.{dropped} "
                                                               "No partial output was kept.]")],
                             "blocked ending not stored as the blocked marker only")
            if text:
                self.assertEqual(out.printed.count(text), 0)  # blocked text is never shown after the fact
        else:
            kept = ("was not confirmed as complete" if kind == "unrecognized"
                    else "is partial and is not a complete answer")
            self.assertEqual(out.history, [user, ("assistant", f"[{NOTES[category]}.{dropped} The text "
                                                               f"below {kept}.]\n\n{text}")],
                             f"{kind} ending not stored with its marker")
            self.assertEqual(out.printed.count(text), 1)
            marked = "unverified" if kind == "unrecognized" else "incomplete"
            self.assertIn(f"It is kept in the conversation marked as {marked}", out.printed)
        if kind != "limit":
            self.assertIn(f"{NOTES[category]} ({value}).", out.printed)
            self.assertEqual("Clawd does not resume paused turns automatically." in out.printed,
                             kind == "paused")
            self.assertEqual("A tool call in this response was not run." in out.printed, bool(tools))
        self.assertEqual(out.usage, [], "abnormal request's usage recorded as a normal one")
        self.assertIn(NO_RETRY, out.printed)
        self.assertIn("at least 11 input" if name in ANTHROPIC_FAMILY else "at least 9 input", out.printed)

    def test_stream_mode_matches_non_stream_mode_and_keeps_every_check(self) -> None:
        for name, rows in ROWS.items():
            for label, wire_args, kind in rows:
                with self.subTest(provider=name, case=label):
                    streamed, provider = self._run(name, wire_args, stream=True)
                    self._assert_outcome(name, wire_args, kind, streamed)
                    self._assert_route(streamed, provider)
                    plain, provider = self._run(name, wire_args, stream=False)
                    self._assert_outcome(name, wire_args, kind, plain)
                    self._assert_route(plain, provider)
                    self.assertEqual(streamed.history, plain.history)
                    self.assertEqual(streamed.usage, plain.usage)
                    if kind == "normal":  # stream mode: raw text once; non-stream: Markdown
                        self.assertEqual(streamed.printed.count(wire_args["text"]), 1)
                        self.assertFalse(streamed.markdown)
                        self.assertTrue(plain.markdown)
                    else:
                        self.assertEqual(streamed.lines, plain.lines)

    def test_instance_level_declaration_gets_the_same_checks(self) -> None:
        for name in ("openai", "anthropic"):
            for label, wire_args, kind in ROWS[name]:
                with self.subTest(provider=name, case=label):
                    streamed, provider = self._run(name, wire_args, stream=True, per_instance=True)
                    self.assertFalse(hasattr(provider, "ledger"))
                    self._assert_outcome(name, wire_args, kind, streamed)
                    self._assert_route(streamed, provider)


# --- N2: the successful reply is shown once, raw, after it is stored --------------------------

class TestStreamModeDisplay(OfflineReplCase):
    def test_builtin_reply_is_shown_once_raw_with_usage(self) -> None:
        provider = declared_unsupported("openai")
        self.play(wire("openai", text="**bold** reply"), wire("openai", text=TRAP))
        out = self.turn(provider, stream=True)
        self.assertEqual(out.history, [("user", DIRECT_PROMPT), ("assistant", "**bold** reply")])
        self.assertEqual(out.lines.count("**bold** reply"), 1)  # the raw text, as one emit
        self.assertFalse(out.markdown)
        self.assertEqual(out.usage, [(9, 5, 14)], "usage not recorded exactly once")
        self.assert_one_non_stream_send(out)
        self.assertEqual(provider.ledger, ["chat"])
        self.assertFalse(out.agent_route)

    def test_plugin_reply_is_shown_once_raw_with_usage(self) -> None:
        provider = ChatPlugin()
        out = self.turn(provider, stream=True)
        self.assertEqual(out.history, [("user", DIRECT_PROMPT), ("assistant", "Plugin reply")])
        self.assertEqual(out.lines.count("Plugin reply"), 1)
        self.assertFalse(out.markdown)
        self.assertEqual(out.usage, [(4, 3, 7)], "usage not recorded exactly once")
        self.assertEqual(provider.ledger, ["chat"])
        self.assertFalse(out.agent_route)


# --- N3: "already shown" is what was actually shown live (stream_started) ---------------------

class TestPartialDisplayFollowsWhatWasShown(OfflineReplCase):
    def test_unsupported_direct_route_prints_the_partial_once(self) -> None:
        for label, error, marked in (
            ("output limit", limit_error(), "[" + LIMIT),
            ("interrupted", finish_error("interrupted", "network_error"), "[" + NOTES["interrupted"]),
        ):
            with self.subTest(label):
                provider = ChatPlugin(action=raising(error))
                out = self.turn(provider, stream=True)
                self.assertTrue(out.history[-1][1].startswith(marked), f"stored outcome: {out.history}")
                self.assertEqual(out.printed.count(PARTIAL), 1)
                self.assertIn(TEXT_ABOVE, out.printed)
                self.assertEqual(provider.ledger, ["chat"])
                self.assertFalse(out.agent_route)

    def test_unsupported_agent_route_prints_the_partial_once(self) -> None:
        """The agent route printed "The text above is partial" with nothing above it."""
        for label, error in (("output limit", limit_error()),
                             ("interrupted", finish_error("interrupted", "network_error"))):
            with self.subTest(label):
                provider = ChatPlugin(action=raising(error))
                out = self.turn(provider, stream=True, prompt=AGENT_PROMPT)
                self.assertEqual(out.printed.count(PARTIAL), 1, "partial text shown this many times")
                self.assertIn(TEXT_ABOVE, out.printed)
                self.assertTrue(out.agent_route)
                self.assertEqual(provider.ledger, ["chat"])

    def test_unsupported_agent_route_real_sdk_output_limit(self) -> None:
        provider = declared_unsupported("openai")
        self.play(wire("openai", text=PARTIAL, finish="length"), wire("openai", text=TRAP))
        out = self.turn(provider, stream=True, prompt=AGENT_PROMPT)
        self.assertTrue(out.history[-1][1].startswith("[" + LIMIT))
        self.assertEqual(out.printed.count(PARTIAL), 1, "partial text shown this many times")
        self.assertIn(TEXT_ABOVE, out.printed)
        self.assert_one_non_stream_send(out)
        self.assertTrue(out.agent_route)
        self.assertEqual(provider.ledger, ["chat"])

    def test_unsupported_blocked_never_claims_streamed_text(self) -> None:
        for prompt in (DIRECT_PROMPT, AGENT_PROMPT):
            with self.subTest(prompt=prompt):
                provider = ChatPlugin(action=raising(finish_error("blocked", "content_filter")))
                out = self.turn(provider, stream=True, prompt=prompt)
                self.assertIn(f"{NOTES['blocked']} (content_filter).", out.printed)
                self.assertNotIn(STREAMED_BLOCKED, out.printed, "claims streamed text that was not shown")
                self.assertEqual(out.printed.count(PARTIAL), 0)
                self.assertEqual(provider.ledger, ["chat"])

    def test_structured_route_follows_the_callback_not_the_stream_setting(self) -> None:
        # label, stream action, times PARTIAL is printed, "streamed text above was blocked" claimed
        cases = (
            ("limit after a live chunk", emit_then(limit_error()), 1, False),
            ("limit before any chunk", raising(limit_error()), 1, False),
            ("interrupted after a live chunk", emit_then(finish_error("interrupted", "x")), 1, False),
            ("interrupted before any chunk", raising(finish_error("interrupted", "x")), 1, False),
            ("blocked after a live chunk", emit_then(finish_error("blocked", "content_filter")), 1, True),
            ("blocked before any chunk", raising(finish_error("blocked", "content_filter")), 0, False),
        )
        for label, action, shown, claimed in cases:
            with self.subTest(label):
                provider = StructuredPlugin(stream_action=action)
                out = self.turn(provider, stream=True)
                self.assertEqual(out.printed.count(PARTIAL), shown, "partial text shown this many times")
                self.assertEqual(STREAMED_BLOCKED in out.printed, claimed, "streamed-text claim")
                self.assertEqual(provider.ledger, ["chat_stream_response"])
                self.assertFalse(out.agent_route)


# --- N4: no production caller of the legacy text stream ---------------------------------------

class TestNoLegacyStreamCaller(unittest.TestCase):
    def test_src_never_calls_chat_stream(self) -> None:
        callers: list[str] = []
        structured_calls = 0
        definitions = 0
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "chat_stream":
                    definitions += 1
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None)
                by_getattr = (name == "getattr" and len(node.args) >= 2
                              and isinstance(node.args[1], ast.Constant)
                              and node.args[1].value == "chat_stream")
                if name == "chat_stream" or by_getattr:
                    callers.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
                if name == "chat_stream_response":
                    structured_calls += 1
        self.assertEqual(callers, [], "production calls to the legacy chat_stream()")
        # The scan sees real code: the method is still defined (required API) and the
        # structured method is still called.
        self.assertGreater(definitions, 0)
        self.assertGreater(structured_calls, 0)


# --- N5: direct BaseProvider plugins ----------------------------------------------------------

class TestDirectPluginProviders(OfflineReplCase):
    def test_plugin_errors_from_chat_reach_the_report_handlers(self) -> None:
        cases = (
            ("output limit", limit_error(), "[" + LIMIT),
            ("finish status", finish_error("context_window", "model_context_window_exceeded"),
             "[" + NOTES["context_window"]),
        )
        for label, error, marker in cases:
            with self.subTest(label):
                provider = ChatPlugin(action=raising(error))
                out = self.turn(provider, stream=True)
                self.assertTrue(out.history[-1][1].startswith(marker), f"stored outcome: {out.history}")
                self.assertIn(NO_RETRY, out.printed)
                self.assertEqual(out.usage, [])
                self.assertEqual(out.traced, 0)
                self.assertEqual(provider.ledger, ["chat"])

    def test_plugin_finish_reason_is_trusted_as_returned(self) -> None:
        """Parity with non-stream mode and the agent route: Clawd does not reclassify it."""
        for stream in (True, False):
            with self.subTest(stream=stream):
                provider = ChatPlugin(action=lambda: plugin_reply(PARTIAL, finish_reason="length"))
                out = self.turn(provider, stream=stream)
                self.assertEqual(out.history, [("user", DIRECT_PROMPT), ("assistant", PARTIAL)])
                self.assertNotIn(LIMIT, out.printed)
                self.assertEqual(out.usage, [(4, 3, 7)])
                self.assertEqual(provider.ledger, ["chat"])


# --- N6: a display failure after a stored reply ------------------------------------------------

class TestDisplayFailureAfterStoredReply(OfflineReplCase):
    def test_console_failure_keeps_the_reply_and_never_resends(self) -> None:
        def fail_on_reply(*args: Any, **kwargs: Any) -> None:
            if args and args[0] == FULL:
                raise OSError("console closed")

        provider = declared_unsupported("openai")
        self.play(wire("openai", text=FULL), wire("openai", text=TRAP))
        out = self.turn(provider, stream=True, console_error=fail_on_reply)
        self.assertEqual(out.history, [("user", DIRECT_PROMPT), ("assistant", FULL)])  # stored first
        self.assertIn("Error: console closed", out.printed)
        self.assertIn("The request failed. Clawd did not issue a fallback retry.", out.printed)
        self.assertEqual(out.traced, 1)
        # Known consequence: the generic error path does not record this request's usage.
        self.assertEqual(out.usage, [])
        self.assert_one_non_stream_send(out)
        self.assertEqual(provider.ledger, ["chat"])
        self.assertFalse(out.agent_route)


# --- E7: an empty chat() reply in stream mode ---------------------------------------------------

class TestEmptyReplyInStreamMode(OfflineReplCase):
    def test_empty_reply_with_and_without_reported_usage(self) -> None:
        for label, usage, recorded in (("reported usage", {"input_tokens": 9, "output_tokens": 0}, [(9, 0, 9)]),
                                       ("no reported usage", {}, [])):
            with self.subTest(label):
                provider = ChatPlugin(action=lambda usage=usage: plugin_reply("", usage=usage))
                out = self.turn(provider, stream=True)
                self.assertEqual(out.history, [])  # rolled back; no assistant reply
                self.assertEqual(out.printed.count(EMPTY_NOTICE), 1)
                self.assertEqual(out.usage, recorded, "usage recorded")  # never fabricated
                self.assertEqual(provider.ledger, ["chat"])  # one request
                self.assertFalse(out.agent_route)
                self.assertEqual(out.traced, 0)


if __name__ == "__main__":
    unittest.main()
