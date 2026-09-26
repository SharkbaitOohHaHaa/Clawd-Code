"""Test agent loop with mocked provider to verify tool invocation."""

import json
import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from src.agent.conversation import Conversation
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatResponse
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from src.tool_system.agent_loop import build_agent_preflight, run_agent_loop, AgentLoopResult


class TestAgentLoop(unittest.TestCase):
    """Test agent loop logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_preflight_filters_active_skill_tool_allowlist(self):
        conversation = Conversation()
        conversation.add_user_message("Use only the approved tools.")
        provider = MagicMock()
        provider.model = "test-model"
        self.context.restrict_tool_allowlist(["Read", "Task"])

        preflight = build_agent_preflight(
            conversation,
            provider,
            self.registry,
            self.context,
        )

        self.assertEqual(
            {schema["name"] for schema in preflight.tool_schemas},
            {"Read", "Agent"},
        )

    def test_agent_loop_restores_temporary_tool_allowlist(self):
        from src.tool_system.protocol import ToolResult
        from src.tool_system.registry import ToolRegistry, ToolSpec

        class ScopeTool:
            def spec(self):
                return ToolSpec(
                    name="Scope",
                    description="narrow tools",
                    input_schema={"type": "object", "additionalProperties": False, "properties": {}},
                    permission_policy="allow",
                )

            def run(self, tool_input, context):
                context.restrict_tool_allowlist(["Scope"])
                return ToolResult(name="Scope", output={"ok": True})

        registry = ToolRegistry([ScopeTool()])
        context = ToolContext(workspace_root=self.workspace)
        conversation = Conversation()
        conversation.add_user_message("scope it")
        provider = MagicMock()
        provider.model = "test-model"
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = [
            ChatResponse(
                content="",
                model="test-model",
                usage={},
                finish_reason="tool_use",
                tool_uses=[{"id": "scope-1", "name": "Scope", "input": {}}],
            ),
            ChatResponse(
                content="done",
                model="test-model",
                usage={},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

        result = run_agent_loop(conversation, provider, registry, context)

        self.assertEqual(result.response_text, "done")
        self.assertIsNone(context.tool_allowlist)

    def test_agent_loop_calls_tool(self):
        """Test agent loop correctly dispatches a tool call from mocked LLM."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        # First response: tool use Write
        mock_tool_use = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(self.workspace / "hello.py"),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_use],
        )

        # Second response: final text after tool result
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        # Verify final response
        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(result.response_text, "File created successfully!")

        # Verify provider was called twice and both OpenAI-compatible
        # turns retain the effective system/workspace context.
        self.assertEqual(mock_provider.chat.call_count, 2)
        first_messages = mock_provider.chat.call_args_list[0].args[0]
        second_messages = mock_provider.chat.call_args_list[1].args[0]
        self.assertEqual(first_messages[0]["role"], "system")
        self.assertEqual(second_messages[0]["role"], "system")
        self.assertEqual(first_messages[0]["content"], second_messages[0]["content"])

        # Verify file was created
        hello_py = self.workspace / "hello.py"
        self.assertTrue(hello_py.exists())
        self.assertEqual(hello_py.read_text(), "print('hello world')")

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_anthropic_refreshes_system_prompt_after_worktree_switch(self):
        root = self.workspace.resolve()

        def git(*args: str) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result

        git("init", "-q")
        git("config", "user.email", "clawd-test@example.invalid")
        git("config", "user.name", "Clawd Test")
        (root / "CLAUDE.md").write_text("Original context.\n", encoding="utf-8")
        (root / "app.py").write_text("value = 1\n", encoding="utf-8")
        git("add", "CLAUDE.md", "app.py")
        git("commit", "-q", "-m", "initial")

        conversation = Conversation()
        conversation.add_user_message("Enter an isolated worktree and continue there.")

        provider = AnthropicProvider(api_key="test-key")
        provider.chat = MagicMock(side_effect=[
            ChatResponse(
                content="Switching worktrees.",
                model=provider.model,
                usage={},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "worktree_1",
                    "name": "EnterWorktree",
                    "input": {"name": "context-refresh"},
                }],
            ),
            ChatResponse(
                content="Now operating in the worktree.",
                model=provider.model,
                usage={},
                finish_reason="stop",
                tool_uses=None,
            ),
        ])
        self.context.permission_handler = lambda name, message, suggestion: (True, False)

        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
        )

        self.assertEqual(result.response_text, "Now operating in the worktree.")
        self.assertEqual(provider.chat.call_count, 2)
        first_system = provider.chat.call_args_list[0].kwargs["system"]
        second_system = provider.chat.call_args_list[1].kwargs["system"]
        self.assertNotEqual(first_system, second_system)
        self.assertNotEqual(self.context.cwd, root)
        self.assertIn(str(self.context.cwd), second_system)
        self.assertNotIn(str(self.context.cwd), first_system)

    def test_openai_compatible_tool_turn_preserves_reasoning_content(self):
        """Thinking providers must receive prior reasoning again on tool turns."""
        conversation = Conversation()
        conversation.add_user_message("Create reasoning.txt")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.side_effect = [
            ChatResponse(
                content="I will create it.",
                model="deepseek-flash",
                usage={"input_tokens": 10, "output_tokens": 5},
                finish_reason="tool_calls",
                reasoning_content="I should use the Write tool.",
                tool_uses=[{
                    "id": "call_1",
                    "name": "Write",
                    "input": {
                        "file_path": str(self.workspace / "reasoning.txt"),
                        "content": "ok",
                    },
                }],
            ),
            ChatResponse(
                content="Done.",
                model="deepseek-flash",
                usage={"input_tokens": 20, "output_tokens": 4},
                finish_reason="stop",
            ),
        ]

        run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
        )

        second_messages = mock_provider.chat.call_args_list[1].args[0]
        assistant = next(
            message
            for message in second_messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        self.assertEqual(
            assistant["reasoning_content"],
            "I should use the Write tool.",
        )

    def test_agent_loop_records_tool_once_via_dispatch(self):
        conversation = Conversation()
        conversation.add_user_message("Create once.txt")

        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        target = self.workspace / "once.txt"
        provider.chat.side_effect = [
            ChatResponse(
                content="Writing.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "tool_once",
                    "name": "Write",
                    "input": {"file_path": str(target), "content": "DO_NOT_LOG_THIS_CONTENT"},
                }],
            ),
            ChatResponse(
                content="Done.",
                model="test-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

        activity = self.workspace / "activity.jsonl"
        changes = self.workspace / "changes.jsonl"
        self.context.instrumentation_enabled = True
        with patch.dict(
            os.environ,
            {
                "CLAWD_ACTIVITY_LEDGER": str(activity),
                "CLAWD_CHANGE_LEDGER": str(changes),
            },
        ):
            result = run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
            )

        self.assertEqual(result.response_text, "Done.")
        activity_events = [
            json.loads(line)
            for line in activity.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        change_events = [
            json.loads(line)
            for line in changes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            [(event["kind"], event["name"], event["status"]) for event in activity_events],
            [("tool", "Write", "ok")],
        )
        self.assertEqual(
            [
                (event["tool"], event["path"], event["operation"])
                for event in change_events
            ],
            [("Write", "once.txt", "create")],
        )
        self.assertNotIn("DO_NOT_LOG_THIS_CONTENT", activity.read_text(encoding="utf-8"))
        self.assertNotIn("DO_NOT_LOG_THIS_CONTENT", changes.read_text(encoding="utf-8"))

    def test_agent_loop_creates_hello_world(self):
        """Test agent loop creates hello.py and writes print('hello world')."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        # First response: tool use Write
        hello_path = self.workspace / "hello.py"
        mock_tool_write = {
            "id": "toolu_123",
            "name": "Write",
            "input": {
                "file_path": str(hello_path),
                "content": "print('hello world')"
            }
        }
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[mock_tool_write],
        )

        # Second response: final
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )

        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertTrue(hello_path.exists())
        self.assertEqual(hello_path.read_text(), "print('hello world')")

    def test_agent_loop_stream_emits_final_text_chunks(self):
        """Streaming mode emits final response chunks without changing the result."""
        conversation = Conversation()
        conversation.add_user_message("Say hello")

        mock_provider = MagicMock()
        mock_provider.SUPPORTS_STRUCTURED_STREAMING = False  # chat() is chosen before any call
        mock_provider.chat.return_value = ChatResponse(
            content="Hello from Clawd!",
            model="test-model",
            usage={"input_tokens": 3, "output_tokens": 4},
            finish_reason="stop",
            tool_uses=None,
        )

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello from Clawd!")
        self.assertEqual(result.response_text, "Hello from Clawd!")
        self.assertEqual(mock_provider.chat.call_count, 1)
        mock_provider.chat_stream_response.assert_not_called()
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[-1].role, "assistant")
        self.assertEqual(conversation.messages[-1].content, "Hello from Clawd!")

    def test_agent_loop_stream_only_emits_final_turn_text(self):
        """Streaming mode skips interim tool-planning text and emits the final answer only."""
        conversation = Conversation()
        conversation.add_user_message("Create a file hello.py with content print('hello world')")

        mock_provider = MagicMock()
        mock_provider.SUPPORTS_STRUCTURED_STREAMING = False  # chat() is chosen before any call
        hello_path = self.workspace / "hello.py"
        mock_response1 = ChatResponse(
            content="I will create the file.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            finish_reason="tool_use",
            tool_uses=[{
                "id": "toolu_123",
                "name": "Write",
                "input": {
                    "file_path": str(hello_path),
                    "content": "print('hello world')",
                },
            }],
        )
        mock_response2 = ChatResponse(
            content="File created successfully!",
            model="test-model",
            usage={"input_tokens": 30, "output_tokens": 10},
            finish_reason="stop",
            tool_uses=None,
        )
        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "File created successfully!")
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertTrue(hello_path.exists())

    def test_agent_loop_stream_uses_structured_provider_streaming_for_tool_turns(self):
        """Structured provider streaming can emit pre-tool text and final text across turns."""
        conversation = Conversation()
        conversation.add_user_message("Create hello.py")

        provider = MagicMock()
        hello_path = self.workspace / "hello.py"

        stream_responses = [
            ChatResponse(
                content="I will create the file.",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 20},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_123",
                    "name": "Write",
                    "input": {
                        "file_path": str(hello_path),
                        "content": "print('hello world')",
                    },
                }],
            ),
            ChatResponse(
                content="File created successfully!",
                model="test-model",
                usage={"input_tokens": 30, "output_tokens": 10},
                finish_reason="stop",
                tool_uses=None,
            ),
        ]

        def stream_side_effect(messages, tools=None, on_text_chunk=None, **kwargs):
            response = stream_responses.pop(0)
            if on_text_chunk is not None and response.content:
                on_text_chunk(response.content)
            return response

        provider.chat_stream_response.side_effect = stream_side_effect
        provider.chat.side_effect = AssertionError("chat() should not be used when structured streaming is available")

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "I will create the file.File created successfully!")
        self.assertEqual(result.response_text, "File created successfully!")
        self.assertEqual(provider.chat_stream_response.call_count, 2)
        self.assertTrue(hello_path.exists())

    def test_preflight_exposes_all_registered_tools_for_specific_requests(self):
        from unittest.mock import patch
        from src.tool_system.agent_loop import build_agent_preflight

        registered_names = [spec.name for spec in self.registry.list_specs()]
        requests = (
            "Ask Gemini this exact question using GeminiThink.",
            "Use Qwen to analyze this image.",
            "Search the web for the latest docs and update README.md.",
        )
        for request in requests:
            with self.subTest(request=request):
                conversation = Conversation()
                conversation.add_user_message(request)
                provider = MagicMock()
                with patch(
                    "src.tool_system.agent_loop._build_effective_system_prompt",
                    return_value="SYSTEM",
                ):
                    preflight = build_agent_preflight(
                        conversation, provider, self.registry, self.context
                    )
                names = [schema["name"] for schema in preflight.tool_schemas]
                self.assertEqual(names, registered_names)

        compound = Conversation()
        compound.add_user_message(
            "Search the web for the latest docs and update README.md."
        )
        provider = MagicMock()
        with patch(
            "src.tool_system.agent_loop._build_effective_system_prompt",
            return_value="SYSTEM",
        ):
            preflight = build_agent_preflight(
                compound, provider, self.registry, self.context
            )
        names = {schema["name"] for schema in preflight.tool_schemas}
        self.assertTrue({"WebSearch", "WebFetch", "Read", "Write", "Edit"} <= names)

    def test_prepare_anthropic_messages_compacts_only_under_context_pressure(self):
        from src.agent.conversation import ToolUseContentBlock
        from src.context_system.microcompact import CLEARED_MESSAGE
        from src.tool_system.agent_loop import _prepare_anthropic_messages

        conversation = Conversation()
        for index in range(4):
            tool_id = f"tool{index}"
            conversation.add_assistant_message([
                ToolUseContentBlock(
                    type="tool_use",
                    id=tool_id,
                    name="Read",
                    input={"file_path": f"{index}.txt"},
                )
            ])
            conversation.add_tool_result_message(tool_id, "x" * 100)

        unpressured = _prepare_anthropic_messages(
            conversation,
            context_window=100_000,
            fixed_input_tokens=0,
            output_reserve_tokens=0,
        )
        unpressured_results = [
            block["content"]
            for message in unpressured
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertEqual(unpressured_results, ["x" * 100] * 4)

        pressured = _prepare_anthropic_messages(
            conversation,
            context_window=1,
            fixed_input_tokens=1,
            output_reserve_tokens=0,
        )
        pressured_results = [
            block["content"]
            for message in pressured
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertEqual(pressured_results[0], CLEARED_MESSAGE)
        self.assertEqual(pressured_results[-1], "x" * 100)

        original_results = [
            block.content
            for message in conversation.messages
            if isinstance(message.content, list)
            for block in message.content
            if getattr(block, "type", "") == "tool_result"
        ]
        self.assertEqual(original_results, ["x" * 100] * 4)

    def test_anthropic_preflight_preserves_tool_results_below_context_pressure(self):
        from unittest.mock import patch
        from src.agent.conversation import ToolUseContentBlock
        from src.providers.anthropic_provider import AnthropicProvider
        from src.tool_system.agent_loop import build_agent_preflight

        conversation = Conversation()
        for index in range(4):
            tool_id = f"tool{index}"
            conversation.add_assistant_message([
                ToolUseContentBlock(
                    type="tool_use",
                    id=tool_id,
                    name="Read",
                    input={"file_path": f"{index}.txt"},
                )
            ])
            conversation.add_tool_result_message(tool_id, "x" * 100)

        provider = AnthropicProvider(api_key="test-key")
        with patch(
            "src.tool_system.agent_loop._build_effective_system_prompt",
            return_value="SYSTEM",
        ):
            preflight = build_agent_preflight(
                conversation, provider, self.registry, self.context
            )

        results = [
            block["content"]
            for message in preflight.api_messages
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertEqual(results, ["x" * 100] * 4)

    def test_preflight_estimate_uses_full_tool_schema_system_and_prepared_messages(self):
        from unittest.mock import patch
        from src.context_system.context_analyzer import count_tool_definition_tokens
        from src.token_estimation import count_messages_tokens, count_tokens
        from src.tool_system.agent_loop import build_agent_preflight

        conversation = Conversation()
        conversation.add_user_message("Fix this source file and update its test.")
        provider = MagicMock()

        with patch(
            "src.tool_system.agent_loop._build_effective_system_prompt",
            return_value="SYSTEM-CONTEXT",
        ):
            preflight = build_agent_preflight(
                conversation, provider, self.registry, self.context
            )

        expected = (
            count_tokens("SYSTEM-CONTEXT")
            + count_messages_tokens(preflight.api_messages)
            + count_tool_definition_tokens(preflight.tool_schemas)
        )
        self.assertEqual(preflight.estimated_input_tokens, expected)
        self.assertEqual(
            [schema["name"] for schema in preflight.tool_schemas],
            [spec.name for spec in self.registry.list_specs()],
        )
        self.assertEqual(
            preflight.api_messages,
            [{"role": "user", "content": "Fix this source file and update its test."}],
        )

    def test_preflight_tool_schema_is_stable_across_user_intent(self):
        from unittest.mock import patch
        from src.tool_system.agent_loop import build_agent_preflight

        expected_names = [spec.name for spec in self.registry.list_specs()]
        requests = (
            "Use GeminiThink.",
            "Search the web for current docs.",
            "Do the unusual operation.",
        )

        provider = MagicMock()
        for request in requests:
            with self.subTest(request=request):
                conversation = Conversation()
                conversation.add_user_message(request)
                with patch(
                    "src.tool_system.agent_loop._build_effective_system_prompt",
                    return_value="SYSTEM",
                ):
                    preflight = build_agent_preflight(
                        conversation, provider, self.registry, self.context
                    )
                self.assertEqual(
                    [schema["name"] for schema in preflight.tool_schemas],
                    expected_names,
                )

    def test_agent_loop_usage_accounting_unchanged_with_preflight(self):
        from unittest.mock import patch
        from src.tool_system.agent_loop import build_agent_preflight

        conversation = Conversation()
        conversation.add_user_message("Fix this source file.")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.return_value = ChatResponse(
            content="Done",
            model="test-model",
            usage={"input_tokens": 123, "output_tokens": 45},
            finish_reason="stop",
            tool_uses=None,
        )
        with patch(
            "src.tool_system.agent_loop._build_effective_system_prompt",
            return_value="SYSTEM",
        ):
            preflight = build_agent_preflight(
                conversation, provider, self.registry, self.context
            )
            result = run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                preflight=preflight,
            )

        self.assertEqual(result.usage, {"input_tokens": 123, "output_tokens": 45})
        provider.chat.assert_called_once()

    def test_agent_loop_stream_falls_back_when_structured_streaming_is_unavailable(self):
        """If the provider lacks structured streaming, the stable synchronous path still works."""
        conversation = Conversation()
        conversation.add_user_message("Say hello")

        provider = MagicMock()
        provider.SUPPORTS_STRUCTURED_STREAMING = False  # declared: chat() without any stream call
        provider.chat.return_value = ChatResponse(
            content="Hello from fallback!",
            model="test-model",
            usage={"input_tokens": 2, "output_tokens": 3},
            finish_reason="stop",
            tool_uses=None,
        )

        chunks: list[str] = []
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            stream=True,
            verbose=False,
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello from fallback!")
        self.assertEqual(result.response_text, "Hello from fallback!")
        provider.chat.assert_called_once()
        provider.chat_stream_response.assert_not_called()

    def _run_failing_stream(self, stream_side_effect, on_text_chunk=None):
        conversation = Conversation()
        conversation.add_user_message("Say hello")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = stream_side_effect
        provider.chat.side_effect = AssertionError("chat() must not resend after a streaming failure")
        chunks: list[str] = []
        with self.assertRaises(Exception) as caught:
            run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                stream=True,
                verbose=False,
                on_text_chunk=on_text_chunk or chunks.append,
            )
        provider.chat.assert_not_called()
        provider.chat_stream_response.assert_called_once()
        return caught.exception, chunks

    def test_stream_failure_before_any_chunk_surfaces_without_resend(self):
        class ProviderStatusError(RuntimeError):
            status_code = 529

        error, chunks = self._run_failing_stream(ProviderStatusError("overloaded"))
        self.assertIsInstance(error, ProviderStatusError)
        self.assertEqual(chunks, [])

    def test_stream_failure_after_chunk_surfaces_without_resend_or_duplicate_text(self):
        def stream_then_fail(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("partial answer")
            raise ConnectionError("stream dropped")

        error, chunks = self._run_failing_stream(stream_then_fail)
        self.assertIsInstance(error, ConnectionError)
        self.assertEqual(chunks, ["partial answer"])

    def test_authentication_error_during_stream_is_not_resent(self):
        class StatusAuthError(RuntimeError):
            status_code = 401

        error, _ = self._run_failing_stream(StatusAuthError("rejected"))
        self.assertIsInstance(error, StatusAuthError)

    def test_not_implemented_after_a_provider_chunk_is_not_treated_as_unsupported(self):
        def stream_then_not_implemented(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("partial")
            raise NotImplementedError("late")

        error, chunks = self._run_failing_stream(stream_then_not_implemented)
        self.assertIsInstance(error, NotImplementedError)
        self.assertEqual(chunks, ["partial"])

    def test_display_callback_not_implemented_is_not_treated_as_unsupported(self):
        def stream_one_chunk(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("hello")
            return ChatResponse(content="hello", model="m", usage={}, finish_reason="stop", tool_uses=None)

        def display_raises(chunk: str) -> None:
            raise NotImplementedError("display failed")

        error, _ = self._run_failing_stream(stream_one_chunk, on_text_chunk=display_raises)
        self.assertIsInstance(error, NotImplementedError)

    def test_attribute_error_during_stream_is_not_treated_as_unsupported(self):
        error, chunks = self._run_failing_stream(AttributeError("provider bug"))
        self.assertIsInstance(error, AttributeError)
        self.assertEqual(chunks, [])

    def test_osv_query_summary_shows_status_and_clawd_statement(self):
        from src.tool_system.agent_loop import summarize_tool_result

        output = {"status": "no_records_found", "statement": "Clawd statement.", "records": [{"summary": "osv text"}]}
        summary = summarize_tool_result("OsvQuery", output)
        self.assertEqual(summary, "OsvQuery · no_records_found · Clawd statement.")
        self.assertNotIn("osv text", summary)

    def test_wrong_stream_response_type_surfaces_without_resend(self):
        error, _ = self._run_failing_stream(lambda *args, **kwargs: {"content": "not a ChatResponse"})
        self.assertIsInstance(error, TypeError)


if __name__ == "__main__":
    unittest.main()
