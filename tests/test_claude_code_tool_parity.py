from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.capabilities import expected_registered_tool_names, load_capability_manifest
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


class TestClaudeCodeToolParity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_registry_matches_capability_manifest(self) -> None:
        manifest = load_capability_manifest()
        actual = sorted(spec.name for spec in self.registry.list_specs())
        self.assertEqual(actual, expected_registered_tool_names(manifest))

        unavailable = {
            name for name, record in manifest["tools"].items()
            if not record.get("registered_expected")
        }
        for name in unavailable:
            with self.subTest(name=name):
                self.assertIsNone(self.registry.get(name))

    def test_send_user_message_is_user_visible_fallback(self) -> None:
        conversation = Conversation()
        conversation.add_user_message("hi")

        mock_provider = MagicMock()
        mock_tool_use = {
            "id": "toolu_1",
            "name": "SendUserMessage",
            "input": {"message": "hello", "status": "normal"},
        }
        mock_response1 = ChatResponse(
            content="",
            model="test",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="tool_use",
            tool_uses=[mock_tool_use],
        )
        mock_response2 = ChatResponse(
            content="",
            model="test",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )
        mock_provider.chat.side_effect = [mock_response1, mock_response2]

        out = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.ctx,
            verbose=False,
        )
        self.assertEqual(out.response_text, "hello")

    def test_tool_search_select(self) -> None:
        out = self.registry.dispatch(
            ToolCall(name="ToolSearch", input={"query": "select:Read"}),
            self.ctx,
        ).output
        self.assertEqual(out["matches"], ["Read"])

    def test_todo_write_roundtrip(self) -> None:
        out1 = self.registry.dispatch(
            ToolCall(
                name="TodoWrite",
                input={"todos": [{"content": "x", "status": "pending", "activeForm": "Doing x"}]},
            ),
            self.ctx,
        ).output
        self.assertEqual(out1["oldTodos"], [])
        self.assertEqual(len(out1["newTodos"]), 1)
        self.assertEqual(len(self.ctx.todos), 1)

        self.registry.dispatch(
            ToolCall(
                name="TodoWrite",
                input={"todos": [{"content": "x", "status": "completed", "activeForm": "Did x"}]},
            ),
            self.ctx,
        )
        self.assertEqual(self.ctx.todos, [])


if __name__ == "__main__":
    unittest.main()

