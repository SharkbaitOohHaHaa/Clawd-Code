from __future__ import annotations

import io
import json
import os
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.permission_handler import PermissionBehavior
from src.tool_system.tools.gemini_think import GeminiThinkTool


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestGeminiThinkTool(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = ToolContext(workspace_root=Path.cwd())
        self.tool = GeminiThinkTool()

    def test_requires_outbound_permission(self) -> None:
        result = self.tool.check_permissions({"question": "Why?"}, self.ctx)
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("Gemini 3.8 Flash", result.message or "")

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ToolInputError, "non-empty string"):
            self.tool.run({"question": "   "}, self.ctx)

    def test_rejects_question_over_limit(self) -> None:
        with self.assertRaisesRegex(ToolInputError, "20,000-character limit"):
            self.tool.run({"question": "x" * 20_001}, self.ctx)

    def test_missing_gemini_key_fails_closed(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ToolInputError, "GEMINI_API_KEY"):
                self.tool.run({"question": "Why?"}, self.ctx)

    def test_calls_interactions_api_and_records_usage(self) -> None:
        api_response = {
            "steps": [{"type": "model_output", "content": [
                {"type": "text", "text": "Use independent perspectives."}
            ]}],
            "usage": {
                "total_input_tokens": 12, "total_output_tokens": 7,
                "total_thought_tokens": 3, "total_tool_use_tokens": 0,
                "total_cached_tokens": 2, "total_tokens": 22,
            },
        }
        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout=0):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["api_key"] = req.get_header("X-goog-api-key")
            return _Response(json.dumps(api_response).encode("utf-8"))

        with patch.dict(os.environ, {"GEMINI_API_KEY": '"dummy-key"'}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
                result = self.tool.run({"question": "  Why multiple models?  "}, self.ctx)

        payload = captured["payload"]
        self.assertEqual(captured["timeout"], 120)
        self.assertEqual(captured["api_key"], "dummy-key")
        self.assertEqual(payload["model"], "gemini-3.8-flash")
        self.assertEqual(payload["input"], "Why multiple models?")
        self.assertFalse(payload["store"])
        self.assertEqual(result.output, {
            "model": "gemini-3.8-flash",
            "answer": "Use independent perspectives.",
        })
        self.assertEqual(self.ctx.usage_records, [{
            "label": "Gemini (gemini-3.8-flash)",
            "input_tokens": 12, "output_tokens": 7,
            "thought_tokens": 3, "tool_use_tokens": 0,
            "cached_tokens": 2, "total_tokens": 22,
        }])

    def test_invalid_json_response_fails(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"}, clear=False):
            with patch.object(urllib.request, "urlopen", return_value=_Response(b"not-json")):
                with self.assertRaisesRegex(RuntimeError, "invalid response"):
                    self.tool.run({"question": "Why?"}, self.ctx)

    def test_default_registry_exposes_gemini_think(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        self.assertIsNotNone(registry.get("GeminiThink"))


if __name__ == "__main__":
    unittest.main()
