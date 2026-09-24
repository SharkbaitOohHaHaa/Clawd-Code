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
from src.tool_system.tools.youtube_gemini import YouTubeAnalyzeTool, _validate_youtube_url


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestYouTubeAnalyzeTool(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = ToolContext(workspace_root=Path.cwd())
        self.tool = YouTubeAnalyzeTool()

    def test_accepts_supported_youtube_video_urls(self) -> None:
        urls = [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "https://www.youtube.com/shorts/abc123",
            "https://www.youtube.com/live/abc123",
            "https://www.youtube.com/embed/abc123",
            "https://www.youtube.com/watch?v=abc123&t=30s&list=PL123",
            "https://youtu.be/abc123?si=share-token&t=30",
        ]
        expected = "https://www.youtube.com/watch?v=abc123"
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(_validate_youtube_url(url), expected)

    def test_rejects_non_youtube_url(self) -> None:
        with self.assertRaises(ToolInputError):
            _validate_youtube_url("https://example.com/watch?v=abc123")

    def test_requires_outbound_permission(self) -> None:
        result = self.tool.check_permissions(
            {"url": "https://www.youtube.com/watch?v=abc123"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("YouTube", result.message or "")

    def test_missing_gemini_key_fails_closed(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ToolInputError, "GEMINI_API_KEY"):
                self.tool.run(
                    {"url": "https://www.youtube.com/watch?v=abc123"},
                    self.ctx,
                )

    def test_calls_interactions_api_and_parses_structured_result(self) -> None:
        analysis = {
            "summary": "A concise summary.",
            "answer": "The requested answer.",
            "key_points": ["One", "Two"],
            "timestamps": [{"time": "00:12", "finding": "Key moment"}],
            "limitations": [],
        }
        api_response = {
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": json.dumps(analysis)}],
                }
            ],
            "usage": {
                "total_input_tokens": 1200,
                "total_output_tokens": 80,
                "total_thought_tokens": 40,
                "total_tool_use_tokens": 300,
                "total_cached_tokens": 10,
                "total_tokens": 1620,
            },
        }
        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout=0):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["api_key"] = req.get_header("X-goog-api-key")
            return _Response(json.dumps(api_response).encode("utf-8"))

        env = {
            "GEMINI_API_KEY": '"dummy-key"',
            "GEMINI_YOUTUBE_MODEL": "gemini-test",
        }

        with patch.dict(os.environ, env, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
                result = self.tool.run(
                    {
                        "url": "https://www.youtube.com/watch?v=abc123",
                        "question": "What happens?",
                    },
                    self.ctx,
                )

        payload = captured["payload"]
        self.assertEqual(captured["timeout"], 120)
        self.assertEqual(captured["api_key"], "dummy-key")
        self.assertEqual(payload["model"], "gemini-test")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["input"][0]["type"], "video")
        self.assertEqual(
            payload["input"][0]["uri"],
            "https://www.youtube.com/watch?v=abc123",
        )
        self.assertEqual(payload["input"][1]["text"], "What happens?")
        self.assertEqual(result.output["analysis"], analysis)
        self.assertEqual(
            self.ctx.usage_records,
            [
                {
                    "label": "Gemini (gemini-test)",
                    "input_tokens": 1200,
                    "output_tokens": 80,
                    "thought_tokens": 40,
                    "tool_use_tokens": 300,
                    "cached_tokens": 10,
                    "total_tokens": 1620,
                }
            ],
        )

    def test_default_registry_exposes_youtube_analyze(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        spec = registry.get("YouTubeAnalyze")
        self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()
