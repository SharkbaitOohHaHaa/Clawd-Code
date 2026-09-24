from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.activity_ledger import append_activity, append_change
from src.observability import runtime_observability_snapshot
from src.provider_event_ledger import append_provider_event
from src.usage_ledger import append_provider_usage


class ObservabilityTests(unittest.TestCase):
    def _env(self, root: Path) -> dict[str, str]:
        return {
            "CLAWD_ACTIVITY_LEDGER": str(root / "activity.jsonl"),
            "CLAWD_CHANGE_LEDGER": str(root / "changes.jsonl"),
            "CLAWD_PROVIDER_EVENT_LEDGER": str(root / "provider-events.jsonl"),
            "CLAWD_USAGE_LEDGER": str(root / "provider-usage.jsonl"),
        }

    def test_snapshot_reports_sanitized_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self._env(root)
            with patch.dict(os.environ, env, clear=False):
                append_activity("tool", "Read", status="ok")
                append_activity("tool", "Write", status="error")
                append_activity("skill", "review", status="error")
                append_change("Edit", "src/example.py", "edit")
                append_provider_usage({"label": "Local (model)", "total_tokens": 3})
                append_provider_event(
                    provider="Demo",
                    operation="chat",
                    stage="provider_attempt",
                    status="failed",
                    model="model",
                    endpoint="https://user:secret@example.com/v1/chat?api_key=SECRET#frag",
                    error_type="TimeoutError",
                    status_code=504,
                )
                snapshot = runtime_observability_snapshot(recent_limit=5)

            raw_provider = (root / "provider-events.jsonl").read_text(encoding="utf-8")

        self.assertEqual(snapshot["tool_calls"], 2)
        self.assertEqual(snapshot["tool_errors"], 1)
        self.assertEqual(snapshot["skill_calls"], 1)
        self.assertEqual(snapshot["skill_errors"], 1)
        self.assertEqual(snapshot["change_events"], 1)
        self.assertEqual(snapshot["provider_events"], 1)
        self.assertEqual(snapshot["provider_failures"], 1)
        self.assertEqual(snapshot["usage_events"], 1)
        self.assertEqual(snapshot["issues"], [])
        self.assertEqual(len(snapshot["recent_errors"]), 3)
        self.assertIn("https://example.com/v1/chat", raw_provider)
        self.assertNotIn("user:secret", raw_provider)
        self.assertNotIn("api_key", raw_provider)
        self.assertNotIn("SECRET", raw_provider)

    def test_observability_write_failures_never_break_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "CLAWD_USAGE_LEDGER": str(root),
                    "CLAWD_PROVIDER_EVENT_LEDGER": str(root),
                },
                clear=False,
            ):
                append_provider_usage({"label": "Demo", "total_tokens": 1})
                append_provider_event(
                    provider="Demo",
                    operation="chat",
                    stage="provider_attempt",
                    status="success",
                    model="model",
                    endpoint="https://example.com/v1",
                )

    def test_malformed_ledgers_report_health_issue_without_leaking_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self._env(root)
            marker = "PRIVATE_MALFORMED_MARKER"
            (root / "activity.jsonl").write_text(marker + "\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=False):
                snapshot = runtime_observability_snapshot()

        self.assertEqual(
            [issue["code"] for issue in snapshot["issues"]],
            ["observability_ledger_malformed"],
        )
        self.assertNotIn(marker, json.dumps(snapshot))


if __name__ == "__main__":
    unittest.main()
