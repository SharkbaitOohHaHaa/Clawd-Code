import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.activity_ledger import activity_counts, append_activity, append_change
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.protocol import ToolCall


class TestActivityLedger(unittest.TestCase):
    def test_counts_only_real_recorded_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "activity.jsonl"
            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                append_activity("skill", "requesting-code-review")
                append_activity("skill", "requesting-code-review")
                append_activity("tool", "Read")
                append_activity("tool", "Read")
                append_activity("tool", "Edit")
                counts = activity_counts()

        self.assertEqual(counts["skill"]["requesting-code-review"], 2)
        self.assertEqual(counts["tool"]["Read"], 2)
        self.assertEqual(counts["tool"]["Edit"], 1)

    def test_dispatch_records_canonical_tools_and_nested_agent_calls_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "activity.jsonl"
            ctx = ToolContext(workspace_root=root, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                direct = registry.dispatch(
                    ToolCall(name="Sleep", input={"seconds": 0}),
                    ctx,
                )
                self.assertFalse(direct.is_error)

                nested = registry.dispatch(
                    ToolCall(
                        name="Task",
                        input={
                            "calls": [{"name": "Sleep", "input": {"seconds": 0}}],
                            "stop_on_error": True,
                        },
                    ),
                    ctx,
                )
                self.assertFalse(nested.is_error)
                counts = activity_counts()
                events = [
                    json.loads(line)
                    for line in ledger.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

        self.assertEqual(counts["tool"]["Sleep"], 2)
        self.assertEqual(counts["tool"]["Agent"], 1)
        self.assertEqual([event["name"] for event in events], ["Sleep", "Sleep", "Agent"])
        self.assertTrue(all(set(event) == {"timestamp", "kind", "name", "status"} for event in events))
        self.assertTrue(all(event["status"] == "ok" for event in events))

    def test_exception_raised_tool_failure_is_recorded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "activity.jsonl"
            ctx = ToolContext(workspace_root=root, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                with self.assertRaises(ToolInputError):
                    registry.dispatch(
                        ToolCall(name="Sleep", input={"seconds": "not-a-number"}),
                        ctx,
                    )
                events = [
                    json.loads(line)
                    for line in ledger.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

        self.assertEqual(len(events), 1)
        self.assertEqual(
            {key: events[0][key] for key in ("kind", "name", "status")},
            {"kind": "tool", "name": "Sleep", "status": "error"},
        )

    def test_failed_skill_result_records_skill_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "activity.jsonl"
            ctx = ToolContext(workspace_root=root, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                result = registry.dispatch(
                    ToolCall(name="Skill", input={"skill": "missing-skill"}),
                    ctx,
                )
                self.assertTrue(result.is_error)
                event = json.loads(ledger.read_text(encoding="utf-8").strip())

        self.assertEqual(
            {key: event[key] for key in ("kind", "name", "status")},
            {"kind": "skill", "name": "missing-skill", "status": "error"},
        )

    def test_change_ledger_contains_only_sanitized_relative_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            activity = root / "activity.jsonl"
            changes = root / "changes.jsonl"
            ctx = ToolContext(workspace_root=workspace, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(
                os.environ,
                {
                    "CLAWD_ACTIVITY_LEDGER": str(activity),
                    "CLAWD_CHANGE_LEDGER": str(changes),
                },
            ):
                target = workspace / "src" / "example.txt"
                result = registry.dispatch(
                    ToolCall(
                        name="Write",
                        input={"file_path": str(target), "content": "PRIVATE_CONTENT_MARKER"},
                    ),
                    ctx,
                )
                self.assertFalse(result.is_error)

                ctx.permission_handler = lambda name, message, suggestion: (True, False)
                sensitive = registry.dispatch(
                    ToolCall(
                        name="Write",
                        input={
                            "file_path": str(workspace / ".env"),
                            "content": "SECRET_MARKER=1",
                        },
                    ),
                    ctx,
                )
                self.assertFalse(sensitive.is_error)

                events = [
                    json.loads(line)
                    for line in changes.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                raw = changes.read_text(encoding="utf-8")

        self.assertEqual(
            [{key: event[key] for key in ("tool", "path", "operation")} for event in events],
            [{"tool": "Write", "path": "src/example.txt", "operation": "create"}],
        )
        self.assertNotIn(str(workspace), raw)
        self.assertNotIn("PRIVATE_CONTENT_MARKER", raw)
        self.assertNotIn("SECRET_MARKER", raw)
        self.assertNotIn(".env", raw)

    def test_notebook_edit_change_ledger_is_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            activity = root / "activity.jsonl"
            changes = root / "changes.jsonl"
            notebook = workspace / "analysis.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "markdown",
                                "id": "cell-1",
                                "metadata": {},
                                "source": ["old\n"],
                            }
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    }
                ),
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=workspace, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(
                os.environ,
                {
                    "CLAWD_ACTIVITY_LEDGER": str(activity),
                    "CLAWD_CHANGE_LEDGER": str(changes),
                },
            ):
                read = registry.dispatch(
                    ToolCall(name="Read", input={"file_path": str(notebook)}),
                    ctx,
                )
                self.assertFalse(read.is_error)
                edited = registry.dispatch(
                    ToolCall(
                        name="NotebookEdit",
                        input={
                            "notebook_path": str(notebook),
                            "cell_id": "cell-1",
                            "new_source": "PRIVATE_NOTEBOOK_CONTENT",
                        },
                    ),
                    ctx,
                )
                self.assertFalse(edited.is_error)
                events = [
                    json.loads(line)
                    for line in changes.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                raw = changes.read_text(encoding="utf-8")

        self.assertEqual(
            [{key: event[key] for key in ("tool", "path", "operation")} for event in events],
            [{"tool": "NotebookEdit", "path": "analysis.ipynb", "operation": "notebook_edit"}],
        )
        self.assertNotIn("PRIVATE_NOTEBOOK_CONTENT", raw)
        self.assertNotIn(str(workspace), raw)

    def test_data_transform_change_ledger_is_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            activity = root / "activity.jsonl"
            changes = root / "changes.jsonl"
            source = workspace / "source.csv"
            target = workspace / "selected.jsonl"
            source.write_text(
                "name,secret\nalice,PRIVATE_FILTER_VALUE\nbob,other\n",
                encoding="utf-8",
            )
            ctx = ToolContext(workspace_root=workspace, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(
                os.environ,
                {
                    "CLAWD_ACTIVITY_LEDGER": str(activity),
                    "CLAWD_CHANGE_LEDGER": str(changes),
                },
            ):
                inspected = registry.dispatch(
                    ToolCall(name="DataInspect", input={"file_path": str(source)}),
                    ctx,
                )
                self.assertFalse(inspected.is_error)
                transformed = registry.dispatch(
                    ToolCall(
                        name="DataTransform",
                        input={
                            "input_path": str(source),
                            "output_path": str(target),
                            "columns": ["name"],
                            "filters": {"secret": "PRIVATE_FILTER_VALUE"},
                        },
                    ),
                    ctx,
                )
                self.assertFalse(transformed.is_error)
                events = [
                    json.loads(line)
                    for line in changes.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                raw = changes.read_text(encoding="utf-8")

        self.assertEqual(
            [{key: event[key] for key in ("tool", "path", "operation")} for event in events],
            [{"tool": "DataTransform", "path": "selected.jsonl", "operation": "data_transform"}],
        )
        self.assertNotIn("PRIVATE_FILTER_VALUE", raw)
        self.assertNotIn("alice", raw)
        self.assertNotIn("source.csv", raw)
        self.assertNotIn(str(workspace), raw)

    def test_denied_sensitive_read_records_only_sanitized_error_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "activity.jsonl"
            secret = root / ".env"
            secret.write_text("TOP_SECRET_VALUE=1", encoding="utf-8")
            ctx = ToolContext(workspace_root=root, instrumentation_enabled=True)
            registry = build_default_registry(include_user_tools=False)

            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                result = registry.dispatch(
                    ToolCall(name="Read", input={"file_path": str(secret)}),
                    ctx,
                )
                self.assertTrue(result.is_error)
                raw = ledger.read_text(encoding="utf-8")
                event = json.loads(raw.strip())

        self.assertEqual(
            {key: event[key] for key in ("kind", "name", "status")},
            {"kind": "tool", "name": "Read", "status": "error"},
        )
        self.assertNotIn(".env", raw)
        self.assertNotIn("TOP_SECRET_VALUE", raw)
        self.assertNotIn("secret-bearing", raw)

    def test_append_change_rejects_absolute_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "changes.jsonl"
            with patch.dict(os.environ, {"CLAWD_CHANGE_LEDGER": str(ledger)}):
                append_change("Write", str(Path(tmp) / "absolute.txt"), "create")
                append_change("Write", "../escape.txt", "create")
                append_change("Write", "safe/file.txt", "create")

            events = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["path"], "safe/file.txt")

    def test_empty_and_malformed_ledger_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "activity.jsonl"
            with patch.dict(os.environ, {"CLAWD_ACTIVITY_LEDGER": str(ledger)}):
                self.assertEqual(dict(activity_counts()["skill"]), {})
                ledger.write_text('not-json\n' + json.dumps({"kind": "tool", "name": ""}) + '\n', encoding="utf-8")
                counts = activity_counts()

        self.assertEqual(dict(counts["skill"]), {})
        self.assertEqual(dict(counts["tool"]), {})


if __name__ == "__main__":
    unittest.main()
