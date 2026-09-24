from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolPermissionError
from src.tool_system.permission_policy import (
    PermissionPolicyConfigError,
    load_permission_context,
)
from src.tool_system.protocol import ToolCall


class PermissionPolicyTests(unittest.TestCase):
    def _write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_policies_use_restrictive_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            ctx = load_permission_context(
                root,
                operator_policy_path=root / "missing-operator.json",
                project_policy_path=root / ".clawd" / "missing-project.json",
            )

            self.assertEqual(ctx.workspace_root, root)
            self.assertEqual(ctx.deny_names, frozenset())
            self.assertEqual(ctx.deny_prefixes, ())
            self.assertEqual(ctx.additional_working_directories, ())
            self.assertFalse(ctx.allow_docs)
            self.assertFalse(ctx.allow_docs_locked_off)

    def test_operator_policy_can_grant_bounded_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as extra_tmp:
            root = Path(tmp).resolve()
            extra = Path(extra_tmp).resolve()
            operator = root / "operator.json"
            self._write(
                operator,
                {
                    "schema_version": 1,
                    "deny_tools": ["Write"],
                    "deny_tool_prefixes": ["Web"],
                    "additional_working_directories": [str(extra)],
                    "allow_docs": True,
                },
            )

            ctx = load_permission_context(
                root,
                operator_policy_path=operator,
                project_policy_path=root / ".clawd" / "missing.json",
            )

            self.assertEqual(ctx.deny_names, frozenset({"write"}))
            self.assertEqual(ctx.deny_prefixes, ("web",))
            self.assertEqual(ctx.additional_working_directories, (extra,))
            self.assertTrue(ctx.allow_docs)
            self.assertEqual(ctx.ensure_path_allowed(extra / "ok.txt"), extra / "ok.txt")

    def test_project_policy_only_reduces_operator_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as extra_tmp:
            root = Path(tmp).resolve()
            extra = Path(extra_tmp).resolve()
            operator = root / "operator.json"
            project = root / ".clawd" / "permissions.json"
            self._write(
                operator,
                {
                    "schema_version": 1,
                    "deny_tools": ["Write"],
                    "deny_tool_prefixes": ["Web"],
                    "additional_working_directories": [str(extra)],
                    "allow_docs": True,
                },
            )
            self._write(
                project,
                {
                    "schema_version": 1,
                    "deny_tools": ["Edit"],
                    "deny_tool_prefixes": ["MCP"],
                    "allow_docs": False,
                },
            )

            ctx = load_permission_context(
                root,
                operator_policy_path=operator,
                project_policy_path=project,
            )

            self.assertEqual(ctx.deny_names, frozenset({"write", "edit"}))
            self.assertEqual(ctx.deny_prefixes, ("mcp", "web"))
            self.assertEqual(ctx.additional_working_directories, (extra,))
            self.assertFalse(ctx.allow_docs)
            self.assertTrue(ctx.allow_docs_locked_off)

    def test_project_policy_cannot_enable_allow_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            project = root / ".clawd" / "permissions.json"
            self._write(project, {"schema_version": 1, "allow_docs": True})

            with self.assertRaisesRegex(PermissionPolicyConfigError, "cannot enable allow_docs"):
                load_permission_context(
                    root,
                    operator_policy_path=root / "missing.json",
                    project_policy_path=project,
                )

    def test_project_policy_rejects_authority_expanding_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            project = root / ".clawd" / "permissions.json"
            self._write(
                project,
                {
                    "schema_version": 1,
                    "additional_working_directories": [str(root)],
                },
            )

            with self.assertRaisesRegex(PermissionPolicyConfigError, "unsupported project"):
                load_permission_context(
                    root,
                    operator_policy_path=root / "missing.json",
                    project_policy_path=project,
                )

    def test_project_policy_must_resolve_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp).resolve()
            outside = Path(outside_tmp).resolve() / "permissions.json"
            self._write(outside, {"schema_version": 1})

            with self.assertRaisesRegex(PermissionPolicyConfigError, "inside the workspace"):
                load_permission_context(
                    root,
                    operator_policy_path=root / "missing.json",
                    project_policy_path=outside,
                )

    def test_operator_additional_working_directories_must_be_absolute_existing_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            operator = root / "operator.json"
            self._write(
                operator,
                {
                    "schema_version": 1,
                    "additional_working_directories": ["relative/path"],
                },
            )
            with self.assertRaisesRegex(PermissionPolicyConfigError, "must be absolute"):
                load_permission_context(
                    root,
                    operator_policy_path=operator,
                    project_policy_path=root / ".clawd" / "missing.json",
                )

            self._write(
                operator,
                {
                    "schema_version": 1,
                    "additional_working_directories": [str(root / "does-not-exist")],
                },
            )
            with self.assertRaisesRegex(PermissionPolicyConfigError, "does not exist"):
                load_permission_context(
                    root,
                    operator_policy_path=operator,
                    project_policy_path=root / ".clawd" / "missing.json",
                )

    def test_invalid_policy_schema_and_unknown_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            operator = root / "operator.json"
            self._write(operator, {"schema_version": 999})
            with self.assertRaisesRegex(PermissionPolicyConfigError, "schema_version"):
                load_permission_context(
                    root,
                    operator_policy_path=operator,
                    project_policy_path=root / ".clawd" / "missing.json",
                )

            self._write(operator, {"schema_version": 1, "mystery": True})
            with self.assertRaisesRegex(PermissionPolicyConfigError, "unsupported operator"):
                load_permission_context(
                    root,
                    operator_policy_path=operator,
                    project_policy_path=root / ".clawd" / "missing.json",
                )

    def test_effective_denies_are_enforced_by_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "blocked.txt"
            target.write_text("hello", encoding="utf-8")
            project = root / ".clawd" / "permissions.json"
            self._write(project, {"schema_version": 1, "deny_tools": ["Read"]})
            permission_context = load_permission_context(
                root,
                operator_policy_path=root / "missing.json",
                project_policy_path=project,
            )
            from src.tool_system.context import ToolContext

            ctx = ToolContext(workspace_root=root, permission_context=permission_context)
            registry = build_default_registry(include_user_tools=False)
            with self.assertRaises(ToolPermissionError):
                registry.dispatch(
                    ToolCall(name="Read", input={"file_path": str(target)}),
                    ctx,
                )


if __name__ == "__main__":
    unittest.main()
