"""Tests for the permission system."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.tool_system.context import ToolContext
from src.tool_system.permission_handler import (
    InteractivePermissionHandler,
    PermissionBehavior,
    PermissionResult,
)
from src.tool_system.permissions import ToolPermissionContext
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools.write import FileWriteTool
from src.tool_system.tools.edit import FileEditTool


class TestPermissionResult(unittest.TestCase):
    def test_allow_returns_correct_behavior(self) -> None:
        result = PermissionResult.allow()
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)
        self.assertIsNone(result.message)
        self.assertIsNone(result.updated_input)

    def test_allow_with_updated_input(self) -> None:
        updated = {"key": "value"}
        result = PermissionResult.allow(updated_input=updated)
        self.assertEqual(result.updated_input, updated)

    def test_deny_returns_correct_behavior(self) -> None:
        result = PermissionResult.deny("test message")
        self.assertEqual(result.behavior, PermissionBehavior.DENY)
        self.assertEqual(result.message, "test message")

    def test_ask_returns_correct_behavior(self) -> None:
        result = PermissionResult.ask("test message", "try this")
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertEqual(result.message, "test message")
        self.assertEqual(result.suggestion, "try this")


class TestWriteToolPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        # Default context has allow_docs=False
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_check_permissions_allow_for_regular_file(self) -> None:
        """Regular files should be allowed without any permission check."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.txt"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_ask_for_md_file_when_docs_disallowed(self) -> None:
        """MD files should trigger 'ask' when allow_docs is False."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.md"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("allow_docs", result.message.lower())

    def test_check_permissions_allow_for_md_file_when_docs_allowed(self) -> None:
        """MD files should be allowed when allow_docs is True."""
        self.ctx.permission_context.allow_docs = True
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.md"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_allow_for_markdown_file(self) -> None:
        """Markdown files should also trigger 'ask' when allow_docs is False."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.markdown"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)


class TestEditToolPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        # Default context has allow_docs=False
        self.ctx = ToolContext(workspace_root=self.root)

        # Create a test file to edit
        self.test_file = self.root / "test.md"
        self.test_file.write_text("original content", encoding="utf-8")
        self.ctx.mark_file_read(self.test_file)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_check_permissions_allow_for_regular_file(self) -> None:
        """Regular files should be allowed without any permission check."""
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.txt"), "old_string": "a", "new_string": "b"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_ask_for_md_file_when_docs_disallowed(self) -> None:
        """MD files should trigger 'ask' when allow_docs is False."""
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.test_file), "old_string": "original", "new_string": "modified"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("allow_docs", result.message.lower())

    def test_check_permissions_allow_for_md_file_when_docs_allowed(self) -> None:
        """MD files should be allowed when allow_docs is True."""
        self.ctx.permission_context.allow_docs = True
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.test_file), "old_string": "original", "new_string": "modified"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)


class TestToolRegistryDispatchPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.registry = ToolRegistry([FileWriteTool()])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dispatch_allows_regular_file(self) -> None:
        """Regular files should work without permission handler."""
        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.txt"), "content": "hello"}),
            self.ctx,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output.get("type"), "create")

    def test_dispatch_denies_md_file_without_handler(self) -> None:
        """MD files should be denied when no permission handler is set."""
        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        error_msg = result.output.get("error", "").lower()
        self.assertTrue(
            "permission" in error_msg or "allow_docs" in error_msg or "blocked" in error_msg,
            f"Expected 'permission', 'allow_docs', or 'blocked' in error, got: {error_msg}",
        )

    def test_dispatch_calls_permission_handler_for_ask(self) -> None:
        """When behavior is 'ask', permission_handler should be called."""
        call_count = 0
        captured_args = None

        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            nonlocal call_count, captured_args
            call_count += 1
            captured_args = (tool_name, message, suggestion)
            return True, False  # Allow

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertEqual(call_count, 1)
        self.assertEqual(captured_args[0], "Write")
        self.assertIn("allow_docs", captured_args[1].lower())
        self.assertFalse(result.is_error)

    def test_dispatch_respects_handler_deny(self) -> None:
        """When permission handler denies, tool should be blocked."""
        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            return False, False  # Deny

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertTrue(result.is_error)
        self.assertIn("denied", result.output.get("error", "").lower())

    def test_dispatch_allows_after_handler_enables_setting(self) -> None:
        """When handler enables setting, tool should proceed."""
        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            # Enable allow_docs
            self.ctx.permission_context.allow_docs = True
            return True, False

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.output.get("type"), "create")


class TestInteractivePermissionHandler(unittest.TestCase):
    def test_project_lock_prevents_session_enable(self) -> None:
        ctx = ToolContext(workspace_root=Path.cwd())
        ctx.permission_context.allow_docs = False
        ctx.permission_context.allow_docs_locked_off = True
        handler = InteractivePermissionHandler(prompt_func=lambda _: "e")
        request = PermissionResult.ask(
            "Writing documentation files is blocked unless allow_docs is enabled"
        )

        self.assertIsNone(handler._can_enable_setting(request, ctx))
        handler._enable_setting(request, ctx)
        self.assertFalse(ctx.permission_context.allow_docs)


class TestPermissionContext(unittest.TestCase):
    def test_default_allow_docs_is_false(self) -> None:
        """By default, allow_docs should be False."""
        pc = ToolPermissionContext()
        self.assertFalse(pc.allow_docs)

    def test_allow_docs_can_be_set_true(self) -> None:
        """allow_docs can be set to True."""
        pc = ToolPermissionContext(allow_docs=True)
        self.assertTrue(pc.allow_docs)

    def test_permission_context_is_mutable(self) -> None:
        """ToolPermissionContext should be mutable to allow runtime changes."""
        pc = ToolPermissionContext()
        self.assertFalse(pc.allow_docs)
        pc.allow_docs = True
        self.assertTrue(pc.allow_docs)


if __name__ == "__main__":
    unittest.main()
