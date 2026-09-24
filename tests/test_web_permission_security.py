from __future__ import annotations

import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.permission_handler import PermissionBehavior
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools.web_fetch import WebFetchTool
from src.tool_system.tools.web_search import WebSearchTool


class TestOutboundWebPermissionSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = ToolContext(workspace_root=Path.cwd())

    def test_web_fetch_requires_user_permission(self) -> None:
        result = WebFetchTool().check_permissions({"url": "https://example.com/?x=secret"}, self.ctx)
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("https://example.com/?x=secret", result.message or "")

    def test_web_search_requires_user_permission(self) -> None:
        result = WebSearchTool().check_permissions({"query": "private project text"}, self.ctx)
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("private project text", result.message or "")

    def test_registry_denies_outbound_web_without_permission_handler(self) -> None:
        fetch_registry = ToolRegistry([WebFetchTool()])
        search_registry = ToolRegistry([WebSearchTool()])
        fetch = fetch_registry.dispatch(
            ToolCall(name="WebFetch", input={"url": "https://example.com/"}),
            self.ctx,
        )
        search = search_registry.dispatch(
            ToolCall(name="WebSearch", input={"query": "private project text"}),
            self.ctx,
        )
        self.assertTrue(fetch.is_error)
        self.assertTrue(search.is_error)
        self.assertIn("allow outbound web request", str(fetch.output).lower())
        self.assertIn("allow web search", str(search.output).lower())


if __name__ == "__main__":
    unittest.main()
