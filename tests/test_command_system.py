"""
Comprehensive tests for the command system.

Tests cover:
- Command type system
- Argument substitution
- Command registry
- Command execution engine
- Built-in commands
- Skills integration
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.command_system import (
    CLEAR_COMMAND,
    COMPACT_COMMAND,
    CONTEXT_COMMAND,
    COST_COMMAND,
    EXIT_COMMAND,
    HELP_COMMAND,
    INIT_COMMAND,
    SKILLS_COMMAND,
    CommandAvailability,
    CommandContext,
    CommandEngine,
    CommandRegistry,
    CommandResult,
    CommandType,
    LocalCommand,
    LocalCommandResult,
    PromptCommand,
    create_command_context,
    execute_command_async,
    execute_command_sync,
    find_commands,
    get_command,
    get_command_name,
    get_command_registry,
    has_command,
    is_command_enabled,
    list_commands,
    meets_availability_requirement,
    parse_argument_names,
    register_builtin_commands,
    register_command,
    substitute_arguments,
)
from src.cost_tracker import CostTracker
from src.history import HistoryLog
from src.skills.trust_registry import SkillTrustRegistry, default_skill_record


@dataclass
class MockConversation:
    """Mock conversation for testing."""

    messages: list = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def clear(self):
        self.messages.clear()


class TestArgumentSubstitution(unittest.TestCase):
    """Tests for argument substitution."""

    def test_simple_positional_args(self):
        """Test simple positional argument substitution."""
        content = "Hello $0 and $1!"
        result = substitute_arguments(content, "Alice Bob")
        self.assertEqual(result, "Hello Alice and Bob!")

    def test_named_args(self):
        """Test named argument substitution."""
        content = "Hello $name, you are $age years old"
        result = substitute_arguments(content, "Alice 30", ["name", "age"])
        self.assertEqual(result, "Hello Alice, you are 30 years old")

    def test_all_args_placeholder(self):
        """Test $ARGUMENTS placeholder."""
        content = "Args: $ARGUMENTS"
        result = substitute_arguments(content, 'foo bar "baz qux"')
        self.assertEqual(result, 'Args: foo bar "baz qux"')

    def test_parse_argument_names_string(self):
        """Test parsing argument names from string."""
        self.assertEqual(
            parse_argument_names("name, age, location"),
            ["name", "age", "location"],
        )

    def test_parse_argument_names_list(self):
        """Test parsing argument names from list."""
        self.assertEqual(
            parse_argument_names(["name", "age"]),
            ["name", "age"],
        )


class TestCommandTypes(unittest.TestCase):
    """Tests for the command type system."""

    def test_prompt_command_creation(self):
        """Test creating a PromptCommand."""
        cmd = PromptCommand(
            name="test-prompt",
            description="Test prompt command",
            progress_message="Testing...",
            content_length=100,
            markdown_content="# Test\n\nHello world",
        )
        self.assertEqual(cmd.command_type, CommandType.PROMPT)
        self.assertEqual(cmd.name, "test-prompt")
        self.assertEqual(cmd.progress_message, "Testing...")

    def test_local_command_creation(self):
        """Test creating a LocalCommand."""
        def mock_call(args: str, context: CommandContext) -> LocalCommandResult:
            return LocalCommandResult(type="text", value=f"Called with: {args}")

        cmd = LocalCommand(
            name="test-local",
            description="Test local command",
            supports_non_interactive=True,
        )
        cmd.set_call(mock_call)

        self.assertEqual(cmd.command_type, CommandType.LOCAL)
        self.assertEqual(cmd.name, "test-local")

    def test_command_enabled_check(self):
        """Test command enabled check."""
        enabled = True

        def check_enabled() -> bool:
            nonlocal enabled
            return enabled

        cmd = PromptCommand(
            name="test",
            description="Test",
            is_enabled=check_enabled,
        )

        self.assertTrue(is_command_enabled(cmd))
        enabled = False
        self.assertFalse(is_command_enabled(cmd))


class TestCommandRegistry(unittest.TestCase):
    """Tests for the command registry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = CommandRegistry()

    def tearDown(self):
        """Clean up test fixtures."""
        get_command_registry().clear()

    def test_register_and_get_command(self):
        """Test registering and getting a command."""
        cmd = PromptCommand(
            name="test",
            description="Test command",
        )
        self.registry.register(cmd)

        retrieved = self.registry.get("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")

    def test_register_with_alias(self):
        """Test registering a command with aliases."""
        cmd = PromptCommand(
            name="test",
            description="Test command",
            aliases=["t", "testing"],
        )
        self.registry.register(cmd)

        self.assertIsNotNone(self.registry.get("t"))
        self.assertIsNotNone(self.registry.get("testing"))
        self.assertEqual(self.registry.get("t").name, "test")

    def test_list_commands(self):
        """Test listing commands."""
        cmd1 = PromptCommand(name="test1", description="Test 1")
        cmd2 = PromptCommand(name="test2", description="Test 2", is_hidden=True)
        self.registry.register(cmd1)
        self.registry.register(cmd2)

        commands = self.registry.list_commands()
        self.assertEqual(len(commands), 1)

    def test_find_commands(self):
        """Test finding commands by search."""
        cmd1 = PromptCommand(name="help", description="Show help")
        cmd2 = PromptCommand(name="hello", description="Say hello")
        self.registry.register(cmd1)
        self.registry.register(cmd2)

        matches = self.registry.find_commands("he")
        self.assertEqual(len(matches), 2)


class TestBuiltinCommands(unittest.TestCase):
    """Tests for built-in commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def test_register_builtin_commands(self):
        """Test canonical command names and compatibility aliases."""
        registry = CommandRegistry()
        register_builtin_commands(registry)

        for name in (
            "help",
            "clear-chat",
            "exit",
            "list-skills",
            "session-usage",
            "usage",
            "context-usage",
            "doctor",
            "compact-context",
            "setup-project",
        ):
            self.assertTrue(registry.has(name), name)

        self.assertFalse(registry.has("api-usage"))

        for alias in (
            "clear",
            "reset",
            "new",
            "quit",
            "q",
            "skills",
            "cost",
            "context",
            "compact",
            "init",
            "?",
        ):
            self.assertTrue(registry.has(alias), alias)

    def test_doctor_reports_healthy_reconciled_state(self):
        healthy = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read", "Write"],
                "expected_registered": ["Read", "Write"],
                "metadata": {},
                "issues": [],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 4},
                "records": {
                    "reviewer": {
                        "review_status": "approved",
                        "activation_status": "active",
                    }
                },
                "issues": [],
            },
            "deferred_features": ["subagent_runtime"],
        }

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch("src.capabilities.reconcile_capabilities", return_value=healthy):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** PASS", result)
        self.assertIn("**Tools/runtime:** PASS — 2/2 expected tools registered", result)
        self.assertIn("**Skills/trust:** PASS — 1 active trusted skill(s); audit chain valid", result)
        self.assertIn("**Deferred capabilities:** subagent_runtime", result)
        self.assertIn("**Network/provider checks:** not run; doctor is local and read-only", result)

    def test_doctor_shows_static_osv_line_without_any_network(self):
        """A5: /doctor describes OSV locally and never contacts it."""
        healthy = {
            "manifest_schema_version": 1,
            "tools": {"registered": ["OsvQuery"], "expected_registered": ["OsvQuery"], "metadata": {}, "issues": []},
            "skills": {"audit_chain": {"valid": True, "entries": 1}, "records": {}, "issues": []},
            "deferred_features": [],
        }

        with patch.dict(
            os.environ,
            {"HOME": str(self.workspace_root), "USERPROFILE": str(self.workspace_root)},
            clear=False,
        ), patch("src.capabilities.reconcile_capabilities", return_value=healthy), \
             patch("socket.create_connection") as connect, patch("socket.getaddrinfo") as dns, \
             patch("src.osv_evidence._open_connection") as osv_open, \
             patch("src.osv_evidence.lookup") as osv_lookup:
            success, result, error = execute_command_sync("doctor", "", self.context)

        for probe in (connect, dns, osv_open, osv_lookup):
            probe.assert_not_called()
        self.assertTrue(success)
        self.assertIn(
            "**Software evidence (OSV):** OsvQuery configured for api.osv.dev only (query, vuln); "
            "approval required per call; one request per call; no retries, redirects or proxy; "
            "/doctor makes no OSV call",
            result,
        )
        self.assertIn("**Network/provider checks:** not run; doctor is local and read-only", result)

    def test_doctor_reports_sanitized_observability_without_failing_on_history(self):
        healthy = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read"],
                "expected_registered": ["Read"],
                "metadata": {},
                "issues": [],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 2},
                "records": {},
                "issues": [],
            },
            "plugins": {"active": [], "records": {}, "issues": []},
            "deferred_features": [],
        }
        observability = {
            "tool_calls": 7,
            "tool_errors": 2,
            "skill_calls": 1,
            "skill_errors": 0,
            "change_events": 3,
            "provider_events": 4,
            "provider_failures": 1,
            "usage_events": 5,
            "issues": [],
            "recent_errors": [
                {
                    "timestamp": "2026-09-23T19:00:00+00:00",
                    "area": "tool",
                    "name": "Read",
                },
                {
                    "timestamp": "2026-09-23T18:59:00+00:00",
                    "area": "provider",
                    "provider": "Demo",
                    "operation": "chat",
                    "stage": "provider_attempt",
                    "error_type": "TimeoutError",
                    "status_code": 504,
                },
            ],
        }

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch(
            "src.capabilities.reconcile_capabilities",
            return_value=healthy,
        ), patch(
            "src.command_system.builtins.runtime_observability_snapshot",
            return_value=observability,
        ):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** PASS", result)
        self.assertIn("**Observability:** PASS — 7 tool call(s), 2 tool error(s)", result)
        self.assertIn("## Recent Runtime Errors", result)
        self.assertIn("tool Read reported an error", result)
        self.assertIn("provider Demo chat/provider_attempt: TimeoutError (HTTP 504)", result)

    def test_doctor_reports_observability_ledger_health_issue(self):
        healthy = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read"],
                "expected_registered": ["Read"],
                "metadata": {},
                "issues": [],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 2},
                "records": {},
                "issues": [],
            },
            "plugins": {"active": [], "records": {}, "issues": []},
            "deferred_features": [],
        }
        observability = {
            "tool_calls": 0,
            "tool_errors": 0,
            "skill_calls": 0,
            "skill_errors": 0,
            "change_events": 0,
            "provider_events": 0,
            "provider_failures": 0,
            "usage_events": 0,
            "recent_errors": [],
            "issues": [
                {
                    "code": "observability_ledger_malformed",
                    "subject": "activity: 1 malformed event(s)",
                }
            ],
        }

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch(
            "src.capabilities.reconcile_capabilities",
            return_value=healthy,
        ), patch(
            "src.command_system.builtins.runtime_observability_snapshot",
            return_value=observability,
        ):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** ISSUES FOUND", result)
        self.assertIn("**Observability:** ISSUE", result)
        self.assertIn("observability/observability_ledger_malformed", result)
        self.assertIn("Review or archive the malformed local JSONL ledger", result)

    def test_doctor_reports_plugin_integrity_issue(self):
        degraded = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read"],
                "expected_registered": ["Read"],
                "metadata": {},
                "issues": [],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 2},
                "records": {},
                "issues": [],
            },
            "plugins": {
                "active": [],
                "records": {"demo": {"state": "review_required"}},
                "issues": [
                    {"code": "plugin_integrity_mismatch", "subject": "demo"}
                ],
            },
            "deferred_features": [],
        }

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch("src.capabilities.reconcile_capabilities", return_value=degraded):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** ISSUES FOUND", result)
        self.assertIn("**Python plugins:** ISSUE", result)
        self.assertIn("plugin/plugin_integrity_mismatch", result)
        self.assertIn("re-review and pin the new exact hash", result)

    def test_doctor_reports_plugin_runtime_registration_issue(self):
        healthy = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read"],
                "expected_registered": ["Read"],
                "metadata": {},
                "issues": [],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 2},
                "records": {},
                "issues": [],
            },
            "plugins": {
                "active": ["demo"],
                "records": {"demo": {"state": "active"}},
                "issues": [],
            },
            "deferred_features": [],
        }
        self.context.config["plugin_runtime_issues"] = [
            {
                "code": "plugin_extension_registration_failed",
                "subject": "demo: command name/alias collision",
            }
        ]

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch("src.capabilities.reconcile_capabilities", return_value=healthy):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** ISSUES FOUND", result)
        self.assertIn("**Python plugins:** ISSUE", result)
        self.assertIn("plugin/plugin_extension_registration_failed", result)
        self.assertIn("conflicts with runtime command/tool contracts", result)

    def test_doctor_reports_runtime_issue_with_actionable_hint(self):
        degraded = {
            "manifest_schema_version": 1,
            "tools": {
                "registered": ["Read"],
                "expected_registered": ["Read"],
                "metadata": {},
                "issues": [
                    {
                        "code": "lsp_runtime_unavailable",
                        "subject": "Pyright 1.1.414",
                    }
                ],
            },
            "skills": {
                "audit_chain": {"valid": True, "entries": 2},
                "records": {},
                "issues": [],
            },
            "deferred_features": [],
        }

        with patch.dict(
            os.environ,
            {
                "HOME": str(self.workspace_root),
                "USERPROFILE": str(self.workspace_root),
            },
            clear=False,
        ), patch("src.capabilities.reconcile_capabilities", return_value=degraded):
            success, result, error = execute_command_sync("doctor", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Status:** ISSUES FOUND", result)
        self.assertIn("tool/lsp_runtime_unavailable", result)
        self.assertIn("HOME/USERPROFILE", result)
        self.assertIn("~/.clawd/lsp/pyright", result)

    def test_context_usage_uses_live_agent_preflight(self):
        from src.agent.conversation import Conversation
        from src.tool_system.agent_loop import build_agent_preflight
        from src.tool_system.context import ToolContext
        from src.tool_system.defaults import build_default_registry

        class ContextProvider:
            model = "context-test-model"

        conversation = Conversation()
        conversation.add_user_message("inspect the current workspace context")
        provider = ContextProvider()
        tool_registry = build_default_registry(include_user_tools=False)
        tool_context = ToolContext(workspace_root=self.workspace_root)
        context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
            config={
                "context_provider": provider,
                "context_tool_registry": tool_registry,
                "context_tool_context": tool_context,
            },
        )
        preflight = build_agent_preflight(
            conversation,
            provider,
            tool_registry,
            tool_context,
        )

        success, result, error = execute_command_sync("context", "", context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("**Model:** context-test-model", result)
        self.assertIn("System prompt", result)
        self.assertIn("System tools", result)
        self.assertIn(f"**Tokens:** {preflight.estimated_input_tokens:,} /", result)

    def test_session_usage_reports_token_breakdown(self):
        """Test that /session-usage shows named provider/model session totals."""
        self.cost_tracker.record_usage(
            "Claude (claude-sonnet-4-6)",
            input_tokens=120,
            output_tokens=30,
        )
        self.cost_tracker.record_usage(
            "Gemini (gemini-3.8-flash)",
            input_tokens=200,
            output_tokens=20,
            thought_tokens=10,
            tool_use_tokens=70,
            total_tokens=300,
        )
        success, result, error = execute_command_sync("session-usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("JR Session Token Usage:", result)
        self.assertIn("Claude (claude-sonnet-4-6):", result)
        self.assertIn("Gemini (gemini-3.8-flash):", result)
        self.assertIn("Thought tokens:  10", result)
        self.assertIn("Tool-use tokens: 70", result)
        self.assertIn("Combined tracked:", result)
        self.assertIn("Input tokens:  320", result)
        self.assertIn("Output tokens: 50", result)
        self.assertIn("Total tokens:  450", result)
        self.assertNotIn("API credit remaining", result)

        alias_success, alias_result, alias_error = execute_command_sync("cost", "", self.context)
        self.assertTrue(alias_success)
        self.assertIsNone(alias_error)
        self.assertEqual(alias_result, result)

    def test_usage_does_not_fake_zero_provider_usage(self):
        """Test that /usage reports unavailable provider access instead of fake zeros."""
        env = {
            "ANTHROPIC_ADMIN_KEY": "",
            "ANTHROPIC_API_KEY": "normal-key",
            "GEMINI_API_KEY": "gemini-key",
        }
        with patch.dict(os.environ, env, clear=False):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(result.startswith("Usage\n\nAPI / Model Usage\n"))
        self.assertIn("Skill Activity", result)
        self.assertIn("Tool Activity", result)
        self.assertIn("not separate billing or API-usage categories", result)
        self.assertIn("ANTHROPIC_ADMIN_KEY is not configured", result)
        self.assertIn("GEMINI_API_KEY alone", result)
        self.assertNotIn("Total tokens:  0", result)

    def test_usage_reports_local_deepseek_provider_usage(self):
        """DeepSeek /usage uses exact request usage captured in JR's ledger."""
        deepseek = {
            "DeepSeek (deepseek-flash)": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "thought_tokens": 50,
                "tool_use_tokens": 0,
                "cached_tokens": 400,
                "total_tokens": 1200,
            }
        }
        env = {
            "ANTHROPIC_ADMIN_KEY": "",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "GOOGLE_OAUTH_CLIENT_FILE": "",
            "GEMINI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins.month_to_date_provider_usage",
            side_effect=lambda prefix: deepseek if prefix == "DeepSeek (" else {},
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("DeepSeek:", result)
        self.assertIn("DeepSeek (deepseek-flash):", result)
        self.assertIn("Input tokens:     1,000", result)
        self.assertIn("Output tokens:    200", result)
        self.assertIn("Cached input:     400", result)
        self.assertIn("Reasoning tokens: 50", result)
        self.assertIn("Total tokens:     1,200", result)

    def test_usage_reports_local_qwen_provider_usage(self):
        """Qwen /usage uses exact request usage captured in JR's ledger."""
        qwen = {
            "Qwen (qwen3.8-max)": {
                "input_tokens": 1500,
                "output_tokens": 250,
                "thought_tokens": 60,
                "tool_use_tokens": 0,
                "cached_tokens": 500,
                "total_tokens": 1750,
            }
        }
        env = {
            "ANTHROPIC_ADMIN_KEY": "",
            "DASHSCOPE_API_KEY": "qwen-key",
            "GOOGLE_OAUTH_CLIENT_FILE": "",
            "GEMINI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins.month_to_date_provider_usage",
            side_effect=lambda prefix: qwen if prefix == "Qwen (" else {},
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("Qwen:", result)
        self.assertIn("Qwen (qwen3.8-max):", result)
        self.assertIn("Input tokens:     1,500", result)
        self.assertIn("Output tokens:    250", result)
        self.assertIn("Cached input:     500", result)
        self.assertIn("Reasoning tokens: 60", result)
        self.assertIn("Total tokens:     1,750", result)

    def test_usage_formats_real_skill_and_tool_activity_with_canonical_names(self):
        env = {"ANTHROPIC_ADMIN_KEY": "", "GEMINI_API_KEY": "", "GOOGLE_OAUTH_CLIENT_FILE": ""}
        activity = {
            "skill": {"requesting-code-review": 3, "systematic-debugging": 1},
            "tool": {"Read": 4, "Edit": 2},
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins.activity_counts", return_value=activity
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("requesting-code-review: 3 uses", result)
        self.assertIn("systematic-debugging: 1 use", result)
        self.assertIn("Total Skill Uses: 4", result)
        self.assertIn("Read: 4 uses", result)
        self.assertIn("Edit: 2 uses", result)
        self.assertIn("Total Tool Uses: 6", result)
        self.assertNotIn("skill_", result)
        self.assertNotIn("SHA-256", result)

    def test_usage_activity_empty_state(self):
        env = {"ANTHROPIC_ADMIN_KEY": "", "GEMINI_API_KEY": "", "GOOGLE_OAUTH_CLIENT_FILE": ""}
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins.activity_counts",
            return_value={"skill": {}, "tool": {}},
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("No skill activity recorded yet.", result)
        self.assertIn("Total Skill Uses: 0", result)
        self.assertIn("No tool activity recorded yet.", result)
        self.assertIn("Total Tool Uses: 0", result)

    def test_usage_reports_real_anthropic_values_when_available(self):
        """Test that provider-side usage values are rendered from the Anthropic usage API."""
        actual = {
            "claude-sonnet-4-6": {
                "input_tokens": 1234,
                "output_tokens": 56,
            }
        }
        env = {
            "ANTHROPIC_ADMIN_KEY": "admin-key",
            "GEMINI_API_KEY": "",
        }
        self.context.permission_handler = lambda name, message, suggestion: (True, False)
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins._anthropic_month_to_date_usage",
            return_value=(actual, 1.25),
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("claude-sonnet-4-6:", result)
        self.assertIn("Input tokens:  1,234", result)
        self.assertIn("Output tokens: 56", result)
        self.assertIn("Total tokens:  1,290", result)
        self.assertIn("Month-to-date spend: $1.25 USD", result)


    def test_usage_requires_authorization_before_provider_retrieval(self):
        env = {
            "ANTHROPIC_ADMIN_KEY": "admin-key",
            "GOOGLE_OAUTH_CLIENT_FILE": "",
        }
        self.context.permission_handler = lambda name, message, suggestion: (False, False)
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins._anthropic_month_to_date_usage"
        ) as anthropic_usage, patch(
            "src.command_system.builtins.activity_counts",
            return_value={"skill": {}, "tool": {}},
        ):
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        anthropic_usage.assert_not_called()
        self.assertIn("Provider retrieval not authorized", result)
        self.assertIn("Skill Activity", result)
        self.assertIn("Tool Activity", result)

    def test_usage_without_permission_handler_never_retrieves_provider_usage(self):
        env = {
            "ANTHROPIC_ADMIN_KEY": "admin-key",
            "GOOGLE_OAUTH_CLIENT_FILE": "",
        }
        self.context.permission_handler = None
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins._anthropic_month_to_date_usage"
        ) as anthropic_usage:
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        anthropic_usage.assert_not_called()
        self.assertIn("Provider retrieval not authorized", result)

    def test_usage_google_denial_does_not_retrieve_provider_usage(self):
        env = {
            "ANTHROPIC_ADMIN_KEY": "",
            "GOOGLE_OAUTH_CLIENT_FILE": "client.json",
            "GOOGLE_OAUTH_TOKEN_FILE": "token.json",
        }
        self.context.permission_handler = lambda name, message, suggestion: (False, False)
        with patch.dict(os.environ, env, clear=False), patch(
            "src.command_system.builtins._google_month_to_date_gemini_usage"
        ) as google_usage:
            success, result, error = execute_command_sync("usage", "", self.context)

        self.assertTrue(success)
        self.assertIsNone(error)
        google_usage.assert_not_called()
        self.assertIn("Provider retrieval not authorized", result)

    def test_google_usage_credentials_do_not_start_oauth_enrollment(self):
        from src.command_system.builtins import _google_user_credentials

        env = {"GOOGLE_OAUTH_TOKEN_FILE": ""}
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(RuntimeError, "will not start OAuth enrollment"):
                _google_user_credentials()

    def test_anthropic_usage_parser_uses_provider_report_values(self):
        """Test month-to-date provider usage parsing without making a network request."""
        from src.command_system.builtins import _anthropic_month_to_date_usage

        usage_payload = {
            "data": [{
                "results": [{
                    "model": "claude-sonnet-4-6",
                    "uncached_input_tokens": 100,
                    "cache_read_input_tokens": 20,
                    "cache_creation": {"ephemeral_1h_input_tokens": 30},
                    "output_tokens": 10,
                }]
            }]
        }
        cost_payload = {
            "data": [{
                "results": [{"amount": "125.0", "currency": "USD"}]
            }]
        }
        with patch(
            "src.command_system.builtins._anthropic_admin_get",
            side_effect=[usage_payload, cost_payload],
        ):
            by_model, usd = _anthropic_month_to_date_usage("admin-key")

        self.assertEqual(by_model["claude-sonnet-4-6"]["input_tokens"], 150)
        self.assertEqual(by_model["claude-sonnet-4-6"]["output_tokens"], 10)
        self.assertEqual(float(usd), 1.25)

    def test_sync_compact_internal_failure_preserves_history(self):
        from src.command_system.builtins import _sync_compact_fallback

        original = [object() for _ in range(12)]
        self.conversation.messages = list(original)

        result = _sync_compact_fallback(self.context)

        self.assertEqual(result.type, "text")
        self.assertIn("conversation preserved", result.value)
        self.assertEqual(self.conversation.messages, original)

    def test_compact_records_named_provider_usage(self):
        from src.command_system.builtins import _compact_async

        class AnthropicProviderFake:
            pass

        provider = AnthropicProviderFake()
        self.context.config["provider"] = provider
        self.context.config["model"] = "claude-sonnet-4-6"
        self.conversation.messages = [object(), object()]

        compact_result = MagicMock()
        compact_result.user_display_message = "Conversation compacted."
        compact_result.usage = {"input_tokens": 900, "output_tokens": 100}
        compact_result.pre_compact_count = 10
        compact_result.post_compact_count = 2
        compact_result.tokens_saved = 5000
        compact_result.trigger = "manual"
        compact_result.summary_text = "Summary"

        with patch(
            "src.compact_service.service.compact_conversation",
            new=AsyncMock(return_value=compact_result),
        ):
            result = asyncio.run(_compact_async("", self.context))

        self.assertIn("Claude (claude-sonnet-4-6)", result.value)
        self.assertIn("Usage this task:", result.value)
        usage = self.cost_tracker.provider_usage["Claude (claude-sonnet-4-6)"]
        self.assertEqual(usage["input_tokens"], 900)
        self.assertEqual(usage["output_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 1000)

    def test_skills_command_with_project_root(self):
        """Test that /skills command can find project skills."""
        from src.skills.create import create_skill

        # Create a skill in the temp directory
        project_skills_dir = Path(self.tmpdir.name) / ".clawd" / "skills"
        project_skills_dir.mkdir(parents=True)

        create_skill(
            directory=project_skills_dir,
            name="test-project-skill",
            description="A test skill in project",
            body="Hello from project skill",
        )

        # Create command context with the project root
        context = create_command_context(
            workspace_root=self.tmpdir.name,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        # Runtime skill discovery is fail-closed: approve + activate the exact test artifact.
        trust_dir = self.workspace_root / "trust"
        skill_root = project_skills_dir / "test-project-skill"
        trust = SkillTrustRegistry(trust_dir)
        record = default_skill_record(
            name="test-project-skill",
            artifact_path=skill_root,
            purpose="command-system test skill",
        )
        trust.register_quarantined(record, initiator="test", reason="test registration")
        trust.mark_reviewed(
            "test-project-skill",
            reviewed_by="test",
            initiator="test",
            reason="test review",
        )
        trust.approve(
            "test-project-skill",
            reviewed_by="test",
            initiator="test",
            reason="test approval",
        )
        trust.activate(
            "test-project-skill",
            initiator="test",
            reason="test activation",
        )

        # Execute /skills command.
        with patch.dict(os.environ, {"CLAWD_SKILL_TRUST_DIR": str(trust_dir)}):
            success, result, error = execute_command_sync("skills", "", context)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIn("test-project-skill", result)


class TestCommandEngine(unittest.IsolatedAsyncioTestCase):
    """Tests for the command engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.registry = CommandRegistry()
        register_builtin_commands(self.registry)

        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        self.engine = CommandEngine(
            registry=self.registry,
            workspace_root=self.workspace_root,
            context=self.context,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    async def test_execute_help_command(self):
        """Test executing /help command."""
        result = await self.engine.execute("/help")

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "help")

    async def test_execute_unknown_command(self):
        """Test executing unknown command."""
        result = await self.engine.execute("/unknown-command")

        self.assertFalse(result.success)
        self.assertIn("Unknown command", result.error or "")


class TestSkillsIntegration(unittest.TestCase):
    """Tests for skills system integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self.tmpdir.name) / "skills"
        self.skills_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def test_skill_to_prompt_command(self):
        """Test converting a skill to a prompt command."""
        from src.command_system.skills_integration import skill_to_prompt_command
        from src.skills.create import create_skill

        skill_path = create_skill(
            directory=self.skills_dir,
            name="test-skill",
            description="Test skill",
            when_to_use="When testing",
            allowed_tools=["Read", "Grep"],
            arguments=["name"],
            body="Hello $name",
        )

        from src.skills.loader import load_skills_from_dir

        skills = load_skills_from_dir(self.skills_dir)
        self.assertEqual(len(skills), 1)

        cmd = skill_to_prompt_command(skills[0])

        self.assertEqual(cmd.name, "test-skill")
        self.assertEqual(cmd.description, "Test skill")
        self.assertEqual(cmd.markdown_content, "Hello $name")


class TestInitCommand(unittest.IsolatedAsyncioTestCase):
    """Tests for the /init command implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.tmpdir.name).resolve()
        self.registry = CommandRegistry()
        register_builtin_commands(self.registry)

        self.conversation = MockConversation()
        self.cost_tracker = CostTracker()
        self.history = HistoryLog()

        self.context = create_command_context(
            workspace_root=self.workspace_root,
            conversation=self.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history,
        )

        self.engine = CommandEngine(
            registry=self.registry,
            workspace_root=self.workspace_root,
            context=self.context,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def _get_init_command(self):
        """Get the /init command from the local registry."""
        return self.registry.get("init")

    def test_init_command_is_prompt_command(self):
        """Test that /init is a PromptCommand, not LocalCommand."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertEqual(init_cmd.command_type, CommandType.PROMPT)
        self.assertIsInstance(init_cmd, PromptCommand)

    def test_init_command_has_correct_description(self):
        """Test that /init has the correct description."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertIn("CLAUDE.md", init_cmd.description)
        self.assertIn("skills", init_cmd.description)
        self.assertNotIn("hooks", init_cmd.description.lower())

    def test_init_command_does_not_offer_unavailable_hook_setup(self):
        """The setup advisor must not promise the intentionally disabled hook runtime."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        content = init_cmd.markdown_content.lower()
        self.assertNotIn("skills and hooks", content)
        self.assertNotIn("hooks only", content)
        self.assertNotIn("set up hooks", content)
        self.assertIn("also set up project skills?", content)

    def test_init_command_has_progress_message(self):
        """Test that /init has a progress message."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertEqual(init_cmd.progress_message, "analyzing your codebase")

    def test_init_command_has_prompt_content(self):
        """Test that /init has the 7-step prompt content."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        self.assertIsInstance(init_cmd, PromptCommand)
        # Verify it contains the key steps
        content = init_cmd.markdown_content
        self.assertIn("Step 1", content)
        self.assertIn("Step 2", content)
        self.assertIn("Step 3", content)
        self.assertIn("Step 4", content)
        self.assertIn("Step 5", content)
        self.assertIn("Step 6", content)
        self.assertIn("Step 7", content)

    def test_init_command_includes_claude_md_instructions(self):
        """Test that /init prompt includes CLAUDE.md creation instructions."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        content = init_cmd.markdown_content
        # Verify it includes key CLAUDE.md content requirements
        self.assertIn("CLAUDE.md", content)
        self.assertIn("build/test/lint", content.lower())
        self.assertIn("code style", content.lower())

    def test_init_command_includes_user_interaction_phases(self):
        """Test that /init prompt includes user interaction phases."""
        init_cmd = self._get_init_command()
        self.assertIsNotNone(init_cmd)
        content = init_cmd.markdown_content
        # Verify it includes AskUserQuestion references
        self.assertIn("AskUserQuestion", content)

    async def test_execute_init_command_via_engine(self):
        """Test executing /init via the async engine."""
        result = await self.engine.execute("/init")

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "setup-project")
        self.assertEqual(result.result_type, "prompt")
        self.assertTrue(result.should_query)
        self.assertEqual(result.display, "user")
        # Verify prompt content was returned
        self.assertTrue(len(result.prompt_content) > 0)
        self.assertEqual(result.prompt_content[0]["type"], "text")

    async def test_execute_init_command_async(self):
        """Test executing /init via execute_command_async."""
        # Register commands to global registry for this test
        from src.command_system import get_command_registry
        registry = get_command_registry()
        register_builtin_commands(registry)

        result = await execute_command_async("init", "", self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.command_name, "setup-project")
        self.assertEqual(result.result_type, "prompt")
        self.assertTrue(len(result.prompt_content) > 0)

    def test_sync_execute_does_not_handle_init(self):
        """Test that sync execution returns error for /init (it's a PromptCommand)."""
        # Register commands to global registry for this test
        from src.command_system import get_command_registry
        registry = get_command_registry()
        register_builtin_commands(registry)

        success, result, error = execute_command_sync("init", "", self.context)
        # Sync execution doesn't handle PromptCommand
        # It will either return False or an error
        if not success:
            # Expected: sync can't handle PromptCommand
            pass
        else:
            # If sync succeeds, it means LocalCommand handled it (shouldn't happen for /init)
            self.fail("/init should not be a LocalCommand")


if __name__ == "__main__":
    unittest.main()
