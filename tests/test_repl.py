"""Tests for REPL functionality."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from rich.markdown import Markdown
from prompt_toolkit.output import DummyOutput

from src.repl import ClawdREPL
from src.repl.core import _is_provider_authentication_error
from src.agent import Session, Conversation
from src.providers.base import BaseProvider, ChatMessage, ChatResponse
from src.providers import clear_plugin_providers, register_plugin_provider
from src.plugins.extensions import (
    PluginExtensionLoadResult,
    PluginExtensions,
    PluginProviderExtension,
)
from src.cost_tracker import CostTracker
from src.tool_system.context import ToolContext
from src.tool_system.mcp_resource_runtime import MCPResourceConfigError
from src.tool_system.permissions import ToolPermissionContext
from src.tool_system.protocol import ToolResult


class TestREPL(unittest.TestCase):
    """Test REPL functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Keep REPL behavior tests meaningful in headless/non-console runners.
        self._prompt_output_patch = patch(
            "prompt_toolkit.output.defaults.create_output",
            return_value=DummyOutput(),
        )
        self._prompt_output_patch.start()
        self.addCleanup(self._prompt_output_patch.stop)

        # Create a temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / ".clawd"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._ledger_env_patch = patch.dict(
            os.environ,
            {
                "CLAWD_ACTIVITY_LEDGER": str(Path(self.temp_dir) / "activity.jsonl"),
                "CLAWD_CHANGE_LEDGER": str(Path(self.temp_dir) / "changes.jsonl"),
            },
        )
        self._ledger_env_patch.start()
        self.addCleanup(self._ledger_env_patch.stop)

        self._permission_policy_patch = patch(
            "src.repl.core.load_permission_context",
            side_effect=lambda workspace_root: ToolPermissionContext(
                workspace_root=Path(workspace_root).resolve()
            ),
        )
        self._permission_policy_loader = self._permission_policy_patch.start()
        self.addCleanup(self._permission_policy_patch.stop)

        # Create a test config
        test_config = {
            "default_provider": "glm",
            "providers": {
                "glm": {
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5"
                }
            }
        }

        config_file = self.config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)

    def test_repl_initialization(self):
        """Test REPL initialization."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session.return_value = Mock()

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    self.assertIsNotNone(repl)
                    self.assertEqual(repl.provider_name, "glm")
                    self.assertFalse(repl.stream)
                    self.assertFalse(repl.multiline_mode)
                    self.assertTrue(repl.tool_context.instrumentation_enabled)
                    self._permission_policy_loader.assert_called_once()
                    self.assertIs(
                        repl.command_context.config["context_provider"],
                        repl.provider,
                    )
                    self.assertIs(
                        repl.command_context.config["context_tool_registry"],
                        repl.tool_registry,
                    )
                    self.assertIs(
                        repl.command_context.config["context_tool_context"],
                        repl.tool_context,
                    )
                    self.assertNotIn("provider", repl.command_context.config)

    def test_repl_starts_with_local_plugin_provider_without_api_key(self):
        constructed = []

        class LocalProvider(BaseProvider):
            def __init__(self, api_key, base_url=None, model=None):
                constructed.append((api_key, base_url, model))
                super().__init__(api_key, base_url, model)

            def chat(self, messages, tools=None, **kwargs):
                return ChatResponse(
                    content="local",
                    model=self.model,
                    usage={},
                    finish_reason="stop",
                )

            def chat_stream(self, messages, tools=None, **kwargs):
                if False:
                    yield ""

            def get_available_models(self):
                return ["local-model"]

        load_result = PluginExtensionLoadResult(
            plugins=[
                PluginExtensions(
                    plugin_name="local-plugin",
                    plugin_version="1.0.0",
                    artifact_sha256="a" * 64,
                    providers=[
                        PluginProviderExtension(
                            name="local-demo",
                            provider_class=LocalProvider,
                            info={
                                "label": "Local Demo",
                                "default_base_url": "http://127.0.0.1:11434/v1",
                                "default_model": "local-model",
                                "available_models": ["local-model"],
                                "requires_api_key": False,
                                "local_only": True,
                            },
                        )
                    ],
                )
            ]
        )

        session = Session(
            session_id="local-session",
            provider="local-demo",
            model="local-model",
        )
        try:
            with patch(
                'src.config.get_config_path',
                return_value=self.config_dir / "config.json",
            ), patch(
                'src.repl.core.load_active_plugin_extensions',
                return_value=load_result,
            ) as load_extensions, patch(
                'src.repl.core.Session.create',
                return_value=session,
            ):
                repl = ClawdREPL(provider_name="local-demo")

            self.assertEqual(repl.provider_name, "local-demo")
            self.assertIsInstance(repl.provider, LocalProvider)
            self.assertEqual(
                constructed,
                [("", "http://127.0.0.1:11434/v1", "local-model")],
            )
            self.assertEqual(load_extensions.call_count, 1)
            self.assertEqual(repl.plugin_extension_issues, [])
        finally:
            clear_plugin_providers()

    def test_relogin_local_plugin_provider_skips_api_key_and_validates_endpoint(self):
        constructed = []

        class LocalProvider(BaseProvider):
            def __init__(self, api_key, base_url=None, model=None):
                constructed.append((api_key, base_url, model))
                super().__init__(api_key, base_url, model)

            def chat(self, messages, tools=None, **kwargs):
                return ChatResponse(
                    content="local",
                    model=self.model,
                    usage={},
                    finish_reason="stop",
                )

            def chat_stream(self, messages, tools=None, **kwargs):
                if False:
                    yield ""

            def get_available_models(self):
                return ["local-model"]

        register_plugin_provider(
            "local-demo",
            LocalProvider,
            {
                "label": "Local Demo",
                "default_base_url": "http://127.0.0.1:11434/v1",
                "default_model": "local-model",
                "available_models": ["local-model"],
                "requires_api_key": False,
                "local_only": True,
            },
            plugin_name="test-plugin",
            artifact_sha256="a" * 64,
        )

        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.provider_name = "glm"
        old_provider = Mock()
        old_provider.model = "glm-4.5"
        repl.provider = old_provider
        repl.session = Mock()
        repl.session.provider = "glm"
        repl.session.model = "glm-4.5"
        repl.command_context = Mock()
        repl.command_context.config = {"context_provider": old_provider}
        try:
            with patch(
                "rich.prompt.Prompt.ask",
                side_effect=[
                    "local-demo",
                    "http://127.0.0.1:5555/v1",
                    "local-custom",
                ],
            ) as prompt, patch(
                "src.config.set_api_key"
            ) as set_api_key, patch(
                "src.config.set_default_provider"
            ) as set_default_provider, patch(
                "src.config.get_provider_config",
                return_value={
                    "api_key": "",
                    "base_url": "http://127.0.0.1:5555/v1",
                    "default_model": "local-custom",
                },
            ):
                result = repl._handle_relogin()

            self.assertTrue(result)
            self.assertEqual(prompt.call_count, 3)
            set_api_key.assert_called_once_with(
                "local-demo",
                api_key="",
                base_url="http://127.0.0.1:5555/v1",
                default_model="local-custom",
            )
            set_default_provider.assert_called_once_with("local-demo")
            self.assertEqual(repl.provider_name, "local-demo")
            self.assertIsInstance(repl.provider, LocalProvider)
            self.assertEqual(repl.session.provider, "local-demo")
            self.assertEqual(repl.session.model, "local-custom")
            self.assertIs(repl.command_context.config["context_provider"], repl.provider)
            self.assertEqual(
                constructed,
                [("", "http://127.0.0.1:5555/v1", "local-custom")],
            )
            self.assertEqual(
                prompt.call_args_list[1].kwargs["default"],
                "http://127.0.0.1:5555/v1",
            )
            self.assertEqual(prompt.call_args_list[2].kwargs["default"], "local-custom")
        finally:
            clear_plugin_providers()

    def test_auth_error_classifier_handles_status_class_message_and_chain(self):
        class StatusAuthError(RuntimeError):
            status_code = 401

        AuthenticationError = type("AuthenticationError", (RuntimeError,), {})

        wrapped = RuntimeError("outer")
        wrapped.__cause__ = StatusAuthError("provider rejected request")

        self.assertTrue(_is_provider_authentication_error(StatusAuthError("no details")))
        self.assertTrue(_is_provider_authentication_error(AuthenticationError("no details")))
        self.assertTrue(_is_provider_authentication_error(RuntimeError("invalid API key")))
        self.assertTrue(_is_provider_authentication_error(wrapped))

        forbidden = RuntimeError("forbidden")
        forbidden.status_code = 403
        self.assertFalse(_is_provider_authentication_error(forbidden))
        self.assertFalse(_is_provider_authentication_error(TimeoutError("provider timeout")))

    def test_chat_auth_failure_does_not_fall_through_to_second_provider_attempt(self):
        class StatusAuthError(RuntimeError):
            status_code = 401

        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                session = Mock()
                session.session_id = "current"
                session.conversation = Conversation()
                mock_session.return_value = session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    provider.chat.side_effect = StatusAuthError("secret-key-material")
                    mock_provider_class.return_value = lambda **_: provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.console.print = Mock()

                    with patch("rich.prompt.Prompt.ask", return_value="n"), patch(
                        "src.repl.core.run_agent_loop"
                    ) as run_agent_loop:
                        repl.chat("hello")

                    provider.chat.assert_called_once()
                    run_agent_loop.assert_not_called()
                    self.assertIsNone(repl._current_status)
                    self.assertEqual(repl.session.conversation.messages, [])
                    rendered = " ".join(
                        str(call.args[0])
                        for call in repl.console.print.call_args_list
                        if call.args
                    )
                    self.assertIn("Authentication Error", rendered)
                    self.assertNotIn("secret-key-material", rendered)

    def test_auth_failure_after_visible_stream_preserves_user_turn(self):
        class StatusAuthError(RuntimeError):
            status_code = 401

        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                session = Mock()
                session.session_id = "current"
                session.conversation = Conversation()
                mock_session.return_value = session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"

                    def stream_then_fail(messages, tools=None, on_text_chunk=None, **kwargs):
                        if on_text_chunk is not None:
                            on_text_chunk("partial")
                        raise StatusAuthError("rejected after visible output")

                    provider.chat_stream_response.side_effect = stream_then_fail
                    mock_provider_class.return_value = lambda **_: provider

                    repl = ClawdREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()

                    with patch("rich.prompt.Prompt.ask", return_value="n"):
                        repl.chat("hello")

                    self.assertEqual(len(repl.session.conversation.messages), 1)
                    self.assertEqual(repl.session.conversation.messages[0].role, "user")
                    self.assertEqual(repl.session.conversation.messages[0].content, "hello")

    def test_auth_failure_preserves_turn_if_agent_activity_already_occurred(self):
        class StatusAuthError(RuntimeError):
            status_code = 401

        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                session = Mock()
                session.session_id = "current"
                session.conversation = Conversation()
                mock_session.return_value = session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = lambda **_: provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.console.print = Mock()

                    def fail_after_activity(*, conversation, **kwargs):
                        conversation.add_assistant_message("partial tool-era activity")
                        raise StatusAuthError("rejected after activity")

                    with patch(
                        "src.repl.core.build_agent_preflight",
                        return_value=Mock(estimated_input_tokens=1),
                    ), patch(
                        "src.repl.core.run_agent_loop",
                        side_effect=fail_after_activity,
                    ), patch(
                        "rich.prompt.Prompt.ask",
                        return_value="n",
                    ):
                        repl.chat("Fix this file")

                    self.assertEqual(len(repl.session.conversation.messages), 2)
                    self.assertEqual(repl.session.conversation.messages[0].role, "user")
                    self.assertEqual(
                        repl.session.conversation.messages[0].content,
                        "Fix this file",
                    )
                    self.assertEqual(
                        repl.session.conversation.messages[1].content,
                        "partial tool-era activity",
                    )

    def test_relogin_provider_constructor_failure_does_not_change_runtime_or_persist(self):
        class BrokenProvider:
            def __init__(self, api_key, base_url=None, model=None):
                raise RuntimeError("constructor failure")

        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.provider_name = "glm"
        old_provider = Mock()
        old_provider.model = "glm-4.5"
        repl.provider = old_provider
        repl.session = Mock()
        repl.session.provider = "glm"
        repl.session.model = "glm-4.5"
        repl.command_context = Mock()
        repl.command_context.config = {"context_provider": old_provider}

        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=[
                "glm",
                "replacement-key",
                "https://open.bigmodel.cn/api/paas/v4",
                "glm-4.5",
            ],
        ), patch(
            "src.config.get_provider_config",
            return_value={
                "api_key": "old-key",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "default_model": "glm-4.5",
            },
        ), patch(
            "src.providers.get_provider_class",
            return_value=BrokenProvider,
        ), patch(
            "src.config.set_api_key"
        ) as set_api_key, patch(
            "src.config.set_default_provider"
        ) as set_default_provider:
            result = repl._handle_relogin()

        self.assertFalse(result)
        set_api_key.assert_not_called()
        set_default_provider.assert_not_called()
        self.assertIs(repl.provider, old_provider)
        self.assertEqual(repl.provider_name, "glm")
        self.assertEqual(repl.session.provider, "glm")
        self.assertEqual(repl.session.model, "glm-4.5")
        self.assertIs(repl.command_context.config["context_provider"], old_provider)

    def test_compact_failure_preserves_conversation_and_restores_provider_config(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                session = Mock()
                session.session_id = "current"
                session.conversation = Conversation()
                session.conversation.add_user_message("keep me")
                mock_session.return_value = session

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch.object(
                        repl,
                        "_try_execute_new_command",
                        side_effect=RuntimeError("compact boom"),
                    ):
                        repl.handle_command("/compact")

                    self.assertEqual(len(repl.session.conversation.messages), 1)
                    self.assertEqual(repl.session.conversation.messages[0].content, "keep me")
                    self.assertNotIn("provider", repl.command_context.config)
                    self.assertNotIn("model", repl.command_context.config)

    def test_compact_context_routes_to_compact_alias(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                session = Mock()
                session.session_id = "current"
                session.conversation = Conversation()
                session.conversation.add_user_message("one")
                session.conversation.add_assistant_message("two")
                mock_session.return_value = session

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch.object(
                        repl,
                        "_try_execute_new_command",
                        return_value=(True, "compacted"),
                    ) as execute:
                        repl.handle_command("/compact-context")

                    execute.assert_called_once_with("compact", "")
                    self.assertNotIn("provider", repl.command_context.config)
                    self.assertNotIn("model", repl.command_context.config)

    def test_doctor_routes_to_local_command_without_provider_config(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.providers.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch.object(
                        repl,
                        "_try_execute_new_command",
                        return_value=(True, "# Clawd Doctor\n\n**Status:** PASS"),
                    ) as execute:
                        repl.handle_command("/doctor")

                    execute.assert_called_once_with("doctor", "")
                    self.assertNotIn("provider", repl.command_context.config)
                    self.assertNotIn("model", repl.command_context.config)

    def test_context_command_does_not_enable_generic_provider_config(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.providers.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch.object(
                        repl,
                        "_try_execute_new_command",
                        return_value=(True, "context report"),
                    ):
                        repl.handle_command("/context")

                    self.assertNotIn("provider", repl.command_context.config)
                    self.assertIs(
                        repl.command_context.config["context_provider"],
                        repl.provider,
                    )

    def test_slash_skill_tool_scope_is_active_only_during_skill_chat(self):
        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))
        repl.tool_registry = Mock()
        repl.tool_registry.get.return_value = object()
        repl._built_in_commands = set()

        def dispatch(_call, context):
            context.restrict_tool_allowlist(["Read"])
            return ToolResult(
                name="Skill",
                output={
                    "success": True,
                    "commandName": "demo",
                    "status": "inline",
                    "allowedTools": ["Read"],
                    "prompt": "Use the approved tool.",
                },
            )

        seen_scopes = []
        repl.tool_registry.dispatch.side_effect = dispatch
        repl.chat = lambda _prompt: seen_scopes.append(repl.tool_context.tool_allowlist)

        handled = repl._try_run_skill_slash("/demo")

        self.assertTrue(handled)
        self.assertEqual(seen_scopes, [frozenset({"read"})])
        self.assertIsNone(repl.tool_context.tool_allowlist)

    def test_plugin_prompt_command_tool_scope_is_temporary(self):
        from src.command_system import CommandRegistry, CommandResult, PromptCommand

        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))
        repl.command_registry = CommandRegistry()
        repl.command_registry.register(
            PromptCommand(
                name="plugin-prompt",
                description="plugin prompt",
                loaded_from="plugin",
                markdown_content="Use the plugin tool.",
                allowed_tools=["Read"],
            )
        )
        seen_scopes = []
        repl.chat = lambda _prompt, **_kwargs: seen_scopes.append(
            repl.tool_context.tool_allowlist
        )

        handled = repl._handle_command_result(
            CommandResult.success_prompt(
                "plugin-prompt",
                [{"type": "text", "text": "Use the plugin tool."}],
            )
        )

        self.assertTrue(handled)
        self.assertEqual(seen_scopes, [frozenset({"read"})])
        self.assertIsNone(repl.tool_context.tool_allowlist)

    def test_plugin_workflow_tool_scope_is_temporary(self):
        from src.command_system import CommandRegistry, CommandResult, PromptCommand

        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))
        repl.command_registry = CommandRegistry()
        repl.command_registry.register(
            PromptCommand(
                name="release-audit",
                description="workflow",
                loaded_from="plugin",
                kind="workflow",
                markdown_content="Audit release.",
                allowed_tools=["Read", "Grep"],
            )
        )
        seen_scopes = []
        repl.chat = lambda _prompt, **_kwargs: seen_scopes.append(
            repl.tool_context.tool_allowlist
        )

        handled = repl._handle_command_result(
            CommandResult.success_prompt(
                "release-audit",
                [{"type": "text", "text": "Audit release."}],
            )
        )

        self.assertTrue(handled)
        self.assertEqual(seen_scopes, [frozenset({"read", "grep"})])
        self.assertIsNone(repl.tool_context.tool_allowlist)

    def test_repl_registers_trusted_plugin_extensions_into_runtime_registries(self):
        from src.command_system import CommandRegistry, LocalCommand, LocalCommandResult
        from src.plugins.extensions import PluginExtensionLoadResult, PluginExtensions
        from src.tool_system.defaults import build_default_registry
        from src.tool_system.protocol import ToolResult
        from src.tool_system.registry import ToolSpec

        class PluginTool:
            def spec(self):
                return ToolSpec(
                    name="PluginUnitRead",
                    description="plugin unit read",
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                    },
                    permission_policy="allow",
                    is_read_only=True,
                )

            def run(self, tool_input, context):
                return ToolResult(name="PluginUnitRead", output={"ok": True})

        command = LocalCommand(
            name="plugin-unit",
            description="plugin unit command",
            loaded_from="plugin",
            supports_non_interactive=True,
        )
        command.set_call(
            lambda args, context: LocalCommandResult(value="plugin:" + args)
        )
        load_result = PluginExtensionLoadResult(
            plugins=[
                PluginExtensions(
                    plugin_name="unit",
                    plugin_version="1.0.0",
                    artifact_sha256="a" * 64,
                    commands=[command],
                    tools=[PluginTool()],
                )
            ]
        )

        repl = ClawdREPL.__new__(ClawdREPL)
        repl.tool_registry = build_default_registry()
        repl.command_registry = CommandRegistry()
        repl.command_context = Mock()
        repl.command_context.config = {}
        global_registry = CommandRegistry()

        with patch(
            "src.repl.core.get_command_registry",
            return_value=global_registry,
        ), patch(
            "src.repl.core.load_active_plugin_extensions",
            return_value=load_result,
        ):
            repl._init_plugin_extensions()

        self.assertIsNotNone(repl.tool_registry.get("PluginUnitRead"))
        self.assertIsNotNone(repl.command_registry.get("plugin-unit"))
        self.assertIsNotNone(global_registry.get("plugin-unit"))
        self.assertEqual(repl.plugin_extension_issues, [])
        self.assertEqual(repl.command_context.config["plugin_runtime_issues"], [])

    def test_repl_bootstraps_mcp_resource_clients(self):
        fake_client = object()
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    with patch(
                        'src.repl.core.load_mcp_resource_clients',
                        return_value={"docs": fake_client},
                    ) as mock_load:
                        repl = ClawdREPL(provider_name="glm")

        mock_load.assert_called_once_with()
        self.assertIs(repl.tool_context.mcp_clients["docs"], fake_client)
        self.assertIsNone(repl.tool_context.mcp_config_error)

    def test_repl_mcp_manifest_error_fails_closed(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    with patch(
                        'src.repl.core.load_mcp_resource_clients',
                        side_effect=MCPResourceConfigError("bad manifest"),
                    ):
                        repl = ClawdREPL(provider_name="glm")

        self.assertEqual(repl.tool_context.mcp_clients, {})
        self.assertEqual(repl.tool_context.mcp_config_error, "bad manifest")

    def test_mcp_explicit_permission_requires_explicit_yes(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawdREPL(provider_name="glm")
        repl.console = Mock()

        with patch("builtins.input", return_value=""):
            allowed, _ = repl._handle_permission_request(
                "MCP",
                "MCP tool 'srv/write' requires explicit approval",
                "require-explicit-yes",
            )
        self.assertFalse(allowed)

        for value in ("y", "Y", "yes", "YES", "1"):
            with self.subTest(value=value), patch("builtins.input", return_value=value):
                allowed, _ = repl._handle_permission_request(
                    "MCP",
                    "MCP tool 'srv/write' requires explicit approval",
                    "require-explicit-yes",
                )
                self.assertTrue(allowed)

        with patch("builtins.input", return_value=""):
            allowed, _ = repl._handle_permission_request(
                "Read",
                "ordinary permission",
                None,
            )
        self.assertFalse(allowed)

    def _permission_prompt_repl(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawdREPL(provider_name="glm")
        repl.console = Mock()
        return repl

    def _printed(self, repl) -> str:
        return " ".join(str(call.args[0]) for call in repl.console.print.call_args_list if call.args)

    def test_every_permission_ask_requires_explicit_choice(self):
        repl = self._permission_prompt_repl()
        # Ordinary menu: 1=Yes, 2=No.
        cases = {
            "": False, "   ": False, "y": True, "Y": True, "yes": True, "YES": True,
            "1": True, "n": False, "no": False, "2": False, "x": False, "3": False, "e": False,
        }
        for value, expected in cases.items():
            with self.subTest(value=value), patch("builtins.input", return_value=value):
                repl.console.reset_mock()
                allowed, _ = repl._handle_permission_request("Read", "ordinary permission", None)
                self.assertIs(allowed, expected)
                printed = self._printed(repl)
                if value.strip() == "":
                    self.assertIn("No choice entered — denied.", printed)
                elif not expected and value not in ("n", "no", "2"):
                    self.assertIn("Invalid choice — denied.", printed)
                else:
                    self.assertNotIn("denied.", printed)

    def test_enable_setting_menu_maps_displayed_numbers(self):
        docs_message = "Writing documentation files is blocked unless allow_docs is enabled"
        # Enable menu: 1=Enable, 2=Yes, 3=No.
        cases = {
            "": (False, False), "   ": (False, False), "1": (True, True), "e": (True, True),
            "2": (True, False), "y": (True, False), "3": (False, False), "n": (False, False),
            "4": (False, False),
        }
        for value, (expected_allowed, expected_enabled) in cases.items():
            with self.subTest(value=value):
                repl = self._permission_prompt_repl()
                repl.tool_context.permission_context.allow_docs = False
                repl.tool_context.permission_context.allow_docs_locked_off = False
                with patch("builtins.input", return_value=value):
                    allowed, _ = repl._handle_permission_request("Write", docs_message, None)
                self.assertIs(allowed, expected_allowed)
                self.assertIs(repl.tool_context.permission_context.allow_docs, expected_enabled)

    def test_run_tool_output_is_printed_without_markup(self):
        """F3: third-party text with Rich tags is shown literally and cannot crash /run-tool."""
        import io
        from rich.console import Console as RichConsole

        repl = self._permission_prompt_repl()
        out = io.StringIO()
        repl.console = RichConsole(file=out, width=10000, color_system=None)
        canned = {
            "status": "records_found",
            "reason": "none",
            "statement": "canned",
            "records": [{"summary": "[/x] [bold]boom :fire: done"}],
            "next_page_token": None,
        }
        with patch("src.tool_system.tools.osv.lookup", return_value=canned), \
             patch("builtins.input", return_value="y"):
            repl.handle_command(
                '/run-tool OsvQuery {"operation": "query", "ecosystem": "PyPI", "name": "jinja2", "version": "2.4.1"}'
            )
        self.assertIn("[/x] [bold]boom :fire: done", out.getvalue())

    def test_osv_tool_results_display_clawd_statement_through_real_on_event(self):
        """J2: the REPL's own on_event path shows Clawd's statement and never OSV text."""
        import io
        from rich.console import Console as RichConsole
        from src.osv_evidence import ERROR_STATUSES, STATUSES
        from src.tool_system.agent_loop import ToolEvent

        for status in STATUSES:
            with self.subTest(status=status):
                repl = self._permission_prompt_repl()
                out = io.StringIO()
                repl.console = RichConsole(file=out, width=100000, color_system=None)
                statement = f"Fixed Clawd statement for {status}. This evidence does not approve any change."
                output = {"status": status, "reason": "none", "statement": statement,
                          "records": [{"summary": "OSV-SUPPLIED-SUMMARY [bold]x"}]}
                is_error = status in ERROR_STATUSES
                if is_error:
                    output["error"] = statement

                def fake_loop(*args, **kwargs):
                    kwargs["on_event"](ToolEvent(kind="tool_result", tool_name="OsvQuery", tool_input={},
                                                 tool_output=output, is_error=is_error))
                    return Mock(response_text="done", usage=None, num_turns=1)

                with patch("src.repl.core.build_agent_preflight", return_value=Mock(estimated_input_tokens=0)), \
                     patch("src.repl.core.run_agent_loop", side_effect=fake_loop):
                    repl.chat("is jinja2 2.4.1 vulnerable?")
                self.assertIn(statement, out.getvalue())
                self.assertNotIn("OSV-SUPPLIED-SUMMARY", out.getvalue())

    def test_project_permission_policy_lock_prevents_enabling_allow_docs(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawdREPL(provider_name="glm")
        repl.console = Mock()
        repl.tool_context.permission_context.allow_docs = False
        repl.tool_context.permission_context.allow_docs_locked_off = True

        repl._enable_permission_setting("allow_docs")
        self.assertFalse(repl.tool_context.permission_context.allow_docs)

        with patch("builtins.input", return_value="e"):
            allowed, _ = repl._handle_permission_request(
                "Write",
                "Writing documentation files is blocked unless allow_docs is enabled",
                "Enable allow_docs to write .md files",
            )
        self.assertFalse(allowed)
        self.assertFalse(repl.tool_context.permission_context.allow_docs)

    def test_high_token_confirmation_uses_lowercase_yn_and_preserves_default_no(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawdREPL(provider_name="glm")

        with patch("builtins.input", return_value="") as mock_input:
            self.assertFalse(repl._confirm_high_token_agent_request(12_345))
            mock_input.assert_called_once_with("Continue? [y/n]> ")

        for value in ("y", "Y", "yes", "YES"):
            with self.subTest(value=value), patch("builtins.input", return_value=value):
                self.assertTrue(repl._confirm_high_token_agent_request(12_345))

        for value in ("n", "N", "no", "NO"):
            with self.subTest(value=value), patch("builtins.input", return_value=value):
                self.assertFalse(repl._confirm_high_token_agent_request(12_345))

    def test_repl_initialization_with_stream_enabled(self):
        """Test REPL can start with stream mode enabled."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session.return_value = Mock()

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm", stream=True)
                    self.assertTrue(repl.stream)

    def test_startup_header_contains_logo_and_metadata(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawdREPL(provider_name="glm")

                    with patch('src.repl.core.Path.cwd', return_value=Path(self.temp_dir)):
                        # Capture stdout to verify fallback path output
                        import io
                        from contextlib import redirect_stdout

                        f = io.StringIO()
                        with redirect_stdout(f):
                            repl._print_startup_header()

                        rendered = f.getvalue()
                        self.assertIn("Clawd Codex", rendered)
                        self.assertIn("glm-4.5", rendered)
                        self.assertIn("GLM Provider", rendered)
                        # Path may be truncated, just check start and end parts
                        self.assertTrue(
                            self.temp_dir[:20] in rendered or self.temp_dir[-20:] in rendered
                        )

    def test_handle_command_exit(self):
        """Test /exit command."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")

                    with self.assertRaises(SystemExit):
                        repl.handle_command("/exit")

    def test_handle_command_clear(self):
        """Test /clear command."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.conversation = Mock()
                mock_session.return_value = mock_session_instance

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.handle_command("/clear")

                    mock_session_instance.conversation.clear.assert_called_once()

    def test_visible_commands_are_organized_and_legacy_aliases_still_work(self):
        """Visible completion uses canonical names while old command aliases remain executable."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.conversation = Mock()
                mock_session.return_value = mock_session_instance

                with patch('src.repl.core.get_provider_class') as mock_provider_class, patch(
                    'src.repl.core.PromptSession'
                ):
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    expected = [
                        "/help",
                        "/exit",
                        "/clear-chat",
                        "/save-session",
                        "/load-session",
                        "/resume",
                        "/compact-context",
                        "/context-usage",
                        "/multiline-input",
                        "/stream-responses",
                        "/render-last-response",
                        "/usage",
                        "/session-usage",
                        "/list-tools",
                        "/run-tool",
                        "/list-skills",
                        "/setup-project",
                        "/doctor",
                    ]
                    words = repl._get_slash_command_words()
                    self.assertEqual(words[:len(expected)], expected)
                    setup_description = next(
                        description
                        for _, commands in repl._visible_command_groups
                        for name, description in commands
                        if name == "/setup-project"
                    )
                    self.assertIn("skills", setup_description.lower())
                    self.assertNotIn("hooks", setup_description.lower())

                    for hidden_alias in (
                        "/cost", "/context", "/compact", "/init",
                        "/clear", "/reset", "/new", "/save", "/load",
                        "/multiline", "/stream", "/render-last", "/tools", "/tool",
                        "/skills", "/quit", "/q", "/?",
                    ):
                        self.assertNotIn(hidden_alias, words)

                    repl.handle_command("/clear-chat")
                    mock_session_instance.conversation.clear.assert_called_once()
                    mock_session_instance.conversation.clear.reset_mock()

                    repl.handle_command("/clear")
                    mock_session_instance.conversation.clear.assert_called_once()

    def test_handle_command_multiline_toggle(self):
        """Test /multiline command."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")

                    # Initially False
                    self.assertFalse(repl.multiline_mode)

                    # Toggle to True
                    repl.handle_command("/multiline")
                    self.assertTrue(repl.multiline_mode)

                    # Toggle back to False
                    repl.handle_command("/multiline")
                    self.assertFalse(repl.multiline_mode)

    def test_handle_command_stream_toggle(self):
        """Test /stream command toggles stream mode safely."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    self.assertFalse(repl.stream)

                    repl.handle_command("/stream on")
                    self.assertTrue(repl.stream)

                    repl.handle_command("/stream off")
                    self.assertFalse(repl.stream)

    def test_handle_command_render_last_renders_markdown(self):
        """Test /render-last re-renders the last assistant response."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session.conversation.add_assistant_message("## Hello\n\n- item")
                mock_session_factory.return_value = mock_session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.console.print = Mock()
                    repl.handle_command("/render-last")

                    self.assertTrue(any(
                        args and isinstance(args[0], Markdown)
                        for args, _kwargs in repl.console.print.call_args_list
                    ))

    def test_handle_command_render_last_without_message(self):
        """Test /render-last handles empty history gracefully."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.console.print = Mock()
                    repl.handle_command("/render-last")

                    self.assertTrue(any(
                        args and "No assistant response available to render." in str(args[0])
                        for args, _kwargs in repl.console.print.call_args_list
                    ))

    def _make_token_warning_repl(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"), \
             patch('src.repl.core.Session.create') as mock_session_factory, \
             patch('src.repl.core.get_provider_class') as mock_provider_class:
            mock_session = Mock()
            mock_session.conversation = Conversation()
            mock_session_factory.return_value = mock_session
            provider = Mock()
            provider.model = "test-model"
            mock_provider_class.return_value = Mock(return_value=provider)
            repl = ClawdREPL(provider_name="glm")
            repl.console.print = Mock()
            return repl, provider, mock_session.conversation

    def test_direct_response_never_builds_or_shows_token_warning(self):
        repl, provider, _conversation = self._make_token_warning_repl()
        provider.chat.return_value = ChatResponse(
            content="Hi",
            model="test-model",
            usage={"input_tokens": 2, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )

        with patch('src.repl.core.build_agent_preflight') as preflight, \
             patch('src.repl.core.append_provider_usage'), \
             patch('builtins.input') as prompt:
            repl.chat("hello")

        provider.chat.assert_called_once()
        preflight.assert_not_called()
        prompt.assert_not_called()
        self.assertNotIn(self._EMPTY_NOTICE, self._printed(repl))

    def test_below_threshold_agent_request_proceeds_without_confirmation(self):
        from src.tool_system.agent_loop import AgentPreflight

        repl, _provider, conversation = self._make_token_warning_repl()
        preflight = AgentPreflight([], "SYSTEM", conversation.get_messages(), 9_999)
        result = Mock(response_text="done", usage=None, num_turns=1)

        with patch('src.repl.core.build_agent_preflight', return_value=preflight), \
             patch('src.repl.core.run_agent_loop', return_value=result) as agent_loop, \
             patch('builtins.input') as prompt:
            repl.chat("Fix this file")

        prompt.assert_not_called()
        agent_loop.assert_called_once()
        self.assertIs(agent_loop.call_args.kwargs["preflight"], preflight)

    def test_above_threshold_agent_request_prompts_once_and_yes_proceeds(self):
        from src.tool_system.agent_loop import AgentPreflight

        repl, _provider, conversation = self._make_token_warning_repl()
        preflight = AgentPreflight([], "SYSTEM", conversation.get_messages(), 10_000)
        result = Mock(response_text="done", usage=None, num_turns=1)

        with patch('src.repl.core.build_agent_preflight', return_value=preflight), \
             patch('src.repl.core.run_agent_loop', return_value=result) as agent_loop, \
             patch('builtins.input', return_value="yes") as prompt:
            repl.chat("Fix this file")

        prompt.assert_called_once()
        agent_loop.assert_called_once()
        self.assertIs(agent_loop.call_args.kwargs["preflight"], preflight)

    def test_above_threshold_agent_request_no_makes_zero_provider_calls(self):
        from src.tool_system.agent_loop import AgentPreflight

        repl, provider, conversation = self._make_token_warning_repl()
        preflight = AgentPreflight([], "SYSTEM", conversation.get_messages(), 12_345)

        with patch('src.repl.core.build_agent_preflight', return_value=preflight), \
             patch('src.repl.core.run_agent_loop') as agent_loop, \
             patch('builtins.input', return_value="no") as prompt:
            repl.chat("Fix this file")

        prompt.assert_called_once()
        agent_loop.assert_not_called()
        provider.chat.assert_not_called()
        provider.chat_stream.assert_not_called()
        provider.chat_stream_response.assert_not_called()
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "user")
        self.assertEqual(conversation.messages[0].content, "Fix this file")

    def test_task_usage_names_claude_and_gemini(self):
        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.provider_name = "anthropic"
        repl.provider = Mock(model="claude-sonnet-4-6")
        repl.cost_tracker = CostTracker()
        repl.command_context = None
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))
        repl.tool_context.record_usage(
            {
                "label": "Gemini (gemini-3.8-flash)",
                "input_tokens": 1200,
                "output_tokens": 80,
                "thought_tokens": 40,
                "tool_use_tokens": 300,
                "cached_tokens": 10,
                "total_tokens": 1620,
            }
        )

        repl._record_and_print_task_usage(
            {"input_tokens": 100, "output_tokens": 20}
        )

        rendered = "\n".join(
            str(args[0])
            for args, _kwargs in repl.console.print.call_args_list
            if args
        )
        self.assertIn("Claude (claude-sonnet-4-6)", rendered)
        self.assertIn("Gemini (gemini-3.8-flash)", rendered)
        self.assertIn("Combined tracked total: 1,740 tokens", rendered)
        self.assertEqual(repl.tool_context.usage_records, [])

    def test_task_usage_names_deepseek_and_persists_provider_details(self):
        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.provider_name = "deepseek"
        repl.provider = Mock(model="deepseek-flash")
        repl.cost_tracker = CostTracker()
        repl.command_context = None
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))

        with patch("src.repl.core.append_provider_usage") as append_usage:
            repl._record_and_print_task_usage(
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "thought_tokens": 8,
                    "cached_tokens": 40,
                    "total_tokens": 120,
                }
            )

        rendered = "\n".join(
            str(args[0])
            for args, _kwargs in repl.console.print.call_args_list
            if args
        )
        self.assertIn("DeepSeek (deepseek-flash)", rendered)
        self.assertIn("8 thought", rendered)
        self.assertIn("40 cached", rendered)
        append_usage.assert_called_once_with({
            "label": "DeepSeek (deepseek-flash)",
            "input_tokens": 100,
            "output_tokens": 20,
            "thought_tokens": 8,
            "tool_use_tokens": 0,
            "cached_tokens": 40,
            "total_tokens": 120,
        })

    def test_task_usage_names_qwen_and_persists_provider_details(self):
        repl = ClawdREPL.__new__(ClawdREPL)
        repl.console = Mock()
        repl.provider_name = "qwen"
        repl.provider = Mock(model="qwen3.8-max")
        repl.cost_tracker = CostTracker()
        repl.command_context = None
        repl.tool_context = ToolContext(workspace_root=Path(self.temp_dir))

        with patch("src.repl.core.append_provider_usage") as append_usage:
            repl._record_and_print_task_usage(
                {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "thought_tokens": 10,
                    "cached_tokens": 50,
                    "total_tokens": 150,
                }
            )

        rendered = "\n".join(
            str(args[0])
            for args, _kwargs in repl.console.print.call_args_list
            if args
        )
        self.assertIn("Qwen (qwen3.8-max)", rendered)
        self.assertIn("10 thought", rendered)
        self.assertIn("50 cached", rendered)
        append_usage.assert_called_once_with({
            "label": "Qwen (qwen3.8-max)",
            "input_tokens": 120,
            "output_tokens": 30,
            "thought_tokens": 10,
            "tool_use_tokens": 0,
            "cached_tokens": 50,
            "total_tokens": 150,
        })

    def test_direct_run_tool_prints_provider_usage_footer(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawdREPL(provider_name="glm")
                    repl.console.print = Mock()

                    def dispatch(call, context):
                        label = (
                            "Qwen (qwen3.8-max)"
                            if call.name == "QwenMediaAnalyze"
                            else "Gemini (gemini-3.8-flash)"
                        )
                        context.usage_records.append({
                            "label": label,
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "thought_tokens": 5,
                            "tool_use_tokens": 0,
                            "cached_tokens": 0,
                            "total_tokens": 120,
                        })
                        return Mock(output={"ok": True})

                    repl.tool_registry.dispatch = Mock(side_effect=dispatch)

                    repl.handle_command('/run-tool QwenMediaAnalyze {"source":"test.mp4"}')
                    repl.handle_command('/run-tool GeminiThink {"question":"test"}')

                    rendered = "\n".join(
                        str(args[0])
                        for args, _kwargs in repl.console.print.call_args_list
                        if args
                    )
                    self.assertIn("Qwen (qwen3.8-max)", rendered)
                    self.assertIn("Gemini (gemini-3.8-flash)", rendered)
                    self.assertGreaterEqual(rendered.count("Usage this task:"), 2)
                    self.assertEqual(repl.tool_context.usage_records, [])

    def test_chat_uses_true_api_stream_for_simple_prompt(self):
        """Simple prompts should use provider.chat_stream when stream mode is enabled."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    # Structured streaming declared unsupported: the legacy chunk stream is chosen.
                    mock_provider.SUPPORTS_STRUCTURED_STREAMING = False
                    mock_provider.chat_stream.return_value = iter(["你", "好"])
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawdREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()

                    with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
                        repl.chat("你是谁")

                    mock_provider.chat_stream.assert_called_once()
                    mock_provider.chat_stream_response.assert_not_called()
                    mock_agent_loop.assert_not_called()
                    self.assertFalse(any(
                        args and isinstance(args[0], Markdown)
                        for args, _kwargs in repl.console.print.call_args_list
                    ))
                    self.assertNotIn(self._EMPTY_NOTICE, self._printed(repl))
                    self.assertEqual(len(mock_session.conversation.messages), 2)
                    self.assertEqual(mock_session.conversation.messages[1].role, "assistant")
                    self.assertEqual(mock_session.conversation.messages[1].content, "你好")

    def test_chat_stream_falls_back_to_agent_loop_for_code_task(self):
        """Code-like prompts keep the existing agent loop path for safety."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session

                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = Mock(return_value=mock_provider)

                    repl = ClawdREPL(provider_name="glm", stream=True)
                    repl.console.print = Mock()

                    def run_agent_loop_side_effect(*args, **kwargs):
                        kwargs["on_text_chunk"]("**done**")
                        return Mock(response_text="**done**", usage=None, num_turns=1)

                    with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
                        mock_agent_loop.side_effect = run_agent_loop_side_effect
                        repl.chat("请读取 README.md 并总结")

                    mock_provider.chat_stream.assert_not_called()
                    mock_agent_loop.assert_called_once()
                    self.assertFalse(any(
                        args and isinstance(args[0], Markdown)
                        for args, _kwargs in repl.console.print.call_args_list
                    ))

    def _direct_failure_repl(self, *, stream: bool, provider: Mock):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session_factory:
                mock_session = Mock()
                mock_session.conversation = Conversation()
                mock_session_factory.return_value = mock_session
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider_class.return_value = Mock(return_value=provider)
                    repl = ClawdREPL(provider_name="glm", stream=stream)
        repl.console.print = Mock()
        return repl

    def _assert_direct_failure_surfaced(self, repl, mock_agent_loop) -> None:
        mock_agent_loop.assert_not_called()
        printed = " ".join(str(call.args[0]) for call in repl.console.print.call_args_list if call.args)
        self.assertIn("Error:", printed)
        self.assertIn("The request failed. Clawd did not issue a fallback retry.", printed)
        self.assertNotIn(self._EMPTY_NOTICE, printed)

    def test_chat_stream_legacy_failure_surfaces_without_agent_loop_request(self):
        """A real legacy-stream failure is shown; it never turns into a second (agent) request."""
        mock_provider = Mock()
        mock_provider.model = "glm-4.5"
        # Structured streaming declared unsupported; the legacy stream then fails before a chunk.
        mock_provider.SUPPORTS_STRUCTURED_STREAMING = False
        mock_provider.chat_stream.side_effect = RuntimeError("stream unavailable")
        repl = self._direct_failure_repl(stream=True, provider=mock_provider)

        with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
            repl.chat("你好呀")

        mock_provider.chat_stream.assert_called_once()
        mock_provider.chat_stream_response.assert_not_called()
        self._assert_direct_failure_surfaced(repl, mock_agent_loop)

    def test_direct_non_stream_failure_surfaces_without_agent_loop_request(self):
        mock_provider = Mock()
        mock_provider.model = "glm-4.5"
        mock_provider.chat.side_effect = RuntimeError("service unavailable")
        repl = self._direct_failure_repl(stream=False, provider=mock_provider)

        with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
            repl.chat("你好呀")

        mock_provider.chat.assert_called_once()
        self._assert_direct_failure_surfaced(repl, mock_agent_loop)

    def test_direct_stream_failure_before_chunk_surfaces_without_agent_loop_request(self):
        mock_provider = Mock()
        mock_provider.model = "glm-4.5"
        mock_provider.chat_stream_response.side_effect = RuntimeError("overloaded")
        repl = self._direct_failure_repl(stream=True, provider=mock_provider)

        with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
            repl.chat("你好呀")

        mock_provider.chat_stream_response.assert_called_once()
        mock_provider.chat_stream.assert_not_called()
        mock_provider.chat.assert_not_called()
        self._assert_direct_failure_surfaced(repl, mock_agent_loop)

    def test_direct_stream_not_implemented_after_chunk_is_not_a_fallback(self):
        def stream_then_not_implemented(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("partial")
            raise NotImplementedError("late")

        mock_provider = Mock()
        mock_provider.model = "glm-4.5"
        mock_provider.chat_stream_response.side_effect = stream_then_not_implemented
        repl = self._direct_failure_repl(stream=True, provider=mock_provider)

        with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
            repl.chat("你好呀")

        mock_provider.chat_stream_response.assert_called_once()
        mock_provider.chat_stream.assert_not_called()
        mock_provider.chat.assert_not_called()
        mock_agent_loop.assert_not_called()
        printed = self._printed(repl)
        self.assertIn("Provider error: the active provider reported an unimplemented operation.", printed)
        self.assertIn("The request failed. Clawd did not issue a fallback retry.", printed)
        self.assertNotIn(self._EMPTY_NOTICE, printed)

    def test_direct_stream_attribute_error_is_never_treated_as_unsupported(self):
        """AttributeError (raised inside the provider, or a missing method) must surface."""
        raising = Mock()
        raising.model = "glm-4.5"
        raising.chat_stream_response.side_effect = AttributeError("provider bug")
        missing_method = Mock(spec=["model", "chat", "chat_stream", "get_available_models"])
        missing_method.model = "glm-4.5"

        for label, provider in (("raised inside provider", raising), ("missing method", missing_method)):
            with self.subTest(label):
                repl = self._direct_failure_repl(stream=True, provider=provider)
                with patch('src.repl.core.run_agent_loop') as mock_agent_loop:
                    repl.chat("你好呀")
                provider.chat_stream.assert_not_called()
                provider.chat.assert_not_called()
                self._assert_direct_failure_surfaced(repl, mock_agent_loop)

    _EMPTY_NOTICE = "The provider returned an empty response. Clawd did not issue a fallback retry."

    @staticmethod
    def _empty(**overrides):
        fields = dict(content="", model="glm-4.5", usage={}, finish_reason="stop", tool_uses=None)
        fields.update(overrides)
        return ChatResponse(**fields)

    @staticmethod
    def _printed(repl) -> str:
        return " ".join(str(call.args[0]) for call in repl.console.print.call_args_list if call.args)

    def _chat_expect_terminal_empty(self, repl):
        """One user turn whose direct reply is empty: it must end without an agent request."""
        before = [(m.role, m.content) for m in repl.session.conversation.messages]
        with patch('src.repl.core.build_agent_preflight') as preflight, \
             patch('src.repl.core.run_agent_loop') as agent_loop, \
             patch('src.repl.core.append_provider_usage') as ledger, \
             patch('builtins.input') as prompt, \
             patch('traceback.print_exc') as print_exc:
            repl.chat("你好呀")
        for untouched in (preflight, agent_loop, prompt, print_exc):
            untouched.assert_not_called()
        printed = self._printed(repl)
        self.assertEqual(printed.count(self._EMPTY_NOTICE), 1)
        self.assertNotIn("Error:", printed)
        self.assertNotIn("The request failed.", printed)
        self.assertEqual(printed.count("Usage this task"), 1)
        self.assertIsNone(repl._current_status)
        # The unanswered user message is rolled back; nothing is stored for the empty reply.
        self.assertEqual([(m.role, m.content) for m in repl.session.conversation.messages], before)
        return ledger, printed

    def test_direct_empty_reply_ends_turn_without_agent_route(self):
        """A sent-but-empty direct reply ends the turn: no second (agent-route) request."""
        usage_9 = {"input_tokens": 9, "output_tokens": 0}
        reasoning = dict(reasoning_content="E7-REASONING-SENTINEL", finish_reason="length")

        def empty_after_blank_chunk(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("")
            return self._empty()

        cases = [
            ("non-stream empty", False, {"chat": self._empty(usage=usage_9)}, "chat", True),
            ("non-stream reasoning only", False,
             {"chat": self._empty(usage={"input_tokens": 9, "output_tokens": 40}, **reasoning)}, "chat", True),
            ("non-stream unexpected tool_uses", False,
             {"chat": self._empty(tool_uses=[{"id": "t", "name": "Read", "input": {}}])}, "chat", False),
            ("stream empty", True, {"chat_stream_response": self._empty()}, "chat_stream_response", False),
            ("stream blank chunk then empty", True,
             {"chat_stream_response": empty_after_blank_chunk}, "chat_stream_response", False),
            ("stream reasoning only", True,
             {"chat_stream_response": self._empty(usage=usage_9, **reasoning)}, "chat_stream_response", True),
            ("legacy stream yields nothing", True,
             {"SUPPORTS_STRUCTURED_STREAMING": False, "chat_stream": iter([])}, "chat_stream", False),
        ]
        for label, stream, behaviour, sent_by, billed in cases:
            with self.subTest(label):
                provider = Mock()
                provider.model = "glm-4.5"
                for method, value in behaviour.items():
                    if method == "SUPPORTS_STRUCTURED_STREAMING":  # a declaration, not a method
                        setattr(provider, method, value)
                    elif callable(value):  # an exception class or a fake method body
                        getattr(provider, method).side_effect = value
                    else:
                        getattr(provider, method).return_value = value
                repl = self._direct_failure_repl(stream=stream, provider=provider)

                ledger, printed = self._chat_expect_terminal_empty(repl)

                getattr(provider, sent_by).assert_called_once()
                for other in {"chat", "chat_stream", "chat_stream_response"} - {sent_by} - set(behaviour):
                    getattr(provider, other).assert_not_called()
                if billed:
                    ledger.assert_called_once()
                    self.assertEqual(ledger.call_args.args[0]["input_tokens"], 9)
                else:
                    ledger.assert_not_called()
                    self.assertIn("did not return token counts", printed)
                self.assertNotIn("E7-REASONING-SENTINEL", printed)

    def test_direct_empty_reply_keeps_conversation_touched_by_something_else(self):
        """The rollback is guarded: it never overwrites a conversation that changed unexpectedly."""
        provider = Mock()
        provider.model = "glm-4.5"
        repl = self._direct_failure_repl(stream=False, provider=provider)

        def chat_that_touches_conversation(*args, **kwargs):
            repl.session.conversation.add_assistant_message("unexpected")
            return self._empty()

        provider.chat.side_effect = chat_that_touches_conversation
        with patch('src.repl.core.run_agent_loop') as agent_loop, \
             patch('src.repl.core.append_provider_usage'):
            repl.chat("你好呀")

        agent_loop.assert_not_called()
        provider.chat.assert_called_once()
        self.assertEqual(
            [(m.role, m.content) for m in repl.session.conversation.messages],
            [("user", "你好呀"), ("assistant", "unexpected")],
        )
        self.assertIn(self._EMPTY_NOTICE, self._printed(repl))

    def test_direct_structured_stream_wrong_type_surfaces_as_error(self):
        """A structured stream that does not return ChatResponse is a provider bug, not 'empty'."""
        for label, returned in (("None", None), ("dict", {"content": "text"})):
            with self.subTest(label):
                provider = Mock()
                provider.model = "glm-4.5"
                provider.chat_stream_response.return_value = returned
                repl = self._direct_failure_repl(stream=True, provider=provider)
                with patch('src.repl.core.run_agent_loop') as agent_loop:
                    repl.chat("你好呀")
                provider.chat_stream_response.assert_called_once()
                provider.chat.assert_not_called()
                provider.chat_stream.assert_not_called()
                self._assert_direct_failure_surfaced(repl, agent_loop)
                self.assertIn("Structured streaming must return ChatResponse", self._printed(repl))

    def test_direct_empty_reply_is_one_provider_call_end_to_end(self):
        """With the real agent loop wired in, an empty direct reply still costs exactly one call."""
        from src.tool_system.agent_loop import AgentPreflight

        provider = Mock()
        provider.model = "glm-4.5"
        provider.chat.side_effect = [
            self._empty(usage={"input_tokens": 9, "output_tokens": 0}),
            ChatResponse(content="agent", model="glm-4.5", usage={}, finish_reason="stop", tool_uses=None),
        ]
        repl = self._direct_failure_repl(stream=False, provider=provider)
        with patch('src.repl.core.build_agent_preflight',
                   return_value=AgentPreflight([], "SYSTEM", [], 1)), \
             patch('src.repl.core.append_provider_usage'):
            repl.chat("你好呀")

        self.assertEqual(provider.chat.call_count, 1)
        self.assertIn(self._EMPTY_NOTICE, self._printed(repl))

    def test_direct_local_payload_failure_still_hands_over_to_agent_route(self):
        """Nothing was sent: the one legitimate handover to the agent route is unchanged."""
        from src.tool_system.agent_loop import AgentPreflight

        for stream in (False, True):
            with self.subTest(stream=stream):
                provider = Mock()
                provider.model = "glm-4.5"
                repl = self._direct_failure_repl(stream=stream, provider=provider)
                result = Mock(response_text="done", usage=None, num_turns=1)
                with patch.object(repl, "_build_direct_stream_payload", side_effect=ValueError("local")), \
                     patch('src.repl.core.build_agent_preflight',
                           return_value=AgentPreflight([], "SYSTEM", [], 1)) as preflight, \
                     patch('src.repl.core.run_agent_loop', return_value=result) as agent_loop:
                    repl.chat("你好呀")

                for method in ("chat", "chat_stream", "chat_stream_response"):
                    getattr(provider, method).assert_not_called()
                preflight.assert_called_once()
                agent_loop.assert_called_once()
                self.assertNotIn(self._EMPTY_NOTICE, self._printed(repl))
                self.assertEqual(repl.session.conversation.messages[0].content, "你好呀")

    _LIMIT_NOTICE = "Incomplete response: the provider stopped at its output limit."

    def _chat_with_guards(self, repl, prompt, *, agent_loop_side_effect=None):
        from src.tool_system.agent_loop import AgentPreflight

        with patch('src.repl.core.build_agent_preflight',
                   return_value=AgentPreflight([], "SYSTEM", [], 1)) as preflight, \
             patch('src.repl.core.run_agent_loop', side_effect=agent_loop_side_effect) as agent_loop, \
             patch('src.repl.core.append_provider_usage') as ledger, \
             patch('rich.prompt.Prompt.ask') as ask, \
             patch('traceback.print_exc') as print_exc:
            repl.chat(prompt)
        ledger.assert_not_called()
        ask.assert_not_called()
        print_exc.assert_not_called()
        return preflight, agent_loop

    def test_output_limit_direct_reply_is_kept_as_partial_not_complete(self):
        """Provider-declared truncation: partial text shown once and stored marked incomplete."""
        from src.providers.base import IncompleteResponseError

        usage = {"input_tokens": 11, "output_tokens": 7}

        def stream_then_limit(messages, tools=None, on_text_chunk=None, **kwargs):
            on_text_chunk("Hello wor")
            raise IncompleteResponseError("output_limit", partial_text="Hello wor", partial_usage=usage)

        cases = (
            ("stream", True, "chat_stream_response", stream_then_limit),
            ("non-stream", False, "chat",
             IncompleteResponseError("output_limit", partial_text="Hello wor", partial_usage=usage)),
        )
        for label, stream, method, side_effect in cases:
            with self.subTest(label):
                provider = Mock()
                provider.model = "glm-4.5"
                getattr(provider, method).side_effect = side_effect
                repl = self._direct_failure_repl(stream=stream, provider=provider)

                preflight, agent_loop = self._chat_with_guards(repl, "你好呀")

                getattr(provider, method).assert_called_once()
                preflight.assert_not_called()
                agent_loop.assert_not_called()
                printed = self._printed(repl)
                self.assertEqual(printed.count("Hello wor"), 1)
                self.assertIn(self._LIMIT_NOTICE, printed)
                self.assertIn("The text above is partial.", printed)
                self.assertIn("Clawd did not issue a fallback retry.", printed)
                self.assertIn("at least 11 input and 7 output tokens were reported (not recorded)", printed)
                for absent in ("Error:", self._EMPTY_NOTICE, "A tool call", "Earlier completed"):
                    self.assertNotIn(absent, printed)
                self.assertEqual(
                    [(m.role, m.content) for m in repl.session.conversation.messages],
                    [("user", "你好呀"),
                     ("assistant", "[Incomplete response: the provider stopped at its output limit. "
                                   "The text below is partial and is not a complete answer.]\n\nHello wor")],
                )

    def test_output_limit_in_agent_route_stores_only_the_marker(self):
        """Tool-only / reasoning-only truncation stores just the marker; never tool calls."""
        from src.providers.base import IncompleteResponseError

        tool_marker = ("[Incomplete response: the provider stopped at its output limit. "
                       "A tool call was cut off and was not run.]")
        cases = (
            ("tool call, first turn", "tool_input_truncated", True, False, tool_marker),
            ("tool call after an earlier turn", "tool_input_truncated", True, True, tool_marker),
            ("reasoning only", "output_limit", False, False,
             "[Incomplete response: the provider stopped at its output limit.]"),
        )
        for label, reason, dropped, earlier_turn, marker in cases:
            with self.subTest(label):
                provider = Mock()
                provider.model = "glm-4.5"
                repl = self._direct_failure_repl(stream=False, provider=provider)

                def agent_turns(*args, **kwargs):
                    if earlier_turn:
                        kwargs["conversation"].add_assistant_message("earlier completed turn")
                    raise IncompleteResponseError(reason, tool_call_dropped=dropped)

                _preflight, agent_loop = self._chat_with_guards(
                    repl, "Fix this file", agent_loop_side_effect=agent_turns
                )

                agent_loop.assert_called_once()
                printed = self._printed(repl)
                self.assertIn(self._LIMIT_NOTICE, printed)
                self.assertIn("did not report token counts", printed)
                self.assertNotIn("The text above is partial.", printed)
                self.assertEqual("A tool call in this response was cut off and was not run." in printed, dropped)
                self.assertEqual("Earlier completed requests in this task were not recorded." in printed,
                                 earlier_turn)
                expected = [("user", "Fix this file")]
                if earlier_turn:
                    expected.append(("assistant", "earlier completed turn"))
                expected.append(("assistant", marker))
                self.assertEqual([(m.role, m.content) for m in repl.session.conversation.messages], expected)

    def test_output_limit_partial_text_is_never_auth_classified(self):
        from src.providers.base import IncompleteResponseError

        provider = Mock()
        provider.model = "glm-4.5"
        provider.chat.side_effect = IncompleteResponseError(
            "output_limit", partial_text="HTTP 401 Unauthorized: invalid api key"
        )
        repl = self._direct_failure_repl(stream=False, provider=provider)

        self._chat_with_guards(repl, "你好呀")

        printed = self._printed(repl)
        self.assertIn(self._LIMIT_NOTICE, printed)
        self.assertNotIn("Authentication Error", printed)
        self.assertEqual(len(repl.session.conversation.messages), 2)

    _FINISH_HEADLINES = {
        "blocked": "Response blocked: the provider stopped this response with a safety or "
                   "content-filter status",
        "interrupted": "Incomplete response: the provider reported that generation was interrupted",
        "context_window": "Incomplete response: the provider stopped at the model's context window",
        "paused": "Incomplete response: the provider paused this turn",
        "unrecognized": "Unverified response: the provider ended a tool-call response with a "
                        "finish status Clawd does not recognize",
    }

    @staticmethod
    def _finish_error(category, value, *, reset=False, dropped=False, partial="Partial answer"):
        from src.providers.base import FinishStatusError

        return FinishStatusError(
            category, finish_value=value, partial_text=partial,
            partial_usage={"input_tokens": 11, "output_tokens": 7},
            tool_call_dropped=dropped, context_reset=reset,
        )

    def _direct_finish_repl(self, stream, error):
        provider = Mock()
        provider.model = "glm-4.5"
        if stream:
            def stream_then_raise(messages, tools=None, on_text_chunk=None, **kwargs):
                on_text_chunk(error.partial_text)
                raise error
            provider.chat_stream_response.side_effect = stream_then_raise
        else:
            provider.chat.side_effect = error
        return provider, self._direct_failure_repl(stream=stream, provider=provider)

    def test_finish_status_direct_reply_is_never_kept_as_complete(self):
        """Provider-declared abnormal finish: fixed notice, marked (or no) partial, nothing recorded."""
        cases = (  # category, value, anthropic refusal reset
            ("blocked", "content_filter", False),
            ("blocked", "refusal", True),
            ("interrupted", "network_error", False),
            ("context_window", "model_context_window_exceeded", False),
            ("paused", "pause_turn", False),
            ("unrecognized", "custom_x", False),
        )
        for category, value, reset in cases:
            for stream in (False, True):
                with self.subTest(category=category, value=value, stream=stream):
                    error = self._finish_error(category, value, reset=reset,
                                               dropped=category == "unrecognized")
                    provider, repl = self._direct_finish_repl(stream, error)

                    preflight, agent_loop = self._chat_with_guards(repl, "你好呀")

                    getattr(provider, "chat_stream_response" if stream else "chat").assert_called_once()
                    preflight.assert_not_called()
                    agent_loop.assert_not_called()
                    printed = self._printed(repl)
                    self.assertIn(f"{self._FINISH_HEADLINES[category]} ({value}).", printed)
                    self.assertIn("Clawd did not issue a fallback retry.", printed)
                    self.assertIn("at least 11 input and 7 output tokens were reported (not recorded)",
                                  printed)
                    for absent in ("Error:", self._EMPTY_NOTICE, "Authentication Error",
                                   "Earlier completed", self._LIMIT_NOTICE):
                        self.assertNotIn(absent, printed)
                    self.assertEqual("Clawd does not resume paused turns automatically." in printed,
                                     category == "paused")
                    self.assertEqual("A tool call in this response was not run." in printed,
                                     category == "unrecognized")
                    history = [(m.role, m.content) for m in repl.session.conversation.messages]
                    if category == "blocked":
                        # Blocked text is never printed after the fact and never kept.
                        self.assertEqual(printed.count("Partial answer"), 1 if stream else 0)
                        self.assertEqual("The streamed text above was blocked" in printed, stream)
                        if reset:
                            self.assertEqual(history, [])
                            self.assertIn("The refused message was removed from the conversation", printed)
                        else:
                            self.assertEqual(history, [
                                ("user", "你好呀"),
                                ("assistant", "[Response blocked: the provider stopped this response with "
                                              "a safety or content-filter status. No partial output was "
                                              "kept.]"),
                            ])
                            self.assertNotIn("The refused message was removed", printed)
                    else:
                        self.assertEqual(printed.count("Partial answer"), 1)
                        kept_as = "unverified" if category == "unrecognized" else "incomplete"
                        self.assertIn(f"It is kept in the conversation marked as {kept_as}", printed)
                        self.assertEqual(history[0], ("user", "你好呀"))
                        self.assertEqual(len(history), 2)
                        marker, partial = history[1][1].split("\n\n")
                        self.assertTrue(marker.startswith(f"[{self._FINISH_HEADLINES[category]}."))
                        self.assertNotIn(value, marker)
                        self.assertEqual(partial, "Partial answer")

    def test_finish_status_rollback_is_only_for_a_clean_anthropic_refusal(self):
        blocked_marker = ("[Response blocked: the provider stopped this response with a safety or "
                          "content-filter status. A tool call was not run. No partial output was kept.]")
        unverified_marker = ("[Unverified response: the provider ended a tool-call response with a "
                             "finish status Clawd does not recognize. A tool call was not run.]")
        cases = (  # label, category, value, reset flag, earlier turn in this task, expected tail
            ("refusal, clean turn", "blocked", "refusal", True, False, None),
            ("refusal after earlier activity", "blocked", "refusal", True, True, blocked_marker),
            ("content_filter, clean turn", "blocked", "content_filter", False, False, blocked_marker),
            ("sensitive, clean turn", "blocked", "sensitive", False, False, blocked_marker),
            # A "refusal" value the provider does not document (e.g. MiniMax) is not Anthropic's rule.
            ("unrecognized refusal value, clean turn", "unrecognized", "refusal", False, False,
             unverified_marker),
        )
        for label, category, value, reset, earlier_turn, marker in cases:
            with self.subTest(label):
                provider = Mock()
                provider.model = "glm-4.5"
                repl = self._direct_failure_repl(stream=False, provider=provider)
                error = self._finish_error(category, value, reset=reset, dropped=True, partial="")

                def agent_turns(*args, **kwargs):
                    if earlier_turn:
                        kwargs["conversation"].add_assistant_message("earlier completed turn")
                    raise error

                _preflight, agent_loop = self._chat_with_guards(
                    repl, "Fix this file", agent_loop_side_effect=agent_turns
                )

                agent_loop.assert_called_once()
                printed = self._printed(repl)
                history = [(m.role, m.content) for m in repl.session.conversation.messages]
                if marker is None:
                    self.assertEqual(history, [])
                    self.assertIn("The refused message was removed from the conversation", printed)
                else:
                    expected = [("user", "Fix this file")]
                    if earlier_turn:
                        expected.append(("assistant", "earlier completed turn"))
                    expected.append(("assistant", marker))
                    self.assertEqual(history, expected)
                    self.assertNotIn("The refused message was removed", printed)
                self.assertEqual("Earlier completed requests in this task were not recorded." in printed,
                                 earlier_turn)
                self.assertIn("A tool call in this response was not run.", printed)
                self.assertIn("at least 11 input and 7 output tokens were reported (not recorded)", printed)

    def test_finish_status_provider_text_is_never_markup_or_auth_classified(self):
        """A real console: provider-controlled text can neither inject markup nor crash chat()."""
        import io
        from rich.console import Console

        cases = (
            ("interrupted", "[/] [red]401 Unauthorized\x1b[0m", "(unprintable value)"),
            ("unrecognized", 401, "(unprintable value)"),
            ("blocked", "content_filter", "(content_filter)"),
            ("unrecognized", "x:smile:", "(x:smile:)"),  # no emoji substitution either
        )
        for category, value, shown in cases:
            for stream in (False, True):
                with self.subTest(category=category, value=value, stream=stream):
                    error = self._finish_error(category, value, dropped=True,
                                               partial="HTTP 401 Unauthorized: invalid api key [/] [bold]x")
                    _provider, repl = self._direct_finish_repl(stream, error)
                    out = io.StringIO()
                    repl.console = Console(file=out, width=200, force_terminal=False, color_system=None)

                    self._chat_with_guards(repl, "你好呀")

                    text = out.getvalue()
                    self.assertIn(shown, text)
                    self.assertNotIn("Authentication Error", text)
                    self.assertNotIn("\x1b", text)
                    if category != "blocked":
                        self.assertIn("invalid api key [/] [bold]x", text)  # printed literally

    def test_finish_status_agent_route_every_category(self):
        """Agent route: fixed notice and marker per category; never recorded, never retried."""
        cases = (  # category, value, marker prefix
            ("blocked", "content_filter", "[Response blocked: the provider stopped this response"),
            ("interrupted", "aborted", "[Incomplete response: the provider reported that generation"),
            ("context_window", "model_context_window_exceeded",
             "[Incomplete response: the provider stopped at the model's context window."),
            ("paused", "pause_turn", "[Incomplete response: the provider paused this turn."),
            ("unrecognized", "custom_x", "[Unverified response: the provider ended a tool-call response"),
        )
        for category, value, marker in cases:
            for stream in (False, True):
                with self.subTest(category=category, stream=stream):
                    provider = Mock()
                    provider.model = "glm-4.5"
                    repl = self._direct_failure_repl(stream=stream, provider=provider)
                    error = self._finish_error(category, value, dropped=True)

                    def agent_turn(*args, **kwargs):
                        if kwargs.get("on_text_chunk") is not None:
                            kwargs["on_text_chunk"](error.partial_text)  # streamed before the status
                        raise error

                    self._chat_with_guards(repl, "Fix this file", agent_loop_side_effect=agent_turn)

                    printed = self._printed(repl)
                    self.assertIn(f"{self._FINISH_HEADLINES[category]} ({value}).", printed)
                    self.assertEqual("Clawd does not resume paused turns automatically." in printed,
                                     category == "paused")
                    self.assertIn("A tool call in this response was not run.", printed)
                    self.assertEqual(printed.count("Partial answer"),
                                     1 if stream or category != "blocked" else 0)
                    history = [(m.role, m.content) for m in repl.session.conversation.messages]
                    self.assertEqual(history[0], ("user", "Fix this file"))
                    self.assertEqual(len(history), 2)
                    self.assertTrue(history[1][1].startswith(marker))
                    self.assertEqual("Partial answer" in history[1][1], category != "blocked")

    def test_finish_status_stream_notice_is_only_about_this_responses_text(self):
        """A blocked turn that streamed nothing itself must not disown earlier (kept) streamed text."""
        for blocked_text in ("", "Blocked text"):
            with self.subTest(blocked_text=blocked_text):
                provider = Mock()
                provider.model = "glm-4.5"
                repl = self._direct_failure_repl(stream=True, provider=provider)
                error = self._finish_error("blocked", "content_filter", partial=blocked_text)

                def agent_turns(*args, **kwargs):
                    kwargs["on_text_chunk"]("Earlier streamed text")
                    kwargs["conversation"].add_assistant_message("Earlier streamed text")
                    if blocked_text:
                        kwargs["on_text_chunk"](blocked_text)
                    raise error

                self._chat_with_guards(repl, "Fix this file", agent_loop_side_effect=agent_turns)

                printed = self._printed(repl)
                self.assertEqual("The streamed text above was blocked" in printed, bool(blocked_text))
                self.assertIn("Earlier completed requests in this task were not recorded.", printed)
                history = [(m.role, m.content) for m in repl.session.conversation.messages]
                self.assertEqual(history[:2], [("user", "Fix this file"),
                                               ("assistant", "Earlier streamed text")])
                self.assertEqual(len(history), 3)
                self.assertNotIn("Blocked text", history[2][1])

    def test_finish_status_is_handled_before_the_auth_classifier(self):
        """Even when its exception context looks like an auth failure, it is never auth-handled."""
        error = self._finish_error("interrupted", "network_error")
        try:
            raise RuntimeError("HTTP 401 Unauthorized")
        except RuntimeError as earlier:
            error.__context__ = earlier
        self.assertTrue(_is_provider_authentication_error(error))  # the classifier alone would
        _provider, repl = self._direct_finish_repl(False, error)

        self._chat_with_guards(repl, "你好呀")  # asserts Prompt.ask was never called

        printed = self._printed(repl)
        self.assertIn(f"{self._FINISH_HEADLINES['interrupted']} (network_error).", printed)
        self.assertNotIn("Authentication Error", printed)
        self.assertEqual(len(repl.session.conversation.messages), 2)

    def test_handle_command_slash_shows_commands_and_skills(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\n"
            "description: say hello\n"
            "---\n"
            "Hello\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWD_SKILLS_DIR": str(skills_dir)}), \
             patch("src.skills.loader.SkillTrustRegistry") as mock_trust_registry:
            mock_trust_registry.return_value.is_active_and_current.return_value = (True, "test fixture")
            with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
                with patch('src.repl.core.Session.create'):
                    with patch('src.providers.get_provider_class') as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawdREPL(provider_name="glm")
                        repl.console.print = Mock()
                        repl.handle_command("/")
                        rendered = "\n".join(
                            str(args[0]) for args, _kwargs in repl.console.print.call_args_list if args
                        )
                        self.assertIn("Available commands and skills", rendered)
                        self.assertIn("/hello", rendered)

    def test_handle_command_slash_prefix_filters(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\n"
            "description: say hello\n"
            "---\n"
            "Hello\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWD_SKILLS_DIR": str(skills_dir)}), \
             patch("src.skills.loader.SkillTrustRegistry") as mock_trust_registry:
            mock_trust_registry.return_value.is_active_and_current.return_value = (True, "test fixture")
            with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
                with patch('src.repl.core.Session.create'):
                    with patch('src.providers.get_provider_class') as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawdREPL(provider_name="glm")
                        repl.console.print = Mock()
                        repl.handle_command("/he")
                        rendered = "\n".join(
                            str(args[0]) for args, _kwargs in repl.console.print.call_args_list if args
                        )
                        self.assertIn("/help", rendered)
                        self.assertIn("/hello", rendered)

    def test_handle_command_skill_invokes_skill_tool_and_chats_with_prompt(self):
        skills_dir = Path(self.temp_dir) / "skills"
        (skills_dir / "hello").mkdir(parents=True, exist_ok=True)
        (skills_dir / "hello" / "SKILL.md").write_text(
            "---\n"
            "description: say hello\n"
            "arguments: [name]\n"
            "---\n"
            "Hello $name\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"CLAWD_SKILLS_DIR": str(skills_dir)}), \
             patch("src.skills.loader.SkillTrustRegistry") as mock_trust_registry:
            mock_trust_registry.return_value.is_active_and_current.return_value = (True, "test fixture")
            with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
                with patch('src.repl.core.Session.create'):
                    with patch('src.providers.get_provider_class') as mock_provider_class:
                        mock_provider = Mock()
                        mock_provider.model = "glm-4.5"
                        mock_provider_class.return_value = mock_provider

                        repl = ClawdREPL(provider_name="glm")
                        repl.chat = Mock()
                        repl.handle_command("/hello bob")
                        args, _kwargs = repl.chat.call_args
                        self.assertIn("Hello bob", args[0])

    def test_save_session(self):
        """Test session saving."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "test_session_123"
                mock_session.return_value = mock_session_instance

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    repl = ClawdREPL(provider_name="glm")
                    repl.save_session()

                    self.assertEqual(mock_session_instance.provider, "glm")
                    self.assertEqual(mock_session_instance.model, "glm-4.5")
                    mock_session_instance.save.assert_called_once()

    def test_load_session(self):
        """Test session loading."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "current_session"
                mock_session.return_value = mock_session_instance

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    with patch('src.repl.core.Session.load') as mock_load:
                        loaded_session = Mock()
                        loaded_session.session_id = "loaded_session_123"
                        loaded_session.provider = "anthropic"
                        loaded_session.model = "claude-sonnet-4-6"
                        loaded_session.conversation = Mock()
                        loaded_session.conversation.messages = []
                        mock_load.return_value = loaded_session

                        repl = ClawdREPL(provider_name="glm")
                        original_provider = repl.provider
                        repl.load_session("loaded_session_123")

                        self.assertEqual(repl.session.session_id, "loaded_session_123")
                        self.assertIs(repl.command_context.conversation, loaded_session.conversation)
                        self.assertIs(repl.provider, original_provider)
                        self.assertEqual(repl.session.provider, "glm")
                        self.assertEqual(repl.session.model, "glm-4.5")
                        self.assertIs(
                            repl.command_context.config["context_provider"],
                            original_provider,
                        )

    def test_resume_direct_id_uses_existing_session_loader(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                current = Mock()
                current.session_id = "current_session"
                current.conversation = Conversation()
                mock_session.return_value = current

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch.object(repl, "load_session") as load_session:
                        repl.handle_command("/resume saved_session")

                    load_session.assert_called_once_with("saved_session")

    def test_resume_picker_excludes_current_and_loads_selection(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                current = Mock()
                current.session_id = "current_session"
                current.conversation = Conversation()
                mock_session.return_value = current

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    saved_current = Session(
                        session_id="current_session",
                        provider="glm",
                        model="glm-4.5",
                    )
                    saved_new = Session(
                        session_id="new_session",
                        provider="glm",
                        model="glm-4.5",
                        updated_at="2026-09-23T17:00:00",
                    )
                    saved_old = Session(
                        session_id="old_session",
                        provider="anthropic",
                        model="claude-sonnet-4-6",
                        updated_at="2026-09-22T17:00:00",
                    )
                    with patch(
                        'src.repl.core.Session.list_saved',
                        return_value=[saved_current, saved_new, saved_old],
                    ), patch(
                        'rich.prompt.Prompt.ask',
                        return_value="2",
                    ), patch.object(repl, "load_session") as load_session:
                        repl.resume_session()

                    load_session.assert_called_once_with("old_session")

    def test_resume_picker_can_cancel_without_loading(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                current = Mock()
                current.session_id = "current_session"
                current.conversation = Conversation()
                mock_session.return_value = current

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    saved = Session(
                        session_id="saved_session",
                        provider="glm",
                        model="glm-4.5",
                    )
                    with patch(
                        'src.repl.core.Session.list_saved',
                        return_value=[saved],
                    ), patch(
                        'rich.prompt.Prompt.ask',
                        return_value="cancel",
                    ), patch.object(repl, "load_session") as load_session:
                        repl.resume_session()

                    load_session.assert_not_called()

    def test_load_invalid_session_reports_error_without_replacing_session(self):
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                current = Mock()
                current.session_id = "current_session"
                current.conversation = Conversation()
                mock_session.return_value = current

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    provider = Mock()
                    provider.model = "glm-4.5"
                    mock_provider_class.return_value = provider

                    repl = ClawdREPL(provider_name="glm")
                    with patch(
                        'src.repl.core.Session.load',
                        side_effect=ValueError("invalid session id"),
                    ):
                        repl.load_session("../bad")

                    self.assertIs(repl.session, current)

    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        with patch('src.config.get_config_path', return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create') as mock_session:
                mock_session_instance = Mock()
                mock_session_instance.session_id = "current_session"
                mock_session.return_value = mock_session_instance

                with patch('src.providers.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider

                    with patch('src.repl.core.Session.load', return_value=None):
                        repl = ClawdREPL(provider_name="glm")
                        original_session = repl.session

                        repl.load_session("nonexistent")

                        # Session should not change
                        self.assertEqual(repl.session, original_session)


class OsvRoutingTests(unittest.TestCase):
    """J1: vulnerability-evidence questions skip the tool-less direct route."""

    @staticmethod
    def _direct(text: str) -> bool:
        return ClawdREPL._should_try_direct_response(object(), text)

    def test_vulnerability_questions_take_the_tool_route(self):
        for text in (
            "is jinja2 2.4.1 vulnerable?",
            "any CVEs for lodash 4.17.20",
            "look up GHSA-abcd-efgh-ijkl",
            "check CVE-2024-3651",
            "osv check pkg:npm/lodash@4.17.20",
            "is there a security advisory for requests 2.0.0",
            "what does PYSEC-2021-66 cover",
        ):
            with self.subTest(text=text):
                self.assertFalse(self._direct(text))

    def test_each_routing_branch_matches_on_its_own(self):
        # Each prompt matches exactly one branch and none of the older code-task markers.
        for text in (
            "tell me about osv",
            "check pkg:npm",
            "explain OSV-2020-111",
            "explain CVE-2024-3651",
            "summarize GHSA-abcd-efgh-ijkl",
            "summarize PYSEC-2021-66",
            "any cves today",
            "is it vulnerable",
            "any security advisory",
        ):
            with self.subTest(text=text):
                self.assertNotIn("/", text)
                self.assertFalse(self._direct(text))

    def test_near_misses_keep_the_direct_route(self):
        for text in (
            "hello",
            "osvaldo says hi",
            "which package manager is best",
            "list pkgs",
            "advisory board meeting today",
            "cvent login help",
        ):
            with self.subTest(text=text):
                self.assertTrue(self._direct(text))


class TestConversation(unittest.TestCase):
    """Test conversation management."""

    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].role, "user")
        self.assertEqual(conv.messages[0].content, "Hello")
        self.assertEqual(conv.messages[1].role, "assistant")

    def test_max_history(self):
        """Test max history limit."""
        conv = Conversation(max_history=3)

        # Add 5 messages
        for i in range(5):
            conv.add_message("user", f"Message {i}")

        # Should only keep last 3
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(conv.messages[0].content, "Message 2")
        self.assertEqual(conv.messages[2].content, "Message 4")

    def test_get_messages(self):
        """Test getting messages in API format."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.add_message("assistant", "Response")

        messages = conv.get_messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "user", "content": "Test"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Response"})

    def test_clear(self):
        """Test clearing conversation."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.clear()

        self.assertEqual(len(conv.messages), 0)

    def test_history_trim_drops_orphan_tool_result(self):
        from src.agent.conversation import ToolUseContentBlock

        conv = Conversation(max_history=2)
        conv.add_assistant_message([
            ToolUseContentBlock(
                id="tool-1",
                name="Read",
                input={"file_path": "demo.txt"},
            )
        ])
        conv.add_tool_result_message("tool-1", "contents")
        conv.add_assistant_message("finished")

        api_messages = conv.get_messages()
        self.assertLessEqual(len(conv.messages), 2)
        self.assertFalse(
            api_messages
            and api_messages[0]["role"] == "user"
            and isinstance(api_messages[0]["content"], list)
            and any(
                block.get("type") == "tool_result"
                for block in api_messages[0]["content"]
                if isinstance(block, dict)
            )
        )

    def test_zero_history_limit_is_safe(self):
        conv = Conversation(max_history=0)
        conv.add_user_message("discard me")
        self.assertEqual(conv.messages, [])

    def test_serialization(self):
        """Test conversation serialization."""
        conv = Conversation()
        conv.add_message("user", "Test")
        conv.add_message("assistant", "Response")

        # Serialize
        data = conv.to_dict()
        self.assertIn("messages", data)
        self.assertEqual(len(data["messages"]), 2)

        # Deserialize
        conv2 = Conversation.from_dict(data)
        self.assertEqual(len(conv2.messages), 2)
        self.assertEqual(conv2.messages[0].content, "Test")


class TestSession(unittest.TestCase):
    """Test session management."""

    def test_create_session(self):
        """Test session creation."""
        session = Session.create("glm", "glm-4.5")

        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.provider, "glm")
        self.assertEqual(session.model, "glm-4.5")
        self.assertEqual(len(session.conversation.messages), 0)

    def test_session_save_load(self):
        """Test session save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / ".clawd" / "sessions"

            with patch('src.agent.session.Path.home', return_value=Path(temp_dir)):
                # Create and save
                session = Session.create("glm", "glm-4.5")
                session.conversation.add_message("user", "Test message")
                session.save()

                # Load
                loaded = Session.load(session.session_id)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.session_id, session.session_id)
                self.assertEqual(len(loaded.conversation.messages), 1)
                self.assertEqual(loaded.conversation.messages[0].content, "Test message")

    def test_list_saved_orders_newest_and_skips_malformed_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / ".clawd" / "sessions"
            session_dir.mkdir(parents=True)

            def write_session(session_id: str, updated_at: str) -> None:
                data = {
                    "session_id": session_id,
                    "provider": "glm",
                    "model": "glm-4.5",
                    "conversation": Conversation().to_dict(),
                    "created_at": "2026-09-20T12:00:00",
                    "updated_at": updated_at,
                }
                (session_dir / f"{session_id}.json").write_text(
                    json.dumps(data),
                    encoding="utf-8",
                )

            write_session("older", "2026-09-21T12:00:00")
            write_session("newest", "2026-09-23T12:00:00")
            write_session("middle", "2026-09-22T12:00:00")
            (session_dir / "broken.json").write_text("{not json", encoding="utf-8")
            (session_dir / "bad name.json").write_text("{}", encoding="utf-8")

            with patch('src.agent.session.Path.home', return_value=root):
                sessions = Session.list_saved(limit=2)

            self.assertEqual([session.session_id for session in sessions], ["newest", "middle"])

    def test_list_saved_ignores_symlinked_session_files(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(temp_dir)
            session_dir = root / ".clawd" / "sessions"
            session_dir.mkdir(parents=True)
            outside = Path(outside_dir) / "external.json"
            outside.write_text(
                json.dumps({
                    "session_id": "external",
                    "provider": "glm",
                    "model": "glm-4.5",
                    "conversation": Conversation().to_dict(),
                    "created_at": "2026-09-23T12:00:00",
                    "updated_at": "2026-09-23T12:00:00",
                }),
                encoding="utf-8",
            )
            try:
                (session_dir / "external.json").symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("file symlinks are unavailable on this platform")

            with patch('src.agent.session.Path.home', return_value=root):
                sessions = Session.list_saved()

            self.assertEqual(sessions, [])


if __name__ == '__main__':
    unittest.main()
