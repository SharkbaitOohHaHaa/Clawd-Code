from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.conversation import Conversation
from src.command_system import (
    CommandEngine,
    CommandRegistry,
    create_command_context,
    register_builtin_commands,
)
from src.cost_tracker import CostTracker
from src.history import HistoryLog
from src.plugins.extensions import (
    load_active_plugin_extensions,
    register_plugin_extensions,
    register_plugin_provider_extensions,
)
from src.providers import (
    clear_plugin_providers,
    get_provider_class,
    get_provider_info,
    validate_provider_runtime_config,
)
from src.plugins.runtime import reconcile_python_plugins
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall


class PluginExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name).resolve()
        self.plugins = self.home / ".clawd" / "plugins"
        self.plugins.mkdir(parents=True)
        self.policy = self.home / ".clawd" / "python_plugins.json"

    def tearDown(self) -> None:
        clear_plugin_providers()
        self.temp.cleanup()

    def _write_plugin(
        self,
        source: str,
        *,
        name: str = "demo",
        extensions: list[str] | None = None,
    ) -> Path:
        plugin_dir = self.plugins / name
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": name,
                    "version": "1.0.0",
                    "entrypoint": "plugin.py",
                    "extensions": extensions or ["commands", "tools"],
                }
            ),
            encoding="utf-8",
        )
        (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
        return plugin_dir

    def _activate(self, name: str = "demo") -> str:
        report = reconcile_python_plugins(
            plugin_root=self.plugins,
            operator_manifest=self.policy,
        )
        digest = report["records"][name]["artifact_sha256"]
        self.policy.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "plugins": {
                        name: {
                            "enabled": True,
                            "artifact_sha256": digest,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return digest

    def _load(self):
        with patch(
            "src.plugins.runtime.default_plugin_root",
            return_value=self.plugins,
        ), patch(
            "src.plugins.runtime.default_operator_manifest_path",
            return_value=self.policy,
        ):
            return load_active_plugin_extensions()

    def test_active_plugin_registers_local_prompt_command_and_read_only_tool(self) -> None:
        self._write_plugin(
            """
from src.command_system.types import LocalCommand, LocalCommandResult, PromptCommand
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolSpec

def local_call(args, context):
    return LocalCommandResult(type="text", value="local:" + args)

local = LocalCommand(
    name="plugin-local",
    description="local plugin command",
    supports_non_interactive=True,
)
local.set_call(local_call)

prompt = PromptCommand(
    name="plugin-prompt",
    description="prompt plugin command",
    markdown_content="plugin prompt $ARGUMENTS",
    allowed_tools=["PluginRead"],
)

class ReadTool:
    def spec(self):
        return ToolSpec(
            name="PluginRead",
            description="read plugin tool",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            permission_policy="allow",
            is_read_only=True,
        )
    def run(self, tool_input, context):
        return ToolResult(name="PluginRead", output={"value": "ok"})

COMMANDS = [local, prompt]
TOOLS = [ReadTool()]
"""
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(loaded.issues, [])
        self.assertEqual(len(loaded.plugins), 1)

        command_registry = CommandRegistry()
        register_builtin_commands(command_registry)
        tool_registry = build_default_registry()
        issues = register_plugin_extensions(
            loaded,
            tool_registry=tool_registry,
            command_registries=(command_registry,),
        )
        self.assertEqual(issues, [])

        local = command_registry.get("plugin-local")
        prompt = command_registry.get("plugin-prompt")
        self.assertIsNotNone(local)
        self.assertIsNotNone(prompt)
        self.assertEqual(local.loaded_from, "plugin")
        self.assertEqual(prompt.loaded_from, "plugin")
        self.assertEqual(prompt.plugin_info["name"], "demo")
        self.assertEqual(prompt.allowed_tools, ["PluginRead"])

        context = create_command_context(
            workspace_root=self.home,
            conversation=Conversation(),
            cost_tracker=CostTracker(),
            history=HistoryLog(),
        )
        engine = CommandEngine(
            registry=command_registry,
            workspace_root=self.home,
            context=context,
        )
        local_result = asyncio.run(engine.execute("/plugin-local hello"))
        self.assertTrue(local_result.success)
        self.assertEqual(local_result.text, "local:hello")

        tool_result = tool_registry.dispatch(
            ToolCall(name="PluginRead", input={}),
            ToolContext(workspace_root=self.home),
        )
        self.assertFalse(tool_result.is_error)
        self.assertEqual(tool_result.output["value"], "ok")

    def test_checked_plugin_tool_uses_existing_permission_handler(self) -> None:
        self._write_plugin(
            """
from src.tool_system.permission_handler import PermissionResult
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolSpec

class CheckedTool:
    def spec(self):
        return ToolSpec(
            name="PluginChecked",
            description="checked plugin tool",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            permission_policy="checked",
        )
    def check_permissions(self, tool_input, context):
        return PermissionResult.ask("confirm plugin action")
    def run(self, tool_input, context):
        return ToolResult(name="PluginChecked", output={"ran": True})

TOOLS = [CheckedTool()]
COMMANDS = []
""",
            extensions=["tools"],
        )
        self._activate()
        loaded = self._load()
        registry = build_default_registry()
        issues = register_plugin_extensions(
            loaded,
            tool_registry=registry,
            command_registries=(),
        )
        self.assertEqual(issues, [])

        context = ToolContext(workspace_root=self.home)
        denied = registry.dispatch(
            ToolCall(name="PluginChecked", input={}),
            context,
        )
        self.assertTrue(denied.is_error)

        prompts = []
        context.permission_handler = lambda name, message, suggestion: (
            prompts.append((name, message)) is None,
            False,
        )
        allowed = registry.dispatch(
            ToolCall(name="PluginChecked", input={}),
            context,
        )
        self.assertFalse(allowed.is_error)
        self.assertEqual(prompts[0][0], "PluginChecked")
        self.assertIn("confirm plugin action", prompts[0][1])

    def test_mutating_allow_plugin_tool_is_rejected_before_registration(self) -> None:
        self._write_plugin(
            """
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolSpec
class UnsafeTool:
    def spec(self):
        return ToolSpec(
            name="UnsafePluginTool",
            description="unsafe",
            input_schema={"type": "object"},
            permission_policy="allow",
            is_read_only=False,
        )
    def run(self, tool_input, context):
        return ToolResult(name="UnsafePluginTool", output={})
TOOLS = [UnsafeTool()]
COMMANDS = []
""",
            extensions=["tools"],
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )

    def test_reserved_command_is_rejected(self) -> None:
        self._write_plugin(
            """
from src.command_system.types import LocalCommand, LocalCommandResult
def call(args, context):
    return LocalCommandResult(value="bad")
command = LocalCommand(name="resume", description="collision")
command.set_call(call)
COMMANDS = [command]
TOOLS = []
""",
            extensions=["commands"],
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )

    def test_tool_collision_rejects_plugin_atomically(self) -> None:
        self._write_plugin(
            """
from src.command_system.types import LocalCommand, LocalCommandResult
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolSpec
def call(args, context):
    return LocalCommandResult(value="should not register")
command = LocalCommand(name="plugin-safe", description="safe")
command.set_call(call)
class CollisionTool:
    def spec(self):
        return ToolSpec(
            name="Read",
            description="collision",
            input_schema={"type": "object"},
            permission_policy="allow",
            is_read_only=True,
        )
    def run(self, tool_input, context):
        return ToolResult(name="Read", output={})
COMMANDS = [command]
TOOLS = [CollisionTool()]
"""
        )
        self._activate()
        loaded = self._load()

        command_registry = CommandRegistry()
        register_builtin_commands(command_registry)
        tool_registry = build_default_registry()
        issues = register_plugin_extensions(
            loaded,
            tool_registry=tool_registry,
            command_registries=(command_registry,),
        )
        self.assertIn(
            "plugin_extension_registration_failed",
            {issue["code"] for issue in issues},
        )
        self.assertIsNone(command_registry.get("plugin-safe"))
        self.assertEqual(tool_registry.get("Read").spec().name, "Read")

    def test_declarative_workflow_registers_as_scoped_prompt_command(self) -> None:
        self._write_plugin(
            """
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolSpec

class WorkflowRead:
    def spec(self):
        return ToolSpec(
            name="WorkflowRead",
            description="workflow read",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            permission_policy="allow",
            is_read_only=True,
        )
    def run(self, tool_input, context):
        return ToolResult(name="WorkflowRead", output={"ok": True})

TOOLS = [WorkflowRead()]
WORKFLOWS = [{
    "name": "release-audit",
    "description": "Audit an internal release",
    "prompt": "Audit release $ARGUMENTS using only approved workflow tools.",
    "allowed_tools": ["WorkflowRead"],
    "aliases": ["audit-release"],
    "argument_hint": "<release>",
}]
""",
            extensions=["workflows", "tools"],
        )
        digest = self._activate()
        loaded = self._load()

        self.assertEqual(loaded.issues, [])
        self.assertEqual(len(loaded.plugins), 1)
        self.assertEqual(len(loaded.plugins[0].workflows), 1)
        workflow = loaded.plugins[0].workflows[0]
        self.assertEqual(workflow.kind, "workflow")
        self.assertEqual(workflow.loaded_from, "plugin")
        self.assertEqual(workflow.allowed_tools, ["WorkflowRead"])
        self.assertEqual(
            workflow.plugin_info,
            {
                "name": "demo",
                "version": "1.0.0",
                "artifact_sha256": digest,
            },
        )

        command_registry = CommandRegistry()
        register_builtin_commands(command_registry)
        tool_registry = build_default_registry()
        issues = register_plugin_extensions(
            loaded,
            tool_registry=tool_registry,
            command_registries=(command_registry,),
        )
        self.assertEqual(issues, [])
        self.assertIsNotNone(tool_registry.get("WorkflowRead"))
        registered = command_registry.get("release-audit")
        self.assertIs(registered, workflow)
        self.assertIs(command_registry.get("audit-release"), workflow)

        context = create_command_context(
            workspace_root=self.home,
            conversation=Conversation(),
            cost_tracker=CostTracker(),
            history=HistoryLog(),
        )
        engine = CommandEngine(
            registry=command_registry,
            workspace_root=self.home,
            context=context,
        )
        result = asyncio.run(engine.execute("/release-audit v1.2.3"))
        self.assertTrue(result.success)
        self.assertEqual(result.result_type, "prompt")
        self.assertEqual(
            result.prompt_content,
            [{
                "type": "text",
                "text": "Audit release v1.2.3 using only approved workflow tools.",
            }],
        )

    def test_workflow_requires_explicit_non_empty_tool_scope(self) -> None:
        self._write_plugin(
            """
WORKFLOWS = [{
    "name": "unsafe-workflow",
    "description": "missing tool scope",
    "prompt": "Do work",
    "allowed_tools": [],
}]
""",
            extensions=["workflows"],
        )
        self._activate()
        loaded = self._load()

        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )
        self.assertIn("non-empty allowed_tools", loaded.issues[0]["subject"])

    def test_workflow_missing_tool_scope_fails_closed(self) -> None:
        self._write_plugin(
            """
WORKFLOWS = [{
    "name": "missing-scope-workflow",
    "description": "missing tool scope",
    "prompt": "Do work",
}]
""",
            extensions=["workflows"],
        )
        self._activate()
        loaded = self._load()

        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )
        self.assertIn("non-empty allowed_tools", loaded.issues[0]["subject"])

    def test_workflow_unsupported_field_fails_closed(self) -> None:
        self._write_plugin(
            """
WORKFLOWS = [{
    "name": "unsupported-field-workflow",
    "description": "unsupported field",
    "prompt": "Do work",
    "allowed_tools": ["Read"],
    "model": "must-not-be-accepted",
}]
""",
            extensions=["workflows"],
        )
        self._activate()
        loaded = self._load()

        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )
        self.assertIn("unsupported WORKFLOWS field", loaded.issues[0]["subject"])

    def test_workflows_export_requires_manifest_declaration(self) -> None:
        self._write_plugin(
            """
WORKFLOWS = [{
    "name": "declared-wrong",
    "description": "manifest mismatch",
    "prompt": "Do work",
    "allowed_tools": ["Read"],
}]
""",
            extensions=["commands"],
        )
        self._activate()
        loaded = self._load()

        self.assertEqual(loaded.plugins, [])
        self.assertIn(
            "plugin_extension_load_failed",
            {issue["code"] for issue in loaded.issues},
        )
        self.assertIn("without declaring workflows", loaded.issues[0]["subject"])

    def test_workflow_unknown_tool_fails_registration_atomically(self) -> None:
        self._write_plugin(
            """
WORKFLOWS = [{
    "name": "unknown-tool-workflow",
    "description": "references missing tool",
    "prompt": "Do work",
    "allowed_tools": ["DefinitelyMissingTool"],
}]
""",
            extensions=["workflows"],
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(loaded.issues, [])

        command_registry = CommandRegistry()
        register_builtin_commands(command_registry)
        tool_registry = build_default_registry()
        issues = register_plugin_extensions(
            loaded,
            tool_registry=tool_registry,
            command_registries=(command_registry,),
        )

        self.assertIn(
            "plugin_extension_registration_failed",
            {issue["code"] for issue in issues},
        )
        self.assertIsNone(command_registry.get("unknown-tool-workflow"))
        self.assertIn("unknown allowed tool", issues[0]["subject"])

    def test_workflow_and_command_share_collision_namespace(self) -> None:
        self._write_plugin(
            """
from src.command_system.types import PromptCommand

COMMANDS = [
    PromptCommand(
        name="same-name",
        description="ordinary command",
        markdown_content="ordinary",
    )
]
WORKFLOWS = [{
    "name": "same-name",
    "description": "workflow collision",
    "prompt": "workflow",
    "allowed_tools": ["Read"],
}]
""",
            extensions=["commands", "workflows"],
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(loaded.issues, [])

        command_registry = CommandRegistry()
        register_builtin_commands(command_registry)
        issues = register_plugin_extensions(
            loaded,
            tool_registry=build_default_registry(),
            command_registries=(command_registry,),
        )

        self.assertIn(
            "plugin_extension_registration_failed",
            {issue["code"] for issue in issues},
        )
        self.assertIsNone(command_registry.get("same-name"))

    def test_provider_only_plugin_registers_without_constructing_provider(self) -> None:
        marker = self.home / "provider_constructed.txt"
        self._write_plugin(
            f"""
from pathlib import Path
from src.providers.base import BaseProvider, ChatResponse

class LocalProvider(BaseProvider):
    def __init__(self, api_key, base_url=None, model=None):
        Path({str(marker)!r}).write_text("constructed", encoding="utf-8")
        super().__init__(api_key, base_url, model or "local-model")
    def chat(self, messages, tools=None, **kwargs):
        return ChatResponse(content="local", model=self.model, usage={{}}, finish_reason="stop")
    def chat_stream(self, messages, tools=None, **kwargs):
        if False:
            yield ""
    def get_available_models(self):
        return ["local-model"]

PROVIDERS = [{{
    "name": "local-demo",
    "label": "Local Demo",
    "provider_class": LocalProvider,
    "default_base_url": "http://127.0.0.1:11434/v1",
    "default_model": "local-model",
    "available_models": ["local-model"],
    "requires_api_key": False,
    "local_only": True,
}}]
""",
            extensions=["providers"],
        )
        self._activate()
        loaded = self._load()

        self.assertEqual(loaded.issues, [])
        self.assertEqual(len(loaded.plugins), 1)
        self.assertEqual(len(loaded.plugins[0].providers), 1)
        self.assertFalse(marker.exists())

        issues = register_plugin_provider_extensions(loaded)
        self.assertEqual(issues, [])
        self.assertFalse(marker.exists())

        info = get_provider_info("local-demo")
        self.assertFalse(info["requires_api_key"])
        self.assertTrue(info["local_only"])
        self.assertEqual(get_provider_class("local-demo").__name__, "LocalProvider")

        validate_provider_runtime_config(
            "local-demo",
            {
                "api_key": "",
                "base_url": "http://localhost:11434/v1",
                "default_model": "local-model",
            },
        )
        with self.assertRaises(ValueError):
            validate_provider_runtime_config(
                "local-demo",
                {
                    "api_key": "",
                    "base_url": "https://example.com/v1",
                    "default_model": "local-model",
                },
            )

        instance = get_provider_class("local-demo")(
            api_key="",
            base_url="http://127.0.0.1:11434/v1",
            model="local-model",
        )
        self.assertTrue(marker.exists())
        self.assertEqual(instance.model, "local-model")

    def test_remote_plugin_provider_requires_api_key(self) -> None:
        self._write_plugin(
            """
from src.providers.base import BaseProvider, ChatResponse

class RemoteProvider(BaseProvider):
    def chat(self, messages, tools=None, **kwargs):
        return ChatResponse(content="remote", model=self.model, usage={}, finish_reason="stop")
    def chat_stream(self, messages, tools=None, **kwargs):
        if False:
            yield ""
    def get_available_models(self):
        return ["remote-model"]

PROVIDERS = [{
    "name": "remote-demo",
    "label": "Remote Demo",
    "provider_class": RemoteProvider,
    "default_base_url": "https://provider.example/v1",
    "default_model": "remote-model",
    "available_models": ["remote-model"],
    "requires_api_key": True,
    "local_only": False,
}]
""",
            extensions=["providers"],
        )
        self._activate()
        loaded = self._load()
        self.assertEqual(register_plugin_provider_extensions(loaded), [])

        with self.assertRaises(ValueError):
            validate_provider_runtime_config(
                "remote-demo",
                {
                    "api_key": "",
                    "base_url": "https://provider.example/v1",
                    "default_model": "remote-model",
                },
            )
        validate_provider_runtime_config(
            "remote-demo",
            {
                "api_key": "test-key",
                "base_url": "https://provider.example/v1",
                "default_model": "remote-model",
            },
        )
        with self.assertRaises(ValueError):
            validate_provider_runtime_config(
                "remote-demo",
                {
                    "api_key": "test-key",
                    "base_url": "file:///tmp/provider.sock",
                    "default_model": "remote-model",
                },
            )
        with self.assertRaises(ValueError):
            validate_provider_runtime_config(
                "remote-demo",
                {
                    "api_key": "test-key",
                    "base_url": "https://user:pass@provider.example/v1",
                    "default_model": "remote-model",
                },
            )

    def test_plugin_provider_collision_with_builtin_is_rejected(self) -> None:
        self._write_plugin(
            """
from src.providers.base import BaseProvider, ChatResponse

class CollisionProvider(BaseProvider):
    def chat(self, messages, tools=None, **kwargs):
        return ChatResponse(content="", model=self.model, usage={}, finish_reason="stop")
    def chat_stream(self, messages, tools=None, **kwargs):
        if False:
            yield ""
    def get_available_models(self):
        return ["fake"]

PROVIDERS = [{
    "name": "openai",
    "label": "Collision",
    "provider_class": CollisionProvider,
    "default_base_url": "https://provider.example/v1",
    "default_model": "fake",
    "available_models": ["fake"],
    "requires_api_key": True,
    "local_only": False,
}]
""",
            extensions=["providers"],
        )
        self._activate()
        loaded = self._load()
        original = get_provider_class("openai")

        issues = register_plugin_provider_extensions(loaded)

        self.assertIn(
            "plugin_provider_registration_failed",
            {issue["code"] for issue in issues},
        )
        self.assertIs(get_provider_class("openai"), original)


if __name__ == "__main__":
    unittest.main()
