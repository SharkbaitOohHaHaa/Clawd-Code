from __future__ import annotations

import dataclasses
import importlib.util
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..command_system.registry import CommandRegistry
from ..command_system.types import Command, CommandBase, PromptCommand
from ..providers import (
    PROVIDER_INFO,
    BaseProvider,
    clear_plugin_providers,
    register_plugin_provider,
    unregister_plugin_provider,
)
from ..tool_system.registry import Tool, ToolRegistry
from .runtime import compute_plugin_artifact_hash, reconcile_python_plugins


_EXTENSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESERVED_COMMAND_NAMES = frozenset({
    "exit", "quit", "q", "help",
    "save", "save-session", "load", "load-session", "resume",
    "multiline", "multiline-input", "stream", "stream-responses",
    "render-last", "render-last-response",
    "tools", "list-tools", "tool", "run-tool",
    "skill", "list-skills",
    "context", "context-usage", "compact", "compact-context",
    "init", "setup-project", "doctor", "usage", "session-usage", "clear-chat",
})


@dataclass
class PluginProviderExtension:
    name: str
    provider_class: type[BaseProvider]
    info: dict[str, Any]


@dataclass
class PluginExtensions:
    plugin_name: str
    plugin_version: str
    artifact_sha256: str
    commands: list[Command] = field(default_factory=list)
    workflows: list[PromptCommand] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    providers: list[PluginProviderExtension] = field(default_factory=list)


@dataclass
class PluginExtensionLoadResult:
    plugins: list[PluginExtensions] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)


class PluginExtensionError(RuntimeError):
    pass


def _as_sequence(value: Any, *, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise PluginExtensionError(f"{label} must be a list or tuple")
    return list(value)


def _normalize_command(
    command: Any,
    *,
    plugin_name: str,
    plugin_version: str,
    artifact_sha256: str,
) -> Command:
    if not isinstance(command, CommandBase):
        raise PluginExtensionError("COMMANDS entries must be Clawd Command objects")
    if not _EXTENSION_NAME_RE.fullmatch(command.name):
        raise PluginExtensionError(f"invalid command name: {command.name!r}")
    if command.name.lower() in _RESERVED_COMMAND_NAMES:
        raise PluginExtensionError(f"reserved command name: {command.name}")
    for alias in command.aliases:
        if not _EXTENSION_NAME_RE.fullmatch(alias):
            raise PluginExtensionError(f"invalid command alias: {alias!r}")
        if alias.lower() in _RESERVED_COMMAND_NAMES:
            raise PluginExtensionError(f"reserved command alias: {alias}")

    changes: dict[str, Any] = {
        "loaded_from": "plugin",
        "version": command.version or plugin_version,
    }
    if isinstance(command, PromptCommand):
        changes["plugin_info"] = {
            "name": plugin_name,
            "version": plugin_version,
            "artifact_sha256": artifact_sha256,
        }
    return dataclasses.replace(command, **changes)


_WORKFLOW_KEYS = frozenset({
    "name",
    "description",
    "prompt",
    "allowed_tools",
    "aliases",
    "argument_hint",
})


def _normalize_workflow(
    workflow: Any,
    *,
    plugin_name: str,
    plugin_version: str,
    artifact_sha256: str,
) -> PromptCommand:
    if not isinstance(workflow, dict):
        raise PluginExtensionError("WORKFLOWS entries must be dictionaries")

    unknown = sorted(set(workflow) - _WORKFLOW_KEYS)
    if unknown:
        raise PluginExtensionError(
            "unsupported WORKFLOWS field(s): " + ", ".join(unknown)
        )

    name = str(workflow.get("name") or "").strip()
    description = str(workflow.get("description") or "").strip()
    prompt = str(workflow.get("prompt") or "").strip()
    if not _EXTENSION_NAME_RE.fullmatch(name):
        raise PluginExtensionError(f"invalid workflow name: {name!r}")
    if name.lower() in _RESERVED_COMMAND_NAMES:
        raise PluginExtensionError(f"reserved workflow name: {name}")
    if not description:
        raise PluginExtensionError(f"workflow {name} description must be non-empty")
    if not prompt:
        raise PluginExtensionError(f"workflow {name} prompt must be non-empty")

    raw_aliases = workflow.get("aliases") or []
    if not isinstance(raw_aliases, (list, tuple)) or not all(
        isinstance(alias, str) and alias.strip()
        for alias in raw_aliases
    ):
        raise PluginExtensionError(f"workflow {name} aliases must be strings")
    aliases = [alias.strip() for alias in raw_aliases]
    for alias in aliases:
        if not _EXTENSION_NAME_RE.fullmatch(alias):
            raise PluginExtensionError(f"invalid workflow alias: {alias!r}")
        if alias.lower() in _RESERVED_COMMAND_NAMES:
            raise PluginExtensionError(f"reserved workflow alias: {alias}")

    raw_tools = workflow.get("allowed_tools")
    if not isinstance(raw_tools, (list, tuple)) or not raw_tools:
        raise PluginExtensionError(
            f"workflow {name} must declare a non-empty allowed_tools list"
        )
    if not all(isinstance(tool, str) and tool.strip() for tool in raw_tools):
        raise PluginExtensionError(
            f"workflow {name} allowed_tools must contain non-empty strings"
        )
    allowed_tools = [tool.strip() for tool in raw_tools]
    if len({tool.lower() for tool in allowed_tools}) != len(allowed_tools):
        raise PluginExtensionError(f"workflow {name} allowed_tools contains duplicates")

    argument_hint = workflow.get("argument_hint")
    if argument_hint is not None and not isinstance(argument_hint, str):
        raise PluginExtensionError(f"workflow {name} argument_hint must be a string")

    return PromptCommand(
        name=name,
        description=description,
        aliases=aliases,
        argument_hint=argument_hint,
        version=plugin_version,
        loaded_from="plugin",
        kind="workflow",
        source="plugin",
        plugin_info={
            "name": plugin_name,
            "version": plugin_version,
            "artifact_sha256": artifact_sha256,
        },
        markdown_content=prompt,
        allowed_tools=allowed_tools,
    )


def _validate_tool(tool: Any) -> Tool:
    spec_fn = getattr(tool, "spec", None)
    run_fn = getattr(tool, "run", None)
    if not callable(spec_fn) or not callable(run_fn):
        raise PluginExtensionError("TOOLS entries must implement spec() and run()")

    spec = spec_fn()
    name = str(getattr(spec, "name", "") or "")
    aliases = tuple(getattr(spec, "aliases", ()) or ())
    if not _EXTENSION_NAME_RE.fullmatch(name):
        raise PluginExtensionError(f"invalid tool name: {name!r}")
    for alias in aliases:
        if not _EXTENSION_NAME_RE.fullmatch(str(alias)):
            raise PluginExtensionError(f"invalid tool alias: {alias!r}")

    policy = getattr(spec, "permission_policy", None)
    if policy not in {"allow", "checked"}:
        raise PluginExtensionError(
            f"plugin tool {name} must use permission_policy 'checked' or read-only 'allow'"
        )
    if policy == "allow":
        if not bool(getattr(spec, "is_read_only", False)) or bool(
            getattr(spec, "is_destructive", False)
        ):
            raise PluginExtensionError(
                f"plugin tool {name} may use 'allow' only when read-only and non-destructive"
            )
    elif not callable(getattr(tool, "check_permissions", None)):
        raise PluginExtensionError(
            f"plugin tool {name} uses 'checked' without check_permissions"
        )
    return tool


def _normalize_provider(provider: Any) -> PluginProviderExtension:
    if not isinstance(provider, dict):
        raise PluginExtensionError("PROVIDERS entries must be dictionaries")

    name = str(provider.get("name") or "").strip().lower()
    provider_class = provider.get("provider_class")
    if not _EXTENSION_NAME_RE.fullmatch(name):
        raise PluginExtensionError(f"invalid provider name: {name!r}")
    if not isinstance(provider_class, type) or not issubclass(provider_class, BaseProvider):
        raise PluginExtensionError("provider_class must subclass BaseProvider")

    info = {
        "label": provider.get("label"),
        "default_base_url": provider.get("default_base_url"),
        "default_model": provider.get("default_model"),
        "available_models": provider.get("available_models"),
        "requires_api_key": provider.get("requires_api_key", True),
        "local_only": provider.get("local_only", False),
    }
    return PluginProviderExtension(
        name=name,
        provider_class=provider_class,
        info=info,
    )


def _load_module(plugin_name: str, entrypoint: Path, artifact_sha256: str) -> types.ModuleType:
    module_name = f"clawd_plugin_{plugin_name}_{artifact_sha256[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise PluginExtensionError("unable to construct plugin module loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_active_plugin_extensions() -> PluginExtensionLoadResult:
    """Import exact-hash active plugins that declare supported extensions."""
    report = reconcile_python_plugins()
    result = PluginExtensionLoadResult(issues=list(report.get("issues") or []))
    records = report.get("records") if isinstance(report.get("records"), dict) else {}

    for plugin_name in report.get("active") or []:
        record = records.get(plugin_name)
        if not isinstance(record, dict):
            continue
        extensions = set(record.get("extensions") or [])
        if not extensions.intersection({"commands", "tools", "providers", "workflows"}):
            continue

        try:
            plugin_root = Path(str(record["root"])).resolve()
            pinned_hash = str(record["operator_hash"])
            current_hash = compute_plugin_artifact_hash(plugin_root)
            if current_hash != pinned_hash:
                raise PluginExtensionError("plugin artifact changed before import")

            entrypoint = (plugin_root / str(record["entrypoint"])).resolve()
            module = _load_module(plugin_name, entrypoint, current_hash)

            after_import_hash = compute_plugin_artifact_hash(plugin_root)
            if after_import_hash != pinned_hash:
                raise PluginExtensionError("plugin artifact changed during import")

            raw_commands = _as_sequence(getattr(module, "COMMANDS", None), label="COMMANDS")
            raw_workflows = _as_sequence(getattr(module, "WORKFLOWS", None), label="WORKFLOWS")
            raw_tools = _as_sequence(getattr(module, "TOOLS", None), label="TOOLS")
            raw_providers = _as_sequence(getattr(module, "PROVIDERS", None), label="PROVIDERS")
            if raw_commands and "commands" not in extensions:
                raise PluginExtensionError("plugin exports COMMANDS without declaring commands")
            if raw_workflows and "workflows" not in extensions:
                raise PluginExtensionError("plugin exports WORKFLOWS without declaring workflows")
            if raw_tools and "tools" not in extensions:
                raise PluginExtensionError("plugin exports TOOLS without declaring tools")
            if raw_providers and "providers" not in extensions:
                raise PluginExtensionError("plugin exports PROVIDERS without declaring providers")

            commands = [
                _normalize_command(
                    command,
                    plugin_name=plugin_name,
                    plugin_version=str(record["version"]),
                    artifact_sha256=pinned_hash,
                )
                for command in raw_commands
            ]
            workflows = [
                _normalize_workflow(
                    workflow,
                    plugin_name=plugin_name,
                    plugin_version=str(record["version"]),
                    artifact_sha256=pinned_hash,
                )
                for workflow in raw_workflows
            ]
            tools = [_validate_tool(tool) for tool in raw_tools]
            providers = [_normalize_provider(provider) for provider in raw_providers]
            result.plugins.append(
                PluginExtensions(
                    plugin_name=plugin_name,
                    plugin_version=str(record["version"]),
                    artifact_sha256=pinned_hash,
                    commands=commands,
                    workflows=workflows,
                    tools=tools,
                    providers=providers,
                )
            )
        except Exception as exc:
            result.issues.append(
                {
                    "code": "plugin_extension_load_failed",
                    "subject": f"{plugin_name}: {exc}",
                }
            )

    result.issues.sort(key=lambda item: (item["code"], item["subject"]))
    return result


def _registry_command_names(registry: CommandRegistry) -> set[str]:
    names: set[str] = set()
    for command in registry.list_commands(include_hidden=True, include_disabled=True):
        names.add(command.name.lower())
        names.update(alias.lower() for alias in command.aliases)
    return names


def _tool_names(registry: ToolRegistry) -> set[str]:
    names: set[str] = set()
    for spec in registry.list_specs():
        names.add(spec.name.lower())
        names.update(alias.lower() for alias in spec.aliases)
    return names


def _plugin_commands(plugin: PluginExtensions) -> list[Command]:
    return [*plugin.commands, *plugin.workflows]


def _validate_no_collisions(
    plugin: PluginExtensions,
    *,
    tool_registry: ToolRegistry,
    command_registries: Iterable[CommandRegistry],
) -> None:
    tool_names = _tool_names(tool_registry)
    pending_tool_names: set[str] = set()
    for tool in plugin.tools:
        spec = tool.spec()
        for raw_name in (spec.name, *spec.aliases):
            name = raw_name.lower()
            if name in tool_names or name in pending_tool_names:
                raise PluginExtensionError(f"tool name/alias collision: {raw_name}")
            pending_tool_names.add(name)

    available_tool_names = tool_names.union(pending_tool_names)
    for workflow in plugin.workflows:
        for raw_tool in workflow.allowed_tools:
            if raw_tool.lower() not in available_tool_names:
                raise PluginExtensionError(
                    f"workflow {workflow.name} references unknown allowed tool: {raw_tool}"
                )

    registries = list(command_registries)
    existing_command_names: set[str] = set()
    for registry in registries:
        existing_command_names.update(_registry_command_names(registry))
    pending_command_names: set[str] = set()
    for command in _plugin_commands(plugin):
        for raw_name in (command.name, *command.aliases):
            name = raw_name.lower()
            if name in existing_command_names or name in pending_command_names:
                raise PluginExtensionError(f"command name/alias collision: {raw_name}")
            pending_command_names.add(name)


def clear_registered_plugin_commands(registry: CommandRegistry) -> None:
    for command in list(
        registry.list_commands(include_hidden=True, include_disabled=True)
    ):
        if command.loaded_from == "plugin":
            registry.unregister(command.name)


def register_plugin_provider_extensions(
    load_result: PluginExtensionLoadResult,
) -> list[dict[str, str]]:
    """Register trusted plugin providers atomically per plugin."""
    clear_plugin_providers()
    issues: list[dict[str, str]] = []

    for plugin in load_result.plugins:
        if not plugin.providers:
            continue
        registered: list[str] = []
        try:
            pending: set[str] = set()
            for provider in plugin.providers:
                name = provider.name.lower()
                if name in PROVIDER_INFO or name in pending:
                    raise PluginExtensionError(f"provider name collision: {provider.name}")
                pending.add(name)

            for provider in plugin.providers:
                register_plugin_provider(
                    provider.name,
                    provider.provider_class,
                    provider.info,
                    plugin_name=plugin.plugin_name,
                    artifact_sha256=plugin.artifact_sha256,
                )
                registered.append(provider.name)
        except Exception as exc:
            for name in reversed(registered):
                try:
                    unregister_plugin_provider(name)
                except Exception:
                    pass
            issues.append(
                {
                    "code": "plugin_provider_registration_failed",
                    "subject": f"{plugin.plugin_name}: {exc}",
                }
            )

    issues.sort(key=lambda item: (item["code"], item["subject"]))
    return issues


def register_plugin_extensions(
    load_result: PluginExtensionLoadResult,
    *,
    tool_registry: ToolRegistry,
    command_registries: Iterable[CommandRegistry],
) -> list[dict[str, str]]:
    """Register validated plugin extensions atomically per plugin."""
    issues = list(load_result.issues)
    registries: list[CommandRegistry] = []
    seen_registry_ids: set[int] = set()
    for registry in command_registries:
        if id(registry) not in seen_registry_ids:
            registries.append(registry)
            seen_registry_ids.add(id(registry))

    for plugin in load_result.plugins:
        try:
            _validate_no_collisions(
                plugin,
                tool_registry=tool_registry,
                command_registries=registries,
            )
            for tool in plugin.tools:
                tool_registry.register(tool)
            for registry in registries:
                for command in _plugin_commands(plugin):
                    registry.register(command)
        except Exception as exc:
            issues.append(
                {
                    "code": "plugin_extension_registration_failed",
                    "subject": f"{plugin.plugin_name}: {exc}",
                }
            )

    issues.sort(key=lambda item: (item["code"], item["subject"]))
    return issues
