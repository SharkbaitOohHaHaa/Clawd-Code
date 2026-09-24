from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol

from ..activity_ledger import append_activity, append_change
from .context import ToolContext
from .permission_handler import PermissionResult
from .permissions import SensitivePathBehavior, sensitive_path_decision
from .protocol import ToolCall, ToolResult
from .schema_validation import validate_json_schema


PermissionPolicy = Literal["allow", "checked", "self_gated", "delegated"]
_VALID_PERMISSION_POLICIES = frozenset({"allow", "checked", "self_gated", "delegated"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    permission_policy: PermissionPolicy | None = None
    aliases: tuple[str, ...] = ()
    is_read_only: bool = False
    is_destructive: bool = False
    strict: bool = False
    max_result_size_chars: int = 20_000


class Tool(Protocol):
    def spec(self) -> ToolSpec: ...

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult: ...

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult: ...


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: list[Tool] = []
        self._by_name: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec()
        policy = spec.permission_policy
        if policy not in _VALID_PERMISSION_POLICIES:
            raise ValueError(
                f"tool {spec.name} must declare permission_policy as one of "
                f"{sorted(_VALID_PERMISSION_POLICIES)}"
            )
        checker = getattr(tool, "check_permissions", None)
        has_checker = callable(checker)
        if policy == "checked" and not has_checker:
            raise ValueError(f"tool {spec.name} declares checked permission policy without check_permissions")
        if policy != "checked" and has_checker:
            raise ValueError(
                f"tool {spec.name} implements check_permissions but declares permission_policy={policy}"
            )

        key = spec.name.lower()
        if key in self._by_name:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools.append(tool)
        self._by_name[key] = tool
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in self._by_name:
                raise ValueError(f"duplicate tool alias: {alias}")
            self._by_name[alias_key] = tool

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools]

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name.lower())

    def _record_change(self, spec: ToolSpec, result: ToolResult, context: ToolContext) -> None:
        if result.is_error or spec.name not in {"Write", "Edit", "NotebookEdit", "DataTransform", "ExitPlanMode"}:
            return
        if not isinstance(result.output, dict):
            return
        raw_path = result.output.get("filePath")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        try:
            resolved = Path(raw_path).expanduser().resolve()
            workspace = Path(context.workspace_root).expanduser().resolve()
            relative = resolved.relative_to(workspace)
        except (OSError, RuntimeError, ValueError):
            return
        if sensitive_path_decision(resolved, operation="write").behavior is not SensitivePathBehavior.ALLOW:
            return

        operation = "write"
        if spec.name == "Write":
            raw_operation = str(result.output.get("type") or "").strip().lower()
            operation = raw_operation if raw_operation in {"create", "update"} else "write"
        elif spec.name == "Edit":
            operation = "edit"
        elif spec.name == "NotebookEdit":
            operation = "notebook_edit"
        elif spec.name == "DataTransform":
            operation = "data_transform"
        elif spec.name == "ExitPlanMode":
            operation = "plan_write"
        append_change(spec.name, relative.as_posix(), operation)

    def _finalize_result(
        self,
        *,
        spec: ToolSpec,
        call: ToolCall,
        result: ToolResult,
        context: ToolContext,
    ) -> ToolResult:
        if result.tool_use_id is None and call.tool_use_id is not None:
            result = ToolResult(
                name=result.name,
                output=result.output,
                is_error=result.is_error,
                tool_use_id=call.tool_use_id,
                content_type=result.content_type,
            )

        if context.instrumentation_enabled:
            if spec.name.lower() == "skill":
                if isinstance(result.output, dict):
                    skill_name = str(result.output.get("commandName") or "").strip()
                    if skill_name:
                        append_activity(
                            "skill",
                            skill_name,
                            status="error" if result.is_error else "ok",
                        )
            else:
                append_activity(
                    "tool",
                    spec.name,
                    status="error" if result.is_error else "ok",
                )
            self._record_change(spec, result, context)
        return result

    def _record_exception_activity(
        self,
        *,
        spec: ToolSpec,
        call: ToolCall,
        context: ToolContext,
    ) -> None:
        if not context.instrumentation_enabled:
            return
        if spec.name.lower() == "skill":
            raw_name = call.input.get("skill")
            if isinstance(raw_name, str):
                skill_name = raw_name.strip().lstrip("/")
                if skill_name:
                    append_activity("skill", skill_name, status="error")
            return
        append_activity("tool", spec.name, status="error")

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        spec = tool.spec()
        try:
            context.ensure_tool_allowed(
                spec.name,
                aliases=(call.name, *spec.aliases),
            )
            validate_json_schema(call.input, spec.input_schema, root_name=spec.name)

            # Permission behavior is explicit; missing policy is rejected at registration.
            if spec.permission_policy == "checked":
                permission_result = tool.check_permissions(call.input, context)
            else:
                permission_result = PermissionResult.allow()
        except Exception:
            self._record_exception_activity(spec=spec, call=call, context=context)
            raise
        if permission_result.behavior.value == "deny":
            return self._finalize_result(
                spec=spec,
                call=call,
                context=context,
                result=ToolResult(
                    name=spec.name,
                    output={"error": permission_result.message or "permission denied"},
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                ),
            )
        if permission_result.behavior.value == "ask":
            # Need user interaction
            if context.permission_handler is None:
                # No handler available, deny by default
                return self._finalize_result(
                    spec=spec,
                    call=call,
                    context=context,
                    result=ToolResult(
                        name=spec.name,
                        output={"error": permission_result.message or "permission required but no handler available"},
                        is_error=True,
                        tool_use_id=call.tool_use_id,
                    ),
                )
            # Call the permission handler
            try:
                allowed, _ = context.permission_handler(
                    spec.name,
                    permission_result.message or f"Tool '{spec.name}' requires permission",
                    permission_result.suggestion,
                )
            except Exception:
                self._record_exception_activity(spec=spec, call=call, context=context)
                raise
            if not allowed:
                return self._finalize_result(
                    spec=spec,
                    call=call,
                    context=context,
                    result=ToolResult(
                        name=spec.name,
                        output={"error": "permission denied by user"},
                        is_error=True,
                        tool_use_id=call.tool_use_id,
                    ),
                )
            # User allowed - proceed with potentially updated input
            if permission_result.updated_input:
                call = ToolCall(
                    name=call.name,
                    input=permission_result.updated_input,
                    tool_use_id=call.tool_use_id,
                )

        try:
            result = tool.run(call.input, context)
        except Exception:
            self._record_exception_activity(spec=spec, call=call, context=context)
            raise
        return self._finalize_result(
            spec=spec,
            call=call,
            result=result,
            context=context,
        )

