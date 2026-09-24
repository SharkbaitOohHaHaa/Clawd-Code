from __future__ import annotations

import json
import uuid
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..permissions import protected_write_reason, sensitive_path_permission
from ..protocol import ToolResult
from ..registry import ToolSpec


def _team_file(context: ToolContext):
    return context.ensure_allowed_path(context.workspace_root / ".clawd" / "team.json")


def _team_file_permission(context: ToolContext, *, operation: str) -> PermissionResult:
    try:
        target = _team_file(context)
    except ToolPermissionError as exc:
        return PermissionResult.deny(str(exc))

    sensitive = sensitive_path_permission(target, operation=operation)
    if sensitive.behavior.value != "allow":
        return sensitive

    protected_reason = protected_write_reason(target, existing=target.exists())
    if protected_reason:
        return PermissionResult.ask(protected_reason)
    return PermissionResult.allow()


class TeamCreateTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamCreate",
            permission_policy="checked",
            description="Create lightweight team metadata. This does not spawn or run subagents.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "team_name": {"type": "string"},
                    "description": {"type": "string"},
                    "agent_type": {"type": "string"},
                },
                "required": ["team_name"],
            },
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> PermissionResult:
        return _team_file_permission(context, operation="write")

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        team_name = tool_input.get("team_name")
        if not isinstance(team_name, str) or not team_name.strip():
            raise ToolInputError("team_name must be a non-empty string")
        description = tool_input.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("description must be a string when provided")
        agent_type = tool_input.get("agent_type")
        if agent_type is not None and not isinstance(agent_type, str):
            raise ToolInputError("agent_type must be a string when provided")

        lead_agent_id = uuid.uuid4().hex[:12]
        team_file = _team_file(context)
        team_file.parent.mkdir(parents=True, exist_ok=True)
        team = {"team_name": team_name, "description": description, "agent_type": agent_type, "lead_agent_id": lead_agent_id}
        team_file.write_text(json.dumps(team, ensure_ascii=False, indent=2), encoding="utf-8")
        context.team = team
        return ToolResult(
            name="TeamCreate",
            output={"team_name": team_name, "team_file_path": str(team_file), "lead_agent_id": lead_agent_id},
        )


class TeamDeleteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TeamDelete",
            permission_policy="checked",
            description="Disband the current team context.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> PermissionResult:
        return _team_file_permission(context, operation="delete")

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.team is None:
            return ToolResult(name="TeamDelete", output={"success": False, "message": "No active team"})
        team_name = context.team.get("team_name")
        team_file = _team_file(context)
        context.team = None
        if team_file.exists():
            try:
                team_file.unlink()
            except Exception:
                pass
        return ToolResult(name="TeamDelete", output={"success": True, "message": "Team deleted", "team_name": team_name})
