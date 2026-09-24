from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec


class SkillTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Skill",
            permission_policy="self_gated",
            description="Execute an approved, active prompt-based SKILL.md skill.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "skill": {"type": "string"},
                    "args": {"type": "string"},
                },
                "required": ["skill"],
            },
            is_destructive=False,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        skill_name = tool_input.get("skill")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ToolInputError("skill must be a non-empty string")
        args = tool_input.get("args", "")
        if not isinstance(args, str):
            raise ToolInputError("args must be a string when provided")

        normalized = skill_name.strip()
        if normalized.startswith("/"):
            normalized = normalized[1:]

        from ...skills.argument_substitution import substitute_arguments
        from ...skills.loader import get_all_skills

        cwd = context.cwd or context.workspace_root
        skills = get_all_skills(project_root=cwd, enforce_trust=True)
        skill = next((item for item in skills if item.name == normalized), None)
        if skill is None:
            return ToolResult(
                name="Skill",
                output={
                    "success": False,
                    "error": f"skill is unknown, inactive, or not approved: {normalized}",
                    "commandName": normalized,
                },
                is_error=True,
            )
        if skill.disable_model_invocation:
            return ToolResult(
                name="Skill",
                output={
                    "success": False,
                    "error": f"skill {normalized} cannot be invoked (disable-model-invocation: true)",
                    "commandName": normalized,
                },
                is_error=True,
            )

        content = substitute_arguments(
            skill.markdown_content,
            args,
            append_if_no_placeholder=True,
            argument_names=skill.arg_names,
        )
        if skill.skill_root:
            content = f"Base directory for this approved skill: {skill.skill_root}\n\n{content}"
            skill_dir = skill.skill_root.replace("\\", "/")
            content = content.replace("$" + "{CLAUDE_SKILL_DIR}", skill_dir)

        if skill.allowed_tools:
            context.restrict_tool_allowlist(list(skill.allowed_tools))

        return ToolResult(
            name="Skill",
            output={
                "success": True,
                "commandName": normalized,
                "status": "inline",
                "allowedTools": list(skill.allowed_tools) if skill.allowed_tools else [],
                "model": skill.model,
                "loadedFrom": skill.loaded_from,
                "skillRoot": skill.skill_root,
                "prompt": content,
            },
        )
