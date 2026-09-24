from __future__ import annotations

from typing import Any

from ...memory import MemoryStore
from ...memory.default_core import CORE_RULES, CORE_RULES_VERSION
from ..context import ToolContext
from ..errors import ToolInputError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec


_CONFIRM_OPERATIONS = {
    "initialize",
    "approve",
    "revoke",
    "supersede",
    "update_project",
    "update_core",
    "recover",
}


class MemoryTool:
    """Controlled interface to JR's persistent memory store."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Memory",
            permission_policy="checked",
            description=(
                "Inspect or update JR persistent memory. Suggestions are CANDIDATE only; "
                "durable activation and policy changes require current user confirmation."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "status", "list", "initialize", "propose", "approve", "revoke",
                            "supersede", "update_project", "update_core", "checkpoint", "recover",
                        ],
                    },
                    "memory_id": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["lesson", "preference"]},
                    "content": {"type": "string"},
                    "source": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "scope": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "enum": ["global", "project"]},
                            "project": {"type": "string"},
                        },
                        "required": ["kind", "project"],
                    },
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "review_at": {"type": "string"},
                    "expires_at": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "superseded_by": {"type": "string"},
                    "project": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["operation"],
            },
            is_destructive=False,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        operation = tool_input.get("operation")
        if operation not in _CONFIRM_OPERATIONS:
            return PermissionResult.allow()
        labels = {
            "initialize": "Initialize JR persistent memory storage?",
            "approve": "Approve this memory as durable ACTIVE memory?",
            "revoke": "Revoke this durable memory?",
            "supersede": "Supersede this durable memory with another entry?",
            "update_project": "Update this project's persistent memory artifact?",
            "update_core": "Update JR CORE-RULES through a versioned, audited policy change?",
            "recover": "Record audit recovery and re-enable memory mutations?",
        }
        return PermissionResult.ask(labels.get(str(operation), "Allow persistent memory mutation?"))

    def _required_text(self, data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ToolInputError(f"{key} must be a non-empty string")
        return value

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        operation = tool_input["operation"]
        if operation == "initialize":
            store = MemoryStore(initialize=True)
            if not store.core_path.exists():
                store.update_core_rules(
                    CORE_RULES,
                    version=CORE_RULES_VERSION,
                    actor="owner-confirmed",
                    reason="Initialize frozen JR core communication and memory rules.",
                    source="owner-approved JR persistent-memory specification",
                    evidence=["implementation acceptance contract"],
                )
                store.append_checkpoint(
                    actor="system",
                    reason="Checkpoint after initial JR core policy installation.",
                )
            return ToolResult(name="Memory", output={"success": True, **store.status()})

        store = MemoryStore()
        if operation == "status":
            return ToolResult(name="Memory", output=store.status())
        if operation == "list":
            return ToolResult(name="Memory", output={"memories": store.list_memories()})
        if operation == "checkpoint":
            event = store.append_checkpoint()
            return ToolResult(name="Memory", output={"success": True, "event": event})

        reason = str(tool_input.get("reason") or "").strip()
        if operation in {"approve", "revoke", "supersede", "update_project", "update_core", "recover"}:
            if not reason:
                raise ToolInputError("reason is required for approved memory mutations")

        if operation == "propose":
            memory_type = self._required_text(tool_input, "memory_type")
            content = self._required_text(tool_input, "content")
            source = self._required_text(tool_input, "source")
            evidence = tool_input.get("evidence", [])
            scope = tool_input.get("scope")
            confidence = tool_input.get("confidence", 80)
            if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
                raise ToolInputError("evidence must be an array of strings")
            if not isinstance(scope, dict):
                raise ToolInputError("scope is required for memory candidates")
            if not isinstance(confidence, int) or isinstance(confidence, bool):
                raise ToolInputError("confidence must be an integer")
            record = store.propose(
                memory_type=memory_type,
                content=content,
                source=source,
                evidence=evidence,
                scope=scope,
                confidence=confidence,
                actor="JR",
                review_at=str(tool_input.get("review_at") or ""),
                expires_at=str(tool_input.get("expires_at") or ""),
                tags=tool_input.get("tags", []),
            )
            return ToolResult(name="Memory", output={"success": True, "candidate": record})

        if operation in {"approve", "revoke"}:
            memory_id = self._required_text(tool_input, "memory_id")
            method = store.approve if operation == "approve" else store.revoke
            record = method(memory_id, actor="owner-confirmed", reason=reason)
            return ToolResult(name="Memory", output={"success": True, "transition": record})

        if operation == "supersede":
            memory_id = self._required_text(tool_input, "memory_id")
            replacement = self._required_text(tool_input, "superseded_by")
            record = store.supersede(
                memory_id,
                superseded_by=replacement,
                actor="owner-confirmed",
                reason=reason,
            )
            return ToolResult(name="Memory", output={"success": True, "transition": record})

        if operation == "update_project":
            project = self._required_text(tool_input, "project")
            content = self._required_text(tool_input, "content")
            source = self._required_text(tool_input, "source")
            version = str(tool_input.get("version") or "")
            event = store.update_project_memory(
                project,
                content,
                version=version,
                actor="owner-confirmed",
                reason=reason,
                source=source,
                evidence=tool_input.get("evidence", []),
                confidence=int(tool_input.get("confidence", 100)),
                review_at=str(tool_input.get("review_at") or ""),
                expires_at=str(tool_input.get("expires_at") or ""),
            )
            return ToolResult(name="Memory", output={"success": True, "event": event})

        if operation == "update_core":
            content = self._required_text(tool_input, "content")
            version = self._required_text(tool_input, "version")
            source = self._required_text(tool_input, "source")
            event = store.update_core_rules(
                content,
                version=version,
                actor="owner-confirmed",
                reason=reason,
                source=source,
                evidence=tool_input.get("evidence", []),
                confidence=int(tool_input.get("confidence", 100)),
                review_at=str(tool_input.get("review_at") or ""),
                expires_at=str(tool_input.get("expires_at") or ""),
            )
            return ToolResult(name="Memory", output={"success": True, "event": event})

        if operation == "recover":
            result = store.recover_audit(actor="owner-confirmed", reason=reason)
            return ToolResult(name="Memory", output={"success": True, **result})

        raise ToolInputError(f"unsupported memory operation: {operation}")
