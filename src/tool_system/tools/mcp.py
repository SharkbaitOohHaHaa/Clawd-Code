from __future__ import annotations

from typing import Any, Protocol

from ..context import ToolContext
from ..errors import ToolInputError
from ..mcp_resource_runtime import MCPResourceRuntimeError, MCPToolPolicy
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec


class MCPClient(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...

    def tool_policy(self, tool_name: str) -> MCPToolPolicy | None: ...

    def can_call_tool(self, tool_name: str) -> bool: ...

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class ListMcpToolsTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ListMcpToolsTool",
            permission_policy="checked",
            description=(
                "List operator-authorized tools advertised by configured local stdio MCP servers. "
                "Server annotations are shown as hints only and never grant execution authority."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"server": {"type": "string"}},
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        if context.mcp_config_error:
            return PermissionResult.deny("MCP configuration is invalid")
        server = tool_input.get("server")
        if server is None:
            return PermissionResult.allow()
        if not isinstance(server, str) or not server.strip():
            return PermissionResult.allow()
        if server not in context.mcp_clients:
            return PermissionResult.deny(f"MCP server is not configured: {server}")
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input.get("server")
        if server is not None and (not isinstance(server, str) or not server.strip()):
            raise ToolInputError("server must be a non-empty string when provided")

        clients: list[tuple[str, Any]]
        if server:
            client = context.mcp_clients.get(server)
            if client is None:
                return ToolResult(
                    name="ListMcpToolsTool",
                    output={"error": f"mcp server not configured: {server}"},
                    is_error=True,
                )
            clients = [(server, client)]
        else:
            clients = list(context.mcp_clients.items())

        tools: list[dict[str, Any]] = []
        for server_name, client in clients:
            try:
                items = client.list_tools()
            except MCPResourceRuntimeError:
                tools.append(
                    {
                        "server": server_name,
                        "error": "MCP tool discovery failed",
                    }
                )
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                tools.append({"server": server_name, **item})
        return ToolResult(name="ListMcpToolsTool", output=tools)


class MCPTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="MCP",
            permission_policy="checked",
            description=(
                "Call an operator-authorized tool exposed by a configured local stdio MCP server. "
                "The tool must be freshly advertised, contract-pinned, and permitted by local policy."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "input": {"type": "object"},
                },
                "required": ["server", "tool"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        if context.mcp_config_error:
            return PermissionResult.deny("MCP configuration is invalid")

        server = tool_input.get("server")
        tool_name = tool_input.get("tool")
        if not isinstance(server, str) or not server.strip():
            return PermissionResult.allow()
        if not isinstance(tool_name, str) or not tool_name.strip():
            return PermissionResult.allow()

        client = context.mcp_clients.get(server)
        if client is None:
            return PermissionResult.deny(f"MCP server is not configured: {server}")
        can_call = getattr(client, "can_call_tool", None)
        policy_getter = getattr(client, "tool_policy", None)
        if not callable(can_call) or not callable(policy_getter):
            return PermissionResult.deny(
                f"MCP server does not support trusted tool execution: {server}"
            )
        if not bool(can_call(tool_name)):
            return PermissionResult.deny(
                "MCP tool was not freshly advertised with a matching operator-pinned contract; "
                "list MCP tools first and review any contract mismatch"
            )

        policy = policy_getter(tool_name)
        if not isinstance(policy, MCPToolPolicy):
            return PermissionResult.deny(
                f"MCP tool has no operator policy: {server}/{tool_name}"
            )
        if policy.approval == "allow":
            return PermissionResult.allow()

        classification = (
            f"read_only={str(policy.read_only).lower()}, "
            f"open_world={str(policy.open_world).lower()}"
        )
        return PermissionResult.ask(
            f"MCP tool '{server}/{tool_name}' requires explicit approval for this call "
            f"({classification}).",
            suggestion="require-explicit-yes",
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input.get("server")
        tool_name = tool_input.get("tool")
        args = tool_input.get("input") or {}
        if not isinstance(server, str) or not server.strip():
            raise ToolInputError("server must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolInputError("tool must be a non-empty string")
        if not isinstance(args, dict):
            raise ToolInputError("input must be an object when provided")

        client = context.mcp_clients.get(server)
        if client is None:
            return ToolResult(
                name="MCP",
                output={"error": f"mcp server not configured: {server}"},
                is_error=True,
            )

        try:
            out = client.call_tool(tool_name, args)
        except MCPResourceRuntimeError as exc:
            return ToolResult(
                name="MCP",
                output={"error": str(exc)},
                is_error=True,
            )

        is_error = bool(out.get("isError")) if isinstance(out, dict) else False
        return ToolResult(
            name="MCP",
            output={
                "server": server,
                "tool": tool_name,
                "output": out,
            },
            is_error=is_error,
        )
