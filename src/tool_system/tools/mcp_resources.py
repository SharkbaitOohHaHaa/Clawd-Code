from __future__ import annotations

from typing import Any, Protocol

from ..context import ToolContext
from ..errors import ToolInputError
from ..mcp_resource_runtime import MCPResourceRuntimeError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec


class _McpResourceClient(Protocol):
    def list_resources(self) -> list[dict[str, Any]]: ...

    def can_read_resource(self, uri: str) -> bool: ...

    def read_resource(self, uri: str) -> dict[str, Any]: ...


class ListMcpResourcesTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ListMcpResourcesTool",
            permission_policy="checked",
            description="List resources from operator-configured local stdio MCP servers.",
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
            return PermissionResult.deny("MCP resource configuration is invalid")
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
                    name="ListMcpResourcesTool",
                    output={"error": f"mcp server not configured: {server}"},
                    is_error=True,
                )
            clients = [(server, client)]
        else:
            clients = list(context.mcp_clients.items())

        resources: list[dict[str, Any]] = []
        for name, client in clients:
            try:
                items = client.list_resources()
            except MCPResourceRuntimeError:
                resources.append({"server": name, "error": "resource listing failed"})
                continue
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        resources.append(
                            {
                                "uri": str(item.get("uri", "")),
                                "name": str(item.get("name", "")),
                                "mimeType": item.get("mimeType"),
                                "description": item.get("description"),
                                "server": name,
                            }
                        )
        return ToolResult(name="ListMcpResourcesTool", output=resources)


class ReadMcpResourceTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReadMcpResourceTool",
            permission_policy="checked",
            description=(
                "Read a resource URI that was previously advertised by the same "
                "operator-configured local stdio MCP server."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
                "required": ["server", "uri"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        if context.mcp_config_error:
            return PermissionResult.deny("MCP resource configuration is invalid")
        server = tool_input.get("server")
        uri = tool_input.get("uri")
        if not isinstance(server, str) or not server.strip():
            return PermissionResult.allow()
        if not isinstance(uri, str) or not uri.strip():
            return PermissionResult.allow()
        client = context.mcp_clients.get(server)
        if client is None:
            return PermissionResult.deny(f"MCP server is not configured: {server}")
        checker = getattr(client, "can_read_resource", None)
        if not callable(checker) or not bool(checker(uri)):
            return PermissionResult.deny(
                "MCP resource URI was not advertised by this server in the current Clawd session; list resources first"
            )
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        server = tool_input.get("server")
        uri = tool_input.get("uri")
        if not isinstance(server, str) or not server.strip():
            raise ToolInputError("server must be a non-empty string")
        if not isinstance(uri, str) or not uri.strip():
            raise ToolInputError("uri must be a non-empty string")

        client = context.mcp_clients.get(server)
        if client is None:
            return ToolResult(
                name="ReadMcpResourceTool",
                output={"error": f"mcp server not configured: {server}"},
                is_error=True,
            )
        try:
            out = client.read_resource(uri)
        except MCPResourceRuntimeError as exc:
            return ToolResult(
                name="ReadMcpResourceTool",
                output={"error": str(exc)},
                is_error=True,
            )
        if isinstance(out, dict) and "contents" in out:
            raw_contents = out.get("contents")
            contents: list[dict[str, Any]] = []
            if isinstance(raw_contents, list):
                for item in raw_contents:
                    if not isinstance(item, dict):
                        continue
                    clean = {
                        "uri": uri,
                        "mimeType": item.get("mimeType"),
                    }
                    if isinstance(item.get("text"), str):
                        clean["text"] = item["text"]
                    elif isinstance(item.get("blob"), str):
                        clean["blob"] = item["blob"]
                    contents.append(clean)
            return ToolResult(
                name="ReadMcpResourceTool",
                output={"contents": contents},
            )
        return ToolResult(
            name="ReadMcpResourceTool",
            output={
                "contents": [
                    {
                        "uri": uri,
                        **(out if isinstance(out, dict) else {"text": str(out)}),
                    }
                ]
            },
        )
