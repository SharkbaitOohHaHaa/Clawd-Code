from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from typing import Any


def _tool_contract_sha256(
    name: str,
    description: str | None,
    input_schema: dict[str, Any],
    output_schema: Any = None,
) -> str:
    raw = json.dumps(
        {
            "name": name,
            "description": description or "",
            "inputSchema": input_schema,
            "outputSchema": output_schema,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strip_meta(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_meta(item)
            for key, item in value.items()
            if key != "_meta"
        }
    if isinstance(value, list):
        return [_strip_meta(item) for item in value]
    return value


async def _run(payload: dict[str, Any]) -> Any:
    from mcp import Client, StdioServerParameters

    action = payload.get("action")
    server = payload.get("server")
    timeout_seconds = float(payload.get("timeout_seconds") or 10.0)
    if action not in {
        "list_resources",
        "read_resource",
        "list_tools",
        "call_tool",
    }:
        raise ValueError("unsupported bridge action")
    if not isinstance(server, dict):
        raise ValueError("server configuration is required")

    command = server.get("command")
    args = server.get("args") or []
    cwd = server.get("cwd")
    if not isinstance(command, str) or not command:
        raise ValueError("server command is required")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("server args must be strings")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("server cwd must be a string")

    params = StdioServerParameters(
        command=command,
        args=args,
        cwd=cwd,
        env=None,
    )
    async with Client(params, read_timeout_seconds=timeout_seconds) as client:
        if action == "list_resources":
            result = await client.list_resources()
            data = result.model_dump(mode="json", by_alias=True)
            return data.get("resources") or []

        if action == "read_resource":
            uri = payload.get("uri")
            if not isinstance(uri, str) or not uri:
                raise ValueError("resource URI is required")
            result = await client.read_resource(uri)
            return _strip_meta(result.model_dump(mode="json", by_alias=True))

        if action == "list_tools":
            tools: list[dict[str, Any]] = []
            cursor: str | None = None
            while True:
                result = await client.list_tools(cursor=cursor)
                data = result.model_dump(mode="json", by_alias=True)
                raw_tools = data.get("tools")
                if isinstance(raw_tools, list):
                    for item in raw_tools:
                        if not isinstance(item, dict):
                            continue
                        tools.append(
                            {
                                "name": item.get("name"),
                                "title": item.get("title"),
                                "description": item.get("description"),
                                "inputSchema": item.get("inputSchema"),
                                "outputSchema": item.get("outputSchema"),
                                "annotations": item.get("annotations"),
                            }
                        )
                next_cursor = data.get("nextCursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    return _strip_meta(tools)
                cursor = next_cursor

        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        expected_contract_sha256 = payload.get("expected_contract_sha256")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool name is required")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if (
            not isinstance(expected_contract_sha256, str)
            or len(expected_contract_sha256) != 64
        ):
            raise ValueError("expected tool contract SHA-256 is required")

        current_contract_sha256: str | None = None
        cursor: str | None = None
        while True:
            tools_result = await client.list_tools(cursor=cursor)
            tools_data = tools_result.model_dump(mode="json", by_alias=True)
            raw_tools = tools_data.get("tools")
            if isinstance(raw_tools, list):
                for item in raw_tools:
                    if not isinstance(item, dict) or item.get("name") != tool_name:
                        continue
                    input_schema = item.get("inputSchema")
                    if not isinstance(input_schema, dict):
                        raise ValueError("tool input schema is unavailable")
                    description = item.get("description")
                    current_contract_sha256 = _tool_contract_sha256(
                        tool_name,
                        description if isinstance(description, str) else "",
                        input_schema,
                        item.get("outputSchema"),
                    )
                    break
            if current_contract_sha256 is not None:
                break
            next_cursor = tools_data.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor

        if current_contract_sha256 != expected_contract_sha256:
            raise ValueError("MCP tool contract changed before execution")

        result = await client.call_tool(tool_name, arguments)
        data = _strip_meta(result.model_dump(mode="json", by_alias=True))
        if data.get("resultType") != "complete":
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "MCP tool requested interactive input; Clawd does not permit MCP elicitation in this runtime.",
                    }
                ],
                "structuredContent": None,
                "resultType": data.get("resultType"),
            }
        return {
            "isError": bool(data.get("isError")),
            "content": data.get("content") if isinstance(data.get("content"), list) else [],
            "structuredContent": data.get("structuredContent"),
            "resultType": data.get("resultType") or "complete",
        }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("bridge payload must be an object")
        timeout_seconds = max(1.0, float(payload.get("timeout_seconds") or 10.0))
        result = asyncio.run(
            asyncio.wait_for(_run(payload), timeout=timeout_seconds + 1.0)
        )
        print(json.dumps({"ok": True, "result": result}, separators=(",", ":"), ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error": exc.__class__.__name__},
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
