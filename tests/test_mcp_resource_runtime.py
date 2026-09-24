from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tool_system.errors import ToolInputError
from src.tool_system.mcp_resource_runtime import (
    MCPResourceConfigError,
    MCPResourceRuntimeError,
    MCPStdioServerConfig,
    MCPToolPolicy,
    TrustedMCPResourceClient,
    load_mcp_resource_clients,
    mcp_tool_contract_sha256,
)


class MCPResourceRuntimeTests(unittest.TestCase):
    def test_manifest_loads_only_explicit_local_stdio_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = root / "server.exe"
            command.write_bytes(b"test")
            manifest = root / "mcp_servers.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "servers": {
                            "docs": {
                                "transport": "stdio",
                                "command": str(command),
                                "args": ["--serve"],
                                "cwd": str(root),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "runtime"
            runtime.mkdir()
            with patch(
                "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                return_value=True,
            ):
                clients = load_mcp_resource_clients(manifest, runtime_root=runtime)

            self.assertEqual(set(clients), {"docs"})
            self.assertEqual(clients["docs"].config.command, command.resolve())
            self.assertEqual(clients["docs"].config.args, ("--serve",))
            self.assertEqual(clients["docs"].config.cwd, root.resolve())

    def test_manifest_rejects_http_and_environment_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = root / "server.exe"
            command.write_bytes(b"test")
            runtime = root / "runtime"
            runtime.mkdir()

            for name, record, expected in (
                (
                    "http",
                    {"transport": "streamable-http", "command": str(command)},
                    "must use transport='stdio'",
                ),
                (
                    "env",
                    {
                        "transport": "stdio",
                        "command": str(command),
                        "env": {"TOKEN": "secret"},
                    },
                    "unsupported settings: env",
                ),
            ):
                with self.subTest(name=name):
                    manifest = root / f"{name}.json"
                    manifest.write_text(
                        json.dumps({"schema_version": 1, "servers": {"srv": record}}),
                        encoding="utf-8",
                    )
                    with patch(
                        "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                        return_value=True,
                    ):
                        with self.assertRaisesRegex(MCPResourceConfigError, expected):
                            load_mcp_resource_clients(manifest, runtime_root=runtime)

    def test_manifest_rejects_unknown_top_level_and_whitespace_server_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = root / "server.exe"
            command.write_bytes(b"test")
            runtime = root / "runtime"
            runtime.mkdir()

            extra_manifest = root / "extra.json"
            extra_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "servers": {},
                        "unexpected": True,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    MCPResourceConfigError,
                    "unsupported settings: unexpected",
                ):
                    load_mcp_resource_clients(extra_manifest, runtime_root=runtime)

            whitespace_manifest = root / "whitespace.json"
            whitespace_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "servers": {
                            " srv ": {
                                "transport": "stdio",
                                "command": str(command),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    MCPResourceConfigError,
                    "leading or trailing whitespace",
                ):
                    load_mcp_resource_clients(whitespace_manifest, runtime_root=runtime)

    def test_schema_v1_cannot_declare_tool_execution_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = root / "server.exe"
            command.write_bytes(b"test")
            manifest = root / "mcp_servers.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "servers": {
                            "srv": {
                                "transport": "stdio",
                                "command": str(command),
                                "tools": {},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = root / "runtime"
            runtime.mkdir()
            with patch(
                "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    MCPResourceConfigError,
                    "unsupported settings: tools",
                ):
                    load_mcp_resource_clients(manifest, runtime_root=runtime)

    def test_schema_v2_requires_ask_for_mutating_or_open_world_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = root / "server.exe"
            command.write_bytes(b"test")
            runtime = root / "runtime"
            runtime.mkdir()

            for tool_record in (
                {
                    "approval": "allow",
                    "read_only": False,
                    "open_world": False,
                },
                {
                    "approval": "allow",
                    "read_only": True,
                    "open_world": True,
                },
            ):
                manifest = root / "mcp_servers.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "servers": {
                                "srv": {
                                    "transport": "stdio",
                                    "command": str(command),
                                    "tools": {"tool": tool_record},
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                with patch(
                    "src.tool_system.mcp_resource_runtime.mcp_resource_runtime_available",
                    return_value=True,
                ):
                    with self.assertRaisesRegex(
                        MCPResourceConfigError,
                        "approval='allow' only when read_only=true and open_world=false",
                    ):
                        load_mcp_resource_clients(manifest, runtime_root=runtime)

    def test_tool_contract_must_be_pinned_and_match_current_advertisement(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        digest = mcp_tool_contract_sha256(
            "lookup",
            "Lookup data",
            schema,
            output_schema,
        )
        policy = MCPToolPolicy(
            name="lookup",
            approval="allow",
            read_only=True,
            open_world=False,
            contract_sha256=digest,
        )
        config = MCPStdioServerConfig(
            name="srv",
            command=Path(__file__).resolve(),
            tools={"lookup": policy},
        )
        client = TrustedMCPResourceClient(
            config=config,
            runtime_python=Path(__file__).resolve(),
            bridge_path=Path(__file__).resolve(),
        )

        with patch.object(
            client,
            "_bridge",
            return_value=[
                {
                    "name": "lookup",
                    "title": "Lookup",
                    "description": "Lookup data",
                    "inputSchema": schema,
                    "outputSchema": output_schema,
                    "annotations": {
                        "readOnlyHint": False,
                        "openWorldHint": True,
                    },
                },
                {
                    "name": "unconfigured",
                    "description": "Must not surface",
                    "inputSchema": {"type": "object"},
                },
            ],
        ):
            listed = client.list_tools()

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "lookup")
        self.assertTrue(listed[0]["trustedContract"])
        self.assertEqual(
            listed[0]["localPolicy"],
            {"approval": "allow", "readOnly": True, "openWorld": False},
        )
        self.assertEqual(listed[0]["serverHints"]["readOnlyHint"], False)
        self.assertTrue(client.can_call_tool("lookup"))

        with self.assertRaises(ToolInputError):
            client.call_tool("lookup", {"unexpected": True})

        changed = dict(schema)
        changed["properties"] = {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        }
        with patch.object(
            client,
            "_bridge",
            return_value=[
                {
                    "name": "lookup",
                    "description": "Lookup data",
                    "inputSchema": changed,
                    "outputSchema": output_schema,
                }
            ],
        ):
            relisted = client.list_tools()
        self.assertFalse(relisted[0]["trustedContract"])
        self.assertFalse(client.can_call_tool("lookup"))

        changed_output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        with patch.object(
            client,
            "_bridge",
            return_value=[
                {
                    "name": "lookup",
                    "description": "Lookup data",
                    "inputSchema": schema,
                    "outputSchema": changed_output_schema,
                }
            ],
        ):
            relisted = client.list_tools()
        self.assertFalse(relisted[0]["trustedContract"])
        self.assertFalse(client.can_call_tool("lookup"))

    def test_tool_output_must_match_pinned_output_schema(self) -> None:
        input_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        digest = mcp_tool_contract_sha256(
            "lookup",
            "Lookup data",
            input_schema,
            output_schema,
        )
        client = TrustedMCPResourceClient(
            config=MCPStdioServerConfig(
                name="srv",
                command=Path(__file__).resolve(),
                tools={
                    "lookup": MCPToolPolicy(
                        name="lookup",
                        approval="allow",
                        read_only=True,
                        open_world=False,
                        contract_sha256=digest,
                    )
                },
            ),
            runtime_python=Path(__file__).resolve(),
            bridge_path=Path(__file__).resolve(),
        )
        with patch.object(
            client,
            "_bridge",
            return_value=[
                {
                    "name": "lookup",
                    "description": "Lookup data",
                    "inputSchema": input_schema,
                    "outputSchema": output_schema,
                }
            ],
        ):
            client.list_tools()

        with patch.object(
            client,
            "_bridge",
            return_value={
                "isError": False,
                "structuredContent": {"value": 7},
                "content": [],
                "resultType": "complete",
            },
        ):
            with self.assertRaisesRegex(
                MCPResourceRuntimeError,
                "violated its pinned output schema",
            ):
                client.call_tool("lookup", {"query": "x"})

        with patch.object(
            client,
            "_bridge",
            return_value={
                "isError": True,
                "structuredContent": None,
                "content": [{"type": "text", "text": "server error"}],
                "resultType": "complete",
            },
        ):
            result = client.call_tool("lookup", {"query": "x"})
        self.assertTrue(result["isError"])

    def test_tool_without_contract_pin_can_be_reviewed_but_not_called(self) -> None:
        policy = MCPToolPolicy(
            name="lookup",
            approval="ask",
            read_only=False,
            open_world=True,
            contract_sha256=None,
        )
        client = TrustedMCPResourceClient(
            config=MCPStdioServerConfig(
                name="srv",
                command=Path(__file__).resolve(),
                tools={"lookup": policy},
            ),
            runtime_python=Path(__file__).resolve(),
            bridge_path=Path(__file__).resolve(),
        )
        with patch.object(
            client,
            "_bridge",
            return_value=[
                {
                    "name": "lookup",
                    "description": "Review me",
                    "inputSchema": {"type": "object"},
                }
            ],
        ):
            listed = client.list_tools()
        self.assertEqual(len(listed), 1)
        self.assertFalse(listed[0]["trustedContract"])
        self.assertFalse(client.can_call_tool("lookup"))
        with self.assertRaisesRegex(
            MCPResourceRuntimeError,
            "contract is not pinned",
        ):
            client.call_tool("lookup", {})

    def test_read_requires_prior_advertisement_from_same_client(self) -> None:
        config = MCPStdioServerConfig(
            name="srv",
            command=Path(__file__).resolve(),
        )
        client = TrustedMCPResourceClient(
            config=config,
            runtime_python=Path(__file__).resolve(),
            bridge_path=Path(__file__).resolve(),
        )
        with self.assertRaisesRegex(MCPResourceRuntimeError, "list resources first"):
            client.read_resource("demo://one")

        with patch.object(
            client,
            "_bridge",
            side_effect=[
                [{"uri": "demo://one", "name": "One", "mimeType": "text/plain"}],
                {"contents": [{"uri": "demo://one", "text": "hello"}]},
            ],
        ):
            listed = client.list_resources()
            self.assertEqual(listed[0]["uri"], "demo://one")
            self.assertTrue(client.can_read_resource("demo://one"))
            self.assertFalse(client.can_read_resource("demo://two"))
            read = client.read_resource("demo://one")
            self.assertEqual(read["contents"][0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
