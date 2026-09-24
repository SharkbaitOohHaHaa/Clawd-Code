from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ToolInputError
from .schema_validation import validate_json_schema


MCP_SDK_VERSION = "2.2.0"
MCP_MANIFEST_SCHEMA_VERSION = 2
MCP_RESOURCE_ONLY_SCHEMA_VERSION = 1
_TOOL_APPROVALS = frozenset({"allow", "ask"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MCPResourceConfigError(RuntimeError):
    pass


class MCPResourceRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPToolPolicy:
    name: str
    approval: str
    read_only: bool
    open_world: bool
    contract_sha256: str | None = None


@dataclass(frozen=True)
class MCPStdioServerConfig:
    name: str
    command: Path
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    tools: dict[str, MCPToolPolicy] = field(default_factory=dict)


def mcp_tool_contract_sha256(
    name: str,
    description: str | None,
    input_schema: dict[str, Any],
    output_schema: Any = None,
) -> str:
    payload = {
        "name": name,
        "description": description or "",
        "inputSchema": input_schema,
        "outputSchema": output_schema,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class TrustedMCPResourceClient:
    config: MCPStdioServerConfig
    runtime_python: Path
    bridge_path: Path
    timeout_seconds: float = 10.0
    _advertised_uris: set[str] = field(default_factory=set)
    _advertised_tools: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _bridge(
        self,
        action: str,
        *,
        uri: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        expected_contract_sha256: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "action": action,
            "server": {
                "command": str(self.config.command),
                "args": list(self.config.args),
                "cwd": str(self.config.cwd) if self.config.cwd else None,
            },
            "timeout_seconds": self.timeout_seconds,
        }
        if uri is not None:
            payload["uri"] = uri
        if tool_name is not None:
            payload["tool_name"] = tool_name
        if arguments is not None:
            payload["arguments"] = arguments
        if expected_contract_sha256 is not None:
            payload["expected_contract_sha256"] = expected_contract_sha256

        try:
            completed = subprocess.run(
                [str(self.runtime_python), str(self.bridge_path)],
                input=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds + 3.0,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' request failed"
            ) from exc

        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' returned an invalid bridge response"
            ) from exc
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' request failed"
            )
        return envelope.get("result")

    def list_resources(self) -> list[dict[str, Any]]:
        raw = self._bridge("list_resources")
        if not isinstance(raw, list):
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' returned an invalid resource list"
            )
        resources: list[dict[str, Any]] = []
        advertised: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            if not uri:
                continue
            advertised.add(uri)
            resources.append(
                {
                    "uri": uri,
                    "name": str(item.get("name") or ""),
                    "mimeType": item.get("mimeType"),
                    "description": item.get("description"),
                }
            )
        self._advertised_uris = advertised
        return resources

    def can_read_resource(self, uri: str) -> bool:
        return uri in self._advertised_uris

    def read_resource(self, uri: str) -> dict[str, Any]:
        if uri not in self._advertised_uris:
            raise MCPResourceRuntimeError(
                "resource URI was not advertised by this server in the current Clawd session; list resources first"
            )
        raw = self._bridge("read_resource", uri=uri)
        if not isinstance(raw, dict):
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' returned invalid resource content"
            )
        return raw

    def list_tools(self) -> list[dict[str, Any]]:
        raw = self._bridge("list_tools")
        if not isinstance(raw, list):
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' returned an invalid tool list"
            )

        advertised: dict[str, dict[str, Any]] = {}
        listed: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            policy = self.config.tools.get(name)
            if not name or policy is None:
                continue
            input_schema = item.get("inputSchema")
            if not isinstance(input_schema, dict):
                continue
            description = item.get("description")
            description_text = description if isinstance(description, str) else ""
            output_schema = item.get("outputSchema")
            contract_sha256 = mcp_tool_contract_sha256(
                name,
                description_text,
                input_schema,
                output_schema,
            )
            trusted = bool(
                policy.contract_sha256
                and policy.contract_sha256 == contract_sha256
            )
            clean = {
                "name": name,
                "title": str(item.get("title") or ""),
                "description": description_text,
                "inputSchema": input_schema,
                "outputSchema": output_schema,
                "contractSha256": contract_sha256,
                "trustedContract": trusted,
                "localPolicy": {
                    "approval": policy.approval,
                    "readOnly": policy.read_only,
                    "openWorld": policy.open_world,
                },
                "serverHints": item.get("annotations")
                if isinstance(item.get("annotations"), dict)
                else {},
            }
            advertised[name] = clean
            listed.append(clean)
        self._advertised_tools = advertised
        return listed

    def tool_policy(self, tool_name: str) -> MCPToolPolicy | None:
        return self.config.tools.get(tool_name)

    def can_call_tool(self, tool_name: str) -> bool:
        advertised = self._advertised_tools.get(tool_name)
        policy = self.config.tools.get(tool_name)
        return bool(
            advertised
            and policy
            and advertised.get("trustedContract") is True
            and policy.contract_sha256
        )

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        advertised = self._advertised_tools.get(tool_name)
        policy = self.config.tools.get(tool_name)
        if advertised is None or policy is None:
            raise MCPResourceRuntimeError(
                "MCP tool was not advertised and locally authorized in the current Clawd session; list MCP tools first"
            )
        if not self.can_call_tool(tool_name):
            raise MCPResourceRuntimeError(
                "MCP tool contract is not pinned or no longer matches the operator-approved contract"
            )
        input_schema = advertised.get("inputSchema")
        if not isinstance(input_schema, dict):
            raise MCPResourceRuntimeError("MCP tool input schema is unavailable")
        validate_json_schema(
            args,
            input_schema,
            root_name=f"MCP.{self.config.name}.{tool_name}",
        )
        raw = self._bridge(
            "call_tool",
            tool_name=tool_name,
            arguments=args,
            expected_contract_sha256=policy.contract_sha256,
        )
        if not isinstance(raw, dict):
            raise MCPResourceRuntimeError(
                f"MCP server '{self.config.name}' returned invalid tool output"
            )

        output_schema = advertised.get("outputSchema")
        if not bool(raw.get("isError")) and output_schema is not None:
            if not isinstance(output_schema, dict):
                raise MCPResourceRuntimeError(
                    f"MCP tool '{self.config.name}/{tool_name}' has an unsupported output schema"
                )
            structured = raw.get("structuredContent")
            if structured is None:
                raise MCPResourceRuntimeError(
                    f"MCP tool '{self.config.name}/{tool_name}' omitted structured content required by its pinned output schema"
                )
            try:
                validate_json_schema(
                    structured,
                    output_schema,
                    root_name=f"MCP.{self.config.name}.{tool_name}.output",
                )
            except ToolInputError as exc:
                raise MCPResourceRuntimeError(
                    f"MCP tool '{self.config.name}/{tool_name}' violated its pinned output schema"
                ) from exc
        return raw


def default_mcp_runtime_root() -> Path:
    return Path.home() / ".clawd" / "mcp" / "runtime"


def default_mcp_manifest_path() -> Path:
    configured = os.environ.get("CLAWD_MCP_SERVERS_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".clawd" / "mcp_servers.json").resolve()


def _runtime_python(runtime_root: Path) -> Path:
    if os.name == "nt":
        return runtime_root / "Scripts" / "python.exe"
    return runtime_root / "bin" / "python"


def mcp_resource_runtime_available(runtime_root: str | Path | None = None) -> bool:
    root = Path(runtime_root or default_mcp_runtime_root()).expanduser().resolve()
    python = _runtime_python(root)
    requirements = root / "requirements.txt"
    if not python.is_file() or not requirements.is_file():
        return False
    if requirements.read_text(encoding="utf-8", errors="replace").strip() != f"mcp=={MCP_SDK_VERSION}":
        return False
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata as m; print(m.version('mcp'))",
            ],
            text=True,
            capture_output=True,
            timeout=5.0,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == MCP_SDK_VERSION


def _validate_tool_policy(tool_name: str, record: Any) -> MCPToolPolicy:
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise MCPResourceConfigError("MCP tool policy names must be non-empty strings")
    if tool_name != tool_name.strip():
        raise MCPResourceConfigError(
            "MCP tool policy names must not contain leading or trailing whitespace"
        )
    if not isinstance(record, dict):
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' must be an object"
        )
    allowed_keys = {
        "approval",
        "read_only",
        "open_world",
        "contract_sha256",
        "enabled",
    }
    extra = sorted(set(record) - allowed_keys)
    if extra:
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' contains unsupported settings: "
            + ", ".join(extra)
        )
    if record.get("enabled", True) is False:
        raise MCPResourceConfigError("disabled")

    approval = record.get("approval")
    if approval not in _TOOL_APPROVALS:
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' approval must be one of: "
            + ", ".join(sorted(_TOOL_APPROVALS))
        )
    read_only = record.get("read_only")
    open_world = record.get("open_world")
    if not isinstance(read_only, bool):
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' read_only must be boolean"
        )
    if not isinstance(open_world, bool):
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' open_world must be boolean"
        )
    if approval == "allow" and (not read_only or open_world):
        raise MCPResourceConfigError(
            f"MCP tool policy '{tool_name}' may use approval='allow' only when "
            "read_only=true and open_world=false"
        )

    contract_sha256 = record.get("contract_sha256")
    if contract_sha256 is not None:
        if (
            not isinstance(contract_sha256, str)
            or not _SHA256_RE.fullmatch(contract_sha256)
        ):
            raise MCPResourceConfigError(
                f"MCP tool policy '{tool_name}' contract_sha256 must be a lowercase SHA-256 hex digest"
            )

    return MCPToolPolicy(
        name=tool_name,
        approval=approval,
        read_only=read_only,
        open_world=open_world,
        contract_sha256=contract_sha256,
    )


def _validate_server(
    name: str,
    record: Any,
    *,
    schema_version: int,
) -> MCPStdioServerConfig:
    if not isinstance(name, str) or not name.strip():
        raise MCPResourceConfigError("MCP server names must be non-empty strings")
    if name != name.strip():
        raise MCPResourceConfigError("MCP server names must not contain leading or trailing whitespace")
    if not isinstance(record, dict):
        raise MCPResourceConfigError(f"MCP server '{name}' must be an object")
    allowed_keys = {"transport", "command", "args", "cwd", "enabled"}
    if schema_version == MCP_MANIFEST_SCHEMA_VERSION:
        allowed_keys.add("tools")
    extra = sorted(set(record) - allowed_keys)
    if extra:
        raise MCPResourceConfigError(
            f"MCP server '{name}' contains unsupported settings: {', '.join(extra)}"
        )
    if record.get("enabled", True) is False:
        raise MCPResourceConfigError("disabled")
    if record.get("transport") != "stdio":
        raise MCPResourceConfigError(
            f"MCP server '{name}' must use transport='stdio' in Phase 1"
        )

    command_raw = record.get("command")
    if not isinstance(command_raw, str) or not command_raw.strip():
        raise MCPResourceConfigError(f"MCP server '{name}' command must be a non-empty string")
    command = Path(command_raw).expanduser()
    if not command.is_absolute():
        raise MCPResourceConfigError(f"MCP server '{name}' command must be an absolute path")
    command = command.resolve()
    if not command.is_file():
        raise MCPResourceConfigError(f"MCP server '{name}' command does not exist")

    args_raw = record.get("args", [])
    if not isinstance(args_raw, list) or not all(isinstance(arg, str) for arg in args_raw):
        raise MCPResourceConfigError(f"MCP server '{name}' args must be an array of strings")

    cwd: Path | None = None
    cwd_raw = record.get("cwd")
    if cwd_raw is not None:
        if not isinstance(cwd_raw, str) or not cwd_raw.strip():
            raise MCPResourceConfigError(f"MCP server '{name}' cwd must be a non-empty string")
        cwd = Path(cwd_raw).expanduser()
        if not cwd.is_absolute():
            raise MCPResourceConfigError(f"MCP server '{name}' cwd must be an absolute path")
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise MCPResourceConfigError(f"MCP server '{name}' cwd does not exist")

    tools: dict[str, MCPToolPolicy] = {}
    if schema_version == MCP_MANIFEST_SCHEMA_VERSION:
        tools_raw = record.get("tools", {})
        if not isinstance(tools_raw, dict):
            raise MCPResourceConfigError(
                f"MCP server '{name}' tools must be an object"
            )
        for tool_name, tool_record in tools_raw.items():
            if isinstance(tool_record, dict) and tool_record.get("enabled", True) is False:
                continue
            policy = _validate_tool_policy(tool_name, tool_record)
            tools[policy.name] = policy

    return MCPStdioServerConfig(
        name=name.strip(),
        command=command,
        args=tuple(args_raw),
        cwd=cwd,
        tools=tools,
    )


def load_mcp_resource_clients(
    manifest_path: str | Path | None = None,
    *,
    runtime_root: str | Path | None = None,
) -> dict[str, TrustedMCPResourceClient]:
    path = Path(manifest_path or default_mcp_manifest_path()).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPResourceConfigError(f"cannot read MCP server manifest: {path}") from exc
    if not isinstance(data, dict):
        raise MCPResourceConfigError("MCP server manifest must be an object")
    schema_version = data.get("schema_version")
    if schema_version not in {
        MCP_RESOURCE_ONLY_SCHEMA_VERSION,
        MCP_MANIFEST_SCHEMA_VERSION,
    }:
        raise MCPResourceConfigError("unsupported MCP server manifest schema version")
    extra_top_level = sorted(set(data) - {"schema_version", "servers"})
    if extra_top_level:
        raise MCPResourceConfigError(
            "MCP server manifest contains unsupported settings: "
            + ", ".join(extra_top_level)
        )
    servers = data.get("servers")
    if not isinstance(servers, dict):
        raise MCPResourceConfigError("MCP server manifest must contain a servers object")

    root = Path(runtime_root or default_mcp_runtime_root()).expanduser().resolve()
    if not mcp_resource_runtime_available(root):
        raise MCPResourceConfigError(
            f"pinned MCP runtime {MCP_SDK_VERSION} is unavailable"
        )
    runtime_python = _runtime_python(root)
    bridge_path = Path(__file__).with_name("mcp_sdk_bridge.py").resolve()

    clients: dict[str, TrustedMCPResourceClient] = {}
    for name, record in servers.items():
        if isinstance(record, dict) and record.get("enabled", True) is False:
            continue
        config = _validate_server(
            name,
            record,
            schema_version=schema_version,
        )
        clients[config.name] = TrustedMCPResourceClient(
            config=config,
            runtime_python=runtime_python,
            bridge_path=bridge_path,
        )
    return clients


def mcp_manifest_status(
    manifest_path: str | Path | None = None,
    *,
    runtime_root: str | Path | None = None,
) -> tuple[bool, str]:
    path = Path(manifest_path or default_mcp_manifest_path()).expanduser().resolve()
    if not path.exists():
        return True, "no MCP servers configured"
    try:
        clients = load_mcp_resource_clients(path, runtime_root=runtime_root)
    except MCPResourceConfigError as exc:
        return False, str(exc)
    return True, f"{len(clients)} MCP stdio server(s) configured"
