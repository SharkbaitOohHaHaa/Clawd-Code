from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..usage_ledger import append_provider_usage
from .errors import ToolPermissionError
from .permissions import ToolPermissionContext
from .task_manager import TaskManager


@dataclass
class ToolContext:
    workspace_root: Path
    permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    cwd: Path | None = None
    read_file_fingerprints: dict[Path, tuple[int, int]] = field(default_factory=dict)
    task_manager: TaskManager = field(default_factory=TaskManager)
    mcp_clients: dict[str, Any] = field(default_factory=dict)
    mcp_config_error: str | None = None
    lsp_client: Any | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan_mode: bool = False
    worktree_root: Path | None = None
    outbox: list[dict[str, Any]] = field(default_factory=list)
    ask_user: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None
    crons: dict[str, dict[str, Any]] = field(default_factory=dict)
    team: dict[str, Any] | None = None
    output_style_name: str | None = None
    output_style_dir: Path | None = None
    usage_records: list[dict[str, Any]] = field(default_factory=list)
    instrumentation_enabled: bool = False
    tool_allowlist: frozenset[str] | None = None

    # Permission handler callback: called when a tool needs user consent.
    # Signature: (tool_name: str, message: str, suggestion: str | None)
    #           -> tuple[bool, bool] (allowed: bool, continue_without_caching: bool)
    # If not set, permission errors will be raised as exceptions.
    permission_handler: Callable[[str, str, Optional[str]], tuple[bool, bool]] | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()
        if self.permission_context.workspace_root is None:
            self.permission_context = ToolPermissionContext.from_iterables(
                self.permission_context.deny_names,
                self.permission_context.deny_prefixes,
                workspace_root=self.workspace_root,
                additional_working_directories=self.permission_context.additional_working_directories,
                allow_docs=self.permission_context.allow_docs,
                allow_docs_locked_off=self.permission_context.allow_docs_locked_off,
            )

    def mark_file_read(self, path: Path) -> None:
        stat = path.stat()
        self.read_file_fingerprints[path.resolve()] = (
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    def was_file_read_and_unchanged(self, path: Path) -> bool:
        resolved = path.resolve()
        fingerprint = self.read_file_fingerprints.get(resolved)
        if fingerprint is None:
            return False
        stat = resolved.stat()
        return fingerprint == (int(stat.st_mtime_ns), int(stat.st_size))

    def ensure_allowed_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        return self.permission_context.ensure_path_allowed(p)

    def is_tool_allowed(self, tool_name: str, aliases: tuple[str, ...] = ()) -> bool:
        names = {tool_name.lower(), *(alias.lower() for alias in aliases)}
        if any(self.permission_context.blocks_tool(name) for name in names):
            return False
        if self.tool_allowlist is None:
            return True
        return any(name in self.tool_allowlist for name in names)

    def ensure_tool_allowed(self, tool_name: str, aliases: tuple[str, ...] = ()) -> None:
        names = {tool_name.lower(), *(alias.lower() for alias in aliases)}
        if any(self.permission_context.blocks_tool(name) for name in names):
            raise ToolPermissionError(f"tool is blocked by permission context: {tool_name}")
        if self.tool_allowlist is not None and not any(
            name in self.tool_allowlist for name in names
        ):
            raise ToolPermissionError(
                f"tool is not allowed by the active skill scope: {tool_name}"
            )

    def restrict_tool_allowlist(self, tool_names: list[str] | tuple[str, ...]) -> None:
        requested = frozenset(
            name.strip().lower()
            for name in tool_names
            if isinstance(name, str) and name.strip()
        )
        if not requested:
            return
        if self.tool_allowlist is None:
            self.tool_allowlist = requested
        else:
            self.tool_allowlist = self.tool_allowlist.intersection(requested)

    def record_usage(self, usage: dict[str, Any]) -> None:
        record = dict(usage)
        self.usage_records.append(record)
        append_provider_usage(record)

    def consume_usage_records(self) -> list[dict[str, Any]]:
        records = list(self.usage_records)
        self.usage_records.clear()
        return records
