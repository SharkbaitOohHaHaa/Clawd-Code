from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

from ..usage_ledger import append_provider_usage
from .errors import ToolPermissionError
from .permissions import ToolPermissionContext
from .task_manager import TaskManager

_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
# POSIX: opening a FIFO or terminal swapped in at a recorded path neither blocks nor
# acquires a controlling terminal. Both flags are absent (0) on Windows.
_NON_BLOCKING_OPEN_FLAGS = getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0)


class ReadFingerprint(NamedTuple):
    """The exact bytes a tool read or wrote: their length and full SHA-256 digest."""

    size: int
    sha256: bytes


def _fingerprint_of_bytes(data: bytes) -> ReadFingerprint:
    if not isinstance(data, bytes):
        raise TypeError(
            "mark_file_read requires the exact bytes the tool read or wrote, "
            f"not {type(data).__name__}"
        )
    return ReadFingerprint(len(data), hashlib.sha256(data).digest())


def _open_without_blocking(path: str | os.PathLike[str], flags: int) -> int:
    # ``flags`` come from open(); in "rb" mode they include O_BINARY on Windows.
    return os.open(path, flags | _NON_BLOCKING_OPEN_FLAGS)


def _file_has_fingerprint(path: Path, recorded: ReadFingerprint) -> bool:
    """True only if ``path`` is a regular file whose bytes are exactly ``recorded``.

    Metadata can prove a change (not a regular file, different size) but never
    equality, which needs every byte. At most ``recorded.size + 1`` bytes are read,
    so a file that grows after the size check is still seen as changed and a grown
    file is never hashed in full.
    """
    # Binary mode is structural: no decoding or newline translation.
    with open(path, "rb", buffering=0, opener=_open_without_blocking) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != recorded.size:
            return False
        digest = hashlib.sha256()
        consumed = 0
        remaining = recorded.size + 1
        while remaining > 0:
            chunk = handle.read(min(_FINGERPRINT_CHUNK_BYTES, remaining))
            if chunk is None:  # a non-blocking read would block: not a proven EOF
                return False
            if not chunk:
                break
            digest.update(chunk)
            consumed += len(chunk)
            remaining -= len(chunk)
    return consumed == recorded.size and digest.digest() == recorded.sha256


@dataclass
class ToolContext:
    workspace_root: Path
    permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    cwd: Path | None = None
    read_file_fingerprints: dict[Path, ReadFingerprint] = field(default_factory=dict)
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

    def mark_file_read(self, path: Path, *, data: bytes) -> None:
        """Record that ``data`` is exactly what this tool operation read from, or wrote
        to, ``path``.

        ``data`` must be the very buffer the tool read (and built its output from) or
        handed to the OS as its write. Never re-read the file to obtain it: a change
        landing in between would then be recorded as seen. This never reads the file;
        ``resolve()`` only canonicalises the key.
        """
        fingerprint = _fingerprint_of_bytes(data)
        self.read_file_fingerprints[path.resolve()] = fingerprint

    def was_file_read_and_unchanged(self, path: Path) -> bool:
        """True only if ``path`` was recorded and its current bytes are exactly the
        recorded bytes.

        Fails closed: a missing file, a directory or other non-regular object, an
        open/read/permission error (including a lock that prevents reading) or an
        invalid path returns False. This is not a lock: the file can still change while
        or after it is checked.
        """
        try:
            resolved = path.resolve()
            recorded = self.read_file_fingerprints.get(resolved)
            if recorded is None:
                return False
            return _file_has_fingerprint(resolved, recorded)
        except (OSError, ValueError, RuntimeError):
            return False

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
