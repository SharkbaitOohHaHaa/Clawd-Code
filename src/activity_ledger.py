from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_ledger_dir() -> Path:
    memory_dir = os.environ.get("CLAWD_MEMORY_DIR", "").strip()
    return Path(memory_dir).expanduser().parent if memory_dir else Path.home() / ".clawd"


def _ledger_path() -> Path:
    configured = os.environ.get("CLAWD_ACTIVITY_LEDGER", "").strip()
    return Path(configured).expanduser() if configured else _default_ledger_dir() / "activity.jsonl"


def _change_ledger_path() -> Path:
    configured = os.environ.get("CLAWD_CHANGE_LEDGER", "").strip()
    return Path(configured).expanduser() if configured else _default_ledger_dir() / "changes.jsonl"


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    """Best-effort instrumentation must never break tool execution."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
    except OSError:
        return


def append_activity(kind: str, name: str, *, status: str = "ok") -> None:
    """Persist one sanitized canonical skill/tool invocation."""
    kind = str(kind or "").strip().lower()
    name = str(name or "").strip()
    status = str(status or "").strip().lower()
    if kind not in {"skill", "tool"} or not name or status not in {"ok", "error"}:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "name": name,
        "status": status,
    }
    _append_jsonl(_ledger_path(), event)


def append_change(tool: str, path: str, operation: str) -> None:
    """Persist sanitized workspace-relative changed-file evidence only."""
    tool = str(tool or "").strip()
    path = str(path or "").strip().replace("\\", "/")
    operation = str(operation or "").strip().lower()
    relative = Path(path)
    if not tool or not path or not operation or relative.is_absolute() or ".." in relative.parts:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "path": relative.as_posix(),
        "operation": operation,
    }
    _append_jsonl(_change_ledger_path(), event)


def activity_counts() -> dict[str, Counter[str]]:
    """Return persisted invocation counts. Malformed events are ignored."""
    totals = {"skill": Counter(), "tool": Counter()}
    path = _ledger_path()
    if not path.is_file():
        return totals
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return totals
    for line in lines:
        try:
            event: Any = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "").strip().lower()
        name = str(event.get("name") or "").strip()
        if kind in totals and name:
            totals[kind][name] += 1
    return totals
