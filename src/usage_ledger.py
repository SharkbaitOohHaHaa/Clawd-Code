from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_FIELDS = (
    "input_tokens",
    "output_tokens",
    "thought_tokens",
    "tool_use_tokens",
    "cached_tokens",
    "total_tokens",
)


def _ledger_path() -> Path:
    configured = os.environ.get("CLAWD_USAGE_LEDGER", "").strip()
    if configured:
        return Path(configured).expanduser()
    memory_dir = os.environ.get("CLAWD_MEMORY_DIR", "").strip()
    base = Path(memory_dir).expanduser().parent if memory_dir else Path.home() / ".clawd"
    return base / "provider-usage.jsonl"


def append_provider_usage(record: dict[str, Any]) -> None:
    """Persist exact provider-reported usage from a completed API response."""
    label = str(record.get("label") or "").strip()
    if not label:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
    }
    for field in _FIELDS:
        try:
            event[field] = max(0, int(record.get(field, 0) or 0))
        except (TypeError, ValueError, OverflowError):
            event[field] = 0

    path = _ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError:
        # Observability must never break an otherwise successful provider response.
        return


def month_to_date_provider_usage(prefix: str) -> dict[str, dict[str, int]]:
    """Aggregate this month's persisted provider-reported usage by label."""
    path = _ledger_path()
    if not path.is_file():
        return {}
    now = datetime.now(timezone.utc)
    month_prefix = f"{now.year:04d}-{now.month:02d}-"
    totals: dict[str, dict[str, int]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return totals
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if not str(event.get("timestamp") or "").startswith(month_prefix):
            continue
        label = str(event.get("label") or "")
        if not label.startswith(prefix):
            continue
        bucket = totals.setdefault(label, {field: 0 for field in _FIELDS})
        for field in _FIELDS:
            try:
                bucket[field] += max(0, int(event.get(field, 0) or 0))
            except (TypeError, ValueError):
                continue
    return totals
