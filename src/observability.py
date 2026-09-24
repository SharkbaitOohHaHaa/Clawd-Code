from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .activity_ledger import _change_ledger_path, _ledger_path as _activity_ledger_path
from .provider_event_ledger import _ledger_path as _provider_event_ledger_path
from .usage_ledger import _ledger_path as _usage_ledger_path


_FAILURE_STATUSES = {"error", "failed", "timeout"}


def _read_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"events": [], "malformed": 0, "read_error": None}
    if not path.is_file():
        return {"events": [], "malformed": 0, "read_error": "not_a_file"}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"events": [], "malformed": 0, "read_error": type(exc).__name__}

    events: list[dict[str, Any]] = []
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        events.append(event)
    return {"events": events, "malformed": malformed, "read_error": None}


def runtime_observability_snapshot(*, recent_limit: int = 5) -> dict[str, Any]:
    """Return a sanitized local-only snapshot of runtime instrumentation."""
    limit = max(0, min(int(recent_limit), 20))
    ledgers = {
        "activity": _read_jsonl(_activity_ledger_path()),
        "changes": _read_jsonl(_change_ledger_path()),
        "provider_events": _read_jsonl(_provider_event_ledger_path()),
        "provider_usage": _read_jsonl(_usage_ledger_path()),
    }

    tool_status = Counter()
    skill_status = Counter()
    recent_errors: list[dict[str, Any]] = []
    for event in ledgers["activity"]["events"]:
        kind = str(event.get("kind") or "").strip().lower()
        name = str(event.get("name") or "").strip()
        status = str(event.get("status") or "").strip().lower()
        if kind not in {"tool", "skill"} or not name or status not in {"ok", "error"}:
            continue
        target = tool_status if kind == "tool" else skill_status
        target[status] += 1
        if status == "error":
            recent_errors.append({
                "timestamp": str(event.get("timestamp") or ""),
                "area": kind,
                "name": name,
            })

    provider_failures = 0
    for event in ledgers["provider_events"]["events"]:
        status = str(event.get("status") or "").strip().lower()
        if status not in _FAILURE_STATUSES:
            continue
        provider_failures += 1
        recent_errors.append({
            "timestamp": str(event.get("timestamp") or ""),
            "area": "provider",
            "provider": str(event.get("provider") or ""),
            "operation": str(event.get("operation") or ""),
            "stage": str(event.get("stage") or ""),
            "error_type": str(event.get("error_type") or ""),
            "status_code": event.get("status_code"),
        })

    recent_errors.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    if limit:
        recent_errors = recent_errors[:limit]
    else:
        recent_errors = []

    issues: list[dict[str, str]] = []
    for label, ledger in ledgers.items():
        read_error = ledger.get("read_error")
        if read_error:
            issues.append({
                "code": "observability_ledger_unreadable",
                "subject": f"{label}: {read_error}",
            })
        malformed = int(ledger.get("malformed") or 0)
        if malformed:
            issues.append({
                "code": "observability_ledger_malformed",
                "subject": f"{label}: {malformed} malformed event(s)",
            })

    return {
        "tool_calls": int(tool_status["ok"] + tool_status["error"]),
        "tool_errors": int(tool_status["error"]),
        "skill_calls": int(skill_status["ok"] + skill_status["error"]),
        "skill_errors": int(skill_status["error"]),
        "change_events": len(ledgers["changes"]["events"]),
        "provider_events": len(ledgers["provider_events"]["events"]),
        "provider_failures": provider_failures,
        "usage_events": len(ledgers["provider_usage"]["events"]),
        "recent_errors": recent_errors,
        "issues": sorted(issues, key=lambda item: (item["code"], item["subject"])),
    }
