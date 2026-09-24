from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _ledger_path() -> Path:
    configured = os.environ.get("CLAWD_PROVIDER_EVENT_LEDGER", "").strip()
    if configured:
        return Path(configured).expanduser()
    memory_dir = os.environ.get("CLAWD_MEMORY_DIR", "").strip()
    base = Path(memory_dir).expanduser().parent if memory_dir else Path.home() / ".clawd"
    return base / "provider-events.jsonl"


def _sanitize_endpoint(value: str) -> str:
    """Keep only non-secret endpoint routing metadata."""
    try:
        parsed = urlsplit(str(value or ""))
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            return ""
        return urlunsplit(
            (parsed.scheme.lower(), f"{host}{port}", parsed.path or "", "", "")
        )
    except (TypeError, ValueError):
        return ""


def append_provider_event(
    *,
    provider: str,
    operation: str,
    stage: str,
    status: str,
    model: str,
    endpoint: str,
    error_type: str = "",
    status_code: int | None = None,
    logical_call_id: str = "",
    attempt: int | None = None,
    elapsed_ms: int | None = None,
    retryable: bool | None = None,
) -> None:
    """Persist non-secret provider execution metadata for debugging."""
    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": str(provider),
        "operation": str(operation),
        "stage": str(stage),
        "status": str(status),
        "model": str(model),
        "endpoint": _sanitize_endpoint(endpoint),
    }
    if error_type:
        event["error_type"] = str(error_type)
    try:
        if status_code is not None:
            event["status_code"] = int(status_code)
        if attempt is not None:
            event["attempt"] = int(attempt)
        if elapsed_ms is not None:
            event["elapsed_ms"] = max(0, int(elapsed_ms))
    except (TypeError, ValueError, OverflowError):
        # Invalid optional metrics are omitted rather than affecting provider work.
        pass
    if logical_call_id:
        event["logical_call_id"] = str(logical_call_id)
    if retryable is not None:
        event["retryable"] = bool(retryable)

    path = _ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n"
            )
    except OSError:
        # Observability must never break an otherwise valid provider operation.
        return
