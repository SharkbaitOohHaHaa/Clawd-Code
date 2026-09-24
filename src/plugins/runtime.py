from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PLUGIN_SCHEMA_VERSION = 1
OPERATOR_SCHEMA_VERSION = 1
_ALLOWED_EXTENSIONS = {"commands", "tools", "providers", "workflows"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HASH_EXCLUDED_DIRS = {".git", "__pycache__"}
_HASH_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


class PluginRuntimeError(RuntimeError):
    """Raised when local Python plugin metadata is unsafe or invalid."""


def default_plugin_root() -> Path:
    return Path.home() / ".clawd" / "plugins"


def default_operator_manifest_path() -> Path:
    return Path.home() / ".clawd" / "python_plugins.json"


def compute_plugin_artifact_hash(root: str | Path) -> str:
    root = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(root)
        if any(part in _HASH_EXCLUDED_DIRS for part in rel.parts):
            continue
        if candidate.suffix.lower() in _HASH_EXCLUDED_SUFFIXES:
            continue
        files.append(candidate)

    for candidate in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        rel = candidate.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _path_has_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _load_plugin_manifest(plugin_dir: Path) -> dict[str, Any]:
    if plugin_dir.is_symlink():
        raise PluginRuntimeError("plugin directory must not be a symlink")
    manifest_path = plugin_dir / "plugin.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PluginRuntimeError("plugin.json must be a regular non-symlink file")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PluginRuntimeError("plugin.json root must be an object")
    if data.get("schema_version") != PLUGIN_SCHEMA_VERSION:
        raise PluginRuntimeError(f"plugin schema_version must be {PLUGIN_SCHEMA_VERSION}")

    name = str(data.get("name") or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise PluginRuntimeError("plugin name is invalid")
    if name != plugin_dir.name:
        raise PluginRuntimeError("plugin name must match its directory name")

    version = str(data.get("version") or "").strip()
    if not version:
        raise PluginRuntimeError("plugin version is required")
    entry_text = str(data.get("entrypoint") or "").strip()
    entry_rel = Path(entry_text)
    if not entry_text or entry_rel.is_absolute() or ".." in entry_rel.parts:
        raise PluginRuntimeError("plugin entrypoint must be a relative in-plugin path")
    if entry_rel.suffix.lower() != ".py":
        raise PluginRuntimeError("plugin entrypoint must be a Python file")
    if _path_has_symlink(plugin_dir, entry_rel):
        raise PluginRuntimeError("plugin entrypoint path must not traverse symlinks")

    root = plugin_dir.resolve()
    entrypoint = (plugin_dir / entry_rel).resolve()
    try:
        entrypoint.relative_to(root)
    except ValueError as exc:
        raise PluginRuntimeError("plugin entrypoint escapes plugin directory") from exc
    if not entrypoint.is_file():
        raise PluginRuntimeError("plugin entrypoint does not exist")

    raw_extensions = data.get("extensions") or []
    if not isinstance(raw_extensions, list):
        raise PluginRuntimeError("plugin extensions must be a list")
    extensions = sorted({str(value).strip() for value in raw_extensions if str(value).strip()})
    unknown = sorted(set(extensions) - _ALLOWED_EXTENSIONS)
    if unknown:
        raise PluginRuntimeError(f"unsupported plugin extensions: {', '.join(unknown)}")

    return {
        "name": name,
        "version": version,
        "description": str(data.get("description") or "").strip(),
        "entrypoint": entry_rel.as_posix(),
        "extensions": extensions,
        "artifact_sha256": compute_plugin_artifact_hash(plugin_dir),
    }


def _load_operator_policy(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if not path.exists():
        return {}, []
    if path.is_symlink() or not path.is_file():
        return {}, [{"code": "plugin_operator_manifest_invalid", "subject": str(path)}]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, [{"code": "plugin_operator_manifest_invalid", "subject": str(path)}]
    if not isinstance(data, dict) or data.get("schema_version") != OPERATOR_SCHEMA_VERSION:
        return {}, [{"code": "plugin_operator_manifest_invalid", "subject": str(path)}]
    raw_plugins = data.get("plugins")
    if raw_plugins is None:
        raw_plugins = {}
    if not isinstance(raw_plugins, dict):
        return {}, [{"code": "plugin_operator_manifest_invalid", "subject": str(path)}]

    policy: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    for name, raw in raw_plugins.items():
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or not isinstance(raw, dict):
            issues.append({"code": "plugin_operator_entry_invalid", "subject": str(name)})
            continue
        enabled = raw.get("enabled", False)
        artifact = str(raw.get("artifact_sha256") or "").lower().strip()
        if not isinstance(enabled, bool) or (artifact and not re.fullmatch(r"[0-9a-f]{64}", artifact)):
            issues.append({"code": "plugin_operator_entry_invalid", "subject": name})
            continue
        policy[name] = {"enabled": enabled, "artifact_sha256": artifact}
    return policy, issues


def reconcile_python_plugins(
    *,
    plugin_root: str | Path | None = None,
    operator_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Discover plugin metadata without importing or executing plugin Python."""
    root = Path(plugin_root or default_plugin_root()).expanduser()
    policy_path = Path(operator_manifest or default_operator_manifest_path()).expanduser()
    policy, issues = _load_operator_policy(policy_path)
    records: dict[str, dict[str, Any]] = {}

    if root.exists() and root.is_symlink():
        issues.append({"code": "plugin_root_symlink", "subject": str(root)})
    elif root.exists() and not root.is_dir():
        issues.append({"code": "plugin_root_invalid", "subject": str(root)})
    elif root.is_dir():
        for plugin_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not plugin_dir.is_dir() and not plugin_dir.is_symlink():
                continue
            try:
                record = _load_plugin_manifest(plugin_dir)
            except (OSError, json.JSONDecodeError, PluginRuntimeError) as exc:
                issues.append({"code": "plugin_manifest_invalid", "subject": f"{plugin_dir.name}: {exc}"})
                continue

            name = record["name"]
            operator = policy.get(name) or {}
            enabled = operator.get("enabled") is True
            pinned_hash = str(operator.get("artifact_sha256") or "")
            active = False
            state = "inactive"
            if enabled and not pinned_hash:
                issues.append({"code": "plugin_operator_hash_missing", "subject": name})
                state = "blocked"
            elif enabled and pinned_hash != record["artifact_sha256"]:
                issues.append({"code": "plugin_integrity_mismatch", "subject": name})
                state = "review_required"
            elif enabled:
                active = True
                state = "active"
            record.update(
                {
                    "root": str(plugin_dir.resolve()),
                    "active": active,
                    "state": state,
                    "operator_hash": pinned_hash,
                }
            )
            records[name] = record

    for name, operator in sorted(policy.items()):
        if operator.get("enabled") is True and name not in records:
            issues.append({"code": "enabled_plugin_missing", "subject": name})

    return {
        "plugin_root": str(root.resolve()),
        "operator_manifest": str(policy_path.resolve()),
        "records": records,
        "active": sorted(name for name, record in records.items() if record["active"]),
        "issues": sorted(issues, key=lambda item: (item["code"], item["subject"])),
    }
