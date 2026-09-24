"""Deterministic capability/trust reconciliation.

This module is read-only: it compares declared capability intent with current
runtime, filesystem, and skill-trust evidence. It never installs, activates,
approves, edits, or repairs capabilities.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

from .plugins.runtime import reconcile_python_plugins
from .skills.trust_registry import REGISTRY_SCHEMA_VERSION, SkillApprovalError, compute_artifact_hash
from .tool_system.defaults import build_default_registry
from .tool_system.mcp_resource_runtime import (
    MCP_SDK_VERSION,
    mcp_manifest_status,
    mcp_resource_runtime_available,
)
from .tool_system.pyright_lsp import pyright_runtime_available
from .tool_system.tools.worktree import git_worktree_runtime_available


MANIFEST_SCHEMA_VERSION = 1
VALID_STATES = {
    "ACTIVE_SUPPORTED",
    "INTENTIONALLY_DISABLED",
    "DEFERRED_NOT_PRODUCTION_READY",
    "CLAWD_SPECIFIC",
}


def load_capability_manifest(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        text = files("src").joinpath("capability_manifest.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported capability manifest schema version")
    states = set(data.get("states") or [])
    if states != VALID_STATES:
        raise ValueError("capability manifest states do not match the supported taxonomy")
    tools = data.get("tools")
    features = data.get("features")
    if not isinstance(tools, dict) or not isinstance(features, dict):
        raise ValueError("capability manifest must contain tools and features objects")
    for section in (tools, features):
        for name, record in section.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ValueError("capability manifest entries must be named objects")
            if record.get("state") not in VALID_STATES:
                raise ValueError(f"invalid capability state for {name}")
    return data


def expected_registered_tool_names(manifest: dict[str, Any] | None = None) -> list[str]:
    manifest = manifest or load_capability_manifest()
    return sorted(
        name
        for name, record in manifest["tools"].items()
        if bool(record.get("registered_expected"))
    )


def render_capability_status_markdown(manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest or load_capability_manifest()
    labels = {
        "ACTIVE_SUPPORTED": "Active / supported",
        "INTENTIONALLY_DISABLED": "Intentionally disabled",
        "DEFERRED_NOT_PRODUCTION_READY": "Deferred / not production-ready",
        "CLAWD_SPECIFIC": "Clawd-specific",
    }
    lines = ["| Capability state | Tools | Features |", "|---|---|---|"]
    for state in (
        "ACTIVE_SUPPORTED",
        "CLAWD_SPECIFIC",
        "INTENTIONALLY_DISABLED",
        "DEFERRED_NOT_PRODUCTION_READY",
    ):
        tools = sorted(name for name, rec in manifest["tools"].items() if rec["state"] == state)
        features = sorted(
            name for name, rec in manifest["features"].items() if rec["state"] == state
        )
        lines.append(
            f"| {labels[state]} | {', '.join(tools)} | {', '.join(features)} |"
        )
    return "\n".join(lines)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _verify_audit_chain_read_only(audit_path: Path) -> dict[str, Any]:
    if not audit_path.exists():
        return {"valid": False, "entries": 0, "error": "skill audit log is missing"}
    previous_hash = ""
    count = 0
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"valid": False, "entries": 0, "error": f"cannot read audit log: {exc}"}
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return {"valid": False, "entries": count, "error": f"invalid JSON at line {line_number}"}
        if not isinstance(entry, dict):
            return {"valid": False, "entries": count, "error": f"non-object entry at line {line_number}"}
        current_hash = str(entry.get("current_entry_hash") or "")
        if str(entry.get("previous_entry_hash") or "") != previous_hash:
            return {"valid": False, "entries": count, "error": f"chain break at line {line_number}"}
        payload = dict(entry)
        payload.pop("current_entry_hash", None)
        expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if not current_hash or current_hash != expected:
            return {"valid": False, "entries": count, "error": f"hash mismatch at line {line_number}"}
        previous_hash = current_hash
        count += 1
    return {"valid": True, "entries": count, "last_hash": previous_hash}
def _candidate_loader_dirs(project_root: str | Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    for raw in (os.environ.get("CLAWD_SKILLS_DIR"), os.environ.get("CLAUDE_SKILLS_DIR")):
        if raw:
            path = Path(raw).expanduser().resolve()
            if path not in dirs:
                dirs.append(path)
    for path in (Path.home() / ".clawd" / "skills", Path.home() / ".claude" / "skills"):
        resolved = path.expanduser().resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    managed = os.environ.get("CLAWD_MANAGED_SKILLS_DIR")
    if managed:
        resolved = Path(managed).expanduser().resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        for path in (root / ".clawd" / "skills", root / ".claude" / "skills"):
            resolved = path.resolve()
            if resolved not in dirs:
                dirs.append(resolved)
    return dirs


def _discover_skill_artifacts(dirs: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for base in dirs:
        if not base.exists() or not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                found.setdefault(entry.name, []).append(entry.resolve())
    return found
def _read_skill_registry(trust_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    registry_path = trust_dir / "skill-registry.json"
    issues: list[dict[str, str]] = []
    if not registry_path.exists():
        return {}, [{"code": "trust_registry_missing", "subject": str(registry_path)}]
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [{"code": "trust_registry_unreadable", "subject": str(exc)}]
    if data.get("schema_version") != REGISTRY_SCHEMA_VERSION or not isinstance(data.get("skills"), dict):
        issues.append({"code": "trust_registry_invalid", "subject": str(registry_path)})
        return {}, issues
    return data["skills"], issues


def _hash_matches(path: Path, expected: str) -> bool | None:
    if not path.exists():
        return None
    try:
        return compute_artifact_hash(path) == expected
    except (OSError, SkillApprovalError):
        return False


def reconcile_skills(
    *,
    trust_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    trust_root = Path(
        trust_dir
        or os.environ.get("CLAWD_SKILL_TRUST_DIR")
        or (Path.home() / ".clawd" / "development-pack")
    ).expanduser().resolve()
    registry, issues = _read_skill_registry(trust_root)
    runtime = _discover_skill_artifacts(_candidate_loader_dirs(project_root))
    candidates = _discover_skill_artifacts([trust_root.parent / "review-candidates"])
    audit = _verify_audit_chain_read_only(trust_root / "skill-audit.jsonl")
    records: dict[str, Any] = {}
    for name, record in sorted(registry.items()):
        if not isinstance(record, dict):
            issues.append({"code": "skill_record_invalid", "subject": name})
            continue
        raw_artifact = str(record.get("artifact_path") or "")
        artifact = Path(raw_artifact).expanduser().resolve() if raw_artifact else None
        runtime_paths = runtime.get(name, [])
        candidate_paths = candidates.get(name, [])
        loader_reachable = artifact is not None and artifact in runtime_paths
        expected_hash = str(record.get("integrity_hash") or "")
        integrity_matches = _hash_matches(artifact, expected_hash) if artifact and expected_hash else None
        approved_active = (
            record.get("review_status") == "approved"
            and record.get("activation_status") == "active"
        )
        if approved_active and not loader_reachable:
            issues.append({"code": "active_skill_not_loader_reachable", "subject": name})
        if approved_active and integrity_matches is False:
            issues.append({"code": "active_skill_integrity_mismatch", "subject": name})
        if runtime_paths and artifact is not None and artifact not in runtime_paths:
            runtime_hashes = []
            for path in runtime_paths:
                try:
                    runtime_hashes.append(compute_artifact_hash(path))
                except (OSError, SkillApprovalError):
                    runtime_hashes.append("")
            if expected_hash not in runtime_hashes:
                issues.append({"code": "runtime_copy_differs_from_trust_artifact", "subject": name})
        # Preserved review candidates are evidence, not loader duplicates.
        # Only multiple loader-reachable copies create runtime ambiguity.
        if len(runtime_paths) > 1:
            issues.append({"code": "duplicate_skill_artifacts", "subject": name})
        records[name] = {
            "review_status": record.get("review_status"),
            "activation_status": record.get("activation_status"),
            "artifact_path": str(artifact) if artifact else "",
            "artifact_exists": bool(artifact and artifact.exists()),
            "integrity_matches": integrity_matches,
            "loader_reachable": loader_reachable,
            "runtime_paths": [str(p) for p in runtime_paths],
            "candidate_paths": [str(p) for p in candidate_paths],
        }
    return {
        "audit_chain": audit,
        "records": records,
        "issues": sorted(issues, key=lambda item: (item["code"], item["subject"])),
    }
def reconcile_tools(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or load_capability_manifest()
    registry = build_default_registry(include_user_tools=False)
    specs = registry.list_specs()
    actual_names = [spec.name for spec in specs]
    expected_names = expected_registered_tool_names(manifest)
    issues: list[dict[str, str]] = []
    actual_set = set(actual_names)
    expected_set = set(expected_names)
    for name in sorted(expected_set - actual_set):
        issues.append({"code": "expected_tool_not_registered", "subject": name})
    for name in sorted(actual_set - expected_set):
        issues.append({"code": "unexpected_tool_registered", "subject": name})
    worktree_feature = manifest["features"].get("git_worktree_runtime")
    if (
        isinstance(worktree_feature, dict)
        and worktree_feature.get("state") == "ACTIVE_SUPPORTED"
        and not git_worktree_runtime_available()
    ):
        issues.append({"code": "git_worktree_runtime_unavailable", "subject": "Git"})

    lsp_record = manifest["tools"].get("LSP")
    if (
        isinstance(lsp_record, dict)
        and lsp_record.get("state") == "ACTIVE_SUPPORTED"
        and bool(lsp_record.get("registered_expected"))
        and not pyright_runtime_available()
    ):
        issues.append({"code": "lsp_runtime_unavailable", "subject": "Pyright 1.1.414"})

    mcp_active = any(
        isinstance(manifest["tools"].get(name), dict)
        and manifest["tools"][name].get("state") == "ACTIVE_SUPPORTED"
        and bool(manifest["tools"][name].get("registered_expected"))
        for name in (
            "ListMcpResourcesTool",
            "ReadMcpResourceTool",
            "ListMcpToolsTool",
            "MCP",
        )
    )
    if mcp_active:
        if not mcp_resource_runtime_available():
            issues.append(
                {"code": "mcp_resource_runtime_unavailable", "subject": f"mcp {MCP_SDK_VERSION}"}
            )
        manifest_ok, manifest_detail = mcp_manifest_status()
        if not manifest_ok:
            issues.append(
                {"code": "mcp_resource_manifest_invalid", "subject": manifest_detail}
            )
    metadata = {}
    for spec in specs:
        record = manifest["tools"].get(spec.name)
        metadata[spec.name] = {
            "state": record.get("state") if isinstance(record, dict) else "UNDECLARED",
            "permission_policy": spec.permission_policy,
            "is_read_only": spec.is_read_only,
            "is_destructive": spec.is_destructive,
        }
        if not isinstance(record, dict):
            issues.append({"code": "registered_tool_missing_from_manifest", "subject": spec.name})
            continue
        expected_policy = record.get("permission_policy")
        if not expected_policy:
            issues.append({"code": "manifest_permission_policy_missing", "subject": spec.name})
        elif expected_policy != spec.permission_policy:
            issues.append({"code": "permission_policy_mismatch", "subject": spec.name})
        if bool(record.get("mutates_state")) and spec.is_read_only:
            issues.append({"code": "read_only_metadata_conflicts_with_mutation", "subject": spec.name})
    return {
        "registered": actual_names,
        "expected_registered": expected_names,
        "metadata": metadata,
        "issues": sorted(issues, key=lambda item: (item["code"], item["subject"])),
    }


def reconcile_capabilities(
    *,
    trust_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_capability_manifest()
    return {
        "manifest_schema_version": manifest["schema_version"],
        "tools": reconcile_tools(manifest),
        "skills": reconcile_skills(trust_dir=trust_dir, project_root=project_root),
        "plugins": reconcile_python_plugins(),
        "deferred_features": sorted(
            name for name, rec in manifest["features"].items()
            if rec["state"] == "DEFERRED_NOT_PRODUCTION_READY"
        ),
    }


def main() -> int:
    print(json.dumps(reconcile_capabilities(), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
