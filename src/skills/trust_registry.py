from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1

REVIEW_QUARANTINED = "quarantined"
REVIEW_REVIEWED = "reviewed"
REVIEW_APPROVED = "approved"
REVIEW_REQUIRED = "review_required"

ACTIVATION_INACTIVE = "inactive"
ACTIVATION_ACTIVE = "active"

_RUNTIME_SENSITIVE_FIELDS = (
    "version",
    "source",
    "pinned_commit",
    "integrity_hash",
    "capabilities",
    "executable_code",
    "hooks",
    "mcp_configuration",
    "filesystem_scope",
    "network_access",
    "dependencies",
    "external_accounts_or_apis",
    "additional_claude_usage",
    "permissions",
    "artifact_path",
)

_HASH_EXCLUDED_DIRS = {".git", "__pycache__"}
_HASH_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


class SkillTrustError(RuntimeError):
    """Base error for the fail-closed skill trust system."""


class AuditChainError(SkillTrustError):
    """Raised when the append-only audit hash chain cannot be verified."""


class SkillApprovalError(SkillTrustError):
    """Raised when a skill cannot be reviewed, approved, or activated safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item) for item in value if str(item)})
    text = str(value).strip()
    return [text] if text else []


def _normalize_network_access(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"enabled": False, "allowed_hosts": []}
    enabled = bool(value.get("enabled", False))
    hosts = _normalize_string_list(value.get("allowed_hosts"))
    return {"enabled": enabled, "allowed_hosts": hosts}


def _normalize_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"expected": False, "level": "none", "reason": ""}
    expected = bool(value.get("expected", False))
    level = str(value.get("level") or ("unknown" if expected else "none"))
    reason = str(value.get("reason") or "")
    return {"expected": expected, "level": level, "reason": reason}


def _resolved_path_string(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def compute_artifact_hash(path: str | Path) -> str:
    """
    Hash the exact reviewed artifact deterministically.

    The relative path and file bytes are both covered. Transient Python caches and
    .git metadata are excluded so normal execution does not invalidate approval.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise SkillApprovalError(f"skill artifact does not exist: {root}")

    digest = hashlib.sha256()

    if root.is_file():
        digest.update(root.name.encode("utf-8"))
        digest.update(b"\0")
        with root.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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


def _security_snapshot(entry: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field in _RUNTIME_SENSITIVE_FIELDS:
        value = deepcopy(entry.get(field))
        if field in {
            "capabilities",
            "filesystem_scope",
            "dependencies",
            "external_accounts_or_apis",
            "permissions",
        }:
            value = _normalize_string_list(value)
        elif field == "network_access":
            value = _normalize_network_access(value)
        elif field == "additional_claude_usage":
            value = _normalize_usage(value)
        elif field in {"executable_code", "hooks", "mcp_configuration"}:
            value = bool(value)
        elif field == "artifact_path" and value:
            value = _resolved_path_string(value)
        snapshot[field] = value
    return snapshot


def compute_approval_fingerprint(entry: dict[str, Any]) -> str:
    """Hash all artifact and declared-capability fields covered by approval."""
    return _sha256_text(_canonical_json(_security_snapshot(entry)))


def default_skill_record(
    *,
    name: str,
    artifact_path: str | Path,
    version: str | None = None,
    source: str = "",
    pinned_commit: str = "",
    purpose: str = "",
    capabilities: Iterable[str] | None = None,
    executable_code: bool = False,
    hooks: bool = False,
    mcp_configuration: bool = False,
    filesystem_scope: Iterable[str] | None = None,
    network_access: dict[str, Any] | None = None,
    dependencies: Iterable[str] | None = None,
    external_accounts_or_apis: Iterable[str] | None = None,
    additional_claude_usage: dict[str, Any] | None = None,
    permissions: Iterable[str] | None = None,
    risks: Iterable[str] | None = None,
) -> dict[str, Any]:
    resolved = _resolved_path_string(artifact_path)
    integrity = compute_artifact_hash(resolved)
    return {
        "name": name,
        "version": version or "",
        "source": source,
        "pinned_commit": pinned_commit,
        "integrity_hash": integrity,
        "artifact_path": resolved,
        "purpose": purpose,
        "capabilities": _normalize_string_list(capabilities),
        "executable_code": bool(executable_code),
        "hooks": bool(hooks),
        "mcp_configuration": bool(mcp_configuration),
        "filesystem_scope": _normalize_string_list(filesystem_scope),
        "network_access": _normalize_network_access(network_access),
        "dependencies": _normalize_string_list(dependencies),
        "external_accounts_or_apis": _normalize_string_list(external_accounts_or_apis),
        "additional_claude_usage": _normalize_usage(additional_claude_usage),
        "permissions": _normalize_string_list(permissions),
        "risks": _normalize_string_list(risks),
        "review_status": REVIEW_QUARANTINED,
        "activation_status": ACTIVATION_INACTIVE,
        "reviewed_by": "",
        "last_audited": "",
        "approved_version": "",
        "approval_notes": "",
        "approval_fingerprint": "",
    }


class SkillTrustRegistry:
    """
    Persistent, fail-closed trust registry for Clawd skills.

    Registry approval applies only to the exact reviewed artifact and declared
    capabilities. Approval never transfers automatically to updates, forks,
    dependency changes, or expanded permissions.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            configured = os.environ.get("CLAWD_SKILL_TRUST_DIR")
            base_dir = configured or (Path.home() / ".clawd" / "development-pack")
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.registry_path = self.base_dir / "skill-registry.json"
        self.audit_path = self.base_dir / "skill-audit.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.registry_path.exists():
            self._atomic_write_registry(
                {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "skills": {},
                }
            )
        if not self.audit_path.exists():
            self.audit_path.touch()
            self.append_audit(
                event="audit_log_created",
                skill_name="",
                initiator="system",
                reason="Initialized Clawd skill audit chain.",
                old_state={},
                new_state={},
                previous_integrity_hash="",
                new_integrity_hash="",
                previous_permissions=[],
                new_permissions=[],
                review_outcome="initialized",
            )

    def _load_registry(self) -> dict[str, Any]:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillTrustError(f"cannot read skill registry safely: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("skills"), dict):
            raise SkillTrustError("skill registry has an invalid structure")
        if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise SkillTrustError("unsupported skill registry schema version")
        return data

    def _atomic_write_registry(self, data: dict[str, Any]) -> None:
        temp = self.registry_path.with_suffix(".json.tmp")
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, self.registry_path)

    def verify_audit_chain(self) -> tuple[int, str]:
        previous_hash = ""
        count = 0
        try:
            lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AuditChainError(f"cannot read audit log: {exc}") from exc

        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuditChainError(f"audit entry {line_number} is invalid JSON") from exc
            if not isinstance(entry, dict):
                raise AuditChainError(f"audit entry {line_number} is not an object")
            current_hash = str(entry.get("current_entry_hash") or "")
            if not current_hash:
                raise AuditChainError(f"audit entry {line_number} has no current_entry_hash")
            if str(entry.get("previous_entry_hash") or "") != previous_hash:
                raise AuditChainError(f"audit chain break at entry {line_number}")
            payload = dict(entry)
            payload.pop("current_entry_hash", None)
            expected = _sha256_text(_canonical_json(payload))
            if expected != current_hash:
                raise AuditChainError(f"audit entry {line_number} hash mismatch")
            previous_hash = current_hash
            count += 1
        return count, previous_hash

    def latest_approved_fingerprint(self, skill_name: str) -> str:
        """Return the approval fingerprint recorded in the latest approval audit entry."""
        self.verify_audit_chain()
        latest = ""
        for raw in self.audit_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            if entry.get("skill") != skill_name:
                continue
            if entry.get("event") == "approved":
                latest = str((entry.get("metadata") or {}).get("approval_fingerprint") or "")
            elif entry.get("event") == "review_required":
                latest = ""
        return latest

    def append_audit(
        self,
        *,
        event: str,
        skill_name: str,
        initiator: str,
        reason: str,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        previous_integrity_hash: str,
        new_integrity_hash: str,
        previous_permissions: Iterable[str],
        new_permissions: Iterable[str],
        review_outcome: str,
        correction_of: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _count, previous_hash = self.verify_audit_chain()
        payload: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "entry_id": str(uuid.uuid4()),
            "previous_entry_hash": previous_hash,
            "timestamp": _utc_now(),
            "event": event,
            "skill": skill_name,
            "initiator": initiator,
            "reason": reason,
            "old_state": old_state,
            "new_state": new_state,
            "previous_integrity_hash": previous_integrity_hash,
            "new_integrity_hash": new_integrity_hash,
            "previous_permissions": _normalize_string_list(previous_permissions),
            "new_permissions": _normalize_string_list(new_permissions),
            "review_outcome": review_outcome,
            "correction_of": correction_of or "",
            "metadata": metadata or {},
        }
        payload["current_entry_hash"] = _sha256_text(_canonical_json(payload))
        line = _canonical_json(payload) + "\n"
        try:
            with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise AuditChainError(f"could not append audit entry: {exc}") from exc
        return payload

    def list_skills(self) -> list[dict[str, Any]]:
        data = self._load_registry()
        return [deepcopy(value) for _, value in sorted(data["skills"].items())]

    def get(self, name: str) -> dict[str, Any] | None:
        data = self._load_registry()
        record = data["skills"].get(name)
        return deepcopy(record) if isinstance(record, dict) else None

    def register_quarantined(
        self,
        record: dict[str, Any],
        *,
        initiator: str,
        reason: str,
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        name = str(record.get("name") or "").strip()
        if not name:
            raise SkillApprovalError("skill name is required")

        new_record = deepcopy(record)
        new_record["name"] = name
        new_record["artifact_path"] = _resolved_path_string(new_record["artifact_path"])
        new_record["integrity_hash"] = compute_artifact_hash(new_record["artifact_path"])
        new_record["review_status"] = REVIEW_QUARANTINED
        new_record["activation_status"] = ACTIVATION_INACTIVE
        new_record["reviewed_by"] = ""
        new_record["last_audited"] = ""
        new_record["approved_version"] = ""
        new_record["approval_notes"] = ""
        new_record["approval_fingerprint"] = ""

        data = self._load_registry()
        old = data["skills"].get(name)
        data["skills"][name] = new_record
        self._atomic_write_registry(data)
        self.append_audit(
            event="registered_quarantined",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(old),
            new_state=_state_of(new_record),
            previous_integrity_hash=str((old or {}).get("integrity_hash") or ""),
            new_integrity_hash=new_record["integrity_hash"],
            previous_permissions=(old or {}).get("permissions") or [],
            new_permissions=new_record.get("permissions") or [],
            review_outcome=REVIEW_QUARANTINED,
            metadata={"source": new_record.get("source", ""), "pinned_commit": new_record.get("pinned_commit", "")},
        )
        return deepcopy(new_record)

    def update_declaration(
        self,
        name: str,
        updates: dict[str, Any],
        *,
        initiator: str,
        reason: str,
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            raise SkillApprovalError(f"unknown skill: {name}")

        before = deepcopy(current)
        for key, value in updates.items():
            if key in {"name", "review_status", "activation_status", "approval_fingerprint"}:
                continue
            if key in {
                "capabilities",
                "filesystem_scope",
                "dependencies",
                "external_accounts_or_apis",
                "permissions",
                "risks",
            }:
                value = _normalize_string_list(value)
            elif key == "network_access":
                value = _normalize_network_access(value)
            elif key == "additional_claude_usage":
                value = _normalize_usage(value)
            elif key in {"executable_code", "hooks", "mcp_configuration"}:
                value = bool(value)
            elif key == "artifact_path":
                value = _resolved_path_string(value)
            current[key] = value

        if "artifact_path" in updates or Path(current["artifact_path"]).exists():
            current["integrity_hash"] = compute_artifact_hash(current["artifact_path"])

        sensitive_changed = _security_snapshot(before) != _security_snapshot(current)
        if sensitive_changed and (
            before.get("review_status") in {REVIEW_REVIEWED, REVIEW_APPROVED}
            or before.get("activation_status") == ACTIVATION_ACTIVE
        ):
            current["review_status"] = REVIEW_REQUIRED
            current["activation_status"] = ACTIVATION_INACTIVE
            current["approval_fingerprint"] = ""

        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="declaration_updated",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome=current.get("review_status") or "",
            metadata={
                "sensitive_change": sensitive_changed,
                "previous_declaration": _security_snapshot(before),
                "new_declaration": _security_snapshot(current),
            },
        )
        return deepcopy(current)

    def mark_reviewed(
        self,
        name: str,
        *,
        reviewed_by: str,
        initiator: str,
        reason: str,
        review_outcome: str = "passed",
        approval_notes: str = "",
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            raise SkillApprovalError(f"unknown skill: {name}")
        before = deepcopy(current)

        current["integrity_hash"] = compute_artifact_hash(current["artifact_path"])
        current["review_status"] = REVIEW_REVIEWED
        current["activation_status"] = ACTIVATION_INACTIVE
        current["reviewed_by"] = reviewed_by
        current["last_audited"] = _utc_now()
        current["approval_notes"] = approval_notes
        current["approval_fingerprint"] = ""

        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="review_completed",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome=review_outcome,
            metadata={"reviewed_by": reviewed_by, "approval_notes": approval_notes},
        )
        return deepcopy(current)

    def approve(
        self,
        name: str,
        *,
        reviewed_by: str,
        initiator: str,
        reason: str,
        approval_notes: str = "",
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            raise SkillApprovalError(f"unknown skill: {name}")
        if current.get("review_status") != REVIEW_REVIEWED:
            raise SkillApprovalError(f"skill must be reviewed before approval: {name}")

        before = deepcopy(current)
        actual_hash = compute_artifact_hash(current["artifact_path"])
        if actual_hash != current.get("integrity_hash"):
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash=actual_hash,
                initiator=initiator,
                reason="Artifact changed after review and before approval.",
            )
            raise SkillApprovalError(f"skill changed after review; re-review required: {name}")

        current["review_status"] = REVIEW_APPROVED
        current["activation_status"] = ACTIVATION_INACTIVE
        current["reviewed_by"] = reviewed_by
        current["last_audited"] = _utc_now()
        current["approved_version"] = str(current.get("version") or "")
        current["approval_notes"] = approval_notes or str(current.get("approval_notes") or "")
        current["approval_fingerprint"] = compute_approval_fingerprint(current)

        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="approved",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome=REVIEW_APPROVED,
            metadata={
                "reviewed_by": reviewed_by,
                "approved_version": current["approved_version"],
                "approval_fingerprint": current["approval_fingerprint"],
            },
        )
        return deepcopy(current)

    def activate(
        self,
        name: str,
        *,
        initiator: str,
        reason: str,
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            raise SkillApprovalError(f"unknown skill: {name}")
        if current.get("review_status") != REVIEW_APPROVED:
            raise SkillApprovalError(f"skill is not approved: {name}")

        before = deepcopy(current)
        actual_hash = compute_artifact_hash(current["artifact_path"])
        expected_fingerprint = compute_approval_fingerprint(current)
        audit_fingerprint = self.latest_approved_fingerprint(name)
        if (
            actual_hash != current.get("integrity_hash")
            or expected_fingerprint != current.get("approval_fingerprint")
            or current.get("approval_fingerprint") != audit_fingerprint
            or str(current.get("version") or "") != str(current.get("approved_version") or "")
        ):
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash=actual_hash,
                initiator=initiator,
                reason="Approved artifact, audit approval, or declared capabilities changed before activation.",
            )
            raise SkillApprovalError(f"approval no longer matches exact artifact/capabilities: {name}")

        current["activation_status"] = ACTIVATION_ACTIVE
        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="activated",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome="active",
            metadata={"approval_fingerprint": current.get("approval_fingerprint", "")},
        )
        return deepcopy(current)

    def deactivate(
        self,
        name: str,
        *,
        initiator: str,
        reason: str,
    ) -> dict[str, Any]:
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            raise SkillApprovalError(f"unknown skill: {name}")
        before = deepcopy(current)
        current["activation_status"] = ACTIVATION_INACTIVE
        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="deactivated",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome="inactive",
        )
        return deepcopy(current)

    def append_correction(
        self,
        *,
        correction_of: str,
        skill_name: str,
        initiator: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not correction_of:
            raise SkillApprovalError("correction_of entry id is required")
        return self.append_audit(
            event="correction",
            skill_name=skill_name,
            initiator=initiator,
            reason=reason,
            old_state={},
            new_state={},
            previous_integrity_hash="",
            new_integrity_hash="",
            previous_permissions=[],
            new_permissions=[],
            review_outcome="correction_appended",
            correction_of=correction_of,
            metadata=metadata,
        )

    def is_active_and_current(
        self,
        name: str,
        *,
        artifact_path: str | Path,
        declared_version: str | None,
        initiator: str = "runtime",
    ) -> tuple[bool, str]:
        """
        Runtime activation gate.

        Any audit-chain failure, artifact change, path change, version change, or
        declared-capability change fails closed.
        """
        self.verify_audit_chain()
        data = self._load_registry()
        current = data["skills"].get(name)
        if not isinstance(current, dict):
            return False, "not registered"
        if current.get("review_status") != REVIEW_APPROVED:
            return False, f"review status is {current.get('review_status') or 'unknown'}"
        if current.get("activation_status") != ACTIVATION_ACTIVE:
            return False, "inactive"

        runtime_path = _resolved_path_string(artifact_path)
        registered_path = _resolved_path_string(current.get("artifact_path") or runtime_path)
        if runtime_path != registered_path:
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash="",
                initiator=initiator,
                reason="Runtime skill path differs from the approved artifact path.",
            )
            return False, "artifact path changed"

        actual_hash = compute_artifact_hash(runtime_path)
        fingerprint = compute_approval_fingerprint(current)
        audit_fingerprint = self.latest_approved_fingerprint(name)
        declared = str(declared_version or "")
        registered_version = str(current.get("version") or "")
        approved_version = str(current.get("approved_version") or "")

        if actual_hash != str(current.get("integrity_hash") or ""):
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash=actual_hash,
                initiator=initiator,
                reason="Approved skill artifact integrity hash changed.",
            )
            return False, "integrity hash changed"

        if (
            fingerprint != str(current.get("approval_fingerprint") or "")
            or str(current.get("approval_fingerprint") or "") != audit_fingerprint
        ):
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash=actual_hash,
                initiator=initiator,
                reason="Declared capabilities or audited approval fingerprint changed.",
            )
            return False, "declared capabilities or audited approval changed"

        if declared != registered_version or registered_version != approved_version:
            self._invalidate_record(
                data,
                name,
                current,
                actual_hash=actual_hash,
                initiator=initiator,
                reason="Runtime or registered skill version differs from approved version.",
            )
            return False, "version changed"

        return True, "approved and active"

    def _invalidate_record(
        self,
        data: dict[str, Any],
        name: str,
        current: dict[str, Any],
        *,
        actual_hash: str,
        initiator: str,
        reason: str,
    ) -> dict[str, Any]:
        before = deepcopy(current)
        if actual_hash:
            current["integrity_hash"] = actual_hash
        current["review_status"] = REVIEW_REQUIRED
        current["activation_status"] = ACTIVATION_INACTIVE
        current["approval_fingerprint"] = ""
        data["skills"][name] = current
        self._atomic_write_registry(data)
        self.append_audit(
            event="review_required",
            skill_name=name,
            initiator=initiator,
            reason=reason,
            old_state=_state_of(before),
            new_state=_state_of(current),
            previous_integrity_hash=str(before.get("integrity_hash") or ""),
            new_integrity_hash=str(current.get("integrity_hash") or ""),
            previous_permissions=before.get("permissions") or [],
            new_permissions=current.get("permissions") or [],
            review_outcome=REVIEW_REQUIRED,
        )
        return deepcopy(current)


def _state_of(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {
        "review_status": record.get("review_status", ""),
        "activation_status": record.get("activation_status", ""),
        "version": record.get("version", ""),
        "approved_version": record.get("approved_version", ""),
        "approval_fingerprint": record.get("approval_fingerprint", ""),
    }
