from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.skills.trust_registry import (
    ACTIVATION_ACTIVE,
    ACTIVATION_INACTIVE,
    REVIEW_APPROVED,
    REVIEW_REQUIRED,
    AuditChainError,
    SkillTrustRegistry,
    compute_artifact_hash,
    default_skill_record,
)


class SkillTrustRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.skill_dir = self.root / "skills" / "demo"
        self.skill_dir.mkdir(parents=True)
        (self.skill_dir / "SKILL.md").write_text(
            "---\\nversion: 1.0.0\\ndescription: demo\\n---\\nDemo body\\n",
            encoding="utf-8",
        )
        self.registry = SkillTrustRegistry(self.root / "trust")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _register_review_approve_activate(self) -> dict:
        record = default_skill_record(
            name="demo",
            version="1.0.0",
            source="https://example.invalid/demo",
            pinned_commit="abc123",
            artifact_path=self.skill_dir,
            purpose="test skill",
            capabilities=["prompt guidance"],
            permissions=["read-workspace"],
        )
        self.registry.register_quarantined(
            record,
            initiator="test",
            reason="test registration",
        )
        self.registry.mark_reviewed(
            "demo",
            reviewed_by="test-reviewer",
            initiator="test",
            reason="review complete",
        )
        self.registry.approve(
            "demo",
            reviewed_by="test-reviewer",
            initiator="test",
            reason="approve exact artifact",
        )
        return self.registry.activate(
            "demo",
            initiator="test",
            reason="activate approved artifact",
        )

    def test_exact_reviewed_artifact_can_activate(self) -> None:
        active = self._register_review_approve_activate()
        self.assertEqual(active["review_status"], REVIEW_APPROVED)
        self.assertEqual(active["activation_status"], ACTIVATION_ACTIVE)
        allowed, reason = self.registry.is_active_and_current(
            "demo",
            artifact_path=self.skill_dir,
            declared_version="1.0.0",
        )
        self.assertTrue(allowed, reason)

    def test_artifact_change_fails_closed_and_requires_review(self) -> None:
        self._register_review_approve_activate()
        (self.skill_dir / "SKILL.md").write_text(
            "---\\nversion: 1.0.0\\ndescription: demo\\n---\\nChanged body\\n",
            encoding="utf-8",
        )
        allowed, reason = self.registry.is_active_and_current(
            "demo",
            artifact_path=self.skill_dir,
            declared_version="1.0.0",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "integrity hash changed")
        record = self.registry.get("demo")
        assert record is not None
        self.assertEqual(record["review_status"], REVIEW_REQUIRED)
        self.assertEqual(record["activation_status"], ACTIVATION_INACTIVE)
        self.assertEqual(record["integrity_hash"], compute_artifact_hash(self.skill_dir))

    def test_loader_path_promotion_requires_reapproval_of_exact_runtime_artifact(self) -> None:
        self._register_review_approve_activate()
        runtime_dir = self.root / "runtime-skills" / "demo"
        runtime_dir.parent.mkdir(parents=True)
        shutil.copytree(self.skill_dir, runtime_dir)

        updated = self.registry.update_declaration(
            "demo",
            {"artifact_path": runtime_dir},
            initiator="test",
            reason="promote reviewed artifact to loader-reachable runtime path",
        )
        self.assertEqual(updated["review_status"], REVIEW_REQUIRED)
        self.assertEqual(updated["activation_status"], ACTIVATION_INACTIVE)
        self.assertEqual(updated["integrity_hash"], compute_artifact_hash(self.skill_dir))

        self.registry.mark_reviewed(
            "demo",
            reviewed_by="test-reviewer",
            initiator="test",
            reason="runtime artifact matches reviewed candidate",
        )
        self.registry.approve(
            "demo",
            reviewed_by="test-reviewer",
            initiator="test",
            reason="approve exact loader-reachable artifact",
        )
        active = self.registry.activate(
            "demo",
            initiator="test",
            reason="activate exact approved runtime artifact",
        )
        self.assertEqual(active["activation_status"], ACTIVATION_ACTIVE)

        allowed, reason = self.registry.is_active_and_current(
            "demo",
            artifact_path=runtime_dir,
            declared_version="1.0.0",
        )
        self.assertTrue(allowed, reason)

    def test_permission_change_invalidates_prior_approval(self) -> None:
        self._register_review_approve_activate()
        updated = self.registry.update_declaration(
            "demo",
            {"permissions": ["read-workspace", "write-workspace"]},
            initiator="test",
            reason="permission expansion",
        )
        self.assertEqual(updated["review_status"], REVIEW_REQUIRED)
        self.assertEqual(updated["activation_status"], ACTIVATION_INACTIVE)

    def test_version_change_fails_closed(self) -> None:
        self._register_review_approve_activate()
        allowed, reason = self.registry.is_active_and_current(
            "demo",
            artifact_path=self.skill_dir,
            declared_version="2.0.0",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "version changed")

    def test_audit_log_is_hash_chained_and_tampering_is_detected(self) -> None:
        self._register_review_approve_activate()
        count, head = self.registry.verify_audit_chain()
        self.assertGreaterEqual(count, 5)
        self.assertEqual(len(head), 64)

        lines = self.registry.audit_path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["reason"] = "tampered"
        lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        self.registry.audit_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

        with self.assertRaises(AuditChainError):
            self.registry.verify_audit_chain()

    def test_corrections_are_appended_and_reference_original_entry(self) -> None:
        original = self.registry.append_audit(
            event="note",
            skill_name="demo",
            initiator="test",
            reason="original",
            old_state={},
            new_state={},
            previous_integrity_hash="",
            new_integrity_hash="",
            previous_permissions=[],
            new_permissions=[],
            review_outcome="noted",
        )
        correction = self.registry.append_correction(
            correction_of=original["entry_id"],
            skill_name="demo",
            initiator="test",
            reason="correct metadata",
            metadata={"replacement": "corrected note"},
        )
        self.assertEqual(correction["correction_of"], original["entry_id"])
        self.registry.verify_audit_chain()


if __name__ == "__main__":
    unittest.main()
