from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.capabilities import (
    expected_registered_tool_names,
    load_capability_manifest,
    reconcile_skills,
    reconcile_tools,
    render_capability_status_markdown,
)
from src.skills.trust_registry import compute_artifact_hash
from src.tool_system.defaults import build_default_registry


class CapabilityManifestTests(unittest.TestCase):
    def test_default_registry_matches_manifest(self) -> None:
        manifest = load_capability_manifest()
        actual = sorted(spec.name for spec in build_default_registry(include_user_tools=False).list_specs())
        self.assertEqual(actual, expected_registered_tool_names(manifest))

    def test_authentication_recovery_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["authentication_recovery"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("prevents direct-response auth failures", feature["reason"])
        self.assertIn("without automatically retrying", feature["reason"])

    def test_context_engine_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["context_engine"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("project map", feature["reason"])
        self.assertIn("does not follow symlinks", feature["reason"])

    def test_permission_policy_configuration_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["permission_policy_configuration"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("operator-permissions.json", feature["reason"])
        self.assertIn("cannot expand operator authority", feature["reason"])

    def test_python_plugin_runtime_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["python_plugin_runtime"]["state"],
            "ACTIVE_SUPPORTED",
        )

    def test_provider_extensions_are_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["provider_extensions"]["state"],
            "ACTIVE_SUPPORTED",
        )

    def test_custom_commands_tools_are_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["custom_commands_tools"]["state"],
            "ACTIVE_SUPPORTED",
        )

    def test_chinese_provider_ecosystem_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["chinese_provider_ecosystem"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        for provider in ("DeepSeek", "Qwen", "GLM", "MiniMax"):
            self.assertIn(provider, feature["reason"])

    def test_developer_quality_tooling_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["developer_quality_tooling"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        for tool in ("uv", "pytest", "Ruff", "Mypy"):
            self.assertIn(tool, feature["reason"])

    def test_enterprise_workflow_extensions_are_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["enterprise_workflow_extensions"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("WORKFLOWS", feature["reason"])
        self.assertIn("tool allowlists", feature["reason"])
        self.assertIn("no background daemon", feature["reason"])

    def test_data_engineering_runtime_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["data_engineering_runtime"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("CSV", feature["reason"])
        self.assertIn("JSONL", feature["reason"])

    def test_runtime_observability_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        feature = manifest["features"]["sanitized_runtime_instrumentation"]
        self.assertEqual(feature["state"], "ACTIVE_SUPPORTED")
        self.assertIn("/doctor", feature["reason"])
        self.assertIn("best-effort", feature["reason"])

    def test_project_setup_advisor_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["project_setup_advisor"]["state"],
            "ACTIVE_SUPPORTED",
        )

    def test_generic_hook_runtime_is_intentionally_disabled(self) -> None:
        manifest = load_capability_manifest()
        hook_runtime = manifest["features"]["hook_runtime"]
        self.assertEqual(hook_runtime["state"], "INTENTIONALLY_DISABLED")
        self.assertIn("dedicated fail-closed chokepoints", hook_runtime["reason"])

    def test_subagent_runtime_is_deferred_and_not_mislabeled(self) -> None:
        manifest = load_capability_manifest()
        subagent_runtime = manifest["features"]["subagent_runtime"]
        self.assertEqual(subagent_runtime["state"], "DEFERRED_NOT_PRODUCTION_READY")
        self.assertIn("No isolated child LLM loop", subagent_runtime["reason"])
        self.assertEqual(manifest["tools"]["Agent"]["state"], "CLAWD_SPECIFIC")
        self.assertIn("sequential tool orchestration", manifest["tools"]["Agent"]["reason"])
        self.assertIn("does not spawn", manifest["tools"]["TeamCreate"]["reason"])

    def test_git_worktree_runtime_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["git_worktree_runtime"]["state"],
            "ACTIVE_SUPPORTED",
        )
        self.assertEqual(manifest["tools"]["EnterWorktree"]["state"], "ACTIVE_SUPPORTED")
        self.assertTrue(manifest["tools"]["EnterWorktree"]["registered_expected"])
        self.assertEqual(manifest["tools"]["EnterWorktree"]["permission_policy"], "checked")
        self.assertEqual(manifest["tools"]["ExitWorktree"]["state"], "ACTIVE_SUPPORTED")
        self.assertTrue(manifest["tools"]["ExitWorktree"]["registered_expected"])
        self.assertEqual(manifest["tools"]["ExitWorktree"]["permission_policy"], "allow")

    def test_git_worktree_runtime_unavailable_is_reported(self) -> None:
        with patch("src.capabilities.git_worktree_runtime_available", return_value=False):
            report = reconcile_tools()
        self.assertIn(
            {"code": "git_worktree_runtime_unavailable", "subject": "Git"},
            report["issues"],
        )

    def test_lsp_runtime_is_declared_active(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(manifest["features"]["lsp_runtime"]["state"], "ACTIVE_SUPPORTED")
        self.assertEqual(manifest["tools"]["LSP"]["state"], "ACTIVE_SUPPORTED")
        self.assertTrue(manifest["tools"]["LSP"]["registered_expected"])
        self.assertEqual(manifest["tools"]["LSP"]["permission_policy"], "checked")

    def test_lsp_runtime_unavailable_is_reported(self) -> None:
        with patch("src.capabilities.pyright_runtime_available", return_value=False):
            report = reconcile_tools()
        self.assertIn(
            {"code": "lsp_runtime_unavailable", "subject": "Pyright 1.1.414"},
            report["issues"],
        )

    def test_mcp_runtime_is_active_with_guarded_tool_execution(self) -> None:
        manifest = load_capability_manifest()
        self.assertEqual(
            manifest["features"]["mcp_resource_runtime"]["state"],
            "ACTIVE_SUPPORTED",
        )
        self.assertEqual(
            manifest["features"]["mcp_runtime"]["state"],
            "ACTIVE_SUPPORTED",
        )
        for name in (
            "ListMcpResourcesTool",
            "ReadMcpResourceTool",
            "ListMcpToolsTool",
            "MCP",
        ):
            self.assertEqual(manifest["tools"][name]["state"], "ACTIVE_SUPPORTED")
            self.assertTrue(manifest["tools"][name]["registered_expected"])
            self.assertEqual(manifest["tools"][name]["permission_policy"], "checked")

    def test_mcp_resource_runtime_unavailable_is_reported(self) -> None:
        with (
            patch("src.capabilities.mcp_resource_runtime_available", return_value=False),
            patch("src.capabilities.mcp_manifest_status", return_value=(True, "ok")),
        ):
            report = reconcile_tools()
        self.assertIn(
            {"code": "mcp_resource_runtime_unavailable", "subject": "mcp 2.2.0"},
            report["issues"],
        )

    def test_mcp_resource_manifest_invalid_is_reported(self) -> None:
        with (
            patch("src.capabilities.mcp_resource_runtime_available", return_value=True),
            patch(
                "src.capabilities.mcp_manifest_status",
                return_value=(False, "invalid test manifest"),
            ),
        ):
            report = reconcile_tools()
        self.assertIn(
            {"code": "mcp_resource_manifest_invalid", "subject": "invalid test manifest"},
            report["issues"],
        )

    def test_deferred_and_disabled_tools_are_not_registered(self) -> None:
        manifest = load_capability_manifest()
        registry = build_default_registry(include_user_tools=False)
        unavailable = [
            name for name, record in manifest["tools"].items()
            if not record.get("registered_expected")
        ]
        for name in unavailable:
            with self.subTest(name=name):
                self.assertIsNone(registry.get(name))

    def test_reconciler_reports_no_tool_contract_drift(self) -> None:
        with (
            patch("src.capabilities.git_worktree_runtime_available", return_value=True),
            patch("src.capabilities.pyright_runtime_available", return_value=True),
            patch("src.capabilities.mcp_resource_runtime_available", return_value=True),
            patch("src.capabilities.mcp_manifest_status", return_value=(True, "ok")),
        ):
            report = reconcile_tools()
        self.assertEqual(report["issues"], [])
        self.assertTrue(
            all(
                item["permission_policy"] in {"allow", "checked", "self_gated", "delegated"}
                for item in report["metadata"].values()
            )
        )

    def test_rendered_status_table_includes_feature_states(self) -> None:
        rendered = render_capability_status_markdown()
        self.assertIn("| Capability state | Tools | Features |", rendered)
        self.assertIn("subagent_runtime", rendered)
        self.assertIn("hook_runtime", rendered)
        deferred_line = next(
            line
            for line in rendered.splitlines()
            if line.startswith("| Deferred / not production-ready |")
        )
        self.assertIn("subagent_runtime", deferred_line)
        disabled_line = next(
            line
            for line in rendered.splitlines()
            if line.startswith("| Intentionally disabled |")
        )
        self.assertIn("Bash", disabled_line)
        self.assertIn("Config", disabled_line)
        self.assertIn("hook_runtime", disabled_line)

    def test_docs_contain_manifest_generated_status_table(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rendered = render_capability_status_markdown()
        readme = (root / "README.md").read_text(encoding="utf-8")
        feature_list = (root / "FEATURE_LIST.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count(rendered), 2)
        self.assertEqual(feature_list.count(rendered), 1)
class SkillReconcilerTests(unittest.TestCase):
    def test_preserved_review_candidate_is_not_a_runtime_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trust = root / ".clawd" / "development-pack"
            candidates = root / ".clawd" / "review-candidates"
            runtime = root / ".clawd" / "skills"
            trust.mkdir(parents=True)
            candidate = candidates / "reviewer"
            runtime_skill = runtime / "reviewer"
            candidate.mkdir(parents=True)
            runtime_skill.mkdir(parents=True)
            skill_text = "---\ndescription: reviewer\n---\nReview safely.\n"
            (candidate / "SKILL.md").write_text(skill_text, encoding="utf-8")
            (runtime_skill / "SKILL.md").write_text(skill_text, encoding="utf-8")
            registry = {
                "schema_version": 1,
                "skills": {
                    "reviewer": {
                        "review_status": "approved",
                        "activation_status": "active",
                        "artifact_path": str(runtime_skill),
                        "integrity_hash": compute_artifact_hash(runtime_skill),
                    }
                },
            }
            (trust / "skill-registry.json").write_text(json.dumps(registry), encoding="utf-8")
            (trust / "skill-audit.jsonl").write_text("", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {
                    "CLAWD_SKILLS_DIR": str(runtime),
                    "CLAWD_SKILL_TRUST_DIR": str(trust),
                    "CLAUDE_SKILLS_DIR": "",
                },
                clear=False,
            ):
                report = reconcile_skills(trust_dir=trust)

            self.assertTrue(report["records"]["reviewer"]["loader_reachable"])
            self.assertNotIn(
                {"code": "duplicate_skill_artifacts", "subject": "reviewer"},
                report["issues"],
            )

    def test_reconciler_is_read_only_and_detects_unreachable_active_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trust = root / ".clawd" / "development-pack"
            candidates = root / ".clawd" / "review-candidates" / "reviewer"
            runtime = root / ".clawd" / "skills"
            trust.mkdir(parents=True)
            candidates.mkdir(parents=True)
            runtime.mkdir(parents=True)
            (candidates / "SKILL.md").write_text(
                "---\ndescription: test reviewer\n---\nReview safely.\n",
                encoding="utf-8",
            )
            registry = {
                "schema_version": 1,
                "skills": {
                    "reviewer": {
                        "review_status": "approved",
                        "activation_status": "active",
                        "artifact_path": str(candidates),
                        "integrity_hash": compute_artifact_hash(candidates),
                    }
                },
            }
            registry_path = trust / "skill-registry.json"
            audit_path = trust / "skill-audit.jsonl"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            audit_path.write_text("", encoding="utf-8")
            before_registry = registry_path.read_bytes()
            before_audit = audit_path.read_bytes()
            with patch.dict(
                "os.environ",
                {
                    "CLAWD_SKILLS_DIR": str(runtime),
                    "CLAWD_SKILL_TRUST_DIR": str(trust),
                },
                clear=False,
            ):
                report = reconcile_skills(trust_dir=trust)

            self.assertIn(
                {"code": "active_skill_not_loader_reachable", "subject": "reviewer"},
                report["issues"],
            )
            self.assertTrue(report["audit_chain"]["valid"])
            self.assertEqual(registry_path.read_bytes(), before_registry)
            self.assertEqual(audit_path.read_bytes(), before_audit)


if __name__ == "__main__":
    unittest.main()
