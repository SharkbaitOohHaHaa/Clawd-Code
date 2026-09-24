from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolPermissionError
from src.tool_system.permission_handler import PermissionBehavior, PermissionResult
from src.tool_system.permissions import SensitivePathBehavior, sensitive_path_decision
from src.tool_system.protocol import ToolCall, ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec


class _MissingPolicyTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(name="MissingPolicy", description="", input_schema={"type": "object"})

    def run(self, tool_input, context) -> ToolResult:
        return ToolResult(name="MissingPolicy", output={})


class _CheckedWithoutCheckerTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="CheckedWithoutChecker",
            description="",
            input_schema={"type": "object"},
            permission_policy="checked",
        )

    def run(self, tool_input, context) -> ToolResult:
        return ToolResult(name="CheckedWithoutChecker", output={})


class _AllowWithCheckerTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="AllowWithChecker",
            description="",
            input_schema={"type": "object"},
            permission_policy="allow",
        )

    def check_permissions(self, tool_input, context) -> PermissionResult:
        return PermissionResult.allow()

    def run(self, tool_input, context) -> ToolResult:
        return ToolResult(name="AllowWithChecker", output={})


class PermissionContractTests(unittest.TestCase):
    def test_registration_fails_closed_without_explicit_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "must declare permission_policy"):
            ToolRegistry([_MissingPolicyTool()])

    def test_checked_policy_requires_checker(self) -> None:
        with self.assertRaisesRegex(ValueError, "without check_permissions"):
            ToolRegistry([_CheckedWithoutCheckerTool()])

    def test_non_checked_policy_rejects_hidden_checker(self) -> None:
        with self.assertRaisesRegex(ValueError, "implements check_permissions"):
            ToolRegistry([_AllowWithCheckerTool()])

    def test_default_registry_has_explicit_policy_for_every_tool(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        self.assertEqual(len(registry.list_specs()), 45)
        self.assertTrue(all(spec.permission_policy for spec in registry.list_specs()))


class SensitivePathSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.registry = build_default_registry(include_user_tools=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builtin_sensitive_path_classification(self) -> None:
        for name in (
            ".env",
            ".env.local",
            "credentials.json",
            "credentials-prod.json",
            "service-account-prod.json",
            "service_account_dev.json",
            "token.json",
            "token-cache.json",
            "oauth-access-token.json",
        ):
            with self.subTest(name=name):
                decision = sensitive_path_decision(self.root / name, operation="read")
                self.assertEqual(decision.behavior, SensitivePathBehavior.ASK)

        for name in (".env.example", ".env.sample", ".env.template"):
            with self.subTest(name=name):
                decision = sensitive_path_decision(self.root / name, operation="read")
                self.assertEqual(decision.behavior, SensitivePathBehavior.ALLOW)

        for name in ("id_rsa", "id_ed25519", "server.pem", "private.key", "cert.p12", "cert.pfx"):
            with self.subTest(name=name):
                decision = sensitive_path_decision(self.root / name, operation="read")
                self.assertEqual(decision.behavior, SensitivePathBehavior.DENY)

    def test_read_env_requires_confirmation_and_never_leaks_on_denial(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("SECRET=not-a-real-secret\n", encoding="utf-8")

        denied = self.registry.dispatch(
            ToolCall(name="Read", input={"file_path": str(env_file)}),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertNotIn("not-a-real-secret", str(denied.output))

        calls: list[str] = []
        self.ctx.permission_handler = lambda name, message, suggestion: (
            calls.append(name) is None,
            False,
        )
        allowed = self.registry.dispatch(
            ToolCall(name="Read", input={"file_path": str(env_file), "limit": 10}),
            self.ctx,
        )
        self.assertFalse(allowed.is_error)
        self.assertEqual(calls, ["Read"])
        self.assertIn("not-a-real-secret", allowed.output["file"]["content"])

    def test_private_key_is_denied_without_prompt(self) -> None:
        key_file = self.root / "id_rsa"
        key_file.write_text("dummy-key-material", encoding="utf-8")
        calls = 0

        def handler(name: str, message: str, suggestion: str | None):
            nonlocal calls
            calls += 1
            return True, False

        self.ctx.permission_handler = handler
        result = self.registry.dispatch(
            ToolCall(name="Read", input={"file_path": str(key_file)}),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        self.assertEqual(calls, 0)
        self.assertNotIn("dummy-key-material", str(result.output))

    def test_write_env_asks_but_template_does_not(self) -> None:
        env_file = self.root / ".env"
        denied = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(env_file), "content": "A=B"}),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertFalse(env_file.exists())

        self.ctx.permission_handler = lambda name, message, suggestion: (True, False)
        allowed = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(env_file), "content": "A=B"}),
            self.ctx,
        )
        self.assertFalse(allowed.is_error)
        self.assertEqual(env_file.read_text(encoding="utf-8"), "A=B")

        template = self.root / ".env.example"
        self.ctx.permission_handler = None
        template_result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(template), "content": "A=example"}),
            self.ctx,
        )
        self.assertFalse(template_result.is_error)

    def test_write_private_key_material_is_denied(self) -> None:
        key_file = self.root / "server.pem"
        self.ctx.permission_handler = lambda name, message, suggestion: (True, False)
        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(key_file), "content": "dummy"}),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        self.assertFalse(key_file.exists())

    def test_grep_filters_sensitive_files_before_reading_contents(self) -> None:
        normal = self.root / "app.txt"
        env_file = self.root / ".env"
        key_file = self.root / "id_rsa"
        normal.write_text("MAGIC normal\n", encoding="utf-8")
        env_file.write_text("MAGIC env-secret\n", encoding="utf-8")
        key_file.write_text("MAGIC key-secret\n", encoding="utf-8")

        result = self.registry.dispatch(
            ToolCall(
                name="Grep",
                input={"pattern": "MAGIC", "path": str(self.root), "output_mode": "content", "-n": True},
            ),
            self.ctx,
        )
        self.assertFalse(result.is_error)
        text = result.output["content"]
        self.assertIn(str(normal), text)
        self.assertNotIn(str(env_file), text)
        self.assertNotIn(str(key_file), text)
        self.assertNotIn("env-secret", text)
        self.assertNotIn("key-secret", text)

    def test_glob_filters_sensitive_names_but_keeps_env_template(self) -> None:
        (self.root / ".env").write_text("A=B", encoding="utf-8")
        template = self.root / ".env.example"
        template.write_text("A=example", encoding="utf-8")
        result = self.registry.dispatch(
            ToolCall(name="Glob", input={"pattern": ".env*", "path": str(self.root)}),
            self.ctx,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["filenames"], [str(template)])

    def test_sensitive_attachment_requires_confirmation(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("A=B", encoding="utf-8")
        call = ToolCall(
            name="SendUserMessage",
            input={"message": "attached", "status": "normal", "attachments": [str(env_file)]},
        )
        denied = self.registry.dispatch(call, self.ctx)
        self.assertTrue(denied.is_error)
        self.assertEqual(self.ctx.outbox, [])

        self.ctx.permission_handler = lambda name, message, suggestion: (True, False)
        allowed = self.registry.dispatch(call, self.ctx)
        self.assertFalse(allowed.is_error)
        self.assertEqual(len(self.ctx.outbox), 1)

    def test_exit_plan_mode_write_uses_checked_permission(self) -> None:
        enter = self.registry.dispatch(ToolCall(name="EnterPlanMode", input={}), self.ctx)
        self.assertFalse(enter.is_error)
        denied = self.registry.dispatch(
            ToolCall(name="ExitPlanMode", input={"plan": "# Plan"}),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertTrue(self.ctx.plan_mode)
        self.assertFalse((self.root / ".clawd" / "plan.md").exists())

        self.ctx.permission_handler = lambda name, message, suggestion: (True, False)
        allowed = self.registry.dispatch(
            ToolCall(name="ExitPlanMode", input={"plan": "# Plan"}),
            self.ctx,
        )
        self.assertFalse(allowed.is_error)
        self.assertFalse(self.ctx.plan_mode)
        self.assertTrue((self.root / ".clawd" / "plan.md").exists())

    def test_exit_plan_mode_sensitive_custom_path_requires_confirmation(self) -> None:
        self.ctx.permission_context.allow_docs = True
        self.registry.dispatch(ToolCall(name="EnterPlanMode", input={}), self.ctx)
        target = self.root / ".env"
        denied = self.registry.dispatch(
            ToolCall(
                name="ExitPlanMode",
                input={"plan": "A=B", "planFilePath": str(target)},
            ),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertFalse(target.exists())
        self.assertTrue(self.ctx.plan_mode)

        self.ctx.permission_handler = lambda name, message, suggestion: (True, False)
        allowed = self.registry.dispatch(
            ToolCall(
                name="ExitPlanMode",
                input={"plan": "A=B", "planFilePath": str(target)},
            ),
            self.ctx,
        )
        self.assertFalse(allowed.is_error)
        self.assertEqual(target.read_text(encoding="utf-8"), "A=B")

    def test_recursive_tools_filter_symlink_escape_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as outside_tmp:
            outside = Path(outside_tmp) / "outside.txt"
            outside.write_text("MAGIC outside-secret", encoding="utf-8")
            link = self.root / "linked.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not supported in this environment")

            grep = self.registry.dispatch(
                ToolCall(
                    name="Grep",
                    input={"pattern": "MAGIC", "path": str(self.root), "output_mode": "content"},
                ),
                self.ctx,
            )
            self.assertFalse(grep.is_error)
            self.assertNotIn("outside-secret", grep.output["content"])
            self.assertNotIn(str(link), grep.output["content"])

            globbed = self.registry.dispatch(
                ToolCall(name="Glob", input={"pattern": "linked.txt", "path": str(self.root)}),
                self.ctx,
            )
            self.assertFalse(globbed.is_error)
            self.assertEqual(globbed.output["filenames"], [])

    def test_recursive_tools_revalidate_default_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as outside_tmp:
            self.ctx.cwd = Path(outside_tmp).resolve()
            with self.assertRaises(Exception):
                self.registry.dispatch(
                    ToolCall(name="Grep", input={"pattern": "x"}),
                    self.ctx,
                )
            with self.assertRaises(Exception):
                self.registry.dispatch(
                    ToolCall(name="Glob", input={"pattern": "**/*"}),
                    self.ctx,
                )

    def test_qwen_local_sensitive_source_uses_sensitive_path_floor(self) -> None:
        qwen = self.registry.get("QwenMediaAnalyze")
        self.assertIsNotNone(qwen)

        key_file = self.root / "private.pem"
        key_file.write_text("dummy", encoding="utf-8")
        denied = qwen.check_permissions(  # type: ignore[union-attr]
            {"source": str(key_file), "media_type": "image"},
            self.ctx,
        )
        self.assertEqual(denied.behavior, PermissionBehavior.DENY)

        env_file = self.root / ".env"
        env_file.write_text("dummy", encoding="utf-8")
        asked = qwen.check_permissions(  # type: ignore[union-attr]
            {"source": str(env_file), "media_type": "image"},
            self.ctx,
        )
        self.assertEqual(asked.behavior, PermissionBehavior.ASK)
        self.assertIn("secret-bearing", asked.message or "")
        self.assertIn("sent to Qwen", asked.message or "")

    def test_team_metadata_write_in_locked_workspace_requires_confirmation(self) -> None:
        locked_root = self.root / "LOCKED" / "baseline"
        locked_root.mkdir(parents=True)
        ctx = ToolContext(workspace_root=locked_root)
        registry = build_default_registry(include_user_tools=False)
        target = locked_root / ".clawd" / "team.json"

        denied = registry.dispatch(
            ToolCall(name="TeamCreate", input={"team_name": "audit"}),
            ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertFalse(target.exists())

        prompts: list[str] = []
        ctx.permission_handler = lambda name, message, suggestion: (
            prompts.append(name) is None,
            False,
        )
        allowed = registry.dispatch(
            ToolCall(name="TeamCreate", input={"team_name": "audit"}),
            ctx,
        )
        self.assertFalse(allowed.is_error)
        self.assertTrue(target.exists())
        self.assertEqual(prompts, ["TeamCreate"])

        ctx.permission_handler = None
        denied_delete = registry.dispatch(ToolCall(name="TeamDelete", input={}), ctx)
        self.assertTrue(denied_delete.is_error)
        self.assertTrue(target.exists())

    def test_team_metadata_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as outside_tmp:
            outside = Path(outside_tmp).resolve()
            link = self.root / ".clawd"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable on this platform")

            result = self.registry.dispatch(
                ToolCall(name="TeamCreate", input={"team_name": "audit"}),
                self.ctx,
            )
            self.assertTrue(result.is_error)
            self.assertFalse((outside / "team.json").exists())

    def test_skill_allowlist_applies_to_delegated_agent_calls(self) -> None:
        self.ctx.restrict_tool_allowlist(["Agent", "Read"])
        target = self.root / "blocked.txt"
        with self.assertRaises(ToolPermissionError):
            self.registry.dispatch(
                ToolCall(
                    name="Agent",
                    input={
                        "calls": [
                            {
                                "name": "Write",
                                "input": {
                                    "file_path": str(target),
                                    "content": "blocked",
                                },
                            }
                        ]
                    },
                ),
                self.ctx,
            )
        self.assertFalse(target.exists())

    def test_agent_delegates_child_permission_to_registry(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("A=B", encoding="utf-8")
        result = self.registry.dispatch(
            ToolCall(
                name="Agent",
                input={
                    "calls": [{"name": "Read", "input": {"file_path": str(env_file)}}],
                    "stop_on_error": True,
                },
            ),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        self.assertTrue(result.output["results"][0]["is_error"])


if __name__ == "__main__":
    unittest.main()
