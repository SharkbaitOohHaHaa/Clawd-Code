from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.context_system import build_context_prompt
from src.context_system.git_context import collect_git_context
from src.context_system.project_map import build_project_map
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


class TestContextSystem(unittest.TestCase):
    def test_build_context_prompt_includes_workspace_and_claude_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("Project rule: always add tests.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

            prompt = build_context_prompt(root)

            self.assertIn("## Runtime Context", prompt)
            self.assertIn("## Project Instructions", prompt)
            self.assertIn("Project rule: always add tests.", prompt)
            self.assertIn("README.md", prompt)
            self.assertIn("pyproject.toml", prompt)
            self.assertIn("## Project Map", prompt)
            self.assertIn("src/app.py", prompt)
            self.assertIn("tests/test_app.py", prompt)
            self.assertIn("## Project Overview", prompt)
            self.assertIn("# Demo", prompt)

    def test_runtime_context_omits_sensitive_top_level_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "normal.json").write_text("{}", encoding="utf-8")
            (root / "credentials-prod.json").write_text("SECRET", encoding="utf-8")
            (root / "token.json").write_text("SECRET", encoding="utf-8")
            (root / ".env").write_text("SECRET=1", encoding="utf-8")

            prompt = build_context_prompt(root)

            self.assertIn("normal.json", prompt)
            self.assertNotIn("credentials-prod.json", prompt)
            self.assertNotIn("token.json", prompt)
            self.assertNotIn(".env", prompt)
            self.assertNotIn("SECRET", prompt)

    def test_project_map_is_bounded_deterministic_and_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "node_modules" / "dependency").mkdir(parents=True)
            (root / ".hidden").mkdir()
            (root / "src" / "pkg" / "module.py").write_text("SECRET_SOURCE_CONTENT", encoding="utf-8")
            (root / "src" / "app.py").write_text("print('app')", encoding="utf-8")
            (root / "tests" / "test_app.py").write_text("def test_ok(): pass", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "credentials-prod.json").write_text("SECRET_CREDENTIAL", encoding="utf-8")
            (root / ".hidden" / "hidden.py").write_text("HIDDEN", encoding="utf-8")
            (root / "node_modules" / "dependency" / "index.py").write_text("VENDOR", encoding="utf-8")
            (root / "image.png").write_bytes(b"not-context")

            first = build_project_map(root)
            second = build_project_map(root)

            self.assertEqual(first, second)
            self.assertIn("src/app.py", first)
            self.assertIn("src/pkg/module.py", first)
            self.assertIn("tests/test_app.py", first)
            self.assertIn("pyproject.toml", first)
            self.assertNotIn("credentials-prod.json", first)
            self.assertFalse(any("node_modules" in item for item in first))
            self.assertFalse(any(".hidden" in item for item in first))
            self.assertNotIn("image.png", first)
            self.assertFalse(any("SECRET_SOURCE_CONTENT" in item for item in first))

    def test_project_map_has_hard_entry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            for index in range(10):
                (root / "src" / f"module_{index:02d}.py").write_text("pass\n", encoding="utf-8")

            project_map = build_project_map(root, max_entries=4)

            self.assertEqual(len(project_map), 5)
            self.assertEqual(project_map[-1], "... [project map truncated]")

    def test_project_map_does_not_follow_external_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            (outside / "outside.py").write_text("OUTSIDE_MAP_SECRET", encoding="utf-8")
            link = root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable on this platform")

            project_map = build_project_map(root)

            self.assertFalse(any("linked" in item for item in project_map))
            self.assertFalse(any("outside.py" in item for item in project_map))

    def test_project_overview_injects_bounded_readme_and_declared_entry_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme_head = "# Demo\n" + ("R" * 3_500)
            (root / "README.md").write_text(readme_head, encoding="utf-8")
            (root / "pyproject.toml").write_text(
                "[project]\nname='demo'\n[project.scripts]\ndemo='src.launch:main'\n",
                encoding="utf-8",
            )
            (root / "src").mkdir()
            (root / "src" / "launch.py").write_text(
                "def main():\n    return 'ENTRY_MARKER'\n",
                encoding="utf-8",
            )

            prompt = build_context_prompt(root)

            self.assertIn("### README.md excerpt", prompt)
            self.assertIn("...[truncated]", prompt)
            self.assertNotIn("R" * 3_100, prompt)
            self.assertIn("### Python entry file: ./src/launch.py", prompt)
            self.assertIn("ENTRY_MARKER", prompt)
            self.assertIn("reference context only", prompt)

    def test_project_overview_detects_script_without_tomllib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname='demo'\n[project.scripts]\ndemo='pkg.launch:main'\n",
                encoding="utf-8",
            )
            (root / "pkg").mkdir()
            (root / "pkg" / "launch.py").write_text("FALLBACK_ENTRY = True\n", encoding="utf-8")

            with patch("src.context_system.builder.tomllib", None):
                prompt = build_context_prompt(root)

            self.assertIn("### Python entry file: ./pkg/launch.py", prompt)
            self.assertIn("FALLBACK_ENTRY = True", prompt)

    def test_project_overview_ignores_external_symlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            (outside / "README.md").write_text("OUTSIDE README SECRET", encoding="utf-8")
            (outside / "launch.py").write_text("OUTSIDE ENTRY SECRET", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                "[project]\nname='demo'\n[project.scripts]\ndemo='src.launch:main'\n",
                encoding="utf-8",
            )
            (root / "src").mkdir()
            try:
                (root / "README.md").symlink_to(outside / "README.md")
                (root / "src" / "launch.py").symlink_to(outside / "launch.py")
            except (OSError, NotImplementedError):
                self.skipTest("file symlinks are unavailable on this platform")

            prompt = build_context_prompt(root)

            self.assertNotIn("OUTSIDE README SECRET", prompt)
            self.assertNotIn("OUTSIDE ENTRY SECRET", prompt)

    def test_project_claude_md_symlink_outside_workspace_is_ignored(self) -> None:
        from src.context_system.claude_md import load_claude_md_context

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            (outside / "CLAUDE.md").write_text("OUTSIDE SECRET", encoding="utf-8")
            link = root / ".clawd"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable on this platform")

            ctx = load_claude_md_context(root)
            combined = "\n".join(f.content for f in ctx.files)
            self.assertNotIn("OUTSIDE SECRET", combined)

    def test_context_prompt_uses_linked_worktree_view(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            def git(*args: str, cwd: Path = root) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )

            self.assertEqual(git("init", "-q").returncode, 0)
            self.assertEqual(git("config", "user.email", "clawd-test@example.invalid").returncode, 0)
            self.assertEqual(git("config", "user.name", "Clawd Test").returncode, 0)
            (root / "CLAUDE.md").write_text("Original instruction.\n", encoding="utf-8")
            (root / "app.py").write_text("value = 1\n", encoding="utf-8")
            self.assertEqual(git("add", "CLAUDE.md", "app.py").returncode, 0)
            self.assertEqual(git("commit", "-q", "-m", "initial").returncode, 0)

            worktree = root / ".git" / "clawd-worktrees" / "demo"
            worktree.parent.mkdir(parents=True)
            self.assertEqual(
                git("worktree", "add", "-q", "-b", "clawd/demo", str(worktree), "HEAD").returncode,
                0,
            )
            (worktree / "CLAUDE.md").write_text("Worktree instruction.\n", encoding="utf-8")
            (worktree / "worktree_only.py").write_text("value = 2\n", encoding="utf-8")

            prompt = build_context_prompt(root, cwd=worktree)

            self.assertIn(f"- Workspace root: {worktree}", prompt)
            self.assertIn("- Current branch: clawd/demo", prompt)
            self.assertIn("Worktree instruction.", prompt)
            self.assertNotIn("Original instruction.", prompt)
            self.assertIn("worktree_only.py", "\n".join(p.name for p in worktree.iterdir()))

    def test_collect_git_context_handles_non_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = collect_git_context(tmp)
            self.assertFalse(ctx.available)

    def test_collect_git_context_rejects_repo_root_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            workspace = parent / "workspace"
            workspace.mkdir()

            fake_top = subprocess.CompletedProcess(
                args=["git"],
                returncode=0,
                stdout=str(parent),
                stderr="",
            )

            with patch(
                "src.context_system.git_context._run_git",
                return_value=fake_top,
            ):
                ctx = collect_git_context(workspace)

            self.assertFalse(ctx.available)
            self.assertEqual(
                ctx.error,
                "git repository root is outside workspace",
            )

    def test_run_git_sanitizes_environment_and_disables_fsmonitor(self) -> None:
        from src.context_system.git_context import _run_git

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {
                    "GIT_DIR": r"C:\evil",
                    "GIT_WORK_TREE": r"C:\evil-tree",
                    "CLAWD_TEST_KEEP": "yes",
                },
                clear=False,
            ):
                with patch(
                    "src.context_system.git_context.subprocess.run"
                ) as mock_run:
                    mock_run.return_value = subprocess.CompletedProcess(
                        args=["git"],
                        returncode=0,
                        stdout="",
                        stderr="",
                    )

                    _run_git(Path(tmp), "status", "--short")

            args, kwargs = mock_run.call_args

            self.assertEqual(
                args[0],
                [
                    "git",
                    "--no-pager",
                    "-c",
                    "core.fsmonitor=false",
                    "status",
                    "--short",
                ],
            )
            self.assertNotIn("GIT_DIR", kwargs["env"])
            self.assertNotIn("GIT_WORK_TREE", kwargs["env"])
            self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(kwargs["env"]["GIT_PAGER"], "cat")
            self.assertEqual(kwargs["env"]["CLAWD_TEST_KEEP"], "yes")
            self.assertEqual(kwargs["timeout"], 5)
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)

    def test_agent_loop_injects_context_prompt_for_non_anthropic(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("Follow the CLAUDE instructions.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")

            ctx = ToolContext(workspace_root=root)

            conversation = Conversation()
            conversation.add_user_message("hello")

            provider = MagicMock()
            provider.chat.return_value = ChatResponse(
                content="ok",
                model="test",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            )

            out = run_agent_loop(conversation, provider, registry, ctx, verbose=False)
            self.assertEqual(out.response_text, "ok")
            system_message = provider.chat.call_args.args[0][0]
            self.assertEqual(system_message["role"], "system")
            self.assertIn("## Runtime Context", system_message["content"])
            self.assertIn("## Project Instructions", system_message["content"])
            self.assertIn("Follow the CLAUDE instructions.", system_message["content"])


if __name__ == "__main__":
    unittest.main()
