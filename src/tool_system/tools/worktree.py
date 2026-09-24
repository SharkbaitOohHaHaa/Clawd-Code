from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec


_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_BRANCH_PREFIX = "clawd/"
_GIT_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class _WorktreePlan:
    workspace: Path
    root: Path
    branch: str
    base_commit: str
    original_dirty: bool


def _git_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
        }
    )
    return env


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "--no-pager", "-c", "core.fsmonitor=false", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolPermissionError("Git is unavailable for worktree operations") from exc


def _git_text(cwd: Path, *args: str) -> str:
    result = _run_git(cwd, *args)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 600:
            detail = detail[-600:]
        message = "Git command failed"
        if detail:
            message += f": {detail}"
        raise ToolPermissionError(message)
    return (result.stdout or "").strip()


def git_worktree_runtime_available() -> bool:
    git = shutil.which("git")
    if not git:
        return False
    try:
        result = subprocess.run(
            [git, "--version"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=5,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and (result.stdout or "").lower().startswith("git version ")


def _validate_name(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolInputError("name must be a non-empty string")
    name = raw.strip()
    if (
        len(name) > 64
        or not _NAME_RE.fullmatch(name)
        or any(part in {".", ".."} for part in name.split("/"))
    ):
        raise ToolInputError("invalid worktree name")
    return name


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _plan_enter(tool_input: dict[str, Any], context: ToolContext) -> _WorktreePlan:
    if context.worktree_root is not None:
        raise ToolPermissionError("already in a worktree session")

    name = _validate_name(tool_input.get("name"))
    workspace = context.workspace_root.resolve()
    git_dir = workspace / ".git"
    if not git_dir.is_dir():
        raise ToolPermissionError(
            "EnterWorktree requires the workspace root to be the main Git worktree"
        )

    top = _git_text(workspace, "rev-parse", "--show-toplevel")
    repo_root = Path(top).resolve()
    if not _same_path(repo_root, workspace):
        raise ToolPermissionError(
            "EnterWorktree requires the workspace root to equal the Git repository root"
        )

    if _git_text(workspace, "rev-parse", "--is-bare-repository").lower() != "false":
        raise ToolPermissionError("EnterWorktree is unavailable for bare repositories")

    base_commit = _git_text(workspace, "rev-parse", "--verify", "HEAD")
    branch = f"{_BRANCH_PREFIX}{name}"
    branch_check = _run_git(workspace, "check-ref-format", "--branch", branch)
    if branch_check.returncode != 0:
        raise ToolInputError("worktree name does not produce a valid Git branch")

    branch_exists = _run_git(
        workspace,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{branch}",
    )
    if branch_exists.returncode == 0:
        raise ToolPermissionError(f"worktree branch already exists: {branch}")
    if branch_exists.returncode not in {0, 1}:
        raise ToolPermissionError("unable to verify whether the worktree branch already exists")

    root = git_dir / "clawd-worktrees" / Path(*name.split("/"))
    context.permission_context.ensure_path_allowed(root)
    if root.exists():
        raise ToolPermissionError(f"worktree path already exists: {root}")

    status = _run_git(workspace, "status", "--porcelain")
    if status.returncode != 0:
        raise ToolPermissionError("unable to inspect Git status before creating worktree")

    return _WorktreePlan(
        workspace=workspace,
        root=root,
        branch=branch,
        base_commit=base_commit,
        original_dirty=bool((status.stdout or "").strip()),
    )


class EnterWorktreeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="EnterWorktree",
            permission_policy="checked",
            description=(
                "Create a real linked Git worktree on a new clawd/<name> branch "
                "from the current committed HEAD and switch this session into it."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    }
                },
                "required": ["name"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        try:
            plan = _plan_enter(tool_input, context)
        except (ToolInputError, ToolPermissionError) as exc:
            return PermissionResult.deny(str(exc))

        dirty_note = ""
        if plan.original_dirty:
            dirty_note = (
                " The original workspace has uncommitted changes; they will remain "
                "there and will NOT be copied into the new worktree."
            )
        return PermissionResult.ask(
            message=(
                f"Create Git worktree branch '{plan.branch}' at '{plan.root}' "
                f"from commit {plan.base_commit[:12]}?{dirty_note}"
            ),
            suggestion="require-explicit-yes",
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        plan = _plan_enter(tool_input, context)
        plan.root.parent.mkdir(parents=True, exist_ok=True)

        created = _run_git(
            plan.workspace,
            "worktree",
            "add",
            "--quiet",
            "-b",
            plan.branch,
            str(plan.root),
            plan.base_commit,
        )
        if created.returncode != 0:
            detail = (created.stderr or created.stdout or "").strip()
            return ToolResult(
                name="EnterWorktree",
                output={"error": detail or "git worktree add failed"},
                is_error=True,
            )

        try:
            actual_root = Path(_git_text(plan.root, "rev-parse", "--show-toplevel")).resolve()
            actual_branch = _git_text(plan.root, "branch", "--show-current")
            actual_head = _git_text(plan.root, "rev-parse", "HEAD")
        except ToolPermissionError as exc:
            return ToolResult(
                name="EnterWorktree",
                output={
                    "error": (
                        "Git worktree was created but verification failed; "
                        f"it was left in place for manual review: {exc}"
                    )
                },
                is_error=True,
            )

        if (
            not _same_path(actual_root, plan.root)
            or actual_branch != plan.branch
            or actual_head != plan.base_commit
        ):
            return ToolResult(
                name="EnterWorktree",
                output={
                    "error": (
                        "Git worktree was created but did not match the requested "
                        "path/branch/base commit; it was left in place for manual review"
                    )
                },
                is_error=True,
            )

        context.worktree_root = plan.root
        context.cwd = plan.root
        context.lsp_client = None
        return ToolResult(
            name="EnterWorktree",
            output={
                "worktreePath": str(plan.root),
                "worktreeBranch": plan.branch,
                "baseCommit": plan.base_commit,
                "originalWorkspaceDirty": plan.original_dirty,
                "message": (
                    f"Created linked Git worktree on {plan.branch} at {plan.root}. "
                    "The session is now working there. Original uncommitted changes "
                    "were not copied."
                ),
            },
        )


class ExitWorktreeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ExitWorktree",
            permission_policy="allow",
            description=(
                "Leave the active worktree session and return to the original workspace. "
                "The linked worktree and branch are preserved."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.worktree_root is None:
            raise ToolPermissionError("not in a worktree session")

        old_root = context.worktree_root
        branch: str | None = None
        if old_root.exists():
            try:
                result = _run_git(old_root, "branch", "--show-current")
            except ToolPermissionError:
                result = None
            if result is not None and result.returncode == 0:
                branch = (result.stdout or "").strip() or None

        context.worktree_root = None
        context.cwd = context.workspace_root
        context.lsp_client = None
        return ToolResult(
            name="ExitWorktree",
            output={
                "worktreePath": str(old_root),
                "worktreeBranch": branch,
                "preserved": True,
                "message": (
                    f"Exited worktree session and returned to {context.workspace_root}. "
                    "The linked worktree and branch were preserved; nothing was deleted."
                ),
            },
        )

