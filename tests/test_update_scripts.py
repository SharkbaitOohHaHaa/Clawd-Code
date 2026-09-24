from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "windows"


class UpdateScriptContractTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (OPS / name).read_text(encoding="utf-8")

    def test_startup_checker_is_passive_and_bounded(self) -> None:
        checker = self._read("Check-ClawdUpdate.ps1")
        lower = checker.lower()

        self.assertIn("ls-remote https://github.com/", lower)
        self.assertIn("$timeoutseconds = 3", lower)
        self.assertIn("write-skipped 'offline'", lower)
        self.assertIn("write-skipped 'git unavailable'", lower)
        self.assertIn("upstream-review-state.json", lower)

        for forbidden in (
            " git fetch ",
            " git pull ",
            " git merge ",
            " git checkout ",
            "copy-item",
            "move-item",
            "set-content",
            "pip install",
            "uv sync",
        ):
            self.assertNotIn(forbidden, lower)
    def test_launcher_runs_checker_asynchronously(self) -> None:
        launcher = self._read("Start Clawd Codex.cmd")
        lower = launcher.lower()
        async_line = (
            'start "" /b powershell -noprofile -executionpolicy bypass '
            '-file "%clawd_root%clawd codex home\\.clawd\\check-clawdupdate.ps1"'
        )
        self.assertIn(async_line, lower)
        self.assertLess(
            lower.index("check-clawdupdate.ps1"),
            lower.index("clawd.exe"),
        )

    def test_review_command_classifies_without_live_update_actions(self) -> None:
        review = self._read("Review-ClawdUpdate.ps1")
        lower = review.lower()

        self.assertIn("git -c $referencepath fetch origin", lower)
        self.assertIn("files we deleted", lower)
        self.assertIn("files we modified", lower)
        self.assertIn("files we haven't touched", lower)
        self.assertIn("$markreviewed", lower)
        self.assertIn("write-reviewstate", lower)
        self.assertIn("review only: no live clawd files", lower)

        for forbidden in (
            " git pull ",
            " git merge ",
            " git checkout ",
            "pip install",
            "uv sync",
        ):
            self.assertNotIn(forbidden, lower)
    def test_review_state_update_is_explicit_and_state_only(self) -> None:
        review = self._read("Review-ClawdUpdate.ps1")
        self.assertIn("if ($MarkReviewed)", review)
        self.assertIn(
            "Move-Item -LiteralPath $temp -Destination $statePath -Force",
            review,
        )
        self.assertNotIn("Set-Content -LiteralPath $liveRoot", review)

    def test_runtime_wrappers_are_path_relative(self) -> None:
        launcher = self._read("Start Clawd Codex.cmd")
        review_wrapper = self._read("Review Clawd Updates.cmd")
        self.assertIn('set "CLAWD_ROOT=%~dp0"', launcher)
        self.assertIn('set "CLAWD_ROOT=%~dp0"', review_wrapper)
        self.assertNotIn("H:\\", launcher)
        self.assertNotIn("H:\\", review_wrapper)


if __name__ == "__main__":
    unittest.main()
