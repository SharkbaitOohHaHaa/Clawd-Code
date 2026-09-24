from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "windows"


class UpdateScriptContractTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (OPS / name).read_text(encoding="utf-8")

    def test_startup_checker_is_passive_bounded_and_tree_scoped(self) -> None:
        checker = self._read("Check-ClawdUpdate.ps1")
        lower = checker.lower()

        self.assertIn("ls-remote https://github.com/", lower)
        self.assertIn("$timeoutseconds = 3", lower)
        self.assertIn("write-skipped 'offline'", lower)
        self.assertIn("write-skipped 'git unavailable'", lower)
        self.assertIn("upstream-review-state.json", lower)
        self.assertIn("clawdupdatekilljob", lower)
        self.assertIn("0x00002000", lower)
        self.assertNotRegex(lower, r"(?m)^\s*&?\s*git\b[^\r\n]*\bfetch\b")

        for forbidden in ("git pull", "git merge", "git checkout", "pip install", "uv sync"):
            self.assertNotIn(forbidden, lower)

    def test_launcher_is_async_and_ignores_commented_env_entries(self) -> None:
        launcher = self._read("Start Clawd Codex.cmd")
        lower = launcher.lower()
        async_line = (
            'start "" /b powershell -noprofile -executionpolicy bypass '
            '-file "%clawd_root%clawd codex home\\.clawd\\check-clawdupdate.ps1"'
        )
        self.assertIn(async_line, lower)
        self.assertLess(lower.index("check-clawdupdate.ps1"), lower.index("clawd.exe"))
        self.assertIn('for /f "usebackq eol=# tokens=1,* delims=="', lower)
        self.assertNotIn("%%a:~0,1", lower)

    def test_review_preserves_status_unicode_and_rename_information(self) -> None:
        review = self._read("Review-ClawdUpdate.ps1")
        lower = review.lower()

        self.assertIn("-c core.quotepath=false diff --name-status -m", lower)
        self.assertIn("resolve-livepath", lower)
        self.assertIn("files we deleted", lower)
        self.assertIn("files we modified", lower)
        self.assertIn("files we haven't touched", lower)
        self.assertIn("status=$status", lower)
        self.assertIn("baselinepath", lower)
        self.assertIn("currentpath", lower)
        self.assertNotIn("--no-renames", lower)

        for forbidden in ("git pull", "git merge", "git checkout", "pip install", "uv sync"):
            self.assertNotIn(forbidden, lower)

    def test_mark_reviewed_uses_exact_explicit_sha(self) -> None:
        review = self._read("Review-ClawdUpdate.ps1")
        self.assertIn("[string]$MarkReviewed", review)
        self.assertIn("$target = if ($MarkReviewed)", review)
        self.assertIn("Write-ReviewState $state $target", review)
        self.assertNotIn("Write-ReviewState $state $remote", review)
        self.assertIn("merge-base --is-ancestor $reviewed $target", review)
        self.assertIn("merge-base --is-ancestor $target $remote", review)
        self.assertIn("re-run with -MarkReviewed $target", review)

    def test_review_state_update_is_state_only(self) -> None:
        review = self._read("Review-ClawdUpdate.ps1")
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

    @unittest.skipUnless(os.name == "nt", "Windows updater behavior")
    def test_windows_powershell_self_tests(self) -> None:
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        self.assertIsNotNone(powershell)

        expected = {
            "Check-ClawdUpdate.ps1": "Startup updater self-test passed.",
            "Review-ClawdUpdate.ps1": "Review classifier self-test passed:",
        }
        for name, marker in expected.items():
            result = subprocess.run(
                [
                    str(powershell),
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(OPS / name),
                    "-SelfTest",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn(marker, output)

    @unittest.skipUnless(os.name == "nt", "Windows batch behavior")
    def test_windows_env_parser_ignores_hash_comments(self) -> None:
        launcher = self._read("Start Clawd Codex.cmd")
        lines = launcher.splitlines()
        for_line = next(line for line in lines if 'for /f "usebackq eol=# tokens=1,* delims=="' in line)
        body_line = next(line for line in lines if 'set "%%A=%%~B"' in line)

        with tempfile.TemporaryDirectory(prefix="clawd-env-parser-") as tmp:
            root = pathlib.Path(tmp)
            env_file = root / "fake.env"
            env_file.write_text(
                "#OLD_KEY=should-not-load\nLIVE_KEY=works\nEMPTY_VALUE=\n",
                encoding="ascii",
            )
            batch = root / "test.cmd"
            adapted_for = re.sub(
                r'\("%CLAWD_ROOT%Secrets\\\.env"\)',
                '("%FAKE_ENV%")',
                for_line.strip(),
                count=1,
            )
            batch.write_text(
                "\r\n".join(
                    [
                        "@echo off",
                        f'set "FAKE_ENV={env_file}"',
                        adapted_for,
                        body_line.strip(),
                        ")",
                        "if defined #OLD_KEY (echo COMMENT_LOADED=YES) else (echo COMMENT_LOADED=NO)",
                        'if "%LIVE_KEY%"=="works" (echo LIVE_KEY=PASS) else (echo LIVE_KEY=FAIL)',
                    ]
                )
                + "\r\n",
                encoding="ascii",
            )
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", str(batch)],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("COMMENT_LOADED=NO", output)
            self.assertIn("LIVE_KEY=PASS", output)


if __name__ == "__main__":
    unittest.main()
