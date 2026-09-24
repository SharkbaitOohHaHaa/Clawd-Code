from __future__ import annotations

import pathlib
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].lower().replace("_", "-")


class PackagingContractTests(unittest.TestCase):
    def test_runtime_dependencies_match_imported_runtime_surface(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertNotIn("python-dotenv", project["project"]["dependencies"])

    def test_requirements_does_not_reintroduce_removed_runtime_dependency(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("python-dotenv", requirements)

    def test_canonical_console_entrypoint_is_unchanged(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["scripts"], {"clawd": "src.cli:main"})

    def test_project_urls_match_readme_repository(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"git clone (https://github\.com/[^\s]+)\.git", readme)
        if match is None:
            self.fail("README clone URL was not found")
        repository = match.group(1)
        self.assertEqual(project["project"]["urls"]["Homepage"], repository)
        self.assertEqual(project["project"]["urls"]["Repository"], repository)
        self.assertEqual(
            project["project"]["urls"]["Documentation"],
            f"{repository}#readme",
        )

    def test_manifest_literal_include_targets_exist(self) -> None:
        for raw_line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("include "):
                continue
            for target in line.split()[1:]:
                if any(marker in target for marker in ("*", "?", "[")):
                    continue
                self.assertTrue((ROOT / target).is_file(), target)

    def test_dev_dependency_group_matches_pip_compatibility_extra(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dev_group = project["dependency-groups"]["dev"]
        dev_extra = project["project"]["optional-dependencies"]["dev"]
        self.assertEqual(dev_group, dev_extra)
        names = {_dependency_name(spec) for spec in dev_group}
        self.assertTrue({"pytest", "ruff", "mypy"}.issubset(names))

    def test_requirements_fallback_covers_project_and_dev_dependencies(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        expected = {
            _dependency_name(spec)
            for spec in (
                project["project"]["dependencies"]
                + project["dependency-groups"]["dev"]
            )
        }
        actual = {
            _dependency_name(line.strip())
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(expected.issubset(actual), sorted(expected - actual))

    def test_uv_lock_runtime_dependencies_match_pyproject(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        package = next(
            item for item in lock["package"] if item["name"] == project["project"]["name"]
        )
        expected = {_dependency_name(spec) for spec in project["project"]["dependencies"]}
        actual = {item["name"] for item in package["dependencies"]}
        self.assertEqual(actual, expected)

    def test_ci_workflow_covers_supported_python_and_release_gates(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(workflow_path.is_file())
        workflow = workflow_path.read_text(encoding="utf-8")

        supported = {
            classifier.rsplit("::", 1)[-1].strip()
            for classifier in project["project"]["classifiers"]
            if classifier.startswith("Programming Language :: Python :: 3.")
        }
        matrix_match = re.search(
            r"matrix:\s*\n\s*python-version:\s*\n"
            r"(?P<versions>(?:\s*-\s*[\"']?3\.\d+[\"']?\s*\n)+)",
            workflow,
        )
        if matrix_match is None:
            self.fail("CI Python version matrix was not found")
        matrix_versions = set(
            re.findall(r"3\.\d+", matrix_match.group("versions"))
        )
        self.assertEqual(matrix_versions, supported)

        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("push:", workflow)
        self.assertIn("pull_request:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request_target:", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 2)
        self.assertEqual(workflow.count("enable-cache: false"), 2)
        self.assertIn("timeout-minutes: 30", workflow)
        self.assertIn("timeout-minutes: 15", workflow)

        uses = re.findall(r"uses:\s+([^@\s]+)@([0-9a-f]{40})", workflow)
        self.assertTrue(uses)
        self.assertTrue(any(name == "actions/checkout" for name, _ in uses))
        self.assertTrue(any(name == "astral-sh/setup-uv" for name, _ in uses))
        self.assertNotRegex(workflow, r"uses:\s+[^\s]+@(v\d+|main|master)\b")

        for command in (
            "uv sync --locked --all-extras --dev",
            "uv run --locked pytest tests/test_capabilities.py -q",
            "uv run --locked pytest -q",
            "uv run --locked ruff check src tests",
            "uv run --locked mypy",
            "uv build --out-dir dist",
            "uv run --locked python -m twine check dist/*",
        ):
            self.assertIn(command, workflow)

        self.assertNotIn("secrets.", workflow)
        self.assertNotRegex(workflow.lower(), r"\b(publish|upload)\b")

    def test_uv_developer_python_is_pinned_to_supported_version(self) -> None:
        self.assertEqual((ROOT / ".python-version").read_text(encoding="utf-8").strip(), "3.12")

    def test_quality_tool_configuration_is_explicit_and_gradual(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["tool"]["pytest"]["ini_options"]["testpaths"], ["tests"])
        self.assertEqual(
            project["tool"]["ruff"]["lint"]["select"],
            ["E9", "F63", "F7", "F82"],
        )
        self.assertEqual(
            project["tool"]["mypy"]["files"],
            [
                "src/config.py",
                "src/capabilities.py",
                "src/context_system/context_analyzer.py",
                "src/tool_system/tools/data.py",
            ],
        )
        self.assertEqual(project["tool"]["mypy"]["follow_imports"], "skip")


if __name__ == "__main__":
    unittest.main()
