from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.plugins.runtime import reconcile_python_plugins
from src.tool_system.defaults import build_default_registry
from src.tool_system.loader import load_tools_from_dir


class PythonPluginRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.plugins = self.root / "plugins"
        self.policy = self.root / "python_plugins.json"
        self.plugins.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_plugin(
        self,
        name: str = "demo",
        *,
        entrypoint: str = "plugin.py",
        code: str = "VALUE = 1\n",
    ) -> Path:
        plugin_dir = self.plugins / name
        plugin_dir.mkdir()
        manifest = {
            "schema_version": 1,
            "name": name,
            "version": "1.0.0",
            "description": "test plugin",
            "entrypoint": entrypoint,
            "extensions": ["tools"],
        }
        (plugin_dir / "plugin.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        if ".." not in Path(entrypoint).parts:
            target = plugin_dir / entrypoint
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
        return plugin_dir

    def _reconcile(self):
        return reconcile_python_plugins(
            plugin_root=self.plugins,
            operator_manifest=self.policy,
        )

    def test_discovery_does_not_execute_plugin_python(self) -> None:
        marker = self.root / "executed.txt"
        self._write_plugin(
            code=(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
            )
        )

        report = self._reconcile()

        self.assertFalse(marker.exists())
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["active"], [])
        self.assertEqual(report["records"]["demo"]["state"], "inactive")
        self.assertRegex(report["records"]["demo"]["artifact_sha256"], r"^[0-9a-f]{64}$")

    def test_activation_requires_exact_hash_and_change_invalidates_it(self) -> None:
        plugin_dir = self._write_plugin()
        first = self._reconcile()
        digest = first["records"]["demo"]["artifact_sha256"]
        self.policy.write_text(
            json.dumps({
                "schema_version": 1,
                "plugins": {"demo": {"enabled": True, "artifact_sha256": digest}},
            }),
            encoding="utf-8",
        )

        active = self._reconcile()
        self.assertEqual(active["active"], ["demo"])
        self.assertEqual(active["records"]["demo"]["state"], "active")
        (plugin_dir / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
        changed = self._reconcile()
        self.assertEqual(changed["active"], [])
        self.assertEqual(changed["records"]["demo"]["state"], "review_required")
        self.assertIn(
            "plugin_integrity_mismatch",
            {issue["code"] for issue in changed["issues"]},
        )

    def test_entrypoint_escape_is_rejected(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        self._write_plugin(entrypoint="../outside.py")

        report = self._reconcile()

        self.assertEqual(report["records"], {})
        self.assertIn(
            "plugin_manifest_invalid",
            {issue["code"] for issue in report["issues"]},
        )

    def test_enabled_plugin_without_hash_is_blocked(self) -> None:
        self._write_plugin()
        self.policy.write_text(
            json.dumps({
                "schema_version": 1,
                "plugins": {"demo": {"enabled": True, "artifact_sha256": ""}},
            }),
            encoding="utf-8",
        )
        report = self._reconcile()

        self.assertEqual(report["active"], [])
        self.assertEqual(report["records"]["demo"]["state"], "blocked")
        self.assertIn(
            "plugin_operator_hash_missing",
            {issue["code"] for issue in report["issues"]},
        )

    def test_enabled_missing_plugin_is_reported(self) -> None:
        self.policy.write_text(
            json.dumps({
                "schema_version": 1,
                "plugins": {
                    "missing": {
                        "enabled": True,
                        "artifact_sha256": "0" * 64,
                    }
                },
            }),
            encoding="utf-8",
        )

        report = self._reconcile()
        self.assertIn(
            "enabled_plugin_missing",
            {issue["code"] for issue in report["issues"]},
        )

    def test_legacy_direct_python_tool_loading_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            load_tools_from_dir(self.root / "tools")
        with self.assertRaises(RuntimeError):
            build_default_registry(include_user_tools=True)


if __name__ == "__main__":
    unittest.main()
