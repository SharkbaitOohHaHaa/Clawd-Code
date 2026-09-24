from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolPermissionError


class TestHardeningSecurity(unittest.TestCase):
    def test_default_registry_excludes_disabled_execution_tools(self) -> None:
        names = {spec.name for spec in build_default_registry().list_specs()}
        for name in {
            "Bash", "Config", "PowerShell",
            "REPL", "RemoteTrigger", "TestingPermission",
        }:
            self.assertNotIn(name, names)

        self.assertIn("Skill", names)
        self.assertIn("NotebookEdit", names)

    def test_default_registry_does_not_autoload_user_python_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            tool_dir = home / ".clawd" / "tools"
            tool_dir.mkdir(parents=True)
            marker = home / "AUTOLOAD-RAN.txt"
            (tool_dir / "canary.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            build_default_registry()
            self.assertFalse(marker.exists())

    def test_workspace_guard_rejects_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            ctx = ToolContext(workspace_root=Path(workspace))
            with self.assertRaises(ToolPermissionError):
                ctx.ensure_allowed_path(Path(outside) / "secret.txt")


if __name__ == "__main__":
    unittest.main()
