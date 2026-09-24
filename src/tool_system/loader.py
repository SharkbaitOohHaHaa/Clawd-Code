from __future__ import annotations

from pathlib import Path


def load_tools_from_dir(directory: str | Path) -> list[object]:
    raise RuntimeError(
        "Direct Python tool discovery/import is disabled. "
        "Use the operator-trusted Python plugin runtime instead."
    )
