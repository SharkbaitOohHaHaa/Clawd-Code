from __future__ import annotations

from pathlib import Path

from .models import ClaudeMdContext, ClaudeMdFile

_PROJECT_CANDIDATES = ("CLAUDE.md", ".clawd/CLAUDE.md", ".claude/CLAUDE.md")
_USER_CANDIDATES = (".clawd/CLAUDE.md", ".claude/CLAUDE.md")


def load_claude_md_context(
    workspace_root: str | Path,
    *,
    cwd: str | Path | None = None,
    max_files: int = 6,
    max_chars_per_file: int = 4_000,
    max_total_chars: int = 12_000,
) -> ClaudeMdContext:
    root = Path(workspace_root).expanduser().resolve()
    current = Path(cwd).expanduser().resolve() if cwd is not None else root

    candidates: list[Path] = []

    home = Path.home()
    for rel in _USER_CANDIDATES:
        path = (home / rel).resolve()
        if path not in candidates:
            candidates.append(path)

    for base in _walk_up_to_root(current, root):
        for rel in _PROJECT_CANDIDATES:
            path = (base / rel).resolve()
            if not _is_within(path, root):
                continue
            if path not in candidates:
                candidates.append(path)

    files: list[ClaudeMdFile] = []
    total_chars = 0
    truncated = False
    for path in candidates:
        if len(files) >= max_files or total_chars >= max_total_chars:
            truncated = True
            break
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        if len(content) > max_chars_per_file:
            content = content[: max_chars_per_file - 32].rstrip() + "\n...[truncated]"
            truncated = True
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            truncated = True
            break
        if len(content) > remaining:
            content = content[: max(0, remaining - 32)].rstrip() + "\n...[truncated]"
            truncated = True
        total_chars += len(content)
        files.append(ClaudeMdFile(path=path, content=content))

    return ClaudeMdContext(files=tuple(files), truncated=truncated)


def _walk_up_to_root(current: Path, root: Path) -> list[Path]:
    current = current.resolve()
    root = root.resolve()
    if current != root:
        try:
            current.relative_to(root)
        except ValueError:
            current = root

    bases: list[Path] = []
    node = current
    while True:
        if node not in bases:
            bases.append(node)
        if node == root:
            break
        parent = node.parent
        if parent == node:
            break
        node = parent
    return bases


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
