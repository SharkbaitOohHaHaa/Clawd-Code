"""Deterministic bounded project-map context for the active workspace."""

from __future__ import annotations

import fnmatch
from pathlib import Path


_IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}
_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".ini",
    ".cfg",
}
_KEY_FILE_NAMES = {
    "Makefile",
    "Dockerfile",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "README.md",
    "CLAUDE.md",
}
_SOURCE_ROOT_NAMES = {"src", "app", "lib", "packages"}
_TEST_ROOT_NAMES = {"tests", "test"}
_SENSITIVE_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "credentials*.json",
    "*credentials*.json",
    "service-account*.json",
    "service_account*.json",
    "token.json",
    "token-*.json",
    "token_*.json",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
)


def build_project_map(
    workspace_root: str | Path,
    *,
    max_depth: int = 3,
    max_entries: int = 80,
) -> tuple[str, ...]:
    """Return a bounded, sorted workspace-relative source/config file map.

    The map is names-only context: it never reads file contents, never follows
    symlinks, and omits hidden/sensitive-looking entries and common generated
    dependency/build directories. A fixed budget retains key project files and,
    when present, representative source and test paths rather than letting one
    large tree consume the whole map.
    """
    if max_depth < 0 or max_entries <= 0:
        return ()

    root = Path(workspace_root).expanduser().resolve()
    if not root.is_dir():
        return ()

    key_files: list[str] = []
    source_files: list[str] = []
    test_files: list[str] = []
    other_files: list[str] = []

    try:
        root_children = [
            child
            for child in root.iterdir()
            if not _skip_name(child.name) and not _safe_is_symlink(child)
        ]
    except OSError:
        return ()
    root_children.sort(key=lambda path: (path.name.lower(), path.name))

    collection_limit = max(max_entries * 4, max_entries)
    for child in root_children:
        try:
            resolved = child.resolve()
        except OSError:
            continue
        if not _is_within(resolved, root):
            continue

        if _safe_is_file(child):
            if _include_file(child.name, child.suffix.lower()):
                target = key_files if child.name in _KEY_FILE_NAMES else other_files
                target.append(child.relative_to(root).as_posix())
            continue

        if not _safe_is_dir(child) or _skip_directory(child.name):
            continue
        lowered = child.name.lower()
        if lowered in _SOURCE_ROOT_NAMES:
            source_files.extend(
                _collect_tree_files(child, root=root, max_depth=max_depth, limit=collection_limit)
            )
        elif lowered in _TEST_ROOT_NAMES:
            test_files.extend(
                _collect_tree_files(child, root=root, max_depth=max_depth, limit=collection_limit)
            )
        else:
            other_files.extend(
                _collect_tree_files(child, root=root, max_depth=max_depth, limit=collection_limit)
            )

    key_files = sorted(set(key_files), key=str.lower)
    source_files = sorted(set(source_files), key=str.lower)
    test_files = sorted(set(test_files), key=str.lower)
    other_files = sorted(set(other_files), key=str.lower)

    selected: list[str] = []
    _extend_bounded(selected, key_files, max_entries)
    remaining = max_entries - len(selected)

    if remaining > 0 and source_files and test_files:
        test_quota = min(len(test_files), max(1, remaining // 4))
        source_quota = min(len(source_files), remaining - test_quota)
        _extend_bounded(selected, source_files[:source_quota], max_entries)
        _extend_bounded(selected, test_files[:test_quota], max_entries)
        _extend_bounded(selected, source_files[source_quota:], max_entries)
        _extend_bounded(selected, test_files[test_quota:], max_entries)
    else:
        _extend_bounded(selected, source_files, max_entries)
        _extend_bounded(selected, test_files, max_entries)

    _extend_bounded(selected, other_files, max_entries)

    candidate_count = len(key_files) + len(source_files) + len(test_files) + len(other_files)
    if candidate_count > len(selected):
        selected.append("... [project map truncated]")
    return tuple(selected)


def _collect_tree_files(
    start: Path,
    *,
    root: Path,
    max_depth: int,
    limit: int,
) -> list[str]:
    collected: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if len(collected) >= limit:
            return
        try:
            children = [
                child
                for child in directory.iterdir()
                if not _skip_name(child.name) and not _safe_is_symlink(child)
            ]
        except OSError:
            return
        children.sort(key=lambda path: (path.name.lower(), path.name))
        for child in children:
            if len(collected) >= limit:
                return
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if not _is_within(resolved, root):
                continue
            if _safe_is_dir(child):
                if not _skip_directory(child.name) and depth < max_depth:
                    visit(child, depth + 1)
                continue
            if _safe_is_file(child) and _include_file(child.name, child.suffix.lower()):
                collected.append(child.relative_to(root).as_posix())

    visit(start, 0)
    return collected


def _extend_bounded(target: list[str], values: list[str], max_entries: int) -> None:
    if len(target) >= max_entries:
        return
    target.extend(values[: max_entries - len(target)])


def _include_file(name: str, suffix: str) -> bool:
    if is_sensitive_context_name(name):
        return False
    return name in _KEY_FILE_NAMES or suffix in _SOURCE_SUFFIXES


def _skip_name(name: str) -> bool:
    return name.startswith(".")


def _skip_directory(name: str) -> bool:
    lowered = name.lower()
    return lowered in _IGNORED_DIR_NAMES or lowered.endswith(".egg-info")


def is_sensitive_context_name(name: str) -> bool:
    """Return whether a filename should be omitted from automatic context surfaces."""
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in _SENSITIVE_FILE_PATTERNS)


def _safe_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return True


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
