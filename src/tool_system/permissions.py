from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import fnmatch
from pathlib import Path
from typing import Iterable

from .errors import ToolPermissionError
from .permission_handler import PermissionResult


def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


class SensitivePathBehavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class SensitivePathDecision:
    behavior: SensitivePathBehavior
    reason: str = ""


_ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template"}
_PRIVATE_KEY_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
_PRIVATE_KEY_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def sensitive_path_decision(path: str | Path, *, operation: str) -> SensitivePathDecision:
    """Classify built-in sensitive path handling for read/write style operations."""
    resolved = Path(path).expanduser().resolve()
    name = resolved.name.lower()

    if name in _PRIVATE_KEY_NAMES or resolved.suffix.lower() in _PRIVATE_KEY_SUFFIXES:
        return SensitivePathDecision(
            SensitivePathBehavior.DENY,
            f"{operation} access to private-key material is denied by default: {resolved}",
        )

    if name in _ENV_TEMPLATE_NAMES:
        return SensitivePathDecision(SensitivePathBehavior.ALLOW)

    is_live_env = name == ".env" or name.startswith(".env.")
    is_credentials_json = (
        fnmatch.fnmatch(name, "credentials*.json")
        or fnmatch.fnmatch(name, "*credentials*.json")
        or fnmatch.fnmatch(name, "service-account*.json")
        or fnmatch.fnmatch(name, "service_account*.json")
    )
    is_token_json = (
        name == "token.json"
        or fnmatch.fnmatch(name, "token-*.json")
        or fnmatch.fnmatch(name, "token_*.json")
        or ("oauth" in name and "token" in name and name.endswith(".json"))
    )
    if is_live_env or is_credentials_json or is_token_json:
        return SensitivePathDecision(
            SensitivePathBehavior.ASK,
            f"{operation} access to a secret-bearing configuration or credential file requires confirmation: {resolved}",
        )

    return SensitivePathDecision(SensitivePathBehavior.ALLOW)


def sensitive_path_permission(path: str | Path, *, operation: str) -> PermissionResult:
    decision = sensitive_path_decision(path, operation=operation)
    if decision.behavior is SensitivePathBehavior.DENY:
        return PermissionResult.deny(decision.reason)
    if decision.behavior is SensitivePathBehavior.ASK:
        return PermissionResult.ask(decision.reason)
    return PermissionResult.allow()


def is_sensitive_path(path: str | Path) -> bool:
    return sensitive_path_decision(path, operation="read").behavior is not SensitivePathBehavior.ALLOW


def protected_write_reason(path: Path, *, existing: bool) -> str | None:
    """Return a confirmation reason for owner-protected project history paths."""
    parts = {part.upper().replace("_", "-") for part in path.parts}
    if "LOCKED" in parts:
        return "This write touches LOCKED known-good history and requires current confirmation"
    if "PAST-ERRORS" in parts and existing:
        return "This write would modify existing PAST-ERRORS history and requires current confirmation"
    return None


@dataclass
class ToolPermissionContext:
    deny_names: frozenset[str] = field(default_factory=frozenset)
    deny_prefixes: tuple[str, ...] = ()
    workspace_root: Path | None = None
    additional_working_directories: tuple[Path, ...] = ()
    allow_docs: bool = False
    allow_docs_locked_off: bool = False

    @classmethod
    def from_iterables(
        cls,
        deny_names: Iterable[str] | None = None,
        deny_prefixes: Iterable[str] | None = None,
        *,
        workspace_root: str | Path | None = None,
        additional_working_directories: Iterable[str | Path] | None = None,
        allow_docs: bool = False,
        allow_docs_locked_off: bool = False,
    ) -> "ToolPermissionContext":
        return cls(
            deny_names=frozenset(name.lower() for name in (deny_names or [])),
            deny_prefixes=tuple(prefix.lower() for prefix in (deny_prefixes or [])),
            workspace_root=_resolve_path(workspace_root) if workspace_root else None,
            additional_working_directories=tuple(
                _resolve_path(p) for p in (additional_working_directories or [])
            ),
            allow_docs=allow_docs,
            allow_docs_locked_off=allow_docs_locked_off,
        )

    def blocks_tool(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return lowered in self.deny_names or any(
            lowered.startswith(prefix) for prefix in self.deny_prefixes
        )

    def allowed_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        if self.workspace_root is not None:
            roots.append(self.workspace_root)
        roots.extend(self.additional_working_directories)
        return tuple(roots)

    def ensure_path_allowed(self, path: str | Path) -> Path:
        resolved = _resolve_path(path)
        roots = self.allowed_roots()
        if not roots:
            return resolved
        if any(_is_within(resolved, root) for root in roots):
            return resolved
        roots_str = ", ".join(str(r) for r in roots)
        raise ToolPermissionError(f"path is outside allowed working directories: {resolved} (allowed: {roots_str})")

