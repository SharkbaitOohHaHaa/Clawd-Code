"""Fail-closed operator and repository permission-policy loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .permissions import ToolPermissionContext


PERMISSION_POLICY_SCHEMA_VERSION = 1


class PermissionPolicyConfigError(RuntimeError):
    """Raised when a permission policy is invalid or expands untrusted authority."""


@dataclass(frozen=True)
class _Policy:
    deny_tools: frozenset[str] = frozenset()
    deny_tool_prefixes: tuple[str, ...] = ()
    additional_working_directories: tuple[Path, ...] = ()
    allow_docs: bool | None = None


def default_operator_permission_policy_path() -> Path:
    """Return the trusted operator permission-policy path."""
    return Path.home() / ".clawd" / "operator-permissions.json"


def default_project_permission_policy_path(workspace_root: str | Path) -> Path:
    """Return the repository restriction-policy path for a workspace."""
    return Path(workspace_root).expanduser().resolve() / ".clawd" / "permissions.json"


def load_permission_context(
    workspace_root: str | Path,
    *,
    operator_policy_path: str | Path | None = None,
    project_policy_path: str | Path | None = None,
) -> ToolPermissionContext:
    """Load and merge operator grants with repository-only restrictions.

    Operator policy may add bounded working directories and enable documentation
    writes. Repository policy may only add tool denials or force documentation
    writes back to prompt-required mode. Repository policy can never add working
    roots or enable a permission the operator did not grant.
    """
    root = Path(workspace_root).expanduser().resolve()
    operator_path = Path(
        operator_policy_path or default_operator_permission_policy_path()
    ).expanduser()
    project_path = Path(
        project_policy_path or default_project_permission_policy_path(root)
    ).expanduser()

    operator = _load_operator_policy(operator_path)
    project = _load_project_policy(project_path, workspace_root=root)

    deny_tools = operator.deny_tools | project.deny_tools
    deny_prefixes = tuple(sorted(set(operator.deny_tool_prefixes) | set(project.deny_tool_prefixes)))
    allow_docs = bool(operator.allow_docs)
    if project.allow_docs is False:
        allow_docs = False

    return ToolPermissionContext.from_iterables(
        deny_names=sorted(deny_tools),
        deny_prefixes=deny_prefixes,
        workspace_root=root,
        additional_working_directories=operator.additional_working_directories,
        allow_docs=allow_docs,
        allow_docs_locked_off=project.allow_docs is False,
    )


def _load_operator_policy(path: Path) -> _Policy:
    data = _read_policy_object(path, label="operator")
    if data is None:
        return _Policy(allow_docs=False)
    _reject_extra_fields(
        data,
        allowed={
            "schema_version",
            "deny_tools",
            "deny_tool_prefixes",
            "additional_working_directories",
            "allow_docs",
        },
        label="operator",
    )
    return _Policy(
        deny_tools=_string_set(data.get("deny_tools", []), "operator deny_tools"),
        deny_tool_prefixes=_string_tuple(
            data.get("deny_tool_prefixes", []),
            "operator deny_tool_prefixes",
        ),
        additional_working_directories=_operator_working_directories(
            data.get("additional_working_directories", [])
        ),
        allow_docs=_optional_bool(data, "allow_docs", default=False, label="operator"),
    )


def _load_project_policy(path: Path, *, workspace_root: Path) -> _Policy:
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise PermissionPolicyConfigError(
            f"cannot resolve project permission policy: {path}"
        ) from exc
    if not _is_within(resolved, workspace_root):
        raise PermissionPolicyConfigError(
            "project permission policy must resolve inside the workspace"
        )

    data = _read_policy_object(path, label="project")
    if data is None:
        return _Policy()
    _reject_extra_fields(
        data,
        allowed={"schema_version", "deny_tools", "deny_tool_prefixes", "allow_docs"},
        label="project",
    )
    allow_docs = _optional_bool(data, "allow_docs", default=None, label="project")
    if allow_docs is True:
        raise PermissionPolicyConfigError(
            "project permission policy cannot enable allow_docs; project policy may only reduce authority"
        )
    return _Policy(
        deny_tools=_string_set(data.get("deny_tools", []), "project deny_tools"),
        deny_tool_prefixes=_string_tuple(
            data.get("deny_tool_prefixes", []),
            "project deny_tool_prefixes",
        ),
        allow_docs=allow_docs,
    )


def _read_policy_object(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise PermissionPolicyConfigError(f"{label} permission policy is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionPolicyConfigError(
            f"cannot read {label} permission policy: {path}"
        ) from exc
    if not isinstance(data, dict):
        raise PermissionPolicyConfigError(f"{label} permission policy must be a JSON object")
    if data.get("schema_version") != PERMISSION_POLICY_SCHEMA_VERSION:
        raise PermissionPolicyConfigError(
            f"{label} permission policy schema_version must be {PERMISSION_POLICY_SCHEMA_VERSION}"
        )
    return data


def _reject_extra_fields(data: dict[str, Any], *, allowed: set[str], label: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise PermissionPolicyConfigError(
            f"unsupported {label} permission policy field(s): {', '.join(extra)}"
        )


def _string_set(value: Any, label: str) -> frozenset[str]:
    return frozenset(_validated_strings(value, label))


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    return tuple(sorted(set(_validated_strings(value, label))))


def _validated_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise PermissionPolicyConfigError(f"{label} must be an array of non-empty strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise PermissionPolicyConfigError(f"{label} must contain non-empty trimmed strings")
        result.append(item.lower())
    return result


def _operator_working_directories(value: Any) -> tuple[Path, ...]:
    if not isinstance(value, list):
        raise PermissionPolicyConfigError(
            "operator additional_working_directories must be an array of absolute directory paths"
        )
    directories: list[Path] = []
    for raw in value:
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise PermissionPolicyConfigError(
                "operator additional_working_directories must contain non-empty trimmed strings"
            )
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise PermissionPolicyConfigError(
                "operator additional_working_directories entries must be absolute"
            )
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise PermissionPolicyConfigError(
                f"cannot resolve operator additional working directory: {candidate}"
            ) from exc
        if not resolved.is_dir():
            raise PermissionPolicyConfigError(
                f"operator additional working directory does not exist: {resolved}"
            )
        if resolved not in directories:
            directories.append(resolved)
    return tuple(directories)


def _optional_bool(
    data: dict[str, Any],
    field: str,
    *,
    default: bool | None,
    label: str,
) -> bool | None:
    if field not in data:
        return default
    value = data[field]
    if not isinstance(value, bool):
        raise PermissionPolicyConfigError(f"{label} {field} must be a boolean")
    return value


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
