from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..compare_write import still_holds
from ..context import ToolContext
from ..errors import ToolExecutionError, ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..permissions import protected_write_reason, sensitive_path_permission
from ..protocol import ToolResult
from ..registry import ToolSpec


_REFUSE_NOTEBOOK = "refusing to edit notebook: file must be read first and unchanged since last read"


def _decode_like_strict_read_text(raw: bytes) -> str:
    """Decode exactly as Path.read_text(encoding="utf-8") does: strict errors and
    universal-newline translation (json.loads(bytes) would also accept BOM/UTF-16)."""
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8") as handle:
        return handle.read()


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _find_cell_index(cells: list[Any], cell_id: str) -> int:
    matches = [
        index
        for index, cell in enumerate(cells)
        if isinstance(cell, dict) and str(cell.get("id") or "") == cell_id
    ]
    if not matches:
        raise ToolInputError(f"notebook cell not found: {cell_id}")
    if len(matches) > 1:
        raise ToolInputError(f"notebook cell id is not unique: {cell_id}")
    return matches[0]


def _new_cell_id(cells: list[Any]) -> str:
    existing = {
        str(cell.get("id") or "")
        for cell in cells
        if isinstance(cell, dict) and cell.get("id")
    }
    for _ in range(32):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    raise ToolExecutionError("failed to generate a unique notebook cell id")


def _write_notebook_atomic(path: Path, data: dict[str, Any], expected: bytes) -> bytes:
    """Replace ``path`` atomically after re-checking that it still holds exactly
    ``expected``, and return exactly the bytes written. The re-check is not atomic with
    the replace: a change (or delete) landing in between is overwritten (or re-created).
    """
    payload = (json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        if not still_holds(path, expected):
            raise ToolInputError(_REFUSE_NOTEBOOK)
        os.replace(temp_path, path)
        temp_path = None
    except OSError as exc:
        raise ToolExecutionError(f"failed to write notebook: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return payload


class NotebookEditTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="NotebookEdit",
            permission_policy="checked",
            description=(
                "Replace, insert, or delete a cell in a Jupyter notebook (.ipynb). "
                "The notebook_path must be absolute. Notebook Python is never executed."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "notebook_path": {"type": "string"},
                    "cell_id": {"type": "string"},
                    "new_source": {"type": "string"},
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                    },
                    "edit_mode": {
                        "type": "string",
                        "enum": ["replace", "insert", "delete"],
                    },
                },
                "required": ["notebook_path", "new_source"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        notebook_path = tool_input.get("notebook_path")
        if not isinstance(notebook_path, str):
            return PermissionResult.allow()

        try:
            path = context.ensure_allowed_path(notebook_path)
        except ToolPermissionError as exc:
            return PermissionResult.deny(str(exc))

        sensitive_result = sensitive_path_permission(path, operation="write")
        if sensitive_result.behavior.value != "allow":
            return sensitive_result

        protected_reason = protected_write_reason(path, existing=True)
        if protected_reason:
            return PermissionResult.ask(message=protected_reason)
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        notebook_path = tool_input["notebook_path"]
        new_source = tool_input["new_source"]
        cell_id = str(tool_input.get("cell_id") or "").strip()
        cell_type = tool_input.get("cell_type")
        edit_mode = str(tool_input.get("edit_mode") or "replace").strip().lower()

        if not isinstance(notebook_path, str):
            raise ToolInputError("notebook_path must be a string")
        if not Path(notebook_path).expanduser().is_absolute():
            raise ToolInputError("notebook_path must be an absolute path")
        if not isinstance(new_source, str):
            raise ToolInputError("new_source must be a string")
        if cell_type is not None and cell_type not in {"code", "markdown"}:
            raise ToolInputError("cell_type must be code or markdown")
        if edit_mode not in {"replace", "insert", "delete"}:
            raise ToolInputError("edit_mode must be replace, insert, or delete")

        path = context.ensure_allowed_path(notebook_path)
        if path.suffix.lower() != ".ipynb":
            raise ToolInputError("NotebookEdit only supports .ipynb files")
        if not path.exists() or not path.is_file():
            raise ToolInputError(f"notebook does not exist: {path}")
        verified = context.read_if_unchanged(path)
        if verified is None:
            raise ToolInputError(_REFUSE_NOTEBOOK)

        try:
            data = json.loads(_decode_like_strict_read_text(verified))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolInputError(f"failed to parse notebook: {exc}") from exc
        if not isinstance(data, dict):
            raise ToolInputError("notebook root must be a JSON object")
        cells = data.get("cells")
        if not isinstance(cells, list):
            raise ToolInputError("notebook must contain a cells array")

        affected_index: int
        affected_id: str
        if edit_mode == "insert":
            if cell_type not in {"code", "markdown"}:
                raise ToolInputError("cell_type is required when edit_mode=insert")
            insert_at = _find_cell_index(cells, cell_id) + 1 if cell_id else 0
            affected_id = _new_cell_id(cells)
            new_cell: dict[str, Any] = {
                "cell_type": cell_type,
                "id": affected_id,
                "metadata": {},
                "source": _source_lines(new_source),
            }
            if cell_type == "code":
                new_cell["execution_count"] = None
                new_cell["outputs"] = []
            cells.insert(insert_at, new_cell)
            affected_index = insert_at
        else:
            if not cell_id:
                raise ToolInputError(f"cell_id is required when edit_mode={edit_mode}")
            affected_index = _find_cell_index(cells, cell_id)
            affected_id = cell_id
            if edit_mode == "delete":
                cells.pop(affected_index)
            else:
                existing = cells[affected_index]
                if not isinstance(existing, dict):
                    raise ToolInputError("target notebook cell must be an object")
                final_type = cell_type or existing.get("cell_type")
                if final_type not in {"code", "markdown"}:
                    raise ToolInputError("target notebook cell type must be code or markdown")
                existing["cell_type"] = final_type
                existing["source"] = _source_lines(new_source)
                if final_type == "code":
                    existing.pop("attachments", None)
                    existing["execution_count"] = None
                    existing["outputs"] = []
                else:
                    existing.pop("execution_count", None)
                    existing.pop("outputs", None)

        written = _write_notebook_atomic(path, data, verified)
        context.mark_file_read(path, data=written)
        return ToolResult(
            name="NotebookEdit",
            output={
                "type": "notebook_edit",
                "filePath": str(path),
                "editMode": edit_mode,
                "cellId": affected_id,
                "cellIndex": affected_index,
                "cellType": None if edit_mode == "delete" else (
                    cells[affected_index].get("cell_type")
                    if isinstance(cells[affected_index], dict)
                    else None
                ),
            },
        )
