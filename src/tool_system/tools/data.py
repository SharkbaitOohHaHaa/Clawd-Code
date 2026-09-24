from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolExecutionError, ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..permissions import protected_write_reason, sensitive_path_permission
from ..protocol import ToolResult
from ..registry import ToolSpec


_MAX_INPUT_BYTES = 10 * 1024 * 1024
_MAX_OUTPUT_BYTES = 20 * 1024 * 1024
_MAX_ROWS = 100_000
_MAX_PREVIEW_ROWS = 200
_SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    raise ToolInputError(
        "data tools support only .csv, .tsv, .json, and .jsonl files"
    )


def _validate_file_size(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolExecutionError(f"failed to inspect data file: {exc}") from exc
    if size > _MAX_INPUT_BYTES:
        raise ToolInputError(
            f"data file exceeds {_MAX_INPUT_BYTES // (1024 * 1024)} MiB limit"
        )
    return size


def _append_bounded(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    if len(rows) >= _MAX_ROWS:
        raise ToolInputError(f"data file exceeds {_MAX_ROWS} row limit")
    rows.append(row)


def _load_delimited(
    path: Path, *, delimiter: str
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return [], []
            if not header or any(not str(name).strip() for name in header):
                raise ToolInputError("delimited data must have non-empty column names")
            if len(set(header)) != len(header):
                raise ToolInputError("delimited data contains duplicate column names")
            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise ToolInputError(
                        f"row {row_number} has {len(values)} field(s); expected {len(header)}"
                    )
                _append_bounded(rows, dict(zip(header, values)))
    except UnicodeDecodeError as exc:
        raise ToolInputError("data file is not valid UTF-8 text") from exc
    except OSError as exc:
        raise ToolExecutionError(f"failed to read data file: {exc}") from exc
    return rows, header


def _load_json(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ToolInputError("data file is not valid UTF-8 text") from exc
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ToolExecutionError(f"failed to read data file: {exc}") from exc

    if not isinstance(value, list):
        raise ToolInputError("JSON data must be a top-level array of objects")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ToolInputError(f"JSON row {index} must be an object")
        _append_bounded(rows, dict(item))
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ToolInputError(
                        f"invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ToolInputError(
                        f"JSONL line {line_number} must contain an object"
                    )
                _append_bounded(rows, dict(item))
    except UnicodeDecodeError as exc:
        raise ToolInputError("data file is not valid UTF-8 text") from exc
    except OSError as exc:
        raise ToolExecutionError(f"failed to read data file: {exc}") from exc
    return rows


def _load_rows(
    path: Path,
) -> tuple[str, int, list[dict[str, Any]], list[str]]:
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        _format_name(path)
    size = _validate_file_size(path)
    fmt = _format_name(path)
    if fmt == "csv":
        rows, columns = _load_delimited(path, delimiter=",")
    elif fmt == "tsv":
        rows, columns = _load_delimited(path, delimiter="\t")
    elif fmt == "json":
        rows = _load_json(path)
        columns = _ordered_columns(rows)
    else:
        rows = _load_jsonl(path)
        columns = _ordered_columns(rows)
    return fmt, size, rows, columns


def _ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for raw_name in row:
            name = str(raw_name)
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


_TYPE_ORDER = {
    "null": 0,
    "boolean": 1,
    "integer": 2,
    "number": 3,
    "string": 4,
    "array": 5,
    "object": 6,
}


def _column_schema(
    rows: list[dict[str, Any]], columns: list[str]
) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in columns:
        types = {
            _value_type(row.get(column))
            for row in rows
            if column in row
        }
        nullable = any(column not in row or row.get(column) is None for row in rows)
        schema.append(
            {
                "name": column,
                "types": sorted(types, key=lambda item: (_TYPE_ORDER.get(item, 99), item)),
                "nullable": nullable,
            }
        )
    return schema


def _normalize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_rows(
    rows: list[dict[str, Any]], *, columns: list[str], output_format: str
) -> str:
    if output_format in {"csv", "tsv"}:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            delimiter="," if output_format == "csv" else "\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _normalize_csv_value(row.get(column)) for column in columns}
            )
        return stream.getvalue()
    if output_format == "json":
        projected = [
            {column: row.get(column) for column in columns}
            for row in rows
        ]
        return json.dumps(projected, ensure_ascii=False, indent=2) + "\n"
    if output_format == "jsonl":
        return "".join(
            json.dumps(
                {column: row.get(column) for column in columns},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    raise ToolInputError(f"unsupported output format: {output_format}")


def _write_new_text(path: Path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise ToolInputError(
            f"transformed output exceeds {_MAX_OUTPUT_BYTES // (1024 * 1024)} MiB limit"
        )
    if not path.parent.exists() or not path.parent.is_dir():
        raise ToolInputError("output parent directory must already exist")
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ToolInputError("refusing to overwrite existing output data file") from exc
    except OSError as exc:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        raise ToolExecutionError(f"failed to write transformed data: {exc}") from exc


class DataInspectTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="DataInspect",
            permission_policy="checked",
            description=(
                "Inspect a bounded local CSV, TSV, JSON-array, or JSONL dataset. "
                "Returns row count, ordered columns, simple type/nullability schema, "
                "and a bounded row preview without executing data-file content."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "preview_rows": {"type": "integer"},
                },
                "required": ["file_path"],
            },
            is_read_only=True,
            strict=True,
            max_result_size_chars=200_000,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str):
            return PermissionResult.allow()
        try:
            path = context.ensure_allowed_path(file_path)
        except ToolPermissionError as exc:
            return PermissionResult.deny(str(exc))
        return sensitive_path_permission(path, operation="read")

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        preview_rows = tool_input.get("preview_rows", 20)
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        if not isinstance(preview_rows, int) or not 1 <= preview_rows <= _MAX_PREVIEW_ROWS:
            raise ToolInputError(
                f"preview_rows must be an integer between 1 and {_MAX_PREVIEW_ROWS}"
            )

        path = context.ensure_allowed_path(file_path)
        if not path.exists() or not path.is_file():
            raise ToolInputError(f"data file does not exist: {path}")
        fmt, size, rows, columns = _load_rows(path)
        context.mark_file_read(path)
        return ToolResult(
            name="DataInspect",
            output={
                "type": "data_preview",
                "filePath": str(path),
                "format": fmt,
                "sizeBytes": size,
                "rowCount": len(rows),
                "columns": _column_schema(rows, columns),
                "rows": rows[:preview_rows],
                "previewRows": min(len(rows), preview_rows),
                "previewTruncated": len(rows) > preview_rows,
            },
        )


class DataTransformTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="DataTransform",
            permission_policy="checked",
            description=(
                "Transform a previously inspected/read CSV, TSV, JSON-array, or JSONL "
                "dataset into a new data file using optional column selection and "
                "exact-match scalar filters. Existing output files are never overwritten."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "filters": {"type": "object"},
                },
                "required": ["input_path", "output_path"],
            },
            is_destructive=True,
            strict=True,
            max_result_size_chars=100_000,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        input_path = tool_input.get("input_path")
        output_path = tool_input.get("output_path")
        if not isinstance(input_path, str) or not isinstance(output_path, str):
            return PermissionResult.allow()
        try:
            source = context.ensure_allowed_path(input_path)
            target = context.ensure_allowed_path(output_path)
        except ToolPermissionError as exc:
            return PermissionResult.deny(str(exc))

        source_result = sensitive_path_permission(source, operation="read")
        if source_result.behavior.value != "allow":
            return source_result
        target_result = sensitive_path_permission(target, operation="write")
        if target_result.behavior.value != "allow":
            return target_result
        protected_reason = protected_write_reason(target, existing=target.exists())
        if protected_reason:
            return PermissionResult.ask(message=protected_reason)
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        input_path = tool_input["input_path"]
        output_path = tool_input["output_path"]
        columns_input = tool_input.get("columns")
        filters = tool_input.get("filters", {})

        if not isinstance(input_path, str) or not isinstance(output_path, str):
            raise ToolInputError("input_path and output_path must be strings")
        if columns_input is not None and (
            not isinstance(columns_input, list)
            or not columns_input
            or any(not isinstance(item, str) or not item for item in columns_input)
        ):
            raise ToolInputError("columns must be a non-empty array of column names")
        if not isinstance(filters, dict):
            raise ToolInputError("filters must be an object")

        for key, value in filters.items():
            if not isinstance(key, str) or not key:
                raise ToolInputError("filter names must be non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ToolInputError("filter values must be scalar JSON values")

        source = context.ensure_allowed_path(input_path)
        target = context.ensure_allowed_path(output_path)
        if source == target:
            raise ToolInputError("input_path and output_path must be different")
        if not source.exists() or not source.is_file():
            raise ToolInputError(f"input data file does not exist: {source}")
        if target.exists():
            raise ToolInputError("refusing to overwrite existing output data file")
        if not context.was_file_read_and_unchanged(source):
            raise ToolInputError(
                "refusing to transform: input data file must be read/inspected first "
                "and unchanged since that read"
            )

        input_format, _size, rows, available_columns = _load_rows(source)
        output_format = _format_name(target)
        available_set = set(available_columns)

        columns = list(columns_input) if columns_input is not None else available_columns
        if not columns and output_format in {"csv", "tsv"}:
            raise ToolInputError(
                "columns are required to write delimited output from a dataset with no discoverable columns"
            )
        if len(set(columns)) != len(columns):
            raise ToolInputError("columns contains duplicate names")
        unknown_columns = (
            [column for column in columns if column not in available_set]
            if available_columns
            else []
        )
        if unknown_columns:
            raise ToolInputError(
                "unknown column(s): " + ", ".join(unknown_columns)
            )
        if filters and not available_columns:
            raise ToolInputError(
                "filters require a dataset with discoverable source columns"
            )
        unknown_filters = [name for name in filters if name not in available_set]
        if unknown_filters:
            raise ToolInputError(
                "unknown filter column(s): " + ", ".join(unknown_filters)
            )

        filtered = [
            row
            for row in rows
            if all(row.get(name) == expected for name, expected in filters.items())
        ]
        payload = _render_rows(
            filtered,
            columns=columns,
            output_format=output_format,
        )
        _write_new_text(target, payload)
        context.mark_file_read(target)
        return ToolResult(
            name="DataTransform",
            output={
                "type": "data_transform",
                "filePath": str(target),
                "inputPath": str(source),
                "inputFormat": input_format,
                "outputFormat": output_format,
                "inputRows": len(rows),
                "outputRows": len(filtered),
                "columns": columns,
                "filterColumns": sorted(filters),
            },
        )
