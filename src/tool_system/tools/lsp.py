from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..permissions import sensitive_path_permission
from ..protocol import ToolResult
from ..pyright_lsp import PyrightLSPClient, PyrightLSPError
from ..registry import ToolSpec


_ALLOWED_OPERATIONS = frozenset(
    {"document_symbols", "hover", "definition", "references"}
)


class LSPClient(Protocol):
    def request(
        self,
        operation: str,
        *,
        file_path: str | Path,
        line: int | None = None,
        character: int | None = None,
    ) -> dict[str, Any]: ...


class LSPTool:
    def _active_workspace_root(self, context: ToolContext) -> Path:
        return (context.worktree_root or context.workspace_root).resolve()

    def _resolve_python_file(self, file_path: str, context: ToolContext) -> Path:
        path = context.ensure_allowed_path(file_path)
        active_root = self._active_workspace_root(context)
        try:
            path.relative_to(active_root)
        except ValueError as exc:
            raise ToolPermissionError(
                f"LSP access is limited to the active workspace: {path}"
            ) from exc
        if path.suffix.lower() not in {".py", ".pyi"}:
            raise ToolInputError("LSP file_path must point to a .py or .pyi file")
        return path

    def _file_uri_within_workspace(self, uri: str, workspace_root: Path) -> bool:
        try:
            parsed = urlparse(uri)
            if parsed.scheme != "file":
                return False
            path = Path(url2pathname(unquote(parsed.path))).resolve()
            path.relative_to(workspace_root)
            return True
        except (OSError, ValueError):
            return False

    def _sanitize_workspace_locations(
        self,
        operation: str,
        response: dict[str, Any],
        workspace_root: Path,
    ) -> dict[str, Any]:
        if operation not in {"document_symbols", "definition", "references"}:
            return response
        raw_result = response.get("result")
        if raw_result is None:
            return response

        def item_allowed(item: Any) -> bool:
            if not isinstance(item, dict):
                return True
            for key in ("uri", "targetUri"):
                uri = item.get(key)
                if isinstance(uri, str):
                    return self._file_uri_within_workspace(uri, workspace_root)
            location = item.get("location")
            if isinstance(location, dict):
                uri = location.get("uri")
                if isinstance(uri, str):
                    return self._file_uri_within_workspace(uri, workspace_root)
            return True

        if isinstance(raw_result, list):
            kept = [item for item in raw_result if item_allowed(item)]
            omitted = len(raw_result) - len(kept)
            cleaned = dict(response)
            cleaned["result"] = kept
            if omitted:
                cleaned["externalLocationsOmitted"] = omitted
            return cleaned

        if isinstance(raw_result, dict) and not item_allowed(raw_result):
            cleaned = dict(response)
            cleaned["result"] = None
            cleaned["externalLocationsOmitted"] = 1
            return cleaned
        return response

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str):
            return PermissionResult.allow()
        try:
            path = self._resolve_python_file(file_path, context)
        except ToolPermissionError as exc:
            return PermissionResult.deny(str(exc))
        except ToolInputError:
            return PermissionResult.allow()
        return sensitive_path_permission(path, operation="read")

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="LSP",
            permission_policy="checked",
            description=(
                "Query the pinned local Pyright language server using bounded "
                "read-only Python code-intelligence operations."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_OPERATIONS),
                    },
                    "file_path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "character": {"type": "integer", "minimum": 1},
                },
                "required": ["operation", "file_path"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        operation = tool_input.get("operation")
        file_path = tool_input.get("file_path")
        line = tool_input.get("line")
        character = tool_input.get("character")

        if operation not in _ALLOWED_OPERATIONS:
            raise ToolInputError(
                "operation must be one of: "
                + ", ".join(sorted(_ALLOWED_OPERATIONS))
            )
        if not isinstance(file_path, str) or not file_path.strip():
            raise ToolInputError("file_path must be a non-empty string")

        needs_position = operation in {"hover", "definition", "references"}
        if needs_position:
            if not isinstance(line, int) or line < 1:
                raise ToolInputError("line must be an integer >= 1 for this operation")
            if not isinstance(character, int) or character < 1:
                raise ToolInputError(
                    "character must be an integer >= 1 for this operation"
                )
        elif line is not None or character is not None:
            raise ToolInputError(
                "line and character are only valid for hover, definition, and references"
            )

        path = self._resolve_python_file(file_path, context)
        if not path.exists():
            return ToolResult(
                name="LSP",
                output={"error": f"file not found: {path}"},
                is_error=True,
            )
        if not path.is_file():
            return ToolResult(
                name="LSP",
                output={"error": f"path is not a file: {path}"},
                is_error=True,
            )

        active_root = self._active_workspace_root(context)
        client: LSPClient
        if context.lsp_client is not None:
            client = context.lsp_client
        else:
            client = PyrightLSPClient(active_root)

        try:
            output = client.request(
                operation,
                file_path=path,
                line=line,
                character=character,
            )
        except PyrightLSPError as exc:
            return ToolResult(
                name="LSP",
                output={"error": str(exc)},
                is_error=True,
            )

        output = self._sanitize_workspace_locations(
            operation,
            output,
            active_root,
        )

        return ToolResult(
            name="LSP",
            output={
                "operation": operation,
                "filePath": str(path),
                "response": output,
            },
        )
