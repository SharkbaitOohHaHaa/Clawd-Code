from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


PYRIGHT_VERSION = "1.1.414"
_ALLOWED_OPERATIONS = {
    "document_symbols": "textDocument/documentSymbol",
    "hover": "textDocument/hover",
    "definition": "textDocument/definition",
    "references": "textDocument/references",
}


class PyrightLSPError(RuntimeError):
    pass


def default_pyright_package_root() -> Path:
    return Path.home() / ".clawd" / "lsp" / "pyright" / "node_modules" / "pyright"


def _load_installed_version(package_root: Path) -> str:
    package_json = package_root / "package.json"
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PyrightLSPError(f"cannot read pinned Pyright package metadata: {exc}") from exc
    return str(data.get("version") or "")


def pyright_runtime_available() -> bool:
    node = shutil.which("node")
    package_root = default_pyright_package_root()
    if not node:
        return False
    if not (package_root / "langserver.index.js").is_file():
        return False
    try:
        return _load_installed_version(package_root) == PYRIGHT_VERSION
    except PyrightLSPError:
        return False


class PyrightLSPClient:
    """One-request-at-a-time read-only Pyright stdio client.

    A fresh language-server process is used for each request. This is slower
    than a persistent session, but it makes lifecycle, workspace scope, and
    cleanup deterministic and prevents stale cross-workspace state.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        package_root: str | Path | None = None,
        node_executable: str | Path | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.package_root = Path(package_root or default_pyright_package_root()).expanduser().resolve()
        node = str(node_executable) if node_executable is not None else shutil.which("node")
        self.node_executable = Path(node).expanduser().resolve() if node else None
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _validate_runtime(self) -> tuple[Path, Path]:
        if self.node_executable is None or not self.node_executable.is_file():
            raise PyrightLSPError("Node.js executable is not available")
        if _load_installed_version(self.package_root) != PYRIGHT_VERSION:
            raise PyrightLSPError(
                f"Pyright runtime must be exactly {PYRIGHT_VERSION}; "
                f"found {_load_installed_version(self.package_root) or 'unknown'}"
            )
        server = self.package_root / "langserver.index.js"
        if not server.is_file():
            raise PyrightLSPError(f"Pyright language-server entrypoint is missing: {server}")
        return self.node_executable, server

    def request(
        self,
        operation: str,
        *,
        file_path: str | Path,
        line: int | None = None,
        character: int | None = None,
    ) -> dict[str, Any]:
        method = _ALLOWED_OPERATIONS.get(operation)
        if method is None:
            raise PyrightLSPError(f"unsupported LSP operation: {operation}")

        path = Path(file_path).expanduser().resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PyrightLSPError(f"LSP file is outside workspace: {path}") from exc
        if path.suffix.lower() not in {".py", ".pyi"}:
            raise PyrightLSPError("Pyright LSP only accepts .py and .pyi files")
        if not path.is_file():
            raise PyrightLSPError(f"LSP file not found: {path}")

        needs_position = operation in {"hover", "definition", "references"}
        if needs_position:
            if not isinstance(line, int) or line < 1:
                raise PyrightLSPError("line must be an integer >= 1")
            if not isinstance(character, int) or character < 1:
                raise PyrightLSPError("character must be an integer >= 1")

        node, server = self._validate_runtime()
        process = subprocess.Popen(
            [str(node), str(server), "--stdio"],
            cwd=str(self.workspace_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise PyrightLSPError("failed to open Pyright stdio pipes")

        incoming: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()
        stderr_lines: list[str] = []
        write_lock = threading.Lock()

        def send(message: dict[str, Any]) -> None:
            raw = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            framed = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw
            with write_lock:
                try:
                    process.stdin.write(framed)
                    process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise PyrightLSPError(f"Pyright stdin closed unexpectedly: {exc}") from exc

        def read_messages() -> None:
            try:
                while True:
                    headers: dict[str, str] = {}
                    while True:
                        line_bytes = process.stdout.readline()
                        if not line_bytes:
                            incoming.put(None)
                            return
                        if line_bytes in {b"\r\n", b"\n"}:
                            break
                        decoded = line_bytes.decode("ascii", errors="replace").strip()
                        if ":" in decoded:
                            key, value = decoded.split(":", 1)
                            headers[key.lower().strip()] = value.strip()
                    length_text = headers.get("content-length")
                    if not length_text:
                        continue
                    body = process.stdout.read(int(length_text))
                    if len(body) != int(length_text):
                        incoming.put(PyrightLSPError("Pyright stdout ended mid-message"))
                        return
                    incoming.put(json.loads(body.decode("utf-8")))
            except BaseException as exc:  # pragma: no cover - transport failure path
                incoming.put(exc)

        def read_stderr() -> None:
            try:
                for raw_line in iter(process.stderr.readline, b""):
                    if len(stderr_lines) < 20:
                        stderr_lines.append(raw_line.decode("utf-8", errors="replace").rstrip())
            except OSError:
                return

        reader = threading.Thread(target=read_messages, name="clawd-pyright-stdout", daemon=True)
        err_reader = threading.Thread(target=read_stderr, name="clawd-pyright-stderr", daemon=True)
        reader.start()
        err_reader.start()

        next_id = 0
        diagnostics: list[dict[str, Any]] = []

        def handle_server_message(message: dict[str, Any]) -> None:
            method_name = message.get("method")
            if method_name == "textDocument/publishDiagnostics":
                params = message.get("params")
                if isinstance(params, dict) and params.get("uri") == path.as_uri():
                    raw = params.get("diagnostics")
                    if isinstance(raw, list):
                        diagnostics[:] = [item for item in raw if isinstance(item, dict)]
                return
            if "id" not in message or not isinstance(method_name, str):
                return
            request_id = message["id"]
            if method_name == "workspace/configuration":
                params = message.get("params")
                items = params.get("items") if isinstance(params, dict) else []
                result = [None for _ in items] if isinstance(items, list) else []
                send({"jsonrpc": "2.0", "id": request_id, "result": result})
            elif method_name in {
                "client/registerCapability",
                "client/unregisterCapability",
                "window/workDoneProgress/create",
            }:
                send({"jsonrpc": "2.0", "id": request_id, "result": None})
            elif method_name == "workspace/workspaceFolders":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": [{"uri": self.workspace_root.as_uri(), "name": self.workspace_root.name}],
                    }
                )
            elif method_name == "workspace/applyEdit":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "applied": False,
                            "failureReason": "Clawd LSP runtime is read-only",
                        },
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "client method not supported"},
                    }
                )

        def wait_for_response(request_id: int, timeout: float | None = None) -> Any:
            deadline = time.monotonic() + (timeout or self.timeout_seconds)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PyrightLSPError(f"Pyright request timed out (id={request_id})")
                try:
                    message = incoming.get(timeout=remaining)
                except queue.Empty as exc:
                    raise PyrightLSPError(f"Pyright request timed out (id={request_id})") from exc
                if message is None:
                    stderr = "\n".join(stderr_lines[-5:])
                    raise PyrightLSPError(
                        "Pyright exited before replying"
                        + (f": {stderr}" if stderr else "")
                    )
                if isinstance(message, BaseException):
                    raise PyrightLSPError(str(message)) from message
                if message.get("id") == request_id and ("result" in message or "error" in message):
                    if "error" in message:
                        raise PyrightLSPError(f"Pyright returned JSON-RPC error: {message['error']}")
                    return message.get("result")
                handle_server_message(message)

        def rpc_request(method_name: str, params: dict[str, Any] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method_name,
            }
            if params is not None:
                payload["params"] = params
            send(payload)
            return wait_for_response(request_id)

        def notify(method_name: str, params: dict[str, Any] | None = None) -> None:
            payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method_name}
            if params is not None:
                payload["params"] = params
            send(payload)

        try:
            rpc_request(
                "initialize",
                {
                    "processId": None,
                    "clientInfo": {"name": "Clawd Codex", "version": "0.1.0"},
                    "rootUri": self.workspace_root.as_uri(),
                    "workspaceFolders": [
                        {"uri": self.workspace_root.as_uri(), "name": self.workspace_root.name}
                    ],
                    "capabilities": {
                        "workspace": {"configuration": False, "workspaceFolders": True},
                        "textDocument": {
                            "documentSymbol": {},
                            "hover": {},
                            "definition": {},
                            "references": {},
                        },
                    },
                },
            )
            notify("initialized", {})
            text = path.read_text(encoding="utf-8", errors="replace")
            notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": path.as_uri(),
                        "languageId": "python",
                        "version": 1,
                        "text": text,
                    }
                },
            )

            params: dict[str, Any] = {"textDocument": {"uri": path.as_uri()}}
            if needs_position:
                assert line is not None and character is not None
                params["position"] = {"line": line - 1, "character": character - 1}
            if operation == "references":
                params["context"] = {"includeDeclaration": True}

            result = rpc_request(method, params)

            drain_until = time.monotonic() + 0.25
            while time.monotonic() < drain_until:
                try:
                    message = incoming.get(timeout=max(0.0, drain_until - time.monotonic()))
                except queue.Empty:
                    break
                if message is None:
                    break
                if isinstance(message, BaseException):
                    break
                handle_server_message(message)

            notify("textDocument/didClose", {"textDocument": {"uri": path.as_uri()}})
            return {
                "operation": operation,
                "result": result,
                "diagnostics": diagnostics,
                "server": {"name": "pyright", "version": PYRIGHT_VERSION},
            }
        finally:
            try:
                if process.poll() is None:
                    try:
                        rpc_request("shutdown")
                    except PyrightLSPError:
                        pass
                    try:
                        notify("exit")
                    except PyrightLSPError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1.0)
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        stream.close()
                    except OSError:
                        pass
