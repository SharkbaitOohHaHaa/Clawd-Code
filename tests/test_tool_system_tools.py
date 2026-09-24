from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.tool_system.context import ToolContext
from src.tool_system.mcp_resource_runtime import MCPToolPolicy
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools import (
    AskUserQuestionTool,
    BashTool,
    BriefTool,
    ConfigTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    LSPTool,
    MCPTool,
    ListMcpToolsTool,
    ListMcpResourcesTool,
    NotebookEditTool,
    ReadMcpResourceTool,
    SkillTool,
    SleepTool,
    TodoWriteTool,
    StructuredOutputTool,
    TaskStopTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskUpdateTool,
    ToolSearchTool,
    WebFetchTool,
    WebSearchTool,
    TeamCreateTool,
    TeamDeleteTool,
    EnterWorktreeTool,
    ExitWorktreeTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
)


class ToolSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class TestReadTool(ToolSystemTests):
    def test_read_returns_cat_n_format(self) -> None:
        p = self.root / "a.txt"
        p.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = FileReadTool()
        out = tool.run({"file_path": str(p), "offset": 2, "limit": 2}, self.ctx).output
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["file"]["content"], "2\tline2\n3\tline3")

    def test_read_allows_relative_path_under_workspace(self) -> None:
        p = self.root / "a.txt"
        p.write_text("x\n", encoding="utf-8")
        tool = FileReadTool()
        out = tool.run({"file_path": "a.txt", "limit": 10}, self.ctx).output
        self.assertEqual(out["type"], "text")
        self.assertIn("1\tx", out["file"]["content"])

    def test_read_returns_file_unchanged_stub(self) -> None:
        p = self.root / "same.txt"
        p.write_text("line\n", encoding="utf-8")
        tool = FileReadTool()
        first = tool.run({"file_path": str(p), "limit": 10}, self.ctx).output
        self.assertEqual(first["type"], "text")
        second = tool.run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(second["type"], "file_unchanged")

    def test_read_fingerprint_detects_same_size_immediate_rewrite(self) -> None:
        p = self.root / "fingerprint.txt"
        p.write_text("AAAA\n", encoding="utf-8")
        FileReadTool().run({"file_path": str(p)}, self.ctx)
        p.write_text("BBBB\n", encoding="utf-8")
        self.assertFalse(self.ctx.was_file_read_and_unchanged(p))

    def test_read_notebook(self) -> None:
        p = self.root / "nb.ipynb"
        p.write_text('{"cells":[{"cell_type":"markdown","source":["hi"]}]}', encoding="utf-8")
        out = FileReadTool().run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(out["type"], "notebook")
        self.assertEqual(len(out["file"]["cells"]), 1)

    def test_read_pdf(self) -> None:
        p = self.root / "x.pdf"
        p.write_bytes(b"%PDF-1.4\n1 0 obj\n")
        out = FileReadTool().run({"file_path": str(p)}, self.ctx).output
        self.assertEqual(out["type"], "pdf")

    def test_read_blocks_device_paths(self) -> None:
        with self.assertRaises(Exception):
            FileReadTool().run({"file_path": "/dev/zero"}, self.ctx)


class TestNotebookEditTool(ToolSystemTests):
    def _write_notebook(self, name: str = "notebook.ipynb") -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": 7,
                            "id": "code-1",
                            "metadata": {"tag": "keep"},
                            "outputs": [
                                {"output_type": "stream", "name": "stdout", "text": ["old\n"]}
                            ],
                            "source": ["print('old')\n"],
                        },
                        {
                            "cell_type": "markdown",
                            "id": "md-1",
                            "metadata": {"note": "keep"},
                            "source": ["old text\n"],
                        },
                    ],
                    "metadata": {"kernelspec": {"name": "python3"}},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _read_first(self, path: Path) -> None:
        result = FileReadTool().run({"file_path": str(path)}, self.ctx)
        self.assertFalse(result.is_error)

    def test_replace_code_cell_preserves_metadata_and_clears_stale_outputs(self) -> None:
        path = self._write_notebook()
        self._read_first(path)

        result = NotebookEditTool().run(
            {
                "notebook_path": str(path),
                "cell_id": "code-1",
                "new_source": "print('new')\nprint('two')",
                "edit_mode": "replace",
            },
            self.ctx,
        )

        data = json.loads(path.read_text(encoding="utf-8"))
        cell = data["cells"][0]
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["cellIndex"], 0)
        self.assertEqual(cell["source"], ["print('new')\n", "print('two')"])
        self.assertEqual(cell["metadata"], {"tag": "keep"})
        self.assertEqual(cell["outputs"], [])
        self.assertIsNone(cell["execution_count"])
        self.assertEqual(data["metadata"], {"kernelspec": {"name": "python3"}})

    def test_replace_markdown_with_code_removes_incompatible_fields(self) -> None:
        path = self._write_notebook()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cells"][1]["attachments"] = {
            "image.png": {"image/png": "encoded-placeholder"}
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        self._read_first(path)

        NotebookEditTool().run(
            {
                "notebook_path": str(path),
                "cell_id": "md-1",
                "new_source": "value = 1\n",
                "cell_type": "code",
                "edit_mode": "replace",
            },
            self.ctx,
        )

        cell = json.loads(path.read_text(encoding="utf-8"))["cells"][1]
        self.assertEqual(cell["cell_type"], "code")
        self.assertEqual(cell["source"], ["value = 1\n"])
        self.assertEqual(cell["metadata"], {"note": "keep"})
        self.assertNotIn("attachments", cell)
        self.assertEqual(cell["outputs"], [])
        self.assertIsNone(cell["execution_count"])

    def test_insert_after_cell_and_delete_target(self) -> None:
        path = self._write_notebook()
        self._read_first(path)
        tool = NotebookEditTool()

        inserted = tool.run(
            {
                "notebook_path": str(path),
                "cell_id": "code-1",
                "new_source": "# inserted\n",
                "cell_type": "markdown",
                "edit_mode": "insert",
            },
            self.ctx,
        )
        inserted_id = inserted.output["cellId"]
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(inserted.output["cellIndex"], 1)
        self.assertEqual(data["cells"][1]["id"], inserted_id)
        self.assertEqual(data["cells"][1]["source"], ["# inserted\n"])
        self.assertEqual(data["cells"][2]["id"], "md-1")

        deleted = tool.run(
            {
                "notebook_path": str(path),
                "cell_id": "md-1",
                "new_source": "",
                "edit_mode": "delete",
            },
            self.ctx,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(deleted.output["editMode"], "delete")
        self.assertNotIn("md-1", [cell.get("id") for cell in data["cells"]])
        self.assertIn(inserted_id, [cell.get("id") for cell in data["cells"]])

    def test_insert_without_cell_id_uses_beginning_and_requires_cell_type(self) -> None:
        path = self._write_notebook()
        self._read_first(path)
        tool = NotebookEditTool()

        with self.assertRaises(Exception):
            tool.run(
                {
                    "notebook_path": str(path),
                    "new_source": "print('missing type')",
                    "edit_mode": "insert",
                },
                self.ctx,
            )

        result = tool.run(
            {
                "notebook_path": str(path),
                "new_source": "print('first')",
                "cell_type": "code",
                "edit_mode": "insert",
            },
            self.ctx,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result.output["cellIndex"], 0)
        self.assertEqual(data["cells"][0]["cell_type"], "code")
        self.assertEqual(data["cells"][0]["outputs"], [])
        self.assertIsNone(data["cells"][0]["execution_count"])

    def test_requires_absolute_read_unchanged_notebook_path(self) -> None:
        path = self._write_notebook()
        tool = NotebookEditTool()
        payload = {
            "notebook_path": str(path),
            "cell_id": "md-1",
            "new_source": "updated",
        }

        with self.assertRaises(Exception):
            tool.run(payload, self.ctx)

        self._read_first(path)
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw + "\n", encoding="utf-8")
        with self.assertRaises(Exception):
            tool.run(payload, self.ctx)

        relative_payload = dict(payload)
        relative_payload["notebook_path"] = path.name
        with self.assertRaises(Exception):
            tool.run(relative_payload, self.ctx)

    def test_rejects_wrong_extension_and_missing_cell(self) -> None:
        wrong = self.root / "notebook.json"
        wrong.write_text('{"cells":[]}', encoding="utf-8")
        self.ctx.mark_file_read(wrong)
        with self.assertRaises(Exception):
            NotebookEditTool().run(
                {
                    "notebook_path": str(wrong),
                    "cell_id": "missing",
                    "new_source": "x",
                },
                self.ctx,
            )

        path = self._write_notebook()
        self._read_first(path)
        with self.assertRaises(Exception):
            NotebookEditTool().run(
                {
                    "notebook_path": str(path),
                    "cell_id": "missing",
                    "new_source": "x",
                },
                self.ctx,
            )

    def test_declares_checked_permission_contract(self) -> None:
        tool = NotebookEditTool()
        path = self._write_notebook()
        spec = tool.spec()
        decision = tool.check_permissions(
            {
                "notebook_path": str(path),
                "cell_id": "code-1",
                "new_source": "x",
            },
            self.ctx,
        )
        self.assertEqual(spec.permission_policy, "checked")
        self.assertTrue(spec.is_destructive)
        self.assertEqual(decision.behavior.value, "allow")


class TestWriteTool(ToolSystemTests):
    def test_write_creates_file(self) -> None:
        tool = FileWriteTool()
        p = self.root / "b.txt"
        out = tool.run({"file_path": str(p), "content": "hello"}, self.ctx).output
        self.assertTrue(p.exists())
        self.assertEqual(out["type"], "create")
        self.assertEqual(out["filePath"], str(p))

    def test_write_requires_read_before_overwrite(self) -> None:
        p = self.root / "c.txt"
        p.write_text("old", encoding="utf-8")
        tool = FileWriteTool()
        with self.assertRaises(Exception):
            tool.run({"file_path": str(p), "content": "new"}, self.ctx)

        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        tool.run({"file_path": str(p), "content": "new"}, self.ctx)
        self.assertEqual(p.read_text(encoding="utf-8"), "new")

    def test_write_blocks_docs_by_default(self) -> None:
        """Writing .md files should require permission when allow_docs is False."""
        tool = FileWriteTool()
        p = self.root / "README.md"
        # Permission check should return 'ask' behavior
        result = tool.check_permissions({"file_path": str(p), "content": "x"}, self.ctx)
        self.assertEqual(result.behavior.value, "ask")
        # But run() itself should NOT raise - it just proceeds (permission is checked elsewhere)
        # Note: run() will still succeed because permission checking moved to check_permissions()


class TestEditTool(ToolSystemTests):
    def test_edit_requires_read(self) -> None:
        p = self.root / "d.txt"
        p.write_text("hello world", encoding="utf-8")
        tool = FileEditTool()
        with self.assertRaises(Exception):
            tool.run({"file_path": str(p), "old_string": "world", "new_string": "you"}, self.ctx)

    def test_edit_replaces_unique(self) -> None:
        p = self.root / "e.txt"
        p.write_text("hello world", encoding="utf-8")
        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        out = FileEditTool().run({"file_path": str(p), "old_string": "world", "new_string": "you"}, self.ctx).output
        self.assertEqual(out["filePath"], str(p))
        self.assertEqual(out["replaceAll"], False)
        self.assertEqual(p.read_text(encoding="utf-8"), "hello you")

    def test_edit_requires_replace_all_for_non_unique(self) -> None:
        p = self.root / "f.txt"
        p.write_text("a a a", encoding="utf-8")
        FileReadTool().run({"file_path": str(p), "limit": 10}, self.ctx)
        with self.assertRaises(Exception):
            FileEditTool().run({"file_path": str(p), "old_string": "a", "new_string": "b"}, self.ctx)
        FileEditTool().run({"file_path": str(p), "old_string": "a", "new_string": "b", "replace_all": True}, self.ctx)
        self.assertEqual(p.read_text(encoding="utf-8"), "b b b")


class TestGlobTool(ToolSystemTests):
    def test_glob_sorts_by_mtime(self) -> None:
        a = self.root / "x1.py"
        b = self.root / "x2.py"
        a.write_text("a", encoding="utf-8")
        time.sleep(0.01)
        b.write_text("b", encoding="utf-8")
        out = GlobTool().run({"pattern": "*.py", "path": str(self.root), "limit": 10}, self.ctx).output
        self.assertEqual(out["filenames"][0], str(b))
        self.assertEqual(out["filenames"][1], str(a))


class TestGrepTool(ToolSystemTests):
    def test_grep_files_with_matches(self) -> None:
        (self.root / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
        (self.root / "b.txt").write_text("nope\n", encoding="utf-8")
        out = GrepTool().run({"pattern": "hello", "path": str(self.root)}, self.ctx).output
        self.assertEqual(out["mode"], "files_with_matches")
        self.assertEqual(out["numFiles"], 1)
        self.assertIn("a.txt", out["filenames"][0])

    def test_grep_content_mode_with_line_numbers(self) -> None:
        (self.root / "a.txt").write_text("hello\nhello\n", encoding="utf-8")
        out = GrepTool().run({"pattern": "hello", "path": str(self.root), "output_mode": "content", "-n": True}, self.ctx).output
        self.assertIn(":1:", out["content"])


@unittest.skipUnless(shutil.which("bash"), "Bash executable is not available in this test environment")
class TestBashTool(ToolSystemTests):
    def test_bash_echo(self) -> None:
        out = BashTool().run({"command": "echo hello"}, self.ctx).output
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hello", out["stdout"])

    def test_bash_blocks_sudo(self) -> None:
        with self.assertRaises(Exception):
            BashTool().run({"command": "sudo echo nope"}, self.ctx)


class TestWebFetchTool(ToolSystemTests):
    def test_web_fetch_blocks_file_scheme(self) -> None:
        with self.assertRaises(Exception):
            WebFetchTool().run({"url": "file:///etc/passwd"}, self.ctx)

    def test_web_fetch_blocks_redirect_to_private_host(self) -> None:
        class _RedirectingOpener:
            def __init__(self, handler):
                self.handler = handler

            def open(self, req, timeout=15):
                self.handler.redirect_request(
                    req,
                    None,
                    302,
                    "Found",
                    {},
                    "http://127.0.0.1/secret",
                )
                raise AssertionError("private redirect was followed")

        def fake_build_opener(handler):
            return _RedirectingOpener(handler)

        def fake_getaddrinfo(host, *args, **kwargs):
            if host == "example.com":
                return [(None, None, None, None, ("93.184.216.34", 0))]
            if host == "127.0.0.1":
                return [(None, None, None, None, ("127.0.0.1", 0))]
            raise AssertionError(f"unexpected host lookup: {host}")

        with patch.object(socket, "getaddrinfo", side_effect=fake_getaddrinfo):
            with patch.object(urllib.request, "build_opener", side_effect=fake_build_opener):
                with self.assertRaisesRegex(
                    Exception,
                    "refusing to fetch localhost/private network URLs",
                ):
                    WebFetchTool().run({"url": "https://example.com/"}, self.ctx)

    def test_web_fetch_extracts_text(self) -> None:
        html_doc = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"

        class _Resp(io.BytesIO):
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _Opener:
            def open(self, req, timeout=15):
                return _Resp(html_doc.encode("utf-8"))

        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ):
            with patch.object(
                urllib.request,
                "build_opener",
                return_value=_Opener(),
            ):
                out = WebFetchTool().run({"url": "https://example.com/"}, self.ctx).output
                self.assertIn("Title", out["content"])
                self.assertIn("Hello world", out["content"])


class TestWebSearchTool(ToolSystemTests):
    def test_web_search_parses_results(self) -> None:
        html_doc = """
        <a class="result__a" href="https://example.com/">Example</a>
        <a class="result__snippet">Snippet</a>
        """

        class _Resp(io.BytesIO):
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch.object(urllib.request, "urlopen", return_value=_Resp(html_doc.encode("utf-8"))):
            out = WebSearchTool().run({"query": "example", "num": 1}, self.ctx).output
            self.assertEqual(len(out["results"]), 1)
            self.assertEqual(out["results"][0]["url"], "https://example.com/")


class TestSleepTool(ToolSystemTests):
    def test_sleep_short(self) -> None:
        start = time.time()
        SleepTool().run({"seconds": 0.01}, self.ctx)
        self.assertGreaterEqual(time.time() - start, 0.0)


class TestTaskStopTool(ToolSystemTests):
    def test_task_stop(self) -> None:
        def target(stop_event):
            while not stop_event.is_set():
                time.sleep(0.01)

        task = self.ctx.task_manager.start(name="loop", target=target)
        out = TaskStopTool().run({"task_id": task.task_id}, self.ctx).output
        self.assertTrue(out["stopped"])


class TestConfigTool(ToolSystemTests):
    def test_config_get_set_roundtrip(self) -> None:
        from src import config as config_mod

        cfg_path = self.root / "config.json"
        cfg_path.write_text(json.dumps(config_mod.get_default_config()), encoding="utf-8")
        with patch("src.config.get_config_path", return_value=cfg_path):
            get_out = ConfigTool().run({"setting": "default_provider"}, self.ctx).output
            self.assertEqual(get_out["operation"], "get")
            set_out = ConfigTool().run({"setting": "default_provider", "value": "openai"}, self.ctx).output
            self.assertEqual(set_out["operation"], "set")
            self.assertEqual(ConfigTool().run({"setting": "default_provider"}, self.ctx).output["value"], "openai")


class TestMCPTool(ToolSystemTests):
    def test_list_mcp_tools_exposes_only_client_sanitized_policy_view(self) -> None:
        class Client:
            def list_tools(self) -> list[dict[str, Any]]:
                return [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "inputSchema": {"type": "object"},
                        "contractSha256": "a" * 64,
                        "trustedContract": True,
                        "localPolicy": {
                            "approval": "allow",
                            "readOnly": True,
                            "openWorld": False,
                        },
                        "serverHints": {"readOnlyHint": False},
                    }
                ]

        self.ctx.mcp_clients["srv"] = Client()
        registry = ToolRegistry(tools=[ListMcpToolsTool()])
        out = registry.dispatch(
            ToolCall(name="ListMcpToolsTool", input={"server": "srv"}),
            self.ctx,
        )
        self.assertFalse(out.is_error)
        self.assertEqual(out.output[0]["name"], "lookup")
        self.assertEqual(out.output[0]["server"], "srv")
        self.assertTrue(out.output[0]["trustedContract"])

    def test_mcp_read_only_closed_world_policy_can_run_without_prompt(self) -> None:
        policy = MCPToolPolicy(
            name="lookup",
            approval="allow",
            read_only=True,
            open_world=False,
            contract_sha256="a" * 64,
        )

        class Client:
            def can_call_tool(self, tool_name: str) -> bool:
                return tool_name == "lookup"

            def tool_policy(self, tool_name: str):
                return policy if tool_name == "lookup" else None

            def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": args["query"]}],
                    "structuredContent": None,
                    "resultType": "complete",
                }

        self.ctx.mcp_clients["srv"] = Client()
        registry = ToolRegistry(tools=[MCPTool()])
        out = registry.dispatch(
            ToolCall(
                name="MCP",
                input={
                    "server": "srv",
                    "tool": "lookup",
                    "input": {"query": "hello"},
                },
            ),
            self.ctx,
        )
        self.assertFalse(out.is_error)
        self.assertEqual(out.output["output"]["content"][0]["text"], "hello")

    def test_mcp_mutating_policy_requires_explicit_confirmation_marker(self) -> None:
        policy = MCPToolPolicy(
            name="write_record",
            approval="ask",
            read_only=False,
            open_world=False,
            contract_sha256="b" * 64,
        )

        class Client:
            def can_call_tool(self, tool_name: str) -> bool:
                return tool_name == "write_record"

            def tool_policy(self, tool_name: str):
                return policy if tool_name == "write_record" else None

            def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": "changed"}],
                    "structuredContent": None,
                    "resultType": "complete",
                }

        prompts: list[tuple[str, str, str | None]] = []

        def permission_handler(tool_name: str, message: str, suggestion: str | None):
            prompts.append((tool_name, message, suggestion))
            return True, False

        self.ctx.mcp_clients["srv"] = Client()
        self.ctx.permission_handler = permission_handler
        registry = ToolRegistry(tools=[MCPTool()])
        out = registry.dispatch(
            ToolCall(
                name="MCP",
                input={
                    "server": "srv",
                    "tool": "write_record",
                    "input": {"value": "x"},
                },
            ),
            self.ctx,
        )
        self.assertFalse(out.is_error)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0][0], "MCP")
        self.assertIn("requires explicit approval", prompts[0][1])
        self.assertEqual(prompts[0][2], "require-explicit-yes")

    def test_mcp_denies_tool_until_current_contract_is_advertised(self) -> None:
        class Client:
            def can_call_tool(self, tool_name: str) -> bool:
                return False

            def tool_policy(self, tool_name: str):
                return MCPToolPolicy(
                    name=tool_name,
                    approval="ask",
                    read_only=False,
                    open_world=True,
                    contract_sha256=None,
                )

            def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("call_tool must not run")

        self.ctx.mcp_clients["srv"] = Client()
        registry = ToolRegistry(tools=[MCPTool()])
        out = registry.dispatch(
            ToolCall(
                name="MCP",
                input={"server": "srv", "tool": "x", "input": {}},
            ),
            self.ctx,
        )
        self.assertTrue(out.is_error)
        self.assertIn("list MCP tools first", out.output["error"])


class TestLSPTool(ToolSystemTests):
    def test_lsp_declares_checked_read_only_contract(self) -> None:
        reg = ToolRegistry(tools=[LSPTool()])
        spec = reg.get("LSP").spec()  # type: ignore[union-attr]
        self.assertEqual(spec.permission_policy, "checked")
        self.assertTrue(spec.is_read_only)
        self.assertEqual(
            set(spec.input_schema["properties"]["operation"]["enum"]),
            {"document_symbols", "hover", "definition", "references"},
        )

    def test_lsp_calls_bounded_client(self) -> None:
        source = self.root / "sample.py"
        source.write_text("value = 1\n", encoding="utf-8")

        class Client:
            def request(self, operation: str, *, file_path, line=None, character=None) -> Any:
                return {
                    "operation": operation,
                    "file_path": str(file_path),
                    "line": line,
                    "character": character,
                }

        self.ctx.lsp_client = Client()
        out = LSPTool().run(
            {
                "operation": "hover",
                "file_path": str(source),
                "line": 1,
                "character": 1,
            },
            self.ctx,
        ).output
        response = out["response"]
        self.assertEqual(response["operation"], "hover")
        self.assertEqual(response["file_path"], str(source))
        self.assertEqual(response["line"], 1)
        self.assertEqual(response["character"], 1)

    def test_lsp_uses_active_worktree_as_workspace_root(self) -> None:
        worktree = self.root / ".git" / "clawd-worktrees" / "demo"
        worktree.mkdir(parents=True)
        source = worktree / "sample.py"
        source.write_text("value = 1\n", encoding="utf-8")
        original = self.root / "original.py"
        original.write_text("value = 2\n", encoding="utf-8")
        self.ctx.worktree_root = worktree
        self.ctx.cwd = worktree

        with patch("src.tool_system.tools.lsp.PyrightLSPClient") as client_cls:
            client_cls.return_value.request.return_value = {
                "operation": "document_symbols",
                "result": [],
                "diagnostics": [],
                "server": {"name": "fake", "version": "test"},
            }
            result = LSPTool().run(
                {"operation": "document_symbols", "file_path": str(source)},
                self.ctx,
            )
        self.assertFalse(result.is_error)
        client_cls.assert_called_once_with(worktree.resolve())

        permission = LSPTool().check_permissions(
            {"operation": "document_symbols", "file_path": str(original)},
            self.ctx,
        )
        self.assertEqual(permission.behavior.value, "deny")
        self.assertIn("active workspace", permission.message)

    def test_lsp_filters_external_result_locations(self) -> None:
        source = self.root / "sample.py"
        source.write_text("value = 1\n", encoding="utf-8")
        outside = self.root.parent / "outside.py"
        outside.write_text("value = 2\n", encoding="utf-8")

        class Client:
            def request(self, operation: str, *, file_path, line=None, character=None) -> Any:
                return {
                    "operation": operation,
                    "result": [
                        {
                            "uri": source.as_uri(),
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 5},
                            },
                        },
                        {
                            "uri": outside.as_uri(),
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 5},
                            },
                        },
                    ],
                    "diagnostics": [],
                    "server": {"name": "fake", "version": "test"},
                }

        try:
            self.ctx.lsp_client = Client()
            out = LSPTool().run(
                {
                    "operation": "definition",
                    "file_path": str(source),
                    "line": 1,
                    "character": 1,
                },
                self.ctx,
            ).output["response"]
            self.assertEqual(len(out["result"]), 1)
            self.assertEqual(out["result"][0]["uri"], source.as_uri())
            self.assertEqual(out["externalLocationsOmitted"], 1)
            self.assertNotIn(str(outside), json.dumps(out))
        finally:
            outside.unlink(missing_ok=True)

    def test_lsp_rejects_workspace_escape(self) -> None:
        outside = self.root.parent / "outside.py"
        outside.write_text("x = 1\n", encoding="utf-8")
        try:
            reg = ToolRegistry(tools=[LSPTool()])
            result = reg.dispatch(
                ToolCall(
                    name="LSP",
                    input={"operation": "document_symbols", "file_path": str(outside)},
                ),
                self.ctx,
            )
            self.assertTrue(result.is_error)
            self.assertIn("outside allowed working directories", result.output["error"])
        finally:
            outside.unlink(missing_ok=True)

    def test_lsp_rejects_non_python_file(self) -> None:
        source = self.root / "sample.txt"
        source.write_text("x\n", encoding="utf-8")
        with self.assertRaisesRegex(Exception, "must point to a .py or .pyi file"):
            LSPTool().run(
                {"operation": "document_symbols", "file_path": str(source)},
                self.ctx,
            )

    def test_lsp_requires_position_for_position_operations(self) -> None:
        source = self.root / "sample.py"
        source.write_text("x = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(Exception, "line must be an integer >= 1"):
            LSPTool().run(
                {"operation": "definition", "file_path": str(source)},
                self.ctx,
            )


class TestSkillTool(ToolSystemTests):
    def test_skill_runs_approved_markdown_skill(self) -> None:
        from src.skills.create import create_skill
        from src.skills.trust_registry import SkillTrustRegistry, default_skill_record

        skills_dir = self.root / "skills"
        skill_file = create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            body="Hello $ARGUMENTS[0]!",
            arguments=["name"],
        )
        skill_root = skill_file.parent
        trust_dir = self.root / "trust"
        trust = SkillTrustRegistry(trust_dir)
        record = default_skill_record(
            name="hello",
            artifact_path=skill_root,
            purpose="tool-system test skill",
        )
        trust.register_quarantined(record, initiator="test", reason="test registration")
        trust.mark_reviewed("hello", reviewed_by="test", initiator="test", reason="test review")
        trust.approve("hello", reviewed_by="test", initiator="test", reason="test approval")
        trust.activate("hello", initiator="test", reason="test activation")

        with patch.dict(
            os.environ,
            {
                "CLAWD_SKILLS_DIR": str(skills_dir),
                "CLAWD_SKILL_TRUST_DIR": str(trust_dir),
            },
        ):
            out = SkillTool().run({"skill": "hello", "args": "bob"}, self.ctx).output
            self.assertTrue(out["success"])
            self.assertIn("Hello bob!", out["prompt"])
            self.assertEqual(out["loadedFrom"], "user")

    def test_skill_blocks_legacy_python_skill(self) -> None:
        skills_dir = self.root / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        marker = self.root / "legacy-ran.txt"
        (skills_dir / "legacy.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n"
            "def run(input, context):\n    return 'hi'\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"CLAWD_SKILLS_DIR": str(skills_dir)}):
            result = SkillTool().run({"skill": "legacy"}, self.ctx)

        self.assertTrue(result.is_error)
        self.assertIn("inactive, or not approved", result.output["error"])
        self.assertFalse(marker.exists())


class TestNewParityTools(ToolSystemTests):
    def test_ask_user_question_uses_handler(self) -> None:
        self.ctx.ask_user = lambda questions: {questions[0]["question"]: "Option A"}
        out = AskUserQuestionTool().run(
            {
                "questions": [
                    {
                        "question": "Choose?",
                        "header": "Choice",
                        "options": [
                            {"label": "Option A", "description": "A"},
                            {"label": "Option B", "description": "B"},
                        ],
                    }
                ]
            },
            self.ctx,
        ).output
        self.assertEqual(out["answers"]["Choose?"], "Option A")

    def test_todo_write(self) -> None:
        out = TodoWriteTool().run(
            {"todos": [{"content": "x", "status": "pending", "activeForm": "Doing x"}]},
            self.ctx,
        ).output
        self.assertEqual(out["newTodos"][0]["content"], "x")

    def test_task_tools_roundtrip(self) -> None:
        created = TaskCreateTool().run({"subject": "T1", "description": "D1"}, self.ctx).output
        task_id = created["task"]["id"]
        listed = TaskListTool().run({}, self.ctx).output
        self.assertEqual(len(listed["tasks"]), 1)
        TaskUpdateTool().run({"taskId": task_id, "status": "completed"}, self.ctx)
        got = TaskGetTool().run({"taskId": task_id}, self.ctx).output
        self.assertEqual(got["task"]["status"], "completed")
        task_out = TaskOutputTool().run({"task_id": task_id}, self.ctx).output
        self.assertEqual(task_out["task"]["task_id"], task_id)

    def test_tool_search(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        out = ToolSearchTool(reg).run({"query": "read"}, self.ctx).output
        self.assertIn("Read", out["matches"])

    def test_cron_tools_roundtrip(self) -> None:
        created = CronCreateTool().run({"cron": "*/5 * * * *", "prompt": "ping"}, self.ctx).output
        cron_id = created["id"]
        listed = CronListTool().run({}, self.ctx).output
        self.assertEqual(len(listed["jobs"]), 1)
        deleted = CronDeleteTool().run({"id": cron_id}, self.ctx).output
        self.assertTrue(deleted["success"])

    def test_structured_output(self) -> None:
        out = StructuredOutputTool().run({"ok": True}, self.ctx).output
        self.assertTrue(out["structured_output"]["ok"])

    def test_mcp_resource_tools_require_advertisement_before_read(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.advertised: set[str] = set()

            def list_resources(self):
                self.advertised = {"x://1"}
                return [{"uri": "x://1", "name": "r1", "mimeType": "text/plain"}]

            def can_read_resource(self, uri: str) -> bool:
                return uri in self.advertised

            def read_resource(self, uri: str):
                return {"contents": [{"uri": uri, "text": "hello"}]}

        self.ctx.mcp_clients["srv"] = Client()
        registry = ToolRegistry(tools=[ListMcpResourcesTool(), ReadMcpResourceTool()])

        denied = registry.dispatch(
            ToolCall(
                name="ReadMcpResourceTool",
                input={"server": "srv", "uri": "x://1"},
            ),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertIn("list resources first", denied.output["error"])

        listed = registry.dispatch(
            ToolCall(name="ListMcpResourcesTool", input={"server": "srv"}),
            self.ctx,
        )
        self.assertFalse(listed.is_error)
        self.assertEqual(listed.output[0]["uri"], "x://1")

        read = registry.dispatch(
            ToolCall(
                name="ReadMcpResourceTool",
                input={"server": "srv", "uri": "x://1"},
            ),
            self.ctx,
        )
        self.assertFalse(read.is_error)
        self.assertEqual(read.output["contents"][0]["text"], "hello")


class TestRegistryAndHelloWorldTool(ToolSystemTests):
    def test_legacy_user_tool_loader_fails_closed(self) -> None:
        user_dir = self.root / "tools"
        user_dir.mkdir(parents=True, exist_ok=True)
        marker = self.root / "executed.txt"
        (user_dir / "hello.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )

        from src.tool_system.loader import load_tools_from_dir

        with self.assertRaises(RuntimeError):
            load_tools_from_dir(user_dir)
        self.assertFalse(marker.exists())


class TestBriefAndAgentTools(ToolSystemTests):
    def test_brief_tool(self) -> None:
        out = BriefTool().run({"text": "abc", "max_chars": 2}, self.ctx).output
        self.assertEqual(out["preview"], "ab…")

    def test_agent_tool_sequences_calls(self) -> None:
        reg = build_default_registry(include_user_tools=False)
        ctx = ToolContext(workspace_root=self.root)
        p = self.root / "x.txt"
        p.write_text("hi", encoding="utf-8")
        call = {"name": "Read", "input": {"file_path": str(p), "limit": 10}}
        out = reg.get("Agent").run({"calls": [call]}, ctx).output  # type: ignore[union-attr]
        self.assertEqual(out["results"][0]["name"], "Read")


class TestTeamTools(ToolSystemTests):
    def test_team_create_roundtrip(self) -> None:
        """Test creating and deleting a team."""
        # Create team
        create_out = TeamCreateTool().run(
            {"team_name": "test-team", "description": "A test team"},
            self.ctx,
        ).output
        self.assertEqual(create_out["team_name"], "test-team")
        self.assertIsNotNone(create_out["lead_agent_id"])
        self.assertEqual(self.ctx.team["team_name"], "test-team")
        self.assertIn("does not spawn or run subagents", TeamCreateTool().spec().description)
        self.assertEqual(self.ctx.task_manager.list(), [])
        self.assertEqual(self.ctx.tasks, {})

        # Verify team file was created
        team_file = self.root / ".clawd" / "team.json"
        self.assertTrue(team_file.exists())

        # Delete team
        delete_out = TeamDeleteTool().run({}, self.ctx).output
        self.assertTrue(delete_out["success"])
        self.assertEqual(delete_out["team_name"], "test-team")
        self.assertIsNone(self.ctx.team)

        # Verify team file was deleted
        self.assertFalse(team_file.exists())

    def test_team_delete_no_team(self) -> None:
        """Test deleting when no team exists."""
        out = TeamDeleteTool().run({}, self.ctx).output
        self.assertFalse(out["success"])
        self.assertEqual(out["message"], "No active team")

    def test_team_create_requires_name(self) -> None:
        """Test team name validation."""
        from src.tool_system.errors import ToolInputError

        with self.assertRaises(ToolInputError):
            TeamCreateTool().run({"team_name": ""}, self.ctx)


class TestWorktreeTools(ToolSystemTests):
    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        if shutil.which("git") is None:
            self.skipTest("Git is unavailable")
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def _init_repo(self) -> None:
        self.assertEqual(self._git("init", "-q").returncode, 0)
        self.assertEqual(self._git("config", "user.email", "clawd-test@example.invalid").returncode, 0)
        self.assertEqual(self._git("config", "user.name", "Clawd Test").returncode, 0)
        (self.root / "a.txt").write_text("one\n", encoding="utf-8")
        self.assertEqual(self._git("add", "a.txt").returncode, 0)
        self.assertEqual(self._git("commit", "-q", "-m", "initial").returncode, 0)

    def _registry_with_approval(self) -> ToolRegistry:
        registry = ToolRegistry(tools=[EnterWorktreeTool(), ExitWorktreeTool()])

        def approve(name: str, message: str, suggestion: str | None):
            self.assertEqual(name, "EnterWorktree")
            self.assertEqual(suggestion, "require-explicit-yes")
            self.assertIn("Create Git worktree branch", message)
            return True, False

        self.ctx.permission_handler = approve
        return registry

    def test_worktree_real_git_roundtrip_preserves_linked_tree(self) -> None:
        self._init_repo()
        registry = self._registry_with_approval()
        entered = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "feature/demo"}),
            self.ctx,
        )
        self.assertFalse(entered.is_error)
        root = Path(entered.output["worktreePath"])
        self.assertEqual(root, self.root / ".git" / "clawd-worktrees" / "feature" / "demo")
        self.assertEqual(entered.output["worktreeBranch"], "clawd/feature/demo")
        self.assertEqual(self.ctx.worktree_root, root)
        self.assertEqual(self.ctx.cwd, root)
        self.assertEqual(self._git("branch", "--show-current", cwd=root).stdout.strip(), "clawd/feature/demo")
        self.assertEqual(self._git("status", "--short").stdout.strip(), "")

        (root / "only-in-worktree.txt").write_text("kept\n", encoding="utf-8")
        exited = registry.dispatch(ToolCall(name="ExitWorktree", input={}), self.ctx)
        self.assertFalse(exited.is_error)
        self.assertTrue(exited.output["preserved"])
        self.assertIsNone(self.ctx.worktree_root)
        self.assertEqual(self.ctx.cwd, self.root)
        self.assertTrue(root.exists())
        self.assertTrue((root / "only-in-worktree.txt").exists())
        self.assertEqual(
            self._git("show-ref", "--verify", "--quiet", "refs/heads/clawd/feature/demo").returncode,
            0,
        )

    def test_worktree_requires_explicit_permission(self) -> None:
        self._init_repo()
        registry = ToolRegistry(tools=[EnterWorktreeTool()])
        denied = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "needs-approval"}),
            self.ctx,
        )
        self.assertTrue(denied.is_error)
        self.assertFalse((self.root / ".git" / "clawd-worktrees" / "needs-approval").exists())
        self.assertNotEqual(
            self._git("show-ref", "--verify", "--quiet", "refs/heads/clawd/needs-approval").returncode,
            0,
        )

    def test_worktree_dirty_original_is_not_copied(self) -> None:
        self._init_repo()
        (self.root / "a.txt").write_text("dirty original\n", encoding="utf-8")
        registry = self._registry_with_approval()
        entered = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "dirty-base"}),
            self.ctx,
        )
        self.assertFalse(entered.is_error)
        self.assertTrue(entered.output["originalWorkspaceDirty"])
        root = Path(entered.output["worktreePath"])
        self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "one\n")
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "dirty original\n")

    def test_worktree_rejects_non_repo_and_duplicate_name(self) -> None:
        registry = ToolRegistry(tools=[EnterWorktreeTool()])
        non_repo = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "demo"}),
            self.ctx,
        )
        self.assertTrue(non_repo.is_error)
        self.assertIn("main Git worktree", non_repo.output["error"])

        self._init_repo()
        registry = self._registry_with_approval()
        first = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "demo"}),
            self.ctx,
        )
        self.assertFalse(first.is_error)
        ExitWorktreeTool().run({}, self.ctx)
        duplicate = registry.dispatch(
            ToolCall(name="EnterWorktree", input={"name": "demo"}),
            self.ctx,
        )
        self.assertTrue(duplicate.is_error)
        self.assertIn("branch already exists", duplicate.output["error"])

    def test_worktree_rejects_parent_path_segments_and_invalid_names(self) -> None:
        from src.tool_system.errors import ToolInputError

        for name in ("../escape", "nested/../../escape", "", "invalid name!", "a" * 65):
            with self.subTest(name=name):
                with self.assertRaises(ToolInputError):
                    EnterWorktreeTool().run({"name": name}, self.ctx)

    def test_worktree_exit_survives_git_unavailable(self) -> None:
        from src.tool_system.errors import ToolPermissionError

        worktree = self.root / ".git" / "clawd-worktrees" / "demo"
        worktree.mkdir(parents=True)
        self.ctx.worktree_root = worktree
        self.ctx.cwd = worktree

        with patch(
            "src.tool_system.tools.worktree._run_git",
            side_effect=ToolPermissionError("Git unavailable"),
        ):
            result = ExitWorktreeTool().run({}, self.ctx)

        self.assertTrue(result.output["preserved"])
        self.assertIsNone(result.output["worktreeBranch"])
        self.assertIsNone(self.ctx.worktree_root)
        self.assertEqual(self.ctx.cwd, self.root)

    def test_worktree_exit_not_in_session(self) -> None:
        from src.tool_system.errors import ToolPermissionError

        with self.assertRaises(ToolPermissionError):
            ExitWorktreeTool().run({}, self.ctx)


class TestPlanModeTools(ToolSystemTests):
    def test_plan_mode_roundtrip(self) -> None:
        """Test entering and exiting plan mode."""
        # Enter plan mode
        enter_out = EnterPlanModeTool().run({}, self.ctx).output
        self.assertTrue(self.ctx.plan_mode)
        self.assertIn("Entered plan mode", enter_out["message"])

        # Exit plan mode
        exit_out = ExitPlanModeTool().run({}, self.ctx).output
        self.assertFalse(self.ctx.plan_mode)
        self.assertFalse(exit_out["isAgent"])
        self.assertTrue(exit_out["hasTaskTool"])

    def test_plan_mode_exit_with_plan(self) -> None:
        """Test exiting plan mode with a plan."""
        EnterPlanModeTool().run({}, self.ctx)

        plan_content = "# My Plan\n\n- Do something\n- Do something else"
        exit_out = ExitPlanModeTool().run({"plan": plan_content}, self.ctx).output

        self.assertEqual(exit_out["plan"], plan_content)
        self.assertIsNotNone(exit_out["filePath"])

        # Verify plan file was created
        plan_file = self.root / ".clawd" / "plan.md"
        self.assertTrue(plan_file.exists())
        self.assertEqual(plan_file.read_text(encoding="utf-8"), plan_content)

    def test_plan_mode_exit_with_custom_path(self) -> None:
        """Test exiting plan mode with custom plan file path."""
        EnterPlanModeTool().run({}, self.ctx)

        custom_path = self.root / "my-plan.md"
        plan_content = "# Custom Plan"
        exit_out = ExitPlanModeTool().run(
            {"plan": plan_content, "planFilePath": str(custom_path)},
            self.ctx,
        ).output

        self.assertEqual(exit_out["filePath"], str(custom_path))
        self.assertTrue(custom_path.exists())

    def test_plan_mode_exit_not_in_mode(self) -> None:
        """Test exiting plan mode when not in it."""
        from src.tool_system.errors import ToolPermissionError

        with self.assertRaises(ToolPermissionError):
            ExitPlanModeTool().run({}, self.ctx)

    def test_plan_mode_plan_validation(self) -> None:
        """Test plan input validation."""
        from src.tool_system.errors import ToolInputError

        EnterPlanModeTool().run({}, self.ctx)

        # Plan must be string
        with self.assertRaises(ToolInputError):
            ExitPlanModeTool().run({"plan": 123}, self.ctx)

        # Plan file path must be string
        with self.assertRaises(ToolInputError):
            ExitPlanModeTool().run({"plan": "x", "planFilePath": 123}, self.ctx)


if __name__ == "__main__":
    unittest.main()
