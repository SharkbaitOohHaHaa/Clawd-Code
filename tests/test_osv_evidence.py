"""OSV evidence client tests (v1a). No real network: a fake socket drives the real http.client."""

from __future__ import annotations

import ast
import http.client
import io
import json
import os
import re
import socket
import ssl
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rich.console import Console

from src import osv_evidence as osv
from src.osv_evidence import OsvInputError, approval_message, build_request, headers_sent, lookup
from src.tool_system.agent_loop import summarize_tool_result
from src.tool_system.context import ToolContext
from src.tool_system.tools.osv import OsvQueryTool

PKG = {"operation": "query", "ecosystem": "PyPI", "name": "jinja2", "version": "2.4.1"}
PKG_BODY = b'{"package":{"ecosystem":"PyPI","name":"jinja2"},"version":"2.4.1"}'
COMMIT = "0123456789abcdef0123456789abcdef01234567"
NO_RECORDS = "OSV returned no matching known vulnerability records for"
BANNED = re.compile(
    r"(?i)\b(safe|clean|secure|verified|not vulnerable|no known vulnerabilities|vulnerability[- ]free)\b"
)


def http_response(
    status: int = 200,
    body: bytes = b"",
    *,
    reason: str = "OK",
    headers: list[tuple[str, str]] | None = None,
    content_type: str | None = "application/json",
    length: bool = True,
) -> bytes:
    lines = [f"HTTP/1.1 {status} {reason}"]
    if content_type is not None:
        lines.append(f"Content-Type: {content_type}")
    lines.extend(f"{k}: {v}" for k, v in (headers or []))
    names = {k.lower() for k, _ in (headers or [])}
    if length and "content-length" not in names and "transfer-encoding" not in names:
        lines.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


def chunked(body: bytes, *, complete: bool = True) -> bytes:
    data = f"{len(body):x}\r\n".encode() + body + b"\r\n"
    return data + (b"0\r\n\r\n" if complete else b"")


def record(vuln_id: str = "GHSA-aaaa-bbbb-cccc", **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"id": vuln_id, "modified": "2026-01-01T00:00:00Z"}
    data.update(extra)
    return data


def as_json(value: Any) -> bytes:
    return json.dumps(value).encode()


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Raw(io.RawIOBase):
    """Serves scripted response bytes; exceptions are raised, ('advance', s) moves the fake clock."""

    def __init__(self, script: list[Any], clock: _Clock) -> None:
        self._script = list(script)
        self._clock = clock

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while self._script:
            item = self._script[0]
            if isinstance(item, BaseException):
                self._script.pop(0)
                raise item
            if isinstance(item, tuple):
                self._script.pop(0)
                self._clock.now += item[1]
                continue
            if not item:
                self._script.pop(0)
                continue
            size = min(len(buffer), len(item))
            buffer[:size] = item[:size]
            if size < len(item):
                self._script[0] = item[size:]
            else:
                self._script.pop(0)
            return size
        return 0


class _FakeSocket:
    def __init__(self, rig: "OsvRig") -> None:
        self._rig = rig

    def setsockopt(self, *args: Any) -> None:
        return None

    def settimeout(self, value: Any) -> None:
        self._rig.timeouts.append(value)

    def gettimeout(self) -> Any:
        return self._rig.timeouts[-1] if self._rig.timeouts else None

    def sendall(self, data: bytes) -> None:
        self._rig.events.append("send")
        self._rig.sent += bytes(data)

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BufferedReader:
        return io.BufferedReader(_Raw(self._rig.script, self._rig.clock))

    def close(self) -> None:
        return None


class OsvRig:
    """Fake network for one test: patches socket creation, TLS wrapping, and the clock."""

    def __init__(self, script: list[Any] | None = None, *, connect_error: BaseException | None = None,
                 wrap_error: BaseException | None = None, connect_advance: float = 0.0) -> None:
        self.script = list(script or [])
        self.connect_error = connect_error
        self.wrap_error = wrap_error
        self.connect_advance = connect_advance
        self.clock = _Clock()
        self.events: list[str] = []
        self.sent = b""
        self.timeouts: list[Any] = []
        self.connect_targets: list[Any] = []
        self.server_hostnames: list[Any] = []
        self.contexts: list[ssl.SSLContext] = []

    @property
    def connects(self) -> int:
        return self.events.count("connect")

    @property
    def requests(self) -> int:
        return self.events.count("request")

    def run(self, request_or_input: Any) -> dict[str, Any]:
        request = request_or_input if isinstance(request_or_input, osv.OsvRequest) else build_request(request_or_input)
        rig = self
        original_request = http.client.HTTPConnection.request

        def fake_create_connection(address: Any, timeout: Any = None, source_address: Any = None) -> _FakeSocket:
            rig.events.append("connect")
            rig.connect_targets.append(address)
            rig.timeouts.append(timeout)
            rig.clock.now += rig.connect_advance
            if rig.connect_error is not None:
                raise rig.connect_error
            return _FakeSocket(rig)

        def fake_wrap(context: ssl.SSLContext, sock: Any, server_hostname: Any = None, **kwargs: Any) -> Any:
            rig.events.append("wrap")
            rig.contexts.append(context)
            rig.server_hostnames.append(server_hostname)
            if rig.wrap_error is not None:
                raise rig.wrap_error
            return sock

        def counting_request(conn: Any, *args: Any, **kwargs: Any) -> Any:
            rig.events.append("request")
            return original_request(conn, *args, **kwargs)

        def no_dns(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("DNS must not be used in tests")

        with patch("socket.create_connection", fake_create_connection), \
                patch.object(ssl.SSLContext, "wrap_socket", fake_wrap), \
                patch.object(http.client.HTTPConnection, "request", counting_request), \
                patch("socket.getaddrinfo", no_dns), \
                patch("time.sleep", side_effect=AssertionError("no sleeping/backoff")), \
                patch.object(osv.time, "monotonic", self.clock):
            result = lookup(request)
        # H1: every lookup in this suite is checked for a single attempt.
        if self.connects > 1 or self.requests > 1:
            raise AssertionError(f"more than one connection or request: {self.events}")
        return result


class NetworkGuard:
    """Records (and refuses) any real socket, DNS, TLS, or sleep use outside the fake transport."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, test: unittest.TestCase) -> None:
        for target in ("socket.create_connection", "socket.getaddrinfo", "time.sleep"):
            patcher = patch(target, self._refuse(target))
            patcher.start()
            test.addCleanup(patcher.stop)
        patcher = patch.object(ssl.SSLContext, "wrap_socket", self._refuse("wrap_socket"))
        patcher.start()
        test.addCleanup(patcher.stop)

    def _refuse(self, name: str) -> Any:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            raise AssertionError(f"real {name} used in an OSV test")
        return refuse


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = NetworkGuard()
        self.guard.install(self)

    def tearDown(self) -> None:
        self.assertEqual(self.guard.calls, [], "a test reached real network, TLS, DNS, or sleep")
    def assert_single_attempt(self, rig: OsvRig) -> None:
        self.assertLessEqual(rig.connects, 1)
        self.assertLessEqual(rig.requests, 1)

    def assert_error_result(self, result: dict[str, Any], status: str, reason: str) -> None:
        self.assertEqual((result["status"], result["reason"]), (status, reason))
        self.assertIn(status, osv.ERROR_STATUSES)
        self.assertEqual(result["error"], result["statement"])
        self.assertIn("No retry was issued", result["statement"])
        self.assertIn("No vulnerability conclusion can be drawn", result["statement"])
        self.assertNotIn(NO_RECORDS, result["statement"])
        self.assertFalse(result["complete"])


class RequestExactnessTests(_Base):
    """Group B: privacy and exact request."""

    def test_b1_exact_body_per_form(self) -> None:
        cases = [
            (PKG, PKG_BODY),
            ({"operation": "query", "purl": "pkg:pypi/jinja2@3.1.4"}, b'{"package":{"purl":"pkg:pypi/jinja2@3.1.4"}}'),
            ({"operation": "query", "purl": "pkg:npm/lodash", "version": "4.17.20"},
             b'{"package":{"purl":"pkg:npm/lodash"},"version":"4.17.20"}'),
            ({"operation": "query", "commit": COMMIT}, b'{"commit":"' + COMMIT.encode() + b'"}'),
        ]
        for tool_input, body in cases:
            with self.subTest(tool_input=tool_input):
                request = build_request(tool_input)
                self.assertEqual(request.body, body)
                self.assertEqual((request.method, request.path), ("POST", "/v1/query"))
        vuln = build_request({"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"})
        self.assertEqual((vuln.method, vuln.path, vuln.body), ("GET", "/v1/vulns/GHSA-aaaa-bbbb-cccc", None))
        continued = build_request({**PKG, "page_token": "tok_1"})
        self.assertEqual(
            continued.body,
            b'{"package":{"ecosystem":"PyPI","name":"jinja2"},"page_token":"tok_1","version":"2.4.1"}',
        )
        self.assertEqual(continued.query_key, build_request(PKG).query_key)

    def test_b2_exact_raw_request_bytes(self) -> None:
        rig = OsvRig([http_response(200, b"{}")])
        rig.run(PKG)
        self.assertEqual(len(PKG_BODY), 66)
        expected = (
            b"POST /v1/query HTTP/1.1\r\n"
            b"Host: api.osv.dev\r\n"
            b"Accept-Encoding: identity\r\n"
            b"Content-Length: 66\r\n"
            b"Accept: application/json\r\n"
            b"User-Agent: clawd-codex/0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"\r\n" + PKG_BODY
        )
        self.assertEqual(rig.sent, expected)
        request = build_request(PKG)
        self.assertEqual([name for name, _ in headers_sent(request)],
                         ["Host", "Accept-Encoding", "Content-Length", "Accept", "User-Agent", "Content-Type"])

        get_rig = OsvRig([http_response(200, as_json(record()))])
        get_rig.run({"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"})
        self.assertEqual(
            get_rig.sent,
            b"GET /v1/vulns/GHSA-aaaa-bbbb-cccc HTTP/1.1\r\n"
            b"Host: api.osv.dev\r\n"
            b"Accept-Encoding: identity\r\n"
            b"Accept: application/json\r\n"
            b"User-Agent: clawd-codex/0.1\r\n"
            b"\r\n",
        )

    def test_b2_b11_prompt_headers_equal_sent_headers_for_post_and_get(self) -> None:
        for tool_input in (PKG, {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"}):
            with self.subTest(operation=tool_input["operation"]):
                request = build_request(tool_input)
                rig = OsvRig([http_response(200, as_json(record()) if request.body is None else b"{}")])
                rig.run(request)
                head = rig.sent.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")[1:]
                sent = [tuple(line.split(": ", 1)) for line in head]
                self.assertEqual(sent, headers_sent(request))
                listed = approval_message(request).split(" and headers ", 1)[1].split(". OSV also sees", 1)[0]
                self.assertEqual(listed, ", ".join(f"{name}: {value}" for name, value in sent))

    def test_b3_fixed_target_and_no_proxy(self) -> None:
        env = {"HTTP_PROXY": "http://proxy.invalid:8080", "HTTPS_PROXY": "http://proxy.invalid:8080",
               "ALL_PROXY": "http://proxy.invalid:8080"}
        with patch.dict(os.environ, env), \
                patch.object(http.client.HTTPConnection, "set_tunnel", side_effect=AssertionError("no tunnel")):
            rig = OsvRig([http_response(200, b"{}")])
            rig.run(PKG)
        self.assertEqual(rig.connect_targets, [("api.osv.dev", 443)])
        self.assertNotIn(b"proxy.invalid", rig.sent)
        self.assertNotIn(b"?", rig.sent.split(b"\r\n", 1)[0])

    def test_b4_secrets_never_leave_the_machine(self) -> None:
        import tempfile

        canary = "CANARY-SECRET-7f3a9c"
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": canary, "OSV_TOKEN": canary}):
            (Path(tmp) / ".env").write_text(f"KEY={canary}\n", encoding="utf-8")
            (Path(tmp) / "requirements.txt").write_text(f"jinja2==2.4.1  # {canary}\n", encoding="utf-8")
            ctx = ToolContext(workspace_root=Path(tmp))
            rigs: list[OsvRig] = []

            def fake_lookup(request: Any) -> dict[str, Any]:
                rig = OsvRig([http_response(200, b"{}")])
                rigs.append(rig)
                return rig.run(request)

            for tool_input in (PKG, {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"}):
                with patch("src.tool_system.tools.osv.lookup", new=fake_lookup):
                    OsvQueryTool().run(tool_input, ctx)
        self.assertEqual(len(rigs), 2)
        for rig in rigs:
            self.assertNotIn(canary.encode(), rig.sent)
            self.assertNotIn(b"requirements", rig.sent)

    def test_b5_invalid_inputs_are_rejected_without_network(self) -> None:
        invalid = [
            {"operation": "query", "purl": "pkg:pypi/jinja2?repository_url=https://x"},
            {"operation": "query", "purl": "pkg:pypi/jinja2#sub"},
            {"operation": "query", "purl": "pkg:pypi/jinja2@3.1.4", "version": "3.1.4"},
            {"operation": "query", "commit": COMMIT, "version": "1.0"},
            {"operation": "query", "name": "jinja2", "version": "1.0"},
            {"operation": "query", "purl": "pkg:pypi/jinja2@1", "name": "jinja2"},
            {"operation": "query", "commit": "abc123"},
            {"operation": "query", "commit": COMMIT.upper()},
            {"operation": "vuln", "vuln_id": "RHSA-2023:1234"},
            {"operation": "vuln", "vuln_id": "GHSA/aaaa"},
            {"operation": "vuln", "vuln_id": "GHSA..x"},
            {"operation": "vuln", "vuln_id": "GHSA-%61"},
            {"operation": "vuln", "vuln_id": "GHSA aaaa"},
            {"operation": "vuln", "vuln_id": "G-" + "a" * 130},
            {"operation": "query", "ecosystem": "Maven", "name": "a:fire:b", "version": "1"},
            {"operation": "query", "purl": "pkg:npm/a:b"},
            {"operation": "query", "ecosystem": "PyPI", "name": "jin ja", "version": "1"},
            {"operation": "query", "ecosystem": "PyPI", "name": "jinja\x07", "version": "1"},
            {"operation": "query", "ecosystem": "PyPI", "name": "[bold]x", "version": "1"},
            {"operation": "query", "ecosystem": "PyPI", "name": 'x"y', "version": "1"},
            {"operation": "query", "ecosystem": "PyPI", "name": "allow_docs", "version": "1"},
            {"operation": "query", "ecosystem": "npm", "name": "x-ALLOW_DOCS-y", "version": "1"},
            {"operation": "query", "purl": "pkg:npm/allow_docs"},
            {"operation": "query", "purl": "pkg:pypi/x-Allow_Docs@1.0"},
            {"operation": "query", "ecosystem": "PyPI", "name": "CANARY[x", "version": "1"},
            {"operation": "query", "purl": "pkg:pypi/CANARY?x=1"},
            {"operation": "vuln", "vuln_id": "CANARY:1"},
            {"operation": "query", "commit": "CANARY"},
            {"operation": "query", "ecosystem": "pypi", "name": "jinja2", "version": "1"},
            {"operation": "query", "ecosystem": "GIT", "name": "x", "version": "1"},
            {**PKG, "page_token": "bad token"},
            {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc", "page_token": "tok"},
            {"operation": "query"},
            {"operation": "query", "commit": COMMIT, "purl": "pkg:npm/x"},
            {"operation": "delete", "commit": COMMIT},
            {"operation": "query", "commit": COMMIT, "extra": "x"},
        ]
        tool = OsvQueryTool()
        ctx = ToolContext(workspace_root=Path.cwd())
        with patch("src.osv_evidence._open_connection", side_effect=AssertionError("no network")) as opener:
            for tool_input in invalid:
                with self.subTest(tool_input=tool_input):
                    with self.assertRaises(OsvInputError):
                        build_request(tool_input)
                    result = tool.check_permissions(tool_input, ctx)
                    self.assertEqual(result.behavior.value, "deny")
                    self.assertNotIn("CANARY", result.message or "")
            opener.assert_not_called()
        colon = tool.check_permissions({"operation": "vuln", "vuln_id": "RHSA-2023:1234"}, ctx)
        self.assertIn("Advisory IDs containing ':' are not supported in v1", colon.message or "")

    def test_b6_run_rejects_invalid_input_without_network(self) -> None:
        from src.tool_system.errors import ToolInputError

        with patch("src.osv_evidence._open_connection", side_effect=AssertionError("no network")) as opener:
            with self.assertRaises(ToolInputError):
                OsvQueryTool().run({"operation": "query", "commit": "nope"}, ToolContext(workspace_root=Path.cwd()))
        opener.assert_not_called()

    def test_b7_b8_case_preserved_and_pypi_normalization_disclosed(self) -> None:
        vuln = build_request({"operation": "vuln", "vuln_id": "GHSA-AbCd-bbbb-cccc"})
        self.assertEqual(vuln.path, "/v1/vulns/GHSA-AbCd-bbbb-cccc")
        normalized = build_request({"operation": "query", "ecosystem": "PyPI", "name": "Jinja_2", "version": "1.0"})
        self.assertEqual(json.loads(normalized.body)["package"]["name"], "jinja-2")
        self.assertEqual(normalized.normalization_applied[0]["from"], "Jinja_2")
        rig = OsvRig([http_response(200, b"{}")])
        result = rig.run(normalized)
        self.assertEqual(result["normalization_applied"][0]["rule"], "PEP 503")
        npm = build_request({"operation": "query", "ecosystem": "npm", "name": "Some_Pkg", "version": "1"})
        self.assertEqual(json.loads(npm.body)["package"]["name"], "Some_Pkg")
        self.assertEqual(npm.normalization_applied, ())

    def test_b9_ask_message_contents_and_rendering(self) -> None:
        names = ["jinja2", "a:fire", "fire:b", "org.x:fire"]
        for name in names:
            with self.subTest(name=name):
                ecosystem = "Maven" if ":" in name else "PyPI"
                request = build_request({"operation": "query", "ecosystem": ecosystem, "name": name, "version": "1:2"})
                message = approval_message(request)
                self.assertIn("api.osv.dev", message)
                self.assertIn(request.body.decode(), message)
                for header_name, value in headers_sent(request):
                    self.assertIn(f"{header_name}: {value}", message)
                self.assertIn("No redirects, proxies or automatic retries", message)
                self.assertIn("does not approve any change", message)
                self.assertNotIn("allow_docs", message.lower())
                self.assertNotIn("documentation files", message.lower())
                out = io.StringIO()
                Console(file=out, width=10000, color_system=None).print("  " + message)
                self.assertEqual(out.getvalue(), "  " + message + "\n")

    def test_b10_tls_and_host(self) -> None:
        rig = OsvRig([http_response(200, b"{}")])
        rig.run(PKG)
        self.assertEqual(rig.server_hostnames, ["api.osv.dev"])
        self.assertEqual(rig.contexts[0].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(rig.contexts[0].check_hostname)

    def test_b11_prompt_body_equals_sent_body(self) -> None:
        request = build_request(PKG)
        rig = OsvRig([http_response(200, b"{}")])
        rig.run(request)
        self.assertTrue(rig.sent.endswith(request.body))
        self.assertIn(request.body.decode(), approval_message(request))
        permission = OsvQueryTool().check_permissions(PKG, ToolContext(workspace_root=Path.cwd()))
        self.assertIsNone(permission.updated_input)
        self.assertIsNone(permission.suggestion)

    def test_b12_connection_is_opened_before_the_single_request(self) -> None:
        rig = OsvRig([http_response(200, b"{}")])
        rig.run(PKG)
        self.assertEqual(rig.events[:3], ["connect", "wrap", "request"])
        self.assertTrue(all(t is not None and t <= 30 for t in rig.timeouts))
        failing = OsvRig(connect_error=TimeoutError("connect timed out"))
        result = failing.run(PKG)
        self.assert_error_result(result, "unavailable", "timeout")
        self.assertEqual((failing.connects, failing.requests), (1, 0))


class FailureTests(_Base):
    """Group D: every failure is explicit, single-attempt, and never 'no records'."""

    def run_case(self, script: list[Any], tool_input: dict[str, Any] = PKG, **rig_kwargs: Any) -> tuple[dict, OsvRig]:
        rig = OsvRig(script, **rig_kwargs)
        result = rig.run(tool_input)
        self.assert_single_attempt(rig)
        return result, rig

    def test_d1_d4_d5_connection_failures(self) -> None:
        cases = [
            (TimeoutError("t"), None, "timeout"),
            (ConnectionRefusedError("r"), None, "connection_error"),
            (ConnectionResetError("r"), None, "connection_error"),
            (socket.gaierror("dns"), None, "connection_error"),
            (None, ssl.SSLCertVerificationError("cert"), "tls_verification_failed"),
            (None, ssl.SSLError("tls"), "tls_error"),
        ]
        for connect_error, wrap_error, reason in cases:
            with self.subTest(reason=reason, error=connect_error or wrap_error):
                result, rig = self.run_case([], connect_error=connect_error, wrap_error=wrap_error)
                self.assert_error_result(result, "unavailable", reason)
                self.assertEqual(rig.requests, 0)

    def test_d2_timeout_mid_body(self) -> None:
        head = http_response(200, b"x" * 10)[:-10]
        result, _ = self.run_case([head, b"xxx", TimeoutError("read")])
        self.assert_error_result(result, "unavailable", "timeout")

    def test_d3_best_effort_deadline_between_chunks(self) -> None:
        with patch.object(osv, "READ_CHUNK_BYTES", 10):
            body = b"x" * 30
            head = http_response(200, body)[:-30]
            result, _ = self.run_case([head, body[:10], ("advance", 25), body[10:20], ("advance", 25), body[20:]])
        self.assert_error_result(result, "unavailable", "deadline_exceeded")

    def test_d4_remote_disconnect_without_status(self) -> None:
        result, _ = self.run_case([])
        self.assert_error_result(result, "unavailable", "connection_error")

    def test_d6_rate_limited(self) -> None:
        result, _ = self.run_case([http_response(429, b"", reason="Too Many", headers=[("Retry-After", "30")])])
        self.assert_error_result(result, "rate_limited", "http_429")
        self.assertEqual(result["retry_after"], "30")
        result, _ = self.run_case([http_response(429, b"", reason="Too Many")])
        self.assertIsNone(result["retry_after"])

    def test_d7_d9_d19_status_mapping(self) -> None:
        cases = [(500, "unavailable", "http_5xx"), (502, "unavailable", "http_5xx"), (503, "unavailable", "http_5xx"),
                 (504, "unavailable", "http_5xx"), (403, "unavailable", "http_4xx"), (418, "unavailable", "http_4xx"),
                 (204, "unavailable", "unexpected_status"), (206, "unavailable", "unexpected_status"),
                 (600, "unavailable", "unexpected_status"), (999, "unavailable", "unexpected_status")]
        for code, status, reason in cases:
            with self.subTest(code=code):
                body = b"" if code == 204 else b"{}"
                result, _ = self.run_case([http_response(code, body, reason="X")])
                self.assert_error_result(result, status, reason)
                self.assertEqual(result["http_status"], code)

    def test_d8_bad_request_message_is_data_only(self) -> None:
        body = as_json({"code": 3, "message": "Invalid query [bold]boom[/bold]"})
        result, _ = self.run_case([http_response(400, body, reason="Bad Request")])
        self.assert_error_result(result, "invalid_request", "http_400")
        self.assertIn("Invalid query", result["osv_error_message"])
        self.assertNotIn("Invalid query", result["statement"])
        self.assertNotIn("Invalid query", result["error"])

    def test_d8_bad_request_body_read_failure_is_still_invalid_request(self) -> None:
        head = http_response(400, b"x" * 50, reason="Bad Request")[:-50]
        result, _ = self.run_case([head, b"xx", TimeoutError("read")])
        self.assert_error_result(result, "invalid_request", "http_400")
        self.assertIsNone(result["osv_error_message"])

    def test_d15b_negative_chunk_size_cannot_bypass_the_cap(self) -> None:
        script = [http_response(200, b"", headers=[("Transfer-Encoding", "chunked")], length=False)
                  + b"-1\r\n" + b"x" * 500]
        with patch.object(osv, "MAX_RESPONSE_BYTES", 100), patch.object(osv, "FRAMING_ALLOWANCE_BYTES", 0):
            result, _ = self.run_case(script)
        self.assert_error_result(result, "unavailable", "response_too_large")

    def test_d3b_deadline_checkpoints_after_connect_and_after_headers(self) -> None:
        after_connect, rig = self.run_case([http_response(200, b"{}")], connect_advance=50)
        self.assert_error_result(after_connect, "unavailable", "deadline_exceeded")
        self.assertEqual(rig.requests, 0)
        after_headers, rig = self.run_case([("advance", 50), http_response(200, b"{}")])
        self.assert_error_result(after_headers, "unavailable", "deadline_exceeded")
        self.assertEqual(rig.requests, 1)

    def test_d10_not_found_status_is_never_no_records(self) -> None:
        for tool_input in (PKG, {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"}):
            with self.subTest(operation=tool_input["operation"]):
                result, _ = self.run_case([http_response(404, b"{}", reason="Not Found")], tool_input)
                self.assert_error_result(result, "http_404", "http_404")
                self.assertTrue(result["statement"].startswith(
                    "OSV returned HTTP 404; Clawd did not infer advisory absence from this response."))

    def test_d11_redirects_are_refused(self) -> None:
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                script = [http_response(code, b"", reason="Moved", headers=[("Location", "https://evil.example/x")])]
                result, rig = self.run_case(script)
                self.assert_error_result(result, "unavailable", "redirect_refused")
                self.assertEqual(rig.connect_targets, [("api.osv.dev", 443)])
                self.assertNotIn("evil.example", json.dumps(result))

    def test_d12_to_d15_framing_failures(self) -> None:
        big = b"x" * 200
        cases = [
            ("short content-length", [http_response(200, b"y" * 30, headers=[("Content-Length", "60")])], "body_incomplete"),
            ("truncated chunked", [http_response(200, b"", headers=[("Transfer-Encoding", "chunked")], length=False)
                                   + chunked(b'{"vulns":', complete=False)], "body_incomplete"),
            ("close delimited", [http_response(200, b"{}", length=False)], "body_incomplete"),
            ("invalid length abc", [http_response(200, b"{}", headers=[("Content-Length", "abc")])], "body_incomplete"),
            ("invalid length -1", [http_response(200, b"{}", headers=[("Content-Length", "-1")])], "body_incomplete"),
            ("two lengths", [http_response(200, b"{}", headers=[("Content-Length", "2"), ("Content-Length", "3")])],
             "body_incomplete"),
            ("over cap by length", [http_response(200, big)], "response_too_large"),
            ("over cap chunked", [http_response(200, b"", headers=[("Transfer-Encoding", "chunked")], length=False)
                                  + chunked(big)], "response_too_large"),
        ]
        with patch.object(osv, "MAX_RESPONSE_BYTES", 100):
            for label, script, reason in cases:
                with self.subTest(label):
                    result, _ = self.run_case(script)
                    self.assert_error_result(result, "unavailable", reason)

    def test_d16_to_d18_malformed_responses(self) -> None:
        bodies = [
            (http_response(200, b"<html></html>", content_type="text/html"), "html"),
            (http_response(200, b"{}", content_type=None), "no content type"),
            (http_response(200, b"{not json"), "invalid json"),
            (http_response(200, b"[]"), "array"),
            (http_response(200, b'{"vulns": {}}'), "vulns object"),
            (http_response(200, b'{"vulns": null}'), "vulns null"),
            (http_response(200, b'{"next_page_token": ""}'), "empty token"),
            (http_response(200, b'{"vulns": [{"modified": "x"}]}'), "record without id"),
            (http_response(200, b'{"vulns": [{"id": "A-1", "modified": "m", "score": NaN}]}'), "nan"),
            (http_response(200, b'{"vulns": [{"id": "A-1", "modified": "m", "id": "A-2"}]}'), "duplicate keys"),
            (http_response(200, b'{"vulns": [{"id": "A-1", "modified": "m", "x": '
                           + b"[" * 100_000 + b"]" * 100_000 + b"}]}"), "deep nesting"),
            (http_response(200, b'{"vulns": [{"id": "A-1", "modified": "m", "summary": "\xff"}]}'), "invalid utf-8"),
            (http_response(200, "{}".encode("utf-16")), "utf-16"),
            (http_response(200, b'{"code": 5, "message": "x"}'), "error-shaped 200"),
        ]
        for script, label in bodies:
            with self.subTest(label):
                result, _ = self.run_case([script])
                self.assert_error_result(result, "unavailable", "malformed_response")
        control, _ = self.run_case([http_response(200, b'{"vulns": [{"id": "A-1", "modified": "m", "score": 1.5}]}')])
        self.assertEqual(control["status"], "records_found")
        result, _ = self.run_case([http_response(200, b'{"summary": "no id"}')],
                                  {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"})
        self.assert_error_result(result, "unavailable", "malformed_response")

    def test_d20_protocol_errors(self) -> None:
        too_many = "HTTP/1.1 200 OK\r\n" + "".join(f"X-H{i}: v\r\n" for i in range(101)) + "\r\n"
        cases = [
            ([b"GARBAGE\r\n\r\n"], "bad status line"),
            ([b"HTTP/1.1 200 " + b"x" * 70000 + b"\r\n\r\n"], "line too long"),
            ([too_many.encode()], "too many headers"),
        ]
        for script, label in cases:
            with self.subTest(label):
                result, _ = self.run_case(script)
                self.assert_error_result(result, "unavailable", "protocol_error")

    def test_d21_internally_conflicting_responses(self) -> None:
        cases = [
            (http_response(200, as_json(record("GHSA-zzzz-zzzz-zzzz", aliases=["CVE-2020-1"]))),
             {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"}),
            (http_response(200, as_json({"vulns": [record("A-1"), record("A-1", modified="2026-02-02T00:00:00Z")]})), PKG),
            (http_response(200, as_json({"vulns": [record("A-1", withdrawn="2026-01-02T00:00:00Z")]})), PKG),
        ]
        for script, tool_input in cases:
            with self.subTest(tool_input=tool_input):
                result, _ = self.run_case([script], tool_input)
                self.assert_error_result(result, "unavailable", "conflicting_response")
                self.assertEqual(result["records"], [])
                self.assertIn(result["conflict_detail"], osv._CONFLICT_DETAILS.values())


class PaginationTests(_Base):
    """Group E: next_page_token means incomplete; continuing needs a new approved call."""

    def test_e1_e2_incomplete_is_a_normal_but_incomplete_result(self) -> None:
        rig = OsvRig([http_response(200, as_json({"vulns": [record("A-1"), record("A-2")], "next_page_token": "tok_1"}))])
        result = rig.run(PKG)
        self.assertEqual((result["status"], result["reason"]), ("incomplete", "incomplete_pagination"))
        self.assertNotIn(result["status"], osv.ERROR_STATUSES)
        self.assertEqual(result["records_shown"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["next_page_token"], "tok_1")
        self.assertIn("incomplete", result["statement"])
        self.assertEqual(rig.requests, 1)
        only_token = OsvRig([http_response(200, b'{"next_page_token": "tok_2"}')]).run(PKG)
        self.assertEqual(only_token["status"], "incomplete")
        self.assertNotIn(NO_RECORDS, only_token["statement"])

    def test_e3_e4_continuation_tokens_are_session_scoped_and_query_bound(self) -> None:
        tool = OsvQueryTool()
        ctx = ToolContext(workspace_root=Path.cwd())
        page = {"vulns": [record("A-1")], "next_page_token": "tok_1"}
        with patch("src.tool_system.tools.osv.lookup",
                   new=lambda request: OsvRig([http_response(200, as_json(page))]).run(request)):
            first = tool.run(PKG, ctx)
        self.assertEqual(first.output["status"], "incomplete")
        self.assertFalse(first.is_error)

        continuation = {**PKG, "page_token": "tok_1"}
        permission = tool.check_permissions(continuation, ctx)
        self.assertEqual(permission.behavior.value, "ask")
        self.assertIn('"page_token":"tok_1"', permission.message or "")

        rejected = [
            {**PKG, "page_token": "never_issued"},
            {"operation": "query", "ecosystem": "PyPI", "name": "requests", "version": "2.0.0", "page_token": "tok_1"},
            {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc", "page_token": "tok_1"},
            {**PKG, "page_token": "bad token!"},
        ]
        with patch("src.osv_evidence._open_connection", side_effect=AssertionError("no network")) as opener:
            for tool_input in rejected:
                with self.subTest(tool_input=tool_input):
                    self.assertEqual(tool.check_permissions(tool_input, ctx).behavior.value, "deny")
            fresh = OsvQueryTool().check_permissions(continuation, ctx)
            self.assertEqual(fresh.behavior.value, "deny")
        opener.assert_not_called()


class ReferencesAndTextTests(_Base):
    """Group F: references are data; third-party text is capped and inert."""

    def test_f1_references_are_never_fetched(self) -> None:
        refs = [{"type": "EVIDENCE" if i % 2 else "DETECTION", "url": f"https://ref{i}.example/x"} for i in range(30)]
        rig = OsvRig([http_response(200, as_json({"vulns": [record(references=refs)]}))])
        with patch("urllib.request.urlopen", side_effect=AssertionError("must not fetch")), \
                patch("src.tool_system.tools.web_fetch.WebFetchTool.run", side_effect=AssertionError("must not fetch")):
            result = rig.run(PKG)
        shown = result["records"][0]["references"]
        self.assertEqual(result["records"][0]["references_total"], 30)
        self.assertEqual(len(shown), osv.MAX_REFERENCES_SHOWN)
        self.assertTrue(all(ref["fetched"] is False for ref in shown))
        self.assertEqual(rig.connects, 1)

    def test_f2_module_imports_no_other_http_clients(self) -> None:
        tree = ast.parse(Path(osv.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        for forbidden in ("urllib.request", "requests", "httpx", "urllib3"):
            self.assertNotIn(forbidden, imported)
        self.assertFalse(any("web_fetch" in name for name in imported))

    def test_f4_third_party_text_is_capped(self) -> None:
        long_record = record(summary="s" * 600, details="d" * 3000)
        result = OsvRig([http_response(200, as_json({"vulns": [long_record]}))]).run(PKG)
        rec = result["records"][0]
        self.assertTrue(rec["summary"].endswith("(truncated by Clawd)"))
        self.assertTrue(rec["details_excerpt"].endswith("(truncated by Clawd)"))
        self.assertLessEqual(len(rec["details_excerpt"]), osv.MAX_DETAILS_CHARS + 30)
        self.assertTrue(result["display_truncated"])


class EvidenceMeaningTests(_Base):
    """Group G: the meaning of each status is exact and never overstated."""

    def all_status_results(self) -> list[dict[str, Any]]:
        return [
            OsvRig([http_response(200, as_json({"vulns": [record()]}))]).run(PKG),
            OsvRig([http_response(200, b"{}")]).run(PKG),
            OsvRig([http_response(200, b'{"next_page_token": "t"}')]).run(PKG),
            OsvRig([http_response(429, b"", reason="X")]).run(PKG),
            OsvRig([http_response(503, b"", reason="X")]).run(PKG),
            OsvRig([http_response(404, b"", reason="X")]).run(PKG),
            OsvRig([http_response(400, b"{}", reason="X")]).run(PKG),
        ]

    def test_g1_no_records_found_exact_statement(self) -> None:
        for body in (b"{}", b'{"vulns": []}'):
            with self.subTest(body=body):
                result = OsvRig([http_response(200, body)]).run(PKG)
                self.assertEqual((result["status"], result["reason"]), ("no_records_found", "none"))
                self.assertTrue(result["complete"])
                self.assertEqual(
                    result["statement"],
                    "OSV returned no matching known vulnerability records for PyPI package jinja2 version 2.4.1 "
                    "at this time. An empty OSV answer is not evidence of absence: OSV may not hold a record, or a "
                    "source may be stale. This evidence does not approve any change.",
                )

    def test_g2_records_found_with_display_cap(self) -> None:
        records = [record(f"GHSA-{i:04d}-bbbb-cccc") for i in range(60)]
        result = OsvRig([http_response(200, as_json({"vulns": records}))]).run(PKG)
        self.assertEqual((result["status"], result["record_count"], result["records_shown"]), ("records_found", 60, 50))
        self.assertIn("50 of 60 shown", result["statement"])
        self.assertTrue(result["display_truncated"])

    def test_g3_withdrawn_record_via_vuln_is_reported(self) -> None:
        payload = record("GHSA-aaaa-bbbb-cccc", withdrawn="2026-03-01T00:00:00Z")
        result = OsvRig([http_response(200, as_json(payload))]).run({"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"})
        self.assertEqual(result["status"], "records_found")
        self.assertEqual(result["records"][0]["withdrawn"], "2026-03-01T00:00:00Z")
        self.assertIn("OSV marks 1 of the shown record(s) as withdrawn.", result["notes"])
        # The fixed §5 records_found statement is unchanged for withdrawn advisories.
        self.assertEqual(
            result["statement"],
            "OSV returned 1 known vulnerability record(s) for advisory GHSA-aaaa-bbbb-cccc at "
            f"{result['observed_at']} (complete response); 1 of 1 shown. The records are shown as OSV "
            "supplied them; OSV's version matching is its own interpretation. "
            "This evidence does not approve any change.",
        )
        self.assertNotIn("withdrawn", result["statement"])

    def test_g4_g5_severity_and_ranges_pass_through(self) -> None:
        ranges = [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"last_affected": "2.9"}, {"limit": "*"}]}]
        with_cvss = record(severity=[{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
                           affected=[{"package": {"ecosystem": "PyPI", "name": "jinja2"}, "ranges": ranges,
                                      "ecosystem_specific": {"severity": "HIGH"}}])
        plain = record("GHSA-bbbb-bbbb-cccc")
        result = OsvRig([http_response(200, as_json({"vulns": [with_cvss, plain]}))]).run(PKG)
        first, second = result["records"]
        self.assertEqual(first["severity"][0]["score"], "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        self.assertEqual(first["affected"][0]["ranges"], ranges)
        self.assertEqual(first["affected"][0]["ecosystem_specific"], {"severity": "HIGH"})
        self.assertIn("database-defined", first["affected"][0]["specific_fields_note"])
        self.assertEqual(second["severity"], [])
        self.assertEqual(second["severity_note"], "OSV supplied no severity")

    def test_g6_vuln_returned_under_alias(self) -> None:
        payload = record("GHSA-zzzz-zzzz-zzzz", aliases=["CVE-2024-3651"])
        result = OsvRig([http_response(200, as_json(payload))]).run({"operation": "vuln", "vuln_id": "CVE-2024-3651"})
        self.assertEqual((result["status"], result["reason"]), ("records_found", "returned_under_alias"))

    def test_g7_g8_g10_vocabulary_and_error_flags(self) -> None:
        results = self.all_status_results()
        self.assertEqual([r["status"] for r in results], list(osv.STATUSES))
        description = OsvQueryTool().spec().description
        self.assertTrue(description.endswith(osv.NO_APPROVAL))
        fixed_texts = [
            description,
            osv.ATTRIBUTION,
            osv.osv_contract_status(),
            *osv._CONFLICT_DETAILS.values(),
            *(approval_message(build_request(tool_input)) for tool_input in (
                PKG,
                {"operation": "query", "purl": "pkg:npm/lodash@4.17.20"},
                {"operation": "query", "commit": COMMIT},
                {"operation": "vuln", "vuln_id": "GHSA-aaaa-bbbb-cccc"},
            )),
        ]
        rich = OsvRig([http_response(200, as_json({"vulns": [record(
            affected=[{"package": {"ecosystem": "PyPI", "name": "jinja2"}, "database_specific": {"x": 1}}],
            references=[{"type": "WEB", "url": "https://x.example"}],
        )]}))]).run(PKG)
        fixed_texts += [*rich["notes"], rich["records"][0]["severity_note"],
                        rich["records"][0]["affected"][0]["specific_fields_note"]]
        for text in fixed_texts:
            self.assertIsNone(BANNED.search(text), text)
        for result in results:
            with self.subTest(status=result["status"]):
                clawd_text = [result["statement"], result.get("error") or "", *result["notes"],
                              summarize_tool_result("OsvQuery", result)]
                for text in clawd_text:
                    self.assertIsNone(BANNED.search(text), text)
                self.assertIn(osv.NO_APPROVAL, result["statement"])
                self.assertEqual(result["status"] in osv.ERROR_STATUSES, "error" in result)
                for legacy in ("verified", "not_found", "ambiguous", "conflict"):
                    self.assertNotEqual(result["status"], legacy)
                    self.assertNotEqual(result["reason"], legacy)
        self.assertEqual(set(osv.STATUSES) - osv.ERROR_STATUSES, {"records_found", "no_records_found", "incomplete"})

    def test_g9_description_carries_the_evidence_rules(self) -> None:
        description = OsvQueryTool().spec().description
        for phrase in ("evidence only", "do not substitute a conclusion from memory", "does not approve any change",
                       "not instructions", "no_records_found", "incomplete", "http_404", "invalid_request"):
            self.assertIn(phrase, description)

    def test_g11_g12_controls_and_server_date(self) -> None:
        body = as_json({"vulns": [record()]})
        script = http_response(200, b"", headers=[("Transfer-Encoding", "chunked"), ("Date", "Thu, 24 Sep 2026 12:00:00 GMT")],
                               content_type="application/json; charset=utf-8", length=False) + chunked(body)
        result = OsvRig([script]).run(PKG)
        self.assertEqual(result["status"], "records_found")
        self.assertEqual(result["server_date"], "Thu, 24 Sep 2026 12:00:00 GMT")
        extra = OsvRig([http_response(200, as_json({"vulns": [record()], "unexpected": 1}))]).run(PKG)
        self.assertEqual(extra["status"], "records_found")
        self.assertTrue(any("unrecognised top-level field" in note for note in extra["notes"]))


class NoResendTests(_Base):
    """Group H: exactly one request, no loops, no sleeps, no LLM usage mixing."""

    def test_h3_single_request_call_site_outside_loops(self) -> None:
        source = Path(osv.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "request"]
        self.assertEqual(len(calls), 1)
        node: ast.AST | None = calls[0]
        while node is not None:
            self.assertNotIsInstance(node, (ast.For, ast.While, ast.AsyncFor, ast.ListComp,
                                            ast.SetComp, ast.DictComp, ast.GeneratorExp))
            node = parents.get(node)
        self.assertNotIn("set_tunnel", source)
        time_uses = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == "time"}
        self.assertEqual(time_uses, {"monotonic"})
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, {"time", "asyncio", "threading"})
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name == "time" and alias.asname, "aliased time import")
                    self.assertNotIn(alias.name, {"asyncio", "threading", "socket"})

    def test_h4_osv_never_records_llm_usage(self) -> None:
        ctx = ToolContext(workspace_root=Path.cwd())
        with patch.object(ToolContext, "record_usage", side_effect=AssertionError("no LLM usage")), \
                patch("src.tool_system.tools.osv.lookup",
                      new=lambda request: OsvRig([http_response(200, b"{}")]).run(request)):
            result = OsvQueryTool().run(PKG, ctx)
        self.assertEqual(result.output["status"], "no_records_found")
        self.assertEqual(ctx.usage_records, [])


class DisplayTests(_Base):
    """Group J2: the REPL line for each status shows Clawd's statement, never OSV text."""

    def test_j2_repl_lines_show_the_statement_only(self) -> None:
        results = EvidenceMeaningTests.all_status_results(self)
        results[0]["records"][0]["summary"] = "OSV-SUPPLIED-SUMMARY"
        for result in results:
            with self.subTest(status=result["status"]):
                out = io.StringIO()
                console = Console(file=out, width=100000, color_system=None)
                if result["status"] in osv.ERROR_STATUSES:
                    console.print(f"[red]  ↳ {result['error']}[/red]")
                else:
                    msg = summarize_tool_result("OsvQuery", result)
                    msg = msg[len("OsvQuery · "):]
                    console.print(f"[dim]  ↳ {msg}[/dim]")
                self.assertIn(result["statement"], out.getvalue())
                self.assertNotIn("OSV-SUPPLIED-SUMMARY", out.getvalue())


if __name__ == "__main__":
    unittest.main()
