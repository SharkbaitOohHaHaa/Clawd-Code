"""Provider SDK policy: one Clawd provider attempt is one SDK request attempt.

Offline only. Credential, base-URL and proxy environment variables are scrubbed, sockets and
DNS are refused, the SDK retry sleep is recorded (never slept), and every HTTP send is counted
at ``httpx.HTTPTransport.handle_request`` (proxy transports are HTTPTransport too).
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import re
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
import zhipuai

from src.providers import _BUILTIN_PROVIDER_NAMES, get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.glm_provider import GLMProvider
from src.providers.minimax_provider import MinimaxProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.qwen_provider import QwenProvider
from src.providers.sdk_policy import (
    ProviderSdkPolicyError,
    provider_sdk_policy_status,
    require_sdk_retry_policy,
    resolve_max_retries,
)

SRC = Path(__file__).resolve().parents[1] / "src"
SCRUBBED_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BASE_URL|PROXY", re.IGNORECASE)
FAKE_KEY = "test-dummy-key"
FAKE_KEYS = {"glm": "dummyid.dummysecret"}
OTHER_HOST = "redirect-target.invalid"
MESSAGES = [{"role": "user", "content": "offline probe"}]
ANTHROPIC_FAMILY = {"anthropic", "minimax"}
OPENAI_FAMILY = {"openai", "deepseek", "qwen"}

CALL_SHAPES: dict[str, Callable[[Any], Any]] = {
    "chat": lambda provider: provider.chat(MESSAGES),
    "chat_stream": lambda provider: list(provider.chat_stream(MESSAGES)),
    "chat_stream_response": lambda provider: provider.chat_stream_response(MESSAGES),
}

REJECTED_RETRY_VALUES: list[Any] = [
    1, 2, -1, True, False, 0.0, -0.0, 1.5, math.inf, math.nan, "0", "", b"0", [], {}, object(),
]


class _Int(int):
    pass


class _FailingBody(httpx.SyncByteStream):
    def __iter__(self):
        raise httpx.ReadError("offline: body read failed")
        yield b""  # pragma: no cover


def _status(code: int, headers: dict[str, str] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            code, headers=headers, json={"error": {"type": "offline", "message": "offline"}}, request=request
        )
    return respond


def _raise(exc_type: type[Exception]) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        raise exc_type("offline", request=request)  # type: ignore[call-arg]
    return respond


def _body_read_fails(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/json"}, stream=_FailingBody(), request=request)


def _redirect(code: int) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == OTHER_HOST:
            return httpx.Response(200, json={}, request=request)
        return httpx.Response(code, headers={"location": f"https://{OTHER_HOST}/collect"}, request=request)
    return respond


FAILURES: dict[str, Callable[[httpx.Request], httpx.Response]] = {
    "408": _status(408),
    "409": _status(409),
    "429": _status(429),
    "500": _status(500),
    "503": _status(503),
    "529": _status(529),
    "400 + x-should-retry": _status(400, {"x-should-retry": "true"}),
    "ConnectError": _raise(httpx.ConnectError),
    "ConnectTimeout": _raise(httpx.ConnectTimeout),
    "ReadTimeout": _raise(httpx.ReadTimeout),
    "RemoteProtocolError": _raise(httpx.RemoteProtocolError),
    # Retried by SDK defaults for non-streaming calls; a streamed body is read after the SDK's
    # retry loop, so for the streaming shapes this case is coverage, not retry proof.
    "200 body read fails": _body_read_fails,
    "cross-origin 307": _redirect(307),
    "cross-origin 308": _redirect(308),
}


def build_provider(name: str) -> Any:
    """Build a built-in the way the REPL does (repl/core.py): api_key, base_url, model only."""
    return get_provider_class(name)(api_key=FAKE_KEYS.get(name, FAKE_KEY), base_url=None, model=None)


_SHARED: dict[str, Any] = {}


def shared_provider(name: str) -> Any:
    """One real provider per built-in (each real SDK client costs an SSL context).

    Only called from OfflineSdkCase tests, so it is built and its client created under the
    scrubbed environment.
    """
    if name not in _SHARED:
        _SHARED[name] = build_provider(name)
    return _SHARED[name]


def sdk_client(provider: Any) -> Any:
    if isinstance(provider, (AnthropicProvider, MinimaxProvider)):
        return provider._ensure_client()
    return provider.client


class OfflineSdkCase(unittest.TestCase):
    """No network, no real keys, no real sleeps; every SDK HTTP send is counted."""

    def setUp(self) -> None:
        clean_env = {name: value for name, value in os.environ.items() if not SCRUBBED_ENV.search(name)}
        self._start(patch.dict(os.environ, clean_env, clear=True))
        self.refused: list[str] = []
        for target in ("socket.create_connection", "socket.getaddrinfo", "socket.socket.connect"):
            self._start(patch(target, self._refuse(target)))
        self.sleeps: list[float] = []
        self._start(patch("time.sleep", side_effect=self.sleeps.append))
        self.sent: list[httpx.Request] = []
        self.respond: Callable[[httpx.Request], httpx.Response] = _status(500)
        def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
            return self._handle(request)

        self._start(patch.object(httpx.HTTPTransport, "handle_request", handle_request))

    def _start(self, patcher: Any) -> None:
        patcher.start()
        self.addCleanup(patcher.stop)

    def _refuse(self, name: str) -> Callable[..., Any]:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.refused.append(name)
            raise AssertionError(f"{name} used in an offline SDK test")
        return refuse

    def _handle(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.sent.append(request)
        return self.respond(request)

    def tearDown(self) -> None:
        self.assertEqual(self.refused, [], "an SDK test reached the network")

    def reset(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.sent.clear()
        self.sleeps.clear()
        self.respond = respond


class TestHarnessObservesSdkDefaults(OfflineSdkCase):
    """Controls: without Clawd's policy the same seam sees SDK retries and redirects."""

    def test_sdk_defaults_retry_every_trigger_and_follow_redirects(self) -> None:
        anthropic_client = anthropic.Anthropic(api_key=FAKE_KEY)
        openai_client = openai.OpenAI(api_key=FAKE_KEY)
        zhipu_client = zhipuai.ZhipuAI(api_key=FAKE_KEYS["glm"])
        calls: dict[str, Callable[[], Any]] = {
            "anthropic": lambda: anthropic_client.messages.create(model="m", max_tokens=8, messages=MESSAGES),
            "openai": lambda: openai_client.chat.completions.create(model="m", messages=MESSAGES),
            "zhipuai": lambda: zhipu_client.chat.completions.create(model="m", messages=MESSAGES),
        }
        retry_sends = {"anthropic": 3, "openai": 3, "zhipuai": 4}
        redirect_sends = {"anthropic": 2, "openai": 2, "zhipuai": 1}
        for case, respond in FAILURES.items():
            redirect = case.startswith("cross-origin")
            for name, call in calls.items():
                with self.subTest(sdk=name, case=case):
                    self.reset(respond)
                    try:
                        call()
                    except Exception:
                        pass  # parsing the stub reply may fail; what was sent is the point
                    expected = (redirect_sends if redirect else retry_sends)[name]
                    self.assertEqual(len(self.sent), expected)
                    self.assertEqual(len(self.sleeps), 0 if redirect else expected - 1)
                    if redirect and expected == 2:
                        self.assertEqual(self.sent[-1].url.host, OTHER_HOST)
                        self.assertIn(b"offline probe", self.sent[-1].content)
                        if name == "anthropic":
                            self.assertEqual(self.sent[-1].headers.get("x-api-key"), FAKE_KEY)


class TestBuiltinProvidersSendOnce(OfflineSdkCase):
    """Every built-in, every call shape, every retry trigger: exactly one HTTP send."""

    def test_every_builtin_call_shape_sends_exactly_once(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            provider = shared_provider(name)
            for shape, call in CALL_SHAPES.items():
                for case, respond in FAILURES.items():
                    with self.subTest(provider=name, call=shape, case=case):
                        self.reset(respond)
                        with self.assertRaises(Exception) as caught:
                            call(provider)
                        self.assertNotIsInstance(caught.exception, ProviderSdkPolicyError)
                        self.assertEqual(len(self.sent), 1)
                        self.assertEqual(self.sleeps, [])
                        self.assertEqual({request.url.host for request in self.sent} & {OTHER_HOST}, set())

    def test_anthropic_family_key_never_reaches_another_host(self) -> None:
        for name in sorted(ANTHROPIC_FAMILY):
            with self.subTest(provider=name):
                provider = shared_provider(name)
                self.reset(_redirect(307))
                with self.assertRaises(anthropic.APIStatusError) as caught:
                    provider.chat(MESSAGES)
                self.assertEqual(caught.exception.status_code, 307)
                self.assertEqual(len(self.sent), 1)
                self.assertNotEqual(self.sent[0].url.host, OTHER_HOST)


class TestBuiltinClientsCarryPolicy(OfflineSdkCase):
    """Real SDK clients (no request): max_retries is the int 0 and redirects are off."""

    def test_builtin_set_is_classified(self) -> None:
        self.assertEqual(set(_BUILTIN_PROVIDER_NAMES), ANTHROPIC_FAMILY | OPENAI_FAMILY | {"glm"})
        named = re.search(r"\(([^)]*)\)", provider_sdk_policy_status())
        assert named is not None
        self.assertEqual({part.strip() for part in named.group(1).split(",")}, set(_BUILTIN_PROVIDER_NAMES))

    def test_real_clients_have_zero_retries_and_no_redirects(self) -> None:
        for name in sorted(_BUILTIN_PROVIDER_NAMES):
            with self.subTest(provider=name):
                client = sdk_client(shared_provider(name))
                self.assertIs(type(client.max_retries), int)
                self.assertEqual(client.max_retries, 0)
                self.assertIs(client._client.follow_redirects, False)
                if name in ANTHROPIC_FAMILY:
                    self.assertIsInstance(client._client, anthropic.DefaultHttpxClient)
                    self.assertEqual(client.timeout, anthropic.DEFAULT_TIMEOUT)
                elif name in OPENAI_FAMILY:
                    self.assertIsInstance(client._client, openai.DefaultHttpxClient)
                    self.assertEqual(client.timeout, openai.DEFAULT_TIMEOUT)
        self.assertEqual(self.sent, [])

    def test_sdk_clients_and_http_clients_are_built_lazily(self) -> None:
        with patch("src.providers.anthropic_provider.anthropic.DefaultHttpxClient") as anthropic_http, \
             patch("src.providers.openai_provider.DefaultHttpxClient") as openai_http, \
             patch("src.providers.anthropic_provider.anthropic.Anthropic") as anthropic_sdk, \
             patch("src.providers.openai_provider.OpenAI") as openai_sdk, \
             patch("src.providers.glm_provider.ZhipuAI") as zhipu_sdk:
            for name in sorted(_BUILTIN_PROVIDER_NAMES):
                build_provider(name)
        for constructor in (anthropic_http, openai_http, anthropic_sdk, openai_sdk, zhipu_sdk):
            constructor.assert_not_called()

    def test_sdk_constructors_still_accept_the_controls(self) -> None:
        for sdk_class in (anthropic.Anthropic, openai.OpenAI):
            parameters = inspect.signature(sdk_class).parameters
            self.assertIn("max_retries", parameters)
            self.assertIn("http_client", parameters)
        self.assertIn("max_retries", inspect.signature(zhipuai.ZhipuAI).parameters)


class TestRetryValueValidation(OfflineSdkCase):
    def test_none_and_zero_resolve_to_the_int_zero(self) -> None:
        for value in (None, 0):
            resolved = resolve_max_retries(value)
            self.assertIs(type(resolved), int)
            self.assertEqual(resolved, 0)

    def test_every_other_value_is_rejected(self) -> None:
        for value in [*REJECTED_RETRY_VALUES, _Int(0)]:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ProviderSdkPolicyError):
                    resolve_max_retries(value)

    def test_openai_family_rejects_before_any_sdk_client_exists(self) -> None:
        for provider_class in (OpenAIProvider, QwenProvider):
            for value in REJECTED_RETRY_VALUES:
                with self.subTest(provider=provider_class.__name__, value=repr(value)), \
                     patch("src.providers.openai_provider.OpenAI") as sdk, \
                     patch("src.providers.openai_provider.DefaultHttpxClient") as http:
                    with self.assertRaises(ProviderSdkPolicyError):
                        provider_class(api_key=FAKE_KEY, max_retries=value)
                    sdk.assert_not_called()
                    http.assert_not_called()

    def test_openai_without_default_http_client_fails_closed(self) -> None:
        provider = OpenAIProvider(api_key=FAKE_KEY)
        with patch("src.providers.openai_provider.DefaultHttpxClient", None), \
             patch("src.providers.openai_provider.OpenAI") as sdk:
            with self.assertRaises(ProviderSdkPolicyError):
                _ = provider.client
            sdk.assert_not_called()
        self.assertIsNone(provider._client)

    def test_deepseek_has_no_retry_parameter(self) -> None:
        self.assertNotIn("max_retries", inspect.signature(DeepSeekProvider).parameters)
        with self.assertRaises(TypeError):
            DeepSeekProvider(api_key=FAKE_KEY, max_retries=0)  # type: ignore[call-arg]


class TestRuntimePolicyCheck(OfflineSdkCase):
    def test_accepts_only_int_zero(self) -> None:
        client = MagicMock(max_retries=0)
        self.assertIs(require_sdk_retry_policy(client), client)
        for value in (2, 1, -1, True, False, 0.0, None, "0", _Int(0), MagicMock()):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ProviderSdkPolicyError):
                    require_sdk_retry_policy(MagicMock(max_retries=value))
        with self.assertRaises(ProviderSdkPolicyError):
            require_sdk_retry_policy(object())

    def test_providers_refuse_and_do_not_cache_an_sdk_that_ignores_the_policy(self) -> None:
        ignored = MagicMock(max_retries=2)
        cases = [
            ("src.providers.anthropic_provider.anthropic.Anthropic", lambda: AnthropicProvider(FAKE_KEY)),
            ("src.providers.minimax_provider.anthropic.Anthropic", lambda: MinimaxProvider(FAKE_KEY)),
            ("src.providers.openai_provider.OpenAI", lambda: OpenAIProvider(FAKE_KEY)),
            ("src.providers.openai_provider.OpenAI", lambda: DeepSeekProvider(FAKE_KEY)),
            ("src.providers.openai_provider.OpenAI", lambda: QwenProvider(FAKE_KEY, max_retries=0)),
            ("src.providers.glm_provider.ZhipuAI", lambda: GLMProvider(FAKE_KEYS["glm"])),
        ]
        for target, make in cases:
            provider = make()
            with self.subTest(provider=type(provider).__name__), patch(target, return_value=ignored), \
                 patch("src.providers.anthropic_provider.anthropic.DefaultHttpxClient"), \
                 patch("src.providers.openai_provider.DefaultHttpxClient"):
                for _ in range(2):
                    with self.assertRaises(ProviderSdkPolicyError):
                        sdk_client(provider)
                cached = provider.client if isinstance(provider, (AnthropicProvider, MinimaxProvider)) else provider._client
                self.assertIsNone(cached)


class TestConstructionSites(unittest.TestCase):
    """Static guard: SDK clients are built only at the four policy sites."""

    SDK_CLIENTS = {
        "Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AsyncAnthropicBedrock", "AnthropicVertex",
        "AsyncAnthropicVertex", "AnthropicAWS", "AsyncAnthropicAWS", "AnthropicFoundry",
        "AsyncAnthropicFoundry", "OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "ZhipuAI",
    }
    # SDK aliases of Anthropic/OpenAI (and async); generic names, so only counted when they come
    # from an SDK module (other libraries, e.g. MCP, also export a `Client`).
    SDK_ALIASES = {"Client", "AsyncClient"}
    SDK_MODULES = ("anthropic", "openai", "zhipuai")
    POLICY_KEYWORDS = {"max_retries", "http_client", "follow_redirects"}
    # (file, callee, keyword) sites allowed to pass a policy keyword; everything else is flagged,
    # which covers copy()/with_options()/partial() overrides under any name.
    ALLOWED_POLICY_KEYWORDS = {
        ("providers/anthropic_provider.py", "Anthropic", "http_client"),
        ("providers/minimax_provider.py", "Anthropic", "http_client"),
        ("providers/anthropic_provider.py", "DefaultHttpxClient", "follow_redirects"),
        ("providers/minimax_provider.py", "DefaultHttpxClient", "follow_redirects"),
        ("providers/openai_provider.py", "DefaultHttpxClient", "follow_redirects"),
        ("tool_system/tools/qwen_media.py", "QwenProvider", "max_retries"),
        ("providers/qwen_provider.py", "__init__", "max_retries"),  # validated by OpenAIProvider
    }

    def _trees(self) -> list[tuple[str, ast.AST]]:
        return [
            (path.relative_to(SRC).as_posix(), ast.parse(path.read_text(encoding="utf-8-sig")))
            for path in sorted(SRC.rglob("*.py"))
        ]

    @staticmethod
    def _called_name(node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _sdk_constructions(self, tree: ast.AST) -> list[str]:
        from_sdk = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in self.SDK_MODULES
            for alias in node.names
        }
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._called_name(node)
            if name in self.SDK_CLIENTS:
                found.append(name)
            elif name in self.SDK_ALIASES and (
                (isinstance(node.func, ast.Name) and name in from_sdk)
                or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in self.SDK_MODULES)
            ):
                found.append(name)
        return found

    def test_sdk_clients_are_constructed_only_at_policy_sites(self) -> None:
        sites = sorted((rel, name) for rel, tree in self._trees() for name in self._sdk_constructions(tree))
        self.assertEqual(sites, [
            ("providers/anthropic_provider.py", "Anthropic"),
            ("providers/glm_provider.py", "ZhipuAI"),
            ("providers/minimax_provider.py", "Anthropic"),
            ("providers/openai_provider.py", "OpenAI"),
        ])

    def test_every_default_http_client_disables_redirects(self) -> None:
        sites = []
        for rel, tree in self._trees():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and self._called_name(node) == "DefaultHttpxClient":
                    keywords = {kw.arg: kw.value for kw in node.keywords}
                    follow = keywords.get("follow_redirects")
                    self.assertTrue(isinstance(follow, ast.Constant) and follow.value is False, rel)
                    sites.append(rel)
        self.assertEqual(sorted(sites), [
            "providers/anthropic_provider.py",
            "providers/minimax_provider.py",
            "providers/openai_provider.py",
        ])

    def test_no_post_construction_retry_override_or_aliased_sdk_import(self) -> None:
        problems = []
        for rel, tree in self._trees():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    callee = self._called_name(node)
                    if callee == "with_options":
                        problems.append(f"{rel}:{node.lineno} with_options")
                    for kw in node.keywords:
                        if kw.arg in self.POLICY_KEYWORDS \
                                and (rel, callee, kw.arg) not in self.ALLOWED_POLICY_KEYWORDS:
                            problems.append(f"{rel}:{node.lineno} {callee}({kw.arg}=...)")
                    if callee == "setattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) \
                            and node.args[1].value in self.POLICY_KEYWORDS:
                        problems.append(f"{rel}:{node.lineno} setattr {node.args[1].value}")
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr in self.POLICY_KEYWORDS:
                        problems.append(f"{rel}:{node.lineno} assigns {target.attr}")
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in self.SDK_MODULES:
                    for alias in node.names:
                        if alias.asname and alias.name in self.SDK_CLIENTS | {"DefaultHttpxClient"}:
                            problems.append(f"{rel}:{node.lineno} aliases {alias.name}")
                # `import openai` would expose its module-level client (SDK default retries, redirects).
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "openai":
                            problems.append(f"{rel}:{node.lineno} import openai")
        self.assertEqual(problems, [])

    def test_doctor_policy_module_imports_no_sdk(self) -> None:
        tree = ast.parse((SRC / "providers" / "sdk_policy.py").read_text(encoding="utf-8-sig"))
        imported = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertEqual(imported, {"__future__", "typing"})


if __name__ == "__main__":
    unittest.main()
