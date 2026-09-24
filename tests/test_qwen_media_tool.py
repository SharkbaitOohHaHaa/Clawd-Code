from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from src.provider_event_ledger import append_provider_event
from src.providers.base import ChatResponse
from src.providers.qwen_provider import QwenProvider
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolExecutionError, ToolInputError, ToolPermissionError
from src.tool_system.permission_handler import PermissionBehavior
from src.tool_system.tools.qwen_media import (
    QwenMediaAnalyzeTool,
    _MAX_IMAGE_BYTES,
    _download_public_media,
    _probe_video_duration,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"safe-test-image"
class _Response(io.BytesIO):
    def __init__(self, data: bytes, *, content_type: str, url: str, length: int | None = None):
        super().__init__(data)
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(data) if length is None else length),
        }
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return self._url


class _Opener:
    def __init__(self, response: _Response):
        self.response = response

    def open(self, req, timeout=0):
        return self.response


class TestQwenMediaAnalyzeTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.tool = QwenMediaAnalyzeTool()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self, **extra: str) -> dict[str, str]:
        env = {
            "QWEN_MEDIA_LIVE_ENABLED": "1",
            "DASHSCOPE_API_KEY": "dummy-key",
        }
        env.update(extra)
        return env

    def test_requires_explicit_outbound_permission(self) -> None:
        result = self.tool.check_permissions(
            {"source": "https://example.com/image.jpg"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("Qwen", result.message or "")

    def test_live_inference_disabled_by_default(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        with patch.dict(
            os.environ,
            {"QWEN_MEDIA_LIVE_ENABLED": "", "DASHSCOPE_API_KEY": "dummy-key"},
            clear=False,
        ), patch.object(QwenProvider, "chat") as chat, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            with self.assertRaisesRegex(ToolPermissionError, "live inference is disabled"):
                self.tool.run({"source": str(image)}, self.ctx)

        chat.assert_not_called()
        self.assertEqual(event.call_args.kwargs["stage"], "validation")
        self.assertEqual(event.call_args.kwargs["status"], "failed")

    def test_missing_key_fails_closed(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        with patch.dict(
            os.environ,
            self._env(DASHSCOPE_API_KEY=""),
            clear=False,
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            with self.assertRaisesRegex(ToolInputError, "DASHSCOPE_API_KEY"):
                self.tool.run({"source": str(image)}, self.ctx)

    def test_local_image_usage_metadata_and_timeout(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        response = ChatResponse(
            content="The image contains a harmless test object.",
            model="qwen3.8-max",
            usage={
                "input_tokens": 120,
                "output_tokens": 20,
                "thought_tokens": 5,
                "cached_tokens": 10,
                "total_tokens": 140,
            },
            finish_reason="stop",
        )
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            return_value=response,
        ) as chat, patch(
            "src.tool_system.context.append_provider_usage"
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            result = self.tool.run(
                {"source": str(image), "question": "What is shown?"},
                self.ctx,
            )

        content = chat.call_args.args[0][1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 4096)
        self.assertEqual(chat.call_args.kwargs["timeout"], 300.0)
        self.assertEqual(result.output["provider"], "Alibaba Qwen")
        self.assertEqual(
            result.output["endpoint"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(result.output["model"], "qwen3.8-max")
        self.assertEqual(result.output["trust"], "untrusted_external_model_output")
        self.assertIn("untrusted_analysis", result.output)
        self.assertNotIn("data:", str(result.output))
        self.assertEqual(
            self.ctx.usage_records,
            [{
                "label": "Qwen (qwen3.8-max)",
                "input_tokens": 120,
                "output_tokens": 20,
                "thought_tokens": 5,
                "tool_use_tokens": 0,
                "cached_tokens": 10,
                "total_tokens": 140,
            }],
        )
        self.assertEqual(event.call_count, 4)
        self.assertEqual(
            [(call.kwargs["stage"], call.kwargs["status"]) for call in event.call_args_list],
            [
                ("execution", "started"),
                ("provider_attempt", "started"),
                ("provider_attempt", "success"),
                ("execution", "success"),
            ],
        )
        logical_call_ids = {
            call.kwargs["logical_call_id"] for call in event.call_args_list
        }
        self.assertEqual(len(logical_call_ids), 1)
        self.assertTrue(next(iter(logical_call_ids)))
        self.assertEqual(event.call_args_list[1].kwargs["attempt"], 1)
        self.assertEqual(event.call_args_list[2].kwargs["attempt"], 1)
        for call in event.call_args_list:
            self.assertEqual(call.kwargs["provider"], "Alibaba Qwen")
            self.assertEqual(call.kwargs["model"], "qwen3.8-max")
            self.assertEqual(
                call.kwargs["endpoint"],
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )

    def test_workspace_traversal_is_rejected(self) -> None:
        outside = self.root.parent / "outside.png"
        outside.write_bytes(_PNG)
        try:
            with patch.dict(os.environ, self._env(), clear=False), patch(
                "src.tool_system.tools.qwen_media.append_provider_event"
            ):
                with self.assertRaises(ToolPermissionError):
                    self.tool.run(
                        {"source": str(self.root / ".." / "outside.png")},
                        self.ctx,
                    )
        finally:
            outside.unlink(missing_ok=True)

    def test_workspace_symlink_escape_is_rejected_when_supported(self) -> None:
        outside_dir = Path(tempfile.mkdtemp()).resolve()
        outside = outside_dir / "outside.png"
        outside.write_bytes(_PNG)
        link = self.root / "escape.png"
        try:
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable on this Windows host")
            with patch.dict(os.environ, self._env(), clear=False), patch(
                "src.tool_system.tools.qwen_media.append_provider_event"
            ):
                with self.assertRaises(ToolPermissionError):
                    self.tool.run({"source": str(link)}, self.ctx)
        finally:
            link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)
            outside_dir.rmdir()

    def test_unsupported_url_protocol_is_rejected(self) -> None:
        with patch.dict(os.environ, self._env(), clear=False), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            with self.assertRaisesRegex(ToolInputError, "workspace path or http/https"):
                self.tool.run(
                    {
                        "source": "ftp://example.com/image.png",
                        "media_type": "image",
                    },
                    self.ctx,
                )

    def test_private_public_url_is_rejected_before_download(self) -> None:
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 0))],
        ), patch.object(urllib.request, "build_opener") as opener:
            with self.assertRaisesRegex(ToolPermissionError, "private network"):
                _download_public_media(
                    "http://localhost/image.png",
                    "image",
                )
        opener.assert_not_called()

    def test_unsafe_redirect_is_rejected(self) -> None:
        class _RedirectingOpener:
            def __init__(self, handler):
                self.handler = handler

            def open(self, req, timeout=0):
                self.handler.redirect_request(
                    req,
                    None,
                    302,
                    "Found",
                    {},
                    "http://127.0.0.1/private.png",
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

        with patch.object(socket, "getaddrinfo", side_effect=fake_getaddrinfo), patch.object(
            urllib.request,
            "build_opener",
            side_effect=fake_build_opener,
        ):
            with self.assertRaisesRegex(ToolPermissionError, "private network"):
                _download_public_media(
                    "https://example.com/image.png",
                    "image",
                )

    def test_public_url_download_size_is_bounded(self) -> None:
        response = _Response(
            _PNG,
            content_type="image/png",
            url="https://example.com/image.png",
            length=_MAX_IMAGE_BYTES + 1,
        )
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ), patch.object(
            urllib.request,
            "build_opener",
            return_value=_Opener(response),
        ):
            with self.assertRaisesRegex(ToolInputError, "safe limit"):
                _download_public_media(
                    "https://example.com/image.png",
                    "image",
                )

    def test_public_url_mime_content_mismatch_is_rejected(self) -> None:
        response = _Response(
            _PNG,
            content_type="image/jpeg",
            url="https://example.com/image.jpg",
        )
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ), patch.object(
            urllib.request,
            "build_opener",
            return_value=_Opener(response),
        ):
            with self.assertRaisesRegex(ToolInputError, "MIME/content mismatch"):
                _download_public_media(
                    "https://example.com/image.jpg",
                    "image",
                )

    def test_video_duration_limits_are_enforced(self) -> None:
        video = self.root / "video.mp4"
        video.write_bytes(b"fake-video")

        short = subprocess.CompletedProcess([], 0, stdout="1.5\n", stderr="")
        with patch(
            "src.tool_system.tools.qwen_media.subprocess.run",
            return_value=short,
        ):
            with self.assertRaisesRegex(ToolInputError, "between 2 seconds and 2 hours"):
                _probe_video_duration(video)

        long = subprocess.CompletedProcess([], 0, stdout="7200.1\n", stderr="")
        with patch(
            "src.tool_system.tools.qwen_media.subprocess.run",
            return_value=long,
        ):
            with self.assertRaisesRegex(ToolInputError, "between 2 seconds and 2 hours"):
                _probe_video_duration(video)

        valid = subprocess.CompletedProcess([], 0, stdout="3.0\n", stderr="")
        with patch(
            "src.tool_system.tools.qwen_media.subprocess.run",
            return_value=valid,
        ):
            self.assertEqual(_probe_video_duration(video), 3.0)

    def test_public_url_download_timeout_is_bounded(self) -> None:
        class _TimeoutOpener:
            def open(self, req, timeout=0):
                self.timeout = timeout
                raise urllib.error.URLError(TimeoutError("timed out"))

        opener = _TimeoutOpener()
        with patch.object(
            socket,
            "getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ), patch.object(
            urllib.request,
            "build_opener",
            return_value=opener,
        ):
            with self.assertRaisesRegex(TimeoutError, "download timed out"):
                _download_public_media(
                    "https://example.com/image.png",
                    "image",
                )
        self.assertEqual(opener.timeout, 15)

    def test_oversized_local_file_is_rejected_before_read(self) -> None:
        image = self.root / "huge.png"
        with image.open("wb") as handle:
            handle.seek(_MAX_IMAGE_BYTES)
            handle.write(b"x")

        with patch.dict(os.environ, self._env(), clear=False), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ), patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("oversized media should not be read"),
        ):
            with self.assertRaisesRegex(ToolInputError, "safe inline limit"):
                self.tool.run({"source": str(image)}, self.ctx)

    def test_provider_event_ledger_records_exact_nonsecret_metadata(self) -> None:
        ledger = self.root / "provider-events.jsonl"
        with patch.dict(
            os.environ,
            {"CLAWD_PROVIDER_EVENT_LEDGER": str(ledger)},
            clear=False,
        ):
            append_provider_event(
                provider="Alibaba Qwen",
                operation="QwenMediaAnalyze",
                stage="execution",
                status="success",
                model="qwen3.8-max",
                endpoint="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                logical_call_id="logical-123",
                attempt=2,
                elapsed_ms=345,
                retryable=True,
            )

        event = json.loads(ledger.read_text(encoding="utf-8").strip())
        self.assertEqual(event["provider"], "Alibaba Qwen")
        self.assertEqual(event["model"], "qwen3.8-max")
        self.assertEqual(
            event["endpoint"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(event["logical_call_id"], "logical-123")
        self.assertEqual(event["attempt"], 2)
        self.assertEqual(event["elapsed_ms"], 345)
        self.assertTrue(event["retryable"])
        self.assertNotIn("api_key", event)
        self.assertNotIn("source", event)
        self.assertNotIn("media", event)

    def test_provider_failure_is_logged_distinctly(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        request = httpx.Request("POST", "https://example.test")
        error = APIError("provider failed", request=request, body=None)
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            side_effect=error,
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            with self.assertRaisesRegex(ToolExecutionError, "provider request failed"):
                self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(event.call_args.kwargs["stage"], "provider")
        self.assertEqual(event.call_args.kwargs["status"], "failed")

    def test_transient_connection_failure_retries_and_groups_attempts(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        request = httpx.Request("POST", "https://example.test")
        error = APIConnectionError(request=request)
        response = ChatResponse(
            content="Recovered after a transient connection failure.",
            model="qwen3.8-max",
            usage={},
            finish_reason="stop",
        )
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            side_effect=[error, response],
        ) as chat, patch(
            "src.tool_system.tools.qwen_media.time.sleep"
        ) as sleep, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            result = self.tool.run({"source": str(image)}, self.ctx)

        self.assertIn("Recovered", result.output["untrusted_analysis"])
        self.assertEqual(chat.call_count, 2)
        sleep.assert_called_once_with(1.0)
        attempts = [
            call.kwargs
            for call in event.call_args_list
            if call.kwargs["stage"] == "provider_attempt"
        ]
        self.assertEqual(
            [(item["attempt"], item["status"]) for item in attempts],
            [(1, "started"), (1, "failed"), (2, "started"), (2, "success")],
        )
        self.assertTrue(attempts[1]["retryable"])
        self.assertIn("elapsed_ms", attempts[1])
        self.assertEqual(
            len({item["logical_call_id"] for item in attempts}),
            1,
        )

    def test_transient_failures_stop_after_two_retries(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        error = APIConnectionError(
            request=httpx.Request("POST", "https://example.test")
        )
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            side_effect=error,
        ) as chat, patch(
            "src.tool_system.tools.qwen_media.time.sleep"
        ) as sleep, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            with self.assertRaisesRegex(ToolExecutionError, "provider request failed"):
                self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(chat.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    def test_429_and_5xx_errors_are_retryable(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        response = ChatResponse(
            content="Recovered.",
            model="qwen3.8-max",
            usage={},
            finish_reason="stop",
        )
        request = httpx.Request("POST", "https://example.test")
        errors = [
            RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=request),
                body=None,
            ),
            InternalServerError(
                "server error",
                response=httpx.Response(503, request=request),
                body=None,
            ),
        ]
        for error in errors:
            with self.subTest(status=error.status_code), patch.dict(
                os.environ,
                self._env(),
                clear=False,
            ), patch.object(
                QwenProvider,
                "chat",
                side_effect=[error, response],
            ) as chat, patch(
                "src.tool_system.tools.qwen_media.time.sleep"
            ) as sleep, patch(
                "src.tool_system.tools.qwen_media.append_provider_event"
            ):
                self.tool.run({"source": str(image)}, self.ctx)

            self.assertEqual(chat.call_count, 2)
            sleep.assert_called_once_with(1.0)

    def test_4xx_validation_error_is_not_retried(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        request = httpx.Request("POST", "https://example.test")
        error = APIStatusError(
            "bad request",
            response=httpx.Response(400, request=request),
            body=None,
        )
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            side_effect=error,
        ) as chat, patch(
            "src.tool_system.tools.qwen_media.time.sleep"
        ) as sleep, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            with self.assertRaisesRegex(ToolExecutionError, "provider request failed"):
                self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(chat.call_count, 1)
        sleep.assert_not_called()

    def test_timeout_and_total_deadline_are_configurable(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        response = ChatResponse(
            content="Within configured deadline.",
            model="qwen3.8-max",
            usage={},
            finish_reason="stop",
        )
        env = self._env(
            QWEN_MEDIA_PROVIDER_TIMEOUT_SECONDS="300",
            QWEN_MEDIA_TOTAL_DEADLINE_SECONDS="45",
            QWEN_MEDIA_MAX_TRANSIENT_RETRIES="0",
        )
        with patch.dict(os.environ, env, clear=False), patch(
            "src.tool_system.tools.qwen_media.time.monotonic",
            return_value=100.0,
        ), patch.object(
            QwenProvider,
            "chat",
            return_value=response,
        ) as chat, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(chat.call_args.kwargs["timeout"], 45.0)

    def test_transient_retry_setting_cannot_exceed_two(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        with patch.dict(
            os.environ,
            self._env(QWEN_MEDIA_MAX_TRANSIENT_RETRIES="3"),
            clear=False,
        ), patch.object(QwenProvider, "chat") as chat, patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ):
            with self.assertRaisesRegex(ToolInputError, "integer from 0 to 2"):
                self.tool.run({"source": str(image)}, self.ctx)

        chat.assert_not_called()

    def test_provider_timeout_is_logged_distinctly(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        error = APITimeoutError(
            request=httpx.Request("POST", "https://example.test")
        )
        with patch.dict(
            os.environ,
            self._env(QWEN_MEDIA_MAX_TRANSIENT_RETRIES="0"),
            clear=False,
        ), patch.object(
            QwenProvider,
            "chat",
            side_effect=error,
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            with self.assertRaisesRegex(ToolExecutionError, "timed out"):
                self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(event.call_args.kwargs["stage"], "timeout")
        self.assertEqual(event.call_args.kwargs["status"], "failed")

    def test_adapter_failure_is_logged_distinctly(self) -> None:
        image = self.root / "image.png"
        image.write_bytes(_PNG)
        response = ChatResponse(
            content="",
            model="qwen3.8-max",
            usage={},
            finish_reason="stop",
        )
        with patch.dict(os.environ, self._env(), clear=False), patch.object(
            QwenProvider,
            "chat",
            return_value=response,
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            with self.assertRaisesRegex(ToolExecutionError, "invalid media response"):
                self.tool.run({"source": str(image)}, self.ctx)

        self.assertEqual(event.call_args.kwargs["stage"], "adapter")
        self.assertEqual(event.call_args.kwargs["status"], "failed")

    def test_execution_failure_is_logged_distinctly(self) -> None:
        with patch.dict(os.environ, self._env(), clear=False), patch(
            "src.tool_system.tools.qwen_media._prepare_media",
            side_effect=OSError("disk error"),
        ), patch(
            "src.tool_system.tools.qwen_media.append_provider_event"
        ) as event:
            with self.assertRaises(OSError):
                self.tool.run(
                    {
                        "source": str(self.root / "image.png"),
                        "media_type": "image",
                    },
                    self.ctx,
                )

        self.assertEqual(event.call_args.kwargs["stage"], "execution")
        self.assertEqual(event.call_args.kwargs["status"], "failed")

    def test_default_registry_has_exactly_one_qwen_media_tool(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        names = [spec.name for spec in registry.list_specs()]
        self.assertEqual(names.count("QwenMediaAnalyze"), 1)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
