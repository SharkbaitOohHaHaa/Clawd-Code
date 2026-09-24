"""Provider SDK request policy: one Clawd provider attempt is one SDK request attempt.

Built-in providers construct their SDK clients with ``max_retries=SDK_MAX_RETRIES`` so the
SDK never re-sends a request underneath Clawd, and the anthropic/openai SDK families are
given ``DefaultHttpxClient(follow_redirects=False)`` so a redirect cannot re-send the prompt
(or, for anthropic, the ``X-Api-Key`` header) to another host. The zhipuai SDK does not
follow redirects. Plugin providers own their HTTP stack and are not enforced.

This module must stay import-light: /doctor reads it without importing any SDK.
"""

from __future__ import annotations

from typing import Any, TypeVar

SDK_MAX_RETRIES = 0

_ClientT = TypeVar("_ClientT")


class ProviderSdkPolicyError(RuntimeError):
    """Raised instead of using an SDK client that could retry underneath Clawd."""


def resolve_max_retries(value: object = None) -> int:
    """Return the fixed SDK retry count; ``None`` means the policy.

    Any explicit value other than the integer 0 is rejected before an SDK client exists
    (bool, negative, float, inf, str and every other value).
    """
    if value is None:
        return SDK_MAX_RETRIES
    if type(value) is not int or value != SDK_MAX_RETRIES:
        raise ProviderSdkPolicyError(
            f"max_retries must be the integer {SDK_MAX_RETRIES}; SDK retries are disabled in Clawd"
        )
    return value


def require_sdk_retry_policy(client: _ClientT) -> _ClientT:
    """Fail closed unless a freshly built SDK client reports ``max_retries == 0`` as an int."""
    retries: Any = getattr(client, "max_retries", None)
    if type(retries) is not int or retries != SDK_MAX_RETRIES:
        raise ProviderSdkPolicyError(
            "The provider SDK client did not apply max_retries=0; Clawd refuses to use it "
            "because the SDK could re-send requests"
        )
    return client


def provider_sdk_policy_status() -> str:
    """Static, local description for /doctor (no SDK import, no provider call)."""
    return (
        "built-in providers (anthropic, minimax, openai, deepseek, qwen, glm) build SDK clients "
        f"with max_retries={SDK_MAX_RETRIES}; redirect following disabled for the anthropic and "
        "openai SDK families (the glm SDK does not follow redirects); plugin providers not "
        "enforced; static, no provider call"
    )
