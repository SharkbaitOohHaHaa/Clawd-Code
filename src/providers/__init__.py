"""LLM Providers for Clawd Codex."""

from __future__ import annotations

import inspect
import ipaddress
import re
from typing import Any, TypedDict
from urllib.parse import urlparse

from .base import BaseProvider, ChatMessage, ChatResponse


# Provider metadata for login/UI
class ProviderInfo(TypedDict, total=False):
    label: str
    default_base_url: str
    default_model: str
    available_models: list[str]
    requires_api_key: bool
    local_only: bool
    plugin_name: str
    artifact_sha256: str


PROVIDER_INFO: dict[str, ProviderInfo] = {
    "anthropic": {
        "label": "Anthropic Claude",
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-6",
        "available_models": [
            # Claude 4 series (latest)
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-0",
            "claude-sonnet-4-20250514",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
            "claude-opus-4-0",
            "claude-opus-4-20250514",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            # Legacy
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "openai": {
        "label": "OpenAI GPT",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.4",
        "available_models": [
            # GPT-5.4 series (latest flagship)
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            # GPT-5.2 series (previous)
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.2-mini",
            "gpt-5.2-nano",
            # GPT-5.3-Codex (coding-specialized)
            "gpt-5.3-codex",
            # Legacy GPT-4 series
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-flash",
        "available_models": [
            "deepseek-flash",
            "deepseek-v4-pro",
        ],
    },
    "qwen": {
        "label": "Alibaba Qwen",
        "default_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3.8-max",
        "available_models": [
            "qwen3.8-max",
            "qwen3.8-flash",
            "qwen3.7-plus",
            "qwen3.7-flash",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
        ],
    },
    "glm": {
        "label": "Zhipu GLM",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5-turbo",
        "available_models": [
            "glm-5.2",
            "glm-5-turbo",
            "glm-5",
            "glm-4.7",
            "glm-4.6",
        ],
    },
    "minimax": {
        "label": "MiniMax AI",
        "default_base_url": "https://api.minimaxi.com/anthropic",
        "default_model": "MiniMax-M3",
        "available_models": [
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "M2-her",
            # Historical
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2",
        ],
    },
}

_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BUILTIN_PROVIDER_NAMES = frozenset(PROVIDER_INFO)
_PLUGIN_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {}


def _is_loopback_base_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_provider_info(name: str, info: dict[str, Any]) -> ProviderInfo:
    if not _PROVIDER_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid provider name: {name!r}")

    label = str(info.get("label") or "").strip()
    base_url = str(info.get("default_base_url") or "").strip()
    default_model = str(info.get("default_model") or "").strip()
    models = info.get("available_models")
    requires_api_key = info.get("requires_api_key", True)
    local_only = info.get("local_only", False)

    if not label:
        raise ValueError("provider label is required")
    if not default_model:
        raise ValueError("provider default_model is required")
    if not isinstance(models, list) or not all(
        isinstance(model, str) and model.strip() for model in models
    ):
        raise ValueError("provider available_models must be a non-empty string list")
    if not models:
        raise ValueError("provider available_models must not be empty")
    normalized_models = [model.strip() for model in models]
    if default_model not in normalized_models:
        raise ValueError("provider default_model must appear in available_models")
    if not isinstance(requires_api_key, bool) or not isinstance(local_only, bool):
        raise ValueError("provider requires_api_key/local_only must be booleans")
    if not requires_api_key and not local_only:
        raise ValueError("credentialless providers must be local_only")
    if local_only and not _is_loopback_base_url(base_url):
        raise ValueError("local_only provider default_base_url must be loopback/localhost")
    if not local_only:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("provider default_base_url must be an http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("provider default_base_url must not embed credentials")

    return ProviderInfo(
        label=label,
        default_base_url=base_url,
        default_model=default_model,
        available_models=normalized_models,
        requires_api_key=requires_api_key,
        local_only=local_only,
    )


def register_plugin_provider(
    name: str,
    provider_class: type[BaseProvider],
    info: dict[str, Any],
    *,
    plugin_name: str,
    artifact_sha256: str,
) -> None:
    """Register a trusted plugin provider without instantiating it."""
    normalized_name = str(name or "").strip().lower()
    if normalized_name in PROVIDER_INFO:
        raise ValueError(f"provider name collision: {normalized_name}")
    if (
        not inspect.isclass(provider_class)
        or not issubclass(provider_class, BaseProvider)
        or inspect.isabstract(provider_class)
    ):
        raise ValueError("provider_class must be a concrete BaseProvider subclass")
    try:
        inspect.signature(provider_class).bind(
            api_key="",
            base_url=None,
            model=None,
        )
    except TypeError as exc:
        raise ValueError(
            "provider_class constructor must accept api_key, base_url, and model"
        ) from exc

    normalized_info = _validate_provider_info(normalized_name, info)
    normalized_info["plugin_name"] = plugin_name
    normalized_info["artifact_sha256"] = artifact_sha256
    PROVIDER_INFO[normalized_name] = normalized_info
    _PLUGIN_PROVIDER_CLASSES[normalized_name] = provider_class
    AVAILABLE_PROVIDERS[normalized_name] = normalized_info["label"]


def unregister_plugin_provider(name: str) -> None:
    normalized_name = str(name or "").strip().lower()
    if normalized_name in _BUILTIN_PROVIDER_NAMES:
        raise ValueError(f"cannot unregister built-in provider: {normalized_name}")
    PROVIDER_INFO.pop(normalized_name, None)
    _PLUGIN_PROVIDER_CLASSES.pop(normalized_name, None)
    AVAILABLE_PROVIDERS.pop(normalized_name, None)


def clear_plugin_providers() -> None:
    for name in list(_PLUGIN_PROVIDER_CLASSES):
        unregister_plugin_provider(name)


def validate_provider_runtime_config(provider_name: str, config: dict[str, Any]) -> None:
    """Validate credential and endpoint policy before provider construction."""
    info = get_provider_info(provider_name)
    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or info.get("default_base_url") or "")
    if info.get("requires_api_key", True) and not api_key:
        raise ValueError("API key not configured")
    if info.get("local_only", False):
        if not _is_loopback_base_url(base_url):
            raise ValueError("local-only provider base_url must be loopback/localhost")
        return

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("provider base_url must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("provider base_url must not embed credentials")


def get_provider_info(provider_name: str) -> ProviderInfo:
    """Get provider info by name."""
    if provider_name not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider_name}")
    return PROVIDER_INFO[provider_name]


def get_provider_class(provider_name: str):
    """Get provider class by name."""
    if provider_name in _PLUGIN_PROVIDER_CLASSES:
        return _PLUGIN_PROVIDER_CLASSES[provider_name]
    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider
    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider
    if provider_name == "deepseek":
        from .deepseek_provider import DeepSeekProvider

        return DeepSeekProvider
    if provider_name == "qwen":
        from .qwen_provider import QwenProvider

        return QwenProvider
    if provider_name == "glm":
        from .glm_provider import GLMProvider

        return GLMProvider
    if provider_name == "minimax":
        from .minimax_provider import MinimaxProvider

        return MinimaxProvider
    raise ValueError(f"Unknown provider: {provider_name}")


# Legacy registry for display purposes
AVAILABLE_PROVIDERS: dict[str, str] = {k: v["label"] for k, v in PROVIDER_INFO.items()}


__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "get_provider_class",
    "get_provider_info",
    "register_plugin_provider",
    "unregister_plugin_provider",
    "clear_plugin_providers",
    "validate_provider_runtime_config",
    "PROVIDER_INFO",
    "AVAILABLE_PROVIDERS",
]
