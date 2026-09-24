"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

try:
    from openai import DefaultHttpxClient  # type: ignore
except ImportError:  # pragma: no cover - missing or too-old openai; fails closed in _create_client
    DefaultHttpxClient = None

from .openai_compatible import OpenAICompatibleProvider
from .sdk_policy import ProviderSdkPolicyError, require_sdk_retry_policy, resolve_max_retries


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider using OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Base URL (optional, for custom endpoints)
            model: Default model (default: gpt-5.4)
            max_retries: SDK retries are fixed at 0. ``None`` uses that policy; any other
                value than the integer 0 is rejected here, before an SDK client exists.
        """
        super().__init__(api_key, base_url, model or "gpt-5.4")
        self._max_retries = resolve_max_retries(max_retries)

    def _create_client(self) -> Any:
        """Create OpenAI SDK client (one request per attempt: no retries, no redirects)."""
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use OpenAIProvider."
            )
        if DefaultHttpxClient is None:
            raise ProviderSdkPolicyError(
                "The installed openai package has no DefaultHttpxClient, so Clawd cannot disable "
                "redirect following; upgrade openai"
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs["max_retries"] = self._max_retries
        # A followed 307/308 would re-POST the prompt, possibly to another host.
        kwargs["http_client"] = DefaultHttpxClient(follow_redirects=False)
        return require_sdk_retry_policy(OpenAI(**kwargs))

    def get_available_models(self) -> list[str]:
        """Get list of available OpenAI models.

        Returns:
            List of model names
        """
        return [
            # GPT-5.4 series (latest flagship)
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            # GPT-5.2 series
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
        ]
