"""DeepSeek provider implementation."""

from __future__ import annotations

from typing import Optional

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider using its OpenAI-compatible Chat Completions API."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-flash"
    FINISH_STATUS_PROFILE = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
        )

    def get_available_models(self) -> list[str]:
        """Return current canonical DeepSeek API model IDs."""
        return ["deepseek-flash", "deepseek-v4-pro"]
