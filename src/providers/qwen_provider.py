"""Qwen provider implementation."""

from __future__ import annotations

from typing import Optional

from .openai_provider import OpenAIProvider


class QwenProvider(OpenAIProvider):
    """Alibaba Cloud Model Studio Qwen provider via OpenAI compatibility."""

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen3.8-max"
    FINISH_STATUS_PROFILE = "qwen"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            max_retries=max_retries,
        )

    def get_available_models(self) -> list[str]:
        """Return current Qwen text/coding models supported by Clawd."""
        return [
            "qwen3.8-max",
            "qwen3.8-flash",
            "qwen3.7-plus",
            "qwen3.7-flash",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
        ]
