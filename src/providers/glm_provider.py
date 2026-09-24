"""GLM (Zhipu AI) provider implementation."""

from __future__ import annotations

from typing import Any, Optional

try:
    from zhipuai import ZhipuAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ZhipuAI = None

from .openai_compatible import OpenAICompatibleProvider


class GLMProvider(OpenAICompatibleProvider):
    """GLM (Zhipu AI) provider using the Zhipu SDK."""

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    DEFAULT_MODEL = "glm-5-turbo"

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize GLM provider.

        Args:
            api_key: Zhipu AI API key
            base_url: Base URL (defaults to the BigModel OpenAI-compatible endpoint)
            model: Default model (default: glm-5-turbo)
        """
        super().__init__(
            api_key,
            base_url or self.DEFAULT_BASE_URL,
            model or self.DEFAULT_MODEL,
        )

    def _create_client(self) -> Any:
        """Create Zhipu AI SDK client."""
        if ZhipuAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "zhipuai package is not installed. Install optional dependencies to use GLMProvider."
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ZhipuAI(**kwargs)

    def get_available_models(self) -> list[str]:
        """Return current GLM model IDs used by Clawd."""
        return [
            "glm-5.2",
            "glm-5-turbo",
            "glm-5",
            "glm-4.7",
            "glm-4.6",
        ]
