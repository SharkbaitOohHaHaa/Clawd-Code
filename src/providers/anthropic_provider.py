"""Anthropic provider implementation."""

from __future__ import annotations

from typing import Generator, Optional, Any

try:
    import anthropic  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _MissingAnthropic:
        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError(
                    "anthropic package is not installed. Install optional dependencies to use AnthropicProvider."
                )

        DefaultHttpxClient = Anthropic

    anthropic = _MissingAnthropic()

from .base import (
    BaseProvider,
    ChatResponse,
    IncompleteResponseError,
    InvalidToolInputError,
    MessageInput,
    TextChunkCallback,
)
from .sdk_policy import SDK_MAX_RETRIES, require_sdk_retry_policy


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Base URL (optional)
            model: Default model (default: claude-sonnet-4-6)
        """
        super().__init__(api_key, base_url, model or "claude-sonnet-4-6")

        self._client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": SDK_MAX_RETRIES}
        if base_url:
            self._client_kwargs["base_url"] = base_url
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        # One Clawd attempt = one SDK request: no SDK retries, and no redirects that could
        # re-send the prompt and X-Api-Key to another host (httpx strips only Authorization).
        client = anthropic.Anthropic(
            **self._client_kwargs,
            http_client=anthropic.DefaultHttpxClient(follow_redirects=False),
        )
        self.client = require_sdk_retry_policy(client)
        return self.client

    def _build_chat_response(self, response: Any) -> ChatResponse:
        """Convert Anthropic SDK response into the shared ChatResponse shape."""
        content_text = ""
        tool_blocks: list[Any] = []

        for block in response.content:
            block_type = getattr(block, "type", "text")
            if block_type == "text":
                text_val = getattr(block, "text", "")
                if text_val is not None:
                    content_text += str(text_val)
            elif block_type == "tool_use":
                tool_blocks.append(block)

        usage = getattr(response, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }
        # The provider says it stopped at max_tokens (streamed or not): never a complete
        # answer, and a tool call built from its partial input must never run.
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise IncompleteResponseError(
                "tool_input_truncated" if tool_blocks else "output_limit",
                partial_text=content_text,
                partial_usage=usage_dict,
                tool_call_dropped=bool(tool_blocks),
            )
        # Only after the sealed output-limit check: a tool input that is not a real object
        # is rejected, never coerced (e.g. a list of pairs into an object) and run.
        tool_uses: list[dict[str, Any]] = []
        for block in tool_blocks:
            tool_input = getattr(block, "input", None)
            if not isinstance(tool_input, dict):
                raise InvalidToolInputError()
            tool_uses.append({
                "id": str(getattr(block, "id", "")),
                "name": str(getattr(block, "name", "")),
                "input": dict(tool_input),
            })
        return ChatResponse(
            content=content_text,
            model=getattr(response, "model", self.model or ""),
            usage=usage_dict,
            finish_reason=str(getattr(response, "stop_reason", "stop")),
            tool_uses=tool_uses if tool_uses else None,
        )

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters (model, max_tokens, temperature, etc.)

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        system = kwargs.pop("system", None)

        # Convert messages to Anthropic format
        anthropic_messages = self._prepare_messages(messages)

        # Make API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            **({"system": system} if system else {}),
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        )

        return self._build_chat_response(response)

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Streaming chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)

        # Convert messages
        anthropic_messages = self._prepare_messages(messages)

        # Stream API call
        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        ) as stream:
            for text in stream.text_stream:
                yield text

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        **kwargs
    ) -> ChatResponse:
        """Stream Anthropic text chunks and return the final structured response."""
        model = self._get_model(**kwargs)
        max_tokens = kwargs.get("max_tokens", 4096)
        system = kwargs.pop("system", None)
        anthropic_messages = self._prepare_messages(messages)

        client = self._ensure_client()
        extra_kwargs: dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools

        streamed_text = ""
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
            **({"system": system} if system else {}),
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "tools"]},
        ) as stream:
            for text in stream.text_stream:
                if not text:
                    continue
                streamed_text += text
                if on_text_chunk is not None:
                    on_text_chunk(text)
            try:
                final_message = stream.get_final_message()
            except Exception:
                final_message = None

        if final_message is not None:
            return self._build_chat_response(final_message)

        return ChatResponse(
            content=streamed_text,
            model=model,
            usage={},
            finish_reason="stop",
            tool_uses=None,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models.

        Returns:
            List of model names
        """
        return [
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
        ]
