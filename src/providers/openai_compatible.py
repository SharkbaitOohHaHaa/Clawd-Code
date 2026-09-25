"""OpenAI-compatible provider base class.

This base class consolidates shared logic for providers that use the
OpenAI-style /chat/completions API (OpenAI, GLM, Minimax, etc.).
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from typing import Any, Generator, Optional

from .base import (
    BaseProvider,
    ChatResponse,
    IncompleteResponseError,
    MessageInput,
    TextChunkCallback,
)


def _reject_non_finite(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):  # e.g. 1e999 overflows to inf
        raise ValueError("non-finite JSON number")
    return number


def _parse_tool_arguments(raw: Any) -> Any:
    """Parse serialized tool-call arguments without inventing input.

    Valid JSON is returned exactly as parsed ({} stays {}, objects are unchanged, non-object
    values keep their type, duplicate keys keep last-wins). Anything else (malformed JSON,
    empty or whitespace-only text, NaN/Infinity, non-string wire values) is returned as the
    raw text, never as {}: every tool schema expects an object, so validation rejects it
    before permission checks or run().
    """
    if not isinstance(raw, str):
        return "" if raw is None else str(raw)
    try:
        return json.loads(raw, parse_constant=_reject_non_finite, parse_float=_finite_float)
    except (ValueError, RecursionError):
        return raw


def _convert_to_openai_tool_schema(anthropic_tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Anthropic tool schema to OpenAI/GLM/Minimax function format.

    Returns None if the schema is invalid (missing type, type is None, or other issues).
    """
    input_schema = anthropic_tool.get("input_schema")
    if not input_schema or not isinstance(input_schema, dict):
        return None
    schema_type = input_schema.get("type")
    if schema_type is None or schema_type == "None":
        return None
    # Some providers (Azure) require type=object to have properties
    if schema_type == "object" and "properties" not in input_schema and "anyOf" not in input_schema and "oneOf" not in input_schema:
        # Try to add an empty properties dict if none provided
        input_schema = {**input_schema, "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool.get("description", ""),
            "parameters": input_schema,
        },
    }


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers using OpenAI-style chat completions API.

    Subclasses must implement:
    - _create_client(): Create and return the provider-specific SDK client
    - get_available_models(): Return list of available model names

    The client is created lazily on first use.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model: Default model to use
        """
        super().__init__(api_key, base_url, model)
        self._client: Optional[Any] = None

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the provider-specific SDK client.

        Built-in subclasses must construct it with ``max_retries=0`` and pass it through
        ``require_sdk_retry_policy`` (see ``sdk_policy``); they create it lazily, here.

        Returns:
            An instance of the provider's SDK client.
        """
        pass

    @property
    def client(self) -> Any:
        """Get or create the SDK client (lazy initialization)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _build_usage_dict(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = int(
            getattr(prompt_details, "cached_tokens", 0)
            or getattr(usage, "prompt_cache_hit_tokens", 0)
            or 0
        )
        reasoning_tokens = int(
            getattr(completion_details, "reasoning_tokens", 0)
            or 0
        )
        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "thought_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        model = self._get_model(**kwargs)

        # Convert messages
        provider_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            converted = [_convert_to_openai_tool_schema(t) for t in tools]
            extra_kwargs["tools"] = [t for t in converted if t is not None]

        # Make API call
        response = self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        # Extract content
        choice = response.choices[0]
        if choice.finish_reason == "length":
            # The provider stopped at its output limit: never a complete answer, and a
            # tool call from its partial arguments must never run.
            tool_calls = getattr(choice.message, "tool_calls", None)
            raise IncompleteResponseError(
                "tool_input_truncated" if tool_calls else "output_limit",
                partial_text=choice.message.content or "",
                partial_usage=self._build_usage_dict(getattr(response, "usage", None)),
                tool_call_dropped=bool(tool_calls),
            )

        # Handle reasoning content (GLM specific, but harmless for others)
        reasoning_content: Optional[str] = None
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content
        ):
            reasoning_content = choice.message.reasoning_content

        # Extract tool calls (OpenAI format -> Anthropic format)
        tool_uses: Optional[list[dict[str, Any]]] = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_uses = []
            for tc in choice.message.tool_calls:
                tool_uses.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": _parse_tool_arguments(getattr(tc.function, "arguments", None)),
                })

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=self._build_usage_dict(getattr(response, "usage", None)),
            finish_reason=choice.finish_reason,
            reasoning_content=reasoning_content,
            tool_uses=tool_uses,
        )

    def chat_stream(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Streaming chat completion.

        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas (Anthropic format)
            **kwargs: Additional parameters

        Yields:
            Chunks of response content
        """
        model = self._get_model(**kwargs)

        # Convert messages
        provider_messages = self._prepare_messages(messages)

        # Convert tools to OpenAI format
        extra_kwargs: dict[str, Any] = {}
        if tools:
            converted = [_convert_to_openai_tool_schema(t) for t in tools]
            extra_kwargs["tools"] = [t for t in converted if t is not None]

        # Stream API call
        stream = self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            stream=True,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        **kwargs
    ) -> ChatResponse:
        """Stream OpenAI-compatible chunks while rebuilding the final response."""
        model = self._get_model(**kwargs)
        provider_messages = self._prepare_messages(messages)

        extra_kwargs: dict[str, Any] = {}
        if tools:
            converted = [_convert_to_openai_tool_schema(t) for t in tools]
            extra_kwargs["tools"] = [t for t in converted if t is not None]

        stream = self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            stream=True,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["model", "tools"]},
        )

        content_parts: list[str] = []
        response_model = model
        finish_reason = "stop"
        reasoning_parts: list[str] = []
        usage_obj: Any = None
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        non_string_argument_indexes: set[int] = set()

        for chunk in stream:
            response_model = getattr(chunk, "model", response_model)
            usage_candidate = getattr(chunk, "usage", None)
            if usage_candidate is not None:
                usage_obj = usage_candidate

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content_piece = getattr(delta, "content", None)
            if content_piece:
                piece = str(content_piece)
                content_parts.append(piece)
                if on_text_chunk is not None:
                    on_text_chunk(piece)

            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                reasoning_parts.append(str(reasoning_piece))

            tool_call_deltas = getattr(delta, "tool_calls", None) or []
            for tc in tool_call_deltas:
                idx = getattr(tc, "index", 0)
                entry = tool_calls_by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})

                tc_id = getattr(tc, "id", None)
                if tc_id:
                    entry["id"] = str(tc_id)

                function = getattr(tc, "function", None)
                if function is not None:
                    fn_name = getattr(function, "name", None)
                    if fn_name:
                        entry["name"] += str(fn_name)
                    fn_args = getattr(function, "arguments", None)
                    if fn_args is not None and not isinstance(fn_args, str):
                        non_string_argument_indexes.add(idx)
                    if fn_args:
                        entry["arguments"] += str(fn_args)

        if finish_reason == "length":
            # Same rule as chat(): checked before any tool arguments are parsed.
            raise IncompleteResponseError(
                "tool_input_truncated" if tool_calls_by_index else "output_limit",
                partial_text="".join(content_parts),
                partial_usage=self._build_usage_dict(usage_obj),
                tool_call_dropped=bool(tool_calls_by_index),
            )

        tool_uses: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_by_index.keys()):
            item = tool_calls_by_index[idx]
            if not item["name"]:
                continue
            if idx in non_string_argument_indexes:
                parsed_args: Any = item["arguments"]  # non-string wire value: never parsed
            else:
                parsed_args = _parse_tool_arguments(item["arguments"])
            tool_uses.append({
                "id": item["id"] or f"tool_call_{idx}",
                "name": item["name"],
                "input": parsed_args,
            })

        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
        return ChatResponse(
            content="".join(content_parts),
            model=response_model,
            usage=self._build_usage_dict(usage_obj),
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
            tool_uses=tool_uses or None,
        )
