"""Base provider abstract class for LLM providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generator, NamedTuple, Optional, TypeAlias


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Represents a chat response."""

    content: str
    model: str
    usage: dict[str, Any]
    finish_reason: str
    reasoning_content: Optional[str] = None
    tool_uses: Optional[list[dict[str, Any]]] = None


class IncompleteResponseError(RuntimeError):
    """The provider reported that generation stopped at its output limit.

    Raised by built-in providers for Anthropic ``stop_reason == "max_tokens"`` and
    OpenAI-compatible ``finish_reason == "length"``, streamed or not. It carries the
    partial text so callers can keep it as partial; it never carries tool calls, so a
    truncated tool call cannot run. Not a NotImplementedError: nothing falls back or
    retries. The message is fixed text (no provider text, no digits) so it can never
    look like an authentication failure.
    """

    _MESSAGES = {
        "output_limit": "The provider stopped at its output limit; the response is incomplete.",
        "tool_input_truncated": (
            "The provider stopped at its output limit while writing a tool call; "
            "the response is incomplete and the tool call was not run."
        ),
    }

    def __init__(
        self,
        reason: str,
        *,
        partial_text: str = "",
        partial_usage: Optional[dict[str, Any]] = None,
        tool_call_dropped: bool = False,
    ) -> None:
        self.reason = reason
        self.partial_text = partial_text
        self.partial_usage = dict(partial_usage or {})
        self.tool_call_dropped = tool_call_dropped
        super().__init__(self._MESSAGES[reason])


class InvalidToolInputError(RuntimeError):
    """A completed Anthropic-family response carried a tool call whose input is not a JSON object.

    Raised instead of coercing it (e.g. a list of pairs into an object), so no tool runs on
    input the model did not provide. Carries no tool input and has a fixed message (no
    provider text, no digits); it is not a NotImplementedError, ValueError, TypeError or SDK
    error, so nothing falls back or retries.
    """

    _MESSAGE = (
        "The provider returned a tool call whose input was not a JSON object; "
        "the response was rejected and no tool was run."
    )

    def __init__(self) -> None:
        super().__init__(self._MESSAGE)


@dataclass(frozen=True)
class _FinishProfile:
    normal: frozenset[str]
    output_limit: str  # handled earlier by the provider (Phase A); never classified here
    abnormal: dict[str, str]  # documented value -> category
    context_reset: frozenset[str] = frozenset()  # documented: reset the conversation afterwards


# The one source of truth for built-in finish/stop values, from each provider's own
# documentation. Matching is exact and case-sensitive. Values not listed here are
# "unrecognized": they never fail a text-only response, but never grant tool authority.
_FINISH_PROFILES: dict[str, _FinishProfile] = {
    "anthropic": _FinishProfile(
        normal=frozenset({"end_turn", "stop_sequence", "tool_use"}),
        output_limit="max_tokens",
        abnormal={
            "refusal": "blocked",
            "pause_turn": "paused",
            "model_context_window_exceeded": "context_window",
        },
        context_reset=frozenset({"refusal"}),
    ),
    "minimax": _FinishProfile(
        normal=frozenset({"end_turn", "tool_use"}), output_limit="max_tokens", abnormal={}
    ),
    "openai": _FinishProfile(
        normal=frozenset({"stop", "tool_calls", "function_call"}),
        output_limit="length",
        abnormal={"content_filter": "blocked"},
    ),
    "deepseek": _FinishProfile(
        normal=frozenset({"stop", "tool_calls"}),
        output_limit="length",
        abnormal={
            "content_filter": "blocked",
            "insufficient_system_resource": "interrupted",
            "aborted": "interrupted",
        },
    ),
    "qwen": _FinishProfile(
        normal=frozenset({"stop", "tool_calls"}), output_limit="length", abnormal={}
    ),
    "glm": _FinishProfile(
        normal=frozenset({"stop", "tool_calls"}),
        output_limit="length",
        abnormal={
            "sensitive": "blocked",
            "network_error": "interrupted",
            "model_context_window_exceeded": "context_window",
        },
    ),
}

_SAFE_FINISH_VALUE = re.compile(r"[A-Za-z0-9_.:-]{1,40}")


class FinishStatus(NamedTuple):
    category: str
    finish_value: Any
    context_reset: bool


def classify_finish_status(
    profile: str, values: list[Any], *, has_tool_calls: bool
) -> Optional[FinishStatus]:
    """Say whether a response must not be returned as a success, from its finish values.

    ``values`` are the finish values the provider reported, in order: one for a response
    that is not streamed, every chunk value other than None or "" for an OpenAI-compatible
    stream (the last one is the final value). A documented abnormal value anywhere wins. Otherwise only
    the final value counts: a documented normal value, the output-limit value (checked
    earlier by the provider) and None or "" (no signal) pass; anything else is
    "unrecognized" and fails only a response that carries tool calls.
    """
    spec = _FINISH_PROFILES[profile]
    for value in values:
        if isinstance(value, str) and value in spec.abnormal:
            return FinishStatus(spec.abnormal[value], value, value in spec.context_reset)
    final = values[-1] if values else None
    if final is None or final == "":
        return None
    if isinstance(final, str) and (final in spec.normal or final == spec.output_limit):
        return None
    if has_tool_calls:
        return FinishStatus("unrecognized", final, False)
    return None


class FinishStatusError(RuntimeError):
    """The provider's finish status says the response did not complete normally.

    Raised by built-in providers for a documented abnormal finish value, and for an
    unrecognized one on a response that carries tool calls; always after the output-limit
    (IncompleteResponseError) and tool-input (InvalidToolInputError) checks. It never
    carries tool calls, so none can run. Not a NotImplementedError, SDK error, ValueError,
    TypeError, AttributeError, KeyError or IndexError: nothing falls back or retries. The
    message is fixed text (no provider text, no digits); the provider's value is only in
    attributes (``safe_finish_value`` is display-safe).
    """

    _MESSAGES = {
        "blocked": (
            "The provider blocked this response with a safety or content-filter status; "
            "the response was not used."
        ),
        "interrupted": (
            "The provider reported that generation was interrupted; the response is "
            "incomplete and was not used."
        ),
        "context_window": (
            "The provider stopped at the model context window; the response is incomplete "
            "and was not used."
        ),
        "paused": "The provider paused this turn; the response is incomplete and was not used.",
        "unrecognized": (
            "The provider ended a tool-call response with an unrecognized finish status; "
            "the response was not used."
        ),
    }

    def __init__(
        self,
        category: str,
        *,
        finish_value: Any,
        partial_text: str = "",
        partial_usage: Optional[dict[str, Any]] = None,
        tool_call_dropped: bool = False,
        context_reset: bool = False,
    ) -> None:
        self.category = category
        self.finish_value = finish_value
        self.safe_finish_value = (
            finish_value
            if isinstance(finish_value, str) and _SAFE_FINISH_VALUE.fullmatch(finish_value)
            else "unprintable value"
        )
        self.partial_text = partial_text
        self.partial_usage = dict(partial_usage or {})
        self.tool_call_dropped = tool_call_dropped
        self.context_reset = context_reset
        super().__init__(self._MESSAGES[category])

    @classmethod
    def from_status(
        cls,
        status: FinishStatus,
        *,
        partial_text: str,
        partial_usage: Optional[dict[str, Any]],
        tool_call_dropped: bool,
    ) -> "FinishStatusError":
        return cls(
            status.category,
            finish_value=status.finish_value,
            partial_text=partial_text,
            partial_usage=partial_usage,
            tool_call_dropped=tool_call_dropped,
            context_reset=status.context_reset,
        )


MessageInput: TypeAlias = ChatMessage | dict[str, Any]
TextChunkCallback: TypeAlias = Callable[[str], None]


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        """Initialize provider.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model: Default model to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            Chat response
        """
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of response content
        """
        pass

    def chat_stream_response(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: TextChunkCallback | None = None,
        **kwargs
    ) -> ChatResponse:
        """Stream a response while also returning the final structured ChatResponse.

        Providers may override this to support tool-aware streaming. The default
        implementation signals that rich streamed responses are unavailable.
        """
        raise NotImplementedError("Structured streaming is not supported by this provider")

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models.

        Returns:
            List of model names
        """
        pass

    def _get_model(self, **kwargs) -> str:
        """Get model from kwargs or use default.

        Args:
            **kwargs: Keyword arguments that may contain 'model'

        Returns:
            Model name to use
        """
        return kwargs.get("model", self.model)

    def _prepare_messages(self, messages: list[MessageInput]) -> list[dict[str, Any]]:
        """Convert provider messages to API dictionary format."""
        return [msg if isinstance(msg, dict) else msg.to_dict() for msg in messages]
