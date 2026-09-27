"""
Compact service for /compact command.

Handles the full compaction flow:
1. Pre-process messages (microcompact, strip images)
2. Call LLM to generate summary
3. Return boundary marker + summary messages
4. Update conversation in place
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from ..agent.conversation import Conversation, Message
from ..context_system.microcompact import microcompact_messages, strip_images_from_messages
from ..providers.base import BaseProvider
from .messages import (
    create_compact_boundary_message,
    create_compact_summary_message,
    get_messages_after_boundary,
    is_compact_boundary_message,
)

# Maximum output tokens for the summary model
COMPACT_MAX_OUTPUT_TOKENS = 4096

# No-tools preamble prepended to the summary prompt
NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
Do NOT use Read, Bash, Grep, or any other tool. Your entire response must be plain text.
"""

SUMMARY_PROMPT_TEMPLATE = NO_TOOLS_PREAMBLE + """
Your task is to create a detailed summary of the conversation so far.

Include:
1. Primary Request and Intent: What the user asked for
2. Key Technical Concepts: Technologies, frameworks, patterns discussed
3. Files and Code Sections: Files examined, modified, or created (with snippets)
4. Errors and Fixes: Problems encountered and how they were resolved
5. Problem Solving: What was accomplished
6. All User Messages: All non-tool-result user messages
7. Pending Tasks: Tasks explicitly requested but not yet done
8. Current Work: What was being worked on immediately before this summary
9. Optional Next Step: Next step in line with recent requests

Respond ONLY with plain text. No tools. No XML tags.
"""

_NOT_DECLARED = object()


class NotEnoughMessagesToCompactError(ValueError):
    """Fewer than two messages after the last compact boundary; nothing was requested."""


class CompactMethodUndeterminedError(RuntimeError):
    """The provider's compaction method could not be decided statically; nothing was requested."""

    def __init__(self) -> None:
        super().__init__("Cannot safely determine the provider's compaction method.")


class CompactSummaryError(RuntimeError):
    """The chosen provider method returned no usable summary text (never retried)."""

    def __init__(self) -> None:
        super().__init__("Provider returned no usable compact summary.")


def _declares_chat_async(provider: Any) -> bool:
    """Decide, before calling the provider, whether compaction calls chat_async or chat.

    chat_async is optional and not part of BaseProvider. The lookup is static
    (inspect.getattr_static on the instance): no property, descriptor, __getattr__, instance
    __getattribute__ or provider method runs to make the choice (a plugin metaclass may still see
    getattr_static's own class lookups). Any statically visible value other than a plain None
    declares it; None is an explicit opt-out, and a chat_async reachable only dynamically
    (__getattr__, proxies) is not declared. If the lookup itself fails, compaction is refused
    before any provider call.
    """
    try:
        declared = inspect.getattr_static(provider, "chat_async", _NOT_DECLARED)
    except Exception as exc:
        raise CompactMethodUndeterminedError() from exc
    return declared is not _NOT_DECLARED and declared is not None


@dataclass
class CompactResult:
    """Result of a compaction operation."""
    boundary_message: Message
    summary_message: Message
    tokens_saved: int
    pre_compact_count: int
    post_compact_count: int
    summary_text: str
    trigger: str = "manual"
    user_display_message: Optional[str] = None
    usage: dict[str, int] | None = None


async def compact_conversation(
    conversation: Conversation,
    provider: BaseProvider,
    model: str,
    custom_instructions: Optional[str] = None,
    trigger: str = "manual",
) -> CompactResult:
    """
    Compact a conversation by summarizing older messages.

    This is the Python equivalent of the TypeScript compactConversation().
    It is NOT a 1:1 port — it omits features unavailable in a Python CLI:
    - Forked agent with cache sharing (no subprocess isolation in Python REPL)
    - Session memory (no disk-backed session memory extraction)
    - Reactive compact (no background compaction agent)
    - Hook system (no pre/post compact hooks in Clawd-Code)

    Args:
        conversation: The live Conversation object (mutated in place, only after a usable summary)
        provider: The LLM provider for generating the summary
        model: Model name for the summary call
        custom_instructions: Optional user instructions for the summarizer
        trigger: "manual" or "auto"

    Returns:
        CompactResult with boundary, summary, and metadata

    Raises:
        NotEnoughMessagesToCompactError: nothing to compact (no provider call).
        CompactMethodUndeterminedError: the provider method could not be decided (no call).
        CompactSummaryError: the provider returned no usable summary text.
        Any exception from the chosen provider method, unchanged. The conversation is then
        left as it was, and no other provider method is tried.
    """
    # Step 1: Get messages after the last boundary (skip already-summarized)
    messages = get_messages_after_boundary(conversation.messages)

    if len(messages) < 2:
        raise NotEnoughMessagesToCompactError("Not enough messages to compact.")

    # Step 2: Count pre-compact tokens
    from ..token_estimation import count_messages_tokens
    api_messages = conversation.get_messages()
    pre_compact_tokens = count_messages_tokens(api_messages)
    pre_compact_count = len(conversation.messages)

    # Step 3: Microcompact — strip images and clear old tool results
    api_messages_stripped = strip_images_from_messages(api_messages)
    compacted_api, tokens_saved = microcompact_messages(api_messages_stripped)

    # Step 4: Build summary prompt
    prompt = SUMMARY_PROMPT_TEMPLATE
    if custom_instructions:
        prompt += f"\n\nAdditional instructions: {custom_instructions}"

    # Build messages for the summary API call
    # System prompt + recent messages + summary request
    summary_request_messages: list[dict[str, Any]] = []

    # Add the last N messages as context (after microcompact)
    # Avoid sending too many old messages to stay within context limits
    context_messages = compacted_api[-20:] if len(compacted_api) > 20 else compacted_api
    for msg in context_messages:
        summary_request_messages.append(msg)

    # Add the summary request as the last user message
    summary_request_messages.append({
        "role": "user",
        "content": prompt,
    })

    # Step 5: Call the LLM to generate summary. Exactly one provider method is chosen before
    # the provider is called, and it is called once. An exception never proves that nothing
    # was sent, so no other provider method is ever tried: any failure (or an unusable reply)
    # ends the compaction and leaves the conversation unchanged.
    request = {
        "messages": summary_request_messages,
        "tools": None,  # No tools during compaction
        "model": model,
        "max_tokens": COMPACT_MAX_OUTPUT_TOKENS,
    }
    if _declares_chat_async(provider):
        response = await provider.chat_async(**request)
    else:
        response = provider.chat(**request)

    content = getattr(response, "content", None)
    summary_text = content.strip() if isinstance(content, str) else ""
    if not summary_text:
        raise CompactSummaryError()

    # Usage is secondary: a valid summary is kept even when its usage metadata is unreadable.
    response_usage: dict[str, int] = {}
    try:
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            response_usage = dict(usage)
    except Exception:
        response_usage = {}

    # Step 6: Create boundary marker
    last_msg_uuid = None
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'uuid'):
            last_msg_uuid = getattr(last_msg, 'uuid', None)
        elif isinstance(last_msg, dict):
            last_msg_uuid = last_msg.get("uuid")

    boundary_msg = create_compact_boundary_message(
        trigger=trigger,
        pre_compact_token_count=pre_compact_tokens,
        last_message_uuid=last_msg_uuid,
    )

    # Step 7: Create summary message
    summary_msg = create_compact_summary_message(summary_text)

    # Step 8: Insert boundary + summary into conversation
    # Find position after the last boundary marker
    boundary_indices = [
        i for i, m in enumerate(conversation.messages)
        if is_compact_boundary_message(m)
    ]

    if boundary_indices:
        insert_pos = max(boundary_indices) + 1
    else:
        insert_pos = 0

    # Remove messages that were summarized (everything from insert_pos onward)
    # Then insert boundary + summary
    if insert_pos == 0:
        # No existing boundary — clear all and start fresh
        conversation.messages.clear()
        conversation.messages.append(boundary_msg)
        conversation.messages.append(summary_msg)
    else:
        # Keep messages before boundary, then boundary + summary
        conversation.messages = list(conversation.messages[:insert_pos])
        conversation.messages.append(boundary_msg)
        conversation.messages.append(summary_msg)

    post_compact_count = len(conversation.messages)

    # Step 9: Build user display message
    saved_str = f"~{tokens_saved:,}" if tokens_saved > 0 else "some"
    user_display = (
        f"Compacted conversation ({saved_str} tokens saved). "
        f"Pre-compact: {pre_compact_tokens:,} tokens. "
        f"Press Ctrl+O to see full summary."
    )

    return CompactResult(
        boundary_message=boundary_msg,
        summary_message=summary_msg,
        tokens_saved=tokens_saved,
        pre_compact_count=pre_compact_count,
        post_compact_count=post_compact_count,
        summary_text=summary_text,
        trigger=trigger,
        user_display_message=user_display,
        usage=response_usage or None,
    )
