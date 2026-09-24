"""
Built-in commands for Clawd Code.

Implements core commands like /help, /clear, /exit, /skills, etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from ..activity_ledger import activity_counts
from ..context_system.builder import build_context_prompt
from ..context_system.context_analyzer import (
    analyze_context,
    format_context_as_markdown,
    get_context_window_for_model,
)
from ..context_system.microcompact import microcompact_messages, strip_images_from_messages
from ..cost_tracker import CostTracker
from ..history import HistoryLog
from ..observability import runtime_observability_snapshot
from ..osv_evidence import osv_contract_status
from ..providers.base import BaseProvider
from ..providers.sdk_policy import provider_sdk_policy_status
from ..usage_ledger import month_to_date_provider_usage
from .engine import CommandContext, CommandResult, LocalCommandResult
from .registry import CommandRegistry, get_command_registry, list_commands
from .types import Command, CommandType, CompactionResult, LocalCommand, PromptCommand


# Official Claude Code /init prompts (Simplified)
NEW_INIT_PROMPT = """Set up a CLAUDE.md file for this repo. CLAUDE.md is loaded into every Claude Code session, so it must be concise — only include what Claude would get wrong without it.

## Step 1: Ask what to set up

Use AskUserQuestion to ask the user:
- "Which CLAUDE.md files should /init set up?" with options: "Project CLAUDE.md" | "Personal CLAUDE.local.md" | "Both project + personal"

Use AskUserQuestion to ask:
- "Also set up project skills?" with options: "Skills" | "No skills"

## Step 2: Explore the codebase

Use tools to understand the project:
- Read key files: README, package.json, pyproject.toml, Cargo.toml, Makefile, existing CLAUDE.md
- Detect: build/test/lint commands, languages, frameworks, project structure
- Detect: code style rules, required env vars, gotchas
- Check for formatter config (ruff, black, prettier, etc.)

## Step 3: Ask follow-up questions (if needed)

Use AskUserQuestion to ask only things you CAN'T figure out from code:
- User's role (e.g., "backend engineer", "new hire")
- Non-obvious workflows or commands
- Communication preferences (terse vs detailed)

## Step 4: Write CLAUDE.md

Write a minimal CLAUDE.md at the project root.

Include:
- Build/test/lint commands that aren't standard (e.g., "uv run pytest" not just "pytest")
- Code style rules that DIFFER from defaults
- Required env vars or setup steps
- Non-obvious gotchas

Exclude:
- File structure (Claude can discover this)
- Standard conventions Claude already knows
- Generic advice

Prefix with:
```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

If CLAUDE.md exists: read it, propose specific improvements.

## Step 5: Write CLAUDE.local.md (if user chose personal or both)

Write CLAUDE.local.md at project root. Add it to .gitignore.

Include:
- User's role and familiarity with codebase
- Personal sandbox URLs, test accounts
- Communication preferences

## Step 6: Create skills (if user chose skills)

Create skills at `.claude/skills/<name>/SKILL.md`:
```yaml
---
name: <skill-name>
description: <what it does>
---

<Instructions>
```

## Step 7: Summary

Tell the user what was set up and suggest any additional optimizations."""

# Fallback prompt for simpler initialization
OLD_INIT_PROMPT = """Please analyze this codebase and create a CLAUDE.md file, which will be given to future instances of Claude Code to operate in this repository.

What to add:
1. Commands that will be commonly used, such as how to build, lint, and run tests. Include the necessary commands to develop in this codebase, such as how to run a single test.
2. High-level code architecture and structure so that future instances can be productive more quickly. Focus on the "big picture" architecture that requires reading multiple files to understand.

Usage notes:
- If there's already a CLAUDE.md, suggest improvements to it.
- When you make the initial CLAUDE.md, do not repeat yourself and do not include obvious instructions like "Provide helpful error messages to users", "Write unit tests for all new utilities", "Never include sensitive information (API keys, tokens) in code or commits".
- Avoid listing every component or file structure that can be easily discovered.
- Don't include generic development practices.
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include the important parts.
- If there is a README.md, make sure to include the important parts.
- Do not make up information such as "Common Development Tasks", "Tips for Development", "Support and Documentation" unless this is expressly included in other files that you read.
- Be sure to prefix the file with the following text:

```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```"""


def clear_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /clear command - clear conversation history.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    if hasattr(context.conversation, "clear"):
        context.conversation.clear()

    if hasattr(context.history, "events"):
        context.history.events.clear()

    return LocalCommandResult(
        type="text",
        value="Conversation cleared.",
    )


def help_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /help command - show available commands.

    Args:
        args: Command arguments (optional search query)
        context: Command context

    Returns:
        LocalCommandResult
    """
    registry = get_command_registry()
    query = args.strip()

    if query:
        commands = registry.find_commands(query, limit=50)
        header = f"Commands matching '{query}':"
    else:
        commands = registry.list_commands(include_hidden=False)
        header = "Available commands:"

    lines = [header, ""]

    for cmd in commands:
        alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name}{alias_str}")
        lines.append(f"      {cmd.description}")
        if cmd.argument_hint:
            lines.append(f"      Usage: /{cmd.name} {cmd.argument_hint}")
        lines.append("")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def skills_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /skills command - list available skills.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    try:
        from ..skills.loader import get_all_skills
        # Pass project_root to find skills in project directories
        skills = get_all_skills(project_root=context.cwd or context.workspace_root)
    except Exception:
        skills = []

    if not skills:
        return LocalCommandResult(
            type="text",
            value="No skills available. Add skills to ~/.clawd/skills/ or ./.clawd/skills/.",
        )

    lines = ["Available skills:", ""]
    for skill in skills:
        lines.append(f"  {skill.name}")
        lines.append(f"      {skill.description}")
        if skill.when_to_use:
            lines.append(f"      When to use: {skill.when_to_use}")
        lines.append("")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def exit_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /exit command - exit the application.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    return LocalCommandResult(
        type="text",
        value="Goodbye!",
    )


def cost_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /session-usage command - show JR-tracked session token usage.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    tracker = context.cost_tracker
    if tracker is None:
        return LocalCommandResult(
            type="text",
            value="Session token tracking not available.",
        )

    lines = ["JR Session Token Usage:", ""]
    provider_usage = getattr(tracker, "provider_usage", {}) or {}

    if provider_usage:
        for label, usage in provider_usage.items():
            lines.append(f"  {label}:")
            lines.append(f"    Input tokens:    {int(usage.get('input_tokens', 0) or 0):,}")
            lines.append(f"    Output tokens:   {int(usage.get('output_tokens', 0) or 0):,}")
            thought = int(usage.get("thought_tokens", 0) or 0)
            tool_use = int(usage.get("tool_use_tokens", 0) or 0)
            cached = int(usage.get("cached_tokens", 0) or 0)
            if thought:
                lines.append(f"    Thought tokens:  {thought:,}")
            if tool_use:
                lines.append(f"    Tool-use tokens: {tool_use:,}")
            if cached:
                lines.append(f"    Cached tokens:   {cached:,}")
            lines.append(f"    Total tokens:    {int(usage.get('total_tokens', 0) or 0):,}")
            lines.append("")

    input_tokens = int(getattr(tracker, "input_tokens", 0) or 0)
    output_tokens = int(getattr(tracker, "output_tokens", 0) or 0)
    lines.append("  Combined tracked:")
    lines.append(f"    Input tokens:  {input_tokens:,}")
    lines.append(f"    Output tokens: {output_tokens:,}")
    lines.append(f"    Total tokens:  {tracker.total_units:,}")

    if tracker.events:
        lines.append("")
        lines.append("  Recent events:")
        for event in tracker.events[-10:]:
            lines.append(f"    - {event}")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def _anthropic_admin_get(path: str, params: list[tuple[str, str]], admin_key: str) -> dict[str, Any]:
    """Fetch a read-only Anthropic organization usage endpoint without exposing credentials."""
    query = urllib.parse.urlencode(params)
    url = f"https://api.anthropic.com{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "anthropic-version": "2023-06-01",
            "x-api-key": admin_key,
            "User-Agent": "Clawd/0.1 usage-report",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError("Anthropic Admin API rejected the usage credential") from exc
        raise RuntimeError(f"Anthropic usage API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic usage API could not be reached: {exc.reason}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Anthropic usage API returned an unexpected response")
    return payload


def _anthropic_month_to_date_usage(admin_key: str) -> tuple[dict[str, dict[str, int]], Decimal]:
    """Return actual Anthropic organization token usage and spend for the current UTC month."""
    now = datetime.now(timezone.utc)
    starting_at = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    common = [("starting_at", starting_at), ("bucket_width", "1d"), ("limit", "31")]

    usage_payload = _anthropic_admin_get(
        "/v1/organizations/usage_report/messages",
        common + [("group_by[]", "model")],
        admin_key,
    )
    by_model: dict[str, dict[str, int]] = {}
    for bucket in usage_payload.get("data", []):
        if not isinstance(bucket, dict):
            continue
        for result in bucket.get("results", []):
            if not isinstance(result, dict):
                continue
            model = str(result.get("model") or "unknown-model")
            totals = by_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            uncached = int(result.get("uncached_input_tokens", result.get("input_tokens", 0)) or 0)
            cache_read = int(result.get("cache_read_input_tokens", 0) or 0)
            cache_write = int(result.get("cache_creation_input_tokens", 0) or 0)
            if not cache_write:
                cache_creation = result.get("cache_creation")
                if isinstance(cache_creation, dict):
                    cache_write = sum(
                        int(value or 0)
                        for key, value in cache_creation.items()
                        if key.endswith("_input_tokens")
                    )
            totals["input_tokens"] += uncached + cache_read + cache_write
            totals["output_tokens"] += int(result.get("output_tokens", 0) or 0)

    cost_payload = _anthropic_admin_get(
        "/v1/organizations/cost_report",
        common,
        admin_key,
    )
    cents = Decimal("0")
    for bucket in cost_payload.get("data", []):
        if not isinstance(bucket, dict):
            continue
        for result in bucket.get("results", []):
            if not isinstance(result, dict):
                continue
            try:
                cents += Decimal(str(result.get("amount", "0") or "0"))
            except InvalidOperation:
                continue

    return by_model, cents / Decimal("100")


def _google_monitoring_get(path: str, params: list[tuple[str, str]], credentials: Any) -> dict[str, Any]:
    """Fetch a read-only Cloud Monitoring endpoint with service-account OAuth."""
    from google.auth.transport.requests import Request

    if not credentials.valid:
        credentials.refresh(Request())
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"https://monitoring.googleapis.com/v3/{path}?{query}",
        headers={"Authorization": f"Bearer {credentials.token}", "User-Agent": "Clawd/0.1 usage-report"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError("Google Cloud Monitoring rejected the service-account credential") from exc
        raise RuntimeError(f"Google Cloud Monitoring returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Google Cloud Monitoring could not be reached: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Google Cloud Monitoring returned an unexpected response")
    return payload


def _google_user_credentials() -> Any:
    """Load existing Google monitoring credentials without starting OAuth enrollment."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    scopes = ["https://www.googleapis.com/auth/monitoring.read"]
    token_file = Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", "")).expanduser()
    if not token_file.is_file():
        raise RuntimeError("Google OAuth token is not configured; /usage will not start OAuth enrollment")

    credentials = Credentials.from_authorized_user_file(str(token_file), scopes=scopes)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_file.write_text(credentials.to_json(), encoding="utf-8")
    if not credentials.valid:
        raise RuntimeError("Google OAuth credentials are not valid; authorize them outside /usage")
    return credentials


def _google_month_to_date_gemini_usage(project_id: str) -> tuple[str, dict[str, dict[str, int]]]:
    """Return month-to-date Gemini API input/output token usage from Cloud Monitoring."""
    credentials = _google_user_credentials()
    project_id = project_id.strip()
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured")

    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    metric_kinds = {
        "output_tokens": ["generativelanguage.googleapis.com/generate_content_usage_output_token_count"],
        "input_tokens": [
            "generativelanguage.googleapis.com/quota/generate_content_free_tier_input_token_count/usage",
            "generativelanguage.googleapis.com/quota/generate_content_paid_tier_input_token_count/usage",
            "generativelanguage.googleapis.com/quota/generate_content_paid_tier_2_input_token_count/usage",
            "generativelanguage.googleapis.com/quota/generate_content_paid_tier_3_input_token_count/usage",
        ],
    }
    by_model: dict[str, dict[str, int]] = {}
    for kind, metric_types in metric_kinds.items():
        for metric_type in metric_types:
            params = [
                ("filter", f'metric.type="{metric_type}"'),
                ("interval.startTime", start.isoformat().replace("+00:00", "Z")),
                ("interval.endTime", now.isoformat().replace("+00:00", "Z")),
                ("view", "FULL"),
            ]
            page_token = ""
            while True:
                page_params = params + ([("pageToken", page_token)] if page_token else [])
                payload = _google_monitoring_get(f"projects/{project_id}/timeSeries", page_params, credentials)
                for series in payload.get("timeSeries", []):
                    if not isinstance(series, dict):
                        continue
                    labels = (series.get("metric") or {}).get("labels") or {}
                    model = str(labels.get("model") or "unknown-model")
                    total = 0
                    for point in series.get("points", []):
                        value = (point.get("value") or {}).get("int64Value", 0)
                        try:
                            total += int(value or 0)
                        except (TypeError, ValueError):
                            continue
                    model_usage = by_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
                    model_usage[kind] += total
                page_token = str(payload.get("nextPageToken") or "")
                if not page_token:
                    break
    return project_id, by_model


def _append_activity_sections(lines: list[str]) -> None:
    """Append real persisted skill/tool invocation counts to the Usage screen."""
    counts = activity_counts()
    lines.extend(["", "Skill Activity"])
    skills = counts["skill"]
    if skills:
        for name, count in sorted(skills.items()):
            lines.append(f"  {name}: {count:,} use{'s' if count != 1 else ''}")
        lines.append(f"  Total Skill Uses: {sum(skills.values()):,}")
    else:
        lines.append("  No skill activity recorded yet.")
        lines.append("  Total Skill Uses: 0")

    lines.extend(["", "Tool Activity"])
    tools = counts["tool"]
    if tools:
        for name, count in sorted(tools.items()):
            lines.append(f"  {name}: {count:,} use{'s' if count != 1 else ''}")
        lines.append(f"  Total Tool Uses: {sum(tools.values()):,}")
    else:
        lines.append("  No tool activity recorded yet.")
        lines.append("  Total Tool Uses: 0")

    lines.extend(["", "Skill Activity and Tool Activity show how Clawd has been working.", "They are not separate billing or API-usage categories."])


def _append_tracked_provider_usage(
    lines: list[str],
    *,
    heading: str,
    label_prefix: str,
    api_key_env: str,
) -> None:
    """Append locally persisted provider-reported month-to-date usage."""
    lines.extend(["", f"  {heading}:"])
    usage_by_label = month_to_date_provider_usage(label_prefix)
    if usage_by_label:
        lines.append("    JR exact provider-reported usage captured month-to-date:")
        for label, usage in sorted(usage_by_label.items()):
            lines.append(f"    {label}:")
            lines.append(f"      Input tokens:     {int(usage.get('input_tokens', 0) or 0):,}")
            lines.append(f"      Output tokens:    {int(usage.get('output_tokens', 0) or 0):,}")
            lines.append(f"      Cached input:     {int(usage.get('cached_tokens', 0) or 0):,}")
            lines.append(f"      Reasoning tokens: {int(usage.get('thought_tokens', 0) or 0):,}")
            lines.append(f"      Total tokens:     {int(usage.get('total_tokens', 0) or 0):,}")
    elif os.environ.get(api_key_env, "").strip():
        lines.append(f"    API key configured; JR has not captured a {heading} request yet.")
    else:
        lines.append(f"    {api_key_env} is not configured.")
    lines.append(
        f"    {heading} totals come from exact provider-reported request usage captured by JR."
    )


def _authorize_usage_provider(context: CommandContext, provider: str) -> bool:
    """Require one-shot user authorization before /usage performs provider network I/O."""
    if context.permission_handler is None:
        return False
    allowed, _ = context.permission_handler(
        "UsageProviderRetrieval",
        f"Allow /usage to retrieve read-only {provider} provider usage over the network?",
        None,
    )
    return bool(allowed)


def usage_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """Handle /usage as an overall usage screen with local data and authorized provider retrieval."""
    lines = ["Usage", "", "API / Model Usage", ""]

    admin_key = os.environ.get("ANTHROPIC_ADMIN_KEY", "").strip().strip('"').strip("'")
    lines.append("  Claude:")
    if admin_key and not _authorize_usage_provider(context, "Anthropic"):
        lines.append("    Provider retrieval not authorized; local usage remains available.")
    elif admin_key:
        try:
            by_model, usd = _anthropic_month_to_date_usage(admin_key)
            if by_model:
                for model, usage in sorted(by_model.items()):
                    input_tokens = int(usage.get("input_tokens", 0) or 0)
                    output_tokens = int(usage.get("output_tokens", 0) or 0)
                    lines.append(f"    {model}:")
                    lines.append(f"      Input tokens:  {input_tokens:,}")
                    lines.append(f"      Output tokens: {output_tokens:,}")
                    lines.append(f"      Total tokens:  {input_tokens + output_tokens:,}")
            else:
                lines.append("    No month-to-date API token usage reported.")
            lines.append(f"    Month-to-date spend: ${usd:,.2f} USD")
        except Exception as exc:
            lines.append(f"    Real usage unavailable: {exc}")
    else:
        lines.append("    Real organization usage unavailable: ANTHROPIC_ADMIN_KEY is not configured.")
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            lines.append("    The normal ANTHROPIC_API_KEY can send requests but cannot read organization usage.")

    _append_tracked_provider_usage(
        lines,
        heading="DeepSeek",
        label_prefix="DeepSeek (",
        api_key_env="DEEPSEEK_API_KEY",
    )
    _append_tracked_provider_usage(
        lines,
        heading="Qwen",
        label_prefix="Qwen (",
        api_key_env="DASHSCOPE_API_KEY",
    )

    lines.append("")
    lines.append("  Gemini:")
    oauth_client = os.environ.get("GOOGLE_OAUTH_CLIENT_FILE", "").strip()
    if not oauth_client:
        lines.append("    Real provider usage unavailable: Google OAuth is not configured.")
        if os.environ.get("GEMINI_API_KEY", "").strip():
            lines.append("    GEMINI_API_KEY alone can send requests but cannot read provider usage.")
        _append_activity_sections(lines)
        return LocalCommandResult(type="text", value="\n".join(lines))
    if not _authorize_usage_provider(context, "Google Cloud Monitoring"):
        lines.append("    Provider retrieval not authorized; local usage remains available.")
        _append_activity_sections(lines)
        return LocalCommandResult(type="text", value="\n".join(lines))
    try:
        project_id, by_model = _google_month_to_date_gemini_usage(
            os.environ.get("GOOGLE_CLOUD_PROJECT", "ldr-gamers")
        )
        lines.append(f"    Google Cloud project: {project_id}")
        if by_model:
            for model, usage in sorted(by_model.items()):
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                lines.append(f"    {model}:")
                lines.append(f"      Input tokens:  {input_tokens:,}")
                lines.append(f"      Output tokens: {output_tokens:,}")
                lines.append(f"      Total tokens:  {input_tokens + output_tokens:,}")
        else:
            lines.append("    Cloud Monitoring returned no month-to-date Gemini token series; this does not mean zero usage.")
            gemini_usage = month_to_date_provider_usage("Gemini (")
            if gemini_usage:
                lines.append("    JR exact provider-reported usage captured month-to-date:")
                for label, usage in sorted(gemini_usage.items()):
                    lines.append(f"      {label}:")
                    lines.append(f"        Input tokens:  {int(usage.get('input_tokens', 0) or 0):,}")
                    lines.append(f"        Output tokens: {int(usage.get('output_tokens', 0) or 0):,}")
                    lines.append(f"        Thought tokens: {int(usage.get('thought_tokens', 0) or 0):,}")
                    lines.append(f"        Tool-use tokens: {int(usage.get('tool_use_tokens', 0) or 0):,}")
                    lines.append(f"        Total tokens:  {int(usage.get('total_tokens', 0) or 0):,}")
            else:
                lines.append("    JR has not captured a Gemini interaction since persistent usage tracking was enabled.")
    except Exception as exc:
        lines.append(f"    Real usage unavailable: {exc}")

    _append_activity_sections(lines)
    return LocalCommandResult(type="text", value="\n".join(lines))


def context_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /context command - show token usage breakdown.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult with Markdown table of context usage
    """
    try:
        # Prefer the exact model-visible first-request preflight used by the REPL.
        # This is local-only assembly: it does not call the provider.
        preflight = None
        context_provider = context.config.get("context_provider")
        context_tool_registry = context.config.get("context_tool_registry")
        context_tool_context = context.config.get("context_tool_context")
        if (
            context_provider is not None
            and context_tool_registry is not None
            and context_tool_context is not None
            and context.conversation is not None
        ):
            try:
                from ..tool_system.agent_loop import build_agent_preflight
                preflight = build_agent_preflight(
                    context.conversation,
                    context_provider,
                    context_tool_registry,
                    context_tool_context,
                )
            except Exception:
                preflight = None

        if preflight is not None:
            conversation_api = preflight.api_messages
            system_prompt = preflight.effective_system_prompt
            tool_schemas = preflight.tool_schemas
            model = str(getattr(context_provider, "model", "") or "claude-sonnet-4-6")
            # CLAUDE.md, Project Overview, Git context, output style, and persistent
            # memory are already contained in the effective system prompt.
            claude_md_content = ""
            mcp_tools: list[dict[str, Any]] = []
            custom_agents: list[dict[str, Any]] = []
            skills_frontmatter_tokens = 0
            skills_count = 0
        else:
            # Legacy/fallback path for direct command-system callers.
            conversation_api: list[dict[str, Any]] = []
            if hasattr(context.conversation, "get_messages"):
                conversation_api = context.conversation.get_messages()
            elif hasattr(context.conversation, "messages"):
                for msg in context.conversation.messages:
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                    conversation_api.append({"role": role, "content": content})

            system_prompt = context.config.get("system_prompt", "")
            tool_schemas = context.config.get("tool_schemas", [])
            mcp_tools = context.config.get("mcp_tools", [])
            custom_agents = context.config.get("custom_agents", [])

            claude_md_content = ""
            try:
                from ..context_system.claude_md import load_claude_md_context
                claude_md = load_claude_md_context(context.workspace_root, cwd=context.cwd)
                if claude_md.files:
                    claude_md_content = "\n".join(
                        getattr(item, "content", "")
                        for item in claude_md.files
                        if getattr(item, "content", "")
                    )
            except Exception:
                pass

            model = context.config.get("model", "claude-sonnet-4-6")
            skills_frontmatter_tokens = context.config.get("skills_tokens", 0)
            skills_count = context.config.get("skills_count", 0)

        # Get API usage from cost tracker
        api_usage = None
        if hasattr(context.cost_tracker, "last_usage"):
            api_usage = context.cost_tracker.last_usage

        # Get auto-compact info from config
        auto_compact_threshold = context.config.get("auto_compact_threshold")
        is_auto_compact_enabled = context.config.get("is_auto_compact_enabled", False)

        data = analyze_context(
            conversation_api_messages=conversation_api,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            claude_md_content=claude_md_content,
            skills_frontmatter_tokens=skills_frontmatter_tokens,
            skills_count=skills_count,
            api_usage=api_usage,
            mcp_tools=mcp_tools,
            custom_agents=custom_agents,
            auto_compact_threshold=auto_compact_threshold,
            is_auto_compact_enabled=is_auto_compact_enabled,
        )

        markdown = format_context_as_markdown(data)
        return LocalCommandResult(type="text", value=markdown)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return LocalCommandResult(type="text", value=f"Context analysis failed: {e}")


def _doctor_issue_hint(code: str) -> str:
    hints = {
        "git_worktree_runtime_unavailable": "Check that Git is installed and available on PATH.",
        "lsp_runtime_unavailable": "Check HOME/USERPROFILE and the pinned Pyright runtime under ~/.clawd/lsp/pyright.",
        "mcp_resource_runtime_unavailable": "Check HOME/USERPROFILE and the pinned MCP runtime under ~/.clawd/mcp/runtime.",
        "mcp_resource_manifest_invalid": "Review ~/.clawd/mcp_servers.json. Doctor does not connect to or repair MCP servers.",
        "expected_tool_not_registered": "Runtime registry is missing a manifest-declared tool; do not auto-repair the registry.",
        "unexpected_tool_registered": "Runtime registry contains an undeclared tool; review capability manifest parity.",
        "registered_tool_missing_from_manifest": "A registered tool is missing from the capability manifest.",
        "manifest_permission_policy_missing": "Capability manifest is missing an explicit permission policy.",
        "permission_policy_mismatch": "Runtime permission policy disagrees with the capability manifest.",
        "read_only_metadata_conflicts_with_mutation": "Tool read-only metadata conflicts with declared state mutation.",
        "trust_registry_missing": "Skill trust registry is missing; do not activate skills until trust state is restored.",
        "trust_registry_unreadable": "Skill trust registry could not be read.",
        "trust_registry_invalid": "Skill trust registry schema is invalid.",
        "active_skill_not_loader_reachable": "An active skill is not reachable from approved loader paths.",
        "active_skill_integrity_mismatch": "An active skill no longer matches its approved integrity hash.",
        "runtime_copy_differs_from_trust_artifact": "A runtime skill copy differs from its trusted artifact.",
        "duplicate_skill_artifacts": "Multiple loader-reachable copies make skill resolution ambiguous.",
        "skill_record_invalid": "A skill trust record is malformed.",
        "plugin_operator_manifest_invalid": "Review ~/.clawd/python_plugins.json; plugin activation stays fail-closed.",
        "plugin_operator_entry_invalid": "Fix the named operator plugin entry before activation.",
        "plugin_root_symlink": "Use a real ~/.clawd/plugins directory; plugin-root symlinks are rejected.",
        "plugin_root_invalid": "Use a real ~/.clawd/plugins directory.",
        "plugin_manifest_invalid": "Fix plugin.json metadata/path validation before activation.",
        "plugin_operator_hash_missing": "Pin the exact reviewed plugin artifact SHA-256 before enabling it.",
        "plugin_integrity_mismatch": "The plugin changed after review; re-review and pin the new exact hash.",
        "enabled_plugin_missing": "Disable the stale operator entry or restore the exact reviewed plugin directory.",
        "plugin_extension_load_failed": "The trusted plugin could not load its declared extensions; review the exact pinned plugin code and exports.",
        "plugin_extension_registration_failed": "The trusted plugin extension conflicts with runtime command/tool contracts or names; fix and re-review it.",
        "plugin_provider_registration_failed": "The trusted provider extension conflicts with provider names or provider safety/config contracts; fix and re-review it.",
        "observability_ledger_unreadable": "Inspect the local runtime ledger path/permissions; doctor does not modify or recreate the ledger.",
        "observability_ledger_malformed": "Review or archive the malformed local JSONL ledger before relying on its diagnostics.",
    }
    return hints.get(code, "Review the reported capability/trust state; doctor does not modify it.")


def doctor_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """Run local, read-only capability and trust diagnostics."""
    try:
        from ..capabilities import reconcile_capabilities

        workspace = Path(context.cwd or context.workspace_root).expanduser().resolve()
        report = reconcile_capabilities(project_root=workspace)
    except Exception as exc:
        return LocalCommandResult(
            type="text",
            value=(
                "# Clawd Doctor\n\n"
                "**Status:** ERROR\n\n"
                f"Local diagnostics could not complete: {exc}\n\n"
                "No repairs, network calls, or provider calls were attempted."
            ),
        )

    tool_report = report.get("tools") if isinstance(report.get("tools"), dict) else {}
    skill_report = report.get("skills") if isinstance(report.get("skills"), dict) else {}
    plugin_report = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    tool_issues = list(tool_report.get("issues") or [])
    skill_issues = list(skill_report.get("issues") or [])
    plugin_issues = list(plugin_report.get("issues") or [])
    observability = runtime_observability_snapshot(recent_limit=5)
    observability_issues = list(observability.get("issues") or [])
    runtime_plugin_issues = context.config.get("plugin_runtime_issues")
    if isinstance(runtime_plugin_issues, list):
        plugin_issues.extend(
            issue for issue in runtime_plugin_issues if isinstance(issue, dict)
        )
    audit = skill_report.get("audit_chain") if isinstance(skill_report.get("audit_chain"), dict) else {}
    audit_valid = bool(audit.get("valid"))
    workspace_ok = workspace.exists() and workspace.is_dir()

    home = Path.home().expanduser().resolve()
    env_home = os.environ.get("HOME", "").strip()
    env_userprofile = os.environ.get("USERPROFILE", "").strip()
    home_mismatch = False
    if env_home and env_userprofile:
        try:
            home_mismatch = (
                Path(env_home).expanduser().resolve()
                != Path(env_userprofile).expanduser().resolve()
            )
        except OSError:
            home_mismatch = True

    has_issues = (
        not workspace_ok
        or home_mismatch
        or bool(tool_issues)
        or bool(skill_issues)
        or bool(plugin_issues)
        or bool(observability_issues)
        or not audit_valid
    )
    status = "ISSUES FOUND" if has_issues else "PASS"

    registered = list(tool_report.get("registered") or [])
    expected = list(tool_report.get("expected_registered") or [])
    records = skill_report.get("records") if isinstance(skill_report.get("records"), dict) else {}
    active_skills = sum(
        1
        for record in records.values()
        if isinstance(record, dict)
        and record.get("review_status") == "approved"
        and record.get("activation_status") == "active"
    )
    active_plugins = list(plugin_report.get("active") or [])
    recent_runtime_errors = list(observability.get("recent_errors") or [])
    deferred = list(report.get("deferred_features") or [])

    lines = [
        "# Clawd Doctor",
        "",
        f"**Status:** {status}",
        f"**Workspace:** {'PASS' if workspace_ok else 'ISSUE'} — {workspace}",
        f"**Runtime home:** {home}",
        (
            f"**Environment:** {'ISSUE' if home_mismatch else 'PASS'}"
            + (
                " — HOME and USERPROFILE resolve to different locations"
                if home_mismatch
                else " — HOME/USERPROFILE alignment accepted"
            )
        ),
        (
            f"**Tools/runtime:** {'PASS' if not tool_issues else 'ISSUE'}"
            f" — {len(registered)}/{len(expected)} expected tools registered"
        ),
        (
            f"**Skills/trust:** {'PASS' if not skill_issues and audit_valid else 'ISSUE'}"
            f" — {active_skills} active trusted skill(s); "
            f"audit chain {'valid' if audit_valid else 'invalid/unavailable'}"
        ),
        (
            f"**Python plugins:** {'PASS' if not plugin_issues else 'ISSUE'}"
            f" — {len(active_plugins)} active exact-hash operator-trusted plugin(s)"
        ),
        (
            f"**Observability:** {'PASS' if not observability_issues else 'ISSUE'}"
            f" — {int(observability.get('tool_calls', 0))} tool call(s), "
            f"{int(observability.get('tool_errors', 0))} tool error(s), "
            f"{int(observability.get('provider_events', 0))} provider event(s), "
            f"{int(observability.get('provider_failures', 0))} provider failure(s), "
            f"{int(observability.get('change_events', 0))} change event(s)"
        ),
        f"**Deferred capabilities:** {', '.join(deferred) if deferred else 'none'}",
        f"**Software evidence (OSV):** {osv_contract_status()}",
        f"**Provider SDK retries:** {provider_sdk_policy_status()}",
        "**Network/provider checks:** not run; doctor is local and read-only",
    ]

    issues: list[tuple[str, str, str]] = []
    if not workspace_ok:
        issues.append(("workspace", "workspace_unavailable", str(workspace)))
    if home_mismatch:
        issues.append(
            (
                "environment",
                "runtime_home_mismatch",
                f"HOME={env_home!r}, USERPROFILE={env_userprofile!r}",
            )
        )
    for issue in tool_issues:
        if isinstance(issue, dict):
            issues.append(
                (
                    "tool",
                    str(issue.get("code") or "unknown_tool_issue"),
                    str(issue.get("subject") or ""),
                )
            )
    for issue in skill_issues:
        if isinstance(issue, dict):
            issues.append(
                (
                    "skill",
                    str(issue.get("code") or "unknown_skill_issue"),
                    str(issue.get("subject") or ""),
                )
            )
    for issue in plugin_issues:
        if isinstance(issue, dict):
            issues.append(
                (
                    "plugin",
                    str(issue.get("code") or "unknown_plugin_issue"),
                    str(issue.get("subject") or ""),
                )
            )
    for issue in observability_issues:
        if isinstance(issue, dict):
            issues.append(
                (
                    "observability",
                    str(issue.get("code") or "unknown_observability_issue"),
                    str(issue.get("subject") or ""),
                )
            )
    if not audit_valid:
        issues.append(
            (
                "skill",
                "skill_audit_chain_invalid",
                str(audit.get("error") or "skill audit chain is unavailable"),
            )
        )

    if issues:
        lines.extend(["", "## Issues"])
        for area, code, subject in issues:
            if code == "workspace_unavailable":
                hint = "Open Clawd from an existing workspace directory."
            elif code == "runtime_home_mismatch":
                hint = "Launch Clawd with one consistent HOME/USERPROFILE runtime home."
            elif code == "skill_audit_chain_invalid":
                hint = "Treat durable skill trust as degraded until the audit chain is restored."
            else:
                hint = _doctor_issue_hint(code)
            lines.append(f"- **{area}/{code}:** {subject} — {hint}")

    if recent_runtime_errors:
        lines.extend(["", "## Recent Runtime Errors"])
        for event in recent_runtime_errors:
            timestamp = str(event.get("timestamp") or "unknown-time")
            area = str(event.get("area") or "runtime")
            if area in {"tool", "skill"}:
                name = str(event.get("name") or "unknown")
                lines.append(f"- {timestamp} — {area} {name} reported an error")
            elif area == "provider":
                provider = str(event.get("provider") or "unknown-provider")
                operation = str(event.get("operation") or "unknown-operation")
                stage = str(event.get("stage") or "unknown-stage")
                error_type = str(event.get("error_type") or "provider failure")
                status_code = event.get("status_code")
                code_text = f" (HTTP {status_code})" if status_code is not None else ""
                lines.append(
                    f"- {timestamp} — provider {provider} {operation}/{stage}: "
                    f"{error_type}{code_text}"
                )

    return LocalCommandResult(type="text", value="\n".join(lines))


async def _compact_async(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Async implementation of compact command.
    """
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(
            type="text",
            value="No conversation to compact.",
        )

    messages = context.conversation.messages
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get provider from config
    provider = context.config.get("provider")
    if provider is None:
        return LocalCommandResult(
            type="text",
            value="Compact requires an LLM provider (not available in this context).",
        )

    model = context.config.get("model", "claude-sonnet-4-6")
    custom_instructions = args.strip() or None

    try:
        # Import here to avoid circular imports
        from ..compact_service.service import compact_conversation

        result = await compact_conversation(
            conversation=context.conversation,
            provider=provider,
            model=model,
            custom_instructions=custom_instructions,
            trigger="manual",
        )

        display_message = result.user_display_message or "Conversation compacted."
        usage = result.usage if isinstance(result.usage, dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens
        if total_tokens > 0 and context.cost_tracker is not None:
            provider_class = type(provider).__name__.lower()
            if "anthropic" in provider_class:
                provider_label = "Claude"
            elif "deepseek" in provider_class:
                provider_label = "DeepSeek"
            elif "qwen" in provider_class:
                provider_label = "Qwen"
            elif "openai" in provider_class:
                provider_label = "OpenAI"
            elif "glm" in provider_class:
                provider_label = "GLM"
            elif "minimax" in provider_class:
                provider_label = "MiniMax"
            else:
                provider_label = type(provider).__name__

            usage_label = f"{provider_label} ({model})" if model else provider_label
            context.cost_tracker.record_usage(
                usage_label,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            display_message += (
                f"\nUsage this task:\n"
                f"  {usage_label}: {input_tokens:,} input, "
                f"{output_tokens:,} output = {total_tokens:,} total tokens"
            )

        return LocalCommandResult(
            type="compact",
            value=display_message,
            compaction_result=CompactionResult(
                pre_compact_count=result.pre_compact_count,
                post_compact_count=result.post_compact_count,
                tokens_saved=result.tokens_saved,
                trigger=result.trigger,
                summary_preview=result.summary_text[:200] if len(result.summary_text) > 200 else result.summary_text,
            ),
        )
    except ValueError as e:
        return LocalCommandResult(type="text", value=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return LocalCommandResult(
            type="text",
            value=f"Compact failed: {e}",
        )


def compact_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /compact command - compact conversation context.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    # Run the async version in a new event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, we can't use asyncio.run
        # Fall back to sync path
        return _sync_compact_fallback(context)
    except RuntimeError:
        # No running event loop — safe to use asyncio.run
        try:
            return asyncio.run(_compact_async(args, context))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return _sync_compact_fallback(context)


def _sync_compact_fallback(context: CommandContext) -> LocalCommandResult:
    """Synchronous fallback when async provider is not available."""
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(type="text", value="No conversation to compact.")

    messages = context.conversation.messages
    original_messages = list(messages)
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get messages after last boundary
    try:
        from ..compact_service.messages import (
            create_compact_boundary_message,
            create_compact_summary_message,
            get_messages_after_boundary,
            is_compact_boundary_message,
        )
        from ..token_estimation import count_messages_tokens

        after_boundary = get_messages_after_boundary(messages)
        if len(after_boundary) < 2:
            return LocalCommandResult(
                type="text",
                value=f"Nothing to compact: only {len(after_boundary)} messages after boundary.",
            )

        # Count tokens
        api_messages = context.conversation.get_messages()
        pre_tokens = count_messages_tokens(api_messages)

        # Strip images and microcompact
        stripped = strip_images_from_messages(api_messages)
        compacted, saved = microcompact_messages(stripped)

        # Find boundary position
        boundary_indices = [
            i for i, m in enumerate(messages)
            if is_compact_boundary_message(m)
        ]

        if boundary_indices:
            insert_pos = max(boundary_indices) + 1
        else:
            insert_pos = 0

        # Create simple text summary
        summary_parts = [f"Conversation had {len(after_boundary)} messages ({pre_tokens:,} tokens)."]
        summary_text = "\n".join(summary_parts)

        boundary = create_compact_boundary_message(
            trigger="manual",
            pre_compact_token_count=pre_tokens,
        )
        summary = create_compact_summary_message(summary_text)

        # Rebuild conversation
        if insert_pos == 0:
            context.conversation.messages.clear()
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)
        else:
            context.conversation.messages = list(messages[:insert_pos])
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)

        return LocalCommandResult(
            type="compact",
            value=f"Compacted: removed {len(after_boundary) - 2} messages ({pre_tokens:,} tokens → ~{saved} saved).",
            compaction_result=CompactionResult(
                pre_compact_count=len(messages),
                post_compact_count=len(context.conversation.messages),
                tokens_saved=saved,
                trigger="manual",
                summary_preview=summary_text[:200],
            ),
        )
    except Exception as e:
        # Compaction failures must not destroy or truncate conversation history.
        context.conversation.messages = original_messages
        return LocalCommandResult(
            type="text",
            value=f"Compact failed safely; conversation preserved: {e}",
        )


# Command definitions
HELP_COMMAND = LocalCommand(
    name="help",
    description="Show available commands",
    aliases=["?"],
    argument_hint="[search_query]",
    supports_non_interactive=True,
)

CLEAR_COMMAND = LocalCommand(
    name="clear-chat",
    description="Clear the current conversation history",
    aliases=["clear", "reset", "new"],
    supports_non_interactive=False,
)

EXIT_COMMAND = LocalCommand(
    name="exit",
    description="Exit the application",
    aliases=["quit", "q"],
    supports_non_interactive=True,
)

SKILLS_COMMAND = LocalCommand(
    name="list-skills",
    description="List available skills",
    aliases=["skills"],
    argument_hint="",
    supports_non_interactive=True,
)

COST_COMMAND = LocalCommand(
    name="session-usage",
    description="Show token usage tracked in the current JR session",
    aliases=["cost"],
    argument_hint="",
    supports_non_interactive=True,
)

USAGE_COMMAND = LocalCommand(
    name="usage",
    description="Show API/model usage plus local skill and tool activity",
    aliases=[],
    argument_hint="",
    supports_non_interactive=True,
)

CONTEXT_COMMAND = LocalCommand(
    name="context-usage",
    description="Show current context-window usage and breakdown",
    aliases=["context"],
    argument_hint="",
    supports_non_interactive=True,
)

DOCTOR_COMMAND = LocalCommand(
    name="doctor",
    description="Run local read-only health and capability diagnostics",
    aliases=[],
    argument_hint="",
    supports_non_interactive=True,
)

COMPACT_COMMAND = LocalCommand(
    name="compact-context",
    description="Compact the conversation to save context space",
    aliases=["compact"],
    argument_hint="",
    supports_non_interactive=True,
)

INIT_COMMAND = PromptCommand(
    name="setup-project",
    description="Set up CLAUDE.md and optional project skills",
    aliases=["init"],
    markdown_content=NEW_INIT_PROMPT,
    progress_message="analyzing your codebase",
    content_length=0,
    source="builtin",
)


# Synchronous versions for REPL integration
def execute_command_sync(cmd_name: str, args: str, context: CommandContext) -> tuple[bool, str | None, str | None]:
    """
    Execute a command synchronously.

    Returns:
        Tuple of (success: bool, result_text: str | None, error: str | None)
    """
    cmd = None
    for builtin_cmd in get_builtin_commands():
        if builtin_cmd.name.lower() == cmd_name.lower() or cmd_name.lower() in [a.lower() for a in builtin_cmd.aliases]:
            cmd = builtin_cmd
            break

    if cmd is None:
        return False, None, f"Unknown command: {cmd_name}"

    try:
        # This is a synchronous wrapper - we directly call the underlying function
        # instead of going through the async call() method
        if cmd is HELP_COMMAND:
            result = help_command_call(args, context)
        elif cmd is CLEAR_COMMAND:
            result = clear_command_call(args, context)
        elif cmd is EXIT_COMMAND:
            result = exit_command_call(args, context)
        elif cmd is SKILLS_COMMAND:
            result = skills_command_call(args, context)
        elif cmd is COST_COMMAND:
            result = cost_command_call(args, context)
        elif cmd is USAGE_COMMAND:
            result = usage_command_call(args, context)
        elif cmd is CONTEXT_COMMAND:
            result = context_command_call(args, context)
        elif cmd is DOCTOR_COMMAND:
            result = doctor_command_call(args, context)
        elif cmd is COMPACT_COMMAND:
            result = compact_command_call(args, context)
        else:
            return False, None, f"Command not implemented for sync execution: {cmd_name}"

        return True, result.value, None
    except Exception as e:
        return False, None, str(e)


# Set the call implementations
HELP_COMMAND.set_call(help_command_call)
CLEAR_COMMAND.set_call(clear_command_call)
EXIT_COMMAND.set_call(exit_command_call)
SKILLS_COMMAND.set_call(skills_command_call)
COST_COMMAND.set_call(cost_command_call)
USAGE_COMMAND.set_call(usage_command_call)
CONTEXT_COMMAND.set_call(context_command_call)
DOCTOR_COMMAND.set_call(doctor_command_call)
COMPACT_COMMAND.set_call(compact_command_call)


def get_builtin_commands() -> list[Command]:
    """Get all built-in commands."""
    return [
        HELP_COMMAND,
        CLEAR_COMMAND,
        EXIT_COMMAND,
        SKILLS_COMMAND,
        COST_COMMAND,
        USAGE_COMMAND,
        CONTEXT_COMMAND,
        DOCTOR_COMMAND,
        COMPACT_COMMAND,
        INIT_COMMAND,
    ]


def register_builtin_commands(registry: CommandRegistry | None = None) -> None:
    """
    Register all built-in commands.

    Args:
        registry: Optional registry to use (uses global if None)
    """
    reg = registry or get_command_registry()
    for cmd in get_builtin_commands():
        reg.register(cmd)


async def execute_command_async(
    cmd_name: str,
    args: str,
    context: CommandContext,
) -> CommandResult:
    """
    Execute a command asynchronously.

    This function handles both LocalCommand and PromptCommand types.
    For PromptCommand, it returns the prompt content that should be sent to the LLM.

    Args:
        cmd_name: Name of the command to execute
        args: Arguments for the command
        context: Command context

    Returns:
        CommandResult with the execution result
    """
    from .engine import CommandEngine

    registry = get_command_registry()
    cmd = registry.get(cmd_name)

    if cmd is None:
        return CommandResult.error(cmd_name, f"Unknown command: {cmd_name}")

    if not cmd.is_enabled():
        return CommandResult.error(cmd_name, f"Command {cmd_name} is disabled")

    engine = CommandEngine(
        registry=registry,
        workspace_root=context.workspace_root,
        context=context,
    )

    # Create a fake command input string for the engine
    command_input = f"/{cmd_name}"
    if args:
        command_input += f" {args}"

    return await engine.execute(command_input)
