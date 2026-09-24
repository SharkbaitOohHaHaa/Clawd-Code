from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    total_units: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[str] = field(default_factory=list)
    provider_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, label: str, units: int) -> None:
        """Backward-compatible generic usage recorder."""
        self.total_units += units
        self.events.append(f"{label}:{units}")

    def record_usage(
        self,
        label: str,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int | None = None,
        thought_tokens: int = 0,
        tool_use_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
        thought_tokens = max(0, int(thought_tokens or 0))
        tool_use_tokens = max(0, int(tool_use_tokens or 0))
        cached_tokens = max(0, int(cached_tokens or 0))
        tracked_total = (
            max(0, int(total_tokens or 0))
            if total_tokens is not None
            else input_tokens + output_tokens
        )

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_units += tracked_total

        bucket = self.provider_usage.setdefault(
            label,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "thought_tokens": 0,
                "tool_use_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            },
        )
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["thought_tokens"] += thought_tokens
        bucket["tool_use_tokens"] += tool_use_tokens
        bucket["cached_tokens"] += cached_tokens
        bucket["total_tokens"] += tracked_total

        detail = (
            f"{label}: input={input_tokens},output={output_tokens},"
            f"total={tracked_total}"
        )
        if thought_tokens:
            detail += f",thought={thought_tokens}"
        if tool_use_tokens:
            detail += f",tool_use={tool_use_tokens}"
        if cached_tokens:
            detail += f",cached={cached_tokens}"
        self.events.append(detail)
