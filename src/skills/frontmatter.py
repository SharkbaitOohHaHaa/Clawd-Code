from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


_COMMA_LIST_KEYS = {"allowed-tools", "arguments", "paths"}


@dataclass(frozen=True)
class FrontmatterParseResult:
    frontmatter: Dict[str, Any]
    body: str


def parse_frontmatter(markdown: str) -> FrontmatterParseResult:
    """
    Minimal frontmatter parser supporting a safe subset of YAML-like syntax:
    - Top-level key: value pairs
    - Booleans: true/false (case-insensitive)
    - Numbers: integers
    - Lists via inline bracket form:
        key: [item1, item2]
    - Lists via hyphen form:
        key:
          - item1
          - item2
    - Lists via comma-separated shorthand for known list fields:
        allowed-tools: Read, Grep
    - Quoted scalar strings (commas remain part of the string)
    - Folded/literal scalar blocks using >, >-, |, |-
    Any unsupported structure falls back to a string.
    """
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return FrontmatterParseResult(frontmatter={}, body=markdown)
    # Find closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return FrontmatterParseResult(frontmatter={}, body=markdown)

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    fm: Dict[str, Any] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if not line.strip():
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, value = _split_key_value(line)
        # List (hyphen form)
        if value == "" and i + 1 < len(fm_lines) and fm_lines[i + 1].lstrip().startswith("- "):
            items: List[str] = []
            i += 1
            while i < len(fm_lines):
                item_line = fm_lines[i]
                if item_line.lstrip().startswith("- "):
                    items.append(item_line.lstrip()[2:].strip())
                    i += 1
                else:
                    break
            fm[key] = [_coerce_scalar(x) for x in items]
            continue

        # Folded/literal YAML-style scalar block.
        if value in {">", ">-", "|", "|-"}:
            block_lines: List[str] = []
            i += 1
            while i < len(fm_lines):
                item_line = fm_lines[i]
                if item_line and not item_line[0].isspace():
                    break
                if item_line:
                    block_lines.append(item_line.lstrip())
                else:
                    block_lines.append("")
                i += 1
            if value.startswith(">"):
                paragraphs: List[str] = []
                current: List[str] = []
                for block_line in block_lines:
                    if block_line == "":
                        if current:
                            paragraphs.append(" ".join(current))
                            current = []
                    else:
                        current.append(block_line)
                if current:
                    paragraphs.append(" ".join(current))
                fm[key] = "\n\n".join(paragraphs)
            else:
                fm[key] = "\n".join(block_lines)
            continue

        # Quoted scalar values must remain scalars even when they contain commas.
        if _is_quoted_scalar(value):
            fm[key] = _coerce_scalar(value)
            i += 1
            continue

        inline_list = _parse_inline_list(value)
        if inline_list is not None:
            fm[key] = inline_list
            i += 1
            continue
        # Comma-separated shorthand is only safe for fields that are known lists.
        # Descriptions and other free-text scalars commonly contain commas.
        if key in _COMMA_LIST_KEYS and "," in value:
            fm[key] = [_coerce_scalar(v.strip()) for v in value.split(",") if v.strip()]
        else:
            fm[key] = _coerce_scalar(value.strip())
        i += 1
    return FrontmatterParseResult(frontmatter=fm, body=body)


def _split_key_value(line: str) -> Tuple[str, str]:
    idx = line.find(":")
    key = line[:idx].strip()
    value = line[idx + 1 :].strip()
    return key, value


def _is_quoted_scalar(value: str) -> bool:
    stripped = value.strip()
    return (
        len(stripped) >= 2
        and stripped[0] in {'"', "'"}
        and stripped[-1] == stripped[0]
    )


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if _is_quoted_scalar(value):
        quote = value[0]
        inner = value[1:-1]
        if quote == '"':
            inner = inner.replace(r'\"', '"').replace(r"\\", "\\")
        else:
            inner = inner.replace("''", "'")
        return inner

    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if value.isdigit():
        try:
            return int(value)
        except Exception:
            pass
    return value


def _parse_inline_list(value: str) -> List[Any] | None:
    stripped = value.strip()
    if len(stripped) < 2 or not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    return [_coerce_scalar(part.strip()) for part in inner.split(",") if part.strip()]
