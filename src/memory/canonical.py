from __future__ import annotations

import json
from typing import Any


CANONICALIZATION_VERSION = "rfc8785-jcs-ijson-v1"
SAFE_INTEGER_MIN = -(2**53 - 1)
SAFE_INTEGER_MAX = 2**53 - 1


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by the memory canonical form."""


class DuplicateKeyError(CanonicalizationError):
    """Raised when JSON contains duplicate object member names."""


def _reject_constant(value: str) -> None:
    raise CanonicalizationError(f"non-I-JSON numeric constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    """Parse strict JSON and reject duplicate keys and NaN/Infinity."""
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalizationError(f"invalid JSON: {exc.msg}") from exc
    _validate_value(value)
    return value


def _validate_string(value: str) -> None:
    for ch in value:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise CanonicalizationError("lone UTF-16 surrogate code points are not allowed")


def _validate_value(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if value < SAFE_INTEGER_MIN or value > SAFE_INTEGER_MAX:
            raise CanonicalizationError("integer is outside the I-JSON safe integer range")
        return
    if isinstance(value, float):
        raise CanonicalizationError(
            "floating-point values are not permitted in memory canonicalization v1"
        )
    if isinstance(value, str):
        _validate_string(value)
        return
    if isinstance(value, list):
        for item in value:
            _validate_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("JSON object keys must be strings")
            _validate_string(key)
            _validate_value(item)
        return
    raise CanonicalizationError(f"unsupported JSON value type: {type(value).__name__}")


def _utf16_sort_key(value: str) -> bytes:
    _validate_string(value)
    return value.encode("utf-16-be")


def _encode_string(value: str) -> str:
    _validate_string(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    """
    Serialize the supported I-JSON subset using RFC 8785 / JCS ordering.

    Memory canonicalization v1 deliberately rejects floating-point values. All
    numeric schema fields are safe integers, which removes cross-language
    number-format ambiguity while remaining JCS-compatible for accepted values.
    """
    _validate_value(value)

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        pieces: list[str] = []
        for key in sorted(value, key=_utf16_sort_key):
            pieces.append(_encode_string(key) + ":" + canonical_json(value[key]))
        return "{" + ",".join(pieces) + "}"
    raise CanonicalizationError("unreachable canonicalization type")


TEST_VECTORS: tuple[tuple[Any, str], ...] = (
    ({"b": 1, "a": 2}, '{"a":2,"b":1}'),
    ({"�": 1, "😀": 2}, '{"😀":2,"�":1}'),
    ({"s": "\b\t\n\f\r\"\\\u0001"}, '{"s":"\\b\\t\\n\\f\\r\\\"\\\\\\u0001"}'),
    (
        {"nested": [True, None, {"z": SAFE_INTEGER_MAX, "a": SAFE_INTEGER_MIN}]},
        '{"nested":[true,null,{"a":-9007199254740991,"z":9007199254740991}]}',
    ),
)


def verify_test_vectors() -> None:
    for index, (value, expected) in enumerate(TEST_VECTORS, start=1):
        actual = canonical_json(value)
        if actual != expected:
            raise CanonicalizationError(
                f"canonicalization test vector {index} failed: {actual!r} != {expected!r}"
            )
