"""Shared payload contracts for authored scenario and publication surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, TypeAlias

PayloadDict: TypeAlias = dict[str, object]
PayloadMap: TypeAlias = Mapping[str, object]


class PayloadSerializable(Protocol):
    """Typed record that can expose a stable JSON-compatible payload."""

    def to_payload(self) -> PayloadDict:
        """Return a JSON-compatible payload mapping."""


def payload_items(value: object) -> tuple[object, ...]:
    """Return list/tuple items as a stable tuple."""
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def payload_string(value: object, default: str = "") -> str:
    """Coerce a payload field to string, preserving the caller's default."""
    return default if value is None else str(value)


def payload_optional_string(value: object) -> str | None:
    """Return a non-empty payload string when present."""
    if isinstance(value, str) and value:
        return value
    return None


def payload_string_tuple(value: object) -> tuple[str, ...]:
    """Return payload string sequences as an immutable tuple."""
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def payload_int(value: object, key: str) -> int | None:
    """Coerce integer-like payload fields with an actionable error message."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"{key} must be an int-compatible value, got {type(value).__name__}")


def payload_count(value: object, key: str) -> int:
    """Coerce a required count field to an integer."""
    count = payload_int(value, key)
    if count is None:
        raise TypeError(f"{key} must be present")
    return count


def payload_float(value: object, key: str) -> float | None:
    """Coerce float-like payload fields with an actionable error message."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{key} must be a float-compatible value, got {type(value).__name__}")


def payload_bool(value: object, default: bool = False) -> bool:
    """Coerce common payload booleans without treating arbitrary strings as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise TypeError(f"Expected a boolean-compatible payload value, got {type(value).__name__}")


def payload_mapping(value: object) -> PayloadMap | None:
    """Return mapping payloads whose keys are all strings."""
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return {str(key): item for key, item in value.items()}


def require_payload_mapping(value: object, *, context: str = "payload") -> PayloadMap:
    """Return a string-keyed payload mapping or raise an actionable error."""
    payload = payload_mapping(value)
    if payload is None:
        raise TypeError(f"Expected string-keyed mapping for {context}")
    return payload


def payload_count_mapping(value: object, *, context: str = "payload") -> dict[str, int]:
    """Return a mapping of string keys to integer counts."""
    return {
        key: payload_count(item, f"{context}.{key}")
        for key, item in require_payload_mapping(value, context=context).items()
    }


def payload_records(records: tuple[PayloadSerializable, ...]) -> list[PayloadDict]:
    """Serialize typed payload records as a stable JSON-compatible list."""
    return [record.to_payload() for record in records]


def canonical_payload_json(payload: PayloadMap) -> str:
    """Serialize a payload mapping to stable canonical JSON."""
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def merge_unique_string_tuples(*groups: tuple[str, ...], skip_empty: bool = False) -> tuple[str, ...]:
    """Merge string tuples while preserving order and removing duplicates."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if (skip_empty and not item) or item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


__all__ = [
    "PayloadDict",
    "PayloadMap",
    "PayloadSerializable",
    "canonical_payload_json",
    "merge_unique_string_tuples",
    "payload_bool",
    "payload_count",
    "payload_count_mapping",
    "payload_float",
    "payload_int",
    "payload_items",
    "payload_mapping",
    "payload_optional_string",
    "payload_records",
    "payload_string",
    "payload_string_tuple",
    "require_payload_mapping",
]
