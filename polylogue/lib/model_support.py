"""Shared coercion helpers for domain-model metadata."""

from __future__ import annotations


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


_CONTEXT_PATTERNS = [
    r"^Contents of .+:",
    r"^<file path=",
]


__all__ = [
    "_CONTEXT_PATTERNS",
    "_coerce_optional_float",
    "_coerce_optional_int",
]
