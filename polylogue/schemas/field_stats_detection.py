"""Detection helpers used by field-statistics collection."""

from __future__ import annotations

import math
import re

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

FORMAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("uuid4", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)),
    ("uuid", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("hex-id", re.compile(r"^[0-9a-f]{24,}$", re.I)),
    ("iso8601", re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")),
    ("unix-epoch-str", re.compile(r"^\d{10}(\.\d+)?$")),
    ("url", re.compile(r"^https?://")),
    ("mime-type", re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+\-]*$", re.I)),
    ("base64", re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
]


def is_dynamic_key(key: str) -> bool:
    if UUID_PATTERN.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))


def _detect_string_format(value: str) -> str | None:
    if not value or len(value) > 500:
        return None
    for fmt_name, pattern in FORMAT_PATTERNS:
        if pattern.match(value):
            return fmt_name
    return None


def _detect_numeric_format(value: float | int) -> str | None:
    if isinstance(value, bool):
        return None
    try:
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return None
        if 946684800.0 <= fval <= 2208988800.0:
            return "unix-epoch"
    except (TypeError, ValueError):
        pass
    return None


__all__ = [
    "_detect_numeric_format",
    "_detect_string_format",
    "FORMAT_PATTERNS",
    "UUID_PATTERN",
    "is_dynamic_key",
]
