"""Detection helpers used by field-statistics collection."""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from functools import lru_cache

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_KEY_PATTERN = re.compile(r"^[0-9a-f]{24,}$", re.IGNORECASE)
_PREFIXED_ID_KEY_PATTERN = re.compile(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", re.IGNORECASE)

FORMAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("uuid4", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)),
    ("uuid", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("hex-id", re.compile(r"^[0-9a-f]{24,}$", re.I)),
    ("iso8601", re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")),
    ("unix-epoch-str", re.compile(r"^\d{10}(\.\d+)?$")),
    ("url", re.compile(r"^https?://")),
    ("mime-type", re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+\-]*$", re.I)),
    ("base64", re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
]

_MAX_STRUCTURAL_KEY_LENGTH = 128
_CONTENT_KEY_MARKERS = frozenset("?<>")
_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35
_DYNAMIC_KEY_RATIO_THRESHOLD = 0.5


@lru_cache(maxsize=4096)
def is_dynamic_key(key: str) -> bool:
    if len(key) > _MAX_STRUCTURAL_KEY_LENGTH:
        return True
    if any(ord(character) < 32 or ord(character) == 127 for character in key):
        return True
    if any(marker in key for marker in _CONTENT_KEY_MARKERS):
        return True
    if UUID_PATTERN.match(key):
        return True
    if _HEX_KEY_PATTERN.match(key):
        return True
    return bool(_PREFIXED_ID_KEY_PATTERN.match(key))


def _looks_pathlike_key(key: str) -> bool:
    if "/" in key or "\\" in key:
        return True
    if key.count(".") >= 2:
        return True
    return ":" in key and len(key) > 2


def is_content_bearing_key(key: str) -> bool:
    """True when a key is free text rather than a structural name.

    Separated from :func:`is_dynamic_key` because the consequence differs. A
    UUID or hex key is merely unhelpful as structure. A key carrying prose --
    punctuation like ``?``/``<``/``>``, control characters, or simply longer
    than any real field name -- is *content*, and emitting it into a schema
    publishes that content verbatim.
    """
    if len(key) > _MAX_STRUCTURAL_KEY_LENGTH:
        return True
    if any(ord(character) < 32 or ord(character) == 127 for character in key):
        return True
    return any(marker in key for marker in _CONTENT_KEY_MARKERS)


def should_collapse_observed_keys(keys: Collection[object]) -> bool:
    """Return whether an observed map must be modeled as a dynamic-key map.

    A single content-bearing key forces collapse, with no cardinality floor.
    The floor used to be the only rule below the high-cardinality threshold,
    and it let real conversation text reach a committed, PUBLIC schema: nine
    user questions became ``properties`` names under
    ``toolUseResult.annotations`` -- an annotations map keyed by the question
    asked -- because nine is under the 24-key floor. ``is_dynamic_key`` would
    have flagged every one of them on the ``?`` marker; this function simply
    never asked it. Structure is inferred from map SHAPE, so a map keyed by
    prose has exactly one honest schema: ``additionalProperties``.

    Identifier-ish keys (UUID, hex, prefixed ids) stay ratio-gated. They are
    not a disclosure risk, and collapsing a mostly-static map because one id
    appeared would lose real structure.
    """
    if not keys:
        return False
    if any(is_content_bearing_key(str(key)) for key in keys):
        return True
    if len(keys) >= _HIGH_CARDINALITY_KEY_THRESHOLD:
        return True
    dynamic = sum(1 for key in keys if is_dynamic_key(str(key)))
    if dynamic / len(keys) >= _DYNAMIC_KEY_RATIO_THRESHOLD:
        return True
    if len(keys) < 24:
        return False
    pathlike = sum(1 for key in keys if _looks_pathlike_key(str(key)))
    return (pathlike / len(keys)) >= _PATHLIKE_KEY_RATIO_THRESHOLD


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
    "should_collapse_observed_keys",
]
