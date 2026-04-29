"""FieldStats accumulation and format detection for schema inference."""

from __future__ import annotations

from polylogue.schemas.field_stats_collection import _collect_field_stats
from polylogue.schemas.field_stats_detection import (
    UUID_PATTERN,
    _detect_numeric_format,
    _detect_string_format,
    is_dynamic_key,
)
from polylogue.schemas.field_stats_models import (
    ENUM_MAX_CARDINALITY,
    ENUM_VALUE_CAP,
    REF_MATCH_THRESHOLD,
    FieldStats,
)

__all__ = [
    "ENUM_MAX_CARDINALITY",
    "ENUM_VALUE_CAP",
    "FieldStats",
    "REF_MATCH_THRESHOLD",
    "_collect_field_stats",
    "_detect_numeric_format",
    "_detect_string_format",
    "UUID_PATTERN",
    "is_dynamic_key",
]
