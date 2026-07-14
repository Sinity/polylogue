"""Typed coercion helpers for model payload decoding."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, date, datetime
from typing import TypeAlias, TypeGuard

PayloadMapping: TypeAlias = Mapping[str, object]


def is_payload_mapping(value: object) -> TypeGuard[PayloadMapping]:
    """Return whether a value is a mapping whose keys are all strings."""
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _iter_items(value: object) -> Iterator[object]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return iter(())
    if isinstance(value, Iterable):
        return iter(value)
    return iter((value,))


def mapping_or_empty(value: object) -> PayloadMapping:
    if is_payload_mapping(value):
        return value
    return {}


def mapping_sequence(value: object) -> tuple[PayloadMapping, ...]:
    if is_payload_mapping(value):
        return (value,)
    return tuple(item for item in _iter_items(value) if is_payload_mapping(item))


def string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if is_payload_mapping(value):
        return ()
    return tuple(str(item) for item in _iter_items(value))


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def optional_datetime(value: object) -> datetime | None:
    """Coerce a value to a UTC-aware datetime or None.

    ISO strings without timezone offsets are treated as UTC.
    Already-aware datetimes are returned as-is (converted to UTC).
    Naive datetimes are assumed to be UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def coerce_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        return int(value)
    return int(str(value))


def coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    return float(str(value))


def string_int_mapping(value: object) -> dict[str, int]:
    return {str(key): coerce_int(item, 0) for key, item in mapping_or_empty(value).items()}


def int_pair(value: object, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    if value is None or is_payload_mapping(value):
        return default
    if isinstance(value, (str, bytes, bytearray)):
        return (coerce_int(value, default[0]), default[1])

    values = list(_iter_items(value))
    if not values:
        return default
    first = coerce_int(values[0], default[0])
    second = coerce_int(values[1], default[1]) if len(values) > 1 else default[1]
    return (first, second)


# Row-value coercion helpers (for DataFrame/query result payloads)
def required_str(value: object) -> str:
    """Coerce a value to a string; raise if None."""
    if value is None:
        raise ValueError("Required string value is None")
    return value if isinstance(value, str) else str(value)


def optional_str(value: object) -> str | None:
    """Coerce a value to a string or None; strict on type."""
    return value if isinstance(value, str) else None


def row_int(value: object) -> int:
    """Coerce a value to int; default to 0 for unparseable or bool."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def row_float(value: object) -> float | None:
    """Coerce a value to float or None; strict on bool (returns None)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


__all__ = [
    "PayloadMapping",
    "coerce_float",
    "coerce_int",
    "int_pair",
    "is_payload_mapping",
    "mapping_or_empty",
    "mapping_sequence",
    "optional_date",
    "optional_datetime",
    "optional_str",
    "optional_string",
    "required_str",
    "row_float",
    "row_int",
    "string_int_mapping",
    "string_sequence",
]
