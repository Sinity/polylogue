"""Typed coercion helpers for model payload decoding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import cast


def mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (cast(Mapping[str, object], value),)
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Iterable):
        return tuple(cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping))
    return ()


def string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def optional_date(value: object) -> date | None:
    if value is None:
        return None
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
    if value is None or isinstance(value, Mapping):
        return default
    if isinstance(value, (str, bytes, bytearray)):
        return (coerce_int(value, default[0]), default[1])
    if not isinstance(value, Iterable):
        return default

    values = list(value)
    if not values:
        return default
    first = coerce_int(values[0], default[0])
    second = coerce_int(values[1], default[1]) if len(values) > 1 else default[1]
    return (first, second)


__all__ = [
    "coerce_float",
    "coerce_int",
    "int_pair",
    "mapping_or_empty",
    "mapping_sequence",
    "optional_date",
    "optional_datetime",
    "optional_string",
    "string_int_mapping",
    "string_sequence",
]
