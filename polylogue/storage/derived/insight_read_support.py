"""Typed hydration helpers for durable insight reads."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TypeVar

_RecordT = TypeVar("_RecordT")
_HydratedT = TypeVar("_HydratedT")


def hydrate_optional(
    record: _RecordT | None,
    hydrate: Callable[[_RecordT], _HydratedT],
) -> _HydratedT | None:
    if record is None:
        return None
    return hydrate(record)


def hydrate_sequence(
    records: Iterable[_RecordT],
    hydrate: Callable[[_RecordT], _HydratedT],
) -> list[_HydratedT]:
    return [hydrate(record) for record in records]


def hydrate_mapping(
    records: Mapping[str, _RecordT],
    hydrate: Callable[[_RecordT], _HydratedT],
) -> dict[str, _HydratedT]:
    return {record_id: hydrate(record) for record_id, record in records.items()}


__all__ = ["hydrate_mapping", "hydrate_optional", "hydrate_sequence"]
