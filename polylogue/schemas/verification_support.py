"""Shared support helpers for schema verification workflows."""

from __future__ import annotations


def bounded_window(
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int]:
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    return bounded_limit, bounded_offset


__all__ = ["bounded_window"]
