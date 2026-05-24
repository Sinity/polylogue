"""Shared daemon process-start timestamp for HTTP surfaces."""

from __future__ import annotations

import time

_STARTED_AT_MONOTONIC: float = time.monotonic()
_STARTED_AT_WALL: float = time.time()


def started_at_wall() -> float:
    """Return the daemon process start wall-clock timestamp."""
    return _STARTED_AT_WALL


def uptime_seconds(*, now_monotonic: float | None = None) -> float:
    """Return process uptime seconds from the shared daemon start anchor."""
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    return max(0.0, now_monotonic - _STARTED_AT_MONOTONIC)


__all__ = ["started_at_wall", "uptime_seconds"]
