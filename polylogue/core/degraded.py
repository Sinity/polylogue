"""Process-local degraded-mode flag.

When a daemon process detects a structural condition that makes ingestion
impossible for the lifetime of the process (most notably a schema-version
mismatch between the binary and the on-disk database), it sets the flag here.
Both the daemon surface and the source-ingest substrate read this flag
cheaply before doing any work.

The module lives in ``core/`` so substrate code (``polylogue.sources``) can
read and write it without violating the layering rule that forbids
``sources/`` from importing ``daemon/``. There is no in-place schema upgrade the daemon
itself can apply, so re-validation only happens on SIGHUP or an explicit
operator action; a process restart picks up a new value naturally.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze_detail(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Wrap a detail mapping in a read-only view so callers can't mutate shared state."""
    if value is None:
        return None
    if isinstance(value, MappingProxyType):
        return value
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class DegradedReason:
    """Why the daemon refuses to ingest until restart or recheck.

    ``detail`` is normalized to a read-only mapping on construction so that
    holders of the original dict — or the value returned by ``degraded_reason()``
    — cannot mutate process-wide shared state outside the lock.
    """

    code: str
    message: str
    detail: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks normal assignment; bypass via object.__setattr__.
        object.__setattr__(self, "detail", _freeze_detail(self.detail))


_lock = threading.Lock()
_state: DegradedReason | None = None


def set_degraded(reason: DegradedReason) -> None:
    """Mark the daemon as degraded. Subsequent ingest entries should short-circuit."""
    global _state
    with _lock:
        _state = reason


def clear_degraded() -> None:
    """Clear the degraded flag (operator-initiated recheck succeeded)."""
    global _state
    with _lock:
        _state = None


def degraded_reason() -> DegradedReason | None:
    """Return the current degraded reason, or None if healthy."""
    with _lock:
        return _state


def is_degraded() -> bool:
    return degraded_reason() is not None


__all__ = [
    "DegradedReason",
    "clear_degraded",
    "degraded_reason",
    "is_degraded",
    "set_degraded",
]
