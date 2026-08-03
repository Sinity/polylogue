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

    ``derived_only`` (polylogue-gbs02): True when the condition affects only
    a rebuildable/derived tier (index.db, embeddings.db) and NOT a durable
    tier (source.db, user.db). Raw acquisition writes only source.db, so a
    derived-only condition must not stop it -- only the full parse/
    materialize/index path (which reads/writes the stale derived tier) needs
    to stay withheld. Defaults to False (the historical behavior: any
    degraded reason fully stops ingestion), so every existing caller that
    doesn't know about this distinction keeps its current, safe behavior.
    """

    code: str
    message: str
    detail: Mapping[str, Any] | None = None
    derived_only: bool = False

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


def is_fully_degraded() -> bool:
    """True only when the current degraded reason blocks ALL ingestion.

    False for a derived-tier-only reason (``derived_only=True``), which must
    still allow raw acquisition (source.db writes) to proceed -- see
    ``DegradedReason.derived_only``. Callers that need the historical
    "ingestion cannot happen at all" check should use this, not
    ``is_degraded()``, which stays True (correctly) for status/health
    reporting purposes in the derived-only case too.
    """
    reason = degraded_reason()
    return reason is not None and not reason.derived_only


__all__ = [
    "DegradedReason",
    "clear_degraded",
    "degraded_reason",
    "is_degraded",
    "is_fully_degraded",
    "set_degraded",
]
