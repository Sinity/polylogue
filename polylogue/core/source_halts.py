"""Process-local per-source ingest halts.

A structural condition can stop one source while the rest stay healthy: a
derived-tier mismatch stops the sources whose ingest reads that tier, and the
process-wide flag in ``polylogue.core.degraded`` is a single slot that cannot
express which. The registry here answers per source, so the catch-up planner
can drop a halted source where work is *selected* instead of paying a chunk
and a writer lease per candidate to be refused where work executes.

Like the process-wide flag this is process-local, and there is no in-place
remedy the daemon can apply, so a halt lasts until an explicit clear or a
restart. It lives beside that flag in ``core/`` so ``polylogue.sources`` can
read and write it without importing ``daemon/``, and in its own module so
that ingest bookkeeping stays outside the derived-schema identity closure
that ``degraded`` belongs to.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import MappingProxyType

from polylogue.core.degraded import DegradedReason

_lock = threading.Lock()
_halts: dict[str, DegradedReason] = {}


def set_source_halt(source_name: str, reason: DegradedReason) -> bool:
    """Halt one source; True when it was not already halted."""
    with _lock:
        already = source_name in _halts
        _halts[source_name] = reason
        return not already


def clear_source_halt(source_name: str) -> None:
    with _lock:
        _halts.pop(source_name, None)


def clear_all_source_halts() -> None:
    with _lock:
        _halts.clear()


def source_halt(source_name: str) -> DegradedReason | None:
    """Return why this source is halted, or None when it can make progress."""
    with _lock:
        return _halts.get(source_name)


def halted_sources() -> Mapping[str, DegradedReason]:
    """Every currently halted source, for planning and status projection."""
    with _lock:
        return MappingProxyType(dict(_halts))


__all__ = [
    "clear_all_source_halts",
    "clear_source_halt",
    "halted_sources",
    "set_source_halt",
    "source_halt",
]
