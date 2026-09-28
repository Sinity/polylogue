"""The cancellation a daemon compute owner publishes to the work it runs.

A derivation pass runs synchronously on a compute thread and cannot be
interrupted from the event loop; the owner shields and settles it on
cancellation. Work that waits without a deadline (a retained preparation in
``polylogue.storage``) must still stop when the owner is cancelled, and it may
not import ``polylogue.daemon``, so the owner's event is published here -- the
same placement argument as ``polylogue.core.write_hold``.
"""

from __future__ import annotations

import contextvars
import threading

#: Set by the compute owner in the context it submits a pass under; the
#: event is set when that owner is cancelled.
compute_cancel: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("compute_cancel", default=None)


def compute_cancel_requested() -> bool:
    """Whether the compute owner running this code has been cancelled."""
    cancelled = compute_cancel.get()
    return cancelled is not None and cancelled.is_set()


__all__ = ["compute_cancel", "compute_cancel_requested"]
