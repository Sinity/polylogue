"""Writer admission for the generic convergence stage engine.

The stage walk used to run entirely inside the daemon writer lease: its caller
wrapped :meth:`DaemonConverger.converge_batch` in ``run_sync``, so a Drive
download, an archive-wide materialization and a transport drain all executed
while every other archive writer was queued behind them.

The engine now runs off the lease. A stage that needs the writer brackets the
short section that actually writes with :func:`admit_stage_write`, which hops
onto the daemon's single writer for that section only. A stage whose execute is
not split yet declares ``writer_admission="whole_execute"`` and the engine
brackets the whole call, so the residual is a named, typed field rather than a
silent property of the caller's control flow.

This lives in ``core`` rather than ``daemon`` because the write sections it
brackets belong to analysis materializers, source-tier publication and
operations owners, none of which may import the daemon surface. The daemon
supplies the admission callable; every other ring only declares where its
write begins and ends.

With no admission bound -- a standalone CLI pass, a unit test, or a caller that
already holds the lease -- the work runs inline. Admission is a routing
decision, never a second write authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

T = TypeVar("T")

#: ``(actor, work) -> result``: run ``work`` under the daemon's sole writer.
StageWriteAdmission = Callable[[str, "Callable[[], Any]"], Any]

_STAGE_WRITE_ADMISSION: ContextVar[StageWriteAdmission | None] = ContextVar(
    "polylogue_stage_write_admission", default=None
)


@contextmanager
def stage_write_admission(admission: StageWriteAdmission | None) -> Iterator[None]:
    """Bind the writer admission the stage engine's writes must travel through."""
    token = _STAGE_WRITE_ADMISSION.set(admission)
    try:
        yield
    finally:
        _STAGE_WRITE_ADMISSION.reset(token)


def stage_write_admission_bound() -> bool:
    """Whether a stage running here would bridge its write to the daemon writer."""
    return _STAGE_WRITE_ADMISSION.get() is not None


def admit_stage_write(actor: str, work: Callable[[], T]) -> T:
    """Run one stage's write section under the daemon writer.

    Falls through to a direct call when no admission is bound or when this
    context already holds the lease, so the same stage body is correct inside a
    standalone pass, a test and the daemon.
    """
    admission = _STAGE_WRITE_ADMISSION.get()
    if admission is None or _lease_already_held():
        return work()
    result: T = admission(actor, work)
    return result


def _lease_already_held() -> bool:
    from polylogue.core.write_lease import current_write_lease

    return current_write_lease() is not None


__all__ = [
    "StageWriteAdmission",
    "admit_stage_write",
    "stage_write_admission",
    "stage_write_admission_bound",
]
