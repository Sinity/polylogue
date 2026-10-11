"""Opt-in observations of actual owner bodies and their artifact consumers.

These references have no execution, reuse, publication or lifetime authority.
Owners supply identities only when they already have complete input evidence;
an unavailable identity still produces a body record. No per-row instrumentation
or new scheduler is involved. CPU comes from the existing synchronous span.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

from polylogue import logging as plog

ENABLED = os.environ.get("POLYLOGUE_TRACE_WORK") == "1"
# Import initialization is serialized. Forked children share the seed but have
# a different PID; neither process nor native-thread ID alone is a lifetime.
_PROCESS_SEED = uuid4().hex if ENABLED else ""
_worker = threading.local()


def _identity() -> tuple[str, str, int]:
    pid = os.getpid()
    if getattr(_worker, "pid", None) != pid:
        _worker.pid = pid
        _worker.generation = uuid4().hex
        _worker.sequence = 0
    _worker.sequence += 1
    return f"{pid}:{_PROCESS_SEED}", _worker.generation, _worker.sequence


@dataclass(slots=True)
class BodyObservation:
    """One synchronous computation body; metadata never certifies its result."""

    _span: plog.Span

    def set_work_key(self, key: str | None, *, unavailable_reason: str | None = None) -> None:
        self._span.set(
            work_key=key, work_key_reason=None if key else unavailable_reason or "input_identity_unavailable"
        )

    def produced(self) -> str:
        """Assign an opaque reference after the owner actually creates a result."""
        reference = uuid4().hex
        self._span.set(artifact_ref=reference)
        return reference


@contextlib.contextmanager
def body(component: str, operation: str, *, work_key: str | None = None) -> Iterator[BodyObservation | None]:
    """Record one actual body, including failed and unmatched started bodies.

    Guard expensive key construction with ENABLED at the caller. With it off,
    this performs no identity allocation, clock sampling, hashing or emission.
    """
    if not ENABLED:
        yield None
        return
    process, worker, sequence = _identity()
    with plog.span(
        "work.body",
        component=component,
        operation=operation,
        attempt_id=uuid4().hex,
        process_generation_id=process,
        worker_generation_id=worker,
        producer_seq=sequence,
        work_key=work_key,
        work_key_reason=None if work_key else "input_identity_unavailable",
        artifact_ref=None,
        span_kind="work",
    ) as active:
        # span.start is DEBUG and may be suppressed. The opt-in body entry must
        # survive ordinary INFO logging to reveal an interrupted body.
        plog.emit("work.body.started")
        observed = BodyObservation(active)
        try:
            yield observed
        finally:
            _process, _worker_id, terminal_sequence = _identity()
            active.set(producer_seq=terminal_sequence)
        active.ok()


def consume(
    component: str,
    operation: str,
    *,
    artifact_ref: str | None,
    work_key: str | None = None,
    outcome: str,
    reason: str | None = None,
) -> None:
    """Observe the owner's existing consumption/publication decision.

    A cache hit is a consumer, never another computation or evidence of avoided
    CPU. Missing producer references are explicit rather than invented.
    """
    if not ENABLED:
        return
    process, worker, sequence = _identity()
    fields: dict[str, object] = {}
    if reason is not None:
        fields["reason"] = reason
    plog.emit(
        "work.artifact.consume",
        component=component,
        operation=operation,
        attempt_id=uuid4().hex,
        process_generation_id=process,
        worker_generation_id=worker,
        producer_seq=sequence,
        artifact_ref=artifact_ref,
        work_key=work_key,
        work_key_reason=None if work_key else "input_identity_unavailable",
        effect=outcome,
        outcome="refused" if "refus" in outcome else "ok",
        **fields,
    )
