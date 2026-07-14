"""Typed in-process event bus for daemon loops (polylogue-yp0).

The daemon runs ~9 concurrent maintenance loops (mapped by
polylogue-9e5.7's lock/starvation audit) coordinating today through
polling and shared tables: convergence checks for debt, embedding
catch-up polls pending counts, FTS status re-derives, alerts re-scan. Each
new daemon feature has meant another loop with its own cadence, its own DB
polling, and its own interaction-risk surface against the daemon's
single-writer invariant.

This module gives the daemon an internal nervous system: one small typed
pub/sub bus, in-process only (no broker, no cross-process delivery — that
is Sinex's territory), that loops can subscribe to instead of polling.

Design contract (see polylogue-yp0's design note for the full rationale):

- **Events are frozen dataclasses.** Each event type is a plain,
  immutable, serialization-friendly record of something that already
  happened (past tense: ``IngestCommitted``, not ``CommitIngest``).
- **Delivery is best-effort and in-process.** ``publish`` calls every
  subscriber for the event's type synchronously (there is no separate
  broker thread/process) and isolates subscriber failures — one
  subscriber raising must never stop delivery to the others, and must
  never silently kill future delivery for that event type. This is the
  concrete form of "a crashed subscriber must not silently stop forever":
  a subscriber that always raises still gets called on every future
  publish; it just never observes success. Callers that need to detect a
  permanently-broken subscriber should track their own health via a slow
  reconcile tick (a periodic self-check independent of bus delivery), not
  rely on the bus for that signal.
- **The bus does not replace correctness-critical polling.** A subscriber
  reacting to ``EmbeddingPending`` is an optimization (wake up sooner);
  the daemon's existing periodic catch-up loops remain the source of
  truth for "did we actually process everything," because bus delivery
  can be missed (e.g. published before any subscriber existed, or during
  a subscriber's own crash-and-restart window). Converting a loop to
  subscribe means "poll less often, react faster," not "stop polling."

**Status**: this bead lands the bus core with full test coverage — the
typed event vocabulary, publish/subscribe/unsubscribe, and the
failure-isolation contract above. Converting an actual daemon loop
(embedding catch-up, convergence debt re-scan, etc.) from polling to
subscribing is deliberately sequenced as follow-up work per the design
note ("Sequence: after 9e5.7 produces the loop inventory, convert with the
map in hand") — touching a live ~9-loop daemon's lock/starvation
invariants needs its own focused pass, not a byproduct of landing the bus
itself. See the child bead filed alongside this module for that
conversion work.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DaemonEvent:
    """Base type for every event the bus can carry.

    Not meant to be published directly — subclass with a concrete,
    past-tense event describing something that already happened.
    """


@dataclass(frozen=True, slots=True)
class IngestCommitted(DaemonEvent):
    """An archive write committed and these sessions changed."""

    cursor: str | None
    session_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CursorMoved(DaemonEvent):
    """A source's ingest cursor advanced."""

    source_name: str
    cursor: str


@dataclass(frozen=True, slots=True)
class ConvergenceStateChanged(DaemonEvent):
    """A convergence stage's state for a session/file transitioned."""

    stage_name: str
    key: str
    from_state: str | None
    to_state: str


@dataclass(frozen=True, slots=True)
class EmbeddingPending(DaemonEvent):
    """New embedding work became available."""

    count: int


@dataclass(frozen=True, slots=True)
class BlobLeaseReleased(DaemonEvent):
    """A blob-GC lease was released, making a blob hash collectible again."""

    blob_hash: str


EventHandler = Callable[[Any], None]
"""A subscriber callback. Bound loosely (``Any``) at the alias level because
Python 3.11 lacks PEP 695 generic type-alias syntax; ``EventBus.subscribe``'s
own signature is the precise, per-call-site typed contract (a handler
registered for ``event_type: type[E]`` must accept ``E``)."""

_E = TypeVar("_E", bound=DaemonEvent)


class EventBus:
    """A small typed in-process pub/sub bus.

    One bus instance is meant to be shared for the lifetime of a daemon
    process (constructed once in ``run_daemon_services`` and threaded to
    whichever components publish/subscribe); it holds no external
    resources and needs no async context of its own — ``publish`` runs
    subscribers synchronously and callers that need async work should
    have their handler schedule it (e.g. ``asyncio.create_task``) rather
    than expecting the bus to await handlers.
    """

    _subscribers: dict[type[DaemonEvent], list[EventHandler]]

    def __init__(self) -> None:
        self._subscribers = defaultdict(list)

    def subscribe(self, event_type: type[_E], handler: Callable[[_E], None]) -> Callable[[], None]:
        """Register ``handler`` for ``event_type``.

        Returns an unsubscribe callable — call it to stop receiving
        ``event_type`` events. Safe to call more than once (idempotent).
        """
        self._subscribers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._subscribers.get(event_type)
            if handlers is not None and handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def publish(self, event: DaemonEvent) -> int:
        """Deliver ``event`` to every subscriber registered for its type.

        Delivery is synchronous and best-effort: a subscriber raising is
        logged and does not prevent delivery to the remaining subscribers
        or poison future publishes (per the module's failure-isolation
        contract). Returns the number of subscribers the event was
        successfully delivered to (a subscriber whose handler raised
        still counts as "attempted," not "delivered").
        """
        handlers = self._subscribers.get(type(event), ())
        delivered = 0
        for handler in tuple(handlers):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "daemon_event_bus_subscriber_failed event=%s handler=%r",
                    type(event).__name__,
                    handler,
                )
                continue
            delivered += 1
        return delivered

    def subscriber_count(self, event_type: type[DaemonEvent]) -> int:
        return len(self._subscribers.get(event_type, ()))


__all__ = [
    "BlobLeaseReleased",
    "ConvergenceStateChanged",
    "CursorMoved",
    "DaemonEvent",
    "EmbeddingPending",
    "EventBus",
    "EventHandler",
    "IngestCommitted",
]
