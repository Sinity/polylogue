from __future__ import annotations

import dataclasses

import pytest

from polylogue.daemon.event_bus import (
    BlobLeaseReleased,
    ConvergenceStateChanged,
    CursorMoved,
    DaemonEvent,
    EmbeddingPending,
    EventBus,
    IngestCommitted,
)


def test_publish_delivers_to_every_subscriber_of_the_event_type() -> None:
    """Seeded positive case: two subscribers on the same event type both fire."""
    bus = EventBus()
    received_a: list[IngestCommitted] = []
    received_b: list[IngestCommitted] = []
    bus.subscribe(IngestCommitted, received_a.append)
    bus.subscribe(IngestCommitted, received_b.append)

    event = IngestCommitted(cursor="c1", session_refs=("s1", "s2"))
    delivered = bus.publish(event)

    assert delivered == 2
    assert received_a == [event]
    assert received_b == [event]


def test_publish_with_no_subscribers_delivers_to_nobody() -> None:
    """Degraded/empty case: publishing an event nobody subscribed to is a no-op,
    not an error — the bus's whole point is that producers don't need to know
    whether any consumer exists yet."""
    bus = EventBus()
    delivered = bus.publish(IngestCommitted(cursor=None, session_refs=()))
    assert delivered == 0


def test_publish_only_delivers_to_subscribers_of_the_matching_event_type() -> None:
    bus = EventBus()
    ingest_events: list[DaemonEvent] = []
    cursor_events: list[DaemonEvent] = []
    bus.subscribe(IngestCommitted, ingest_events.append)
    bus.subscribe(CursorMoved, cursor_events.append)

    bus.publish(IngestCommitted(cursor="c1", session_refs=("s1",)))

    assert len(ingest_events) == 1
    assert cursor_events == []


def test_subscriber_exception_does_not_prevent_delivery_to_other_subscribers() -> None:
    """A crashed subscriber must not silently stop forever (module contract):
    one raising subscriber must not poison delivery to its siblings on the
    same publish call."""
    bus = EventBus()
    received: list[IngestCommitted] = []

    def boom(_event: IngestCommitted) -> None:
        raise RuntimeError("simulated subscriber failure")

    bus.subscribe(IngestCommitted, boom)
    bus.subscribe(IngestCommitted, received.append)

    event = IngestCommitted(cursor="c1", session_refs=())
    delivered = bus.publish(event)

    assert delivered == 1  # only the non-raising subscriber counts as delivered
    assert received == [event]


def test_subscriber_exception_does_not_poison_future_publishes() -> None:
    """A subscriber that always raises still gets called on every future
    publish — the bus never silently drops it after one failure."""
    bus = EventBus()
    attempts = 0

    def always_boom(_event: IngestCommitted) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("always fails")

    bus.subscribe(IngestCommitted, always_boom)

    bus.publish(IngestCommitted(cursor="c1", session_refs=()))
    bus.publish(IngestCommitted(cursor="c2", session_refs=()))
    bus.publish(IngestCommitted(cursor="c3", session_refs=()))

    assert attempts == 3


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[IngestCommitted] = []
    unsubscribe = bus.subscribe(IngestCommitted, received.append)

    bus.publish(IngestCommitted(cursor="c1", session_refs=()))
    unsubscribe()
    bus.publish(IngestCommitted(cursor="c2", session_refs=()))

    assert len(received) == 1
    assert received[0].cursor == "c1"


def test_unsubscribe_is_idempotent() -> None:
    bus = EventBus()
    unsubscribe = bus.subscribe(IngestCommitted, lambda _e: None)
    unsubscribe()
    unsubscribe()  # must not raise


def test_subscriber_count_reflects_registrations_and_unsubscribes() -> None:
    bus = EventBus()
    assert bus.subscriber_count(IngestCommitted) == 0
    unsubscribe = bus.subscribe(IngestCommitted, lambda _e: None)
    assert bus.subscriber_count(IngestCommitted) == 1
    unsubscribe()
    assert bus.subscriber_count(IngestCommitted) == 0


def test_all_five_declared_event_types_are_frozen_dataclasses() -> None:
    """Pins the design note's typed-event vocabulary — a regression here
    means an event type lost its frozen/immutable guarantee or was
    accidentally removed from the public surface."""
    for event_type in (
        IngestCommitted,
        CursorMoved,
        ConvergenceStateChanged,
        EmbeddingPending,
        BlobLeaseReleased,
    ):
        assert issubclass(event_type, DaemonEvent)
        assert dataclasses.is_dataclass(event_type)
        params = getattr(event_type, "__dataclass_params__")  # noqa: B009
        assert params.frozen


def test_event_dataclasses_are_immutable() -> None:
    event = CursorMoved(source_name="codex-session", cursor="abc123")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.cursor = "mutated"  # type: ignore[misc]
