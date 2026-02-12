"""Tests for the structured event bus."""
from __future__ import annotations

import pytest

from polylogue.pipeline.event_bus import (
    EventBus,
    ParseFailure,
    ParseProgress,
    PipelineEvent,
    SyncCompleted,
    SyncStarted,
    get_event_bus,
    reset_event_bus,
)


class TestEventBus:
    """Tests for EventBus subscription and dispatch."""

    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe(SyncCompleted, lambda e: received.append(e))
        event = SyncCompleted(new_conversations=5)
        bus.emit(event)
        assert len(received) == 1
        assert received[0].new_conversations == 5

    def test_type_filtering(self):
        bus = EventBus()
        sync_events = []
        parse_events = []
        bus.subscribe(SyncCompleted, lambda e: sync_events.append(e))
        bus.subscribe(ParseProgress, lambda e: parse_events.append(e))

        bus.emit(SyncCompleted(new_conversations=1))
        bus.emit(ParseProgress(conversation_id="abc"))

        assert len(sync_events) == 1
        assert len(parse_events) == 1

    def test_wildcard_subscription(self):
        bus = EventBus()
        all_events = []
        bus.subscribe(PipelineEvent, lambda e: all_events.append(e))

        bus.emit(SyncStarted(source_names=("inbox",)))
        bus.emit(SyncCompleted(new_conversations=2))
        bus.emit(ParseFailure(error="bad json"))

        assert len(all_events) == 3

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        def handler(e):
            received.append(e)
        bus.subscribe(SyncCompleted, handler)
        bus.unsubscribe(SyncCompleted, handler)
        bus.emit(SyncCompleted(new_conversations=1))
        assert len(received) == 0

    def test_handler_exception_does_not_halt_dispatch(self):
        bus = EventBus()
        results = []

        def failing(e):
            raise RuntimeError("handler failed")

        def succeeding(e):
            results.append(e)

        bus.subscribe(SyncCompleted, failing)
        bus.subscribe(SyncCompleted, succeeding)
        bus.emit(SyncCompleted(new_conversations=1))
        assert len(results) == 1  # Second handler still ran

    def test_handler_count(self):
        bus = EventBus()
        bus.subscribe(SyncCompleted, lambda e: None)
        bus.subscribe(ParseProgress, lambda e: None)
        bus.subscribe(PipelineEvent, lambda e: None)
        assert bus.handler_count == 3

    def test_event_timestamp(self):
        event = SyncCompleted(new_conversations=0)
        assert event.timestamp is not None

    def test_frozen_events(self):
        event = SyncCompleted(new_conversations=5)
        with pytest.raises(AttributeError):
            event.new_conversations = 10  # type: ignore


class TestGlobalEventBus:
    """Tests for the global event bus singleton."""

    def test_get_returns_same_instance(self):
        reset_event_bus()
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_clears_bus(self):
        reset_event_bus()
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2
