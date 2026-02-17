"""Structured event bus for the pipeline lifecycle.

Provides typed event classes for each pipeline stage and a central
``EventBus`` for subscription and dispatch.

Event hierarchy (all frozen dataclasses):

    PipelineEvent (base)
    ├── SyncStarted        — emitted when a sync begins
    ├── SourceDiscovered    — emitted when source files are resolved
    ├── ParseProgress       — emitted after each conversation is parsed
    ├── ParseFailure        — emitted when a file/payload fails to parse
    ├── RenderComplete      — emitted when rendering finishes
    ├── IndexUpdated        — emitted when the FTS/vector index is updated
    ├── SchemaDriftDetected — emitted when schema inference detects drift
    └── SyncCompleted       — emitted when a sync finishes (replaces SyncEvent)

Usage::

    bus = EventBus()
    bus.subscribe(SyncCompleted, my_handler)
    bus.emit(SyncCompleted(new_conversations=5, ...))

Handlers are called synchronously in subscription order.  Exceptions in
a handler are logged but do not prevent subsequent handlers from running.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from polylogue.lib.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

E = TypeVar("E", bound="PipelineEvent")


@dataclass(frozen=True)
class PipelineEvent:
    """Base class for all pipeline events.

    All events carry a UTC timestamp and an optional source identifier.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source: str = ""


@dataclass(frozen=True)
class SyncStarted(PipelineEvent):
    """Emitted when a pipeline sync begins."""

    source_names: tuple[str, ...] = ()
    preview: bool = False


@dataclass(frozen=True)
class SourceDiscovered(PipelineEvent):
    """Emitted when source files are resolved for processing."""

    source_name: str = ""
    file_count: int = 0
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParseProgress(PipelineEvent):
    """Emitted after each conversation is successfully parsed."""

    source_name: str = ""
    conversation_id: str = ""
    provider: str = ""
    title: str = ""
    is_new: bool = True


@dataclass(frozen=True)
class ParseFailure(PipelineEvent):
    """Emitted when a file or payload fails to parse."""

    source_name: str = ""
    path: str = ""
    error: str = ""
    error_type: str = ""


@dataclass(frozen=True)
class RenderComplete(PipelineEvent):
    """Emitted when conversation rendering finishes."""

    rendered_count: int = 0
    output_path: str = ""
    format: str = ""


@dataclass(frozen=True)
class IndexUpdated(PipelineEvent):
    """Emitted when the FTS or vector index is updated."""

    indexed_messages: int = 0
    index_type: str = "fts5"  # "fts5" | "vector" | "hybrid"
    is_rebuild: bool = False


@dataclass(frozen=True)
class SchemaDriftDetected(PipelineEvent):
    """Emitted when schema inference detects drift in provider formats."""

    provider: str = ""
    field: str = ""
    expected: str = ""
    actual: str = ""
    severity: str = "warning"  # "warning" | "error"


@dataclass(frozen=True)
class SyncCompleted(PipelineEvent):
    """Emitted when a pipeline sync finishes.

    This is the structured replacement for the original ``SyncEvent`` in
    ``events.py``, carrying richer metadata about the completed run.
    """

    new_conversations: int = 0
    updated_conversations: int = 0
    skipped_conversations: int = 0
    new_messages: int = 0
    duration_ms: int = 0
    run_id: str = ""
    counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Handler type
# ---------------------------------------------------------------------------

EventHandler = Callable[[Any], None]


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Central event dispatcher.

    Supports typed subscriptions: handlers receive only events of the
    exact type they subscribed to (no inheritance matching — subscribe
    to ``PipelineEvent`` to receive *all* events via a wildcard).

    Thread-safety: The bus is designed for single-threaded pipeline
    execution.  For multi-threaded use, wrap ``emit()`` calls or use
    a queue-based adapter.

    Example::

        bus = EventBus()

        def on_sync(event: SyncCompleted) -> None:
            print(f"Synced {event.new_conversations} new conversations")

        bus.subscribe(SyncCompleted, on_sync)
        bus.emit(SyncCompleted(new_conversations=42))
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: The event class to subscribe to.  Use
                ``PipelineEvent`` to subscribe to all events.
            handler: Callable that receives the event instance.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Remove a previously registered handler.

        No-op if the handler was not registered.
        """
        handlers = self._handlers.get(event_type)
        if handlers:
            with suppress(ValueError):
                handlers.remove(handler)

    def emit(self, event: PipelineEvent) -> None:
        """Dispatch an event to all matching handlers.

        Handlers registered for the exact event type are called first,
        followed by handlers registered for ``PipelineEvent`` (wildcard).
        Exceptions in handlers are logged but do not halt dispatch.
        """
        event_type = type(event)

        # Exact-type handlers
        for handler in self._handlers.get(event_type, ()):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event handler %s failed for %s",
                    getattr(handler, "__name__", repr(handler)),
                    event_type.__name__,
                )

        # Wildcard handlers (PipelineEvent) — skip if event IS PipelineEvent
        if event_type is not PipelineEvent:
            for handler in self._handlers.get(PipelineEvent, ()):
                try:
                    handler(event)
                except Exception:
                    logger.exception(
                        "Wildcard handler %s failed for %s",
                        getattr(handler, "__name__", repr(handler)),
                        event_type.__name__,
                    )

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all event types."""
        return sum(len(h) for h in self._handlers.values())


# ---------------------------------------------------------------------------
# Convenience: global default bus
# ---------------------------------------------------------------------------

_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the application-wide event bus singleton."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_event_bus() -> None:
    """Reset the global event bus.  For tests."""
    global _default_bus
    _default_bus = None


__all__ = [
    "PipelineEvent",
    "SyncStarted",
    "SourceDiscovered",
    "ParseProgress",
    "ParseFailure",
    "RenderComplete",
    "IndexUpdated",
    "SchemaDriftDetected",
    "SyncCompleted",
    "EventBus",
    "EventHandler",
    "get_event_bus",
    "reset_event_bus",
]
