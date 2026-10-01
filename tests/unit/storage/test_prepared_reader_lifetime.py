"""Sealed preparation iterators transport rows without live SQLite handles."""

from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from polylogue.core.enums import Role
from polylogue.core.sql_settlement import retained_native_sql_owners
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSessionEvent
from polylogue.sources.prepared_message_sink import (
    SqliteAttachmentSink,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
)

pytestmark = pytest.mark.uses_real_clock("Actual producer and consumer threads prove native iterator cleanup affinity.")


@pytest.mark.parametrize("kind", ["messages", "attachments", "events", "ordered-events", "provider-ids"])
@pytest.mark.parametrize("action", ["resume", "abandon"])
def test_sealed_prepared_iterator_can_resume_or_be_abandoned_on_another_thread(
    tmp_path: Path, kind: str, action: str
) -> None:
    path = tmp_path / "prepared.db"
    store = SqliteMessageStore(path)
    messages = store.new_sink()
    attachments = store.new_attachment_sink()
    events = store.new_event_sink()
    for index in range(700):
        messages.append(ParsedMessage(provider_message_id=f"m-{index:04d}", role=Role.USER, text=f"row {index}"))
        attachments.append(
            ParsedAttachment(provider_attachment_id=f"a-{index:04d}", message_provider_id=f"m-{index:04d}")
        )
        events.append(ParsedSessionEvent(event_type="turn_context", timestamp="2026-01-01T00:00:00Z"))
    store.conn.commit()
    store.close()
    sealed_messages = SqliteMessageSink(path, messages.session_ordinal, count=700)
    carrier: Iterable[object]
    if kind == "messages":
        carrier = sealed_messages
    elif kind == "attachments":
        carrier = SqliteAttachmentSink(path, attachments.session_ordinal, count=700)
    elif kind == "events":
        carrier = SqliteSessionEventSink(path, events.session_ordinal, count=700)
    elif kind == "ordered-events":
        carrier = SqliteSessionEventSink(path, events.session_ordinal, count=700).iter_ordered({"turn_context": 1})
    else:
        carrier = sealed_messages.provider_message_ids(include_none=False)

    def start() -> tuple[object, Iterator[object]]:
        iterator = iter(carrier)
        first = next(iterator)
        assert retained_native_sql_owners() == ()
        return first, iterator

    def finish(iterator: Iterator[object]) -> list[object]:
        remaining = list(iterator)
        assert retained_native_sql_owners() == ()
        return remaining

    with ThreadPoolExecutor(max_workers=1) as producer, ThreadPoolExecutor(max_workers=1) as consumer:
        _first, iterator = producer.submit(start).result()
        if action == "resume":
            assert len(consumer.submit(finish, iterator).result()) == 699
        else:
            assert isinstance(iterator, Generator)
            consumer.submit(iterator.close).result()
            assert consumer.submit(retained_native_sql_owners).result() == ()
    assert retained_native_sql_owners() == ()
