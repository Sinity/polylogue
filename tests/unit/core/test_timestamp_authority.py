"""Timestamp authority reads the timeline only when a side needs it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import polylogue.sources.prepared_message_sink as sink_module
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.timestamp_authority import normalize_session_timestamps, session_evidence_timestamps
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore

_T0 = 1_780_000_000_000


class CountedMessages:
    """A message sequence that counts how many messages were visited."""

    def __init__(self, points: list[int | None]) -> None:
        self._messages = [SimpleNamespace(occurred_at_ms=point) for point in points]
        self.visits = 0

    def __iter__(self) -> Iterator[SimpleNamespace]:
        for message in self._messages:
            self.visits += 1
            yield message


def _iso(millis: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(millis / 1000, UTC).isoformat()


def _session(messages: object, *, created: int | None, updated: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        messages=messages,
        session_events=(),
        created_at=_iso(created) if created is not None else None,
        updated_at=_iso(updated) if updated is not None else None,
        created_at_provenance="unknown",
        updated_at_provenance="unknown",
    )


def test_a_valid_producer_pair_decides_without_visiting_messages() -> None:
    """Anti-vacuity: reading the timeline before checking the producer pair
    (the retired order) makes ``visits`` equal the message count."""
    messages = CountedMessages([_T0 + index for index in range(3000)])

    pair = session_evidence_timestamps(_session(messages, created=_T0 - 5, updated=_T0 + 9000))

    assert pair == (_T0 - 5, _T0 + 9000)
    assert messages.visits == 0


def test_an_inverted_producer_pair_is_ordered_without_visiting_messages() -> None:
    messages = CountedMessages([_T0])

    assert session_evidence_timestamps(_session(messages, created=_T0 + 10, updated=_T0)) == (_T0, _T0 + 10)
    assert messages.visits == 0


@pytest.mark.parametrize(
    ("created", "updated", "expected"),
    [
        (None, None, (_T0 + 1, _T0 + 7)),
        (_T0 - 3, None, (_T0 - 3, _T0 + 7)),
        (None, _T0 + 20, (_T0 + 1, _T0 + 20)),
    ],
)
def test_a_missing_producer_side_is_derived_from_the_timeline(
    created: int | None, updated: int | None, expected: tuple[int, int]
) -> None:
    messages = CountedMessages([None, _T0 + 7, _T0 + 1, None])

    assert session_evidence_timestamps(_session(messages, created=created, updated=updated)) == expected
    assert messages.visits == 4


def _stored_session(tmp_path: Path, points: list[int | None]) -> tuple[SqliteMessageStore, SqliteMessageSink]:
    store = SqliteMessageStore(tmp_path / "prepared.db")
    sink = store.new_sink()
    sink.extend(
        ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text="hi", occurred_at_ms=point)
        for index, point in enumerate(points)
    )
    return store, sink


def test_a_sealed_prepared_session_answers_its_timeline_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk-backed session reports the same derived pair as the list form,
    from its stored rows. Anti-vacuity: without ``occurred_at_bounds`` the
    authority walks the sink and ``decodes`` counts every message."""
    points = [_T0 + 40, None, _T0 + 2, _T0 + 90]
    store, sink = _stored_session(tmp_path, points)
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    list_session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="s",
        messages=[
            ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text="hi", occurred_at_ms=point)
            for index, point in enumerate(points)
        ],
    )
    decodes = 0
    real_decode = sink_module._from_text_json

    def counting_decode(model: type[ParsedMessage], encoded: str) -> ParsedMessage:
        nonlocal decodes
        decodes += 1
        return real_decode(model, encoded)

    monkeypatch.setattr(sink_module, "_from_text_json", counting_decode)

    disk_pair = session_evidence_timestamps(_session(sealed, created=None, updated=None))

    assert decodes == 0
    assert disk_pair == session_evidence_timestamps(list_session) == (_T0 + 2, _T0 + 90)


def test_a_changed_timeline_changes_the_derived_pair(tmp_path: Path) -> None:
    """A derived pair is never trusted over the timeline it came from: a
    normalized session whose messages change reports the new extrema."""
    store, sink = _stored_session(tmp_path, [_T0 + 5, _T0 + 6])
    session = ParsedSession(source_name=Provider.CODEX, provider_session_id="s", messages=[])
    session = session.model_copy(update={"messages": sink})
    normalized = normalize_session_timestamps(session)
    assert normalized.updated_at_provenance == "derived"
    assert session_evidence_timestamps(normalized) == (_T0 + 5, _T0 + 6)

    sink.append(ParsedMessage(provider_message_id="late", role=Role.USER, text="later", occurred_at_ms=_T0 + 500))

    assert session_evidence_timestamps(normalized) == (_T0 + 5, _T0 + 500)
    store.close()
