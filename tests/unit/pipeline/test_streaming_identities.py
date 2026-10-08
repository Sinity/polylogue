"""Prepared disk sequences keep the canonical identity contract."""

from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import _relation
from polylogue.core.enums import BlockType, Provider
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.pipeline import ids
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore


def test_disk_projection_transfers_files_and_closes_handles_before_iterator_yields(tmp_path: Path) -> None:
    def build() -> ids.SessionRevisionProjection:
        store = SqliteMessageStore(tmp_path / "transfer.db")
        try:
            sink = store.new_sink()
            sink.extend(
                ParsedMessage(provider_message_id=f"message-{number}", role=Role.USER, text=str(number))
                for number in range(700)
            )
            return ids.session_revision_projection(_session([], [], []).model_copy(update={"messages": sink}))
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=1) as producer:
        projection = producer.submit(build).result()
    with ThreadPoolExecutor(max_workers=1) as first_consumer:
        iterator = iter(projection.message_hashes)
        first = first_consumer.submit(next, iterator).result()
    # The first consumer is gone; resuming and abandoning the iterator must
    # not require that thread to close a native connection.
    assert isinstance(first, bytes)
    assert len(tuple(iterator)) == 699
    abandoned = iter(projection.message_contents)
    with ThreadPoolExecutor(max_workers=1) as second_consumer:
        second_consumer.submit(next, abandoned).result()
    assert isinstance(abandoned, Generator)
    abandoned.close()
    assert len(projection.message_contents) == 700
    projection.close()
    projection.close()
    with pytest.raises(RuntimeError):
        next(iter(projection.message_hashes))


@pytest.mark.uses_real_clock("A failed page close retains its actual creator thread and artifact.")
def test_projection_close_preserves_artifact_until_failed_reader_close_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from typing import Any, cast

    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    prepared = SqliteMessageStore(tmp_path / "prepared.db")
    sink = prepared.new_sink()
    sink.append(ParsedMessage(provider_message_id="retained", role=Role.USER, text="retained row"))
    prepared.conn.commit()
    prepared.close()
    sink = SqliteMessageSink(prepared.path, sink.session_ordinal, count=len(sink))
    projection = ids.session_revision_projection(_session([], [], []).model_copy(update={"messages": sink}))
    actual_connect = sqlite3.connect
    opened: list[ControlledConnection] = []

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if isinstance(database, str) and "projection.db?mode=ro" in database:
            kwargs["factory"] = ControlledConnection
            connection = actual_connect(database, *args, **kwargs)
            assert isinstance(connection, ControlledConnection)
            connection.close_failure = OSError("synthetic projection page close failure")
            opened.append(connection)
            return connection
        return cast(sqlite3.Connection, actual_connect(database, *args, **kwargs))

    def failed_page() -> NativeSQLCustodyOwner:
        with pytest.raises(NativeConnectionSettlementError) as refused:
            next(iter(projection.message_hashes))
        return refused.value.owner

    monkeypatch.setattr(sqlite3, "connect", connect)
    with ThreadPoolExecutor(max_workers=1) as reader:
        owner = reader.submit(failed_page).result()
        assert len(opened) == 1
        with pytest.raises(NativeConnectionSettlementError) as refused:
            projection.close()
        assert refused.value.owner is owner
        artifact = projection._artifact_owner
        assert artifact is not None
        assert (Path(artifact._scratch.name) / "projection.db").is_file()
        opened[0].close_failure = None
        reader.submit(owner.close).result()
    projection.close()
    assert not Path(artifact._scratch.name).exists()


def _session(
    messages: list[ParsedMessage], events: list[ParsedSessionEvent], attachments: list[ParsedAttachment]
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="prepared-identity",
        title="caf\u00e9",
        created_at="2024-01-01T00:00:00Z",
        messages=messages,
        session_events=events,
        attachments=attachments,
    )


def test_disk_hash_matches_canonical_tree_with_owners_events_and_nfc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = [
        ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="cafe\u0301",
            timestamp="2024-01-01T00:00:01Z",
            position=position,
            blocks=[ParsedContentBlock(type=BlockType.DOCUMENT, metadata={"ref": reference})],
        )
        for position, reference in enumerate(("document-a", "document-b"))
    ]
    events = [
        ParsedSessionEvent(
            event_type="citation",
            timestamp="2024-01-01T00:00:02Z",
            payload={"label": "cafe\u0301"},
        )
    ]
    attachments = [
        ParsedAttachment(
            provider_attachment_id=reference,
            message_provider_id="",
            message_position=position,
            name="same.txt",
            mime_type="text/plain",
        )
        for position, reference in enumerate(("document-a", "document-b"))
    ]
    attachments.reverse()
    ordinary = _session(messages, events, attachments)
    expected = ids.session_content_hash(ordinary)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        message_sink = store.new_sink()
        message_sink.extend(messages)
        event_sink = store.new_event_sink()
        event_sink.extend(events)
        prepared = ordinary.model_copy(update={"messages": message_sink, "session_events": event_sink})

        def reject_tree(*args: object, **kwargs: object) -> None:
            raise AssertionError("disk-backed hash rematerialized the session tree")

        monkeypatch.setattr(ids, "_session_hash_components", reject_tree)
        assert ids.session_content_hash(prepared) == expected
        assert ids.session_content_hash(prepared.model_copy(update={"title": "cafe\u0301"})) == expected
    finally:
        store.close()


def test_disk_identity_occurrences_match_public_tuple_and_survive_unrelated_insert(tmp_path: Path) -> None:
    repeated = ParsedMessage(provider_message_id="", role=Role.USER, text="same")
    other = ParsedMessage(provider_message_id="", role=Role.USER, text="other")
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        sink.extend([repeated, other, repeated])
        offsets = {ids.message_content_identity(repeated): 4}
        expected = ids.message_content_identities([repeated, other, repeated], occurrence_offsets=offsets)
        with ids.disk_message_content_identities(sink, occurrence_offsets=offsets) as actual:
            assert tuple(actual) == expected
            assert actual[-1] == expected[-1]
            assert actual[1:3] == list(expected[1:3])
        sink.append(other)
        with ids.disk_message_content_identities(sink, occurrence_offsets=offsets) as actual:
            assert actual[2] == expected[2]
            assert actual[3][1] == 1
    finally:
        store.close()


def test_disk_hash_preserves_ambiguous_attachment_fallback(tmp_path: Path) -> None:
    repeated = [
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same", position=position)
        for position in (0, 1)
    ]
    attachment = ParsedAttachment(
        provider_attachment_id="document",
        message_provider_id="",
        message_position=0,
        name="same.txt",
        mime_type="text/plain",
    )
    ordinary = _session(repeated, [], [attachment])
    expected = ids.session_content_hash(ordinary)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        sink.extend(repeated)
        prepared = ordinary.model_copy(update={"messages": sink})
        assert ids.session_content_hash(prepared) == expected
    finally:
        store.close()


def test_disk_hash_preserves_private_stable_owner_evidence(tmp_path: Path) -> None:
    coordinate = MessageOwnerCoordinate(stable_key="provider-stable-anchor")
    message = ParsedMessage(
        provider_message_id="",
        role=Role.ASSISTANT,
        text="owner",
        owner_coordinate=coordinate,
    )
    attachment = ParsedAttachment(
        provider_attachment_id="document",
        message_provider_id="",
        owner_coordinate=coordinate,
        name="document.txt",
        mime_type="text/plain",
    )
    ordinary = _session([message], [], [attachment])
    expected = ids.session_content_hash(ordinary)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        sink.append(message)
        assert sink[0].owner_coordinate == coordinate
        prepared = ordinary.model_copy(update={"messages": sink})
        assert ids.session_content_hash(prepared) == expected
    finally:
        store.close()


def test_disk_owner_keys_match_public_resolution_at_each_ordinal(tmp_path: Path) -> None:
    messages = [
        ParsedMessage(provider_message_id="native", role=Role.USER, text="first", position=0),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same", position=1),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same", position=2),
        ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="stable",
            owner_coordinate=MessageOwnerCoordinate(stable_key="stable-owner", position=3),
        ),
    ]
    expected = ids.message_owner_resolution(messages)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        sink.extend(messages)
        with ids.disk_message_owner_resolution(sink) as actual:
            assert len(actual.keys) == len(messages)
            assert tuple(actual.keys) == expected.keys
            assert actual.keys[-1] == expected.keys[-1]
            assert actual.keys[1:3] == list(expected.keys[1:3])
            assert set(actual.ambiguous_keys) == expected.ambiguous_keys
            assert dict(actual.by_physical_coordinate) == expected.by_physical_coordinate
            assert dict(actual.by_stable_key) == expected.by_stable_key
            assert dict(actual.unique_provider_keys) == expected.unique_provider_keys
    finally:
        store.close()


def test_set_based_disk_owner_resolution_matches_public_resolution_under_every_collision(tmp_path: Path) -> None:
    """The disk resolver's set-based counts and keys equal the in-memory law.

    Duplicate native ids, duplicate stable keys, a stable key colliding with
    another message's key, shared physical coordinates and identical content
    each change which anchor wins; a count or a lookup taken over the wrong
    rows turns one of these comparisons red.
    """
    messages = [
        ParsedMessage(provider_message_id="dup", role=Role.USER, text="a", position=0),
        ParsedMessage(provider_message_id="dup", role=Role.USER, text="b", position=1),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same", position=2),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same", position=2),
        ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="x",
            owner_coordinate=MessageOwnerCoordinate(stable_key="shared", position=4),
        ),
        ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="y",
            owner_coordinate=MessageOwnerCoordinate(stable_key="shared", position=5),
        ),
        ParsedMessage(
            provider_message_id="unique",
            role=Role.USER,
            text="ü",
            owner_coordinate=MessageOwnerCoordinate(stable_key="ünique-stable", position=6),
        ),
        ParsedMessage(provider_message_id="", role=Role.USER, text="lonely", position=7),
    ]
    expected = ids.message_owner_resolution(messages)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        for message in messages:
            sink.append(message)
        with ids.disk_message_owner_resolution(sink) as actual:
            assert tuple(actual.keys) == expected.keys
            assert set(actual.ambiguous_keys) == expected.ambiguous_keys
            assert set(actual.ambiguous_stable_keys) == expected.ambiguous_stable_keys
            assert set(actual.ambiguous_provider_ids) == expected.ambiguous_provider_ids
            assert set(actual.ambiguous_physical_coordinates) == expected.ambiguous_physical_coordinates
            assert dict(actual.by_physical_coordinate.items()) == expected.by_physical_coordinate
            assert dict(actual.by_stable_key.items()) == expected.by_stable_key
            assert dict(actual.unique_provider_keys.items()) == expected.unique_provider_keys
    finally:
        store.close()


def test_disk_revision_projection_matches_every_canonical_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [
        ParsedMessage(provider_message_id="native", role=Role.USER, text="first"),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
        ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="same"),
    ]
    events = [
        ParsedSessionEvent(event_type="citation", timestamp="2024-01-01T00:00:02Z", payload={"label": "cafe\u0301"})
    ]
    attachments = [
        ParsedAttachment(
            provider_attachment_id="document",
            message_provider_id="native",
            name="doc.txt",
            mime_type="text/plain",
            inline_bytes=b"document bytes",
        )
    ]
    ordinary = _session(messages, events, attachments)
    expected = ids.session_revision_projection(ordinary)
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        sink = store.new_sink()
        sink.extend(messages)
        event_sink = store.new_event_sink()
        event_sink.extend(events)
        prepared = ordinary.model_copy(update={"messages": sink, "session_events": event_sink})

        def reject_tree(*args: object, **kwargs: object) -> None:
            raise AssertionError("disk projection rematerialized the session tree")

        monkeypatch.setattr(ids, "_session_hash_components", reject_tree)
        actual = ids.session_revision_projection(prepared)
        assert actual.session_hash == expected.session_hash
        assert tuple(actual.message_hashes) == expected.message_hashes
        assert tuple(actual.event_hashes) == expected.event_hashes
        for field in (
            "message_contents",
            "mutable_message_identities",
            "attachment_identities",
            "attachment_contents",
            "event_contents",
            "anchor_free_event_identities",
        ):
            assert frozenset(getattr(actual, field)) == getattr(expected, field)
        assert _relation(expected, actual) == "equal"
        actual.close()
        expected.close()
    finally:
        store.close()


@pytest.mark.parametrize(
    ("older", "newer", "expected"),
    [
        (["first"], ["first", "second"], "b_contains_a"),
        (["same"], ["same", "same"], "b_contains_a"),
        (["first"], ["changed"], "conflict"),
    ],
)
def test_disk_revision_relation_preserves_growth_and_conflict(
    tmp_path: Path, older: list[str], newer: list[str], expected: str
) -> None:
    def session(texts: list[str]) -> ParsedSession:
        return _session([ParsedMessage(provider_message_id="", role=Role.USER, text=text) for text in texts], [], [])

    old = session(older)
    new = session(newer)
    expected_relation = _relation(ids.session_revision_projection(old), ids.session_revision_projection(new))
    assert expected_relation == expected
    stores = []
    try:
        prepared = []
        for ordinal, original in enumerate((old, new)):
            store = SqliteMessageStore(tmp_path / f"prepared-{ordinal}.db")
            stores.append(store)
            sink = store.new_sink()
            sink.extend(original.messages)
            prepared.append(original.model_copy(update={"messages": sink}))
        assert (
            _relation(ids.session_revision_projection(prepared[0]), ids.session_revision_projection(prepared[1]))
            == expected_relation
        )
    finally:
        for store in stores:
            store.close()


def test_disk_revision_relation_pairs_remeasured_event_anchor(tmp_path: Path) -> None:
    message = ParsedMessage(provider_message_id="native", role=Role.ASSISTANT, text="answer")

    def session(anchor: str, state: str = "completed") -> ParsedSession:
        return _session(
            [message],
            [
                ParsedSessionEvent(
                    event_type="generation_lifecycle",
                    timestamp="13.0",
                    source_message_provider_id=anchor,
                    payload={
                        "state": state,
                        "evidence_source": "provider_native",
                        "fidelity": "exact",
                        "duration_semantics": "provider_reported_elapsed",
                        "elapsed_duration_ms": 13000,
                    },
                )
            ],
            [],
        )

    originals = [session("anchor-a"), session("anchor-b"), session("anchor-b", "interrupted")]
    stores = []
    try:
        prepared = []
        for ordinal, original in enumerate(originals):
            store = SqliteMessageStore(tmp_path / f"event-{ordinal}.db")
            stores.append(store)
            message_sink = store.new_sink()
            message_sink.extend(original.messages)
            event_sink = store.new_event_sink()
            event_sink.extend(original.session_events)
            prepared.append(original.model_copy(update={"messages": message_sink, "session_events": event_sink}))
        projections = [ids.session_revision_projection(item) for item in prepared]
        assert projections[0].event_contents != projections[1].event_contents
        assert _relation(projections[0], projections[1]) == "equal"
        assert _relation(projections[0], projections[2]) == "conflict"
    finally:
        for store in stores:
            store.close()
