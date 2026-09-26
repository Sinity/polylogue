"""Prepared disk sequences keep the canonical identity contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session_revision_membership import _relation
from polylogue.core.enums import BlockType, Provider
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.pipeline import ids
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.prepared_message_sink import SqliteMessageStore


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
