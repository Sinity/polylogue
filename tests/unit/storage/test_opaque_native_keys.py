"""Exact Source names survive prepared storage and Index/material identity lowering."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.core.identity_law import message_id, split_message_local_id
from polylogue.core.message_native_identity import (
    message_native_key,
    source_native_id_from_json,
    source_native_id_json,
)
from polylogue.pipeline.ids import disk_message_owner_resolution, message_owner_resolution, session_content_hash
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.prepared_message_sink import ScratchSessionSpill, SqliteMessageStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.index_writer import write_fixture_index_session
from tests.unit.sinex.test_material_adapter import _decoded_publication
from tests.unit.storage.test_archive_tiers_write import _connect

_NAMES = ("\ud800", "\ud800\udc00", "\U00010000", "eda080", "x:s:eda080", " dup ")


@pytest.mark.parametrize("disk", [False, True])
def test_native_keys_keep_exact_source_names_through_index_and_publication(tmp_path: Path, disk: bool) -> None:
    messages = [
        ParsedMessage(
            provider_message_id=native,
            position=i,
            role=Role.USER,
            text="same",
            parent_message_provider_id=_NAMES[i - 1] if i else None,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="same")],
        )
        for i, native in enumerate(_NAMES)
    ]
    store = SqliteMessageStore(tmp_path / "prepared.db")
    sink = store.new_sink()
    sink.extend(messages)
    operand = sink if disk else messages
    assert [message.provider_message_id for message in sink] == list(_NAMES)
    assert set(sink.provider_message_ids(include_none=True)) == set(_NAMES)
    with disk_message_owner_resolution(operand) as owners:
        assert set(owners.unique_provider_keys) == set(_NAMES)
        assert not owners.ambiguous_provider_ids
    session = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="native-law", messages=[]).model_copy(
        update={"messages": operand}
    )
    conn = _connect(tmp_path / "index.db")
    sid = write_fixture_index_session(conn, session)
    rows = conn.execute(
        "SELECT message_id, parent_message_id, source_native_id_json FROM messages ORDER BY position"
    ).fetchall()
    assert [row["message_id"] for row in rows] == [message_id(sid, native) for native in _NAMES]
    assert [source_native_id_from_json(row["source_native_id_json"]) for row in rows] == list(_NAMES)
    assert [row["parent_message_id"] for row in rows[1:]] == [row["message_id"] for row in rows[:-1]]
    _payload, decoded = _decoded_publication(session)
    assert [message.native_id for message in decoded.messages] == list(_NAMES)
    assert [message.source_native_id for message in decoded.messages] == list(_NAMES)
    decoded_sid = decoded.session["session_id"]
    assert isinstance(decoded_sid, str)
    for message, native in zip(decoded.messages, _NAMES, strict=True):
        assert split_message_local_id(message.message_id, parent_session_id=decoded_sid) == (
            native,
            None,
            0,
        )
    conn.close()
    store.close()


def test_native_keys_are_injective_and_independent_of_mutable_content() -> None:
    keys = [message_native_key(native) for native in _NAMES]
    assert len(set(keys)) == len(_NAMES)
    for native in _NAMES:
        assert source_native_id_from_json(source_native_id_json(native)) == native
        first = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="same-native",
            messages=[ParsedMessage(provider_message_id=native, role=Role.USER, text="before")],
        )
        edited = first.model_copy(deep=True)
        edited.messages[0].text = "after"
        assert message_id("session", native) == message_id("session", edited.messages[0].provider_message_id)
        assert session_content_hash(first) != session_content_hash(edited)
    pair = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="unicode",
        messages=[ParsedMessage(provider_message_id="\ud800\udc00", role=Role.USER, text="same")],
    )
    scalar = pair.model_copy(deep=True)
    scalar.messages[0].provider_message_id = "\U00010000"
    assert session_content_hash(pair) != session_content_hash(scalar)


@pytest.mark.parametrize("native", ["\ud800\udc00", "\U00010000"])
def test_asserted_branch_publication_restores_exact_source_native_name(native: str) -> None:
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child",
        parent_session_provider_id="parent",
        branch_point_provider_message_id=native,
        messages=[ParsedMessage(provider_message_id="child-message", role=Role.USER, text="tail")],
    )
    _payload, decoded = _decoded_publication(session)
    assert decoded.lineage[0]["branch_point_message_native_id"] == native


def test_prepared_reference_and_chatgpt_owner_collections_keep_exact_names(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "prepared-reference.db")
    attachments = store.new_attachment_sink()
    events = store.new_event_sink()
    entries = ScratchSessionSpill(store).entries()
    for index, native in enumerate(_NAMES):
        attachments.append(ParsedAttachment(provider_attachment_id="file", message_provider_id=native, name="neutral"))
        events.append(ParsedSessionEvent(event_type="neutral", source_message_provider_id=native, payload={}))
        entries.add(
            None, index, native, ParsedMessage(provider_message_id=native, position=index, role=Role.USER, text="same")
        )
        assert entries.provider_for_node(native) == native
        assert native in entries.emitted_provider_ids()
    assert [attachment.message_provider_id for attachment in attachments] == list(_NAMES)
    assert [event.source_message_provider_id for event in events] == list(_NAMES)
    assert [message.provider_message_id for message in entries.ordered()] == list(_NAMES)
    assert entries.last_emitted_among(frozenset(_NAMES)) == _NAMES[-1]
    store.close()


@pytest.mark.parametrize("duplicates", [False, True])
def test_shard_owner_lookup_preserves_exact_native_keys_and_ambiguities(tmp_path: Path, duplicates: bool) -> None:
    # JSON alone encodes a scalar and its UTF-16 surrogate pair identically.
    # Both Source names must remain separate in the sealed lookup and set.
    names = (*_NAMES, *_NAMES[1:3]) if duplicates else _NAMES
    messages = [
        ParsedMessage(provider_message_id=native, position=i, role=Role.USER, text=f"body {i}")
        for i, native in enumerate(names)
    ]
    expected = message_owner_resolution(messages)
    session = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="native-owner-law", messages=messages)
    shard = prepare_session_shard(tmp_path / "shards", [session])
    restored = shard.sessions[0].owner_resolution
    assert dict(restored.unique_provider_keys) == dict(expected.unique_provider_keys)
    assert set(restored.ambiguous_provider_ids) == set(expected.ambiguous_provider_ids)
    for native in _NAMES:
        assert (native in restored.ambiguous_provider_ids) is (duplicates and native in _NAMES[1:3])
        if native in expected.unique_provider_keys:
            assert restored.unique_provider_keys[native] == expected.unique_provider_keys[native]
    assert "unknown" not in restored.ambiguous_provider_ids
    assert "" not in restored.ambiguous_provider_ids
    assert restored.unique_provider_keys.get("") is None
