"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import IO

import ijson
import pytest

from polylogue.core.enums import Provider, Role
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import iter_grok_export_events
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import (
    _ACTIVE_PARENT_LOOKUP_SQL,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.source_builders import ChatGPTExportBuilder


def _prepared_artifact(tmp_path: Path) -> tuple[PreparedJsonl, MessageOwnerCoordinate]:
    path = tmp_path / "prepared.db"
    store = SqliteMessageStore(path)
    messages = store.new_sink()
    events = store.new_event_sink()
    coordinate = MessageOwnerCoordinate(position=0, variant_index=2)
    messages.append(
        ParsedMessage(
            provider_message_id="message-1",
            role=Role.USER,
            text="A neutral prompt",
            variant_index=2,
            parent_message_position=0,
            owner_coordinate=coordinate,
        )
    )
    events.append(
        ParsedSessionEvent(
            event_type="turn_context",
            boundary_message_position=0,
        )
    )
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-1",
        messages=[],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="attachment-1",
                message_position=0,
                message_variant_index=2,
                owner_coordinate=coordinate,
                inline_bytes=b"\x00\xff",
                precomputed_blob=("a" * 64, 2),
            )
        ],
    ).model_copy(update={"messages": messages, "session_events": events})
    session.content_hash = session_content_hash(session)
    shard = prepare_session_shard(tmp_path, [session])
    _write_artifact(
        store,
        "b" * 64,
        [session],
        enrichment_digest="c" * 64,
        enrichment_index_path="/index.db",
    )
    store.close()

    artifact = PreparedJsonl.seal(
        "b" * 64,
        path,
        shard.path,
        enrichment_digest="c" * 64,
        enrichment_index_path="/index.db",
    )
    return artifact, coordinate


def test_prepared_active_path_parent_lookup_uses_provider_index(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "active-path.db")
    try:
        plan = store.conn.execute("EXPLAIN QUERY PLAN " + _ACTIVE_PARENT_LOOKUP_SQL, (0, "tail")).fetchall()
        assert any("prepared_message_provider" in str(row[3]) for row in plan)

        messages = store.new_sink()
        for index in range(500):
            messages.append(
                ParsedMessage(
                    provider_message_id=f"message-{index}",
                    parent_message_provider_id=f"message-{index - 1}" if index else None,
                    role=Role.USER,
                    text="A neutral prompt",
                    is_active_leaf=index == 499,
                )
            )
        messages.normalize_active_path()
        assert all(message.is_active_path for message in messages)
    finally:
        store.close()


def test_prepared_artifact_preserves_private_linkage_and_refuses_changed_seal(tmp_path: Path) -> None:
    artifact, coordinate = _prepared_artifact(tmp_path)
    artifact.verify_files(full=True)
    restored = list(artifact.iter_sessions())
    assert len(restored) == 1
    assert isinstance(restored[0].messages, SqliteMessageSink)
    assert isinstance(restored[0].session_events, SqliteSessionEventSink)
    assert restored[0].messages[0].owner_coordinate == coordinate
    assert restored[0].messages[0].parent_message_position == 0
    assert restored[0].session_events[0].boundary_message_position == 0
    assert restored[0].attachments[0].owner_coordinate == coordinate
    assert restored[0].attachments[0].inline_bytes == b"\x00\xff"
    assert restored[0].attachments[0].precomputed_blob == ("a" * 64, 2)

    assert artifact.sessions_path is not None
    os.chmod(artifact.sessions_path, 0o600)
    with sqlite3.connect(artifact.sessions_path) as conn:
        conn.execute("UPDATE artifact_seal SET enrichment_digest = ?", ("d" * 64,))
    with pytest.raises(ValueError, match="identity changed"):
        list(artifact.iter_sessions())


def test_prepared_artifact_refuses_same_count_row_change_and_file_replacement(tmp_path: Path) -> None:
    artifact, _coordinate = _prepared_artifact(tmp_path)
    assert artifact.sessions_path is not None
    assert artifact.shard_path is not None
    assert artifact.sessions_seal is not None

    # The row count and SQL seal remain unchanged; the closed-file digest
    # still catches a rewritten message if stat evidence is made to match.
    os.chmod(artifact.sessions_path, 0o600)
    with sqlite3.connect(artifact.sessions_path) as conn:
        conn.execute(
            "UPDATE prepared_message SET message_json = replace(message_json, ?, ?)",
            ("A neutral prompt", "A different text"),
        )
    changed = artifact.sessions_path.stat()
    forged_stat = replace(
        artifact.sessions_seal,
        device=changed.st_dev,
        inode=changed.st_ino,
        size=changed.st_size,
        mtime_ns=changed.st_mtime_ns,
        ctime_ns=changed.st_ctime_ns,
    )
    with pytest.raises(ValueError, match="content changed"):
        replace(artifact, sessions_seal=forged_stat).verify_files(full=True)

    replacement = tmp_path / "replacement.db"
    shutil.copyfile(artifact.shard_path, replacement)
    os.replace(replacement, artifact.shard_path)
    with pytest.raises(ValueError, match="identity changed"):
        artifact.verify_files(full=False)


def _claude_document(session_id: str) -> dict[str, object]:
    return {
        "uuid": session_id,
        "name": session_id,
        "chat_messages": [
            {"uuid": "repeated", "sender": "human", "text": "First neutral prompt"},
            {"uuid": "repeated", "sender": "assistant", "text": "Second neutral answer"},
        ],
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_bundle_worker_stream_preserves_parser_and_duplicate_identity_rows(tmp_path: Path, wrapped: bool) -> None:
    documents = [_claude_document("one"), _claude_document("two")]
    payload: object = {"sessions": documents} if wrapped else documents
    source = tmp_path / "claude-sessions.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    expected = parse_payload(
        Provider.CLAUDE_AI, list(_iter_json_stream(BytesIO(source.read_bytes()), source.name)), "fallback"
    )
    assert [message.provider_message_id for message in expected[0].messages] == ["repeated", "repeated"]
    for session in expected:
        session.content_hash = session_content_hash(session)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())
    assert [(session.provider_session_id, session.content_hash) for session in actual] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def test_bundle_worker_does_not_construct_a_whole_document_record_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "many.json"
    with source.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index in range(300):
            if index:
                handle.write(",")
            handle.write(json.dumps(_claude_document(f"session-{index}")))
        handle.write("]")

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-document decode or parse was used")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert sum(1 for _ in artifact.iter_sessions()) == 300


@pytest.mark.parametrize("provider", [Provider.DRIVE, Provider.GEMINI, Provider.UNKNOWN])
def test_generic_single_object_stream_matches_parser_with_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    record = {
        "id": "generic-session",
        "name": "Neutral session",
        "createdAt": "2025-01-01T00:00:00Z",
        "messages": [
            *({"id": "repeated", "role": "user", "text": f"Neutral prompt {index}"} for index in range(400)),
            {"id": 1e20, "role": "assistant", "text": "Numeric ID answer", "timestamp": 1.25},
            {"id": 10**30, "role": "user", "text": "Large integer ID prompt"},
        ],
    }
    source = tmp_path / "session.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    [expected] = parse_payload(provider, record, "fallback")
    expected.content_hash = session_content_hash(expected)
    expected_shard = prepare_session_shard(tmp_path / "expected", [expected])

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("generic object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.decoder_json.json.load", refuse_whole_document)
    decoded = 0
    first_appended_after: int | None = None
    original_items = ijson.items
    original_append = SqliteMessageSink.append

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        for item in original_items(*args, **kwargs):
            decoded += 1
            yield item

    def tracked_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal first_appended_after
        if first_appended_after is None:
            first_appended_after = decoded
        original_append(self, value)

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(SqliteMessageSink, "append", tracked_append)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        provider.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert first_appended_after == 1
    [actual] = artifact.iter_sessions()
    assert (actual.provider_session_id, actual.title, actual.created_at, actual.content_hash) == (
        expected.provider_session_id,
        expected.title,
        expected.created_at,
        expected.content_hash,
    )
    assert [message.provider_message_id for message in actual.messages] == [
        *(["repeated"] * 400),
        "1e+20",
        str(10**30),
    ]
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def test_generic_single_object_stream_discards_corrupt_suffix(tmp_path: Path) -> None:
    source = tmp_path / "damaged.json"
    source.write_text(
        '{"id":"generic-session","messages":[{"role":"user","text":"A neutral prompt"}]} trailing',
        encoding="utf-8",
    )
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_grok_single_object_streams_responses_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses: list[dict[str, object]] = [
        {"response": {"sender": "human" if index % 2 == 0 else "grok", "message": f"Turn {index}"}}
        for index in range(350)
    ]
    responses.extend([dict(responses[-1]), {"sender": "human", "message": ""}])
    record = {
        "conversations": [
            {
                "conversation": {"title": "First", "create_time": 1712000000, "ignored": {"values": list(range(1000))}},
                "responses": responses,
            },
            {"responses": responses},
            {"responses": [{"sender": "human", "message": "After metadata"}], "conversation": {"title": "Last"}},
        ]
    }
    source = tmp_path / "grok.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    expected = parse_payload(Provider.GROK, record, "fallback")
    for session in expected:
        session.content_hash = session_content_hash(session)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Grok object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    decoded = 0
    first_append_after: int | None = None
    original_events = iter_grok_export_events
    original_append = SqliteMessageSink.append

    def tracked_events(handle: IO[bytes], *, include_item: Callable[[int], bool] | None = None) -> object:
        nonlocal decoded
        for event, value in original_events(handle, include_item=include_item):
            if event == "response":
                decoded += 1
            yield event, value

    def tracked_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal first_append_after
        if first_append_after is None:
            first_append_after = decoded
        original_append(self, value)

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_grok_export_events", tracked_events)
    monkeypatch.setattr(SqliteMessageSink, "append", tracked_append)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert first_append_after == 1
    actual = list(artifact.iter_sessions())
    assert [(session.provider_session_id, session.content_hash) for session in actual] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as prepared:
        assert prepared.execute("SELECT COUNT(*) FROM prepared_message").fetchone()[0] == sum(
            len(session.messages) for session in expected
        )
    artifact.discard()


def test_grok_single_object_corrupt_suffix_leaves_no_artifact(tmp_path: Path) -> None:
    source = tmp_path / "damaged-grok.json"
    source.write_text(
        '{"conversations":[{"conversation":{"title":"T"},"responses":[{"sender":"human","message":"Hi"}]}]} trailing',
        encoding="utf-8",
    )
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert list(directory.glob("*.db")) == []


def test_grok_empty_conversation_keeps_direct_parse_session(tmp_path: Path) -> None:
    record = {"conversations": [{"conversation": {"title": "Empty"}, "responses": []}]}
    source = tmp_path / "empty-grok.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered is False
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.GROK, record, "fallback")
    assert (actual.provider_session_id, actual.title, list(actual.messages)) == (
        expected.provider_session_id,
        expected.title,
        expected.messages,
    )
    artifact.discard()


def test_grok_single_object_changed_during_stream_defers_and_discards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "changing-grok.json"
    source.write_text(
        json.dumps(
            {"conversations": [{"conversation": {"title": "T"}, "responses": [{"sender": "human", "message": "Hi"}]}]}
        ),
        encoding="utf-8",
    )
    original_events = iter_grok_export_events

    def changing_events(handle: IO[bytes], *, include_item: Callable[[int], bool] | None = None) -> object:
        for event, value in original_events(handle, include_item=include_item):
            yield event, value
            if event == "response":
                with source.open("a", encoding="utf-8") as writer:
                    writer.write(" ")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_grok_export_events", changing_events)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.deferred is True
    assert artifact.error is not None
    assert list(directory.glob("*.db")) == []


def test_grok_future_wire_type_keeps_parser_admission_event(tmp_path: Path) -> None:
    record = {
        "conversations": [
            {
                "conversation": {"title": "T"},
                "responses": [{"sender": "human", "message": "Hi", "type": "future_response"}],
            }
        ]
    }
    source = tmp_path / "future-grok.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.GROK, record, "fallback")
    assert list(actual.session_events) == expected.session_events
    assert [event.event_type for event in actual.session_events] == ["grok_unknown_input"]
    artifact.discard()


def test_chatgpt_bundle_worker_keeps_original_positions_after_skipped_siblings(tmp_path: Path) -> None:
    records = [
        {"unrelated": "sibling"},
        ChatGPTExportBuilder("conversation-1").add_node("user", "First neutral prompt").build(),
        {"mapping": {"bad": {"id": "bad"}}},
        ChatGPTExportBuilder("conversation-2").add_node("assistant", "Second neutral answer").build(),
    ]
    source = tmp_path / "conversations-000.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    expected = parse_payload(
        Provider.CHATGPT, list(_iter_json_stream(BytesIO(source.read_bytes()), source.name)), "fallback"
    )
    expected = require_positive_conversational_evidence(expected, provider=Provider.CHATGPT, source_path=str(source))
    for session in expected:
        session.content_hash = session_content_hash(session)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]


def test_whole_json_preparation_transforms_and_indexes_each_session_without_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        ChatGPTExportBuilder(f"conversation-{index}").add_node("user", f"Neutral prompt {index}").build()
        for index in range(80)
    ]
    source = tmp_path / "conversations.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    transformed: list[str] = []

    def transform(session: ParsedSession) -> ParsedSession:
        transformed.append(session.provider_session_id)
        return session

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_records=lambda items: items,
        prepare_session=transform,
    )
    assert artifact.error is None
    assert len(transformed) == 80
    sequence = artifact.session_sequence()

    def forbidden_iteration(_self: PreparedJsonl) -> object:
        raise AssertionError("keyed sibling lookup must not materialize or scan the session cohort")

    monkeypatch.setattr(PreparedJsonl, "iter_sessions", forbidden_iteration)
    session_id = f"{origin_from_provider(Provider.CHATGPT).value}:conversation-73"
    assert sequence.by_session_id(session_id).provider_session_id == "conversation-73"


def test_retained_top_level_chatgpt_object_keeps_fallback_and_sidecar_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.sources.revision_backfill as revision_backfill
    from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex

    record = ChatGPTExportBuilder("single-object").add_node("user", "A neutral prompt").build()
    record.pop("create_time", None)
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    for node in mapping.values():
        assert isinstance(node, dict)
        message = node.get("message")
        if isinstance(message, dict):
            message.pop("create_time", None)
            message["content"] = {
                "content_type": "multimodal_text",
                "parts": [
                    {"content_type": "text", "text": "A neutral prompt"},
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": "file-service://file-sidecar",
                        "width": 1,
                        "height": 1,
                        "size_bytes": 1,
                    },
                ],
            }
    payload = json.dumps(record).encode("utf-8")
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(payload)
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)

    evidence_providers: list[Provider] = []
    asset_index = ChatGPTAssetIndex.build(
        library_files_payload=[],
        asset_file_names_payload={"file-sidecar.dat": "resolved-sidecar.png"},
    )

    def sidecar_evidence(**kwargs: object) -> dict[str, object]:
        provider = kwargs["provider"]
        assert isinstance(provider, Provider)
        evidence_providers.append(provider)
        return {"chatgpt_asset_index": asset_index}

    monkeypatch.setattr(revision_backfill, "_retained_enrichment_sidecar_data", sidecar_evidence)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-raw",
        Provider.CHATGPT.value,
        blob_hash,
        str(tmp_path / "snapshot.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        "2025-01-02T03:04:05Z",
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered is True
    sessions = list(artifact.session_sequence())
    assert len(sessions) == 1
    assert sessions[0].created_at == "2025-01-02T03:04:05+00:00"
    assert sessions[0].updated_at == "2025-01-02T03:04:05+00:00"
    assert len(sessions[0].attachments) == 1
    assert sessions[0].attachments[0].name == "resolved-sidecar.png"
    assert evidence_providers == [Provider.CHATGPT]
    assert artifact.enrichment_digest is not None
    artifact.discard()


def test_bundle_worker_discards_partial_artifact_on_corrupt_suffix(tmp_path: Path) -> None:
    source = tmp_path / "damaged.json"
    source.write_text("[" + json.dumps(_claude_document("first")) + ", {broken}]", encoding="utf-8")
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_singleton_chatgpt_array_keeps_existing_parse_identity(tmp_path: Path) -> None:
    source = tmp_path / "one.json"
    source.write_text(
        json.dumps([ChatGPTExportBuilder("one").add_node("user", "A neutral prompt").build()]),
        encoding="utf-8",
    )
    expected = parse_payload(
        Provider.CHATGPT, list(_iter_json_stream(BytesIO(source.read_bytes()), source.name)), "fallback"
    )
    for session in expected:
        session.content_hash = session_content_hash(session)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
