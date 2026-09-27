"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO

import ijson
import pytest

from polylogue.core.enums import Provider, Role
from polylogue.core.json import JSONValue
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import iter_grok_export_events
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import (
    _ACTIVE_PARENT_LOOKUP_SQL,
    ChatGPTNodeMapping,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    read_chatgpt_mapping_object,
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
    scratch_root = tmp_path / "prepared"
    attempt_directory = scratch_root / "attempt-generic"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        provider.value,
        "fallback",
        is_stream=False,
        shard_directory=str(scratch_root),
        attempt_directory=attempt_directory,
    )
    assert artifact.error is None
    assert first_appended_after == 1
    assert artifact.sessions_path is not None and artifact.sessions_path.parent == attempt_directory
    assert artifact.shard_path is not None and artifact.shard_path.parent == attempt_directory
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
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )
    artifact.discard()
    assert not attempt_directory.exists()


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


def test_hermes_snapshot_stream_matches_parser_and_spills_before_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "session_id": "neutral-hermes",
        "model": "neutral-model",
        "platform": "linux",
        "session_start": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:02:00Z",
        "system_prompt": "Be concise.",
        "message_count": 300,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": f"Neutral prompt {index}", "tool_call_id": "repeat"} for index in range(300)
        ]
        + [
            {
                "role": "assistant",
                "content": "done",
                "tool_call_id": "repeat",
                "finish_reason": "stop",
                "codex_reasoning_items": [{"type": "reasoning", "text": "A neutral thought"}],
                "tool_calls": [
                    {
                        "id": "lookup-1",
                        "function": {"name": "lookup", "arguments": "{}"},
                        "extra_content": {"trace": "synthetic"},
                    }
                ],
            },
            {"role": "assistant", "content": "", "type": "future_turn"},
        ],
    }
    source = tmp_path / "session_neutral.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    [expected] = parse_payload(Provider.HERMES, record, "fallback", source_path=str(source))
    expected.content_hash = session_content_hash(expected)
    expected_shard = prepare_session_shard(tmp_path / "baseline", [expected])

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Hermes snapshot decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    decoded = 0
    first_written_after: int | None = None
    original_items = ijson.items
    original_append = SqliteMessageSink.append

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        for item in original_items(*args, **kwargs):
            if len(args) > 1 and args[1] == "messages.item":
                decoded += 1
            yield item

    def tracked_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal first_written_after
        if first_written_after is None and decoded:
            first_written_after = decoded
        original_append(self, value)

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(SqliteMessageSink, "append", tracked_append)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert first_written_after == 1
    [actual] = artifact.iter_sessions()
    assert [message.provider_message_id for message in actual.messages] == [
        message.provider_message_id for message in expected.messages
    ]
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.unit_accounting == expected.unit_accounting
    assert actual.content_hash == expected.content_hash
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )
    artifact.discard()


def test_hermes_snapshot_stream_discards_corrupt_suffix(tmp_path: Path) -> None:
    source = tmp_path / "session_damaged.json"
    source.write_text(
        '{"session_id":"neutral","platform":"linux","messages":[{"role":"user","content":"hello"}]} trailing',
        encoding="utf-8",
    )
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_hermes_snapshot_retained_callbacks_keep_stream_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "session_id": "retained-hermes",
        "platform": "linux",
        "messages": [{"role": "user", "content": "Neutral prompt"}],
    }
    source = tmp_path / "session_retained.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Hermes snapshot decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    witnesses: list[tuple[dict[str, JSONValue], list[JSONValue]]] = []

    def classify(envelope: dict[str, JSONValue], messages: Sequence[JSONValue]) -> bool:
        witnesses.append((envelope, list(messages)))
        return True

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
        prepare_records=lambda records: records,
        classify_hermes_object=classify,
    )
    assert artifact.error is None
    assert witnesses == [({"session_id": "retained-hermes", "platform": "linux"}, record["messages"])]
    assert [session.title for session in artifact.iter_sessions()] == ["retained-hermes"]
    artifact.discard()


def test_hermes_snapshot_stream_refuses_source_mutation_after_spill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "session_mutating.json"
    source.write_text(
        json.dumps({"session_id": "neutral", "platform": "linux", "messages": [{"role": "user", "content": "one"}]}),
        encoding="utf-8",
    )
    original_append = SqliteMessageSink.append
    changed = False

    def mutate_after_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal changed
        original_append(self, value)
        if not changed:
            changed = True
            source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")

    monkeypatch.setattr(SqliteMessageSink, "append", mutate_after_append)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert changed
    assert artifact.deferred
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_generic_retained_callbacks_keep_bounded_message_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "session.json"
    record = {
        "id": "retained-generic",
        "messages": [
            {"id": f"message-{index}", "role": "user", "text": f"Neutral prompt {index}"} for index in range(100)
        ],
    }
    source.write_text(json.dumps(record), encoding="utf-8")
    expected = parse_payload(Provider.DRIVE, record, "fallback")[0]

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained generic object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    first_written_after: int | None = None
    decoded = 0
    original_items = ijson.items
    original_append = SqliteMessageSink.append

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        for item in original_items(*args, **kwargs):
            decoded += 1
            yield item

    def tracked_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal first_written_after
        if first_written_after is None:
            first_written_after = decoded
        original_append(self, value)

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(SqliteMessageSink, "append", tracked_append)
    finalized: list[str] = []

    def finalize(sessions: list[ParsedSession]) -> list[ParsedSession]:
        finalized.extend(session.provider_session_id for session in sessions)
        return sessions

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_records=lambda records: records,
        prepare_sessions=finalize,
        classify_generic_object=lambda _envelope, messages: bool(messages),
    )
    assert artifact.error is None
    assert first_written_after == 65
    assert finalized == ["retained-generic"]
    [actual] = artifact.iter_sessions()
    assert (actual.provider_session_id, actual.content_hash) == (
        expected.provider_session_id,
        session_content_hash(expected),
    )
    assert [message.text for message in actual.messages] == [message.text for message in expected.messages]
    artifact.discard()


def test_retained_generic_object_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    record = {
        "id": "retained-drive",
        "messages": [{"id": "repeated", "role": "user", "text": f"Neutral prompt {index}"} for index in range(120)],
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode())
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained replay decoded the complete object")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-raw",
        Provider.DRIVE.value,
        blob_hash,
        str(tmp_path / "session.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        "2025-01-02T03:04:05Z",
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.DRIVE, record, "session")
    assert actual.provider_session_id == expected.provider_session_id
    assert [message.provider_message_id for message in actual.messages] == [
        message.provider_message_id for message in expected.messages
    ]
    assert [message.text for message in actual.messages] == [message.text for message in expected.messages]
    assert actual.created_at == "2025-01-02T03:04:05+00:00"
    artifact.discard()


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


def test_retained_grok_streams_responses_with_replay_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources import revision_backfill

    responses = [
        {"response": {"sender": "human" if index % 2 == 0 else "grok", "message": f"Turn {index}"}}
        for index in range(350)
    ]
    responses.append(dict(responses[-1]))
    record = {
        "conversations": [
            {"conversation": {"title": "Retained", "create_time": 1712000000}, "responses": responses},
            {"conversation": {"title": "Empty"}, "responses": []},
            {"responses": responses},
        ],
        "id": "not-a-Beads-interaction",
        "kind": "export",
        "created_at": "2025-01-02T03:04:05Z",
        "issue_id": "not-an-issue",
        "extra": "not-a-Beads-map",
        "type": "export",
        "version": "v1",
        "mapping": "metadata",
    }
    fallback_timestamp = "2025-01-02T03:04:05Z"
    source_path = str(tmp_path / "prod-grok-backend.json")
    expected = require_positive_conversational_evidence(
        parse_payload(Provider.GROK, record, Path(source_path).stem),
        provider=Provider.GROK,
        source_path=source_path,
    )
    expected = [normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp) for session in expected]
    for session in expected:
        session.content_hash = session_content_hash(session)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode("utf-8"))
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Grok object decoded as a whole document")

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
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-raw",
        Provider.GROK.value,
        blob_hash,
        source_path,
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        fallback_timestamp,
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered is True
    assert first_append_after == 1
    actual = list(artifact.iter_sessions())
    assert [(session.provider_session_id, session.content_hash) for session in actual] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    assert [session.created_at for session in actual] == [session.created_at for session in expected]
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as prepared:
        assert prepared.execute("SELECT COUNT(*) FROM prepared_message").fetchone()[0] == len(responses)
    artifact.discard()

    sidecar = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-sidecar",
        Provider.GROK.value,
        blob_hash,
        str(tmp_path / "agent-neutral.meta.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "sidecar-prepared"),
        fallback_timestamp,
    )
    assert sidecar.error is None
    assert list(sidecar.iter_sessions()) == []
    sidecar.discard()

    analysis_artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-analysis-path",
        Provider.GROK.value,
        blob_hash,
        str(tmp_path / "analysis" / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "analysis-prepared"),
        fallback_timestamp,
    )
    assert analysis_artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in analysis_artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    analysis_artifact.discard()

    beads_record = {**record, "extra": {}}
    beads_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(beads_record).encode("utf-8"))
    beads_artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-beads-overlap",
        Provider.GROK.value,
        beads_hash,
        source_path,
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "beads-prepared"),
        fallback_timestamp,
    )
    assert beads_artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in beads_artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    beads_artifact.discard()

    beads_analysis_artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-beads-analysis-path",
        Provider.GROK.value,
        beads_hash,
        str(tmp_path / "analysis" / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "beads-analysis-prepared"),
        fallback_timestamp,
    )
    assert beads_analysis_artifact.error is None
    assert list(beads_analysis_artifact.iter_sessions()) == []
    beads_analysis_artifact.discard()

    messages_record = {key: value for key, value in record.items() if key not in {"type", "version"}}
    messages_record["messages"] = [{"role": "user", "content": "Root metadata"}]
    messages_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(messages_record).encode("utf-8"))
    messages_artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-grok-messages-overlap",
        Provider.GROK.value,
        messages_hash,
        str(tmp_path / "analysis" / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "messages-prepared"),
        fallback_timestamp,
    )
    assert messages_artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in messages_artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    messages_artifact.discard()


def test_retained_grok_corrupt_suffix_leaves_no_publishable_artifact(tmp_path: Path) -> None:
    from polylogue.sources import revision_backfill

    blob_root = tmp_path / "blob"
    payload = b'{"conversations":[{"conversation":{},"responses":[{"sender":"human","message":"Hi"}]}]} trailing'
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(payload)
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
    directory = tmp_path / "prepared"
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-corrupt-grok",
        Provider.GROK.value,
        blob_hash,
        str(tmp_path / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(directory),
        None,
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert artifact.shard_path is None
    assert list(directory.glob("*.db")) == []


def test_retained_grok_future_wire_keeps_parser_admission_event(tmp_path: Path) -> None:
    from polylogue.sources import revision_backfill

    record = {
        "conversations": [
            {
                "conversation": {"title": "Future"},
                "responses": [{"sender": "human", "message": "Hi", "type": "future_response"}],
            }
        ]
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode("utf-8"))
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-future-grok",
        Provider.GROK.value,
        blob_hash,
        str(tmp_path / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        None,
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered is False
    [session] = artifact.iter_sessions()
    assert [event.event_type for event in session.session_events] == ["grok_unknown_input"]
    artifact.discard()


def test_retained_grok_hook_overlap_keeps_artifact_taxonomy(tmp_path: Path) -> None:
    from polylogue.sources import revision_backfill

    record = {
        "conversations": [
            {"conversation": {"title": "Ambiguous"}, "responses": [{"sender": "human", "message": "Hi"}]}
        ],
        "event_type": "SessionStart",
        "session_id": "hook-session",
        "timestamp": "2025-01-02T03:04:05Z",
        "provider": "codex",
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode("utf-8"))
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-ambiguous-grok",
        Provider.GROK.value,
        blob_hash,
        str(tmp_path / "prod-grok-backend.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        None,
    )
    assert artifact.error is None
    assert list(artifact.iter_sessions()) == []
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


def test_chatgpt_mapping_object_spills_before_eof_with_bounded_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = ChatGPTExportBuilder("bounded-mapping").add_node("user", "A neutral prompt").build()
    source_mapping = record["mapping"]
    assert isinstance(source_mapping, dict)
    node = next(iter(source_mapping.values()))
    record["mapping"] = {f"node-{index}": node for index in range(3000)}
    source = json.dumps(record).encode()

    class BoundedReader(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= 128 * 1024
            return super().read(size)

    handle = BoundedReader(source)
    first_write: list[int] = []
    original_put = ChatGPTNodeMapping.put

    def observe_put(mapping: ChatGPTNodeMapping, key: str, value: object, ordinal: int) -> None:
        if not first_write:
            first_write.append(handle.tell())
        original_put(mapping, key, value, ordinal)

    monkeypatch.setattr(ChatGPTNodeMapping, "put", observe_put)
    with sqlite3.connect(tmp_path / "nodes.db") as conn:
        result = read_chatgpt_mapping_object(handle, conn)
        assert result is not None
        assert len(result[1]) == 3000
        assert conn.execute("SELECT COUNT(*) FROM chatgpt_node").fetchone()[0] == 3000
    assert first_write and first_write[0] < len(source)


def test_chatgpt_mapping_object_preparation_matches_parser_and_duplicate_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = (
        ChatGPTExportBuilder("mapping-parity")
        .add_node("user", "A neutral prompt")
        .add_node("assistant", "A neutral answer")
        .build()
    )
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    first_key = next(iter(mapping))
    duplicate = json.dumps(mapping[first_key])
    source = tmp_path / "chatgpt.json"
    encoded = json.dumps(record)
    marker = json.dumps(first_key) + ": " + duplicate
    encoded = encoded.replace(marker, marker + ", " + marker, 1)
    source.write_text(encoded, encoding="utf-8")
    expected = parse_payload(Provider.CHATGPT, [json.loads(encoded)], "fallback")[0]

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ChatGPT object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = artifact.load_sessions()[0]
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert [attachment.model_dump(mode="json") for attachment in actual.attachments] == [
        attachment.model_dump(mode="json") for attachment in expected.attachments
    ]
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_native_object_preparation_preserves_complete_parser_output(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/chatgpt/native-conversation-v1.json")
    source = tmp_path / "conversation.json"
    source.write_bytes(fixture.read_bytes())
    expected = parse_payload(Provider.CHATGPT, [json.loads(source.read_bytes())], "fallback")[0]
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = artifact.load_sessions()[0]
    collected = actual.model_copy(
        update={"messages": list(actual.messages), "session_events": list(actual.session_events)}
    )
    assert collected.model_dump(mode="json", exclude={"content_hash"}) == expected.model_dump(
        mode="json", exclude={"content_hash"}
    )
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_mapping_object_corrupt_suffix_discards_scratch(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("mapping-corrupt").add_node("user", "A neutral prompt").build()
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record) + " trailing", encoding="utf-8")
    directory = tmp_path / "scratch"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


@pytest.mark.parametrize("failure", ["mutation", "parser"])
def test_chatgpt_mapping_object_failure_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    record = ChatGPTExportBuilder("mapping-failure").add_node("user", "A neutral prompt").build()
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    directory = tmp_path / "scratch"
    if failure == "mutation":
        original = read_chatgpt_mapping_object

        def mutate_after_read(handle: BinaryIO, conn: sqlite3.Connection) -> object:
            result = original(handle, conn)
            source.write_text(json.dumps(record) + " ", encoding="utf-8")
            return result

        monkeypatch.setattr("polylogue.sources.prepared_jsonl.read_chatgpt_mapping_object", mutate_after_read)
    else:

        def fail_parser(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("synthetic parse worker failure")

        monkeypatch.setattr("polylogue.sources.prepared_jsonl.chatgpt.parse", fail_parser)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


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


def test_assigned_attempt_discard_preserves_sibling_attempt_and_sentinel(tmp_path: Path) -> None:
    source = tmp_path / "conversation.json"
    source.write_text(
        json.dumps([ChatGPTExportBuilder("attempt-carrier").add_node("user", "Neutral prompt").build()]),
        encoding="utf-8",
    )
    scratch_root = tmp_path / "parse-shards"
    attempt = scratch_root / "attempt-owned"
    sibling = scratch_root / "attempt-sibling"
    sibling.mkdir(parents=True)
    sentinel = sibling / "keep.txt"
    sentinel.write_text("unrelated carrier")

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(scratch_root),
        attempt_directory=attempt,
    )
    assert artifact.error is None
    assert artifact.attempt_directory == attempt
    assert artifact.sessions_path is not None and artifact.sessions_path.parent == attempt
    assert artifact.shard_path is not None and artifact.shard_path.parent == attempt
    artifact.discard()

    assert not attempt.exists()
    assert sentinel.read_text() == "unrelated carrier"


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
