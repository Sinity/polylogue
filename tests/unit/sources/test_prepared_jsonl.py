"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Mapping, Sequence
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
from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
from polylogue.sources.decoder_json import claude_design_object_envelope, iter_grok_export_events
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence
from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver
from polylogue.sources.live.tool_result_sidecars import _MAX_SIDECAR_FILE_BYTES
from polylogue.sources.parsers import chatgpt, local_agent
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import (
    _ACTIVE_PARENT_LOOKUP_SQL,
    ChatGPTNodeMapping,
    SqliteAttachmentSink,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    read_chatgpt_mapping_object,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarFile, RetainedSidecarScope
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
    assert isinstance(restored[0].attachments, SqliteAttachmentSink)
    assert restored[0].messages[0].owner_coordinate == coordinate
    assert restored[0].messages[0].parent_message_position == 0
    assert restored[0].session_events[0].boundary_message_position == 0
    assert restored[0].attachments[0].owner_coordinate == coordinate
    assert restored[0].attachments[0].inline_bytes == b"\x00\xff"
    assert restored[0].attachments[0].precomputed_blob == ("a" * 64, 2)
    first_attachment = restored[0].attachments[0]
    second_attachment = restored[0].attachments[0]
    assert first_attachment is not second_attachment
    assert first_attachment.acquisition_key == second_attachment.acquisition_key
    assert session_content_hash(restored[0]) == restored[0].content_hash
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        metadata = json.loads(conn.execute("SELECT metadata_json FROM prepared_session").fetchone()[0])
        assert "attachments" not in metadata
        assert conn.execute("SELECT COUNT(*) FROM prepared_attachment").fetchone()[0] == 1

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


def test_claude_design_object_stream_matches_direct_parser_and_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "uuid": "design-session",
        "project": {"uuid": "neutral-project"},
        "title": "Neutral design",
        "created_at": "2026-01-01T00:00:00Z",
        "messages": [
            {
                "uuid": f"u-{index}",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": f"Neutral prompt {index}",
                    "authorAccountUuid": "neutral-account",
                    "authorName": "Author",
                    "timestamp": f"2026-01-01T00:00:{index % 60:02d}Z",
                    "attachments": (
                        [{"id": "neutral-attachment", "name": "brief.txt", "type": "text", "content": "Neutral brief"}]
                        if index == 0
                        else []
                    ),
                },
            }
            for index in range(300)
        ]
        + [
            {
                "uuid": "u-0",
                "role": "assistant",
                "content": {
                    "role": "assistant",
                    "content": "",
                    "contentBlocks": [{"type": "text", "text": "Neutral answer"}],
                    "turnChanges": {"reason": "complete"},
                },
            }
        ],
    }
    source = tmp_path / "design-chat.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    [expected] = parse_payload(Provider.CLAUDE_DESIGN, payload, "fallback")
    expected.content_hash = session_content_hash(expected)
    expected_shard = prepare_session_shard(tmp_path / "expected", [expected])

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-document decode or parse was used")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    decoded = 0
    first_written_after: int | None = None
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
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_DESIGN.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        classify_claude_design_object=lambda envelope, sample: envelope["project"] is None and len(sample) == 64,
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    assert first_written_after == 65
    [actual] = artifact.iter_sessions()
    assert isinstance(actual.messages, SqliteMessageSink)
    assert isinstance(actual.session_events, SqliteSessionEventSink)
    assert actual.content_hash == expected.content_hash
    assert actual.unit_accounting == expected.unit_accounting
    assert (
        actual.model_copy(
            update={"messages": list(actual.messages), "session_events": list(actual.session_events)}
        ).model_dump()
        == expected.model_dump()
    )
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(artifact.shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def test_claude_design_object_probe_keeps_future_wire_types_on_admission_route(tmp_path: Path) -> None:
    content = {"role": "user", "content": "Neutral", "type": "future_turn"}
    payload = {
        "uuid": "design-session",
        "project": {},
        "messages": [{"role": "user", "content": content}],
    }
    assert claude_design_object_envelope(BytesIO(json.dumps(payload).encode())) is None
    source = tmp_path / "future-design.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_DESIGN.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.CLAUDE_DESIGN, payload, "fallback")
    assert list(actual.session_events) == expected.session_events
    assert actual.unit_accounting == expected.unit_accounting
    content["type"] = "known_turn"
    assert claude_design_object_envelope(BytesIO(json.dumps(payload).encode())) is not None
    payload.update(
        event_type="PreToolUse", session_id="hook-session", timestamp="2026-01-01T00:00:00Z", provider="codex"
    )
    assert claude_design_object_envelope(BytesIO(json.dumps(payload).encode())) is None


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


def test_hermes_snapshot_retained_callbacks_discard_corrupt_suffix(tmp_path: Path) -> None:
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
        prepare_sessions=lambda sessions: sessions,
        prepare_records=lambda records: records,
        classify_hermes_object=lambda _envelope, _messages: True,
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


def test_retained_hermes_extracted_transcript_is_not_a_session(tmp_path: Path) -> None:
    from polylogue.sources import revision_backfill

    record = {
        "session_id": "copied-extract",
        "platform": "linux",
        "transcript": "source.json",
        "content": "Copied text",
        "messages": [{"role": "user", "content": "Neutral source prompt"}],
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode())
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-extract",
        Provider.HERMES.value,
        blob_hash,
        str(tmp_path / "sessions" / "session_extract.json"),
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


def test_hermes_snapshot_stream_uses_parser_future_type_priority(tmp_path: Path) -> None:
    record = {
        "session_id": "future-priority",
        "platform": "linux",
        "nested": {"type": "future_inner"},
        "messages": [{"role": "user", "content": "Neutral prompt"}],
        "type": "future_outer",
    }
    source = tmp_path / "session_future.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    [expected] = parse_payload(Provider.HERMES, record, "fallback", source_path=str(source))
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.content_hash == session_content_hash(expected)
    assert session_content_hash(actual) == session_content_hash(expected)
    artifact.discard()


@pytest.mark.parametrize("marker", ["atif", "state", "verification"])
def test_hermes_snapshot_shortcut_preserves_higher_priority_dispatch(tmp_path: Path, marker: str) -> None:
    record: dict[str, object] = {
        "session_id": "priority",
        "platform": "linux",
        "messages": [{"role": "user", "content": "Neutral prompt"}],
    }
    if marker == "atif":
        record.update(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [{"source": "agent", "message": "Observer step"}],
            }
        )
    else:
        record["polylogue_artifact"] = "hermes_state_db" if marker == "state" else "hermes_verification_evidence_db"
        record["state_db_path" if marker == "state" else "verification_db_path"] = "missing.db"
    source = tmp_path / "session_priority.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    if marker == "atif":
        assert artifact.error is None
        [session] = artifact.iter_sessions()
        [expected] = parse_payload(Provider.HERMES, record, "fallback", source_path=str(source))
        assert session.provider_session_id == expected.provider_session_id
        assert any(event.event_type == "hermes_llm_request_span" for event in session.session_events)
    else:
        assert artifact.error is not None
        assert artifact.sessions_path is None
    artifact.discard()


def test_gemini_cli_object_spills_and_matches_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "sessionId": "process-1",
        "projectHash": "project-1",
        "kind": "subagent",
        "startTime": "2026-01-01T00:00:00Z",
        "messages": [
            {"id": "repeat", "type": "user", "content": "A neutral request", "timestamp": "2026-01-01T00:00:01Z"},
            {"id": "repeat", "type": "gemini", "content": "A neutral answer", "model": "gemini-test"},
            {"id": "repeat", "type": "gemini", "content": "Another answer", "tokens": {"input": 3}},
        ],
        "lastUpdated": "2026-01-01T00:00:04Z",
        "userMessageCount": 1,
        "hasUserOrAssistantMessage": True,
        "memoryScratchpad": {"workflowSummary": "Neutral summary"},
    }
    source = tmp_path / "session.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    [expected] = parse_payload(Provider.GEMINI_CLI, record, "fallback")
    original_items = ijson.items
    from polylogue.sources import prepared_jsonl

    original_append = prepared_jsonl._append_gemini_raw_message
    appended = 0
    observed_spill = False

    def tracked_append(conn: sqlite3.Connection, ordinal: int, item: object) -> None:
        nonlocal appended
        original_append(conn, ordinal, item)
        appended += 1

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal observed_spill
        for index, item in enumerate(original_items(*args, **kwargs)):
            if index == 1:
                assert appended == 1
                observed_spill = True
            yield item

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(prepared_jsonl, "_append_gemini_raw_message", tracked_append)
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl._iter_json_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("whole-object fallback")),
    )
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert observed_spill
    [actual] = artifact.iter_sessions()
    assert actual.content_hash == session_content_hash(expected)
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.unit_accounting == expected.unit_accounting
    assert [message.is_active_leaf for message in actual.messages] == [False, False, True]
    artifact.discard()


@pytest.mark.parametrize("with_sidecar_scope", [False, True])
def test_gemini_cli_object_corrupt_suffix_discards_scratch(tmp_path: Path, with_sidecar_scope: bool) -> None:
    source = tmp_path / "project" / "chats" / "session.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        '{"sessionId":"process-1","kind":"chat","messages":[{"id":"m1","type":"user","content":"Hi"}]} trailing',
        encoding="utf-8",
    )
    if with_sidecar_scope:
        outputs = tmp_path / "project" / "tool-outputs" / "session-process-1"
        outputs.mkdir(parents=True)
        (outputs / "run_1.txt").write_text("Full neutral output", encoding="utf-8")
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
        sidecar_resolver=FilesystemSidecarResolver() if with_sidecar_scope else None,
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_gemini_cli_object_source_change_defers_and_discards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "session.json"
    source.write_text(
        json.dumps({"sessionId": "process-1", "kind": "chat", "messages": [{"type": "user", "content": "Hi"}]}),
        encoding="utf-8",
    )
    from polylogue.sources import prepared_jsonl

    original_envelope = prepared_jsonl._gemini_cli_envelope

    def changing_envelope(handle: BinaryIO) -> dict[str, JSONValue] | None:
        envelope = original_envelope(handle)
        with source.open("a", encoding="utf-8") as writer:
            writer.write(" ")
        return envelope

    monkeypatch.setattr(prepared_jsonl, "_gemini_cli_envelope", changing_envelope)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.deferred
    assert artifact.error is not None
    assert list(directory.glob("*.db")) == []


def test_gemini_cli_object_keeps_sidecar_debt_on_existing_scope(tmp_path: Path) -> None:
    chats = tmp_path / "project" / "chats"
    chats.mkdir(parents=True)
    (tmp_path / "project" / "tool-outputs" / "session-process-1").mkdir(parents=True)
    source = chats / "session.json"
    record = {
        "sessionId": "process-1",
        "kind": "chat",
        "messages": [
            {
                "id": "m1",
                "type": "gemini",
                "toolCalls": [
                    {
                        "id": "tool-1",
                        "name": "run_shell_command",
                        "resultDisplay": "For full output see: tool-outputs/session-process-1/missing.txt",
                    }
                ],
            }
        ],
    }
    source.write_text(json.dumps(record), encoding="utf-8")
    resolver = FilesystemSidecarResolver()
    [expected] = parse_payload(
        Provider.GEMINI_CLI, record, "fallback", source_path=str(source), sidecar_resolver=resolver
    )
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        sidecar_resolver=resolver,
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert any(event.event_type == "gemini_cli_tool_output_sidecar" for event in actual.session_events)
    artifact.discard()


def _gemini_message_payloads(session: ParsedSession) -> list[dict[str, object]]:
    payloads = [message.model_dump(mode="json") for message in session.messages]
    for message in payloads:
        blocks = message["blocks"]
        assert isinstance(blocks, list)
        for block in blocks:
            # The sealed reader resolves the parser's nullable legacy outcome
            # from the tool status when it validates a persisted block.
            block.pop("tool_outcome", None)
    return payloads


def test_gemini_cli_sidecar_scope_streams_and_matches_object_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources import prepared_jsonl

    def tool(tool_id: str, output: str) -> dict[str, object]:
        return {
            "id": tool_id,
            "name": "run_shell_command",
            "status": "success",
            "result": [
                {"functionResponse": {"id": tool_id, "name": "run_shell_command", "response": {"output": output}}}
            ],
        }

    pointer = "For full output see: tool-outputs/session-process-1/"
    record = {
        "sessionId": "process-1",
        "kind": "chat",
        "startTime": "2026-01-01T00:00:00Z",
        "messages": [
            {"id": "m1", "type": "gemini", "toolCalls": [tool("run_1", pointer + "missing-alias.txt")]},
            {
                "id": "m2",
                "type": "gemini",
                "toolCalls": [
                    tool("run_1", "short"),
                    tool("run_1_long", "tiny"),
                    tool("fallback", pointer + "pointer.txt"),
                    tool("absent", pointer + "missing.txt"),
                    tool("oversize", "tiny"),
                    tool("unreadable", "tiny"),
                ],
            },
        ],
    }
    source = tmp_path / "project" / "chats" / "session.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record), encoding="utf-8")

    def entry(name: str, text: str, *, size: int | None = None, unreadable: bool = False) -> RetainedSidecarFile:
        def read() -> str:
            if unreadable:
                raise OSError("synthetic read failure")
            return text

        return RetainedSidecarFile(name, len(text) if size is None else size, 1_700_000_000_000, read)

    scope = RetainedSidecarScope(
        scope_key="retained-gemini",
        available=True,
        files=(
            entry("run_1.txt", "first complete output"),
            entry("run_1.txt", "second complete output"),
            entry("prefix_run_1_long_slug.txt", "long id output"),
            entry("pointer.txt", "pointer fallback output"),
            entry("oversize.txt", "not read", size=_MAX_SIDECAR_FILE_BYTES + 1),
            entry("unreadable.txt", "not read", unreadable=True),
        ),
    )

    class Resolver:
        def claude_code_scope(self, *_args: object) -> RetainedSidecarScope:
            return scope

        def gemini_cli_scope(self, *_args: object) -> RetainedSidecarScope:
            return scope

    resolver = Resolver()
    [expected] = parse_payload(
        Provider.GEMINI_CLI, record, "fallback", source_path=str(source), sidecar_resolver=resolver
    )
    appended = 0
    original_append = prepared_jsonl._append_gemini_raw_message
    original_items = ijson.items

    def tracked_append(conn: sqlite3.Connection, ordinal: int, item: object) -> None:
        nonlocal appended
        original_append(conn, ordinal, item)
        appended += 1

    def tracked_items(*args: object, **kwargs: object) -> object:
        for index, item in enumerate(original_items(*args, **kwargs)):
            if index == 1:
                assert appended == 1
            yield item

    monkeypatch.setattr(prepared_jsonl, "_append_gemini_raw_message", tracked_append)
    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(prepared_jsonl, "_iter_json_stream", lambda *_a, **_k: pytest.fail("whole-object fallback"))
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        sidecar_resolver=resolver,
    )
    assert artifact.error is None
    assert appended == 2
    [actual] = artifact.iter_sessions()
    assert actual.content_hash == session_content_hash(expected)
    assert _gemini_message_payloads(actual) == _gemini_message_payloads(expected)
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    sidecar_events = [
        event.payload for event in actual.session_events if event.event_type == "gemini_cli_tool_output_sidecar"
    ]
    assert [event["acquisition_status"] for event in sidecar_events] == [
        "matched",
        "matched",
        "matched",
        "matched",
        "debt",
        "debt",
        "debt",
    ]
    assert [(event["filename"], event["tool_use_id"]) for event in sidecar_events[:4]] == [
        ("pointer.txt", "fallback"),
        ("prefix_run_1_long_slug.txt", "run_1_long"),
        ("run_1.txt", "run_1"),
        ("run_1.txt", "run_1"),
    ]
    assert [event["reason"] for event in sidecar_events[4:]] == [
        "size_exceeded",
        "read_error:OSError",
        "expected_sidecar_not_retained",
    ]
    assert any(block.text == "second complete output" for message in actual.messages for block in message.blocks)
    artifact.discard()


def test_retained_gemini_sidecar_replay_uses_sealed_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources import revision_backfill
    from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

    record = {
        "sessionId": "retained-process",
        "kind": "chat",
        "startTime": "2026-01-01T00:00:00Z",
        "messages": [
            {"id": "user", "type": "user", "content": "Run a neutral command"},
            {
                "id": "answer",
                "type": "gemini",
                "toolCalls": [
                    {
                        "id": "run_1",
                        "name": "run_shell_command",
                        "status": "success",
                        "result": [
                            {
                                "functionResponse": {
                                    "id": "run_1",
                                    "name": "run_shell_command",
                                    "response": {
                                        "output": "For full output see: tool-outputs/session-retained-process/run_1.txt",
                                    },
                                }
                            }
                        ],
                    }
                ],
            },
        ],
    }
    source_path = str(tmp_path / "old-project" / "chats" / "session.json")
    scope = RetainedSidecarScope(
        scope_key="retained-only",
        available=True,
        files=(RetainedSidecarFile("run_1.txt", 19, None, lambda: "Retained full output"),),
    )
    monkeypatch.setattr(RetainedSidecarResolver, "gemini_cli_scope", lambda *_args: scope)
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode())
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl._iter_json_stream", lambda *_a, **_k: pytest.fail("whole-object replay")
    )
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "retained-gemini",
        Provider.GEMINI_CLI.value,
        blob_hash,
        source_path,
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        None,
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(
        Provider.GEMINI_CLI,
        record,
        "session",
        source_path=source_path,
        sidecar_resolver=RetainedSidecarResolver(tmp_path),
    )
    assert actual.provider_session_id == expected.provider_session_id
    assert [message.provider_message_id for message in actual.messages] == [
        message.provider_message_id for message in expected.messages
    ]
    assert _gemini_message_payloads(actual) == _gemini_message_payloads(expected)
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    artifact.discard()


def test_gemini_sidecar_join_failure_discards_unsealed_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources.prepared_message_sink import GeminiToolOutputIndex

    record = {
        "sessionId": "process-1",
        "kind": "chat",
        "messages": [{"id": f"m{index}", "type": "user", "content": "Neutral"} for index in range(2)],
    }
    source = tmp_path / "project" / "chats" / "session.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record), encoding="utf-8")
    outputs = tmp_path / "project" / "tool-outputs" / "session-process-1"
    outputs.mkdir(parents=True)
    original_observe = GeminiToolOutputIndex.observe
    observed = 0

    def broken_observe(self: GeminiToolOutputIndex, message: object) -> None:
        nonlocal observed
        original_observe(self, message)
        observed += 1
        if observed == 2:
            raise RuntimeError("synthetic join failure")

    monkeypatch.setattr(GeminiToolOutputIndex, "observe", broken_observe)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
        sidecar_resolver=FilesystemSidecarResolver(),
    )
    assert "synthetic join failure" in (artifact.error or "")
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_gemini_sidecar_source_mutation_discards_unsealed_scratch(tmp_path: Path) -> None:
    source = tmp_path / "project" / "chats" / "session.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "sessionId": "process-1",
                "kind": "chat",
                "messages": [
                    {
                        "id": "m1",
                        "type": "gemini",
                        "toolCalls": [
                            {
                                "id": "run_1",
                                "name": "run_shell_command",
                                "status": "success",
                                "result": [{"functionResponse": {"response": {"output": "short"}}}],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def changing_read() -> str:
        with source.open("a", encoding="utf-8") as writer:
            writer.write(" ")
        return "Full neutral output"

    scope = RetainedSidecarScope(
        scope_key="changed-source",
        available=True,
        files=(RetainedSidecarFile("run_1.txt", 19, None, changing_read),),
    )

    class Resolver:
        def claude_code_scope(self, *_args: object) -> RetainedSidecarScope:
            return scope

        def gemini_cli_scope(self, *_args: object) -> RetainedSidecarScope:
            return scope

    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
        sidecar_resolver=Resolver(),
    )
    assert artifact.deferred
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


def test_gemini_cli_object_preserves_future_wire_admission(tmp_path: Path) -> None:
    record = {
        "sessionId": "process-1",
        "kind": "chat",
        "messages": [
            {"id": "m1", "type": "user", "content": "A neutral request"},
            {"id": "m2", "type": "future_turn", "content": "A neutral future turn"},
        ],
    }
    source = tmp_path / "session.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    [expected] = parse_payload(Provider.GEMINI_CLI, record, "fallback")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.unit_accounting == expected.unit_accounting
    assert any(event.event_type == "gemini_cli_unknown_input" for event in actual.session_events)
    artifact.discard()


def test_gemini_cli_turnless_stub_accepts_sidecar_resolver(tmp_path: Path) -> None:
    stub: dict[str, JSONValue] = {"sessionId": "process-1", "projectHash": "project-1", "kind": "chat"}
    chats = tmp_path / "project" / "chats"
    chats.mkdir(parents=True)
    outputs = tmp_path / "project" / "tool-outputs" / "session-process-1"
    outputs.mkdir(parents=True)
    (outputs / "other-checkpoint.txt").write_text("Neutral tool output", encoding="utf-8")
    session = local_agent.parse_gemini_cli(
        stub, "fallback", source_path=chats / "session.jsonl", sidecar_resolver=FilesystemSidecarResolver()
    )
    assert session.messages == []
    assert not any(event.event_type == "gemini_cli_tool_output_sidecar" for event in session.session_events)
    assert session.unit_accounting is not None
    session.unit_accounting.assert_conserved()


def test_gemini_cli_object_honors_direct_record_transform(tmp_path: Path) -> None:
    source = tmp_path / "session.json"
    source.write_text(
        json.dumps({"sessionId": "process-1", "kind": "chat", "messages": [{"type": "user", "content": "Hi"}]}),
        encoding="utf-8",
    )
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_records=lambda _records: iter(()),
    )
    assert artifact.error is None
    assert list(artifact.iter_sessions()) == []
    artifact.discard()


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


def test_chatgpt_mapping_children_spill_preserves_duplicate_and_member_order(tmp_path: Path) -> None:
    first = '{"id":"node","children":["old"]}'
    second = '{"id":"node","children":["later", "first", "later"]}'
    source = (
        '{"conversation_id":"conversation", "current_node":"node", "create_time":1, '
        f'"mapping":{{"node":{first},"node":{second}}}}}'
    ).encode()
    with sqlite3.connect(tmp_path / "nodes.db") as conn:
        result = read_chatgpt_mapping_object(BytesIO(source), conn)
        assert result is not None
        mapping = result[1]
        assert mapping.children_are_all_strings()
        assert chatgpt._mapping_nodes_are_valid(mapping.shallow_view())
        assert mapping.shallow_node("node") == {"id": "node", "children": []}
        assert mapping["node"] == {"id": "node", "children": ["later", "first", "later"]}
        assert list(mapping.iter_children("node")) == ["later", "first", "later"]
        assert conn.execute("SELECT COUNT(*) FROM chatgpt_child").fetchone()[0] == 3


def test_chatgpt_large_children_array_prepares_without_rebuilding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = ChatGPTExportBuilder("large-children").add_node("user", "Prompt").add_node("assistant", "Answer").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    parent, child = mapping.values()
    assert isinstance(parent, dict) and isinstance(child, dict)
    child["parent"] = parent["id"]
    parent["children"] = [f"absent-{index}" for index in range(20_000)] + [child["id"]]
    record["current_node"] = child["id"]
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    def refuse_rebuild(_self: ChatGPTNodeMapping, _key: str) -> object:
        raise AssertionError("large children array was rebuilt during the prepared path")

    monkeypatch.setattr(ChatGPTNodeMapping, "__getitem__", refuse_rebuild)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())[0]
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


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
    actual = list(artifact.iter_sessions())[0]
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


def test_chatgpt_simple_mapping_normalizes_one_node_at_a_time_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = ChatGPTExportBuilder("simple-spill").add_node("user", "First").add_node("assistant", "Second").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    first, second = mapping.values()
    assert isinstance(first, dict) and isinstance(second, dict)
    second["parent"] = first["id"]
    # Non-monotonic timestamps exercise the parser's ordering contract after
    # scratch sorting; the parent edge exercises branch and active-path state.
    second_message = second["message"]
    first_message = first["message"]
    assert isinstance(second_message, dict) and isinstance(first_message, dict)
    first_message.update(
        {"update_time": 2.0, "status": "finished_successfully", "end_turn": True, "weight": 1.0, "recipient": "all"}
    )
    second_message.update(
        {"update_time": 3.0, "status": "finished_successfully", "end_turn": True, "weight": 1.0, "recipient": "all"}
    )
    second_message["author"] = {"role": "assistant", "name": "assistant", "metadata": {}}
    second_message["create_time"] = 1.0
    for index in range(300):
        node_id = f"extra-{index}"
        mapping[node_id] = {
            "id": node_id,
            "parent": first["id"],
            "message": {
                "id": node_id,
                "author": {"role": "assistant"},
                "create_time": float(index + 3),
                "content": {"content_type": "text", "parts": [f"Neutral answer {index}"]},
            },
        }
    record["current_node"] = "extra-299"
    first["children"] = ["extra-1", "extra-0"]  # declared order overrides their mapping order
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    normalized_sizes: list[int] = []
    original = chatgpt.extract_messages_from_mapping

    def observe(mapping: Mapping[str, object], *args: object, **kwargs: object) -> object:
        normalized_sizes.append(len(mapping))
        assert len(mapping) <= 1
        return original(mapping, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chatgpt, "extract_messages_from_mapping", observe)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    assert len(normalized_sizes) == len(mapping) + 1  # includes the empty envelope shell
    actual = list(artifact.iter_sessions())[0]
    assert [item.model_dump(mode="json") for item in actual.messages] == [
        item.model_dump(mode="json") for item in expected.messages
    ]
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.unit_accounting == expected.unit_accounting
    assert actual.active_leaf_message_provider_id == expected.active_leaf_message_provider_id
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_retained_chatgpt_simple_mapping_replays_sealed_messages(tmp_path: Path) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    record = (
        ChatGPTExportBuilder("retained-simple")
        .add_node("user", "Neutral prompt")
        .add_node("assistant", "Neutral answer")
        .build()
    )
    payload = json.dumps(record).encode()
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(payload)
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)
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
    sessions = list(artifact.session_sequence())
    assert len(sessions) == 1
    assert isinstance(sessions[0].messages, SqliteMessageSink)
    assert [message.text for message in sessions[0].messages] == ["Neutral prompt", "Neutral answer"]
    assert sessions[0].content_hash == session_content_hash(sessions[0])
    artifact.discard()


def test_chatgpt_object_finalizer_receives_disk_sidecars_and_empty_result_cleans_rows(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("carrier-callback").add_node("user", "Neutral prompt").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    node = next(iter(mapping.values()))
    assert isinstance(node, dict)
    message = node["message"]
    assert isinstance(message, dict)
    message["metadata"] = {"attachments": [{"id": "file-neutral", "name": "brief.txt"}]}
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    observed: list[tuple[bool, bool]] = []

    def reject_after_observation(sessions: list[ParsedSession]) -> list[ParsedSession]:
        session = sessions[0]
        observed.append(
            (
                isinstance(session.attachments, SqliteAttachmentSink),
                isinstance(session.session_events, SqliteSessionEventSink),
            )
        )
        return []

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
        prepare_sessions=reject_after_observation,
    )
    assert artifact.error is None
    assert observed == [(True, True)]
    assert list(artifact.iter_sessions()) == []
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        for table in ("prepared_session", "prepared_message", "prepared_event", "prepared_attachment"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    artifact.discard()


def test_chatgpt_object_finalizer_enriches_sealed_attachment_and_event_rows(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("carrier-sidecar").add_node("user", "Neutral prompt").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    node = next(iter(mapping.values()))
    assert isinstance(node, dict)
    message = node["message"]
    assert isinstance(message, dict)
    message["metadata"] = {"attachments": [{"id": "file-neutral"}]}
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    index = ChatGPTAssetIndex.build(
        library_files_payload=[], asset_file_names_payload={"file-neutral.dat": "brief.txt"}
    )

    def enrich(session: ParsedSession) -> ParsedSession:
        assert isinstance(session.attachments, SqliteAttachmentSink)
        return ChatGPTAssemblySpec().enrich_session(session, {"chatgpt_asset_index": index})

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
        prepare_session=enrich,
    )
    assert artifact.error is None
    [session] = artifact.iter_sessions()
    assert [attachment.name for attachment in session.attachments] == ["brief.txt"]
    assert [event.event_type for event in session.session_events] == ["chatgpt_asset_resolution"]
    assert session_content_hash(session) == session.content_hash
    artifact.discard()


def test_chatgpt_missing_current_node_uses_collecting_parser(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("missing-current").add_node("user", "Neutral prompt").build()
    record["current_node"] = "missing-node"
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())[0]
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


@pytest.mark.parametrize("default_model_slug", [123, ""])
def test_chatgpt_simple_mapping_matches_default_model_coercion(tmp_path: Path, default_model_slug: object) -> None:
    record = ChatGPTExportBuilder("model-coercion").add_node("assistant", "Neutral answer").build()
    record["default_model_slug"] = default_model_slug
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())[0]
    assert [message.model_name for message in actual.messages] == [message.model_name for message in expected.messages]
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_simple_mapping_preserves_empty_parent_key_on_active_path(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("empty-parent").add_node("user", "First").add_node("assistant", "Second").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    first, second = mapping.values()
    assert isinstance(first, dict) and isinstance(second, dict)
    first["id"] = ""
    first_message = first["message"]
    assert isinstance(first_message, dict)
    first_message["id"] = "parent-message"
    second["parent"] = ""
    record["mapping"] = {"": first, "node-2": second}
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())[0]
    assert [message.is_active_path for message in actual.messages] == [
        message.is_active_path for message in expected.messages
    ]
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_empty_current_node_uses_collecting_parser(tmp_path: Path) -> None:
    record = ChatGPTExportBuilder("empty-current").add_node("user", "Neutral prompt").build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    node = next(iter(mapping.values()))
    assert isinstance(node, dict)
    node["id"] = ""
    record["mapping"] = {"": node}
    record["current_node"] = ""
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())[0]
    assert [message.is_active_path for message in actual.messages] == [
        message.is_active_path for message in expected.messages
    ]
    assert actual.active_leaf_message_provider_id == expected.active_leaf_message_provider_id
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_sandbox_fallback_emits_truncation_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    links = " ".join(f"sandbox:/mnt/data/file-{index}.txt" for index in range(513))
    record = ChatGPTExportBuilder("sandbox-links").add_node("assistant", links).build()
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    emitted: list[str] = []

    def observe(event: str, **_kwargs: object) -> None:
        emitted.append(event)

    monkeypatch.setattr(chatgpt, "emit", observe)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    assert emitted.count("sources.chatgpt.sandbox_links_bounded") == 1
    assert len(list(artifact.iter_sessions())[0].attachments) == 512
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
    actual = list(artifact.iter_sessions())[0]
    collected = actual.model_copy(
        update={"messages": list(actual.messages), "session_events": list(actual.session_events)}
    )
    assert collected.model_dump(mode="json", exclude={"content_hash"}) == expected.model_dump(
        mode="json", exclude={"content_hash"}
    )
    assert actual.content_hash == session_content_hash(expected)
    assert session_content_hash(actual) == session_content_hash(expected)
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


def test_retained_claude_design_object_uses_streamed_replay_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    record = {
        "uuid": "design-retained",
        "project": {"uuid": "neutral-project"},
        "messages": [
            {
                "uuid": f"u-{index}",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": f"Neutral {index}",
                    "authorAccountUuid": "neutral-account",
                    "timestamp": f"2026-01-01T00:00:{index % 60:02d}Z",
                },
            }
            for index in range(300)
        ],
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode())
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    for path, tier in ((source_db, ArchiveTier.SOURCE), (index_db, ArchiveTier.INDEX)):
        with sqlite3.connect(path) as conn:
            initialize_archive_tier(conn, tier)

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Design object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    artifact = revision_backfill.prepare_retained_jsonl_artifact(
        "synthetic-raw",
        Provider.CLAUDE_DESIGN.value,
        blob_hash,
        str(tmp_path / "design_chats" / "session.json"),
        "full",
        None,
        str(blob_root),
        str(source_db),
        str(index_db),
        str(tmp_path / "prepared"),
        "2025-01-02T03:04:05Z",
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    [actual] = artifact.iter_sessions()
    assert actual.provider_session_id == "design-retained"
    assert len(actual.messages) == 300
    assert len(actual.session_events) == 300
    assert actual.created_at == "2026-01-01T00:00:00+00:00"
    assert actual.updated_at == "2026-01-01T00:00:59+00:00"
