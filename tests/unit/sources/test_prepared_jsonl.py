"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import (
    _ACTIVE_PARENT_LOOKUP_SQL,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
)
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
