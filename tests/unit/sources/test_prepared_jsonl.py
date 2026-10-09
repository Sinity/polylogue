"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO, cast

import ijson
import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.core.json import JSONValue
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
from polylogue.sources.decoder_json import claude_design_object_envelope, iter_grok_export_events
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver
from polylogue.sources.parsers import chatgpt, local_agent
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.base_models import ParsedContentBlock
from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex
from polylogue.sources.prepared_jsonl import PreparedJsonl, PreparedSessionSequence, _write_artifact, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import (
    _EARLIER_PARENT_OCCURRENCE_SQL,
    _LAST_PARENT_OCCURRENCE_SQL,
    ChatGPTNodeMapping,
    ScratchSessionSpill,
    SqliteAttachmentSink,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    read_chatgpt_mapping_object,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarFile, RetainedSidecarScope
from tests.infra.json_values import iter_owned_json_values


def test_prepared_jsonl_retry_progress_keeps_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench.run import WorkProgressTail
    from polylogue.core import work_progress

    monkeypatch.setattr(work_progress, "PROGRESS_INTERVAL_S", 0)
    emitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(work_progress, "emit", lambda event, **fields: emitted.append((event, fields)))
    payload = (
        b'{"type":"session_meta","payload":{"id":"retry-progress"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m-1","role":"user",'
        b'"content":[{"type":"input_text","text":"stable source"}]}}\n'
    )
    blob_hash = hashlib.sha256(payload).hexdigest()
    events_path = tmp_path / "events.jsonl"
    tail = WorkProgressTail(events_path)
    counts: list[int] = []
    unit_ids: set[str] = set()
    productive_ids: set[str] = set()

    for attempt in ("first", "retry"):
        source = tmp_path / f"{attempt}.jsonl"
        source.write_bytes(payload)
        artifact = prepare_jsonl_blob(
            str(source),
            "codex/stable.jsonl",
            Provider.CODEX.value,
            "fallback",
            is_stream=True,
            shard_directory=str(tmp_path / f"{attempt}-shards"),
            source_sha256=blob_hash,
            strict_jsonl_records=True,
        )
        artifact.discard()
        with events_path.open("a", encoding="utf-8") as handle:
            for event, fields in emitted:
                if event == "daemon.work.progress":
                    unit_ids.add(str(fields["unit_id"]))
                    productive_ids.add(str(fields["productive_id"]))
                    handle.write(json.dumps({"event": event, **fields}) + "\n")
        emitted.clear()
        counts.append(tail.poll())

    assert len(unit_ids) == 2
    assert len(productive_ids) == 1
    assert counts[0] > 0
    assert counts[1] == counts[0]
    tail.close()


from polylogue.sources.value_bounds import MAX_STORABLE_VALUE_BYTES
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_jsonl import prepared_source_fixture, retained_parser_fixture
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.parametrize("late_conversation", [False, True])
def test_prepared_stream_classification_uses_records_after_old_sample(tmp_path: Path, late_conversation: bool) -> None:
    source = tmp_path / "session.jsonl"
    payload = (json.dumps({"type": "file-history-snapshot"}) + "\n") * 65
    if late_conversation:
        payload += (
            json.dumps(
                {
                    "type": "user",
                    "uuid": "message",
                    "sessionId": "session",
                    "message": {"role": "user", "content": "hello"},
                }
            )
            + "\n"
        )
    source.write_text(payload, encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_CODE.value,
        "session",
        is_stream=True,
        shard_directory=str(tmp_path / "prepared"),
    )
    try:
        assert artifact.error is None
        proof = artifact.stream_classification()
        assert proof is not None
        assert proof.proved_non_session is not late_conversation
        sessions = list(artifact.iter_sessions())
        assert len(sessions) == int(late_conversation)
        if sessions:
            assert [message.text for message in sessions[0].messages] == ["hello"]
    finally:
        artifact.discard()


@pytest.mark.parametrize("terminated", [False, True])
def test_prepared_classification_obeys_parser_tail_boundary(tmp_path: Path, terminated: bool) -> None:
    source = tmp_path / "session.jsonl"
    payload = (json.dumps({"type": "file-history-snapshot"}) + "\n") * 65
    source.write_text(payload + '{"broken":' + ("\n" if terminated else ""), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_CODE.value,
        "session",
        is_stream=True,
        shard_directory=str(tmp_path / "prepared"),
    )
    try:
        if terminated:
            assert artifact.error is not None
            assert artifact.sessions_path is None
        else:
            assert artifact.error is None
            proof = artifact.stream_classification()
            assert proof is not None and proof.proved_non_session
            assert proof.record_count == 65
            assert list(artifact.iter_sessions()) == []
    finally:
        artifact.discard()


def test_prepared_beads_refusal_keeps_explicit_proof_with_unknown_kind(tmp_path: Path) -> None:
    from polylogue.archive.artifact_taxonomy import ArtifactKind

    source = tmp_path / "interactions.jsonl"
    source.write_text(
        json.dumps(
            {"id": "interaction", "kind": "field_change", "created_at": "synthetic", "issue_id": "task", "extra": {}}
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.UNKNOWN.value,
        "interaction",
        is_stream=True,
        shard_directory=str(tmp_path / "prepared"),
    )
    try:
        assert artifact.error is None
        proof = artifact.stream_classification()
        assert proof is not None and proof.proved_non_session
        assert proof.classification.kind is ArtifactKind.UNKNOWN
        assert list(artifact.iter_sessions()) == []
    finally:
        artifact.discard()


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
        for sql, parameters in (
            (_EARLIER_PARENT_OCCURRENCE_SQL, (0, "tail", 7)),
            (_LAST_PARENT_OCCURRENCE_SQL, (0, "tail")),
        ):
            plan = store.conn.execute("EXPLAIN QUERY PLAN " + sql, parameters).fetchall()
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


def test_composed_attachment_binds_current_carrier_and_preserves_original_evidence(tmp_path: Path) -> None:
    artifact, coordinate = _prepared_artifact(tmp_path)
    original = artifact.session_sequence()[0]
    original_key = original.attachments[0].acquisition_key
    composed_path = tmp_path / "composed.db"
    store = SqliteMessageStore(composed_path)
    try:
        _write_artifact(
            store,
            "e" * 64,
            [original],
            enrichment_digest="c" * 64,
            enrichment_index_path="/index.db",
        )
    finally:
        store.close()
    attachments = SqliteAttachmentSink(composed_path, 0, count=1)
    restored = attachments[0]
    assert restored.acquisition_key == (str(composed_path), 0, 0)
    assert restored.acquisition_key != original_key
    assert original.attachments[0].acquisition_key == original_key
    assert restored.owner_coordinate == coordinate
    assert restored.inline_bytes == original.attachments[0].inline_bytes
    assert restored.precomputed_blob == original.attachments[0].precomputed_blob


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


def _count_message_decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    decodes = [0]
    real = ParsedMessage.model_validate_json

    def counting(data: str | bytes, *args: object, **kwargs: object) -> ParsedMessage:
        decodes[0] += 1
        return real(data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ParsedMessage, "model_validate_json", counting)
    return decodes


def test_sealed_session_decodes_once_across_walks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publication walks a session's messages many times; each walk after the
    first reuses the decoded messages.

    Anti-vacuity: without the retained decode, the second and third walks
    validate every message again, so the count is 3 instead of 1.
    """
    artifact, coordinate = _prepared_artifact(tmp_path)
    (session,) = list(artifact.iter_sessions())
    decodes = _count_message_decodes(monkeypatch)
    first = list(session.messages)
    second = list(session.messages)
    assert isinstance(session.messages, SqliteMessageSink)
    third = list(session.messages.iter_from(0))
    assert decodes[0] == 1
    assert [message.model_dump() for message in second] == [message.model_dump() for message in first]
    assert third[0].owner_coordinate == coordinate
    assert session.messages[0] is first[0]


def test_discarded_sessions_are_decoded_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A discarded carrier releases its retained decode."""
    from polylogue.sources import prepared_message_sink

    artifact, _coordinate = _prepared_artifact(tmp_path)
    assert artifact.sessions_path is not None
    (session,) = list(artifact.iter_sessions())
    decodes = _count_message_decodes(monkeypatch)
    list(session.messages)
    prepared_message_sink.discard_decoded_sessions(artifact.sessions_path)
    list(session.messages)
    assert decodes[0] == 2


def test_oversized_session_walks_replay_a_spool_not_the_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session above half the budget stays out of memory, and its later walks unpickle.

    Anti-vacuity: without the spool every walk of an oversized session
    re-validates each message from JSON, so the second and third walks
    raise the count to three decodes per message.
    """
    from polylogue.sources import prepared_message_sink

    artifact, coordinate = _prepared_artifact(tmp_path)
    assert artifact.sessions_path is not None
    prepared_message_sink.discard_decoded_sessions(artifact.sessions_path)
    monkeypatch.setattr(prepared_message_sink._DECODED_SESSIONS, "budget_bytes", 8)
    (session,) = list(artifact.iter_sessions())
    assert isinstance(session.messages, SqliteMessageSink)
    decodes = _count_message_decodes(monkeypatch)
    first = list(session.messages)
    second = list(session.messages)
    suffix = list(session.messages.iter_from(1))
    assert decodes[0] == len(first)
    retained = [
        key for key in prepared_message_sink._DECODED_SESSIONS._entries if key[0] == str(artifact.sessions_path)
    ]
    assert retained == []
    assert second == first
    assert second[0] is not first[0]
    assert suffix == first[1:]
    assert first[0].owner_coordinate == coordinate

    prepared_message_sink.discard_decoded_sessions(artifact.sessions_path)
    list(session.messages)
    assert decodes[0] == 2 * len(first)


def _tool_turn(session: int, index: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"s{session}-m{index}",
        role=Role.ASSISTANT,
        text=f"turn {index}",
        blocks=[
            ParsedContentBlock(type=BlockType.TEXT, text="neutral narration " * 4),
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="shell",
                tool_id=f"s{session}-t{index}",
                tool_input={"command": "ls", "options": {"long": True, "paths": ["a", "b", "c"]}},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT, tool_id=f"s{session}-t{index}", text="out " * 40, is_error=False
            ),
        ],
    )


def test_decoded_session_cache_holds_no_more_memory_than_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decoded-session LRU's real resident memory stays within its declared budget.

    Sessions are walked until the LRU has evicted; the traced memory that
    clearing it releases is what it held.

    Anti-vacuity: charge each retained message its sealed JSON bytes instead
    of :func:`_decoded_size` and the LRU keeps about four times its budget of
    decoded messages.
    """
    import gc
    import tracemalloc

    from polylogue.sources import prepared_message_sink

    budget = 1024 * 1024
    path = tmp_path / "prepared.db"
    store = SqliteMessageStore(path)
    sessions: list[ParsedSession] = []
    for ordinal in range(16):
        messages = store.new_sink()
        for index in range(30):
            messages.append(_tool_turn(ordinal, index))
        session = ParsedSession(
            source_name=Provider.CODEX, provider_session_id=f"session-{ordinal}", messages=[]
        ).model_copy(update={"messages": messages, "session_events": store.new_event_sink()})
        session.content_hash = session_content_hash(session)
        sessions.append(session)
    shard = prepare_session_shard(tmp_path, sessions)
    _write_artifact(store, "b" * 64, sessions, enrichment_digest="c" * 64, enrichment_index_path="/index.db")
    store.close()
    artifact = PreparedJsonl.seal(
        "b" * 64, path, shard.path, enrichment_digest="c" * 64, enrichment_index_path="/index.db"
    )
    sealed = list(artifact.iter_sessions())
    monkeypatch.setattr(prepared_message_sink._DECODED_SESSIONS, "budget_bytes", budget)
    prepared_message_sink._DECODED_SESSIONS.clear()
    gc.collect()
    tracemalloc.start()
    try:
        for session in sealed:
            for _message in session.messages:
                pass
        gc.collect()
        retained_entries = len(prepared_message_sink._DECODED_SESSIONS._entries)
        charged = prepared_message_sink._DECODED_SESSIONS._bytes
        holding = tracemalloc.get_traced_memory()[0]
        prepared_message_sink._DECODED_SESSIONS.clear()
        gc.collect()
        held = holding - tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    # The LRU retained sessions and evicted others without exceeding its budget.
    assert 0 < retained_entries < len(sealed), retained_entries
    assert 0 < charged <= budget, (charged, budget, retained_entries)
    assert held <= budget, (held, budget, retained_entries)


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
        Provider.CLAUDE_AI, list(iter_owned_json_values(BytesIO(source.read_bytes()), source.name)), "fallback"
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    assert first_written_after == 1
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
    )
    assert artifact.error is None
    proof = artifact.stream_classification()
    assert proof is not None and not proof.proved_non_session
    assert proof.classification.provider is Provider.HERMES
    assert proof.record_count == 1
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

    bootstrap_archive_root(tmp_path)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.HERMES.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "sessions" / "session_extract.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime=None,
    ) as (artifact, _reader):
        assert artifact.error is None
        assert list(artifact.iter_sessions()) == []


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
        "polylogue.sources.prepared_jsonl.owned_json_records",
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
            entry("oversize.txt", "not read", size=MAX_STORABLE_VALUE_BYTES + 1),
            entry("unreadable.txt", "not read", unreadable=True),
        ),
    )

    class Resolver:
        def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
            return scope

        def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
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
    monkeypatch.setattr(prepared_jsonl, "owned_json_records", lambda *_a, **_k: pytest.fail("whole-object fallback"))
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
    assert _gemini_message_payloads(actual) == _gemini_message_payloads(expected)
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.content_hash == session_content_hash(expected)
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
    from polylogue.core.hashing import hash_text

    assert [event["content_hash"] for event in sidecar_events[2:4]] == [
        hash_text("first complete output"),
        hash_text("second complete output"),
    ]
    assert [event["reason"] for event in sidecar_events[4:]] == [
        # A sidecar no SQLite cell can hold; smaller ones are joined whole.
        "value_bound_refused",
        "read_error:OSError",
        "expected_sidecar_not_retained",
    ]
    assert any(block.text == "second complete output" for message in actual.messages for block in message.blocks)
    artifact.discard()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_prepared_gemini_sidecars_preserve_stderr_and_independent_display(tmp_path: Path, checkpoint: bool) -> None:
    """Replacing prepared primary carriers with text alone loses stderr/display."""
    from polylogue.sources.parsers.local_agent import TOOL_RESULT_DISPLAY_MEDIA_TYPE

    messages = [
        {
            "id": "answer",
            "type": "gemini",
            "timestamp": "2026-01-01T00:00:01Z",
            "content": "",
            "toolCalls": [
                {
                    "id": "stderr-call",
                    "name": "terminal",
                    "result": [
                        {
                            "functionResponse": {
                                "id": "stderr-call",
                                "response": {
                                    "output": "<tool_output_masked>For full output see: stderr-call.txt</tool_output_masked>",
                                    "error": "DISTINCT_STDERR",
                                },
                            }
                        }
                    ],
                },
                {
                    "id": "display-call",
                    "name": "terminal",
                    "resultDisplay": "DISPLAY_OUTPUT",
                    "result": [{"functionResponse": {"id": "display-call", "response": {"output": "MODEL_OUTPUT"}}}],
                },
            ],
        }
    ]
    header = {
        "sessionId": "sidecar-streams",
        "projectHash": "neutral",
        "kind": "main",
        "startTime": "2026-01-01T00:00:00Z",
    }
    payload = [header, *messages] if checkpoint else {**header, "messages": messages}
    source = tmp_path / ("checkpoint.jsonl" if checkpoint else "session.json")
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in payload) if checkpoint else json.dumps(payload),
        encoding="utf-8",
    )
    full_stderr = "FULL_STDOUT"
    full_display = "FULL_MODEL_OUTPUT_THAT_IS_LONGER_THAN_INLINE"
    scope = RetainedSidecarScope(
        scope_key="neutral-streams",
        available=True,
        files=(
            RetainedSidecarFile("stderr-call.txt", len(full_stderr), None, lambda: full_stderr),
            RetainedSidecarFile("display-call.txt", len(full_display), None, lambda: full_display),
        ),
    )

    class Resolver:
        def gemini_cli_scope(self, *_args: object) -> RetainedSidecarScope:
            return scope

    resolver = Resolver()
    [expected] = parse_payload(
        Provider.GEMINI_CLI, payload, "fallback", source_path=str(source), sidecar_resolver=resolver
    )
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "fallback",
        is_stream=False,
        strict_jsonl_records=checkpoint,
        shard_directory=str(tmp_path / "prepared"),
        sidecar_resolver=resolver,
    )
    try:
        assert artifact.error is None, artifact.error
        [actual] = artifact.iter_sessions()
        for session in (expected, actual):
            results = [
                block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
            ]
            assert [block.text for block in results] == [
                f"{full_stderr}\nDISTINCT_STDERR",
                full_display,
                "DISPLAY_OUTPUT",
            ]
            assert results[-1].media_type == TOOL_RESULT_DISPLAY_MEDIA_TYPE
        assert _gemini_message_payloads(actual) == _gemini_message_payloads(expected)
        assert actual.content_hash == session_content_hash(expected)
    finally:
        artifact.discard()


def test_retained_gemini_sidecar_replay_uses_sealed_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
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
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.owned_json_records", lambda *_a, **_k: pytest.fail("whole-object replay")
    )
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GEMINI_CLI.value),
        blob_hash=blob_hash,
        source_path=source_path,
        directory=Path(str(tmp_path / "prepared")),
        file_mtime=None,
    ) as (artifact, _reader):
        assert artifact.error is None
        [actual] = artifact.iter_sessions()
        [expected] = parse_payload(
            Provider.GEMINI_CLI,
            record,
            "session",
            source_path=source_path,
            sidecar_resolver=_reader.retained_sidecar_resolver(),
        )
        assert actual.provider_session_id == expected.provider_session_id
        assert [message.provider_message_id for message in actual.messages] == [
            message.provider_message_id for message in expected.messages
        ]
        assert _gemini_message_payloads(actual) == _gemini_message_payloads(expected)
        assert [event.model_dump(mode="json") for event in actual.session_events] == [
            event.model_dump(mode="json") for event in expected.session_events
        ]


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
        def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
            return scope

        def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
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


def test_gemini_cli_complete_object_preserves_its_declared_messages(tmp_path: Path) -> None:
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
    )
    assert artifact.error is None
    [session] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.GEMINI_CLI, json.loads(source.read_text()), "fallback")
    assert session.provider_session_id == expected.provider_session_id
    assert [message.model_dump(mode="json") for message in session.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert [message.text for message in session.messages] == ["Hi"]
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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

    def finalize(sessions: PreparedSessionSequence) -> Iterable[ParsedSession]:
        finalized.extend(session.provider_session_id for session in sessions)
        return sessions

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=finalize,
    )
    assert artifact.error is None
    assert first_written_after == 1
    assert finalized == ["retained-generic"]
    [actual] = artifact.iter_sessions()
    assert (actual.provider_session_id, actual.content_hash) == (
        expected.provider_session_id,
        session_content_hash(expected),
    )
    assert [message.text for message in actual.messages] == [message.text for message in expected.messages]
    artifact.discard()


def test_retained_generic_object_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    bootstrap_archive_root(tmp_path)
    record = {
        "id": "retained-drive",
        "messages": [{"id": "repeated", "role": "user", "text": f"Neutral prompt {index}"} for index in range(120)],
    }
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(record).encode())
    source_db = tmp_path / "source.db"

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained replay decoded the complete object")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.DRIVE.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "session.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (artifact, _reader):
        assert artifact.error is None
        [actual] = artifact.iter_sessions()
        [expected] = parse_payload(Provider.DRIVE, record, "session")
        assert actual.provider_session_id == expected.provider_session_id
        assert [message.provider_message_id for message in actual.messages] == [
            message.provider_message_id for message in expected.messages
        ]
        assert [message.text for message in actual.messages] == [message.text for message in expected.messages]
        assert actual.created_at == "2025-01-02T03:04:05+00:00"


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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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
        assert prepared.execute(
            "SELECT COUNT(*) FROM prepared_message WHERE session_ordinal IN "
            "(SELECT message_ordinal FROM prepared_session)"
        ).fetchone()[0] == sum(len(session.messages) for session in expected)
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


def test_grok_empty_conversation_is_refused_at_preparation(tmp_path: Path) -> None:
    """The preparation owner admits sessions on every branch, callback or not.

    Anti-vacuity: before the owner applied the rule itself, a Grok export
    prepared without a callback sealed the empty conversation that every
    publishing route then refuses.
    """
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
    assert artifact.positive_evidence_filtered is True
    assert list(artifact.iter_sessions()) == []
    direct = parse_payload(Provider.GROK, record, "fallback")
    assert direct == []
    assert admit_parsed_sessions_for_publication(direct, provider=Provider.GROK, source_path=str(source)) == []
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


def test_grok_future_wire_type_keeps_parser_admission_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "type": "future_export",
        "conversations": [
            {
                "conversation": {"title": "T", "kind": "unknown_conversation"},
                "responses": [{"sender": "human", "message": "Hi", "type": "future_response"}],
            },
            {
                "conversation": {"title": "Known"},
                "responses": [{"sender": "human", "message": "Hello"}],
            },
        ],
    }
    source = tmp_path / "future-grok.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    expected = parse_payload(Provider.GROK, record, "fallback")

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("future-typed Grok export decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())
    assert [list(session.session_events) for session in actual] == [session.session_events for session in expected]
    assert [session.unit_accounting for session in actual] == [session.unit_accounting for session in expected]
    assert [[event.payload for event in session.session_events] for session in actual] == [
        [{"source_index": 1, "wire_type": "unknown_conversation"}],
        [],
    ]
    artifact.discard()


def test_retained_grok_streams_responses_with_replay_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    bootstrap_archive_root(tmp_path)
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
    expected = admit_parsed_sessions_for_publication(
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

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Grok object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=source_path,
        directory=Path(str(tmp_path / "prepared")),
        file_mtime=fallback_timestamp,
    ) as (artifact, _reader):
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
        with sqlite3.connect(artifact.shard_path) as prepared:
            assert prepared.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == len(responses)

    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "agent-neutral.meta.json"),
        directory=Path(str(tmp_path / "sidecar-prepared")),
        file_mtime=fallback_timestamp,
    ) as (sidecar, _reader):
        # ``agent-*.meta.json`` is a content-blind sidecar marker, not an
        # OriginSpec ``fact`` path: session-shaped content there stays refused.
        assert sidecar.error is None
        assert list(sidecar.iter_sessions()) == []

    fact_hash, _ = BlobStore(Path(str(source_db)).parent / "blob").write_from_bytes(
        b'{"agent":"neutral","facts":[{"key":"status","value":"ready"}]}'
    )
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.GROK,
        blob_hash=fact_hash,
        source_path=str(tmp_path / "agent-fact.meta.json"),
        directory=tmp_path / "fact-prepared",
        file_mtime=fallback_timestamp,
    ) as (fact, _reader):
        assert fact.error is None
        assert list(fact.iter_sessions()) == []
        proof = fact.stream_classification()
        assert proof is not None and proof.proved_non_session

    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "analysis" / "prod-grok-backend.json"),
        directory=Path(str(tmp_path / "analysis-prepared")),
        file_mtime=fallback_timestamp,
    ) as (analysis_artifact, _reader):
        assert analysis_artifact.error is None
        assert [
            (session.provider_session_id, session.content_hash) for session in analysis_artifact.iter_sessions()
        ] == [(session.provider_session_id, session.content_hash) for session in expected]

    beads_record = {**record, "extra": {}}
    beads_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(beads_record).encode("utf-8"))
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=beads_hash,
        source_path=source_path,
        directory=Path(str(tmp_path / "beads-prepared")),
        file_mtime=fallback_timestamp,
    ) as (beads_artifact, _reader):
        assert beads_artifact.error is None
        assert list(beads_artifact.iter_sessions()) == []
        assert beads_artifact.blob_hash == beads_hash

    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=beads_hash,
        source_path=str(tmp_path / "analysis" / "prod-grok-backend.json"),
        directory=Path(str(tmp_path / "beads-analysis-prepared")),
        file_mtime=fallback_timestamp,
    ) as (beads_analysis_artifact, _reader):
        assert beads_analysis_artifact.error is None
        assert list(beads_analysis_artifact.iter_sessions()) == []

    messages_record = {key: value for key, value in record.items() if key not in {"type", "version"}}
    messages_record["messages"] = [{"role": "user", "content": "Root metadata"}]
    messages_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(messages_record).encode("utf-8"))
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=messages_hash,
        source_path=str(tmp_path / "analysis" / "prod-grok-backend.json"),
        directory=Path(str(tmp_path / "messages-prepared")),
        file_mtime=fallback_timestamp,
    ) as (messages_artifact, _reader):
        assert messages_artifact.error is None
        assert [
            (session.provider_session_id, session.content_hash) for session in messages_artifact.iter_sessions()
        ] == [(session.provider_session_id, session.content_hash) for session in expected]


def test_retained_grok_corrupt_suffix_leaves_no_publishable_artifact(tmp_path: Path) -> None:

    bootstrap_archive_root(tmp_path)
    blob_root = tmp_path / "blob"
    payload = b'{"conversations":[{"conversation":{},"responses":[{"sender":"human","message":"Hi"}]}]} trailing'
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(payload)
    source_db = tmp_path / "source.db"
    directory = tmp_path / "prepared"
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "prod-grok-backend.json"),
        directory=Path(str(directory)),
        file_mtime=None,
    ) as (artifact, _reader):
        assert artifact.error is not None
        assert artifact.sessions_path is None
        assert artifact.shard_path is None
        assert list(directory.glob("*.db")) == []


def test_retained_grok_future_wire_keeps_parser_admission_event(tmp_path: Path) -> None:

    bootstrap_archive_root(tmp_path)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "prod-grok-backend.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime=None,
    ) as (artifact, _reader):
        assert artifact.error is None
        assert artifact.positive_evidence_filtered
        [session] = artifact.iter_sessions()
        [expected] = parse_payload(Provider.GROK, record, "fallback")
        assert list(session.session_events) == expected.session_events
        assert session.unit_accounting == expected.unit_accounting
        assert [event.event_type for event in session.session_events] == ["grok_unknown_input"]


def test_retained_grok_hook_overlap_keeps_artifact_taxonomy(tmp_path: Path) -> None:

    bootstrap_archive_root(tmp_path)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.GROK.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "prod-grok-backend.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime=None,
    ) as (artifact, _reader):
        assert artifact.error is None
        assert list(artifact.iter_sessions()) == []


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
        Provider.CHATGPT, list(iter_owned_json_values(BytesIO(source.read_bytes()), source.name)), "fallback"
    )
    expected = admit_parsed_sessions_for_publication(expected, provider=Provider.CHATGPT, source_path=str(source))
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
    bootstrap_archive_root(tmp_path)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.CHATGPT.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "snapshot.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (artifact, _reader):
        assert artifact.error is None
        assert artifact.positive_evidence_filtered is True
        sessions = list(artifact.session_sequence())
        assert len(sessions) == 1
        assert sessions[0].created_at == "2025-01-02T03:04:05+00:00"
        assert sessions[0].updated_at == "2025-01-02T03:04:05+00:00"
        assert len(sessions[0].attachments) == 1
        assert sessions[0].attachments[0].name == "resolved-sidecar.png"
        assert evidence_providers == [Provider.CHATGPT]
        # The original seal owns retained evidence; a second artifact-level
        # enrichment owner is refused by the prepared Raw publication route.
        assert artifact.enrichment_digest is None
        assert artifact.enrichment_index_path is None
        assert artifact.blob_hash == blob_hash
        assert tuple(artifact.iter_attachment_claims()) == ()


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
    second = '{"id":"node","children":["discarded"],"children":["later", "first", "later"]}'
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

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
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


def _refuse_collecting_chatgpt_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if the prepared route falls back to the list-backed entry store.

    The collecting parser holds every normalized message in one list; the
    prepared route must keep them in scratch. Refusing the list store makes
    any fallback to it (or to ``extract_messages_from_mapping``) red.
    """

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepared ChatGPT route collected messages in memory")

    monkeypatch.setattr(chatgpt, "_ListMessageEntries", refuse)


def test_chatgpt_mapping_normalizes_into_scratch_with_parser_parity(
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

    _refuse_collecting_chatgpt_entries(monkeypatch)
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


def test_chatgpt_text_nodes_spill_attachment_metadata_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = ChatGPTExportBuilder("attachment-spill")
    for index in range(120):
        builder.add_node("user", f"Neutral prompt {index}")
    record = builder.build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    for index, node in enumerate(mapping.values()):
        assert isinstance(node, dict)
        message = node["message"]
        assert isinstance(message, dict)
        message["metadata"] = {
            "attachments": [{"id": f"file-{index}", "name": f"document-{index}.txt"}],
            "targeted_reply": f"Quoted turn {index}",
            "is_visually_hidden_from_conversation": index % 2 == 0,
        }
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    _refuse_collecting_chatgpt_entries(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = next(artifact.iter_sessions())
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert [attachment.model_dump(mode="json") for attachment in actual.attachments] == [
        attachment.model_dump(mode="json") for attachment in expected.attachments
    ]
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert actual.unit_accounting == expected.unit_accounting
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_retained_chatgpt_simple_mapping_replays_sealed_messages(tmp_path: Path) -> None:

    bootstrap_archive_root(tmp_path)
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
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.CHATGPT.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "snapshot.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.session_sequence())
        assert len(sessions) == 1
        assert isinstance(sessions[0].messages, SqliteMessageSink)
        assert [message.text for message in sessions[0].messages] == ["Neutral prompt", "Neutral answer"]
        assert sessions[0].content_hash == session_content_hash(sessions[0])


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

    def reject_after_observation(sessions: PreparedSessionSequence) -> Iterable[ParsedSession]:
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


def _stored_messages(session: ParsedSession) -> list[dict[str, object]]:
    """Messages as the writer stores them: active path settled, tool outcomes derived.

    Scratch retains the original parser fields and a separate normalized
    writer operand. Compare that operand with the collecting writer lowering.
    """
    from polylogue.core.sources import origin_from_provider
    from polylogue.sources.prepared_message_sink import normalize_active_branch
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    if isinstance(session.messages, SqliteMessageSink):
        return [
            message.model_dump(mode="json")
            for message in session.messages.normalized_messages(
                session.session_events, origin=origin_from_provider(session.source_name)
            )
        ]
    messages = derive_tool_outcomes(
        normalize_active_branch(list(session.messages)),
        list(session.session_events),
        origin=origin_from_provider(session.source_name),
    )
    return [message.model_dump(mode="json") for message in messages]


def test_chatgpt_missing_current_node_prepares_in_scratch(tmp_path: Path) -> None:
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
    assert _stored_messages(actual) == _stored_messages(expected)
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


def test_chatgpt_sandbox_links_spill_every_attachment(tmp_path: Path) -> None:
    """Every distinct sandbox link reaches the scratch attachment carrier.

    Anti-vacuity: a per-message attachment cap in the parser makes the count
    fall short of 1500.
    """
    links = " ".join(f"sandbox:/mnt/data/file-{index}.txt" for index in range(1500))
    record = ChatGPTExportBuilder("sandbox-links").add_node("assistant", links).build()
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
    attachments = list(artifact.iter_sessions())[0].attachments
    assert isinstance(attachments, SqliteAttachmentSink)
    assert len(attachments) == 1500
    artifact.discard()


def test_chatgpt_native_object_preparation_preserves_complete_parser_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complex nodes (tool output, multimodal parts, reasoning timings) spill too.

    Anti-vacuity: route any node shape back to the collecting parser and
    ``_refuse_collecting_chatgpt_entries`` turns this red.
    """
    fixture = Path("tests/fixtures/chatgpt/native-conversation-v1.json")
    source = tmp_path / "conversation.json"
    source.write_bytes(fixture.read_bytes())
    expected = parse_payload(Provider.CHATGPT, [json.loads(source.read_bytes())], "fallback")[0]
    _refuse_collecting_chatgpt_entries(monkeypatch)
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
    collected = actual.model_copy(update={"session_events": list(actual.session_events)})
    assert collected.model_dump(mode="json", exclude={"content_hash", "messages"}) == expected.model_dump(
        mode="json", exclude={"content_hash", "messages"}
    )
    assert _stored_messages(actual) == _stored_messages(expected)
    # The carrier's hash is sealed before shard lowering settles the scratch
    # messages in place, so it is compared, not recomputed from them.
    assert actual.content_hash == session_content_hash(expected)
    artifact.discard()


def test_chatgpt_complex_nodes_prepare_in_scratch_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cross-node rule holds when messages live in scratch.

    Covers the rules the collecting parser applied to its in-memory list:
    tool results owned through a tool chain, a code-interpreter run record,
    authorship evidence, a generation timing owned by a node that emits no
    message, image pointers and sandbox links becoming attachments, declared
    sibling order, and timestamp ordering with an untimestamped node.

    Anti-vacuity: ``_refuse_collecting_chatgpt_entries`` fails the route if it
    normalizes through the list store; drop any spilled rule and the parity
    comparison below turns red.
    """
    builder = ChatGPTExportBuilder("complex-spill").title("Complex spill")
    builder.add_node("user", "Plot the data", node_id="u1")
    builder.add_node(
        "assistant",
        "Running code",
        node_id="a1",
        metadata={
            "reasoning_start_time": 10.0,
            "reasoning_end_time": 14.5,
            "model_slug": "model-a",
        },
    )
    builder.add_node("tool", "stdout line", node_id="t1", metadata={"name": "python"})
    builder.add_node("tool", "second result", node_id="t2", metadata={"name": "python"})
    builder.add_node(
        "assistant",
        "Saved [chart](sandbox:/mnt/data/chart.png) and [table](sandbox:/mnt/data/t.csv)",
        node_id="a2",
        metadata={"finished_duration_sec": 3.25, "model_slug": "model-b"},
    )
    builder.add_node("user", "Alternative", node_id="u2")
    record = builder.build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    parents = {"a1": "u1", "t1": "a1", "t2": "t1", "a2": "t2", "u2": "u1"}
    for key, parent in parents.items():
        mapping[key]["parent"] = parent
    mapping["u1"]["children"] = ["u2", "a1"]
    a1 = mapping["a1"]["message"]
    a1["channel"] = "commentary"
    a1["author"] = {"role": "assistant", "metadata": {"real_author": "tool:web.run"}}
    a1["metadata"]["aggregate_result"] = {
        "status": "failed_with_in_kernel_exception",
        "run_id": "run-1",
        "start_time": 11.0,
        "end_time": 12.0,
        "in_kernel_exception": {"name": "ValueError", "args": ["bad"]},
        "messages": [{"message_type": "stream", "text": "partial"}],
    }
    mapping["u1"]["message"]["content"] = {
        "content_type": "multimodal_text",
        "parts": [
            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-img", "size_bytes": 7},
            "Plot the data",
        ],
    }
    mapping["u2"]["message"]["create_time"] = None
    # A timing on a node whose message is refused (no role) is owned by the
    # latest emitted message of its branch instead.
    mapping["orphan"] = {
        "id": "orphan",
        "parent": "a2",
        "children": [],
        "message": {
            "id": "orphan",
            "author": {"role": "tool"},
            "create_time": 20.0,
            "content": {"content_type": "text", "parts": [""]},
            "metadata": {"finished_duration_sec": 1.0},
        },
    }
    record["current_node"] = "a2"
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    _refuse_collecting_chatgpt_entries(monkeypatch)
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
    assert isinstance(actual.messages, SqliteMessageSink)
    collected = actual.model_copy(
        update={"attachments": list(actual.attachments), "session_events": list(actual.session_events)}
    )
    assert collected.model_dump(mode="json", exclude={"content_hash", "messages"}) == expected.model_dump(
        mode="json", exclude={"content_hash", "messages"}
    )
    assert _stored_messages(actual) == _stored_messages(expected)
    assert {event.event_type for event in expected.session_events} >= {
        "generation_lifecycle",
        "chatgpt_code_interpreter_run",
        "chatgpt_message_authorship",
    }
    assert len(expected.attachments) >= 3
    assert actual.content_hash == session_content_hash(expected)
    # Parser-only scratch (Codex P2, #5643): the sealed artifact carries no
    # sibling-order or string-map table, only the publication tables.
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not tables & {"chatgpt_sibling", "chatgpt_node", "chatgpt_child", "scratch_string_map"}
    artifact.discard()


def _lower_storable_limit(monkeypatch: pytest.MonkeyPatch, bound: int) -> None:
    """Stand in for SQLite's value limit so a test needs no gigabyte string."""
    import polylogue.sources.value_bounds as value_bounds

    monkeypatch.setattr(value_bounds, "MAX_STORABLE_VALUE_BYTES", bound)


@pytest.mark.parametrize("oversized", ["part", "title", "mapping_key"])
def test_chatgpt_object_refuses_a_value_sqlite_cannot_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized: str
) -> None:
    """An unstorable scalar refuses the whole object, typed and not retryable.

    Anti-vacuity: remove the check from ``normalize_ijson_stdlib_numbers`` or
    ``require_storable_string`` and the object prepares with the value intact.
    """
    _lower_storable_limit(monkeypatch, 64)
    giant = "x" * 65
    builder = ChatGPTExportBuilder("bounded").add_node("user", giant if oversized == "part" else "Prompt")
    record = builder.build()
    if oversized == "title":
        record["title"] = giant
    if oversized == "mapping_key":
        mapping = record["mapping"]
        assert isinstance(mapping, dict)
        node = mapping.pop("node-1")
        mapping[giant] = node
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
    assert artifact.error is not None
    assert artifact.error.startswith("ValueBoundRefusedError: value_bound_refused:")
    assert artifact.deferred is False
    assert list((tmp_path / "scratch").glob("prepared-*.db")) == []


def test_streamed_generic_object_refuses_a_value_sqlite_cannot_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other object-streaming routes share the limit through the decoder.

    Anti-vacuity: remove the string arm from ``normalize_ijson_stdlib_numbers``
    and the generic ``messages`` route prepares the oversized part.
    """
    _lower_storable_limit(monkeypatch, 64)
    record = {
        "id": "generic-bounded",
        "messages": [
            {"role": "user", "content": "Prompt"},
            {"role": "assistant", "content": "y" * 65},
        ],
    }
    source = tmp_path / "generic.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.UNKNOWN.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is not None
    assert "value_bound_refused" in artifact.error
    assert artifact.deferred is False


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
        Provider.CHATGPT, list(iter_owned_json_values(BytesIO(source.read_bytes()), source.name)), "fallback"
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

    bootstrap_archive_root(tmp_path)
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

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Design object decoded as a whole document")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse_whole_document)
    with retained_parser_fixture(
        root=Path(str(source_db)).parent,
        provider=Provider.from_string(Provider.CLAUDE_DESIGN.value),
        blob_hash=blob_hash,
        source_path=str(tmp_path / "design_chats" / "session.json"),
        directory=Path(str(tmp_path / "prepared")),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (artifact, _reader):
        assert artifact.error is None
        assert artifact.positive_evidence_filtered
        [actual] = artifact.iter_sessions()
        assert actual.provider_session_id == "design-retained"
        assert len(actual.messages) == 300
        assert len(actual.session_events) == 300
        assert actual.created_at == "2026-01-01T00:00:00+00:00"
        assert actual.updated_at == "2026-01-01T00:00:59+00:00"


def test_event_sink_batch_insert_matches_sequential_inserts(tmp_path: Path) -> None:
    """Anti-vacuity: number batch insertions from the post-insert sequence and
    the order differs from sequential ``insert`` calls at shifted indices."""
    from polylogue.sources.parsers.base import ParsedSessionEvent
    from polylogue.sources.prepared_message_sink import SqliteMessageStore

    store = SqliteMessageStore(tmp_path / "events.db")
    try:
        sink = store.new_event_sink()
        expected = [ParsedSessionEvent(event_type=f"e{index}", payload={}) for index in range(6)]
        for event in expected:
            sink.append(event)
        insertions = [(0, "a"), (2, "b"), (2, "c"), (6, "d")]
        sink.insert_sorted((index, ParsedSessionEvent(event_type=name, payload={})) for index, name in insertions)
        for offset, (index, name) in enumerate(insertions):
            expected.insert(index + offset, ParsedSessionEvent(event_type=name, payload={}))
        assert [event.event_type for event in sink] == [event.event_type for event in expected]
    finally:
        store.close()


def test_event_sink_early_batch_insert_renumbers_in_linear_time(tmp_path: Path) -> None:
    """An early compaction's many contexts ahead of many later events stays fast.

    Anti-vacuity: renumber each existing event by counting the insertions
    before it and this case visits ~9e8 index entries, taking minutes.
    """
    import time

    from polylogue.sources.parsers.base import ParsedSessionEvent
    from polylogue.sources.prepared_message_sink import SqliteMessageStore

    count = 30_000
    store = SqliteMessageStore(tmp_path / "events.db")
    try:
        sink = store.new_event_sink()
        for index in range(count):
            sink.append(ParsedSessionEvent(event_type=f"e{index}", payload={}))
        started = time.perf_counter()
        sink.insert_sorted(
            (index // 2, ParsedSessionEvent(event_type=f"c{index}", payload={})) for index in range(count)
        )
        assert time.perf_counter() - started < 30
        events = [event.event_type for event in sink]
        assert len(events) == 2 * count
        assert events[:3] == ["c0", "c1", "e0"]
        assert events[-1] == f"e{count - 1}"
    finally:
        store.close()


def test_sink_json_keeps_literal_escape_text_and_json_mode_fields(tmp_path: Path) -> None:
    """Literal ``\\ud800`` text is not a surrogate escape, and a real one keeps
    JSON-mode conversions such as hex digests.

    Anti-vacuity: treat the literal text as an escape and validate in Python
    mode, and the paste evidence digest comes back as its hex text.
    """
    from polylogue.sources.parsers.base_models import ParsedPasteEvidence
    from polylogue.sources.prepared_message_sink import _from_text_json, _message_json

    for text in ("literal \\ud800 text", "real \ud800 surrogate"):
        evidence = ParsedPasteEvidence(content_hash=b"\x01" * 32, source_marker=text)
        message = ParsedMessage(provider_message_id="m1", role=Role.USER, text=text, paste_spans=[evidence])
        assert _from_text_json(ParsedMessage, _message_json(message)).paste_spans == [evidence]


def test_sink_surrogate_decode_keeps_excluded_parser_coordinates() -> None:
    """A surrogate-bearing message keeps its parser-only coordinates.

    Anti-vacuity: drop the excluded-field carry-over in ``_from_text_json`` and
    ``parent_message_position`` comes back ``None``.
    """
    from polylogue.sources.prepared_message_sink import _from_text_json, _message_json

    message = ParsedMessage(provider_message_id="m2", role=Role.USER, text="real \ud800 surrogate")
    message = message.model_copy(update={"parent_message_position": 1})
    decoded = _from_text_json(ParsedMessage, _message_json(message))

    assert decoded.text == "real \ud800 surrogate"
    assert decoded.parent_message_position == 1


def test_storable_value_limit_is_sqlites_own() -> None:
    """The refusal threshold is the linked SQLite's value limit, in UTF-8 bytes.

    Anti-vacuity: replace ``MAX_STORABLE_VALUE_BYTES`` with a chosen number or
    count characters instead of encoded bytes and one of these asserts fails.
    """
    import polylogue.sources.value_bounds as value_bounds

    with sqlite3.connect(":memory:") as conn:
        assert conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) == value_bounds.MAX_STORABLE_VALUE_BYTES
    four_byte = "\U0001f600" * 4
    assert value_bounds.require_storable_string(four_byte, bound=16) == four_byte
    with pytest.raises(value_bounds.ValueBoundRefusedError):
        value_bounds.require_storable_string(four_byte + "x", bound=16)


def test_browser_capture_envelope_with_a_mapping_keeps_the_capture_route(tmp_path: Path) -> None:
    """Preparation keeps the dispatcher's detector precedence.

    Anti-vacuity: drop the ``browser_capture.looks_like`` guard on the
    ChatGPT mapping spill and the decoy mapping wins, sealing zero messages
    instead of the envelope's captured turns.
    """
    from polylogue.sources.parsers import browser_capture

    payload = json.loads(Path("tests/fixtures/chatgpt/native-browser-capture-v1.json").read_text(encoding="utf-8"))
    payload["conversation_id"] = "decoy-conversation"
    payload["id"] = "decoy-conversation"
    payload["create_time"] = 1_700_000_000.0
    payload["current_node"] = "decoy-chatgpt-node"
    payload["mapping"] = {"decoy-chatgpt-node": {"id": "decoy-chatgpt-node", "parent": None, "children": []}}
    assert browser_capture.looks_like(payload)
    expected = parse_payload(Provider.CHATGPT, [payload], "fallback")
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())
    assert [len(session.messages) for session in actual] == [len(session.messages) for session in expected]
    assert all(len(session.messages) > 0 for session in actual)
    artifact.discard()


def test_sibling_index_includes_appended_tails(tmp_path: Path) -> None:
    """A sibling grown by append ingest contributes its accepted tail's tool ids.

    The chain is the newest full revision plus the appends descended from
    it. A later append for the same start offset competes with the first;
    the chain stops there rather than include a superseded branch. An
    append of an older full revision never extends a newer one, even when
    the newer one has the length the append starts at.

    Anti-vacuity: select only the full revision in
    ``RetainedSidecarResolver._retained_siblings`` and ``toolu_tail`` is
    missing; concatenate every append and ``toolu_branch`` appears; chain by
    offset alone and ``toolu_stale`` joins the newer revision of ``agent-c``.
    """

    bootstrap_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    blob_root = tmp_path / "blob"
    store = BlobStore(blob_root)
    session_dir = tmp_path / "project" / "session-1"
    tail_sibling = (session_dir / "subagents" / "agent-a.jsonl").as_posix()
    branch_sibling = (session_dir / "subagents" / "agent-b.jsonl").as_posix()
    rebased_sibling = (session_dir / "subagents" / "agent-c.jsonl").as_posix()

    def tool_use(tool_id: str) -> bytes:
        return (
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}]},
                }
            ).encode()
            + b"\n"
        )

    base_hash, base_size = store.write_from_bytes(tool_use("toolu_base"))
    tail_hash, tail_size = store.write_from_bytes(tool_use("toolu_tail"))
    branch_hash, branch_size = store.write_from_bytes(tool_use("toolu_branch"))
    newer_hash, newer_size = store.write_from_bytes(tool_use("toolu_newr"))
    stale_hash, stale_size = store.write_from_bytes(tool_use("toolu_stale"))
    assert newer_size == base_size
    rows = (
        ("a-base", tail_sibling, base_hash, base_size, "full", 1, None, None, None),
        ("a-tail", tail_sibling, tail_hash, tail_size, "append", 2, base_size, base_size + tail_size, "a-base"),
        ("b-base", branch_sibling, base_hash, base_size, "full", 1, None, None, None),
        ("b-one", branch_sibling, tail_hash, tail_size, "append", 2, base_size, base_size + tail_size, "b-base"),
        (
            "b-two",
            branch_sibling,
            branch_hash,
            branch_size,
            "append",
            3,
            base_size,
            base_size + branch_size + 7,
            "b-base",
        ),
        ("c-old", rebased_sibling, base_hash, base_size, "full", 1, None, None, None),
        ("c-stale", rebased_sibling, stale_hash, stale_size, "append", 2, base_size, base_size + stale_size, "c-old"),
        ("c-new", rebased_sibling, newer_hash, newer_size, "full", 3, None, None, None),
    )
    with sqlite3.connect(source_db) as conn:
        for raw_id, path, blob_hash, size, kind, acquired, start, end, predecessor in rows:
            conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, "
                "revision_kind, append_start_offset, append_end_offset, predecessor_raw_id) "
                "VALUES (?, 'claude-code-session', ?, ?, ?, ?, ?, ?, ?, ?)",
                (raw_id, path, bytes.fromhex(blob_hash), size, acquired, kind, start, end, predecessor),
            )
            # Each observation records its raw_payload receipt, in observation order.
            conn.execute(
                "INSERT OR REPLACE INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, "
                "acquired_at_ms) VALUES (?, ?, 'raw_payload', ?, ?, ?)",
                (bytes.fromhex(blob_hash), raw_id, path, size, acquired),
            )
    with prepared_source_fixture(tmp_path) as reader:
        resolver = reader.retained_sidecar_resolver()
        from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

        assert isinstance(resolver, RetainedSidecarResolver)
        siblings = resolver._retained_siblings(None, session_dir.parent / "session-1.jsonl")
        tool_ids = {
            sibling.coordinate: {
                block["id"]
                for record in sibling.open_records()
                if isinstance(record, dict)
                for block in record["message"]["content"]
            }
            for sibling in siblings
        }
    assert tool_ids == {
        tail_sibling: {"toolu_base", "toolu_tail"},
        branch_sibling: {"toolu_base"},
        rebased_sibling: {"toolu_newr"},
    }


def test_a_historical_append_does_not_extend_a_reselected_baseline(tmp_path: Path) -> None:
    """``A -> A+X -> B -> A``: the current ``A`` is not followed by the old ``X``.

    Anti-vacuity (Codex P1, #5643): follow any append whose predecessor and
    offsets match and ``toolu_x`` from the historical tail reappears, though
    the live ``A`` file holds no such call.
    """

    bootstrap_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    blob_root = tmp_path / "blob"
    store = BlobStore(blob_root)
    session_dir = tmp_path / "project" / "session-1"
    sibling = (session_dir / "subagents" / "agent-a.jsonl").as_posix()

    def tool_use(tool_id: str) -> bytes:
        return (
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}]},
                }
            ).encode()
            + b"\n"
        )

    a_hash, a_size = store.write_from_bytes(tool_use("toolu_a"))
    x_hash, x_size = store.write_from_bytes(tool_use("toolu_x"))
    b_hash, b_size = store.write_from_bytes(tool_use("toolu_b") + tool_use("toolu_b2"))
    rows = {
        "a": (a_hash, a_size, "full", None, None, None),
        "x": (x_hash, x_size, "append", a_size, a_size + x_size, "a"),
        "b": (b_hash, b_size, "full", None, None, None),
    }
    with sqlite3.connect(source_db) as conn:
        for raw_id, (blob_hash, size, kind, start, end, predecessor) in rows.items():
            conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, "
                "revision_kind, append_start_offset, append_end_offset, predecessor_raw_id) "
                "VALUES (?, 'claude-code-session', ?, ?, ?, 1, ?, ?, ?, ?)",
                (raw_id, sibling, bytes.fromhex(blob_hash), size, kind, start, end, predecessor),
            )
        # Observed A, A+X, B, then A again: the returning A re-records its
        # receipt (INSERT OR REPLACE, as ``source_write`` does), which makes it
        # the newest observation.
        for observed_ms, raw_id in enumerate(("a", "x", "b", "a"), start=1):
            blob_hash, size, *_rest = rows[raw_id]
            conn.execute(
                "INSERT OR REPLACE INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, "
                "acquired_at_ms) VALUES (?, ?, 'raw_payload', ?, ?, ?)",
                (bytes.fromhex(blob_hash), raw_id, sibling, size, observed_ms),
            )
    with prepared_source_fixture(tmp_path) as reader:
        resolver = reader.retained_sidecar_resolver()
        from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

        assert isinstance(resolver, RetainedSidecarResolver)
        (found,) = resolver._retained_siblings(None, session_dir.parent / "session-1.jsonl")
        tool_ids = {
            block["id"]
            for record in found.open_records()
            if isinstance(record, dict)
            for block in record["message"]["content"]
        }

    assert tool_ids == {"toolu_a"}


def test_sibling_baseline_follows_the_newest_durable_receipt(tmp_path: Path) -> None:
    """A sibling whose bytes returned to an earlier value replays that value.

    Content-addressed admission reuses revision A's raw row when a sibling
    goes A -> B -> A, so ``raw_sessions.acquired_at_ms`` still orders B last;
    A's re-recorded ``raw_payload`` receipt is the newest in receipt order
    (``raw_receipt_order_sql``) and names A as current.

    Anti-vacuity (Codex P1, #5643): rank full revisions by
    ``raw_sessions.acquired_at_ms`` and the baseline is B, so the sibling
    index names ``toolu_b`` instead of the current file's ``toolu_a``.
    """

    bootstrap_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    blob_root = tmp_path / "blob"
    store = BlobStore(blob_root)
    session_dir = tmp_path / "project" / "session-1"
    sibling = (session_dir / "subagents" / "agent-a.jsonl").as_posix()

    def tool_use(tool_id: str) -> bytes:
        record = {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}]},
        }
        return json.dumps(record).encode() + b"\n"

    a_hash, a_size = store.write_from_bytes(tool_use("toolu_a"))
    b_hash, b_size = store.write_from_bytes(tool_use("toolu_b"))
    revisions = {"rev-a": (a_hash, a_size, 1), "rev-b": (b_hash, b_size, 2)}
    with sqlite3.connect(source_db) as conn:
        for raw_id, (blob_hash, size, first_seen) in revisions.items():
            conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, "
                "revision_kind) VALUES (?, 'claude-code-session', ?, ?, ?, ?, 'full')",
                (raw_id, sibling, bytes.fromhex(blob_hash), size, first_seen),
            )
        # A, B, then A again: the returning A re-records its receipt.
        for observed_ms, raw_id in enumerate(("rev-a", "rev-b", "rev-a"), start=1):
            blob_hash, size, _first_seen = revisions[raw_id]
            conn.execute(
                "INSERT OR REPLACE INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, "
                "acquired_at_ms) VALUES (?, ?, 'raw_payload', ?, ?, ?)",
                (bytes.fromhex(blob_hash), raw_id, sibling, size, observed_ms),
            )
    with prepared_source_fixture(tmp_path) as reader:
        resolver = reader.retained_sidecar_resolver()
        from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

        assert isinstance(resolver, RetainedSidecarResolver)
        (resolved,) = resolver._retained_siblings(None, session_dir.parent / "session-1.jsonl")
        tool_ids = {
            block["id"]
            for record in resolved.open_records()
            if isinstance(record, dict)
            for block in record["message"]["content"]
        }
    assert tool_ids == {"toolu_a"}


def test_chatgpt_spill_keeps_every_per_node_collection_in_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Events, sibling ordinals and sandbox dedup stay in scratch; order matches.

    Anti-vacuity: each monkeypatch below fails if the spilled parse falls
    back to an in-memory collection: ``_sibling_ordinals`` builds a dict over
    every node, a builtin ``set`` dedups sandbox links, and the admission
    wrapper's ``list(session_events)`` replaces the scratch event sink. The
    ``"nan"`` timestamp pins the finite-timestamp rule: with NaN kept as a
    float, the collecting sort and SQLite (NaN stored as NULL) order the
    messages differently.
    """
    builder = ChatGPTExportBuilder("scratch-only")
    builder.add_node("user", "first", node_id="n1")
    builder.add_node("assistant", "see sandbox:/mnt/data/a.txt and sandbox:/mnt/data/b.txt", node_id="n2")
    builder.add_node("user", "third", node_id="n3")
    record = builder.build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    mapping["n1"]["message"]["create_time"] = "nan"
    mapping["n2"]["message"]["create_time"] = 2.0
    mapping["n3"]["message"]["create_time"] = 1.0
    mapping["n2"]["message"]["metadata"] = {"targeted_reply": "quoted"}
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]

    def refuse_ordinals(_mapping: object) -> object:
        raise AssertionError("sibling ordinals collected in memory")

    original_paths = chatgpt._sandbox_file_paths

    def scratch_seen_only(text: str, seen: object = None) -> object:
        assert seen is not None and not isinstance(seen, set), "sandbox dedup used an in-memory set"
        return original_paths(text, seen)  # type: ignore[arg-type]

    monkeypatch.setattr(chatgpt, "_sibling_ordinals", refuse_ordinals)
    monkeypatch.setattr(chatgpt, "_sandbox_file_paths", scratch_seen_only)
    store = SqliteMessageStore(tmp_path / "scratch.db")
    try:
        (tmp_path / "chatgpt.json").write_text(json.dumps(record), encoding="utf-8")
        with (tmp_path / "chatgpt.json").open("rb") as source_handle:
            read = read_chatgpt_mapping_object(source_handle, store.conn)
        assert read is not None
        envelope, node_mapping = read
        session = chatgpt.parse(
            {**envelope, "mapping": node_mapping.shallow_view()}, "fallback", spill=ScratchSessionSpill(store)
        )
        assert isinstance(session.session_events, SqliteSessionEventSink)
        assert [message.provider_message_id for message in session.messages] == [
            message.provider_message_id for message in expected.messages
        ]
        assert [event.model_dump(mode="json") for event in session.session_events] == [
            event.model_dump(mode="json") for event in expected.session_events
        ]
        assert len(session.attachments) == len(expected.attachments) == 2
    finally:
        store.close()


def test_chatgpt_node_mapping_numbers_sibling_ordinals_in_one_scan(tmp_path: Path) -> None:
    """Scratch sibling ordinals equal ``_sibling_ordinals`` and cost one scan.

    Anti-vacuity: a per-lookup prefix ``COUNT(*)`` (quadratic over a wide
    sibling set) shows up as traced ``COUNT`` statements; skipping the
    renumbering after a later ``put`` returns the stale ordinal of ``b``,
    whose re-put moved it to another parent.
    """
    nodes: dict[str, object] = {
        "root": {"id": "root", "parent": None},
        "a": {"id": "a", "parent": "root"},
        "b": {"id": "b", "parent": "root"},
        "c": {"id": "c", "parent": "root"},
        "d": {"id": "d", "parent": "a"},
    }
    conn = sqlite3.connect(tmp_path / "scratch.db")
    try:
        mapping = ChatGPTNodeMapping(conn)
        for ordinal, (key, node) in enumerate(nodes.items()):
            mapping.put(key, node, ordinal)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        assert {key: mapping.sibling_ordinal(key) for key in nodes} == chatgpt._sibling_ordinals(nodes)
        assert sum("INSERT INTO chatgpt_sibling" in sql for sql in statements) == 1
        assert not any("COUNT(" in sql for sql in statements)
        conn.set_trace_callback(None)

        nodes["b"] = {"id": "b", "parent": "a"}
        mapping.put("b", nodes["b"], 5)
        assert {key: mapping.sibling_ordinal(key) for key in nodes} == chatgpt._sibling_ordinals(nodes)
    finally:
        conn.close()


def test_chatgpt_refuses_values_that_only_combine_past_the_cell_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Individually storable parts that combine into an unstorable cell refuse.

    Anti-vacuity: drop the serialized-record check in ``_message_json`` (and
    ``ChatGPTNodeMapping.put``) and preparation stores the combined record.
    """
    _lower_storable_limit(monkeypatch, 200)
    builder = ChatGPTExportBuilder("combined").add_node("user", "a" * 150, "b" * 150)
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(builder.build()), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CHATGPT.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "scratch"),
    )
    assert artifact.error is not None and "value_bound_refused" in artifact.error
    assert artifact.deferred is False


@pytest.mark.parametrize(
    "envelope",
    [
        "Output too large. Showing first , and last , characters",
        "Output too large. Showing first " + "9" * 5000 + " and last 1 characters",
    ],
)
def test_gemini_malformed_excerpt_counts_are_an_unquantified_mask(tmp_path: Path, envelope: str) -> None:
    """A mask line whose counts are not numbers still observes the tool call.

    Anti-vacuity: convert the comma-only or 5000-digit count with ``int`` and
    ``observe`` raises ``ValueError``, aborting the whole session.
    """
    from polylogue.sources.prepared_message_sink import GeminiToolOutputIndex

    conn = sqlite3.connect(tmp_path / "scratch.db")
    try:
        index = GeminiToolOutputIndex(conn)
        index.observe(
            {"toolCalls": [{"id": "call-1", "result": [{"functionResponse": {"response": {"output": envelope}}}]}]}
        )
        assert conn.execute("SELECT masked, complete_len FROM gemini_tool_owner").fetchall() == [(1, 1)]
    finally:
        conn.close()


def test_gemini_sidecar_decoded_past_the_cell_limit_is_typed_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar under the byte limit that decodes past it is refused as debt.

    Anti-vacuity: store the decoded text unchecked and the oversized
    replacement reaches SQLite (an untyped ``DataError`` at the real limit)
    instead of a ``value_bound_refused`` debt row.
    """
    from polylogue.sources import value_bounds
    from polylogue.sources.live.tool_result_sidecars import SidecarMatch
    from polylogue.sources.prepared_message_sink import GeminiToolOutputIndex

    monkeypatch.setattr(value_bounds, "MAX_STORABLE_VALUE_BYTES", 64)
    conn = sqlite3.connect(tmp_path / "scratch.db")
    try:
        index = GeminiToolOutputIndex(conn)
        index.observe({"toolCalls": [{"id": "call-1", "result": []}]})
        # 30 invalid bytes decode to 30 U+FFFD characters: 90 UTF-8 bytes.
        scope = RetainedSidecarScope(
            scope_key="scope",
            files=(
                RetainedSidecarFile(
                    filename="call-1.txt", byte_size=30, file_mtime_ms=None, read_text=lambda: "�" * 30
                ),
            ),
            available=True,
        )
        results = list(index.join(scope))
        assert not any(isinstance(result, SidecarMatch) for result in results)
        assert conn.execute("SELECT filename, reason FROM gemini_tool_debt").fetchall() == [
            ("call-1.txt", "value_bound_refused")
        ]
    finally:
        conn.close()


def test_chatgpt_message_attachment_ids_read_each_attachment_once() -> None:
    """Pointer dedupe reads each of a message's attachments once in total.

    Anti-vacuity: rescan ``attachments[start:]`` per pointer and the reads
    grow quadratically with the message's attachments and pointers.
    """
    reads = 0

    class CountingList(list[ParsedAttachment]):
        def __getitem__(self, index: object) -> object:  # type: ignore[override]
            nonlocal reads
            reads += 1
            return super().__getitem__(cast(int, index))

    count = 400
    attachments = CountingList(
        ParsedAttachment(provider_attachment_id=f"file-{index}", message_provider_id="m1") for index in range(count)
    )
    index = chatgpt._MessageAttachmentIds(attachments, 0, {})
    assert [index.find(f"file-{position}") for position in range(count)] == list(range(count))
    assert index.find("file-missing") is None
    assert reads == count


def test_chatgpt_scratch_row_past_the_length_limit_is_typed(tmp_path: Path) -> None:
    """A normalized message row SQLite refuses as too big is a typed refusal.

    Anti-vacuity: insert without typing the refusal and the parse fails with
    an untyped ``sqlite3.DataError``.
    """
    from polylogue.sources.prepared_message_sink import _ScratchChatGPTEntries
    from polylogue.sources.value_bounds import ValueBoundRefusedError

    store = SqliteMessageStore(tmp_path / "scratch.db")
    try:
        if not hasattr(store.conn, "setlimit"):
            pytest.skip("this sqlite driver cannot lower its length limit")
        entries = _ScratchChatGPTEntries(store.conn)
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4_000)
        message = ParsedMessage(provider_message_id="m1", role=Role.USER, text="x" * 1_500, position=0)
        with pytest.raises(ValueBoundRefusedError, match="value_bound_refused"):
            entries.add(None, 1, "n" * 3_000, message)
    finally:
        store.close()


def test_chatgpt_generation_timings_live_in_the_spill_database(tmp_path: Path) -> None:
    """The spilled parse selects generation timings in its scratch database.

    Anti-vacuity: select them in process memory and the scratch database has
    no candidate row for the timed generation.
    """
    builder = ChatGPTExportBuilder("timed")
    builder.add_node("user", "question", node_id="n1")
    builder.add_node("assistant", "answer", node_id="n2")
    record = builder.build()
    mapping = record["mapping"]
    assert isinstance(mapping, dict)
    mapping["n2"]["message"]["metadata"] = {"finished_duration_sec": 2.5}
    expected = parse_payload(Provider.CHATGPT, [record], "fallback")[0]
    store = SqliteMessageStore(tmp_path / "scratch.db")
    try:
        session = chatgpt.parse(record, "fallback", spill=ScratchSessionSpill(store))
        assert store.conn.execute("SELECT COUNT(*) FROM temp.gt_candidate").fetchone()[0] == 1
        assert [message.duration_ms for message in session.messages] == [
            message.duration_ms for message in expected.messages
        ]
        assert [event.model_dump(mode="json") for event in session.session_events] == [
            event.model_dump(mode="json") for event in expected.session_events
        ]
    finally:
        store.close()


def test_an_unused_oversized_bundle_field_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A member field no parser stores is never held to SQLite's cell limit.

    Anti-vacuity (Codex P2, #5643): bound every decoded scalar and a
    conversation with one oversized ignored field is refused whole.
    """
    from polylogue.sources import value_bounds
    from polylogue.sources.decoder_json import iter_json_container_records

    monkeypatch.setattr(value_bounds, "MAX_STORABLE_VALUE_BYTES", 16)
    payload = b'[{"id": "c1", "ignored": "' + b"x" * 64 + b'"}]'
    (record,) = list(iter_json_container_records(BytesIO(payload), "item"))
    assert isinstance(record, dict) and record["id"] == "c1"


def test_removing_a_scratch_tree_releases_its_decodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: removing replay scratch without evicting leaves the walked
    session's decode resident, so the next walk decodes nothing."""
    from polylogue.sources import prepared_message_sink

    artifact, _coordinate = _prepared_artifact(tmp_path)
    (session,) = list(artifact.iter_sessions())
    decodes = _count_message_decodes(monkeypatch)
    list(session.messages)
    prepared_message_sink.discard_decoded_sessions_under(tmp_path)
    list(session.messages)
    assert decodes[0] == 2


def test_sink_surrogate_decode_parses_once_without_a_dump_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A surrogate-bearing row is parsed once and validated, with no marked
    copy, dump or restore pass over a possibly near-limit value.

    Anti-vacuity: validate a marked copy and dump it to restore surrogates,
    and the patched ``model_dump`` fails the decode.
    """
    from polylogue.sources.prepared_message_sink import _from_text_json, _message_json

    message = ParsedMessage(provider_message_id="m3", role=Role.USER, text="x" * 4096 + "\ud800")
    encoded = _message_json(message)

    def no_dump(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dumped the validated row to restore surrogates")

    monkeypatch.setattr(ParsedMessage, "model_dump", no_dump)
    assert _from_text_json(ParsedMessage, encoded).text == message.text


@pytest.mark.parametrize(
    "cause",
    [OSError("source unavailable"), AssertionError("parser invariant"), RuntimeError("worker failure")],
)
def test_non_decode_stream_failures_remain_retryable(cause: BaseException) -> None:
    from polylogue.sources.decoder_json import PartialJsonStreamError
    from polylogue.sources.prepared_jsonl import classify_decode_failure, terminal_decode_evidence

    error = PartialJsonStreamError("synthetic.json", recovered=1, offset=None, cause=cause)
    assert classify_decode_failure(error) is None
    assert terminal_decode_evidence(error, provider=Provider.CHATGPT) is None
    assert terminal_decode_evidence(error, provider=Provider.UNKNOWN) is None


@pytest.mark.parametrize("count", [24, 192])
def test_gemini_checkpoint_preparation_spools_records_before_eof_without_retaining_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    import gc
    import weakref
    from collections.abc import Iterator

    import polylogue.sources.prepared_jsonl as prepared

    source = tmp_path / "checkpoint.jsonl"
    header = {"sessionId": "bounded", "projectHash": "neutral", "kind": "main", "startTime": "2026-05-02T09:00:00.000Z"}
    records = [
        header,
        *(
            {
                "id": f"message-{index}",
                "type": "user",
                "timestamp": "2026-05-02T09:00:01.000Z",
                "content": f"neutral-{index}",
            }
            for index in range(count)
        ),
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    del records
    decoded = 0
    live = 0
    peak = 0
    first_append: int | None = None
    original_records = prepared.owned_json_records
    original_append = prepared._append_gemini_raw_message

    class TrackedRecord(dict[str, JSONValue]):
        pass

    def retired() -> None:
        nonlocal live
        live -= 1

    @contextmanager
    def observe_records(
        handle: BinaryIO | IO[bytes], path_name: str, unpack_lists: bool = True, *, fail_on_decode_error: bool = False
    ) -> Generator[Iterable[object], None, None]:
        with original_records(
            handle, path_name, unpack_lists=unpack_lists, fail_on_decode_error=fail_on_decode_error
        ) as records:

            def tracked_records() -> Iterator[object]:
                nonlocal decoded, live, peak
                for record in records:
                    assert isinstance(record, dict)
                    gc.collect()
                    tracked = TrackedRecord(record)
                    live += 1
                    peak = max(peak, live)
                    weakref.finalize(tracked, retired)
                    decoded += 1
                    yield tracked

            yield tracked_records()

    def observe_append(conn: sqlite3.Connection, ordinal: int, item: object) -> None:
        nonlocal first_append
        if first_append is None:
            first_append = decoded
        original_append(conn, ordinal, item)

    def refuse_collected(*args: object, **kwargs: object) -> object:
        raise AssertionError("checkpoint fell back to whole-input parse")

    monkeypatch.setattr(prepared, "owned_json_records", observe_records)
    monkeypatch.setattr(prepared, "_append_gemini_raw_message", observe_append)
    monkeypatch.setattr(prepared, "iter_parsed_payload", refuse_collected)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "unused",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
    )
    assert artifact.error is None, artifact.error
    assert decoded == count + 1
    assert first_append == 2
    assert peak <= 4  # Header, parser's prior record, producer's current record.
    [session] = artifact.iter_sessions()
    assert session.provider_session_id == "bounded:main:2026-05-02T09:00:00.000Z"
    assert [message.provider_message_id for message in session.messages] == [
        f"message-{index}" for index in range(count)
    ]
    assert [message.text for message in session.messages] == [f"neutral-{index}" for index in range(count)]
    artifact.discard()


@pytest.mark.parametrize("suffix", ["valid", "invalid_record", "corrupt_json"])
def test_gemini_checkpoint_preparation_preserves_replacement_patch_and_suffix_refusal(
    tmp_path: Path, suffix: str
) -> None:
    header = {
        "sessionId": "replacement",
        "projectHash": "neutral",
        "kind": "main",
        "startTime": "2026-05-02T09:00:00.000Z",
    }
    before = {"id": "replaced", "type": "user", "timestamp": "2026-05-02T09:00:01.000Z", "content": "old"}
    after = {"id": "retained", "type": "user", "timestamp": "2026-05-02T09:00:01.000Z", "content": "new"}
    records = [header, before, {"$set": {"messages": [after], "lastUpdated": "2026-05-02T09:00:02.000Z"}}]
    source = tmp_path / "checkpoint.jsonl"
    text = "".join(json.dumps(record) + "\n" for record in records)
    if suffix == "invalid_record":
        text += '{"unrecognized":"neutral"}\n'
    elif suffix == "corrupt_json":
        text += '{"unfinished":\n'
    source.write_text(text, encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "unused",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        strict_jsonl_records=True,
    )
    if suffix == "valid":
        assert artifact.error is None, artifact.error
        [session] = artifact.iter_sessions()
        assert [message.provider_message_id for message in session.messages] == ["retained"]
        assert [message.text for message in session.messages] == ["new"]
        assert session.updated_at == "2026-05-02T09:00:02.000Z"
    elif suffix == "invalid_record":
        assert artifact.error is None, artifact.error
        assert list(artifact.iter_sessions()) == []
    else:
        assert artifact.error is not None
        assert artifact.sessions_path is None
    artifact.discard()


@pytest.mark.parametrize("replace_future", [False, True])
def test_gemini_checkpoint_preparation_admits_only_final_transcript_wire_evidence(
    tmp_path: Path, replace_future: bool
) -> None:
    header = {
        "sessionId": "wire-evidence",
        "projectHash": "neutral",
        "kind": "main",
        "startTime": "2026-05-02T09:00:00.000Z",
    }
    future = {
        "id": "future",
        "type": "future_neutral",
        "timestamp": "2026-05-02T09:00:01.000Z",
        "content": "future authored text",
    }
    known = {
        "id": "known",
        "type": "user",
        "timestamp": "2026-05-02T09:00:02.000Z",
        "content": "retained authored text",
    }
    records = [header, future, {"$set": {"messages": [known]}}] if replace_future else [header, known, future]
    source = tmp_path / "checkpoint.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GEMINI_CLI.value,
        "unused",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        strict_jsonl_records=True,
    )
    assert artifact.error is None, artifact.error
    [actual] = artifact.iter_sessions()
    [expected] = parse_payload(Provider.GEMINI_CLI, records, "unused", source_path=str(source))
    assert actual.unit_accounting == expected.unit_accounting
    assert [event.model_dump() for event in actual.session_events] == [
        event.model_dump() for event in expected.session_events
    ]
    assert [message.provider_message_id for message in actual.messages] == [
        message.provider_message_id for message in expected.messages
    ]
    assert actual.messages[0].text == "retained authored text"
    assert any(event.event_type == "gemini_cli_unknown_input" for event in actual.session_events) is (
        not replace_future
    )
    artifact.discard()


@pytest.mark.parametrize("provider", [Provider.DRIVE, Provider.GEMINI])
@pytest.mark.parametrize("count", [8, 64])
def test_bare_drive_jsonl_preparation_reuses_chunk_stream_without_retaining_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider, count: int
) -> None:
    import gc
    import weakref
    from collections.abc import Iterator

    import polylogue.sources.prepared_jsonl as prepared

    records = [{"id": f"chunk-{index}", "role": "user", "text": f"neutral-{index}"} for index in range(count)]
    expected = parse_payload(provider, records, "bare", source_path="bare.jsonl")[0]
    source = tmp_path / "bare.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    del records
    live = 0
    peak = 0
    decoded = 0
    original_records = prepared.owned_json_records

    class TrackedRecord(dict[str, JSONValue]):
        pass

    def retired() -> None:
        nonlocal live
        live -= 1

    @contextmanager
    def observe_records(
        handle: BinaryIO | IO[bytes], path_name: str, unpack_lists: bool = True, *, fail_on_decode_error: bool = False
    ) -> Generator[Iterable[object], None, None]:
        with original_records(
            handle, path_name, unpack_lists=unpack_lists, fail_on_decode_error=fail_on_decode_error
        ) as records:

            def tracked_records() -> Iterator[object]:
                nonlocal live, peak, decoded
                for record in records:
                    assert isinstance(record, dict)
                    gc.collect()
                    tracked = TrackedRecord(record)
                    live += 1
                    peak = max(peak, live)
                    decoded += 1
                    weakref.finalize(tracked, retired)
                    yield tracked

            yield tracked_records()

    def refuse_collected(*args: object, **kwargs: object) -> object:
        raise AssertionError("bare chunks fell back to whole-input parse")

    monkeypatch.setattr(prepared, "owned_json_records", observe_records)
    monkeypatch.setattr(prepared, "iter_parsed_payload", refuse_collected)
    artifact = prepare_jsonl_blob(
        str(source),
        "bare.jsonl",
        provider.value,
        "bare",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
    )
    try:
        assert artifact.error is None, artifact.error
        [session] = artifact.iter_sessions()
        assert decoded >= count * 2  # Eligibility and the canonical parser's original-source passes.
        assert peak <= 4
        assert [message.provider_message_id for message in session.messages] == [
            f"chunk-{index}" for index in range(count)
        ]
        assert [message.text for message in session.messages] == [f"neutral-{index}" for index in range(count)]
        assert _stored_messages(session) == _stored_messages(expected)
        assert list(session.session_events) == list(expected.session_events)
        assert session.content_hash == session_content_hash(expected)
    finally:
        artifact.discard()


@pytest.mark.parametrize("future_wire", [False, True])
def test_bare_drive_jsonl_preparation_preserves_future_wire_admission(tmp_path: Path, future_wire: bool) -> None:
    records = [{"id": "chunk", "role": "user", "text": "authored neutral text"}]
    if future_wire:
        records[0]["type"] = "future_neutral"
    expected = parse_payload(Provider.DRIVE, records, "bare", source_path="bare.jsonl")[0]
    source = tmp_path / "bare.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        "bare.jsonl",
        Provider.DRIVE.value,
        "bare",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    try:
        assert artifact.error is None, artifact.error
        [session] = artifact.iter_sessions()
        assert [message.text for message in session.messages] == ["authored neutral text"]
        assert session.unit_accounting == expected.unit_accounting
        assert list(session.session_events) == list(expected.session_events)
    finally:
        artifact.discard()


@pytest.mark.parametrize("shape", ["empty", "container", "browser", "mixed"])
def test_drive_chunk_stream_selection_preserves_existing_lowering_precedence(shape: str) -> None:
    from polylogue.browser_capture.models import BROWSER_CAPTURE_KIND, BROWSER_CAPTURE_SCHEMA_VERSION
    from polylogue.sources.dispatch import is_drive_chunk_sequence

    chunk: dict[str, JSONValue] = {"role": "user", "text": "neutral"}
    browser: dict[str, JSONValue] = {
        **chunk,
        "polylogue_capture_kind": BROWSER_CAPTURE_KIND,
        "schema_version": BROWSER_CAPTURE_SCHEMA_VERSION,
        "session": {},
        "provenance": {},
    }
    alternatives: dict[str, list[JSONValue]] = {
        "empty": [],
        "container": [chunk, {"chunks": []}],
        "browser": [browser],
        "mixed": [chunk, {"neutral_metadata": True}],
    }
    records = alternatives[shape]
    assert is_drive_chunk_sequence(iter(records)) is (shape == "mixed")


@pytest.mark.parametrize("provider", [Provider.DRIVE, Provider.GEMINI])
@pytest.mark.parametrize("count", [8, 64])
def test_bare_drive_json_array_preparation_streams_canonical_chunk_operands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider, count: int
) -> None:
    import gc
    import weakref

    import polylogue.sources.prepared_jsonl as prepared

    records = [{"id": f"chunk-{index}", "role": "user", "text": f"neutral-{index}"} for index in range(count)]
    records[-1]["type"] = "future_neutral"
    expected = parse_payload(provider, records, "bare", source_path="bare.json")[0]
    source = tmp_path / "bare.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    del records
    live = 0
    peak = 0
    decoded = 0
    from polylogue.sources.decoder_json import normalize_ijson_stdlib_numbers

    original_normalize = normalize_ijson_stdlib_numbers

    class TrackedRecord(dict[str, JSONValue]):
        pass

    def retired() -> None:
        nonlocal live
        live -= 1

    def observe_normalized(value: object) -> object:
        nonlocal live, peak, decoded
        normalized = original_normalize(value)
        if not isinstance(normalized, dict) or not str(normalized.get("id", "")).startswith("chunk-"):
            return normalized
        gc.collect()
        tracked = TrackedRecord(normalized)
        live += 1
        peak = max(peak, live)
        decoded += 1
        weakref.finalize(tracked, retired)
        return tracked

    def refuse_collected(*args: object, **kwargs: object) -> object:
        raise AssertionError("root chunks fell back to whole-input parse")

    monkeypatch.setattr(prepared, "normalize_ijson_stdlib_numbers", observe_normalized)
    monkeypatch.setattr(prepared, "iter_parsed_payload", refuse_collected)
    artifact = prepare_jsonl_blob(
        str(source),
        "bare.json",
        provider.value,
        "bare",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
    )
    try:
        assert artifact.error is None, artifact.error
        [session] = artifact.iter_sessions()
        assert decoded >= count * 2
        assert peak <= 4
        assert [message.provider_message_id for message in session.messages] == [
            f"chunk-{index}" for index in range(count)
        ]
        assert [message.text for message in session.messages] == [f"neutral-{index}" for index in range(count)]
        assert _stored_messages(session) == _stored_messages(expected)
        assert list(session.session_events) == list(expected.session_events)
        assert session.unit_accounting == expected.unit_accounting
        assert session.content_hash == session_content_hash(expected)
    finally:
        artifact.discard()


@pytest.mark.parametrize("wrapper", [False, True])
def test_decoded_record_tape_preserves_complete_wrapper_scope_and_replay(wrapper: bool) -> None:
    """Both decoder framings preserve late records and independently repeat ordinals."""
    from contextlib import closing

    from polylogue.sources.decoder_json import DecodedRecordSequence

    expected = [{"identity": str(index), "text": "neutral", "empty": []} for index in range(1201)]
    document = {"ignored": [0, False], "sessions": expected} if wrapper else expected
    with closing(
        DecodedRecordSequence(iter_owned_json_values(BytesIO(json.dumps(document).encode()), "neutral-wrapper.json"))
    ) as records:
        assert len(records) == 1201
        assert records[0] == expected[0]
        assert records[1200] == expected[1200]
        assert records[-1] == expected[-1]
        assert list(records) == expected
        assert list(records) == expected
    with pytest.raises(RuntimeError):
        len(records)


def test_nonseekable_decoder_keeps_borrowed_stream_open_and_preserves_late_records() -> None:
    class Nonseekable(BytesIO):
        def seekable(self) -> bool:
            return False

        def seek(self, *_args: object, **_kwargs: object) -> int:
            raise OSError("synthetic nonseekable source")

    expected = [{"ordinal": index, "text": "neutral"} for index in range(1201)]
    borrowed = Nonseekable(json.dumps(expected).encode())
    assert list(iter_owned_json_values(borrowed, "neutral-array.json")) == expected
    assert not borrowed.closed
    borrowed.close()


def test_record_recovery_preserves_stdlib_surrogates_nonfinite_and_last_wrapper() -> None:
    import math
    from contextlib import closing

    from polylogue.sources.decoder_json import DecodedRecordSequence

    document = b'{"sessions":[{"old":true}],"ignored":[1,2],"sessions":[{"id":"a"},{"id":"b","text":"\xed\xa0\x80","number":NaN}]}'
    with closing(DecodedRecordSequence(iter_owned_json_values(BytesIO(document), "neutral-wrapper.json"))) as records:
        assert len(records) == 2
        assert records[0] == {"id": "a"}
        last = records[1]
        assert isinstance(last, dict)
        assert last["id"] == "b" and last["text"] == "\ud800"
        assert isinstance(last["number"], float) and math.isnan(last["number"])


def test_atif_cohort_finalization_preserves_parent_first_and_all_child_headers(tmp_path: Path) -> None:
    document = {
        "schema_version": "ATIF-v1.7",
        "session_id": "neutral-parent",
        "steps": [{"source": "agent", "message": "Neutral parent observation"}],
        "subagent_trajectories": [
            {
                "session_id": f"neutral-child-{ordinal:04}",
                "steps": [{"source": "agent", "message": f"Neutral child observation {ordinal}"}],
            }
            for ordinal in range(1001)
        ],
    }
    source = tmp_path / "hermes" / "trajectory.json"
    source.parent.mkdir()
    source.write_text(json.dumps(document), encoding="utf-8")
    observed: list[str] = []

    def finalize(sessions: PreparedSessionSequence) -> Iterable[ParsedSession]:
        observed.extend(sessions.iter_provider_session_ids())
        assert len(sessions) == 1002
        return sessions

    import hashlib

    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=finalize,
    )
    try:
        assert artifact.error is None
        assert artifact.blob_hash == original_hash
        assert len(observed) == 1002
        assert "neutral-parent" in observed[0]
        assert "neutral-child-0000" in observed[1]
        assert "neutral-child-1000" in observed[-1]
        assert [session.provider_session_id for session in artifact.iter_sessions()] == observed
        assert artifact.sessions_path is not None
        with sqlite3.connect(artifact.sessions_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM prepared_session").fetchone()[0] == 1002
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM prepared_event WHERE event_type='hermes_subagent_span'"
                ).fetchone()[0]
                == 1001
            )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    finally:
        artifact.discard()


def test_atof_prepared_cohort_waits_for_late_conflicting_delegation(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "llm",
            "scope_category": "start",
            "uuid": f"neutral-{ordinal}",
            "timestamp": "2026-07-18T09:00:08Z",
            "name": "hermes.llm.request",
            "metadata": {"session_id": "parent-a"},
            "data": {},
        }
        for ordinal in range(1201)
    ]
    for owner in ("parent-a", "parent-b"):
        records.append(
            {
                "atof_version": "0.1",
                "kind": "mark",
                "uuid": f"delegation-{owner}",
                "timestamp": "2026-07-18T09:00:08Z",
                "name": "hermes.subagent.start",
                "metadata": {"session_id": owner},
                "data": {"child_session_id": "neutral-child"},
            }
        )
    source = tmp_path / "hermes" / "events.jsonl"
    source.parent.mkdir()
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=True,
        shard_directory=str(tmp_path / "prepared"),
        prepare_sessions=lambda sessions: sessions,
    )
    try:
        assert artifact.error is None
        sessions = artifact.session_sequence()
        assert len(sessions) == 2
        observed = [session.provider_session_id for session in sessions]
        assert "parent-a" in observed[0] and "parent-b" in observed[1]
        assert not any("neutral-child" in identity for identity in observed)
        for session in sessions:
            claims = [event for event in session.session_events if event.event_type == "hermes_subagent_span"]
            assert len(claims) == 1
            assert claims[0].payload["delegation_edge_asserted"] is False
        assert artifact.sessions_path is not None
        with sqlite3.connect(artifact.sessions_path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM prepared_event WHERE event_type='hermes_llm_request_span'"
                ).fetchone()[0]
                == 1201
            )
    finally:
        artifact.discard()


def test_decoded_record_tape_closes_input_iterator_after_retention_failure() -> None:
    from collections.abc import Iterator

    from polylogue.sources.decoder_json import DecodedRecordSequence

    closed = False
    failure = OSError("synthetic original record read failure")

    def records() -> Iterator[JSONValue]:
        nonlocal closed
        try:
            yield {"neutral": "retained first record"}
            raise failure
        finally:
            closed = True

    with pytest.raises(OSError) as observed:
        DecodedRecordSequence(records())
    assert observed.value is failure
    assert closed


def test_decoded_record_tape_preserves_input_and_close_failures() -> None:
    from builtins import BaseExceptionGroup

    from polylogue.sources.decoder_json import DecodedRecordSequence

    original = OSError("synthetic original record read failure")
    settlement = OSError("synthetic original iterator close failure")

    class Records:
        def __iter__(self) -> Records:
            return self

        def __next__(self) -> object:
            raise original

        def close(self) -> None:
            raise settlement

    with pytest.raises(BaseExceptionGroup) as observed:
        DecodedRecordSequence(Records())  # type: ignore[arg-type]
    assert observed.value.exceptions == (original, settlement)


@pytest.mark.parametrize("construction_failure", [False, True])
def test_cohort_finalizer_retains_original_failure_and_settles_output_close(
    tmp_path: Path, construction_failure: bool
) -> None:
    from builtins import BaseExceptionGroup
    from collections.abc import Iterator

    original = OSError("synthetic original cohort interpretation failure")
    settlement = OSError("synthetic original finalized iterator close failure")
    close_count = 0
    reached = False

    class Output:
        def __iter__(self) -> Iterator[ParsedSession]:
            if construction_failure:
                raise original
            return self

        def __next__(self) -> ParsedSession:
            raise original

        def close(self) -> None:
            nonlocal close_count
            close_count += 1
            raise settlement

    def finalize(sessions: PreparedSessionSequence) -> Iterable[ParsedSession]:
        nonlocal reached
        assert len(sessions) == 1
        assert "neutral-parent" in next(sessions.iter_provider_session_ids())
        reached = True
        return Output()

    source = tmp_path / "hermes" / "trajectory.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "neutral-parent",
                "steps": [{"source": "agent", "message": "Neutral observation"}],
            }
        ),
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()
    directory = tmp_path / "prepared"
    with pytest.raises(BaseExceptionGroup) as observed:
        prepare_jsonl_blob(
            str(source),
            str(source),
            Provider.HERMES.value,
            "fallback",
            is_stream=False,
            shard_directory=str(directory),
            prepare_sessions=finalize,
        )
    assert observed.value.exceptions == (original, settlement)
    assert reached and close_count == 1
    assert source.read_bytes() == original_bytes
    assert not list(directory.glob("*.db"))


@pytest.mark.parametrize("wire", ["document", "array", "jsonl"])
@pytest.mark.parametrize(
    ("provider", "record", "kind"),
    [
        (
            Provider.HERMES,
            {
                "session_id": "copied-neutral",
                "transcript": "neutral.json",
                "content": "Copied neutral prompt",
                "messages": [{"role": "user", "content": "Neutral prompt"}],
            },
            "extracted_transcript_corpus",
        ),
        (
            Provider.GROK,
            {
                "conversations": [
                    {"conversation": {"title": "Neutral"}, "responses": [{"sender": "human", "message": "Hi"}]}
                ],
                "event_type": "SessionStart",
                "session_id": "neutral-hook",
                "timestamp": "2025-01-02T03:04:05Z",
                "provider": "codex",
            },
            "hook_event",
        ),
    ],
)
def test_retained_candidacy_preserves_content_refusal_across_wire_shapes(
    provider: Provider, record: dict[str, JSONValue], kind: str, wire: str
) -> None:
    from io import BytesIO

    from polylogue.archive.artifact_taxonomy.runtime import classify_artifact_records, classify_artifact_stream

    encoded = json.dumps([record] if wire == "array" else record).encode()
    if wire == "jsonl":
        encoded += b"\n"
    observed = classify_artifact_stream(
        BytesIO(encoded),
        provider=provider,
        source_path="neutral.jsonl" if wire == "jsonl" else "neutral.json",
        wire_format="jsonl" if wire == "jsonl" else "json",
    )
    canonical = classify_artifact_records([record], provider=provider, source_path="neutral.json")
    assert observed.proved_non_session is True
    assert observed.classification.parse_as_session is False
    assert observed.classification.kind.value == canonical.classification.kind.value == kind


@pytest.mark.parametrize("provider", [Provider.DRIVE, Provider.GEMINI])
@pytest.mark.parametrize("wire", ["json", "jsonl"])
def test_bare_drive_chunks_account_for_skipped_and_future_records(
    tmp_path: Path, provider: Provider, wire: str
) -> None:
    from polylogue.sources.parsers.base_models import AdmissionDisposition, AdmissionUnit

    records: list[JSONValue] = [
        {"id": "known", "role": "user", "text": "Neutral authored material"},
        {"text": "Missing required role"},
        17,
        {"type": "future_neutral", "text": "Future missing role"},
    ]
    source = tmp_path / f"chunks.{wire}"
    source.write_text(
        json.dumps(records) if wire == "json" else "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    [expected] = parse_payload(provider, records, "chunks", source_path=str(source))
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        provider.value,
        "chunks",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    try:
        assert artifact.error is None, artifact.error
        [actual] = artifact.iter_sessions()
        assert [message.text for message in actual.messages] == ["Neutral authored material"]
        assert actual.unit_accounting == expected.unit_accounting
        assert actual.unit_accounting is not None
        actual.unit_accounting.assert_conserved()
        assert actual.unit_accounting.expected[AdmissionUnit.OUTER_RECORD] == 4
        assert [outcome.disposition for outcome in actual.unit_accounting.iter_outcomes()] == [
            AdmissionDisposition.MATERIALIZED,
            AdmissionDisposition.TYPED_REFUSAL,
            AdmissionDisposition.TYPED_REFUSAL,
            AdmissionDisposition.TYPED_UNKNOWN,
        ]
        assert list(actual.session_events) == list(expected.session_events)
    finally:
        artifact.discard()


def test_event_sort_preserves_ties_and_uses_exact_ordinal_lookup(tmp_path: Path) -> None:
    from polylogue.sources.parsers.base import ParsedSessionEvent
    from polylogue.sources.prepared_message_sink import SqliteMessageStore

    store = SqliteMessageStore(tmp_path / "event-sort.sqlite")
    plan: list[str] = []
    try:
        events = store.new_event_sink()
        untouched = store.new_event_sink()
        untouched.append(ParsedSessionEvent(event_type="untouched", payload={"position": -1}))
        original = [
            ParsedSessionEvent(
                timestamp=(None, "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z")[index % 3],
                event_type=("early", "late")[index % 2],
                payload={"position": index},
            )
            for index in range(100)
        ]
        events.extend(original)
        tiers = {"early": -1, "late": 1}

        def observe(statement: str) -> None:
            if statement.startswith("UPDATE prepared_event SET event_ordinal = -1 - ("):
                plan.extend(row[3] for row in store.conn.execute("EXPLAIN QUERY PLAN " + statement))

        store.conn.set_trace_callback(observe)
        events.sort_in_place(tiers)
        store.conn.set_trace_callback(None)
        expected = sorted(
            enumerate(original), key=lambda item: (item[1].timestamp or "", tiers[item[1].event_type], item[0])
        )
        assert [event.payload["position"] for event in events] == [index for index, _event in expected]
        assert len(untouched) == 1 and untouched[0].payload["position"] == -1
        assert any("SEARCH o USING INDEX prepared_event_order_key" in step for step in plan), plan
        assert not any("SCAN o" in step for step in plan), plan
        events.append(ParsedSessionEvent(event_type="sidecar", timestamp="2025-01-01T00:00:00Z"))
        assert events[-1].event_type == "sidecar"
        assert (
            store.conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name='prepared_event_order'").fetchone()[
                0
            ]
            == 0
        )
    finally:
        store.close()
