"""A single AI Studio chunked prompt streams through sealed preparation."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path

import ijson
import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import drive_chunked_prompt_envelope
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.parsers import drive
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import SqliteMessageSink
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.retained_jsonl import retained_raw_fixture


def _chunks(count: int) -> list[object]:
    """Chunks out of time order, with branch evidence, attachments and ID-less turns."""
    chunks: list[object] = []
    for index in range(count):
        chunk: dict[str, object] = {
            "id": f"chunk-{index}",
            "role": "user" if index % 2 == 0 else "model",
            "text": f"Neutral turn {index}",
            "createTime": f"2026-01-01T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}Z",
        }
        chunks.append(chunk)
    chunks.reverse()
    chunks.append({"role": "user", "text": "Neutral idless turn", "createTime": "2026-01-01T00:00:03Z"})
    chunks.append({"role": "model", "text": "Neutral idless answer"})
    chunks.append(
        {
            "id": "branch-answer",
            "role": "model",
            "text": "Neutral branch answer",
            "createTime": "2026-01-01T00:00:04Z",
            "branchParent": {"id": "chunk-2"},
        }
    )
    chunks.append(
        {
            "id": "chunk-parent",
            "role": "model",
            "text": "Neutral parent",
            "branchChildren": [{"id": "chunk-5"}, {"promptId": "prompt-child"}],
        }
    )
    chunks.append(
        {
            "id": "with-document",
            "role": "user",
            "text": "Neutral attachment",
            "driveDocument": {"id": "neutral-document"},
            "tokenCount": 12,
        }
    )
    chunks.append({"id": "thought", "role": "model", "text": "Neutral thought", "isThought": True})
    chunks.append({"id": "skipped", "text": "No role"})
    return chunks


def _prompt(count: int = 300, *, wrapped: bool = True) -> dict[str, object]:
    document: dict[str, object] = {
        "id": "neutral-prompt",
        "displayName": "Neutral prompt",
        "runSettings": {"model": "models/neutral", "temperature": 0.5},
        "systemInstruction": {"text": "Neutral instruction"},
    }
    if wrapped:
        document["chunkedPrompt"] = {"chunks": _chunks(count), "pendingInputs": [{"text": "Neutral draft"}]}
    else:
        document["chunks"] = _chunks(count)
    return document


def _expected(provider: Provider, payload: dict[str, object], source: Path) -> ParsedSession:
    [expected] = admit_parsed_sessions_for_publication(
        parse_payload(provider, [payload], "fallback"), provider=provider, source_path=str(source)
    )
    expected.content_hash = session_content_hash(expected)
    return expected


def _assert_same_publication(actual: ParsedSession, expected: ParsedSession, shard_path: Path, tmp_path: Path) -> None:
    assert actual.content_hash == expected.content_hash
    assert actual.unit_accounting == expected.unit_accounting
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert [(item.model_dump(mode="json"), item.message_position) for item in actual.attachments] == [
        (item.model_dump(mode="json"), item.message_position) for item in expected.attachments
    ]
    assert [
        (message.provider_message_id, message.parent_message_provider_id, message.parent_message_position)
        for message in actual.messages
    ] == [
        (message.provider_message_id, message.parent_message_provider_id, message.parent_message_position)
        for message in expected.messages
    ]
    exclude = {"messages", "session_events", "attachments"}
    assert actual.model_dump(mode="json", exclude=exclude) == expected.model_dump(mode="json", exclude=exclude)
    expected_shard = prepare_session_shard(tmp_path / "expected", [expected])
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def _refuse_whole_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-document decode or parse was used")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse)


@pytest.mark.parametrize("provider", [Provider.DRIVE, Provider.GEMINI])
@pytest.mark.parametrize("wrapped", [True, False])
def test_chunked_prompt_streams_messages_before_eof_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider, wrapped: bool
) -> None:
    payload = _prompt(wrapped=wrapped)
    source = tmp_path / "prompt.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(provider, payload, source)
    assert len(expected.messages) == 306
    assert any(message.parent_message_position is not None for message in expected.messages)
    assert expected.pending_drafts if wrapped else not expected.pending_drafts
    assert expected.attachments

    _refuse_whole_document(monkeypatch)
    original_items = ijson.items
    original_append = SqliteMessageSink.append
    decoded_in_pass: list[int] = []
    first_append_pass: tuple[int, int] | None = None

    def tracked_items(*args: object, **kwargs: object) -> object:
        decoded_in_pass.append(0)
        for item in original_items(*args, **kwargs):
            decoded_in_pass[-1] += 1
            yield item

    def tracked_append(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal first_append_pass
        if first_append_pass is None:
            first_append_pass = (len(decoded_in_pass) - 1, decoded_in_pass[-1])
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
    assert artifact.positive_evidence_filtered
    assert first_append_pass is not None
    message_pass, decoded_before_first_row = first_append_pass
    assert decoded_before_first_row == 1
    assert decoded_in_pass[message_pass] == len(_chunks(300))
    [actual] = artifact.iter_sessions()
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)


def test_chunked_prompt_future_wire_type_keeps_parser_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = _chunks(20)
    future_chunk = chunks[4]
    assert isinstance(future_chunk, dict)
    future_chunk["type"] = "future_chunk"
    payload = {**_prompt(20), "chunkedPrompt": {"chunks": chunks}}
    envelope = drive_chunked_prompt_envelope(BytesIO(json.dumps(payload).encode()))
    assert envelope is not None and envelope[0]["__admission_future_type"] == "future_chunk"
    source = tmp_path / "future.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(Provider.DRIVE, payload, source)
    assert [event.payload for event in expected.session_events if event.event_type == "drive_unknown_input"] == [
        {"source_index": 1, "wire_type": "future_chunk"}
    ]
    _refuse_whole_document(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)


@pytest.mark.parametrize(
    "document",
    [
        {**_prompt(3), "messages": []},
        {**_prompt(3), "sessionId": "neutral"},
        {**_prompt(3), "mapping": {}},
        {**_prompt(3), "role": "user"},
        {**_prompt(3), "sessions": []},
        {**_prompt(3), "chunks": []},
        {**_prompt(3), "chunkedPrompt": {"pendingInputs": []}},
        {**_prompt(3), "chunkedPrompt": {"chunks": {"not": "an array"}}},
    ],
)
def test_chunked_prompt_probe_leaves_other_lowerings_to_the_object_parser(document: dict[str, object]) -> None:
    assert drive_chunked_prompt_envelope(BytesIO(json.dumps(document).encode())) is None


def test_chunked_prompt_corrupt_suffix_leaves_no_artifact(tmp_path: Path) -> None:
    source = tmp_path / "damaged.json"
    source.write_text(json.dumps(_prompt(20))[:-3] + ", {broken", encoding="utf-8")
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


@pytest.mark.parametrize("failure", ["mutation", "parser"])
def test_chunked_prompt_failure_after_spill_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "prompt.json"
    source.write_text(json.dumps(_prompt(20)), encoding="utf-8")
    original_append = SqliteMessageSink.append
    written = 0

    def append_then_fail(self: SqliteMessageSink, value: ParsedMessage) -> None:
        nonlocal written
        original_append(self, value)
        written += 1
        if written == 5:
            if failure == "mutation":
                source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
            else:
                raise RuntimeError("synthetic parse worker failure")

    monkeypatch.setattr(SqliteMessageSink, "append", append_then_fail)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert written >= 5
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert artifact.deferred is (failure == "mutation")
    assert list(directory.glob("*.db")) == []


def test_retained_chunked_prompt_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    payload = _prompt()
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(payload).encode())
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.DRIVE,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "drive" / "Neutral prompt.json"),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (reader, raw_id):
        _refuse_whole_document(monkeypatch)
        artifact = revision_backfill.prepare_retained_jsonl_artifact(
            reader, raw_id, directory=BlobStore(blob_root)._ensure_private_staging_root() / "prepared"
        )
        try:
            assert artifact.error is None, artifact.error
            assert artifact.positive_evidence_filtered
            [actual] = artifact.iter_sessions()
            assert actual.provider_session_id == "neutral-prompt"
            assert len(actual.messages) == 306
        finally:
            artifact.discard()


def test_repeated_ancestor_key_keeps_the_document_on_the_object_parser() -> None:
    """The decoder keeps the last ``chunkedPrompt``; streaming must not take the first one's chunks.

    Anti-vacuity: without the duplicate-key guard the probe returns an
    envelope that streams the discarded turn.
    """
    document = (
        b'{"id":"p","chunkedPrompt":{"chunks":[{"role":"user","text":"discarded"}]},'
        b'"chunkedPrompt":{"pendingInputs":[]}}'
    )
    assert drive_chunked_prompt_envelope(BytesIO(document)) is None


def test_a_source_key_named_like_probe_metadata_stays_on_the_object_parser() -> None:
    """Anti-vacuity: keeping the envelope lets the streamed parser read the source value as probe metadata."""
    document = (
        b'{"id":"p","__admission_future_type":"future_x","chunkedPrompt":{"chunks":[{"role":"user","text":"hello"}]}}'
    )
    assert drive_chunked_prompt_envelope(BytesIO(document)) is None


def test_chunked_prompt_keeps_ordering_and_branch_rows_in_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ambiguous, prompt-declared and single-parent branch children match the parser.

    The streamed route writes its per-message ordering rows to the artifact's
    scratch database, never to a resident in-memory one, and drops them
    before sealing.
    """
    chunks = [
        {"id": "a", "role": "user", "text": "Neutral a", "createTime": "2026-01-01T00:00:02+02:00"},
        {"id": "b", "role": "model", "text": "Neutral b", "branchChildren": ["c", {"id": "d"}]},
        {"id": "e", "role": "model", "text": "Neutral e", "branchChildren": [{"messageId": "d"}]},
        {"id": "c", "role": "user", "text": "Neutral c", "createTime": "2026-01-01T00:00:01Z"},
        {"id": "d", "role": "user", "text": "Neutral d"},
        {"id": "f", "role": "model", "text": "Neutral f", "branchParent": {"promptId": "other-prompt"}},
        {"id": "g", "role": "user", "text": "Neutral g", "branchChildren": [{"promptId": "a"}]},
    ]
    payload: dict[str, object] = {"id": "neutral-branches", "chunkedPrompt": {"chunks": chunks}}
    source = tmp_path / "prompt.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(Provider.DRIVE, payload, source)
    assert expected.parent_session_provider_id == "other-prompt"
    parents = {message.provider_message_id: message.parent_message_provider_id for message in expected.messages}
    assert parents["c"] == "b"
    assert parents["d"] is None  # declared under two parents

    _refuse_whole_document(monkeypatch)

    scratch_files: list[str] = []
    original_init = drive._ChunkOrder.__init__

    def record_scratch(self: drive._ChunkOrder, conn: sqlite3.Connection) -> None:
        scratch_files.append(conn.execute("PRAGMA database_list").fetchone()[2])
        original_init(self, conn)

    monkeypatch.setattr(drive._ChunkOrder, "__init__", record_scratch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.DRIVE.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    # An in-memory database has no file: the chunk rows went to the
    # artifact's scratch file, and only the chunkless admission stub that
    # follows used memory.
    assert scratch_files[0] and scratch_files[1:] == [""]
    [actual] = artifact.iter_sessions()
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not {table for table in tables if table.startswith("drive_")}
