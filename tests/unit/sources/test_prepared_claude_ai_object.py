"""A single claude.ai conversation object streams through sealed preparation."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path

import ijson
import pytest

from polylogue.core.enums import Provider, Role
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import claude_ai_object_envelope
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.claude import common as claude_common
from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import ClaudeChatEvidence, SqliteMessageStore, normalize_active_branch
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.retained_jsonl import retained_raw_fixture


def _conversation(message_count: int = 300) -> dict[str, object]:
    """A branched conversation carrying every normalization the parser owns."""
    messages: list[dict[str, object]] = []
    for index in range(message_count):
        message: dict[str, object] = {
            "uuid": f"m-{index}",
            "sender": "human" if index % 2 == 0 else "assistant",
            "text": f"Neutral turn {index}",
            "created_at": f"2026-01-01T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}Z",
            "parent_message_uuid": f"m-{index - 1}" if index else None,
        }
        messages.append(message)
    messages[3]["updated_at"] = "2026-01-02T00:00:00Z"
    messages[3]["edited_at"] = "2026-01-02T00:00:00Z"
    messages[4]["attachments"] = [
        {"file_name": "brief.txt", "file_type": "text/plain", "extracted_content": "Neutral brief", "file_size": 13}
    ]
    messages[5]["content"] = [
        {"type": "text", "text": "Neutral"},
        {"type": "tool_use", "name": "web_search", "input": {"query": "neutral"}},
        {"type": "tool_result", "name": "web_search", "content": [{"type": "text", "text": "Neutral result"}]},
    ]
    messages[6]["compaction_summary"] = [
        {"text": "Neutral summary", "start_timestamp": "2026-01-01T00:00:00Z", "stop_timestamp": "2026-01-01T00:01:00Z"}
    ]
    messages[7]["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    # A sibling variant, a duplicate native id, an ID-less turn and an empty turn.
    messages.append(
        {
            "uuid": "m-9",
            "sender": "assistant",
            "text": "Neutral variant",
            "created_at": "2026-01-01T00:00:08Z",
            "parent_message_uuid": "m-8",
        }
    )
    messages.append({"sender": "human", "text": "Neutral idless", "parent_message_uuid": "m-8"})
    messages.append({"uuid": "m-empty", "sender": "assistant", "text": "", "parent_message_uuid": "m-2"})
    return {
        "uuid": "claude-conversation",
        "name": "Neutral conversation",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-03T00:00:00Z",
        "model": "neutral-model",
        "settings": {"effort_level": "high", "thinking_mode": "auto"},
        "summary": "Neutral provider summary",
        "status": "complete",
        "current_leaf_message_uuid": f"m-{message_count - 1}",
        "chat_messages": messages,
        "files": [{"file_name": "brief.txt", "file_type": "text/plain"}, {"file_name": "other.txt"}],
    }


def _expected(payload: dict[str, object], source: Path) -> ParsedSession:
    [expected] = admit_parsed_sessions_for_publication(
        parse_payload(Provider.CLAUDE_AI, [payload], "fallback"),
        provider=Provider.CLAUDE_AI,
        source_path=str(source),
    )
    expected.content_hash = session_content_hash(expected)
    return expected


def _assert_same_publication(actual: ParsedSession, expected: ParsedSession, shard_path: Path, tmp_path: Path) -> None:
    assert actual.content_hash == expected.content_hash
    assert actual.unit_accounting == expected.unit_accounting
    assert [event.model_dump(mode="json") for event in actual.session_events] == [
        event.model_dump(mode="json") for event in expected.session_events
    ]
    assert [attachment.model_dump(mode="json") for attachment in actual.attachments] == [
        attachment.model_dump(mode="json") for attachment in expected.attachments
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

    def refuse_resident(self: object, _value: object) -> None:
        raise AssertionError("streamed Claude AI evidence was held in a resident store")

    monkeypatch.setattr(claude_common._ResidentEvidence, "put", refuse_resident)
    monkeypatch.setattr(claude_common._ResidentAttachmentRows, "put", refuse_resident)


def test_claude_ai_object_streams_evidence_before_eof_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _conversation()
    source = tmp_path / "conversation.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(payload, source)
    assert len(expected.messages) == 302
    assert {event.event_type for event in expected.session_events} >= {
        "model_configuration",
        "message_revision",
        "claude_ai_compaction_summary",
        "normalization_diagnostic",
        "provider_session_status",
        "claude_ai_conversation_summary",
    }
    assert len(expected.attachments) == 2

    _refuse_whole_document(monkeypatch)
    decoded = 0
    first_spilled_after: int | None = None
    original_items = ijson.items
    original_put = ClaudeChatEvidence.put

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        for item in original_items(*args, **kwargs):
            decoded += 1
            yield item

    def tracked_put(self: ClaudeChatEvidence, evidence: object) -> None:
        nonlocal first_spilled_after
        if first_spilled_after is None:
            first_spilled_after = decoded
        original_put(self, evidence)  # type: ignore[arg-type]

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(ClaudeChatEvidence, "put", tracked_put)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    assert first_spilled_after == 1
    [actual] = artifact.iter_sessions()
    assert artifact.shard_path is not None and artifact.sessions_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not tables & {"claude_evidence", "claude_attachment"}


def test_claude_ai_object_future_wire_type_keeps_parser_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _conversation(12)
    messages = payload["chat_messages"]
    assert isinstance(messages, list)
    messages[2]["metadata"] = {"kind": "future_metadata"}
    messages[5]["type"] = "future_turn"
    claude_object = claude_ai_object_envelope(BytesIO(json.dumps(payload).encode()))
    assert claude_object is not None and claude_object[0]["__admission_future_type"] == "future_metadata"
    source = tmp_path / "future.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(payload, source)
    assert [event.payload for event in expected.session_events if event.event_type == "claude_ai_unknown_input"] == [
        {"source_index": 1, "wire_type": "future_metadata"}
    ]

    _refuse_whole_document(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    [actual] = artifact.iter_sessions()
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)


@pytest.mark.parametrize(
    "extra",
    [
        {"sessions": []},
        {"account_uuid": "neutral-account", "conversations_memory": "Neutral"},
        {"docs": [], "prompt_template": "Neutral"},
        {"polylogue_capture_kind": "browser_capture"},
        {"chat_messages": {"not": "an array"}},
    ],
)
def test_claude_ai_probe_leaves_rerouted_shapes_to_the_object_parser(tmp_path: Path, extra: dict[str, object]) -> None:
    payload = {**_conversation(12), **extra}
    assert claude_ai_object_envelope(BytesIO(json.dumps(payload).encode())) is None


def test_claude_ai_object_without_identity_keeps_bundle_fallback_identity(tmp_path: Path) -> None:
    payload = _conversation(12)
    del payload["uuid"]
    source = tmp_path / "anonymous.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(payload, source)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    [actual] = artifact.iter_sessions()
    assert actual.provider_session_id == expected.provider_session_id == "fallback-0"
    assert actual.content_hash == expected.content_hash


def test_claude_ai_object_corrupt_suffix_leaves_no_artifact(tmp_path: Path) -> None:
    source = tmp_path / "damaged.json"
    source.write_text(json.dumps(_conversation(20))[:-2] + ", {broken", encoding="utf-8")
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


@pytest.mark.parametrize("failure", ["mutation", "parser"])
def test_claude_ai_object_failure_after_spill_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "conversation.json"
    source.write_text(json.dumps(_conversation(20)), encoding="utf-8")
    original_put = ClaudeChatEvidence.put
    spilled = 0

    def put_then_fail(self: ClaudeChatEvidence, evidence: object) -> None:
        nonlocal spilled
        original_put(self, evidence)  # type: ignore[arg-type]
        spilled += 1
        if spilled == 10:
            if failure == "mutation":
                source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
            else:
                raise RuntimeError("synthetic parse worker failure")

    monkeypatch.setattr(ClaudeChatEvidence, "put", put_then_fail)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert spilled >= 10
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert artifact.deferred is (failure == "mutation")
    assert list(directory.glob("*.db")) == []


def test_retained_claude_ai_object_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    payload = _conversation()
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(payload).encode())
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_AI,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "claude" / "conversation.json"),
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
            assert actual.provider_session_id == "claude-conversation"
            assert len(actual.messages) == 302
            assert actual.created_at == "2026-01-01T00:00:00+00:00"
        finally:
            artifact.discard()


def test_sink_active_path_walk_starts_at_the_leaf_row(tmp_path: Path) -> None:
    messages = [
        ParsedMessage(provider_message_id="a", role=Role.USER, text="one"),
        ParsedMessage(provider_message_id="b", role=Role.ASSISTANT, text="two", parent_message_provider_id="a"),
        ParsedMessage(provider_message_id="x", role=Role.USER, text="three"),
        ParsedMessage(
            provider_message_id="b",
            role=Role.ASSISTANT,
            text="four",
            parent_message_provider_id="x",
            is_active_leaf=False,
        ),
    ]
    messages[1] = messages[1].model_copy(update={"is_active_leaf": True})
    resident = normalize_active_branch(list(messages))
    store = SqliteMessageStore(tmp_path / "scratch.db")
    sink = store.new_sink()
    sink.extend(messages)
    normalized = list(sink.normalize_active_path())
    store.close()
    assert [message.is_active_path for message in normalized] == [message.is_active_path for message in resident]
    # Only the leaf occurrence and its own parent chain: the later "b"
    # (parent "x") is a different message that repeats the provider id.
    assert [message.is_active_path for message in resident] == [True, True, None, None]


@pytest.mark.parametrize("prepared", [False, True], ids=["conventional", "prepared"])
def test_real_claude_repeated_occurrence_events_keep_their_message(tmp_path: Path, prepared: bool) -> None:
    from contextlib import closing

    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "claude-ai" / "event-occurrences.json"
    source = tmp_path / "conversation.json"
    source.write_bytes(fixture.read_bytes())
    artifact = None
    try:
        if prepared:
            artifact = prepare_jsonl_blob(
                str(source),
                str(source),
                Provider.CLAUDE_AI.value,
                "fallback",
                is_stream=False,
                shard_directory=str(tmp_path / "prepared"),
            )
            assert artifact.error is None
            with closing(artifact.iter_sessions()) as sessions:
                parsed = next(sessions)
        else:
            parsed = _expected(json.loads(source.read_text()), source)
        with closing(connect_measured(tmp_path / "index.db")) as conn, conn:
            conn.row_factory = sqlite3.Row
            initialize_archive_tier(conn, ArchiveTier.INDEX)
            sid = write_fixture_index_session(conn, parsed)
            rows = conn.execute(
                "SELECT e.event_type, e.payload_json, b.text FROM session_events e "
                "JOIN blocks b ON b.message_id = e.source_message_id "
                "WHERE e.session_id = ? AND b.block_type = 'text'",
                (sid,),
            ).fetchall()
        summaries = {json.loads(row[1])["summary"]: row[2] for row in rows if row[0] == "claude_ai_compaction_summary"}
        assert summaries == {"first summary": "first occurrence", "second summary": "second occurrence"}
        configurations = {json.loads(row[1])["model"]: row[2] for row in rows if row[0] == "model_configuration"}
        assert configurations == {"synthetic-first": "first occurrence", "synthetic-second": "second occurrence"}
        revisions = {json.loads(row[1])["updated_at"]: row[2] for row in rows if row[0] == "message_revision"}
        # The parser stores provider timestamps as normalized ISO-8601.
        assert revisions == {
            "2026-01-01T00:01:00+00:00": "first occurrence",
            "2026-01-01T00:03:00+00:00": "second occurrence",
        }
    finally:
        if artifact is not None:
            artifact.discard()
