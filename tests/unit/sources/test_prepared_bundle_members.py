"""Bundle members, claude.ai lineage and Grok roots stream through sealed preparation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources import prepared_jsonl
from polylogue.sources.decoder_json import (
    claude_ai_object_envelope,
    grok_export_item_count,
    grok_taxonomy_witness,
    iter_container_member_files,
    scan_container_members,
)
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, bundle_member_sessions, parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.claude import common as claude_common
from polylogue.sources.parsers.claude.lineage_graph import ClaudeLineageGraph, LineageNode
from polylogue.sources.prepared_jsonl import PreparedJsonl, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import ClaudeChatEvidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.retained_jsonl import retained_parser_fixture
from tests.infra.source_builders import ChatGPTExportBuilder
from tests.unit.sources.test_prepared_claude_ai_object import _conversation


def _design(session_id: str) -> dict[str, object]:
    return {
        "uuid": session_id,
        "project": {"uuid": "neutral-project"},
        "title": "Neutral design",
        "created_at": "2026-01-01T00:00:00Z",
        "messages": [
            {
                "uuid": f"{session_id}-u-{index}",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": f"Neutral prompt {index}",
                    "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                    "attachments": [
                        {"id": f"attachment-{index}", "name": f"brief-{index}.txt", "type": "text", "content": "Brief"}
                    ],
                },
            }
            for index in range(3)
        ],
    }


def _chatgpt(conversation_id: str, turns: int = 3) -> dict[str, object]:
    builder = ChatGPTExportBuilder(conversation_id)
    for index in range(turns):
        builder.add_node("user" if index % 2 == 0 else "assistant", f"Neutral turn {index}")
    record = builder.build()
    record["create_time"] = 1704067200.0
    return record


def _expected(provider: Provider, source: Path) -> list[ParsedSession]:
    expected = admit_parsed_sessions_for_publication(
        parse_payload(provider, list(_iter_json_stream(BytesIO(source.read_bytes()), source.name)), "fallback"),
        provider=provider,
        source_path=str(source),
    )
    for session in expected:
        session.content_hash = session_content_hash(session)
    return expected


def _assert_publication(artifact: PreparedJsonl, expected: list[ParsedSession], tmp_path: Path) -> None:
    assert artifact.error is None, artifact.error
    actual = list(artifact.iter_sessions())
    assert [(session.provider_session_id, session.content_hash) for session in actual] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]
    for prepared, baseline in zip(actual, expected, strict=True):
        assert [event.model_dump(mode="json") for event in prepared.session_events] == [
            event.model_dump(mode="json") for event in baseline.session_events
        ]
        assert [attachment.model_dump(mode="json") for attachment in prepared.attachments] == [
            attachment.model_dump(mode="json") for attachment in baseline.attachments
        ]
        exclude = {"messages", "session_events", "attachments"}
        assert prepared.model_dump(mode="json", exclude=exclude) == baseline.model_dump(mode="json", exclude=exclude)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)
    assert artifact.shard_path is not None
    with sqlite3.connect(expected_shard.path) as baseline_db, sqlite3.connect(artifact.shard_path) as prepared_db:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared_db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline_db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    required_tables = {
        "prepared_session",
        "prepared_message",
        "prepared_event",
        "prepared_attachment",
        "artifact_seal",
        "prepared_message_normalization",
        "prepared_sidecar_publication",
        "prepared_attachment_publication",
        "prepared_classification",
        "prepared_codex_state",
        "prepared_codex_thread",
        "prepared_codex_spawn",
        "prepared_codex_state_part",
        "prepared_streamed_json_array",
        "prepared_streamed_json_array_value",
    }
    assert tables == required_tables
    sessions_path = artifact.sessions_path
    shard_path = artifact.shard_path
    artifact.discard()
    assert not sessions_path.exists()
    assert not shard_path.exists()


def _collected_members(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record which members reach the collecting bundle lowering."""
    collected: list[int] = []
    original = bundle_member_sessions

    def tracked(provider: Provider, record: object, fallback_id: str, index: int, **kwargs: object) -> object:
        collected.append(index)
        return original(provider, record, fallback_id, index, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(prepared_jsonl, "bundle_member_sessions", tracked)
    return collected


_BUNDLES: dict[str, tuple[Provider, object, list[int]]] = {
    # payload, and the members only the collecting lowering may read
    "chatgpt-siblings": (
        Provider.CHATGPT,
        [{"unrelated": "sibling"}, _chatgpt("c-1"), {"mapping": {"bad": {"id": "bad"}}}, _chatgpt("c-2"), 5, [1]],
        [0, 2],
    ),
    "chatgpt-singleton": (Provider.CHATGPT, [_chatgpt("c-only")], []),
    "claude-array": (
        Provider.CLAUDE_AI,
        [_conversation(40), {**_conversation(12), "uuid": "second"}, {"uuid": "memory", "account_uuid": "a"}],
        [2],
    ),
    "claude-sessions": (
        Provider.CLAUDE_AI,
        {"meta": 1, "sessions": [_conversation(20), {**_conversation(8), "uuid": "b"}]},
        [],
    ),
    "claude-singleton": (Provider.CLAUDE_AI, [_conversation(30)], []),
    "design-array": (Provider.CLAUDE_DESIGN, [_design("d-1"), _design("d-2")], []),
}


@pytest.mark.parametrize("case", sorted(_BUNDLES))
def test_bundle_members_stream_with_parser_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    provider, payload, collecting = _BUNDLES[case]
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(provider, source)
    assert expected

    collected = _collected_members(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        provider.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    _assert_publication(artifact, expected, tmp_path)
    assert artifact.positive_evidence_filtered
    assert collected == collecting
    assert list((tmp_path / "prepared").glob("member-*")) == []


def test_bundle_member_spills_before_the_container_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [_conversation(60), {**_conversation(10), "uuid": "second"}, {**_conversation(10), "uuid": "third"}]
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected = _expected(Provider.CLAUDE_AI, source)

    def refuse_member_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a conversation member was decoded whole")

    monkeypatch.setattr(prepared_jsonl, "iter_json_container_records", refuse_member_decode)
    members_written = 0
    first_spill_after: int | None = None
    original_files = iter_container_member_files
    original_put = ClaudeChatEvidence.put

    def tracked_files(*args: object, **kwargs: object) -> Iterator[int | None]:
        nonlocal members_written
        for member in original_files(*args, **kwargs):  # type: ignore[arg-type]
            members_written += 1
            yield member

    def tracked_put(self: ClaudeChatEvidence, evidence: object) -> None:
        nonlocal first_spill_after
        if first_spill_after is None:
            first_spill_after = members_written
        original_put(self, evidence)  # type: ignore[arg-type]

    monkeypatch.setattr(prepared_jsonl, "iter_container_member_files", tracked_files)
    monkeypatch.setattr(ClaudeChatEvidence, "put", tracked_put)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    _assert_publication(artifact, expected, tmp_path)
    # The first member's records reach scratch while two members are unread.
    assert first_spill_after == 1


def test_member_files_carry_one_member_at_a_time_with_bundle_numbers(tmp_path: Path) -> None:
    payload = b'[{"a": 1.0, "b": [2.5, {"c": "\\u00e9"}], "d": null}, 7, {"e": true, "f": {}}]'
    member_path = tmp_path / "member.json"
    seen: list[object] = []
    for index in iter_container_member_files(BytesIO(payload), "item", member_path):
        seen.append((index, json.loads(member_path.read_bytes()) if index is not None else None))
    # An integral number reads back as the bundle lowering reads it: an int.
    assert seen == [(0, {"a": 1, "b": [2.5, {"c": "é"}], "d": None}), (None, None), (2, {"e": True, "f": {}})]


def test_member_scan_reads_shapes_and_bounded_witnesses(tmp_path: Path) -> None:
    members = [{"session": {"x": 1}, "schema_version": 1, "mapping": {str(key): {} for key in range(100)}}, "scalar"]
    shapes: list[object] = []
    witnesses: list[object] = []

    def observe(index: int, shape: object, witness: object) -> None:
        shapes.append(shape)
        witnesses.append(witness)

    count = scan_container_members(
        BytesIO(json.dumps({"sessions": members}).encode()),
        "sessions.item",
        shape_keys=frozenset({"session", "schema_version"}),
        witnesses=1,
        on_member=observe,
    )
    assert count == 2
    assert shapes == [{"session": {}, "schema_version": 1}, "scalar"]
    first = witnesses[0]
    assert isinstance(first, dict) and len(first["mapping"]) == 64
    assert witnesses[1] is None
    # A decoder keeps only the last repeated ``sessions``; the stream refuses.
    repeated = b'{"sessions": [{"a": 1}], "sessions": [{"b": 2}]}'
    assert (
        scan_container_members(
            BytesIO(repeated), "sessions.item", shape_keys=frozenset(), witnesses=0, on_member=observe
        )
        is None
    )


def test_bundle_member_with_an_unstorable_ignored_field_keeps_the_collecting_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.sources.value_bounds as value_bounds

    member = _conversation(9)
    member["ignored_blob"] = "x" * 16384
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps([member, {**_conversation(10), "uuid": "second"}]), encoding="utf-8")
    expected = _expected(Provider.CLAUDE_AI, source)
    monkeypatch.setattr(value_bounds, "MAX_STORABLE_VALUE_BYTES", 8192)
    collected = _collected_members(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    _assert_publication(artifact, expected, tmp_path)
    assert collected == [0]


def test_bundle_corrupt_suffix_leaves_no_artifact_or_member_file(tmp_path: Path) -> None:
    source = tmp_path / "damaged.json"
    source.write_text("[" + json.dumps(_conversation(8)) + ', {"uuid": "broken"', encoding="utf-8")
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
    assert list(directory.iterdir()) == []


def test_bundle_source_mutation_between_passes_defers_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps([_conversation(8), {**_conversation(8), "uuid": "second"}]), encoding="utf-8")
    original_scan = scan_container_members

    def scan_then_append(*args: object, **kwargs: object) -> int | None:
        count = original_scan(*args, **kwargs)  # type: ignore[arg-type]
        source.write_text(
            json.dumps([_conversation(8), {**_conversation(8), "uuid": "second"}, {**_conversation(8), "uuid": "x"}]),
            encoding="utf-8",
        )
        return count

    monkeypatch.setattr(prepared_jsonl, "scan_container_members", scan_then_append)
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
    assert artifact.deferred
    assert artifact.sessions_path is None
    assert list(directory.iterdir()) == []


def test_bundle_member_failure_after_spill_discards_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps([_conversation(8), {**_conversation(8), "uuid": "second"}]), encoding="utf-8")
    original_put = ClaudeChatEvidence.put
    puts = 0

    def failing_put(self: ClaudeChatEvidence, evidence: object) -> None:
        nonlocal puts
        puts += 1
        if puts == 12:
            raise RuntimeError("worker lost mid-member")
        original_put(self, evidence)  # type: ignore[arg-type]

    monkeypatch.setattr(ClaudeChatEvidence, "put", failing_put)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None and "worker lost mid-member" in artifact.error
    assert not artifact.deferred
    assert list(directory.iterdir()) == []


@contextmanager
def _retained(tmp_path: Path, payload: object, provider: Provider, source_path: Path) -> Iterator[PreparedJsonl]:
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(json.dumps(payload).encode("utf-8"))
    with retained_parser_fixture(
        root=tmp_path,
        provider=provider,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=BlobStore(tmp_path / "blob")._ensure_private_staging_root()
        / f"prepared-{source_path.parent.name}-{source_path.stem}",
        file_mtime="2025-01-02T03:04:05Z",
    ) as (artifact, _reader):
        yield artifact


def test_retained_bundle_streams_members_and_keeps_artifact_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [_conversation(20), {**_conversation(9), "uuid": "second"}]
    source_path = tmp_path / "claude" / "conversations.json"
    expected = admit_parsed_sessions_for_publication(
        parse_payload(Provider.CLAUDE_AI, payload, source_path.stem),
        provider=Provider.CLAUDE_AI,
        source_path=str(source_path),
    )
    expected = [
        normalize_session_timestamps(session, fallback_timestamp="2025-01-02T03:04:05Z") for session in expected
    ]
    collected = _collected_members(monkeypatch)
    with _retained(tmp_path, payload, Provider.CLAUDE_AI, source_path) as artifact:
        assert artifact.error is None
        assert [(session.provider_session_id, session.created_at) for session in artifact.iter_sessions()] == [
            (session.provider_session_id, session.created_at) for session in expected
        ]
        assert collected == []

    # An ``agent-*.meta.json`` path is a content-blind sidecar marker: unlike
    # an OriginSpec ``fact`` path, session-shaped content there stays refused.
    with _retained(tmp_path, payload, Provider.CLAUDE_AI, tmp_path / "agent-neutral.meta.json") as sidecar:
        assert sidecar.error is None
        assert list(sidecar.iter_sessions()) == []

    fact_payload = {"agent": "neutral", "facts": [{"key": "status", "value": "ready"}]}
    with _retained(tmp_path, fact_payload, Provider.CLAUDE_AI, tmp_path / "agent-fact.meta.json") as fact:
        assert fact.error is None
        assert list(fact.iter_sessions()) == []
        proof = fact.stream_classification()
        assert proof is not None and proof.proved_non_session


def test_claude_lineage_graph_and_attachments_stay_in_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _conversation(80)
    messages = payload["chat_messages"]
    assert isinstance(messages, list)
    # A cycle and a record that repeats its native id both reach the graph.
    messages[20]["parent_message_uuid"] = "m-25"
    messages.append(dict(messages[30], text="Neutral richer duplicate with more text"))
    payload["attachments"] = [{"file_name": "brief.txt", "file_type": "text/plain"}, {"file_name": "loose.txt"}]
    source = tmp_path / "conversation.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    [expected] = admit_parsed_sessions_for_publication(
        parse_payload(Provider.CLAUDE_AI, [payload], "fallback"), provider=Provider.CLAUDE_AI, source_path=str(source)
    )
    expected.content_hash = session_content_hash(expected)
    assert claude_common.CLAUDE_LINEAGE_CYCLE_INGEST_FLAG in expected.ingest_flags
    assert len(expected.attachments) == 3

    object_result = claude_ai_object_envelope(BytesIO(source.read_bytes()))
    assert object_result is not None
    envelope, arrays = object_result
    assert arrays == ("attachments", "files")
    assert "attachments" not in envelope and "files" not in envelope and "chat_messages" not in envelope

    graph_writes: list[bool] = []
    original_observe = ClaudeLineageGraph.observe

    def tracked_observe(self: ClaudeLineageGraph, node: LineageNode, richer_on_tie: Callable[[int, int], bool]) -> None:
        # ``_owned`` marks the object parser's in-memory graph.
        graph_writes.append(self._owned)
        original_observe(self, node, richer_on_tie)

    def refuse_resident(self: object, _value: object) -> None:
        raise AssertionError("streamed attachment rows were held resident")

    monkeypatch.setattr(ClaudeLineageGraph, "observe", tracked_observe)
    monkeypatch.setattr(claude_common._ResidentAttachmentRows, "put", refuse_resident)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    _assert_publication(artifact, [expected], tmp_path)
    # Every record reached the graph on the scratch connection.
    assert len(graph_writes) == len(messages) and not any(graph_writes)


def test_claude_non_array_conversation_attachments_keep_streaming(tmp_path: Path) -> None:
    payload = _conversation(10)
    payload["attachments"] = {"not": "a list"}
    source = tmp_path / "conversation.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    object_result = claude_ai_object_envelope(BytesIO(source.read_bytes()))
    assert object_result is not None
    assert object_result[0]["attachments"] == {"not": "a list"}
    assert object_result[1] == ("files",)
    [expected] = admit_parsed_sessions_for_publication(
        parse_payload(Provider.CLAUDE_AI, [payload], "fallback"), provider=Provider.CLAUDE_AI, source_path=str(source)
    )
    expected.content_hash = session_content_hash(expected)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.CLAUDE_AI.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    _assert_publication(artifact, [expected], tmp_path)


_HOOK_OVERLAP = {
    "conversations": [{"conversation": {"title": "Ambiguous"}, "responses": [{"sender": "human", "message": "Hi"}]}],
    "event_type": "SessionStart",
    "session_id": "hook-session",
    "timestamp": "2025-01-02T03:04:05Z",
    "provider": "codex",
}


@pytest.mark.parametrize("hook_record", [False, True])
def test_grok_root_admission_preserves_hook_taxonomy_and_provider_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook_record: bool
) -> None:
    payload = _HOOK_OVERLAP if hook_record else {"conversations": _HOOK_OVERLAP["conversations"]}
    raw = json.dumps(payload).encode()
    # Low-level parser selection does not override canonical artifact admission.
    assert grok_export_item_count(BytesIO(raw)) == (None if hook_record else 1)
    assert grok_export_item_count(BytesIO(raw), detect=False) == 1
    source = tmp_path / "prod-grok-backend.json"
    source.write_bytes(raw)
    expected = admit_parsed_sessions_for_publication(
        parse_payload(Provider.GROK, payload, "fallback"), provider=Provider.GROK, source_path=str(source)
    )
    if hook_record:
        expected = []
    for session in expected:
        session.content_hash = session_content_hash(session)

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Grok root decoded as a whole document")

    monkeypatch.setattr(prepared_jsonl, "_iter_json_stream", refuse_whole_document)
    monkeypatch.setattr(prepared_jsonl, "iter_parsed_payload", refuse_whole_document)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.GROK.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert [(session.provider_session_id, session.content_hash) for session in artifact.iter_sessions()] == [
        (session.provider_session_id, session.content_hash) for session in expected
    ]

    if hook_record:
        proof = artifact.stream_classification()
        assert proof is not None and proof.proved_non_session
        assert proof.classification.kind.value == "hook_event"


def test_grok_replay_classifies_a_bounded_root_witness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "conversations": [
            {"conversation": {"title": f"T{index}"}, "responses": [{"sender": "human", "message": "Hi"}] * 70}
            for index in range(70)
        ],
        "kind": "export",
        "created_at": 1712000000.5,
        "messages": [{"role": "user", "content": "Root metadata"}] * 70,
    }
    witness = grok_taxonomy_witness(BytesIO(json.dumps(record).encode()))
    assert isinstance(witness, dict)
    conversations = witness["conversations"]
    assert isinstance(conversations, list) and len(conversations) == 64
    assert isinstance(conversations[0], dict) and len(conversations[0]["responses"]) == 64  # type: ignore[arg-type]
    assert witness["kind"] == "export" and witness["created_at"] == 1712000000.5
    assert isinstance(witness["messages"], list) and len(witness["messages"]) == 64

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained Grok root decoded as a whole document")

    monkeypatch.setattr(prepared_jsonl, "_iter_json_stream", refuse_whole_document)
    monkeypatch.setattr(prepared_jsonl, "iter_parsed_payload", refuse_whole_document)
    # The hook-shaped root is judged by taxonomy on the stream route, not by
    # a whole-document fallback, and stays a non-session artifact.
    with _retained(tmp_path, _HOOK_OVERLAP, Provider.GROK, tmp_path / "prod-grok-backend.json") as artifact:
        assert artifact.error is None
        assert list(artifact.iter_sessions()) == []
        proof = artifact.stream_classification()
        assert proof is not None and proof.proved_non_session
