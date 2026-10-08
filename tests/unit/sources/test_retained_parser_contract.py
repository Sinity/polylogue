"""Original retained parser and bounded artifact contracts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, MutableSequence
from contextlib import closing
from io import BytesIO
from pathlib import Path

import pytest

import polylogue.sources.prepared_jsonl as prepared_jsonl
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.sources import revision_backfill
from polylogue.sources.dispatch import PayloadRecord, parse_generic_messages_stream
from polylogue.sources.fallback_identity import fallback_session_id
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.prepared_jsonl import DecodeFailure, PreparedDecodeError, terminal_decode_evidence
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.raw_authority import iter_parser_census_logical_keys
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_jsonl import retained_parser_fixture
from tests.infra.retained_parser_payloads import (
    _CLAUDE_ASSISTANT_RECORD,
    _CLAUDE_USER_RECORD,
    _chatgpt_session,
    _relationship_index_jsonl_bytes,
    _single_session_state_db_bytes,
)


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        (
            "/archive/drive-cache/gemini/Branch_of_Br-144383b77f2f293fb94ec8647f3632e4.json",
            "Branch_of_Br-144383b77f2f293fb94ec8647f3632e4",
        ),
        (
            "/project/session/subagents/agent-aba750c3c29cb63e0.jsonl",
            "agent-aba750c3c29cb63e0",
        ),
        ("/exports/conversation-0123456789abcdef.json", "conversation"),
        ("/exports/archive.zip:conversation-0123456789abcdef.jsonl", "conversation"),
    ],
)
def test_retained_fallback_identity_preserves_provider_path_contract(source_path: str, expected: str) -> None:
    assert fallback_session_id(source_path, "raw-id") == expected


def test_retained_fallback_identity_uses_raw_id_without_source_path() -> None:
    assert fallback_session_id(None, "raw-id") == "raw-id"


def test_parse_one_replays_single_session_state_db_bytes_via_temp_spill(tmp_path: Path) -> None:
    """Original acquired state exports replay through the retained non-JSON artifact carrier."""

    payload = _single_session_state_db_bytes(tmp_path)

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.HERMES,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "hermes-home" / "state.db"),
        directory=tmp_path / "prepared",
        prepare=revision_backfill.prepare_retained_non_json_artifact,
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert len(sessions) == 1
        assert sessions[0].messages
        assert sessions[0].messages[0].text == "hi"


@pytest.mark.parametrize("suffix", ["json", "txt"])
def test_retained_generic_message_object_alias_uses_streaming_message_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    """A generic single-object Drive export streams even under a neutral filename."""
    bootstrap_archive_root(tmp_path)
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"neutral message {index}"}
        for index in range(400)
    ]
    payload = json.dumps({"id": "neutral-drive-object", "title": "Neutral export", "messages": messages}).encode()
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)

    def refuse_collecting_replay(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("single-object JSON alias used collecting retained replay")

    monkeypatch.setattr(revision_backfill, "parse_retained_raw_sessions", refuse_collecting_replay)
    streamed_records = 0
    original_parse = parse_generic_messages_stream

    def observe_stream(
        provider: Provider,
        envelope: PayloadRecord,
        records: Iterable[object],
        fallback_id: str,
        *,
        message_sink: MutableSequence[ParsedMessage],
    ) -> ParsedSession | None:
        nonlocal streamed_records

        def count_records() -> Iterable[object]:
            nonlocal streamed_records
            for record in records:
                streamed_records += 1
                yield record

        return original_parse(provider, envelope, count_records(), fallback_id, message_sink=message_sink)

    monkeypatch.setattr(prepared_jsonl, "parse_generic_messages_stream", observe_stream)
    prepare = (
        revision_backfill.prepare_retained_jsonl_artifact
        if suffix == "json"
        else revision_backfill.prepare_retained_non_json_artifact
    )
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.DRIVE,
        blob_hash=blob_hash,
        source_path=f"neutral-drive.{suffix}",
        directory=tmp_path / f"prepared-{suffix}",
        prepare=prepare,
    ) as (artifact, _reader):
        assert artifact.error is None
        (session,) = artifact.iter_sessions()
        assert session.provider_session_id == "neutral-drive-object"
        assert [message.text for message in session.messages] == [item["content"] for item in messages]
        assert session.content_hash is not None
    assert streamed_records == len(messages)


def test_retained_generic_message_object_alias_rejects_invalid_suffix_before_publish(tmp_path: Path) -> None:
    """The shape probe and streamed preparation both require complete JSON EOF."""
    bootstrap_archive_root(tmp_path)
    payload = b'{"id":"neutral-drive-object","messages":[{"role":"user","content":"kept?"}]} trailing'
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.DRIVE,
        blob_hash=blob_hash,
        source_path="neutral-drive.txt",
        directory=tmp_path / "prepared",
        prepare=revision_backfill.prepare_retained_non_json_artifact,
    ) as (artifact, _reader):
        assert artifact.error is not None
        assert artifact.sessions_path is None
        assert artifact.shard_path is None


@pytest.mark.parametrize("cancel", [False, True])
def test_non_json_validation_failure_discards_sealed_artifact_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cancel: bool
) -> None:
    """Validation refusal and cancellation both settle the sealed non-JSON artifact."""
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from tests.infra.retained_jsonl import retained_raw_fixture

    archive = tmp_path / "archive"
    bootstrap_archive_root(archive)
    payload = json.dumps({"conversations": [_chatgpt_session("cleanup", "user", "assistant")]}).encode()
    blob_hash, _size = BlobStore(archive / "blob").write_from_bytes(payload)
    preparation = tmp_path / "prepared"
    preparation.mkdir()
    import polylogue.schemas as schemas

    validation_failure: BaseException = (
        DaemonOperationCancelled("validation cancellation sentinel")
        if cancel
        else RuntimeError("validation cleanup sentinel")
    )

    def fail_validation(*_args: object, **_kwargs: object) -> object:
        raise validation_failure

    monkeypatch.setattr(schemas, "validate_retained_document", fail_validation)
    with retained_raw_fixture(
        root=archive,
        provider=Provider.CHATGPT,
        blob_hash=blob_hash,
        source_path="neutral-original.capture",
    ) as (reader, raw_id):
        expected_error = DaemonOperationCancelled if cancel else RetainedPreparationRetryableError
        with pytest.raises(expected_error) as raised:
            revision_backfill.prepare_retained_non_json_artifact(
                reader,
                raw_id,
                directory=preparation,
                validation_mode=ValidationMode.ADVISORY,
            )

    primary = raised.value if cancel else raised.value.__cause__
    if cancel:
        assert primary is validation_failure
    else:
        assert isinstance(primary, RuntimeError)
    assert str(primary) == str(validation_failure)
    assert list(preparation.iterdir()) == []


def test_unknown_retained_stream_replay_scans_past_oversized_first_record_without_eager_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """UNKNOWN JSONL scans past an oversized first record before streaming replay."""
    bootstrap_archive_root(tmp_path)
    payload = (
        json.dumps({"opaque": "x" * 9_000}, sort_keys=True).encode() + b"\n"
        b'{"type":"session_meta","payload":{"id":"unknown-stream","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
        b'"content":[{"type":"input_text","text":"prefix detected replay"}]}}\n'
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    def refuse_eager_material(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained parser eagerly loaded complete Raw bytes")

    monkeypatch.setattr(PreparedSessionSourceRead, "raw_revision_material", refuse_eager_material)
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="unknown-member.jsonl",
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert [session.provider_session_id for session in sessions] == ["unknown-stream"]


def test_unknown_retained_codex_record_scans_provider_key_past_8k_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A late Codex discriminator in one oversized record remains visible."""
    bootstrap_archive_root(tmp_path)
    late_session_meta = json.dumps(
        {
            "padding": "x" * (8192 + 512),
            "type": "session_meta",
            "payload": {"id": "late-codex", "timestamp": "2026-06-01T00:00:00Z"},
        },
        separators=(",", ":"),
    ).encode()
    payload = (
        late_session_meta
        + b"\n"
        + b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
        + b'"content":[{"type":"input_text","text":"late discriminator"}]}}\n'
    )
    assert b'"type":"session_meta"' not in late_session_meta[:8192]

    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    def refuse_eager_material(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained parser eagerly loaded complete Raw bytes")

    monkeypatch.setattr(PreparedSessionSourceRead, "raw_revision_material", refuse_eager_material)
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="late-codex.jsonl",
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert [session.provider_session_id for session in sessions] == ["late-codex"]


@pytest.mark.parametrize("terminated", [False, True])
def test_unknown_retained_partial_tail_and_terminated_corruption_have_distinct_outcomes(
    tmp_path: Path, terminated: bool
) -> None:
    from tests.infra.retained_jsonl import retained_raw_fixture

    bootstrap_archive_root(tmp_path)
    payload = b'{"padding":"' + b"x" * 80_000 + b'","type":"session_meta","payload":{"id":"late"}'
    payload += b"\n" if terminated else b""
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with (
        retained_raw_fixture(
            root=tmp_path,
            provider=Provider.UNKNOWN,
            blob_hash=blob_hash,
            source_path=str(tmp_path / "huge.jsonl"),
        ) as (reader, raw_id),
        reader.open_raw_revision_material(raw_id) as (_provider, stream, source_path, _kind),
    ):
        if terminated:
            with pytest.raises(PreparedDecodeError) as refused:
                revision_backfill._resolved_retained_provider(stream, source_path)
            assert refused.value.kind is DecodeFailure.JSONL_RECORD
        else:
            provider, _evidence = revision_backfill._resolved_retained_provider(stream, source_path)
            assert provider is Provider.UNKNOWN
        assert stream.tell() == 0


def test_unknown_retained_oversized_provider_record_never_uses_eager_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive prefix evidence survives a record larger than the total scan cap."""
    bootstrap_archive_root(tmp_path)
    payload = (
        json.dumps(
            {
                "sessionId": "oversized-only-provider-record",
                "uuid": "message-1",
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": "x" * 80_000}]},
            }
        ).encode()
        + b"\n"
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    def refuse_eager_material(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained parser eagerly loaded complete Raw bytes")

    monkeypatch.setattr(PreparedSessionSourceRead, "raw_revision_material", refuse_eager_material)
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="oversized-only.jsonl",
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert [session.provider_session_id for session in sessions] == ["oversized-only-provider-record"]


def test_unknown_retained_stream_census_worker_scans_past_oversized_first_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production census worker must discover a later bounded JSONL record."""
    bootstrap_archive_root(tmp_path)
    payload = (
        json.dumps({"opaque": "x" * 9_000}, sort_keys=True).encode()
        + b"\n"
        + b'{"type":"session_meta","payload":{"id":"unknown-worker","timestamp":"2026-06-01T00:00:00Z"}}\n'
        + b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
        + b'"content":[{"type":"input_text","text":"worker replay"}]}}\n'
    )
    import sys
    from builtins import BaseExceptionGroup

    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: pytest.fail("UNKNOWN stream preparation must not eagerly read the blob"),
    )
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "unknown-worker.jsonl"),
    ) as (reader, raw_id):
        original = reader.raw_revision_descriptor(raw_id)
        assert original[1] == blob_hash
        artifact = revision_backfill.prepare_retained_jsonl_artifact(reader, raw_id, directory=tmp_path / "prepared")
        try:
            assert artifact.error is None
            assert artifact.blob_hash == blob_hash
            assert [session.provider_session_id for session in artifact.iter_sessions()] == ["unknown-worker"]
            assert reader.raw_revision_descriptor(raw_id) == original
        finally:
            primary = sys.exception()
            try:
                artifact.discard()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("UNKNOWN worker and close failed", [primary, cleanup]) from None
                raise


def test_unknown_retained_nonstream_jsonl_keeps_complete_payload_fallback(tmp_path: Path) -> None:
    """Positive bounded document evidence may select eager non-stream replay."""
    bootstrap_archive_root(tmp_path)
    document = _chatgpt_session("large-jsonl-document", "bounded evidence")
    document["padding"] = "x" * 9_000
    payload = json.dumps(document, sort_keys=True).encode() + b"\n"
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="large-chatgpt.jsonl",
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert [session.provider_session_id for session in sessions] == ["large-jsonl-document"]


def test_unknown_retained_document_scans_past_oversized_leading_value(tmp_path: Path) -> None:
    """A complete ChatGPT document must scan beyond its bounded prefix.

    The raw is intentionally a source-only UNKNOWN ``conversations.json``
    whose provider-defining fields follow an oversized leading value. This
    drives the historical replay chokepoint against a real archive, rather
    than testing the structural scanner in isolation.
    """
    bootstrap_archive_root(tmp_path)
    document = {"padding": "x" * 9_000, **_chatgpt_session("large-document", "bounded evidence")}
    payload = json.dumps([document]).encode()
    assert b'"mapping"' not in payload[:8192]
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="export/conversations.json",
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> tuple[revision_backfill.PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    results = asyncio.run(replay())
    _assert_complete_original_parser_receipt(tmp_path, raw_id)
    assert results and all(result.scanned == 0 and result.quarantined == 0 for result in results)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [("chatgpt-export:large-document",)]


def test_unknown_retained_document_preserves_long_scalars_and_late_provider_fields(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    text = "x" * 128_000
    payload = json.dumps(
        {
            "padding": text,
            "uuid": "late-claude-provider",
            "name": "Complete retained document",
            "chat_messages": [
                {"uuid": "message-1", "sender": "human", "text": text, "created_at": "2026-08-13T00:00:00Z"}
            ],
        }
    ).encode()
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="export/late-provider.json",
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())
        assert [session.provider_session_id for session in sessions] == ["late-claude-provider"]
        assert sessions[0].messages[0].text == text


def test_unknown_retained_array_ignores_fragment_only_mapping_before_real_provider(tmp_path: Path) -> None:
    """An unrelated mapping fragment cannot claim a whole document sequence."""
    bootstrap_archive_root(tmp_path)
    payload = json.dumps(
        [
            {"mapping": {"foreign-node": {"message": None}}, "metadata": "not a conversation"},
            {
                "uuid": "later-claude-provider",
                "name": "Later Claude provider",
                "chat_messages": [
                    {
                        "uuid": "claude-message",
                        "sender": "human",
                        "text": "real provider evidence",
                        "created_at": "2026-08-13T00:00:00Z",
                    }
                ],
            },
        ]
    ).encode()
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path="export/unknown-array.json",
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> tuple[revision_backfill.PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    results = asyncio.run(replay())
    _assert_complete_original_parser_receipt(tmp_path, raw_id)
    assert results and all(result.scanned == 0 and result.quarantined == 0 for result in results)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [
            ("claude-ai-export:later-claude-provider",)
        ]


def test_prepared_session_spill_uses_its_actual_owned_index_directory(tmp_path: Path) -> None:
    """The canonical artifact keeps private Blob custody and the exact inactive Index destination."""
    import asyncio
    import sys
    from builtins import BaseExceptionGroup

    from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationReplacement
    from polylogue.storage.index_generation import rebuild_source_evidence_snapshot
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    payload = (
        b'{"type":"user","sessionId":"scratch-placement","uuid":"user-1",'
        b'"message":{"role":"user","content":"original scratch"}}\n'
    )
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(tmp_path / ".claude" / "projects" / "neutral" / "scratch-placement.jsonl"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def establish_source() -> None:
        async with prepared_live_convergence_owner(archive_root) as owner:
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    asyncio.run(establish_source())
    store = IndexGenerationStore.for_archive_root(archive_root)
    with write_lease("test.parser-generation.create", archive_root=archive_root):
        generation = store.create(
            owner_id="test-parser-generation", source_snapshot=rebuild_source_evidence_snapshot(archive_root)
        )

    async def exercise() -> None:
        async with prepared_live_convergence_owner(archive_root) as owner:
            retained: list[RawObservationReplacement] = []

            def prepare() -> None:
                index_path = Path(generation.index_path)
                adapter = make_raw_observation_derivation(
                    archive_root,
                    compute_adapter=owner._compute_adapter,
                    index_db_path=index_path,
                    owned_generation=generation,
                )
                frame = raw_observation_frame(archive_root, raw_ids=(raw_id,), index_db_path=index_path)
                replacement = adapter.compute(frame, raw_id, replay_current=True)
                retained.append(replacement)
                try:
                    assert not replacement.needs_source_census and not replacement.needs_source_classification
                    assert frame.source_revision == str(index_path.resolve())
                    assert replacement.reference_seal is not None
                    assert replacement.reference_seal.has_tier_capability("index")
                    assert replacement.reference_seal.index_path == index_path.resolve()
                    scratch = replacement.scratch_directory
                    assert scratch is not None and scratch.exists()
                    assert scratch.parent == BlobStore(archive_root / "blob").staging_root
                    assert replacement.prepared_inputs is not None
                    artifact = replacement.prepared_inputs[raw_id].prepared_artifact
                    assert artifact is not None and artifact.error is None
                    assert artifact.sessions_path is not None and artifact.sessions_path.exists()
                    assert artifact.shard_path is not None and artifact.shard_path.exists()
                    # Each prepared artifact owns one directory inside the
                    # preparation's scratch, so a stale one is discarded alone.
                    artifact_directory = artifact.sessions_path.parent
                    assert artifact_directory.parent == scratch
                    assert artifact.shard_path.parent == artifact_directory
                finally:
                    primary = sys.exception()
                    try:
                        replacement.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup("pinned scratch and close failed", [primary, cleanup]) from None
                        raise
                    else:
                        retained.remove(replacement)
                        assert replacement.scratch_directory is not None
                        assert not replacement.scratch_directory.exists()

            await owner.run_prepared_sync(
                "test.pinned-parser.prepare",
                prepare,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=len(payload),
            )

    try:
        asyncio.run(exercise())
    finally:
        primary = sys.exception()
        try:
            with write_lease("test.parser-generation.discard", archive_root=archive_root):
                assert store.discard_if_inactive(generation)
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("pinned parser and generation close failed", [primary, cleanup]) from None
            raise


@pytest.mark.parametrize(
    "source_path_suffix",
    [
        "subagents/agent-deadbeef.meta.json",
        "workflows/wf-run-1.json",
        "subagents/workflows/wf-run-1/journal.jsonl",
        "jobs/session-a/adopt.json",
    ],
)
def test_parse_one_refuses_declared_fact_artifacts(tmp_path: Path, source_path_suffix: str) -> None:
    """Declared fact artifacts remain durable evidence without becoming sessions."""
    source_path = tmp_path / ".claude" / "projects" / "proj" / "sess" / source_path_suffix
    payload = json.dumps({"agentId": "agent-deadbeef", "transcriptPath": "agent-deadbeef.jsonl"}).encode("utf-8")

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert sessions == []


def test_parse_one_recovery_accepts_session_evidence_at_a_declared_fact_path(tmp_path: Path) -> None:
    """Source-only raw recovery decodes evidence before assigning fact taxonomy."""
    source_path = tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl"
    payload = (
        b'{"parentUuid":null,"type":"user","sessionId":"wf","message":{"role":"user","content":"recover me"},'
        b'"uuid":"user-1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"user-1","type":"assistant","sessionId":"wf","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"recovered"}]},"uuid":"assistant-1",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert len(sessions) == 1
        assert [message.text for message in sessions[0].messages] == ["recover me", "recovered"]


def test_parse_stream_recovery_accepts_session_evidence_at_a_declared_fact_path(tmp_path: Path) -> None:
    """The streamed replay route must inspect fact-path records before refusing them."""
    source_path = tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl"
    payload = BytesIO(
        b'{"parentUuid":null,"type":"user","sessionId":"wf","message":{"role":"user","content":"recover me"},'
        b'"uuid":"user-1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"user-1","type":"assistant","sessionId":"wf","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"recovered"}]},"uuid":"assistant-1",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload.getvalue())
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert len(sessions) == 1
        assert [message.text for message in sessions[0].messages] == ["recover me", "recovered"]


def test_backfill_scans_declared_stream_past_non_session_prefix(tmp_path: Path) -> None:
    """Later Claude records outrank an arbitrarily long fact-artifact prefix.

    The production backfill route must not turn the first 64 non-session
    records into permanent artifact authority when later records prove a
    session.  The archive assertion fails if replay rejects that bounded
    prefix before parsing the rest of the retained JSONL.
    """
    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    payload = _relationship_index_jsonl_bytes(64) + (
        b'{"parentUuid":null,"type":"user","sessionId":"late-session","message":{"role":"user","content":"late evidence"},'
        b'"uuid":"late-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"late-user","type":"assistant","sessionId":"late-session","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"late reply"}]},"uuid":"late-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=source_path,
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> tuple[revision_backfill.PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    results = asyncio.run(replay())
    _assert_complete_original_parser_receipt(tmp_path, raw_id)
    assert results and all(result.scanned == 0 and result.quarantined == 0 for result in results)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [("claude-code-session:late-session",)]


def test_parse_one_refuses_non_conversational_content_with_no_path_rule(tmp_path: Path) -> None:
    """Regression for polylogue-9ykn: a record with no OriginSpec path rule
    at all (so ``_is_declared_non_session_artifact``'s path check alone would
    admit it) must still be refused when its CONTENT carries no positive
    conversation evidence.

    Before this fix, this replay chokepoint (``polylogue ops reset --index``
    / ``devtools`` rebuild-index) only consulted ``artifact_rule_for_path`` --
    a path-pattern allowlist -- so a file with no matching path pattern, like
    a third-party analysis index sitting under
    ``~/.claude/projects/<proj>/analysis/index/``, sailed through unchanged
    on every rebuild even after the live daemon ingest path (which also
    consults the richer content classifier, ``classify_artifact``) learned to
    refuse it. The two "single chokepoints" disagreeing is exactly the
    location-as-identity defect recurring at a second layer.
    """
    source_path = tmp_path / ".claude" / "projects" / "proj" / "analysis" / "index" / "conversation_relationships.jsonl"
    payload = _relationship_index_jsonl_bytes()

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert sessions == []


def test_parse_stream_refuses_non_conversational_content_with_no_path_rule(tmp_path: Path) -> None:
    """Streaming-path sibling of the test above (large multi-GiB JSONL never
    materializes fully; the content-classification sample is bounded to the
    first 64 records instead)."""
    source_path = tmp_path / ".claude" / "projects" / "proj" / "analysis" / "index" / "conversation_relationships.jsonl"
    payload = BytesIO(_relationship_index_jsonl_bytes())

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload.getvalue())
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert sessions == []


def test_prepared_retained_unheadered_codex_append_keeps_its_acquired_native_identity(tmp_path: Path) -> None:
    """A genuine forward append uses its accepted baseline identity without altering acquired bytes."""
    import hashlib
    import sys
    from builtins import BaseExceptionGroup

    from polylogue.archive.revision_authority import append_source_revision
    from polylogue.sources.acquisition_boundary import bound_profile_identity, bound_source_observation, open_bound_path
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.retained_jsonl import prepared_source_fixture

    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "sessions" / "append.jsonl"
    source_path.parent.mkdir(parents=True)
    baseline = b'{"type":"session_meta","payload":{"id":"append-owner"}}\n'
    delta = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
    )
    source_path.write_bytes(baseline)
    baseline_revision = hashlib.sha256(baseline).hexdigest()
    key = "codex-session:append-owner"
    with open_bound_path(source_path, None) as original:
        profile = bound_profile_identity(original)
        baseline_profile_key = profile.key if profile else None
        canonical, observation = bound_source_observation(original)
        assert canonical is not None and observation is not None
        acquired = original.read()
        assert acquired == baseline
        with (
            write_lease("test.codex.append.baseline", archive_root=tmp_path),
            ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
        ):
            baseline_raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=acquired,
                source_path=str(source_path),
                canonical_source_path=canonical,
                captured_profile_key=profile.key if profile else None,
                native_id="append-owner",
                acquired_at_ms=1,
                file_mtime_ms=observation[3] // 1_000_000,
                revision=RawRevisionEnvelope(
                    key, RawRevisionKind.FULL, baseline_revision, 0, authority=RawRevisionAuthority.BYTE_PROVEN
                ),
            )
            archive.commit()
    with source_path.open("ab") as writer:
        writer.write(delta)
    with open_bound_path(source_path, None) as original:
        profile = bound_profile_identity(original)
        append_canonical, observation = bound_source_observation(original)
        assert (profile.key if profile else None) == baseline_profile_key
        assert append_canonical == canonical and observation is not None
        acquired = original.read()
        assert acquired == baseline + delta
        original_delta = acquired[len(baseline) :]
        assert original_delta == delta
        with (
            write_lease("test.codex.append.range", archive_root=tmp_path),
            ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
        ):
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=original_delta,
                source_path=str(source_path),
                canonical_source_path=append_canonical,
                captured_profile_key=profile.key if profile else None,
                native_id="append-owner",
                acquired_at_ms=2,
                file_mtime_ms=observation[3] // 1_000_000,
                revision=RawRevisionEnvelope(
                    key,
                    RawRevisionKind.APPEND,
                    append_source_revision(baseline_revision, hashlib.sha256(original_delta).hexdigest()),
                    1,
                    predecessor_source_revision=baseline_revision,
                    predecessor_raw_id=baseline_raw_id,
                    baseline_raw_id=baseline_raw_id,
                    append_start_offset=len(baseline),
                    append_end_offset=len(acquired),
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )
            archive.commit()
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_revision_descriptor(raw_id)[1] == hashlib.sha256(delta).hexdigest()
        artifact = revision_backfill.prepare_retained_jsonl_artifact(reader, raw_id, directory=tmp_path / "prepared")
        try:
            assert artifact.error is None
            (session,) = artifact.iter_sessions()
            assert session.provider_session_id == "append-owner"
            assert [message.text for message in session.messages] == ["one"]
        finally:
            primary = sys.exception()
            try:
                artifact.discard()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("append parsing and artifact close failed", [primary, cleanup]) from None
                raise


def test_parse_one_still_replays_real_claude_code_sessions_with_no_path_rule(tmp_path: Path) -> None:
    """Guard against the regression-direction failure mode: the content gate
    added for polylogue-9ykn must not start refusing genuine Claude Code
    session records that (like most session JSONL files) carry no matching
    OriginSpec path rule."""
    source_path = tmp_path / ".claude" / "projects" / "proj" / "analysis" / "index" / "sess-real.jsonl"
    record = {
        "uuid": "u1",
        "parentUuid": None,
        "sessionId": "sess-real",
        "type": "user",
        "message": {"role": "user", "content": "hello"},
        "timestamp": "2026-05-01T00:00:00.000Z",
    }
    payload = (json.dumps(record) + "\n").encode("utf-8")

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    blob_hash, _size = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=archive_root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        sessions = list(artifact.iter_sessions())

        assert len(sessions) == 1
        assert sessions[0].messages


def test_historical_backfill_replays_single_session_state_db(tmp_path: Path) -> None:
    """The supplied retained owner publishes a single-session state export from original Raw evidence."""

    bootstrap_archive_root(tmp_path)
    payload = _single_session_state_db_bytes(tmp_path)
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.HERMES,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "hermes-home" / "state.db"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> tuple[revision_backfill.PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    results = asyncio.run(replay())
    _assert_complete_original_parser_receipt(tmp_path, raw_id)

    assert results and all(result.scanned == 0 for result in results)
    assert sum(result.replayed_logical_sources for result in results) == 1
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)
    assert all(result.quarantined == 0 for result in results)


def test_historical_backfill_streams_codex_raw_without_eager_blob_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap_archive_root(tmp_path)
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused (never becomes a session) --
    # append one real message record so this fixture keeps testing what it
    # means to test (stream-safe blob I/O), not the now-refused empty shape.
    payload = (
        b'{"type":"session_meta","payload":{"id":"streamed"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CODEX,
        blob_hash=blob_hash,
        source_path="streamed.jsonl",
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> tuple[revision_backfill.PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: pytest.fail("stream-safe revision replay must not eagerly read a blob"),
    )

    results = asyncio.run(replay())
    _assert_complete_original_parser_receipt(tmp_path, raw_id)

    assert results and all(result.scanned == 0 for result in results)
    assert sum(result.replayed_logical_sources for result in results) == 1
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)


def test_retained_replay_refuses_a_malformed_middle_record_with_a_terminal_census(tmp_path: Path) -> None:
    """Original complete malformed records settle terminally without publishing a partial session."""
    import asyncio

    from polylogue.storage.blob_store import BlobStore
    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import prepared_source_fixture, retained_raw_fixture

    bootstrap_archive_root(tmp_path)
    payload = _CLAUDE_USER_RECORD + b'{"type":"user","message":oops}\n' + _CLAUDE_ASSISTANT_RECORD
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "projects" / "proj" / "strict-replay.jsonl"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            refused: list[str] = []
            result = (
                await owner.replay_retained_raw_ids(
                    (raw_id,), on_terminal_refusal=lambda _keys, refusal: refused.append(refusal.raw_id)
                )
            ).require_complete()
            assert result == ()
            assert refused == [raw_id]

    asyncio.run(replay())
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_parser_census_is_current(raw_id)
        assert reader.raw_terminal_decode_refusal(raw_id) is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        status, keys = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone()
        artifact_kinds = {
            str(row[0]) for row in conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,))
        }
        (parse_error,) = conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
    assert status == "complete"
    assert tuple(iter_parser_census_logical_keys(keys)) == ()
    assert RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value in artifact_kinds
    assert parse_error is not None
    asyncio.run(replay())
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "payload",
    [b"{", b'{"title": "cut", "mapping": {"n": {"id": "n", "message": '],
    ids=["opening-brace", "truncated-mapping"],
)
def test_retained_replay_settles_an_undecodable_json_document_as_terminal(tmp_path: Path, payload: bytes) -> None:
    """Original undecodable documents retain the same terminal census law as malformed records."""
    import asyncio

    from polylogue.storage.blob_store import BlobStore
    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import prepared_source_fixture, retained_raw_fixture

    bootstrap_archive_root(tmp_path)
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CHATGPT,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "exports" / "conversation.json"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            refused: list[str] = []
            result = (
                await owner.replay_retained_raw_ids(
                    (raw_id,), on_terminal_refusal=lambda _keys, refusal: refused.append(refusal.raw_id)
                )
            ).require_complete()
            assert result == ()
            assert refused == [raw_id]

    asyncio.run(replay())
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_parser_census_is_current(raw_id)
        assert reader.raw_terminal_decode_refusal(raw_id) is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (status,) = conn.execute("SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)).fetchone()
        artifact_kinds = {
            str(row[0]) for row in conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,))
        }
    assert status == "complete"
    assert RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value in artifact_kinds
    asyncio.run(replay())
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_retained_replay_parses_the_complete_prefix_before_an_unterminated_tail(tmp_path: Path) -> None:
    """Original acquisition parses complete records and refuses terminated corruption."""
    from polylogue.storage.blob_store import BlobStore
    from tests.infra.retained_jsonl import retained_parser_fixture

    bootstrap_archive_root(tmp_path)
    source_path = str(tmp_path / "projects" / "proj" / "strict-replay.jsonl")
    complete = _CLAUDE_USER_RECORD + _CLAUDE_ASSISTANT_RECORD
    payload = complete + b'{"parentUuid":"assistant-1","type":"user","mess'
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=source_path,
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None
        assert artifact.blob_hash == blob_hash
        (session,) = artifact.iter_sessions()
        assert [message.text for message in session.messages] == ["kept", "reply"]

    corrupt = complete + b'{"type":"user","message":oops}\n'
    corrupt_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(corrupt)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=corrupt_hash,
        source_path=source_path,
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.blob_hash == corrupt_hash
        assert artifact.error is not None
        assert artifact.decode_failure is not None
        assert (
            terminal_decode_evidence(
                revision_backfill.retained_parse_exception(artifact.error, artifact.decode_failure),
                provider=Provider.CLAUDE_CODE,
            )
            is RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT
        )
        assert artifact.sessions_path is None
        with pytest.raises(RuntimeError):
            next(artifact.iter_sessions())


def test_prepared_decode_refusal_keeps_its_kind_across_the_artifact_boundary(tmp_path: Path) -> None:
    """Original malformed records retain their structured refusal after preparation."""
    from polylogue.storage.blob_store import BlobStore
    from tests.infra.retained_jsonl import retained_parser_fixture

    payload = _CLAUDE_USER_RECORD + b'{"type":"user","message":oops}\n' + _CLAUDE_ASSISTANT_RECORD
    bootstrap_archive_root(tmp_path)
    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "projects" / "proj" / "strict-replay.jsonl"),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is not None
        assert artifact.decode_failure is not None
        assert artifact.unsupported_shape is False
        assert artifact.blob_hash == blob_hash
        assert (
            terminal_decode_evidence(
                revision_backfill.retained_parse_exception(artifact.error, artifact.decode_failure),
                provider=Provider.CLAUDE_CODE,
            )
            is RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT
        )
        assert artifact.sessions_path is None
        with pytest.raises(RuntimeError):
            next(artifact.iter_sessions())
    assert revision_backfill.retained_parse_exception("neutral failure", None).__class__ is RuntimeError


def _assert_complete_original_parser_receipt(root: Path, raw_id: str) -> None:
    """The census receipt is separate from the returned terminal replay receipt."""
    from contextlib import closing

    with closing(sqlite3.connect(root / "source.db")) as conn:
        receipt = conn.execute("SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)).fetchone()
    assert receipt == ("complete",)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"contentKey":"fact","agentId":"agent"}\n',
        b'{"parentUuid":"foreign-parent","uuid":"fragment-only"}\n',
        b'{"type":"assistant","sessionId":"header-only","uuid":"header"}\n',
    ],
)
def test_retained_fact_recovery_requires_accepted_session_evidence(tmp_path: Path, payload: bytes) -> None:
    from io import BytesIO

    from polylogue.archive.artifact_taxonomy import classify_artifact_stream, strong_path_classification

    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    source_path = tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl"
    fact = strong_path_classification(source_path, provider=Provider.CLAUDE_CODE)
    assert fact is not None and not fact.parse_as_session
    # At a ``fact`` path, enveloped records reach the parser as candidacy;
    # the parser's finding of no accepted session decides the refusal, and
    # the input keeps the path's fact classification.
    ordinary = classify_artifact_stream(
        BytesIO(payload), provider=Provider.CLAUDE_CODE, source_path=source_path, wire_format="jsonl"
    )
    if ordinary.proved_non_session:
        assert ordinary.classification == fact
    else:
        assert ordinary.classification.parse_as_session
    blob_hash, _size = BlobStore(root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.error is None and list(artifact.iter_sessions()) == []
        selected = artifact.stream_classification()
        assert selected is not None and selected.proved_non_session
        assert selected.classification == fact


def test_retained_fact_recovery_refuses_an_incomplete_provider_record(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    source_path = tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl"
    payload = b'{"type":"user","sessionId":"wf","message":{"role":"user","content":"unfinished'
    blob_hash, _size = BlobStore(root / "blob").write_from_bytes(payload)
    with retained_parser_fixture(
        root=root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
    ) as (artifact, _reader):
        assert artifact.parsed_prefix_size == 0
        if artifact.error is None:
            assert list(artifact.iter_sessions()) == []
        else:
            assert artifact.sessions_path is None and artifact.decode_failure is not None
