from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from polylogue.core.enums import ArtifactSupportStatus, Origin, ValidationStatus
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHistorySidecar,
    ArchiveHookEvent,
    ArchiveRawArtifactEnvelope,
    ArchiveRawSessionEnvelope,
    ArchiveSourceArtifact,
    ArchiveSourceBlobRef,
    deterministic_blob_hash,
    deterministic_history_sidecar_id,
    deterministic_raw_session_id,
    list_hook_events,
    list_raw_artifacts,
    read_archive_raw_session_envelope,
    read_history_sidecar,
    read_hook_event,
    read_raw_artifact,
    write_history_sidecar,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def test_archive_tiers_source_writer_materializes_raw_session_with_blob_ref(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    payload = b'{"kind":"session","messages":["hello"]}'
    sidecar_payload: dict[str, object] = {"history": [{"event": "paste", "message_id": "m1"}]}
    sidecar_id = write_history_sidecar(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/record.jsonl",
        payload=sidecar_payload,
        observed_at_ms=1_767_000_000_010,
    )
    computed_blob_hash = deterministic_blob_hash(payload)
    expected_raw_id = deterministic_raw_session_id(
        Origin.CLAUDE_CODE_SESSION,
        "/tmp/record.jsonl",
        0,
        computed_blob_hash,
        native_id="session-1",
    )

    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/record.jsonl",
        source_index=0,
        native_id="session-1",
        payload=payload,
        acquired_at_ms=1_767_000_000_000,
        parsed_at_ms=1_767_000_000_050,
        validation_status=ValidationStatus.PASSED,
        validation_drift_count=0,
        additional_blob_refs=(
            ArchiveSourceBlobRef(
                blob_hash=deterministic_blob_hash(b"attach"),
                ref_type="attachment",
                source_path="/tmp/record.jsonl",
                size_bytes=6,
                acquired_at_ms=1_767_000_000_001,
            ),
        ),
        artifact=ArchiveSourceArtifact(
            artifact_id="artifact-1",
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            artifact_kind="session_export",
            classification_reason="expected",
            support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
            parse_as_session=True,
            schema_eligible=True,
            first_observed_at_ms=1_767_000_000_100,
            last_observed_at_ms=1_767_000_000_100,
        ),
        hook_event=ArchiveHookEvent(
            hook_event_id="hook-1",
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            event_type="source_opened",
            payload={"path": "/tmp/record.jsonl"},
            observed_at_ms=1_767_000_000_120,
            session_native_id="session-1",
        ),
    )

    assert raw_id == expected_raw_id

    envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert isinstance(envelope, ArchiveRawSessionEnvelope)
    assert envelope.raw_id == expected_raw_id
    assert envelope.origin == Origin.CLAUDE_CODE_SESSION.value
    assert envelope.native_id == "session-1"
    assert envelope.source_path == "/tmp/record.jsonl"
    assert envelope.blob_hash == computed_blob_hash
    assert envelope.blob_size == len(payload)
    assert envelope.validation_status == ValidationStatus.PASSED.value
    assert len(envelope.blob_refs) == 2
    assert {blob.ref_type for blob in envelope.blob_refs} == {"raw_payload", "attachment"}
    assert envelope.artifact_ids == ("artifact-1",)
    assert envelope.hook_event_ids == ("hook-1",)
    assert envelope.history_sidecar_ids == (sidecar_id,)

    sidecar = read_history_sidecar(conn, sidecar_id)
    assert isinstance(sidecar, ArchiveHistorySidecar)
    assert sidecar.payload == sidecar_payload
    assert sidecar.content_hash == deterministic_blob_hash(b'{"history":[{"event":"paste","message_id":"m1"}]}')

    artifact = read_raw_artifact(conn, "artifact-1")
    assert artifact == ArchiveRawArtifactEnvelope(
        artifact_id="artifact-1",
        raw_id=raw_id,
        origin=Origin.CLAUDE_CODE_SESSION.value,
        source_path="/tmp/record.jsonl",
        source_index=0,
        artifact_kind="session_export",
        support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE.value,
        classification_reason="expected",
        parse_as_session=True,
        schema_eligible=True,
        malformed_jsonl_lines=0,
        decode_error=None,
        cohort_id=None,
        link_group_key=None,
        sidecar_agent_type=None,
        first_observed_at_ms=1_767_000_000_100,
        last_observed_at_ms=1_767_000_000_100,
    )
    assert list_raw_artifacts(conn, raw_id=raw_id) == (artifact,)

    hook_event = read_hook_event(conn, "hook-1")
    assert hook_event == ArchiveHookEvent(
        hook_event_id="hook-1",
        origin=Origin.CLAUDE_CODE_SESSION.value,
        source_path="/tmp/record.jsonl",
        event_type="source_opened",
        payload={"path": "/tmp/record.jsonl"},
        observed_at_ms=1_767_000_000_120,
        native_id=None,
        session_native_id="session-1",
    )
    assert list_hook_events(conn, origin=Origin.CLAUDE_CODE_SESSION, session_native_id="session-1") == (hook_event,)


def test_archive_tiers_source_writer_replays_hook_events_idempotently(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    payload = b'{"kind":"session","messages":["hello"]}'
    hook_event = ArchiveHookEvent(
        hook_event_id="hook-1",
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/record.jsonl",
        event_type="source_opened",
        payload={"path": "/tmp/record.jsonl"},
        observed_at_ms=1_767_000_000_120,
        session_native_id="session-1",
    )

    write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/record.jsonl",
        source_index=0,
        native_id="session-1",
        payload=payload,
        acquired_at_ms=1_767_000_000_000,
        hook_event=hook_event,
    )
    write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/record.jsonl",
        source_index=0,
        native_id="session-1",
        payload=payload,
        acquired_at_ms=1_767_000_000_000,
        hook_event=replace(hook_event, payload={"path": "/tmp/record.jsonl", "replayed": True}),
    )

    assert list_hook_events(conn, origin=Origin.CLAUDE_CODE_SESSION, session_native_id="session-1") == (
        replace(hook_event, payload={"path": "/tmp/record.jsonl", "replayed": True}),
    )


def test_archive_tiers_source_writer_deterministic_ids() -> None:
    payload = b"stable"
    blob_hash = deterministic_blob_hash(payload)
    raw_id_a = deterministic_raw_session_id(
        Origin.CLAUDE_CODE_SESSION, "/tmp/record.jsonl", 1, blob_hash, native_id="same"
    )
    raw_id_b = deterministic_raw_session_id(
        Origin.CLAUDE_CODE_SESSION, "/tmp/record.jsonl", 1, blob_hash, native_id="same"
    )

    assert raw_id_a == raw_id_b
    assert raw_id_a != deterministic_raw_session_id(
        Origin.CHATGPT_EXPORT, "/tmp/record.jsonl", 1, blob_hash, native_id="same"
    )
    assert deterministic_history_sidecar_id(Origin.CLAUDE_CODE_SESSION, "/tmp/history.jsonl", blob_hash) == (
        deterministic_history_sidecar_id(Origin.CLAUDE_CODE_SESSION, "/tmp/history.jsonl", blob_hash)
    )
    assert deterministic_history_sidecar_id(Origin.CLAUDE_CODE_SESSION, "/tmp/history.jsonl", blob_hash) != (
        deterministic_history_sidecar_id(Origin.CODEX_SESSION, "/tmp/history.jsonl", blob_hash)
    )
