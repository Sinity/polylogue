from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider, ValidationStatus
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    ArchiveRawArtifactEnvelope,
    ArchiveRawSessionEnvelope,
    ArchiveSourceArtifact,
    ArchiveSourceBlobRef,
    deterministic_blob_hash,
    deterministic_raw_session_id,
    list_hook_events,
    list_raw_artifacts,
    read_archive_raw_session_envelope,
    read_capture_mode_resolution,
    read_hook_event,
    read_raw_artifact,
    record_raw_container_coordinate,
    upsert_raw_artifact,
    write_source_hook_event,
    write_source_raw_session,
    write_source_raw_session_blob_ref,
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
        capture_mode=Provider.CLAUDE_CODE,
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

    assert (
        conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0] == "claude-code"
    )

    assert raw_id == expected_raw_id

    envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert isinstance(envelope, ArchiveRawSessionEnvelope)
    assert envelope.raw_id == expected_raw_id
    assert envelope.origin == Origin.CLAUDE_CODE_SESSION.value
    assert envelope.capture_mode == Provider.CLAUDE_CODE.value
    assert envelope.native_id == "session-1"
    assert envelope.source_path == "/tmp/record.jsonl"
    assert envelope.blob_hash == computed_blob_hash
    assert envelope.blob_size == len(payload)
    assert envelope.validation_status == ValidationStatus.PASSED.value
    assert len(envelope.blob_refs) == 2
    assert {blob.ref_type for blob in envelope.blob_refs} == {"raw_payload", "attachment"}
    assert envelope.artifact_ids == ("artifact-1",)
    assert envelope.hook_event_ids == ("hook-1",)

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


def test_raw_writer_refuses_invalid_storage_blob_category_before_commit(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        with pytest.raises(ValueError, match="ref_type"):
            write_source_raw_session(
                conn,
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="/tmp/record.jsonl",
                source_index=0,
                payload=b"payload",
                acquired_at_ms=1,
                additional_blob_refs=(
                    ArchiveSourceBlobRef(
                        blob_hash=deterministic_blob_hash(b"attachment"),
                        ref_type="not-a-blob-category",  # type: ignore[arg-type]
                        source_path="/tmp/record.jsonl",
                        size_bytes=10,
                        acquired_at_ms=1,
                    ),
                ),
            )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
    finally:
        conn.close()


def test_container_coordinate_writer_refuses_invalid_format_before_persistence(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            source_index=0,
            payload=b"payload",
            acquired_at_ms=1,
        )
        with pytest.raises(ValueError, match="coordinate_format"):
            record_raw_container_coordinate(
                conn,
                raw_id,
                coordinate_format="not-a-format",  # type: ignore[arg-type]
                entry_ordinal=0,
                split_index=0,
                addressing_mode=None,
            )
        assert conn.execute("SELECT COUNT(*) FROM raw_container_coordinates").fetchone()[0] == 0
    finally:
        conn.close()


def test_hook_writer_refuses_invalid_storage_carrier_role_before_persistence(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    hook_event = ArchiveHookEvent(
        hook_event_id="hook-invalid-role",
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/hook.jsonl",
        event_type="source_opened",
        payload={},
        observed_at_ms=1,
    )
    with pytest.raises(ValueError, match="carrier_role"):
        write_source_hook_event(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/hook.jsonl",
            payload=b"{}",
            acquired_at_ms=1,
            raw_id="raw-hook",
            hook_event=hook_event,
            carrier_role="not-a-carrier-role",  # type: ignore[arg-type]
        )
    assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hook_event_carriers").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize(
    ("column", "reader_field"),
    [
        ("origin", "origin"),
        ("capture_mode", "capture_mode"),
        ("validation_status", "validation_status"),
        ("validation_mode", "validation_mode"),
    ],
)
def test_raw_session_hydration_refuses_unowned_membership_values(
    tmp_path: Path, column: str, reader_field: str
) -> None:
    """A SQL bypass cannot leak an unowned durable token through a typed reader."""
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            source_index=0,
            payload=b"payload",
            acquired_at_ms=1,
        )
        conn.execute(f"UPDATE raw_sessions SET {column}=? WHERE raw_id=?", ("not-a-vocabulary-member", raw_id))

        with pytest.raises(ValueError, match=reader_field):
            read_archive_raw_session_envelope(conn, raw_id)
    finally:
        conn.close()


def test_raw_session_hydration_refuses_unowned_blob_ref_type(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            source_index=0,
            payload=b"payload",
            acquired_at_ms=1,
        )
        # Model corrupt legacy bytes by disabling SQLite CHECK enforcement for
        # this one fixture; production opens retain normal constraint checks.
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute("UPDATE blob_refs SET ref_type='not-a-vocabulary-member' WHERE ref_id=?", (raw_id,))

        with pytest.raises(ValueError, match="ref_type"):
            read_archive_raw_session_envelope(conn, raw_id)
    finally:
        conn.close()


def test_artifact_and_hook_hydration_refuse_unowned_domain_values(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            source_path="/tmp/record.jsonl",
            source_index=0,
            payload=b"payload",
            acquired_at_ms=1,
            artifact=ArchiveSourceArtifact(
                artifact_id="artifact-1",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="/tmp/record.jsonl",
                source_index=0,
                artifact_kind="session_export",
                classification_reason="fixture",
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
            ),
            hook_event=ArchiveHookEvent(
                hook_event_id="hook-1",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="/tmp/record.jsonl",
                event_type="source_opened",
                payload={},
                observed_at_ms=1,
            ),
        )
        conn.execute("UPDATE raw_artifacts SET support_status='invalid' WHERE artifact_id='artifact-1'")
        conn.execute("UPDATE raw_hook_events SET origin='invalid' WHERE hook_event_id='hook-1'")

        with pytest.raises(ValueError, match="artifact.support_status"):
            read_raw_artifact(conn, "artifact-1")
        with pytest.raises(ValueError, match="hook_event.origin"):
            read_hook_event(conn, "hook-1")
    finally:
        conn.close()


def test_source_artifact_upsert_keeps_coordinate_deduplication_and_raw_failure_fanout(
    tmp_path: Path,
) -> None:
    conn = _connect(tmp_path / "source.db")
    raw_ids = [
        write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/shared.jsonl",
            source_index=0,
            payload=f"payload-{suffix}".encode(),
            acquired_at_ms=index + 1,
        )
        for index, suffix in enumerate(("old", "new"))
    ]

    upsert_raw_artifact(
        conn,
        raw_ids[0],
        ArchiveSourceArtifact(
            artifact_id="failure-old",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/shared.jsonl",
            source_index=0,
            artifact_kind="deferred_cas_frontier",
            classification_reason="deferred_cas_frontier",
            support_status=ArtifactSupportStatus.PARTIAL_DECODE,
        ),
    )
    upsert_raw_artifact(
        conn,
        raw_ids[1],
        ArchiveSourceArtifact(
            artifact_id="failure-new",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/shared.jsonl",
            source_index=0,
            artifact_kind="deferred_cas_frontier",
            classification_reason="deferred_cas_frontier",
            support_status=ArtifactSupportStatus.PARTIAL_DECODE,
        ),
    )
    upsert_raw_artifact(
        conn,
        raw_ids[0],
        ArchiveSourceArtifact(
            artifact_id="failure-old-refresh",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/shared.jsonl",
            source_index=0,
            artifact_kind="terminal_corrupt_input",
            classification_reason="terminal_corrupt_input",
            support_status=ArtifactSupportStatus.DECODE_FAILED,
        ),
    )

    rows = conn.execute(
        """
        SELECT raw_id, origin, source_path, source_index, artifact_kind, support_status
        FROM raw_artifacts
        WHERE origin = ? AND source_path = ? AND source_index = 0
        ORDER BY raw_id
        """,
        (Origin.CODEX_SESSION.value, "/tmp/shared.jsonl"),
    ).fetchall()
    assert {tuple(row) for row in rows} == {
        (
            raw_ids[0],
            Origin.CODEX_SESSION.value,
            "/tmp/shared.jsonl",
            0,
            "terminal_corrupt_input",
            ArtifactSupportStatus.DECODE_FAILED.value,
        ),
        (
            raw_ids[1],
            Origin.CODEX_SESSION.value,
            "/tmp/shared.jsonl",
            0,
            "deferred_cas_frontier",
            ArtifactSupportStatus.PARTIAL_DECODE.value,
        ),
    }

    upsert_raw_artifact(
        conn,
        raw_ids[0],
        ArchiveSourceArtifact(
            artifact_id="ordinary-coordinate",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/shared.jsonl",
            source_index=0,
            artifact_kind="session_export",
            classification_reason="session_export",
            support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
        ),
    )
    ordinary = conn.execute(
        """
        SELECT artifact_id, raw_id, artifact_kind
        FROM raw_artifacts
        WHERE origin = ? AND source_path = ? AND source_index = 0
          AND artifact_kind = 'session_export'
        """,
        (Origin.CODEX_SESSION.value, "/tmp/shared.jsonl"),
    ).fetchone()
    assert ordinary is not None
    assert tuple(ordinary) == ("ordinary-coordinate", raw_ids[0], "session_export")


def test_source_artifact_upsert_refreshes_current_equal_time_carrier(tmp_path: Path) -> None:
    """The current raw may refine its own coordinate even at the same timestamp."""
    conn = _connect(tmp_path / "source.db")
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/current.jsonl",
        source_index=0,
        payload=b"current",
        acquired_at_ms=1,
    )
    upsert_raw_artifact(
        conn,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id="deferred-current",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/current.jsonl",
            source_index=0,
            artifact_kind="deferred_cas_frontier",
            classification_reason="deferred",
            support_status=ArtifactSupportStatus.PARTIAL_DECODE,
        ),
    )
    upsert_raw_artifact(
        conn,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id="terminal-current",
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/current.jsonl",
            source_index=0,
            artifact_kind="terminal_corrupt_input",
            classification_reason="corrupt",
            support_status=ArtifactSupportStatus.DECODE_FAILED,
        ),
    )

    row = conn.execute(
        "SELECT artifact_id, artifact_kind, support_status, classification_reason FROM raw_artifacts"
    ).fetchone()
    assert row is not None
    assert tuple(row) == (
        "deferred-current",
        "terminal_corrupt_input",
        ArtifactSupportStatus.DECODE_FAILED.value,
        "corrupt",
    )


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
        hook_event=hook_event,
    )

    assert list_hook_events(conn, origin=Origin.CLAUDE_CODE_SESSION, session_native_id="session-1") == (hook_event,)


def test_archive_tiers_source_writer_keeps_multiple_raw_captures_for_one_native_id(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")

    first_raw_id = write_source_raw_session(
        conn,
        origin=Origin.CHATGPT_EXPORT,
        source_path="/captures/direct.json",
        source_index=0,
        native_id="conversation-1",
        payload=b'{"title":"direct"}',
        acquired_at_ms=1_767_000_000_000,
    )
    second_raw_id = write_source_raw_session(
        conn,
        origin=Origin.CHATGPT_EXPORT,
        source_path="/captures/browser.json",
        source_index=0,
        native_id="conversation-1",
        payload=b'{"title":"browser"}',
        acquired_at_ms=1_767_000_000_100,
    )

    rows = conn.execute(
        """
        SELECT raw_id, source_path
        FROM raw_sessions
        WHERE origin = ? AND native_id = ?
        ORDER BY source_path
        """,
        (Origin.CHATGPT_EXPORT.value, "conversation-1"),
    ).fetchall()

    assert first_raw_id != second_raw_id
    assert [(row["raw_id"], row["source_path"]) for row in rows] == [
        (second_raw_id, "/captures/browser.json"),
        (first_raw_id, "/captures/direct.json"),
    ]


def test_source_writers_backfill_legacy_capture_mode_on_duplicate_raw_id(tmp_path: Path) -> None:
    """A post-migration re-acquisition enriches, but never replaces, NULL provenance."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"chunkedPrompt":{"chunks":[]}}'
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        source_path="/captures/aistudio.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=1,
    )
    assert conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0] is None

    assert (
        write_source_raw_session(
            conn,
            origin=Origin.AISTUDIO_DRIVE,
            capture_mode=Provider.DRIVE,
            source_path="/captures/aistudio.json",
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )
        == raw_id
    )
    assert conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0] == "drive"

    blob_hash = deterministic_blob_hash(b"blob-backed aistudio")
    blob_raw_id = write_source_raw_session_blob_ref(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        source_path="/captures/aistudio-blob.json",
        source_index=0,
        blob_hash=blob_hash,
        blob_size=len(b"blob-backed aistudio"),
        acquired_at_ms=2,
    )
    assert conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (blob_raw_id,)).fetchone()[0] is None
    assert (
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.AISTUDIO_DRIVE,
            capture_mode=Provider.DRIVE,
            source_path="/captures/aistudio-blob.json",
            source_index=0,
            blob_hash=blob_hash,
            blob_size=len(b"blob-backed aistudio"),
            acquired_at_ms=2,
        )
        == blob_raw_id
    )
    assert (
        conn.execute("SELECT capture_mode FROM raw_sessions WHERE raw_id = ?", (blob_raw_id,)).fetchone()[0] == "drive"
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


def test_source_reference_commit_atomically_consumes_publication_reservation(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    conn = _connect(db_path)
    payload = b"reserved raw payload"
    blob_hash = deterministic_blob_hash(payload)
    conn.execute(
        """
        INSERT INTO blob_publication_reservations (
            publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
        ) VALUES ('receipt-1', ?, ?, 'publisher', 1)
        """,
        (blob_hash, len(payload)),
    )
    conn.commit()

    write_source_raw_session(
        conn,
        origin=Origin.CHATGPT_EXPORT,
        source_path="/captures/reserved.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=2,
        blob_publication_receipt_id="receipt-1",
        manage_transaction=False,
    )

    observer = sqlite3.connect(db_path)
    try:
        assert observer.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1
        assert observer.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        conn.commit()
        assert observer.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
        assert observer.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    finally:
        observer.close()
        conn.close()


def test_capture_mode_resolution_ambiguous_gemini_then_drive(tmp_path: Path) -> None:
    """A content-identical GEMINI export observed first, then a live DRIVE
    acquisition of the exact same bytes, must retain BOTH observed modes
    (polylogue-buns AC1) instead of the DRIVE observation silently vanishing
    behind the first-known cache."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"chunkedPrompt": {"chunks": []}}'

    raw_id = write_source_raw_session(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.GEMINI,
        source_path="/tmp/export.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=1_000,
    )
    write_source_raw_session(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.DRIVE,
        source_path="/tmp/export.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=2_000,
        raw_id=raw_id,
    )

    resolution = read_capture_mode_resolution(conn, raw_id)
    assert resolution.status == "ambiguous"
    assert set(resolution.modes) == {Provider.GEMINI, Provider.DRIVE}
    # Order reflects first-observation time.
    assert resolution.modes == (Provider.GEMINI, Provider.DRIVE)

    # The cached convenience column is unchanged: first-known-mode wins.
    envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert envelope.capture_mode == Provider.GEMINI.value

    # AC3: blob dedup is untouched -- one raw_payload ref for this raw_id.
    blob_ref_count = conn.execute(
        "SELECT COUNT(*) FROM blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'",
        (raw_id,),
    ).fetchone()[0]
    assert blob_ref_count == 1
    conn.close()


def test_capture_mode_resolution_ambiguous_drive_then_gemini(tmp_path: Path) -> None:
    """Mirror order of the above: a live DRIVE acquisition observed first,
    then a GEMINI export of the same bytes -- both orders must behave the
    same way (polylogue-buns AC4)."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"chunkedPrompt": {"chunks": []}}'

    raw_id = write_source_raw_session(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.DRIVE,
        source_path="/tmp/live-drive.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=1_000,
    )
    write_source_raw_session(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.GEMINI,
        source_path="/tmp/live-drive.json",
        source_index=0,
        payload=payload,
        acquired_at_ms=2_000,
        raw_id=raw_id,
    )

    resolution = read_capture_mode_resolution(conn, raw_id)
    assert resolution.status == "ambiguous"
    assert set(resolution.modes) == {Provider.GEMINI, Provider.DRIVE}
    assert resolution.modes == (Provider.DRIVE, Provider.GEMINI)

    envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert envelope.capture_mode == Provider.DRIVE.value

    blob_ref_count = conn.execute(
        "SELECT COUNT(*) FROM blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'",
        (raw_id,),
    ).fetchone()[0]
    assert blob_ref_count == 1
    conn.close()


def test_capture_mode_resolution_unambiguous_single_observation(tmp_path: Path) -> None:
    """A raw with exactly one acquisition mode ever observed reads as unambiguous,
    not merely as an implicit absence of ambiguity."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"kind":"session"}'
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode=Provider.CLAUDE_CODE,
        source_path="/tmp/single.jsonl",
        source_index=0,
        payload=payload,
        acquired_at_ms=1_000,
    )
    # Repeat acquisition of the identical raw is a no-op for ambiguity.
    write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode=Provider.CLAUDE_CODE,
        source_path="/tmp/single.jsonl",
        source_index=0,
        payload=payload,
        acquired_at_ms=2_000,
        raw_id=raw_id,
    )

    resolution = read_capture_mode_resolution(conn, raw_id)
    assert resolution.status == "unambiguous"
    assert resolution.modes == (Provider.CLAUDE_CODE,)
    conn.close()


def test_capture_mode_resolution_unknown_when_never_observed(tmp_path: Path) -> None:
    """A raw acquired without a capture_mode reads as unknown, not unambiguous."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"kind":"session"}'
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        source_path="/tmp/no-capture-mode.jsonl",
        source_index=0,
        payload=payload,
        acquired_at_ms=1_000,
    )

    resolution = read_capture_mode_resolution(conn, raw_id)
    assert resolution.status == "unknown"
    assert resolution.modes == ()
    conn.close()


def test_capture_mode_resolution_ambiguous_via_blob_ref_writer(tmp_path: Path) -> None:
    """The memory-bounded streaming writer (write_source_raw_session_blob_ref)
    must retain both acquisition modes exactly like the in-memory writer."""
    conn = _connect(tmp_path / "source.db")
    payload = b'{"chunkedPrompt": {"chunks": []}}'
    blob_hash = deterministic_blob_hash(payload)

    raw_id = write_source_raw_session_blob_ref(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.GEMINI,
        source_path="/tmp/blobref-export.json",
        source_index=0,
        blob_hash=blob_hash,
        blob_size=len(payload),
        acquired_at_ms=1_000,
    )
    write_source_raw_session_blob_ref(
        conn,
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode=Provider.DRIVE,
        source_path="/tmp/blobref-export.json",
        source_index=0,
        blob_hash=blob_hash,
        blob_size=len(payload),
        acquired_at_ms=2_000,
        raw_id=raw_id,
    )

    resolution = read_capture_mode_resolution(conn, raw_id)
    assert resolution.status == "ambiguous"
    assert resolution.modes == (Provider.GEMINI, Provider.DRIVE)
    conn.close()


def test_fresh_source_ddl_declares_no_parser_census_timestamp(tmp_path: Path) -> None:
    """polylogue-48bos: ``raw_authority_parser_census`` carries no census clock.

    The retired ``censused_at_ms`` column was declared
    ``INTEGER NOT NULL CHECK(censused_at_ms >= 0)`` as if it held a wall
    clock, but the complete production writer set --
    ``record_current_parser_source_census``
    (``storage/sqlite/archive_tiers/revision_governance.py``) and
    ``record_resource_blocked_revision_census``
    (``sources/revision_backfill.py``) -- bound the bare SQL literal ``0``
    on both the insert and the ``ON CONFLICT`` update path, and no reader
    anywhere selected it. Every row it ever durably held carried the same
    non-fact.

    Concrete input: a fresh source tier. Wrong observable outcome
    prevented: a fresh archive re-declaring a NOT NULL column that promises
    "when was this raw's authority census taken" and can never answer it.

    Anti-vacuity: restoring the column to the DDL makes the column-set
    assertion red; the retained-column assertion is its partner, so a change
    that dropped the whole table rather than the one dead column also fails.
    """
    conn = _connect(tmp_path / "source.db")
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(raw_authority_parser_census)")}
    assert "censused_at_ms" not in columns
    assert columns == {"raw_id", "parser_fingerprint", "status", "logical_keys_json", "detail"}
    # The sibling census table keeps its timestamp: it has real writers that
    # bind acquired_at_ms/observed_at_ms, so the retirement is targeted.
    membership_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(raw_membership_census)")}
    assert "censused_at_ms" in membership_columns
    conn.close()


def test_parser_census_writers_persist_a_row_without_a_timestamp(tmp_path: Path) -> None:
    """polylogue-48bos: both production writers still record their receipt.

    Concrete input: a fresh source tier holding one raw session, driven
    through the complete production writer set for
    ``raw_authority_parser_census`` --
    ``record_current_parser_source_census`` and
    ``record_resource_blocked_revision_census``.

    Wrong observable outcome prevented: a retirement that removed the column
    from the DDL and left a writer naming it, which turns every authority
    census into an ``OperationalError`` at the write boundary rather than a
    recorded receipt.

    Anti-vacuity: reinstating ``censused_at_ms`` in either INSERT makes
    exactly that writer's arm red with "table raw_authority_parser_census
    has no column named censused_at_ms"; asserting the receipt row's
    contents keeps a writer that silently wrote nothing from passing.
    """
    from polylogue.sources.revision_backfill import record_resource_blocked_revision_census
    from polylogue.storage.sqlite.archive_tiers.revision_governance import record_current_parser_source_census

    conn = _connect(tmp_path / "source.db")
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        capture_mode=Provider.CODEX,
        source_path="/tmp/session.jsonl",
        source_index=0,
        native_id="session-1",
        payload=b'{"kind":"session"}',
        acquired_at_ms=1_767_000_000_000,
        parsed_at_ms=1_767_000_000_050,
        validation_status=ValidationStatus.PASSED,
        validation_drift_count=0,
    )
    conn.commit()

    record_current_parser_source_census(conn, raw_id, parser_sessions=[])
    conn.commit()
    row = conn.execute(
        "SELECT parser_fingerprint, status, logical_keys_json, detail FROM raw_authority_parser_census WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    assert row is not None
    assert str(row["status"]) in {"complete", "failed"}
    assert str(row["detail"])

    conn.close()
    record_resource_blocked_revision_census(
        tmp_path,
        (raw_id,),
        max_payload_bytes=1,
        total_payload_bytes=2,
        stream_safe=True,
    )
    verify = sqlite3.connect(tmp_path / "source.db")
    verify.row_factory = sqlite3.Row
    blocked = verify.execute(
        "SELECT status, detail FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
    ).fetchone()
    assert str(blocked["status"]) == "failed"
    assert "exceeds envelope" in str(blocked["detail"])
    verify.close()


def test_newer_raw_replaces_a_terminal_carrier(tmp_path: Path) -> None:
    """A replaced source file gets its own artifact receipt, not the old refusal.

    ``artifact_observation_id`` is stable by (source, path, index) and carries
    no raw identity, so when a path holding a terminal carrier is replaced by
    different bytes, the new raw's ordinary classification lands on the SAME
    ``artifact_id``. The terminal guard compared kinds only, so it suppressed
    that update even though the newer raw had already won the timestamp/rowid
    ordering -- leaving the newest coordinate with no receipt of its own, which
    the raw-frontier check reports as ``source_raws_without_accepted_head``.

    Anti-vacuity: the second half re-observes the SAME raw with an ordinary
    classification and the terminal carrier must survive, so the guard cannot
    become "any ordinary observation wins".
    """
    conn = _connect(tmp_path / "source.db")
    observation_id = artifact_observation_id(
        source_name=Origin.CODEX_SESSION.value, source_path="/tmp/replaced.jsonl", source_index=0
    )
    old_raw = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/replaced.jsonl",
        source_index=0,
        payload=b"corrupt bytes",
        acquired_at_ms=1,
    )
    upsert_raw_artifact(
        conn,
        old_raw,
        ArchiveSourceArtifact(
            artifact_id=observation_id,
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/replaced.jsonl",
            source_index=0,
            artifact_kind="terminal_corrupt_input",
            classification_reason="corrupt",
            support_status=ArtifactSupportStatus.DECODE_FAILED,
        ),
    )

    new_raw = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/replaced.jsonl",
        source_index=0,
        payload=b"a valid replacement export",
        acquired_at_ms=2,
    )
    assert new_raw != old_raw
    upsert_raw_artifact(
        conn,
        new_raw,
        ArchiveSourceArtifact(
            artifact_id=observation_id,
            origin=Origin.CODEX_SESSION,
            source_path="/tmp/replaced.jsonl",
            source_index=0,
            artifact_kind="session_export",
            classification_reason="session_export",
            support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
        ),
    )
    row = conn.execute(
        "SELECT raw_id, artifact_kind FROM raw_artifacts WHERE artifact_id = ?", (observation_id,)
    ).fetchone()
    assert row is not None
    assert (row["raw_id"], row["artifact_kind"]) == (new_raw, "session_export")

    # The same raw re-observing its own coordinate must NOT take back its own
    # terminal refusal.
    same_raw_id = artifact_observation_id(
        source_name=Origin.CODEX_SESSION.value, source_path="/tmp/kept.jsonl", source_index=0
    )
    kept_raw = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/tmp/kept.jsonl",
        source_index=0,
        payload=b"kept corrupt bytes",
        acquired_at_ms=3,
    )
    for kind, status, reason in (
        ("terminal_corrupt_input", ArtifactSupportStatus.DECODE_FAILED, "corrupt"),
        ("session_export", ArtifactSupportStatus.SUPPORTED_PARSEABLE, "session_export"),
    ):
        upsert_raw_artifact(
            conn,
            kept_raw,
            ArchiveSourceArtifact(
                artifact_id=same_raw_id,
                origin=Origin.CODEX_SESSION,
                source_path="/tmp/kept.jsonl",
                source_index=0,
                artifact_kind=kind,
                classification_reason=reason,
                support_status=status,
            ),
        )
    kept = conn.execute(
        "SELECT raw_id, artifact_kind FROM raw_artifacts WHERE artifact_id = ?", (same_raw_id,)
    ).fetchone()
    assert kept is not None
    assert (kept["raw_id"], kept["artifact_kind"]) == (kept_raw, "terminal_corrupt_input")
