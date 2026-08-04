"""Adversarial corpus acceptance checks over the real seeded archive route."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ArchiveVerificationCheck,
    ArchiveVerificationReport,
    verify_archive,
)
from tests.infra.workload_artifacts import SeededArchiveArtifact, clone_seeded_archive


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _clone(artifact: SeededArchiveArtifact, root: Path) -> Path:
    return clone_seeded_archive(artifact, root).root


def _check(root: Path, name: str) -> tuple[ArchiveVerificationReport, ArchiveVerificationCheck]:
    report = verify_archive(root, checks=(name,))
    assert len(report.checks) == 1
    check = report.checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    return report, check


def _first_session_id(root: Path) -> str:
    with _connect(root / "index.db") as conn:
        row = conn.execute("SELECT session_id FROM sessions ORDER BY session_id LIMIT 1").fetchone()
    assert row is not None
    return str(row[0])


def _insert_raw(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    origin: str,
    native_id: str | None,
    logical_source_key: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions(
            raw_id, origin, native_id, source_path, blob_hash, blob_size,
            acquired_at_ms, logical_source_key
        ) VALUES (?, ?, ?, ?, ?, 10, 100, ?)
        """,
        (
            raw_id,
            origin,
            native_id,
            f"/fixture/{raw_id}",
            hashlib.sha256(raw_id.encode()).digest(),
            logical_source_key,
        ),
    )


def test_real_production_route_is_green_and_read_only(
    corpus_fidelity_archive: SeededArchiveArtifact,
) -> None:
    source_before = hashlib.sha256((corpus_fidelity_archive.root / "source.db").read_bytes()).digest()
    index_before = hashlib.sha256((corpus_fidelity_archive.root / "index.db").read_bytes()).digest()

    report = verify_archive(
        corpus_fidelity_archive.root,
        checks=("corpus-absences", "corpus-attachment-fidelity", "corpus-revision-fidelity"),
    )

    assert not report.blocking
    assert all(check.status is OutcomeStatus.OK for check in report.checks)
    assert hashlib.sha256((corpus_fidelity_archive.root / "source.db").read_bytes()).digest() == source_before
    assert hashlib.sha256((corpus_fidelity_archive.root / "index.db").read_bytes()).digest() == index_before


def test_absence_gate_catches_unindexed_membership_document(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "absence")
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-absent-document",
            origin="codex-session",
            native_id="absent-document",
            logical_source_key="codex:absent-document",
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES (?, ?, ?, 'rev-1', ?, 4)
            """,
            ("raw-absent-document", "codex:absent-document", "absent-document", b"a" * 32),
        )

    report, check = _check(root, "corpus-absences")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["absent_total"] == 1
    assert check.evidence["absent_by_origin_cause"] == {"codex-session/settled-yet-absent": 1}


def test_absence_gate_does_not_hide_raw_without_identity(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "unattributable")
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-unattributable",
            origin="aistudio-drive",
            native_id=None,
            logical_source_key=None,
        )

    report, check = _check(root, "corpus-absences")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["absent_total"] == 0
    assert check.evidence["raws_without_attributable_identity"] == 1
    assert check.evidence["unattributable_sample"] == ["raw-unattributable"]


def test_attachment_gate_only_blocks_unfetched_and_reports_typed_unavailable(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "attachments")
    session_id = _first_session_id(root)
    with _connect(root / "index.db") as conn:
        message_row = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
        ).fetchone()
        assert message_row is not None
        message_id = str(message_row[0])
        conn.execute(
            "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('attachment-unfetched', 'unfetched')"
        )
        conn.execute(
            "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('attachment-unavailable', 'unavailable')"
        )
        conn.execute(
            """
            INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin)
            VALUES ('attachment-unfetched', ?, ?, 0, 'drive')
            """,
            (session_id, message_id),
        )
        conn.execute(
            """
            INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin)
            VALUES ('attachment-unavailable', ?, ?, 1, 'oauth')
            """,
            (session_id, message_id),
        )

    report, check = _check(root, "corpus-attachment-fidelity")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["refs_unfetched"] == 1
    assert check.evidence["refs_unavailable"] == 1
    assert check.evidence["breakdown"]["aistudio-drive/drive/unfetched"] == 1

    with _connect(root / "index.db") as conn:
        conn.execute(
            "UPDATE attachments SET acquisition_status = 'unavailable' WHERE attachment_id = 'attachment-unfetched'"
        )
    report, check = _check(root, "corpus-attachment-fidelity")
    assert not report.blocking
    assert check.status is OutcomeStatus.OK
    assert check.evidence["refs_unfetched"] == 0
    assert check.evidence["refs_unavailable"] == 2


def test_revision_gate_catches_smaller_index_than_best_recorded_revision(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "revision")
    session_id = _first_session_id(root)
    origin, native_id = session_id.split(":", 1)
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-best-revision",
            origin=origin,
            native_id=native_id,
            logical_source_key=f"codex:{native_id}",
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES (?, ?, ?, 'rev-best', ?, 100000)
            """,
            ("raw-best-revision", f"codex:{native_id}", native_id, b"b" * 32),
        )

    report, check = _check(root, "corpus-revision-fidelity")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unexplained_shortfall"] == 1
    assert check.evidence["worst"][0]["session_id"] == session_id
    assert check.evidence["worst"][0]["best_recorded_messages"] == 100000


def test_revision_gate_explains_event_reclassification_without_hiding_shortfall(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "event-reclassification")
    session_id = _first_session_id(root)
    origin, native_id = session_id.split(":", 1)
    with _connect(root / "index.db") as conn:
        message_count = int(
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        )
        event_count = int(
            conn.execute("SELECT COUNT(*) FROM session_events WHERE session_id = ?", (session_id,)).fetchone()[0]
        )
        next_position = int(
            conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM session_events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO session_events(session_id, position, event_type, summary)
            VALUES (?, ?, 'fixture-reclassified', 'event represented a historical message')
            """,
            (session_id, next_position),
        )
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-event-reclassification",
            origin=origin,
            native_id=native_id,
            logical_source_key=f"fixture:{native_id}",
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES (?, ?, ?, 'rev-reclassified', ?, ?)
            """,
            (
                "raw-event-reclassification",
                f"fixture:{native_id}",
                native_id,
                b"g" * 32,
                message_count + event_count + 1,
            ),
        )

    report, check = _check(root, "corpus-revision-fidelity")

    assert not report.blocking
    assert check.status is OutcomeStatus.OK
    assert check.evidence["unexplained_shortfall"] == 0
    assert check.evidence["explained_by_event_reclassification"] == 1
