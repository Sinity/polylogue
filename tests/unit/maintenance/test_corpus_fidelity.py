"""Adversarial corpus acceptance checks over the real seeded archive route."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

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


def test_absence_gate_excludes_membershipless_non_session_artifact(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "non-session-artifact")
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-membershipless-non-session",
            origin="codex-session",
            native_id="settings-artifact",
            logical_source_key="codex:settings-artifact",
        )
        conn.execute(
            """
            INSERT INTO raw_membership_census(
                raw_id, parser_fingerprint, status, member_count, censused_at_ms
            ) VALUES (?, 'fixture', 'non_session', 0, 100)
            """,
            ("raw-membershipless-non-session",),
        )

    report, check = _check(root, "corpus-absences")

    assert not report.blocking
    assert check.status is OutcomeStatus.OK
    assert check.evidence["absent_total"] == 0
    assert check.evidence["membershipless_non_session_artifacts_excluded"] == 1


def test_attachment_gate_requires_provenance_for_unavailable_refs(
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
            "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('attachment-unavailable-untyped', 'unavailable')"
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
        conn.execute(
            """
            INSERT INTO attachment_refs(attachment_id, session_id, message_id, position)
            VALUES ('attachment-unavailable-untyped', ?, ?, 2)
            """,
            (session_id, message_id),
        )

    report, check = _check(root, "corpus-attachment-fidelity")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["refs_unfetched"] == 1
    assert check.evidence["refs_unavailable"] == 2
    assert check.evidence["refs_unavailable_without_provenance"] == 1
    assert check.evidence["breakdown"]["aistudio-drive/drive/unfetched"] == 1

    with _connect(root / "index.db") as conn:
        conn.execute(
            "UPDATE attachments SET acquisition_status = 'unavailable' WHERE attachment_id = 'attachment-unfetched'"
        )
    report, check = _check(root, "corpus-attachment-fidelity")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["refs_unfetched"] == 0
    assert check.evidence["refs_unavailable"] == 3
    assert check.evidence["refs_unavailable_without_provenance"] == 1

    with _connect(root / "index.db") as conn:
        conn.execute(
            """
            UPDATE attachment_refs
            SET source_url = 'https://fixture.invalid/unavailable.bin'
            WHERE attachment_id = 'attachment-unavailable-untyped'
            """
        )
    report, check = _check(root, "corpus-attachment-fidelity")
    assert not report.blocking
    assert check.status is OutcomeStatus.OK
    assert check.evidence["refs_unfetched"] == 0
    assert check.evidence["refs_unavailable"] == 3
    assert check.evidence["refs_unavailable_without_provenance"] == 0


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


@pytest.mark.parametrize(
    ("violation", "expected_check"),
    (
        ("absent", "corpus-absences"),
        ("attachment", "corpus-attachment-fidelity"),
        ("revision", "corpus-revision-fidelity"),
    ),
)
def test_candidate_index_corpus_gate_reads_durable_source_and_inactive_index(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
    violation: str,
    expected_check: str,
) -> None:
    """Each candidate runner reads the real inactive index, not active state.

    Each case leaves default verification of the active index clear. The
    revision case adds matching durable source evidence before reducing only
    the candidate, so removing a candidate runner, or accidentally routing
    it through the active index, leaves the selected candidate gate green.
    """
    root = _clone(corpus_fidelity_archive, tmp_path / violation)
    candidate_index = tmp_path / f"{violation}-candidate-index.db"
    shutil.copy2(root / "index.db", candidate_index)
    session_id = _first_session_id(root)

    assert not verify_archive(root, checks=(expected_check,)).blocking
    with _connect(candidate_index) as conn:
        if violation == "attachment":
            message_row = conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
            ).fetchone()
            assert message_row is not None
            conn.execute(
                "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('candidate-unfetched', 'unfetched')"
            )
            conn.execute(
                "INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin) "
                "VALUES ('candidate-unfetched', ?, ?, 99, 'drive')",
                (session_id, message_row[0]),
            )
        elif violation == "absent":
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        else:
            origin, native_id = session_id.split(":", 1)
            message_row = conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1", (session_id,)
            ).fetchone()
            assert message_row is not None
            with _connect(root / "source.db") as source:
                _insert_raw(
                    source,
                    raw_id="candidate-revision-source",
                    origin=origin,
                    native_id=native_id,
                    logical_source_key=f"fixture:{native_id}",
                )
                message_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                source.execute(
                    """
                    INSERT INTO raw_session_memberships(
                        raw_id, logical_source_key, provider_session_id, source_revision,
                        normalized_content_hash, message_count
                    ) VALUES ('candidate-revision-source', ?, ?, 'candidate-revision', ?, ?)
                    """,
                    (f"fixture:{native_id}", native_id, b"c" * 32, message_count),
                )
            conn.execute("DELETE FROM messages WHERE session_id = ? AND message_id != ?", (session_id, message_row[0]))
            conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))

    report = verify_archive(root, checks=(expected_check,), index_path_override=candidate_index)

    assert not verify_archive(root, checks=(expected_check,)).blocking
    assert report.blocking
    assert len(report.checks) == 1
    check = report.checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    assert check.status is OutcomeStatus.ERROR


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
            conn.execute(
                """
                SELECT COUNT(*) FROM session_events
                WHERE session_id = ?
                  AND (source_message_id IS NOT NULL OR source_message_provider_id IS NOT NULL)
                """,
                (session_id,),
            ).fetchone()[0]
        )
        next_position = int(
            conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM session_events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO session_events(
                session_id, source_message_provider_id, position, event_type, summary
            ) VALUES (?, 'fixture-missing-message', ?, 'message_revision', 'event represented a historical message')
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


def test_revision_gate_rejects_unattributed_event_as_message_replacement(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
) -> None:
    root = _clone(corpus_fidelity_archive, tmp_path / "unattributed-event")
    session_id = _first_session_id(root)
    origin, native_id = session_id.split(":", 1)
    with _connect(root / "index.db") as conn:
        message_count = int(
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        )
        attributed_event_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM session_events
                WHERE session_id = ?
                  AND (source_message_id IS NOT NULL OR source_message_provider_id IS NOT NULL)
                """,
                (session_id,),
            ).fetchone()[0]
        )
        next_position = int(
            conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM session_events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO session_events(session_id, position, event_type, summary)
            VALUES (?, ?, 'fixture-arbitrary', 'unrelated timeline event')
            """,
            (session_id, next_position),
        )
    with _connect(root / "source.db") as conn:
        _insert_raw(
            conn,
            raw_id="raw-unattributed-event",
            origin=origin,
            native_id=native_id,
            logical_source_key=f"fixture:{native_id}",
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES (?, ?, ?, 'rev-unattributed-event', ?, ?)
            """,
            (
                "raw-unattributed-event",
                f"fixture:{native_id}",
                native_id,
                b"h" * 32,
                message_count + attributed_event_count + 1,
            ),
        )

    report, check = _check(root, "corpus-revision-fidelity")

    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unexplained_shortfall"] == 1
