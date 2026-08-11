"""Crash-window and rollback proofs for the source-backed audit head."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.audit_continuity import (
    AuditContinuityCoordinator,
    AuditContinuityError,
    AuditMutation,
)


def _mutation(number: int) -> AuditMutation:
    return AuditMutation(
        kind="test-audit-write",
        mutation_id=f"mutation:{number}",
        created_at_ms=number,
        payload={"number": number},
    )


def _apply(conn: sqlite3.Connection, mutation: AuditMutation) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, 1)",
        (f"archive:{mutation.mutation_id}", mutation.created_at_ms),
    )
    return mutation.mutation_id


def test_same_inode_stale_audit_copy_is_rejected(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    coordinator = AuditContinuityCoordinator(tmp_path)
    coordinator.execute(_mutation(1), _apply)
    stale_bytes = (tmp_path / "audit.db").read_bytes()
    coordinator.execute(_mutation(2), _apply)

    # Deliberately overwrite rather than replace the path. The inode remains
    # stable, so this proves the source/audit head catches the rollback that
    # the former st_dev/st_ino receipt accepted.
    audit_path = tmp_path / "audit.db"
    inode = audit_path.stat().st_ino
    audit_path.write_bytes(stale_bytes)
    assert audit_path.stat().st_ino == inode

    with pytest.raises(AuditContinuityError, match="regressed|replaced"):
        coordinator.reconcile(_apply)


def test_crash_before_source_prepare_leaves_no_command(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "before_source_prepare":
            raise RuntimeError("crash before source prepare")

    with pytest.raises(RuntimeError, match="crash before source"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)


def test_pending_command_replays_after_audit_rollback(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("crash before audit commit")

    with pytest.raises(RuntimeError, match="crash before"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM archive_authority").fetchone()[0] == 1


def test_pending_command_promotes_after_audit_commit(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_audit_commit":
            raise RuntimeError("crash before source promotion")

    with pytest.raises(RuntimeError, match="crash before"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None


def test_second_mutation_refuses_while_first_command_is_pending(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("leave pending")

    with pytest.raises(RuntimeError, match="leave pending"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)
    with pytest.raises(AuditContinuityError, match="already pending"):
        AuditContinuityCoordinator(tmp_path).execute(_mutation(2), _apply)


@pytest.mark.parametrize(
    ("crash_phase", "error"),
    [
        ("after_source_prepare", "crash after rebind prepare"),
        ("after_audit_commit", "crash after rebind audit commit"),
    ],
)
def test_rebind_replays_from_its_bound_image_after_each_wal_crash_window(
    tmp_path: Path, crash_phase: str, error: str
) -> None:
    """A rebind WAL command can complete after either replayable crash window."""

    initialize_active_archive_root(tmp_path)
    audit_sha256 = hashlib.sha256((tmp_path / "audit.db").read_bytes()).hexdigest()
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "rebind" and phase == crash_phase:
            raise RuntimeError(error)
        original_phase(self, phase, mutation)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
        with pytest.raises(RuntimeError, match=error):
            AuditContinuityCoordinator(tmp_path).seed_or_rebind(
                mutation_id="rebind:crash",
                now_ms=1,
                evidence={"audit_image_sha256": audit_sha256},
            )

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit:
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        )


def test_rebind_rejects_a_stale_in_place_image_before_blessing_it(tmp_path: Path) -> None:
    """Rebinding checks image bytes, not only a stable path or inode."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    stale_bytes = audit_path.read_bytes()
    inode = audit_path.stat().st_ino
    with sqlite3.connect(audit_path) as audit:
        audit.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, 1)",
            ("newer-audit-image", 1),
        )
        audit.commit()
    expected_image_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    audit_path.write_bytes(stale_bytes)
    assert audit_path.stat().st_ino == inode

    with pytest.raises(AuditContinuityError, match="image changed before continuity rebind"):
        AuditContinuityCoordinator(tmp_path).seed_or_rebind(
            mutation_id="rebind:stale-image",
            now_ms=1,
            evidence={"audit_image_sha256": expected_image_sha256},
        )
