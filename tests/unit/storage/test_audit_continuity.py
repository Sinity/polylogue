"""Crash-window and rollback proofs for the source-backed audit head."""

from __future__ import annotations

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
