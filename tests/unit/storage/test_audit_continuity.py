"""Crash-window and rollback proofs for the source-backed audit head."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.audit_continuity import (
    AUDIT_CONTINUITY_GENESIS_HEAD_SHA256,
    AuditContinuityCoordinator,
    AuditContinuityError,
    AuditMutation,
    audit_semantic_sha256,
)


def test_genesis_head_is_shared_by_fresh_ddl_and_additive_migrations() -> None:
    migration_paths = (
        Path("polylogue/storage/sqlite/migrations/audit/002_audit_continuity_head.sql"),
        Path("polylogue/storage/sqlite/migrations/source/032_audit_continuity_control.sql"),
    )

    assert AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 in AUDIT_DDL
    assert AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 in SOURCE_DDL
    assert all(AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 in path.read_text(encoding="utf-8") for path in migration_paths)


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
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)


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


@pytest.mark.parametrize(
    ("dropped_table", "error"),
    [
        ("audit_continuity_control", "current source schema"),
        ("audit_continuity_head", "current audit schema"),
    ],
)
def test_current_schema_missing_a_continuity_table_is_damage(tmp_path: Path, dropped_table: str, error: str) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / ("source.db" if dropped_table.endswith("control") else "audit.db")
    with sqlite3.connect(path) as connection:
        connection.execute(f"DROP TABLE {dropped_table}")
        connection.commit()

    with pytest.raises(AuditContinuityError, match=error):
        AuditContinuityCoordinator(tmp_path).is_available()


@pytest.mark.parametrize(
    ("path_name", "table", "legacy_version"),
    [("source.db", "audit_continuity_control", 31), ("audit.db", "audit_continuity_head", 1)],
)
def test_legitimate_one_sided_precontinuity_schema_window_stays_in_standby(
    tmp_path: Path, path_name: str, table: str, legacy_version: int
) -> None:
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / path_name) as connection:
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"PRAGMA user_version = {legacy_version}")
        connection.commit()

    assert not AuditContinuityCoordinator(tmp_path).is_available()


def test_empty_fresh_archive_can_use_the_genesis_continuity_head(tmp_path: Path) -> None:
    """Genesis is valid only when it describes an empty freshly-created audit journal."""

    initialize_active_archive_root(tmp_path)

    assert AuditContinuityCoordinator(tmp_path).is_available()


def test_semantic_hash_reads_audit_without_creating_backup_sidecars(tmp_path: Path) -> None:
    """Read-only continuity validation never changes an immutable backup image."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    for suffix in ("-wal", "-shm", "-journal"):
        audit_path.with_name(f"audit.db{suffix}").unlink(missing_ok=True)

    assert len(audit_semantic_sha256(audit_path)) == 64
    assert not any(audit_path.with_name(f"audit.db{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_second_mutation_refuses_while_first_command_is_pending(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("leave pending")

    with pytest.raises(RuntimeError, match="leave pending"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)
    with pytest.raises(AuditContinuityError, match="already pending"):
        AuditContinuityCoordinator(tmp_path).execute(_mutation(2), _apply)


def test_rejected_audit_transaction_aborts_its_prepared_command(tmp_path: Path) -> None:
    """A deterministic reject cannot leave the source WAL blocking later work."""

    initialize_active_archive_root(tmp_path)

    def reject(_conn: sqlite3.Connection, _mutation: AuditMutation) -> object:
        raise ValueError("already consumed")

    coordinator = AuditContinuityCoordinator(tmp_path)
    with pytest.raises(ValueError, match="already consumed"):
        coordinator.execute(_mutation(1), reject)

    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)
    assert coordinator.execute(_mutation(2), _apply) == "mutation:2"
    with closing(sqlite3.connect(tmp_path / "audit.db")) as audit:
        assert audit.execute("SELECT generation, mutation_id FROM audit_continuity_head").fetchone() == (
            1,
            "mutation:2",
        )


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
    with closing(sqlite3.connect(audit_path)) as audit:
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


def test_populated_precontinuity_audit_is_bound_before_normal_coordination(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "audit.db") as audit:
        audit.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES ('legacy:archive', 1, 1)"
        )
        audit.execute("DROP TABLE audit_continuity_head")
        audit.execute("PRAGMA user_version = 1")
        audit.executescript(Path("polylogue/storage/sqlite/migrations/audit/002_audit_continuity_head.sql").read_text())
        audit.execute("PRAGMA user_version = 2")
        audit.commit()
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("DROP TABLE audit_continuity_control")
        source.execute("PRAGMA user_version = 31")
        source.executescript(
            Path("polylogue/storage/sqlite/migrations/source/032_audit_continuity_control.sql").read_text()
        )
        source.execute("PRAGMA user_version = 32")
        source.commit()
    expected = audit_semantic_sha256(tmp_path / "audit.db")
    coordinator = AuditContinuityCoordinator(tmp_path)

    with pytest.raises(AuditContinuityError, match="post-migration binding"):
        coordinator.is_available()
    coordinator.bind_precontinuity_audit(
        mutation_id=f"precontinuity-audit:{expected}", now_ms=1, audit_semantic_sha256=expected
    )

    assert coordinator.is_available()
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit:
        assert (
            source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control"
            ).fetchone()
            == audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        )


def test_precontinuity_binding_rejects_a_substituted_genesis_audit_image(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "audit.db") as audit:
        audit.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES ('legacy:archive', 1, 1)"
        )
        audit.execute("DROP TABLE audit_continuity_head")
        audit.execute("PRAGMA user_version = 1")
        audit.executescript(Path("polylogue/storage/sqlite/migrations/audit/002_audit_continuity_head.sql").read_text())
        audit.execute("PRAGMA user_version = 2")
        audit.commit()
    expected = audit_semantic_sha256(tmp_path / "audit.db")
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("DROP TABLE audit_continuity_control")
        source.execute("PRAGMA user_version = 31")
        source.executescript(
            Path("polylogue/storage/sqlite/migrations/source/032_audit_continuity_control.sql").read_text()
        )
        source.execute("PRAGMA user_version = 32")
        source.commit()
    replacement_root = tmp_path / "replacement"
    initialize_active_archive_root(replacement_root)
    with sqlite3.connect(replacement_root / "audit.db") as audit:
        audit.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES ('substituted:archive', 1, 1)"
        )
        audit.commit()
    (replacement_root / "audit.db").replace(tmp_path / "audit.db")

    with pytest.raises(AuditContinuityError, match="differs from its authenticated migration evidence"):
        AuditContinuityCoordinator(tmp_path).bind_precontinuity_audit(
            mutation_id=f"precontinuity-audit:{expected}", now_ms=1, audit_semantic_sha256=expected
        )
