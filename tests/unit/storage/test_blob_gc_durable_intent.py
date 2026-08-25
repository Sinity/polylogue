"""Crash-safe durable intent for physical blob deletion.

These tests drive the production GC entry point against fresh temporary
archives.  They deliberately observe durable rows at the filesystem mutation
boundary: an unlink without an already-committed exact member intent is a
test failure, even if a later summary row would look successful.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

import polylogue.storage.blob_gc as blob_gc
from polylogue.storage.blob_liveness import BlobLiveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.hook_payload_ref_reconciliation import HookPayloadRefMatchStage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import migrate_archive_tier


def _backdate(store: BlobStore, blob_hash: str) -> None:
    path = store.blob_path(blob_hash)
    old = time.time() - 3600
    os.utime(path, (old, old))


def _member_rows(source_db: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(source_db) as conn:
        return conn.execute(
            "SELECT generation_id, hex(blob_hash), candidate_liveness, outcome "
            "FROM gc_generation_members ORDER BY generation_id, blob_hash"
        ).fetchall()


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_gc_commits_exact_member_intent_before_any_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The durable member row exists before the production unlink seam runs.

    Anti-vacuity twin: moving unlink ahead of the intent commit makes the
    patched unlink observe no matching durable member and fails this test.
    """
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"intent before unlink")
    _backdate(store, blob_hash)

    original_unlink = Path.unlink

    def assert_intent_then_unlink(path: Path, missing_ok: bool = False) -> None:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            row = conn.execute(
                "SELECT outcome FROM gc_generation_members WHERE blob_hash = ?",
                (bytes.fromhex(blob_hash),),
            ).fetchone()
        assert row == ("pending",), "unlink ran before durable exact member intent"
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", assert_intent_then_unlink)

    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 1
    assert _member_rows(tmp_path / "source.db") == [
        (report.generation_id, blob_hash.upper(), "unreferenced", "removed")
    ]


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_pending_member_retries_after_fresh_liveness_and_absence_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after intent has exact retry and post-unlink restart semantics."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"restartable exact intent")
    _backdate(store, blob_hash)

    original_final = blob_gc._final_gc_member_liveness
    calls = 0

    def crash_before_final(
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection | None,
        candidate_hash: str,
        *,
        legacy_hook_stage: HookPayloadRefMatchStage,
    ) -> tuple[BlobLiveness, BlobLiveness]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fault after intent commit")
        return original_final(source_conn, index_conn, candidate_hash, legacy_hook_stage=legacy_hook_stage)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_before_final)
    with pytest.raises(RuntimeError, match="after intent commit"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    pending = _member_rows(tmp_path / "source.db")
    assert len(pending) == 1
    assert pending[0][1:] == (blob_hash.upper(), "unreferenced", "pending")
    assert store.exists(blob_hash)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)
    retry = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert retry.generation_id == pending[0][0]
    assert retry.deleted_count == 1
    assert not store.exists(blob_hash)

    second_hash, _ = store.write_from_bytes(b"absence after unlink")
    _backdate(store, second_hash)
    original_outcome = blob_gc._commit_gc_member_outcome
    raised = False

    def crash_before_outcome(
        source_conn: sqlite3.Connection,
        *,
        generation_id: str,
        blob_hash: str,
        outcome: str,
        detail: str | None = None,
    ) -> None:
        nonlocal raised
        if not raised:
            raised = True
            raise RuntimeError("fault after unlink")
        original_outcome(
            source_conn,
            generation_id=generation_id,
            blob_hash=blob_hash,
            outcome=outcome,
            detail=detail,
        )

    monkeypatch.setattr(blob_gc, "_commit_gc_member_outcome", crash_before_outcome)
    with pytest.raises(RuntimeError, match="after unlink"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert not store.exists(second_hash)

    monkeypatch.setattr(blob_gc, "_commit_gc_member_outcome", original_outcome)
    reconciled = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert reconciled.deleted_count == 0
    assert any(
        row[1:] == (second_hash.upper(), "unreferenced", "reconciled_removed")
        for row in _member_rows(tmp_path / "source.db")
    )


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_gc_does_not_terminalize_a_generation_with_an_unknown_member(tmp_path: Path) -> None:
    """A completed generation is impossible while one member remains pending.

    Anti-vacuity twin: a finalizer that writes terminal counters without
    requiring every member outcome would mark this manually seeded generation
    complete.
    """
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO gc_generations (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES ('pending-generation', 1, NULL, 0, 0)"
        )
        conn.execute(
            "INSERT INTO gc_generation_members "
            "(generation_id, blob_hash, candidate_liveness, candidate_mtime_ns, candidate_size_bytes, source_schema_version, "
            "index_schema_version, index_generation, archive_identity_digest, code_identity, intent_committed_at_ms, outcome) "
            "VALUES ('pending-generation', ?, 'unreferenced', 1, 1, 1, 1, 'index', ?, 'test', 1, 'pending')",
            (b"p" * 32, "0" * 64),
        )
        conn.commit()

    (tmp_path / "index.db").unlink()
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", tmp_path / "blob")

    assert report.blocked_reason is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT completed_at_ms FROM gc_generations WHERE generation_id = 'pending-generation'"
        ).fetchone() == (None,)


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_intent_commit_failure_and_absent_without_intent_never_become_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No intent means no unlink, and an absent non-member has no GC success."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"intent commit failure")
    _backdate(store, blob_hash)

    def reject_intent(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("simulated durable intent failure")

    monkeypatch.setattr(blob_gc, "_commit_gc_generation_intent", reject_intent)
    with pytest.raises(sqlite3.OperationalError, match="intent failure"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert store.exists(blob_hash)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM gc_generations").fetchone() == (0,)

    store.blob_path(blob_hash).unlink()
    monkeypatch.undo()
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert report.deleted_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM gc_generation_members").fetchone() == (0,)


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_partial_generation_restarts_only_its_exact_pending_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial batch keeps its denominator and does not allocate a new plan."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    first_hash, _ = store.write_from_bytes(b"partial first")
    second_hash, _ = store.write_from_bytes(b"partial second")
    _backdate(store, first_hash)
    _backdate(store, second_hash)

    original_final = blob_gc._final_gc_member_liveness
    calls = 0

    def crash_second(
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection | None,
        candidate_hash: str,
        *,
        legacy_hook_stage: HookPayloadRefMatchStage,
    ) -> tuple[BlobLiveness, BlobLiveness]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fault during partial batch")
        return original_final(source_conn, index_conn, candidate_hash, legacy_hook_stage=legacy_hook_stage)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_second)
    with pytest.raises(RuntimeError, match="partial batch"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)

    rows = _member_rows(tmp_path / "source.db")
    assert {row[1:] for row in rows} == {
        (first_hash.upper(), "unreferenced", "removed"),
        (second_hash.upper(), "unreferenced", "pending"),
    }
    generation_id = rows[0][0]
    assert not store.exists(first_hash)
    assert store.exists(second_hash)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)
    retry = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)
    assert retry.generation_id == generation_id
    assert retry.deleted_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT completed_at_ms, reclaimed_count FROM gc_generations WHERE generation_id = ?", (generation_id,)
            ).fetchone()[1]
            == 2
        )


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
@pytest.mark.parametrize("kind", ["referent", "reservation"])
def test_final_recheck_closes_pending_member_as_still_live_when_newly_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Fresh canonical liveness, not the accepted plan, decides a retry."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_bytes(f"new {kind}".encode())
    _backdate(store, blob_hash)
    original_final = blob_gc._final_gc_member_liveness

    def introduce_protection(
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection | None,
        candidate_hash: str,
        *,
        legacy_hook_stage: HookPayloadRefMatchStage,
    ) -> tuple[BlobLiveness, BlobLiveness]:
        if kind == "referent":
            source_conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
                "VALUES ('introduced-live', 'codex-session', '/live', ?, ?, 1)",
                (bytes.fromhex(blob_hash), size),
            )
        else:
            source_conn.execute(
                "INSERT INTO blob_publication_reservations "
                "(publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms) "
                "VALUES ('introduced-reservation', ?, ?, 'test', 1)",
                (bytes.fromhex(blob_hash), size),
            )
        return original_final(source_conn, index_conn, candidate_hash, legacy_hook_stage=legacy_hook_stage)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", introduce_protection)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 0
    assert store.exists(blob_hash)
    assert _member_rows(tmp_path / "source.db")[0][3] == "skipped_still_live"


def test_v33_source_migrates_additively_to_exact_gc_member_intent(tmp_path: Path) -> None:
    """The v34 durable addition needs no backup and preserves the v33 table."""
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute("DROP INDEX idx_gc_generation_members_pending")
        conn.execute("DROP TABLE gc_generation_members")
        conn.execute("PRAGMA user_version = 33")
        conn.execute(
            "INSERT INTO gc_generations (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES ('before-v34', 1, 1, 0, 0)"
        )
        conn.commit()
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)
        assert result.applied_versions == (34,)
        assert conn.execute("PRAGMA user_version").fetchone() == (34,)
        assert conn.execute("SELECT generation_id FROM gc_generations").fetchone() == ("before-v34",)
        assert conn.execute("SELECT COUNT(*) FROM gc_generation_members").fetchone() == (0,)


def test_gc_refuses_a_missing_source_tier_before_planning(tmp_path: Path) -> None:
    """An index entry point never treats a missing durable source tier as empty."""
    initialize_active_archive_root(tmp_path)
    (tmp_path / "source.db").rename(tmp_path / "source.db.unavailable")

    report = blob_gc.run_blob_gc_report(tmp_path / "index.db", tmp_path / "blob")

    assert report.blocked_reason is not None
    assert "source tier is unavailable" in report.blocked_reason
