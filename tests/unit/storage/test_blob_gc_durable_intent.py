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


def test_marker_creation_fsyncs_file_and_containing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a new namespace marker must fsync both the marker and its directory.

    Regression for a review finding: closing the marker file makes neither
    its contents nor its directory entry durable, so a power loss between
    marker creation and the (already-fsynced) ``gc_generations`` commit
    could leave a durable pending generation whose marker never reached
    disk. Anti-vacuity: if the fsync calls in
    ``blob_gc._blob_namespace_identity``'s marker-creation branch are
    removed, ``os.fsync`` is called zero times here instead of at least
    once for the marker fd and once for the directory fd, and this test
    goes red.
    """
    blob_root = tmp_path / "blob"
    blob_root.mkdir()

    fsynced_fds: list[int] = []
    real_fsync = os.fsync

    def _recording_fsync(fd: int) -> None:
        fsynced_fds.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _recording_fsync)

    marker_path = blob_root / ".polylogue-blob-namespace"
    assert not marker_path.exists()

    identity = blob_gc._blob_namespace_identity(blob_root, create_marker=True)

    assert marker_path.exists()
    assert identity.marker == marker_path.read_text(encoding="ascii")
    # At least two fsync calls: one for the marker file's own fd, one for
    # the containing directory's fd (durability of the new directory entry).
    assert len(fsynced_fds) >= 2


def _member_rows(source_db: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(source_db) as conn:
        return conn.execute(
            "SELECT generation_id, hex(blob_hash), outcome FROM gc_generation_members ORDER BY generation_id, blob_hash"
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

    original_unlink = blob_gc._unlink_observed_gc_member

    def assert_intent_then_unlink(observed: blob_gc._ObservedBlobObject) -> None:
        with sqlite3.connect(tmp_path / "source.db") as conn:
            row = conn.execute(
                "SELECT outcome FROM gc_generation_members WHERE blob_hash = ?",
                (bytes.fromhex(blob_hash),),
            ).fetchone()
        assert row == ("pending",), "unlink ran before durable exact member intent"
        original_unlink(observed)

    monkeypatch.setattr(blob_gc, "_unlink_observed_gc_member", assert_intent_then_unlink)

    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 1
    assert _member_rows(tmp_path / "source.db") == [(report.generation_id, blob_hash.upper(), "removed")]


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_gc_mid_batch_unlink_crash_leaves_durable_intent_for_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash during unlink preserves each completed member outcome durably."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    first_hash, _ = store.write_from_bytes(b"first crash-window member")
    second_hash, _ = store.write_from_bytes(b"second crash-window member")
    _backdate(store, first_hash)
    _backdate(store, second_hash)

    original_unlink = blob_gc._unlink_observed_gc_member
    calls = 0

    def crash_on_second_unlink(observed: blob_gc._ObservedBlobObject) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fault during unlink batch")
        original_unlink(observed)

    monkeypatch.setattr(blob_gc, "_unlink_observed_gc_member", crash_on_second_unlink)
    with pytest.raises(RuntimeError, match="during unlink batch"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        generation = conn.execute(
            "SELECT generation_id, completed_at_ms, reclaimed_count FROM gc_generations"
        ).fetchone()
        members = conn.execute(
            "SELECT hex(blob_hash), outcome FROM gc_generation_members ORDER BY blob_hash"
        ).fetchall()

    assert generation is not None
    assert generation[1] is None
    expected_hashes = sorted((first_hash.upper(), second_hash.upper()))
    assert generation[2] == 0
    assert members == [(expected_hashes[0], "removed"), (expected_hashes[1], "pending")]
    assert not store.exists(expected_hashes[0].lower())
    assert store.exists(expected_hashes[1].lower())
    # Anti-vacuity: deferring outcome commits until the loop ends leaves both
    # members pending after the second unlink raises.


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
    assert pending[0][1:] == (blob_hash.upper(), "pending")
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
    assert any(row[1:] == (second_hash.upper(), "reconciled_removed") for row in _member_rows(tmp_path / "source.db"))


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_pending_member_refuses_a_swapped_blob_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restart must not terminalize intent against a replacement namespace.

    Anti-vacuity: a retry that treats any readable missing path as reconciled
    removal would complete the pending member after the namespace swap.
    """
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"namespace-bound pending intent")
    _backdate(store, blob_hash)
    original_final = blob_gc._final_gc_member_liveness

    def crash_after_intent(*_args: object, **_kwargs: object) -> tuple[BlobLiveness, BlobLiveness]:
        raise RuntimeError("leave namespace-bound intent pending")

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_after_intent)
    with pytest.raises(RuntimeError, match="namespace-bound intent pending"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)

    observed_namespace = tmp_path / "observed-blob-namespace"
    store.root.rename(observed_namespace)
    store.root.mkdir()

    retry = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert retry.blocked_reason == "blob namespace authority changed since GC intent was committed"
    assert (observed_namespace / blob_hash[:2] / blob_hash[2:]).exists()
    assert _member_rows(tmp_path / "source.db")[0][1:] == (blob_hash.upper(), "pending")


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_pending_member_recovers_after_device_number_change_with_stable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remount-style device change cannot wedge a marker-bound intent.

    Anti-vacuity: binding the intent to ``st_dev`` instead of the owned marker
    makes the patched root stat block the pending generation.
    """
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"remount stable marker")
    _backdate(store, blob_hash)
    original_final = blob_gc._final_gc_member_liveness
    monkeypatch.setattr(
        blob_gc,
        "_final_gc_member_liveness",
        lambda _source, _index, _hash, *, legacy_hook_stage: (_ for _ in ()).throw(
            RuntimeError("leave intent pending")
        ),
    )
    with pytest.raises(RuntimeError, match="leave intent pending"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)

    original_stat = Path.stat

    def remounted_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        result = original_stat(path, follow_symlinks=follow_symlinks)
        if path == store.root:
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev + 1, *result[3:]))
        return result

    monkeypatch.setattr(Path, "stat", remounted_stat)
    retry = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert retry.blocked_reason is None
    assert retry.deleted_count == 1
    assert not store.exists(blob_hash)


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_member_reconciliation_keeps_the_observed_namespace_after_batch_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A swap after descriptor admission cannot redirect an old intent's effect."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"swap after batch validation")
    _backdate(store, blob_hash)
    original_final = blob_gc._final_gc_member_liveness
    original_root = tmp_path / "original-namespace"
    swapped = False

    def swap_after_batch(
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection | None,
        candidate_hash: str,
        *,
        legacy_hook_stage: HookPayloadRefMatchStage,
    ) -> tuple[BlobLiveness, BlobLiveness]:
        nonlocal swapped
        result = original_final(source_conn, index_conn, candidate_hash, legacy_hook_stage=legacy_hook_stage)
        if not swapped:
            swapped = True
            store.root.rename(original_root)
            store.root.mkdir()
            (store.root / ".polylogue-blob-namespace").write_text("f" * 32, encoding="ascii")
            replacement = store.blob_path(blob_hash)
            replacement.parent.mkdir()
            replacement.write_bytes(b"replacement namespace bytes")
        return result

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", swap_after_batch)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 1
    assert report.blocked_reason is None
    assert not (original_root / blob_hash[:2] / blob_hash[2:]).exists()
    assert store.exists(blob_hash)
    assert _member_rows(tmp_path / "source.db")[0][2] == "removed"


@pytest.mark.uses_real_clock("backdates a temporary blob to pass production GC's age gate")
def test_member_unlink_stays_in_observed_namespace_after_root_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final effect uses the object handle observed before the swap.

    Anti-vacuity: replacing dirfd-relative unlink with ``Path.unlink`` makes
    the replacement namespace object disappear after this production-seam
    swap between final object observation and effect.
    """
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"observed namespace object")
    _backdate(store, blob_hash)
    original_unlink = blob_gc._unlink_observed_gc_member
    observed_root = tmp_path / "observed-namespace"

    def swap_root_after_observation(observed: blob_gc._ObservedBlobObject) -> None:
        store.root.rename(observed_root)
        store.root.mkdir()
        (store.root / ".polylogue-blob-namespace").write_text("e" * 32, encoding="ascii")
        replacement = store.blob_path(blob_hash)
        replacement.parent.mkdir()
        replacement.write_bytes(b"replacement namespace object")
        original_unlink(observed)

    monkeypatch.setattr(blob_gc, "_unlink_observed_gc_member", swap_root_after_observation)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 1
    assert not (observed_root / blob_hash[:2] / blob_hash[2:]).exists()
    assert store.blob_path(blob_hash).read_bytes() == b"replacement namespace object"
    assert _member_rows(tmp_path / "source.db") == [(report.generation_id, blob_hash.upper(), "removed")]


def test_authorized_abandonment_terminalizes_exact_intent_without_blob_effect(tmp_path: Path) -> None:
    """The mutation authority writes source/audit only, never blob bytes or marker.

    Anti-vacuity: calling the legacy GC execution path from the actuator, or
    omitting the executor receipt, either unlinks the candidate or leaves no
    durable audit operation for this exact generation.
    """
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_actuators import (
        PendingBlobGCGenerationAbandonActuator,
        PendingBlobGCGenerationAbandonArgs,
    )
    from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor

    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"operator adjudication")
    marker = blob_gc._blob_namespace_identity(store.root, create_marker=True).marker
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO gc_generations (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes, blob_namespace_marker) "
            "VALUES ('blocked', 1, NULL, 0, 0, ?)",
            (marker,),
        )
        conn.execute(
            "INSERT INTO gc_generation_members (generation_id, blob_hash, candidate_size_bytes, intent_committed_at_ms, outcome) "
            "VALUES ('blocked', ?, 1, 1, 'pending')",
            (bytes.fromhex(blob_hash),),
        )
    assert blob_gc.inspect_pending_gc_generations(tmp_path / "source.db") == [
        blob_gc.PendingGCGeneration("blocked", 1, 1, marker)
    ]
    actuator = PendingBlobGCGenerationAbandonActuator()
    args = PendingBlobGCGenerationAbandonArgs(tmp_path, "blocked")
    executor = OperationExecutor.for_archive_root(tmp_path)
    binding = runtime_operation_binding(actuator)
    principal = MutationPrincipal(
        "user:test",
        frozenset({"archive.blob_gc.abandon_pending_generation"}),
        "cli",
        "test",
    )
    preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=tmp_path)
    authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="confirm_flag")
    receipt = executor.execute_bound(binding, preview, authorization, args)

    assert receipt.affected_count == 1
    assert receipt.receipt_ref is not None
    assert receipt.domain_receipt["blob_effect"] == "none"
    assert receipt.domain_receipt["namespace_rebound"] is False
    assert store.exists(blob_hash)
    assert blob_gc._blob_namespace_identity(store.root).marker == marker
    assert _member_rows(tmp_path / "source.db") == [("blocked", blob_hash.upper(), "failed")]
    retry_preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=tmp_path)
    retry_authorization = executor.authorize_bound(
        binding, retry_preview, principal, confirmation_strength="confirm_flag"
    )
    retry = executor.execute_bound(binding, retry_preview, retry_authorization, args)
    assert retry.status == "already_satisfied"
    assert retry.affected_count == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM operation_attempts AS attempt "
            "JOIN operation_runs AS run ON run.operation_id = attempt.operation_id "
            "WHERE run.operation_name = ?",
            ("mutate-abandon-pending-blob-gc-generation",),
        ).fetchone() == (2,)


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
            "(generation_id, blob_hash, candidate_size_bytes, intent_committed_at_ms, outcome) "
            "VALUES ('pending-generation', ?, 1, 1, 'pending')",
            (b"p" * 32,),
        )
        conn.commit()

    (tmp_path / "index.db").unlink()
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", tmp_path / "blob")

    assert report.blocked_reason is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT completed_at_ms FROM gc_generations WHERE generation_id = 'pending-generation'"
        ).fetchone() == (None,)


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_finalizer_refuses_terminal_summary_while_a_member_outcome_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production execution reaches finalization but cannot skip a pending guard.

    Anti-vacuity twin: deleting the pending-member predicate in
    ``_finalize_gc_generation`` makes this generation terminal even though the
    patched outcome writer deliberately leaves one exact member unexplained.
    """
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    first_hash, _ = store.write_from_bytes(b"first finalization guard")
    second_hash, _ = store.write_from_bytes(b"second finalization guard")
    _backdate(store, first_hash)
    _backdate(store, second_hash)
    original_outcome = blob_gc._commit_gc_member_outcome

    def leave_second_pending(
        source_conn: sqlite3.Connection,
        *,
        generation_id: str,
        blob_hash: str,
        outcome: str,
        detail: str | None = None,
    ) -> None:
        if blob_hash == second_hash:
            return
        original_outcome(
            source_conn,
            generation_id=generation_id,
            blob_hash=blob_hash,
            outcome=outcome,
            detail=detail,
        )

    monkeypatch.setattr(blob_gc, "_commit_gc_member_outcome", leave_second_pending)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)

    assert report.generation_completed is False
    assert report.generation_id is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT completed_at_ms FROM gc_generations WHERE generation_id = ?", (report.generation_id,)
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT COUNT(*) FROM gc_generation_members WHERE generation_id = ? AND outcome = 'pending'",
            (report.generation_id,),
        ).fetchone() == (1,)


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_pending_member_blocks_on_unreadable_shard_not_object_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required shard failure leaves durable intent pending and names its blocker."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"unreadable shard")
    _backdate(store, blob_hash)
    original_final = blob_gc._final_gc_member_liveness

    def crash_after_intent(*args: object, **kwargs: object) -> tuple[BlobLiveness, BlobLiveness]:
        raise RuntimeError("leave exact intent pending")

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_after_intent)
    with pytest.raises(RuntimeError, match="leave exact intent pending"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)

    original_open = os.open

    def deny_required_shard(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == blob_hash[:2] and dir_fd is not None:
            raise PermissionError("simulated unreadable shard")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", deny_required_shard)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.blocked_reason is not None
    assert "blob namespace shard" in report.blocked_reason
    assert _member_rows(tmp_path / "source.db") == [(report.generation_id, blob_hash.upper(), "pending")]


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_direct_unlink_resumes_pending_generation_before_planning_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-retention's direct entry point shares recurring GC's pending gate."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    pending_hash, _ = store.write_from_bytes(b"pending direct gate")
    _backdate(store, pending_hash)
    original_final = blob_gc._final_gc_member_liveness

    def crash_after_intent(*args: object, **kwargs: object) -> tuple[BlobLiveness, BlobLiveness]:
        raise RuntimeError("pending direct gate")

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_after_intent)
    with pytest.raises(RuntimeError, match="pending direct gate"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", original_final)
    next_hash, _ = store.write_from_bytes(b"next direct gate")
    _backdate(store, next_hash)

    deleted, _bytes, errors = blob_gc.unlink_unreferenced_blob_hashes_under_exclusion(
        tmp_path / "source.db", tmp_path / "index.db", store.root, {next_hash}
    )

    assert (deleted, errors) == (1, ())
    assert not store.exists(pending_hash)
    assert store.exists(next_hash)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM gc_generations").fetchone() == (1,)


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_empty_intent_is_terminal_in_its_commit_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No-member plan cannot wedge restart recovery between intent and finalization."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_bytes(b"referenced means empty plan")
    _backdate(store, blob_hash)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
            "VALUES ('empty-plan-ref', 'codex-session', '/fixture', ?, ?, 1)",
            (bytes.fromhex(blob_hash), size),
        )
        conn.commit()

    def fail_if_called(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("empty intent must already be terminal")

    monkeypatch.setattr(blob_gc, "_finalize_gc_generation", fail_if_called)
    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.generation_written is True
    assert report.generation_completed is True
    restarted = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)
    assert restarted.blocked_reason is None
    assert restarted.generation_written is True


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_report_counts_this_resume_while_generation_counts_all_durable_outcomes(tmp_path: Path) -> None:
    """A restart cannot present a generation total as this invocation's deletion count."""
    initialize_active_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    pending_hash, pending_size = store.write_from_bytes(b"pending report counter")
    _backdate(store, pending_hash)
    generation_id = "mixed-outcome-generation"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes, "
            "blob_namespace_marker) VALUES (?, 1, NULL, 0, 0, ?)",
            (generation_id, blob_gc._blob_namespace_identity(store.root, create_marker=True).marker),
        )
        conn.execute(
            "INSERT INTO gc_generation_members "
            "(generation_id, blob_hash, candidate_size_bytes, intent_committed_at_ms, outcome, outcome_at_ms) "
            "VALUES (?, ?, 7, 1, 'removed', 1)",
            (generation_id, b"r" * 32),
        )
        conn.execute(
            "INSERT INTO gc_generation_members "
            "(generation_id, blob_hash, candidate_size_bytes, intent_committed_at_ms, outcome) "
            "VALUES (?, ?, ?, 1, 'pending')",
            (generation_id, bytes.fromhex(pending_hash), pending_size),
        )
        conn.commit()

    report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root)

    assert report.deleted_count == 1
    assert report.reclaimed_bytes == pending_size
    assert report.generation_completed is True
    assert report.generation_reclaimed_count == 2
    assert report.generation_reclaimed_bytes == pending_size + 7


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
    """A partial batch keeps its denominator and resumes exact pending members."""
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
        (min(first_hash, second_hash).upper(), "removed"),
        (max(first_hash, second_hash).upper(), "pending"),
    }
    generation_id = rows[0][0]
    assert not store.exists(min(first_hash, second_hash))
    assert store.exists(max(first_hash, second_hash))

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
    assert _member_rows(tmp_path / "source.db")[0][2] == "skipped_still_live"


def test_v33_source_migrates_additively_to_exact_gc_member_intent(tmp_path: Path) -> None:
    """The v34/v35 durable additions need no backup and preserve v33 rows."""
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute("DROP VIEW source_item_reconciliation")
        conn.execute("DROP INDEX idx_source_items_disposition")
        conn.execute("DROP INDEX idx_source_items_raw_id")
        conn.execute("DROP INDEX idx_hook_event_carriers_event")
        conn.execute("DROP TABLE source_items")
        conn.execute("DROP TABLE source_generations")
        conn.execute("DROP TABLE hook_event_carriers")
        conn.execute("DROP INDEX idx_gc_generation_members_pending")
        conn.execute("DROP TABLE gc_generation_members")
        conn.execute("DROP TABLE gc_generations")
        conn.execute(
            "CREATE TABLE gc_generations (generation_id TEXT PRIMARY KEY, started_at_ms INTEGER NOT NULL, "
            "completed_at_ms INTEGER, reclaimed_count INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_count >= 0), "
            "reclaimed_bytes INTEGER NOT NULL DEFAULT 0 CHECK(reclaimed_bytes >= 0)) STRICT"
        )
        conn.execute("PRAGMA user_version = 33")
        conn.execute(
            "INSERT INTO gc_generations (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES ('before-v34', 1, 1, 0, 0)"
        )
        conn.execute(
            "INSERT INTO gc_generations (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES ('memberless-pre035', 2, NULL, 0, 0)"
        )
        conn.commit()
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None, target_version=35)
        assert result.applied_versions == (34, 35)
        assert conn.execute("PRAGMA user_version").fetchone() == (35,)
        assert conn.execute("SELECT generation_id FROM gc_generations").fetchone() == ("before-v34",)
        assert conn.execute("SELECT COUNT(*) FROM gc_generation_members").fetchone() == (0,)
        assert conn.execute(
            "SELECT completed_at_ms FROM gc_generations WHERE generation_id = 'memberless-pre035'"
        ).fetchone() == (2,)
        assert "blob_namespace_marker" in {row[1] for row in conn.execute("PRAGMA table_info(gc_generations)")}
        assert [row[1] for row in conn.execute("PRAGMA table_info(gc_generation_members)")] == [
            "generation_id",
            "blob_hash",
            "candidate_size_bytes",
            "intent_committed_at_ms",
            "outcome",
            "outcome_at_ms",
            "outcome_detail",
        ]


def test_gc_refuses_a_missing_source_tier_before_planning(tmp_path: Path) -> None:
    """An index entry point never treats a missing durable source tier as empty."""
    initialize_active_archive_root(tmp_path)
    (tmp_path / "source.db").rename(tmp_path / "source.db.unavailable")

    report = blob_gc.run_blob_gc_report(tmp_path / "index.db", tmp_path / "blob")

    assert report.blocked_reason is not None
    assert "source tier is unavailable" in report.blocked_reason
