"""Regression tests for blob GC safety invariants.

Verifies two historical GC regressions stay fixed:

1. canonical liveness must check ``raw_sessions.blob_hash``
2. ``_candidate_blobs`` must walk all 256 prefix subdirectories
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import polylogue.storage.blob_gc as blob_gc
from polylogue.logging import capture
from polylogue.storage.blob_gc import (
    _candidate_blobs,
    read_gc_history,
    run_blob_gc,
    run_blob_gc_report,
    unlink_unreferenced_blob_hashes_under_exclusion,
)
from polylogue.storage.blob_liveness import BlobLiveness, LivenessState, inspect_blob_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index_sibling(db_path: str | Path) -> Path:
    """Bootstrap the production index tier GC requires beside a fixture."""
    index_path = Path(db_path).with_name("index.db")
    initialize_archive_database(index_path, ArchiveTier.INDEX)
    return index_path


def _make_db(path: str | Path | None = None) -> sqlite3.Connection:
    """Create an in-memory or file-based production source-tier fixture."""
    target = str(path) if path else ":memory:"
    if path is not None and Path(path).name == "index.db":
        initialize_archive_database(Path(path), ArchiveTier.INDEX)
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        return conn
    if path is not None:
        _make_index_sibling(path)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def _make_source_db(path: str | Path) -> sqlite3.Connection:
    return _make_db(path)


# ---------------------------------------------------------------------------
# Canonical decision: raw_id is observation identity, never blob liveness
# ---------------------------------------------------------------------------


def test_canonical_liveness_does_not_treat_raw_id_as_blob_reference() -> None:
    """A raw observation ID that happens to look like a hash cannot pin bytes."""
    conn = _make_db()
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('abc123def456', 'codex-session', 'test.json', ?, 42, 1)""",
        (b"x" * 32,),
    )
    conn.commit()
    assert inspect_blob_liveness(conn, "abc123def456").state is LivenessState.UNREFERENCED
    conn.close()


def test_canonical_liveness_rejects_unknown_hash() -> None:
    """A blob not in raw_sessions is not referenced."""
    conn = _make_db()
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('known-hash-1', 'codex-session', 'test.json', ?, 10, 1)""",
        (b"y" * 32,),
    )
    conn.commit()
    assert inspect_blob_liveness(conn, "unknown-dead-hash").state is LivenessState.UNREFERENCED
    conn.close()


def test_canonical_liveness_empty_table() -> None:
    """With no raw_sessions rows, nothing is referenced."""
    conn = _make_db()
    assert inspect_blob_liveness(conn, "any-hash").state is LivenessState.UNREFERENCED
    conn.close()


def test_canonical_liveness_recognizes_archive_source_hash(tmp_path: Path) -> None:
    """source references are BLOB hashes, not legacy raw_id text."""
    blob_hash = "a" * 64
    source_conn = _make_source_db(tmp_path / "source.db")
    source_conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES (?, 'codex-session', '/fixture', ?, ?, 1)""",
        ("raw-1", bytes.fromhex(blob_hash), 42),
    )
    source_conn.commit()

    decision = inspect_blob_liveness(source_conn, blob_hash)
    assert decision.state is LivenessState.LIVE
    assert decision.surfaces == ("source.db.raw_sessions",)
    source_conn.close()


def test_final_gc_recheck_uses_one_connection_when_source_and_index_alias(tmp_path: Path) -> None:
    """The legacy single-file repair route must not lock its own database twice."""

    db_path = tmp_path / "legacy.db"
    conn = _make_source_db(db_path)
    conn.close()
    store = BlobStore(tmp_path / "blobs")
    blob_hash, _size = store.write_from_bytes(b"legacy single-file candidate")

    deleted, _bytes, errors = unlink_unreferenced_blob_hashes_under_exclusion(db_path, db_path, store.root, {blob_hash})

    # The one-file legacy schema cannot satisfy current index ownership, so
    # this route must fail closed. The anti-vacuity condition is that a second
    # BEGIN IMMEDIATE on db_path would instead report "database is locked".
    assert deleted == 0
    assert any("index.attachments is missing" in error for error in errors)
    assert not any("database is locked" in error for error in errors)
    assert store.exists(blob_hash)


def test_final_gc_stage_build_is_constant_for_10k_candidates_and_large_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final writer lock builds the legacy anti-join once, never per hash."""
    import polylogue.storage.hook_payload_ref_reconciliation as reconciliation

    source_db = tmp_path / "source.db"
    conn = _make_source_db(source_db)
    (tmp_path / "blobs").mkdir()
    candidates = {f"{index:064x}" for index in range(10_000)}
    conn.executemany(
        """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', '/ledger', 1, 1)""",
        ((bytes.fromhex(blob_hash), f"dangling-{index}") for index, blob_hash in enumerate(sorted(candidates))),
    )
    conn.commit()
    conn.close()

    stage_builds = 0
    readiness_checks = 0
    original_build = reconciliation._build_match_stage
    original_readiness = reconciliation._match_stage_readiness

    def count_stage_builds(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal stage_builds
        stage_builds += 1
        return original_build(*args, **kwargs)

    def count_readiness_checks(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal readiness_checks
        readiness_checks += 1
        return original_readiness(*args, **kwargs)

    monkeypatch.setattr(reconciliation, "_build_match_stage", count_stage_builds)
    monkeypatch.setattr(reconciliation, "_match_stage_readiness", count_readiness_checks)
    deleted, deleted_bytes, errors = unlink_unreferenced_blob_hashes_under_exclusion(
        source_db, source_db.with_name("index.db"), tmp_path / "blobs", candidates
    )

    assert (deleted, deleted_bytes, errors) == (0, 0, ())
    # This is an operation-count contract, rather than a timing threshold:
    # restoring per-hash readiness/stage construction makes it 10,000.
    assert stage_builds == 1
    # The initial absent-stage check and post-build attestation are the only
    # full readiness passes. A per-hash anti-join validation is 10,002 here.
    assert readiness_checks == 2


def test_final_gc_synthetic_batch_uses_one_writer_connection_and_one_final_match_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure the selected bounded-batch design against the old per-member shape.

    The synthetic store is local to this test.  The selected design has one
    final writer connection/stage for 32 candidates; the retired per-member
    transaction design would make both counts 32.  This count-based
    measurement is stable where a wall-clock threshold would not be.
    """
    import polylogue.storage.hook_payload_ref_reconciliation as reconciliation
    from polylogue.storage import blob_gc

    source_db = tmp_path / "source.db"
    _make_source_db(source_db).close()
    store = BlobStore(tmp_path / "blobs")
    hashes = {store.write_from_bytes(f"synthetic gc {number}".encode())[0] for number in range(32)}
    final_connection_ids: set[int] = set()
    final_stage_builds = 0
    original_final = blob_gc._final_gc_member_liveness
    original_stage_build = reconciliation._build_match_stage

    def observe_final(source_conn, *args, **kwargs):  # type: ignore[no-untyped-def]
        final_connection_ids.add(id(source_conn))
        return original_final(source_conn, *args, **kwargs)

    def observe_stage(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal final_stage_builds
        final_stage_builds += 1
        return original_stage_build(*args, **kwargs)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", observe_final)
    monkeypatch.setattr(reconciliation, "_build_match_stage", observe_stage)
    deleted, _bytes, errors = unlink_unreferenced_blob_hashes_under_exclusion(
        source_db, source_db.with_name("index.db"), store.root, hashes
    )

    assert (deleted, errors) == (32, ())
    assert len(final_connection_ids) == 1
    # One matcher stage is needed for non-destructive planning and one for the
    # final writer batch; neither grows with the member count.
    assert final_stage_builds == 2


# ---------------------------------------------------------------------------
# _candidate_blobs — regression: must walk 256 prefix subdirectories
# ---------------------------------------------------------------------------


def test_candidate_blobs_finds_blobs_in_multiple_prefix_dirs(tmp_path: Path) -> None:
    """Blobs spread across prefix directories should all be found."""
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    hashes = set()
    for i in range(10):
        h, _ = blob_store.write_from_bytes(f"payload {i}".encode())
        hashes.add(h)

    # Verify blobs are spread across multiple prefix dirs
    prefix_dirs = {h[:2] for h in hashes}
    assert len(prefix_dirs) >= 1  # At least one prefix dir

    candidates = _candidate_blobs(blob_root, older_than=0)
    found_hashes = {h for h, _ in candidates}
    assert found_hashes == hashes, f"Candidate walk missed blobs: expected {hashes}, got {found_hashes}"


def test_candidate_blobs_respects_older_than(tmp_path: Path) -> None:
    """Blobs newer than older_than should be excluded."""
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    blob_store.write_from_bytes(b"fresh blob")

    # With very high older_than, no blobs should be returned
    candidates = _candidate_blobs(blob_root, older_than=3600)
    assert len(candidates) == 0

    # With older_than=0, all blobs should be found
    candidates = _candidate_blobs(blob_root, older_than=0)
    assert len(candidates) == 1


def test_candidate_blobs_empty_dir(tmp_path: Path) -> None:
    """A missing root is a namespace blocker, not an empty candidate census."""
    with pytest.raises(RuntimeError, match="blob namespace"):
        _candidate_blobs(tmp_path / "nonexistent", older_than=0)


def test_candidate_blobs_skips_dotfiles(tmp_path: Path) -> None:
    """Files starting with '.' (temp files) should be skipped."""
    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    prefix_dir = blob_root / "aa"
    prefix_dir.mkdir()
    # Create a temp file that should be skipped
    (prefix_dir / ".blob.temp").write_bytes(b"temp")
    # Create a real blob manually — a valid 64-char lowercase-hex hash.
    remainder = "bb" * 31
    (prefix_dir / remainder).write_bytes(b"real")

    candidates = _candidate_blobs(blob_root, older_than=0)
    found = {h for h, _ in candidates}
    assert "aa" + remainder in found
    assert "aa.blob.temp" not in found


def test_candidate_blobs_skips_non_two_char_prefix_dirs(tmp_path: Path) -> None:
    """Directories not matching the two-char prefix pattern should be skipped."""
    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    # Valid prefix dir with a valid 64-char lowercase-hex blob hash.
    (blob_root / "ab").mkdir()
    remainder = "cd" * 31
    (blob_root / "ab" / remainder).write_bytes(b"real")
    # Non-prefix dir
    (blob_root / "not-a-prefix").mkdir()

    candidates = _candidate_blobs(blob_root, older_than=0)
    found = {h for h, _ in candidates}
    assert "ab" + remainder in found
    assert not any(h.startswith("not") for h in found)


def test_candidate_blobs_filters_invalid_namespace_entries(tmp_path: Path) -> None:
    """A shard file whose combined name is not exactly 64 lowercase-hex chars
    must never be shortlisted.

    Regression for a review finding: ``_candidate_blobs`` used to shortlist
    any aged regular file regardless of name shape, so a foreign or
    partially-written entry in a valid-looking shard directory would reach
    ``_commit_gc_generation_intent`` and violate the 32-byte ``blob_hash``
    CHECK constraint (or raise ``ValueError`` from ``bytes.fromhex``).
    ``BlobStore.iter_namespace`` already classifies these as invalid
    namespace entries; GC must leave them alone. If the length/hex guard in
    ``_candidate_blobs`` is removed, this test goes red because the
    short/uppercase/non-hex entries would reappear in the candidate set.
    """
    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    prefix_dir = blob_root / "aa"
    prefix_dir.mkdir()

    valid_remainder = "11" * 31  # 62 chars + "aa" prefix = 64
    (prefix_dir / valid_remainder).write_bytes(b"real")
    (prefix_dir / "tooshort").write_bytes(b"short")  # not 64 chars total
    (prefix_dir / ("zz" * 31)).write_bytes(b"nonhex")  # not lowercase hex
    (prefix_dir / (valid_remainder.upper())).write_bytes(b"uppercase")  # not lowercase hex

    candidates = _candidate_blobs(blob_root, older_than=0)
    found = {h for h, _ in candidates}
    assert found == {"aa" + valid_remainder}


# ---------------------------------------------------------------------------
# run_blob_gc integration (lightweight)
# ---------------------------------------------------------------------------


def test_run_blob_gc_empty_store(tmp_path: Path) -> None:
    """GC on an empty blob store should succeed with 0 deletions."""
    db_path = tmp_path / "archive.db"
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()

    conn = _make_db(db_path)
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_dir), max_batch=10)
    assert deleted == 0


def test_run_blob_gc_preserves_referenced_blobs(tmp_path: Path) -> None:
    """GC must not delete blobs that are still referenced in raw_sessions."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    # Create a blob and a matching raw_sessions row
    h, _ = blob_store.write_from_bytes(b"referenced content")

    conn = _make_db(db_path)
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('raw', 'codex-session', 'test.json', ?, ?, 1)""",
        (bytes.fromhex(h), len(b"referenced content")),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    assert deleted == 0
    # Blob still on disk
    assert blob_store.exists(h)


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_preserves_archive_source_referenced_blobs(tmp_path: Path) -> None:
    """GC run from ``index.db`` must preserve blobs referenced by sibling ``source.db``."""
    index_db_path = tmp_path / "index.db"
    source_db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    blob_hash, _ = blob_store.write_from_bytes(b"archive referenced content")
    _backdate(blob_store, blob_hash)

    index_conn = _make_db(index_db_path)
    index_conn.close()

    source_conn = _make_source_db(source_db_path)
    source_conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES (?, 'codex-session', '/fixture', ?, ?, 1)""",
        ("raw-v1", bytes.fromhex(blob_hash), len(b"archive referenced content")),
    )
    source_conn.commit()
    source_conn.close()

    deleted = run_blob_gc(str(index_db_path), str(blob_root), max_batch=10)

    assert deleted == 0
    assert blob_store.exists(blob_hash)
    # A pass ran but reclaimed nothing because sibling source.db references it.
    history = read_gc_history(str(index_db_path), limit=1)
    assert len(history) == 1
    assert history[0].reclaimed_count == 0
    assert history[0].reclaimed_bytes == 0


def test_run_blob_gc_max_batch_bound(tmp_path: Path) -> None:
    """GC should never exceed max_batch even with many orphans."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    # Create several orphan blobs
    for i in range(5):
        blob_store.write_from_bytes(f"orphan {i}".encode())

    conn = _make_db(db_path)
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=2)
    # May be 0 due to MIN_AGE_S, but should never exceed max_batch
    assert 0 <= deleted <= 2


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_bounds_final_lock_rechecks_with_many_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Referenced candidates are filtered before the destructive lock."""
    from polylogue.storage import blob_gc

    db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)
    conn = _make_db(db_path)
    for index in range(40):
        blob_hash, size = store.write_from_bytes(f"referenced-{index}".encode())
        _backdate(store, blob_hash)
        conn.execute(
            """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES (?, 'codex-session', ?, ?, ?, 1)""",
            (f"raw-{index}", f"{index}.json", bytes.fromhex(blob_hash), size),
        )
    orphan_hash, _ = store.write_from_bytes(b"bounded orphan")
    _backdate(store, orphan_hash)
    conn.commit()
    conn.close()

    lock_rechecks = 0
    original_inspect = blob_gc.inspect_blob_liveness

    def count_lock_rechecks(conn, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal lock_rechecks
        if conn.in_transaction:
            lock_rechecks += 1
        return original_inspect(conn, *args, **kwargs)

    monkeypatch.setattr(blob_gc, "inspect_blob_liveness", count_lock_rechecks)
    report = run_blob_gc_report(db_path, blob_root, max_batch=2)

    assert report.candidate_count == 41
    assert report.deleted_count == 1
    assert lock_rechecks <= 2
    assert not store.exists(orphan_hash)


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_does_not_stage_again_when_all_candidates_are_referenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)
    conn = _make_db(db_path)
    blob_hash, size = store.write_from_bytes(b"referenced")
    _backdate(store, blob_hash)
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('raw', 'codex-session', 'referenced.json', ?, ?, 1)""",
        (bytes.fromhex(blob_hash), size),
    )
    conn.commit()
    conn.close()

    from polylogue.storage import blob_gc

    final_member_checks = 0
    original_final_member_liveness = blob_gc._final_gc_member_liveness

    def count_final_member_checks(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal final_member_checks
        final_member_checks += 1
        return original_final_member_liveness(*args, **kwargs)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", count_final_member_checks)

    report = run_blob_gc_report(db_path, blob_root, max_batch=10)

    assert report.deleted_count == 0
    assert report.skipped_referenced == 1
    # An empty deletion plan records one generation but must not enter the
    # per-member final recheck route.
    assert final_member_checks == 0
    assert report.generation_written is True


def test_run_blob_gc_nonexistent_blob_dir(tmp_path: Path) -> None:
    """GC on nonexistent directory should return 0 without crash."""
    db_path = tmp_path / "archive.db"
    conn = _make_db(db_path)
    conn.close()
    deleted = run_blob_gc(str(db_path), str(tmp_path / "nonexistent"), max_batch=10)
    assert deleted == 0


# ---------------------------------------------------------------------------
# #1190 — sharded unlink path + accurate deleted counter
# ---------------------------------------------------------------------------


def _backdate(blob_store: BlobStore, blob_hash: str, *, seconds: float = 3600) -> None:
    """Backdate a blob's mtime past MIN_AGE_S so it is GC-eligible."""
    import os

    path = blob_store.blob_path(blob_hash)
    past = __import__("time").time() - seconds
    os.utime(path, (past, past))


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_unlinks_sharded_path_and_increments_counter(tmp_path: Path) -> None:
    """#1190 regression: an orphan blob present at the sharded path
    ``{root}/{prefix}/{remainder}`` must actually be removed and the
    ``deleted`` counter must increment by exactly 1.

    Before the fix, ``run_blob_gc`` unlinked ``{root}/{full_hash}``
    (a path that never exists for a real blob), and ``missing_ok=True``
    silently swallowed the failure. The counter still bumped, so the
    function reported successful reclamation while leaving the blob on
    disk.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"orphan to reclaim")
    sharded = blob_store.blob_path(h)
    assert sharded.is_file()
    assert (blob_root / h).exists() is False  # never lived at the flat path

    _backdate(blob_store, h)

    _make_db(db_path).close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)

    assert deleted == 1, "deleted counter must match actual unlinks"
    assert not sharded.exists(), "sharded blob must actually be removed from disk"


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_does_not_increment_when_file_already_missing(tmp_path: Path) -> None:
    """#1190 regression: when the candidate file has vanished between
    discovery and unlink (concurrent reclaimer, stale candidate, manual
    cleanup), the ``deleted`` counter must NOT increment and the recorded
    generation row reclaims nothing.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"will-vanish")
    sharded = blob_store.blob_path(h)
    _backdate(blob_store, h)

    # Race simulation: file disappears after _candidate_blobs() lists it
    # but before run_blob_gc unlinks it. We approximate this by removing
    # the file ourselves before the call — _candidate_blobs has already
    # seen the dirent; the unlink will hit FileNotFoundError.
    # But _candidate_blobs runs inside run_blob_gc. We instead remove the
    # underlying file out-of-band BEFORE the call but AFTER recording the
    # candidate via a wrapper: simpler — patch the candidate listing to
    # report this hash even though the file is now gone.
    sharded.unlink()
    assert not sharded.exists()

    _make_db(db_path).close()

    # Re-create a sibling so _candidate_blobs sees the directory; then
    # patch the listing to include the vanished hash too.
    from polylogue.storage import blob_gc as gc_mod

    real_listing = gc_mod._candidate_blobs

    def patched(root: Path, *, older_than: float) -> list[tuple[str, float]]:
        out = list(real_listing(root, older_than=older_than))
        out.append((h, 0.0))
        return out

    gc_mod._candidate_blobs = patched  # type: ignore[assignment]
    try:
        deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    finally:
        gc_mod._candidate_blobs = real_listing

    assert deleted == 0, "counter must not bump when no file was actually unlinked"

    # The pass still recorded a generation row that reclaimed nothing.
    history = read_gc_history(str(db_path), limit=1)
    assert len(history) == 1
    assert history[0].reclaimed_count == 0
    assert history[0].reclaimed_bytes == 0


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_dry_run_does_not_delete_or_record_generation(tmp_path: Path) -> None:
    """#1190 ambitious-expansion: --dry-run previews without committing.

    A dry-run must:
      - NOT remove any file from disk;
      - NOT insert a row into ``gc_generations`` (no generation slot consumed);
      - still return the would-be count.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"dry-run orphan")
    _backdate(blob_store, h)
    _make_db(db_path).close()

    would_delete = run_blob_gc(str(db_path), str(blob_root), max_batch=10, dry_run=True)

    assert would_delete == 1
    assert blob_store.exists(h), "dry-run must never touch disk"

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM gc_generations").fetchone()
    finally:
        conn.close()
    assert row[0] == 0, "dry-run must not consume a generation slot"


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_dry_run_does_not_create_namespace_marker(tmp_path: Path) -> None:
    """A dry-run preview must not write the namespace marker.

    Regression for a review finding: every ``dry_run=True`` call used to
    reach the unconditional ``create_marker=True`` path, mutating an archive
    that had no marker yet and failing against read-only backup media. If
    ``run_blob_gc_report`` is changed back to bind (and create) a namespace
    marker unconditionally, this test goes red because the marker file would
    exist after a dry run against a marker-less archive.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"dry-run orphan")
    _backdate(blob_store, h)
    _make_db(db_path).close()

    marker_path = blob_root / ".polylogue-blob-namespace"
    assert not marker_path.exists()

    would_delete = run_blob_gc(str(db_path), str(blob_root), max_batch=10, dry_run=True)

    assert would_delete == 1
    assert not marker_path.exists(), "a read-only preview must not create the namespace marker"


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_run_blob_gc_records_reclaim_counters(tmp_path: Path) -> None:
    """#1743: each committed pass writes a typed ``gc_generations`` row
    capturing the reclaimed blob count and freed bytes — the durable
    audit trail that replaced the JSON evidence column.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    referenced_hash, _ = blob_store.write_from_bytes(b"keep me")
    orphan_payload = b"delete me"
    orphan_hash, _ = blob_store.write_from_bytes(orphan_payload)
    _backdate(blob_store, referenced_hash)
    _backdate(blob_store, orphan_hash)

    conn = _make_db(db_path)
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('raw', 'codex-session', 'x.json', ?, 1, 1)""",
        (bytes.fromhex(referenced_hash),),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    assert deleted == 1

    history = read_gc_history(str(db_path), limit=1)
    assert len(history) == 1
    row = history[0]
    assert row.reclaimed_count == 1
    assert row.reclaimed_bytes == len(orphan_payload)
    assert row.generation_id.startswith("gc-")
    assert row.completed_at_ms is not None
    assert row.started_at_ms <= row.completed_at_ms


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_read_gc_history_returns_recent_passes(tmp_path: Path) -> None:
    """#1743: ``read_gc_history`` surfaces one typed row per committed pass,
    so a ``gc-history`` operator surface can show recent reclamation without
    bespoke SQLite tooling.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    for i in range(3):
        h, _ = blob_store.write_from_bytes(f"orphan-{i}".encode())
        _backdate(blob_store, h)
        run_blob_gc(str(db_path), str(blob_root), max_batch=10)

    history = read_gc_history(str(db_path), limit=10)
    assert len(history) == 3
    assert all(row.reclaimed_count == 1 for row in history)
    assert all(row.generation_id.startswith("gc-") for row in history)


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call"
)
def test_run_blob_gc_refuses_when_the_index_tier_cannot_be_read(tmp_path: Path) -> None:
    """An unavailable reference tier is a blocker, never proof of non-reference.

    ``attachments.blob_hash`` lives in the index tier and nowhere else -- 1,240
    distinct attachment payloads (~425 MB) on the live archive are reachable
    only that way. The tier is rebuildable and reached through a symlink into
    a generations directory, so a reset or an interrupted generation promotion
    leaves the path absent. Reading that absence as "unreferenced" unlinks
    durable bytes that every other surface agrees nothing else holds.
    """
    db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)
    index_path = tmp_path / "index.db"

    attachment_hash, _size = store.write_from_bytes(b"attachment payload")
    control_hash, _control_size = store.write_from_bytes(b"nothing references me")
    _backdate(store, attachment_hash)
    _backdate(store, control_hash)
    conn = _make_db(db_path)
    conn.commit()
    conn.close()

    # The only reference to the attachment bytes anywhere in the archive.
    index_conn = sqlite3.connect(str(index_path))
    index_conn.execute(
        "INSERT INTO attachments (attachment_id, blob_hash, acquisition_status) VALUES ('attachment', ?, 'acquired')",
        (bytes.fromhex(attachment_hash),),
    )
    index_conn.commit()
    index_conn.close()

    # The control blob proves this pass reaches the liveness check at all: it
    # clears the same age gate and the same candidate scan, and it is
    # collected. Without it, a blob retained for an unrelated reason (age,
    # an empty candidate walk) would satisfy every assertion below.
    assert run_blob_gc(db_path, blob_root, max_batch=10) == 1
    assert not store.exists(control_hash)
    assert store.exists(attachment_hash)

    # Now the index tier is gone -- the shape a reset or a swapped generation
    # leaves behind. Nothing else in the archive names the attachment blob.
    second_control_hash, _second_size = store.write_from_bytes(b"nothing references me either")
    _backdate(store, second_control_hash)
    index_path.unlink()

    report = run_blob_gc_report(db_path, blob_root, max_batch=10)

    # The data-loss claim first: without the gate this is the assertion that
    # fails, and it fails because the bytes are gone.
    assert store.exists(attachment_hash), "unlinked a blob whose only reference tier was unreadable"
    assert report.blocked_reason is not None
    assert "index tier" in report.blocked_reason
    assert report.deleted_count == 0
    assert report.generation_written is False
    # The refusal is total, not selective: even a blob that was provably
    # collectable a moment ago is left alone while a tier cannot be read.
    assert store.exists(second_control_hash)


@pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call"
)
def test_run_blob_gc_serializes_the_recheck_against_an_index_tier_writer(tmp_path: Path) -> None:
    """The recheck+unlink window must exclude writers on every tier it reads.

    Invariant 3 claims the final reference recheck and the unlink are
    serialized under a write lock. A read-only sibling connection excludes
    nobody, so a concurrent commit could create the very reference the pass is
    about to decide does not exist. Holding an index-tier write transaction
    here must therefore make the destructive pass fail rather than delete.
    """
    db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)

    orphan_hash, _size = store.write_from_bytes(b"orphan payload")
    _backdate(store, orphan_hash)
    conn = _make_db(db_path)
    conn.commit()
    conn.close()

    # Stand in for a concurrent process mid-write on the index tier.
    writer = sqlite3.connect(str(tmp_path / "index.db"), timeout=0.1)
    writer.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            run_blob_gc(db_path, blob_root, max_batch=10)
    finally:
        writer.rollback()
        writer.close()

    assert store.exists(orphan_hash), "unlinked while an index-tier writer held the reference table"

    # With no competing writer the same pass reclaims the blob, so the
    # assertion above is about serialization, not a blanket refusal.
    assert run_blob_gc(db_path, blob_root, max_batch=10) == 1
    assert not store.exists(orphan_hash)


# ---------------------------------------------------------------------------
# polylogue-kdt2f -- a replaced (wiped/rebuilt) index tier cannot prove absence
# ---------------------------------------------------------------------------


def _replace_index_tier(index_path: Path) -> None:
    """Wipe and re-bootstrap the index tier, as a reset or rebuild does."""
    index_path.unlink()
    initialize_archive_database(index_path, ArchiveTier.INDEX)


@pytest.mark.uses_real_clock(
    "backdates real blob mtimes via os.utime; blob_gc.py's age gate compares them against a real time.time() call"
)
def test_run_blob_gc_protects_index_only_blobs_after_the_index_tier_is_replaced(tmp_path: Path) -> None:
    """A blob only ``index.attachments`` claims survives a wiped, re-bootstrapped index.

    ``index.db`` is rebuildable; a wipe leaves a schema-current, row-empty file
    whose silence GC used to read as "unreferenced". On the live archive 1,240
    attachment payloads (~425 MB) have no ``blob_refs`` row, no direct source
    owner and no publication receipt, so the index's silence was their death
    warrant and the pass reported clean success with ``blocked_reason=None``.

    Anti-vacuity: the first pass here runs against the *materialized* index and
    reclaims nothing, which records the witness and proves the candidate walk,
    the age gate and the liveness route are all reached. Deleting the witness
    file, or making ``inspect_blob_liveness`` ignore
    ``index_authority_blocker`` again, unlinks the attachment blob in the
    second pass and leaves ``blocked_reason`` None.
    """
    source_db_path = tmp_path / "source.db"
    index_db_path = tmp_path / "index.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)

    attachment_hash, _size = store.write_from_bytes(b"index-only attachment payload")
    _backdate(store, attachment_hash)

    source_conn = _make_source_db(source_db_path)
    source_conn.commit()
    source_conn.close()

    index_conn = sqlite3.connect(str(index_db_path))
    index_conn.execute(
        "INSERT INTO attachments (attachment_id, blob_hash, acquisition_status) VALUES ('a1', ?, 'acquired')",
        (bytes.fromhex(attachment_hash),),
    )
    index_conn.commit()
    index_conn.close()

    # Pass one: the index is materialized and owns the blob. Nothing is
    # reclaimed and GC records what the tier held.
    first = run_blob_gc_report(source_db_path, blob_root, max_batch=10)
    assert first.blocked_reason is None
    assert first.deleted_count == 0
    assert first.skipped_referenced == 1
    assert store.exists(attachment_hash)

    _replace_index_tier(index_db_path)

    second = run_blob_gc_report(source_db_path, blob_root, max_batch=10)
    assert store.exists(attachment_hash), "a replaced index tier's silence unlinked an index-only blob"
    assert second.deleted_count == 0
    assert second.blocked_reason is not None
    assert "replacement file" in second.blocked_reason


@pytest.mark.uses_real_clock(
    "backdates real blob mtimes via os.utime; blob_gc.py's age gate compares them against a real time.time() call"
)
def test_run_blob_gc_still_collects_orphans_against_an_index_that_never_held_blobs(tmp_path: Path) -> None:
    """An index that never owned a blob is not a wiped index, and GC keeps reclaiming.

    This is the over-blocking failure mode a pass-level preflight refusal has:
    a source-decidable archive whose index is legitimately empty must still
    collect genuinely unreferenced bytes, and a source-referenced blob must
    still be retained, on both the first and a subsequent pass (the second pass
    runs against a witness this code recorded itself).

    Anti-vacuity: the referenced blob clears the same age gate and candidate
    walk as the orphan, so ``deleted_count == 1`` cannot be satisfied by an
    empty or skipped pass. Widening the gate to "the index holds no rows"
    turns the orphan deletions below red.
    """
    source_db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    store = BlobStore(blob_root)

    referenced_hash, size = store.write_from_bytes(b"source referenced payload")
    orphan_hash, _orphan_size = store.write_from_bytes(b"nothing references me")
    _backdate(store, referenced_hash)
    _backdate(store, orphan_hash)

    source_conn = _make_source_db(source_db_path)
    source_conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms)
        VALUES ('raw-1', 'codex-session', '/fixture', ?, ?, 1)""",
        (bytes.fromhex(referenced_hash), size),
    )
    source_conn.commit()
    source_conn.close()

    first = run_blob_gc_report(source_db_path, blob_root, max_batch=10)
    assert first.blocked_reason is None
    assert first.deleted_count == 1
    assert store.exists(referenced_hash)
    assert not store.exists(orphan_hash)

    second_orphan, _ = store.write_from_bytes(b"also nothing references me")
    _backdate(store, second_orphan)
    second = run_blob_gc_report(source_db_path, blob_root, max_batch=10)
    assert second.blocked_reason is None
    assert second.deleted_count == 1
    assert store.exists(referenced_hash)
    assert not store.exists(second_orphan)


# ---------------------------------------------------------------------------
# polylogue-n9t0p -- every refusal inside the locked execution window emits
# ---------------------------------------------------------------------------


def _backdate_blob(store: BlobStore, blob_hash: str) -> None:
    path = store.blob_path(blob_hash)
    old = time.time() - 3600
    os.utime(path, (old, old))


def _pending_generation(tmp_path: Path, payloads: list[bytes], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Leave one pending GC generation whose members are all still unlinked.

    Uses the production fault seam ``_final_gc_member_liveness``: a crash after
    durable member intent is exactly the state a restart resumes through
    ``_execute_gc_generation_members``, the locked window under test.
    """
    bootstrap_archive_root(tmp_path)
    store = BlobStore(tmp_path / "blob")
    hashes = []
    for payload in payloads:
        blob_hash, _ = store.write_from_bytes(payload)
        _backdate_blob(store, blob_hash)
        hashes.append(blob_hash)

    def crash_after_intent(*_args: Any, **_kwargs: Any) -> tuple[BlobLiveness, BlobLiveness]:
        raise RuntimeError("leave GC intent pending")

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", crash_after_intent)
    with pytest.raises(RuntimeError, match="leave GC intent pending"):
        blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=len(payloads))
    monkeypatch.undo()
    return hashes


def _refusals(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record.get("event") == "storage.blob_gc.refused"]


def _blocked_liveness(reason: str) -> BlobLiveness:
    return BlobLiveness(state=LivenessState.BLOCKED, blockers=(reason,))


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
@pytest.mark.parametrize(
    ("fault", "needle"),
    [
        ("generation_namespace", "forced namespace mismatch"),
        ("namespace_identity", "forced namespace unavailable"),
        ("window_preflight", "forced window preflight blocker"),
        ("legacy_hook_stage", "forced rekey matcher failure"),
    ],
)
def test_locked_window_refusals_emit_a_refused_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, needle: str
) -> None:
    """Each pre-unlink refusal inside the locked window emits, never returns silently.

    Anti-vacuity: deleting the ``_emit_gc_refusal`` call at the matching return
    site leaves ``report.blocked_reason`` set with an empty refusal stream, and
    that parameter case goes red. A refusal that emitted nothing is
    indistinguishable from a pass that ran and reclaimed nothing -- the defect
    this test pins.
    """
    _pending_generation(tmp_path, [b"n9t0p locked-window member"], monkeypatch)
    store = BlobStore(tmp_path / "blob")

    if fault == "generation_namespace":
        monkeypatch.setattr(blob_gc, "_generation_namespace_matches", lambda *_a, **_k: needle)
    elif fault == "namespace_identity":
        real_identity = blob_gc._blob_namespace_identity

        def failing_identity(*args: Any, **kwargs: Any) -> Any:
            # Only the locked window's own identity read is the site under
            # test; the earlier namespace-match check has its own refusal.
            if sys._getframe(1).f_code.co_name == "_execute_gc_generation_members":
                raise blob_gc._BlobNamespaceUnavailableError(needle)
            return real_identity(*args, **kwargs)

        monkeypatch.setattr(blob_gc, "_blob_namespace_identity", failing_identity)
    elif fault == "window_preflight":
        real_liveness = blob_gc.inspect_blob_liveness

        def blocked_preflight(conn: Any, blob_hash: str, **kwargs: Any) -> BlobLiveness:
            if blob_hash == "":
                return _blocked_liveness(needle)
            return real_liveness(conn, blob_hash, **kwargs)

        monkeypatch.setattr(blob_gc, "inspect_blob_liveness", blocked_preflight)
    else:

        def failing_stage(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError(needle)

        monkeypatch.setattr(blob_gc, "prepare_match_stage", failing_stage)

    with capture() as records:
        report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=1)

    assert report.blocked_reason is not None and needle in report.blocked_reason
    assert report.deleted_count == 0
    matching = [record for record in _refusals(records) if needle in str(record.get("error_detail"))]
    assert matching, "a locked-window refusal emitted nothing"
    assert matching[0]["outcome"] == "refused"
    assert matching[0]["phase"] == "execute"
    assert matching[0]["reclaimed"] == 0
    assert matching[0]["bytes"] == 0


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_final_recheck_blocker_after_a_deletion_reports_the_partial_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocker in the final locked recheck reports the deletions already made.

    Anti-vacuity: without the emit at the ``protection.blockers`` return inside
    the member loop, the stream carries no refusal at all and a reader sees
    only a GC that unlinked one blob -- an aborted pass rendered as a clean
    one. Dropping the counts from that emit turns the ``reclaimed``/``bytes``
    assertions red.
    """
    hashes = _pending_generation(tmp_path, [b"n9t0p member one", b"n9t0p member two"], monkeypatch)
    store = BlobStore(tmp_path / "blob")
    ordered = sorted(hashes)
    real_final = blob_gc._final_gc_member_liveness
    seen: list[str] = []

    def block_the_second_member(
        source_conn: Any, index_conn: Any, blob_hash: str, **kwargs: Any
    ) -> tuple[BlobLiveness, BlobLiveness]:
        seen.append(blob_hash)
        if len(seen) >= 2:
            return _blocked_liveness("forced recheck blocker"), BlobLiveness(state=LivenessState.UNREFERENCED)
        return real_final(source_conn, index_conn, blob_hash, **kwargs)

    monkeypatch.setattr(blob_gc, "_final_gc_member_liveness", block_the_second_member)

    with capture() as records:
        report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)

    assert report.deleted_count == 1, "the first member must be unlinked before the blocker"
    assert not store.exists(ordered[0])
    assert store.exists(ordered[1])
    assert report.blocked_reason == "forced recheck blocker"

    matching = [record for record in _refusals(records) if "forced recheck blocker" in str(record.get("error_detail"))]
    assert matching, "a blocker in the final locked recheck emitted nothing after a real deletion"
    assert matching[0]["outcome"] == "refused"
    assert matching[0]["phase"] == "final_recheck"
    assert matching[0]["reclaimed"] == report.deleted_count == 1
    assert matching[0]["bytes"] == report.reclaimed_bytes > 0


@pytest.mark.uses_real_clock("backdates temporary blobs to pass production GC's age gate")
def test_namespace_loss_during_the_unlink_pass_reports_the_partial_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the blob namespace mid-unlink emits a refusal carrying the partial work.

    Anti-vacuity: removing the emit at the ``_BlobNamespaceUnavailableError``
    handler around the member loop leaves an empty refusal stream while one
    blob is already gone.
    """
    hashes = _pending_generation(tmp_path, [b"n9t0p unlink one", b"n9t0p unlink two"], monkeypatch)
    store = BlobStore(tmp_path / "blob")
    ordered = sorted(hashes)
    real_unlink = blob_gc._unlink_observed_gc_member
    calls = {"n": 0}

    def lose_namespace_on_second(observed: Any) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise blob_gc._BlobNamespaceUnavailableError("forced namespace loss mid-unlink")
        real_unlink(observed)

    monkeypatch.setattr(blob_gc, "_unlink_observed_gc_member", lose_namespace_on_second)

    with capture() as records:
        report = blob_gc.run_blob_gc_report(tmp_path / "source.db", store.root, max_batch=2)

    assert report.deleted_count == 1
    assert not store.exists(ordered[0])
    assert report.blocked_reason == "forced namespace loss mid-unlink"

    matching = [record for record in _refusals(records) if "forced namespace loss" in str(record.get("error_detail"))]
    assert matching, "a namespace loss during the unlink pass emitted nothing"
    assert matching[0]["outcome"] == "refused"
    assert matching[0]["phase"] == "unlink"
    assert matching[0]["reclaimed"] == 1
    assert matching[0]["bytes"] == report.reclaimed_bytes > 0
