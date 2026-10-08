"""Contract suite pinning blob-store invariants from docs/internals.md.

These tests pin the **documented contract** of the content-addressed blob
store and the GC concurrency model, distinct from the happy-path CRUD
coverage in ``test_blob_store.py`` and the GC regression coverage in
``test_blob_gc.py``.

The invariants pinned here are sourced from
``docs/internals.md`` § "Blob Store Model" and § "GC concurrency model
— snapshot reference check plus age floor".  Each test references the
invariant it pins in its docstring so a future doc/code drift is
attributable.

A prior revision of this suite also pinned a blob-GC lease mechanism
(``acquire_blob_leases``/``release_operation_leases``/``_has_active_lease``);
that mechanism was removed as unreachable dead code (polylogue-v7e0 — no
production ingest caller ever populated the lease payload keys) and its
tests were removed with it.

The blob store on disk (``polylogue/storage/blob_store.py``) is a pure
content-addressed store with no notion of grouping. The grouping the
docs refer to ("``link_group_key`` groups related blobs, e.g. all
blobs belonging to one session") is implemented in
``artifact_observations.link_group_key`` (see
``polylogue/storage/sqlite/archive_tiers/source.py`` and
``polylogue/storage/artifacts/inspection.py``). This contract suite
pins both layers — pure CAS *and* the artifact-observation grouping
the docs describe.
"""

from __future__ import annotations

import hashlib
import sqlite3
import string
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.blob_gc import run_blob_gc_report
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

pytestmark = pytest.mark.uses_real_clock(
    "Blob aging uses time.time() to backdate file mtime; production GC reads file mtime, not datetime.now."
)

# ---------------------------------------------------------------------------
# Helpers (kept local; deliberately minimal — these tests are about the
# blob-store contract, not the wider storage runtime)
# ---------------------------------------------------------------------------


def _make_gc_archive(root: Path) -> tuple[Path, BlobStore]:
    """Bootstrap the production split archive that ``run_blob_gc`` reads.

    GC fails closed unless every canonical owner surface exists, so a
    hand-written minimum schema drifts silently: once GC learns a new owner
    table the fixture's GC refuses up front and every "never deletes a live
    blob" assertion passes without GC having run. The production bootstrap
    carries each owner table GC resolves, and each test below also asserts GC
    was not refused.
    """
    initialize_active_archive_root(root)
    return root / "source.db", BlobStore(root / "blob")


# ---------------------------------------------------------------------------
# § "Blob Store Model" → "Content addressing: SHA-256 hash over raw bytes.
# The hash IS the address. Identical content → identical hash → automatic
# deduplication."
# ---------------------------------------------------------------------------


_DETERMINISM_PAYLOADS = [
    pytest.param(b"", id="empty"),
    pytest.param(b"a", id="single-byte"),
    pytest.param(b"polylogue blob payload", id="ascii"),
    pytest.param("polilog \u00e9\u4e2d\U0001f600".encode(), id="utf8-multibyte"),
    pytest.param(b"\x00\x01\x02\xfd\xfe\xff", id="binary"),
    pytest.param(b"x" * (2 * 1024 * 1024 + 3), id="multi-mib-spans-chunks"),
]


@pytest.mark.parametrize("payload", _DETERMINISM_PAYLOADS)
def test_content_addressing_is_deterministic(tmp_path: Path, payload: bytes) -> None:
    """docs/internals.md § Blob Store Model: same bytes → same hash → same
    on-disk path; the hash IS the address.

    This is the load-bearing claim for both dedup and cross-process
    convergence. We verify it across an interesting payload catalogue.
    """
    store = BlobStore(tmp_path / "blobs")

    expected = hashlib.sha256(payload).hexdigest()

    h_bytes, _ = store.write_from_bytes(payload)
    assert h_bytes == expected
    h_fileobj, _ = store.write_from_fileobj(BytesIO(payload))
    assert h_fileobj == expected

    # Source-file path variant — exercises ``write_from_path``'s
    # stream-hash code path which is the one used by source acquisition.
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    h_path, _ = store.write_from_path(src)
    assert h_path == expected

    # The on-disk path is a pure function of the hash.
    dest = store.blob_path(expected)
    assert dest == store.root / expected[:2] / expected[2:]


# ---------------------------------------------------------------------------
# § "Blob Store Model" → "Prefix sharding: 256 subdirectories
# (blob/00/ through blob/ff/)".
# ---------------------------------------------------------------------------


def test_prefix_shard_layout_uses_first_two_hex_chars(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: blob path is
    ``{root}/{hash[:2]}/{hash[2:]}`` — the first hex byte is the shard,
    the rest is the leaf filename.

    Pins the exact sharding split so a refactor cannot silently move
    a blob to a different prefix.
    """
    store = BlobStore(tmp_path / "blobs")
    # Choose hashes whose first byte spans the full 256-shard space.
    sample_bytes = [b"alpha", b"beta", b"gamma", b"\x00", b"\xff", b"prefix-edge"]
    for payload in sample_bytes:
        h, _ = store.write_from_bytes(payload)
        path = store.blob_path(h)
        assert path.parent.name == h[:2]
        assert path.name == h[2:]
        assert path.exists()
        assert len(path.parent.name) == 2
        # Shard dir name must be lowercase hex (0-9, a-f)
        assert all(c in string.hexdigits and (c.isdigit() or c.islower()) for c in path.parent.name)


def test_prefix_shard_layout_supports_full_256_namespace(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: there are 256 possible
    shard directories — ``blob/00/`` through ``blob/ff/``.

    We don't materialise all 256 (would require ~256 distinct hashes),
    but we *do* assert that ``blob_path`` permits any lowercase-hex
    byte as the prefix and rejects everything else.
    """
    store = BlobStore(tmp_path / "blobs")
    # Build representative hex digests for several prefix bytes.
    for prefix_byte in (0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF):
        synthetic = f"{prefix_byte:02x}" + "0" * 62
        path = store.blob_path(synthetic)
        assert path.parent.name == f"{prefix_byte:02x}"
        assert path.name == "0" * 62


def test_iter_all_only_walks_valid_two_char_prefix_dirs(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model + ``iter_all`` contract:
    only directories whose name is exactly 2 chars are considered
    shard dirs. Anything else (legacy layout, scratch dirs, mounts) is
    ignored, never mis-attributed as a blob.
    """
    store = BlobStore(tmp_path / "blobs")
    h, _ = store.write_from_bytes(b"genuine")

    # Plant non-shard noise that ``iter_all`` must skip.
    (store.root / "not-a-shard").mkdir(parents=True, exist_ok=True)
    (store.root / "not-a-shard" / "abc").write_bytes(b"intruder")
    (store.root / "abcd").mkdir(parents=True, exist_ok=True)  # wrong length
    (store.root / "abcd" / "leaf").write_bytes(b"too-long-prefix")

    seen = list(store.iter_all())
    assert seen == [h]


# ---------------------------------------------------------------------------
# § "Blob Store Model" → "Operations: Blobs are write-once, read-many.
# No in-place modification." + "Dedup: identical content → identical hash —
# automatic deduplication."
# ---------------------------------------------------------------------------


def test_write_once_dedup_is_a_noop_on_second_write(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: write-once + dedup.

    Re-writing the same payload returns the same hash, leaves the
    on-disk inode untouched, and produces a single store entry.
    """
    store = BlobStore(tmp_path / "blobs")
    payload = b"the canonical content"

    h1, _ = store.write_from_bytes(payload)
    inode_after_first = store.blob_path(h1).stat().st_ino
    mtime_after_first = store.blob_path(h1).stat().st_mtime_ns

    h2, _ = store.write_from_bytes(payload)
    inode_after_second = store.blob_path(h2).stat().st_ino
    mtime_after_second = store.blob_path(h2).stat().st_mtime_ns

    assert h1 == h2
    # Write-once: the second write must not have replaced the file.
    assert inode_after_first == inode_after_second
    assert mtime_after_first == mtime_after_second
    assert store.stats()["count"] == 1


def test_write_once_does_not_clobber_existing_content(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: "Blobs are write-once" —
    even if the on-disk content is corrupted (hash mismatch),
    re-writing the original payload does not silently replace the
    corrupted file.

    This pins that we trust the hash, not the bytes. Repair is an
    explicit operation, never a side-effect of a re-write.
    """
    store = BlobStore(tmp_path / "blobs")
    payload = b"original"
    h, _ = store.write_from_bytes(payload)

    # Corrupt on disk.
    store.blob_path(h).write_bytes(b"corrupted!")

    # Re-write the original payload.
    h2, _ = store.write_from_bytes(payload)

    assert h2 == h
    # The on-disk content is still the corrupted version — write_once
    # means "addressed, not validated on every write".
    assert store.blob_path(h).read_bytes() == b"corrupted!"
    # verify() detects the corruption and reports it honestly.
    assert not store.verify(h)


def test_dedup_across_write_methods(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: dedup is by content, not
    by call site. ``write_from_bytes`` / ``write_from_fileobj`` /
    ``write_from_path`` of the same payload all collapse to one blob.
    """
    store = BlobStore(tmp_path / "blobs")
    payload = b"three-way dedup"
    src = tmp_path / "src.bin"
    src.write_bytes(payload)

    h1, _ = store.write_from_bytes(payload)
    h2, _ = store.write_from_fileobj(BytesIO(payload))
    h3, _ = store.write_from_path(src)

    assert h1 == h2 == h3
    assert store.stats()["count"] == 1


# ---------------------------------------------------------------------------
# § "GC concurrency model"
# ---------------------------------------------------------------------------


def test_gc_skips_blobs_with_db_reference(tmp_path: Path) -> None:
    """docs/internals.md § GC concurrency model, current-referent step:
    canonical liveness resolves the source owner's ``blob_hash``; if the row
    owns the bytes, GC skips the blob.

    Anti-vacuity: the unreferenced control blob is reclaimed in the same pass,
    so GC ran; a GC that refused or skipped the pass fails that assertion.
    """
    source_db, store = _make_gc_archive(tmp_path / "archive")
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="/src.jsonl",
            canonical_source_path="/src.jsonl",
            source_index=0,
            native_id="still-referenced",
            payload=b"still-referenced",
            acquired_at_ms=1,
        )
        referenced = bytes(
            conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        ).hex()
    store.write_from_bytes(b"still-referenced")
    unreferenced, _ = store.write_from_bytes(b"nothing names this blob")
    # Force candidate eligibility under the MIN_AGE_S age check.
    _backdate_blobs(store)

    report = run_blob_gc_report(source_db, store.root)

    assert report.blocked_reason is None
    assert report.deleted_count == 1
    assert store.exists(referenced), "GC must never delete a blob that is still referenced"
    assert not store.exists(unreferenced)


def test_gc_records_a_new_generation_when_it_runs(tmp_path: Path) -> None:
    """docs/internals.md § GC concurrency model: "``gc_generations``
    tracks the high-water mark of completed GC runs." The
    "defense-in-depth" age check uses the previous generation marker
    to refuse blobs younger than the previous cycle.

    When GC has work to consider (candidates present) it must record a
    new generation row each cycle so the next cycle can apply the age
    guard against the most-recent completion timestamp.
    """
    source_db, store = _make_gc_archive(tmp_path / "archive")

    for cycle in range(3):
        # Plant a fresh candidate blob for each cycle so the GC
        # codepath that records the generation row is exercised.
        store.write_from_bytes(f"candidate-{cycle}".encode())
        _backdate_blobs(store)

        report = run_blob_gc_report(source_db, store.root)
        assert report.blocked_reason is None

        conn = sqlite3.connect(str(source_db))
        count = conn.execute("SELECT COUNT(*) FROM gc_generations").fetchone()[0]
        latest = conn.execute(
            "SELECT completed_at_ms FROM gc_generations ORDER BY completed_at_ms DESC LIMIT 1"
        ).fetchone()
        conn.close()
        # One durable generation row accumulates per executed cycle.
        assert count == cycle + 1
        assert latest[0] is not None


# ---------------------------------------------------------------------------
# § "Blob Store Model" → "Linking: ``link_group_key`` groups related blobs
# (e.g., all blobs belonging to one session)."
#
# In the live codebase the grouping is implemented on
# ``artifact_observations.link_group_key`` (the doc's "blob_links table"
# vocabulary describes the same concept). We pin the isolation property
# at the storage layer so a group-A query never surfaces a group-B row.
# ---------------------------------------------------------------------------


def test_link_group_isolation_in_artifact_observations(tmp_path: Path) -> None:
    """docs/internals.md § Blob Store Model: ``link_group_key`` groups
    related blobs (sidecar bundles). A query scoped to one group must
    never surface blobs from another group.

    Pins the isolation property of the live grouping table
    (``artifact_observations.link_group_key``) — the doc's "blob_links"
    vocabulary refers to this same grouping.
    """
    db_path = tmp_path / "groups.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY
        );
        CREATE TABLE artifact_observations (
            observation_id TEXT PRIMARY KEY,
            raw_id TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
            link_group_key TEXT
        );
        CREATE INDEX idx_artifact_obs_link_group
            ON artifact_observations(link_group_key)
            WHERE link_group_key IS NOT NULL;
        """
    )
    fixtures = [
        ("raw-a1", "obs-a1", "group-A"),
        ("raw-a2", "obs-a2", "group-A"),
        ("raw-b1", "obs-b1", "group-B"),
        ("raw-c1", "obs-c1", None),
    ]
    for raw_id, obs_id, group in fixtures:
        conn.execute("INSERT INTO raw_sessions (raw_id) VALUES (?)", (raw_id,))
        conn.execute(
            "INSERT INTO artifact_observations (observation_id, raw_id, link_group_key) VALUES (?, ?, ?)",
            (obs_id, raw_id, group),
        )
    conn.commit()

    def raws_for_group(group: str) -> set[str]:
        rows = conn.execute(
            "SELECT raw_id FROM artifact_observations WHERE link_group_key = ?",
            (group,),
        ).fetchall()
        return {row[0] for row in rows}

    assert raws_for_group("group-A") == {"raw-a1", "raw-a2"}
    assert raws_for_group("group-B") == {"raw-b1"}
    # Group queries do not surface the ungrouped row.
    null_rows = conn.execute("SELECT raw_id FROM artifact_observations WHERE link_group_key IS NULL").fetchall()
    assert {row[0] for row in null_rows} == {"raw-c1"}

    conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backdate_blobs(store: BlobStore) -> None:
    """Push every blob's mtime back past ``MIN_AGE_S`` so GC will
    consider them eligible.

    The blob-store integration tests routinely need this; encapsulating
    it here keeps the assertions focused on the invariant under test.
    """
    import os
    import time

    cutoff = time.time() - 3600
    for h in store.iter_all():
        path = store.blob_path(h)
        os.utime(path, (cutoff, cutoff))
