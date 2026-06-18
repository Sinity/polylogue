"""Property tests for ``BlobStore`` orphan detection and cleanup (#818-A1).

Pins behavioural invariants of the orphan-detection contract that
unit tests cover only by example. Two key invariants:

1. Sample size is strictly bounded by ``max_sample``. The detection
   API must never let a pathological orphan count blow up the
   result payload.
2. Counts and bytes report exactly what is on disk minus what the DB
   knows about — independent of how the two sets are constructed.

Operating against a real ``BlobStore`` (under ``tmp_path``) keeps the
tests honest: any silent change to the iteration order or bookkeeping
will fail here even if the unit tests still pass.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.blob_store import BlobStore

_HEALTH_SUPPRESS = [
    HealthCheck.function_scoped_fixture,
    HealthCheck.too_slow,
]


def _store(tmp_path: Path) -> BlobStore:
    """Return a fresh BlobStore rooted under tmp_path."""
    root = tmp_path / "blob"
    root.mkdir(parents=True, exist_ok=True)
    return BlobStore(root)


@given(
    referenced=st.integers(min_value=0, max_value=20),
    orphans=st.integers(min_value=0, max_value=80),
    max_sample=st.integers(min_value=1, max_value=20),
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=_HEALTH_SUPPRESS,
)
def test_detect_orphans_count_and_bytes_match_disk_minus_db(
    tmp_path_factory: object, referenced: int, orphans: int, max_sample: int
) -> None:
    """For any (referenced, orphan) split, detect_orphans returns the right
    count and a bounded sample.
    """
    # Hypothesis runs many examples — give each its own tmp path so blobs
    # don't collide across runs.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store(Path(tmpdir))

        ref_hashes: set[str] = set()
        orphan_hashes: set[str] = set()
        orphan_bytes_total = 0

        for i in range(referenced):
            payload = f"ref-{i}".encode()
            h, _ = store.write_from_bytes(payload)
            ref_hashes.add(h)

        for j in range(orphans):
            payload = f"orphan-{j}".encode() + b"\0" * j  # vary size
            h, n = store.write_from_bytes(payload)
            # Skip collisions with referenced set (degenerate cases).
            if h in ref_hashes:
                continue
            orphan_hashes.add(h)
            orphan_bytes_total += n

        result = store.detect_orphans(ref_hashes, max_sample=max_sample)

        # Invariant 1: count matches the orphan set exactly.
        assert result.orphan_count == len(orphan_hashes)

        # Invariant 2: bytes match the orphan-set total byte size.
        assert result.orphan_bytes == orphan_bytes_total

        # Invariant 3: sample is bounded by max_sample.
        assert len(result.orphan_samples) <= max_sample

        # Invariant 4: every sampled hash is genuinely an orphan
        # (in the orphan set, not in the referenced set).
        sampled = set(result.orphan_samples)
        assert sampled.issubset(orphan_hashes)
        assert sampled.isdisjoint(ref_hashes)


@given(orphans=st.integers(min_value=1, max_value=30))
@settings(max_examples=20, deadline=None, suppress_health_check=_HEALTH_SUPPRESS)
def test_cleanup_dry_run_never_deletes(orphans: int) -> None:
    """``cleanup_orphans(dry_run=True)`` must always leave every blob on disk.

    This is the safety boundary for ``polylogue ops doctor`` previews.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store(Path(tmpdir))

        hashes: set[str] = set()
        for i in range(orphans):
            h, _ = store.write_from_bytes(f"orphan-{i}".encode())
            hashes.add(h)

        result = store.cleanup_orphans(hashes, dry_run=True)

        assert result.dry_run is True
        assert result.deleted_count == 0
        assert result.deleted_bytes == 0
        # would_delete_count must equal what's actually on disk for the
        # provided set (caller passed a clean set, not corrupted hashes).
        assert result.would_delete_count == len(hashes)
        # All blobs still on disk.
        for h in hashes:
            assert store.exists(h) is True
