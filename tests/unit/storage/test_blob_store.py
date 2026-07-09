from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

from polylogue.storage.blob_store import (
    BlobStore,
    BlobVerifyAllResult,
    BlobVerifyFailure,
)

# ---------------------------------------------------------------------------
# Write round-trip and dedup (existing)
# ---------------------------------------------------------------------------


def test_write_from_fileobj_round_trips_content(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"streamed blob content"

    blob_hash, blob_size = blob_store.write_from_fileobj(BytesIO(payload))

    assert blob_hash == hashlib.sha256(payload).hexdigest()
    assert blob_size == len(payload)
    assert blob_store.read_all(blob_hash) == payload


def test_write_from_fileobj_deduplicates_existing_blob(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"same payload"

    first_hash, first_size = blob_store.write_from_bytes(payload)
    second_hash, second_size = blob_store.write_from_fileobj(BytesIO(payload))

    assert second_hash == first_hash
    assert second_size == first_size == len(payload)
    assert blob_store.stats()["count"] == 1


def test_write_from_fileobj_invokes_heartbeat(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"x" * (2 * 1024 * 1024 + 1)

    beats = 0

    def heartbeat() -> None:
        nonlocal beats
        beats += 1

    blob_store.write_from_fileobj(BytesIO(payload), heartbeat=heartbeat)

    assert beats >= 2


# ---------------------------------------------------------------------------
# Dedup across different write methods
# ---------------------------------------------------------------------------


def test_write_from_bytes_deduplicates(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"dedup test"

    h1, s1 = blob_store.write_from_bytes(payload)
    h2, s2 = blob_store.write_from_bytes(payload)

    assert h1 == h2
    assert s1 == s2
    assert blob_store.stats()["count"] == 1


# ---------------------------------------------------------------------------
# Single-blob verify
# ---------------------------------------------------------------------------


def test_verify_existing_blob_passes(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h, _ = blob_store.write_from_bytes(b"verify me")
    assert blob_store.verify(h)


def test_verify_nonexistent_blob_fails(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    fake_hash = hashlib.sha256(b"nonexistent").hexdigest()
    assert not blob_store.verify(fake_hash)


def test_verify_corrupted_blob_fails(tmp_path: Path) -> None:
    """A blob whose on-disk content has been altered must fail verification."""
    blob_store = BlobStore(tmp_path / "blobs")
    h, _ = blob_store.write_from_bytes(b"original content")
    path = blob_store.blob_path(h)
    # Corrupt the file
    path.write_bytes(b"corrupted!!!")
    assert not blob_store.verify(h)


# ---------------------------------------------------------------------------
# verify_all — batch integrity
# ---------------------------------------------------------------------------


def test_verify_all_empty_store(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    result = blob_store.verify_all()
    assert result.passed
    assert result.checked == 0
    assert result.failed_count == 0
    assert not result.truncated


def test_verify_all_all_pass(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.write_from_bytes(b"alpha")
    blob_store.write_from_bytes(b"beta")
    blob_store.write_from_bytes(b"gamma")

    result = blob_store.verify_all()
    assert result.passed
    assert result.checked == 3
    assert result.failed_count == 0
    assert not result.truncated
    assert result.checked_bytes > 0


def test_verify_all_detects_hash_mismatch(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h1, _ = blob_store.write_from_bytes(b"good blob one")
    h2, _ = blob_store.write_from_bytes(b"good blob two")

    # Corrupt the second blob
    blob_store.blob_path(h2).write_bytes(b"tampered content!")

    result = blob_store.verify_all()
    assert not result.passed
    assert result.checked == 2
    assert result.failed_count == 1
    assert result.failures[0].hash == h2
    assert result.failures[0].reason == "hash_mismatch"


def test_verify_all_stops_at_max_failures(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    hashes = []
    for i in range(5):
        h, _ = blob_store.write_from_bytes(f"payload {i}".encode())
        hashes.append(h)

    # Corrupt only the first blob
    blob_store.blob_path(hashes[0]).write_bytes(b"bad")

    result = blob_store.verify_all(max_failures=2)
    # Should not be truncated — only 1 failure, under max
    assert not result.truncated

    # Now corrupt 3 blobs, set max_failures to 2
    for h in hashes[:3]:
        blob_store.blob_path(h).write_bytes(b"corrupted")
    result = blob_store.verify_all(max_failures=2)
    assert result.truncated
    assert result.failed_count == 2


def test_verify_all_handles_removed_prefix_dir(tmp_path: Path) -> None:
    """Blob files in a prefix dir that was removed between checks."""
    blob_store = BlobStore(tmp_path / "blobs")
    h, _ = blob_store.write_from_bytes(b"another blob")
    prefix_dir = blob_store.blob_path(h).parent
    blob_store.blob_path(h).unlink()
    os.rmdir(prefix_dir)
    result = blob_store.verify_all()
    # The prefix dir is gone, so iter_all yields nothing
    assert result.checked == 0


# ---------------------------------------------------------------------------
# detect_orphans
# ---------------------------------------------------------------------------


def test_detect_orphans_none_when_all_referenced(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h1, _ = blob_store.write_from_bytes(b"a")
    h2, _ = blob_store.write_from_bytes(b"b")

    result = blob_store.detect_orphans({h1, h2})
    assert result.orphan_count == 0
    assert result.orphan_bytes == 0
    assert result.orphan_samples == ()


def test_detect_orphans_finds_unreferenced(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h1, _ = blob_store.write_from_bytes(b"referenced")
    h2, _ = blob_store.write_from_bytes(b"orphan")

    # Only h1 is in the DB
    result = blob_store.detect_orphans({h1})
    assert result.orphan_count == 1
    assert result.orphan_bytes > 0
    assert result.orphan_samples == (h2,)


def test_detect_orphans_all_orphaned(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.write_from_bytes(b"x")
    blob_store.write_from_bytes(b"y")

    # Empty DB — all blobs are orphans
    result = blob_store.detect_orphans(set())
    assert result.orphan_count == 2
    assert result.orphan_bytes > 0
    assert len(result.orphan_samples) == 2


def test_detect_orphans_respects_max_sample(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    for i in range(20):
        blob_store.write_from_bytes(f"payload-{i}".encode())

    result = blob_store.detect_orphans(set(), max_sample=5)
    assert result.orphan_count == 20
    assert len(result.orphan_samples) == 5


# ---------------------------------------------------------------------------
# cleanup_orphans — dry-run
# ---------------------------------------------------------------------------


def test_cleanup_orphans_dry_run_reports_what_would_be_deleted(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h1, _ = blob_store.write_from_bytes(b"keep me")
    h2, _ = blob_store.write_from_bytes(b"delete me")

    # dry_run=True (default)
    result = blob_store.cleanup_orphans({h2})
    assert result.dry_run is True
    assert result.deleted_count == 0
    assert result.deleted_bytes == 0
    assert result.would_delete_count == 1
    assert result.would_delete_bytes > 0
    # Files are still on disk
    assert blob_store.exists(h1)
    assert blob_store.exists(h2)


def test_cleanup_orphans_dry_run_empty_set(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    result = blob_store.cleanup_orphans(set(), dry_run=True)
    assert result.would_delete_count == 0
    assert result.would_delete_bytes == 0


def test_cleanup_orphans_dry_run_nonexistent_hash(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    fake_hash = hashlib.sha256(b"ghost").hexdigest()
    result = blob_store.cleanup_orphans({fake_hash}, dry_run=True)
    # Hash is valid format but file doesn't exist
    assert result.would_delete_count == 0


def test_cleanup_orphans_dry_run_skips_invalid_hash(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    # Invalid hash format
    result = blob_store.cleanup_orphans({"not-a-valid-hex-!@#$"}, dry_run=True)
    assert result.would_delete_count == 0


# ---------------------------------------------------------------------------
# cleanup_orphans — apply
# ---------------------------------------------------------------------------


def test_cleanup_orphans_apply_deletes_blobs(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h1, s1 = blob_store.write_from_bytes(b"keep me")
    h2, s2 = blob_store.write_from_bytes(b"delete me")

    result = blob_store.cleanup_orphans({h2}, dry_run=False)
    assert result.dry_run is False
    assert result.deleted_count == 1
    assert result.deleted_bytes == s2
    assert result.errors == 0

    # h1 still there, h2 gone
    assert blob_store.exists(h1)
    assert not blob_store.exists(h2)


def test_cleanup_orphans_apply_multiple(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    hashes = {}
    for i in range(5):
        h, s = blob_store.write_from_bytes(f"blob-{i}".encode())
        hashes[h] = s

    to_delete = set(list(hashes.keys())[:3])
    result = blob_store.cleanup_orphans(to_delete, dry_run=False)
    assert result.deleted_count == 3
    assert result.errors == 0

    for h in to_delete:
        assert not blob_store.exists(h)


def test_cleanup_orphans_apply_nonexistent_is_noop(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    fake_hash = hashlib.sha256(b"ghost").hexdigest()
    result = blob_store.cleanup_orphans({fake_hash}, dry_run=False)
    assert result.deleted_count == 0
    assert result.errors == 0


def test_cleanup_orphans_apply_invalid_hash_reported_as_error(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    result = blob_store.cleanup_orphans({"bad!!!hex"}, dry_run=False)
    assert result.deleted_count == 0
    assert result.errors == 1
    assert len(result.error_details) == 1


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_existing_blob(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    h, _ = blob_store.write_from_bytes(b"transient")
    assert blob_store.exists(h)
    assert blob_store.remove(h)
    assert not blob_store.exists(h)


def test_remove_nonexistent_blob(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    fake = hashlib.sha256(b"never-written").hexdigest()
    assert not blob_store.remove(fake)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_empty_store(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    s = blob_store.stats()
    assert s["count"] == 0
    assert s["total_bytes"] == 0


def test_stats_with_blobs(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.write_from_bytes(b"alpha")
    blob_store.write_from_bytes(b"beta")
    s = blob_store.stats()
    assert s["count"] == 2
    assert s["total_bytes"] == len(b"alpha") + len(b"beta")


# ---------------------------------------------------------------------------
# iter_all
# ---------------------------------------------------------------------------


def test_iter_all_skips_temp_files(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.write_from_bytes(b"real blob")

    # Create a fake temp file (like ones during writes)
    prefix_dir = blob_store.root / "ab"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    temp_file = prefix_dir / ".blob.temp123"
    temp_file.write_bytes(b"temp content")

    hashes = list(blob_store.iter_all())
    assert len(hashes) == 1  # Only the real blob, not .blob.* temp


def test_iter_all_skips_non_prefix_dirs(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.write_from_bytes(b"real blob")

    # Create a non-prefix directory (name not 2 chars)
    (blob_store.root / "not-a-prefix-dir").mkdir(parents=True, exist_ok=True)

    hashes = list(blob_store.iter_all())
    assert len(hashes) == 1


# ---------------------------------------------------------------------------
# blob_path validation
# ---------------------------------------------------------------------------


def test_blob_path_rejects_non_hex(tmp_path: Path) -> None:
    blob = BlobStore(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="invalid blob hash"):
        blob.blob_path("not hex!!")


def test_blob_path_rejects_uppercase(tmp_path: Path) -> None:
    blob = BlobStore(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="invalid blob hash"):
        blob.blob_path("AABBCCDDEEFF0011223344556677889900AABBCCDD")


def test_blob_path_rejects_truncated_hash(tmp_path: Path) -> None:
    """jsy: a 63-char (one short of a real SHA-256 digest) hash must be
    rejected, not silently accepted as if length didn't matter."""
    blob = BlobStore(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="invalid blob hash"):
        blob.blob_path("a" * 63)


def test_blob_path_rejects_over_long_hash(tmp_path: Path) -> None:
    """jsy: a 65-char hash (one over) must be rejected, not truncated/accepted."""
    blob = BlobStore(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="invalid blob hash"):
        blob.blob_path("a" * 65)


def test_blob_path_rejects_trailing_newline(tmp_path: Path) -> None:
    """jsy: the former `^[0-9a-f]+$` pattern (via .match, not .fullmatch)
    accepted a trailing newline since bare `$` matches just before one --
    fullmatch on a fixed-length pattern must reject it."""
    blob = BlobStore(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="invalid blob hash"):
        blob.blob_path("a" * 64 + "\n")


def test_blob_path_accepts_exactly_64_hex_chars(tmp_path: Path) -> None:
    blob = BlobStore(tmp_path)
    path = blob.blob_path("a" * 64)
    assert path == tmp_path / "aa" / ("a" * 62)


# ---------------------------------------------------------------------------
# Content-addressing invariant
# ---------------------------------------------------------------------------


def test_identical_content_same_hash_across_methods(tmp_path: Path) -> None:
    """Same content produces same hash regardless of write method."""
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"identical across methods"

    h1, _ = blob_store.write_from_bytes(payload)
    h2, _ = blob_store.write_from_fileobj(BytesIO(payload))

    assert h1 == h2
    assert blob_store.stats()["count"] == 1


# ---------------------------------------------------------------------------
# Result dataclass properties
# ---------------------------------------------------------------------------


def test_blob_verify_all_result_passed_property() -> None:
    r = BlobVerifyAllResult(checked=10, checked_bytes=100, failures=(), truncated=False)
    assert r.passed
    assert r.failed_count == 0


def test_blob_verify_all_result_failed_property() -> None:
    f = BlobVerifyFailure(hash="aabb", reason="hash_mismatch")
    r = BlobVerifyAllResult(checked=10, checked_bytes=100, failures=(f,), truncated=False)
    assert not r.passed
    assert r.failed_count == 1
