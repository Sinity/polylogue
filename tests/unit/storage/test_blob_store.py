from __future__ import annotations

import hashlib
import os
import stat
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import (
    BlobNamespaceEntryKind,
    BlobNamespaceIssue,
    BlobStore,
    BlobVerifyAllResult,
    BlobVerifyFailure,
    get_blob_store,
    reset_blob_store,
)

# ---------------------------------------------------------------------------
# Write round-trip and dedup (existing)
# ---------------------------------------------------------------------------


def test_publish_fsyncs_staged_file_before_close_and_shard_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blob bytes and their shard entry reach stable storage in order.

    Anti-vacuity: removing either the staged-file ``fsync`` or the
    post-replace shard-directory ``fsync`` makes the corresponding ordering
    assertion fail.
    """
    blob_store = BlobStore(tmp_path / "blobs")
    events: list[tuple[str, int | None]] = []
    real_close = os.close
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_close(fd: int) -> None:
        events.append(("close", fd))
        real_close(fd)

    def recording_fsync(fd: int) -> None:
        events.append(("fsync", fd))
        real_fsync(fd)

    def recording_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        real_replace(source, destination)
        events.append(("replace", None))

    monkeypatch.setattr(os, "close", recording_close)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)

    blob_store.write_from_bytes(b"durable blob")

    file_fsync_index = next(index for index, event in enumerate(events) if event[0] == "fsync")
    file_fsync_fd = events[file_fsync_index][1]
    assert file_fsync_fd is not None
    assert events[file_fsync_index + 1] == ("close", file_fsync_fd)

    replace_index = next(index for index, event in enumerate(events) if event[0] == "replace")
    directory_fsync_index = next(
        index for index in range(replace_index + 1, len(events)) if events[index][0] == "fsync"
    )
    directory_fsync_fd = events[directory_fsync_index][1]
    assert directory_fsync_fd is not None
    assert directory_fsync_index > replace_index
    assert events[directory_fsync_index + 1] == ("close", directory_fsync_fd)


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


def test_verify_all_reports_file_backed_root(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    root.write_bytes(b"not a directory")

    result = BlobStore(root).verify_all()

    assert result.checked == 0
    assert [(failure.reason, failure.path, failure.detail) for failure in result.failures] == [
        ("invalid_namespace_entry", ".", ".: stat_failed")
    ]


def test_verify_all_reports_malformed_leaf_without_hash_path_parsing(tmp_path: Path) -> None:
    """A malformed leaf is a namespace fault, not a malformed hash exception."""
    blob_store = BlobStore(tmp_path / "blobs")
    valid_hash, _ = blob_store.write_from_bytes(b"valid")
    shard = blob_store.root / valid_hash[:2]
    malformed_leaf = shard / f"{valid_hash[2:]}-wal"
    malformed_leaf.write_bytes(b"sqlite wal sidecar")

    entries = tuple(blob_store.iter_namespace())
    assert [(entry.kind, entry.relative_path, entry.issue) for entry in entries] == [
        (BlobNamespaceEntryKind.BLOB, f"{valid_hash[:2]}/{valid_hash[2:]}", None),
        (
            BlobNamespaceEntryKind.INVALID_SHARD_ENTRY,
            f"{valid_hash[:2]}/{valid_hash[2:]}-wal",
            BlobNamespaceIssue.INVALID_LEAF_NAME,
        ),
    ]

    result = blob_store.verify_all()

    assert result.checked == 1
    assert result.failed_count == 1
    assert result.failures[0].hash == ""
    assert result.failures[0].reason == "invalid_namespace_entry"
    assert result.failures[0].path == f"{valid_hash[:2]}/{valid_hash[2:]}-wal"
    assert result.failures[0].detail.endswith("invalid_leaf_name")


def test_verify_all_rejects_foreign_sqlite_sidecars_at_namespace_root(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_hash, _ = blob_store.write_from_bytes(b"valid")
    for suffix in ("-wal", "-shm"):
        (blob_store.root / f"{blob_hash}{suffix}").write_bytes(b"sidecar")

    result = blob_store.verify_all()

    assert result.checked == 1
    assert [(failure.path, failure.detail) for failure in result.failures] == [
        (f"{blob_hash}-shm", f"{blob_hash}-shm: invalid_shard_name"),
        (f"{blob_hash}-wal", f"{blob_hash}-wal: invalid_shard_name"),
    ]


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


def test_pending_publication_is_noncanonical_and_recovers_to_pristine_namespace(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_hash, _ = blob_store.write_from_bytes(b"real blob")
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", blob_store.root)
    pending_hash, _ = publisher.write_from_bytes(b"interrupted write")
    pending_path = publisher.blob_path(pending_hash)
    try:
        entries = tuple(blob_store.iter_namespace())
        verified = blob_store.verify_all()

        assert pending_path.parent == blob_store.staging_root
        assert pending_path.exists()
        assert [(entry.relative_path, entry.issue) for entry in entries] == [
            (f".staging/{pending_path.name}", BlobNamespaceIssue.STAGED_WORK_FILE),
            (blob_hash[:2] + "/" + blob_hash[2:], None),
        ]
        assert not verified.passed
        assert [(failure.path, failure.reason) for failure in verified.failures] == [
            (f".staging/{pending_path.name}", "invalid_namespace_entry")
        ]
        assert list(blob_store.iter_all()) == [blob_hash]
    finally:
        publisher.discard_pending()
    assert not pending_path.exists()
    assert blob_store.staging_root.is_dir()
    assert not tuple(blob_store.staging_root.iterdir())
    assert blob_store.verify_all().passed


def test_blob_staging_is_owner_only(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.staging_root.mkdir(parents=True, mode=0o755)
    blob_store.staging_root.chmod(0o755)

    prepared = blob_store.prepare_from_bytes(b"private staging")
    try:
        assert stat.S_IMODE(blob_store.staging_root.stat().st_mode) == 0o700
        assert stat.S_IMODE(prepared.temporary_path.stat().st_mode) == 0o600
    finally:
        blob_store.discard_prepared(prepared)


def test_blob_staging_rejects_a_preexisting_symlink(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    blob_store.root.mkdir()
    blob_store.staging_root.symlink_to(redirect, target_is_directory=True)

    with pytest.raises(RuntimeError, match="not a private directory"):
        blob_store.prepare_from_bytes(b"must not escape")

    assert not tuple(redirect.iterdir())


def test_blob_staging_rejects_a_preexisting_file(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    blob_store.root.mkdir()
    blob_store.staging_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a private directory"):
        blob_store.allocate_staging_path(prefix="snapshot-", suffix=".db")


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


# ---------------------------------------------------------------------------
# get_blob_store() lazy-singleton thread safety (polylogue-xikl.2)
# ---------------------------------------------------------------------------


def test_get_blob_store_singleton_is_race_safe_under_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent first access must not construct duplicate BlobStore instances.

    ``get_blob_store()``'s check-then-set used to be unguarded; the daemon's
    real ``archive_query_executor`` already dispatches concurrent request
    handlers onto real OS threads today, so N threads racing the very first
    ``get_blob_store()`` call is a live scenario, not a hypothetical one. A
    short delay is injected into ``BlobStore.__init__`` to force the
    interleaving window open regardless of scheduling luck: before the fix
    this reliably produced multiple distinct instances (last writer wins);
    with the lock guarding the whole check-then-construct section, exactly
    one thread ever constructs the instance.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    reset_blob_store()

    original_init = BlobStore.__init__

    def delayed_init(self: BlobStore, *args: object, **kwargs: object) -> None:
        time.sleep(0.02)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BlobStore, "__init__", delayed_init)

    results: list[BlobStore] = []
    results_lock = threading.Lock()

    def worker() -> None:
        store = get_blob_store()
        with results_lock:
            results.append(store)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    try:
        assert len(results) == 8
        assert len({id(store) for store in results}) == 1, (
            "concurrent first access constructed more than one BlobStore instance"
        )
    finally:
        reset_blob_store()


def test_new_shard_entry_persists_blob_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a shard must persist its entry in the blob root, not only inside it.

    ``publish_prepared`` fsynced ``dest.parent`` only. That persists the entries
    *inside* the shard; the shard's own directory entry in the blob root stayed
    unpersisted, so a power loss after the durable source-db reference commit
    could take the whole new shard with it while publication had reported
    success.

    The assertion observes the ``os.fsync`` calls the production route actually
    makes and resolves each descriptor back to a path, so it does not depend on
    any private helper existing.

    Anti-vacuity: drop the root fsync and the first assertion fails -- the root
    never appears among the fsynced directories. The second assertion is the
    opposite direction: republishing into an existing shard must not fsync the
    root again, so a blanket "always fsync the root" is refuted too.
    """
    store = BlobStore(tmp_path / "blob")
    fsynced: list[Path] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        try:
            fsynced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        except OSError:
            pass
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)

    store.write_from_bytes(b"first blob in a fresh shard")
    root = (tmp_path / "blob").resolve()
    assert root in fsynced, f"blob root was never persisted; fsynced={fsynced}"

    fsynced.clear()
    # A second blob whose hash shares the first one's shard prefix is hard to
    # arrange; republishing the same bytes into the now-existing shard is the
    # reachable "shard already exists" case and must not re-persist the root.
    store.write_from_bytes(b"first blob in a fresh shard")
    assert root not in fsynced


def _payload_in_shard(shard: str, *, seed: int) -> bytes:
    """Find deterministic bytes whose SHA-256 begins with *shard*.

    Two blobs land in the same shard directory only when their digests share a
    two-hex prefix, so the batch-durability law below cannot be written from
    arbitrary content. Brute force is ~256 attempts per hit and deterministic.
    """
    counter = seed
    while True:
        candidate = f"polylogue-rk0it shard probe {counter}".encode()
        if hashlib.sha256(candidate).hexdigest().startswith(shard):
            return candidate
        counter += 1


def _directory_fsync_recorder(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the path of every directory ``os.fsync`` the production route makes."""
    recorded: list[Path] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        try:
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            target = None
        if target is not None and target.is_dir():
            recorded.append(target)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)
    return recorded


def test_publish_many_persists_each_shard_once_per_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A published batch pays one directory fsync per touched shard, not per blob.

    ``publish_many`` was a plain loop over ``publish_prepared``, so four blobs
    sharing one hash prefix fsynced the same shard directory four times for one
    directory's worth of durability (polylogue-rk0it AC5). The durability
    boundary is the batch: every touched shard, and the root when the batch
    created a shard, is persisted before ``publish_many`` returns.

    Anti-vacuity, both directions:
      * restore the per-blob loop and the shard count becomes 4, not 1;
      * drop the fsyncs entirely and the ``== 1`` assertions become ``== 0``.
    Durability is pinned separately: every blob is readable at its final path
    after the call, so a batch that skipped the ``os.replace`` cannot pass.
    """
    store = BlobStore(tmp_path / "blob")
    root = (tmp_path / "blob").resolve()
    shard = "a1"
    payloads = []
    seed = 0
    for _ in range(4):
        payload = _payload_in_shard(shard, seed=seed)
        payloads.append(payload)
        seed = int(payload.rsplit(b" ", 1)[1]) + 1
    assert len({hashlib.sha256(p).hexdigest() for p in payloads}) == 4

    prepared = [store.prepare_from_bytes(payload) for payload in payloads]
    fsynced = _directory_fsync_recorder(monkeypatch)
    published = store.publish_many(prepared)

    shard_directory = (root / shard).resolve()
    assert fsynced.count(shard_directory) == 1, (
        f"one shard needs one fsync per batch, saw {fsynced.count(shard_directory)}; fsynced={fsynced}"
    )
    assert fsynced.count(root) == 1, f"the new shard's own root entry must be persisted once; fsynced={fsynced}"
    assert len(fsynced) == 2, f"a four-blob single-shard batch persisted {len(fsynced)} directories: {fsynced}"

    assert len(published) == 4
    for payload, (hash_hex, size) in zip(payloads, published, strict=True):
        assert hash_hex == hashlib.sha256(payload).hexdigest()
        assert size == len(payload)
        assert store.blob_path(hash_hex).read_bytes() == payload


def test_publish_many_persists_every_shard_it_touches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deduplicating the fsync must not drop a shard the batch actually wrote.

    Anti-vacuity: fsync only the first shard (or only the root) and the missing
    directory's ``in`` assertion fails. This is the opposite direction from the
    count law above -- a blanket "persist nothing twice" that persisted only one
    directory would pass that test and fail this one.
    """
    store = BlobStore(tmp_path / "blob")
    root = (tmp_path / "blob").resolve()
    payloads = [_payload_in_shard("b2", seed=0), _payload_in_shard("c3", seed=0)]
    prepared = [store.prepare_from_bytes(payload) for payload in payloads]

    fsynced = _directory_fsync_recorder(monkeypatch)
    store.publish_many(prepared)

    assert (root / "b2").resolve() in fsynced
    assert (root / "c3").resolve() in fsynced
    assert fsynced.count(root) == 1, f"two new shards still persist the root once; fsynced={fsynced}"


def test_publish_many_existing_shards_leave_root_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch that creates no shard leaves the blob root alone.

    Anti-vacuity: replace the batch's ``shard_created`` bookkeeping with an
    unconditional root fsync and this goes red, so the cheap "always persist the
    root" shortcut is refused.
    """
    store = BlobStore(tmp_path / "blob")
    root = (tmp_path / "blob").resolve()
    first = _payload_in_shard("d4", seed=0)
    second = _payload_in_shard("d4", seed=int(first.rsplit(b" ", 1)[1]) + 1)
    store.publish_many([store.prepare_from_bytes(first)])

    prepared = [store.prepare_from_bytes(second)]
    fsynced = _directory_fsync_recorder(monkeypatch)
    store.publish_many(prepared)

    assert (root / "d4").resolve() in fsynced
    assert root not in fsynced, f"an existing-shard batch re-persisted the root; fsynced={fsynced}"
