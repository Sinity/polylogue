"""Content-addressed blob store on the local filesystem.

Blobs are stored as immutable files under a two-level directory structure:
``{root}/{hash[:2]}/{hash[2:]}``, where hash is the SHA-256 hex digest of
the content. This is the same hash used as ``raw_id`` in the
``raw_sessions`` table — no separate addressing scheme needed.

Writes are atomic (tempfile + ``os.replace``). Files are never modified
after creation. Deduplication is free: identical content produces the
same hash, so the second write is a no-op.

The primary motivation is to avoid loading multi-GB files into Python
memory. ``write_from_path`` streams the file in 1 MiB chunks, hashing
as it goes, then copies to the store — peak memory is one chunk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Valid blob hash: lowercase hex only
_VALID_HEX = re.compile(r"^[0-9a-f]+$")

Heartbeat = Callable[[], None]


def _write_all(fd: int, data: bytes) -> None:
    """Write all *data* to *fd*, retrying on partial writes."""
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written == 0:
            raise OSError("write() returned 0 — possible disk full or closed fd")
        offset += written


class BlobStore:
    """Content-addressed blob store backed by the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def blob_path(self, hash_hex: str) -> Path:
        """Return the filesystem path for a blob by its hex digest."""
        if not _VALID_HEX.match(hash_hex):
            raise ValueError(f"invalid blob hash: {hash_hex!r} — expected lowercase hex string")
        return self.root / hash_hex[:2] / hash_hex[2:]

    def exists(self, hash_hex: str) -> bool:
        """Check whether a blob exists on disk."""
        return self.blob_path(hash_hex).exists()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_from_path(
        self,
        source: Path,
        *,
        heartbeat: Heartbeat | None = None,
    ) -> tuple[str, int]:
        """Stream-hash a file and copy it to the store.

        Reads the source in 1 MiB chunks — never loads the full file into
        Python memory. Returns ``(sha256_hex, byte_count)``.

        If a blob with the same hash already exists, the write is skipped
        (content-addressed deduplication).
        """
        # Single-pass: hash and write to temp file simultaneously. The temp
        # lives at the blob-store root so the final ``os.replace`` to the
        # sharded ``aa/bb/...`` destination stays on the same filesystem.
        # Ensure the root exists; in a fresh archive (and in tests) it has
        # not been created yet, and ``tempfile.mkstemp`` would raise
        # ``FileNotFoundError`` — which the source-acquisition layer would
        # then mis-attribute as a TOCTOU race against the source file.
        self.root.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path: str | None = None
        try:
            hasher = hashlib.sha256()
            size = 0
            fd, tmp_path = tempfile.mkstemp(dir=self.root, prefix=".blob.")
            with open(source, "rb") as src:
                while True:
                    chunk = src.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    _write_all(fd, chunk)
                    size += len(chunk)
                    if heartbeat is not None:
                        with suppress(Exception):
                            heartbeat()
            os.close(fd)
            fd = None

            hash_hex = hasher.hexdigest()
            dest = self.blob_path(hash_hex)

            if dest.exists():
                os.unlink(tmp_path)
                tmp_path = None
                return hash_hex, size

            dest.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return hash_hex, size

    def write_from_fileobj(
        self,
        source: IO[bytes],
        *,
        heartbeat: Heartbeat | None = None,
    ) -> tuple[str, int]:
        """Stream-hash an open binary file-like object into the store.

        Reads from ``source`` in 1 MiB chunks, hashing and writing to a
        temporary file in one pass. Returns ``(sha256_hex, byte_count)``.
        """
        hasher = hashlib.sha256()
        size = 0
        self.root.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self.root, prefix=".blob.")
            while True:
                chunk = source.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
                _write_all(fd, chunk)
                if heartbeat is not None:
                    with suppress(Exception):
                        heartbeat()

            hash_hex = hasher.hexdigest()
            dest = self.blob_path(hash_hex)
            if dest.exists():
                os.close(fd)
                fd = None
                os.unlink(tmp_path)
                tmp_path = None
                return hash_hex, size

            dest.parent.mkdir(parents=True, exist_ok=True)
            os.close(fd)
            fd = None
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return hash_hex, size

    def write_from_bytes(self, data: bytes) -> tuple[str, int]:
        """Hash in-memory bytes and write to the store.

        Returns ``(sha256_hex, len(data))``. Skips write if blob exists.
        """
        hash_hex = hashlib.sha256(data).hexdigest()
        dest = self.blob_path(hash_hex)

        if dest.exists():
            return hash_hex, len(data)

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dest.parent, prefix=".blob.")
            _write_all(fd, data)
            os.close(fd)
            fd = None
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return hash_hex, len(data)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def open(self, hash_hex: str) -> BinaryIO:
        """Open a blob for reading. Caller must close the handle."""
        path = self.blob_path(hash_hex)
        return builtins_open(path, "rb")

    def read_prefix(self, hash_hex: str, n: int = 65536) -> bytes:
        """Read the first *n* bytes of a blob."""
        path = self.blob_path(hash_hex)
        with builtins_open(path, "rb") as f:
            return f.read(n)

    def read_all(self, hash_hex: str) -> bytes:
        """Read the full blob content. Use for small blobs only."""
        return self.blob_path(hash_hex).read_bytes()

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify(self, hash_hex: str) -> bool:
        """Re-hash the blob on disk and verify it matches the expected hash."""
        path = self.blob_path(hash_hex)
        if not path.exists():
            return False
        hasher = hashlib.sha256()
        with builtins_open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest() == hash_hex

    def iter_all(self) -> Iterator[str]:
        """Yield all blob hashes present on disk."""
        if not self.root.exists():
            return
        for prefix_dir in sorted(self.root.iterdir()):
            if not prefix_dir.is_dir() or len(prefix_dir.name) != 2:
                continue
            for blob_file in sorted(prefix_dir.iterdir()):
                if blob_file.is_file() and not blob_file.name.startswith(".blob."):
                    yield prefix_dir.name + blob_file.name

    def remove(self, hash_hex: str) -> bool:
        """Remove a blob from the store. Returns True if it existed."""
        path = self.blob_path(hash_hex)
        if path.exists():
            path.unlink()
            return True
        return False

    def stats(self) -> dict[str, int]:
        """Return blob store statistics."""
        count = 0
        total_bytes = 0
        for hash_hex in self.iter_all():
            count += 1
            total_bytes += self.blob_path(hash_hex).stat().st_size
        return {"count": count, "total_bytes": total_bytes}

    # ------------------------------------------------------------------
    # Batch integrity and maintenance
    # ------------------------------------------------------------------

    def verify_all(
        self,
        *,
        max_failures: int = 10,
        heartbeat: Heartbeat | None = None,
    ) -> BlobVerifyAllResult:
        """Re-hash every blob on disk and verify content integrity.

        Reads and hashes each blob in 1 MiB chunks. Stops after
        *max_failures* failures to keep output bounded. Returns a
        summary with count, bytes checked, and failure details.

        This is intentionally a filesystem-only integrity check — it
        does not consult the database.  That keeps it usable even when
        the database is unavailable or corrupted.
        """
        checked = 0
        checked_bytes = 0
        failures: list[BlobVerifyFailure] = []

        for hash_hex in self.iter_all():
            checked += 1
            path = self.blob_path(hash_hex)
            try:
                file_size = path.stat().st_size
                checked_bytes += file_size
            except OSError:
                failures.append(BlobVerifyFailure(hash=hash_hex, reason="stat_failed"))
                if len(failures) >= max_failures:
                    break
                continue

            hasher = hashlib.sha256()
            try:
                with builtins_open(path, "rb") as f:
                    while True:
                        chunk = f.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        hasher.update(chunk)
            except OSError as exc:
                failures.append(BlobVerifyFailure(hash=hash_hex, reason="read_error", detail=str(exc)))
                if len(failures) >= max_failures:
                    break
                continue

            actual = hasher.hexdigest()
            if actual != hash_hex:
                failures.append(
                    BlobVerifyFailure(
                        hash=hash_hex,
                        reason="hash_mismatch",
                        detail=f"expected {hash_hex[:16]}..., got {actual[:16]}...",
                    )
                )
                if len(failures) >= max_failures:
                    break

            if heartbeat is not None:
                with suppress(Exception):
                    heartbeat()

        return BlobVerifyAllResult(
            checked=checked,
            checked_bytes=checked_bytes,
            failures=tuple(failures),
            truncated=len(failures) >= max_failures,
        )

    def detect_orphans(
        self,
        db_referenced_ids: set[str],
        *,
        max_sample: int = 10,
    ) -> OrphanDetectionResult:
        """Find blobs on disk that have no corresponding DB reference.

        Walks the blob store directory (one blob at a time, bounded
        memory) and compares against *db_referenced_ids* (the set of
        ``raw_id`` values from ``raw_sessions``).

        Returns count, total bytes, and a representative sample of
        orphan hashes.  Blob files that are temporary (``.blob.*``
        prefix) are excluded from the walk by ``iter_all()``.
        """
        orphan_count = 0
        orphan_bytes = 0
        orphan_samples: list[str] = []

        for hash_hex in self.iter_all():
            if hash_hex in db_referenced_ids:
                continue
            orphan_count += 1
            with suppress(OSError):
                orphan_bytes += self.blob_path(hash_hex).stat().st_size
            if len(orphan_samples) < max_sample:
                orphan_samples.append(hash_hex)

        return OrphanDetectionResult(
            orphan_count=orphan_count,
            orphan_bytes=orphan_bytes,
            orphan_samples=tuple(orphan_samples),
        )

    def cleanup_orphans(
        self,
        orphan_hashes: set[str],
        *,
        dry_run: bool = True,
    ) -> CleanupOrphansResult:
        """Delete orphaned blobs from the filesystem.

        Safety: *dry_run* defaults to ``True`` — callers must
        explicitly opt in to deletion.  The *orphan_hashes* set should
        be produced by ``detect_orphans()`` immediately before cleanup
        to avoid TOCTOU races against concurrent ingest.

        Returns per-blob results and aggregate stats.
        """
        if dry_run:
            would_delete_count = 0
            would_delete_bytes = 0
            for hash_hex in orphan_hashes:
                if not _VALID_HEX.match(hash_hex):
                    continue
                path = self.blob_path(hash_hex)
                if path.exists():
                    would_delete_count += 1
                    with suppress(OSError):
                        would_delete_bytes += path.stat().st_size
            return CleanupOrphansResult(
                deleted_count=0,
                deleted_bytes=0,
                errors=0,
                error_details=(),
                dry_run=True,
                would_delete_count=would_delete_count,
                would_delete_bytes=would_delete_bytes,
            )

        deleted_count = 0
        deleted_bytes = 0
        errors = 0
        error_details: list[str] = []

        for hash_hex in orphan_hashes:
            if not _VALID_HEX.match(hash_hex):
                errors += 1
                error_details.append(f"invalid hash: {hash_hex[:32]}...")
                continue
            path = self.blob_path(hash_hex)
            if not path.exists():
                continue
            try:
                file_size = path.stat().st_size
                path.unlink()
                deleted_count += 1
                deleted_bytes += file_size
            except OSError as exc:
                errors += 1
                error_details.append(f"{hash_hex[:16]}...: {exc}")

        return CleanupOrphansResult(
            deleted_count=deleted_count,
            deleted_bytes=deleted_bytes,
            errors=errors,
            error_details=tuple(error_details),
            dry_run=False,
            would_delete_count=0,
            would_delete_bytes=0,
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobVerifyFailure:
    """A single blob integrity verification failure."""

    hash: str
    reason: str  # stat_failed | read_error | hash_mismatch
    detail: str = ""


@dataclass(frozen=True)
class BlobVerifyAllResult:
    """Summary of a batch blob integrity verification pass."""

    checked: int
    checked_bytes: int
    failures: tuple[BlobVerifyFailure, ...]
    truncated: bool  # True when stopped early at max_failures

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    @property
    def failed_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True)
class OrphanDetectionResult:
    """Result of scanning the blob store for unreferenced blobs."""

    orphan_count: int
    orphan_bytes: int
    orphan_samples: tuple[str, ...]  # up to max_sample representative hashes


@dataclass(frozen=True)
class CleanupOrphansResult:
    """Result of an orphan blob cleanup operation."""

    deleted_count: int
    deleted_bytes: int
    errors: int
    error_details: tuple[str, ...]
    dry_run: bool
    would_delete_count: int
    would_delete_bytes: int


# Avoid shadowing by the method name
builtins_open = open

# Module-level singleton, lazily initialized
_DEFAULT_STORE: BlobStore | None = None


def get_blob_store() -> BlobStore:
    """Return the default blob store instance."""
    global _DEFAULT_STORE
    from polylogue.paths import blob_store_root

    root = blob_store_root()
    if _DEFAULT_STORE is None or _DEFAULT_STORE.root != root:
        _DEFAULT_STORE = BlobStore(root)
    return _DEFAULT_STORE


def reset_blob_store() -> None:
    """Reset the singleton (for testing)."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = None


def load_raw_content(raw_id: str) -> bytes:
    """Load raw content from the blob store by raw_id.

    Convenience wrapper around ``get_blob_store().read_all(raw_id)``.
    Suitable for small-to-medium blobs. For large files (JSONL), prefer
    streaming via ``get_blob_store().blob_path(raw_id)`` directly.
    """
    return get_blob_store().read_all(raw_id)


__all__ = [
    "BlobStore",
    "BlobVerifyAllResult",
    "BlobVerifyFailure",
    "CleanupOrphansResult",
    "OrphanDetectionResult",
    "get_blob_store",
    "load_raw_content",
    "reset_blob_store",
]
