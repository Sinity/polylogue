"""Content-addressed blob store on the local filesystem.

Blobs are stored as immutable files under a two-level directory structure:
``{root}/{hash[:2]}/{hash[2:]}``, where hash is the SHA-256 hex digest of
the content. This is the same hash used as ``raw_id`` in the
``raw_conversations`` table — no separate addressing scheme needed.

Writes are atomic (tempfile + ``os.replace``). Files are never modified
after creation. Deduplication is free: identical content produces the
same hash, so the second write is a no-op.

The primary motivation is to avoid loading multi-GB files into Python
memory. ``write_from_path`` streams the file in 128 KB chunks, hashing
as it goes, then copies to the store — peak memory is one chunk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

_CHUNK_SIZE = 128 * 1024  # 128 KB


class BlobStore:
    """Content-addressed blob store backed by the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def blob_path(self, hash_hex: str) -> Path:
        """Return the filesystem path for a blob by its hex digest."""
        return self.root / hash_hex[:2] / hash_hex[2:]

    def exists(self, hash_hex: str) -> bool:
        """Check whether a blob exists on disk."""
        return self.blob_path(hash_hex).exists()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_from_path(self, source: Path) -> tuple[str, int]:
        """Stream-hash a file and copy it to the store.

        Reads the source in 128 KB chunks — never loads the full file into
        Python memory. Returns ``(sha256_hex, byte_count)``.

        If a blob with the same hash already exists, the write is skipped
        (content-addressed deduplication).
        """
        hasher = hashlib.sha256()
        size = 0
        with open(source, "rb") as src:
            while True:
                chunk = src.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)

        hash_hex = hasher.hexdigest()
        dest = self.blob_path(hash_hex)

        if dest.exists():
            return hash_hex, size

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dest.parent, prefix=".blob.")
            with open(source, "rb") as src:
                while True:
                    chunk = src.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    os.write(fd, chunk)
            os.close(fd)
            fd = None
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return hash_hex, size

    def write_from_fileobj(self, source: BinaryIO) -> tuple[str, int]:
        """Stream-hash an open binary file-like object into the store.

        Reads from ``source`` in 128 KB chunks, hashing and writing to a
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
                os.write(fd, chunk)

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
            os.write(fd, data)
            os.close(fd)
            fd = None
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
                if blob_file.is_file():
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


# Avoid shadowing by the method name
builtins_open = open

# Module-level singleton, lazily initialized
_DEFAULT_STORE: BlobStore | None = None


def get_blob_store() -> BlobStore:
    """Return the default blob store instance."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        from polylogue.paths import blob_store_root

        _DEFAULT_STORE = BlobStore(blob_store_root())
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


__all__ = ["BlobStore", "get_blob_store", "load_raw_content", "reset_blob_store"]
