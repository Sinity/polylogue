"""Exact-byte staging of a retained blob from bytes that prove its identity.

A retained blob is content-addressed: its SHA-256 digest and recorded size are
its whole identity. Any replay that still holds those exact bytes -- a direct
source file window, or a payload a backup reacquired -- may stage them into a
blob namespace; nothing else may. Staging streams through the namespace's
private workspace and is kept only when both digest and size match, so a
drifted, truncated, or grown source is refused instead of published.

Staging performs no archive mutation. Publication into the live store stays
with the archive writer (``ArchiveBlobPublisher.queue_prepared`` then
``flush``); a closed backup package publishes into its own namespace.
"""

from __future__ import annotations

import io
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from polylogue.storage.blob_store import BlobStore, BlobVerificationCancelledError, PreparedBlob


class _ExactWindowReader(io.RawIOBase):
    """Read at most ``remaining`` bytes, polling cancellation between reads."""

    def __init__(self, raw: IO[bytes], *, remaining: int, blob_hash: str, stop: Callable[[], bool] | None) -> None:
        super().__init__()
        self._raw = raw
        self._remaining = remaining
        self._blob_hash = blob_hash
        self._stop = stop

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        if self._stop is not None and self._stop():
            raise BlobVerificationCancelledError(self._blob_hash)
        if self._remaining <= 0:
            return 0
        view = memoryview(buffer)  # type: ignore[arg-type]
        chunk = self._raw.read(min(len(view), self._remaining))
        if not chunk:
            return 0
        self._remaining -= len(chunk)
        view[: len(chunk)] = chunk
        return len(chunk)


def stage_exact_blob(
    store: BlobStore,
    source: IO[bytes],
    *,
    blob_hash: str,
    size_bytes: int,
    stop: Callable[[], bool] | None = None,
) -> PreparedBlob | None:
    """Stage at most ``size_bytes`` from ``source``; keep them only when they are the blob.

    Returns the prepared blob when the staged bytes have exactly the recorded
    size and digest, and ``None`` (with the staged file discarded) otherwise.
    """
    store.blob_path(blob_hash)  # validates the digest before anything is staged
    if size_bytes < 0:
        return None
    prepared = store.prepare_from_fileobj(
        io.BufferedReader(_ExactWindowReader(source, remaining=size_bytes, blob_hash=blob_hash, stop=stop))
    )
    if prepared.size_bytes == size_bytes and prepared.hash_hex == blob_hash:
        return prepared
    store.discard_prepared(prepared)
    return None


@dataclass(frozen=True, slots=True)
class SourceByteWindow:
    """A half-open byte range of a direct source file."""

    start: int
    end: int


def direct_source_windows(
    *,
    size_bytes: int,
    append_start_offset: int | None,
    append_end_offset: int | None,
) -> tuple[SourceByteWindow, ...]:
    """Every window of a direct source that can hold a raw's recorded bytes.

    A full observation is the file's prefix of the recorded size (a later
    append leaves it there). An append raw retains either its recorded window
    or, when admission kept the whole observed prefix, ``[0, end)`` -- which
    is the same prefix window. Only windows of exactly the recorded size are
    candidates; the digest decides among them.
    """
    windows = [SourceByteWindow(0, size_bytes)]
    if (
        append_start_offset is not None
        and append_end_offset is not None
        and append_start_offset > 0
        and append_end_offset - append_start_offset == size_bytes
    ):
        windows.append(SourceByteWindow(append_start_offset, append_end_offset))
    return tuple(windows)


def stage_exact_direct_source_blob(
    store: BlobStore,
    *,
    source_path: Path,
    blob_hash: str,
    size_bytes: int,
    append_start_offset: int | None = None,
    append_end_offset: int | None = None,
    stop: Callable[[], bool] | None = None,
) -> tuple[PreparedBlob | None, str | None]:
    """Stage a retained raw's exact bytes from its recorded direct source file.

    Returns ``(prepared, None)`` on an exact match, or ``(None, reason)`` with
    a stable token: ``source_missing`` (no file at the recorded path),
    ``source_not_regular_file``, or ``hash_mismatch`` (no candidate window
    holds the recorded bytes). Any other read fault propagates as ``OSError``
    so the caller keeps it retryable.
    """
    try:
        metadata = os.stat(source_path)
    except FileNotFoundError:
        return None, "source_missing"
    if not stat.S_ISREG(metadata.st_mode):
        return None, "source_not_regular_file"
    for window in direct_source_windows(
        size_bytes=size_bytes,
        append_start_offset=append_start_offset,
        append_end_offset=append_end_offset,
    ):
        if window.end > metadata.st_size:
            continue
        with source_path.open("rb") as handle:
            handle.seek(window.start)
            prepared = stage_exact_blob(
                store,
                handle,
                blob_hash=blob_hash,
                size_bytes=window.end - window.start,
                stop=stop,
            )
        if prepared is not None:
            return prepared, None
    return None, "hash_mismatch"


__all__ = [
    "SourceByteWindow",
    "direct_source_windows",
    "stage_exact_blob",
    "stage_exact_direct_source_blob",
]
