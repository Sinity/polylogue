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

import hashlib
import io
import os
import sqlite3
import stat
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO

from polylogue.archive.revision_authority import raw_receipt_order_sql
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_coordinates import relocated_source_path, split_zip_member_text
from polylogue.core.sources import provider_from_origin
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


class RetainedBlobSourceKind(StrEnum):
    """Which recorded source window can hold a retained raw's exact bytes.

    The values are the proof kinds a backup package records, so a package
    and raw derivation name the same window the same way.
    """

    #: The file's ``[0, size)`` prefix, for a row that is not a full
    #: observation at source index 0 (the whole file, when it has not grown).
    DIRECT_FILE = "direct_file_sha256"
    #: The ``[0, size)`` prefix of a full observation the file grew past.
    HISTORICAL_SNAPSHOT_PREFIX = "historical_snapshot_prefix_sha256"
    #: A recorded append window ``[start, end)``, or the ``[0, end)`` prefix
    #: when admission retained the whole observed prefix.
    APPEND_WINDOW = "live_append_segment_sha256"
    #: A pre-offset append row, replayed from the end of the preceding full
    #: observation of the same path.
    LEGACY_APPEND_WINDOW = "historical_append_segment_sha256"
    #: The ZIP member the row's container coordinate names, replayed through
    #: acquisition's ZIP admission.
    ZIP_MEMBER = "zip_reacquired_payload"


@dataclass(frozen=True, slots=True)
class RetainedBlobSource:
    """One candidate source of a retained raw's bytes; ``window`` is ``None`` for a ZIP member."""

    kind: RetainedBlobSourceKind
    window: SourceByteWindow | None = None


_RAW_SOURCE_EVIDENCE_COLUMNS = """
    lower(hex(raw_sessions.blob_hash)) AS blob_hash,
    raw_sessions.raw_id AS raw_id,
    raw_sessions.raw_id AS ref_id,
    raw_sessions.origin AS origin,
    raw_sessions.capture_mode AS capture_mode,
    raw_sessions.acquired_at_ms AS acquired_at_ms,
    raw_sessions.source_path AS source_path,
    raw_sessions.source_index AS source_index,
    raw_sessions.revision_kind AS revision_kind,
    raw_sessions.append_start_offset AS append_start_offset,
    raw_sessions.append_end_offset AS append_end_offset,
    raw_sessions.blob_size AS size_bytes,
    coordinate.coordinate_format AS coordinate_format,
    coordinate.entry_ordinal AS entry_ordinal,
    coordinate.split_index AS split_index,
    coordinate.addressing_mode AS addressing_mode,
    coordinate.content_identity AS content_identity
"""


def read_raw_source_evidence(conn: sqlite3.Connection, raw_id: str) -> dict[str, object] | None:
    """The recorded acquisition evidence of one raw, in the row shape both replay routes read."""
    cursor = conn.execute(
        f"SELECT {_RAW_SOURCE_EVIDENCE_COLUMNS} FROM raw_sessions "
        "LEFT JOIN raw_container_coordinates coordinate ON coordinate.raw_id = raw_sessions.raw_id "
        "WHERE raw_sessions.raw_id = ?",
        (raw_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {str(column[0]): value for column, value in zip(cursor.description, row, strict=True)}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def is_recorded_container_member(row: Mapping[str, object]) -> bool:
    """Whether the row records a ZIP container coordinate."""
    return (
        row.get("coordinate_format") == "zip-v2"
        or row.get("entry_ordinal") is not None
        or row.get("addressing_mode") is not None
    )


def is_legacy_append_without_window(row: Mapping[str, object]) -> bool:
    """A pre-offset Codex or Claude Code append row: ``source_index`` -1 and no byte window."""
    provider = Provider.from_string(str(row.get("capture_mode") or ""))
    if provider is Provider.UNKNOWN:
        provider = provider_from_origin(Origin.from_string(str(row.get("origin") or "")))
    return (
        provider in {Provider.CODEX, Provider.CLAUDE_CODE}
        and _optional_int(row.get("source_index")) == -1
        and str(row.get("revision_kind") or "") in {"", "unknown"}
        and row.get("append_start_offset") is None
        and row.get("append_end_offset") is None
    )


def _live_zip_split(source_path: str, root: Path) -> tuple[str, str] | None:
    """Split ``<container>:<member>`` at a prefix that is a real ZIP here.

    The container path may itself hold colons (a Windows drive, a legal POSIX
    filename), so every colon is tried, shortest container first.
    """
    start = 0
    while (separator_at := source_path.find(":", start)) != -1:
        start = separator_at + 1
        if separator_at == 0 or separator_at == len(source_path) - 1:
            continue
        candidate = relocated_source_path(Path(source_path[:separator_at]), root)
        if candidate.is_file() and zipfile.is_zipfile(candidate):
            return source_path[:separator_at], source_path[start:]
    return None


def resolved_source_path(source_path: str, root: Path, *, container_member: bool) -> str:
    """Resolve a recorded acquisition path against the archive root in force.

    A container member keeps its ``:<member>`` suffix and re-anchors its
    container; a direct path is re-anchored whole, so a colon in a loose
    file's name stays file-name data.
    """
    split = (_live_zip_split(source_path, root) or split_zip_member_text(source_path)) if container_member else None
    outer, member = split if split is not None else (source_path, None)
    path = relocated_source_path(Path(outer), root)
    return f"{path}:{member}" if member is not None else str(path)


def is_container_member_row(row: Mapping[str, object], root: Path) -> bool:
    """Whether a row names a ZIP member: a recorded coordinate, or a legacy path whose prefix is a live ZIP."""
    if is_recorded_container_member(row):
        return True
    source_path = row.get("source_path")
    return isinstance(source_path, str) and _live_zip_split(source_path, root) is not None


def legacy_append_start(
    conn: sqlite3.Connection,
    row: Mapping[str, object],
    *,
    resolved_path: str,
) -> int | None:
    """Where a window-less append row's bytes begin in its source file, or ``None``.

    The row continues the byte chain of the same path (as recorded, or
    re-anchored at the root in force). The chain's anchor is the latest
    observation received before the row whose end is recorded: a full
    observation at source index 0 (its size) or a windowed append (its end
    offset). Every window-less append received between the anchor and the
    row extends the chain by its size, since appends are contiguous, and the
    row starts where the chain ends. "Received before" is the durable
    ``raw_payload`` receipt order (``raw_receipt_order_sql``), never
    ``acquired_at_ms``: a wall-clock step back between observations must not
    hide the anchor or pick a later one. A row with no receipt has no order
    and no start. The start is an inference; the caller proves it by hashing
    the window.
    """
    raw_id = row.get("raw_id") or row.get("ref_id")
    source_path = row.get("source_path")
    if not isinstance(raw_id, str) or not isinstance(source_path, str):
        return None
    own_order = conn.execute(
        "SELECT MAX(rowid) FROM blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'", (raw_id,)
    ).fetchone()[0]
    if own_order is None:
        return None
    anchor_order = raw_receipt_order_sql("anchor")
    anchor = conn.execute(
        f"""
        SELECT COALESCE(anchor.append_end_offset, anchor.blob_size), {anchor_order}
        FROM raw_sessions AS anchor
        WHERE anchor.source_path IN (?, ?)
          AND (
              anchor.append_end_offset IS NOT NULL
              OR (
                  anchor.source_index = 0
                  AND anchor.revision_kind IN ('full', 'unknown')
                  AND anchor.append_start_offset IS NULL
              )
          )
          AND {anchor_order} < ?
        ORDER BY {anchor_order} DESC, anchor.raw_id DESC
        LIMIT 1
        """,
        (source_path, resolved_path, own_order),
    ).fetchone()
    if anchor is None or _optional_int(anchor[0]) is None:
        return None
    chained_order = raw_receipt_order_sql("chained")
    (chained_size,) = conn.execute(
        f"""
        SELECT COALESCE(SUM(chained.blob_size), 0)
        FROM raw_sessions AS chained
        WHERE chained.source_path IN (?, ?)
          AND chained.source_index = -1
          AND chained.revision_kind = 'unknown'
          AND chained.append_start_offset IS NULL
          AND chained.append_end_offset IS NULL
          AND {chained_order} > ?
          AND {chained_order} < ?
        """,
        (source_path, resolved_path, anchor[1], own_order),
    ).fetchone()
    return int(anchor[0]) + int(chained_size)


@dataclass(frozen=True, slots=True)
class RetainedBlobSources:
    """Where one retained raw's bytes can be replayed from under the archive root in force."""

    #: The recorded source path re-anchored at the root in force.
    source_path: str
    container_member: bool
    candidates: tuple[RetainedBlobSource, ...]


def retained_blob_sources(conn: sqlite3.Connection, row: Mapping[str, object], *, root: Path) -> RetainedBlobSources:
    """Every recorded source window that can prove one raw's retained bytes.

    This is the one owner of that decision: backup recoverability and raw
    derivation's blob restoration both read their candidates here, so a raw
    one route can prove is one the other can restore. It re-anchors the
    recorded path at the archive root in force, decides whether the row is a
    ZIP member, and, for a window-less append row, finds where its bytes
    start (:func:`legacy_append_start`). ``row`` is the raw's recorded evidence
    (:func:`read_raw_source_evidence`, or a row of the same shape) and
    ``conn`` reads the source tier. Candidates are proofs only once the bytes
    read from them hash to the raw's recorded identity; the caller checks
    that.
    """
    recorded = row.get("source_path")
    if not isinstance(recorded, str) or not recorded:
        return RetainedBlobSources("", False, ())
    container_member = is_container_member_row(row, root)
    resolved = resolved_source_path(recorded, root, container_member=container_member)
    legacy_start = (
        legacy_append_start(conn, row, resolved_path=resolved)
        if not container_member and is_legacy_append_without_window(row)
        else None
    )
    return RetainedBlobSources(
        resolved,
        container_member,
        _source_candidates(row, container_member=container_member, legacy_append_start=legacy_start),
    )


def _source_candidates(
    row: Mapping[str, object],
    *,
    container_member: bool,
    legacy_append_start: int | None,
) -> tuple[RetainedBlobSource, ...]:
    if container_member:
        return (RetainedBlobSource(RetainedBlobSourceKind.ZIP_MEMBER),)
    size = _optional_int(row.get("size_bytes"))
    if size is None or size < 0:
        return ()
    start = _optional_int(row.get("append_start_offset"))
    end = _optional_int(row.get("append_end_offset"))
    if start is not None and end is not None:
        windows = [SourceByteWindow(start, end)] if 0 <= start <= end else []
        if end >= 0 and start != 0:
            # Admission may retain the whole observed prefix while recording
            # the tail's offsets; the blob is then ``[0, end)``.
            windows.append(SourceByteWindow(0, end))
        return tuple(RetainedBlobSource(RetainedBlobSourceKind.APPEND_WINDOW, window) for window in windows)
    if is_legacy_append_without_window(row):
        if legacy_append_start is None:
            return ()
        return (
            RetainedBlobSource(
                RetainedBlobSourceKind.LEGACY_APPEND_WINDOW,
                SourceByteWindow(legacy_append_start, legacy_append_start + size),
            ),
        )
    full_at_origin = (
        str(row.get("revision_kind") or "") in {"full", "unknown"} and _optional_int(row.get("source_index")) == 0
    )
    kind = RetainedBlobSourceKind.HISTORICAL_SNAPSHOT_PREFIX if full_at_origin else RetainedBlobSourceKind.DIRECT_FILE
    return (RetainedBlobSource(kind, SourceByteWindow(0, size)),)


def source_window_holds_blob(
    source_path: Path,
    window: SourceByteWindow,
    *,
    blob_hash: str,
    stop: Callable[[], bool] | None = None,
) -> tuple[bool, str | None]:
    """Whether one direct-source window hashes to the retained blob, streaming it.

    Returns ``(True, None)`` on a match, or ``(False, reason)`` with
    ``source_missing``, ``source_not_regular_file``, ``window_beyond_source``
    (the file is shorter than the window), or ``hash_mismatch``. Memory stays
    bounded whatever the window's size. Any other read fault propagates as
    ``OSError``.
    """
    try:
        metadata = os.stat(source_path)
    except FileNotFoundError:
        return False, "source_missing"
    if not stat.S_ISREG(metadata.st_mode):
        return False, "source_not_regular_file"
    if window.end > metadata.st_size:
        return False, "window_beyond_source"
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        handle.seek(window.start)
        reader = _ExactWindowReader(handle, remaining=window.end - window.start, blob_hash=blob_hash, stop=stop)
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return (True, None) if digest.hexdigest() == blob_hash else (False, "hash_mismatch")


def stage_exact_source_window_blob(
    store: BlobStore,
    *,
    source_path: Path,
    window: SourceByteWindow,
    blob_hash: str,
    stop: Callable[[], bool] | None = None,
) -> tuple[PreparedBlob | None, str | None]:
    """Stage one direct-source window when its bytes are exactly the retained blob.

    Streams the window, so memory stays bounded whatever its size. Returns
    ``(prepared, None)`` on an exact match, or ``(None, reason)`` with the
    tokens :func:`read_source_window` uses plus ``hash_mismatch``. Any other
    read fault propagates as ``OSError`` so the caller keeps it retryable.
    """
    try:
        metadata = os.stat(source_path)
    except FileNotFoundError:
        return None, "source_missing"
    if not stat.S_ISREG(metadata.st_mode):
        return None, "source_not_regular_file"
    if window.end > metadata.st_size:
        return None, "window_beyond_source"
    with source_path.open("rb") as handle:
        handle.seek(window.start)
        prepared = stage_exact_blob(
            store,
            handle,
            blob_hash=blob_hash,
            size_bytes=window.end - window.start,
            stop=stop,
        )
    return (prepared, None) if prepared is not None else (None, "hash_mismatch")


__all__ = [
    "RetainedBlobSource",
    "RetainedBlobSourceKind",
    "RetainedBlobSources",
    "SourceByteWindow",
    "is_container_member_row",
    "is_legacy_append_without_window",
    "is_recorded_container_member",
    "legacy_append_start",
    "read_raw_source_evidence",
    "resolved_source_path",
    "retained_blob_sources",
    "source_window_holds_blob",
    "stage_exact_blob",
    "stage_exact_source_window_blob",
]
