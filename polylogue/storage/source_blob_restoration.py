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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO

from polylogue.archive.revision_authority import raw_receipt_order_sql
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_coordinates import (
    read_captured_zip_coordinate_receipt,
)
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


_RAW_SOURCE_EVIDENCE_COLUMNS = f"""
    {raw_receipt_order_sql("raw_sessions")} AS receipt_order,
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
    coordinate.content_identity AS content_identity,
    coordinate.captured_coordinate AS captured_coordinate
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


def retained_source_location(row: Mapping[str, object], root: Path) -> tuple[str, bool]:
    """Resolve explicit retained coordinates or a literal loose-file path.

    A member namespace is authority only when its acquisition receipt names it.
    Colons and ZIP-looking suffixes in loose paths remain filename data.
    """
    source = str(row.get("source_path") or "")

    def relocate(path: Path) -> Path:
        for directory in ("inbox", "browser-capture", "hooks"):
            if directory in path.parts:
                candidate = root.joinpath(*path.parts[path.parts.index(directory) :])
                try:
                    candidate.stat()
                except (FileNotFoundError, NotADirectoryError):
                    continue
                else:
                    return candidate
        return path

    receipt = row.get("captured_coordinate")
    if receipt is not None:
        if not isinstance(receipt, str):
            raise ValueError("captured ZIP coordinate receipt must be text")
        coordinate = read_captured_zip_coordinate_receipt(receipt)
        return f"{relocate(Path(coordinate.canonical_container))}:{coordinate.member_name}", True

    return str(relocate(Path(source))), is_recorded_container_member(row)


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


def read_prior_full_source_receipts(conn: sqlite3.Connection, row: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    """The latest proven full receipt before this windowless append receipt."""
    order = _optional_int(row.get("receipt_order"))
    if order is None or not is_legacy_append_without_window(row):
        return ()
    rank = raw_receipt_order_sql("prior")
    cursor = conn.execute(
        f"SELECT {rank}, prior.blob_size FROM raw_sessions AS prior "
        "WHERE prior.source_path = ? AND prior.source_index = 0 "
        "AND prior.revision_kind IN ('full', 'unknown') "
        f"AND {rank} < ? ORDER BY {rank} DESC LIMIT 1",
        (row["source_path"], order),
    )
    try:
        predecessor = cursor.fetchone()
        return () if predecessor is None else ((int(predecessor[0]), int(predecessor[1])),)
    finally:
        cursor.close()


def retained_blob_source_candidates(
    row: Mapping[str, object],
    *,
    container_member: bool,
    prior_full_observations: Iterable[tuple[int, int]] = (),
) -> tuple[RetainedBlobSource, ...]:
    """Every recorded source window that can prove one raw's retained bytes.

    This is the one owner of that decision: backup recoverability and raw
    derivation's blob restoration both read their candidates here, so a raw
    one route can prove is one the other can restore. ``row`` is the raw's
    recorded evidence (:func:`read_raw_source_evidence`); ``container_member``
    is the caller's container decision (a relocated archive root can move
    the container). ``prior_full_observations`` are ``(receipt_order,
    size)`` of the full observations at the same path, read only for a
    legacy append row. The order is the latest durable raw-payload receipt order, not their possibly inverted wall-clock timestamps.
    Candidates are proofs only once the bytes read from
    them hash to the raw's recorded identity; the caller checks that.
    """
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
        receipt_order = _optional_int(row.get("receipt_order"))
        predecessors = (
            [(order, full_size) for order, full_size in prior_full_observations if order < receipt_order]
            if receipt_order is not None
            else []
        )
        if not predecessors:
            return ()
        _timestamp, predecessor_end = max(predecessors)
        return (
            RetainedBlobSource(
                RetainedBlobSourceKind.LEGACY_APPEND_WINDOW,
                SourceByteWindow(predecessor_end, predecessor_end + size),
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
    "SourceByteWindow",
    "is_legacy_append_without_window",
    "is_recorded_container_member",
    "read_raw_source_evidence",
    "read_prior_full_source_receipts",
    "source_window_holds_blob",
    "retained_blob_source_candidates",
    "retained_source_location",
    "stage_blob_from_recorded_source",
    "stage_exact_blob",
    "stage_exact_source_window_blob",
]


def stage_blob_from_recorded_source(
    conn: sqlite3.Connection,
    archive_root: Path,
    blob_store: BlobStore,
    raw_id: str,
    *,
    blob_hash: str,
    source_path: str,
    stop: Callable[[], bool] | None = None,
) -> tuple[PreparedBlob | None, str | None]:
    """Stage one absent blob from the first recorded source window holding its exact bytes.

    The candidate windows come from ``retained_blob_source_candidates``,
    the owner backup recoverability reads too. A ZIP member is replayed
    through acquisition's ZIP admission (``zip_reacquired_unit``)
    and staged only when the replayed value is byte-identical to the
    blob; a structural-only match is ``inexact_payload``. Returns the
    staged blob, or ``None`` with the last candidate's refusal reason.
    """
    row = read_raw_source_evidence(conn, raw_id)
    if row is None:
        raise KeyError(raw_id)
    prior_full_observations = read_prior_full_source_receipts(conn, row)
    source_path, container_member = retained_source_location(row, archive_root)
    candidates = retained_blob_source_candidates(
        row,
        container_member=container_member,
        prior_full_observations=prior_full_observations,
    )
    if not candidates:
        return None, "no_source_window"
    reason: str | None = None
    for candidate in candidates:
        if candidate.window is not None:
            prepared, reason = stage_exact_source_window_blob(
                blob_store,
                source_path=Path(source_path),
                window=candidate.window,
                blob_hash=blob_hash,
                stop=stop,
            )
        else:
            from polylogue.storage.source_zip_replay import zip_reacquired_unit

            # The resolved unit streams from its member again: a preserved
            # member can be gigabytes, so its bytes are never held whole.
            unit, reason = zip_reacquired_unit(row, source_path=source_path, zip_payload_cache={})
            prepared = None
            if unit is not None and unit.open_payload is not None:
                try:
                    with unit.open_payload() as unit_stream:
                        prepared = stage_exact_blob(
                            blob_store,
                            unit_stream,
                            blob_hash=blob_hash,
                            size_bytes=unit.size_bytes,
                            stop=stop,
                        )
                except (OSError, zipfile.BadZipFile, LookupError, ContentIdentityRefusal) as exc:
                    reason = f"error:{type(exc).__name__}"
                else:
                    reason = None if prepared is not None else "inexact_payload"
        if prepared is not None:
            return prepared, None
    return None, reason
