"""ZIP validation and extraction helpers for source ingestion."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact_path
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from .assembly import SidecarData
from .cursor import _record_cursor_failure
from .parsers.base import ParsedSession, RawSessionData

logger = get_logger(__name__)

MAX_COMPRESSION_RATIO = 1000
MAX_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024

#: Aggregate ceiling (bytes) on the sum of declared ``file_size`` across every
#: entry admitted from a single ZIP archive (polylogue-lqxx). The per-entry
#: cap above bounds one entry but does nothing to stop a "zip bomb by many
#: entries": thousands of entries each just under the 10 GiB per-entry cap
#: would still sum to an unbounded total. 64 GiB is chosen to comfortably
#: exceed any real single-archive GDPR/Takeout export this repo has observed
#: (the largest full raw corpus recorded across this operator's *entire*
#: archive history, spanning every session ever ingested, is ~52.1 GiB --
#: see `sources/revision_backfill.py`'s newest-revision-raws comment) while
#: still bounding aggregate decompression well below the terabyte-scale
#: totals a many-small-entries zip bomb would otherwise reach. It is also in
#: the same order of magnitude as the daemon's other whole-pass resource
#: envelopes (e.g. the whale-pass `raw_authority_whale_payload_bytes`
#: default of 8 GiB for a *single* stream-safe-gated component).
MAX_AGGREGATE_UNCOMPRESSED_SIZE = 64 * 1024 * 1024 * 1024

#: Chunk size used when bounding ZIP entry decompression. Read in fixed
#: windows so a malicious entry cannot allocate more than this much extra
#: memory beyond the running total before the ceiling check fires.
_ZIP_READ_CHUNK_SIZE = 1024 * 1024


class ZipBombError(Exception):
    """Raised when an entry's real decompressed size exceeds the hard cap.

    The declared header sizes (``ZipInfo.file_size`` / ``compress_size``)
    are attacker-controllable, so they are used only for an early cheap
    skip. The authoritative ceiling is enforced here against the actual
    bytes produced by decompression.
    """


class _BoundedZipReader(io.RawIOBase):
    """Wrap a ZIP entry stream and abort once ``max_bytes`` is exceeded.

    Every read is counted against the real decompressed byte total. If the
    total would cross ``max_bytes`` the reader raises :class:`ZipBombError`
    instead of returning the bytes, so downstream consumers never receive
    an over-cap payload regardless of the entry's declared sizes.
    """

    def __init__(self, raw: IO[bytes], *, max_bytes: int, entry_name: str) -> None:
        super().__init__()
        self._raw = raw
        self._max_bytes = max_bytes
        self._entry_name = entry_name
        self._total = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer)  # type: ignore[arg-type]
        chunk = self._raw.read(len(view))
        if not chunk:
            return 0
        self._total += len(chunk)
        if self._total > self._max_bytes:
            raise ZipBombError(
                f"ZIP entry {self._entry_name!r} exceeded the {self._max_bytes}-byte decompression ceiling during read"
            )
        view[: len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def open_bounded_zip_entry(
    zf: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int = MAX_UNCOMPRESSED_SIZE,
) -> io.BufferedReader:
    """Open a ZIP entry with a hard real-byte decompression ceiling.

    Returns a buffered stream that raises :class:`ZipBombError` if the
    actual decompressed size would exceed ``max_bytes``. This does not
    trust the (forgeable) declared header sizes — the ceiling is enforced
    against bytes produced by the decompressor itself.
    """
    raw = zf.open(name)
    return io.BufferedReader(_BoundedZipReader(raw, max_bytes=max_bytes, entry_name=name))


class ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_cursor_state", "_zip_path", "_aggregate_total")

    def __init__(
        self,
        provider_hint: str | Provider,
        *,
        cursor_state: CursorStatePayload | None,
        zip_path: Path,
    ) -> None:
        del provider_hint
        self._cursor_state = cursor_state
        self._zip_path = zip_path
        self._aggregate_total = 0

    def filter_entries(self, entries: list[zipfile.ZipInfo]) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries and record failures in cursor state."""
        for info in entries:
            if info.is_dir():
                continue
            name = info.filename
            lower_name = name.lower()

            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    logger.warning(
                        "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                        name,
                        self._zip_path,
                        ratio,
                    )
                    _record_cursor_failure(
                        self._cursor_state,
                        f"{self._zip_path}:{name}",
                        f"Suspicious compression ratio: {ratio:.1f}",
                    )
                    continue

            if info.file_size > MAX_UNCOMPRESSED_SIZE:
                logger.warning(
                    "Skipping oversized file %s in %s: %d bytes exceeds limit",
                    name,
                    self._zip_path,
                    info.file_size,
                )
                _record_cursor_failure(
                    self._cursor_state,
                    f"{self._zip_path}:{name}",
                    f"File size {info.file_size} exceeds limit",
                )
                continue

            if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                # Aggregate cap: the per-entry check above bounds one entry,
                # but a zip bomb built from many entries each just under the
                # per-entry cap would otherwise sum to an unbounded total.
                # Check the running total of declared uncompressed sizes
                # against MAX_AGGREGATE_UNCOMPRESSED_SIZE before yielding, so
                # rejection happens from central-directory metadata alone --
                # before any entry is opened/decompressed.
                projected_total = self._aggregate_total + info.file_size
                if projected_total > MAX_AGGREGATE_UNCOMPRESSED_SIZE:
                    logger.warning(
                        "Skipping %s in %s: aggregate uncompressed size %d would exceed the %d-byte archive-wide limit",
                        name,
                        self._zip_path,
                        projected_total,
                        MAX_AGGREGATE_UNCOMPRESSED_SIZE,
                    )
                    _record_cursor_failure(
                        self._cursor_state,
                        f"{self._zip_path}:{name}",
                        f"Aggregate uncompressed size {projected_total} exceeds archive-wide limit "
                        f"{MAX_AGGREGATE_UNCOMPRESSED_SIZE}",
                    )
                    continue

                self._aggregate_total = projected_total
                yield info


def zip_entry_session_artifact(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Stream a JSONL member before applying a terminal artifact path rule."""
    if not info.filename.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        return None
    from polylogue.archive.raw_payload.decode import jsonl_session_artifact

    with open_bounded_zip_entry(zf, info.filename) as handle:
        return jsonl_session_artifact(handle, provider=provider)


def zip_entry_provider_hint(entry_name: str, fallback_provider: str | Provider) -> Provider:
    del entry_name
    return Provider.from_string(fallback_provider)


def process_zip(
    zip_path: Path,
    *,
    provider_hint: Provider,
    should_group: bool,
    file_mtime: str | None,
    capture_raw: bool,
    cursor_state: CursorStatePayload | None,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    sidecar_data: SidecarData | None = None,
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Process a ZIP file, yielding sessions from its entries.

    ``sidecar_data`` (bd polylogue-8ac0) threads the source-scan-level
    provider assembly sidecars (e.g. ChatGPT's ``chatgpt_asset_index``/
    ``chatgpt_dat_blobs``, discovered once per source by
    ``_setup_source_walk`` before any entry is parsed) into every entry's
    ``_ParseContext`` so ``_SessionEmitter.emit``'s ``enrich_session`` hook
    actually fires for ZIP-bundle sources. Without it, every entry got an
    empty sidecar mapping and provider-assembly enrichment silently never ran
    for ZIP-shaped sources -- the common shape for a GDPR/Takeout export.
    """
    del should_group

    from polylogue.paths import blob_store_root
    from polylogue.storage.blob_publication import flush_blob_publications, publication_receipt_id

    from .cursor import _ParseContext
    from .dispatch import GROUP_PROVIDERS
    from .emitter import _SessionEmitter

    resolved_sidecar_data: SidecarData = sidecar_data if sidecar_data is not None else {}

    store = blob_store or BlobStore(blob_root or blob_store_root())

    validator = ZipEntryValidator(
        provider_hint,
        cursor_state=cursor_state,
        zip_path=zip_path,
    )

    with zipfile.ZipFile(zip_path) as zf:
        for info in validator.filter_entries(zf.infolist()):
            name = info.filename
            entry_provider_hint = zip_entry_provider_hint(name, provider_hint)
            path_classification = classify_artifact_path(name, provider=entry_provider_hint)
            session_artifact = zip_entry_session_artifact(zf, info, provider=entry_provider_hint)
            if (
                path_classification is not None
                and not path_classification.parse_as_session
                and session_artifact is None
            ):
                continue
            entry_should_group = entry_provider_hint in GROUP_PROVIDERS
            ctx = _ParseContext(
                provider_hint=entry_provider_hint,
                should_group=entry_should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                sidecar_data=resolved_sidecar_data,
            )
            emitter = _SessionEmitter(ctx)
            precomputed_raw: RawSessionData | None = None
            try:
                if capture_raw and entry_should_group:
                    # ``open_bounded_zip_entry`` enforces a hard real-byte
                    # ceiling during decompression, independent of the
                    # entry's (forgeable) declared header sizes.
                    with open_bounded_zip_entry(zf, name) as handle:
                        blob_hash, blob_size = store.write_from_fileobj(handle)
                    receipt_id = publication_receipt_id(store, blob_hash)
                    flush_blob_publications(store)
                    precomputed_raw = RawSessionData(
                        raw_bytes=b"",
                        source_path=f"{zip_path}:{name}",
                        source_index=None,
                        file_mtime=file_mtime,
                        provider_hint=entry_provider_hint,
                        blob_hash=blob_hash,
                        blob_size=blob_size,
                        blob_publication_receipt_id=receipt_id,
                    )
                with open_bounded_zip_entry(zf, name) as handle:
                    yield from emitter.emit(
                        handle,
                        name,
                        precomputed_raw=precomputed_raw,
                        session_artifact=session_artifact,
                    )
            except ZipBombError as exc:
                logger.warning(
                    "Skipping ZIP entry %s in %s: %s",
                    name,
                    zip_path,
                    exc,
                )
                _record_cursor_failure(
                    cursor_state,
                    f"{zip_path}:{name}",
                    str(exc),
                )
                continue


__all__ = [
    "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZipBombError",
    "ZipEntryValidator",
    "open_bounded_zip_entry",
    "process_zip",
    "zip_entry_session_artifact",
    "zip_entry_provider_hint",
]
