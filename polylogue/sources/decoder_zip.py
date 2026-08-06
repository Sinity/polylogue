"""ZIP validation and extraction helpers for source ingestion."""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact_path
from polylogue.archive.zip_admission import (
    _ZIP_READ_CHUNK_SIZE,
    MAX_AGGREGATE_UNCOMPRESSED_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    ZIP_JSON_SUFFIXES,
    ZipAdmission,
    ZipBombError,
    open_bounded_zip_entry,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from .assembly import SidecarData
from .cursor import _record_cursor_failure
from .parsers.base import ParsedSession, RawSessionData

logger = get_logger(__name__)


class ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_cursor_state", "_zip_path", "_admission")

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
        self._admission = ZipAdmission(zip_path=zip_path)

    def filter_entries(
        self,
        entries: list[zipfile.ZipInfo],
        *,
        allowed_suffixes: Collection[str] = ZIP_JSON_SUFFIXES,
        on_rejected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries and record failures in cursor state.

        ``allowed_suffixes`` selects which member kinds a caller needs, while
        this validator remains the sole owner of the security checks. The
        yielded object is the exact central-directory ``ZipInfo`` that was
        admitted. Callers must pass it through to ``open_bounded_zip_entry``;
        reopening by filename can select a different duplicate member.

        ``on_rejected`` lets read-only surfaces report the same admission
        decisions without duplicating the security checks.
        """

        def reject(info: zipfile.ZipInfo, reason: str) -> None:
            _record_cursor_failure(
                self._cursor_state,
                f"{self._zip_path}:{info.filename}",
                reason.capitalize() if reason.startswith("aggregate") else reason,
            )
            if on_rejected is not None:
                on_rejected(info, reason)

        yield from self._admission.filter_entries(
            entries,
            allowed_suffixes=allowed_suffixes,
            on_rejected=reject,
        )


def zip_entry_session_artifact(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Decode a member before applying a terminal artifact path rule."""
    from polylogue.archive.raw_payload.decode import jsonl_session_artifact

    lower_name = info.filename.lower()
    if lower_name.endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        with open_bounded_zip_entry(zf, info) as handle:
            return jsonl_session_artifact(handle, provider=provider)
    if not lower_name.endswith(".json"):
        return None
    try:
        with open_bounded_zip_entry(zf, info) as handle:
            payload = json_loads(handle.read())
    except JSONDecodeError:
        return None
    # Deliberately omit source_path. The caller is asking whether decoded
    # content can override a non-session path rule, so reapplying that rule
    # here would make the evidence check circular.
    from polylogue.archive.artifact_taxonomy import classify_artifact

    artifact = classify_artifact(payload, provider=provider)
    return artifact if artifact.parse_as_session else None


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
            session_artifact: ArtifactClassification | None = None
            if path_classification is not None and not path_classification.parse_as_session:
                session_artifact = zip_entry_session_artifact(zf, info, provider=entry_provider_hint)
                if session_artifact is None:
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
                    with open_bounded_zip_entry(zf, info) as handle:
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
                with open_bounded_zip_entry(zf, info) as handle:
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
    "_ZIP_READ_CHUNK_SIZE",
    "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZIP_JSON_SUFFIXES",
    "ZipBombError",
    "ZipEntryValidator",
    "open_bounded_zip_entry",
    "process_zip",
    "zip_entry_session_artifact",
    "zip_entry_provider_hint",
]
