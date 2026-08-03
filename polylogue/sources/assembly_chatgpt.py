"""ChatGPT provider assembly — asset-name and sandbox-file sidecar resolution.

bd polylogue-0hwv / polylogue-dt5s / polylogue-2m2e: the 2026-07-29
GDPR/Takeout export ships ``.dat`` asset bytes and two sibling JSON files that
name them (``conversation_asset_file_names.json``, ``library_files.json``).
Neither the ZIP-bundle path nor the extracted-directory path has any other
place they'd naturally be read from: they are cross-conversation lookup
tables, not conversation shards themselves. This module discovers them once
per source scan (``discover_sidecars``) and joins every emitted ChatGPT
session's attachments against the resulting index (``enrich_session``), using
the standard provider-assembly extension point (``sources/assembly.py``) so
no change is needed to the ZIP/directory walking or emitter plumbing.

Resolution results are recorded as ``session_events`` rather than new
attachment/schema columns (index.db is a derived tier; a schema bump needs a
declared delta class) — same precedent as this file's neighbors
(``chatgpt.py``'s ``chatgpt_block_metadata``/``repo_identity_evidence``
events). ``provider_file_id`` IS updated in place when an id-grade match is
found (tiers 1-4 of the sandbox resolver, or any ``.dat`` id resolution) —
that is a real identity strengthening, not a guess.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore

from .assembly import SidecarData
from .parsers.base import ParsedAttachment, ParsedSession, ParsedSessionEvent
from .parsers.chatgpt_sidecars import ChatGPTAssetIndex, _normalize_file_id

logger = get_logger(__name__)

_LIBRARY_FILES_NAME = "library_files.json"
_ASSET_NAMES_NAME = "conversation_asset_file_names.json"
_DAT_SUFFIX = ".dat"


def _read_json_file(path: Path) -> object | None:
    try:
        with path.open("rb") as handle:
            data: object = json.load(handle)
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("chatgpt_sidecar_read_failed", path=str(path), error=str(exc))
        return None


def _read_json_zip_member(zip_path: Path, member_name: str) -> object | None:
    try:
        with zipfile.ZipFile(zip_path) as zf, zf.open(member_name) as handle:
            data: object = json.load(handle)
            return data
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        logger.debug(
            "chatgpt_sidecar_zip_member_unavailable",
            zip_path=str(zip_path),
            member=member_name,
            error=str(exc),
        )
        return None


def _dat_asset_id(basename: str) -> str:
    bare = basename[: -len(_DAT_SUFFIX)] if basename.lower().endswith(_DAT_SUFFIX) else basename
    return _normalize_file_id(bare)


def _acquire_dat_blobs_from_zip(zip_path: Path, store: BlobStore) -> dict[str, tuple[str, int]]:
    """Stream every ``.dat`` ZIP member into *store*, keyed by asset id.

    Mirrors ``decoder_zip.py``'s ``capture_raw`` streaming pattern (bounded
    decompression via ``open_bounded_zip_entry``, no full-file memory load)
    but scans the whole archive up front rather than the main
    ``ZipEntryValidator`` per-entry loop, which only ever admits
    ``.json``/``.jsonl`` entries (``session_only=True``) and would otherwise
    never see a ``.dat`` member at all.
    """
    from polylogue.storage.blob_publication import flush_blob_publications

    from .decoder_zip import (
        MAX_COMPRESSION_RATIO,
        MAX_UNCOMPRESSED_SIZE,
        ZipBombError,
        open_bounded_zip_entry,
    )

    acquired: dict[str, tuple[str, int]] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.lower().endswith(_DAT_SUFFIX):
                    continue
                dat_id = _dat_asset_id(Path(name).name)
                if dat_id in acquired:
                    continue
                if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                    logger.warning("chatgpt_dat_suspicious_compression_ratio", path=str(zip_path), member=name)
                    continue
                if info.file_size > MAX_UNCOMPRESSED_SIZE:
                    logger.warning(
                        "chatgpt_dat_oversized",
                        path=str(zip_path),
                        member=name,
                        size=info.file_size,
                    )
                    continue
                try:
                    with open_bounded_zip_entry(zf, name) as handle:
                        blob_hash, size = store.write_from_fileobj(handle)
                except ZipBombError:
                    logger.warning("chatgpt_dat_zip_bomb", path=str(zip_path), member=name)
                    continue
                except (KeyError, zipfile.BadZipFile, OSError) as exc:
                    logger.debug(
                        "chatgpt_dat_read_failed",
                        path=str(zip_path),
                        member=name,
                        error=str(exc),
                    )
                    continue
                acquired[dat_id] = (blob_hash, size)
    except (OSError, zipfile.BadZipFile) as exc:
        logger.warning("chatgpt_dat_zip_open_failed", path=str(zip_path), error=str(exc))
        return acquired
    if acquired:
        flush_blob_publications(store)
    return acquired


def _acquire_dat_blobs_from_directory(directory: Path, store: BlobStore) -> dict[str, tuple[str, int]]:
    """Stream sibling ``*.dat`` files from an extracted export directory.

    ``ChatGPTAssemblySpec.discover_sidecars`` already walks this directory
    (looking for ``library_files.json``/``conversation_asset_file_names.json``);
    the ``.dat`` files themselves sit right next to ``conversations-*.json``
    as ordinary files, streamed via ``BlobStore.write_from_path`` (no
    full-file memory load).
    """
    from polylogue.storage.blob_publication import flush_blob_publications

    from .decoder_zip import MAX_UNCOMPRESSED_SIZE

    acquired: dict[str, tuple[str, int]] = {}
    try:
        candidates = sorted(directory.glob(f"*{_DAT_SUFFIX}"))
    except OSError:
        return acquired
    for dat_path in candidates:
        dat_id = _dat_asset_id(dat_path.name)
        try:
            size_on_disk = dat_path.stat().st_size
        except OSError:
            continue
        if size_on_disk > MAX_UNCOMPRESSED_SIZE:
            logger.warning("chatgpt_dat_oversized", path=str(dat_path), size=size_on_disk)
            continue
        try:
            blob_hash, size = store.write_from_path(dat_path)
        except OSError as exc:
            logger.warning("chatgpt_dat_read_failed", path=str(dat_path), error=str(exc))
            continue
        acquired[dat_id] = (blob_hash, size)
    if acquired:
        flush_blob_publications(store)
    return acquired


class ChatGPTAssemblySpec:
    """ChatGPT provider assembly — ``.dat``/sandbox-file sidecar resolution."""

    def discover_sidecars(
        self,
        source_paths: list[Path],
        *,
        blob_store: BlobStore | None = None,
    ) -> SidecarData:
        """Discover ``library_files.json``/``conversation_asset_file_names.json``
        and, when ``blob_store`` is given, stream every ``.dat`` asset's bytes
        into the content-addressed blob store (bd polylogue-8ac0).

        Resolves via each source path's containing directory rather than
        requiring the sidecar filenames to appear verbatim in
        ``source_paths``: a full source-directory walk already includes them
        as siblings, but a daemon single-file catch-up re-parse
        (``ingest_batch/_core.py``'s ``discover_sidecars([Path(source_path)])``)
        passes only the one shard being reprocessed. Climbing to that shard's
        parent directory and globbing for the two known sidecar filenames
        there covers both call shapes with the same code, mirroring how
        ``CodexAssemblySpec`` climbs to a stable anchor directory.

        ``.dat`` bytes are acquired the same way: a ZIP source streams every
        ``.dat`` member through ``BlobStore.write_from_fileobj`` (bounded
        decompression, no full-file memory load — mirrors
        ``decoder_zip.py``'s ``capture_raw`` branch); an extracted-directory
        source streams sibling ``*.dat`` files through
        ``BlobStore.write_from_path``. ``blob_store`` is ``None`` for callers
        that only need sidecar metadata (e.g. non-session artifact admission),
        so this stays a no-op there.
        """
        library_files_payload: object | None = None
        asset_names_payload: object | None = None
        dat_blobs: dict[str, tuple[str, int]] = {}
        seen_dirs: set[Path] = set()
        for path in source_paths:
            if path.suffix.lower() == ".zip":
                if library_files_payload is None:
                    library_files_payload = _read_json_zip_member(path, _LIBRARY_FILES_NAME)
                if asset_names_payload is None:
                    asset_names_payload = _read_json_zip_member(path, _ASSET_NAMES_NAME)
                if blob_store is not None:
                    dat_blobs.update(_acquire_dat_blobs_from_zip(path, blob_store))
                continue
            directory = path.parent
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            if library_files_payload is None:
                candidate = directory / _LIBRARY_FILES_NAME
                if candidate.is_file():
                    library_files_payload = _read_json_file(candidate)
            if asset_names_payload is None:
                candidate = directory / _ASSET_NAMES_NAME
                if candidate.is_file():
                    asset_names_payload = _read_json_file(candidate)
            if blob_store is not None:
                dat_blobs.update(_acquire_dat_blobs_from_directory(directory, blob_store))
        index = ChatGPTAssetIndex.build(
            library_files_payload=library_files_payload,
            asset_file_names_payload=asset_names_payload,
        )
        result: SidecarData = {"chatgpt_asset_index": index}
        if dat_blobs:
            result["chatgpt_dat_blobs"] = dat_blobs
        return result

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        if conv.source_name is not Provider.CHATGPT or not conv.attachments:
            return conv
        index = sidecar_data.get("chatgpt_asset_index")
        dat_blobs = sidecar_data.get("chatgpt_dat_blobs") or {}
        if (index is None or index.is_empty) and not dat_blobs:
            return conv
        if index is None:
            index = ChatGPTAssetIndex.empty()

        new_attachments: list[ParsedAttachment] = []
        new_events: list[ParsedSessionEvent] = []
        changed = False
        for attachment in conv.attachments:
            resolved, event = _resolve_attachment(
                attachment, index, thread_id=conv.provider_session_id, dat_blobs=dat_blobs
            )
            new_attachments.append(resolved)
            if resolved is not attachment:
                changed = True
            if event is not None:
                new_events.append(event)
        if not changed and not new_events:
            return conv
        return conv.model_copy(
            update={
                "attachments": new_attachments,
                "session_events": [*conv.session_events, *new_events],
            }
        )


def _resolve_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    *,
    thread_id: str,
    dat_blobs: Mapping[str, tuple[str, int]],
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    if attachment.attachment_kind == "sandbox_file":
        # bd polylogue-dt5s: sandbox links carry no bytes of their own -- the
        # export/capture never ships the Code-Interpreter container's file,
        # so there is nothing in ``dat_blobs`` to join against here.
        return _resolve_sandbox_attachment(attachment, index, thread_id=thread_id)
    return _resolve_dat_attachment(attachment, index, dat_blobs)


def _resolve_dat_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    dat_blobs: Mapping[str, tuple[str, int]],
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    resolved = index.resolve_dat(attachment.provider_attachment_id)
    blob = dat_blobs.get(_normalize_file_id(attachment.provider_attachment_id))
    if resolved is None and blob is None:
        return attachment, None
    update: dict[str, object] = {}
    if resolved is not None:
        if attachment.name is None and resolved.name is not None:
            update["name"] = resolved.name
        if attachment.mime_type is None and resolved.mime_type is not None:
            update["mime_type"] = resolved.mime_type
        if attachment.size_bytes is None and resolved.size_bytes is not None:
            update["size_bytes"] = resolved.size_bytes
        if attachment.provider_file_id is None:
            update["provider_file_id"] = resolved.file_id
    if blob is not None and attachment.inline_bytes is None and attachment.precomputed_blob is None:
        # bd polylogue-8ac0: bytes already streamed into the blob store during
        # sidecar discovery (`_acquire_dat_blobs_from_zip`/`_from_directory`).
        # Recording the (hash, size) pair here -- rather than re-reading the
        # source bytes -- lets `ingest_batch/_core.py` mark the attachment
        # acquired without re-hashing already-written bytes.
        update["precomputed_blob"] = blob
        if attachment.size_bytes is None:
            update["size_bytes"] = blob[1]
    new_attachment = attachment.model_copy(update=update) if update else attachment
    event: ParsedSessionEvent | None = None
    if resolved is not None:
        event = ParsedSessionEvent(
            event_type="chatgpt_asset_resolution",
            source_message_provider_id=attachment.message_provider_id,
            payload={
                "attachment_id": attachment.provider_attachment_id,
                "resolved_name": resolved.name,
                "resolved_mime_type": resolved.mime_type,
                "resolved_size_bytes": resolved.size_bytes,
                "provider_sha256": resolved.sha256_digest,
                "resolution_source": resolved.source,
                "blob_acquired": blob is not None,
            },
        )
    return new_attachment, event


def _resolve_sandbox_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    *,
    thread_id: str,
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    file_name = attachment.name
    if not file_name:
        return attachment, None
    resolution = index.resolve_sandbox(
        message_id=attachment.message_provider_id,
        thread_id=thread_id,
        file_name=file_name,
    )
    new_attachment = attachment
    if resolution.file is not None and attachment.provider_file_id is None:
        new_attachment = attachment.model_copy(update={"provider_file_id": resolution.file.file_id})
    payload: dict[str, object] = {
        "sandbox_file_name": file_name,
        "resolution_tier": resolution.tier,
        "resolution_method": resolution.method,
    }
    if resolution.file is not None:
        payload["resolved_file_id"] = resolution.file.file_id
        payload["resolved_mime_type"] = resolution.file.mime_type
        payload["resolved_size_bytes"] = resolution.file.file_size_bytes
        payload["provider_sha256"] = resolution.file.sha256_digest
        payload["matched_name"] = resolution.matched_name
    event = ParsedSessionEvent(
        event_type="chatgpt_sandbox_file_resolution",
        source_message_provider_id=attachment.message_provider_id,
        payload=payload,
    )
    return new_attachment, event


__all__ = ["ChatGPTAssemblySpec"]
