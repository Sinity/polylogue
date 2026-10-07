"""ChatGPT provider assembly — asset-name and sandbox-file sidecar resolution.

bd polylogue-0hwv / polylogue-dt5s / polylogue-2m2e: a GDPR/Takeout export
ships asset bytes and two sibling JSON files that name them
(``conversation_asset_file_names.json``, ``library_files.json``). Asset
members carry their provider file id in the member name and are claimed by
that id, never by suffix: one export vintage names them ``file-<id>.dat``,
another ships them under their real extensions (``.png``/``.wav``/``.pdf``/
…) or under none at all, in per-conversation subdirectories.
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
(``chatgpt.py``'s ``chatgpt_block_metadata`` events). ``provider_file_id`` IS updated in place when an id-grade match is
found (tiers 1-4 of the sandbox resolver, or any asset-member id resolution) —
that is a real identity strengthening, not a guess.
"""

from __future__ import annotations

import mimetypes
import os
import re
import stat
import zipfile
from collections.abc import Iterator, Mapping
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import ijson

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.blob_store import BlobStore

from .assembly import SidecarData
from .parsers.base import ParsedAttachment, ParsedSession, ParsedSessionEvent
from .parsers.chatgpt_sidecars import ChatGPTAssetIndex, _normalize_file_id
from .prepared_message_sink import SqliteAttachmentSink, SqliteSessionEventSink

logger = get_logger(__name__)

_LIBRARY_FILES_NAME = "library_files.json"
_ASSET_NAMES_NAME = "conversation_asset_file_names.json"
_ASSET_MEMBER_ID_RE = re.compile(r"(?:\A|#)(file[-_][A-Za-z0-9]+)")


def _read_index_file(path: Path, index: ChatGPTAssetIndex, *, library: bool) -> bool:
    from .acquisition_boundary import open_bound_container
    from .source_staging import bind_source_input

    try:
        with (
            TemporaryDirectory(prefix="polylogue-chatgpt-sidecar-") as scratch,
            bind_source_input(path) as binding,
            open_bound_container(
                BlobStore(Path(scratch)),
                binding,
            ) as source,
        ):
            return index.load_stream(source.stream, library=library)
    except (OSError, ValueError, ijson.JSONError) as exc:
        logger.warning("chatgpt_sidecar_read_failed", path=str(path), error=str(exc))
        return False


def _read_chatgpt_zip_sidecars(
    zip_path: Path,
    store: BlobStore | None,
    index: ChatGPTAssetIndex,
    claimed: set[str],
) -> set[str]:
    """Read admitted JSON sidecars and stream every admitted asset member.

    ``ZipInfo`` identity is preserved from central-directory admission through
    decompression. In particular, a later duplicate filename cannot replace an
    earlier member by making ``ZipFile.open(name)`` resolve through the archive's
    name map. Both JSON and binary payloads are consumed through the same
    exact central-directory entry and validated through end of stream.
    """
    from .decoder_zip import ZIP_JSON_SUFFIXES, ZipEntryValidator, open_zip_entry

    targets = {_LIBRARY_FILES_NAME, _ASSET_NAMES_NAME}
    seen_targets: set[str] = set()
    selected: set[str] = set()
    group = index.begin_asset_group()
    try:
        from .acquisition_boundary import open_bound_container
        from .source_staging import bind_source_input

        with (
            TemporaryDirectory(prefix="polylogue-chatgpt-container-") as scratch,
            bind_source_input(zip_path) as binding,
            open_bound_container(
                BlobStore(Path(scratch)),
                binding,
            ) as physical,
            zipfile.ZipFile(physical.stream) as zf,
        ):
            validator = ZipEntryValidator("chatgpt", cursor_state=None, zip_path=zip_path)
            entries = validator.filter_entries(
                zf.infolist(),
                allowed_suffixes=ZIP_JSON_SUFFIXES,
                allowed_path=_is_asset_member,
            )
            for info in entries:
                check_compute_cancelled()
                # An asset member is claimed by its name, not its suffix: the
                # 2026-04-23 export ships assets under their real extensions
                # (and some under none), including a handful named `.json`.
                asset_id = _member_asset_id(Path(info.filename).name)
                if asset_id is not None:
                    if store is None or index.has_asset_member(group, asset_id, info.filename):
                        continue
                    try:
                        with open_zip_entry(zf, info) as handle:
                            blob_hash, size = store.write_from_fileobj(handle, heartbeat=check_compute_cancelled)
                    except (KeyError, zipfile.BadZipFile, OSError) as exc:
                        logger.debug(
                            "chatgpt_asset_read_failed",
                            path=str(zip_path),
                            member=info.filename,
                            error=str(exc),
                        )
                        continue
                    index.record_asset(group, asset_id, info.filename, (blob_hash, size))
                    continue
                if info.filename not in targets or info.filename in seen_targets or info.filename in claimed:
                    continue
                seen_targets.add(info.filename)
                try:
                    with open_zip_entry(zf, info) as source:
                        if index.load_stream(source, library=info.filename == _LIBRARY_FILES_NAME):
                            selected.add(info.filename)
                except (OSError, KeyError, zipfile.BadZipFile, ValueError, ijson.JSONError) as exc:
                    logger.debug(
                        "chatgpt_sidecar_zip_member_unavailable",
                        zip_path=str(zip_path),
                        member=info.filename,
                        error=str(exc),
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        logger.debug("chatgpt_sidecar_zip_open_failed", zip_path=str(zip_path), error=str(exc))
    index.finish_asset_group(group)
    return selected


def _member_asset_id(basename: str) -> str | None:
    """Return the asset id an export member's name carries, or ``None``.

    Every carrier names an asset by embedding its provider file id in the
    member name, and every shape puts that id either first or right after a
    ``#`` separator, terminated by ``-`` or ``.``:

        file-078R8dTqVR9lYSLVmOsCh6ht.dat
        <conversation>/image/file_<32hex>-<uuid>.png
        dalle-generations/file-<id>-<uuid>.webp
        file-<id>-<original name>.pdf
        <7hex>#file_<32hex>#p_0.jpg-p_0.jpg
        file-<id>-<uuid>                      (no extension at all)

    The id is what joins: it is the key ``library_files.json`` uses and what
    an attachment's ``provider_file_id`` normalizes to. Anchoring the match
    keeps non-asset members out — ``conversation_asset_file_names.json``
    contains the substring ``file_names`` but does not start with it.
    """
    match = _ASSET_MEMBER_ID_RE.search(basename)
    if match is None:
        return None
    return _normalize_file_id(match.group(1))


def _asset_rendition_key(asset_id: str, member_name: str) -> str:
    """Keep each physical rendition addressable under its provider file id."""
    return f"{asset_id}#{member_name}"


def _is_asset_member(name: str) -> bool:
    return _member_asset_id(Path(name).name) is not None


def _acquire_asset_blobs_from_directory(directory: Path, store: BlobStore, index: ChatGPTAssetIndex) -> None:
    """Stream an extracted export directory's asset files into the blob store.

    ``ChatGPTAssemblySpec.discover_sidecars`` already anchors on this directory
    (looking for ``library_files.json``/``conversation_asset_file_names.json``).
    Assets sit beside ``conversations-*.json`` and, in the extension-carrying
    export shape, under per-conversation ``image/``/``audio/`` subdirectories,
    so the walk is recursive; each file is streamed via
    ``BlobStore.write_from_fileobj`` over a no-follow regular-file handle.
    """
    from polylogue.storage.blob_publication import flush_blob_publications

    group = index.begin_asset_group()
    for asset_path in _walk_asset_files(directory):
        check_compute_cancelled()
        asset_id = _member_asset_id(asset_path.name)
        if asset_id is None:
            continue
        member = asset_path.relative_to(directory).as_posix()
        if index.has_asset_member(group, asset_id, member):
            continue
        try:
            # Inspect the opened object, not a followed path. NOFOLLOW closes
            # the leaf-symlink race; NONBLOCK prevents a substituted FIFO from
            # hanging acquisition before its regular-file check.
            descriptor = os.open(asset_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as handle:
                observed = os.fstat(handle.fileno())
                if not stat.S_ISREG(observed.st_mode):
                    continue
                # Streamed into the blob store, so a large asset costs no
                # memory; a size cap here would drop a valid asset.
                blob_hash, size = store.write_from_fileobj(handle, heartbeat=check_compute_cancelled)
        except OSError as exc:
            emit(
                "sources.chatgpt.asset_refused",
                level=WARNING,
                outcome="error",
                reason="read_failed",
                path=str(asset_path),
                error_type=type(exc).__name__,
            )
            continue
        index.record_asset(group, asset_id, member, (blob_hash, size))
    index.finish_asset_group(group)
    flush_blob_publications(store)


def _walk_asset_files(directory: Path) -> Iterator[Path]:
    def read_failed(error: OSError) -> None:
        raise error

    for root, dirnames, filenames in os.walk(directory, onerror=read_failed):
        check_compute_cancelled()
        dirnames.sort()
        root_path = Path(root)
        for filename in sorted(filenames):
            check_compute_cancelled()
            if _member_asset_id(filename) is None:
                continue
            candidate = root_path / filename
            try:
                if stat.S_ISREG(candidate.lstat().st_mode):
                    yield candidate
            except OSError as exc:
                emit(
                    "sources.chatgpt.asset_refused",
                    level=WARNING,
                    outcome="error",
                    reason="stat_failed",
                    path=str(candidate),
                    error_type=type(exc).__name__,
                )


class ChatGPTAssemblySpec:
    """ChatGPT provider assembly — asset-member/sandbox-file sidecar resolution."""

    def discover_sidecars(
        self,
        source_paths: list[Path],
        *,
        blob_store: BlobStore | None = None,
    ) -> SidecarData:
        """Discover ``library_files.json``/``conversation_asset_file_names.json``
        and, when ``blob_store`` is given, stream every asset member's bytes
        into the content-addressed blob store (bd polylogue-8ac0).

        Resolves via each source path's containing directory rather than
        requiring the sidecar filenames to appear verbatim in
        ``source_paths``: a full source-directory walk already includes them
        as siblings, but a daemon single-file catch-up re-parse
        (which discovers sidecars for one source path)
        passes only the one shard being reprocessed. Climbing to that shard's
        parent directory and globbing for the two known sidecar filenames
        there covers both call shapes with the same code, mirroring how
        ``CodexAssemblySpec`` climbs to a stable anchor directory.

        Asset bytes are acquired the same way: a ZIP source streams every
        member whose name carries a file id through
        ``BlobStore.write_from_fileobj`` (bounded decompression, no full-file
        memory load — mirrors ``decoder_zip.py``'s ``capture_raw`` branch); an
        extracted-directory source streams the same members from disk through
        ``BlobStore.write_from_fileobj`` after opening without symlink following.
        ``blob_store`` is ``None`` for callers
        that only need sidecar metadata (e.g. non-session artifact admission),
        so this stays a no-op there.
        """
        index = ChatGPTAssetIndex()
        claimed: set[str] = set()
        seen_dirs: set[Path] = set()
        try:
            for path in source_paths:
                check_compute_cancelled()
                if path.suffix.lower() == ".zip":
                    claimed.update(_read_chatgpt_zip_sidecars(path, blob_store, index, claimed))
                    if blob_store is not None:
                        from polylogue.storage.blob_publication import flush_blob_publications

                        flush_blob_publications(blob_store)
                    continue
                directory = path.parent
                if directory in seen_dirs:
                    continue
                seen_dirs.add(directory)
                for name in (_LIBRARY_FILES_NAME, _ASSET_NAMES_NAME):
                    candidate = directory / name
                    if (
                        name not in claimed
                        and candidate.is_file()
                        and _read_index_file(
                            candidate,
                            index,
                            library=name == _LIBRARY_FILES_NAME,
                        )
                    ):
                        claimed.add(name)
                if blob_store is not None:
                    _acquire_asset_blobs_from_directory(directory, blob_store, index)
            index.seal()
        except BaseException:
            index.close()
            raise
        result: SidecarData = {"chatgpt_asset_index": index}
        if index.asset_blobs:
            result["chatgpt_asset_blobs"] = index.asset_blobs
        return result

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        if conv.source_name is not Provider.CHATGPT or not conv.attachments:
            return conv
        index = sidecar_data.get("chatgpt_asset_index")
        asset_blobs = sidecar_data.get("chatgpt_asset_blobs") or {}
        if (index is None or index.is_empty) and not asset_blobs:
            return conv
        if index is None:
            index = ChatGPTAssetIndex.empty()
        temporary_index = sidecar_data.get("chatgpt_asset_index") is None
        try:
            if isinstance(conv.attachments, SqliteAttachmentSink):
                attachments = conv.attachments
                events = conv.session_events
                if (
                    attachments._writer is None
                    or not isinstance(events, SqliteSessionEventSink)
                    or events._writer is not attachments._writer
                ):
                    raise TypeError("ChatGPT sidecar enrichment requires one writable prepared carrier")
                conn = attachments._writer
                savepoint = "chatgpt_sidecar_enrichment"
                event_count = len(events)
                attachment_count = len(attachments)
                conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    write_position = 0
                    with attachments.original_items_for_rewrite() as originals:
                        for attachment in originals:
                            for item, event in _resolve_attachment_renditions(
                                attachment,
                                index,
                                thread_id=conv.provider_session_id,
                                asset_blobs=asset_blobs,
                            ):
                                if item is not None:
                                    if write_position >= attachment_count:
                                        attachments.append(item)
                                    else:
                                        attachments[write_position] = item
                                    write_position += 1
                                if event is not None:
                                    events.append(event)
                except BaseException:
                    conn.execute(f"ROLLBACK TO {savepoint}")
                    conn.execute(f"RELEASE {savepoint}")
                    events._count = event_count
                    attachments._count = attachment_count
                    raise
                conn.execute(f"RELEASE {savepoint}")
                return conv

            new_attachments: list[ParsedAttachment] = []
            new_events: list[ParsedSessionEvent] = []
            changed = False
            for attachment in conv.attachments:
                resolved_count = 0
                for item, event in _resolve_attachment_renditions(
                    attachment,
                    index,
                    thread_id=conv.provider_session_id,
                    asset_blobs=asset_blobs,
                ):
                    if item is not None:
                        new_attachments.append(item)
                        resolved_count += 1
                        changed |= item is not attachment
                    if event is not None:
                        new_events.append(event)
                changed |= resolved_count != 1
            if not changed and not new_events:
                return conv
            return conv.model_copy(
                update={
                    "attachments": new_attachments,
                    "session_events": [*conv.session_events, *new_events],
                }
            )
        finally:
            if temporary_index:
                index.close()


def _rendition_keys(asset_blobs: Mapping[str, tuple[str, int]], asset: str) -> Iterator[str]:
    from .parsers.chatgpt_sidecars import _AssetBlobs

    if isinstance(asset_blobs, _AssetBlobs):
        yield from asset_blobs.rendition_keys(asset)
    else:
        # Explicit caller-provided mappings retain their existing sorted-key
        # semantics; production discovery uses the paged index above.
        prefix = f"{asset}#"
        yield from (key for key in sorted(asset_blobs) if key.startswith(prefix))


def _resolve_attachment_renditions(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    *,
    thread_id: str,
    asset_blobs: Mapping[str, tuple[str, int]],
) -> Iterator[tuple[ParsedAttachment | None, ParsedSessionEvent | None]]:
    """Resolve one attachment into every physical member it names.

    Discovery keys a single member by its bare asset id and, once a second
    member normalizes to the same id, every member by ``asset_id#member``.
    No member of an ambiguous set is authoritatively the attachment's primary
    bytes, so each becomes its own attachment carrying its own blob; the
    pointer's own bytes, when it carries any, are kept as well.
    """
    asset_id = _normalize_file_id(attachment.provider_attachment_id)
    prefix = f"{asset_id}#"
    member_keys = (
        _rendition_keys(asset_blobs, asset_id)
        if attachment.attachment_kind != "sandbox_file" and asset_id not in asset_blobs
        else ()
    )
    iterator = iter(member_keys)
    first = next(iterator, None)
    if first is None:
        resolved, event = _resolve_attachment(attachment, index, thread_id=thread_id, asset_blobs=asset_blobs)
        yield resolved, event
        return
    base, base_event = _resolve_asset_attachment(attachment, index, {})
    if base_event is not None:
        yield None, base_event
    if attachment.inline_bytes is not None or attachment.precomputed_blob is not None:
        yield base, None
    for key in chain((first,), iterator):
        member = key[len(prefix) :]
        blob_hash, blob_size = asset_blobs[key]
        name = ParsedAttachment.sanitize_name(Path(member).name)
        rendition_id = _asset_rendition_key(attachment.provider_attachment_id, member)
        rendition = base.model_copy(
            update={
                "provider_attachment_id": rendition_id,
                # A provider file id the pointer or library already carried
                # is the provider's identity; the member's id is the fallback.
                "provider_file_id": base.provider_file_id or asset_id,
                "name": name,
                # A member name without a known extension keeps the media
                # type the pointer or library already declared.
                "mime_type": mimetypes.guess_type(name or "")[0] or base.mime_type,
                "size_bytes": blob_size,
                "path": None,
                "inline_bytes": None,
                "precomputed_blob": (blob_hash, blob_size),
                "prepared_carrier_key": None,
            }
        )
        yield (
            rendition,
            ParsedSessionEvent(
                event_type="chatgpt_asset_resolution",
                source_message_provider_id=attachment.message_provider_id,
                payload={
                    "attachment_id": rendition_id,
                    # A provider file id the pointer or library already carried
                    # is the provider's identity; the member's id is the fallback.
                    "provider_file_id": base.provider_file_id or asset_id,
                    "member_name": member,
                    "resolved_name": name,
                    "resolved_mime_type": rendition.mime_type,
                    "resolved_size_bytes": blob_size,
                    "resolution_source": "asset_member",
                    "blob_acquired": True,
                },
            ),
        )


def _resolve_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    *,
    thread_id: str,
    asset_blobs: Mapping[str, tuple[str, int]],
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    if attachment.attachment_kind == "sandbox_file":
        # bd polylogue-dt5s: sandbox links carry no bytes of their own -- the
        # export/capture never ships the Code-Interpreter container's file,
        # so there is nothing in ``asset_blobs`` to join against here.
        return _resolve_sandbox_attachment(attachment, index, thread_id=thread_id)
    return _resolve_asset_attachment(attachment, index, asset_blobs)


def _resolve_asset_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
    asset_blobs: Mapping[str, tuple[str, int]],
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    resolved = index.resolve_dat(attachment.provider_attachment_id)
    asset_id = _normalize_file_id(attachment.provider_attachment_id)
    # Only the bare key is this attachment's own blob; an ambiguous member set
    # is expanded by ``_resolve_attachment_renditions``.
    blob = asset_blobs.get(asset_id)
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
        # sidecar discovery (`_read_chatgpt_zip_sidecars` /
        # `_acquire_asset_blobs_from_directory`).
        # Recording the (hash, size) pair here -- rather than re-reading the
        # source bytes -- lets the session writer mark the attachment
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
