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
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.logging import get_logger

from .assembly import SidecarData
from .parsers.base import ParsedAttachment, ParsedSession, ParsedSessionEvent
from .parsers.chatgpt_sidecars import ChatGPTAssetIndex

logger = get_logger(__name__)

_LIBRARY_FILES_NAME = "library_files.json"
_ASSET_NAMES_NAME = "conversation_asset_file_names.json"


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


class ChatGPTAssemblySpec:
    """ChatGPT provider assembly — ``.dat``/sandbox-file sidecar resolution."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover ``library_files.json``/``conversation_asset_file_names.json``.

        Resolves via each source path's containing directory rather than
        requiring the sidecar filenames to appear verbatim in
        ``source_paths``: a full source-directory walk already includes them
        as siblings, but a daemon single-file catch-up re-parse
        (``ingest_batch/_core.py``'s ``discover_sidecars([Path(source_path)])``)
        passes only the one shard being reprocessed. Climbing to that shard's
        parent directory and globbing for the two known sidecar filenames
        there covers both call shapes with the same code, mirroring how
        ``CodexAssemblySpec`` climbs to a stable anchor directory.
        """
        library_files_payload: object | None = None
        asset_names_payload: object | None = None
        seen_dirs: set[Path] = set()
        for path in source_paths:
            if path.suffix.lower() == ".zip":
                if library_files_payload is None:
                    library_files_payload = _read_json_zip_member(path, _LIBRARY_FILES_NAME)
                if asset_names_payload is None:
                    asset_names_payload = _read_json_zip_member(path, _ASSET_NAMES_NAME)
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
        index = ChatGPTAssetIndex.build(
            library_files_payload=library_files_payload,
            asset_file_names_payload=asset_names_payload,
        )
        return {"chatgpt_asset_index": index}

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        if conv.source_name is not Provider.CHATGPT or not conv.attachments:
            return conv
        index = sidecar_data.get("chatgpt_asset_index")
        if index is None or index.is_empty:
            return conv

        new_attachments: list[ParsedAttachment] = []
        new_events: list[ParsedSessionEvent] = []
        changed = False
        for attachment in conv.attachments:
            resolved, event = _resolve_attachment(attachment, index, thread_id=conv.provider_session_id)
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
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    if attachment.attachment_kind == "sandbox_file":
        return _resolve_sandbox_attachment(attachment, index, thread_id=thread_id)
    return _resolve_dat_attachment(attachment, index)


def _resolve_dat_attachment(
    attachment: ParsedAttachment,
    index: ChatGPTAssetIndex,
) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
    resolved = index.resolve_dat(attachment.provider_attachment_id)
    if resolved is None:
        return attachment, None
    update: dict[str, object] = {}
    if attachment.name is None and resolved.name is not None:
        update["name"] = resolved.name
    if attachment.mime_type is None and resolved.mime_type is not None:
        update["mime_type"] = resolved.mime_type
    if attachment.size_bytes is None and resolved.size_bytes is not None:
        update["size_bytes"] = resolved.size_bytes
    if attachment.provider_file_id is None:
        update["provider_file_id"] = resolved.file_id
    new_attachment = attachment.model_copy(update=update) if update else attachment
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
