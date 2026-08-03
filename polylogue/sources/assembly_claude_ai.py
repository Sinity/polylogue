"""Claude AI provider assembly — sole-copy attachment recovery sidecar.

bd polylogue-4zqh3: a Claude.ai browser-captured session can carry attachment
references whose bytes were never fetched at capture time (upload-only
metadata, no ``extracted_content``) and are permanently gone from claude.ai
itself once the message/attachment scrolls out of the API's retention
window. When those bytes are later recovered out of band -- a Borg backup,
an adjacent knowledge-base file, a manually-preserved copy -- the durable,
reindex-safe way to bind them back to the archived session is the same
provider-assembly extension point ``assembly_chatgpt.py`` uses for ``.dat``
asset bytes: resolve them at *parse time*, from evidence sitting next to the
source being (re-)ingested, not by patching ``index.db`` directly (a derived
tier a later full reindex would silently discard).

This spec looks for a small, purpose-built sidecar file named
``attachment-recovery.json`` next to (or in a same-or-parent directory of)
each ``source_paths`` entry:

    {"attachments": [{"native_id": "<attachment provider id>",
                       "path": "<file path, relative to the manifest>"}]}

``native_id`` must equal ``ParsedAttachment.provider_attachment_id`` (the
bare id Claude.ai assigns the attachment -- see
``base_support.attachment_from_meta``). For each entry whose file exists,
``discover_sidecars`` streams it into the blob store (when one is given) and
records ``(sha256_hex, byte_count)`` keyed by ``native_id``; ``enrich_session``
joins that back onto matching attachments as ``precomputed_blob``, exactly
the field ``ingest_batch/_core.py``'s ``preacquired_attachment_blobs`` loop
already knows how to record as ``acquired`` without re-hashing.

This does not run automatically against already-ingested sessions -- it
fires when the source carrying the manifest is (re-)walked, e.g. a one-off
``polylogue import`` pointed at a recovery-packet directory that places
both the manifest and the recovered files next to the original captured
document. See polylogue-4zqh3's bead notes for the concrete recovery-packet
path and the one attachment currently known to have a confirmed native-id
match (a corpus-wide reindex is the trigger this exists to survive).
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore

from .assembly import SidecarData
from .parsers.base import ParsedAttachment, ParsedSession

logger = get_logger(__name__)

_MANIFEST_NAME = "attachment-recovery.json"


def _read_manifest(path: Path) -> object | None:
    try:
        with path.open("rb") as handle:
            data: object = json.load(handle)
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("claude_ai_attachment_recovery_manifest_read_failed", path=str(path), error=str(exc))
        return None


def _acquire_recovered_blobs(
    manifest_dir: Path, payload: object, store: BlobStore | None
) -> dict[str, tuple[str, int]]:
    acquired: dict[str, tuple[str, int]] = {}
    if not isinstance(payload, dict):
        return acquired
    entries = payload.get("attachments")
    if not isinstance(entries, list):
        return acquired
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        native_id = entry.get("native_id")
        rel_path = entry.get("path")
        if not isinstance(native_id, str) or not native_id or not isinstance(rel_path, str) or not rel_path:
            continue
        file_path = (manifest_dir / rel_path).resolve()
        if not file_path.is_file():
            logger.warning(
                "claude_ai_attachment_recovery_file_missing",
                native_id=native_id,
                path=str(file_path),
            )
            continue
        if store is None:
            continue
        try:
            blob_hash, size = store.write_from_path(file_path)
        except OSError as exc:
            logger.warning(
                "claude_ai_attachment_recovery_write_failed",
                native_id=native_id,
                path=str(file_path),
                error=str(exc),
            )
            continue
        acquired[native_id] = (blob_hash, size)
    return acquired


class ClaudeAIAssemblySpec:
    """Claude AI provider assembly — sole-copy attachment recovery sidecar."""

    def discover_sidecars(
        self,
        source_paths: list[Path],
        *,
        blob_store: BlobStore | None = None,
    ) -> SidecarData:
        recovered: dict[str, tuple[str, int]] = {}
        seen_dirs: set[Path] = set()
        for path in source_paths:
            directory = path if path.is_dir() else path.parent
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            manifest_path = directory / _MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            payload = _read_manifest(manifest_path)
            if payload is None:
                continue
            recovered.update(_acquire_recovered_blobs(directory, payload, blob_store))
        result: SidecarData = {}
        if recovered:
            result["claude_ai_recovered_blobs"] = recovered
        return result

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        if conv.source_name is not Provider.CLAUDE_AI or not conv.attachments:
            return conv
        recovered = sidecar_data.get("claude_ai_recovered_blobs")
        if not recovered:
            return conv

        new_attachments: list[ParsedAttachment] = []
        changed = False
        for attachment in conv.attachments:
            blob = recovered.get(attachment.provider_attachment_id)
            if blob is None or attachment.inline_bytes is not None or attachment.precomputed_blob is not None:
                new_attachments.append(attachment)
                continue
            update: dict[str, object] = {"precomputed_blob": blob}
            if attachment.size_bytes is None:
                update["size_bytes"] = blob[1]
            new_attachments.append(attachment.model_copy(update=update))
            changed = True
        if not changed:
            return conv
        return conv.model_copy(update={"attachments": new_attachments})


__all__ = ["ClaudeAIAssemblySpec"]
