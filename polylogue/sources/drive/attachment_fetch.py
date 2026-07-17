"""Live Drive-hosted attachment byte acquisition (polylogue-83u.2).

Gemini/AI-Studio Drive exports reference some attachments (`driveDocument(s)`,
`driveImage`, `driveAudio`, `driveVideo`) by Drive file id rather than
embedding bytes inline. `DriveSourceClient.download_bytes` can fetch those
bytes, but only while a live, authenticated client is available — and the
ingest pipeline decouples acquire (live client, streaming) from parse
(subprocess, no client) for memory-bounded streaming. This module is the one
place both are available together: it must run INSIDE the iterator scope that
owns the live client (`iter_drive_raw_data`), before that scope closes.

Rather than inventing a second attachment-acquisition path, this fetches the
bytes and injects them into the raw JSON payload under
`DRIVE_LIVE_FETCH_DATA_KEY` (base64). The existing chunk parser
(`drive_support_attachments.attachment_from_doc`) already knows how to turn
that sidecar into `ParsedAttachment.inline_bytes`, which the existing
publish-then-write path (`ingest_batch/_core.py`) already turns into a
content-addressed blob with a true SHA-256 and `acquisition_status="acquired"`
— zero changes needed downstream of the parser.

Per-file fetch failures and oversize files are left unresolved: the attachment
stays `upload_origin="drive"` with no `inline_bytes`, i.e. honestly unfetched.
Nothing here fabricates a hash or a size for bytes it did not actually read.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass, field

from polylogue.core.json import JSONValue
from polylogue.logging import get_logger

from ..parsers.drive_support_attachments import (
    DRIVE_DOC_FIELD_NAMES,
    DRIVE_LIVE_FETCH_DATA_KEY,
    DRIVE_MEDIA_FIELD_NAMES,
)

logger = get_logger(__name__)

_LIVE_FETCH_FIELDS: frozenset[str] = frozenset({*DRIVE_DOC_FIELD_NAMES, *DRIVE_MEDIA_FIELD_NAMES})

# Generous headroom over typical Drive/AI-Studio attachment sizes; a
# pathological reference should not be allowed to balloon ingest memory.
DEFAULT_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024


@dataclass(slots=True)
class DriveAttachmentFetchStats:
    """Outcome of one `fetch_live_drive_attachment_bytes` pass."""

    fetched_count: int = 0
    fetched_bytes: int = 0
    failed_count: int = 0
    skipped_too_large_count: int = 0
    failures: list[str] = field(default_factory=list)


def _doc_file_id(doc: JSONValue) -> str | None:
    if isinstance(doc, str):
        return doc or None
    if isinstance(doc, dict):
        for key in ("id", "fileId", "driveId"):
            value = doc.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _inject(doc: JSONValue, data_b64: str) -> JSONValue:
    if isinstance(doc, str):
        return {"id": doc, DRIVE_LIVE_FETCH_DATA_KEY: data_b64}
    if isinstance(doc, dict):
        updated = dict(doc)
        updated[DRIVE_LIVE_FETCH_DATA_KEY] = data_b64
        return updated
    return doc


def _resolve_doc(
    doc: JSONValue,
    download_bytes: Callable[[str], bytes],
    stats: DriveAttachmentFetchStats,
    *,
    max_bytes: int,
) -> JSONValue:
    if isinstance(doc, dict) and DRIVE_LIVE_FETCH_DATA_KEY in doc:
        return doc
    file_id = _doc_file_id(doc)
    if file_id is None:
        return doc
    try:
        raw = download_bytes(file_id)
    except Exception as exc:  # any transport/auth failure leaves it honestly unfetched
        stats.failed_count += 1
        stats.failures.append(f"{file_id}: {exc}")
        logger.warning("Failed to fetch live Drive attachment %s: %s", file_id, exc)
        return doc
    if len(raw) > max_bytes:
        stats.skipped_too_large_count += 1
        logger.warning(
            "Skipping live fetch for Drive attachment %s: %d bytes exceeds cap %d",
            file_id,
            len(raw),
            max_bytes,
        )
        return doc
    try:
        data_b64 = base64.b64encode(raw).decode("ascii")
    except (ValueError, binascii.Error) as exc:
        stats.failed_count += 1
        stats.failures.append(f"{file_id}: {exc}")
        return doc
    stats.fetched_count += 1
    stats.fetched_bytes += len(raw)
    return _inject(doc, data_b64)


def _resolve_field_value(
    value: JSONValue,
    download_bytes: Callable[[str], bytes],
    stats: DriveAttachmentFetchStats,
    *,
    max_bytes: int,
) -> JSONValue:
    if isinstance(value, list):
        return [_resolve_doc(item, download_bytes, stats, max_bytes=max_bytes) for item in value]
    return _resolve_doc(value, download_bytes, stats, max_bytes=max_bytes)


def _walk(
    node: JSONValue,
    download_bytes: Callable[[str], bytes],
    stats: DriveAttachmentFetchStats,
    *,
    max_bytes: int,
) -> JSONValue:
    if isinstance(node, dict):
        updated: dict[str, JSONValue] = {}
        for key, value in node.items():
            if key in _LIVE_FETCH_FIELDS:
                updated[key] = _resolve_field_value(value, download_bytes, stats, max_bytes=max_bytes)
            else:
                updated[key] = _walk(value, download_bytes, stats, max_bytes=max_bytes)
        return updated
    if isinstance(node, list):
        return [_walk(item, download_bytes, stats, max_bytes=max_bytes) for item in node]
    return node


def fetch_live_drive_attachment_bytes(
    payload: JSONValue,
    download_bytes: Callable[[str], bytes],
    *,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
) -> tuple[JSONValue, DriveAttachmentFetchStats]:
    """Resolve live Drive-hosted attachment references to inline bytes.

    Recursively walks ``payload`` for `driveDocument(s)`/`drive_document`/
    `driveImage`/`driveAudio`/`driveVideo` references and downloads each via
    ``download_bytes`` (expected to be a live `DriveSourceAPI.download_bytes`
    bound method, called while its client is still open). Fetched bytes are
    injected as base64 under `DRIVE_LIVE_FETCH_DATA_KEY` so
    `drive_support_attachments.attachment_from_doc` picks them up as
    `ParsedAttachment.inline_bytes`. Inline/file-data/YouTube references are
    never touched — they already have their own honest handling.

    Returns the (possibly rewritten) payload and fetch stats; the caller
    decides whether the rewritten payload is worth re-serializing.
    """
    stats = DriveAttachmentFetchStats()
    resolved = _walk(payload, download_bytes, stats, max_bytes=max_attachment_bytes)
    return resolved, stats


__all__ = [
    "DEFAULT_MAX_ATTACHMENT_BYTES",
    "DriveAttachmentFetchStats",
    "fetch_live_drive_attachment_bytes",
]
