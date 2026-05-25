"""Per-conversation provenance read surface (#1125).

Surfaces the source artifact and ingest metadata that produced a given
conversation, with optional bounded access to the raw payload bytes.

Contract:

- Provenance metadata (source path, acquisition timestamp, content hash,
  raw blob size, validation/quarantine state) is always returned for a
  known conversation. Source paths are sanitized for display — the home
  directory is rendered as ``~`` so absolute filesystem locations do not
  leak unnecessarily into the reader.
- The raw payload preview is opt-in (``?include_raw=1``) and is capped
  at :data:`RAW_PREVIEW_MAX_BYTES`. The cap is enforced server-side and
  cannot be widened by a client query parameter.
- Quarantined / unreadable raw artifacts surface explicitly via the
  ``quarantined`` flag plus a ``quarantine_reason`` string, instead of
  silently omitting the row.

The endpoint reads from ``conversations`` and ``raw_conversations`` via a
synchronous SQLite connection. No archive mutations occur. Raw bytes are
read from the content-addressed blob store using
:func:`polylogue.storage.blob_store.get_blob_store`'s ``read_prefix``
helper so the full blob is never materialized for the preview.
"""

from __future__ import annotations

import base64
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from polylogue.paths import db_path
from polylogue.storage.blob_store import get_blob_store

# Hard server-side cap on the preview window. The endpoint will never
# return more than this many bytes of raw content regardless of what a
# client requests. Sized to bound a single response while still being
# large enough to show typical session headers and the first message.
RAW_PREVIEW_MAX_BYTES: Final[int] = 16 * 1024


@dataclass(frozen=True)
class ProvenanceRow:
    """Joined conversation + raw row used to assemble a response."""

    conversation_id: str
    source_name: str
    content_hash: str
    raw_id: str | None
    source_path: str | None
    source_name: str | None
    blob_size: int | None
    acquired_at: str | None
    file_mtime: str | None
    parsed_at: str | None
    parse_error: str | None
    validated_at: str | None
    validation_status: str | None
    validation_error: str | None


def _display_source_path(raw_path: str | None) -> str | None:
    """Return a presentation-friendly source path.

    Replaces the user home prefix with ``~`` so we do not leak absolute
    filesystem locations unnecessarily. Returns ``None`` for falsy input
    so the surface can distinguish "no source recorded" from a path that
    happens to be the empty string.
    """

    if not raw_path:
        return None
    home = os.path.expanduser("~")
    if home and home != "/" and (raw_path == home or raw_path.startswith(home + os.sep)):
        return "~" + raw_path[len(home) :]
    return raw_path


def fetch_provenance_row(conversation_id: str) -> ProvenanceRow | None:
    """Load the joined provenance row for *conversation_id*.

    Returns ``None`` when no conversation with that id exists. When the
    conversation exists but has no raw_id (legacy / hand-seeded rows),
    the raw fields come back as ``None`` and the caller surfaces the
    "no raw artifact" state explicitly.
    """

    dbp = db_path()
    if not dbp.exists():
        return None
    conn = sqlite3.connect(str(dbp))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT
                c.conversation_id   AS conversation_id,
                c.source_name     AS source_name,
                c.content_hash      AS content_hash,
                c.raw_id            AS raw_id,
                r.source_path       AS source_path,
                r.source_name       AS source_name,
                r.blob_size         AS blob_size,
                r.acquired_at       AS acquired_at,
                r.file_mtime        AS file_mtime,
                r.parsed_at         AS parsed_at,
                r.parse_error       AS parse_error,
                r.validated_at      AS validated_at,
                r.validation_status AS validation_status,
                r.validation_error  AS validation_error
            FROM conversations AS c
            LEFT JOIN raw_conversations AS r ON r.raw_id = c.raw_id
            WHERE c.conversation_id = ?
            """,
            (conversation_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return ProvenanceRow(
        conversation_id=str(row["conversation_id"]),
        content_hash=str(row["content_hash"] or ""),
        raw_id=(str(row["raw_id"]) if row["raw_id"] is not None else None),
        source_path=(str(row["source_path"]) if row["source_path"] is not None else None),
        source_name=(str(row["source_name"]) if row["source_name"] is not None else None),
        blob_size=(int(row["blob_size"]) if row["blob_size"] is not None else None),
        acquired_at=(str(row["acquired_at"]) if row["acquired_at"] is not None else None),
        file_mtime=(str(row["file_mtime"]) if row["file_mtime"] is not None else None),
        parsed_at=(str(row["parsed_at"]) if row["parsed_at"] is not None else None),
        parse_error=(str(row["parse_error"]) if row["parse_error"] is not None else None),
        validated_at=(str(row["validated_at"]) if row["validated_at"] is not None else None),
        validation_status=(str(row["validation_status"]) if row["validation_status"] is not None else None),
        validation_error=(str(row["validation_error"]) if row["validation_error"] is not None else None),
    )


def _quarantine_state(row: ProvenanceRow) -> tuple[bool, str | None]:
    """Classify the raw artifact's quarantine state.

    Returns ``(quarantined, reason)``. A raw artifact is treated as
    quarantined when it failed parsing, failed validation, or has no
    raw_id at all (no recorded source artifact).
    """

    if row.raw_id is None:
        return True, "no_raw_artifact"
    if row.parse_error:
        return True, "parse_error"
    if row.validation_status == "failed":
        return True, "validation_failed"
    return False, None


def _build_raw_preview(
    raw_id: str,
    blob_size: int | None,
    *,
    requested_bytes: int | None,
) -> dict[str, object]:
    """Read and encode a bounded raw payload preview.

    Server-side cap is :data:`RAW_PREVIEW_MAX_BYTES`. A larger
    ``requested_bytes`` is clamped to the cap; a smaller positive value
    is honored as-is. Returns a structured envelope that carries the
    cap, the actual bytes returned, the truncation flag, and either the
    decoded text (when the prefix is valid UTF-8) or a base64-encoded
    binary payload.

    On a missing blob, returns a structured ``unavailable`` envelope
    rather than raising — the caller surfaces this as part of the
    provenance response so the operator sees the unreadable state
    explicitly.
    """

    if requested_bytes is None or requested_bytes <= 0:
        cap = RAW_PREVIEW_MAX_BYTES
    else:
        cap = min(int(requested_bytes), RAW_PREVIEW_MAX_BYTES)

    store = get_blob_store()
    try:
        data = store.read_prefix(raw_id, cap)
    except FileNotFoundError:
        return {
            "available": False,
            "reason": "blob_missing",
            "max_bytes": cap,
            "bytes_returned": 0,
            "truncated": False,
        }
    except OSError as exc:
        return {
            "available": False,
            "reason": f"blob_unreadable: {exc.__class__.__name__}",
            "max_bytes": cap,
            "bytes_returned": 0,
            "truncated": False,
        }

    bytes_returned = len(data)
    total_size = blob_size if blob_size is not None else bytes_returned
    truncated = bytes_returned < total_size or (bytes_returned == cap and total_size > cap)

    envelope: dict[str, object] = {
        "available": True,
        "max_bytes": cap,
        "bytes_returned": bytes_returned,
        "total_size_bytes": total_size,
        "truncated": bool(truncated),
    }
    try:
        envelope["encoding"] = "utf-8"
        envelope["text"] = data.decode("utf-8")
    except UnicodeDecodeError:
        envelope["encoding"] = "base64"
        envelope["base64"] = base64.b64encode(data).decode("ascii")
    return envelope


def build_provenance_payload(
    conversation_id: str,
    *,
    include_raw: bool = False,
    requested_bytes: int | None = None,
) -> dict[str, object] | None:
    """Assemble the JSON payload for ``GET /api/conversations/{id}/provenance``.

    Returns ``None`` when the conversation does not exist so the caller
    can emit a 404. Otherwise returns the full structured envelope. The
    raw preview is only attached when ``include_raw`` is true, and even
    then is bounded by :data:`RAW_PREVIEW_MAX_BYTES`.
    """

    row = fetch_provenance_row(conversation_id)
    if row is None:
        return None
    quarantined, quarantine_reason = _quarantine_state(row)

    raw_path = row.source_path
    sanitized_path = _display_source_path(raw_path)
    home = os.path.expanduser("~") or None
    is_absolute = bool(raw_path and Path(raw_path).is_absolute())
    contains_home = bool(home and raw_path and (raw_path == home or raw_path.startswith(home + os.sep)))

    payload: dict[str, object] = {
        "conversation_id": row.conversation_id,
        "provider": row.source_name or None,
        "content_hash": row.content_hash or None,
        "raw_id": row.raw_id,
        "source_path_display": sanitized_path,
        "source_name": row.source_name,
        "source_path_is_absolute": is_absolute,
        "source_path_contains_home": contains_home,
        "acquired_at": row.acquired_at,
        "file_mtime": row.file_mtime,
        "parsed_at": row.parsed_at,
        "parse_error": row.parse_error,
        "validated_at": row.validated_at,
        "validation_status": row.validation_status,
        "validation_error": row.validation_error,
        "blob_size_bytes": row.blob_size,
        "quarantined": quarantined,
        "quarantine_reason": quarantine_reason,
        "raw_preview_cap_bytes": RAW_PREVIEW_MAX_BYTES,
        "raw_preview_included": False,
    }

    if include_raw:
        if row.raw_id is None:
            payload["raw_preview_included"] = True
            payload["raw_preview"] = {
                "available": False,
                "reason": "no_raw_artifact",
                "max_bytes": RAW_PREVIEW_MAX_BYTES,
                "bytes_returned": 0,
                "truncated": False,
            }
        else:
            payload["raw_preview_included"] = True
            payload["raw_preview"] = _build_raw_preview(
                row.raw_id,
                row.blob_size,
                requested_bytes=requested_bytes,
            )

    return payload


__all__ = [
    "RAW_PREVIEW_MAX_BYTES",
    "ProvenanceRow",
    "build_provenance_payload",
    "fetch_provenance_row",
]
