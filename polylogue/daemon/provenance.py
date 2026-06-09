"""Per-session provenance read surface (#1125).

Surfaces the source artifact and ingest metadata that produced a given
session, with optional bounded access to the raw payload bytes.

Contract:

- Provenance metadata (source path, acquisition timestamp, content hash,
  raw blob size, validation/quarantine state) is always returned for a
  known session. Source paths are sanitized for display — the home
  directory is rendered as ``~`` so absolute filesystem locations do not
  leak unnecessarily into the reader.
- The raw payload preview is opt-in (``?include_raw=1``) and is capped
  at :data:`RAW_PREVIEW_MAX_BYTES`. The cap is enforced server-side and
  cannot be widened by a client query parameter.
- Quarantined / unreadable raw artifacts surface explicitly via the
  ``quarantined`` flag plus a ``quarantine_reason`` string, instead of
  silently omitting the row.

The endpoint reads session metadata from ``index.db`` and, when available,
raw acquisition metadata from ``source.db``.
No archive mutations occur. Raw bytes are read from the content-addressed blob store using
:func:`polylogue.storage.blob_store.get_blob_store`'s ``read_prefix``
helper so the full blob is never materialized for the preview.
"""

from __future__ import annotations

import base64
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from polylogue.paths import active_index_db_path
from polylogue.storage.blob_store import get_blob_store

# Hard server-side cap on the preview window. The endpoint will never
# return more than this many bytes of raw content regardless of what a
# client requests. Sized to bound a single response while still being
# large enough to show typical session headers and the first message.
RAW_PREVIEW_MAX_BYTES: Final[int] = 16 * 1024


@dataclass(frozen=True)
class ProvenanceRow:
    """Joined session + raw row used to assemble a response."""

    session_id: str
    content_hash: str
    raw_id: str | None
    raw_blob_id: str | None
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


def _iso_from_epoch_ms(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        epoch_ms = value
    elif isinstance(value, float):
        epoch_ms = int(value)
    elif isinstance(value, str):
        try:
            epoch_ms = int(value)
        except ValueError:
            return None
    else:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, UTC).isoformat()


def _fetch_archive_provenance_row(archive_db: Path, session_id: str) -> ProvenanceRow | None:
    if not archive_db.exists():
        return None
    source_db = archive_db.with_name("source.db")
    conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if source_db.exists():
            conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
            row = conn.execute(
                """
                SELECT
                    s.session_id              AS session_id,
                    s.origin                  AS source_name,
                    lower(hex(s.content_hash)) AS content_hash,
                    s.raw_id                  AS raw_id,
                    r.source_path             AS source_path,
                    r.blob_hash               AS blob_hash,
                    r.blob_size               AS blob_size,
                    r.acquired_at_ms          AS acquired_at_ms,
                    r.file_mtime_ms           AS file_mtime_ms,
                    r.parsed_at_ms            AS parsed_at_ms,
                    r.parse_error             AS parse_error,
                    r.validated_at_ms         AS validated_at_ms,
                    r.validation_status       AS validation_status,
                    r.validation_error        AS validation_error
                FROM sessions AS s
                LEFT JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                WHERE s.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
            SELECT
                s.session_id              AS session_id,
                s.origin                  AS source_name,
                lower(hex(s.content_hash)) AS content_hash,
                s.raw_id                  AS raw_id,
                NULL                      AS source_path,
                NULL                      AS blob_hash,
                NULL                      AS blob_size,
                NULL                      AS acquired_at_ms,
                NULL                      AS file_mtime_ms,
                NULL                      AS parsed_at_ms,
                NULL                      AS parse_error,
                NULL                      AS validated_at_ms,
                NULL                      AS validation_status,
                NULL                      AS validation_error
            FROM sessions AS s
            WHERE s.session_id = ?
                """,
                (session_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    blob_hash = row["blob_hash"]
    raw_blob_id = bytes(blob_hash).hex() if blob_hash is not None else None
    return ProvenanceRow(
        session_id=str(row["session_id"]),
        source_name=(str(row["source_name"]) if row["source_name"] is not None else None),
        content_hash=str(row["content_hash"] or ""),
        raw_id=(str(row["raw_id"]) if row["raw_id"] is not None else None),
        raw_blob_id=raw_blob_id,
        source_path=(str(row["source_path"]) if row["source_path"] is not None else None),
        blob_size=(int(row["blob_size"]) if row["blob_size"] is not None else None),
        acquired_at=_iso_from_epoch_ms(row["acquired_at_ms"]),
        file_mtime=_iso_from_epoch_ms(row["file_mtime_ms"]),
        parsed_at=_iso_from_epoch_ms(row["parsed_at_ms"]),
        parse_error=(str(row["parse_error"]) if row["parse_error"] is not None else None),
        validated_at=_iso_from_epoch_ms(row["validated_at_ms"]),
        validation_status=(str(row["validation_status"]) if row["validation_status"] is not None else None),
        validation_error=(str(row["validation_error"]) if row["validation_error"] is not None else None),
    )


def fetch_provenance_row(session_id: str) -> ProvenanceRow | None:
    """Load the joined provenance row for *session_id*.

    Returns ``None`` when no session with that id exists. When the
    session exists but has no raw_id (unsupported or hand-seeded rows),
    the raw fields come back as ``None`` and the caller surfaces the
    "no raw artifact" state explicitly.
    """

    dbp = active_index_db_path()
    archive_db = dbp if dbp.name == "index.db" else dbp.with_name("index.db")
    if archive_db.exists():
        return _fetch_archive_provenance_row(archive_db, session_id)
    if not dbp.exists():
        return _fetch_archive_provenance_row(archive_db, session_id)
    conn = sqlite3.connect(str(dbp))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT
                c.session_id   AS session_id,
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
            FROM sessions AS c
            LEFT JOIN raw_sessions AS r ON r.raw_id = c.raw_id
            WHERE c.session_id = ?
            """,
            (session_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return ProvenanceRow(
        session_id=str(row["session_id"]),
        source_name=(str(row["source_name"]) if row["source_name"] is not None else None),
        content_hash=str(row["content_hash"] or ""),
        raw_id=(str(row["raw_id"]) if row["raw_id"] is not None else None),
        raw_blob_id=(str(row["raw_id"]) if row["raw_id"] is not None else None),
        source_path=(str(row["source_path"]) if row["source_path"] is not None else None),
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
    session_id: str,
    *,
    include_raw: bool = False,
    requested_bytes: int | None = None,
) -> dict[str, object] | None:
    """Assemble the JSON payload for ``GET /api/sessions/{id}/provenance``.

    Returns ``None`` when the session does not exist so the caller
    can emit a 404. Otherwise returns the full structured envelope. The
    raw preview is only attached when ``include_raw`` is true, and even
    then is bounded by :data:`RAW_PREVIEW_MAX_BYTES`.
    """

    row = fetch_provenance_row(session_id)
    if row is None:
        return None
    quarantined, quarantine_reason = _quarantine_state(row)

    raw_path = row.source_path
    sanitized_path = _display_source_path(raw_path)
    home = os.path.expanduser("~") or None
    is_absolute = bool(raw_path and Path(raw_path).is_absolute())
    contains_home = bool(home and raw_path and (raw_path == home or raw_path.startswith(home + os.sep)))

    payload: dict[str, object] = {
        "session_id": row.session_id,
        "origin": row.source_name or None,
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
        if row.raw_blob_id is None:
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
                row.raw_blob_id,
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
