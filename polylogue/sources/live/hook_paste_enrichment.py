"""Post-ingest paste-evidence enrichment from UserPromptSubmit hook events.

#1654: hook events carry ground-truth paste markers (``[Pasted text #N]``) in
the UserPromptSubmit payload. They arrive after the session JSONL was
ingested, so the initial materialization may have missed them.

The evidence is read from the durable source tier's ``raw_hook_events`` rows,
whose ``payload_json`` is the whole spool envelope the drain committed. That
is the only carrier an event keeps for its whole life: a pending envelope is
moved to ``acknowledged`` the moment the drain commits it, so enriching from
the spool would see an event exactly once and never again. The consequence is
a drain-cadence lag -- an event enriches on the first batch after the drain
that admitted it, not on the batch that ingested its session -- which the
per-batch rescan and the whole-archive pass both absorb.

It is called by the daemon after each live-ingest batch completes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from polylogue.archive.message.paste_detection import has_paste_indicator
from polylogue.core.enums import PasteBoundary
from polylogue.core.hook_payload import hook_record_field, matched_reader_keys
from polylogue.logging import get_logger
from polylogue.storage.introspection import table_exists as _table_exists

logger = get_logger(__name__)

#: Only UserPromptSubmit events carry paste ground truth.
_PASTE_EVENT_TYPE = "UserPromptSubmit"
#: Tolerance for matching a hook event timestamp to a message sort_key
#: (milliseconds — same as history.jsonl matching in assembly_claude_code.py).
_TIMESTAMP_TOLERANCE_MS = 3000


def _scoped_hook_keys(session_ids: Iterable[str]) -> list[tuple[str, str]]:
    """``(origin, native_id)`` pairs addressing one session's hook events each.

    ``raw_hook_events`` is indexed on ``(origin, session_native_id, ...)``, so
    an archive session id is split back into both halves rather than matched
    on the native id alone: the leading column is what keeps the scan bounded
    by the batch instead of by the archive's whole hook history.
    """
    keys: dict[tuple[str, str], None] = {}
    for session_id in session_ids:
        origin, separator, native_id = str(session_id).partition(":")
        if not separator or not origin or not native_id:
            continue
        keys[(origin, native_id)] = None
    return list(keys)


def _iter_hook_paste_events(source_db: Path, session_ids: Iterable[str] | None = None) -> list[dict[str, object]]:
    """Durable UserPromptSubmit hook envelopes that carry paste evidence.

    ``session_ids`` (archive ``origin:native_id`` ids) bounds the read to those
    sessions' events; ``None`` reads every UserPromptSubmit event in the tier.
    """
    if not source_db.exists():
        return []
    events: list[dict[str, object]] = []
    try:
        connection = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.debug("hook_paste: could not open %s", source_db, exc_info=True)
        return []
    try:
        if not _table_exists(connection, "raw_hook_events"):
            return []
        queries: list[tuple[str, tuple[str, ...]]]
        if session_ids is None:
            queries = [("SELECT payload_json FROM raw_hook_events WHERE event_type = ?", (_PASTE_EVENT_TYPE,))]
        else:
            keys = _scoped_hook_keys(session_ids)
            if not keys:
                return []
            queries = [
                (
                    "SELECT payload_json FROM raw_hook_events "
                    "WHERE origin = ? AND session_native_id = ? AND event_type = ?",
                    (origin, native_id, _PASTE_EVENT_TYPE),
                )
                for origin, native_id in keys
            ]
        for sql, parameters in queries:
            for (payload_json,) in connection.execute(sql, parameters):
                record = _decode_envelope(payload_json)
                if record is None:
                    continue
                if not has_paste_indicator(record):
                    continue
                events.append(record)
    except sqlite3.Error:
        logger.debug("hook_paste: could not read hook events from %s", source_db, exc_info=True)
    finally:
        connection.close()
    return events


def _decode_envelope(payload_json: object) -> dict[str, object] | None:
    if not isinstance(payload_json, str):
        return None
    try:
        record = json.loads(payload_json)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _hook_epoch_ms(event: dict[str, object]) -> float:
    timestamp = hook_record_field(event, "timestamp")
    if not timestamp:
        return 0.0
    try:
        from datetime import datetime

        hook_ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return hook_ts.timestamp() * 1000
    except (ValueError, OSError):
        return 0.0


def _archive_index_path(db_path: Path) -> Path | None:
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    return index_db if index_db.exists() else None


def _archive_source_path(db_path: Path) -> Path:
    from polylogue.storage.archive_identity import ArchiveLocation

    return ArchiveLocation.resolve(db_path.parent).active_tier("source").configured_path


def _enrich_archive_paste_from_hooks(index_db: Path, events: list[dict[str, object]]) -> int:
    conn = sqlite3.connect(str(index_db))
    updated = 0
    updated_sessions: set[str] = set()
    try:
        if not _table_exists(conn, "sessions") or not _table_exists(conn, "messages"):
            return 0
        for event in events:
            session_id = hook_record_field(event, "session_id")
            hook_epoch_ms = _hook_epoch_ms(event)
            if not session_id or hook_epoch_ms <= 0:
                # Ground-truth paste evidence that cannot be keyed is dropped
                # here, and a key-name mismatch reads exactly like a field that
                # was never sent. Naming the keys that did resolve separates
                # "this generation is undescribed" from "the field is absent".
                payload = event.get("payload")
                logger.warning(
                    "hook_paste: %s record carries paste evidence but no readable session key; "
                    "reader keys matched=%s payload keys=%s",
                    hook_record_field(event, "event_type"),
                    sorted(matched_reader_keys(event)),
                    sorted(payload) if isinstance(payload, dict) else [],
                )
                continue
            rows = conn.execute(
                """
                SELECT m.message_id, m.session_id
                FROM messages AS m
                JOIN sessions AS s ON s.session_id = m.session_id
                WHERE (s.session_id = ? OR s.native_id = ?)
                  AND m.role = 'user'
                  AND m.has_paste = 0
                  AND m.occurred_at_ms IS NOT NULL
                  AND abs(m.occurred_at_ms - ?) < ?
                ORDER BY abs(m.occurred_at_ms - ?), m.position
                LIMIT 1
                """,
                (
                    str(session_id),
                    str(session_id),
                    hook_epoch_ms,
                    _TIMESTAMP_TOLERANCE_MS,
                    hook_epoch_ms,
                ),
            ).fetchall()
            for message_id, matched_session_id in rows:
                conn.execute(
                    """
                    UPDATE messages
                    SET has_paste = 1,
                        paste_boundary = COALESCE(paste_boundary, ?)
                    WHERE message_id = ?
                    """,
                    (PasteBoundary.HASH_ONLY.value, message_id),
                )
                if _table_exists(conn, "paste_spans"):
                    content_hash = sha256(f"hook-paste\0{message_id}".encode()).digest()
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO paste_spans (
                            message_id, session_id, position, start_offset, end_offset, content_hash, boundary_state
                        ) VALUES (?, ?, 0, 0, 0, ?, ?)
                        """,
                        (message_id, matched_session_id, content_hash, PasteBoundary.HASH_ONLY.value),
                    )
                updated_sessions.add(str(matched_session_id))
                updated += 1
        for session_id in updated_sessions:
            conn.execute(
                """
                UPDATE sessions
                SET paste_count = (
                    SELECT COUNT(*) FROM messages WHERE session_id = ? AND has_paste = 1
                )
                WHERE session_id = ?
                """,
                (session_id, session_id),
            )
        if updated:
            conn.commit()
    finally:
        conn.close()
    return updated


def enrich_paste_from_hooks(db_path: Path, *, session_ids: Iterable[str] | None = None) -> int:
    """Read durable hook events and update has_paste on matching messages.

    ``db_path`` is the caller's ops.db path (``archive_root / "ops.db"``), so
    both tiers are derived from its parent rather than from an ambient global
    (polylogue-o7hx): this enrichment always inspects the archive it was
    actually called for, never a different one that happens to be the
    process-wide default.

    ``session_ids`` (archive ``origin:native_id`` ids) bounds the read to those
    sessions' hook events; ``None`` reads every UserPromptSubmit event.

    Returns the number of messages updated.
    """
    events = _iter_hook_paste_events(_archive_source_path(db_path), session_ids)
    if not events:
        return 0

    archive_index = _archive_index_path(db_path)
    if archive_index is None:
        return 0
    updated = _enrich_archive_paste_from_hooks(archive_index, events)
    if updated:
        logger.info("hook_paste: enriched %d archive message(s) from hook events", updated)
    return updated
