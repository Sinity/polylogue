"""Post-ingest paste-evidence enrichment from UserPromptSubmit hook events.

#1654: hook events written to the hooks sidecar directory by
``polylogue-hook`` carry ground-truth paste markers (``[Pasted text #N]``)
in the UserPromptSubmit payload. These arrive after the session
JSONL was ingested, so the initial materialization may have missed them.

This module provides a lightweight enrichment step that reads hook
sidecar JSONL files from disk and updates ``has_paste`` on matching
messages. It is called by the daemon after each live-ingest batch
completes.
"""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path

from polylogue.archive.message.paste_detection import has_paste_indicator
from polylogue.core.enums import PasteBoundary
from polylogue.logging import get_logger
from polylogue.paths import hooks_sidecar_dir

logger = get_logger(__name__)

#: Only UserPromptSubmit events carry paste ground truth.
_PASTE_EVENT_TYPE = "UserPromptSubmit"
#: Tolerance for matching a hook event timestamp to a message sort_key
#: (milliseconds — same as history.jsonl matching in assembly_claude_code.py).
_TIMESTAMP_TOLERANCE_MS = 3000


def _iter_hook_paste_events(hooks_dir: Path) -> list[dict[str, object]]:
    """Scan hook sidecar JSONL files, return UserPromptSubmit events with paste."""
    events: list[dict[str, object]] = []
    if not hooks_dir.exists():
        return events
    for jsonl_path in hooks_dir.glob("*.jsonl"):
        try:
            with open(jsonl_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if record.get("event_type") != _PASTE_EVENT_TYPE:
                        continue
                    if not has_paste_indicator(record):
                        continue
                    events.append(record)
        except OSError:
            logger.debug("hook_paste: could not read %s", jsonl_path, exc_info=True)
    return events


def _hook_field(event: dict[str, object], key: str) -> object:
    value = event.get(key)
    if value:
        return value
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload.get(key)
    return None


def _hook_epoch_ms(event: dict[str, object]) -> float:
    timestamp = _hook_field(event, "timestamp")
    if not timestamp:
        return 0.0
    try:
        from datetime import datetime

        hook_ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return hook_ts.timestamp() * 1000
    except (ValueError, OSError):
        return 0.0


def _archive_index_path(db_path: Path) -> Path | None:
    if db_path.name == "index.db":
        return db_path if db_path.exists() else None
    index_db = db_path.with_name("index.db")
    return index_db if index_db.exists() else None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _enrich_archive_paste_from_hooks(index_db: Path, events: list[dict[str, object]]) -> int:
    conn = sqlite3.connect(str(index_db))
    updated = 0
    updated_sessions: set[str] = set()
    try:
        if not _table_exists(conn, "sessions") or not _table_exists(conn, "messages"):
            return 0
        for event in events:
            session_id = _hook_field(event, "session_id")
            hook_epoch_ms = _hook_epoch_ms(event)
            if not session_id or hook_epoch_ms <= 0:
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
                            message_id, session_id, start_offset, end_offset, content_hash, boundary
                        ) VALUES (?, ?, 0, 0, ?, ?)
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


def enrich_paste_from_hooks(db_path: Path) -> int:
    """Scan hook sidecar files and update has_paste on matching messages.

    Returns the number of messages updated.
    """
    events = _iter_hook_paste_events(hooks_sidecar_dir())
    if not events:
        return 0

    archive_index = _archive_index_path(db_path)
    if archive_index is not None:
        updated = _enrich_archive_paste_from_hooks(archive_index, events)
        if updated:
            logger.info("hook_paste: enriched %d archive message(s) from hook sidecar events", updated)
        return updated

    conn = sqlite3.connect(str(db_path))
    updated = 0
    try:
        for event in events:
            session_id = _hook_field(event, "session_id")
            hook_epoch_ms = _hook_epoch_ms(event)
            if not session_id or hook_epoch_ms <= 0:
                continue

            # Find messages whose session carries this session_id
            # and whose sort_key falls within the tolerance window.
            rows = conn.execute(
                """
                SELECT m.message_id
                FROM messages m
                JOIN sessions c ON c.session_id = m.session_id
                WHERE c.provider_meta IS NOT NULL
                  AND json_extract(c.provider_meta, '$.session_id') = ?
                  AND m.role = 'user'
                  AND m.has_paste = 0
                  AND abs(m.sort_key - ?) < ?
                LIMIT 1
                """,
                (str(session_id), hook_epoch_ms, _TIMESTAMP_TOLERANCE_MS),
            ).fetchall()

            for (message_id,) in rows:
                conn.execute(
                    "UPDATE messages SET has_paste = 1 WHERE message_id = ?",
                    (message_id,),
                )
                updated += 1

        if updated:
            conn.commit()
    finally:
        conn.close()

    if updated:
        logger.info("hook_paste: enriched %d message(s) from hook sidecar events", updated)
    return updated
