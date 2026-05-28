"""Post-ingest paste-evidence enrichment from UserPromptSubmit hook events.

#1654: hook events written to the hooks sidecar directory by
``polylogue-hook`` carry ground-truth paste markers (``[Pasted text #N]``)
in the UserPromptSubmit payload. These arrive after the conversation
JSONL was ingested, so the initial materialization may have missed them.

This module provides a lightweight enrichment step that reads hook
sidecar JSONL files from disk and updates ``has_paste`` on matching
messages. It is called by the daemon after each live-ingest batch
completes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.paste_detection import has_paste_indicator
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


def enrich_paste_from_hooks(db_path: Path) -> int:
    """Scan hook sidecar files and update has_paste on matching messages.

    Returns the number of messages updated.
    """
    events = _iter_hook_paste_events(hooks_sidecar_dir())
    if not events:
        return 0

    conn = sqlite3.connect(str(db_path))
    updated = 0
    try:
        for event in events:
            session_id = event.get("session_id")
            timestamp_str = event.get("timestamp")
            if not session_id or not timestamp_str:
                continue

            # Parse the hook event timestamp to a unix-epoch float for
            # comparison against messages.sort_key.
            hook_epoch_ms: float = 0.0
            try:
                from datetime import datetime

                hook_ts = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
                hook_epoch_ms = hook_ts.timestamp() * 1000
            except (ValueError, OSError):
                continue
            if hook_epoch_ms <= 0:
                continue

            # Find messages whose conversation carries this session_id
            # and whose sort_key falls within the tolerance window.
            rows = conn.execute(
                """
                SELECT m.message_id
                FROM messages m
                JOIN conversations c ON c.conversation_id = m.conversation_id
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
