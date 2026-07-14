"""Cross-database integrity checks for archive user overlays."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from polylogue.storage.sqlite.introspection import (
    table_exists,
)


@dataclass(frozen=True, slots=True)
class ArchiveUserOverlayOrphan:
    """A user assertion whose target is absent from index.db."""

    table: str
    row_id: str
    target_type: str
    target_id: str


_ARCHIVE_TARGETS: dict[str, tuple[str, str]] = {
    "session": ("sessions", "session_id"),
    "message": ("messages", "message_id"),
    "block": ("blocks", "block_id"),
    "attachment": ("attachments", "attachment_id"),
    "paste_span": ("paste_spans", "paste_id"),
    "work_event": ("session_work_events", "event_id"),
    "phase": ("session_phases", "phase_id"),
    "thread": ("threads", "thread_id"),
}


def find_archive_user_overlay_orphans(
    user_conn: sqlite3.Connection,
    archive_conn: sqlite3.Connection,
) -> tuple[ArchiveUserOverlayOrphan, ...]:
    """Return user assertions whose archive target cannot be resolved."""
    user_conn.row_factory = sqlite3.Row
    archive_conn.row_factory = sqlite3.Row
    orphans: list[ArchiveUserOverlayOrphan] = []
    if not _table_exists(user_conn, "assertions"):
        return ()
    rows = user_conn.execute(
        """
        SELECT assertion_id, target_ref
        FROM assertions
        WHERE kind IN ('mark', 'annotation', 'note')
          AND COALESCE(status, '') != 'deleted'
        ORDER BY assertion_id
        """
    ).fetchall()
    for row in rows:
        target = _split_target_ref(str(row["target_ref"]))
        if target is None:
            continue
        target_type, target_id = target
        if not _target_exists(archive_conn, target_type, target_id):
            orphans.append(
                ArchiveUserOverlayOrphan(
                    table="assertions",
                    row_id=str(row["assertion_id"]),
                    target_type=target_type,
                    target_id=target_id,
                )
            )
    return tuple(orphans)


def _split_target_ref(target_ref: str) -> tuple[str, str] | None:
    if ":" not in target_ref:
        return None
    target_type, target_id = target_ref.split(":", 1)
    if target_type not in _ARCHIVE_TARGETS or not target_id:
        return None
    return target_type, target_id


def _target_exists(conn: sqlite3.Connection, target_type: str, target_id: str) -> bool:
    target_table, target_column = _ARCHIVE_TARGETS[target_type]
    return (
        conn.execute(
            f"SELECT 1 FROM {target_table} WHERE {target_column} = ? LIMIT 1",
            (target_id,),
        ).fetchone()
        is not None
    )


__all__ = ["ArchiveUserOverlayOrphan", "find_archive_user_overlay_orphans"]
