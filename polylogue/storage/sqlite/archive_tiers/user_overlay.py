"""Cross-database integrity checks for archive user overlays."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArchiveUserOverlayOrphan:
    """A user overlay row whose target is absent from index.db."""

    table: str
    row_id: str
    target_type: str
    target_id: str


_ARCHIVE_TARGETS: dict[str, tuple[str, str]] = {
    "session": ("sessions", "session_id"),
    "message": ("messages", "message_id"),
    "block": ("blocks", "block_id"),
    "attachment": ("attachments", "attachment_id"),
    "paste_span": ("paste_spans", "paste_span_id"),
    "work_event": ("session_work_events", "event_id"),
    "phase": ("session_phases", "phase_id"),
    "thread": ("threads", "thread_id"),
}


def find_archive_user_overlay_orphans(
    user_conn: sqlite3.Connection,
    archive_conn: sqlite3.Connection,
) -> tuple[ArchiveUserOverlayOrphan, ...]:
    """Return mark/annotation rows whose archive target cannot be resolved."""
    user_conn.row_factory = sqlite3.Row
    archive_conn.row_factory = sqlite3.Row
    orphans: list[ArchiveUserOverlayOrphan] = []
    for table, id_column in (("marks", "mark_id"), ("annotations", "annotation_id")):
        orphans.extend(_orphans_for_table(user_conn, archive_conn, table=table, id_column=id_column))
    orphans.extend(
        _orphans_for_table(
            user_conn,
            archive_conn,
            table="blackboard_notes",
            id_column="note_id",
            skip_global_targets=True,
        )
    )
    return tuple(orphans)


def _orphans_for_table(
    user_conn: sqlite3.Connection,
    archive_conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    skip_global_targets: bool = False,
) -> Iterable[ArchiveUserOverlayOrphan]:
    if not _table_exists(user_conn, table):
        return
    rows = user_conn.execute(
        f"""
        SELECT {id_column} AS row_id, target_type, target_id
        FROM {table}
        ORDER BY {id_column}
        """
    ).fetchall()
    for row in rows:
        if skip_global_targets and row["target_type"] is None and row["target_id"] is None:
            continue
        if row["target_type"] is None or row["target_id"] is None:
            yield ArchiveUserOverlayOrphan(
                table=table,
                row_id=str(row["row_id"]),
                target_type=str(row["target_type"]),
                target_id=str(row["target_id"]),
            )
            continue
        target = _ARCHIVE_TARGETS.get(str(row["target_type"]))
        if target is None:
            yield ArchiveUserOverlayOrphan(
                table=table,
                row_id=str(row["row_id"]),
                target_type=str(row["target_type"]),
                target_id=str(row["target_id"]),
            )
            continue
        target_table, target_column = target
        found = archive_conn.execute(
            f"SELECT 1 FROM {target_table} WHERE {target_column} = ? LIMIT 1",
            (row["target_id"],),
        ).fetchone()
        if found is None:
            yield ArchiveUserOverlayOrphan(
                table=table,
                row_id=str(row["row_id"]),
                target_type=str(row["target_type"]),
                target_id=str(row["target_id"]),
            )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


__all__ = ["ArchiveUserOverlayOrphan", "find_archive_user_overlay_orphans"]
