"""User-tier assertion overlay audit helpers (#1883)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.core.json import JSONDocument, json_document
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind

_ASSERTION_BACKED_SURFACES: Mapping[str, AssertionKind] = {
    "marks": AssertionKind.MARK,
    "annotations": AssertionKind.ANNOTATION,
    "suppressions": AssertionKind.SUPPRESSION,
    "corrections": AssertionKind.CORRECTION,
    "saved_views": AssertionKind.SAVED_QUERY,
    "recall_packs": AssertionKind.RECALL_PACK,
    "workspaces": AssertionKind.WORKSPACE_NOTE,
    "blackboard_notes": AssertionKind.NOTE,
    "transform_candidates": AssertionKind.TRANSFORM_CANDIDATE,
}

_TABLE_BACKED_SURFACES: Mapping[str, tuple[str, str]] = {
    "session_tags": (
        "session_tags",
        "simple session metadata table; not an assertion lifecycle claim",
    ),
    "session_metadata": (
        "session_metadata",
        "simple key/value session metadata table; not an assertion lifecycle claim",
    ),
}

_REMOVED_OVERLAY_TABLES: tuple[str, ...] = (
    "marks",
    "annotations",
    "corrections",
    "suppressions",
    "saved_views",
    "recall_packs",
    "workspaces",
    "blackboard_notes",
)


@dataclass(frozen=True, slots=True)
class UserOverlayAuditSurface:
    """Audit row for one user-overlay surface."""

    name: str
    storage: str
    assertion_kind: str | None
    table: str | None
    total_count: int
    active_count: int
    deleted_count: int
    candidate_count: int
    rationale: str

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "storage": self.storage,
                "assertion_kind": self.assertion_kind,
                "table": self.table,
                "total_count": self.total_count,
                "active_count": self.active_count,
                "deleted_count": self.deleted_count,
                "candidate_count": self.candidate_count,
                "rationale": self.rationale,
            }
        )


@dataclass(frozen=True, slots=True)
class UserOverlayAudit:
    """Live audit of how user-overlay surfaces are stored."""

    surfaces: tuple[UserOverlayAuditSurface, ...]
    legacy_tables_present: tuple[str, ...]

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "surfaces": [surface.to_dict() for surface in self.surfaces],
                "legacy_tables_present": list(self.legacy_tables_present),
            }
        )


def audit_user_overlay_storage(conn: sqlite3.Connection) -> UserOverlayAudit:
    """Return the live user-overlay storage audit for a user-tier connection."""

    surfaces: list[UserOverlayAuditSurface] = []
    kind_counts = _assertion_counts_by_kind(conn)
    for surface_name, kind in _ASSERTION_BACKED_SURFACES.items():
        counts = kind_counts.get(kind.value, {})
        total = sum(counts.values())
        active = counts.get("active", 0) + counts.get("", 0)
        surfaces.append(
            UserOverlayAuditSurface(
                name=surface_name,
                storage="assertions",
                assertion_kind=kind.value,
                table="assertions",
                total_count=total,
                active_count=active,
                deleted_count=counts.get("deleted", 0),
                candidate_count=counts.get("candidate", 0),
                rationale="assertion lifecycle claim",
            )
        )

    for surface_name, (table_name, rationale) in _TABLE_BACKED_SURFACES.items():
        count = _table_row_count(conn, table_name)
        surfaces.append(
            UserOverlayAuditSurface(
                name=surface_name,
                storage="table",
                assertion_kind=None,
                table=table_name,
                total_count=count,
                active_count=count,
                deleted_count=0,
                candidate_count=0,
                rationale=rationale,
            )
        )

    legacy_tables_present = tuple(table for table in _REMOVED_OVERLAY_TABLES if _table_exists(conn, table))
    return UserOverlayAudit(surfaces=tuple(surfaces), legacy_tables_present=legacy_tables_present)


def _assertion_counts_by_kind(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    if not _table_exists(conn, "assertions"):
        return {}
    rows = conn.execute(
        """
        SELECT kind, COALESCE(status, '') AS status, COUNT(*) AS count
        FROM assertions
        GROUP BY kind, COALESCE(status, '')
        """
    ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        kind = str(row[0])
        status = str(row[1])
        counts.setdefault(kind, {})[status] = int(row[2] or 0)
    return counts


def _table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


__all__ = [
    "UserOverlayAudit",
    "UserOverlayAuditSurface",
    "audit_user_overlay_storage",
]
