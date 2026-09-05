"""Read-only structural closure law for acquired blob references."""

from __future__ import annotations

import sqlite3


def raw_reference_closure_predicate(raw_alias: str = "r", ref_alias: str = "b") -> str:
    """Return the canonical exact-one raw-payload reference predicate."""
    return f"""
        (
            (
            SELECT COUNT(*) FROM blob_refs {ref_alias}
            WHERE {ref_alias}.ref_type = 'raw_payload'
              AND {ref_alias}.ref_id = {raw_alias}.raw_id
              AND {ref_alias}.blob_hash = {raw_alias}.blob_hash
            ) != 1
            OR (
            SELECT COUNT(*) FROM blob_refs {ref_alias}
            WHERE {ref_alias}.ref_type = 'raw_payload'
              AND {ref_alias}.ref_id = {raw_alias}.raw_id
            ) != 1
        )
    """


def closure_counts(source_conn: sqlite3.Connection, index_conn: sqlite3.Connection) -> dict[str, int]:
    """Return exact structural closure counts without parsing or mutation."""
    raw_missing = int(
        source_conn.execute(
            f"""
            SELECT COUNT(*) FROM raw_sessions r
            WHERE {raw_reference_closure_predicate()}
            """
        ).fetchone()[0]
    )
    attachment_missing = int(
        index_conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE a.acquisition_status = 'acquired'
              AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return {"raw_missing_exact_count": raw_missing, "acquired_attachment_missing_ref_count": attachment_missing}


__all__ = ["closure_counts", "raw_reference_closure_predicate"]
