"""Safety predicates shared by reset previews and daemon mutation handlers."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.raw_coordinates import zip_member_container


def unresolvable_raw_source_count(archive_root: Path) -> int:
    """Count retained raw rows whose original source path cannot be reacquired.

    A ZIP member is reacquirable only when replaying it from the retained
    container still yields the recorded payload: a container that was replaced
    by another valid ZIP without the member, or with different member bytes,
    no longer holds the evidence ``source.db`` is the only copy of.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return 0
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(source_db, validate_schema=False)
    try:
        rows = conn.execute(
            "SELECT source_path, lower(hex(blob_hash)), COUNT(*) FROM raw_sessions WHERE source_path IS NOT NULL "
            "AND source_path != '' GROUP BY source_path, blob_hash"
        ).fetchall()
    finally:
        conn.close()

    at_risk = 0
    member_rows: dict[str, int] = {}
    for source_path, blob_hash, count in rows:
        text = str(source_path)
        if Path(text).exists():
            continue
        # ZIP rows address members as ``<container>:<member>``. The member
        # suffix is not a filesystem path; the container must hold it.
        if zip_member_container(text) is None:
            at_risk += int(count)
            continue
        member_rows[str(blob_hash)] = member_rows.get(str(blob_hash), 0) + int(count)
    if member_rows:
        from polylogue.daemon.backup import _source_recoverability_proofs

        proven = {
            proof["blob_hash"]
            for proof in _source_recoverability_proofs(
                source_db, root=archive_root, missing_hashes=set(member_rows), immutable=False
            )
        }
        at_risk += sum(count for blob_hash, count in member_rows.items() if blob_hash not in proven)
    return at_risk


__all__ = ["unresolvable_raw_source_count"]
