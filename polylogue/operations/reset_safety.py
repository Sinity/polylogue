"""Safety predicates shared by reset previews and daemon mutation handlers."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.raw_coordinates import zip_member_container


def unresolvable_raw_source_count(archive_root: Path) -> int:
    """Count retained raw rows whose original source path cannot be reacquired."""
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return 0
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(source_db, validate_schema=False)
    try:
        rows = conn.execute(
            "SELECT source_path, COUNT(*) FROM raw_sessions WHERE source_path IS NOT NULL "
            "AND source_path != '' GROUP BY source_path"
        ).fetchall()
    finally:
        conn.close()

    def reacquirable(source_path: object) -> bool:
        # ZIP rows address members as ``<container>:<member>``. The member
        # suffix is not a filesystem path; check the retained container.
        text = str(source_path)
        return Path(text).exists() or zip_member_container(text) is not None

    return sum(int(count) for source_path, count in rows if not reacquirable(source_path))


__all__ = ["unresolvable_raw_source_count"]
