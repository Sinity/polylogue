"""Safety predicates shared by reset previews and daemon mutation handlers."""

from __future__ import annotations

from pathlib import Path


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
    return sum(int(count) for source_path, count in rows if not Path(str(source_path)).exists())


__all__ = ["unresolvable_raw_source_count"]
