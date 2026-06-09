"""Source-file cursor persistence queries.

File-stat cursor tracking (device/inode/size/mtime) shares the unified
``ingest_cursor`` table (ops tier) with the daemon's richer cursor state. These
helpers only read/write the stat columns; ``ON CONFLICT`` preserves the other
cursor fields (byte offsets, fingerprints, record counts) the daemon manages.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


def _now_ms() -> int:
    return int(time.time() * 1000)


async def get_known_source_cursors(
    conn: aiosqlite.Connection,
) -> dict[str, dict[str, object]]:
    """Return {source_path: {st_dev, st_ino, st_size, mtime_ns}} for cursor tracking."""
    result: dict[str, dict[str, object]] = {}
    cursor = await conn.execute("SELECT source_path, st_dev, st_ino, stat_size, mtime_ns FROM ingest_cursor")
    while True:
        rows = list(await cursor.fetchmany(1000))
        if not rows:
            break
        for row in rows:
            fields: dict[str, object] = {}
            for out_col, db_col in (
                ("st_dev", "st_dev"),
                ("st_ino", "st_ino"),
                ("st_size", "stat_size"),
                ("mtime_ns", "mtime_ns"),
            ):
                val = row[db_col]
                if val is not None:
                    fields[out_col] = val
            if fields:
                result[row["source_path"]] = fields
    return result


async def upsert_source_file_cursor(
    conn: aiosqlite.Connection,
    source_path: str,
    *,
    st_dev: int | None = None,
    st_ino: int | None = None,
    st_size: int | None = None,
    mtime_ns: int | None = None,
    transaction_depth: int,
) -> None:
    """Upsert the stat columns of one ``ingest_cursor`` row."""
    await conn.execute(
        """
        INSERT INTO ingest_cursor (source_path, st_dev, st_ino, stat_size, mtime_ns, updated_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            st_dev = COALESCE(EXCLUDED.st_dev, ingest_cursor.st_dev),
            st_ino = COALESCE(EXCLUDED.st_ino, ingest_cursor.st_ino),
            stat_size = COALESCE(EXCLUDED.stat_size, ingest_cursor.stat_size),
            mtime_ns = COALESCE(EXCLUDED.mtime_ns, ingest_cursor.mtime_ns),
            updated_at_ms = EXCLUDED.updated_at_ms
        """,
        (source_path, st_dev, st_ino, st_size, mtime_ns, _now_ms()),
    )
    if transaction_depth == 0:
        await conn.commit()


__all__ = [
    "get_known_source_cursors",
    "upsert_source_file_cursor",
]
