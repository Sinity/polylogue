"""Source-file cursor persistence queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


async def get_known_source_cursors(
    conn: aiosqlite.Connection,
) -> dict[str, dict[str, object]]:
    """Return {source_path: {st_dev, st_ino, st_size, mtime_ns}} for cursor tracking."""
    result: dict[str, dict[str, object]] = {}
    cursor = await conn.execute("SELECT source_path, st_dev, st_ino, st_size, mtime_ns FROM source_file_cursor")
    while True:
        rows = await cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            fields: dict[str, object] = {}
            for col in ("st_dev", "st_ino", "st_size", "mtime_ns"):
                val = row[col]
                if val is not None:
                    fields[col] = val
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
    """Upsert one source file cursor row."""
    await conn.execute(
        """
        INSERT INTO source_file_cursor (source_path, st_dev, st_ino, st_size, mtime_ns)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            st_dev = COALESCE(EXCLUDED.st_dev, source_file_cursor.st_dev),
            st_ino = COALESCE(EXCLUDED.st_ino, source_file_cursor.st_ino),
            st_size = COALESCE(EXCLUDED.st_size, source_file_cursor.st_size),
            mtime_ns = COALESCE(EXCLUDED.mtime_ns, source_file_cursor.mtime_ns)
        """,
        (source_path, st_dev, st_ino, st_size, mtime_ns),
    )
    if transaction_depth == 0:
        await conn.commit()


__all__ = [
    "get_known_source_cursors",
    "upsert_source_file_cursor",
]
