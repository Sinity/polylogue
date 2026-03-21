"""Publication manifest queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _parse_json
from polylogue.storage.store import PublicationRecord, _json_or_none

__all__ = [
    "get_latest_publication",
    "record_publication",
]


async def get_latest_publication(
    conn: aiosqlite.Connection,
    publication_kind: str,
) -> PublicationRecord | None:
    """Fetch the most recent publication record for one publication kind."""
    cursor = await conn.execute(
        """
        SELECT *
        FROM publications
        WHERE publication_kind = ?
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        (publication_kind,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return PublicationRecord(
        publication_id=row["publication_id"],
        publication_kind=row["publication_kind"],
        generated_at=row["generated_at"],
        output_dir=row["output_dir"],
        duration_ms=row["duration_ms"],
        manifest=_parse_json(row["manifest_json"], field="manifest_json", record_id=row["publication_id"]),
    )


async def record_publication(
    conn: aiosqlite.Connection,
    record: PublicationRecord,
    transaction_depth: int,
) -> None:
    """Persist one publication manifest."""
    await conn.execute(
        """
        INSERT INTO publications (
            publication_id,
            publication_kind,
            generated_at,
            output_dir,
            duration_ms,
            manifest_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.publication_id,
            record.publication_kind,
            record.generated_at,
            record.output_dir,
            record.duration_ms,
            _json_or_none(record.manifest),
        ),
    )
    if transaction_depth == 0:
        await conn.commit()
