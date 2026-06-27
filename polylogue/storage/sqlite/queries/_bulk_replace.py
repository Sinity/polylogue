"""Shared DELETE-then-INSERT helper for durable insight write queries.

Several ``replace_*_bulk`` functions in this package follow the same
shape: DELETE rows whose id column is in a caller-supplied set, then
``executemany`` an INSERT built from a caller-supplied column list and
extractor, then commit when ``transaction_depth == 0``.

:func:`replace_insight_rows` consolidates that boilerplate. Per-site
code becomes a thin wrapper that supplies the dynamic column list and
the per-record value extractor — typically 10-15 lines instead of 25-35.

The helper is intentionally faithful to the previous handwritten SQL:
one ``DELETE ... WHERE <id_column> IN (?, ?, ...)`` statement using
every id value verbatim (no de-duplication, no chunking), and one
``INSERT`` produced by :func:`build_insert_sql`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import aiosqlite

from polylogue.storage.insights.session.storage import build_insert_sql

__all__ = ["replace_insight_rows"]

T = TypeVar("T")


async def replace_insight_rows(
    conn: aiosqlite.Connection,
    *,
    table: str,
    id_column: str,
    id_values: Sequence[str],
    columns: Sequence[str],
    records: Sequence[T],
    extractor: Callable[[T], tuple[Any, ...]],
    transaction_depth: int,
    or_replace: bool = False,
) -> None:
    """DELETE rows matching ``id_values`` then INSERT ``records``.

    ``id_values`` becomes the operand of ``WHERE {id_column} IN (...)``.
    When empty, the DELETE is skipped entirely (no ``IN ()`` is emitted).

    ``records`` are passed one-by-one through ``extractor`` to produce
    the tuple bound to ``executemany``. When empty, no INSERT runs.

    The ``transaction_depth == 0`` commit mirrors the per-site rule:
    nested callers (depth > 0) defer commit to the outermost frame.
    """
    if id_values:
        placeholders = ", ".join("?" for _ in id_values)
        await conn.execute(
            f"DELETE FROM {table} WHERE {id_column} IN ({placeholders})",
            tuple(id_values),
        )
    if records:
        await conn.executemany(
            build_insert_sql(table, columns, or_replace=or_replace),
            [extractor(record) for record in records],
        )
    if transaction_depth == 0:
        await conn.commit()
