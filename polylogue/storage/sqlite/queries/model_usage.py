"""Canonical per-session model-usage reads (``session_model_usage`` table).

``session_model_usage`` is the single substrate table both cost/usage rollups
and ``session_profiles`` must agree with (polylogue-r7p6). It is populated
provider-neutrally by ``storage/sqlite/archive_tiers/write.py`` from whichever
usage evidence a given origin actually reports -- Codex-style cumulative
``token_count`` events (disjoint-lane mapped) or per-message token sums.
Session-profile materialization reads it back here instead of recomputing an
independent estimate from per-message fields, which Codex sessions rarely
populate (their real usage arrives as periodic cumulative events, not
per-message ``usage`` blocks) and which previously undercounted Codex token
totals by roughly 1000x.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.archive.semantic.cost_records import ModelUsageTotals

_MODEL_USAGE_BATCH_SQL = """
SELECT session_id, model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
FROM session_model_usage
WHERE session_id IN ({placeholders})
ORDER BY session_id, model_name
"""


def _row_to_model_usage_totals(row: sqlite3.Row) -> ModelUsageTotals:
    return ModelUsageTotals(
        model_name=row["model_name"],
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
        cache_read_tokens=int(row["cache_read_tokens"] or 0),
        cache_write_tokens=int(row["cache_write_tokens"] or 0),
    )


async def get_model_usage_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ModelUsageTotals]]:
    """Batch-read per-session ``session_model_usage`` rows (async)."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            _MODEL_USAGE_BATCH_SQL.format(placeholders=placeholders),
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[ModelUsageTotals]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        result[str(row["session_id"])].append(_row_to_model_usage_totals(row))
    return dict(result)


def sync_model_usage_batch(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ModelUsageTotals]]:
    """Batch-read per-session ``session_model_usage`` rows (sync)."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        _MODEL_USAGE_BATCH_SQL.format(placeholders=placeholders),
        tuple(session_ids),
    ).fetchall()
    result: dict[str, list[ModelUsageTotals]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        result[str(row["session_id"])].append(_row_to_model_usage_totals(row))
    return dict(result)


__all__ = ["get_model_usage_batch", "sync_model_usage_batch"]
