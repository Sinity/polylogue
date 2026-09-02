"""Recompute a session profile's cost lanes from canonical usage evidence.

#4225 removed the duplicated cost and token columns from ``session_profiles``,
making ``session_model_usage`` the single authority. The read path therefore
recomputes the profile's cost lanes rather than reading stored copies, through
:func:`compute_session_cost` -- the same function the materializer uses, so the
provenance vocabulary (``provider_reported`` / ``catalog_priced`` / ``mixed`` /
``unknown``) has exactly one implementation.

The pricing catalog is consulted per model, not per session, and a batch read
groups one ``session_model_usage`` query by session, so a batch of N profiles
costs N dictionary lookups over one query rather than N queries or N catalog
loads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.archive.semantic.cost_compute import compute_session_cost
from polylogue.archive.semantic.cost_records import ModelUsageTotals

if TYPE_CHECKING:
    from polylogue.storage.insights.session.records import SessionProfileRecord

__all__ = [
    "ProfileCostLanes",
    "apply_profile_cost_lanes",
    "profile_cost_lanes",
    "read_model_usage_batch_async",
    "read_model_usage_batch_sync",
]

_MODEL_USAGE_SQL = """
SELECT session_id, model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
FROM session_model_usage
WHERE session_id IN ({placeholders})
"""


class ProfileCostLanes(dict[str, object]):
    """The recomputed cost fields overlaid onto a ``SessionProfileRecord``."""


def _group(rows: Iterable[Sequence[object]]) -> dict[str, list[ModelUsageTotals]]:
    grouped: dict[str, list[ModelUsageTotals]] = {}
    for row in rows:
        grouped.setdefault(str(row[0]), []).append(
            ModelUsageTotals(
                model_name=str(row[1] or ""),
                input_tokens=int(row[2] or 0),
                output_tokens=int(row[3] or 0),
                cache_read_tokens=int(row[4] or 0),
                cache_write_tokens=int(row[5] or 0),
            )
        )
    return grouped


def read_model_usage_batch_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ModelUsageTotals]]:
    """One query for every session's usage rows, grouped by session."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(_MODEL_USAGE_SQL.format(placeholders=placeholders), tuple(session_ids)).fetchall()
    return _group(rows)


async def read_model_usage_batch_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ModelUsageTotals]]:
    """Async sibling of :func:`read_model_usage_batch_sync`."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    cursor = await conn.execute(_MODEL_USAGE_SQL.format(placeholders=placeholders), tuple(session_ids))
    return _group(await cursor.fetchall())


def profile_cost_lanes(model_usage: Sequence[ModelUsageTotals]) -> ProfileCostLanes:
    """Cost and token lanes for one session, from its usage rows alone.

    A session with no usage rows produces no breakdowns, which
    :func:`compute_session_cost` reports as ``unknown`` provenance and zero
    cost -- absent evidence, not a known-zero bill.
    """
    summary = compute_session_cost(None, model_usage=model_usage, estimate_if_missing=False)
    return ProfileCostLanes(
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_cache_read_tokens=summary.total_cache_read_tokens,
        total_cache_write_tokens=summary.total_cache_write_tokens,
        total_cost_usd=summary.total_api_cost_usd,
        total_credit_cost=summary.total_credit_cost,
        cost_provenance=summary.cost_provenance,
        cost_is_estimated=summary.cost_provenance != "provider_reported",
    )


def apply_profile_cost_lanes(
    record: SessionProfileRecord,
    model_usage: Mapping[str, Sequence[ModelUsageTotals]],
) -> SessionProfileRecord:
    """Return ``record`` with its cost lanes recomputed from ``model_usage``."""
    lanes = profile_cost_lanes(model_usage.get(str(record.session_id), ()))
    return record.model_copy(update=lanes)
