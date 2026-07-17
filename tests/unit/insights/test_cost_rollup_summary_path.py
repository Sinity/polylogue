"""Regression: cost_rollups uses typed summary cost rows (#1671).

Before #1671 ``list_cost_rollup_insights`` loaded every Session with
its full message stream so it could call ``estimate_session_cost``.
On a 7.9K-session archive that pulled 3.7M message rows into Python
and hung the MCP stdio loop past its 60s deadline (#1621).

The current path reads only typed session/profile cost rows and model-usage
rows. Sessions without typed cost evidence surface as ``status="unavailable"``
rather than triggering message hydration.

The contract this test pins:

* ``list_cost_rollup_insights`` aggregates typed cost rows — a session whose
  profile carries ``cost_usd`` contributes that exact amount to the rollup.
* The rollup completes without loading message bodies (no
  ``SessionRepository.list`` / ``get_many`` calls); the only
  session read is ``list_summaries_by_query``.
* Sessions whose profile lacks cost evidence appear in the rollup as
  ``status="unavailable"`` rather than triggering an expensive
  message-hydration fallback.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.api import Polylogue
from polylogue.insights.archive import CostRollupInsightQuery
from polylogue.storage.sqlite.archive_tiers.write import upsert_session_profile_costs
from tests.infra.storage_records import SessionBuilder


@pytest.mark.asyncio
async def test_cost_rollups_aggregate_typed_cost_rows_without_message_load(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-priced")
        .provider("claude-code")
        .title("Priced session")
        .updated_at("2026-03-01T10:00:00+00:00")
        .add_message("u1", role="user", text="hello")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-no-cost")
        .provider("claude-code")
        .title("No cost evidence")
        .updated_at("2026-03-01T11:00:00+00:00")
        .add_message("u1", role="user", text="hello")
        .save()
    )
    priced_id = "conv-priced"
    no_cost_id = "conv-no-cost"
    with sqlite3.connect(db_path) as conn:
        priced_row = conn.execute(
            "SELECT session_id FROM sessions WHERE native_id = ?",
            ("ext-conv-priced",),
        ).fetchone()
        no_cost_row = conn.execute(
            "SELECT session_id FROM sessions WHERE native_id = ?",
            ("ext-conv-no-cost",),
        ).fetchone()
        assert priced_row is not None
        assert no_cost_row is not None
        priced_id = str(priced_row[0])
        no_cost_id = str(no_cost_row[0])
        upsert_session_profile_costs(
            conn,
            priced_id,
            cost_usd=2.50,
            cost_is_estimated=False,
            cost_provenance="exact",
            priced_with="origin-reported",
            priced_at_ms=1_772_360_400_000,
        )
        conn.execute(
            """
            INSERT INTO session_model_usage (session_id, model_name, cost_provenance)
            VALUES (?, 'claude-sonnet-4-5', 'origin_reported')
            """,
            (priced_id,),
        )
        upsert_session_profile_costs(
            conn,
            no_cost_id,
            cost_usd=0.0,
            cost_is_estimated=True,
            cost_provenance="none",
        )

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    # The archive cost rollup reads typed session_profiles/session_model_usage
    # facts without loading message bodies.

    target = "polylogue.storage.repository.archive.sessions.RepositoryArchiveSessionMixin.list_by_query"
    with patch(target, side_effect=AssertionError("messages loaded")) as spy:
        rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(origin="claude-code-session"))
        assert spy.call_count == 0

    total = sum(rollup.total_usd for rollup in rollups)
    assert total == pytest.approx(2.50)
    merged_status_counts: dict[str, int] = {}
    for rollup in rollups:
        for key, value in rollup.status_counts.items():
            merged_status_counts[key] = merged_status_counts.get(key, 0) + value
    assert merged_status_counts.get("exact", 0) == 1
    assert merged_status_counts.get("unavailable", 0) == 1
    unavailable = next(rollup for rollup in rollups if rollup.unavailable_session_count)
    assert unavailable.priced_session_count == 0
    assert unavailable.confidence is None
