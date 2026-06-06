"""Regression: cost_rollups uses summary-only path (#1671).

Before #1671 ``list_cost_rollup_insights`` loaded every Session with
its full message stream so it could call ``estimate_session_cost``.
On a 7.9K-session archive that pulled 3.7M message rows into Python
and hung the MCP stdio loop past its 60s deadline (#1621).

The new path reads only session rows (with ``provider_meta``) and
derives cost from ``provider_meta`` alone via
``estimate_cost_from_provider_meta``. Sessions without usable cost
evidence in ``provider_meta`` surface as ``status="unavailable"`` rather
than triggering message hydration.

The contract this test pins:

* ``list_cost_rollup_insights`` aggregates costs from ``provider_meta``
  alone — a session whose ``provider_meta`` carries ``total_cost_usd``
  contributes that exact amount to the rollup.
* The rollup completes without loading message bodies (no
  ``SessionRepository.list`` / ``get_many`` calls); the only
  session read is ``list_summaries_by_query``.
* Sessions whose ``provider_meta`` lacks cost evidence appear in the
  rollup as ``status="unavailable"`` rather than triggering an
  expensive message-hydration fallback.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.api import Polylogue
from polylogue.insights.archive import CostRollupInsightQuery
from tests.infra.storage_records import SessionBuilder


@pytest.mark.asyncio
async def test_cost_rollups_aggregate_provider_meta_without_message_load(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-priced")
        .provider("claude-code")
        .title("Priced session")
        .provider_meta({"total_cost_usd": 2.50, "model": "claude-sonnet-4-5"})
        .updated_at("2026-03-01T10:00:00+00:00")
        .add_message("u1", role="user", text="hello")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-no-cost")
        .provider("claude-code")
        .title("No cost evidence")
        .provider_meta({})
        .updated_at("2026-03-01T11:00:00+00:00")
        .add_message("u1", role="user", text="hello")
        .save()
    )

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    # The archive cost rollup reads materialized session_profiles.cost_usd,
    # which the rebuild populates from provider_meta (origin_meta) without
    # loading message bodies.
    await archive.rebuild_insights()

    target = "polylogue.storage.repository.archive.sessions.RepositoryArchiveSessionMixin.list_by_query"
    with patch(target, side_effect=AssertionError("messages loaded")) as spy:
        rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(provider="claude-code"))
        assert spy.call_count == 0

    total = sum(rollup.total_usd for rollup in rollups)
    assert total == pytest.approx(2.50)
    merged_status_counts: dict[str, int] = {}
    for rollup in rollups:
        for key, value in rollup.status_counts.items():
            merged_status_counts[key] = merged_status_counts.get(key, 0) + value
    assert merged_status_counts.get("exact", 0) == 1
    assert merged_status_counts.get("unavailable", 0) == 1
