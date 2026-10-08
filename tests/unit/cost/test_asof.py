"""Cost outlook uses exact as-of evidence before daily aggregation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from polylogue.analysis.archive import ArchiveInsightProvenance, SessionCostInsight, SessionCostInsightQuery
from polylogue.api import Polylogue
from polylogue.archive.semantic.pricing import CostEstimatePayload
from polylogue.cost.aggregation import session_costs_to_daily_usd
from polylogue.cost.outlook import DailyUsage, build_cycle_outlook
from polylogue.cost.plans import SubscriptionPlan
from tests.infra.storage_records import SessionBuilder, db_setup


def test_daily_outlook_excludes_future_days_and_exclusive_cycle_end() -> None:
    plan = SubscriptionPlan(
        name="calendar", provider="test", display_name="Calendar", monthly_cost_usd=0, cycle_anchor_day=1
    )
    rows = [
        DailyUsage(day=date(2026, month, day), basis="usd", amount=amount)
        for month, day, amount in ((4, 30, 100), (5, 2, 10), (5, 20, 90), (6, 1, 100))
    ]
    early = build_cycle_outlook(plan, rows, now=datetime(2026, 5, 11, tzinfo=UTC))
    late = build_cycle_outlook(plan, rows, now=datetime(2026, 5, 31, 23, tzinfo=UTC))
    assert early is not None and late is not None
    assert early.cycle_to_date == {"usd": 10.0}
    assert late.cycle_to_date == {"usd": 100.0}


@pytest.mark.asyncio
async def test_api_outlook_is_unchanged_when_later_archive_facts_arrive(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = workspace_env["archive_root"] / "cost-plan.toml"
    config.write_text("""[[cost.subscription.plans]]
name = "calendar"
provider = "test"
display_name = "Calendar"
monthly_cost_usd = 0.0
cycle_anchor_day = 1
""")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config))
    path = db_setup(workspace_env)

    def add(name: str, timestamp: str, amount: float) -> None:
        (
            SessionBuilder(path, name)
            .provider("claude-code")
            .created_at(timestamp)
            .updated_at(timestamp)
            .reported_cost_usd(amount)
            .add_message(name + "-m", role="assistant", text="neutral usage")
            .save()
        )

    add("past", "2026-05-11T08:00:00Z", 10.0)
    now = datetime(2026, 5, 11, 12, tzinfo=UTC)
    async with Polylogue(archive_root=path.parent, db_path=path) as archive:
        before = await archive.cost_outlook("calendar", now=now)
        assert before is not None and before.cycle_to_date == {"usd": 10.0}
        add("same-day-future", "2026-05-11T18:00:00Z", 90.0)
        add("later", "2026-05-20T00:00:00Z", 50.0)
        add("end", "2026-06-01T00:00:00Z", 100.0)
        add("before-cycle", "2026-04-30T23:59:59Z", 200.0)
        after = await archive.cost_outlook("calendar", now=now)
        assert after is not None
        assert after.cycle_to_date == before.cycle_to_date
        assert after.projected_total == before.projected_total
        at_instant = await archive.cost_outlook("calendar", now=datetime(2026, 5, 11, 18, tzinfo=UTC))
        advanced = await archive.cost_outlook("calendar", now=datetime(2026, 5, 21, tzinfo=UTC))
        assert at_instant is not None and at_instant.cycle_to_date == {"usd": 100.0}
        assert advanced is not None and advanced.cycle_to_date == {"usd": 150.0}


def test_daily_aggregation_bounds_exact_instant_before_utc_fold() -> None:
    rows = [
        SessionCostInsight(
            session_id=str(i),
            origin="claude-code-session",
            created_at=timestamp,
            estimate=CostEstimatePayload(
                origin="claude-code-session", session_id=str(i), status="exact", total_usd=amount
            ),
            provenance=ArchiveInsightProvenance(
                materializer_version=1,
                materialized_at="2026-05-17T00:00:00Z",
                source_updated_at=None,
                source_sort_key=None,
            ),
        )
        for i, (timestamp, amount) in enumerate(
            (("2026-05-12T01:00:00+02:00", 10), ("2026-05-11T23:00:00Z", 5), ("2026-05-11T23:00:00.000001Z", 90))
        )
    ]
    cutoff = datetime(2026, 5, 11, 23, tzinfo=UTC)
    assert session_costs_to_daily_usd(rows, as_of=cutoff) == [DailyUsage(day=date(2026, 5, 11), basis="usd", amount=15)]
    assert sum(row.amount for row in session_costs_to_daily_usd(rows)) == 105


@pytest.mark.asyncio
async def test_api_outlook_attributes_later_updated_session_to_creation_time(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = workspace_env["archive_root"] / "cost-plan.toml"
    config.write_text("""[[cost.subscription.plans]]
name = "calendar"
provider = "test"
display_name = "Calendar"
monthly_cost_usd = 0.0
cycle_anchor_day = 1
""")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config))
    path = db_setup(workspace_env)
    (
        SessionBuilder(path, "later-update")
        .provider("claude-code")
        .created_at("2026-05-11T08:00:00Z")
        .updated_at("2026-05-20T00:00:00Z")
        .reported_cost_usd(10)
        .add_message("m", role="assistant", text="neutral usage")
        .save()
    )
    async with Polylogue(archive_root=path.parent, db_path=path) as archive:
        outlook = await archive.cost_outlook("calendar", now=datetime(2026, 5, 11, 12, tzinfo=UTC))
        assert outlook is not None and outlook.cycle_to_date == {"usd": 10.0}
        # Generic cost queries retain their canonical source/sort time contract.
        generic = await archive.list_session_cost_insights(
            SessionCostInsightQuery(until="2026-05-11T12:00:00Z", limit=None)
        )
        assert generic == []
        created = await archive.list_session_cost_insights(
            SessionCostInsightQuery(until="2026-05-11T12:00:00Z", time_basis="created", limit=None)
        )
        assert len(created) == 1 and created[0].created_at is not None
        assert datetime.fromisoformat(created[0].created_at) == datetime(2026, 5, 11, 8, tzinfo=UTC)
        assert (
            SessionCostInsightQuery.model_validate_json(
                SessionCostInsightQuery(time_basis="created").model_dump_json()
            ).time_basis
            == "created"
        )
