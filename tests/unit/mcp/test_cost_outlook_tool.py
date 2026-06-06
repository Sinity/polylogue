"""MCP ``cost_outlook`` tool contract tests (#1138)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.cost.outlook import (
    CycleOutlook,
    CycleWindow,
    OverageRow,
    ProjectionMethod,
    QuotaPressure,
    QuotaPressureMissing,
)
from polylogue.cost.plans import OverageRule, PlanLookupError, QuotaBasis
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface_async,
    make_polylogue_mock,
)


def _quota_outlook() -> CycleOutlook:
    window = CycleWindow(
        start=datetime(2026, 5, 1, tzinfo=UTC),
        end=datetime(2026, 5, 31, tzinfo=UTC),
        now=datetime(2026, 5, 11, tzinfo=UTC),
        elapsed_days=10.0,
        remaining_days=20.0,
        total_days=30.0,
    )
    return CycleOutlook(
        plan_name="claude-pro",
        window=window,
        cycle_to_date={"credits": 10_000.0, "usd": 4.5},
        burn_rate_per_day={"credits": 1_000.0, "usd": 0.45},
        projected_total={"credits": 30_000.0, "usd": 13.5},
        projection_method=ProjectionMethod.linear,
        quota_pressure=QuotaPressure(
            basis=QuotaBasis.credits,
            quota=21_700.0,
            used=10_000.0,
            projected=30_000.0,
            used_ratio=10_000.0 / 21_700.0,
            projected_ratio=30_000.0 / 21_700.0,
            breach_day=None,
        ),
        overage_rows=(
            OverageRow(
                basis=QuotaBasis.credits,
                actual_overage=0.0,
                projected_overage=8_300.0,
                overage_rule=OverageRule.metered,
                rate_usd_per_unit=0.001,
                projected_overage_cost_usd=8.3,
            ),
        ),
        coverage_ratio=1.0,
        incomplete_days=(),
        confidence=0.9,
    )


class TestCostOutlookToolContract:
    @pytest.mark.asyncio
    async def test_returns_typed_outlook_payload(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.cost_outlook = AsyncMock(return_value=_quota_outlook())
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_outlook"].fn,
                plan="claude-pro",
            )

        payload = json.loads(raw)
        assert payload["plan_name"] == "claude-pro"
        assert payload["projection_method"] == "linear"
        assert set(payload["cycle_to_date"]) == {"credits", "usd"}
        # Subscription-equivalent and API-equivalent never merged.
        assert payload["cycle_to_date"]["credits"] != payload["cycle_to_date"]["usd"]
        assert payload["quota_pressure"]["basis"] == "credits"
        assert payload["overage_rows"][0]["overage_rule"] == "metered"

    @pytest.mark.asyncio
    async def test_no_cycle_window_returns_typed_error(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.cost_outlook = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_outlook"].fn,
                plan="github-copilot-pro",
            )

        payload = json.loads(raw)
        assert payload["code"] == "no_cycle_window"
        assert payload["tool"] == "cost_outlook"
        assert "github-copilot-pro" in payload["detail"]

    @pytest.mark.asyncio
    async def test_unknown_plan_returns_typed_error(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.cost_outlook = AsyncMock(side_effect=PlanLookupError("Unknown subscription plan 'mystery'."))
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_outlook"].fn,
                plan="mystery",
            )

        payload = json.loads(raw)
        assert payload["code"] == "not_found"
        assert payload["tool"] == "cost_outlook"
        assert "mystery" in payload["detail"]

    @pytest.mark.asyncio
    async def test_invalid_method_returns_typed_error(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_outlook"].fn,
                plan="claude-pro",
                method="not-a-method",
            )

        payload = json.loads(raw)
        assert payload["code"] == "invalid_argument"
        assert payload["tool"] == "cost_outlook"
        assert "not-a-method" in payload["detail"]

    @pytest.mark.asyncio
    async def test_no_quota_plan_returns_pressure_missing(self, mcp_server: MCPServerUnderTest) -> None:
        no_quota = _quota_outlook().model_copy(
            update={
                "plan_name": "chatgpt-plus",
                "quota_pressure": QuotaPressureMissing(),
                "overage_rows": (),
                "cycle_to_date": {"usd": 1.0},
                "burn_rate_per_day": {"usd": 0.1},
                "projected_total": {"usd": 3.0},
            }
        )
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.cost_outlook = AsyncMock(return_value=no_quota)
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_outlook"].fn,
                plan="chatgpt-plus",
            )

        payload = json.loads(raw)
        # QuotaPressureMissing must serialise with an explicit reason,
        # never as a fabricated zero-quota row.
        assert payload["quota_pressure"]["reason"] == "no_quota_configured"
        assert payload["overage_rows"] == []


class TestCostOutlookAggregator:
    """Unit tests for ``session_costs_to_daily_usd`` (#1138 facade glue)."""

    def test_aggregates_by_day_skipping_zero_and_unparseable(self) -> None:
        from polylogue.archive.semantic.pricing import CostEstimatePayload, CostUsagePayload
        from polylogue.cost.aggregation import session_costs_to_daily_usd
        from polylogue.insights.archive import SessionCostInsight
        from polylogue.insights.archive_models import ArchiveInsightProvenance

        provenance = ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-05-04T00:00:00Z")

        def _insight(conv: str, ts: str | None, total_usd: float) -> SessionCostInsight:
            return SessionCostInsight(
                session_id=conv,
                source_name="claude",
                created_at=ts,
                estimate=CostEstimatePayload(
                    source_name="claude",
                    session_id=conv,
                    status="exact" if total_usd > 0 else "unavailable",
                    total_usd=total_usd,
                    usage=CostUsagePayload(),
                ),
                provenance=provenance,
            )

        rows = session_costs_to_daily_usd(
            [
                _insight("a", "2026-05-01T10:00:00Z", 1.0),
                _insight("b", "2026-05-01T22:30:00Z", 0.25),
                _insight("c", "2026-05-02T01:00:00+00:00", 2.0),
                _insight("d", "2026-05-03T00:00:00Z", 0.0),  # skipped (zero)
                _insight("e", None, 5.0),  # skipped (no timestamp)
                _insight("f", "not-a-date", 5.0),  # skipped (unparseable)
            ]
        )

        assert [r.day.isoformat() for r in rows] == ["2026-05-01", "2026-05-02"]
        assert rows[0].amount == pytest.approx(1.25)
        assert rows[1].amount == pytest.approx(2.0)
        assert all(r.basis == "usd" for r in rows)
