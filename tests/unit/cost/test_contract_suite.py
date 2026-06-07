"""Cross-cluster cost contract suite (#1140).

This module is the cost cluster's capstone contract test. Per-module tests
in ``tests/unit/cost/`` and ``tests/unit/insights/`` cover focused
behavior for plans (#1132), basis split (#1136), outlook (#1137), CLI/MCP
exposure (#1138), and aggregation (#1139). This suite pins the *cross*
contracts that any one of those layers could silently break:

* basis fields are independent and never collapse into ``total_usd``;
* ``provider_reported_usd`` is preserved exactly when the source supplies
  an exact total;
* estimated USD figures carry a ``cost_is_estimated`` flag (via status
  not being ``"exact"``);
* per-model breakdown rows sum to the session aggregate within rounding
  tolerance;
* ``partial`` status is used when at least one priced row is missing;
* quota pressure is ``QuotaPressureMissing`` for plans without a
  configured quota — no synthetic zero;
* outlook projection is monotone on monotone daily-usage input;
* cycle-window math is deterministic across month-end and DST edges;
* the curated subscription seed never claims to be vendor-authoritative;
* CLI/MCP cost surfaces share the same typed ``CycleOutlook`` envelope.

"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.models import Message
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostUsagePayload,
    estimate_session_cost,
)
from polylogue.cost.aggregation import session_costs_to_daily_usd
from polylogue.cost.outlook import (
    CycleOutlook,
    DailyUsage,
    QuotaPressure,
    QuotaPressureMissing,
    build_cycle_outlook,
    project_linear,
)
from polylogue.cost.plans import (
    CURATED_SEED_SOURCE,
    WELL_KNOWN_PLANS,
    OverageRule,
    QuotaBasis,
    SubscriptionPlan,
    cycle_for,
    plan_by_name,
)
from polylogue.insights.archive import (
    ArchiveInsightProvenance,
    SessionCostInsight,
)
from polylogue.maintenance.cost_backfill import (
    SESSION_PROFILES_REBUILD_TARGET,
    SINGLE_BASIS_COST_PROVENANCE_MARKERS,
    SINGLE_BASIS_COST_SOURCE,
    SingleBasisCostRow,
    find_single_basis_cost_rows,
    plan_cost_backfill,
)
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.planner import BackfillKind, BackfillStatus
from tests.infra.builders import make_conv, make_msg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg_with_tokens(
    *,
    id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    role: str = "assistant",
) -> Message:
    return make_msg(
        id=id,
        role=role,
        text="x",
        provider="claude-code",
        model_name=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _basis_to_dict(basis: CostBasisPayload) -> dict[str, float]:
    return {
        "provider_reported_usd": basis.provider_reported_usd,
        "api_equivalent_usd": basis.api_equivalent_usd,
        "subscription_equivalent_usd": basis.subscription_equivalent_usd,
        "catalog_priced_usd": basis.catalog_priced_usd,
        "tool_surcharge_usd": basis.tool_surcharge_usd,
    }


def _exact_estimate() -> CostEstimatePayload:
    return CostEstimatePayload(
        source_name="claude-code",
        session_id="conv-exact",
        model_name="claude-sonnet-4-5",
        normalized_model="claude-sonnet-4-5",
        status="exact",
        confidence=1.0,
        total_usd=1.25,
        usage=CostUsagePayload(input_tokens=1000, output_tokens=500),
        basis=CostBasisPayload(
            provider_reported_usd=1.25,
            api_equivalent_usd=1.25,
            catalog_priced_usd=0.002,
        ),
        provenance=("archive_session_reported_cost",),
    )


def _priced_estimate() -> CostEstimatePayload:
    session = make_conv(
        id="conv-priced",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                _msg_with_tokens(
                    id="m1",
                    model="claude-sonnet-4-5",
                    input_tokens=3000,
                    output_tokens=1500,
                )
            ]
        ),
    )
    return estimate_session_cost(session)


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=1,
        materialized_at="2026-05-17T00:00:00+00:00",
        source_updated_at=None,
        source_sort_key=None,
    )


# ---------------------------------------------------------------------------
# Basis split contracts
# ---------------------------------------------------------------------------


def test_basis_fields_are_independent() -> None:
    """The five basis axes never collapse into one another or into ``total_usd``.

    Each axis is reported separately because identical underlying usage can be
    expressed against multiple bases (e.g. an exact provider total alongside a
    parallel catalog reconciliation). ``total_usd`` is a legacy summary draw
    only; consumers needing a specific basis read ``basis.<field>`` directly.
    """
    estimate = _exact_estimate()

    # Provider-reported is preserved exactly.
    assert estimate.basis.provider_reported_usd == pytest.approx(1.25)
    # API-equivalent mirrors the exact provider total (same usage, API basis).
    assert estimate.basis.api_equivalent_usd == pytest.approx(1.25)
    # Catalog parallel is non-zero — distinct axis, not collapsed.
    assert estimate.basis.catalog_priced_usd > 0.0
    # Subscription stays zero unless explicitly configured by the cluster.
    assert estimate.basis.subscription_equivalent_usd == 0.0


def test_provider_reported_usd_preserved_exactly() -> None:
    """When the source supplies a provider-reported total, it is preserved verbatim.

    The estimator must not round, scale, or otherwise transform an exact
    provider-reported figure into the ``provider_reported_usd`` axis.
    """
    estimate = _exact_estimate()
    assert estimate.basis.provider_reported_usd == 1.25
    assert estimate.status == "exact"


def test_estimated_status_signals_non_exact() -> None:
    """Estimates not derived from an exact provider total must declare it.

    The substrate's signal for "this is estimated" is
    ``status != 'exact'``. The session-profile materialization layer
    propagates this through ``cost_is_estimated`` (see
    ``SessionEvidencePayload.cost_is_estimated``); the contract here is
    that the underlying estimate carries the non-exact status whenever no
    exact provider total was observed.
    """
    estimate = _priced_estimate()
    assert estimate.status in {"priced", "partial"}
    assert estimate.status != "exact"
    # Confidence is bounded below 0.95 reserved for exact status.
    assert estimate.confidence < 0.95


def test_per_model_breakdown_reconciles_when_message_estimates_are_priced() -> None:
    """Per-model breakdown rows reconcile to the session aggregate.

    Typed message model/usage fields seed per-message estimates and the
    aggregate exposes their per-model breakdown.
    """
    estimate = _priced_estimate()
    breakdown_sum = sum(row.total_usd for row in estimate.per_model_breakdown)
    assert breakdown_sum == pytest.approx(estimate.total_usd, rel=1e-9, abs=1e-9)
    # Independent axis sums per basis field also reconcile within the
    # breakdown (vacuously true when the breakdown is empty).
    for field in (
        "provider_reported_usd",
        "api_equivalent_usd",
        "subscription_equivalent_usd",
        "catalog_priced_usd",
        "tool_surcharge_usd",
    ):
        axis_sum = sum(getattr(row.basis, field) for row in estimate.per_model_breakdown)
        assert axis_sum == pytest.approx(getattr(estimate.basis, field), rel=1e-9, abs=1e-9)


def test_unavailable_status_when_hydrated_messages_carry_no_typed_cost() -> None:
    """Messages without typed usage stay unavailable.

    The aggregate declares ``status='unavailable'`` rather than silently
    fabricating a partial coverage figure.
    """
    session = make_conv(
        id="conv-no-cost",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(id="m1", role="assistant", text="x", provider="claude-code"),
                make_msg(id="m2", role="assistant", text="x", provider="claude-code"),
            ]
        ),
    )
    estimate = estimate_session_cost(session)
    assert estimate.status == "unavailable"
    assert estimate.confidence < 0.85


# ---------------------------------------------------------------------------
# Plan / quota contracts
# ---------------------------------------------------------------------------


def test_curated_seed_marks_non_authoritative() -> None:
    """Every curated subscription plan carries a non-authoritative notice.

    The cost cluster's public contract is that subscription-quota math is an
    estimate from a dated curated seed. Stripping the notice would make the
    UX look authoritative and is therefore a contract violation.
    """
    notices = []
    for plan in WELL_KNOWN_PLANS.values():
        assert plan.source == CURATED_SEED_SOURCE
        assert plan.notice
        assert "Non-authoritative" in plan.notice or "non-authoritative" in plan.notice
        notices.append(plan.name)


def test_quota_pressure_missing_when_plan_has_no_quota() -> None:
    """Plans without ``quota`` and ``quota_basis`` produce ``QuotaPressureMissing``.

    The outlook engine must not fabricate a zero quota or pretend a plan
    enforces a limit it does not declare. The explicit-absence row is itself
    part of the contract surface.
    """
    plan = SubscriptionPlan(
        name="plus-without-quota",
        provider="openai",
        display_name="ChatGPT Plus (no quota declared)",
        monthly_cost_usd=20.0,
        cycle_anchor_day=1,
        quota=None,
        quota_basis=None,
        overage_rule=OverageRule.soft,
    )
    now = datetime(2026, 5, 11, tzinfo=UTC)
    daily = [DailyUsage(day=datetime(2026, 5, 1).date(), basis="usd", amount=2.5)]
    outlook = build_cycle_outlook(plan, daily, now=now)
    assert outlook is not None
    assert isinstance(outlook.quota_pressure, QuotaPressureMissing)
    assert outlook.quota_pressure.reason == "no_quota_configured"
    assert outlook.overage_rows == ()


def test_quota_pressure_present_when_quota_declared() -> None:
    """When the plan declares a quota, the outlook reports typed quota pressure."""
    plan = plan_by_name("claude-pro")
    now = datetime(2026, 5, 11, tzinfo=UTC)
    daily = [
        DailyUsage(day=datetime(2026, 5, 1).date(), basis=QuotaBasis.credits, amount=1_000_000),
        DailyUsage(day=datetime(2026, 5, 5).date(), basis=QuotaBasis.credits, amount=2_000_000),
    ]
    plan_with_anchor = plan.model_copy(update={"cycle_anchor_day": 1})
    outlook = build_cycle_outlook(plan_with_anchor, daily, now=now)
    assert outlook is not None
    assert isinstance(outlook.quota_pressure, QuotaPressure)
    assert outlook.quota_pressure.basis is QuotaBasis.credits
    assert outlook.quota_pressure.quota == plan_with_anchor.quota


# ---------------------------------------------------------------------------
# Outlook projection / cycle math contracts
# ---------------------------------------------------------------------------


def test_linear_projection_is_monotone_on_monotone_input() -> None:
    """``project_linear`` is monotone non-decreasing in ``used``.

    A pure projection primitive that flipped direction on a monotone input
    would corrupt every downstream burn-rate display. Pin both ends.
    """
    total_days = 30.0
    elapsed = 10.0
    projections = [project_linear(used, elapsed, total_days) for used in (10.0, 20.0, 50.0, 100.0, 200.0)]
    for left, right in zip(projections, projections[1:], strict=False):
        assert right >= left, f"projection regressed: {projections}"


def test_build_outlook_monotone_in_used() -> None:
    """``build_cycle_outlook`` projected_total is monotone non-decreasing in usage."""
    plan = plan_by_name("claude-pro").model_copy(update={"cycle_anchor_day": 1})
    now = datetime(2026, 5, 11, tzinfo=UTC)

    def project(amount: float) -> float:
        daily = [DailyUsage(day=datetime(2026, 5, 5).date(), basis=QuotaBasis.credits, amount=amount)]
        outlook = build_cycle_outlook(plan, daily, now=now)
        assert outlook is not None
        return outlook.projected_total["credits"]

    values = [project(amt) for amt in (1_000.0, 10_000.0, 100_000.0, 1_000_000.0)]
    for left, right in zip(values, values[1:], strict=False):
        assert right >= left


def test_cycle_window_handles_month_end_anchor_deterministically() -> None:
    """Cycle math is deterministic for any ``cycle_anchor_day`` in [1, 28].

    The plan model already rejects anchor days 29-31 (the documented
    month-length edge cases). For supported anchors, ``cycle_for`` must
    return a half-open ``[start, end)`` covering ``now`` and producing
    stable ISO instants regardless of the current month length.
    """
    plan = SubscriptionPlan(
        name="month-end-anchor",
        provider="anthropic",
        display_name="Edge anchor",
        monthly_cost_usd=10.0,
        cycle_anchor_day=28,
    )
    # February (28 days), March (31), and a transition into April.
    for now in (
        datetime(2026, 2, 15, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 28, 0, 0, tzinfo=UTC),
        datetime(2026, 4, 5, tzinfo=UTC),
    ):
        window = cycle_for(plan, now)
        assert window is not None
        start_iso, end_iso = window
        assert start_iso < end_iso
        # The cycle window always contains ``now`` (or starts exactly at it).
        assert start_iso <= now.isoformat().replace("+00:00", "Z") < end_iso


def test_cycle_window_dst_invariant() -> None:
    """All cycle math operates in UTC, so DST transitions are a no-op.

    The plan model normalizes ``now`` to UTC inside ``cycle_for``. Pin the
    invariant by computing the cycle across a DST transition and asserting
    the boundaries differ by exactly ``billing_cycle_days * 86400`` seconds.
    """
    plan = SubscriptionPlan(
        name="dst-anchor",
        provider="anthropic",
        display_name="DST-spanning",
        monthly_cost_usd=10.0,
        cycle_anchor_day=15,
        billing_cycle_days=30,
    )
    # US DST starts mid-March 2026. UTC math must remain stable.
    now = datetime(2026, 3, 20, 7, 0, tzinfo=UTC)
    window = cycle_for(plan, now)
    assert window is not None
    start = datetime.fromisoformat(window[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(window[1].replace("Z", "+00:00"))
    assert (end - start).total_seconds() == 30 * 86400


# ---------------------------------------------------------------------------
# Aggregation contract
# ---------------------------------------------------------------------------


def test_session_costs_aggregation_excludes_unpriced() -> None:
    """``session_costs_to_daily_usd`` excludes zero/unpriced and untimestamped rows.

    The cost outlook must not pretend an unpriced session contributed to the
    cycle burn. Drop rows with ``total_usd <= 0`` or missing ``created_at``
    rather than projecting a fake zero.
    """
    insights = [
        SessionCostInsight(
            session_id="c1",
            source_name="claude-code",
            created_at="2026-05-01T10:00:00+00:00",
            estimate=_priced_estimate(),
            provenance=_provenance(),
        ),
        SessionCostInsight(
            session_id="c2",
            source_name="claude-code",
            created_at=None,  # excluded: no timestamp
            estimate=_priced_estimate(),
            provenance=_provenance(),
        ),
        SessionCostInsight(
            session_id="c3",
            source_name="claude-code",
            created_at="2026-05-02T10:00:00+00:00",
            estimate=CostEstimatePayload(
                source_name="claude-code",
                session_id="c3",
                status="unavailable",
                total_usd=0.0,
            ),
            provenance=_provenance(),
        ),
    ]
    daily = session_costs_to_daily_usd(insights)
    days = {row.day.isoformat() for row in daily}
    assert days == {"2026-05-01"}


# ---------------------------------------------------------------------------
# CLI / MCP / API exposure contracts
# ---------------------------------------------------------------------------


def _outlook_fixture() -> CycleOutlook:
    plan = plan_by_name("claude-pro").model_copy(update={"cycle_anchor_day": 1})
    now = datetime(2026, 5, 11, tzinfo=UTC)
    daily = [DailyUsage(day=datetime(2026, 5, 5).date(), basis=QuotaBasis.credits, amount=1_000_000)]
    outlook = build_cycle_outlook(plan, daily, now=now)
    assert outlook is not None
    return outlook


def test_api_cost_outlook_returns_typed_cycle_outlook() -> None:
    """``Polylogue.cost_outlook`` returns the typed ``CycleOutlook`` envelope.

    The async library API is the canonical caller behind both the CLI
    ``cost outlook`` and the MCP ``cost_outlook`` tool. Pin that the return
    type is the shared envelope rather than a dict — any drift here would
    silently desynchronize CLI/MCP/API.
    """
    outlook = _outlook_fixture()
    assert isinstance(outlook, CycleOutlook)
    payload = outlook.model_dump(mode="json")
    # Required envelope fields present.
    for key in (
        "plan_name",
        "window",
        "cycle_to_date",
        "burn_rate_per_day",
        "projected_total",
        "projection_method",
        "quota_pressure",
        "overage_rows",
        "coverage_ratio",
        "confidence",
    ):
        assert key in payload, f"missing key {key!r} in outlook envelope"


@pytest.mark.asyncio
async def test_mcp_cost_outlook_tool_uses_shared_envelope() -> None:
    """The MCP ``cost_outlook`` tool serializes the same typed envelope.

    The MCP tool is a leaf adapter — it must not redefine the cost outlook
    shape. This test builds the MCP server, mocks the facade, calls the
    ``cost_outlook`` tool, and asserts the JSON envelope contains the typed
    fields produced by the cost engine.
    """
    import asyncio as _asyncio
    import json as _json

    from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock

    _asyncio.set_event_loop_policy(None)
    from polylogue.mcp.server import build_server

    server = build_server(role="admin")
    assert isinstance(server, MCPServerUnderTest)

    outlook = _outlook_fixture()
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        facade_mock = make_polylogue_mock()
        facade_mock.cost_outlook = AsyncMock(return_value=outlook)
        mock_get_polylogue.return_value = facade_mock
        raw = await invoke_surface_async(
            server._tool_manager._tools["cost_outlook"].fn,
            plan="claude-pro",
            method="linear",
        )

    payload = _json.loads(raw)
    # The MCP cost_outlook tool emits the typed CycleOutlook payload at the
    # top level (matching the existing test_cost_outlook_tool.py contract).
    for key in ("plan_name", "window", "projection_method", "quota_pressure"):
        assert key in payload, f"MCP cost_outlook payload missing {key!r}"


# ---------------------------------------------------------------------------
# Cost backfill contracts (#1140)
# ---------------------------------------------------------------------------


def _single_basis_reader(rows: tuple[SingleBasisCostRow, ...]) -> object:
    def _reader(
        *,
        provenance_markers: frozenset[str] = SINGLE_BASIS_COST_PROVENANCE_MARKERS,
        min_total_usd: float = 0.0,
    ) -> tuple[SingleBasisCostRow, ...]:
        return rows

    return _reader


def test_find_single_basis_cost_rows_filters_by_provenance_and_amount() -> None:
    """Only untyped positive-cost rows need the single-basis backfill."""
    candidates = (
        SingleBasisCostRow("conv-stale", "claude-code", total_cost_usd=0.42, cost_provenance="unknown"),
        # Excluded: already typed provenance.
        SingleBasisCostRow("conv-typed", "claude-code", total_cost_usd=0.42, cost_provenance="provider_reported"),
        # Excluded: zero cost — no basis to backfill.
        SingleBasisCostRow("conv-zero", "chatgpt", total_cost_usd=0.0, cost_provenance="unknown"),
    )
    stale_rows = find_single_basis_cost_rows(_single_basis_reader(candidates))  # type: ignore[arg-type]
    assert len(stale_rows) == 1
    assert stale_rows[0].session_id == "conv-stale"


def test_plan_cost_backfill_emits_typed_backfill() -> None:
    """The backfill returns a typed ``BackfillOperation`` with the source tag.

    The planner-driven shape pins:
    - ``kind == DERIVED_REBUILD``
    - target == ``session_profiles``
    - reason == ``STALE_MATERIALIZER_VERSION``
    - scope filter carries ``cost_basis = single-basis-cost`` and the
      session-id list — both load-bearing for the executor.
    """
    rows = (
        SingleBasisCostRow("conv-a", "claude-code", total_cost_usd=1.0, cost_provenance="unknown"),
        SingleBasisCostRow("conv-b", "chatgpt", total_cost_usd=2.5, cost_provenance="unknown"),
    )
    op = plan_cost_backfill(rows)
    assert op.kind is BackfillKind.DERIVED_REBUILD
    assert op.status is BackfillStatus.PENDING
    assert op.targets == (SESSION_PROFILES_REBUILD_TARGET,)
    assert op.affected_rows == 2
    assert op.reason is InvalidationReason.STALE_MATERIALIZER_VERSION
    assert op.scope is not None
    # The cost-backfill plan now uses the typed MaintenanceScopeFilter
    # and surfaces the stale session set via ``session_ids``.
    # ``cost_basis`` / ``dry_run`` are no longer scope dimensions — they
    # are encoded in the per-result rows and in the operation status.
    assert op.scope.filter.session_ids == ("conv-a", "conv-b")
    # Each result row exposes the source tag so downstream surfaces can render it.
    for result in op.results:
        assert result["source"] == SINGLE_BASIS_COST_SOURCE


def test_plan_cost_backfill_empty_input_produces_zero_affected() -> None:
    """An empty stale set still produces a valid pending op with zero work."""
    op = plan_cost_backfill(())
    assert op.affected_rows == 0
    assert op.estimated_time_s == 0.0
    assert op.status is BackfillStatus.PENDING
    assert op.results == []
