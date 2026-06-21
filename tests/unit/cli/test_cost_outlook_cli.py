"""Tests for ``polylogue analyze --cost-outlook``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from polylogue.cost.outlook import (
    CycleOutlook,
    CycleWindow,
    OverageRow,
    ProjectionMethod,
    QuotaPressure,
    QuotaPressureMissing,
)
from polylogue.cost.plans import OverageRule, PlanLookupError, QuotaBasis


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


def _no_quota_outlook() -> CycleOutlook:
    window = CycleWindow(
        start=datetime(2026, 5, 1, tzinfo=UTC),
        end=datetime(2026, 5, 31, tzinfo=UTC),
        now=datetime(2026, 5, 11, tzinfo=UTC),
        elapsed_days=10.0,
        remaining_days=20.0,
        total_days=30.0,
    )
    return CycleOutlook(
        plan_name="chatgpt-plus",
        window=window,
        cycle_to_date={"usd": 1.25},
        burn_rate_per_day={"usd": 0.125},
        projected_total={"usd": 3.75},
        projection_method=ProjectionMethod.linear,
        quota_pressure=QuotaPressureMissing(),
        overage_rows=(),
        coverage_ratio=1.0,
        incomplete_days=(),
        confidence=0.6,
    )


class TestCostOutlookJsonContract:
    def test_json_output_serializes_full_payload(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(return_value=_quota_outlook())

            result = cli_runner.invoke(
                click_cli, ["analyze", "--cost-outlook", "--plan", "claude-pro", "--format", "json"]
            )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["plan_name"] == "claude-pro"
        assert payload["projection_method"] == "linear"
        # Both bases reach the payload, never merged.
        assert set(payload["cycle_to_date"]) == {"credits", "usd"}
        # Quota pressure carries the basis label.
        assert payload["quota_pressure"]["basis"] == "credits"
        # Overage rows include rule and projected cost in USD.
        assert payload["overage_rows"][0]["overage_rule"] == "metered"
        assert payload["overage_rows"][0]["projected_overage_cost_usd"] == pytest.approx(8.3)

    def test_unknown_plan_exits_with_typed_message(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(side_effect=PlanLookupError("Unknown subscription plan 'mystery'."))

            result = cli_runner.invoke(
                click_cli, ["analyze", "--cost-outlook", "--plan", "mystery", "--format", "json"]
            )

        assert result.exit_code != 0
        assert "mystery" in result.output

    def test_no_cycle_anchor_returns_json_null_outlook(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(return_value=None)

            result = cli_runner.invoke(
                click_cli, ["analyze", "--cost-outlook", "--plan", "github-copilot-pro", "--format", "json"]
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload == {
            "plan_name": "github-copilot-pro",
            "outlook": None,
            "reason": "no_cycle_anchor",
        }


class TestCostOutlookPlainContract:
    def test_plain_mode_labels_bases_and_estimates(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(return_value=_quota_outlook())

            result = cli_runner.invoke(
                click_cli,
                ["--plain", "analyze", "--cost-outlook", "--plan", "claude-pro", "--format", "plaintext"],
            )

        assert result.exit_code == 0, result.output
        out = result.output
        # Estimate marker visible.
        assert "(estimate)" in out
        # USD basis labelled as API-equivalent — never bare "cost".
        assert "USD (API-equivalent)" in out
        # Credit-basis quota figures labelled with basis.
        assert "Quota pressure (credits)" in out
        # Subscription-quota math caveat must surface.
        assert "non-authoritative" in out

    def test_plain_mode_surfaces_absent_quota_explicitly(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(return_value=_no_quota_outlook())

            result = cli_runner.invoke(
                click_cli,
                ["--plain", "analyze", "--cost-outlook", "--plan", "chatgpt-plus", "--format", "plaintext"],
            )

        assert result.exit_code == 0, result.output
        # Absence of quota must be reported, not hidden as zero.
        assert "not configured" in result.output

    def test_plain_mode_no_cycle_anchor_message(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(return_value=None)

            result = cli_runner.invoke(
                click_cli,
                ["--plain", "analyze", "--cost-outlook", "--plan", "github-copilot-pro", "--format", "plaintext"],
            )

        assert result.exit_code == 0
        assert "cycle_anchor_day" in result.output


class TestCostOutlookMethodChoice:
    def test_trailing_mean_method_passes_through(self, cli_runner: CliRunner) -> None:
        captured: dict[str, object] = {}

        async def fake_outlook(plan_name: str, *, method: ProjectionMethod) -> CycleOutlook:
            captured["plan_name"] = plan_name
            captured["method"] = method
            return _quota_outlook()

        with patch("polylogue.api.Polylogue") as mock_cls:
            poly = mock_cls.return_value
            poly.__aenter__ = AsyncMock(return_value=poly)
            poly.__aexit__ = AsyncMock(return_value=None)
            poly.cost_outlook = AsyncMock(side_effect=fake_outlook)

            result = cli_runner.invoke(
                click_cli,
                [
                    "analyze",
                    "--cost-outlook",
                    "--plan",
                    "claude-pro",
                    "--method",
                    "trailing-7d-mean",
                    "--format",
                    "json",
                ],
            )

        assert result.exit_code == 0, result.output
        assert captured["method"] is ProjectionMethod.trailing_7d_mean
