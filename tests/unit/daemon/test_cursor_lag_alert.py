"""Tests for per-source-family cursor-lag SLO alerts (#1232)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon.cursor_lag_alert import (
    DEFAULT_CRITICAL_S,
    DEFAULT_DEDUP_WINDOW_S,
    DEFAULT_ERROR_S,
    DEFAULT_WARNING_S,
    CursorLagDedupState,
    CursorLagThresholds,
    FamilyLagThreshold,
    evaluate_cursor_lag,
    load_thresholds_from_config,
)
from polylogue.daemon.cursor_lag_status import (
    CursorLagFamilySummary,
    CursorLagSummary,
)
from polylogue.daemon.health import HealthSeverity, HealthTier


def _summary(families: list[CursorLagFamilySummary]) -> CursorLagSummary:
    return CursorLagSummary(
        tracked_file_count=sum(f.tracked_file_count for f in families),
        stuck_file_count=sum(f.stuck_file_count for f in families),
        idle_file_count=sum(f.idle_file_count for f in families),
        max_lag_s=max((f.max_lag_s for f in families), default=0.0),
        family_summaries=families,
    )


def _family(
    *, family: str = "claude-code-session", stuck: int = 1, max_lag_s: float = 0.0, idle: int = 0
) -> CursorLagFamilySummary:
    return CursorLagFamilySummary(
        family=family,
        tracked_file_count=stuck + idle,
        stuck_file_count=stuck,
        idle_file_count=idle,
        max_lag_s=max_lag_s,
    )


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def test_for_family_falls_back_to_global_defaults_when_no_override() -> None:
    thresholds = CursorLagThresholds(default_warning_s=10, default_error_s=100, default_critical_s=1000)
    assert thresholds.for_family("claude-code-session") == (10, 100, 1000)


def test_for_family_applies_family_specific_overrides() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=60,
        default_error_s=600,
        default_critical_s=3600,
        families={"claude-code-session": FamilyLagThreshold(warning_s=5, error_s=30, critical_s=120)},
    )
    assert thresholds.for_family("claude-code-session") == (5, 30, 120)
    assert thresholds.for_family("chatgpt-export") == (60, 600, 3600)


def test_for_family_partial_override_inherits_unset_fields() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10,
        default_error_s=20,
        default_critical_s=30,
        families={"codex-session": FamilyLagThreshold(critical_s=100)},
    )
    # warning_s and error_s fall back to defaults; critical_s overridden
    assert thresholds.for_family("codex-session") == (10, 20, 100)


def test_for_family_clamps_misconfigured_ladder_monotonically() -> None:
    # Operator-misconfigured ladders (e.g. error < warning, critical < error)
    # must be clamped so the alert ladder remains monotonic; otherwise a
    # severity band collapses and one severity level becomes unreachable.
    thresholds = CursorLagThresholds(
        default_warning_s=100,
        default_error_s=50,  # below warning
        default_critical_s=10,  # below error
    )
    assert thresholds.for_family("anything") == (100, 100, 100)


# ---------------------------------------------------------------------------
# Severity resolution
# ---------------------------------------------------------------------------


def test_evaluate_emits_no_alerts_when_no_stuck_files() -> None:
    # Idle-only family must never produce a lag alert regardless of lag value.
    thresholds = CursorLagThresholds(default_warning_s=10, default_error_s=20, default_critical_s=30)
    summary = _summary([_family(stuck=0, idle=3, max_lag_s=0.0)])
    state = CursorLagDedupState()

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    assert alerts == []


def test_evaluate_emits_no_alerts_when_below_warning() -> None:
    thresholds = CursorLagThresholds(default_warning_s=300, default_error_s=600, default_critical_s=1800)
    summary = _summary([_family(stuck=1, max_lag_s=100.0)])
    state = CursorLagDedupState()

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    assert alerts == []


def test_evaluate_emits_warning_when_lag_crosses_warning_threshold() -> None:
    thresholds = CursorLagThresholds(default_warning_s=60, default_error_s=600, default_critical_s=3600)
    summary = _summary([_family(stuck=1, max_lag_s=120.0)])
    state = CursorLagDedupState()

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == HealthSeverity.WARNING
    assert alert.tier == HealthTier.MEDIUM
    assert alert.check_name == "cursor_lag[claude-code-session]"
    assert "stuck" in alert.message


def test_evaluate_emits_error_when_lag_crosses_error_threshold() -> None:
    thresholds = CursorLagThresholds(default_warning_s=60, default_error_s=300, default_critical_s=3600)
    summary = _summary([_family(stuck=1, max_lag_s=500.0)])
    state = CursorLagDedupState()

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    assert len(alerts) == 1
    assert alerts[0].severity == HealthSeverity.ERROR


def test_evaluate_emits_critical_when_lag_crosses_critical_threshold() -> None:
    thresholds = CursorLagThresholds(default_warning_s=60, default_error_s=300, default_critical_s=600)
    summary = _summary([_family(stuck=1, max_lag_s=10_000.0)])
    state = CursorLagDedupState()

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    assert len(alerts) == 1
    assert alerts[0].severity == HealthSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Dedup and escalation
# ---------------------------------------------------------------------------


def test_evaluate_dedups_repeated_alerts_within_window() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10, default_error_s=600, default_critical_s=3600, dedup_window_s=600
    )
    summary = _summary([_family(stuck=1, max_lag_s=120.0)])
    state = CursorLagDedupState()

    first = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=1000.0)
    second = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=1100.0)

    assert len(first) == 1
    assert second == []


def test_evaluate_reemits_after_dedup_window_expires() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10, default_error_s=600, default_critical_s=3600, dedup_window_s=600
    )
    summary = _summary([_family(stuck=1, max_lag_s=120.0)])
    state = CursorLagDedupState()

    first = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=1000.0)
    second = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=1700.0)

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.WARNING


def test_evaluate_escalation_warning_to_error_fires_through_dedup_window() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10, default_error_s=300, default_critical_s=3600, dedup_window_s=3600
    )
    state = CursorLagDedupState()

    first = evaluate_cursor_lag(
        _summary([_family(stuck=1, max_lag_s=100.0)]),
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_cursor_lag(
        _summary([_family(stuck=1, max_lag_s=500.0)]),
        thresholds=thresholds,
        state=state,
        now=10.0,
    )

    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.ERROR


def test_evaluate_escalation_error_to_critical_fires_through_dedup_window() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10, default_error_s=100, default_critical_s=1000, dedup_window_s=3600
    )
    state = CursorLagDedupState()

    first = evaluate_cursor_lag(
        _summary([_family(stuck=1, max_lag_s=200.0)]),
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_cursor_lag(
        _summary([_family(stuck=1, max_lag_s=5_000.0)]),
        thresholds=thresholds,
        state=state,
        now=10.0,
    )

    assert first[0].severity == HealthSeverity.ERROR
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.CRITICAL


def test_evaluate_emits_resolution_alert_when_lag_clears() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10, default_error_s=600, default_critical_s=3600, dedup_window_s=3600
    )
    state = CursorLagDedupState()

    first = evaluate_cursor_lag(
        _summary([_family(stuck=1, max_lag_s=120.0)]),
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_cursor_lag(_summary([]), thresholds=thresholds, state=state, now=10.0)
    third = evaluate_cursor_lag(_summary([]), thresholds=thresholds, state=state, now=20.0)

    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.OK
    # Once cleared, no further OK alerts fire on subsequent quiet runs.
    assert third == []


def test_evaluate_independent_families_emit_independent_alerts() -> None:
    thresholds = CursorLagThresholds(
        default_warning_s=10,
        default_error_s=600,
        default_critical_s=3600,
        families={"claude-code-session": FamilyLagThreshold(warning_s=5)},
    )
    state = CursorLagDedupState()
    summary = _summary(
        [
            _family(family="claude-code-session", stuck=1, max_lag_s=20.0),
            _family(family="chatgpt-export", stuck=1, max_lag_s=20.0),
        ]
    )

    alerts = evaluate_cursor_lag(summary, thresholds=thresholds, state=state, now=0.0)

    by_check = {a.check_name: a for a in alerts}
    assert "cursor_lag[claude-code-session]" in by_check
    assert "cursor_lag[chatgpt-export]" in by_check
    assert by_check["cursor_lag[claude-code-session]"].severity == HealthSeverity.WARNING
    assert by_check["cursor_lag[chatgpt-export]"].severity == HealthSeverity.WARNING


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_thresholds_from_config_reads_polylogue_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(
        """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600
default_critical_s = 7200
dedup_window_s = 900

[health.cursor_lag.families.claude-code-session]
warning_s = 30
error_s = 120
critical_s = 600

[health.cursor_lag.families.chatgpt-export]
warning_s = 14400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_thresholds_from_config(cfg)

    assert thresholds.default_warning_s == 60
    assert thresholds.default_error_s == 600
    assert thresholds.default_critical_s == 7200
    assert thresholds.dedup_window_s == 900
    assert thresholds.families["claude-code-session"] == FamilyLagThreshold(warning_s=30, error_s=120, critical_s=600)
    # chatgpt-export only sets warning_s; error_s and critical_s fall through.
    assert thresholds.for_family("chatgpt-export") == (14400, 14400, 14400)
    assert thresholds.for_family("claude-code-session") == (30, 120, 600)
    assert thresholds.for_family("codex-session") == (60, 600, 7200)


def test_load_thresholds_defaults_when_no_config_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text("[archive]\nroot = '/tmp/archive'\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_thresholds_from_config(cfg)

    assert thresholds.default_warning_s == DEFAULT_WARNING_S
    assert thresholds.default_error_s == DEFAULT_ERROR_S
    assert thresholds.default_critical_s == DEFAULT_CRITICAL_S
    assert thresholds.dedup_window_s == DEFAULT_DEDUP_WINDOW_S
    assert thresholds.families == {}
