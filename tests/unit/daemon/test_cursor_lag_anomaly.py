"""Tests for the cursor-lag anomaly-band alerts (#1349)."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon.cursor_lag_anomaly import (
    DEFAULT_BASELINE_MIN_SAMPLES,
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_DEDUP_WINDOW_S,
    DEFAULT_ERROR_MULTIPLIER,
    DEFAULT_MIN_LAG_S,
    DEFAULT_WARNING_MULTIPLIER,
    CursorLagAnomalyDedupState,
    CursorLagAnomalyThresholds,
    FamilyAnomalyOverride,
    evaluate_cursor_lag_anomaly,
    load_anomaly_thresholds_from_config,
)
from polylogue.daemon.cursor_lag_baseline import FamilyBaseline
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


def _baseline(
    *, family: str = "claude-code-session", p95: float = 10.0, samples: int = 60, confident: bool = True
) -> FamilyBaseline:
    return FamilyBaseline(
        family=family,
        sample_count=samples,
        rolling_median_lag_s=p95 * 0.5,
        rolling_p95_lag_s=p95,
        window_started_at="2026-05-11T00:00:00+00:00",
        confident=confident,
    )


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


def test_for_family_returns_global_defaults_when_no_override() -> None:
    thresholds = CursorLagAnomalyThresholds()
    assert thresholds.for_family("claude-code-session") == (
        True,
        DEFAULT_WARNING_MULTIPLIER,
        DEFAULT_ERROR_MULTIPLIER,
        DEFAULT_MIN_LAG_S,
        DEFAULT_BASELINE_WINDOW_DAYS,
        DEFAULT_BASELINE_MIN_SAMPLES,
    )


def test_for_family_applies_family_overrides() -> None:
    thresholds = CursorLagAnomalyThresholds(
        warning_multiplier=5.0,
        error_multiplier=20.0,
        families={
            "claude-code-session": FamilyAnomalyOverride(warning_multiplier=3.0, error_multiplier=10.0, min_lag_s=60)
        },
    )
    enabled, warn, err, min_lag, _w, _m = thresholds.for_family("claude-code-session")
    assert enabled is True
    assert warn == 3.0
    assert err == 10.0
    assert min_lag == 60


def test_for_family_clamps_error_below_warning_to_warning() -> None:
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=10.0, error_multiplier=5.0)
    enabled, warn, err, *_ = thresholds.for_family("anything")
    assert warn == 10.0
    assert err == 10.0  # clamped


def test_for_family_respects_per_family_disable() -> None:
    thresholds = CursorLagAnomalyThresholds(families={"noisy": FamilyAnomalyOverride(enabled=False)})
    enabled, *_ = thresholds.for_family("noisy")
    assert enabled is False


# ---------------------------------------------------------------------------
# Severity resolution
# ---------------------------------------------------------------------------


def test_disabled_globally_emits_no_alerts() -> None:
    thresholds = CursorLagAnomalyThresholds(enabled=False)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=10_000.0)])

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": _baseline(p95=1.0)}, thresholds=thresholds, state=state, now=0.0
    )
    assert alerts == []


def test_idle_family_emits_no_alerts_regardless_of_baseline() -> None:
    thresholds = CursorLagAnomalyThresholds()
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=0, max_lag_s=0.0, idle=3)])

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": _baseline(p95=1.0)}, thresholds=thresholds, state=state, now=0.0
    )
    assert alerts == []


def test_unconfident_baseline_silences_alert_regardless_of_multiplier() -> None:
    # AC #2: confidence gate must hold.
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=3.0, error_multiplier=10.0)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=10_000.0)])
    baseline = _baseline(p95=1.0, samples=10, confident=False)

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert alerts == []


def test_below_absolute_floor_silences_alert_regardless_of_multiplier() -> None:
    # AC #3: absolute-floor gate must hold. 10x ratio over a 0.1s baseline
    # would normally fire — but 1s absolute lag is below min_lag_s.
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=3.0, error_multiplier=10.0, min_lag_s=30)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=1.0)])
    baseline = _baseline(p95=0.1, samples=100)

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert alerts == []


def test_degenerate_zero_baseline_is_silenced() -> None:
    # A p95 of zero would make every observation "infinite multiplier";
    # the anomaly check refuses to alert in that case — the static
    # ladder still catches absolute lag.
    thresholds = CursorLagAnomalyThresholds()
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=10_000.0)])
    baseline = _baseline(p95=0.0, samples=100)

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert alerts == []


def test_warning_multiplier_threshold_fires_warning() -> None:
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=60.0)])
    baseline = _baseline(p95=10.0)  # 6x ratio

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert len(alerts) == 1
    assert alerts[0].severity == HealthSeverity.WARNING
    assert alerts[0].tier == HealthTier.MEDIUM
    assert alerts[0].check_name == "cursor_lag_anomaly[claude-code-session]"


def test_error_multiplier_threshold_fires_error() -> None:
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=300.0)])
    baseline = _baseline(p95=10.0)  # 30x ratio

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert len(alerts) == 1
    assert alerts[0].severity == HealthSeverity.ERROR


def test_anomaly_has_no_critical_tier() -> None:
    # Even an absurd multiplier never escalates beyond ERROR. Only the
    # static ladder can page on the critical band.
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10)
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=10_000.0)])
    baseline = _baseline(p95=10.0)  # 1000x ratio

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=0.0
    )
    assert len(alerts) == 1
    # Anomaly has no CRITICAL tier on purpose — even 1000x cannot escalate
    # beyond ERROR. Only the static ladder can page on the critical band.
    assert alerts[0].severity == HealthSeverity.ERROR


# ---------------------------------------------------------------------------
# Dedup and escalation
# ---------------------------------------------------------------------------


def test_repeated_alerts_within_dedup_window_are_suppressed() -> None:
    thresholds = CursorLagAnomalyThresholds(
        warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10, dedup_window_s=600
    )
    state = CursorLagAnomalyDedupState()
    summary = _summary([_family(stuck=1, max_lag_s=60.0)])
    baseline = _baseline(p95=10.0)

    first = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=1000.0
    )
    second = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=1100.0
    )
    assert len(first) == 1
    assert second == []


def test_escalation_fires_immediately_through_dedup_window() -> None:
    thresholds = CursorLagAnomalyThresholds(
        warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10, dedup_window_s=3600
    )
    state = CursorLagAnomalyDedupState()
    baseline = _baseline(p95=10.0)

    first = evaluate_cursor_lag_anomaly(
        _summary([_family(stuck=1, max_lag_s=60.0)]),  # 6x → warning
        {"claude-code-session": baseline},
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_cursor_lag_anomaly(
        _summary([_family(stuck=1, max_lag_s=300.0)]),  # 30x → error
        {"claude-code-session": baseline},
        thresholds=thresholds,
        state=state,
        now=10.0,
    )
    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.ERROR


def test_resolution_alert_fires_once_when_anomaly_clears() -> None:
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10)
    state = CursorLagAnomalyDedupState()
    baseline = _baseline(p95=10.0)

    first = evaluate_cursor_lag_anomaly(
        _summary([_family(stuck=1, max_lag_s=60.0)]),
        {"claude-code-session": baseline},
        thresholds=thresholds,
        state=state,
        now=0.0,
    )
    second = evaluate_cursor_lag_anomaly(
        _summary([_family(stuck=1, max_lag_s=20.0)]),  # back below warning
        {"claude-code-session": baseline},
        thresholds=thresholds,
        state=state,
        now=10.0,
    )
    third = evaluate_cursor_lag_anomaly(
        _summary([_family(stuck=1, max_lag_s=20.0)]),
        {"claude-code-session": baseline},
        thresholds=thresholds,
        state=state,
        now=20.0,
    )

    assert first[0].severity == HealthSeverity.WARNING
    assert len(second) == 1
    assert second[0].severity == HealthSeverity.OK
    assert third == []


def test_per_family_disable_drops_pending_dedup_state() -> None:
    # An operator can silence one noisy family without disabling the
    # whole anomaly surface. When a family is disabled mid-run, its
    # pending dedup state is cleared so a re-enable does not spuriously
    # emit a resolution alert against a baseline that was never breached.
    thresholds = CursorLagAnomalyThresholds(
        warning_multiplier=5.0,
        error_multiplier=20.0,
        min_lag_s=10,
        families={"claude-code-session": FamilyAnomalyOverride(enabled=False)},
    )
    state = CursorLagAnomalyDedupState()
    # Pre-seed dedup state as if a previous tick had fired.
    state.last_emit_at["claude-code-session"] = ("warning", 0.0)
    summary = _summary([_family(stuck=1, max_lag_s=60.0)])
    baseline = _baseline(p95=10.0)

    alerts = evaluate_cursor_lag_anomaly(
        summary, {"claude-code-session": baseline}, thresholds=thresholds, state=state, now=10.0
    )
    assert alerts == []
    assert "claude-code-session" not in state.last_emit_at


def test_independent_families_fire_independently() -> None:
    thresholds = CursorLagAnomalyThresholds(warning_multiplier=5.0, error_multiplier=20.0, min_lag_s=10)
    state = CursorLagAnomalyDedupState()
    summary = _summary(
        [
            _family(family="claude-code-session", stuck=1, max_lag_s=60.0),
            _family(family="chatgpt-export", stuck=1, max_lag_s=60.0),
        ]
    )
    baselines = {
        "claude-code-session": _baseline(family="claude-code-session", p95=10.0),
        "chatgpt-export": _baseline(family="chatgpt-export", p95=10.0),
    }

    alerts = evaluate_cursor_lag_anomaly(summary, baselines, thresholds=thresholds, state=state, now=0.0)
    by_check = {a.check_name: a.severity for a in alerts}
    assert by_check["cursor_lag_anomaly[claude-code-session]"] == HealthSeverity.WARNING
    assert by_check["cursor_lag_anomaly[chatgpt-export]"] == HealthSeverity.WARNING


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_anomaly_thresholds_reads_polylogue_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(
        """
[health.cursor_lag]
default_warning_s = 60
default_error_s = 600
default_critical_s = 7200

anomaly_enabled = true
anomaly_baseline_window_days = 3
anomaly_baseline_min_samples = 30
anomaly_warning_multiplier = 4.0
anomaly_error_multiplier = 15.0
anomaly_min_lag_s = 45
retention_days = 21

[health.cursor_lag.families.claude-code-session]
anomaly_warning_multiplier = 2.0
anomaly_error_multiplier = 8.0

[health.cursor_lag.families.noisy-family]
anomaly_enabled = false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_anomaly_thresholds_from_config(cfg)

    assert thresholds.enabled is True
    assert thresholds.baseline_window_days == 3
    assert thresholds.baseline_min_samples == 30
    assert thresholds.warning_multiplier == 4.0
    assert thresholds.error_multiplier == 15.0
    assert thresholds.min_lag_s == 45
    assert thresholds.retention_days == 21

    enabled, warn, err, *_ = thresholds.for_family("claude-code-session")
    assert enabled is True
    assert warn == 2.0
    assert err == 8.0

    enabled, *_ = thresholds.for_family("noisy-family")
    assert enabled is False


def test_load_anomaly_thresholds_defaults_when_no_config_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text("[archive]\nroot = '/tmp/archive'\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg_path))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()
    thresholds = load_anomaly_thresholds_from_config(cfg)
    assert thresholds.warning_multiplier == DEFAULT_WARNING_MULTIPLIER
    assert thresholds.error_multiplier == DEFAULT_ERROR_MULTIPLIER
    assert thresholds.baseline_window_days == DEFAULT_BASELINE_WINDOW_DAYS
    assert thresholds.baseline_min_samples == DEFAULT_BASELINE_MIN_SAMPLES
    assert thresholds.min_lag_s == DEFAULT_MIN_LAG_S
    assert thresholds.dedup_window_s == DEFAULT_DEDUP_WINDOW_S
