"""Auto-calibrating cursor-lag anomaly alerts (#1349, ambitious-expansion of #1232).

The static cursor-lag SLO ladder in :mod:`polylogue.daemon.cursor_lag_alert`
fires at hardcoded thresholds. That ladder is the hard "page-me-now" floor —
it is preserved verbatim by this module. The anomaly band adds a *second*,
softer alert signal that catches stalls anomalous-for-this-family even when
absolute lag has not yet crossed the global default.

Three fail-open properties keep the auto-calibration safe:

1. **Additive, not substitutive.** Anomaly alerts use a distinct
   ``check_name = "cursor_lag_anomaly[family]"``. Their dedup state is
   independent; an anomaly emission does not suppress a static alert, and
   removing this check from ``_run_medium_checks`` is behavior-equivalent
   to PR #1346 master.
2. **No CRITICAL tier.** Only the static ladder can page on the critical
   band — anomaly evidence is informational/escalatory.
3. **Confidence gate + absolute floor.** No alert fires below
   ``anomaly_baseline_min_samples`` (default 50) samples in the rolling
   window, or below ``anomaly_min_lag_s`` (default 30s) absolute lag.
   Prevents "0.5s vs 0.05s baseline → 10x!" noise.

Configuration table (``polylogue.toml``)::

    [health.cursor_lag]
    # Existing static-ladder keys preserved from #1232 / PR #1346.
    default_warning_s = 300
    default_error_s = 1800
    default_critical_s = 7200

    # New anomaly-band keys:
    anomaly_enabled = true
    anomaly_baseline_window_days = 7
    anomaly_baseline_min_samples = 50
    anomaly_warning_multiplier = 5.0
    anomaly_error_multiplier = 20.0
    anomaly_min_lag_s = 30
    retention_days = 14

    [health.cursor_lag.families.claude-code-session]
    # Existing per-family static overrides still apply. New per-family
    # anomaly knobs (all optional, all fall back to defaults):
    anomaly_warning_multiplier = 3.0
    anomaly_error_multiplier = 10.0

When a family's worst stuck lag falls back below the warning multiplier, a
single ``severity = ok`` resolution alert fires. Severity escalations and
resolutions always fire immediately through the dedup window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polylogue.config import PolylogueConfig, load_polylogue_config
from polylogue.daemon.cursor_lag_baseline import FamilyBaseline
from polylogue.daemon.cursor_lag_status import CursorLagSummary
from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

# Defaults are intentionally conservative. The anomaly band is most useful
# at multipliers small enough to catch real anomalies (~5×) but high enough
# to ignore normal jitter. The absolute floor prevents false positives at
# very-low-baseline families.
DEFAULT_ANOMALY_ENABLED = True
DEFAULT_BASELINE_WINDOW_DAYS = 7
DEFAULT_BASELINE_MIN_SAMPLES = 50
DEFAULT_WARNING_MULTIPLIER = 5.0
DEFAULT_ERROR_MULTIPLIER = 20.0
DEFAULT_MIN_LAG_S = 30
DEFAULT_RETENTION_DAYS = 14
DEFAULT_DEDUP_WINDOW_S = 3600


@dataclass(frozen=True, slots=True)
class FamilyAnomalyOverride:
    """Per-source-family overrides for the anomaly band.

    ``None`` fields fall back to global defaults. Family-level
    ``anomaly_enabled`` lets the operator silence a single noisy family
    without disabling the whole anomaly surface.
    """

    enabled: bool | None = None
    warning_multiplier: float | None = None
    error_multiplier: float | None = None
    min_lag_s: int | None = None
    baseline_window_days: int | None = None
    baseline_min_samples: int | None = None


@dataclass(frozen=True, slots=True)
class CursorLagAnomalyThresholds:
    """Resolved threshold configuration for the cursor-lag anomaly band."""

    enabled: bool = DEFAULT_ANOMALY_ENABLED
    baseline_window_days: int = DEFAULT_BASELINE_WINDOW_DAYS
    baseline_min_samples: int = DEFAULT_BASELINE_MIN_SAMPLES
    warning_multiplier: float = DEFAULT_WARNING_MULTIPLIER
    error_multiplier: float = DEFAULT_ERROR_MULTIPLIER
    min_lag_s: int = DEFAULT_MIN_LAG_S
    retention_days: int = DEFAULT_RETENTION_DAYS
    dedup_window_s: int = DEFAULT_DEDUP_WINDOW_S
    families: dict[str, FamilyAnomalyOverride] = field(default_factory=dict)

    def for_family(self, family: str) -> tuple[bool, float, float, int, int, int]:
        """Return resolved ``(enabled, warn_mul, err_mul, min_lag_s, window_days, min_samples)``.

        ``error_multiplier`` is clamped to be at least the warning multiplier
        so the alert ladder stays monotonic — a misconfigured ``err < warn``
        would otherwise collapse the warning band.
        """
        override = self.families.get(family)
        enabled = self.enabled
        warn = self.warning_multiplier
        err = self.error_multiplier
        min_lag = self.min_lag_s
        window = self.baseline_window_days
        min_samples = self.baseline_min_samples
        if override is not None:
            if override.enabled is not None:
                enabled = override.enabled
            if override.warning_multiplier is not None:
                warn = override.warning_multiplier
            if override.error_multiplier is not None:
                err = override.error_multiplier
            if override.min_lag_s is not None:
                min_lag = override.min_lag_s
            if override.baseline_window_days is not None:
                window = override.baseline_window_days
            if override.baseline_min_samples is not None:
                min_samples = override.baseline_min_samples
        if err < warn:
            err = warn
        return enabled, warn, err, min_lag, window, min_samples


@dataclass(slots=True)
class CursorLagAnomalyDedupState:
    """In-memory dedup state keyed by family.

    Persists for the daemon process lifetime; resets on restart. Independent
    of the static-ladder dedup state — an anomaly emission cannot suppress a
    static escalation and vice versa.
    """

    last_emit_at: dict[str, tuple[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_anomaly_thresholds_from_config(
    cfg: PolylogueConfig | None = None,
) -> CursorLagAnomalyThresholds:
    """Load anomaly thresholds from ``[health.cursor_lag]``.

    The anomaly keys coexist with the static-ladder keys in the same TOML
    table — they are namespaced by the ``anomaly_`` prefix and the
    ``retention_days`` key. Per-family overrides reuse the existing
    ``[health.cursor_lag.families.<name>]`` sub-table.
    """
    cfg = cfg if cfg is not None else load_polylogue_config()
    raw = cfg.raw.get("health_cursor_lag")
    if not isinstance(raw, dict):
        return CursorLagAnomalyThresholds()
    enabled = _coerce_bool(raw.get("anomaly_enabled"), DEFAULT_ANOMALY_ENABLED)
    window_days = _coerce_int(raw.get("anomaly_baseline_window_days"), DEFAULT_BASELINE_WINDOW_DAYS)
    min_samples = _coerce_int(raw.get("anomaly_baseline_min_samples"), DEFAULT_BASELINE_MIN_SAMPLES)
    warning_multiplier = _coerce_float(raw.get("anomaly_warning_multiplier"), DEFAULT_WARNING_MULTIPLIER)
    error_multiplier = _coerce_float(raw.get("anomaly_error_multiplier"), DEFAULT_ERROR_MULTIPLIER)
    min_lag_s = _coerce_int(raw.get("anomaly_min_lag_s"), DEFAULT_MIN_LAG_S)
    retention_days = _coerce_int(raw.get("retention_days"), DEFAULT_RETENTION_DAYS)
    dedup_window_s = _coerce_int(raw.get("anomaly_dedup_window_s"), DEFAULT_DEDUP_WINDOW_S)
    families_raw = raw.get("families")
    families: dict[str, FamilyAnomalyOverride] = {}
    if isinstance(families_raw, dict):
        for family, overrides in families_raw.items():
            if not isinstance(family, str) or not isinstance(overrides, dict):
                continue
            families[family] = FamilyAnomalyOverride(
                enabled=_coerce_optional_bool(overrides.get("anomaly_enabled")),
                warning_multiplier=_coerce_optional_float(overrides.get("anomaly_warning_multiplier")),
                error_multiplier=_coerce_optional_float(overrides.get("anomaly_error_multiplier")),
                min_lag_s=_coerce_optional_int(overrides.get("anomaly_min_lag_s")),
                baseline_window_days=_coerce_optional_int(overrides.get("anomaly_baseline_window_days")),
                baseline_min_samples=_coerce_optional_int(overrides.get("anomaly_baseline_min_samples")),
            )
    return CursorLagAnomalyThresholds(
        enabled=enabled,
        baseline_window_days=window_days,
        baseline_min_samples=min_samples,
        warning_multiplier=warning_multiplier,
        error_multiplier=error_multiplier,
        min_lag_s=min_lag_s,
        retention_days=retention_days,
        dedup_window_s=dedup_window_s,
        families=families,
    )


# ---------------------------------------------------------------------------
# Anomaly evaluation
# ---------------------------------------------------------------------------


def evaluate_cursor_lag_anomaly(
    summary: CursorLagSummary,
    baselines: dict[str, FamilyBaseline],
    *,
    thresholds: CursorLagAnomalyThresholds,
    state: CursorLagAnomalyDedupState,
    now: float | None = None,
) -> list[HealthAlert]:
    """Emit alerts for per-family cursor-lag anomalies.

    Returns one alert per family whose ``(current_max_lag_s, baseline_p95)``
    pair crosses a multiplier threshold since the last emission, subject to:

    - the family is enabled in config (global + per-family),
    - the baseline is confident (sample_count >= ``baseline_min_samples``),
    - current absolute lag >= ``min_lag_s`` (absolute floor),
    - at least one stuck file in this family (no anomaly alert for an idle
      family even if a stale baseline says it should be low).

    Severity escalations (warning -> error) and resolutions (any -> ok)
    always fire immediately through the dedup window.
    """
    if not thresholds.enabled:
        return []

    now_ts = now if now is not None else time.time()
    iso_now = datetime.fromtimestamp(now_ts, tz=UTC).isoformat()

    family_state: dict[str, _Snapshot] = {}
    for entry in summary.family_summaries:
        family_state[entry.family] = _Snapshot(
            family=entry.family,
            max_lag_s=entry.max_lag_s,
            stuck_file_count=entry.stuck_file_count,
        )

    alerts: list[HealthAlert] = []
    # Consider every observed family plus any family with a pending non-ok
    # emission (so its resolution alert can fire).
    candidate_families: set[str] = set(family_state) | {
        name for name, (severity, _) in state.last_emit_at.items() if severity != HealthSeverity.OK.value
    }
    for family in sorted(candidate_families):
        snapshot = family_state.get(family) or _Snapshot(family=family)
        enabled, warn_mul, err_mul, min_lag, _window, _min_samples = thresholds.for_family(family)
        if not enabled:
            # Disabled per-family — never alert and clear any pending dedup
            # state so a re-enable does not spuriously fire a resolution.
            state.last_emit_at.pop(family, None)
            continue
        baseline = baselines.get(family)
        severity = _resolve_severity(
            snapshot=snapshot,
            baseline=baseline,
            warning_multiplier=warn_mul,
            error_multiplier=err_mul,
            min_lag_s=min_lag,
        )
        prev = state.last_emit_at.get(family)

        if not _should_emit(severity, prev, now_ts, thresholds.dedup_window_s):
            continue

        alerts.append(
            _build_alert(
                family=family,
                snapshot=snapshot,
                baseline=baseline,
                warning_multiplier=warn_mul,
                error_multiplier=err_mul,
                severity=severity,
                iso_now=iso_now,
            )
        )
        state.last_emit_at[family] = (severity.value, now_ts)

    return alerts


@dataclass(frozen=True, slots=True)
class _Snapshot:
    family: str
    max_lag_s: float = 0.0
    stuck_file_count: int = 0


def _resolve_severity(
    *,
    snapshot: _Snapshot,
    baseline: FamilyBaseline | None,
    warning_multiplier: float,
    error_multiplier: float,
    min_lag_s: int,
) -> HealthSeverity:
    # Idle family — never an anomaly.
    if snapshot.stuck_file_count == 0:
        return HealthSeverity.OK
    # Absolute floor — bias against false positives at very-low-baseline
    # families where small variance translates to large multipliers.
    if snapshot.max_lag_s < min_lag_s:
        return HealthSeverity.OK
    # Unconfident baseline — the warm-up window is silent rather than wrong.
    if baseline is None or not baseline.confident:
        return HealthSeverity.OK
    # Degenerate baseline — a p95 of zero would make every observation
    # "infinite multiplier"; treat as silent because we have no signal to
    # anomaly-against. The static ladder still catches absolute lag.
    if baseline.rolling_p95_lag_s <= 0.0:
        return HealthSeverity.OK
    multiplier = snapshot.max_lag_s / baseline.rolling_p95_lag_s
    if multiplier >= error_multiplier:
        return HealthSeverity.ERROR
    if multiplier >= warning_multiplier:
        return HealthSeverity.WARNING
    return HealthSeverity.OK


def _should_emit(
    severity: HealthSeverity,
    prev: tuple[str, float] | None,
    now_ts: float,
    dedup_window_s: int,
) -> bool:
    """Match the static-ladder dedup discipline.

    - First-ever non-ok observation: emit.
    - Severity changed from the last emission: emit (escalations and
      resolutions surface immediately).
    - Same severity within ``dedup_window_s``: suppress.
    - Same severity after ``dedup_window_s``: re-emit so a long-running
      anomaly surfaces in the notification log at least once per window.
    - Resolution (now OK) when previous was already OK or absent: suppress.
    """
    if prev is None:
        return severity != HealthSeverity.OK
    prev_severity, prev_ts = prev
    if severity.value != prev_severity:
        return True
    if severity == HealthSeverity.OK:
        return False
    return (now_ts - prev_ts) >= dedup_window_s


def _build_alert(
    *,
    family: str,
    snapshot: _Snapshot,
    baseline: FamilyBaseline | None,
    warning_multiplier: float,
    error_multiplier: float,
    severity: HealthSeverity,
    iso_now: str,
) -> HealthAlert:
    check_name = f"cursor_lag_anomaly[{family}]"
    baseline_p95 = baseline.rolling_p95_lag_s if baseline is not None else 0.0
    sample_count = baseline.sample_count if baseline is not None else 0
    if severity == HealthSeverity.OK:
        message = (
            f"cursor-lag anomaly for {family} cleared (baseline p95 {_fmt(baseline_p95)} over {sample_count} samples)"
        )
    else:
        multiplier = snapshot.max_lag_s / baseline_p95 if baseline_p95 > 0 else 0.0
        if severity == HealthSeverity.WARNING:
            ladder = f"warning at {warning_multiplier:.1f}x, error at {error_multiplier:.1f}x"
        else:  # ERROR — anomaly has no CRITICAL tier on purpose
            ladder = f"error threshold {error_multiplier:.1f}x"
        message = (
            f"cursor-lag anomaly for {family}: worst lag {_fmt(snapshot.max_lag_s)} "
            f"is {multiplier:.1f}x rolling p95 baseline {_fmt(baseline_p95)} "
            f"({sample_count} samples; {ladder})"
        )
    return HealthAlert(
        check_name=check_name,
        tier=HealthTier.MEDIUM,
        severity=severity,
        message=message,
        checked_at=iso_now,
        consecutive_failures=0,
    )


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Process-wide singleton for the periodic health loop
# ---------------------------------------------------------------------------


_DEDUP_STATE = CursorLagAnomalyDedupState()


def get_default_dedup_state() -> CursorLagAnomalyDedupState:
    """Return the daemon's shared anomaly dedup state.

    Used by the periodic health loop. Tests construct their own state for
    hermetic evaluation.
    """
    return _DEDUP_STATE


def reset_default_dedup_state() -> None:
    """Clear the daemon's shared anomaly dedup state.

    Exposed for daemon shutdown/restart in tests; not called in production.
    """
    _DEDUP_STATE.last_emit_at.clear()


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
    return None


__all__ = [
    "DEFAULT_ANOMALY_ENABLED",
    "DEFAULT_BASELINE_MIN_SAMPLES",
    "DEFAULT_BASELINE_WINDOW_DAYS",
    "DEFAULT_DEDUP_WINDOW_S",
    "DEFAULT_ERROR_MULTIPLIER",
    "DEFAULT_MIN_LAG_S",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_WARNING_MULTIPLIER",
    "CursorLagAnomalyDedupState",
    "CursorLagAnomalyThresholds",
    "FamilyAnomalyOverride",
    "evaluate_cursor_lag_anomaly",
    "get_default_dedup_state",
    "load_anomaly_thresholds_from_config",
    "reset_default_dedup_state",
]
