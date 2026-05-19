"""Per-source-family cursor-lag SLO alert evaluation (#1232).

The daemon tracks per-file ingest progress in ``live_cursor``
(:mod:`polylogue.sources.live.cursor`). When the cursor for a source file
stops advancing while there is still unprocessed data — either the file grew
beyond the cursor offset or a parse/convergence failure is pending — that
source has incurred ingest *lag*. This module turns the per-family lag
projection from :mod:`polylogue.daemon.cursor_lag_status` into typed,
deduplicated :class:`~polylogue.daemon.health.HealthAlert` rows with an
escalation ladder (``warning → error → critical``) driven by per-family
thresholds in ``polylogue.toml``.

Configuration table (``polylogue.toml``)::

    [health.cursor_lag]
    default_warning_s = 300        # warn when any stuck cursor lags this long
    default_error_s = 1800         # escalate to ERROR at this lag
    default_critical_s = 7200      # escalate to CRITICAL at this lag
    dedup_window_s = 3600          # don't re-fire the same alert within N seconds

    [health.cursor_lag.families.claude-code-session]
    warning_s = 60
    error_s = 300
    critical_s = 1800

    [health.cursor_lag.families.chatgpt-export]
    warning_s = 7200
    error_s = 86400

Idle cursors (caught up, no failure) do not produce alerts at any lag — only
stuck cursors do. This is the "stuck vs idle" distinction the SLO depends on:
without it, a long-quiet ``chatgpt-export`` would page the operator at every
threshold crossing despite there being no actual ingest backlog.

When the worst stuck-lag in a family falls back below its warning threshold,
a single ``severity = ok`` resolution alert fires. Dedup state is in-memory
and process-wide; tests pass their own state for hermetic evaluation.

The escalation ladder ``warning → error → critical`` always fires immediately
on a severity change, even within the dedup window — operators should learn
about an escalation the moment it happens, not after the window expires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polylogue.config import PolylogueConfig, load_polylogue_config
from polylogue.daemon.cursor_lag_status import CursorLagSummary
from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

# Default per-family thresholds when neither the global default nor a family
# override is set. The defaults bias toward conservative SLO ladder steps
# (5m / 30m / 2h) suitable for a mixed personal archive; per-family overrides
# tighten or relax for high-traffic or long-tail sources respectively.
DEFAULT_WARNING_S = 300
DEFAULT_ERROR_S = 1800
DEFAULT_CRITICAL_S = 7200
DEFAULT_DEDUP_WINDOW_S = 3600


@dataclass(frozen=True, slots=True)
class FamilyLagThreshold:
    """Per-source-family lag thresholds (seconds).

    A ``None`` field falls back to the global default.
    """

    warning_s: int | None = None
    error_s: int | None = None
    critical_s: int | None = None


@dataclass(frozen=True, slots=True)
class CursorLagThresholds:
    """Resolved threshold configuration for cursor-lag alerts."""

    default_warning_s: int = DEFAULT_WARNING_S
    default_error_s: int = DEFAULT_ERROR_S
    default_critical_s: int = DEFAULT_CRITICAL_S
    dedup_window_s: int = DEFAULT_DEDUP_WINDOW_S
    families: dict[str, FamilyLagThreshold] = field(default_factory=dict)

    def for_family(self, family: str) -> tuple[int, int, int]:
        """Return ``(warning_s, error_s, critical_s)`` thresholds for ``family``.

        Misconfigured ladders (where a higher-severity threshold is smaller
        than a lower-severity one) are clamped upward so the alert ladder
        stays monotonic — otherwise a band would collapse and one severity
        level would become unreachable.
        """
        override = self.families.get(family)
        warning = override.warning_s if override and override.warning_s is not None else self.default_warning_s
        error = override.error_s if override and override.error_s is not None else self.default_error_s
        critical = override.critical_s if override and override.critical_s is not None else self.default_critical_s
        if error < warning:
            error = warning
        if critical < error:
            critical = error
        return warning, error, critical


@dataclass(slots=True)
class CursorLagDedupState:
    """In-memory dedup state keyed by family.

    Persists for the daemon process lifetime; resets on restart. Stores the
    most recently emitted ``(severity, unix-epoch)`` per family so escalations
    fire immediately while same-severity repeats are suppressed for
    ``dedup_window_s`` seconds.
    """

    last_emit_at: dict[str, tuple[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_thresholds_from_config(cfg: PolylogueConfig | None = None) -> CursorLagThresholds:
    """Load cursor-lag thresholds from the resolved Polylogue config.

    Reads the ``health.cursor_lag`` raw dict written by
    :func:`polylogue.config._merge_toml`. Unknown families are accepted — the
    operator may pre-declare overrides for families the local archive has not
    yet seen.
    """
    cfg = cfg if cfg is not None else load_polylogue_config()
    raw = cfg.raw.get("health_cursor_lag")
    if not isinstance(raw, dict):
        return CursorLagThresholds()
    default_warning = _coerce_int(raw.get("default_warning_s"), DEFAULT_WARNING_S)
    default_error = _coerce_int(raw.get("default_error_s"), DEFAULT_ERROR_S)
    default_critical = _coerce_int(raw.get("default_critical_s"), DEFAULT_CRITICAL_S)
    dedup_window_s = _coerce_int(raw.get("dedup_window_s"), DEFAULT_DEDUP_WINDOW_S)
    families_raw = raw.get("families")
    families: dict[str, FamilyLagThreshold] = {}
    if isinstance(families_raw, dict):
        for family, overrides in families_raw.items():
            if not isinstance(family, str) or not isinstance(overrides, dict):
                continue
            families[family] = FamilyLagThreshold(
                warning_s=_coerce_optional_int(overrides.get("warning_s")),
                error_s=_coerce_optional_int(overrides.get("error_s")),
                critical_s=_coerce_optional_int(overrides.get("critical_s")),
            )
    return CursorLagThresholds(
        default_warning_s=default_warning,
        default_error_s=default_error,
        default_critical_s=default_critical,
        dedup_window_s=dedup_window_s,
        families=families,
    )


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------


def evaluate_cursor_lag(
    summary: CursorLagSummary,
    *,
    thresholds: CursorLagThresholds,
    state: CursorLagDedupState,
    now: float | None = None,
) -> list[HealthAlert]:
    """Emit alerts for per-family cursor-lag SLO breaches.

    Returns one alert per family whose worst stuck lag crossed a threshold
    since the last emission. Families whose worst lag returns below the
    warning threshold get a single resolution alert.

    Dedup: within ``thresholds.dedup_window_s`` seconds of the previous
    emission for the same family at the same severity, no new alert fires.
    Severity escalations (warning -> error -> critical) and resolutions
    (any -> ok) always fire immediately.
    """
    now_ts = now if now is not None else time.time()
    iso_now = datetime.fromtimestamp(now_ts, tz=UTC).isoformat()

    family_state: dict[str, CursorLagFamilySnapshot] = {}
    for entry in summary.family_summaries:
        family_state[entry.family] = CursorLagFamilySnapshot(
            family=entry.family,
            max_lag_s=entry.max_lag_s,
            stuck_file_count=entry.stuck_file_count,
        )

    alerts: list[HealthAlert] = []
    # The universe of families to consider: anything currently observed plus
    # any family with a previously-recorded non-ok emission (so its
    # resolution alert can fire).
    candidate_families: set[str] = set(family_state) | {
        name for name, (severity, _) in state.last_emit_at.items() if severity != HealthSeverity.OK.value
    }
    for family in sorted(candidate_families):
        snapshot = family_state.get(family) or CursorLagFamilySnapshot(family=family)
        warn_s, err_s, crit_s = thresholds.for_family(family)
        severity = _resolve_severity(
            max_lag_s=snapshot.max_lag_s,
            stuck_count=snapshot.stuck_file_count,
            warning_s=warn_s,
            error_s=err_s,
            critical_s=crit_s,
        )
        prev = state.last_emit_at.get(family)

        if not _should_emit(severity, prev, now_ts, thresholds.dedup_window_s):
            continue

        alerts.append(
            _build_alert(
                family=family,
                snapshot=snapshot,
                warning_s=warn_s,
                error_s=err_s,
                critical_s=crit_s,
                severity=severity,
                iso_now=iso_now,
            )
        )
        state.last_emit_at[family] = (severity.value, now_ts)

    return alerts


@dataclass(frozen=True, slots=True)
class CursorLagFamilySnapshot:
    """Per-evaluation snapshot of one family's observed lag state."""

    family: str
    max_lag_s: float = 0.0
    stuck_file_count: int = 0


def _resolve_severity(
    *,
    max_lag_s: float,
    stuck_count: int,
    warning_s: int,
    error_s: int,
    critical_s: int,
) -> HealthSeverity:
    if stuck_count == 0:
        return HealthSeverity.OK
    if max_lag_s >= critical_s:
        return HealthSeverity.CRITICAL
    if max_lag_s >= error_s:
        return HealthSeverity.ERROR
    if max_lag_s >= warning_s:
        return HealthSeverity.WARNING
    return HealthSeverity.OK


def _should_emit(
    severity: HealthSeverity,
    prev: tuple[str, float] | None,
    now_ts: float,
    dedup_window_s: int,
) -> bool:
    """Decide whether to emit an alert given dedup state.

    - First-ever non-ok observation: emit.
    - Severity changed from the last emission: emit (escalations and
      resolutions surface immediately).
    - Same severity within ``dedup_window_s``: suppress.
    - Same severity after ``dedup_window_s``: re-emit so a long-running
      incident shows up in the notification log at least once per window.
    - Resolution (now OK) when previous emission was already OK or there
      is no previous emission: suppress.
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
    snapshot: CursorLagFamilySnapshot,
    warning_s: int,
    error_s: int,
    critical_s: int,
    severity: HealthSeverity,
    iso_now: str,
) -> HealthAlert:
    check_name = f"cursor_lag[{family}]"
    if severity == HealthSeverity.OK:
        message = f"cursor lag for {family} cleared (warn>={warning_s}s)"
    else:
        lag_text = _format_seconds(snapshot.max_lag_s)
        if severity == HealthSeverity.WARNING:
            message = (
                f"cursor lag for {family}: {snapshot.stuck_file_count} stuck file(s), "
                f"worst lag {lag_text} (warning at {warning_s}s, error at {error_s}s)"
            )
        elif severity == HealthSeverity.ERROR:
            message = (
                f"cursor lag for {family}: {snapshot.stuck_file_count} stuck file(s), "
                f"worst lag {lag_text} (error threshold {error_s}s, critical at {critical_s}s)"
            )
        else:  # CRITICAL
            message = (
                f"cursor lag for {family}: {snapshot.stuck_file_count} stuck file(s), "
                f"worst lag {lag_text} (critical threshold {critical_s}s)"
            )
    return HealthAlert(
        check_name=check_name,
        tier=HealthTier.MEDIUM,
        severity=severity,
        message=message,
        checked_at=iso_now,
        consecutive_failures=0,
    )


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Process-wide singleton for the periodic health loop
# ---------------------------------------------------------------------------


_DEDUP_STATE = CursorLagDedupState()


def get_default_dedup_state() -> CursorLagDedupState:
    """Return the daemon's shared dedup state.

    Used by the periodic health loop. Tests should construct their own
    :class:`CursorLagDedupState` instance.
    """
    return _DEDUP_STATE


def reset_default_dedup_state() -> None:
    """Reset the daemon's shared dedup state.

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


__all__ = [
    "DEFAULT_CRITICAL_S",
    "DEFAULT_DEDUP_WINDOW_S",
    "DEFAULT_ERROR_S",
    "DEFAULT_WARNING_S",
    "CursorLagDedupState",
    "CursorLagFamilySnapshot",
    "CursorLagThresholds",
    "FamilyLagThreshold",
    "evaluate_cursor_lag",
    "get_default_dedup_state",
    "load_thresholds_from_config",
    "reset_default_dedup_state",
]
