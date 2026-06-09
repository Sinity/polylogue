"""Per-source-family convergence-debt alert evaluation (#1226).

The daemon records post-ingest convergence failures in
``live_convergence_debt``. A simple ``failed_count > 0`` rule fires
indiscriminately: a stuck claude-code-session debt is more urgent than a
stuck chatgpt one, and a low background count of debt across a long-running
archive is normal noise. This module turns the raw debt summary into
typed, deduplicated :class:`~polylogue.daemon.health.HealthAlert` rows
using per-source-family thresholds drawn from ``polylogue.toml``.

Configuration table (``polylogue.toml``)::

    [health.convergence_debt]
    default_warning = 1           # debt count that produces a WARNING alert
    default_error = 10            # debt count that produces an ERROR alert
    dedup_window_s = 3600         # don't re-fire the same alert within N seconds

    [health.convergence_debt.families.claude-code-session]
    warning = 1
    error = 5

    [health.convergence_debt.families.chatgpt-export]
    warning = 25
    error = 200

When a family falls back below its warning threshold, a single
``severity = ok`` resolution alert fires. After firing (warning, error or
resolution) the in-memory state suppresses duplicate alerts for
``dedup_window_s`` seconds so the periodic health loop does not flood
notification backends.

The module is intentionally pure aside from one process-wide
:class:`AlertDedupState` that the periodic health loop owns. Tests pass
their own dedup state to keep evaluations hermetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polylogue.config import PolylogueConfig, load_polylogue_config
from polylogue.daemon.convergence_debt_status import ConvergenceDebtSummary
from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

# Default thresholds when neither the global default nor a family override is set.
DEFAULT_WARNING_COUNT = 1
DEFAULT_ERROR_COUNT = 10
DEFAULT_DEDUP_WINDOW_S = 3600


@dataclass(frozen=True, slots=True)
class FamilyThreshold:
    """Per-source-family debt thresholds.

    A ``None`` field falls back to the global default.
    """

    warning: int | None = None
    error: int | None = None


@dataclass(frozen=True, slots=True)
class ConvergenceDebtThresholds:
    """Resolved threshold configuration for convergence-debt alerts."""

    default_warning: int = DEFAULT_WARNING_COUNT
    default_error: int = DEFAULT_ERROR_COUNT
    dedup_window_s: int = DEFAULT_DEDUP_WINDOW_S
    families: dict[str, FamilyThreshold] = field(default_factory=dict)

    def for_family(self, family: str) -> tuple[int, int]:
        """Return ``(warning, error)`` thresholds for ``family``."""
        override = self.families.get(family)
        warning = override.warning if override and override.warning is not None else self.default_warning
        error = override.error if override and override.error is not None else self.default_error
        # Error threshold must never be below the warning threshold,
        # otherwise the warning band collapses.
        if error < warning:
            error = warning
        return warning, error


@dataclass(slots=True)
class AlertDedupState:
    """In-memory dedup state keyed by ``(family, severity)``.

    Persists for the daemon process lifetime; resets on restart. Stores
    the unix epoch at which the most recent alert was emitted plus the
    severity that was emitted, so an escalation from WARNING -> ERROR
    fires immediately even within the dedup window.
    """

    last_emit_at: dict[str, tuple[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_thresholds_from_config(cfg: PolylogueConfig | None = None) -> ConvergenceDebtThresholds:
    """Load convergence-debt thresholds from the resolved Polylogue config.

    Reads the ``health.convergence_debt`` raw dict written by
    :func:`polylogue.config._merge_toml`. Unknown families are accepted —
    the operator may pre-declare overrides for families that the local
    archive has not yet seen.
    """
    cfg = cfg if cfg is not None else load_polylogue_config()
    raw = cfg.raw.get("health_convergence_debt")
    if not isinstance(raw, dict):
        return ConvergenceDebtThresholds()
    default_warning = _coerce_int(raw.get("default_warning"), DEFAULT_WARNING_COUNT)
    default_error = _coerce_int(raw.get("default_error"), DEFAULT_ERROR_COUNT)
    dedup_window_s = _coerce_int(raw.get("dedup_window_s"), DEFAULT_DEDUP_WINDOW_S)
    families_raw = raw.get("families")
    families: dict[str, FamilyThreshold] = {}
    if isinstance(families_raw, dict):
        for family, overrides in families_raw.items():
            if not isinstance(family, str) or not isinstance(overrides, dict):
                continue
            families[family] = FamilyThreshold(
                warning=_coerce_optional_int(overrides.get("warning")),
                error=_coerce_optional_int(overrides.get("error")),
            )
    return ConvergenceDebtThresholds(
        default_warning=default_warning,
        default_error=default_error,
        dedup_window_s=dedup_window_s,
        families=families,
    )


# ---------------------------------------------------------------------------
# Source-family inference
# ---------------------------------------------------------------------------


# Static map of WatchSource.name -> canonical source-family token. We keep
# this here (not in core.sources) so the alert module stays self-contained
# and does not pull live-watch internals into the substrate.
_WATCHSOURCE_TO_FAMILY: dict[str, str] = {
    "claude-code": "claude-code-session",
    "codex": "codex-session",
    "gemini-cli": "gemini-cli-session",
    "hermes": "hermes-session",
    "antigravity": "antigravity-session",
    "inbox": "inbox",
    "hooks": "hooks",
    "aistudio": "gemini-export",
}


def watchsource_name_to_family(name: str) -> str:
    """Map a watch-source name to its canonical source-family token.

    Returns ``"unknown"`` for unrecognized names.
    """
    return _WATCHSOURCE_TO_FAMILY.get(name, "unknown")


def source_family_for_path(path: Path | str) -> str:
    """Infer the source-family token from a source-file path.

    Matches the path against the configured ``WatchSource`` roots and
    returns the matching family. Returns ``"unknown"`` if no root matches
    or the watch-source name is not recognized.
    """
    try:
        from polylogue.sources.live.watcher import default_sources
    except Exception:
        return "unknown"

    try:
        resolved = Path(path).resolve(strict=False)
    except OSError:
        return "unknown"
    for src in default_sources():
        try:
            src_root = src.root.resolve(strict=False)
        except OSError:
            continue
        try:
            resolved.relative_to(src_root)
        except ValueError:
            continue
        return watchsource_name_to_family(src.name)
    return "unknown"


def source_family_for_subject(subject_type: str, subject_id: str) -> str:
    """Infer the source-family token from a debt subject.

    For ``source_path`` subjects, the subject id is the filesystem path
    and is matched against ``default_sources()``. For ``session_id``
    subjects, source-family attribution requires a DB lookup and is
    currently treated as ``"unknown"`` — those rows fall under the global
    default threshold. Improving session-id attribution is tracked
    as follow-up work.
    """
    if subject_type == "source_path":
        return source_family_for_path(subject_id)
    return "unknown"


# ---------------------------------------------------------------------------
# Per-family aggregation
# ---------------------------------------------------------------------------


def aggregate_debt_by_family(summary: ConvergenceDebtSummary) -> dict[str, int]:
    """Bucket the recent-debt items by source family.

    The summary's ``recent`` list is bounded (10 items by default) but it
    is the only per-subject view the status payload carries. That is
    sufficient to ground a per-family threshold check because the alert
    fires on the visible failed/retry-due counts, not on long-tail
    rollups: the operator-visible severity follows what an operator
    would see in ``polylogue status`` or ``/health``.
    """
    counts: dict[str, int] = {}
    for item in summary.recent:
        family = source_family_for_subject(item.subject_type, item.subject_id)
        counts[family] = counts.get(family, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------


def evaluate_convergence_debt(
    summary: ConvergenceDebtSummary,
    *,
    thresholds: ConvergenceDebtThresholds,
    state: AlertDedupState,
    now: float | None = None,
) -> list[HealthAlert]:
    """Emit alerts for per-family convergence-debt threshold breaches.

    Returns one alert per family whose debt count crossed its warning or
    error threshold since the last emission. Families whose count
    returns below the warning threshold get a single resolution alert.

    Dedup: within ``thresholds.dedup_window_s`` seconds of the previous
    emission for the same family at the same severity, no new alert
    fires. Severity escalations (warning -> error) always fire.
    """
    now_ts = now if now is not None else time.time()
    iso_now = datetime.fromtimestamp(now_ts, tz=UTC).isoformat()
    family_counts = aggregate_debt_by_family(summary)

    alerts: list[HealthAlert] = []
    # Determine the universe of families we might emit for: anything with
    # current debt, plus any family with a previously-recorded non-ok
    # emission (so we can fire its resolution alert).
    candidate_families = set(family_counts) | {
        family for family, (severity, _) in state.last_emit_at.items() if severity != HealthSeverity.OK.value
    }
    for family in sorted(candidate_families):
        count = family_counts.get(family, 0)
        warn, err = thresholds.for_family(family)
        severity = _resolve_severity(count, warn, err)
        prev = state.last_emit_at.get(family)

        if not _should_emit(severity, prev, now_ts, thresholds.dedup_window_s):
            continue

        alerts.append(_build_alert(family=family, count=count, warn=warn, err=err, severity=severity, iso_now=iso_now))
        state.last_emit_at[family] = (severity.value, now_ts)

    return alerts


def _resolve_severity(count: int, warn: int, err: int) -> HealthSeverity:
    if count >= err:
        return HealthSeverity.ERROR
    if count >= warn:
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
      resolutions must surface immediately).
    - Same severity within ``dedup_window_s``: suppress.
    - Same severity after ``dedup_window_s``: re-emit (so a long-running
      incident shows up in the notification log at least once per window).
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
    count: int,
    warn: int,
    err: int,
    severity: HealthSeverity,
    iso_now: str,
) -> HealthAlert:
    check_name = f"convergence_debt[{family}]"
    if severity == HealthSeverity.OK:
        message = f"convergence debt for {family} cleared (threshold warn>={warn})"
    elif severity == HealthSeverity.WARNING:
        message = f"convergence debt for {family}: {count} item(s) (warning at {warn}, error at {err})"
    else:
        message = f"convergence debt for {family}: {count} item(s) (error threshold {err})"
    return HealthAlert(
        check_name=check_name,
        tier=HealthTier.MEDIUM,
        severity=severity,
        message=message,
        checked_at=iso_now,
        consecutive_failures=0,
    )


# ---------------------------------------------------------------------------
# Process-wide singleton for the periodic health loop
# ---------------------------------------------------------------------------


_DEDUP_STATE = AlertDedupState()


def get_default_dedup_state() -> AlertDedupState:
    """Return the daemon's shared dedup state.

    Used by the periodic health loop. Tests should construct their own
    :class:`AlertDedupState` instance.
    """
    return _DEDUP_STATE


def reset_default_dedup_state() -> None:
    """Reset the daemon's shared dedup state.

    Exposed for daemon shutdown/restart in tests; not called in
    production.
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
    "DEFAULT_DEDUP_WINDOW_S",
    "DEFAULT_ERROR_COUNT",
    "DEFAULT_WARNING_COUNT",
    "AlertDedupState",
    "ConvergenceDebtThresholds",
    "FamilyThreshold",
    "aggregate_debt_by_family",
    "evaluate_convergence_debt",
    "get_default_dedup_state",
    "load_thresholds_from_config",
    "reset_default_dedup_state",
    "source_family_for_path",
    "source_family_for_subject",
    "watchsource_name_to_family",
]
