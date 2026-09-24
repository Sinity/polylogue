"""Human-facing archive convergence warnings for query result surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

logger = get_logger(__name__)


def convergence_warning_line(active_archive: Path | None = None) -> str | None:
    """Return a concise warning when archive results may be partial.

    ``None`` means "checked, and results are complete". Failing to determine
    readiness is not that: this used to swallow every exception and return
    ``None``, so a missing table, an unreadable ops tier, or a schema drift
    rendered exactly like a healthy archive and partial results were presented as
    complete. An unanswerable check now says so.
    """
    try:
        from polylogue.paths import archive_root
        from polylogue.storage.archive_readiness import (
            RawMaterializationAssessmentState,
            assess_raw_materialization,
            raw_materialization_readiness_snapshot,
        )

        root = active_archive or archive_root()
        raw_readiness = raw_materialization_readiness_snapshot(root)
        assessment = assess_raw_materialization(raw_readiness)
        if assessment.state is RawMaterializationAssessmentState.POPULATED_CONVERGED:
            return None
        return _raw_materialization_warning(
            raw_readiness,
            is_unmeasured=assessment.state is RawMaterializationAssessmentState.UNMEASURED,
            assessment_reason=assessment.reason,
            assessment_detail=assessment.detail,
        )
    except Exception:
        logger.warning("convergence warning probe failed; reporting readiness as undetermined", exc_info=True)
        return _UNDETERMINED


#: The caveat every undeterminable readiness check renders, whether the probe
#: raised or returned an ``available: False`` snapshot.
_UNDETERMINED = "Archive convergence state could not be determined; results may be partial."


def _raw_materialization_warning(
    readiness: dict[str, object],
    *,
    is_unmeasured: bool,
    assessment_reason: str,
    assessment_detail: str | None,
) -> str | None:
    if is_unmeasured:
        # An unavailable snapshot returns NORMALLY (a missing source/index
        # tier, an unreadable ops tier, a schema drift), so the caller's
        # ``except`` arm never fires for it. Returning ``None`` here claimed
        # "checked, and results are complete" for a check that never ran, and
        # discarded the reason the snapshot carried.
        detail = assessment_detail or readiness.get("error") or readiness.get("reason")
        if not detail and assessment_reason == "zero_denominator":
            detail = "no raw artifacts: raw materialization is undefined at a zero denominator, not converged"
        if detail:
            return f"{_UNDETERMINED} ({detail})"
        return _UNDETERMINED
    rows = (
        _safe_int(readiness.get("actionable"))
        + _safe_int(readiness.get("blocked"))
        + _safe_int(readiness.get("affected_open"))
    )
    if rows <= 0:
        unchecked = _safe_int(readiness.get("affected_unchecked")) or _safe_int(readiness.get("unchecked"))
        if unchecked > 0:
            raw_count = _safe_int(readiness.get("raw_artifact_count"))
            materialized_count = _safe_int(readiness.get("materialized_raw_artifact_count"))
            prefix = ""
            if raw_count > 0:
                prefix = f"{materialized_count:,}/{raw_count:,} raw artifact(s) materialized; "
            return (
                f"Archive materialization needs classification: {prefix}{unchecked:,} raw/index join gap(s) found; "
                "results may be partial until daemon convergence classifies them."
            )
        return f"Archive raw materialization is not converged ({assessment_reason}); results may be partial."
    category_counts = readiness.get("category_counts")
    parse_failed = 0
    if isinstance(category_counts, dict):
        parse_failed = _safe_int(category_counts.get("parse_failed"))
    parse_failed = parse_failed or _safe_int(readiness.get("affected_actionable"))
    details = ""
    if parse_failed > 0:
        details = f"; {parse_failed:,} parse-failed raw artifact(s)"
    return (
        f"Archive has raw materialization debt: {rows:,} issue group(s){details}; "
        "results may be partial for affected source artifacts."
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["convergence_warning_line"]
