"""Human-facing archive convergence warnings for query result surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def convergence_warning_line(active_archive: Path | None = None) -> str | None:
    """Return a concise warning when archive results may be partial."""
    try:
        from polylogue.paths import archive_root
        from polylogue.storage.archive_readiness import (
            active_rebuild_index_attempts,
            raw_materialization_readiness_snapshot,
            raw_materialization_ready,
        )

        root = active_archive or archive_root()
        attempts = active_rebuild_index_attempts(root / "ops.db")
        if attempts:
            return _active_rebuild_warning(attempts)

        raw_readiness = raw_materialization_readiness_snapshot(root)
        if raw_materialization_ready(raw_readiness):
            return None
        return _raw_materialization_warning(raw_readiness)
    except Exception:
        return None


def _active_rebuild_warning(attempts: list[dict[str, object]]) -> str:
    count = len(attempts)
    materialized = sum(_safe_int(attempt.get("materialized_count")) for attempt in attempts)
    parsed = sum(_safe_int(attempt.get("parsed_raw_count")) for attempt in attempts)
    progress = ""
    if materialized or parsed:
        progress = f" ({materialized:,} sessions materialized from {parsed:,} parsed raw rows)"
    return f"Archive is converging: {count:,} index rebuild attempt(s) active{progress}; results may be partial."


def _raw_materialization_warning(readiness: dict[str, object]) -> str | None:
    if not readiness.get("available", False):
        return None
    rows = (
        _safe_int(readiness.get("actionable")) + _safe_int(readiness.get("blocked")) + _safe_int(readiness.get("open"))
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
        return None
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
