"""Aggregate/governance derived-model builders."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_rows
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION


def build_aggregate_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "session_tag_rollups": DerivedModelStatus(
            name="session_tag_rollups",
            ready=bool(metrics["tag_rollups_ready"]),
            detail=(
                f"Session tag rollups ready ({metrics['tag_rollup_rows']:,}/{metrics['expected_tag_rollup_rows']:,} rows)"
                if bool(metrics["tag_rollups_ready"])
                else f"Session tag rollups pending ({metrics['tag_rollup_rows']:,}/{metrics['expected_tag_rollup_rows']:,} rows)"
            ),
            source_rows=int(metrics["expected_tag_rollup_rows"]),
            materialized_rows=int(metrics["tag_rollup_rows"]),
            pending_rows=pending_rows(int(metrics["expected_tag_rollup_rows"]), int(metrics["tag_rollup_rows"])),
            stale_rows=int(metrics["stale_tag_rollup_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_tag_rollup_rows"]) == 0),
        ),
        "day_session_summaries": DerivedModelStatus(
            name="day_session_summaries",
            ready=bool(metrics["day_summaries_ready"]),
            detail=(
                f"Day session summaries ready ({metrics['day_summary_rows']:,}/{metrics['expected_day_summary_rows']:,} rows)"
                if bool(metrics["day_summaries_ready"])
                else f"Day session summaries pending ({metrics['day_summary_rows']:,}/{metrics['expected_day_summary_rows']:,} rows)"
            ),
            source_rows=int(metrics["expected_day_summary_rows"]),
            materialized_rows=int(metrics["day_summary_rows"]),
            pending_rows=pending_rows(int(metrics["expected_day_summary_rows"]), int(metrics["day_summary_rows"])),
            stale_rows=int(metrics["stale_day_summary_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_day_summary_rows"]) == 0),
        ),
        "week_session_summaries": DerivedModelStatus(
            name="week_session_summaries",
            ready=bool(metrics["week_summaries_ready"]),
            detail=(
                "Week session summaries ready (derived from day-session summaries)"
                if bool(metrics["week_summaries_ready"])
                else "Week session summaries pending (day-session summaries not ready)"
            ),
            source_rows=int(metrics["expected_day_summary_rows"]),
            materialized_rows=int(metrics["day_summary_rows"]),
            pending_rows=pending_rows(int(metrics["expected_day_summary_rows"]), int(metrics["day_summary_rows"])),
            stale_rows=int(metrics["stale_day_summary_rows"]),
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            matches_version=bool(int(metrics["stale_day_summary_rows"]) == 0),
        ),
    }


__all__ = ["build_aggregate_statuses"]
