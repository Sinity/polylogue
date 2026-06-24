"""Live/archive-oriented validation lane declarations."""

from __future__ import annotations

from devtools.lane_models import LaneEntry
from polylogue.scenarios import (
    PipelineProbeInputMode,
    PipelineProbeRequest,
    build_live_insight_surface_lanes,
    build_live_operational_surface_lanes,
    build_memory_budget_operational_surface_lanes,
    devtools_execution,
    memory_budget_execution,
    pipeline_probe_execution,
    polylogue_execution,
)


def _live_insight_lanes() -> dict[str, LaneEntry]:
    return {
        spec.name: LaneEntry(
            name=spec.name,
            description=spec.description,
            timeout_s=spec.timeout_s,
            category="live",
            execution=polylogue_execution(*spec.args),
            tags=spec.tags,
        )
        for spec in build_live_insight_surface_lanes()
    }


def _live_operational_lanes() -> dict[str, LaneEntry]:
    return {
        spec.name: LaneEntry(
            name=spec.name,
            description=spec.description,
            timeout_s=spec.timeout_s,
            category="live",
            execution=polylogue_execution(*spec.args),
            tags=spec.tags,
        )
        for spec in build_live_operational_surface_lanes()
    }


def _memory_budget_operational_lanes() -> dict[str, LaneEntry]:
    return {
        spec.name: LaneEntry(
            name=spec.name,
            description=spec.description,
            timeout_s=spec.timeout_s,
            category="live",
            execution=memory_budget_execution(
                spec.max_rss_mb or 1024,
                polylogue_execution(*spec.args),
            ),
            tags=spec.tags,
        )
        for spec in build_memory_budget_operational_surface_lanes()
    }


LIVE_LANES: dict[str, LaneEntry] = {
    "live-archive-subset-parse-probe": LaneEntry(
        name="live-archive-subset-parse-probe",
        description="Live archive medium archive-subset parse probe with persisted manifest/workdir artifacts",
        timeout_s=1800,
        category="live",
        execution=pipeline_probe_execution(
            PipelineProbeRequest(
                input_mode=PipelineProbeInputMode.ARCHIVE_SUBSET,
                stage="parse",
                sample_per_provider=50,
                workdir="/tmp/polylogue-live-archive-subset-parse-probe",
                json_out="/tmp/polylogue-live-archive-subset-parse-probe.json",
            )
        ),
        tags=("live", "probe", "parse"),
    ),
    "live-archive-smoke": LaneEntry(
        name="live-archive-smoke",
        description="Manual lab archive-smoke lane",
        timeout_s=1800,
        category="live",
        execution=devtools_execution("lab smoke", "run", "archive-smoke", "--live", "--tier", "0", "--json"),
    ),
    **_live_operational_lanes(),
    **_live_insight_lanes(),
    **_memory_budget_operational_lanes(),
}

__all__ = ["LIVE_LANES"]
