"""Live/archive-oriented validation lane declarations."""

from __future__ import annotations

from functools import partial

from devtools.validation_lane_base import memory_budget_lane as _memory_budget_lane
from devtools.validation_lane_base import pipeline_probe_lane as _pipeline_probe_lane
from devtools.validation_lane_base import polylogue_lane as _polylogue_lane
from polylogue.scenarios import (
    PipelineProbeInputMode,
    build_live_operational_surface_lanes,
    build_live_product_surface_lanes,
    build_memory_budget_operational_surface_lanes,
    polylogue_execution,
)

memory_budget_lane = partial(_memory_budget_lane, category="live")
pipeline_probe_lane = partial(_pipeline_probe_lane, category="live")
polylogue_lane = partial(_polylogue_lane, category="live")


def _live_product_lanes() -> dict[str, object]:
    return {
        spec.name: polylogue_lane(
            spec.name,
            spec.description,
            spec.timeout_s,
            *spec.args,
            tags=spec.tags,
        )
        for spec in build_live_product_surface_lanes()
    }


def _live_operational_lanes() -> dict[str, object]:
    return {
        spec.name: polylogue_lane(
            spec.name,
            spec.description,
            spec.timeout_s,
            *spec.args,
            tags=spec.tags,
        )
        for spec in build_live_operational_surface_lanes()
    }


def _memory_budget_operational_lanes() -> dict[str, object]:
    return {
        spec.name: memory_budget_lane(
            spec.name,
            spec.description,
            spec.timeout_s,
            max_rss_mb=spec.max_rss_mb or 1024,
            execution=polylogue_execution(*spec.args),
            tags=spec.tags,
        )
        for spec in build_memory_budget_operational_surface_lanes()
    }

LIVE_LANES = {
    "live-archive-subset-parse-probe": pipeline_probe_lane(
        "live-archive-subset-parse-probe",
        "Live archive medium archive-subset parse probe with persisted manifest/workdir artifacts",
        1800,
        input_mode=PipelineProbeInputMode.ARCHIVE_SUBSET,
        stage="parse",
        sample_per_provider=50,
        workdir="/tmp/polylogue-live-archive-subset-parse-probe",
        json_out="/tmp/polylogue-live-archive-subset-parse-probe.json",
        tags=("live", "probe", "parse"),
    ),
    "live-exercises": polylogue_lane(
        "live-exercises",
        "Manual live archive showcase/QA exercise lane",
        1800,
        "audit",
        "--live",
        "--only",
        "exercises",
        "--tier",
        "0",
        "--json",
    ),
    **_live_operational_lanes(),
    **_live_product_lanes(),
    **_memory_budget_operational_lanes(),
}


__all__ = ["LIVE_LANES"]
