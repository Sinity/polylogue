"""Shared validation-lane config and command builders."""

from __future__ import annotations

from devtools.lane_models import LaneEntry
from polylogue.scenarios import (
    CorpusRequest,
    ExecutionSpec,
    PipelineProbeInputMode,
    PipelineProbeRequest,
    command_execution,
    composite_execution,
    devtools_execution,
    memory_budget_execution,
    pipeline_probe_execution,
    polylogue_execution,
    pytest_execution,
)


def cli_lane(
    name: str,
    description: str,
    timeout_s: int,
    executable: str,
    *args: str,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=command_execution(executable, *args),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def pytest_lane(
    name: str,
    description: str,
    timeout_s: int,
    *args: str,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=pytest_execution(*args),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def devtools_lane(
    name: str,
    description: str,
    timeout_s: int,
    subcommand: str,
    *args: str,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=devtools_execution(subcommand, *args),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def pipeline_probe_lane(
    name: str,
    description: str,
    timeout_s: int,
    *,
    stage: str = "all",
    input_mode: PipelineProbeInputMode | str = PipelineProbeInputMode.SYNTHETIC,
    corpus_request: CorpusRequest | None = None,
    sample_per_provider: int | None = None,
    workdir: str | None = None,
    json_out: str | None = None,
    max_total_ms: float | None = None,
    max_peak_rss_mb: float | None = None,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=pipeline_probe_execution(
            PipelineProbeRequest(
                stage=stage,
                input_mode=input_mode,
                corpus_request=corpus_request,
                sample_per_provider=sample_per_provider,
                workdir=workdir,
                json_out=json_out,
                max_total_ms=max_total_ms,
                max_peak_rss_mb=max_peak_rss_mb,
            )
        ),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def polylogue_lane(
    name: str,
    description: str,
    timeout_s: int,
    *args: str,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=polylogue_execution(*args),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def memory_budget_lane(
    name: str,
    description: str,
    timeout_s: int,
    *,
    max_rss_mb: int,
    execution: ExecutionSpec,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=memory_budget_execution(max_rss_mb, execution),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def composite_lane(
    name: str,
    description: str,
    timeout_s: int,
    *sub_lanes: str,
    category: str,
    family: str | None = None,
    origin: str = "authored.validation-lane.composite",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        family=family,
        execution=composite_execution(*sub_lanes),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


__all__ = [
    "cli_lane",
    "composite_lane",
    "devtools_lane",
    "LaneEntry",
    "memory_budget_lane",
    "pipeline_probe_lane",
    "polylogue_lane",
    "pytest_lane",
]
