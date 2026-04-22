"""Executable pipeline stage specifications."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

StageContextPolicy = Literal["none", "requested_or_leaf", "requested_or_all_after_parse"]


@dataclass(frozen=True, slots=True)
class PipelineStageSpec:
    """Static execution facts for one pipeline leaf stage."""

    name: str
    log_stage: str
    context_policy: StageContextPolicy = "none"
    suspend_fts_triggers: bool = False
    pipeline_managed: bool = True

    def execution_stage(
        self,
        *,
        requested_stage: str,
        explicit_sequence: bool,
        executed_stages: Collection[str],
    ) -> str:
        """Return the stage token expected by the existing stage implementation."""
        if self.context_policy == "none":
            return self.name
        if not explicit_sequence:
            return requested_stage
        if self.context_policy == "requested_or_leaf":
            return self.name
        if "parse" in executed_stages:
            return "all"
        return self.name


PIPELINE_STAGE_SPECS: dict[str, PipelineStageSpec] = {
    "acquire": PipelineStageSpec(name="acquire", log_stage="acquire"),
    "schema": PipelineStageSpec(name="schema", log_stage="schema"),
    "parse": PipelineStageSpec(
        name="parse",
        log_stage="ingest",
        context_policy="requested_or_leaf",
        suspend_fts_triggers=True,
    ),
    "materialize": PipelineStageSpec(
        name="materialize",
        log_stage="materialize",
        context_policy="requested_or_all_after_parse",
    ),
    "render": PipelineStageSpec(
        name="render",
        log_stage="render",
        context_policy="requested_or_all_after_parse",
        suspend_fts_triggers=True,
    ),
    "site": PipelineStageSpec(name="site", log_stage="site"),
    "index": PipelineStageSpec(
        name="index",
        log_stage="index",
        context_policy="requested_or_all_after_parse",
        suspend_fts_triggers=True,
    ),
    "embed": PipelineStageSpec(name="embed", log_stage="embed", pipeline_managed=False),
}


def stage_specs_for_sequence(stage_sequence: Sequence[str]) -> tuple[PipelineStageSpec, ...]:
    """Resolve normalized leaf stage names to executable stage specs."""
    return tuple(PIPELINE_STAGE_SPECS[stage_name] for stage_name in stage_sequence)


def stage_sequence_suspends_fts(stage_specs: Sequence[PipelineStageSpec]) -> bool:
    """Return whether any stage in a sequence needs FTS triggers suspended."""
    return any(stage.suspend_fts_triggers for stage in stage_specs)


__all__ = [
    "PIPELINE_STAGE_SPECS",
    "PipelineStageSpec",
    "StageContextPolicy",
    "stage_sequence_suspends_fts",
    "stage_specs_for_sequence",
]
