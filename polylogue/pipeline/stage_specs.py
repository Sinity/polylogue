"""Executable pipeline stage specifications and contract validation.

Each :class:`PipelineStageSpec` declares which named state slots it
consumes (``inputs``) and produces (``outputs``). At each stage
boundary, :func:`validate_stage_contract` verifies that every required
input was produced by an upstream stage that already executed in the
current run. On violation it raises :class:`StageContractError`.

The structural validator catches misconfigured stage sequences such as
``[acquire, materialize]`` (parse skipped) or ``[acquire, render]``
(parse skipped). Single-stage runs skip validation, since their inputs
come from durable archive state rather than upstream pipeline outputs.

See `#447 <https://github.com/Sinity/polylogue/issues/447>`_.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

StageContextPolicy = Literal["none", "requested_or_leaf", "requested_or_all_after_parse"]


@dataclass(frozen=True, slots=True)
class StageInput:
    """One named state slot a stage requires before it can execute."""

    name: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class PipelineStageSpec:
    """Static execution facts for one pipeline leaf stage."""

    name: str
    log_stage: str
    context_policy: StageContextPolicy = "none"
    suspend_fts_triggers: bool = False
    pipeline_managed: bool = True
    inputs: tuple[StageInput, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)

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


class StageContractError(RuntimeError):
    """Raised when a stage runs with required inputs not produced upstream."""

    def __init__(self, *, stage: str, missing: Sequence[str], reason: str) -> None:
        formatted = ", ".join(sorted(missing))
        super().__init__(f"Stage {stage!r} missing required input(s) {{{formatted}}}: {reason}")
        self.stage = stage
        self.missing: tuple[str, ...] = tuple(missing)
        self.reason = reason


PIPELINE_STAGE_SPECS: dict[str, PipelineStageSpec] = {
    "acquire": PipelineStageSpec(
        name="acquire",
        log_stage="acquire",
        outputs=("raw_artifacts",),
    ),
    "schema": PipelineStageSpec(
        name="schema",
        log_stage="schema",
        outputs=("schemas",),
    ),
    "parse": PipelineStageSpec(
        name="parse",
        log_stage="ingest",
        context_policy="requested_or_leaf",
        suspend_fts_triggers=True,
        inputs=(StageInput(name="raw_artifacts", required=False),),
        outputs=("processed_ids",),
    ),
    "materialize": PipelineStageSpec(
        name="materialize",
        log_stage="materialize",
        context_policy="requested_or_all_after_parse",
        inputs=(StageInput(name="processed_ids"),),
        outputs=("materialized",),
    ),
    "render": PipelineStageSpec(
        name="render",
        log_stage="render",
        context_policy="requested_or_all_after_parse",
        suspend_fts_triggers=True,
        inputs=(StageInput(name="processed_ids"),),
        outputs=("rendered",),
    ),
    "site": PipelineStageSpec(
        name="site",
        log_stage="site",
        inputs=(StageInput(name="rendered", required=False),),
    ),
    "index": PipelineStageSpec(
        name="index",
        log_stage="index",
        context_policy="requested_or_all_after_parse",
        suspend_fts_triggers=True,
        inputs=(StageInput(name="processed_ids"),),
    ),
    "embed": PipelineStageSpec(name="embed", log_stage="embed", pipeline_managed=False),
}


def stage_specs_for_sequence(stage_sequence: Sequence[str]) -> tuple[PipelineStageSpec, ...]:
    """Resolve normalized leaf stage names to executable stage specs."""
    return tuple(PIPELINE_STAGE_SPECS[stage_name] for stage_name in stage_sequence)


def stage_sequence_suspends_fts(stage_specs: Sequence[PipelineStageSpec]) -> bool:
    """Return whether any stage in a sequence needs FTS triggers suspended."""
    return any(stage.suspend_fts_triggers for stage in stage_specs)


def validate_stage_contract(
    spec: PipelineStageSpec,
    *,
    executed_specs: Sequence[PipelineStageSpec],
) -> None:
    """Verify ``spec``'s required inputs were produced by an executed upstream stage.

    Single-stage runs (``executed_specs`` empty) skip validation: their inputs
    come from durable archive state, not upstream pipeline outputs. Multi-stage
    runs that omit a required producer raise :class:`StageContractError`.
    """
    if not executed_specs:
        return
    produced: set[str] = {output for upstream in executed_specs for output in upstream.outputs}
    missing = [
        stage_input.name for stage_input in spec.inputs if stage_input.required and stage_input.name not in produced
    ]
    if missing:
        executed_names = [upstream.name for upstream in executed_specs]
        raise StageContractError(
            stage=spec.name,
            missing=missing,
            reason=f"upstream stages {executed_names} did not produce them",
        )


__all__ = [
    "PIPELINE_STAGE_SPECS",
    "PipelineStageSpec",
    "StageContextPolicy",
    "StageContractError",
    "StageInput",
    "stage_sequence_suspends_fts",
    "stage_specs_for_sequence",
    "validate_stage_contract",
]
