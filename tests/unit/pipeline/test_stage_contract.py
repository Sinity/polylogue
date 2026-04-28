"""Stage contract validation tests.

See `#447 <https://github.com/Sinity/polylogue/issues/447>`_.
"""

from __future__ import annotations

import pytest

from polylogue.pipeline.stage_specs import (
    PIPELINE_STAGE_SPECS,
    PipelineStageSpec,
    StageContractError,
    StageInput,
    stage_specs_for_sequence,
    validate_stage_contract,
)


def test_single_stage_run_skips_validation() -> None:
    """A stage with no upstream executions is permitted; data may live in DB."""
    materialize = PIPELINE_STAGE_SPECS["materialize"]
    validate_stage_contract(materialize, executed_specs=())


def test_full_sequence_satisfies_all_inputs() -> None:
    """The default ``all`` sequence satisfies every stage's required inputs."""
    sequence = stage_specs_for_sequence(("acquire", "parse", "materialize", "render", "site", "index"))
    executed: list[PipelineStageSpec] = []
    for spec in sequence:
        validate_stage_contract(spec, executed_specs=executed)
        executed.append(spec)


def test_reprocess_sequence_satisfies_inputs() -> None:
    """``reprocess`` (parse, materialize, render, site, index) satisfies the contract."""
    sequence = stage_specs_for_sequence(("parse", "materialize", "render", "site", "index"))
    executed: list[PipelineStageSpec] = []
    for spec in sequence:
        validate_stage_contract(spec, executed_specs=executed)
        executed.append(spec)


def test_publish_sequence_satisfies_inputs() -> None:
    """``publish`` (render, site) satisfies the contract — site's input is optional."""
    sequence = stage_specs_for_sequence(("render", "site"))
    executed: list[PipelineStageSpec] = []
    for spec in sequence:
        validate_stage_contract(spec, executed_specs=executed)
        executed.append(spec)


def test_skipped_parse_violates_materialize_contract() -> None:
    """``[acquire, materialize]`` skips parse → materialize lacks ``processed_ids``."""
    acquire = PIPELINE_STAGE_SPECS["acquire"]
    materialize = PIPELINE_STAGE_SPECS["materialize"]
    with pytest.raises(StageContractError) as excinfo:
        validate_stage_contract(materialize, executed_specs=(acquire,))
    assert excinfo.value.stage == "materialize"
    assert "processed_ids" in excinfo.value.missing


def test_skipped_parse_violates_render_contract() -> None:
    """``[acquire, render]`` likewise lacks ``processed_ids`` for render."""
    acquire = PIPELINE_STAGE_SPECS["acquire"]
    render = PIPELINE_STAGE_SPECS["render"]
    with pytest.raises(StageContractError) as excinfo:
        validate_stage_contract(render, executed_specs=(acquire,))
    assert excinfo.value.stage == "render"
    assert "processed_ids" in excinfo.value.missing


def test_skipped_parse_violates_index_contract() -> None:
    """``[acquire, index]`` lacks ``processed_ids`` for index."""
    acquire = PIPELINE_STAGE_SPECS["acquire"]
    index = PIPELINE_STAGE_SPECS["index"]
    with pytest.raises(StageContractError) as excinfo:
        validate_stage_contract(index, executed_specs=(acquire,))
    assert "processed_ids" in excinfo.value.missing


def test_optional_input_does_not_violate() -> None:
    """``site`` declares ``rendered`` as optional; running without render is OK."""
    site = PIPELINE_STAGE_SPECS["site"]
    parse = PIPELINE_STAGE_SPECS["parse"]
    validate_stage_contract(site, executed_specs=(parse,))


def test_violation_message_contains_stage_and_missing() -> None:
    """Error message names both the stage and the missing inputs."""
    materialize = PIPELINE_STAGE_SPECS["materialize"]
    acquire = PIPELINE_STAGE_SPECS["acquire"]
    with pytest.raises(StageContractError) as excinfo:
        validate_stage_contract(materialize, executed_specs=(acquire,))
    message = str(excinfo.value)
    assert "materialize" in message
    assert "processed_ids" in message


def test_custom_spec_with_required_input_fires() -> None:
    """A bespoke spec with required input names not produced upstream fails."""
    custom = PipelineStageSpec(
        name="custom",
        log_stage="custom",
        inputs=(StageInput(name="not_produced"),),
    )
    upstream = PIPELINE_STAGE_SPECS["acquire"]
    with pytest.raises(StageContractError) as excinfo:
        validate_stage_contract(custom, executed_specs=(upstream,))
    assert excinfo.value.missing == ("not_produced",)


def test_every_stage_input_is_produced_by_some_pipeline_stage() -> None:
    """Each declared input matches an output declared by a pipeline-managed stage."""
    produced: set[str] = set()
    for spec in PIPELINE_STAGE_SPECS.values():
        produced.update(spec.outputs)
    for spec in PIPELINE_STAGE_SPECS.values():
        for stage_input in spec.inputs:
            assert stage_input.name in produced, (
                f"Stage {spec.name!r} declares input {stage_input.name!r} with no producer in PIPELINE_STAGE_SPECS"
            )
