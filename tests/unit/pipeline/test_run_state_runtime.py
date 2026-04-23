from __future__ import annotations

from polylogue.pipeline.run_state import RunExecutionState
from polylogue.pipeline.services.parsing import ParseResult
from polylogue.pipeline.stage_models import AcquireResult, ValidateResult


def test_run_execution_state_records_stage_results_and_finalizes_drift() -> None:
    state = RunExecutionState()
    state.record_acquire(AcquireResult(acquired=2, skipped=1, errors=3))
    state.record_validation(None)
    state.record_validation(ValidateResult(validated=4, invalid=5, drift=6, skipped_no_schema=7, errors=8))

    parse_result = ParseResult()
    parse_result.counts["messages"] = 4
    parse_result.counts["attachments"] = 2
    parse_result.changed_counts["conversations"] = 1
    parse_result.changed_counts["messages"] = 4
    parse_result.changed_counts["attachments"] = 2
    parse_result.processed_ids = {"conv-1", "conv-2"}
    parse_result.parse_failures = 9
    state.record_parse(parse_result)
    state.record_schema_generation(generated=10, failed=11)
    state.record_materialize(materialized=12)
    state.record_render(rendered=13, failures=[{"conversation_id": "conv-1", "error": "boom"}])

    drift = state.finalize()

    assert state.counts.acquired == 2
    assert state.counts.skipped == 1
    assert state.counts.acquire_errors == 3
    assert state.counts.validated == 4
    assert state.counts.validation_invalid == 5
    assert state.counts.validation_drift == 6
    assert state.counts.validation_skipped_no_schema == 7
    assert state.counts.validation_errors == 8
    assert state.counts.conversations == 2
    assert state.counts.messages == 4
    assert state.counts.attachments == 2
    assert state.counts.parse_failures == 9
    assert state.counts.schemas_generated == 10
    assert state.counts.schemas_failed == 11
    assert state.counts.materialized == 12
    assert state.counts.rendered == 13
    assert state.counts.render_failures == 1
    assert state.render_failures == [{"conversation_id": "conv-1", "error": "boom"}]
    assert state.processed_ids == {"conv-1", "conv-2"}
    assert state.changed_counts.conversations == 1
    assert state.changed_counts.messages == 4
    assert state.changed_counts.attachments == 2
    assert state.counts.new_conversations == 1
    assert state.counts.changed_conversations == 1
    assert drift.new.conversations == 1
    assert drift.changed.conversations == 1


def test_run_execution_state_finalize_without_failures_or_validation_keeps_defaults() -> None:
    state = RunExecutionState()
    drift = state.finalize()

    assert drift.new.conversations == 0
    assert drift.changed.conversations == 0
    assert state.counts.render_failures is None
    assert state.render_failures == []


def test_run_execution_state_record_parse_and_render_keep_default_false_paths() -> None:
    state = RunExecutionState()

    parse_result = ParseResult()
    parse_result.counts["messages"] = 2
    parse_result.processed_ids = {"conv-1"}
    state.record_parse(parse_result)
    state.record_render(rendered=2, failures=[])

    assert state.counts.messages == 2
    assert state.counts.conversations == 1
    assert state.counts.parse_failures is None
    assert state.counts.rendered == 2
    assert state.counts.render_failures is None
