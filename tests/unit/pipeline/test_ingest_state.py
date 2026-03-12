"""Tests for typed ingest-state transition guards."""

from __future__ import annotations

import pytest

from polylogue.pipeline.services.parsing import IngestPhase, IngestState


def test_ingest_state_happy_path_transitions() -> None:
    state = IngestState(source_names=("inbox",), parse_requested=True)
    assert state.phase == IngestPhase.INIT

    state.record_acquired(["raw-1", "raw-2"])
    assert state.phase == IngestPhase.ACQUIRED
    assert state.acquired_raw_ids == ["raw-1", "raw-2"]

    state.record_validation_candidates(["raw-1", "raw-2", "raw-3"])
    state.record_validation_result(["raw-1", "raw-3"])
    assert state.phase == IngestPhase.VALIDATED
    assert state.parseable_raw_ids == ["raw-1", "raw-3"]

    state.record_parse_candidates(["raw-3", "raw-1"])
    state.record_parse_completed()
    assert state.phase == IngestPhase.PARSED
    assert state.parse_raw_ids == ["raw-3", "raw-1"]


def test_ingest_state_rejects_out_of_order_transition() -> None:
    state = IngestState(source_names=("inbox",), parse_requested=True)
    with pytest.raises(RuntimeError, match="expected phase acquired"):
        state.record_validation_candidates(["raw-1"])


def test_ingest_state_rejects_unexpected_validation_ids() -> None:
    state = IngestState(source_names=("inbox",), parse_requested=True)
    state.record_acquired(["raw-1"])
    state.record_validation_candidates(["raw-1"])
    with pytest.raises(ValueError, match="outside validation candidates"):
        state.record_validation_result(["raw-2"])


def test_ingest_state_rejects_unexpected_parse_ids() -> None:
    state = IngestState(source_names=("inbox",), parse_requested=True)
    state.record_acquired(["raw-1"])
    state.record_validation_candidates(["raw-1"])
    state.record_validation_result(["raw-1"])
    with pytest.raises(ValueError, match="outside validation candidates"):
        state.record_parse_candidates(["raw-2"])


def test_ingest_state_allows_persisted_prevalidated_parse_ids() -> None:
    state = IngestState(source_names=("inbox",), parse_requested=True)
    state.record_acquired([])
    state.record_validation_candidates([])
    state.record_validation_result([])
    state.record_parse_candidates(
        ["raw-prevalidated"],
        persisted_validated_raw_ids=["raw-prevalidated"],
    )
    state.record_parse_completed()
    assert state.phase == IngestPhase.PARSED
