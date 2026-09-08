"""Streaming evidence must retain statistics without retaining source rows."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.generation import evidence as evidence_module
from polylogue.schemas.generation.evidence import SchemaEvidenceAccumulator, collect_source_evidence
from polylogue.schemas.source_inference import SourceObservation


def observation(records: list[JSONValue], source: str = "session-a") -> SourceObservation:
    return SourceObservation(source, "a" * 64, "claude-code", "session_record_stream", iter(records))


def test_structure_pass_skips_statistics_and_matches_complete_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The structure pass must not walk values through the statistics collector."""
    records: list[JSONValue] = [{"role": "user", "text": "x" * 1000}, {"role": "assistant", "optional": 4}]
    complete = collect_source_evidence(observation(records))

    def refuse_statistics(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("structure-only collection reached statistics")

    monkeypatch.setattr(evidence_module, "_collect_field_stats", refuse_statistics)
    structure = collect_source_evidence(observation(records), include_statistics=False)
    assert structure.fields == {}
    assert structure.current_structure == complete.current_structure
    assert structure.shape_hashes == complete.shape_hashes
    assert structure.current_record_count == complete.current_record_count == 2
    assert structure.current_source_count == complete.current_source_count == 1


def test_repeated_shapes_are_merged_once_while_every_value_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated shape merging must not scale with the number of identical shapes."""
    real_merge = evidence_module._merge_structure
    merges = 0

    def count_merge(left: JSONDocument, right: JSONDocument) -> JSONDocument:
        nonlocal merges
        merges += 1
        return real_merge(left, right)

    monkeypatch.setattr(evidence_module, "_merge_structure", count_merge)
    records: list[JSONValue] = [{"text": "x" * length} for length in range(1, 101)]
    result = collect_source_evidence(observation(records))
    assert merges == 1
    stats = result.field_stats["$.text"]
    assert result.current_record_count == stats.value_count == 100
    assert stats.string_length_distribution.minimum == 1
    assert stats.string_length_distribution.maximum == 100
    assert stats.string_length_distribution.mean == pytest.approx(50.5)


def test_stream_fold_drops_source_objects_and_keeps_current_and_historical_denominators() -> None:
    """Keeping completed evidence rows makes coordinator memory grow with source count."""
    accumulator = SchemaEvidenceAccumulator()
    references = []
    for index in range(40):
        row = collect_source_evidence(observation([{"value": index}], source=f"source-{index}"))
        references.append(weakref.ref(row))
        accumulator.add(row)
        del row
    historical = collect_source_evidence(observation([{"retired_field": True}]), is_current=False)
    accumulator.add(historical)
    del historical
    gc.collect()
    assert all(reference() is None for reference in references)

    result = accumulator.finish()
    assert result.current_source_count == result.current_record_count == 40
    assert result.historical_source_count == result.historical_record_count == 1
    historical_properties = result.historical_structure["properties"]
    assert isinstance(historical_properties, dict)
    assert "retired_field" in historical_properties
    assert "$.retired_field" not in result.field_stats
    stats = result.field_stats["$.value"]
    assert stats.total_samples == stats.value_count == 40
    assert stats.numeric_distribution.mean == pytest.approx(19.5)
    assert stats.numeric_distribution.minimum == 0
    assert stats.numeric_distribution.maximum == 39


def test_stream_fold_rejects_mixed_normalization() -> None:
    """Combining raw dictionary paths with wildcard paths changes denominators."""
    row = collect_source_evidence(observation([{"field": 1}]))
    accumulator = SchemaEvidenceAccumulator()
    accumulator.add(row)
    with pytest.raises(ValueError, match="different dynamic-key normalization"):
        accumulator.add(replace(row, normalization_paths=("$.mapping",)))
