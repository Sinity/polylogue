"""Regression coverage for compact incremental schema evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.models import EQUALITY_EVIDENCE_CAP
from polylogue.schemas.generation.evidence import (
    SchemaEvidence,
    collect_evidence,
    collect_source_evidence,
    merge_evidence,
)
from polylogue.schemas.generation.schema_builder import emit_schema_from_evidence
from polylogue.schemas.observation import ProviderConfig


@dataclass(frozen=True)
class _Observation:
    logical_source_id: str
    revision_sha256: str
    subject: str
    element_kind: str
    records: Iterable[object]
    is_current: bool = True


def _object(value: JSONValue) -> JSONDocument:
    assert isinstance(value, dict)
    return value


def test_evidence_measures_original_long_strings_before_compaction() -> None:
    """A 10k scalar must not inherit the observation sample's 1024-byte cap."""
    value = "a" * 5_000 + "\n" + "b" * 5_000
    evidence = collect_evidence(
        [
            _Observation(
                logical_source_id="session-a",
                revision_sha256="a" * 64,
                subject="claude-code",
                element_kind="session_record_stream",
                records=({"message": value},),
            )
        ]
    )

    field = evidence.field_stats["$.message"]
    assert field.string_length_distribution.maximum == len(value)
    assert field.newline_distribution.maximum == 1


def test_historical_shape_is_retained_without_historical_workload_weight() -> None:
    current = _Observation(
        logical_source_id="session-a",
        revision_sha256="a" * 64,
        subject="claude-code",
        element_kind="session_record_stream",
        records=({"current": 1},),
    )
    historical = _Observation(
        logical_source_id="session-a",
        revision_sha256="b" * 64,
        subject="claude-code",
        element_kind="session_record_stream",
        records=({"retained_only_historically": 2},),
    )

    evidence = collect_evidence([current], historical_observations=[historical])

    assert "$.current" in evidence.field_stats
    assert "$.retained_only_historically" not in evidence.field_stats
    assert evidence.current_record_count == 1
    assert evidence.historical_record_count == 1
    assert "retained_only_historically" in _object(evidence.structure["properties"])


def test_identical_records_in_independent_sessions_remain_two_observations() -> None:
    records = ({"message": "same text"},)
    first = _Observation("session-a", "a" * 64, "claude-code", "session_record_stream", records)
    second = _Observation("session-b", "b" * 64, "claude-code", "session_record_stream", records)

    evidence = collect_evidence([first, second])

    assert evidence.current_source_count == 2
    assert evidence.current_record_count == 2
    assert evidence.field_stats["$.message"].string_length_distribution.count == 2


def test_missing_paths_keep_the_global_current_document_denominator() -> None:
    left = _Observation("session-a", "a" * 64, "claude-code", "session_record_stream", ({"left": 1},))
    right = _Observation("session-b", "b" * 64, "claude-code", "session_record_stream", ({"right": 2},))

    evidence = collect_evidence([left, right])

    assert evidence.field_stats["$.left"].total_samples == 2
    assert evidence.field_stats["$.right"].total_samples == 2
    assert evidence.field_stats["$.left"].document_frequency == 0.5
    assert evidence.field_stats["$.right"].document_frequency == 0.5


def test_current_head_replaces_old_statistics_while_old_shape_stays_historical() -> None:
    current = _Observation(
        "session-a", "b" * 64, "claude-code", "session_record_stream", ({"current_length": "x" * 40},)
    )
    old_revision = _Observation(
        "session-a", "a" * 64, "claude-code", "session_record_stream", ({"old_length": "x" * 4},)
    )

    evidence = collect_evidence([current], historical_observations=[old_revision])

    assert evidence.current_source_count == 1
    assert evidence.field_stats["$.current_length"].string_length_distribution.maximum == 40
    assert "$.old_length" not in evidence.field_stats
    assert "old_length" in _object(evidence.structure["properties"])


def test_ninth_array_record_and_deep_field_survive_reduced_evidence() -> None:
    nested: dict[str, JSONValue] = {"tail": "present"}
    for index in range(9):
        nested = {f"level_{index}": nested}
    records = ({"items": [{"head": index} for index in range(8)] + [nested]},)
    observation = _Observation("session-a", "a" * 64, "claude-code", "session_record_stream", records)

    evidence = collect_evidence([observation])
    schema = evidence.structure
    item_schema = _object(_object(_object(schema["properties"])["items"])["items"])
    assert "level_8" in _object(item_schema["properties"])
    assert any(path.endswith(".tail") for path in evidence.field_stats)


def test_dynamic_map_document_denominator_is_not_summed_from_siblings() -> None:
    observation = _Observation(
        "session-a",
        "a" * 64,
        "claude-code",
        "session_record_stream",
        (
            {"map": {"first question?": {"left": 1}}},
            {"map": {"second question?": {"right": 2}}},
        ),
    )

    evidence = collect_evidence([observation])

    left = evidence.field_stats["$.map.*.left"]
    right = evidence.field_stats["$.map.*.right"]
    assert left.total_samples == right.total_samples == 2
    assert left.document_encountered_count == right.document_encountered_count == 1
    assert left.document_frequency == right.document_frequency == 0.5


def test_serialized_evidence_has_no_literal_and_emits_same_safe_semantics() -> None:
    private_literal = "do-not-persist-this-private-message"
    observation = _Observation(
        "session-a",
        "a" * 64,
        "claude-code",
        "session_record_stream",
        ({"role": "user", "message": private_literal}, {"role": "assistant", "message": private_literal}),
    )
    evidence = collect_evidence([observation])
    restored = SchemaEvidence.from_json(evidence.to_json())
    config = ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code",
        sample_granularity="record",
        record_type_key="type",
    )

    warm_schema, _ = emit_schema_from_evidence("claude-code", config, restored, privacy_config=None)
    cold_schema, _ = emit_schema_from_evidence("claude-code", config, evidence, privacy_config=None)
    serialized = json.dumps(evidence.to_json(), sort_keys=True)
    public = json.dumps(warm_schema, sort_keys=True)

    assert private_literal not in serialized
    assert "session-a" not in serialized
    assert private_literal not in public
    assert warm_schema == cold_schema
    assert restored.field_stats["$.role"].value_session_ids["user"]


def test_reduced_evidence_preserves_mutual_exclusion_annotations() -> None:
    observation = _Observation(
        "session-a",
        "a" * 64,
        "claude-code",
        "session_record_stream",
        ({"role": "user", "left": 1}, {"role": "assistant", "right": 2}),
    )
    config = ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code",
        sample_granularity="record",
        record_type_key="type",
    )
    evidence = SchemaEvidence.from_json(collect_evidence([observation]).to_json())

    schema, _ = emit_schema_from_evidence("claude-code", config, evidence, privacy_config=None)

    assert schema["x-polylogue-mutually-exclusive"] == [{"fields": ["left", "right"], "parent": "$"}]


def test_merge_order_and_one_pass_source_collection_are_deterministic() -> None:
    consumed: list[int] = []

    def records() -> Iterator[JSONValue]:
        for value in (1, 2, 3):
            consumed.append(value)
            yield {"value": value}

    one_pass = _Observation("session-a", "a" * 64, "claude-code", "session_record_stream", records())
    second = _Observation("session-b", "b" * 64, "claude-code", "session_record_stream", ({"value": 4},))
    first_evidence = collect_source_evidence(one_pass)
    second_evidence = collect_source_evidence(second)

    assert consumed == [1, 2, 3]
    assert (
        merge_evidence([first_evidence, second_evidence]).to_json()
        == merge_evidence([second_evidence, first_evidence]).to_json()
    )


def test_equality_evidence_stays_bounded_during_source_reduction() -> None:
    contributions = [
        collect_source_evidence(
            _Observation(
                f"session-{index}",
                f"{index:064x}",
                "claude-code",
                "session_record_stream",
                ({"reference_id": f"private-value-{index}"},),
            )
        )
        for index in range(EQUALITY_EVIDENCE_CAP + 32)
    ]

    merged = merge_evidence(contributions).field_stats["$.reference_id"]

    assert len(merged.equality_hash_counts) == EQUALITY_EVIDENCE_CAP
