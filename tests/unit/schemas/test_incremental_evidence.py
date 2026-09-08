"""Regression coverage for compact incremental schema evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.field_stats.evidence import (
    finalize_field_stats,
    merge_field_stats,
    merge_field_stats_into,
    serialize_field_stats,
)
from polylogue.schemas.field_stats.models import EQUALITY_EVIDENCE_CAP, FieldStats
from polylogue.schemas.generation.evidence import (
    SchemaEvidence,
    collect_evidence,
    collect_sample_evidence,
    collect_source_evidence,
    merge_evidence,
)
from polylogue.schemas.generation.schema_builder import emit_schema_from_evidence
from polylogue.schemas.generation.workload_profiles import _field_profiles
from polylogue.schemas.inference.relational.foreign_keys import detect_foreign_keys
from polylogue.schemas.inference.relational.time import detect_time_deltas
from polylogue.schemas.inference.semantic.message_scoring import score_role
from polylogue.schemas.inference.semantic.runtime import infer_semantic_roles, select_best_roles
from polylogue.schemas.observation import ProviderConfig


@dataclass(frozen=True)
class _Observation:
    logical_source_id: str
    revision_sha256: str
    subject: str
    element_kind: str
    records: Iterable[JSONValue]
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
    records: tuple[JSONValue, ...] = ({"message": "same text"},)
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
    items: list[JSONValue] = [{"head": index} for index in range(8)]
    items.append(nested)
    records: tuple[JSONValue, ...] = ({"items": items},)
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


def test_live_evidence_matches_round_trip_without_raw_enum_or_relation_changes() -> None:
    private_literal = "brief"
    records = [{"label": private_literal, "id": f"node-{index}", "parent_id": f"node-{index}"} for index in range(6)]
    evidence = collect_sample_evidence(
        records,
        session_ids=[f"session-{index}" for index in range(6)],
    )
    restored = SchemaEvidence.from_json(evidence.to_json())
    config = ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code",
        sample_granularity="record",
        record_type_key="type",
    )

    live_schema, _ = emit_schema_from_evidence("claude-code", config, evidence, privacy_config=None)
    restored_schema, _ = emit_schema_from_evidence("claude-code", config, restored, privacy_config=None)

    assert private_literal not in json.dumps(live_schema, sort_keys=True)
    assert live_schema == restored_schema
    assert detect_foreign_keys(evidence.field_stats) == detect_foreign_keys(restored.field_stats)


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


def test_streaming_field_stats_fold_matches_batch_merge_after_finalization() -> None:
    first = collect_source_evidence(
        _Observation(
            "session-a", "a" * 64, "claude-code", "session_record_stream", ({"mapping": {"a": {}}, "ref": "a"},)
        ),
        dynamic_paths=["$.mapping"],
    )
    second = collect_source_evidence(
        _Observation(
            "session-b", "b" * 64, "claude-code", "session_record_stream", ({"mapping": {"b": {}}, "ref": "b"},)
        ),
        dynamic_paths=["$.mapping"],
    )
    summaries = [first.field_stats, second.field_stats]

    expected = merge_field_stats(summaries, total_samples=2)
    streamed: dict[str, FieldStats] = {}
    for summary in summaries:
        merge_field_stats_into(streamed, summary)
    finalize_field_stats(streamed, total_samples=2)

    assert {path: serialize_field_stats(stats) for path, stats in streamed.items()} == {
        path: serialize_field_stats(stats) for path, stats in expected.items()
    }


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


def test_reduced_evidence_retains_detected_mapping_reference() -> None:
    node_ids = [f"node-{index:08x}" for index in range(60)]
    records = [{"mapping": {node_id: {} for node_id in node_ids}, "current_node": node_id} for node_id in node_ids]

    raw_stats = _collect_field_stats(records)
    reduced_stats = SchemaEvidence.from_json(collect_sample_evidence(records).to_json()).field_stats

    assert raw_stats["$.current_node"].ref_target == "$.mapping"
    assert reduced_stats["$.current_node"].ref_target == "$.mapping"
    assert [(relation.source_path, relation.target_path) for relation in detect_foreign_keys(raw_stats)] == [
        ("$.current_node", "$.mapping")
    ]
    assert [(relation.source_path, relation.target_path) for relation in detect_foreign_keys(reduced_stats)] == [
        ("$.current_node", "$.mapping")
    ]


def test_merged_mapping_reference_uses_global_overlap_not_a_source_local_conclusion() -> None:
    node_ids = [f"node-{index:08x}" for index in range(60)]
    matched: list[JSONDocument] = [
        {"mapping": {node_id: {} for node_id in node_ids}, "current_node": node_id} for node_id in node_ids
    ]
    unmatched: list[JSONDocument] = [{"current_node": f"unmatched-{index}"} for index in range(140)]
    dynamic_paths = ["$.mapping"]

    raw_relations = detect_foreign_keys(_collect_field_stats([*matched, *unmatched], dynamic_paths=dynamic_paths))
    merged_stats = merge_evidence(
        [
            collect_source_evidence(
                _Observation("source-a", "a" * 64, "claude-code", "session_record_stream", matched),
                dynamic_paths=dynamic_paths,
            ),
            collect_source_evidence(
                _Observation("source-b", "b" * 64, "claude-code", "session_record_stream", unmatched),
                dynamic_paths=dynamic_paths,
            ),
        ]
    ).field_stats

    assert raw_relations == []
    assert merged_stats["$.current_node"].ref_target is None
    assert [
        relation for relation in detect_foreign_keys(merged_stats) if relation.source_path == "$.current_node"
    ] == []


def test_merged_mapping_reference_detects_global_overlap_when_each_source_is_below_local_cardinality() -> None:
    node_ids = [f"node-{index:08x}" for index in range(60)]
    records: list[JSONDocument] = [
        {"mapping": {node_id: {} for node_id in node_ids}, "current_node": node_id} for node_id in node_ids
    ]
    dynamic_paths = ["$.mapping"]

    merged_stats = merge_evidence(
        [
            collect_source_evidence(
                _Observation("source-a", "a" * 64, "claude-code", "session_record_stream", records[:30]),
                dynamic_paths=dynamic_paths,
            ),
            collect_source_evidence(
                _Observation("source-b", "b" * 64, "claude-code", "session_record_stream", records[30:]),
                dynamic_paths=dynamic_paths,
            ),
        ]
    ).field_stats

    assert [(relation.source_path, relation.target_path) for relation in detect_foreign_keys(merged_stats)] == [
        ("$.current_node", "$.mapping")
    ]
    assert merged_stats["$.current_node"].ref_target == "$.mapping"


def test_merged_mapping_reference_compares_within_a_truncated_target_hash_domain() -> None:
    node_ids = [f"node-{index:08x}" for index in range(1_000)]
    records: list[JSONDocument] = [
        {"mapping": {node_id: {} for node_id in node_ids}, "current_node": node_id} for node_id in node_ids[:60]
    ]
    evidence = collect_source_evidence(
        _Observation("source-a", "a" * 64, "claude-code", "session_record_stream", records),
        dynamic_paths=["$.mapping"],
    )
    merged_stats = merge_evidence([evidence]).field_stats

    assert merged_stats["$.mapping"].truncated_evidence["object_key_hashes"]
    assert merged_stats["$.current_node"].ref_target == "$.mapping"
    assert [(relation.source_path, relation.target_path) for relation in detect_foreign_keys(merged_stats)] == [
        ("$.current_node", "$.mapping")
    ]


def test_reduced_evidence_keeps_hashes_when_safe_values_are_present_for_foreign_keys() -> None:
    records = [{"id": value, "parent_id": value} for value in ["user", *(f"item-{index}" for index in range(6))]]

    raw_relations = detect_foreign_keys(_collect_field_stats(records))
    reduced_relations = detect_foreign_keys(collect_sample_evidence(records).field_stats)

    assert [(relation.source_path, relation.target_path) for relation in raw_relations] == [("$.parent_id", "$.id")]
    assert [(relation.source_path, relation.target_path) for relation in reduced_relations] == [("$.parent_id", "$.id")]


def test_reduced_foreign_key_hashes_do_not_double_count_safe_values() -> None:
    references = ["user", *(f"item-{index}" for index in range(8))]
    records = [{"parent_id": value} for value in references] + [{"id": value} for value in references[:5]]

    raw_relations = detect_foreign_keys(_collect_field_stats(records))
    reduced_relations = detect_foreign_keys(collect_sample_evidence(records).field_stats)

    assert raw_relations == []
    assert reduced_relations == []


def test_equality_hash_cap_does_not_drop_slash_shape_evidence() -> None:
    values = sorted(
        (f"/private/{index}" for index in range(400)),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    stats = FieldStats(path="$.title")

    for value in values:
        stats.observe_equality_value(value)

    assert len(stats.equality_hash_counts) == EQUALITY_EVIDENCE_CAP
    assert stats.slash_value_count == len(values)


def test_shape_evidence_reports_the_same_unretained_observation_lower_bound_for_streams_and_samples() -> None:
    records: list[JSONDocument] = [{f"field_{index}": 1} for index in range(512)]
    for _ in range(10):
        records.append({"extra_field": 1})
    observation = _Observation("session-a", "a" * 64, "claude-code", "session_record_stream", records)

    streamed = collect_source_evidence(observation)
    sampled = collect_sample_evidence(records)

    assert len(streamed.shape_hashes) == len(sampled.shape_hashes) == 512
    assert streamed.unretained_shape_observation_lower_bound == 10
    assert sampled.unretained_shape_observation_lower_bound == 10
    assert "distinct_shapes_overflow" not in _object(streamed.to_json()["denominators"])


def test_reduced_evidence_preserves_caseful_known_role_values_for_scoring() -> None:
    records = [{"role": "USER"}, {"role": "ASSISTANT"}]

    raw_candidate = score_role("$.role", _collect_field_stats(records)["$.role"])
    reduced_candidate = score_role("$.role", collect_sample_evidence(records).field_stats["$.role"])

    assert raw_candidate is not None
    assert reduced_candidate == raw_candidate
    assert reduced_candidate.role == "message_role"


def test_reduced_evidence_uses_complete_equality_entropy_for_session_title_selection() -> None:
    records = [
        {"title": "user" if index == 0 else f"Neutral title {index}", "label": f"Neutral label {index}"}
        for index in range(21)
    ]

    raw_best = select_best_roles(infer_semantic_roles(_collect_field_stats(records), artifact_kind="session_document"))
    reduced_best = select_best_roles(
        infer_semantic_roles(collect_sample_evidence(records).field_stats, artifact_kind="session_document")
    )

    assert raw_best["session_title"].path == "$.title"
    assert reduced_best["session_title"].path == "$.title"


def test_public_numeric_annotations_exclude_magnitudes_and_preserve_private_evidence() -> None:
    """Coordinates, IDs, and unknown numeric content must not leak through distribution annotations."""
    config = ProviderConfig(name=Provider.CLAUDE_CODE, description="Synthetic", sample_granularity="record")
    summaries = []
    for number in (12.3456789, 67.8912345):
        evidence = collect_sample_evidence(
            [{"latitude": number, "user_id": number, "arbitrary": number, "text": "synthetic", "items": [1, 2]}]
        )
        private = evidence.to_json()
        assert evidence.field_stats["$.latitude"].numeric_distribution.minimum == number
        schema, _ = emit_schema_from_evidence("claude-code", config, evidence, privacy_config=None)
        assert evidence.to_json() == private
        properties = _object(schema["properties"])
        summary = _object(_object(properties["latitude"])["x-polylogue-observed-distribution"])
        assert summary["numeric"] == {"count": 1, "non_finite_count": 0}
        summaries.append(summary["numeric"])
        for name in ("latitude", "user_id", "arbitrary"):
            node = _object(properties[name])
            assert "x-polylogue-range" not in node
            assert _object(node["x-polylogue-observed-distribution"])["numeric"] == summary["numeric"]
        text_distribution = _object(_object(properties["text"])["x-polylogue-observed-distribution"])
        assert "string_length" in text_distribution
        items_distribution = _object(_object(properties["items"])["x-polylogue-observed-distribution"])
        assert "array_length" in items_distribution
        profile = _field_profiles(schema)
        assert profile["$.latitude"]["numeric"] == summary["numeric"]
        assert str(number) not in json.dumps({"schema": schema, "profile": profile})
    assert summaries[0] == summaries[1]


def test_public_timestamp_annotations_exclude_ranges_and_empirical_deltas() -> None:
    """Timestamp role diagnostics and relations cannot reintroduce private numeric magnitudes."""
    evidence = collect_sample_evidence(
        [{"created_at": 1_700_000_000 + index, "updated_at": 1_700_000_500 + index} for index in range(4)]
    )
    assert detect_time_deltas(evidence.field_stats)
    # Legacy reduced fields can carry numeric bounds without format counters.
    for field in evidence.field_stats.values():
        field.detected_formats.clear()
    candidates = infer_semantic_roles(evidence.field_stats)
    assert any("range" in candidate.evidence for candidate in candidates)
    config = ProviderConfig(name=Provider.CLAUDE_CODE, description="Synthetic", sample_granularity="record")
    schema, _ = emit_schema_from_evidence("claude-code", config, evidence, privacy_config=None)
    assert "x-polylogue-time-deltas" not in schema
    for node in _object(schema["properties"]).values():
        child = _object(node)
        assert "x-polylogue-range" not in child
        assert "range" not in _object(child.get("x-polylogue-evidence", {}))


def test_schema_frequency_preserves_rare_observations() -> None:
    """Rounding to three decimal places misrepresents observed fields as absent."""
    from polylogue.schemas.generation.field_annotations import annotate_schema

    stats = FieldStats(path="$.rare", total_samples=3_000_000, document_encountered_count=5)
    schema = annotate_schema({"type": "string"}, {"$.rare": stats}, "$.rare")
    assert schema["x-polylogue-frequency"] == 5 / 3_000_000
