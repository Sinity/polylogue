"""Behavioral contracts for bounded schema workload profiles."""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.scenarios import (
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
    lineage_replay_canary_spec,
    tool_pairing_canary_spec,
)
from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.field_stats.distributions import CategoricalSketch, DistributionSketch
from polylogue.schemas.generation.archive_workload_profile import (
    build_archive_workload_profile,
    write_archive_workload_profile,
)
from polylogue.schemas.generation.field_annotations import annotate_schema
from polylogue.schemas.generation.models import _PackageAccumulator, _UnitMembership
from polylogue.schemas.generation.workload_profiles import (
    _TOOL_IDS_PER_SCOPE_CAP,
    build_package_workload_profile,
    workload_profile_identity,
)
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.workload_tiers import WorkloadScaleTier, WorkloadSelectivityTier
from polylogue.sources.dispatch import parse_payload
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)


def test_distribution_sketch_is_bounded_mergeable_and_preserves_tails() -> None:
    left = DistributionSketch()
    right = DistributionSketch()
    combined = DistributionSketch()
    values = [0, 1, 2, 3, 10, 100, 1_000, 1_000_000]
    for value in values[:4]:
        left.observe(value)
    for value in values[4:]:
        right.observe(value)
    for value in values:
        combined.observe(value)

    left.merge(right)

    assert left.count == combined.count == len(values)
    assert left.minimum == combined.minimum == 0
    assert left.maximum == combined.maximum == 1_000_000
    assert left.mean == pytest.approx(combined.mean)
    assert left.buckets == combined.buckets
    assert len(left.buckets) <= 1_025
    upper_quantile = left.quantile(0.99)
    assert upper_quantile is not None
    assert upper_quantile >= 100_000


def test_distribution_sketch_observes_repeated_values_without_expansion() -> None:
    sketch = DistributionSketch()
    sketch.observe_repeated(0, 1_000_000)
    sketch.observe_repeated(10, 2)

    assert sketch.count == 1_000_002
    assert sketch.minimum == 0
    assert sketch.maximum == 10
    assert sketch.mean == pytest.approx(20 / 1_000_002)
    assert sketch.buckets[0] == 1_000_000


def test_workload_profile_identity_normalizes_unicode_content() -> None:
    composed = {
        "label": "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        "nested": {
            "r\N{LATIN SMALL LETTER E WITH ACUTE}sum\N{LATIN SMALL LETTER E WITH ACUTE}": [
                "na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve"
            ]
        },
    }
    decomposed = {
        "label": "cafe\N{COMBINING ACUTE ACCENT}",
        "nested": {"re\N{COMBINING ACUTE ACCENT}sume\N{COMBINING ACUTE ACCENT}": ["nai\N{COMBINING DIAERESIS}ve"]},
    }

    assert workload_profile_identity(composed) == workload_profile_identity(decomposed)


def test_categorical_sketch_retains_every_observation_without_values() -> None:
    left = CategoricalSketch()
    right = CategoricalSketch()
    for index in range(5_000):
        target = left if index % 2 else right
        target.observe(f"private-value-{index % 997}")
    left.merge(right)

    payload = left.to_payload()
    assert payload["count"] == 5_000
    assert 850 <= cast(int, payload["estimated_distinct"]) <= 1_150
    assert len(left.buckets) <= 256
    assert "private-value" not in json.dumps(payload)


def test_categorical_sketch_keeps_unobserved_hll_registers_sparse() -> None:
    sketch = CategoricalSketch()
    assert sketch.registers == {}

    for value in ("alpha", "beta", "alpha"):
        sketch.observe(value)

    assert 0 < len(sketch.registers) <= 2
    assert sketch.estimated_distinct == 2


def test_field_collection_retains_full_counts_while_bounding_legacy_evidence() -> None:
    samples = [
        {
            "items": list(range(index % 37)),
            "text": "x" * (index % 503),
            "nullable": None if index % 3 == 0 else index,
        }
        for index in range(5_000)
    ]

    stats = _collect_field_stats(samples)

    assert stats["$.items"].array_length_distribution.count == 5_000
    assert stats["$.items"].array_length_distribution.maximum == 36
    assert len(stats["$.items"].array_lengths) == 2_000
    assert stats["$.items"].truncated_evidence["array_length_samples"] == 3_000
    assert stats["$.text"].string_length_distribution.count == 5_000
    assert stats["$.text"].categorical_distribution.count == 5_000
    assert stats["$.items"].object_key_distribution.count == 0
    assert stats["$.nullable"].document_encountered_count == 5_000
    assert stats["$.nullable"].document_non_null_count == 3_333
    assert stats["$.nullable"].null_count == 1_667
    assert len(stats["$.nullable"].documents_present) == 2_000
    assert stats["$.nullable"].document_frequency == 1.0

    numeric_stats = _collect_field_stats([{"value": 1.0}, {"value": float("inf")}, {"value": float("nan")}])
    assert numeric_stats["$.value"].numeric_distribution.count == 1
    assert numeric_stats["$.value"].numeric_distribution.non_finite_count == 2


def test_array_annotations_use_full_sketch_extrema_after_sample_cap() -> None:
    samples: list[JSONDocument] = [{"items": []} for _ in range(2_000)]
    samples.append({"items": list(range(2_500))})
    schema: JSONDocument = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
    }

    annotated = annotate_schema(schema, _collect_field_stats(samples))
    items = cast(dict[str, JSONValue], cast(dict[str, JSONValue], annotated["properties"])["items"])

    assert items["x-polylogue-array-lengths"] == [0, 2_500]


def test_annotations_drive_synthetic_numeric_and_array_distributions() -> None:
    samples = [
        {"count": 1, "items": [1]},
        {"count": 1, "items": [1]},
        {"count": 1_000, "items": list(range(20))},
    ]
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    }
    annotated = annotate_schema(schema, _collect_field_stats(samples))
    corpus = SyntheticCorpus(
        annotated,
        WireFormat(encoding="json"),
        "test",
    )

    generated = [corpus._generate_from_schema(annotated, random.Random(seed)) for seed in range(100)]
    records = [record for record in generated if isinstance(record, dict)]
    counts = [record["count"] for record in records]
    item_values = [record["items"] for record in records]
    assert all(isinstance(items, list) for items in item_values)
    lengths = [len(cast(list[JSONValue], items)) for items in item_values]

    assert set(counts) <= {1, 1_000}
    assert 1 in lengths
    assert max(lengths) >= 19


def test_synthetic_generation_bounds_inferred_tail_payloads() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "x-polylogue-array-lengths": [2_500, 2_500],
            }
        },
        "x-polylogue-string-lengths": [
            {"path": "$.items[*]", "min": 1_000_000, "max": 1_000_000, "avg": 1_000_000, "stddev": 0}
        ],
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")

    generated = cast(dict[str, JSONValue], corpus._generate_from_schema(schema, random.Random(7)))
    items = cast(list[JSONValue], generated["items"])
    generated_strings = [item for item in items if isinstance(item, str)]

    assert len(items) == 32
    assert generated_strings
    assert max(map(len, generated_strings)) <= 4_096


def test_synthetic_generation_can_explicitly_preserve_tail_limits() -> None:
    array_schema: JSONDocument = {
        "type": "array",
        "items": {"type": "integer"},
        "x-polylogue-array-lengths": [2_500, 2_500],
    }
    string_schema: JSONDocument = {
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "x-polylogue-string-lengths": [{"path": "$.body", "min": 8_192, "max": 8_192, "avg": 8_192, "stddev": 0}],
    }
    unbounded_array = SyntheticCorpus(
        array_schema,
        WireFormat(encoding="json"),
        "test",
        max_array_items=None,
    )
    unbounded_string = SyntheticCorpus(
        string_schema,
        WireFormat(encoding="json"),
        "test",
        max_string_length=None,
    )

    items = cast(list[JSONValue], unbounded_array._generate_from_schema(array_schema, random.Random(7)))
    record = cast(dict[str, JSONValue], unbounded_string._generate_from_schema(string_schema, random.Random(7)))

    assert len(items) == 2_500
    assert 8_190 <= len(cast(str, record["body"])) <= 8_192


def test_registry_roundtrips_separate_workload_profile_artifact(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    package = SchemaVersionPackage(
        provider="chatgpt",
        version="v1",
        anchor_kind="session_document",
        default_element_kind="session_document",
        first_seen="2026-01-01T00:00:00+00:00",
        last_seen="2026-01-02T00:00:00+00:00",
        bundle_scope_count=2,
        sample_count=3,
        workload_profile_file="workload-profile.json.gz",
        elements=[
            SchemaElementManifest(
                element_kind="session_document",
                schema_file="session_document.schema.json.gz",
                sample_count=3,
                artifact_count=3,
            )
        ],
    )
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        packages=[package],
        latest_version="v1",
        default_version="v1",
        recommended_version="v1",
    )
    profile = {
        "profile_version": 1,
        "provider": "chatgpt",
        "package_version": "v1",
        "privacy_policy": "standard",
        "elements": {"session_document": {"sample_count": 3}},
    }

    registry.replace_provider_packages(
        "chatgpt",
        catalog,
        {"v1": {"session_document": {"type": "object"}}},
        package_workload_profiles={"v1": profile},
    )

    assert registry.get_workload_profile("chatgpt", "v1") == profile
    assert registry.get_element_schema("chatgpt", version="v1") is not None


def test_synthetic_generation_selects_joint_structural_variants() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "alpha": {"type": "integer", "x-polylogue-frequency": 0.0},
            "beta": {"type": "integer", "x-polylogue-frequency": 0.0},
            "unrelated": {"type": "integer"},
        },
    }
    workload_profile: JSONDocument = {
        "profile_id": "workload-profile:joint-test",
        "elements": {
            "session_document": {
                "structural_variants": [
                    {
                        "count": 10,
                        "tokens": ["field:alpha", "field:beta", "type:alpha:number"],
                    }
                ]
            }
        },
    }
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="json"),
        "test",
        element_kind="session_document",
        workload_profile=workload_profile,
    )

    batch = corpus.generate_batch(count=5, seed=7)
    decoded = [json.loads(item) for item in batch.raw_items]

    assert all(set(item) == {"alpha", "beta"} for item in decoded)
    assert batch.report.workload_profile_id == "workload-profile:joint-test"
    assert sum(batch.report.structural_variant_counts.values()) == 5


def test_joint_variant_selection_does_not_perturb_content_rng() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "alpha": {"type": "integer"},
            "beta": {"type": "integer"},
        },
        "required": ["alpha", "beta"],
    }
    baseline = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")
    profiled = SyntheticCorpus(
        schema,
        WireFormat(encoding="json"),
        "test",
        element_kind="session_document",
        workload_profile={
            "profile_id": "workload-profile:all-fields",
            "elements": {
                "session_document": {"structural_variants": [{"count": 1, "tokens": ["field:alpha", "field:beta"]}]}
            },
        },
    )

    assert profiled.generate(count=3, seed=9) == baseline.generate(count=3, seed=9)


def _codex_record_stream_corpus(*, relationships: JSONDocument | None = None) -> SyntheticCorpus:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "payload": {"type": "object"},
            "timestamp": {"type": "string"},
            "type": {"type": "string"},
            "unrelated": {"type": "string"},
        },
    }
    bucket_tokens: list[JSONValue] = [
        token
        for bucket in ("session_meta", "response_item", "event_msg", "turn_context")
        for token in (
            f"bucket:type:{bucket}",
            f"field:type:{bucket}:payload",
            f"type:type:{bucket}:payload:object",
            f"field:type:{bucket}:timestamp",
            f"type:type:{bucket}:timestamp:string",
            f"field:type:{bucket}:type",
            f"type:type:{bucket}:type:string",
        )
    ]
    workload_profile: JSONDocument = {
        "profile_id": "workload-profile:codex-stream",
        "elements": {"session_record_stream": {"structural_variants": [{"count": 1, "tokens": bucket_tokens}]}},
    }
    if relationships is not None:
        workload_profile["relationships"] = relationships
    return SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "codex",
        element_kind="session_record_stream",
        workload_profile=workload_profile,
    )


def test_synthetic_codex_realizes_record_stream_variant_through_parser() -> None:
    corpus = _codex_record_stream_corpus()

    batch = corpus.generate_batch(
        count=1,
        seed=19,
        messages_per_session=range(2, 3),
        session_native_ids=("synthetic-session",),
    )
    records = [json.loads(line) for line in batch.raw_items[0].splitlines()]

    assert {record["type"] for record in records} == {
        "session_meta",
        "response_item",
        "event_msg",
        "turn_context",
    }
    assert all("unrelated" not in record for record in records)
    sessions = parse_payload("codex", records, "fallback")
    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "synthetic-session"
    assert sessions[0].messages


def test_profile_generated_tool_pair_reaches_production_action_view(tmp_path: Path) -> None:
    corpus = _codex_record_stream_corpus(
        relationships={
            "tool_results": {
                "calls": 1,
                "results": 1,
                "paired": 1,
                "error_results": 1,
                "functions_exec_calls": 1,
            }
        }
    )
    started = time.perf_counter()
    batch = corpus.generate_batch(
        count=1,
        seed=23,
        messages_per_session=range(4, 5),
        session_native_ids=("tool-pair-canary",),
    )
    generated_at = time.perf_counter()
    records = [json.loads(line) for line in batch.raw_items[0].splitlines()]
    sessions = parse_payload("codex", records, "fallback")
    parsed_at = time.perf_counter()

    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session_id = write_parsed_session_to_archive(conn, sessions[0])
    materialized_at = time.perf_counter()
    actions = conn.execute(
        "SELECT tool_name, is_error, exit_code FROM actions WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    queried_at = time.perf_counter()
    spec = tool_pairing_canary_spec(
        profile_id=batch.report.workload_profile_id or "",
        archive_id="archive:test:tool-pair-canary",
    )
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:test",
        runtime_id="python:test",
        archive_id="archive:test:tool-pair-canary",
        generation_id="synthetic:seed:23",
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="generate", wall_ms=(generated_at - started) * 1_000),
            WorkloadPhaseObservation(name="parse", wall_ms=(parsed_at - generated_at) * 1_000),
            WorkloadPhaseObservation(name="materialize", wall_ms=(materialized_at - parsed_at) * 1_000),
            WorkloadPhaseObservation(name="query", wall_ms=(queried_at - materialized_at) * 1_000),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=True, quiescent=True),
        ),
        cleanup_complete=True,
    )

    assert [tuple(action) for action in actions] == [("exec", 1, 1)]
    assert batch.report.relationship_variant_counts == {
        "error_result": 1,
        "functions_exec": 1,
        "paired": 1,
    }
    assert receipt.spec.inputs[0].profile_id == batch.report.workload_profile_id
    assert receipt.receipt_id.startswith("workload-receipt:sha256:")
    conn.close()


def test_profile_generation_activates_every_observed_tool_relationship_class() -> None:
    relationship_names = (
        "paired",
        "missing",
        "orphan",
        "duplicate_calls",
        "duplicate_results",
        "out_of_order",
        "call_without_id",
        "result_without_id",
    )
    tool_results: JSONDocument = {
        "calls": 8,
        "results": 8,
        "error_results": 8,
        "functions_exec_calls": 8,
        "result_record_distance": {"quantiles": {"p99": 4}, "max": 4},
    }
    tool_results.update(dict.fromkeys(relationship_names, 1))
    corpus = _codex_record_stream_corpus(relationships={"tool_results": tool_results})

    batch = corpus.generate_batch(
        count=len(relationship_names),
        seed=31,
        messages_per_session=range(4, 5),
        session_native_ids=tuple(f"relationship-{index}" for index in range(len(relationship_names))),
    )

    for raw in batch.raw_items:
        records = [json.loads(line) for line in raw.splitlines()]
        sessions = parse_payload("codex", records, "fallback")
        assert len(sessions) == 1
        assert sessions[0].messages
    assert {name: batch.report.relationship_variant_counts[name] for name in relationship_names} == dict.fromkeys(
        relationship_names, 1
    )
    assert batch.report.relationship_variant_counts["functions_exec"] == len(relationship_names)
    assert batch.report.relationship_variant_counts["error_result"] == len(relationship_names)
    assert batch.report.relationship_variant_counts["late_result"] == 3


def test_profile_generated_lineage_replays_through_production_write_and_read(tmp_path: Path) -> None:
    corpus = _codex_record_stream_corpus(relationships={"lineage": {"parent_references": 1}})
    started = time.perf_counter()
    batch = corpus.generate_batch(
        count=2,
        seed=29,
        messages_per_session=range(4, 5),
        session_native_ids=("lineage-parent", "lineage-child"),
    )
    generated_at = time.perf_counter()
    parsed = [
        parse_payload("codex", [json.loads(line) for line in raw.splitlines()], "fallback")[0]
        for raw in batch.raw_items
    ]
    parsed_at = time.perf_counter()

    conn = sqlite3.connect(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    parent_id = write_parsed_session_to_archive(conn, parsed[0])
    child_id = write_parsed_session_to_archive(conn, parsed[1])
    materialized_at = time.perf_counter()
    child_physical_count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (child_id,),
    ).fetchone()[0]
    envelope = read_archive_session_envelope(conn, child_id)
    composed_at = time.perf_counter()
    link = conn.execute(
        "SELECT resolved_dst_session_id, inheritance, branch_point_message_id "
        "FROM session_links WHERE src_session_id = ?",
        (child_id,),
    ).fetchone()
    spec = lineage_replay_canary_spec(
        profile_id=batch.report.workload_profile_id or "",
        archive_id="archive:test:lineage-canary",
    )
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:test",
        runtime_id="python:test",
        archive_id="archive:test:lineage-canary",
        generation_id="synthetic:seed:29",
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="generate", wall_ms=(generated_at - started) * 1_000),
            WorkloadPhaseObservation(name="parse", wall_ms=(parsed_at - generated_at) * 1_000),
            WorkloadPhaseObservation(name="materialize", wall_ms=(materialized_at - parsed_at) * 1_000),
            WorkloadPhaseObservation(name="compose", wall_ms=(composed_at - materialized_at) * 1_000),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=True, quiescent=True),
        ),
        cleanup_complete=True,
    )

    assert batch.report.lineage_edge_count == 1
    assert parsed[1].parent_session_provider_id == "lineage-parent"
    assert tuple(link)[:2] == (parent_id, "prefix-sharing")
    assert link["branch_point_message_id"] is not None
    assert child_physical_count < len(envelope.messages)
    assert envelope.lineage_complete is True
    assert receipt.spec.inputs[0].profile_id == batch.report.workload_profile_id
    assert receipt.receipt_id.startswith("workload-receipt:sha256:")
    conn.close()


def test_workload_profile_models_tool_relationships_without_persisting_ids() -> None:
    call_unit = SchemaUnit(
        cluster_payload={},
        schema_samples=[
            {"type": "function_call", "name": "functions.exec", "call_id": "private-call-1"},
            {"type": "function_call", "name": "other", "call_id": "private-call-2"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "private-call-3"},
        ],
        artifact_kind="session_record_stream",
        bundle_scope="session-a",
        profile_tokens=("record:function_call", "field:call_id"),
    )
    result_unit = SchemaUnit(
        cluster_payload={},
        schema_samples=[
            {
                "type": "function_call_output",
                "call_id": "private-call-1",
                "status": "failed",
                "parentUuid": "private-parent",
                "isSidechain": True,
            },
            {"type": "custom_tool_call_output", "call_id": "private-call-3"},
            {"type": "function_call_output", "call_id": "private-orphan"},
        ],
        artifact_kind="session_record_stream",
        bundle_scope="session-a",
        profile_tokens=("record:function_call_output", "field:call_id"),
    )
    memberships = [
        _UnitMembership(call_unit, "family-a"),
        _UnitMembership(result_unit, "family-a"),
    ]
    package = _PackageAccumulator(
        provider="codex",
        anchor_family_id="family-a",
        anchor_kind="session_record_stream",
        memberships=memberships,
        bundle_scopes={"session-a"},
    )
    profile = build_package_workload_profile(
        provider="codex",
        version="v1",
        package=package,
        element_schemas={"session_record_stream": {"type": "object"}},
        privacy_policy="standard",
        observation_outcomes={
            "total": 2,
            "status_counts": {"included": 1, "decode_failed": 1},
            "reason_counts": {"payload_decode_failed": 1, "observed_schema_units": 1},
        },
    )

    relationships = cast(dict[str, JSONValue], profile["relationships"])
    tool_results = cast(dict[str, JSONValue], relationships["tool_results"])
    lineage = cast(dict[str, JSONValue], relationships["lineage"])
    assert tool_results["calls"] == 3
    assert tool_results["results"] == 3
    assert tool_results["paired"] == 2
    assert tool_results["missing"] == 1
    assert tool_results["orphan"] == 1
    assert tool_results["error_results"] == 1
    assert tool_results["functions_exec_calls"] == 2
    assert lineage["parent_references"] == 1
    assert lineage["sidechain_true"] == 1
    serialized = json.dumps(profile, sort_keys=True)
    assert "private-call" not in serialized
    assert "private-parent" not in serialized
    elements = cast(dict[str, JSONValue], profile["elements"])
    stream = cast(dict[str, JSONValue], elements["session_record_stream"])
    loss = cast(dict[str, JSONValue], stream["loss_inventory"])
    all_observations = cast(dict[str, JSONValue], loss["all_observations"])
    assert all_observations["count"] == 2
    provenance = cast(dict[str, JSONValue], profile["provenance"])
    outcomes = cast(dict[str, JSONValue], provenance["observation_outcomes"])
    assert outcomes["status_counts"] == {"included": 1, "decode_failed": 1}


def test_tool_totals_remain_exact_when_pairing_identity_sketch_is_full() -> None:
    observed_calls = _TOOL_IDS_PER_SCOPE_CAP + 3
    unit = SchemaUnit(
        cluster_payload={},
        schema_samples=[
            {"type": "custom_tool_call", "name": "exec", "call_id": f"private-{index}"}
            for index in range(observed_calls)
        ],
        artifact_kind="session_record_stream",
        bundle_scope="large-session",
        profile_tokens=("record:custom_tool_call", "field:call_id"),
    )
    package = _PackageAccumulator(
        provider="codex",
        anchor_family_id="family-a",
        anchor_kind="session_record_stream",
        memberships=[_UnitMembership(unit, "family-a")],
        bundle_scopes={"large-session"},
    )

    profile = build_package_workload_profile(
        provider="codex",
        version="v1",
        package=package,
        element_schemas={"session_record_stream": {"type": "object"}},
        privacy_policy="standard",
    )

    relationships = cast(dict[str, JSONValue], profile["relationships"])
    tool_results = cast(dict[str, JSONValue], relationships["tool_results"])
    loss_inventory = cast(dict[str, JSONValue], relationships["loss_inventory"])
    assert tool_results["calls"] == observed_calls
    assert tool_results["functions_exec_calls"] == observed_calls
    assert loss_inventory["tool_ids_over_capacity"] == 3


def test_archive_profile_preserves_composition_without_private_dimension_values(tmp_path: Path) -> None:
    index_path = tmp_path / "index.db"
    with sqlite3.connect(index_path) as conn:
        conn.executescript(
            """
            PRAGMA user_version = 24;
            CREATE TABLE sessions (
                session_id TEXT, origin TEXT, session_kind TEXT, branch_type TEXT,
                title_source TEXT, parent_session_id TEXT, title TEXT,
                instructions_text TEXT, git_branch TEXT, git_repository_url TEXT,
                provider_project_ref TEXT, message_count INTEGER, word_count INTEGER,
                tool_use_count INTEGER, thinking_count INTEGER, paste_count INTEGER,
                user_message_count INTEGER, authored_user_message_count INTEGER,
                assistant_message_count INTEGER, system_message_count INTEGER,
                tool_message_count INTEGER, user_word_count INTEGER,
                authored_user_word_count INTEGER, assistant_word_count INTEGER,
                reported_duration_ms INTEGER, created_at_ms INTEGER,
                updated_at_ms INTEGER, sort_key_ms INTEGER
            );
            CREATE TABLE messages (
                session_id TEXT, position INTEGER, variant_index INTEGER, role TEXT,
                message_type TEXT, material_origin TEXT, model_name TEXT,
                user_context_text TEXT, word_count INTEGER, input_tokens INTEGER,
                output_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, duration_ms INTEGER
            );
            CREATE TABLE blocks (
                session_id TEXT, tool_id TEXT, block_type TEXT, tool_name TEXT,
                text TEXT, tool_input TEXT
            );
            CREATE TABLE session_links (
                src_session_id TEXT, resolved_dst_session_id TEXT, link_type TEXT,
                inheritance TEXT, status TEXT
            );
            INSERT INTO sessions VALUES
                ('s1', 'codex-session', 'standard', NULL, 'origin', NULL,
                 'secret title', 'secret instructions', 'private-branch',
                 'ssh://private/repo', 'private-project', 2, 30, 1, 0, 0,
                 1, 1, 1, 0, 0, 10, 10, 20, 100, 1000, 2000, 2000),
                ('s2', 'claude-code-session', 'standard', 'subagent', 'path', 's1',
                 'other title', NULL, 'private-branch', 'ssh://private/repo',
                 'other-project', 1, 8, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 8,
                 NULL, 1500, 2500, 2500),
                ('s3', 'codex-session', 'standard', NULL, 'origin', NULL,
                 NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, NULL, 1600, 2600, 2600);
            INSERT INTO messages VALUES
                ('s1', 0, 0, 'user', 'message', 'human_authored',
                 'private-model-name', 'private context', 10, 12, 0, 0, 0, 20),
                ('s1', 1, 0, 'assistant', 'message', 'assistant_authored',
                 'private-model-name', NULL, 20, 20, 30, 5, 2, 80),
                ('s2', 0, 0, 'assistant', 'thinking', 'assistant_authored',
                 'other-private-model', NULL, 8, 8, 9, 0, 0, NULL);
            INSERT INTO blocks VALUES
                ('s1', 'private-tool-id', 'tool_use', 'private_tool_name',
                 NULL, '{"command":"private command"}'),
                ('s1', 'private-tool-id', 'tool_result', 'private_tool_name',
                 'private output', NULL),
                ('s2', NULL, 'thinking', NULL, 'private reasoning', NULL),
                ('s2', NULL, 'tool_use', 'private-unidentified-tool', NULL, NULL),
                ('s2', '', 'tool_result', NULL, 'private unidentified output', NULL);
            INSERT INTO session_links VALUES ('s2', 's1', 'subagent', 'spawned-fresh', NULL);
            """
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executescript(
            """
            PRAGMA user_version = 3;
            CREATE TABLE raw_sessions (
                origin TEXT, revision_kind TEXT, revision_authority TEXT,
                capture_mode TEXT, validation_status TEXT, blob_size INTEGER,
                append_start_offset INTEGER, append_end_offset INTEGER,
                logical_source_key TEXT, parsed_at_ms INTEGER, parse_error TEXT,
                acquired_at_ms INTEGER, acquisition_generation INTEGER
            );
            INSERT INTO raw_sessions VALUES
                ('codex-session', 'append', 'byte_proven', 'codex', 'passed',
                 4096, 100, 300, '/private/source/path', 20, NULL, 10, 7),
                ('codex-session', 'full', 'asserted', 'codex', 'passed',
                 8192, NULL, NULL, '/private/source/path', 30, NULL, 20, 8);
            """
        )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE ingest_cursor (
                origin TEXT, failure_count INTEGER, record_count INTEGER,
                stat_size INTEGER, byte_offset INTEGER, excluded INTEGER
            );
            CREATE TABLE convergence_debt (
                stage TEXT, status TEXT, target_type TEXT, attempts INTEGER
            );
            CREATE TABLE cursor_lag_samples (
                family TEXT, severity TEXT, lag_ms INTEGER, stuck_file_count INTEGER
            );
            INSERT INTO ingest_cursor VALUES ('codex-session', 1, 20, 1000, 750, 0);
            INSERT INTO convergence_debt VALUES ('private-stage', 'deferred', 'private-target', 2);
            INSERT INTO cursor_lag_samples VALUES ('private-family', 'warning', 3000, 1);
            """
        )

    profile = build_archive_workload_profile(
        index_path,
        package_bundle_scope_counts={"codex": {"v1": 2}},
    )
    repeated = build_archive_workload_profile(
        index_path,
        package_bundle_scope_counts={"codex": {"v1": 2}},
    )

    assert profile is not None
    assert repeated == profile
    index = cast(dict[str, JSONValue], profile["index"])
    row_counts = cast(dict[str, JSONValue], index["row_counts"])
    action_shapes = cast(dict[str, JSONValue], index["action_shapes"])
    pairing = cast(dict[str, JSONValue], action_shapes["tool_pairing"])
    tool_uses = cast(dict[str, JSONValue], action_shapes["tool_uses_per_session"])
    tool_results = cast(dict[str, JSONValue], action_shapes["tool_results_per_session"])
    archive_mix = cast(dict[str, JSONValue], profile["archive_mix"])
    scale_tiers = cast(dict[str, JSONValue], profile["scale_tiers"])
    selectivity_tiers = cast(dict[str, JSONValue], profile["selectivity_tiers"])
    assert row_counts == {"sessions": 3, "messages": 3, "blocks": 5}
    assert pairing["paired"] == 1
    assert pairing["unknown_identity_uses"] == 1
    assert pairing["unknown_identity_results"] == 1
    assert tool_uses["count"] == 3
    assert tool_uses["min"] == 0
    assert tool_results["count"] == 3
    assert tool_results["min"] == 0
    assert archive_mix["package_bundle_scope_counts"] == {"codex": {"v1": 2}}
    ci_tier = cast(dict[str, JSONValue], scale_tiers[WorkloadScaleTier.CI_ACTIVATION.value])
    archive_1x = cast(dict[str, JSONValue], scale_tiers[WorkloadScaleTier.ARCHIVE_1X.value])
    archive_10x = cast(dict[str, JSONValue], scale_tiers[WorkloadScaleTier.ARCHIVE_10X.value])
    assert ci_tier["row_counts"] == row_counts
    assert ci_tier["selectivity_tier"] == WorkloadSelectivityTier.EXACT_ONE.value
    assert archive_1x["row_counts"] == row_counts
    assert archive_10x["row_counts"] == {"sessions": 30, "messages": 30, "blocks": 50}
    exact_tier = cast(dict[str, JSONValue], selectivity_tiers[WorkloadSelectivityTier.EXACT_ONE.value])
    assert exact_tier["cardinality"] == 1
    assert profile["profile_id"] == repeated["profile_id"]

    serialized = json.dumps(profile, sort_keys=True)
    for private_value in (
        "secret title",
        "secret instructions",
        "private-branch",
        "ssh://private/repo",
        "private-project",
        "private-model-name",
        "private context",
        "private_tool_name",
        "private command",
        "private-tool-id",
        "private output",
        "private reasoning",
        "private unidentified output",
        "/private/source/path",
        "private-stage",
        "private-target",
        "private-family",
    ):
        assert private_value not in serialized

    path = write_archive_workload_profile(tmp_path / "staged", profile)
    first_bytes = path.read_bytes()
    write_archive_workload_profile(tmp_path / "staged", profile)
    assert path.read_bytes() == first_bytes
    assert json.loads(gzip.decompress(first_bytes)) == profile
