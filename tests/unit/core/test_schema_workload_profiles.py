"""Behavioral contracts for bounded schema workload profiles."""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
import time
from collections.abc import Mapping, Sequence
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
    build_package_workload_profile,
    workload_profile_identity,
)
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.workload_profile_rows import DepthGrammars, WorkloadProfile, iter_workload_profile_chunks
from polylogue.schemas.workload_tiers import WorkloadScaleTier, WorkloadSelectivityTier
from polylogue.sources.dispatch import parse_payload
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.schema_workload_profiles import inferred_record_profile


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
    assert stats["$.items"].array_length_distribution.minimum == 0
    assert "array_length_samples" not in stats["$.items"].truncated_evidence
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
    stats = _collect_field_stats(samples)
    annotated = annotate_schema(schema, stats)
    count_node = cast(JSONDocument, cast(JSONDocument, annotated["properties"])["count"])
    distribution = cast(JSONDocument, count_node["x-polylogue-observed-distribution"])
    # Explicitly supplied numeric distributions remain usable by synthetic consumers.
    distribution["numeric"] = stats["$.count"].numeric_distribution.to_payload()
    corpus = SyntheticCorpus(
        annotated,
        WireFormat(encoding="json"),
        "test",
    )

    generated = [corpus._generate_from_schema(annotated, random.Random(seed)) for seed in range(100)]
    records = [record for record in generated if isinstance(record, Mapping)]
    counts = [record["count"] for record in records]
    item_values = [record["items"] for record in records]
    assert all(isinstance(items, Sequence) and not isinstance(items, str | bytes) for items in item_values)
    lengths = [len(items) for items in item_values if isinstance(items, Sequence)]

    assert set(counts) <= {1, 1_000}
    assert 1 in lengths
    assert max(lengths) >= 19
    corpus._relation_solver.close()


def _point_distribution(value: int) -> JSONDocument:
    """A one-bucket observed histogram pinned to exactly ``value`` by its min and max."""
    return {"count": 1, "histogram": [[80, 1]], "log_base": 1.1, "min": value, "max": value}


def test_synthetic_generation_keeps_observed_tail_sizes() -> None:
    """Removing a size cap from generation: a 2,500-item array and 8,192-character strings survive."""
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "string",
                    "x-polylogue-observed-distribution": {"string_length": _point_distribution(8_192)},
                },
                "x-polylogue-observed-distribution": {"array_length": _point_distribution(2_500)},
            }
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")

    generated = cast(dict[str, JSONValue], corpus._generate_from_schema(schema, random.Random(7)))
    items = cast(list[JSONValue], generated["items"])

    assert len(items) == 2_500
    assert {len(cast(str, item)) for item in items} == {8_192}


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


def test_synthetic_generation_refuses_retired_private_field_variants() -> None:
    with pytest.raises(ValueError, match="retired"):
        SyntheticCorpus(
            {"type": "object", "properties": {"alpha": {"type": "integer"}}},
            WireFormat(encoding="json"),
            "test",
            element_kind="session_document",
            workload_profile={
                "elements": {
                    "session_document": {
                        "structural_variants": [{"count": 10, "tokens": ["field:alpha", "field:beta"]}]
                    }
                }
            },
        )


def test_public_record_profile_generation_is_seeded_and_keeps_field_presence_owner() -> None:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["assistant"]},
            "uuid": {"type": "string", "x-polylogue-values": ["neutral-message"]},
            "alpha": {"type": "integer", "x-polylogue-frequency": 0.0},
            "beta": {"type": "integer", "x-polylogue-frequency": 1.0},
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["assistant"]},
                    "content": {"type": "string", "x-polylogue-values": ["neutral body"]},
                },
            },
        },
    }
    profile = inferred_record_profile(
        "claude-code",
        schema,
        [{"type": "assistant", "message": {"role": "assistant", "content": "neutral body"}}],
        element="session_record_stream",
    )
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    batch = corpus.generate_batch(count=5, seed=7)
    assert batch.raw_items == corpus.generate_batch(count=5, seed=7).raw_items
    decoded = [json.loads(item) for item in batch.raw_items]
    assert all("alpha" not in item and "beta" in item for item in decoded)
    assert batch.report.workload_profile_id == profile["profile_id"]
    assert sum(batch.report.sampled_record_kind_counts.values()) == 5


def _codex_record_stream_corpus(*, lineage: bool = False) -> SyntheticCorpus:
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "x-polylogue-values": ["session_meta", "response_item", "event_msg", "turn_context"],
            },
            "timestamp": {"type": "string"},
            "unrelated": {"type": "string", "x-polylogue-frequency": 0.0},
            "payload": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "x-polylogue-values": ["message", "token_count"],
                        "x-polylogue-frequency": 0.5,
                    },
                    "role": {"type": "string", "x-polylogue-values": ["user"], "x-polylogue-frequency": 0.25},
                    "id": {"type": "string"},
                    "cwd": {"type": "string"},
                    "model": {"type": "string"},
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["input_text"]},
                                "text": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }
    samples: list[JSONDocument] = [
        {"type": "session_meta", "payload": {"id": "neutral-session"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "neutral body"}]},
        },
        {"type": "event_msg", "payload": {"type": "token_count"}},
        {"type": "turn_context", "payload": {"cwd": "/workspace", "model": "neutral-model"}},
    ]
    if lineage:
        payload = cast(JSONDocument, cast(JSONDocument, schema["properties"])["payload"])
        properties = cast(JSONDocument, payload["properties"])
        properties["forked_from_id"] = {"type": "string"}
        prefix = samples[1]
        tail: JSONDocument = {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "neutral tail"}]},
        }
        units = [
            SchemaUnit(
                cluster_payload={},
                schema_samples=[{"type": "session_meta", "payload": {"id": "lineage-parent"}}, prefix],
                artifact_kind="session_record_stream",
                bundle_scope="neutral-parent",
            ),
            SchemaUnit(
                cluster_payload={},
                schema_samples=[
                    {"type": "session_meta", "payload": {"id": "lineage-child", "forked_from_id": "lineage-parent"}},
                    prefix,
                    tail,
                ],
                artifact_kind="session_record_stream",
                bundle_scope="neutral-child",
            ),
        ]
        profile = build_package_workload_profile(
            provider="codex",
            version="v1",
            package=_PackageAccumulator(
                provider="codex",
                anchor_family_id="neutral-family",
                anchor_kind="session_record_stream",
                memberships=[_UnitMembership(unit, "neutral-family") for unit in units],
            ),
            element_schemas={"session_record_stream": schema},
            privacy_policy="standard",
        )
    else:
        profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    return SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
    )


def _measured_tool_relationship_corpus(variant: str, *, native_id: str) -> SyntheticCorpus:
    call: JSONDocument = {
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "call_id": "neutral-tool", "name": "exec", "input": "printf neutral"},
    }
    result: JSONDocument = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "neutral-tool",
            "output": {"output": "neutral failure", "metadata": {"exit_code": 1}},
        },
    }
    if variant == "call_without_id":
        cast(JSONDocument, call["payload"]).pop("call_id")
    if variant == "result_without_id":
        cast(JSONDocument, result["payload"]).pop("call_id")
    operands = {
        "paired": [call, result],
        "missing": [call],
        "orphan": [result],
        "duplicate_calls": [call, call, result],
        "duplicate_results": [call, result, result],
        "out_of_order": [result, call],
        "call_without_id": [call],
        "result_without_id": [result],
    }[variant]
    samples: list[JSONDocument] = [{"type": "session_meta", "payload": {"id": native_id}}, *operands]

    def shape(value: JSONValue) -> JSONDocument:
        if isinstance(value, dict):
            return {
                "type": "object",
                "required": list(value),
                "properties": {key: shape(child) for key, child in value.items()},
            }
        if isinstance(value, str):
            return {"type": "string", "x-polylogue-values": [value]}
        if isinstance(value, int):
            return {"type": "integer", "x-polylogue-range": [value, value]}
        raise AssertionError("neutral tool fixture contains an undeclared value")

    schema: JSONDocument = {"oneOf": [shape(sample) for sample in samples]}
    # Duplicate physical events are a measured relationship, not duplicated
    # union branches that would make oneOf support ambiguous.
    unique: list[JSONValue] = []
    for branch in cast(list[JSONValue], schema["oneOf"]):
        if branch not in unique:
            unique.append(branch)
    schema["oneOf"] = unique
    profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    return SyntheticCorpus(
        schema, WireFormat(encoding="jsonl"), "codex", element_kind="session_record_stream", workload_profile=profile
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
    corpus = _measured_tool_relationship_corpus("paired", native_id="tool-pair-canary")
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

    conn = connect_measured(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session_id = write_fixture_index_session(conn, sessions[0])
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
    assert batch.report.relationship_variant_counts == {"paired": 1}
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
    for relationship in relationship_names:
        corpus = _measured_tool_relationship_corpus(relationship, native_id="relationship-" + relationship)
        batch = corpus.generate_batch(count=1, seed=31, session_native_ids=("relationship-" + relationship,))
        records = [json.loads(line) for line in batch.raw_items[0].splitlines()]
        sessions = parse_payload("codex", records, "fallback")
        assert len(sessions) == 1
        assert sessions[0].messages
        assert batch.report.relationship_variant_counts[relationship] == 1
        # No late template can add unmeasured calls, results or filler records.
        expected_count = (
            4
            if relationship in {"duplicate_calls", "duplicate_results"}
            else 3
            if relationship in {"paired", "out_of_order"}
            else 2
        )
        assert len(records) == expected_count
        assert (
            batch.raw_items
            == corpus.generate_batch(count=1, seed=31, session_native_ids=("relationship-" + relationship,)).raw_items
        )


def test_profile_generated_lineage_replays_through_production_write_and_read(tmp_path: Path) -> None:
    corpus = _codex_record_stream_corpus(lineage=True)
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

    conn = connect_measured(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    parent_id = write_fixture_index_session(conn, parsed[0])
    child_id = write_fixture_index_session(conn, parsed[1])
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
    serialized = b"".join(iter_workload_profile_chunks(profile)).decode()
    assert "private-call" not in serialized
    assert "private-parent" not in serialized
    elements = cast(dict[str, JSONValue], profile["elements"])
    stream = cast(dict[str, JSONValue], elements["session_record_stream"])
    structure = cast(JSONDocument, stream["record_structure"])
    assert sum(cast(int, count) for count in cast(JSONDocument, structure["record_kinds"]).values()) == 6
    assert "field_profiles" not in stream
    provenance = cast(dict[str, JSONValue], profile["provenance"])
    outcomes = cast(dict[str, JSONValue], provenance["observation_outcomes"])
    assert outcomes["status_counts"] == {"included": 1, "decode_failed": 1}


def test_tool_pairing_keeps_every_identity_beyond_the_retired_capacity() -> None:
    observed_calls = 4_099
    records: list[JSONDocument] = [
        cast(JSONDocument, {"type": "custom_tool_call", "name": "exec", "call_id": f"private-{index}"})
        for index in range(observed_calls)
    ] + [
        cast(JSONDocument, {"type": "custom_tool_call_output", "call_id": f"private-{index}"})
        for index in range(observed_calls)
    ]
    unit = SchemaUnit(
        cluster_payload={},
        schema_samples=records,
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
    assert tool_results["calls"] == observed_calls
    assert tool_results["functions_exec_calls"] == observed_calls
    assert tool_results["paired"] == observed_calls
    assert tool_results["missing"] == tool_results["orphan"] == 0
    assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()


def test_original_journal_pairs_noncontiguous_scope_fragments_without_hydrating_units(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal

    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        fragments: tuple[tuple[str, list[JSONDocument]], ...] = (
            ("first", [{"type": "function_call", "call_id": "private-first"}]),
            ("second", [{"type": "function_call_output", "call_id": "private-first"}]),
            ("first", [{"type": "function_call_output", "call_id": "private-first"}]),
            ("empty", []),
        )
        for scope, records in fragments:
            unit = SchemaUnit(
                cluster_payload={},
                schema_samples=records,
                artifact_kind="session_record_stream",
                bundle_scope=scope,
            )
            unit_id = journal.append_unit(unit)
            journal.assign_profile_family(unit_id, "family")
        memberships = journal.memberships()
        journal._connection.execute("SAVEPOINT original_profile_window")
        # Reading the workload view must not construct a whole unit. Its
        # independent full sample traversal still groups original fragments.
        original_unit_reader = journal._unit_from_row
        journal._unit_from_row = lambda *args, **kwargs: pytest.fail("whole-unit hydration")  # type: ignore[method-assign]
        try:
            samples = list(memberships.iter_scope_samples(order="bundle"))
            assert [original.bundle_scope for original in samples] == ["empty", "first", "first", "second"]
            profile = build_package_workload_profile(
                provider="codex",
                version="v1",
                package=_PackageAccumulator(
                    provider="codex",
                    anchor_family_id="family",
                    anchor_kind="session_record_stream",
                    memberships=memberships,
                ),
                element_schemas={"session_record_stream": {"type": "object"}},
                privacy_policy="standard",
            )
        finally:
            journal._unit_from_row = original_unit_reader  # type: ignore[method-assign]
        relationships = cast(JSONDocument, profile["relationships"])
        tools = cast(JSONDocument, relationships["tool_results"])
        assert tools["paired"] == 1
        assert tools["orphan"] == 1
        assert tools["missing"] == 0
        counts = cast(JSONDocument, relationships["scope_record_count"])
        assert counts["count"] == 3 and counts["min"] == 0 and counts["max"] == 2
        assert "private-first" not in b"".join(iter_workload_profile_chunks(profile)).decode()
        assert not journal._connection.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE name = 'workload_tool_events'"
        ).fetchone()
        # The profile scan must not commit the original journal transaction.
        journal._connection.execute("ROLLBACK TO original_profile_window")
        journal._connection.execute("RELEASE original_profile_window")


def test_synthetic_profile_refuses_the_retired_duplicate_field_authority() -> None:
    with pytest.raises(ValueError):
        SyntheticCorpus(
            {"type": "object"},
            WireFormat(encoding="json"),
            "test",
            workload_profile={"elements": {"session_document": {"field_profiles": {}}}},
        )


def test_package_profile_keeps_every_scope_without_publishing_private_field_variants() -> None:
    memberships = [
        _UnitMembership(
            SchemaUnit(
                cluster_payload={},
                schema_samples=[],
                artifact_kind="session_document",
                raw_id=f"neutral-{index}",
                profile_tokens=(f"field:neutral_{index}",),
            ),
            "family",
        )
        for index in range(300)
    ]
    profile = build_package_workload_profile(
        provider="chatgpt",
        version="v1",
        package=_PackageAccumulator(
            provider="chatgpt",
            anchor_family_id="family",
            anchor_kind="session_document",
            memberships=memberships,
        ),
        element_schemas={
            "session_document": {
                "type": "object",
                "x-polylogue-observed-distribution": {"object_fanout": _point_distribution(1)},
            }
        },
        privacy_policy="standard",
    )
    element = cast(JSONDocument, cast(JSONDocument, profile["elements"])["session_document"])
    structure = cast(JSONDocument, element["record_structure"])
    counts = cast(JSONDocument, structure["scope_record_count"])
    assert counts["count"] == 300 and counts["min"] == 0 and counts["max"] == 0
    assert structure["record_kinds"] == {} and structure["transitions"] == []
    assert "neutral_" not in b"".join(iter_workload_profile_chunks(profile)).decode()
    assert "structural_variants" not in element
    assert "field_profiles" not in element and "loss_inventory" not in element
    assert "x-polylogue-observed-distribution" not in b"".join(iter_workload_profile_chunks(profile)).decode()


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

    serialized = b"".join(iter_workload_profile_chunks(profile)).decode()
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


def test_canonical_archive_measurements_refuse_missing_columns_instead_of_emitting_partial_profiles() -> None:
    from polylogue.schemas.generation.archive_workload_profile import (
        _anonymous_cardinality_profile,
        _mix,
        _scan_table_profile,
    )

    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('neutral')")
        for observe in (
            lambda: _mix(conn, "sessions", "origin"),
            lambda: _scan_table_profile(conn, "sessions", mix_columns=("origin",)),
            lambda: _scan_table_profile(conn, "sessions", length_columns=("title",)),
            lambda: _anonymous_cardinality_profile(conn, table="sessions", column="git_branch"),
        ):
            with pytest.raises(sqlite3.OperationalError):
                observe()


def test_package_profile_uses_public_wire_order_and_ignores_tool_looking_arguments() -> None:
    records: list[JSONDocument] = [
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "private-call",
                        "name": "Read",
                        "input": {
                            "type": "function_call_output",
                            "call_id": "private-orphan",
                            "parentUuid": "private-parent",
                        },
                    }
                ]
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "private-call", "content": "neutral"}]},
        },
    ]
    unit = SchemaUnit(
        cluster_payload={}, schema_samples=records, artifact_kind="session_record_stream", bundle_scope="neutral"
    )
    profile = build_package_workload_profile(
        provider="claude-code",
        version="v1",
        package=_PackageAccumulator(
            provider="claude-code",
            anchor_family_id="family",
            anchor_kind="session_record_stream",
            memberships=[_UnitMembership(unit, "family")],
        ),
        element_schemas={"session_record_stream": {"type": "object"}},
        privacy_policy="standard",
    )
    structure = cast(
        JSONDocument,
        cast(JSONDocument, cast(JSONDocument, profile["elements"])["session_record_stream"])["record_structure"],
    )
    assert structure["initial_kinds"] == {"assistant_tool_use": 1}
    assert structure["transitions"] == [{"from": "assistant_tool_use", "to": "user_tool_result", "count": 1}]
    gaps = cast(JSONDocument, structure["adjacent_timestamp_gap_seconds"])
    assert gaps["count"] == 1 and gaps["min"] == 2 and gaps["max"] == 2
    relationships = cast(JSONDocument, profile["relationships"])
    tools = cast(JSONDocument, relationships["tool_results"])
    assert tools["calls"] == 1 and tools["results"] == 1 and tools["paired"] == 1 and tools["orphan"] == 0
    assert cast(JSONDocument, relationships["lineage"])["parent_references"] == 0
    assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()


def test_measured_public_blocks_keep_variable_positions_and_original_branch_support() -> None:
    from polylogue.schemas.record_structure import classify_claude_code_record

    text: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["text"]},
            "text": {"type": "string", "x-polylogue-values": ["neutral body"]},
        },
    }
    tool: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["tool_use"]},
            "id": {"type": "string", "x-polylogue-format": "uuid4"},
            "name": {"type": "string", "x-polylogue-values": ["exec"]},
            "input": {"type": "object"},
        },
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["assistant"]},
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["assistant"]},
                    "content": {"type": "array", "items": {"oneOf": [text, tool]}},
                },
            },
        },
    }
    samples: list[JSONDocument] = []
    for kinds in (("text", "tool_use", "text"), ("tool_use", "text", "text"), ("text", "text", "tool_use")):
        blocks: list[JSONValue] = [
            {"type": kind, "text": "neutral body"}
            if kind == "text"
            else {"type": kind, "id": "00000000-0000-4000-8000-000000000001", "name": "exec", "input": {}}
            for kind in kinds
        ]
        samples.append({"type": "assistant", "message": {"role": "assistant", "content": blocks}})
    # The real field annotation owner must retain the same neutral identity
    # width as the record relation producer; a format label alone is not a
    # measured width distribution.
    tool_samples = [
        block
        for sample in samples
        for block in cast(list[JSONDocument], cast(JSONDocument, sample["message"])["content"])
        if block["type"] == "tool_use"
    ]
    annotate_schema(tool, _collect_field_stats(tool_samples))
    profile = inferred_record_profile("claude-code", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    batch = corpus.generate_batch(count=20, seed=29, messages_per_session=range(1, 2))
    assert batch.raw_items == corpus.generate_batch(count=20, seed=29, messages_per_session=range(1, 2)).raw_items
    positions: set[int] = set()
    for raw in batch.raw_items:
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == 3
        for record in records:
            assert classify_claude_code_record(record) == "assistant_tool_use"
            blocks = record["message"]["content"]
            assert len(blocks) == 3
            assert sum(isinstance(block, dict) and block["type"] == "tool_use" for block in blocks) == 1
            for index, block in enumerate(blocks):
                assert isinstance(block, dict)
                if block["type"] == "tool_use":
                    positions.add(index)
                    assert "input" in block and "text" not in block
                else:
                    assert block["text"] == "neutral body" and "input" not in block
        parsed = parse_payload("claude-code", records, "neutral-public-grammar")
        assert parsed and sum(len(session.messages) for session in parsed) == 3
    assert len(positions) > 1
    assert batch.report.sampled_record_kind_counts == {"assistant_tool_use": 60}


def test_measured_document_scope_reuses_public_role_and_field_owners() -> None:
    item: JSONDocument = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "sender": {"type": "string", "x-polylogue-values": ["human", "assistant"]},
            "text": {"type": "string", "x-polylogue-values": ["neutral body"]},
        },
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "name": {"type": "string", "x-polylogue-values": ["Neutral session"]},
            "chat_messages": {"type": "array", "items": item},
        },
    }
    samples: list[JSONDocument] = [
        {
            "uuid": "neutral-session",
            "chat_messages": [
                {"uuid": "neutral-user", "sender": "human", "text": "neutral body"},
                {"uuid": "neutral-assistant", "sender": "assistant", "text": "neutral body"},
            ],
        }
    ]
    profile = inferred_record_profile("claude-ai", schema, samples, element="session_document")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="json", messages_path="chat_messages"),
        "claude-ai",
        element_kind="session_document",
        workload_profile=profile,
    )
    batch = corpus.generate_batch(count=1, seed=31, messages_per_session=range(1, 2))
    document = json.loads(batch.raw_items[0])
    assert [message["sender"] for message in document["chat_messages"]] == ["human", "assistant"]
    assert all(message["text"] == "neutral body" for message in document["chat_messages"])
    parsed = parse_payload("claude-ai", document, "neutral-public-document")
    assert parsed and sum(len(session.messages) for session in parsed) == 2
    assert batch.report.sampled_record_kind_counts == {"user_message": 1, "assistant_message": 1}


def test_measured_role_carriers_preserve_parser_precedence_and_absence() -> None:
    from polylogue.schemas.record_structure import public_role_carriers
    from polylogue.sources.parsers.claude.code_parser import _record_role

    role_schema: JSONDocument = {
        "oneOf": [
            {"type": "string", "x-polylogue-values": ["human", "assistant", "system"]},
            {"type": "null"},
        ]
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["user"]},
            "role": role_schema,
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "message": {
                "type": "object",
                "properties": {
                    "role": role_schema,
                    "content": {"type": "string", "x-polylogue-values": ["neutral body"]},
                },
            },
        },
    }
    samples: list[JSONDocument] = [
        {"type": "user", "role": "system", "message": {"role": "assistant", "content": "neutral body"}},
        {"type": "user", "role": "human", "message": {"role": None, "content": "neutral body"}},
        {"type": "user", "message": {"content": "neutral body"}},
    ]
    profile = inferred_record_profile("claude-code", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    batch = corpus.generate_batch(count=30, seed=11)
    expected = {
        public_role_carriers(record): _record_role(cast(dict[str, object], record), record["message"])
        for record in samples
    }
    seen = set()
    for raw in batch.raw_items:
        records = [json.loads(line) for line in raw.splitlines()]
        for record in records:
            carrier = public_role_carriers(record)
            assert carrier in expected
            seen.add(carrier)
            assert _record_role(record, record["message"]) == expected[carrier]
        parsed = parse_payload("claude-code", records, "neutral-role-joints")
        assert parsed and sum(len(session.messages) for session in parsed) == 3
    assert seen == set(expected)


def test_measured_document_empty_nodes_keep_null_absence_and_real_graph() -> None:
    from polylogue.schemas.field_stats.distributions import DistributionSketch
    from polylogue.schemas.record_structure import record_form, record_kind
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS

    parts_lengths = DistributionSketch()
    parts_lengths.observe(1)
    identifier: JSONDocument = {"type": "string", "x-polylogue-format": "uuid4"}
    message: JSONDocument = {
        "type": "object",
        "properties": {
            "id": identifier,
            "author": {
                "type": "object",
                "properties": {"role": {"type": "string", "x-polylogue-values": ["user", "assistant"]}},
            },
            "content": {
                "type": "object",
                "properties": {
                    "content_type": {"type": "string", "x-polylogue-values": ["text"]},
                    "parts": {
                        "type": "array",
                        "items": {"type": "string", "x-polylogue-values": ["neutral body"]},
                        "x-polylogue-observed-distribution": {"array_length": parts_lengths.to_payload()},
                    },
                },
            },
        },
    }
    node: JSONDocument = {
        "type": "object",
        "properties": {
            "id": identifier,
            "parent": {"type": ["string", "null"]},
            "children": {"type": "array", "items": {"type": "string"}},
            "message": {"oneOf": [message, {"type": "null"}]},
        },
    }
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "id": identifier,
            "title": {"type": "string", "x-polylogue-values": ["Neutral session"]},
            "current_node": {"type": "string"},
            "mapping": {"type": "object", "additionalProperties": node},
        },
    }
    samples: list[JSONDocument] = [
        {
            "mapping": {
                "root": {"message": None},
                "empty": {},
                "user": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["neutral body"]},
                    }
                },
                "assistant": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["neutral body"]},
                    }
                },
            }
        }
    ]
    profile = inferred_record_profile("chatgpt", schema, samples, element="session_document")
    corpus = SyntheticCorpus(
        schema, PROVIDER_WIRE_FORMATS["chatgpt"], "chatgpt", element_kind="session_document", workload_profile=profile
    )
    batch = corpus.generate_batch(count=20, seed=31)
    forms = set()
    for raw in batch.raw_items:
        document = json.loads(raw)
        mapping = document["mapping"]
        assert len(mapping) == 4 and document["current_node"] in mapping
        assert sum(record_kind("chatgpt", record) == "record:empty" for record in mapping.values()) == 2
        for key, record in mapping.items():
            assert record["id"] == key
            if record_kind("chatgpt", record) == "record:empty":
                forms.add(record_form("chatgpt", record))
                assert "message" not in record or record["message"] is None
            for child in record["children"]:
                assert mapping[child]["parent"] == key
        parsed = parse_payload("chatgpt", document, "neutral-empty-tree")
        assert parsed and sum(len(session.messages) for session in parsed) == 2
    assert forms == {"chatgpt-message-absent", "chatgpt-message-null"}


def test_original_opaque_metrics_preserve_scoped_zero_counts_lengths_and_package_exclusion(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal

    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        for scope in ("private-one", "private-empty"):
            unit_id = journal.append_unit(
                SchemaUnit(
                    cluster_payload={}, schema_samples=[], artifact_kind="session_record_stream", bundle_scope=scope
                )
            )
            journal.assign_profile_family(unit_id, "family")
        for raw_id, scope, length in (
            ("a", "private-one", 0),
            ("b", "private-one", 10_000_003),
            ("outside", "private-other", 999),
        ):
            journal.record_payload(
                raw_id=raw_id, artifact_kind="tool_result_sidecar", bundle_scope=scope, byte_count=length
            )
            journal.record_terminal(
                raw_id=raw_id,
                status="intentionally_excluded",
                artifact_kind="tool_result_sidecar",
                source_path=f"/neutral/{scope}/tool-results/{raw_id}.txt",
                reason="opaque",
            )
        journal._connection.execute("SAVEPOINT original")
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": {"type": "object"}},
            privacy_policy="standard",
        )
        metrics = cast(JSONDocument, profile["opaque_payloads"])
        assert metrics["scope_count"] == 2
        kinds = cast(JSONDocument, metrics["kinds"])
        sidecars = cast(JSONDocument, kinds["tool_result_sidecar"])
        lengths = cast(JSONDocument, sidecars["bytes"])
        counts = cast(JSONDocument, sidecars["count_per_scope"])
        assert lengths["count"] == 2 and lengths["min"] == 0 and lengths["max"] == 10_000_003
        assert counts["count"] == 2 and counts["min"] == 0 and counts["max"] == 2
        assert counts["mean"] == 1
        assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()
        journal._connection.execute("ROLLBACK TO original")


def test_original_scoped_sidecar_index_conditions_byte_sketches_on_actual_references(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal

    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        records = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "private-direct", "content": "inline"}],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "private-persisted",
                            "content": "Full output saved to: /neutral/owner/tool-results/pointer.txt",
                        }
                    ],
                },
            },
        ]
        for scope, fragments in (("owner", records[:1]), ("empty", []), ("owner", records[1:])):
            unit_id = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=cast(list[JSONDocument], fragments),
                    artifact_kind="session_record_stream",
                    bundle_scope=scope,
                )
            )
            journal.assign_profile_family(unit_id, "family")
        for raw_id, filename, byte_count in (
            ("a", "private-direct.txt", 3),
            ("b", "pointer.txt", 10_000_003),
            ("c", "unowned.txt", 0),
        ):
            journal.record_payload(
                raw_id=raw_id, artifact_kind="tool_result_sidecar", bundle_scope="owner", byte_count=byte_count
            )
            journal.record_terminal(
                raw_id=raw_id,
                status="intentionally_excluded",
                artifact_kind="tool_result_sidecar",
                source_path=f"/neutral/owner/tool-results/{filename}",
                reason="opaque",
            )
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": {"type": "object"}},
            privacy_policy="standard",
        )
        relationships = cast(JSONDocument, profile["relationships"])
        opaque = cast(JSONDocument, relationships["opaque_tool_results"])
        classes = cast(JSONDocument, opaque["reference_classes"])
        for kind, length in (("direct_inline", 3), ("persisted_truncated", 10_000_003), ("orphan", 0)):
            measured = cast(JSONDocument, classes[kind])
            sizes = cast(JSONDocument, measured["bytes"])
            counts = cast(JSONDocument, measured["count_per_scope"])
            assert sizes["count"] == 1 and sizes["min"] == sizes["max"] == length
            assert counts["count"] == 2 and counts["min"] == 0 and counts["max"] == 1
        per_result = cast(JSONDocument, opaque["per_tool_result"])
        for mode, kind in (("string:inline", "direct_inline"), ("string:truncated", "persisted_truncated")):
            structure = cast(JSONDocument, per_result[mode])
            file_counts = cast(JSONDocument, structure["file_count"])
            assert file_counts["count"] == 1 and file_counts["min"] == file_counts["max"] == 1
            assert structure["initial_classes"] == structure["terminal_classes"] == {kind: 1}
            assert structure["transitions"] == []
        assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()
        assert "pointer.txt" not in b"".join(iter_workload_profile_chunks(profile)).decode()


def test_original_opaque_result_paths_keep_empty_and_multiple_reference_files(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal

    records: list[JSONDocument] = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "private-one",
                        "content": "Full output saved to: /neutral/a.txt",
                    },
                    {"type": "tool_result", "tool_use_id": "private-empty", "content": None},
                    {"type": "tool_result", "tool_use_id": "private-absent"},
                ],
            },
        }
    ]
    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        uid = journal.append_unit(
            SchemaUnit(
                cluster_payload={}, schema_samples=records, artifact_kind="session_record_stream", bundle_scope="owner"
            )
        )
        journal.assign_profile_family(uid, "family")
        for raw_id, filename in (("a", "a.txt"), ("b", "private-one.txt")):
            journal.record_payload(
                raw_id=raw_id, artifact_kind="tool_result_sidecar", bundle_scope="owner", byte_count=5
            )
            journal.record_terminal(
                raw_id=raw_id,
                status="intentionally_excluded",
                artifact_kind="tool_result_sidecar",
                source_path=f"/neutral/owner/tool-results/{filename}",
                reason="opaque",
            )
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": {"type": "object"}},
            privacy_policy="standard",
        )
        opaque = cast(JSONDocument, cast(JSONDocument, profile["relationships"])["opaque_tool_results"])
        paths = cast(JSONDocument, opaque["per_tool_result"])
        files = cast(JSONDocument, paths["string:truncated"])
        count = cast(JSONDocument, files["file_count"])
        assert count["count"] == 1 and count["min"] == count["max"] == 2
        inline = cast(JSONDocument, files["inline_length"])
        assert inline["min"] == inline["max"] == len("Full output saved to: /neutral/a.txt")
        assert files["initial_classes"] == {"persisted_truncated": 1}
        assert files["terminal_classes"] == {"direct_truncated": 1}
        assert files["transitions"] == [{"from": "persisted_truncated", "to": "direct_truncated", "count": 1}]
        for mode in ("null:inline", "absent:inline"):
            empty = cast(JSONDocument, paths[mode])
            count = cast(JSONDocument, empty["file_count"])
            assert count["count"] == 1 and count["min"] == count["max"] == 0
            assert empty["initial_classes"] == empty["terminal_classes"] == {} and empty["transitions"] == []
            inline = cast(JSONDocument, empty["inline_length"])
            assert inline["min"] == inline["max"] == 0
        assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()


def test_opaque_reference_paths_use_original_native_grammar_and_refuse_unmeasured_edges() -> None:
    from polylogue.schemas.synthetic.record_profiles import OpaqueReferenceKinds
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    profile: JSONDocument = {
        "initial_classes": {"persisted_truncated": 1},
        "terminal_classes": {"direct_truncated": 1},
        "transitions": [{"from": "persisted_truncated", "to": "direct_truncated", "count": 1}],
    }
    owner = ForeignKeyGraph()
    try:
        path = OpaqueReferenceKinds("string:truncated", profile, 2, owner)
        assert list(path.sample(random.Random(1))) == ["persisted_truncated", "direct_truncated"]
        conditional = OpaqueReferenceKinds("string:truncated", profile, 2, owner, presence="direct+persisted")
        assert list(conditional.sample(random.Random(1))) == ["persisted_truncated", "direct_truncated"]
        with pytest.raises(ValueError):
            list(
                OpaqueReferenceKinds("string:truncated", profile, 2, owner, presence="direct").sample(random.Random(1))
            )
        empty = OpaqueReferenceKinds(
            "null:inline", {"initial_classes": {}, "terminal_classes": {}, "transitions": []}, 0, owner
        )
        assert list(empty.sample(random.Random(1))) == []
        with pytest.raises(ValueError):
            OpaqueReferenceKinds("string:inline", profile, 2, owner)
        with pytest.raises(ValueError):
            list(OpaqueReferenceKinds("string:truncated", profile, 3, owner).sample(random.Random(1)))
    finally:
        owner.close()
    with pytest.raises(RuntimeError):
        list(path.sample(random.Random(1)))


def test_original_reference_grammar_preserves_prior_pointer_for_repeated_result_id() -> None:
    from polylogue.schemas.synthetic.record_profiles import OpaqueReferenceKinds
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph
    from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

    with ToolResultIndexAccumulator(disk_backed=True) as index:
        for content in ("Full output saved to: /neutral/file.txt", None):
            index.observe({"message": {"content": [{"type": "tool_result", "tool_use_id": "one", "content": content}]}})
        references = index.iter_tool_result_references()
        try:
            reference = next(references)
        finally:
            references.close()
        assert reference.content_representation == "null" and reference.inline_length == 0
        match = index.reference_for_filename("file.txt")
        assert match is not None and match.reference_kind == "persisted" and not match.was_truncated
        owner = ForeignKeyGraph()
        try:
            path = OpaqueReferenceKinds(
                "null:inline",
                {
                    "initial_classes": {"persisted_inline": 1},
                    "terminal_classes": {"persisted_inline": 1},
                    "transitions": [],
                },
                1,
                owner,
            )
            assert list(path.sample(random.Random(1))) == ["persisted_inline"]
        finally:
            owner.close()


def test_original_result_history_conditions_prior_pointers_and_same_record_order(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.synthetic.record_profiles import OpaqueResultHistoryKinds
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    with ObservationJournal.create(root=tmp_path / "journal") as journal:
        # Both histories end at null, but only the first owns a preceding
        # saved-to pointer. Several blocks share one record position; original
        # native insertion order must retain their actual predecessor relation.
        for scope, contents in (("first", ["Full output saved to: /neutral/prior.txt", None]), ("second", [None])):
            uid = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    artifact_kind="session_record_stream",
                    bundle_scope=scope,
                    schema_samples=[
                        {
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": "private-original", "content": content}
                                    for content in contents
                                ],
                            },
                        }
                    ],
                )
            )
            journal.assign_profile_family(uid, "family")
            if scope == "first":
                journal.record_payload(
                    raw_id="opaque", artifact_kind="tool_result_sidecar", bundle_scope=scope, byte_count=9
                )
                journal.record_terminal(
                    raw_id="opaque",
                    status="intentionally_excluded",
                    artifact_kind="tool_result_sidecar",
                    source_path="/neutral/first/tool-results/prior.txt",
                    reason="opaque",
                )
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            privacy_policy="standard",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": {"type": "object"}},
        )
        opaque = cast(JSONDocument, cast(JSONDocument, profile["relationships"])["opaque_tool_results"])
        per_result = cast(JSONDocument, opaque["per_tool_result"])
        mode = cast(JSONDocument, per_result["null:inline"])
        histories = cast(JSONDocument, mode["result_history"])
        assert set(histories) == {"none", "persisted"}
        history = cast(JSONDocument, histories["persisted"])
        counts = cast(JSONDocument, history["record_count"])
        assert counts["count"] == 1 and counts["min"] == counts["max"] == 2
        file_counts = cast(JSONDocument, history["file_count_by_identity_history_and_pointers"])
        assert len(file_counts) == 1
        file_count = cast(JSONDocument, next(iter(file_counts.values())))
        assert file_count["min"] == file_count["max"] == 1
        assert history["initial_modes"] == {"string:truncated": 1}
        assert history["terminal_modes"] == {"null:inline": 1}
        assert history["transitions"] == [{"from": "string:truncated", "to": "null:inline", "count": 1}]
        assert mode["initial_classes"] == mode["terminal_classes"] == {"persisted_inline": 1}
        assert (
            "private-original" not in b"".join(iter_workload_profile_chunks(profile)).decode()
            and "prior.txt" not in b"".join(iter_workload_profile_chunks(profile)).decode()
        )
        owner = ForeignKeyGraph()
        try:
            path = OpaqueResultHistoryKinds("null:inline", history, 2, owner)
            assert list(path.sample(random.Random(2))) == ["string:truncated", "null:inline"]
            empty_files = cast(JSONDocument, histories["none"])
            one = OpaqueResultHistoryKinds("null:inline", empty_files, 1, owner)
            assert list(one.sample(random.Random(2))) == ["null:inline"]
            with pytest.raises(ValueError):
                list(OpaqueResultHistoryKinds("null:inline", history, 1, owner).sample(random.Random(2)))
            with pytest.raises(ValueError):
                OpaqueResultHistoryKinds("string:truncated", history, 2, owner)
            with pytest.raises(ValueError):
                list(OpaqueResultHistoryKinds("null:inline", history, 3, owner).sample(random.Random(2)))
        finally:
            owner.close()
        with pytest.raises(RuntimeError):
            list(path.sample(random.Random(2)))


def test_original_result_shapes_bind_before_draw_and_preserve_prior_reference() -> None:
    from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["user"]},
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["user"]},
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
                                "tool_use_id": {"type": "string", "x-polylogue-values": ["neutraltool"]},
                                "content": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
    }
    sample: JSONDocument = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "neutraltool",
                    "content": "Full output saved to: /neutral/prior.txt\nneutral body",
                },
                {"type": "tool_result", "tool_use_id": "neutraltool", "content": None},
            ],
        },
    }
    annotate_schema(schema, _collect_field_stats([sample]))
    profile = inferred_record_profile("claude-code", schema, [sample], element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    for seed in range(8):
        sessions = corpus.iter_sessions(count=1, seed=seed)
        try:
            session = next(sessions)
            record = cast(JSONDocument, json.loads(b"".join(session.chunks)))
            blocks = cast(list[JSONDocument], cast(JSONDocument, record["message"])["content"])
            assert len(blocks) == 2
            assert isinstance(blocks[0]["content"], str)
            assert blocks[1]["content"] is None
            assert blocks[0]["tool_use_id"] == blocks[1]["tool_use_id"] == "neutraltool"
            with ToolResultIndexAccumulator(disk_backed=True) as observed:
                observed.observe(record)
                # The generated original pointer field owns the filename; a
                # later null field remains null while preserving its history.
                from polylogue.core.tool_result_pointer import PERSISTED_OUTPUT_POINTER_PATTERN

                pointer = PERSISTED_OUTPUT_POINTER_PATTERN.search(blocks[0]["content"])
                assert pointer is not None
                match = observed.reference_for_filename(Path(pointer.group(1)).name)
                assert match is not None and match.reference_kind == "persisted" and not match.was_truncated
            with pytest.raises(StopIteration):
                next(sessions)
        finally:
            sessions.close()


def test_physical_streams_keep_bundle_children_out_of_the_parent_timeline(tmp_path: Path) -> None:
    """Joining files by bundle invents a parent transition and timestamp gap."""
    from polylogue.core.json import json_document
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.workload_profiles import PhysicalScopeCounts

    with ObservationJournal.create(root=tmp_path / "physical-journal") as journal:
        records = (
            ("session_record_stream", "private-root", "private-root.jsonl", "user", "2025-01-01T00:00:00Z"),
            (
                "agent_transcript",
                "private-agent",
                "private-root/subagents/agent-private.jsonl",
                "assistant",
                "2025-01-01T00:00:10Z",
            ),
            ("session_record_stream", "private-root", "private-root.jsonl", "assistant", "2025-01-01T00:00:01Z"),
        )
        for kind, raw_id, source_path, role, timestamp in records:
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=[
                        {
                            "type": role,
                            "timestamp": timestamp,
                            "message": {"role": role, "content": [{"type": "text", "text": "neutral"}]},
                        }
                    ],
                    artifact_kind=kind,
                    raw_id=raw_id,
                    source_path=source_path,
                    bundle_scope="private-bundle",
                )
            )
            journal.assign_profile_family(unit, "neutral-physical-family")
        memberships = journal.memberships()
        profile = build_package_workload_profile(
            provider="claude-code",
            version="neutral",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="neutral-physical-family",
                anchor_kind="session_record_stream",
                memberships=memberships,
            ),
            element_schemas={"session_record_stream": {"type": "object"}, "agent_transcript": {"type": "object"}},
            privacy_policy="standard",
        )
        files = list(memberships.iter_scope_file_counts())
        assert files == [("private-bundle", 0, 1, 1, 0, 0, 0)]
        counts = PhysicalScopeCounts.from_payload(json_document(profile["physical_scopes"])["count_joint"])
        assert tuple(counts.bins.values()) == (((0, 1, 1, 0, 0, 0), 1),)
        elements = json_document(profile["elements"])
        parent = json_document(json_document(elements["session_record_stream"])["record_structure"])
        child = json_document(json_document(elements["agent_transcript"])["record_structure"])
        assert json_document(parent["scope_record_count"])["max"] == 2
        assert json_document(child["scope_record_count"])["max"] == 1
        assert parent["transitions"] == [{"from": "user_text", "to": "assistant_text", "count": 1}]
        assert json_document(parent["adjacent_timestamp_gap_seconds"])["min"] == 1
        assert json_document(parent["adjacent_timestamp_gap_seconds"])["max"] == 1
        encoded = b"".join(iter_workload_profile_chunks(profile)).decode()
        assert "private-root" not in encoded
        assert "private-agent" not in encoded
        assert "private-bundle" not in encoded


def test_selection_enrolls_each_declared_element_from_the_same_original_package(tmp_path: Path) -> None:
    """A main-only selection loses child field constraints before their draws."""
    from polylogue.schemas.synthetic.selection import select_synthetic_schema

    registry = SchemaRegistry(storage_root=tmp_path / "selected-schemas")
    schemas: dict[str, JSONDocument] = {
        "session_record_stream": {"type": "object", "properties": {"root_operand": {"type": "integer"}}},
        "agent_transcript": {"type": "object", "properties": {"child_operand": {"type": "boolean"}}},
    }
    package = SchemaVersionPackage(
        provider="claude-code",
        version="neutral-v1",
        anchor_kind="session_record_stream",
        default_element_kind="session_record_stream",
        first_seen="2025-01-01T00:00:00Z",
        last_seen="2025-01-01T00:00:01Z",
        bundle_scope_count=1,
        sample_count=2,
        workload_profile_file="workload-profile.json.gz",
        elements=[
            SchemaElementManifest(
                element_kind=kind, schema_file=f"{kind}.schema.json.gz", sample_count=1, artifact_count=1
            )
            for kind in schemas
        ],
    )
    catalog = SchemaPackageCatalog(
        provider="claude-code",
        packages=[package],
        latest_version="neutral-v1",
        default_version="neutral-v1",
        recommended_version="neutral-v1",
    )
    profile: JSONDocument = {"elements": {kind: {"sample_count": 1} for kind in schemas}}
    registry.replace_provider_packages(
        "claude-code", catalog, {"neutral-v1": schemas}, package_workload_profiles={"neutral-v1": profile}
    )
    selection = select_synthetic_schema("claude-code", registry_factory=lambda: registry)
    assert selection.package_version == "neutral-v1"
    assert set(selection.element_schemas) == set(schemas)
    for kind in schemas:
        assert selection.element_schemas[kind] == registry.get_element_schema(
            "claude-code", version="neutral-v1", element_kind=kind
        )
        assert selection.element_schemas[kind]["properties"] == schemas[kind]["properties"]
    assert selection.schema is selection.element_schemas["session_record_stream"]
    corpus = SyntheticCorpus.from_selection(selection)
    assert corpus.element_schemas == selection.element_schemas
    assert corpus.element_schemas["agent_transcript"]["properties"] == {"child_operand": {"type": "boolean"}}


def test_original_session_headers_prove_exact_prefix_and_preserve_ambiguous_parents(tmp_path: Path) -> None:
    from polylogue.core.json import json_document
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.workload_profiles import SessionPrefixCounts

    first: JSONDocument = {
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": "neutral first"},
    }
    second: JSONDocument = {
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": "neutral second"},
    }
    tail: JSONDocument = {
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": "neutral tail"},
    }
    streams: tuple[tuple[str, list[JSONDocument]], ...] = (
        ("parent", [{"type": "session_meta", "payload": {"id": "private-parent"}}, first, second]),
        (
            "fork",
            [
                {"type": "session_meta", "payload": {"id": "private-child", "forked_from_id": "private-parent"}},
                first,
                second,
                tail,
            ],
        ),
        (
            "later-meta",
            [
                {"type": "session_meta", "payload": {"id": "private-root"}},
                {"type": "session_meta", "payload": {"id": "private-replayed", "forked_from_id": "private-parent"}},
                tail,
            ],
        ),
        (
            "outside",
            [
                {"type": "session_meta", "payload": {"id": "private-outside", "parent_thread_id": "private-absent"}},
                tail,
            ],
        ),
        ("duplicate-a", [{"type": "session_meta", "payload": {"id": "private-duplicated"}}, first]),
        ("duplicate-b", [{"type": "session_meta", "payload": {"id": "private-duplicated"}}, second]),
        (
            "ambiguous",
            [
                {
                    "type": "session_meta",
                    "payload": {"id": "private-ambiguous", "forked_from_id": "private-duplicated"},
                },
                first,
            ],
        ),
        (
            "spawn",
            [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "private-spawn",
                        "source": {"subagent": {"thread_spawn": {"parent_thread_id": "private-parent"}}},
                    },
                },
                tail,
            ],
        ),
    )
    with ObservationJournal.create(root=tmp_path / "lineage-journal") as journal:
        for key, records in streams:
            unit_id = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=records,
                    artifact_kind="session_record_stream",
                    raw_id="private-raw-" + key,
                    bundle_scope="private-bundle",
                )
            )
            journal.assign_profile_family(unit_id, "family")
        # Metadata sharing a bundle and header-shaped fields is not a
        # transcript parent, even when its private id happens to match.
        metadata = journal.append_unit(
            SchemaUnit(
                cluster_payload={},
                schema_samples=[{"type": "session_meta", "payload": {"id": "private-parent"}}],
                artifact_kind="agent_sidecar_meta",
                raw_id="private-metadata",
                bundle_scope="private-bundle",
            )
        )
        journal.assign_profile_family(metadata, "family")
        journal._connection.execute("SAVEPOINT original_lineage_window").close()
        profile = build_package_workload_profile(
            provider="codex",
            version="v1",
            package=_PackageAccumulator(
                provider="codex",
                anchor_family_id="family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": {"type": "object"}},
            privacy_policy="standard",
        )
        lineage = json_document(profile["session_lineage"])
        assert lineage["outcomes"] == {
            "no_explicit_parent": 4,
            "matched:prefix:forked_from_id": 1,
            "matched:spawn:thread_spawn.parent_thread_id": 1,
            "unretained_parent": 1,
            "ambiguous_parent": 1,
        }
        classes = lineage["header_classes"]
        assert isinstance(classes, list)
        fork_header = next(
            row
            for row in classes
            if isinstance(row, dict) and row["outcome"] == "matched" and row["carrier"] == "prefix:forked_from_id"
        )
        assert fork_header["form"] == "codex-envelope"
        assert fork_header["fields"] == {"forked_from_id": "string", "parent_thread_id": "absent", "source": "absent"}
        spawn_header = next(
            row
            for row in classes
            if isinstance(row, dict)
            and row["outcome"] == "matched"
            and row["carrier"] == "spawn:thread_spawn.parent_thread_id"
        )
        assert spawn_header["fields"] == {
            "forked_from_id": "absent",
            "parent_thread_id": "absent",
            "source": "object",
            "source.subagent": "object",
            "source.subagent.thread_spawn": "object",
            "source.subagent.thread_spawn.parent_thread_id": "string",
        }
        joints = json_document(lineage["matched_prefix_joints"])
        fork = SessionPrefixCounts.from_payload(joints["prefix:forked_from_id"])
        spawn = SessionPrefixCounts.from_payload(joints["spawn:thread_spawn.parent_thread_id"])
        assert {point for rows in fork.extrema.values() for point in rows} == {(4, 3, 3, 2, 2)}
        assert {point for rows in spawn.extrema.values() for point in rows} == {(2, 3, 1, 2, 0)}
        structures = json_document(lineage["record_structures"])
        fork_structure = json_document(json_document(structures["prefix:forked_from_id"])["codex-envelope"])
        fork_counts = json_document(fork_structure["scope_record_count"])
        # This carrier also has the original ambiguous-parent stream. Its
        # own two-record grammar remains visible, not rewritten as the fork.
        assert fork_counts["count"] == 2 and fork_counts["min"] == 2 and fork_counts["max"] == 4
        spawn_counts = json_document(
            json_document(json_document(structures["spawn:thread_spawn.parent_thread_id"])["codex-envelope"])[
                "scope_record_count"
            ]
        )
        assert spawn_counts["count"] == 1 and spawn_counts["min"] == spawn_counts["max"] == 2
        assert "private-" not in b"".join(iter_workload_profile_chunks(profile)).decode()
        assert not journal._connection.execute(
            "SELECT name FROM sqlite_temp_master WHERE name LIKE 'workload_session_%'"
        ).fetchall()
        journal._connection.execute("ROLLBACK TO original_lineage_window").close()
        journal._connection.execute("RELEASE original_lineage_window").close()


@pytest.mark.parametrize("header_type", ["absent", "null", "empty"])
def test_flat_codex_header_and_role_message_keep_the_original_wire_carriers(header_type: str) -> None:
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.record_structure import codex_session_header
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS

    header: JSONDocument = {"id": "00000000-0000-4000-8000-000000000001", "timestamp": "2025-06-01T00:00:00Z"}
    if header_type != "absent":
        header["type"] = None if header_type == "null" else ""
    message: JSONDocument = {"role": "user", "content": [{"type": "input_text", "text": "neutral message"}]}
    samples = [header, message]
    # The declared neutral branches use the original inference producer;
    # neither header ID nor message body is pasted after unrelated draws.
    schema: JSONDocument = {
        "oneOf": [generate_schema_from_samples([record], max_stats_samples=0, provider="codex") for record in samples]
    }
    profile = inferred_record_profile("codex", schema, samples, element="session_record_stream")
    selected = SyntheticSchemaSelection(
        "codex",
        "neutral-flat",
        "session_record_stream",
        schema,
        PROVIDER_WIRE_FORMATS["codex"],
        workload_profile=profile,
    )
    for seed in (0, 19, 71):
        corpus = SyntheticCorpus.from_selection(selected)
        result = corpus.generate_batch(count=1, messages_per_session=range(2, 3), seed=seed)
        records = [json.loads(line) for line in result.raw_items[0].splitlines()]
        assert len(records) == 2
        assert codex_session_header(records[0]) == records[0]
        assert "payload" not in records[0]
        assert ("type" in records[0]) == (header_type != "absent")
        if header_type != "absent":
            assert records[0]["type"] == header["type"]
        assert "type" not in records[1] and records[1]["role"] == "user"
        assert "id" not in records[1] and "timestamp" not in records[1]
        assert records[1]["content"] and records[1]["content"][0]["type"] == "input_text"
        parsed = parse_payload("codex", records, "neutral-flat", source_path="neutral-flat.jsonl")
        assert len(parsed) == 1 and len(parsed[0].messages) == 1
        assert parsed[0].provider_session_id == records[0]["id"]


def test_first_header_parent_binding_filters_original_branches_before_draws() -> None:
    import random

    from polylogue.schemas.synthetic.models import GeneratedRecord, SchemaRecord
    from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph
    from polylogue.schemas.synthetic.wire_formats import WireFormat

    fields: SchemaRecord = {"forked_from_id": "absent", "parent_thread_id": "string", "source": "null"}
    header_class: SchemaRecord = {"form": "codex-envelope", "carrier": "prefix:parent_thread_id", "fields": fields}
    original_leaf = annotate_schema(
        {"type": "object", "properties": {"identity": {"type": "string"}}},
        _collect_field_stats([{"identity": "child1"}, {"identity": "prior1"}]),
    )
    leaf = cast(SchemaRecord, cast(JSONDocument, original_leaf["properties"])["identity"])
    wrong: SchemaRecord = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "payload": {"type": "object", "properties": {"id": leaf, "source": {"type": "object"}}},
        },
    }
    good: SchemaRecord = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "payload": {
                "type": "object",
                "properties": {"id": leaf, "parent_thread_id": leaf, "source": {"type": "null"}},
            },
        },
    }
    schema: SchemaRecord = {"oneOf": [wrong, good]}
    structure: SchemaRecord = {
        "record_forms_by_kind": {"session_meta": {"codex-envelope": 1}},
        "content_blocks_by_record_kind": {
            "session_meta": {"codex-envelope": {"absent": {"representations": {"absent": 1}}}}
        },
    }
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        plan = PublicRecordPlan(
            "codex", structure, "session_meta", random.Random(1), owner, schema, header_class=header_class
        )
        plan.bind_session_header(header_class, native_id="child1", parent_id="prior1")
        assert not plan.compatible(wrong, "$")
        assert plan.compatible(good, "$")
        corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "codex")
        corpus._public_record_plan = plan
        # Use the same plan owner as the actual field walker; no detached
        # plain mapping substitutes for generated native field custody.
        corpus._relation_solver.fk_graph = owner
        result = corpus._generate_from_schema(schema, random.Random(2))
        assert isinstance(result, GeneratedRecord)
        payload = result["payload"]
        assert isinstance(payload, GeneratedRecord)
        assert payload["parent_thread_id"] == "prior1" and payload["id"] == "child1"
        assert payload["source"] is None and "forked_from_id" not in payload
        with pytest.raises(ValueError, match="schema branch"):
            incompatible = PublicRecordPlan(
                "codex", structure, "session_meta", random.Random(1), owner, wrong, header_class=header_class
            )
            incompatible.bind_session_header(header_class, native_id="child1", parent_id="prior1")
        with pytest.raises(ValueError, match="schema branch"):
            unsupported = PublicRecordPlan(
                "codex", structure, "session_meta", random.Random(1), owner, good, header_class=header_class
            )
            unsupported.bind_session_header(header_class, native_id="child1", parent_id="too-wide-parent")
    finally:
        owner.close()


def test_headerless_codex_scope_uses_original_counts_without_inventing_header(tmp_path: Path) -> None:
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS

    records: list[JSONDocument] = [
        {"type": "message", "role": "user", "content": "neutral user"},
        {"type": "message", "role": "assistant", "content": "neutral assistant"},
    ]
    schema = generate_schema_from_samples(records, max_stats_samples=0, provider="codex")
    with ObservationJournal.create(root=tmp_path / "headerless") as journal:
        unit = journal.append_unit(
            SchemaUnit(
                cluster_payload={},
                schema_samples=records,
                artifact_kind="session_record_stream",
                raw_id="neutral-raw",
                bundle_scope="neutral-bundle",
            )
        )
        journal.assign_profile_family(unit, "neutral")
        profile = build_package_workload_profile(
            provider="codex",
            version="neutral",
            package=_PackageAccumulator(
                provider="codex",
                anchor_family_id="neutral",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": schema},
            privacy_policy="standard",
        )
    selection = SyntheticSchemaSelection(
        "codex", "neutral", "session_record_stream", schema, PROVIDER_WIRE_FORMATS["codex"], workload_profile=profile
    )
    corpus = SyntheticCorpus.from_selection(selection)
    batch = corpus.generate_batch(count=1, messages_per_session=range(17, 18), seed=83)
    generated = [json.loads(line) for line in batch.artifacts[0].raw_bytes.splitlines()]
    assert len(generated) == 2
    assert [record["role"] for record in generated] == ["user", "assistant"]
    assert all(record["type"] == "message" and "payload" not in record for record in generated)
    assert batch.artifacts[0].facts.native_session_id is None
    assert batch.artifacts[0].message_count == 2
    with pytest.raises(ValueError):
        corpus.generate_batch(count=1, seed=83, session_native_ids=("undeclared",))


def test_blank_original_parent_is_refused_before_it_can_become_a_reference() -> None:
    from polylogue.schemas.record_structure import codex_parent_field_forms
    from polylogue.schemas.synthetic.models import SchemaRecord
    from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    header_class: SchemaRecord = {
        "form": "codex-envelope",
        "carrier": "no_explicit_parent",
        "fields": cast(SchemaRecord, codex_parent_field_forms({"forked_from_id": " \t"})),
    }
    assert header_class["fields"] == {"forked_from_id": "blank", "parent_thread_id": "absent", "source": "absent"}
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "payload": {"type": "object", "properties": {"forked_from_id": {"type": "string"}}},
        },
    }
    structure: SchemaRecord = {
        "record_forms_by_kind": {"session_meta": {"codex-envelope": 1}},
        "content_blocks_by_record_kind": {
            "session_meta": {"codex-envelope": {"absent": {"representations": {"absent": 1}}}}
        },
    }
    owner = ForeignKeyGraph()
    owner._owner()
    try:
        plan = PublicRecordPlan("codex", structure, "session_meta", random.Random(1), owner, schema)
        with pytest.raises(ValueError, match="field class is unsupported"):
            plan.bind_session_header(header_class, native_id=None, parent_id=None)
    finally:
        owner.close()


def test_dispatcher_carrier_forms_constrain_original_schema_before_field_draws() -> None:
    from polylogue.schemas.record_structure import claude_dispatch_field_forms

    agent: JSONDocument = {"type": "string", "x-polylogue-values": ["neutral-child"]}
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["user"]},
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "toolUseResult": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "object", "properties": {"agentId": agent}},
                ]
            },
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["user"]},
                    "content": {"type": "string", "x-polylogue-values": ["neutral body"]},
                },
            },
        },
    }
    samples: list[JSONDocument] = [
        {"type": "user", "message": {"role": "user", "content": "neutral body"}},
        {"type": "user", "toolUseResult": None, "message": {"role": "user", "content": "neutral body"}},
        {
            "type": "user",
            "toolUseResult": {"agentId": "neutral-child"},
            "message": {"role": "user", "content": "neutral body"},
        },
    ]
    profile = inferred_record_profile("claude-code", schema, samples, element="session_record_stream")
    corpus = SyntheticCorpus(
        schema,
        WireFormat(encoding="jsonl"),
        "claude-code",
        element_kind="session_record_stream",
        workload_profile=profile,
    )
    expected = {json.dumps(claude_dispatch_field_forms(record), sort_keys=True) for record in samples}
    seen = set()
    for raw in corpus.generate_batch(count=20, seed=7).raw_items:
        records = [json.loads(line) for line in raw.splitlines()]
        for record in records:
            form = json.dumps(claude_dispatch_field_forms(record), sort_keys=True)
            assert form in expected
            seen.add(form)
            if isinstance(record.get("toolUseResult"), dict):
                assert record["toolUseResult"]["agentId"] == "neutral-child"
        parsed = parse_payload("claude-code", records, "neutral-dispatch-fields")
        assert parsed and len(parsed[0].messages) == 3
    assert seen == expected
    incompatible = json.loads(json.dumps(schema))
    del incompatible["properties"]["toolUseResult"]
    with pytest.raises(ValueError, match="dispatcher form has no declared schema branch"):
        SyntheticCorpus(
            incompatible,
            WireFormat(encoding="jsonl"),
            "claude-code",
            element_kind="session_record_stream",
            workload_profile=profile,
        ).generate_batch(count=20, seed=7)


def test_original_dispatch_child_counts_keep_missing_and_ambiguous_associations(tmp_path: Path) -> None:
    from polylogue.core.json import json_document
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.workload_profiles import AgentChildCounts

    def dispatch(agent: str) -> JSONDocument:
        return {
            "type": "user",
            "toolUseResult": {"agentId": agent},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "private-tool", "content": "neutral"}],
            },
        }

    with ObservationJournal.create(root=tmp_path / "agent-journal") as journal:
        declarations: list[tuple[str, str, str, list[JSONDocument]]] = [
            (
                "session_record_stream",
                "private-root",
                "private-root.jsonl",
                [
                    dispatch("private-matched"),
                    dispatch("private-missing"),
                    dispatch("private-repeated"),
                    dispatch("private-repeated"),
                ],
            ),
            (
                "agent_transcript",
                "private-child",
                "private-root/subagents/agent-private-matched.jsonl",
                [
                    {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}},
                    dispatch("private-nested"),
                ],
            ),
            (
                "agent_sidecar_meta",
                "private-meta",
                "private-root/subagents/agent-private-matched.meta.json",
                [{"agentId": "private-matched"}],
            ),
            (
                "agent_transcript",
                "private-nested-child",
                "private-root/subagents/agent-private-nested.jsonl",
                [{"type": "assistant", "message": {"role": "assistant", "content": "neutral"}}],
            ),
            (
                "agent_transcript",
                "private-repeated-child",
                "private-root/subagents/agent-private-repeated.jsonl",
                [
                    {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}},
                ],
            ),
        ]
        for kind, raw_id, path, samples in declarations:
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=samples,
                    artifact_kind=kind,
                    raw_id=raw_id,
                    source_path=path,
                    bundle_scope="private-scope",
                )
            )
            journal.assign_profile_family(unit, "neutral-family")
        profile = build_package_workload_profile(
            provider="claude-code",
            version="neutral",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="neutral-family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={kind: {"type": "object"} for kind, _, _, _ in declarations},
            privacy_policy="standard",
        )
        elements = json_document(profile["elements"])
        structure = json_document(json_document(elements["session_record_stream"])["record_structure"])
        contents = json_document(
            json_document(json_document(structure["content_blocks_by_record_kind"])["user_tool_result"])[
                "claude-message"
            ]
        )
        forms = json_document(contents["array"])["dispatch_forms"]
        assert isinstance(forms, list)
        observed = {}
        for row in forms:
            assert isinstance(row, dict)
            counts = AgentChildCounts.from_payload(row["child_count_joint"])
            observed[row["children"]] = (row["count"], tuple(counts.bins.values()))
        assert observed == {
            "matched": (1, (((1, 2, 1, len("private-matched"), len("agent-private-matched")), 1),)),
            "missing_transcript": (1, (((0, 0, 0, len("private-missing"), len("agent-private-missing")), 1),)),
            "ambiguous": (2, (((1, 1, 0, len("private-repeated"), len("agent-private-repeated")), 2),)),
        }
        assert isinstance(profile, WorkloadProfile)
        by_depth = profile.depth_grammars
        assert isinstance(by_depth, DepthGrammars)
        assert set(by_depth) == {"1", "2"}
        level_one = json_document(by_depth["1"])
        level_two = json_document(by_depth["2"])
        assert json_document(level_one["scope_record_count"])["min"] == 2
        assert json_document(level_one["scope_record_count"])["max"] == 2
        assert level_one["transitions"] == [{"from": "assistant_text", "to": "user_tool_result", "count": 1}]
        assert json_document(level_two["scope_record_count"])["min"] == 1
        assert json_document(level_two["scope_record_count"])["max"] == 1
        assert level_two["record_kinds"] == {"assistant_text": 1}
        encoded = b"".join(iter_workload_profile_chunks(profile)).decode()
        assert all(
            value not in encoded for value in ("private-root", "private-tool", "private-matched", "private-scope")
        )
        assert not journal._connection.execute(
            "SELECT name FROM sqlite_temp_master WHERE name LIKE 'workload_agent_%'"
        ).fetchall()


def test_original_dispatch_transcript_is_emitted_by_collector_writer_and_fresh_build(tmp_path: Path) -> None:
    from tempfile import TemporaryDirectory

    from devtools.fresh_build_bench.corpus import corpus_from_synthetic, verify_manifest
    from polylogue.scenarios import CorpusSpec
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from tests.infra.schema_workload_profiles import inferred_physical_profile

    leaf: JSONDocument = {"type": "string", "x-polylogue-values": ["neutral"]}
    parent: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["user"]},
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "toolUseResult": {
                "type": "object",
                "properties": {"agentId": {"type": "string", "x-polylogue-values": ["neutral-a"]}},
            },
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "x-polylogue-values": ["user"]},
                    "content": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "x-polylogue-values": ["tool_result"]},
                                "tool_use_id": {"type": "string", "x-polylogue-values": ["neutral-tool"]},
                                "content": leaf,
                            },
                        },
                    },
                },
            },
        },
    }
    child: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "x-polylogue-values": ["assistant"]},
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
            "message": {
                "type": "object",
                "properties": {"role": {"type": "string", "x-polylogue-values": ["assistant"]}, "content": leaf},
            },
        },
    }
    schemas = {"session_record_stream": parent, "agent_transcript": child}
    units = [
        SchemaUnit(
            cluster_payload={},
            artifact_kind="session_record_stream",
            raw_id="neutral-parent",
            source_path="neutral-parent.jsonl",
            bundle_scope="neutral-agent-scope",
            schema_samples=[
                {
                    "type": "user",
                    "toolUseResult": {"agentId": "neutral-a"},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "neutral-tool", "content": "neutral"}],
                    },
                }
            ],
        ),
        SchemaUnit(
            cluster_payload={},
            artifact_kind="agent_transcript",
            raw_id="neutral-child",
            source_path="neutral-parent/subagents/agent-neutral-a.jsonl",
            bundle_scope="neutral-agent-scope",
            schema_samples=[
                {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}},
                {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}},
            ],
        ),
    ]
    profile = inferred_physical_profile("claude-code", schemas, units)
    selection = SyntheticSchemaSelection(
        "claude-code", "v1", "session_record_stream", parent, WireFormat(encoding="jsonl"), profile, schemas
    )
    corpus = SyntheticCorpus.from_selection(selection)
    collected = corpus.generate_batch(count=1, seed=3)
    artifact = collected.artifacts[0]
    assert len(artifact.adjuncts) == 1
    adjunct = artifact.adjuncts[0]
    assert (adjunct.name, adjunct.directory, adjunct.artifact_kind) == (
        "agent-neutral-a.jsonl",
        "subagents",
        "agent_transcript",
    )
    main_records = [json.loads(line) for line in artifact.raw_bytes.splitlines()]
    child_records = [json.loads(line) for line in adjunct.raw_bytes.splitlines()]
    assert len(main_records) == 1 and len(child_records) == 2
    assert all(record["type"] == "assistant" and record["message"]["content"] == "neutral" for record in child_records)
    parsed = parse_payload("claude-code", main_records, "neutral-parent", source_path="neutral-parent.jsonl")
    assert parsed and [
        event.payload["child_provider_id"]
        for event in parsed[0].session_events
        if event.event_type == "claude_delegation_progress"
    ] == ["agent-neutral-a"]
    parsed_child = parse_payload(
        "claude-code", child_records, "agent-neutral-a", source_path="neutral-parent/subagents/agent-neutral-a.jsonl"
    )
    assert parsed_child and len(parsed_child[0].messages) == 2
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=1, seed=3), tmp_path / "agent-writer", prefix="session"
    )
    assert len(written.files) == 2
    assert written.session_files[0].read_bytes() == artifact.raw_bytes
    assert (
        written.session_files[0].parent / written.session_files[0].stem / adjunct.directory / adjunct.name
    ).read_bytes() == adjunct.raw_bytes
    with TemporaryDirectory(prefix="neutral-agent-driver-", dir="/realm/tmp/work") as directory:
        root = Path(directory) / "corpus"
        manifest = corpus_from_synthetic(root, corpus=corpus, seed=3, count=1)
        verify_manifest(root, manifest)
        assert manifest["file_count"] == 2
        assert manifest["total_bytes"] == len(artifact.raw_bytes) + len(adjunct.raw_bytes)
        main = root / "home/.claude/projects/schema/session-000000.jsonl"
        assert main.read_bytes() == artifact.raw_bytes
        assert (main.parent / main.stem / "subagents" / adjunct.name).read_bytes() == adjunct.raw_bytes


@pytest.mark.parametrize(
    "metadata_sample,claim,transcript",
    [
        ({"agentId": "neutral-a"}, "neutral-a", True),
        ({"agentId": "neutral-a"}, "neutral-a", False),
        ({"agentId": None, "agent_id": "agent-neutral-a"}, "agent-neutral-a", True),
        ({"agentId": "", "sessionId": None}, "agent-neutral-a", True),
        ({"agentId": "neutral-b"}, "independent", True),
    ],
)
def test_original_metadata_alias_is_bound_before_draw_and_emitted_with_parent(
    tmp_path: Path, metadata_sample: JSONDocument, claim: str, transcript: bool
) -> None:
    from polylogue.scenarios import CorpusSpec
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from polylogue.sources.parsers.claude.orchestration import parse_claude_orchestration_artifact
    from tests.infra.schema_workload_profiles import inferred_physical_profile

    parent_sample: JSONDocument = {
        "type": "user",
        "toolUseResult": {"agentId": "neutral-a"},
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "neutral-tool", "content": "neutral"}],
        },
    }
    child_sample: JSONDocument = {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}}

    def schema_for(sample: JSONDocument) -> JSONDocument:
        # The existing inference owns every field distribution and branch.
        from polylogue.schemas.generation.schema_builder import generate_schema_from_samples

        return generate_schema_from_samples([sample], max_stats_samples=0, provider="claude-code")

    parent_schema, child_schema, meta_schema = (
        schema_for(parent_sample),
        schema_for(child_sample),
        schema_for(metadata_sample),
    )
    schemas = {
        "session_record_stream": parent_schema,
        "agent_transcript": child_schema,
        "agent_sidecar_meta": meta_schema,
    }
    units = [
        SchemaUnit(
            cluster_payload={},
            schema_samples=[sample],
            artifact_kind=kind,
            raw_id=kind,
            source_path=path,
            bundle_scope="neutral-metadata-scope",
        )
        for kind, path, sample in [
            ("session_record_stream", "neutral-parent.jsonl", parent_sample),
            ("agent_transcript", "neutral-parent/subagents/agent-neutral-a.jsonl", child_sample),
            ("agent_sidecar_meta", "neutral-parent/subagents/agent-neutral-a.meta.json", metadata_sample),
        ]
    ]
    if not transcript:
        units = [unit for unit in units if unit.artifact_kind != "agent_transcript"]
    profile = inferred_physical_profile("claude-code", schemas, units)
    selection = SyntheticSchemaSelection(
        "claude-code", "v1", "session_record_stream", parent_schema, WireFormat(encoding="jsonl"), profile, schemas
    )
    corpus = SyntheticCorpus.from_selection(selection)
    artifacts = corpus.generate_batch(count=1, seed=4).artifacts
    assert len(artifacts[0].adjuncts) == 1 + int(transcript)
    metadata = [file for file in artifacts[0].adjuncts if file.artifact_kind == "agent_sidecar_meta"]
    assert len(metadata) == 1
    from polylogue.core.claude_code_children import claude_agent_transcript_stem

    parent_record = json.loads(artifacts[0].raw_bytes)
    agent = parent_record["toolUseResult"]["agentId"]
    stem = claude_agent_transcript_stem(agent)
    assert metadata[0].name == stem + ".meta.json" and metadata[0].directory == "subagents"
    expected = dict(metadata_sample)
    if metadata_sample.get("agentId") == "neutral-a":
        expected["agentId"] = agent
        expected_claim = agent
    elif metadata_sample.get("agent_id") == "agent-neutral-a":
        expected["agent_id"] = stem
        expected_claim = stem
    elif claim == "independent":
        emitted = json.loads(metadata[0].raw_bytes)
        assert emitted["agentId"] not in {"", agent, stem}
        assert len(emitted["agentId"]) == len(cast(str, metadata_sample["agentId"]))
        expected["agentId"] = emitted["agentId"]
        expected_claim = emitted["agentId"]
    else:
        expected_claim = stem
    actual = json.loads(metadata[0].raw_bytes)
    assert actual == expected
    assert claim == (
        "independent"
        if metadata_sample.get("agentId") == "neutral-b"
        else "neutral-a"
        if metadata_sample.get("agentId") == "neutral-a"
        else "agent-neutral-a"
    )
    parsed = parse_claude_orchestration_artifact("neutral-parent/subagents/" + metadata[0].name, metadata[0].raw_bytes)
    assert parsed is not None and len(parsed.facts) == 1
    assert parsed.facts[0].agent_id == expected_claim
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=1, seed=4), tmp_path / "meta-writer", prefix="session"
    )
    assert len(written.files) == 2 + int(transcript)
    main = written.session_files[0]
    assert (main.parent / main.stem / "subagents" / metadata[0].name).read_bytes() == metadata[0].raw_bytes


def test_original_child_opaque_file_joins_all_siblings_before_serialization(tmp_path: Path) -> None:
    from devtools.fresh_build_bench.corpus import corpus_from_synthetic, verify_manifest
    from polylogue.scenarios import CorpusSpec
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

    parent: JSONDocument = {
        "type": "user",
        "toolUseResult": {"agentId": "neutral-a"},
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "parent-tool", "content": "neutral"}],
        },
    }
    child: JSONDocument = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "child-tool", "content": "neutral"}],
        },
    }
    schemas = {
        kind: generate_schema_from_samples([sample], max_stats_samples=0, provider="claude-code")
        for kind, sample in [("session_record_stream", parent), ("agent_transcript", child)]
    }
    with ObservationJournal.create(root=tmp_path / "child-opaque-journal") as journal:
        for kind, path, sample in [
            ("session_record_stream", "neutral-parent.jsonl", parent),
            ("agent_transcript", "neutral-parent/subagents/agent-neutral-a.jsonl", child),
        ]:
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=[sample],
                    artifact_kind=kind,
                    raw_id=kind,
                    source_path=path,
                    bundle_scope="neutral-sibling-scope",
                )
            )
            journal.assign_profile_family(unit, "neutral-family")
        journal.record_payload(
            raw_id="neutral-payload",
            artifact_kind="tool_result_sidecar",
            bundle_scope="neutral-sibling-scope",
            byte_count=32769,
        )
        journal.record_terminal(
            raw_id="neutral-payload",
            status="intentionally_excluded",
            artifact_kind="tool_result_sidecar",
            source_path="neutral-parent/tool-results/child-tool.txt",
            reason="opaque",
        )
        profile = build_package_workload_profile(
            provider="claude-code",
            version="v1",
            package=_PackageAccumulator(
                provider="claude-code",
                anchor_family_id="neutral-family",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas=schemas,
            privacy_policy="standard",
        )
    selection = SyntheticSchemaSelection(
        "claude-code",
        "v1",
        "session_record_stream",
        schemas["session_record_stream"],
        WireFormat(encoding="jsonl"),
        profile,
        schemas,
    )
    corpus = SyntheticCorpus.from_selection(selection)
    artifact = corpus.generate_batch(count=1, seed=7).artifacts[0]
    transcripts = [file for file in artifact.adjuncts if file.artifact_kind == "agent_transcript"]
    opaque = [file for file in artifact.adjuncts if file.artifact_kind == "opaque_payload"]
    assert len(transcripts) == 1 and len(opaque) == 1
    assert opaque[0].reference_kind == "direct_inline" and opaque[0].directory == "tool-results"
    assert opaque[0].raw_bytes == b"x" * 32769
    with ToolResultIndexAccumulator(disk_backed=True) as observed:
        for content in (artifact.raw_bytes, transcripts[0].raw_bytes):
            for line in content.splitlines():
                observed.observe(json.loads(line))
        match = observed.reference_for_filename(opaque[0].name)
        assert match is not None and match.reference_kind == "direct" and not match.was_truncated
        child_record = json.loads(transcripts[0].raw_bytes)
        assert match.tool_use_id == child_record["message"]["content"][0]["tool_use_id"]
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=1, seed=7), tmp_path / "child-files", prefix="session"
    )
    assert len(written.files) == 3
    main = written.session_files[0]
    for file in artifact.adjuncts:
        assert (main.parent / main.stem / file.directory / file.name).read_bytes() == file.raw_bytes
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="neutral-child-opaque-", dir="/realm/tmp/work") as directory:
        root = Path(directory) / "corpus"
        manifest = corpus_from_synthetic(root, corpus=corpus, seed=7, count=1)
        verify_manifest(root, manifest)
        assert manifest["file_count"] == 3
        assert manifest["total_bytes"] == len(artifact.raw_bytes) + sum(
            len(file.raw_bytes) for file in artifact.adjuncts
        )
        fresh_main = root / "home/.claude/projects/schema/session-000000.jsonl"
        for file in artifact.adjuncts:
            assert (fresh_main.parent / fresh_main.stem / file.directory / file.name).read_bytes() == file.raw_bytes


@pytest.mark.parametrize(
    "sample",
    [
        "neutral-name",
        "12345678-1234-4234-8234-123456789abc",
        "abc123",
        "https://neutral.test/item",
        "neutral@example.test",
        "text/plain",
        "bmV1dHJhbA==",
        "2025-06-01T00:00:00Z",
        "1720000000.123456",
        "Full output saved to: /neutral/result.txt\nneutral body",
    ],
)
def test_independent_metadata_conditions_original_string_format_before_spelling(sample: str) -> None:
    import random

    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.record_structure import claude_agent_metadata_field_forms
    from polylogue.schemas.synthetic.observed_values import existing_value_is_supported

    document: JSONDocument = {"agentId": sample}
    schema = generate_schema_from_samples([document], max_stats_samples=0, provider="claude-code")
    corpus = SyntheticCorpus(schema, WireFormat(encoding="jsonl"), "claude-code")
    metadata: JSONDocument = {
        "fields": cast(JSONDocument, claude_agent_metadata_field_forms(document)),
        "carrier": "agentId",
        "relation": "independent",
    }
    try:
        corpus._relation_solver.fk_graph.new_record(schema)
        generated = corpus._generate_agent_metadata(schema, metadata, sample, "agent-" + sample, random.Random(4))
        value = generated["agentId"]
        assert isinstance(value, str) and value not in {"", sample, "agent-" + sample}
        assert len(value) == len(sample) and value.count("\n") == sample.count("\n")
        assert existing_value_is_supported(
            cast(JSONDocument, cast(JSONDocument, schema["properties"])["agentId"]), value
        )
    finally:
        corpus._relation_solver.fk_graph.close()


def test_prefixed_dispatch_names_are_distinct_before_original_children_are_drawn() -> None:
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from tests.infra.schema_workload_profiles import inferred_physical_profile

    parents: list[JSONDocument] = [
        {
            "type": "user",
            "toolUseResult": {"agentId": "agent-neutral-" + suffix},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "neutral-tool-" + suffix, "content": "neutral"}],
            },
        }
        for suffix in ("a", "b")
    ]
    leaf: JSONDocument = {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}}
    schemas = {
        "session_record_stream": generate_schema_from_samples(parents, max_stats_samples=0, provider="claude-code"),
        "agent_transcript": generate_schema_from_samples([leaf], max_stats_samples=0, provider="claude-code"),
    }
    # This fixture declares neutral result IDs. Agent spellings remain solely
    # the inferred anonymous category and original prefix relation's draw.
    tool_schema = schemas["session_record_stream"]
    for field in ("properties", "message", "properties", "content", "items", "properties", "tool_use_id"):
        tool_schema = cast(JSONDocument, tool_schema[field])
    tool_schema["x-polylogue-values"] = ["neutral-tool-a", "neutral-tool-b"]
    units = [
        SchemaUnit(
            cluster_payload={},
            schema_samples=parents,
            artifact_kind="session_record_stream",
            raw_id="root",
            source_path="root.jsonl",
            bundle_scope="neutral-unique-children",
        )
    ]
    units.extend(
        SchemaUnit(
            cluster_payload={},
            schema_samples=[leaf],
            artifact_kind="agent_transcript",
            raw_id="child-" + suffix,
            source_path="root/subagents/agent-neutral-" + suffix + ".jsonl",
            bundle_scope="neutral-unique-children",
        )
        for suffix in ("a", "b")
    )
    profile = inferred_physical_profile("claude-code", schemas, units)
    selection = SyntheticSchemaSelection(
        "claude-code",
        "v1",
        "session_record_stream",
        schemas["session_record_stream"],
        WireFormat(encoding="jsonl"),
        profile,
        schemas,
    )
    corpus = SyntheticCorpus.from_selection(selection)
    artifact = corpus.generate_batch(count=1, seed=2).artifacts[0]
    records = [json.loads(line) for line in artifact.raw_bytes.splitlines()]
    identities = [record["toolUseResult"]["agentId"] for record in records]
    assert len(identities) == 2 and len(set(identities)) == 2
    assert all(identity.startswith("agent-") and len(identity) == len("agent-neutral-a") for identity in identities)
    assert {file.name for file in artifact.adjuncts} == {identity + ".jsonl" for identity in identities}
    assert len(artifact.adjuncts) == 2
    assert all(len(file.raw_bytes.splitlines()) == 1 for file in artifact.adjuncts)


def test_depth_grammar_carrier_round_trip_keeps_exact_original_deep_tree_and_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    from polylogue.core.digest import RECEIPT, canonical_bytes
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
    from polylogue.schemas.workload_profile_rows import read_workload_profile
    from tests.infra.schema_workload_profiles import inferred_physical_profile

    def dispatch(depth: int) -> JSONDocument:
        return {
            "type": "user",
            "toolUseResult": {"agentId": f"neutral-{depth:03d}"},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "neutral-tool", "content": "neutral"}],
            },
        }

    depth_count = 24
    leaf: JSONDocument = {"type": "assistant", "message": {"role": "assistant", "content": "neutral"}}
    samples = [dispatch(depth) for depth in range(depth_count)]
    schemas = {
        "session_record_stream": generate_schema_from_samples(samples[:1], max_stats_samples=0, provider="claude-code"),
        "agent_transcript": generate_schema_from_samples(
            [*samples[1:], leaf], max_stats_samples=0, provider="claude-code"
        ),
    }
    units = [
        SchemaUnit(
            cluster_payload={},
            schema_samples=[samples[0]],
            artifact_kind="session_record_stream",
            raw_id="neutral-root",
            source_path="neutral-root.jsonl",
            bundle_scope="neutral-tree",
        ),
        *[
            SchemaUnit(
                cluster_payload={},
                schema_samples=[samples[depth] if depth < depth_count else leaf],
                artifact_kind="agent_transcript",
                raw_id=f"neutral-child-{depth}",
                source_path=f"neutral-root/subagents/agent-neutral-{depth - 1:03d}.jsonl",
                bundle_scope="neutral-tree",
            )
            for depth in range(1, depth_count + 1)
        ],
    ]
    # This helper retires the original journal before returning the public tape.
    profile = inferred_physical_profile("claude-code", schemas, units)
    assert isinstance(profile, WorkloadProfile)
    assert profile.depth_grammars is not None
    assert len(profile.depth_grammars) == depth_count
    chunks = b"".join(iter_workload_profile_chunks(profile))
    materialized = json.loads(chunks)
    assert chunks == canonical_bytes(materialized, RECEIPT)
    assert set(materialized["elements"]["agent_transcript"]["record_structures_by_depth"]) == {
        str(depth) for depth in range(1, depth_count + 1)
    }
    with gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(chunks, mtime=0))) as retained:
        loaded = read_workload_profile(retained)
    assert loaded["profile_id"] == workload_profile_identity(loaded)
    assert loaded.depth_grammars is not None
    assert loaded.depth_grammars[str(depth_count)]["record_kinds"] == {"assistant_text": 1}
    package = SchemaVersionPackage(
        provider="claude-code",
        version="v1",
        anchor_kind="session_record_stream",
        default_element_kind="session_record_stream",
        first_seen="",
        last_seen="",
        bundle_scope_count=1,
        sample_count=depth_count + 1,
        workload_profile_file="workload-profile.json.gz",
        elements=[SchemaElementManifest(kind, kind + ".schema.json.gz", 1, 1) for kind in schemas],
    )
    registry = SchemaRegistry(storage_root=tmp_path / "published")
    registry.replace_provider_packages(
        "claude-code",
        SchemaPackageCatalog(provider="claude-code", packages=[package], default_version="v1"),
        {"v1": schemas},
        package_workload_profiles={"v1": profile},
    )
    before_failure = {
        path.relative_to(registry.storage_root): path.read_bytes()
        for path in registry.storage_root.rglob("*")
        if path.is_file()
    }
    original_write = gzip.GzipFile.write
    reached = []

    def fail_profile_write(writer: gzip.GzipFile, data: bytes) -> int:
        written = original_write(writer, data)
        if str(getattr(writer.fileobj, "name", "")).endswith("workload-profile.json.gz"):
            reached.append(len(data))
            raise OSError("neutral original profile publication fault")
        return written

    with monkeypatch.context() as fault:
        fault.setattr(gzip.GzipFile, "write", fail_profile_write)
        with pytest.raises(OSError):
            registry.replace_provider_packages(
                "claude-code",
                SchemaPackageCatalog(provider="claude-code", packages=[package], default_version="v1"),
                {"v1": schemas},
                package_workload_profiles={"v1": profile},
            )
    assert reached
    assert before_failure == {
        path.relative_to(registry.storage_root): path.read_bytes()
        for path in registry.storage_root.rglob("*")
        if path.is_file()
    }
    assert profile.depth_grammars[str(depth_count)]["record_kinds"] == {"assistant_text": 1}
    assert profile["profile_id"] == workload_profile_identity(profile)
    registry.replace_provider_packages(
        "claude-code",
        SchemaPackageCatalog(provider="claude-code", packages=[package], default_version="v1"),
        {"v1": schemas},
        package_workload_profiles={"v1": profile},
    )
    published = registry.get_workload_profile("claude-code", "v1")
    assert isinstance(published, WorkloadProfile)
    loaded = published
    assert isinstance(loaded, WorkloadProfile)
    assert loaded.depth_grammars is not None
    assert loaded["profile_id"] == workload_profile_identity(loaded)
    assert len(loaded.depth_grammars) == depth_count
    committed = registry.load_committed_version_document("claude-code", "v1", "workload-profile.json.gz")
    assert isinstance(committed, WorkloadProfile) and committed.depth_grammars is not None
    assert committed.depth_grammars[str(depth_count)] == loaded.depth_grammars[str(depth_count)]
    # The fixture envelope follows the same profile codec, with its existing
    # exact-reference hash encoding; no nested dict copy can drop child rows.
    from polylogue.schemas.workload_profile_rows import read_workload_profile_document

    envelope = {"entries": [{"workload_profile": loaded}], "neutral_label": "ñ"}
    carried = b"".join(iter_workload_profile_chunks(envelope, normalize_unicode=False))
    decoded = read_workload_profile_document(io.BytesIO(carried))
    assert isinstance(decoded, dict) and decoded["neutral_label"] == "ñ"
    carried_profile = decoded["entries"][0]["workload_profile"]
    assert isinstance(carried_profile, WorkloadProfile) and carried_profile.depth_grammars is not None
    assert len(carried_profile.depth_grammars) == depth_count
    assert carried == b"".join(iter_workload_profile_chunks(decoded, normalize_unicode=False))
    assert "record_structures_by_depth" not in cast(
        JSONDocument, cast(JSONDocument, loaded["elements"])["agent_transcript"]
    )
    selection = SyntheticSchemaSelection(
        "claude-code",
        "v1",
        "session_record_stream",
        schemas["session_record_stream"],
        WireFormat(encoding="jsonl"),
        loaded,
        schemas,
    )
    corpus = SyntheticCorpus.from_selection(selection)
    first = corpus.generate_batch(count=1, seed=83).artifacts[0]
    second = corpus.generate_batch(count=1, seed=83).artifacts[0]
    assert len(first.adjuncts) == depth_count
    assert [(file.name, file.raw_bytes) for file in first.adjuncts] == [
        (file.name, file.raw_bytes) for file in second.adjuncts
    ]
    child_records = [json.loads(file.raw_bytes) for file in first.adjuncts]
    assert sum("toolUseResult" in record for record in child_records) == depth_count - 1
    assert child_records[-1]["type"] == "assistant" and "toolUseResult" not in child_records[-1]


def test_depth_grammar_loading_retains_rows_not_a_python_depth_dictionary() -> None:
    import io
    import tracemalloc

    from polylogue.schemas.workload_profile_rows import read_workload_profile

    rows = DepthGrammars()
    grammar: JSONDocument = {"record_kinds": {"assistant_text": 1}, "neutral_structure": "x" * 2048}
    for depth in range(1, 2049):
        rows.append(str(depth), grammar)
    profile = WorkloadProfile({"elements": {"agent_transcript": {}}, "profile_version": 1}, rows)
    encoded = gzip.compress(b"".join(iter_workload_profile_chunks(profile)), mtime=0)
    tracemalloc.start()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(encoded)) as captured:
            loaded = read_workload_profile(captured)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert loaded.depth_grammars is not None and len(loaded.depth_grammars) == 2048
    assert loaded.depth_grammars["2048"] == grammar
    # A whole decoded dictionary would retain over four MiB of string bodies.
    assert peak < 2 * 1024 * 1024


def test_depth_profile_stream_retains_native_rows_until_cancelled_last_borrower() -> None:
    import gc
    import sqlite3
    import weakref
    from contextlib import closing

    rows = DepthGrammars()
    rows.append("24", {"record_kinds": {"assistant_text": 1}})
    native = rows._connection
    with closing(native.execute("PRAGMA database_list")) as address:
        path = Path(address.fetchone()[2])
    native.execute("SELECT 1").close()
    retained = weakref.ref(rows)
    profile = WorkloadProfile({"elements": {"agent_transcript": {}}, "profile_version": 1}, rows)
    with pytest.raises(TypeError):
        json.dumps(profile)
    stream = iter_workload_profile_chunks(profile)
    assert next(stream) == b"{"
    del profile, rows
    gc.collect()
    assert retained() is not None and path.exists()
    stream.close()
    gc.collect()
    assert retained() is None and not path.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        native.execute("SELECT 1")


def test_wire_memo_identity_includes_original_depth_rows() -> None:
    from tests.infra.wire_support import _content_digest

    first, second = DepthGrammars(), DepthGrammars()
    first.append("24", {"record_kinds": {"assistant_text": 1}})
    second.append("24", {"record_kinds": {"tool_result": 1}})
    fields = {"elements": {"agent_transcript": {}}, "profile_version": 1}
    assert _content_digest(WorkloadProfile(fields, first)) != _content_digest(WorkloadProfile(fields, second))


def test_depth_profile_codec_keeps_large_counts_and_original_numeric_encoding() -> None:
    import io

    from polylogue.core.digest import RECEIPT, canonical_bytes
    from polylogue.schemas.workload_profile_rows import read_workload_profile

    rows = DepthGrammars()
    grammar: JSONDocument = {"record_kinds": {"assistant_text": 2**80}, "ratio": 1e-9}
    rows.append("24", grammar)
    profile = WorkloadProfile({"elements": {"agent_transcript": {}}, "profile_version": 1}, rows)
    encoded = b"".join(iter_workload_profile_chunks(profile))
    assert encoded == canonical_bytes(json.loads(encoded), RECEIPT)
    loaded = read_workload_profile(io.BytesIO(encoded))
    assert loaded.depth_grammars is not None and loaded.depth_grammars["24"] == grammar
    assert encoded == b"".join(iter_workload_profile_chunks(loaded))


def test_depth_profile_reference_codec_preserves_original_row_keys_and_wire_labels() -> None:
    import io

    from polylogue.core.digest import RECEIPT, REFERENCE, canonical_bytes
    from polylogue.core.json import json_document
    from polylogue.schemas.workload_profile_rows import read_workload_profile

    grammar = json_document({"record_kinds": {"neutral-n\u0303": 1}, "wire_label": "n\u0303"})
    rows = DepthGrammars()
    rows.append("24", grammar)
    profile = WorkloadProfile({"elements": {"agent_transcript": {}}}, rows)
    ordinary = {"elements": {"agent_transcript": {"record_structures_by_depth": {"24": grammar}}}}
    reference = b"".join(iter_workload_profile_chunks(profile, normalize_unicode=False))
    assert reference == canonical_bytes(ordinary, REFERENCE)
    loaded = read_workload_profile(io.BytesIO(reference))
    assert loaded.depth_grammars is not None
    assert loaded.depth_grammars["24"] == grammar
    assert b"".join(iter_workload_profile_chunks(loaded, normalize_unicode=False)) == reference
    assert b"".join(iter_workload_profile_chunks(profile)) == canonical_bytes(ordinary, RECEIPT)
    assert b"".join(iter_workload_profile_chunks(loaded)) == canonical_bytes(ordinary, RECEIPT)
    assert rows["24"] == grammar  # Receipt serialization never rewrites retained wire evidence.


def test_depth_profile_duplicate_public_key_is_a_typed_invalid_document() -> None:
    import io

    from polylogue.schemas.workload_profile_rows import read_workload_profile

    with pytest.raises(ValueError, match="duplicate workload profile depth"):
        read_workload_profile(
            io.BytesIO(b'{"elements":{"agent_transcript":{"record_structures_by_depth":{"24":{},"24":{}}}}}')
        )
