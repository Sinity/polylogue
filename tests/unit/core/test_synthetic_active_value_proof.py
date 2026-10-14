"""Alias proof follows actual operands without granting catalog support."""

from __future__ import annotations

import random

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.classification import unsupported_schema_constructs
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.observed_values import existing_value_is_supported


def test_null_operand_does_not_activate_object_descendants_or_relations() -> None:
    schema: SchemaRecord = {
        "type": ["null", "object"],
        "properties": {"id": {"type": "string", "x-polylogue-ref": "$.mapping"}},
    }
    assert existing_value_is_supported(schema, None)
    assert "x-polylogue-ref" in unsupported_schema_constructs(schema)
    with pytest.raises(ValueError):
        existing_value_is_supported(schema, {"id": "not-a-proved-graph-link"})
    assert existing_value_is_supported(schema, {})
    counted: SchemaRecord = {
        **schema,
        "x-polylogue-observed-distribution": {"type_counts": {"null": 2, "object": 2}},
    }
    assert existing_value_is_supported(counted, None)
    assert existing_value_is_supported(counted, {})


def test_alias_union_checks_only_branches_matching_the_actual_operand() -> None:
    schema: SchemaRecord = {
        "oneOf": [
            {"type": "object", "properties": {"id": {"type": "string", "pattern": "unimplemented"}}},
            {"type": "null"},
        ]
    }
    assert existing_value_is_supported(schema, None)
    assert "pattern" in unsupported_schema_constructs(schema)
    with pytest.raises(ValueError):
        existing_value_is_supported(schema, {"id": "active"})
    assert not existing_value_is_supported({"oneOf": [{"type": "null"}, {"type": "null"}]}, None)


def test_active_unknown_annotation_and_relational_custody_still_refuse() -> None:
    with pytest.raises(ValueError):
        existing_value_is_supported({"type": "null", "x-unknown-law": True}, None)
    schema: SchemaRecord = {
        "type": "object",
        "properties": {"cached_input_tokens": {"type": "integer"}, "input_tokens": {"type": "integer"}},
        "x-polylogue-numeric-order": [{"lower": "cached_input_tokens", "upper": "input_tokens"}],
    }
    with pytest.raises(ValueError, match="relational solver custody"):
        existing_value_is_supported(schema, {"cached_input_tokens": 1, "input_tokens": 2})
    assert not existing_value_is_supported({"type": "string"}, None)


def test_numeric_union_epoch_and_null_observation_match_actual_scalar_support() -> None:
    numeric: SchemaRecord = {"type": ["null", "integer", "number"], "x-polylogue-format": "unix-epoch"}
    empty: SchemaRecord = {"type": "null", "x-polylogue-observed-distribution": {"type_counts": {"null": 3}}}
    assert not unsupported_schema_constructs(numeric)
    assert not unsupported_schema_constructs(empty)
    assert existing_value_is_supported(numeric, None)
    assert existing_value_is_supported(numeric, 1700000000)
    assert existing_value_is_supported(numeric, 1700000000.5)
    assert existing_value_is_supported(empty, None)
    for schema in (numeric, empty):
        corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")
        try:
            for seed in range(12):
                value = corpus._generate_from_schema(schema, random.Random(seed))
                assert value is None or isinstance(value, int | float)
                assert existing_value_is_supported(schema, value)
        finally:
            corpus._relation_solver.close()


def test_array_alias_keeps_active_prefix_and_forbidden_tail_constraints() -> None:
    schema: SchemaRecord = {
        "type": "array",
        "prefixItems": [{"type": "integer"}, {"type": "string", "x-polylogue-values": ["neutral"]}],
        "items": False,
    }
    assert existing_value_is_supported(schema, [1, "neutral"])
    assert existing_value_is_supported(schema, [1])
    assert not existing_value_is_supported(schema, ["neutral", 1])
    assert not existing_value_is_supported(schema, [1, "neutral", 3])
    assert not existing_value_is_supported({"type": "number"}, float("inf"))


def test_measured_document_message_form_conditions_nullable_parent_before_draws() -> None:
    import json

    from polylogue.core.json import JSONDocument
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS
    from tests.infra.schema_workload_profiles import inferred_record_profile

    original: JSONDocument = {
        "id": "c1e3e4e2-1d69-4dc5-a0b2-c2b6b1a77b03",
        "mapping": {
            "05014f56-9d38-4af5-8b1f-f2a7b8ec2c64": {
                "id": "05014f56-9d38-4af5-8b1f-f2a7b8ec2c64",
                "parent": None,
                "children": ["4b19dcc3-e408-4e59-b609-7a1b561a8fa6"],
                "message": None,
            },
            "4b19dcc3-e408-4e59-b609-7a1b561a8fa6": {
                "id": "4b19dcc3-e408-4e59-b609-7a1b561a8fa6",
                "parent": "05014f56-9d38-4af5-8b1f-f2a7b8ec2c64",
                "children": [],
                "message": {
                    "id": "c1ea565a-8180-4e28-9f5b-cc871e83d16d",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["neutral"]},
                },
            },
        },
        "current_node": "4b19dcc3-e408-4e59-b609-7a1b561a8fa6",
    }
    schema = generate_schema_from_samples([original], provider="chatgpt", max_stats_samples=0)
    profile = inferred_record_profile("chatgpt", schema, [original], element="session_document")
    for seed in range(12):
        corpus = SyntheticCorpus(
            schema,
            PROVIDER_WIRE_FORMATS["chatgpt"],
            "chatgpt",
            element_kind="session_document",
            workload_profile=profile,
        )
        try:
            batch = corpus.generate_batch(count=1, seed=seed)
            document = json.loads(batch.raw_items[0])
            assert batch.artifacts[0].message_count == batch.artifacts[0].facts.message_count == 1
            nodes = list(document["mapping"].values())
            assert len(nodes) == 2
            assert nodes[0]["message"] is None
            assert nodes[1]["message"]["author"]["role"] == "assistant"
            assert sum(len(part) for part in nodes[1]["message"]["content"]["parts"]) == len("neutral")
        finally:
            corpus._relation_solver.close()
