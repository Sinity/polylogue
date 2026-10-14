"""Original anonymous character evidence survives replay and actual draws."""

import json
from collections.abc import Mapping

import pytest

from polylogue.schemas.field_stats.collection import _collect_field_stats
from polylogue.schemas.field_stats.evidence import (
    deserialize_field_stats,
    merge_field_stats_into,
    serialize_field_stats,
)
from polylogue.schemas.generation.field_annotations import annotate_schema
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.observed_values import existing_value_is_supported, measured_string_formats


def test_original_character_length_joint_survives_replay_merge_and_driver() -> None:
    samples = [{"text": "abcdefghij"} for _ in range(70)] + [{"text": "é\n" * 3 + "é" * 6} for _ in range(30)]
    original = _collect_field_stats(samples)
    left = {
        name: deserialize_field_stats(serialize_field_stats(value))
        for name, value in _collect_field_stats(samples[:80]).items()
    }
    right = {
        name: deserialize_field_stats(serialize_field_stats(value))
        for name, value in _collect_field_stats(samples[80:]).items()
    }
    merge_field_stats_into(left, right)
    for name, shape in original["$.text"].string_formats[""].character_shapes.items():
        merged = left["$.text"].string_formats[""].character_shapes[name]
        assert merged.length.to_state() == shape.length.to_state()
        assert merged.newlines.to_state() == shape.newlines.to_state()
    schema = annotate_schema({"type": "object", "properties": {"text": {"type": "string"}}}, left)
    assert "abcdefghij" not in json.dumps(schema)
    assert "é\n" * 3 + "é" * 6 not in json.dumps(schema)
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-character-control")
    try:
        values = [json.loads(row)["text"] for row in corpus.generate(count=1000, seed=31)]
        assert all(len(value) == (10 if value.isascii() else 12) for value in values)
        assert all(value.count("\n") == (0 if value.isascii() else 3) for value in values)
        assert 200 <= sum(not value.isascii() for value in values) <= 400
        properties = schema["properties"]
        assert isinstance(properties, dict)
        node = properties["text"]
        assert isinstance(node, dict)
        assert all(existing_value_is_supported(node, value) for value in values)
    finally:
        corpus._relation_solver.close()


def test_short_measured_non_ascii_string_always_carries_original_class() -> None:
    stats = _collect_field_stats([{"text": "é"} for _ in range(10)])
    schema = annotate_schema({"type": "object", "properties": {"text": {"type": "string"}}}, stats)
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral-character-control")
    try:
        for row in corpus.generate(count=32, seed=7):
            value = json.loads(row)["text"]
            assert len(value) == 1
            assert not value.isascii()
    finally:
        corpus._relation_solver.close()


def test_old_measured_shape_without_character_partitions_refuses_invented_class() -> None:
    parent = annotate_schema(
        {"type": "object", "properties": {"text": {"type": "string"}}}, _collect_field_stats([{"text": "neutral"}])
    )
    properties = parent["properties"]
    assert isinstance(properties, dict)
    schema = properties["text"]
    assert isinstance(schema, dict)
    observed = schema["x-polylogue-observed-distribution"]
    assert isinstance(observed, Mapping)
    formats = observed["string_formats"]
    assert isinstance(formats, dict)
    for shape in formats.values():
        assert isinstance(shape, dict)
        shape.pop("character_shapes")
    with pytest.raises(ValueError):
        measured_string_formats(schema)


def test_character_identity_partition_keeps_pending_membership_and_scope_reset() -> None:
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    owner = ForeignKeyGraph()
    owner._owner()
    try:
        first = owner.fresh_shape_identity("tool", 1, 0, character_class="non_ascii")
        assert isinstance(first, str) and len(first) == 1 and not first.isascii()
        assert owner.fresh_shape_identity("tool", 1, 0, character_class="non_ascii") == first
        owner.observe_tool_identity(first, "call", "new", {"type": "string"}, "")
        following = owner.fresh_shape_identity("tool", 1, 0, character_class="non_ascii")
        assert isinstance(following, str) and len(following) == 1 and not following.isascii()
        assert following != first
        ascii_value = owner.fresh_shape_identity("tool", 1, 0, character_class="ascii")
        assert isinstance(ascii_value, str) and len(ascii_value) == 1 and ascii_value.isascii()
        owner.advance_scope()
        assert owner.fresh_shape_identity("tool", 1, 0, character_class="non_ascii") == first
    finally:
        owner.close()


def test_measured_missing_all_partitions_refuses_but_unmeasured_declared_support_remains() -> None:
    parent = annotate_schema(
        {"type": "object", "properties": {"text": {"type": "string"}}}, _collect_field_stats([{"text": "neutral"}])
    )
    properties = parent["properties"]
    assert isinstance(properties, dict)
    schema = properties["text"]
    assert isinstance(schema, dict)
    observed = schema["x-polylogue-observed-distribution"]
    assert isinstance(observed, dict)
    observed.pop("string_formats")
    with pytest.raises(ValueError):
        measured_string_formats(schema)
    with pytest.raises(ValueError):
        existing_value_is_supported(schema, "neutral")
    assert measured_string_formats({"type": "string", "minLength": 2, "maxLength": 3}) == {}


def test_width_draw_does_not_relabel_original_character_newline_observations() -> None:
    import random

    from polylogue.schemas.synthetic.observed_values import measured_character_shapes
    from polylogue.schemas.synthetic.runtime import _condition_character_shape, _condition_width_partition

    parent = annotate_schema(
        {"type": "object", "properties": {"text": {"type": "string"}}},
        _collect_field_stats([{"text": "!" * 10}] * 3 + [{"text": "!" * 20}] * 2),
    )
    properties = parent["properties"]
    assert isinstance(properties, dict)
    schema = properties["text"]
    assert isinstance(schema, dict)
    formats = measured_string_formats(schema)
    original = formats[""]
    projected = _condition_width_partition(original, width=10, geometry=None)
    assert projected is not None
    draws = measured_character_shapes(projected, 3)
    original_classes = original["character_shapes"]
    assert isinstance(original_classes, dict)
    original_ascii = original_classes["ascii"]
    assert isinstance(original_ascii, dict)
    assert draws["ascii"]["newline_count"] == original_ascii["newline_count"]
    conditioned = _condition_character_shape(
        {"type": "string", "x-polylogue-observed-distribution": projected}, random.Random(31)
    )
    observed = conditioned["x-polylogue-observed-distribution"]
    assert isinstance(observed, dict)
    assert observed["character_class"] == "ascii"
    drawn_length = observed["string_length"]
    assert isinstance(drawn_length, dict)
    assert drawn_length["min"] == drawn_length["max"] == 10
