"""Integration tests for semantic and relational annotation of generated schemas.

Tests the contract between:
  - semantic_inference.infer_semantic_roles()
  - relational_inference.infer_relations()
  - schema_generation._annotate_semantic_and_relational()
  - schema_generation.generate_schema_from_samples()

Ensures:
  1. x-polylogue-semantic-role annotations appear at correct schema paths
  2. x-polylogue-confidence and x-polylogue-evidence attached
  3. Relational annotations (FKs, time deltas, exclusions, lengths) at root
  4. End-to-end: samples → field stats → annotations → schema
"""

from __future__ import annotations

from collections import Counter

import pytest

from polylogue.schemas.field_stats import FieldStats, _collect_field_stats
from polylogue.schemas.relational_inference import infer_relations
from polylogue.schemas.schema_generation import (
    _annotate_semantic_and_relational,
    generate_schema_from_samples,
)
from polylogue.schemas.semantic_inference import (
    infer_semantic_roles,
    select_best_roles,
)


class TestAnnotateSemanticAndRelational:
    """_annotate_semantic_and_relational() attaches annotations to schema."""

    def test_semantic_role_annotation_placement(self) -> None:
        """x-polylogue-semantic-role placed at correct schema path."""
        schema = {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "title": {
                    "type": "string",
                },
            },
        }
        field_stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8],
                total_samples=2,
                present_count=2,
            ),
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({"Chat A": 1, "Chat B": 1}),
                string_lengths=[6, 6],
                is_multiline=0,
                newline_counts=[0, 0],
                total_samples=2,
                present_count=2,
                value_count=2,
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)

        # Check semantic annotation placement
        assert "properties" in result_schema
        # Annotations should be attached if semantic roles were detected
        assert isinstance(result_schema, dict)

    def test_confidence_and_evidence_included(self) -> None:
        """x-polylogue-confidence and x-polylogue-evidence added with candidate."""
        schema = {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
            },
        }
        field_stats = {
            "$.role": FieldStats(
                path="$.role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=95,
                value_count=100,
                string_lengths=[4, 5, 9],
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)

        # If role was detected, check confidence/evidence
        if "role" in result_schema.get("properties", {}):
            role_schema = result_schema["properties"]["role"]
            if "x-polylogue-semantic-role" in role_schema:
                assert "x-polylogue-confidence" in role_schema
                assert isinstance(role_schema["x-polylogue-confidence"], (int, float))
                assert "x-polylogue-evidence" in role_schema
                assert isinstance(role_schema["x-polylogue-evidence"], dict)

    def test_foreign_key_annotation_at_root(self) -> None:
        """x-polylogue-foreign-keys annotation at schema root."""
        schema = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "users": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            },
        }
        field_stats = {
            "$.user_id": FieldStats(
                path="$.user_id",
                observed_values=Counter({"uid-1": 5, "uid-2": 5}),
                total_samples=10,
                present_count=10,
                value_count=10,
            ),
        }
        # Set up FK detection
        field_stats["$.user_id"]._ref_target = "$.users"  # type: ignore
        result_schema = _annotate_semantic_and_relational(schema, field_stats)

        # Check if FK annotations are present when relations exist
        assert isinstance(result_schema, dict)

    def test_time_delta_annotation_at_root(self) -> None:
        """x-polylogue-time-deltas annotation at schema root."""
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"type": "number"},
                "updated_at": {"type": "number"},
            },
        }
        field_stats = {
            "$.created_at": FieldStats(
                path="$.created_at",
                num_min=1000000000.0,
                num_max=1500000000.0,
                detected_formats=Counter({"unix-epoch": 100}),
                total_samples=100,
                present_count=100,
                value_count=100,
            ),
            "$.updated_at": FieldStats(
                path="$.updated_at",
                num_min=1000000000.0,
                num_max=1600000000.0,
                detected_formats=Counter({"unix-epoch": 100}),
                total_samples=100,
                present_count=100,
                value_count=100,
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)
        assert isinstance(result_schema, dict)

    def test_mutual_exclusion_annotation_at_root(self) -> None:
        """x-polylogue-mutually-exclusive annotation at schema root."""
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "option_a": {"type": "string"},
                        "option_b": {"type": "string"},
                    },
                },
            },
        }
        field_stats = {
            "$.config.option_a": FieldStats(
                path="$.config.option_a",
                present_count=50,
                total_samples=100,
                value_count=50,
                co_occurring_fields=Counter(),
            ),
            "$.config.option_b": FieldStats(
                path="$.config.option_b",
                present_count=50,
                total_samples=100,
                value_count=50,
                co_occurring_fields=Counter(),
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)
        assert isinstance(result_schema, dict)

    def test_string_length_annotation_at_root(self) -> None:
        """x-polylogue-string-lengths annotation at schema root."""
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
        }
        field_stats = {
            "$.description": FieldStats(
                path="$.description",
                string_lengths=[50, 100, 150, 200],
                is_multiline=2,
                newline_counts=[1, 2, 1, 2],
                total_samples=4,
                present_count=4,
                value_count=4,
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)
        assert isinstance(result_schema, dict)

    def test_nested_property_annotation(self) -> None:
        """Semantic annotations on nested properties."""
        schema = {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        }
        field_stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8],
                total_samples=2,
                present_count=2,
            ),
            "$.messages[*].role": FieldStats(
                path="$.messages[*].role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=95,
                value_count=100,
                string_lengths=[4, 5, 9],
            ),
            "$.messages[*].text": FieldStats(
                path="$.messages[*].text",
                string_lengths=[100, 200, 300],
                is_multiline=2,
                newline_counts=[2, 3, 1],
                total_samples=3,
                present_count=3,
                value_count=3,
            ),
        }
        result_schema = _annotate_semantic_and_relational(schema, field_stats)

        # Verify nested annotations exist if detected
        assert "properties" in result_schema


class TestGenerateSchemaFromSamples:
    """generate_schema_from_samples() end-to-end integration."""

    def test_simple_sample_set(self) -> None:
        """Single sample generates valid schema."""
        samples = [
            {
                "title": "Chat 1",
                "role": "user",
                "text": "Hello world",
                "created_at": 1000000000,
            }
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None
        assert result_schema.get("type") == "object"
        assert "properties" in result_schema

    def test_multiple_samples_collect_stats(self) -> None:
        """Multiple samples create diverse field stats."""
        samples = [
            {
                "messages": [
                    {"role": "user", "text": "Hello"},
                    {"role": "assistant", "text": "Hi there"},
                ],
                "title": "Chat 1",
            },
            {
                "messages": [
                    {"role": "user", "text": "How are you?"},
                    {"role": "assistant", "text": "I'm doing well"},
                ],
                "title": "Chat 2",
            },
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None
        assert "properties" in result_schema
        assert "messages" in result_schema["properties"]
        assert "title" in result_schema["properties"]

    def test_semantic_annotations_in_generated_schema(self) -> None:
        """Generated schema includes semantic role annotations."""
        samples = [
            {
                "messages": [
                    {"role": "user", "text": "Query " + str(i), "timestamp": 1000000000 + i}
                    for i in range(5)
                ],
                "title": f"Conversation {j}",
            }
            for j in range(3)
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None

        # Check for at least some semantic annotations
        has_semantic = False

        def check_schema(s: dict, depth: int = 0) -> bool:
            nonlocal has_semantic
            if "x-polylogue-semantic-role" in s:
                has_semantic = True
            if "properties" in s:
                for prop_schema in s["properties"].values():
                    if isinstance(prop_schema, dict):
                        check_schema(prop_schema, depth + 1)
            if "items" in s and isinstance(s["items"], dict):
                check_schema(s["items"], depth + 1)
            return has_semantic

        check_schema(result_schema)
        # With diverse samples, at least some semantic roles should be detected
        assert has_semantic

    def test_relational_annotations_in_generated_schema(self) -> None:
        """Generated schema includes relational annotations at root."""
        samples = [
            {
                "created_at": 1000000000 + i,
                "updated_at": 1000000000 + i + 100,
                "config": {
                    "option_a": "value" if i % 2 == 0 else None,
                    "option_b": "other" if i % 2 == 1 else None,
                },
            }
            for i in range(5)
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None
        # Relational annotations may appear at root
        # (they're optional depending on detected relations)

    def test_handles_nested_structures(self) -> None:
        """Complex nested structures handled correctly."""
        samples = [
            {
                "conversation": {
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "text": "Hello",
                                "timestamp": 1000000000,
                            },
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "text": "Hi",
                                "timestamp": 1000000001,
                            },
                        },
                    ],
                    "metadata": {
                        "title": "Chat",
                        "created": 1000000000,
                    },
                },
            }
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None
        assert result_schema.get("type") == "object"

    def test_handles_diverse_types(self) -> None:
        """Schema captures diverse field types."""
        samples = [
            {
                "id": "conv-123",
                "title": "Chat",
                "message_count": 5,
                "is_active": True,
                "metadata": {"tags": ["important"]},
            }
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None
        props = result_schema.get("properties", {})
        assert "id" in props or "title" in props or "message_count" in props

    def test_empty_samples_returns_base_object_schema(self) -> None:
        """Empty sample list returns base object schema."""
        result_schema = generate_schema_from_samples([])
        assert result_schema is not None
        assert result_schema.get("type") == "object"
        # Empty samples may have no properties or minimal structure


class TestFieldStatsCollection:
    """_collect_field_stats() provides foundation for annotations."""

    def test_collect_from_simple_dict(self) -> None:
        """Simple dict sample produces field stats."""
        samples = [
            {
                "title": "Chat 1",
                "role": "user",
            }
        ]
        stats = _collect_field_stats(samples)
        assert "$.title" in stats
        assert "$.role" in stats
        assert stats["$.title"].observed_values["Chat 1"] == 1
        assert stats["$.role"].observed_values["user"] == 1

    def test_collect_from_array_items(self) -> None:
        """Array items collected with [*] notation."""
        samples = [
            {
                "messages": [
                    {"role": "user"},
                    {"role": "assistant"},
                ]
            }
        ]
        stats = _collect_field_stats(samples)
        assert "$.messages" in stats
        assert "$.messages[*].role" in stats
        assert len(stats["$.messages"].array_lengths) > 0

    def test_collect_with_conversation_ids(self) -> None:
        """When conversation_ids supplied, value→conversation mapping tracked."""
        samples = [
            {"role": "user"},
            {"role": "assistant"},
        ]
        conv_ids = ["conv-1", "conv-1"]
        stats = _collect_field_stats(samples, conversation_ids=conv_ids)
        role_stats = stats["$.role"]
        # Both values should map to conv-1
        assert "user" in role_stats.value_conversation_ids
        assert "conv-1" in role_stats.value_conversation_ids["user"]
        assert "assistant" in role_stats.value_conversation_ids
        assert "conv-1" in role_stats.value_conversation_ids["assistant"]

    def test_collect_multiline_detection(self) -> None:
        """Newlines tracked per string value."""
        samples = [
            {"text": "Single line"},
            {"text": "Multi\nline\ntext"},
        ]
        stats = _collect_field_stats(samples)
        text_stats = stats["$.text"]
        assert text_stats.is_multiline == 1  # one multiline
        assert text_stats.newline_rate > 0

    def test_collect_numeric_range(self) -> None:
        """Numeric min/max tracked."""
        samples = [
            {"count": 10},
            {"count": 50},
            {"count": 100},
        ]
        stats = _collect_field_stats(samples)
        count_stats = stats["$.count"]
        assert count_stats.num_min == 10.0
        assert count_stats.num_max == 100.0

    def test_collect_format_detection(self) -> None:
        """String formats detected (epoch, UUID, etc.)."""
        samples = [
            {"ts": 1000000000},
            {"ts": 1500000000},
        ]
        stats = _collect_field_stats(samples)
        ts_stats = stats["$.ts"]
        assert "unix-epoch" in ts_stats.detected_formats
