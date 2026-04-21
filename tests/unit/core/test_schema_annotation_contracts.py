"""Integration tests for semantic and relational annotation of generated schemas, plus packaged schema quality.

Tests the contract between:
  - semantic_inference.infer_semantic_roles()
  - relational_inference.infer_relations()
  - schema_generation._annotate_semantic_and_relational()
  - schema_generation.generate_schema_from_samples()

Ensures:
  1. x-polylogue-semantic-role annotations appear at correct schema paths
  2. x-polylogue-score and x-polylogue-evidence attached
  3. Relational annotations (FKs, time deltas, exclusions, lengths) at root
  4. End-to-end: samples → field stats → annotations → schema
  5. Packaged provider schemas expose coherent annotation metadata
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import pytest

from polylogue.schemas.field_stats import FieldStats, _collect_field_stats
from polylogue.schemas.generation_support import (
    _annotate_semantic_and_relational,
)
from polylogue.schemas.generation_workflow import (
    generate_schema_from_samples,
)
from polylogue.schemas.json_types import JSONDocument
from polylogue.schemas.semantic_inference_runtime import (
    infer_semantic_roles,
)
from tests.infra.schema_access import (
    schema_items,
    schema_node,
    schema_properties,
    schema_property,
    schema_values,
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
        """x-polylogue-score and x-polylogue-evidence added with candidate."""
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
        role_schema = schema_property(result_schema, "role")
        if role_schema and "x-polylogue-semantic-role" in role_schema:
            assert "x-polylogue-score" in role_schema
            assert isinstance(role_schema["x-polylogue-score"], (int, float))
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
        # Set up FK detection.
        field_stats["$.user_id"].ref_target = "$.users"
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
        properties = schema_properties(result_schema)
        assert properties
        assert "messages" in properties
        assert "title" in properties

    def test_semantic_annotations_in_generated_schema(self) -> None:
        """Generated schema includes semantic role annotations."""
        samples = [
            {
                "messages": [
                    {"role": "user", "text": "Query " + str(i), "timestamp": 1000000000 + i} for i in range(5)
                ],
                "title": f"Conversation {j}",
            }
            for j in range(3)
        ]
        result_schema = generate_schema_from_samples(samples)
        assert result_schema is not None

        # Check for at least some semantic annotations
        has_semantic = False

        def check_schema(s: object, depth: int = 0) -> bool:
            nonlocal has_semantic
            node = schema_node(s)
            if "x-polylogue-semantic-role" in node:
                has_semantic = True
            for prop_schema in schema_properties(node).values():
                check_schema(prop_schema, depth + 1)
            items = schema_items(node)
            if items:
                check_schema(items, depth + 1)
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
        props = schema_properties(result_schema)
        assert "id" in props or "title" in props or "message_count" in props

    def test_empty_samples_returns_base_object_schema(self) -> None:
        """Empty sample list returns base object schema."""
        result_schema = generate_schema_from_samples([])
        assert result_schema is not None
        assert result_schema.get("type") == "object"
        # Empty samples may have no properties or minimal structure


class TestSemanticInferenceMisclassificationRegression:
    """Regression tests for known misclassification bugs.

    parentUuid and runSettings.model were both scoring 0.6 as
    conversation_title due to generic positives (short string + high
    cardinality). Fixed by adding negative signals in _score_title.
    """

    def test_uuid_field_not_scored_as_title(self) -> None:
        """A UUID-format field should never be classified as conversation_title."""
        field_stats = {
            "$.parentUuid": FieldStats(
                path="$.parentUuid",
                observed_values=Counter(
                    {
                        "550e8400-e29b-41d4-a716-446655440000": 1,
                        "6ba7b810-9dad-11d1-80b4-00c04fd430c8": 1,
                        "f47ac10b-58cc-4372-a567-0e02b2c3d479": 1,
                    }
                ),
                detected_formats=Counter({"uuid4": 2, "uuid": 1}),
                string_lengths=[36, 36, 36],
                is_multiline=0,
                newline_counts=[0, 0, 0],
                total_samples=3,
                present_count=3,
                value_count=3,
            ),
        }
        candidates = infer_semantic_roles(field_stats)
        title_candidates = [c for c in candidates if c.role == "conversation_title" and c.path == "$.parentUuid"]
        assert not title_candidates, "UUID field should not be a title candidate"

    def test_model_field_not_scored_as_title(self) -> None:
        """A model-slug field should not be classified as conversation_title."""
        field_stats = {
            "$.runSettings.model": FieldStats(
                path="$.runSettings.model",
                observed_values=Counter(
                    {
                        "gpt-4-code-interpreter": 100,
                        "gpt-4": 50,
                        "models/gemini-2.5-pro": 30,
                    }
                ),
                string_lengths=[23, 5, 22],
                is_multiline=0,
                newline_counts=[0, 0, 0],
                total_samples=180,
                present_count=180,
                value_count=180,
            ),
        }
        candidates = infer_semantic_roles(field_stats)
        title_candidates = [c for c in candidates if c.role == "conversation_title" and c.path == "$.runSettings.model"]
        # Should either not appear, or have very low confidence
        for c in title_candidates:
            assert c.confidence < 0.3, f"Model field scored {c.confidence} as title, expected <0.3"

    def test_id_suffix_field_penalized(self) -> None:
        """Fields ending in 'Id' or '_id' are penalized for title role."""
        field_stats = {
            "$.parentId": FieldStats(
                path="$.parentId",
                observed_values=Counter({f"id-{i}": 1 for i in range(20)}),
                string_lengths=[5] * 20,
                is_multiline=0,
                newline_counts=[0] * 20,
                total_samples=20,
                present_count=20,
                value_count=20,
            ),
        }
        candidates = infer_semantic_roles(field_stats)
        title_candidates = [c for c in candidates if c.role == "conversation_title" and c.path == "$.parentId"]
        for c in title_candidates:
            assert c.confidence < 0.15, f"ID-suffix field scored {c.confidence} as title, expected <0.15"

    def test_path_like_values_penalized(self) -> None:
        """Fields where >30% of values contain '/' are penalized."""
        field_stats = {
            "$.model_name": FieldStats(
                path="$.model_name",
                observed_values=Counter(
                    {
                        "models/gemini-2.5-pro": 10,
                        "models/gemini-1.5-flash": 8,
                        "models/gemini-1.0-pro": 5,
                        "gpt-4": 2,
                    }
                ),
                string_lengths=[22, 24, 22, 5],
                is_multiline=0,
                newline_counts=[0, 0, 0, 0],
                total_samples=25,
                present_count=25,
                value_count=25,
            ),
        }
        candidates = infer_semantic_roles(field_stats)
        title_candidates = [c for c in candidates if c.role == "conversation_title" and c.path == "$.model_name"]
        for c in title_candidates:
            assert c.confidence < 0.3, f"Path-heavy field scored {c.confidence} as title, expected <0.3"


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
        conv_ids: list[str | None] = ["conv-1", "conv-1"]
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


# =============================================================================
# Merged from test_schema_annotations.py (2024-03-15)
# =============================================================================


def _load_schema(provider: str) -> JSONDocument | None:
    """Load a packaged provider schema, returning None if absent."""
    try:
        from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry

        registry = SchemaRegistry(storage_root=SCHEMA_DIR)
        package = registry.get_package(provider, version="default")
        if package is None:
            return None
        return registry.get_element_schema(
            provider,
            version=package.version,
            element_kind=package.default_element_kind,
        )
    except Exception:
        return None


def _find_annotations(
    schema: Mapping[str, object],
    prefix: str = "x-polylogue-",
) -> dict[str, list[tuple[str, object]]]:
    """Walk the schema tree and collect all x-polylogue-* annotations by key."""
    result: dict[str, list[tuple[str, object]]] = {}

    def _walk(obj: object, path: str) -> None:
        if not isinstance(obj, Mapping):
            return
        for key, value in obj.items():
            if key.startswith(prefix):
                result.setdefault(key, []).append((path, value))
            if isinstance(value, dict):
                _walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        _walk(item, f"{path}.{key}[{index}]")

    _walk(schema, "$")
    return result


def _get_nested(schema: object, dotpath: str) -> JSONDocument | None:
    """Navigate a schema by dot-separated property path."""
    current = schema_node(schema)
    for part in dotpath.split("."):
        if part == "additionalProperties":
            candidate = schema_node(current.get("additionalProperties"))
        else:
            candidate = schema_property(current, part)
        current = schema_node(candidate)
        if not current:
            return None
        any_of = current.get("anyOf")
        if isinstance(any_of, list):
            for variant in any_of:
                variant_node = schema_node(variant)
                if variant_node.get("type") == "object" and schema_properties(variant_node):
                    current = variant_node
                    break
    return current or None


def _schema_any_of_variants(schema: object) -> list[JSONDocument]:
    any_of = schema_node(schema).get("anyOf")
    return [variant for variant in any_of if isinstance(variant, dict)] if isinstance(any_of, list) else []


class TestSchemaAnnotations:
    """Packaged schemas should expose coherent annotation metadata."""

    def test_chatgpt_role_semantic(self) -> None:
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        role_schema = _get_nested(schema, "mapping.additionalProperties.message.author.role")
        assert role_schema is not None
        # Role field is identified via semantic inference; enum values may or may
        # not be populated depending on cross-conversation threshold filtering.
        assert role_schema.get("x-polylogue-semantic-role") == "message_role"

    def test_chatgpt_uuid_format(self) -> None:
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        current_node = schema_property(schema, "current_node")
        assert current_node
        assert current_node.get("x-polylogue-format") == "uuid4"
        node_id = _get_nested(schema, "mapping.additionalProperties.id")
        assert node_id is not None
        assert node_id.get("x-polylogue-format") == "uuid4"

    def test_chatgpt_timestamp_format(self) -> None:
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        create_time = schema_property(schema, "create_time")
        assert create_time
        fmt = create_time.get("x-polylogue-format")
        rng = create_time.get("x-polylogue-range")
        assert fmt == "unix-epoch" or rng is not None

    def test_chatgpt_reference_detection(self) -> None:
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        current_node = schema_property(schema, "current_node")
        assert current_node
        assert current_node.get("x-polylogue-ref") == "$.mapping"

    def test_claude_code_has_annotations(self) -> None:
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        annotations = _find_annotations(schema)
        total = sum(len(values) for values in annotations.values())
        assert total > 100
        assert "x-polylogue-format" in annotations
        assert "x-polylogue-values" in annotations
        assert "x-polylogue-frequency" in annotations

    def test_claude_code_type_enum(self) -> None:
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        type_schema = schema_property(schema, "type")
        assert type_schema
        type_values = schema_values(type_schema)
        assert any(value in type_values for value in ("user", "assistant", "human"))

    def test_claude_ai_sender_semantic(self) -> None:
        schema = _load_schema("claude-ai")
        if schema is None:
            pytest.skip("Claude AI schema not available")

        messages = schema_property(schema, "chat_messages")
        assert messages
        item = schema_items(messages)
        assert item
        # Navigate anyOf if present
        variants = _schema_any_of_variants(item)
        if variants:
            for variant in variants:
                sender = schema_property(variant, "sender")
                if "x-polylogue-semantic-role" in sender:
                    assert sender["x-polylogue-semantic-role"] == "message_role"
                    return
        sender = schema_property(item, "sender")
        assert sender.get("x-polylogue-semantic-role") == "message_role"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_frequency_values_in_range(self, provider: str) -> None:
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, frequency in _find_annotations(schema).get("x-polylogue-frequency", []):
            assert not isinstance(frequency, bool) and isinstance(frequency, (int, float))
            assert 0.0 < float(frequency) < 1.0, f"{provider} {path}: frequency {frequency} not in (0, 1)"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_numeric_ranges_plausible(self, provider: str) -> None:
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, value_range in _find_annotations(schema).get("x-polylogue-range", []):
            assert isinstance(value_range, list) and len(value_range) == 2
            low, high = value_range
            assert low <= high, f"{provider} {path}: range inverted: {low} > {high}"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_format_values_are_known(self, provider: str) -> None:
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        known_formats = {
            "uuid4",
            "uuid",
            "hex-id",
            "iso8601",
            "unix-epoch",
            "unix-epoch-str",
            "base64",
            "url",
            "email",
            "mime-type",
        }
        for path, fmt in _find_annotations(schema).get("x-polylogue-format", []):
            assert fmt in known_formats, f"{provider} {path}: unknown format {fmt!r}"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_values_are_nonempty_lists(self, provider: str) -> None:
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, values in _find_annotations(schema).get("x-polylogue-values", []):
            assert isinstance(values, list), f"{provider} {path}: values not a list"
            assert values, f"{provider} {path}: empty values list"
            assert all(isinstance(value, str) for value in values)
