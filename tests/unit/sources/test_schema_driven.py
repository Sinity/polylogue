from __future__ import annotations

import copy

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.core.json import JSONValue, json_document
from polylogue.schemas.registry import SchemaRegistry
from tests.infra.strategies.schema_driven import schema_conformant_payload, strip_schema_extensions


def _collect_schema_paths(schema: object, *, path: str = "root") -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    if isinstance(schema, dict):
        value = schema.get("$schema")
        if isinstance(value, str):
            matches.append((path, value))
        for key, child in schema.items():
            matches.extend(_collect_schema_paths(child, path=f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, child in enumerate(schema):
            matches.extend(_collect_schema_paths(child, path=f"{path}[{index}]"))
    return matches


def test_strip_schema_extensions_removes_nested_metaschema_declarations() -> None:
    raw_schema = SchemaRegistry().get_schema("chatgpt", version="latest")
    assert raw_schema is not None

    cleaned = strip_schema_extensions(copy.deepcopy(raw_schema))
    schema_paths = _collect_schema_paths(cleaned)

    assert schema_paths == [("root", "https://json-schema.org/draft/2020-12/schema")]


def test_strip_schema_extensions_translates_polylogue_values_to_enum() -> None:
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "x-polylogue-values": ["user", "assistant"],
            },
        },
    }

    cleaned = strip_schema_extensions(copy.deepcopy(raw_schema))

    assert isinstance(cleaned, dict)
    properties = cleaned["properties"]
    assert isinstance(properties, dict)
    role_schema = properties["role"]
    assert isinstance(role_schema, dict)
    assert "x-polylogue-values" not in role_schema
    assert role_schema["enum"] == ["user", "assistant"]


def test_distributional_mode_still_respects_polylogue_values_enum() -> None:
    """Both modes translate x-polylogue-values -- distributional adds MORE
    constraints on top, it never loosens the finite value-domain one."""
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "x-polylogue-values": ["user", "assistant"]},
        },
    }

    for mode in ("adversarial", "distributional"):
        cleaned = strip_schema_extensions(copy.deepcopy(raw_schema), mode=mode)
        assert isinstance(cleaned, dict)
        properties = cleaned["properties"]
        assert isinstance(properties, dict)
        role_schema = properties["role"]
        assert isinstance(role_schema, dict)
        assert role_schema["enum"] == ["user", "assistant"], mode


def test_distributional_mode_threads_range_into_minimum_maximum() -> None:
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "elapsedTimeSeconds": {
                "type": "integer",
                "x-polylogue-range": [2.0, 566.0],
            },
        },
    }

    adversarial = strip_schema_extensions(copy.deepcopy(raw_schema), mode="adversarial")
    distributional = strip_schema_extensions(copy.deepcopy(raw_schema), mode="distributional")

    assert isinstance(adversarial, dict)
    adversarial_properties = adversarial["properties"]
    assert isinstance(adversarial_properties, dict)
    adversarial_field = adversarial_properties["elapsedTimeSeconds"]
    assert isinstance(adversarial_field, dict)
    assert "minimum" not in adversarial_field and "maximum" not in adversarial_field

    assert isinstance(distributional, dict)
    distributional_properties = distributional["properties"]
    assert isinstance(distributional_properties, dict)
    distributional_field = distributional_properties["elapsedTimeSeconds"]
    assert isinstance(distributional_field, dict)
    assert distributional_field["minimum"] == 2.0
    assert distributional_field["maximum"] == 566.0


def test_distributional_mode_threads_array_lengths_into_min_max_items() -> None:
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "chat_messages": {
                "type": "array",
                "items": {"type": "string"},
                "x-polylogue-array-lengths": [1, 379],
            },
        },
    }

    distributional = strip_schema_extensions(copy.deepcopy(raw_schema), mode="distributional")
    assert isinstance(distributional, dict)
    properties = distributional["properties"]
    assert isinstance(properties, dict)
    field = properties["chat_messages"]
    assert isinstance(field, dict)
    assert field["minItems"] == 1
    assert field["maxItems"] == 379


def test_distributional_mode_promotes_high_frequency_field_to_required() -> None:
    """A field observed present >= the high-frequency threshold is forced
    present in distributional mode; adversarial mode leaves it optional."""
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "x-polylogue-frequency": 0.999},
            "rare_field": {"type": "string", "x-polylogue-frequency": 0.01},
        },
    }

    adversarial = strip_schema_extensions(copy.deepcopy(raw_schema), mode="adversarial")
    distributional = strip_schema_extensions(copy.deepcopy(raw_schema), mode="distributional")

    assert isinstance(adversarial, dict)
    assert "required" not in adversarial

    assert isinstance(distributional, dict)
    assert distributional["required"] == ["id"]


def test_distributional_mode_maps_known_format_tokens() -> None:
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "created_at": {"type": "string", "x-polylogue-format": "iso8601"},
            "uuid": {"type": "string", "x-polylogue-format": "uuid4"},
        },
    }

    distributional = strip_schema_extensions(copy.deepcopy(raw_schema), mode="distributional")
    assert isinstance(distributional, dict)
    properties = distributional["properties"]
    assert isinstance(properties, dict)
    created_at_schema = properties["created_at"]
    uuid_schema = properties["uuid"]
    assert isinstance(created_at_schema, dict)
    assert isinstance(uuid_schema, dict)
    assert created_at_schema["format"] == "date-time"
    # uuid4 has no safe standard-format equivalent -- deliberately unmapped.
    assert "format" not in uuid_schema


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_distributional_mode_generated_values_respect_range_annotation(data: st.DataObject) -> None:
    """Measurable distribution difference: distributional-mode draws are
    always inside the annotated observed range; adversarial mode has no such
    guarantee (an unconstrained JSON Schema integer spans a far wider domain)."""
    raw_schema: JSONValue = {
        "type": "object",
        "properties": {
            "elapsedTimeSeconds": {"type": "integer", "x-polylogue-range": [2, 566]},
        },
        "required": ["elapsedTimeSeconds"],
    }
    distributional = json_document(strip_schema_extensions(copy.deepcopy(raw_schema), mode="distributional"))

    from hypothesis_jsonschema import from_schema

    payload = data.draw(from_schema(distributional))
    assert isinstance(payload, dict)
    value = payload["elapsedTimeSeconds"]
    assert 2 <= value <= 566


def test_schema_conformant_payload_accepts_mode_parameter() -> None:
    """Smoke-check the public entry point threads `mode` through to the
    registry-backed strategy cache without raising (real registry schema)."""
    from hypothesis import strategies as hyp_st

    for mode in ("adversarial", "distributional"):
        strategy = schema_conformant_payload("claude-code", mode=mode)
        assert isinstance(strategy, hyp_st.SearchStrategy)
