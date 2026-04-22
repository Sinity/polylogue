from __future__ import annotations

import copy

from polylogue.lib.json import JSONValue
from polylogue.schemas.registry import SchemaRegistry
from tests.infra.strategies.schema_driven import strip_schema_extensions


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
