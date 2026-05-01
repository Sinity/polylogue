"""Test helpers for narrowing generated schema payloads."""

from __future__ import annotations

from polylogue.core.json import JSONDocument, JSONValue, json_document


def schema_node(value: object) -> JSONDocument:
    return json_document(value)


def schema_properties(schema: object) -> JSONDocument:
    return json_document(schema_node(schema).get("properties"))


def schema_property(schema: object, name: str) -> JSONDocument:
    return json_document(schema_properties(schema).get(name))


def schema_items(schema: object) -> JSONDocument:
    return json_document(schema_node(schema).get("items"))


def schema_values(schema: object) -> list[JSONValue]:
    values = schema_node(schema).get("x-polylogue-values")
    return list(values) if isinstance(values, list) else []


__all__ = [
    "schema_items",
    "schema_node",
    "schema_properties",
    "schema_property",
    "schema_values",
]
