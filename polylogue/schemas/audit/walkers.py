"""Schema audit loading and traversal helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias

from polylogue.core.json import JSONDocument, JSONValue, json_document, json_document_list
from polylogue.schemas.runtime_registry import SchemaRegistry

SchemaPath = str
SchemaStringValues = list[str]
SchemaRoleRecord = tuple[SchemaPath, str, float]
SchemaValueRecord = tuple[SchemaPath, SchemaStringValues]
SchemaNode: TypeAlias = JSONDocument

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)
_SCHEMA_COMPOSITE_KEYWORDS = ("anyOf", "oneOf", "allOf")


def _load_committed_schema(provider: str) -> SchemaNode | None:
    """Load a committed provider schema."""
    schema_root = Path(__file__).resolve().parent / "providers"
    return SchemaRegistry(storage_root=schema_root).get_schema(provider, version="default")


def _string_values(value: JSONValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _child_schema_nodes(schema: SchemaNode, path: SchemaPath) -> Iterable[tuple[SchemaPath, SchemaNode]]:
    properties = json_document(schema.get("properties"))
    for name, prop in properties.items():
        child = json_document(prop)
        if child:
            yield f"{path}.{name}", child

    additional_properties = json_document(schema.get("additionalProperties"))
    if additional_properties:
        yield f"{path}.*", additional_properties

    items = json_document(schema.get("items"))
    if items:
        yield f"{path}[*]", items

    for keyword in _SCHEMA_COMPOSITE_KEYWORDS:
        for child in json_document_list(schema.get(keyword)):
            yield path, child


def _walk_values(schema: SchemaNode, path: SchemaPath = "$") -> list[SchemaValueRecord]:
    """Walk every nested JSON value and collect x-polylogue-values entries.

    JSON Schema adds schema-bearing keywords over time, and several of them
    are map or list containers rather than ``properties`` children. Walking
    all nested objects keeps the privacy predicate fail-closed for ``$defs``,
    ``patternProperties``, conditionals, array keywords, and future schema
    containers without maintaining another keyword allowlist.
    """
    results: list[SchemaValueRecord] = []
    values = _string_values(schema.get("x-polylogue-values"))
    if values:
        results.append((path, values))

    for key, child in schema.items():
        if key == "x-polylogue-values":
            continue
        child_path = f"{path}.{key}"
        if isinstance(child, dict):
            results.extend(_walk_values(child, child_path))
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, dict):
                    results.extend(_walk_values(item, f"{child_path}[{index}]"))
                elif isinstance(item, list):
                    results.extend(_walk_nested_values(item, f"{child_path}[{index}]"))
    return results


def _walk_nested_values(value: list[JSONValue], path: SchemaPath) -> list[SchemaValueRecord]:
    results: list[SchemaValueRecord] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(item, dict):
            results.extend(_walk_values(item, item_path))
        elif isinstance(item, list):
            results.extend(_walk_nested_values(item, item_path))
    return results


def _walk_semantic_roles(schema: SchemaNode, path: SchemaPath = "$") -> list[SchemaRoleRecord]:
    """Walk schema and collect (path, role, confidence) tuples."""
    results: list[SchemaRoleRecord] = []
    role = schema.get("x-polylogue-semantic-role")
    confidence = schema.get("x-polylogue-score", 0.0)
    if isinstance(role, str):
        results.append((path, role, float(confidence) if isinstance(confidence, int | float) else 0.0))

    for child_path, child in _child_schema_nodes(schema, path):
        results.extend(_walk_semantic_roles(child, child_path))
    return results


__all__ = [
    "_HEX_RE",
    "_UUID_RE",
    "_load_committed_schema",
    "_walk_semantic_roles",
    "_walk_values",
]
