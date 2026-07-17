"""Dynamic-key collapse helpers for schema generation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping

try:
    from genson import SchemaBuilder

    GENSON_AVAILABLE = True
except ImportError:
    SchemaBuilder = None
    GENSON_AVAILABLE = False

from polylogue.core.json import JSONDocument, JSONValue, json_document, json_document_list
from polylogue.schemas.field_stats.detection import is_dynamic_key, should_collapse_observed_keys

_STRUCTURAL_DEDUP_WINDOW = 1_024


def merge_schemas(schemas: Iterable[JSONDocument]) -> JSONDocument:
    """Merge multiple schemas into one using genson when available."""
    if not GENSON_AVAILABLE:
        return next(iter(schemas), {})
    builder = SchemaBuilder()
    observed = False
    for schema in schemas:
        builder.add_schema(schema)
        observed = True
    return json_document(builder.to_schema()) if observed else {}


def _schema_types(schema: Mapping[str, object]) -> set[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    return set()


def _required_names(schema: Mapping[str, object]) -> set[str]:
    required = schema.get("required")
    return {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()


def _merge_observed_structure_pair(left: JSONDocument, right: JSONDocument) -> JSONDocument:
    if not left:
        return right
    if not right:
        return left

    merged: JSONDocument = {}
    schema_types = sorted(_schema_types(left) | _schema_types(right))
    if len(schema_types) == 1:
        merged["type"] = schema_types[0]
    elif schema_types:
        type_values: list[JSONValue] = list(schema_types)
        merged["type"] = type_values

    left_properties = json_document(left.get("properties"))
    right_properties = json_document(right.get("properties"))
    property_names = sorted(set(left_properties) | set(right_properties))
    properties: JSONDocument = {}
    for name in property_names:
        left_schema = json_document(left_properties.get(name))
        right_schema = json_document(right_properties.get(name))
        properties[name] = _merge_observed_structure_pair(left_schema, right_schema)

    required = sorted(_required_names(left) & _required_names(right))
    left_additional = json_document(left.get("additionalProperties"))
    right_additional = json_document(right.get("additionalProperties"))
    additional = _merge_observed_structure_pair(left_additional, right_additional)

    already_high_cardinality = (
        left.get("x-polylogue-high-cardinality-keys") is True or right.get("x-polylogue-high-cardinality-keys") is True
    )
    if properties and (already_high_cardinality or should_collapse_observed_keys(properties.keys())):
        additional = merge_observed_structure_schemas([additional, *map(json_document, properties.values())])
        properties = {}
        required = []
        merged["x-polylogue-high-cardinality-keys"] = True
        merged["x-polylogue-dynamic-keys"] = True

    if properties:
        merged["properties"] = properties
    if required:
        required_values: list[JSONValue] = list(required)
        merged["required"] = required_values

    left_items = json_document(left.get("items"))
    right_items = json_document(right.get("items"))
    items = _merge_observed_structure_pair(left_items, right_items)
    if items:
        merged["items"] = items
    if additional:
        merged["additionalProperties"] = additional
        merged["x-polylogue-dynamic-keys"] = True

    for marker in ("x-polylogue-dynamic-keys", "x-polylogue-high-cardinality-keys"):
        if left.get(marker) is True or right.get(marker) is True:
            merged[marker] = True
    return merged


def merge_observed_structure_schemas(schemas: Iterable[JSONDocument]) -> JSONDocument:
    """Incrementally merge structural schemas without retaining property history."""
    merged: JSONDocument = {}
    seen_identities: set[bytes] = set()
    identity_order: deque[bytes] = deque()
    for schema in schemas:
        identity = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).digest()
        if identity in seen_identities:
            continue
        if len(identity_order) >= _STRUCTURAL_DEDUP_WINDOW:
            seen_identities.remove(identity_order.popleft())
        seen_identities.add(identity)
        identity_order.append(identity)
        merged = _merge_observed_structure_pair(merged, schema)
    return merged


def dynamic_object_paths(schema: Mapping[str, object], path: str = "$") -> set[str]:
    """Return schema paths whose observed object keys were collapsed."""
    paths: set[str] = set()
    if schema.get("x-polylogue-dynamic-keys") is True:
        paths.add(path)
    for name, child in json_document(schema.get("properties")).items():
        paths.update(dynamic_object_paths(json_document(child), f"{path}.{name}"))
    items = json_document(schema.get("items"))
    if items:
        paths.update(dynamic_object_paths(items, f"{path}[*]"))
    additional = json_document(schema.get("additionalProperties"))
    if additional:
        paths.update(dynamic_object_paths(additional, f"{path}.*"))
    return paths


def observed_structure_schema(value: object) -> JSONDocument:
    """Build a bounded structural schema before Genson sees dynamic keys.

    Genson creates one node per object property.  Feeding it provider payloads
    directly therefore retains unbounded ID-, path-, and content-shaped keys
    until the later schema-collapse pass.  This recursive projection preserves
    every observed value shape while merging dynamic-key values incrementally
    into ``additionalProperties``.
    """
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        array_schema: JSONDocument = {"type": "array"}
        items = merge_observed_structure_schemas(observed_structure_schema(item) for item in value)
        if items:
            array_schema["items"] = items
        return array_schema
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported schema observation value: {type(value).__name__}")

    collapse_all = should_collapse_observed_keys(value.keys())
    properties: JSONDocument = {}
    required: list[JSONValue] = []
    for key, child in value.items():
        key_text = str(key)
        if collapse_all or is_dynamic_key(key_text):
            continue
        properties[key_text] = observed_structure_schema(child)
        required.append(key_text)

    object_schema: JSONDocument = {"type": "object"}
    if properties:
        object_schema["properties"] = properties
        object_schema["required"] = required
    dynamic_values = merge_observed_structure_schemas(
        observed_structure_schema(child) for key, child in value.items() if collapse_all or is_dynamic_key(str(key))
    )
    if dynamic_values:
        object_schema["additionalProperties"] = dynamic_values
        object_schema["x-polylogue-dynamic-keys"] = True
        if collapse_all:
            object_schema["x-polylogue-high-cardinality-keys"] = True
    return object_schema


def collapse_dynamic_keys(schema: JSONDocument) -> JSONDocument:
    """Collapse dynamic key properties into additionalProperties."""
    properties = json_document(schema.get("properties"))
    if properties:
        static_props: JSONDocument = {}
        dynamic_schemas: list[JSONDocument] = []
        key_names = list(properties.keys())

        for key, value in properties.items():
            collapsed_value = collapse_dynamic_keys(json_document(value))
            if is_dynamic_key(key):
                dynamic_schemas.append(collapsed_value)
            else:
                static_props[key] = collapsed_value

        if should_collapse_observed_keys(key_names):
            dynamic_schemas.extend(json_document(value) for value in static_props.values() if json_document(value))
            static_props = {}
            schema["x-polylogue-high-cardinality-keys"] = True

        if dynamic_schemas:
            schema["properties"] = static_props
            schema["additionalProperties"] = merge_schemas(dynamic_schemas)
            schema["x-polylogue-dynamic-keys"] = True
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [item for item in required if isinstance(item, str) and item in static_props]
        else:
            schema["properties"] = static_props

    items = json_document(schema.get("items"))
    if items:
        schema["items"] = collapse_dynamic_keys(items)

    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = json_document_list(schema.get(keyword))
        if variants:
            schema[keyword] = [collapse_dynamic_keys(item) for item in variants]

    additional_properties = json_document(schema.get("additionalProperties"))
    if additional_properties:
        schema["additionalProperties"] = collapse_dynamic_keys(additional_properties)

    return schema


__all__ = [
    "GENSON_AVAILABLE",
    "SchemaBuilder",
    "collapse_dynamic_keys",
    "dynamic_object_paths",
    "merge_schemas",
    "merge_observed_structure_schemas",
    "observed_structure_schema",
]
