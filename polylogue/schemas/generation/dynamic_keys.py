"""Dynamic-key collapse helpers for schema generation."""

from __future__ import annotations

try:
    from genson import SchemaBuilder

    GENSON_AVAILABLE = True
except ImportError:
    SchemaBuilder = None
    GENSON_AVAILABLE = False

from polylogue.lib.json import JSONDocument, json_document, json_document_list
from polylogue.schemas.field_stats import is_dynamic_key

_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35


def _looks_pathlike_key(key: str) -> bool:
    if "/" in key or "\\" in key:
        return True
    if key.count(".") >= 2:
        return True
    return ":" in key and len(key) > 2


def _should_collapse_high_cardinality_keys(keys: list[str]) -> bool:
    if len(keys) >= _HIGH_CARDINALITY_KEY_THRESHOLD:
        return True
    if len(keys) < 24:
        return False
    pathlike = sum(1 for key in keys if _looks_pathlike_key(key))
    return (pathlike / len(keys)) >= _PATHLIKE_KEY_RATIO_THRESHOLD


def merge_schemas(schemas: list[JSONDocument]) -> JSONDocument:
    """Merge multiple schemas into one using genson when available."""
    if not GENSON_AVAILABLE:
        return schemas[0] if schemas else {}
    builder = SchemaBuilder()
    for schema in schemas:
        builder.add_schema(schema)
    return json_document(builder.to_schema())


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

        if _should_collapse_high_cardinality_keys(key_names):
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
    "merge_schemas",
]
