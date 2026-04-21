"""Field-stat-driven schema annotation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import overload

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.json_types import JSONDocument, JSONValue
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)

_ENUM_VALUE_CAP = 200
_ENUM_OUTPUT_CAP = 20
_ENUM_MIN_COUNT = 2
_ENUM_MIN_FREQ = 0.03


def _enum_values(
    field_stats: FieldStats,
    *,
    path: str,
    min_conversation_count: int,
    privacy_config: object | None,
) -> list[JSONValue]:
    total = max(field_stats.value_count, 1)
    min_count = _ENUM_MIN_COUNT if total >= 20 else 1
    min_freq = _ENUM_MIN_FREQ if total >= 50 else 0.0
    values: list[JSONValue] = []
    for value, count in field_stats.observed_values.most_common(_ENUM_VALUE_CAP):
        if not _is_safe_enum_value(value, path=path, config=privacy_config):
            continue
        if count < min_count:
            continue
        if min_freq and (count / total) < min_freq:
            continue
        if min_conversation_count > 1 and field_stats.value_conversation_ids:
            conversation_count = len(field_stats.value_conversation_ids.get(value, set()))
            if conversation_count < min_conversation_count:
                continue
        values.append(value)
        if len(values) >= _ENUM_OUTPUT_CAP:
            break
    return values


def _schema_node(value: JSONValue | object) -> JSONDocument | None:
    return value if isinstance(value, dict) else None


@overload
def annotate_schema(
    schema: JSONDocument,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: object | None = None,
) -> JSONDocument: ...


@overload
def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: object | None = None,
) -> JSONValue: ...


def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: object | None = None,
) -> JSONValue:
    """Apply x-polylogue-* annotations to a schema from collected field stats."""
    schema_node = _schema_node(schema)
    if schema_node is None:
        return schema

    field_stats = stats.get(path)
    if field_stats:
        if "[*]" not in path:
            freq = field_stats.frequency
            if 0.0 < freq < 0.95:
                schema_node["x-polylogue-frequency"] = round(freq, 3)

        fmt = field_stats.dominant_format
        if fmt:
            schema_node["x-polylogue-format"] = fmt

        id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        if (
            field_stats.is_enum_like
            and field_stats.observed_values
            and fmt not in id_formats
            and not _is_content_field(path)
        ):
            enum_values = _enum_values(
                field_stats,
                path=path,
                min_conversation_count=min_conversation_count,
                privacy_config=privacy_config,
            )
            if enum_values:
                schema_node["x-polylogue-values"] = enum_values

        if field_stats.num_min is not None and field_stats.num_max is not None:
            schema_node["x-polylogue-range"] = [field_stats.num_min, field_stats.num_max]

        if field_stats.array_lengths:
            schema_node["x-polylogue-array-lengths"] = [
                min(field_stats.array_lengths),
                max(field_stats.array_lengths),
            ]

        if (
            field_stats.is_multiline
            and field_stats.value_count
            and field_stats.is_multiline / field_stats.value_count > 0.3
        ):
            schema_node["x-polylogue-multiline"] = True

        if field_stats.ref_target:
            schema_node["x-polylogue-ref"] = field_stats.ref_target

    properties = _schema_node(schema_node.get("properties"))
    if properties is not None:
        for prop_name, prop_schema in properties.items():
            properties[prop_name] = annotate_schema(
                prop_schema,
                stats,
                f"{path}.{prop_name}",
                min_conversation_count=min_conversation_count,
                privacy_config=privacy_config,
            )

    additional_properties = _schema_node(schema_node.get("additionalProperties"))
    if additional_properties is not None:
        schema_node["additionalProperties"] = annotate_schema(
            additional_properties,
            stats,
            f"{path}.*",
            min_conversation_count=min_conversation_count,
            privacy_config=privacy_config,
        )

    items = _schema_node(schema_node.get("items"))
    if items is not None:
        schema_node["items"] = annotate_schema(
            items,
            stats,
            f"{path}[*]",
            min_conversation_count=min_conversation_count,
            privacy_config=privacy_config,
        )

    for keyword in ("anyOf", "oneOf", "allOf"):
        keyword_items = schema_node.get(keyword)
        if isinstance(keyword_items, Sequence) and not isinstance(keyword_items, (str, bytes, bytearray)):
            schema_node[keyword] = [
                annotate_schema(
                    item,
                    stats,
                    path,
                    min_conversation_count=min_conversation_count,
                    privacy_config=privacy_config,
                )
                for item in keyword_items
            ]

    return schema_node


@overload
def remove_nested_required(schema: JSONDocument, depth: int = 0) -> JSONDocument: ...


@overload
def remove_nested_required(schema: JSONValue, depth: int = 0) -> JSONValue: ...


def remove_nested_required(schema: JSONValue, depth: int = 0) -> JSONValue:
    """Remove nested required declarations from generated schemas."""
    schema_node = _schema_node(schema)
    if schema_node is None:
        return schema

    if depth > 0 and "required" in schema_node:
        del schema_node["required"]

    properties = _schema_node(schema_node.get("properties"))
    if properties is not None:
        for key, prop in properties.items():
            properties[key] = remove_nested_required(prop, depth + 1)

    additional_properties = _schema_node(schema_node.get("additionalProperties"))
    if additional_properties is not None:
        schema_node["additionalProperties"] = remove_nested_required(additional_properties, depth + 1)

    items = _schema_node(schema_node.get("items"))
    if items is not None:
        schema_node["items"] = remove_nested_required(items, depth + 1)

    for keyword in ("anyOf", "oneOf", "allOf"):
        keyword_items = schema_node.get(keyword)
        if isinstance(keyword_items, Sequence) and not isinstance(keyword_items, (str, bytes, bytearray)):
            schema_node[keyword] = [remove_nested_required(item, depth + 1) for item in keyword_items]

    return schema_node


__all__ = [
    "annotate_schema",
    "remove_nested_required",
]
