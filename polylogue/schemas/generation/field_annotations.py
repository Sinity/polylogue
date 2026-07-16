"""Field-stat-driven schema annotation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import overload

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)
from polylogue.schemas.privacy_config import SchemaPrivacyConfig

_ENUM_VALUE_CAP = 200
_ENUM_OUTPUT_CAP = 20
_ENUM_MIN_COUNT = 2
_ENUM_MIN_FREQ = 0.03

# Semantic roles whose enum values are structural identifiers
# (e.g. record types, message roles).  These fields bypass the
# cross-session privacy threshold because their values are
# protocol-level tokens, not user content.
_STRUCTURAL_ROLE_VALUES = frozenset({"message_role"})


def _enum_values(
    field_stats: FieldStats,
    *,
    path: str,
    min_session_count: int,
    privacy_config: SchemaPrivacyConfig | None,
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
        if min_session_count > 1 and field_stats.value_session_ids:
            session_count = len(field_stats.value_session_ids.get(value, set()))
            if session_count < min_session_count:
                continue
        values.append(value)
        if len(values) >= _ENUM_OUTPUT_CAP:
            break
    return values


def _schema_node(value: JSONValue | object) -> JSONDocument | None:
    return value if isinstance(value, dict) else None


def _observed_distribution_payload(field_stats: FieldStats) -> JSONDocument:
    payload: JSONDocument = {
        "documents": field_stats.total_samples,
        "encountered_documents": field_stats.document_encountered_count,
        "non_null_documents": field_stats.document_non_null_count,
        "null_observations": field_stats.null_count,
        "missing_documents": max(0, field_stats.total_samples - field_stats.document_encountered_count),
        "type_counts": dict(sorted(field_stats.type_counts.items())),
    }
    distributions = {
        "numeric": field_stats.numeric_distribution,
        "string_length": field_stats.string_length_distribution,
        "newline_count": field_stats.newline_distribution,
        "array_length": field_stats.array_length_distribution,
        "object_fanout": field_stats.object_fanout_distribution,
    }
    for name, distribution in distributions.items():
        if distribution.count or distribution.non_finite_count:
            payload[name] = distribution.to_payload()
    if field_stats.ordered_pair_count:
        payload["ordered_numeric_pairs"] = {
            "count": field_stats.ordered_pair_count,
            "nondecreasing_count": field_stats.ordered_increasing_pair_count,
            "nondecreasing_rate": field_stats.monotonicity_score,
        }
    if field_stats.co_occurring_fields:
        payload["co_occurring_fields"] = dict(sorted(field_stats.co_occurring_fields.items()))
    if field_stats.categorical_distribution.count:
        payload["categorical"] = field_stats.categorical_distribution.to_payload()
    if field_stats.object_key_distribution.count:
        payload["object_keys"] = field_stats.object_key_distribution.to_payload()
    if field_stats.co_occurrence_distribution.count:
        payload["co_occurrence_hashes"] = field_stats.co_occurrence_distribution.to_payload()
    if field_stats.boolean_counts:
        payload["boolean_counts"] = dict(sorted(field_stats.boolean_counts.items()))
    if field_stats.detected_formats:
        payload["format_counts"] = dict(sorted(field_stats.detected_formats.items()))
    if field_stats.truncated_evidence:
        payload["loss_inventory"] = dict(sorted(field_stats.truncated_evidence.items()))
    return payload


@overload
def annotate_schema(
    schema: JSONDocument,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_session_count: int = 1,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONDocument: ...


@overload
def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_session_count: int = 1,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONValue: ...


def annotate_schema(
    schema: JSONValue,
    stats: Mapping[str, FieldStats],
    path: str = "$",
    *,
    min_session_count: int = 1,
    privacy_config: SchemaPrivacyConfig | None = None,
) -> JSONValue:
    """Apply x-polylogue-* annotations to a schema from collected field stats."""
    schema_node = _schema_node(schema)
    if schema_node is None:
        return schema

    field_stats = stats.get(path)
    if field_stats:
        schema_node["x-polylogue-observed-distribution"] = _observed_distribution_payload(field_stats)
        freq = field_stats.document_frequency
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
            # Structural fields (e.g. message_role) bypass the
            # cross-session privacy threshold — their values
            # are protocol-level tokens, not user content.
            sem_role = schema_node.get("x-polylogue-semantic-role")
            effective_min_conv = (
                1 if isinstance(sem_role, str) and sem_role in _STRUCTURAL_ROLE_VALUES else min_session_count
            )
            enum_values = _enum_values(
                field_stats,
                path=path,
                min_session_count=effective_min_conv,
                privacy_config=privacy_config,
            )
            if enum_values:
                schema_node["x-polylogue-values"] = enum_values

        if field_stats.num_min is not None and field_stats.num_max is not None:
            schema_node["x-polylogue-range"] = [field_stats.num_min, field_stats.num_max]

        if field_stats.array_length_distribution.count:
            minimum = field_stats.array_length_distribution.minimum
            maximum = field_stats.array_length_distribution.maximum
            if minimum is not None and maximum is not None:
                schema_node["x-polylogue-array-lengths"] = [int(minimum), int(maximum)]
        elif field_stats.array_lengths:
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
                min_session_count=min_session_count,
                privacy_config=privacy_config,
            )

    additional_properties = _schema_node(schema_node.get("additionalProperties"))
    if additional_properties is not None:
        schema_node["additionalProperties"] = annotate_schema(
            additional_properties,
            stats,
            f"{path}.*",
            min_session_count=min_session_count,
            privacy_config=privacy_config,
        )

    items = _schema_node(schema_node.get("items"))
    if items is not None:
        schema_node["items"] = annotate_schema(
            items,
            stats,
            f"{path}[*]",
            min_session_count=min_session_count,
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
                    min_session_count=min_session_count,
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
