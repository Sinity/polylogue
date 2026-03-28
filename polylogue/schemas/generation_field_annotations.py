"""Field-stat-driven schema annotation helpers."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)

_ENUM_VALUE_CAP = 200
_ENUM_OUTPUT_CAP = 20
_ENUM_MIN_COUNT = 2
_ENUM_MIN_FREQ = 0.03


def annotate_schema(
    schema: dict[str, Any],
    stats: dict[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: Any | None = None,
) -> dict[str, Any]:
    """Apply x-polylogue-* annotations to a schema from collected field stats."""
    if not isinstance(schema, dict):
        return schema

    field_stats = stats.get(path)
    if field_stats:
        if "[*]" not in path:
            freq = field_stats.frequency
            if 0.0 < freq < 0.95:
                schema["x-polylogue-frequency"] = round(freq, 3)

        fmt = field_stats.dominant_format
        if fmt:
            schema["x-polylogue-format"] = fmt

        id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        if (
            field_stats.is_enum_like
            and field_stats.observed_values
            and fmt not in id_formats
            and not _is_content_field(path)
        ):
            total = max(field_stats.value_count, 1)
            min_count = _ENUM_MIN_COUNT if total >= 20 else 1
            min_freq = _ENUM_MIN_FREQ if total >= 50 else 0.0
            sorted_vals: list[Any] = []
            for value, count in field_stats.observed_values.most_common(_ENUM_VALUE_CAP):
                if isinstance(value, str):
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
                sorted_vals.append(value)
                if len(sorted_vals) >= _ENUM_OUTPUT_CAP:
                    break
            if sorted_vals:
                schema["x-polylogue-values"] = sorted_vals

        if field_stats.num_min is not None and field_stats.num_max is not None:
            schema["x-polylogue-range"] = [field_stats.num_min, field_stats.num_max]

        if field_stats.array_lengths:
            schema["x-polylogue-array-lengths"] = [
                min(field_stats.array_lengths),
                max(field_stats.array_lengths),
            ]

        if (
            field_stats.is_multiline
            and field_stats.value_count
            and field_stats.is_multiline / field_stats.value_count > 0.3
        ):
            schema["x-polylogue-multiline"] = True

        ref_target = getattr(field_stats, "_ref_target", None)
        if ref_target:
            schema["x-polylogue-ref"] = ref_target

    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            schema["properties"][prop_name] = annotate_schema(
                prop_schema,
                stats,
                f"{path}.{prop_name}",
                min_conversation_count=min_conversation_count,
                privacy_config=privacy_config,
            )

    if isinstance(schema.get("additionalProperties"), dict):
        schema["additionalProperties"] = annotate_schema(
            schema["additionalProperties"],
            stats,
            f"{path}.*",
            min_conversation_count=min_conversation_count,
            privacy_config=privacy_config,
        )

    if isinstance(schema.get("items"), dict):
        schema["items"] = annotate_schema(
            schema["items"],
            stats,
            f"{path}[*]",
            min_conversation_count=min_conversation_count,
            privacy_config=privacy_config,
        )

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [
                annotate_schema(
                    item,
                    stats,
                    path,
                    min_conversation_count=min_conversation_count,
                    privacy_config=privacy_config,
                )
                for item in schema[keyword]
            ]

    return schema


def remove_nested_required(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Remove nested required declarations from generated schemas."""
    if not isinstance(schema, dict):
        return schema

    if depth > 0 and "required" in schema:
        del schema["required"]

    if "properties" in schema:
        for key, prop in schema["properties"].items():
            schema["properties"][key] = remove_nested_required(prop, depth + 1)

    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, dict):
        schema["additionalProperties"] = remove_nested_required(additional_properties, depth + 1)

    if "items" in schema:
        schema["items"] = remove_nested_required(schema["items"], depth + 1)

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [remove_nested_required(item, depth + 1) for item in schema[keyword]]

    return schema


__all__ = [
    "annotate_schema",
    "remove_nested_required",
]
