"""Schema generation annotation and shape-normalization helpers."""

from __future__ import annotations

from typing import Any

try:
    from genson import SchemaBuilder

    GENSON_AVAILABLE = True
except ImportError:
    SchemaBuilder = None
    GENSON_AVAILABLE = False

from polylogue.schemas.field_stats import FieldStats, is_dynamic_key
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)
from polylogue.schemas.relational_inference import infer_relations
from polylogue.schemas.semantic_inference import infer_semantic_roles, select_best_roles

_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35
_ENUM_VALUE_CAP = 200
_ENUM_OUTPUT_CAP = 20
_ENUM_MIN_COUNT = 2
_ENUM_MIN_FREQ = 0.03


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


def collapse_dynamic_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse dynamic key properties into additionalProperties."""
    if not isinstance(schema, dict):
        return schema

    if "properties" in schema:
        props = schema["properties"]
        static_props: dict[str, Any] = {}
        dynamic_schemas: list[dict[str, Any]] = []
        key_names = list(props.keys())

        for key, value in props.items():
            collapsed_value = collapse_dynamic_keys(value)
            if is_dynamic_key(key):
                dynamic_schemas.append(collapsed_value)
            else:
                static_props[key] = collapsed_value

        if _should_collapse_high_cardinality_keys(key_names):
            dynamic_schemas.extend(static_props.values())
            static_props = {}
            schema["x-polylogue-high-cardinality-keys"] = True

        if dynamic_schemas:
            schema["properties"] = static_props
            schema["additionalProperties"] = _merge_schemas(dynamic_schemas)
            schema["x-polylogue-dynamic-keys"] = True
            if "required" in schema:
                schema["required"] = [r for r in schema["required"] if r in static_props]
        else:
            schema["properties"] = static_props

    if "items" in schema:
        schema["items"] = collapse_dynamic_keys(schema["items"])

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [collapse_dynamic_keys(s) for s in schema[keyword]]

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = collapse_dynamic_keys(schema["additionalProperties"])

    return schema


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple schemas into one using genson."""
    if not GENSON_AVAILABLE:
        return schemas[0] if schemas else {}
    builder = SchemaBuilder()
    for schema in schemas:
        builder.add_schema(schema)
    return dict(builder.to_schema())


def _annotate_schema(
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

        _id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        if (
            field_stats.is_enum_like
            and field_stats.observed_values
            and fmt not in _id_formats
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
                        n_convs = len(field_stats.value_conversation_ids.get(value, set()))
                        if n_convs < min_conversation_count:
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

        if field_stats.is_multiline and field_stats.value_count and field_stats.is_multiline / field_stats.value_count > 0.3:
            schema["x-polylogue-multiline"] = True

        ref_target = getattr(field_stats, "_ref_target", None)
        if ref_target:
            schema["x-polylogue-ref"] = ref_target

    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            schema["properties"][prop_name] = _annotate_schema(
                prop_schema,
                stats,
                f"{path}.{prop_name}",
                min_conversation_count=min_conversation_count,
                privacy_config=privacy_config,
            )

    if isinstance(schema.get("additionalProperties"), dict):
        schema["additionalProperties"] = _annotate_schema(
            schema["additionalProperties"],
            stats,
            f"{path}.*",
            min_conversation_count=min_conversation_count,
        )

    if isinstance(schema.get("items"), dict):
        schema["items"] = _annotate_schema(
            schema["items"],
            stats,
            f"{path}[*]",
            min_conversation_count=min_conversation_count,
        )

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [
                _annotate_schema(s, stats, path, min_conversation_count=min_conversation_count)
                for s in schema[key]
            ]

    return schema


def _annotate_semantic_and_relational(
    schema: dict[str, Any],
    field_stats: dict[str, FieldStats],
) -> dict[str, Any]:
    candidates = infer_semantic_roles(field_stats)
    best_roles = select_best_roles(candidates)
    role_by_path: dict[str, tuple[str, float, dict]] = {}
    for role, candidate in best_roles.items():
        role_by_path[candidate.path] = (role, candidate.confidence, candidate.evidence)

    def _attach_semantic(s: dict[str, Any], path: str = "$") -> None:
        if not isinstance(s, dict):
            return
        if path in role_by_path:
            role, confidence, evidence = role_by_path[path]
            s["x-polylogue-semantic-role"] = role
            s["x-polylogue-confidence"] = round(confidence, 3)
            s["x-polylogue-evidence"] = evidence

        if "properties" in s:
            for prop_name, prop_schema in s["properties"].items():
                _attach_semantic(prop_schema, f"{path}.{prop_name}")
        if isinstance(s.get("additionalProperties"), dict):
            _attach_semantic(s["additionalProperties"], f"{path}.*")
        if isinstance(s.get("items"), dict):
            _attach_semantic(s["items"], f"{path}[*]")
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in s:
                for sub in s[keyword]:
                    _attach_semantic(sub, path)

    _attach_semantic(schema)
    relations = infer_relations(field_stats)

    if relations.foreign_keys:
        schema["x-polylogue-foreign-keys"] = [
            {
                "source": fk.source_path,
                "target": fk.target_path,
                "match_ratio": round(fk.match_ratio, 3),
            }
            for fk in relations.foreign_keys
        ]

    if relations.time_deltas:
        schema["x-polylogue-time-deltas"] = [
            {
                "field_a": td.field_a,
                "field_b": td.field_b,
                "min_delta": round(td.min_delta, 1),
                "max_delta": round(td.max_delta, 1),
                "avg_delta": round(td.avg_delta, 1),
            }
            for td in relations.time_deltas
        ]

    if relations.mutual_exclusions:
        schema["x-polylogue-mutually-exclusive"] = [
            {
                "parent": me.parent_path,
                "fields": sorted(me.field_names),
            }
            for me in relations.mutual_exclusions
        ]

    if relations.string_lengths:
        schema["x-polylogue-string-lengths"] = [
            {
                "path": sl.path,
                "min": sl.min_length,
                "max": sl.max_length,
                "avg": round(sl.avg_length, 1),
                "stddev": round(sl.stddev, 1),
            }
            for sl in relations.string_lengths
        ]

    return schema


def _remove_nested_required(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    if depth > 0 and "required" in schema:
        del schema["required"]

    if "properties" in schema:
        for key, prop in schema["properties"].items():
            schema["properties"][key] = _remove_nested_required(prop, depth + 1)

    ap = schema.get("additionalProperties")
    if isinstance(ap, dict):
        schema["additionalProperties"] = _remove_nested_required(ap, depth + 1)

    if "items" in schema:
        schema["items"] = _remove_nested_required(schema["items"], depth + 1)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_remove_nested_required(s, depth + 1) for s in schema[key]]

    return schema


__all__ = [
    "GENSON_AVAILABLE",
    "_annotate_schema",
    "_annotate_semantic_and_relational",
    "_merge_schemas",
    "_remove_nested_required",
    "collapse_dynamic_keys",
]
