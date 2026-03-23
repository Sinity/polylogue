"""Schema generation from provider data samples.

Provides functions to build and annotate JSON schemas from raw data samples.
Uses genson for structural inference and adds x-polylogue-* annotations from
field statistics.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from genson import SchemaBuilder
    GENSON_AVAILABLE = True
except ImportError:
    GENSON_AVAILABLE = False

from polylogue.paths import db_path as default_db_path
from polylogue.schemas.field_stats import (
    UUID_PATTERN,
    FieldStats,
    _collect_field_stats,
    is_dynamic_key,
)
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaVersionPackage,
)
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
    _looks_high_entropy_token,
)
from polylogue.schemas.redaction_report import (
    FieldReport,
    RedactionDecision,
    SchemaReport,
)
from polylogue.schemas.registry import ClusterManifest, SchemaCluster, SchemaRegistry
from polylogue.schemas.relational_inference import (
    infer_relations,
)
from polylogue.schemas.sampling import (
    PROVIDERS,
    ProviderConfig,
    SchemaUnit,
    _resolve_provider_config,
    iter_schema_units,
    profile_cluster_id,
    profile_similarity,
)
from polylogue.schemas.semantic_inference import (
    infer_semantic_roles,
    select_best_roles,
)
from polylogue.schemas.shape_fingerprint import _structure_fingerprint
from polylogue.types import Provider

_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35
_STRUCTURE_EXEMPLARS_PER_FINGERPRINT = 8
_PROFILE_CORE_MIN_RATIO = 0.5
_PROFILE_MAX_TOKENS = 128
_PROFILE_SIMILARITY_THRESHOLDS = {
    "conversation_document": 0.86,
    "conversation_record_stream": 0.8,
    "subagent_conversation_stream": 0.8,
}
_ANCHOR_ELEMENT_KINDS = {
    "conversation_document",
    "conversation_record_stream",
}

# Thresholds used by _annotate_schema
_ENUM_VALUE_CAP = 200  # max values to iterate per field in annotation
_ENUM_OUTPUT_CAP = 20  # max values to emit in x-polylogue-values (schema size bound)
_ENUM_MIN_COUNT = 2  # suppress one-off values on large corpora (privacy + stability)
_ENUM_MIN_FREQ = 0.03  # suppress extremely rare enum values on large corpora


@dataclass
class GenerationResult:
    """Result of schema generation."""

    provider: str
    schema: dict[str, Any] | None
    sample_count: int
    error: str | None = None
    redaction_report: Any | None = None  # SchemaReport when privacy tracking enabled
    versions: list[str] = field(default_factory=list)
    default_version: str | None = None
    cluster_count: int = 0
    package_count: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.schema is not None and self.error is None


@dataclass
class _ProviderBundle:
    result: GenerationResult
    catalog: SchemaPackageCatalog | None = None
    package_schemas: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    manifest: ClusterManifest | None = None


# =============================================================================
# Schema Utilities
# =============================================================================


def _looks_pathlike_key(key: str) -> bool:
    """Heuristic for key names that likely encode file/user-specific paths."""
    if "/" in key or "\\" in key:
        return True
    if key.count(".") >= 2:
        return True
    return ":" in key and len(key) > 2


def _should_collapse_high_cardinality_keys(keys: list[str]) -> bool:
    """Collapse key-heavy objects that are likely data maps, not fixed schemas."""
    if len(keys) >= _HIGH_CARDINALITY_KEY_THRESHOLD:
        return True
    if len(keys) < 24:
        return False
    pathlike = sum(1 for key in keys if _looks_pathlike_key(key))
    return (pathlike / len(keys)) >= _PATHLIKE_KEY_RATIO_THRESHOLD


def collapse_dynamic_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse dynamic key properties into additionalProperties.

    Handles objects like ChatGPT's `mapping` where keys are UUIDs.
    Also handles high-cardinality objects with path-like keys (file trees).
    Static keys are preserved in `properties`; dynamic keys are merged into
    `additionalProperties`.
    """
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
            merged = _merge_schemas(dynamic_schemas)
            schema["additionalProperties"] = merged
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
# =============================================================================
# Field Statistics & Schema Annotations
# =============================================================================


def _annotate_schema(
    schema: dict[str, Any],
    stats: dict[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
    privacy_config: Any | None = None,
) -> dict[str, Any]:
    """Apply x-polylogue-* annotations to a schema based on collected field stats.

    Walks the schema tree in parallel with the stats paths. Adds:
    - x-polylogue-values: observed enum values for low-cardinality string fields
    - x-polylogue-format: detected string/numeric format
    - x-polylogue-range: [min, max] for numeric fields
    - x-polylogue-frequency: field presence frequency (omitted if ~1.0)
    - x-polylogue-ref: JSON path this field references
    - x-polylogue-array-lengths: [min, max] for array fields
    - x-polylogue-multiline: true for fields with multiline content
    """
    if not isinstance(schema, dict):
        return schema

    field_stats = stats.get(path)

    # Annotate this node
    if field_stats:
        # Frequency annotation (omit if always present)
        # Skip for array item paths — denominator is item count, not sample count
        if "[*]" not in path:
            freq = field_stats.frequency
            if 0.0 < freq < 0.95:
                schema["x-polylogue-frequency"] = round(freq, 3)

        # Format detection (checked first — format suppresses enum for ID-like fields)
        fmt = field_stats.dominant_format
        if fmt:
            schema["x-polylogue-format"] = fmt

        # String enum values (skip if format indicates generated/unique values,
        # or if the field is known to contain user content)
        _id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        if (field_stats.is_enum_like
                and field_stats.observed_values
                and fmt not in _id_formats
                and not _is_content_field(path)):
            # Top values by frequency — cap output, filter out content/PII.
            # On larger datasets, require repeat observations to avoid leaking
            # one-off values while retaining structural enums.
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
                    # Cross-conversation privacy threshold: suppress values seen in
                    # fewer than min_conversation_count distinct conversations.
                    # Only enforced when conversation tracking is available.
                    if (
                        min_conversation_count > 1
                        and field_stats.value_conversation_ids
                    ):
                        n_convs = len(field_stats.value_conversation_ids.get(value, set()))
                        if n_convs < min_conversation_count:
                            continue
                sorted_vals.append(value)
                if len(sorted_vals) >= _ENUM_OUTPUT_CAP:
                    break
            if sorted_vals:
                schema["x-polylogue-values"] = sorted_vals

        # Numeric range
        if field_stats.num_min is not None and field_stats.num_max is not None:
            schema["x-polylogue-range"] = [field_stats.num_min, field_stats.num_max]

        # Array length range
        if field_stats.array_lengths:
            schema["x-polylogue-array-lengths"] = [
                min(field_stats.array_lengths),
                max(field_stats.array_lengths),
            ]

        # Multiline content
        if field_stats.is_multiline and field_stats.value_count and field_stats.is_multiline / field_stats.value_count > 0.3:
                schema["x-polylogue-multiline"] = True

        # Reference detection
        ref_target = getattr(field_stats, "_ref_target", None)
        if ref_target:
            schema["x-polylogue-ref"] = ref_target

    # Recurse into properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            schema["properties"][prop_name] = _annotate_schema(
                prop_schema, stats, f"{path}.{prop_name}",
                min_conversation_count=min_conversation_count,
                privacy_config=privacy_config,
            )

    # Recurse into additionalProperties
    if isinstance(schema.get("additionalProperties"), dict):
        schema["additionalProperties"] = _annotate_schema(
            schema["additionalProperties"], stats, f"{path}.*",
            min_conversation_count=min_conversation_count,
        )

    # Recurse into array items
    if isinstance(schema.get("items"), dict):
        schema["items"] = _annotate_schema(
            schema["items"], stats, f"{path}[*]",
            min_conversation_count=min_conversation_count,
        )

    # Recurse into anyOf/oneOf/allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [
                _annotate_schema(s, stats, path, min_conversation_count=min_conversation_count)
                for s in schema[key]
            ]

    return schema


def _build_redaction_report(
    provider: str,
    stats: dict[str, FieldStats],
    schema: dict[str, Any],
    *,
    privacy_config: Any | None = None,
    privacy_level: str = "standard",
) -> SchemaReport:
    """Build a redaction report by comparing field stats against the annotated schema.

    Walks field stats and for each enum-like field, checks which values ended up
    in x-polylogue-values vs. which were rejected. Non-invasive: runs after
    _annotate_schema has already completed.
    """
    report = SchemaReport(provider=provider, privacy_level=privacy_level)

    # Collect all x-polylogue-values from the final schema
    schema_values: dict[str, set[str]] = {}

    def _collect_schema_values(s: dict[str, Any], path: str = "$") -> None:
        if not isinstance(s, dict):
            return
        vals = s.get("x-polylogue-values")
        if isinstance(vals, list):
            schema_values[path] = {str(v) for v in vals}
        if "properties" in s:
            for name, prop in s["properties"].items():
                _collect_schema_values(prop, f"{path}.{name}")
        if isinstance(s.get("additionalProperties"), dict):
            _collect_schema_values(s["additionalProperties"], f"{path}.*")
        if isinstance(s.get("items"), dict):
            _collect_schema_values(s["items"], f"{path}[*]")
        for kw in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(kw, []):
                if isinstance(sub, dict):
                    _collect_schema_values(sub, path)

    _collect_schema_values(schema)

    # Walk field stats and compare
    for path, fs in stats.items():
        if not fs.is_enum_like or not fs.observed_values:
            continue

        report.total_fields += 1
        report.fields_with_enums += 1
        included_in_schema = schema_values.get(path, set())
        field_report = FieldReport(path=path)

        if _is_content_field(path):
            field_report.content_field_blocked = True

        for value, count in fs.observed_values.most_common():
            if not isinstance(value, str):
                continue
            report.total_values_considered += 1

            if value in included_in_schema:
                decision = RedactionDecision(
                    path=path, value=value, action="included", count=count,
                )
                field_report.included_values.append(value)
            else:
                # Determine rejection reason
                reason = "unknown"
                if _is_content_field(path):
                    reason = "content_field"
                elif _looks_high_entropy_token(value):
                    reason = "high_entropy"
                elif not _is_safe_enum_value(value, path=path, config=privacy_config):
                    reason = "unsafe_value"
                else:
                    reason = "threshold"  # cross-conv or min_count

                decision = RedactionDecision(
                    path=path, value=value, action="rejected",
                    reason=reason, count=count,
                )
                field_report.rejected.append(decision)

                # Track borderline decisions (high count but rejected)
                if count >= 100:
                    decision.risk = "medium"
                    report.borderline_decisions.append(decision)

            report.add_decision(decision)

        if field_report.included_values or field_report.rejected:
            report.field_reports.append(field_report)

    return report


def _annotate_semantic_and_relational(
    schema: dict[str, Any],
    field_stats: dict[str, FieldStats],
) -> dict[str, Any]:
    """Attach semantic role and relational annotations to a schema.

    This is called after ``_annotate_schema()`` to layer on semantic and
    relational intelligence without modifying the base annotation pass.

    Semantic annotations are attached at the field level:
        - x-polylogue-semantic-role
        - x-polylogue-confidence
        - x-polylogue-evidence

    Relational annotations are attached at the schema root:
        - x-polylogue-foreign-keys
        - x-polylogue-time-deltas
        - x-polylogue-mutually-exclusive
        - x-polylogue-string-lengths
    """
    # --- Semantic roles ---
    candidates = infer_semantic_roles(field_stats)
    best_roles = select_best_roles(candidates)

    # Map role assignments back to their schema paths
    role_by_path: dict[str, tuple[str, float, dict]] = {}
    for role, candidate in best_roles.items():
        role_by_path[candidate.path] = (role, candidate.confidence, candidate.evidence)

    # Walk schema and attach semantic annotations
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

    # --- Relational annotations ---
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


# =============================================================================
# Schema Generation
# =============================================================================


def _remove_nested_required(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Remove 'required' arrays from nested objects.

    Genson marks fields as required if they appear in all samples, but this
    is too strict for real data where fields can be optional. We keep top-level
    required (depth=0) but remove from nested objects.
    """
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


@dataclass
class _ProfileSummary:
    artifact_kind: str
    profile_tokens: tuple[str, ...]
    dominant_keys: list[str]
    sample_count: int = 0
    schema_sample_count: int = 0
    representative_paths: list[str] = field(default_factory=list)


@dataclass
class _ClusterAccumulator:
    artifact_kind: str
    dominant_keys: list[str]
    sample_count: int = 0  # number of clustered units (documents or raw streams)
    schema_sample_count: int = 0  # number of individual samples contributing to the schema
    representative_paths: list[str] = field(default_factory=list)
    profile_token_counts: Counter[str] = field(default_factory=Counter)
    member_profiles: set[tuple[str, ...]] = field(default_factory=set)
    reservoir_samples: list[dict[str, Any]] = field(default_factory=list)
    reservoir_conv_ids: list[str | None] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(42))
    exact_structure_ids: set[str] = field(default_factory=set)
    bundle_scopes: set[str] = field(default_factory=set)
    first_seen: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True)
class _UnitMembership:
    unit: SchemaUnit
    profile_family_id: str


@dataclass
class _PackageAccumulator:
    provider: str
    anchor_family_id: str
    anchor_kind: str
    memberships: list[_UnitMembership] = field(default_factory=list)
    bundle_scopes: set[str] = field(default_factory=set)
    representative_paths: list[str] = field(default_factory=list)
    profile_family_ids: set[str] = field(default_factory=set)
    first_seen: str | None = None
    last_seen: str | None = None


def _artifact_priority(artifact_kind: str) -> int:
    priorities = {
        "conversation_document": 120,
        "conversation_record_stream": 120,
        "subagent_conversation_stream": 90,
    }
    return priorities.get(artifact_kind, 0)


def _dominant_keys_for_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        return sorted(payload.keys())
    if isinstance(payload, list):
        first_dict = next((item for item in payload if isinstance(item, dict)), None)
        if isinstance(first_dict, dict):
            return sorted(first_dict.keys())
    return []


def _cluster_sort_key(item: tuple[str, _ClusterAccumulator]) -> tuple[int, int, int]:
    _cluster_id, acc = item
    return (_artifact_priority(acc.artifact_kind), acc.sample_count, acc.schema_sample_count)


def _parse_observed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _update_observed_window(acc: _ClusterAccumulator | _PackageAccumulator, observed_at: str | None) -> None:
    parsed = _parse_observed_at(observed_at)
    if parsed is None:
        return
    iso = parsed.astimezone(timezone.utc).isoformat()
    if acc.first_seen is None or _parse_observed_at(acc.first_seen) is None or parsed < _parse_observed_at(acc.first_seen):
        acc.first_seen = iso
    if acc.last_seen is None or _parse_observed_at(acc.last_seen) is None or parsed > _parse_observed_at(acc.last_seen):
        acc.last_seen = iso


def _cluster_reservoir_size(config: ProviderConfig, max_samples: int | None) -> int:
    if max_samples is not None:
        return max(64, min(max_samples, 500))
    if config.sample_granularity == "record":
        return 192
    return 500


def _profile_similarity_threshold(artifact_kind: str) -> float:
    return _PROFILE_SIMILARITY_THRESHOLDS.get(artifact_kind, 0.84)


def _cluster_profile_tokens(acc: _ClusterAccumulator) -> tuple[str, ...]:
    token_counts = getattr(acc, "profile_token_counts", None)
    if not token_counts:
        dominant_keys = getattr(acc, "dominant_keys", ())
        return tuple(f"field:{key}" for key in dominant_keys[:_PROFILE_MAX_TOKENS])

    min_count = 1 if acc.sample_count <= 2 else max(2, int(acc.sample_count * _PROFILE_CORE_MIN_RATIO))

    tokens = sorted(
        token
        for token, count in token_counts.items()
        if count >= min_count
    )
    if not tokens:
        tokens = [
            token
            for token, _count in sorted(
                token_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_PROFILE_MAX_TOKENS]
        ]
    return tuple(tokens[:_PROFILE_MAX_TOKENS])


def _cluster_similarity(acc: _ClusterAccumulator, profile_tokens: tuple[str, ...]) -> float:
    if not profile_tokens:
        return 0.0
    return profile_similarity(set(_cluster_profile_tokens(acc)), set(profile_tokens))


def _merge_representative_paths(target: list[str], source: list[str]) -> None:
    for path in source:
        if path not in target and len(target) < 5:
            target.append(path)


def _membership_observed_window(memberships: list[_UnitMembership]) -> tuple[str, str] | tuple[None, None]:
    first_seen: str | None = None
    last_seen: str | None = None
    for membership in memberships:
        parsed = _parse_observed_at(membership.unit.observed_at)
        if parsed is None:
            continue
        iso = parsed.astimezone(timezone.utc).isoformat()
        if first_seen is None or parsed < _parse_observed_at(first_seen):
            first_seen = iso
        if last_seen is None or parsed > _parse_observed_at(last_seen):
            last_seen = iso
    return first_seen, last_seen


def _merge_dominant_keys(target: list[str], source: list[str]) -> list[str]:
    merged = list(dict.fromkeys([*target, *source]))
    return merged[:20]


def _new_cluster_accumulator(
    *,
    artifact_kind: str,
    dominant_keys: list[str],
) -> _ClusterAccumulator:
    return _ClusterAccumulator(
        artifact_kind=artifact_kind,
        dominant_keys=dominant_keys[:20],
    )


def _merge_profile_summary(
    acc: _ClusterAccumulator,
    summary: _ProfileSummary,
) -> None:
    acc.sample_count += summary.sample_count
    acc.schema_sample_count += summary.schema_sample_count
    acc.profile_token_counts.update(dict.fromkeys(summary.profile_tokens, summary.sample_count))
    acc.member_profiles.add(summary.profile_tokens)
    acc.dominant_keys = _merge_dominant_keys(acc.dominant_keys, summary.dominant_keys)
    _merge_representative_paths(acc.representative_paths, summary.representative_paths)


def _merge_cluster_accumulators(
    target: _ClusterAccumulator,
    source: _ClusterAccumulator,
    *,
    reservoir_size: int | None = None,
) -> None:
    target.sample_count += source.sample_count
    target.schema_sample_count += source.schema_sample_count
    target.profile_token_counts.update(source.profile_token_counts)
    target.member_profiles.update(source.member_profiles)
    target.dominant_keys = _merge_dominant_keys(target.dominant_keys, source.dominant_keys)
    _merge_representative_paths(target.representative_paths, source.representative_paths)
    target.exact_structure_ids.update(source.exact_structure_ids)
    target.bundle_scopes.update(source.bundle_scopes)
    _update_observed_window(target, source.first_seen)
    _update_observed_window(target, source.last_seen)
    if reservoir_size is not None and source.reservoir_samples:
        combined = list(zip(target.reservoir_samples, target.reservoir_conv_ids, strict=False))
        combined.extend(zip(source.reservoir_samples, source.reservoir_conv_ids, strict=False))
        if len(combined) > reservoir_size:
            target.rng.shuffle(combined)
            combined = combined[:reservoir_size]
        target.reservoir_samples = [sample for sample, _conv_id in combined]
        target.reservoir_conv_ids = [conv_id for _sample, conv_id in combined]


def _refine_coarse_clusters(
    coarse_clusters: list[_ClusterAccumulator],
    *,
    reservoir_size: int | None = None,
) -> list[_ClusterAccumulator]:
    clusters = list(coarse_clusters)
    while True:
        best_pair: tuple[int, int, float] | None = None
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                if left.artifact_kind != right.artifact_kind:
                    continue
                score = profile_similarity(
                    set(_cluster_profile_tokens(left)),
                    set(_cluster_profile_tokens(right)),
                )
                if score < _profile_similarity_threshold(left.artifact_kind):
                    continue
                if best_pair is None or score > best_pair[2]:
                    best_pair = (left_index, right_index, score)

        if best_pair is None:
            return clusters

        left_index, right_index, _score = best_pair
        left = clusters[left_index]
        right = clusters[right_index]
        if (
            right.sample_count > left.sample_count
            or (
                right.sample_count == left.sample_count
                and right.schema_sample_count > left.schema_sample_count
            )
        ):
            left_index, right_index = right_index, left_index
            left, right = right, left

        _merge_cluster_accumulators(left, right, reservoir_size=reservoir_size)
        del clusters[right_index]


def _update_cluster_reservoir(
    acc: _ClusterAccumulator,
    sample: dict[str, Any],
    conversation_id: str | None,
    *,
    reservoir_size: int,
) -> None:
    acc.schema_sample_count += 1
    if len(acc.reservoir_samples) < reservoir_size:
        acc.reservoir_samples.append(sample)
        acc.reservoir_conv_ids.append(conversation_id)
        return

    slot = acc.rng.randint(0, acc.schema_sample_count - 1)
    if slot < reservoir_size:
        acc.reservoir_samples[slot] = sample
        acc.reservoir_conv_ids[slot] = conversation_id


def _collect_cluster_accumulators(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
) -> tuple[dict[str, _ClusterAccumulator], list[_UnitMembership], int, dict[str, int]]:
    """Cluster schema units into profile families and retain bundle memberships."""
    units = list(iter_schema_units(provider, db_path=db_path, max_samples=max_samples))
    profile_summaries: dict[tuple[str, tuple[str, ...]], _ProfileSummary] = {}
    total_schema_samples = 0
    artifact_counts: dict[str, int] = {}

    for unit in units:
        summary_key = (unit.artifact_kind, unit.profile_tokens)
        summary = profile_summaries.get(summary_key)
        if summary is None:
            summary = _ProfileSummary(
                artifact_kind=unit.artifact_kind,
                profile_tokens=unit.profile_tokens,
                dominant_keys=_dominant_keys_for_payload(unit.cluster_payload)[:20],
            )
            profile_summaries[summary_key] = summary

        summary.sample_count += 1
        summary.schema_sample_count += len(unit.schema_samples)
        artifact_counts[unit.artifact_kind] = artifact_counts.get(unit.artifact_kind, 0) + 1

        if (
            unit.source_path
            and unit.source_path not in summary.representative_paths
            and len(summary.representative_paths) < 5
        ):
            summary.representative_paths.append(unit.source_path)

        total_schema_samples += len(unit.schema_samples)

    coarse_clusters: list[_ClusterAccumulator] = []
    ordered_summaries = sorted(
        profile_summaries.items(),
        key=lambda item: (
            _artifact_priority(item[1].artifact_kind),
            item[1].sample_count,
            item[1].schema_sample_count,
        ),
        reverse=True,
    )

    for _summary_key, summary in ordered_summaries:
        best_index: int | None = None
        best_score = 0.0
        for index, acc in enumerate(coarse_clusters):
            if acc.artifact_kind != summary.artifact_kind:
                continue
            score = _cluster_similarity(acc, summary.profile_tokens)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= _profile_similarity_threshold(summary.artifact_kind):
            _merge_profile_summary(coarse_clusters[best_index], summary)
            continue

        acc = _new_cluster_accumulator(
            artifact_kind=summary.artifact_kind,
            dominant_keys=summary.dominant_keys,
        )
        _merge_profile_summary(acc, summary)
        coarse_clusters.append(acc)
    coarse_clusters = _refine_coarse_clusters(coarse_clusters)

    clusters: dict[str, _ClusterAccumulator] = {}
    for acc in coarse_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        existing = clusters.get(cluster_id)
        if existing is None:
            clusters[cluster_id] = _new_cluster_accumulator(
                artifact_kind=acc.artifact_kind,
                dominant_keys=acc.dominant_keys,
            )
            clusters[cluster_id].representative_paths = list(acc.representative_paths)
            clusters[cluster_id].profile_token_counts.update(acc.profile_token_counts)
        else:
            existing.dominant_keys = _merge_dominant_keys(existing.dominant_keys, acc.dominant_keys)
            _merge_representative_paths(existing.representative_paths, acc.representative_paths)
            existing.profile_token_counts.update(acc.profile_token_counts)

    summary_cluster_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    for acc in coarse_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        for profile_tokens in acc.member_profiles:
            summary_cluster_ids[(acc.artifact_kind, profile_tokens)] = cluster_id

    memberships: list[_UnitMembership] = []
    for unit in units:
        cluster_id = summary_cluster_ids[(unit.artifact_kind, unit.profile_tokens)]
        acc = clusters[cluster_id]
        acc.sample_count += 1
        acc.schema_sample_count += len(unit.schema_samples)
        acc.profile_token_counts.update(unit.profile_tokens)
        acc.member_profiles.add(unit.profile_tokens)
        acc.exact_structure_ids.add(unit.exact_structure_id)
        if unit.bundle_scope:
            acc.bundle_scopes.add(unit.bundle_scope)
        _update_observed_window(acc, unit.observed_at)
        if (
            unit.source_path
            and unit.source_path not in acc.representative_paths
            and len(acc.representative_paths) < 5
        ):
            acc.representative_paths.append(unit.source_path)
        memberships.append(_UnitMembership(unit=unit, profile_family_id=cluster_id))

    refined_clusters = _refine_coarse_clusters(
        list(clusters.values()),
        reservoir_size=reservoir_size,
    )
    final_clusters: dict[str, _ClusterAccumulator] = {}
    for acc in refined_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        existing = final_clusters.get(cluster_id)
        if existing is None:
            final_clusters[cluster_id] = acc
        else:
            _merge_cluster_accumulators(existing, acc, reservoir_size=reservoir_size)

    membership_cluster_map: dict[str, str] = {}
    for cluster_id, acc in final_clusters.items():
        for profile_tokens in acc.member_profiles:
            membership_cluster_map[profile_cluster_id(acc.artifact_kind, profile_tokens)] = cluster_id

    normalized_memberships: list[_UnitMembership] = []
    for membership in memberships:
        normalized_cluster_id = membership_cluster_map.get(membership.profile_family_id, membership.profile_family_id)
        normalized_memberships.append(
            _UnitMembership(unit=membership.unit, profile_family_id=normalized_cluster_id)
        )
        final_acc = final_clusters[normalized_cluster_id]
        for sample in membership.unit.schema_samples:
            _update_cluster_reservoir(
                final_acc,
                sample,
                membership.unit.conversation_id,
                reservoir_size=reservoir_size,
            )

    return final_clusters, normalized_memberships, total_schema_samples, artifact_counts


def _generate_cluster_schema(
    provider: str,
    config: ProviderConfig,
    samples: list[dict[str, Any]],
    conv_ids: list[str | None],
    *,
    privacy_config: Any | None,
) -> tuple[dict[str, Any], SchemaReport | None]:
    """Generate one schema version from the bounded cluster reservoir."""
    if not samples:
        return {"type": "object", "description": "No samples available"}, None

    builder = SchemaBuilder()
    fingerprint_counts: dict[Any, int] = {}
    for sample in samples:
        fingerprint = _structure_fingerprint(sample)
        seen = fingerprint_counts.get(fingerprint, 0)
        if seen < _STRUCTURE_EXEMPLARS_PER_FINGERPRINT:
            builder.add_object(sample)
            fingerprint_counts[fingerprint] = seen + 1

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)
    schema = _remove_nested_required(schema)
    if config.sample_granularity == "record":
        schema.pop("required", None)

    conv_ids_for_stats: list[str | None] | None = (
        conv_ids if any(conv_id is not None for conv_id in conv_ids) else None
    )
    field_stats = _collect_field_stats(
        samples,
        conversation_ids=conv_ids_for_stats,
    )
    schema = _annotate_schema(
        schema,
        field_stats,
        min_conversation_count=3,
        privacy_config=privacy_config,
    )
    schema = _annotate_semantic_and_relational(schema, field_stats)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    redaction_report = _build_redaction_report(
        provider,
        field_stats,
        schema,
        privacy_config=privacy_config,
        privacy_level=getattr(privacy_config, "level", "standard")
        if privacy_config else "standard",
    )
    return schema, redaction_report


def _apply_schema_metadata(
    schema: dict[str, Any],
    *,
    provider: str,
    config: ProviderConfig,
    schema_sample_count: int,
    anchor_profile_family_id: str,
    artifact_kind: str,
    observed_artifact_count: int,
) -> None:
    schema["title"] = f"{provider} export format ({artifact_kind})"
    schema["description"] = config.description
    schema["x-polylogue-generated-at"] = datetime.now(tz=timezone.utc).isoformat()
    schema["x-polylogue-sample-count"] = schema_sample_count
    schema["x-polylogue-generator"] = "polylogue.schemas.schema_inference"
    schema["x-polylogue-sample-granularity"] = config.sample_granularity
    schema["x-polylogue-anchor-profile-family-id"] = anchor_profile_family_id
    schema["x-polylogue-observed-artifact-count"] = observed_artifact_count
    schema["x-polylogue-artifact-kind"] = artifact_kind


def _membership_scope_key(membership: _UnitMembership) -> str:
    unit = membership.unit
    return (
        unit.bundle_scope
        or unit.raw_id
        or unit.source_path
        or f"{membership.profile_family_id}:{unit.artifact_kind}:{unit.exact_structure_id}"
    )


def _dedupe_bundle_memberships(memberships: list[_UnitMembership]) -> dict[str, list[_UnitMembership]]:
    scoped: dict[str, list[_UnitMembership]] = {}
    for membership in memberships:
        scoped.setdefault(_membership_scope_key(membership), []).append(membership)

    deduped: dict[str, list[_UnitMembership]] = {}
    for scope, items in scoped.items():
        items = sorted(
            items,
            key=lambda item: (
                item.unit.observed_at or "",
                item.unit.source_path or "",
                item.profile_family_id,
            ),
        )
        seen: set[tuple[str, str]] = set()
        retained: list[_UnitMembership] = []
        for membership in items:
            dedupe_key = (membership.unit.artifact_kind, membership.unit.exact_structure_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            retained.append(membership)
        deduped[scope] = retained
    return deduped


def _attach_package_membership(
    package: _PackageAccumulator,
    membership: _UnitMembership,
    *,
    scope: str,
) -> None:
    package.memberships.append(membership)
    package.bundle_scopes.add(scope)
    package.profile_family_ids.add(membership.profile_family_id)
    if membership.unit.source_path:
        _merge_representative_paths(package.representative_paths, [membership.unit.source_path])
    _update_observed_window(package, membership.unit.observed_at)


def _build_package_candidates(
    provider: str,
    *,
    memberships: list[_UnitMembership],
    clusters: dict[str, _ClusterAccumulator],
) -> tuple[list[_PackageAccumulator], dict[str, int]]:
    scoped = _dedupe_bundle_memberships(memberships)
    packages: dict[str, _PackageAccumulator] = {}
    orphan_adjunct_counts: Counter[str] = Counter()

    for scope, items in scoped.items():
        anchor_families = sorted(
            {
                membership.profile_family_id
                for membership in items
                if membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS
            }
        )

        if not anchor_families:
            for membership in items:
                orphan_adjunct_counts[membership.unit.artifact_kind] += 1
            continue

        for family_id in anchor_families:
            acc = packages.get(family_id)
            if acc is None:
                cluster = clusters[family_id]
                acc = _PackageAccumulator(
                    provider=provider,
                    anchor_family_id=family_id,
                    anchor_kind=cluster.artifact_kind,
                )
                packages[family_id] = acc
            for membership in items:
                if membership.profile_family_id == family_id and membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(acc, membership, scope=scope)

        if len(anchor_families) == 1:
            target = packages[anchor_families[0]]
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(target, membership, scope=scope)
        else:
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    orphan_adjunct_counts[membership.unit.artifact_kind] += 1

    ordered = sorted(
        packages.values(),
        key=lambda item: (
            _parse_observed_at(item.first_seen) or datetime.max.replace(tzinfo=timezone.utc),
            -len(item.bundle_scopes),
            item.anchor_family_id,
        ),
    )
    return ordered, dict(orphan_adjunct_counts)


def _element_profile_tokens(memberships: list[_UnitMembership]) -> list[str]:
    token_counts: Counter[str] = Counter()
    for membership in memberships:
        token_counts.update(membership.unit.profile_tokens)
    if not token_counts:
        return []
    min_count = max(1, len(memberships) // 2)
    tokens = sorted(token for token, count in token_counts.items() if count >= min_count)
    if not tokens:
        tokens = [
            token
            for token, _count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:_PROFILE_MAX_TOKENS]
        ]
    return tokens[:_PROFILE_MAX_TOKENS]


def generate_schema_from_samples(
    samples: list[dict[str, Any]],
    *,
    annotate: bool = True,
    max_stats_samples: int = 500,
    max_genson_samples: int | None = None,
) -> dict[str, Any]:
    """Generate JSON schema from samples using genson, with optional annotations."""
    if not GENSON_AVAILABLE:
        raise ImportError("genson is required for schema generation. Install with: pip install genson")

    if not samples:
        return {"type": "object", "description": "No samples available"}

    genson_samples = samples
    if max_genson_samples and len(samples) > max_genson_samples:
        import random
        rng = random.Random(0)
        genson_samples = rng.sample(samples, max_genson_samples)

    builder = SchemaBuilder()
    for sample in genson_samples:
        builder.add_object(sample)

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)
    schema = _remove_nested_required(schema)

    if annotate:
        stats_samples = samples
        if max_stats_samples and len(samples) > max_stats_samples:
            import random
            rng = random.Random(42)
            stats_samples = rng.sample(samples, max_stats_samples)

        field_stats = _collect_field_stats(stats_samples)
        schema = _annotate_schema(schema, field_stats)
        schema = _annotate_semantic_and_relational(schema, field_stats)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    return schema


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> GenerationResult:
    """Generate the default inferred schema for a provider."""
    return _build_provider_bundle(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
    ).result


def _build_provider_bundle(
    provider: str,
    *,
    db_path: Path | None,
    max_samples: int | None,
    privacy_config: Any | None,
) -> _ProviderBundle:
    """Generate all inferred schema versions plus the default result for a provider."""
    provider_token = Provider.from_string(provider)
    if provider_token not in PROVIDERS:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error=f"Unknown provider: {provider}. Known: {[str(item) for item in PROVIDERS]}",
            ),
        )
    if db_path is None:
        db_path = default_db_path()
    if not GENSON_AVAILABLE:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error="genson not installed",
            ),
        )

    config = _resolve_provider_config(provider_token)

    try:
        clusters, memberships, sample_count, artifact_counts = _collect_cluster_accumulators(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            reservoir_size=_cluster_reservoir_size(config, max_samples),
        )
        if not clusters:
            return _ProviderBundle(
                result=GenerationResult(
                    provider=str(provider_token),
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                ),
            )
        packages, orphan_adjunct_counts = _build_package_candidates(
            str(provider_token),
            memberships=memberships,
            clusters=clusters,
        )
        if not packages:
            return _ProviderBundle(
                result=GenerationResult(
                    provider=str(provider_token),
                    schema=None,
                    sample_count=sample_count,
                    error="No anchor-backed schema packages found",
                    cluster_count=len(clusters),
                    artifact_counts=artifact_counts,
                ),
                manifest=ClusterManifest(
                    provider=provider_token,
                    clusters=[
                        SchemaCluster(
                            cluster_id=cluster_id,
                            provider=provider_token,
                            sample_count=acc.sample_count,
                            first_seen=acc.first_seen or "",
                            last_seen=acc.last_seen or "",
                            representative_paths=acc.representative_paths,
                            dominant_keys=acc.dominant_keys,
                            confidence=1.0,
                            artifact_kind=acc.artifact_kind,
                            profile_tokens=list(_cluster_profile_tokens(acc)),
                            exact_structure_ids=sorted(acc.exact_structure_ids),
                            bundle_scope_count=len(acc.bundle_scopes),
                        )
                        for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True)
                    ],
                    artifact_counts=artifact_counts,
                ),
            )

        total_units = max(sum(acc.sample_count for acc in clusters.values()), 1)
        package_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        package_reports: dict[str, dict[str, SchemaReport | None]] = {}
        catalog_packages: list[SchemaVersionPackage] = []
        cluster_to_package_version: dict[str, str] = {}

        for index, package_acc in enumerate(packages, start=1):
            version = f"v{index}"
            package_schemas[version] = {}
            package_reports[version] = {}

            element_memberships: dict[str, list[_UnitMembership]] = {}
            for membership in package_acc.memberships:
                element_memberships.setdefault(membership.unit.artifact_kind, []).append(membership)

            elements: list[SchemaElementManifest] = []
            total_package_samples = 0
            for element_kind, kind_memberships in sorted(
                element_memberships.items(),
                key=lambda item: (_artifact_priority(item[0]), item[0]),
                reverse=True,
            ):
                schema_samples: list[dict[str, Any]] = []
                conv_ids: list[str | None] = []
                representative_paths: list[str] = []
                exact_structure_ids = sorted({membership.unit.exact_structure_id for membership in kind_memberships})
                profile_family_ids = sorted({membership.profile_family_id for membership in kind_memberships})
                element_bundle_scopes = sorted(
                    {
                        membership.unit.bundle_scope
                        for membership in kind_memberships
                        if membership.unit.bundle_scope
                    }
                )
                element_first_seen, element_last_seen = _membership_observed_window(kind_memberships)
                for membership in kind_memberships:
                    schema_samples.extend(membership.unit.schema_samples)
                    conv_ids.extend([membership.unit.conversation_id] * len(membership.unit.schema_samples))
                    if membership.unit.source_path:
                        _merge_representative_paths(representative_paths, [membership.unit.source_path])

                total_package_samples += len(schema_samples)
                schema, redaction_report = _generate_cluster_schema(
                    provider,
                    config,
                    schema_samples,
                    conv_ids,
                    privacy_config=privacy_config,
                )
                _apply_schema_metadata(
                    schema,
                    provider=str(provider_token),
                    config=config,
                    schema_sample_count=len(schema_samples),
                    anchor_profile_family_id=package_acc.anchor_family_id,
                    artifact_kind=element_kind,
                    observed_artifact_count=len(kind_memberships),
                )
                schema["x-polylogue-package-version"] = version
                schema["x-polylogue-profile-family-ids"] = profile_family_ids
                schema["x-polylogue-exact-structure-ids"] = exact_structure_ids
                if element_first_seen:
                    schema["x-polylogue-element-first-seen"] = element_first_seen
                if element_last_seen:
                    schema["x-polylogue-element-last-seen"] = element_last_seen
                schema["x-polylogue-element-bundle-scope-count"] = len(element_bundle_scopes)
                schema["x-polylogue-anchor-profile-family-id"] = package_acc.anchor_family_id
                schema["x-polylogue-package-profile-family-ids"] = sorted(package_acc.profile_family_ids)
                package_schemas[version][element_kind] = schema
                package_reports[version][element_kind] = redaction_report
                elements.append(
                    SchemaElementManifest(
                        element_kind=element_kind,
                        schema_file=f"{element_kind}.schema.json.gz",
                        sample_count=len(schema_samples),
                        artifact_count=len(kind_memberships),
                        first_seen=element_first_seen or "",
                        last_seen=element_last_seen or "",
                        bundle_scope_count=len(element_bundle_scopes),
                        bundle_scopes=element_bundle_scopes,
                        exact_structure_ids=exact_structure_ids,
                        profile_family_ids=profile_family_ids,
                        profile_tokens=_element_profile_tokens(kind_memberships),
                        representative_paths=representative_paths,
                        observed_artifact_count=len(kind_memberships),
                    )
                )

            package = SchemaVersionPackage(
                provider=provider_token,
                version=version,
                anchor_kind=package_acc.anchor_kind,
                default_element_kind=package_acc.anchor_kind,
                first_seen=package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
                last_seen=package_acc.last_seen or package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
                bundle_scope_count=len(package_acc.bundle_scopes),
                sample_count=total_package_samples,
                anchor_profile_family_id=package_acc.anchor_family_id,
                bundle_scopes=sorted(package_acc.bundle_scopes),
                profile_family_ids=sorted(package_acc.profile_family_ids),
                representative_paths=package_acc.representative_paths,
                elements=elements,
            )
            catalog_packages.append(package)
            for cluster_id in package.profile_family_ids:
                cluster_to_package_version[cluster_id] = version

        latest_version = catalog_packages[-1].version if catalog_packages else None
        catalog = SchemaPackageCatalog(
            provider=provider_token,
            packages=catalog_packages,
            latest_version=latest_version,
            default_version=latest_version,
            recommended_version=latest_version,
            orphan_adjunct_counts=orphan_adjunct_counts,
        )
        manifest_clusters: list[SchemaCluster] = []
        for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True):
            manifest_clusters.append(
                SchemaCluster(
                    cluster_id=cluster_id,
                    provider=provider_token,
                    sample_count=acc.sample_count,
                    first_seen=acc.first_seen or "",
                    last_seen=acc.last_seen or "",
                    representative_paths=acc.representative_paths,
                    dominant_keys=acc.dominant_keys,
                    confidence=round(min(1.0, acc.sample_count / max(total_units * 0.1, 1)), 3),
                    artifact_kind=acc.artifact_kind,
                    profile_tokens=list(_cluster_profile_tokens(acc)),
                    exact_structure_ids=sorted(acc.exact_structure_ids),
                    bundle_scope_count=len(acc.bundle_scopes),
                    promoted_package_version=cluster_to_package_version.get(cluster_id),
                )
            )
        manifest = ClusterManifest(
            provider=provider_token,
            clusters=manifest_clusters,
            artifact_counts=artifact_counts,
            default_version=catalog.default_version,
        )
        default_package = catalog.package(catalog.default_version) if catalog.default_version else None
        default_schema = (
            package_schemas[default_package.version][default_package.default_element_kind]
            if default_package is not None
            else None
        )
        default_redaction_report = (
            package_reports[default_package.version][default_package.default_element_kind]
            if default_package is not None
            else None
        )
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=default_schema,
                sample_count=sample_count,
                redaction_report=default_redaction_report,
                versions=[package.version for package in catalog.packages],
                default_version=catalog.default_version,
                cluster_count=len(clusters),
                package_count=len(catalog.packages),
                artifact_counts=artifact_counts,
            ),
            catalog=catalog,
            package_schemas=package_schemas,
            manifest=manifest,
        )
    except Exception as e:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error=str(e),
            ),
        )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> list[GenerationResult]:
    """Generate versioned schemas for all (or specified) providers."""
    if db_path is None:
        db_path = default_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []

    for provider in provider_list:
        bundle = _build_provider_bundle(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
        )
        result = bundle.result
        results.append(result)

        if result.success and bundle.manifest is not None and bundle.catalog is not None:
            registry = SchemaRegistry(storage_root=output_dir)
            registry.replace_provider_packages(provider, bundle.catalog, bundle.package_schemas)
            registry.save_cluster_manifest(bundle.manifest)

            for legacy_name in (f"{provider}.schema.json.gz", f"{provider}.schema.json"):
                legacy_path = output_dir / legacy_name
                if legacy_path.exists():
                    legacy_path.unlink()

    return results


__all__ = [
    "GenerationResult",
    "_annotate_schema",
    "_merge_schemas",
    "_remove_nested_required",
    "_structure_fingerprint",
    "collapse_dynamic_keys",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
    "is_dynamic_key",
    "UUID_PATTERN",
]
