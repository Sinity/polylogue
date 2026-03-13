"""Schema generation from provider data samples.

Provides functions to build and annotate JSON schemas from raw data samples.
Uses genson for structural inference and adds x-polylogue-* annotations from
field statistics.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from genson import SchemaBuilder
    GENSON_AVAILABLE = True
except ImportError:
    GENSON_AVAILABLE = False

from polylogue.schemas.field_stats import (
    FieldStats,
    UUID_PATTERN,
    _collect_field_stats,
    is_dynamic_key,
)
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)
from polylogue.schemas.sampling import (
    PROVIDERS,
    ProviderConfig,
    _iter_samples_from_db,
    _iter_samples_from_sessions,
    load_samples_from_db,
    load_samples_from_sessions,
)
from polylogue.paths import db_path as default_db_path

_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35
_STRUCTURE_EXEMPLARS_PER_FINGERPRINT = 8
_FINGERPRINT_MAX_DEPTH = 8
_FINGERPRINT_ARRAY_SAMPLE = 8

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

    @property
    def success(self) -> bool:
        return self.schema is not None and self.error is None


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


def _structure_fingerprint(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _FINGERPRINT_MAX_DEPTH,
) -> Any:
    """Build a hashable structural fingerprint for schema-dedup heuristics.

    The fingerprint intentionally ignores concrete scalar values and keeps only
    structural/type shape so repeated record variants can be collapsed before
    feeding Genson.
    """
    if depth >= max_depth:
        return ("depth-limit", type(value).__name__)

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number",)
    if isinstance(value, str):
        return ("string",)

    if isinstance(value, list):
        item_shapes = {
            _structure_fingerprint(item, depth=depth + 1, max_depth=max_depth)
            for item in value[:_FINGERPRINT_ARRAY_SAMPLE]
        }
        return ("array", tuple(sorted(item_shapes, key=repr)))

    if isinstance(value, dict):
        props: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = value[key]
            normalized_key = "*" if is_dynamic_key(key) else key
            props.append(
                (
                    normalized_key,
                    _structure_fingerprint(child, depth=depth + 1, max_depth=max_depth),
                )
            )
        return ("object", tuple(props))

    return ("other", type(value).__name__)


# =============================================================================
# Field Statistics & Schema Annotations
# =============================================================================


def _annotate_schema(
    schema: dict[str, Any],
    stats: dict[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
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
                    if not _is_safe_enum_value(value, path=path):
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

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    return schema


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> GenerationResult:
    """Generate schema for a provider."""
    if db_path is None:
        db_path = default_db_path()
    if not GENSON_AVAILABLE:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error="genson not installed",
        )

    config = PROVIDERS.get(provider)
    if not config:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error=f"Unknown provider: {provider}. Known: {list(PROVIDERS.keys())}",
        )

    try:
        if max_samples is not None:
            if config.db_provider_name:
                samples = load_samples_from_db(
                    config.db_provider_name,
                    db_path=db_path,
                    max_samples=max_samples,
                )
            elif config.session_dir:
                samples = load_samples_from_sessions(
                    config.session_dir,
                    max_sessions=config.max_sessions,
                    max_samples=max_samples,
                    record_type_key=config.record_type_key,
                )
            else:
                samples = []

            if not samples:
                return GenerationResult(
                    provider=provider,
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                )

            schema = generate_schema_from_samples(
                samples,
                max_genson_samples=max_samples,
            )
            sample_count = len(samples)
        else:
            import random

            builder = SchemaBuilder()
            reservoir: list[dict[str, Any]] = []
            reservoir_size = 500
            reservoir_rng = random.Random(42)
            sample_count = 0
            fingerprint_counts: dict[Any, int] = {}

            if config.db_provider_name:
                sample_iter = _iter_samples_from_db(
                    config.db_provider_name,
                    db_path=db_path,
                    config=config,
                )
            elif config.session_dir:
                sample_iter = _iter_samples_from_sessions(
                    config.session_dir,
                    max_sessions=config.max_sessions,
                )
            else:
                sample_iter = iter(())

            for sample in sample_iter:
                sample_count += 1
                fingerprint = _structure_fingerprint(sample)
                seen = fingerprint_counts.get(fingerprint, 0)
                if seen < _STRUCTURE_EXEMPLARS_PER_FINGERPRINT:
                    builder.add_object(sample)
                    fingerprint_counts[fingerprint] = seen + 1

                if len(reservoir) < reservoir_size:
                    reservoir.append(sample)
                else:
                    j = reservoir_rng.randint(0, sample_count - 1)
                    if j < reservoir_size:
                        reservoir[j] = sample

            if sample_count == 0:
                return GenerationResult(
                    provider=provider,
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                )

            schema = builder.to_schema()
            schema = collapse_dynamic_keys(schema)
            schema = _remove_nested_required(schema)
            if config.sample_granularity == "record":
                schema.pop("required", None)

            field_stats = _collect_field_stats(reservoir)
            schema = _annotate_schema(schema, field_stats)

            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

        schema["title"] = f"{provider} export format"
        schema["description"] = config.description

        schema["$id"] = f"polylogue://schemas/{provider}/v1"
        schema["x-polylogue-version"] = 1
        schema["x-polylogue-generated-at"] = datetime.now(tz=timezone.utc).isoformat()
        schema["x-polylogue-sample-count"] = sample_count
        schema["x-polylogue-generator"] = "polylogue.schemas.schema_inference"
        schema["x-polylogue-sample-granularity"] = config.sample_granularity

        return GenerationResult(
            provider=provider,
            schema=schema,
            sample_count=sample_count,
        )
    except Exception as e:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error=str(e),
        )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
) -> list[GenerationResult]:
    """Generate schemas for all (or specified) providers."""
    if db_path is None:
        db_path = default_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []

    for provider in provider_list:
        result = generate_provider_schema(
            provider,
            db_path=db_path,
            max_samples=max_samples,
        )
        results.append(result)

        if result.success and result.schema:
            output_path = output_dir / f"{provider}.schema.json.gz"
            compressed = gzip.compress(
                json.dumps(result.schema, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            )
            output_path.write_bytes(compressed)
            legacy_path = output_dir / f"{provider}.schema.json"
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
