"""Schema emission helpers for generation workflows."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from typing import TypeAlias

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas.field_stats.stats import _collect_field_stats
from polylogue.schemas.generation.support import (
    GENSON_AVAILABLE,
    SchemaBuilder,
    _annotate_schema,
    _annotate_semantic_and_relational,
    _build_redaction_report,
    _remove_nested_required,
    collapse_dynamic_keys,
)
from polylogue.schemas.observation import ProviderConfig
from polylogue.schemas.pinning import load_pins
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.redaction_report import SchemaReport
from polylogue.schemas.shape_fingerprint import _structure_fingerprint

SchemaInput: TypeAlias = Mapping[str, object]
SchemaPayload: TypeAlias = JSONDocument


def _load_rejected_pins(provider: str) -> dict[str, set[str]]:
    # load_pins already degrades loudly (missing file -> empty PinSet, corrupt
    # file -> logged warning + empty PinSet); swallowing anything beyond that
    # would silently drop operator rejection decisions from generated schemas.
    pins = load_pins(provider)
    rejected: dict[str, set[str]] = {}
    for pin in pins.pins:
        if pin.action == "reject":
            rejected.setdefault(pin.path, set()).add(pin.role)
    return rejected


MutableSchemaPayload: TypeAlias = JSONDocument


_STRUCTURE_EXEMPLARS_PER_FINGERPRINT = 8


def _generate_cluster_schema(
    provider: str,
    config: ProviderConfig,
    samples: Collection[SchemaInput],
    conv_ids: Collection[str | None],
    *,
    privacy_config: SchemaPrivacyConfig | None,
    full_corpus: bool = False,
    artifact_kind: str | None = None,
) -> tuple[MutableSchemaPayload, SchemaReport | None]:
    if not samples:
        return {"type": "object", "description": "No samples available"}, None

    builder = SchemaBuilder()
    fingerprint_counts: dict[object, int] = {}
    exemplar_cap = None if full_corpus else _STRUCTURE_EXEMPLARS_PER_FINGERPRINT
    for sample in samples:
        fingerprint = _structure_fingerprint(sample)
        seen = fingerprint_counts.get(fingerprint, 0)
        if exemplar_cap is None or seen < exemplar_cap:
            builder.add_object(dict(sample))
            fingerprint_counts[fingerprint] = seen + 1

    schema = collapse_dynamic_keys(json_document(builder.to_schema()))
    schema = _remove_nested_required(schema)
    if config.sample_granularity == "record":
        schema.pop("required", None)

    conv_ids_for_stats: Collection[str | None] | None = (
        conv_ids if any(conv_id is not None for conv_id in conv_ids) else None
    )
    field_stats = _collect_field_stats(samples, session_ids=conv_ids_for_stats)
    pins = _load_rejected_pins(provider)
    schema = _annotate_semantic_and_relational(schema, field_stats, artifact_kind=artifact_kind, pins=pins)
    schema = _annotate_schema(
        schema,
        field_stats,
        min_session_count=3,
        privacy_config=privacy_config,
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    redaction_report = _build_redaction_report(
        provider,
        field_stats,
        schema,
        privacy_config=privacy_config,
        privacy_level=privacy_config.level if privacy_config else "standard",
    )
    return schema, redaction_report


def _apply_schema_metadata(
    schema: MutableSchemaPayload,
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
    schema["x-polylogue-generator"] = "polylogue.schemas.operator.schema_inference"
    schema["x-polylogue-sample-granularity"] = config.sample_granularity
    schema["x-polylogue-anchor-profile-family-id"] = anchor_profile_family_id
    schema["x-polylogue-observed-artifact-count"] = observed_artifact_count
    schema["x-polylogue-artifact-kind"] = artifact_kind


def generate_schema_from_samples(
    samples: Sequence[SchemaInput],
    *,
    annotate: bool = True,
    max_stats_samples: int = 500,
    max_genson_samples: int | None = None,
    provider: str | None = None,
) -> MutableSchemaPayload:
    if not GENSON_AVAILABLE:
        raise ImportError("genson is required for schema generation. Install with: pip install genson")

    if not samples:
        return {"type": "object", "description": "No samples available"}

    genson_samples = list(samples)
    if max_genson_samples and len(samples) > max_genson_samples:
        import random

        genson_samples = random.Random(0).sample(list(samples), max_genson_samples)

    builder = SchemaBuilder()
    for sample in genson_samples:
        builder.add_object(dict(sample))

    schema = collapse_dynamic_keys(json_document(builder.to_schema()))
    schema = _remove_nested_required(schema)

    if annotate:
        stats_samples = list(samples)
        if max_stats_samples and len(samples) > max_stats_samples:
            import random

            stats_samples = random.Random(42).sample(list(samples), max_stats_samples)

        field_stats = _collect_field_stats(stats_samples)
        schema = _annotate_schema(schema, field_stats)
        pins = _load_rejected_pins(provider) if provider else {}
        schema = _annotate_semantic_and_relational(schema, field_stats, pins=pins)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


__all__ = [
    "SchemaInput",
    "_apply_schema_metadata",
    "_generate_cluster_schema",
    "generate_schema_from_samples",
]
