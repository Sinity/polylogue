"""Schema-observation payload extraction runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.json import JSONDocument, JSONValue, is_json_value, json_document
from polylogue.lib.raw_payload import extract_payload_samples, record_bucket_key
from polylogue.schemas.observation_identity import derive_bundle_scope, schema_cluster_id
from polylogue.schemas.observation_models import ProviderConfig, SchemaClusterPayload, SchemaUnit
from polylogue.types import Provider

SchemaSample: TypeAlias = JSONDocument

_SCHEMA_SAMPLE_STRING_LIMIT = 1024


@dataclass(frozen=True)
class _ObservationContext:
    provider_name: Provider
    source_path: Path | None
    source_path_text: str | None
    raw_id: str | None
    observed_at: str | None
    bundle_scope: str | None
    effective_max_samples: int | None


@dataclass(frozen=True)
class _ObservedSchemaUnit:
    cluster_payload: SchemaClusterPayload
    schema_samples: list[SchemaSample]
    artifact_kind: str
    conversation_id: str | None
    profile_tokens: tuple[str, ...]


def _build_observation_context(
    *,
    provider_name: Provider,
    source_path: str | Path | None,
    raw_id: str | None,
    observed_at: str | None,
    config: ProviderConfig,
    max_samples: int | None,
) -> _ObservationContext:
    provider_token = Provider.from_string(provider_name)
    source_path_obj = Path(source_path) if source_path is not None else None
    return _ObservationContext(
        provider_name=provider_token,
        source_path=source_path_obj,
        source_path_text=str(source_path_obj) if source_path_obj is not None else None,
        raw_id=raw_id,
        observed_at=observed_at,
        bundle_scope=derive_bundle_scope(provider_token, source_path_obj),
        effective_max_samples=max_samples if max_samples is not None else config.schema_sample_cap,
    )


def _to_schema_unit(observed: _ObservedSchemaUnit, context: _ObservationContext) -> SchemaUnit:
    return SchemaUnit(
        cluster_payload=observed.cluster_payload,
        schema_samples=observed.schema_samples,
        artifact_kind=observed.artifact_kind,
        conversation_id=observed.conversation_id,
        raw_id=context.raw_id,
        source_path=context.source_path_text,
        bundle_scope=context.bundle_scope,
        observed_at=context.observed_at,
        exact_structure_id=schema_cluster_id(observed.cluster_payload, observed.artifact_kind),
        profile_tokens=observed.profile_tokens,
    )


def _eligible_artifact_kind(
    payload: SchemaClusterPayload,
    *,
    context: _ObservationContext,
) -> str | None:
    artifact = classify_artifact(
        payload,
        provider=context.provider_name,
        source_path=context.source_path,
    )
    return artifact.cohort if artifact.schema_eligible else None


def _extract_record_observation(
    normalized_payload: JSONValue,
    *,
    context: _ObservationContext,
    config: ProviderConfig,
) -> _ObservedSchemaUnit | None:
    artifact_kind = _eligible_artifact_kind(normalized_payload, context=context)
    if artifact_kind is None:
        return None

    samples = _compact_schema_samples(
        extract_payload_samples(
            normalized_payload,
            sample_granularity="record",
            max_samples=context.effective_max_samples,
            record_type_key=config.record_type_key,
        )
    )
    if not samples:
        return None

    return _ObservedSchemaUnit(
        cluster_payload=normalized_payload,
        schema_samples=samples,
        artifact_kind=artifact_kind,
        conversation_id=context.raw_id,
        profile_tokens=_record_profile_tokens(
            samples,
            record_type_key=config.record_type_key,
        ),
    )


def _extract_document_observations(
    normalized_payload: JSONValue,
    *,
    context: _ObservationContext,
) -> list[_ObservedSchemaUnit]:
    documents = _compact_schema_samples(
        extract_payload_samples(
            normalized_payload,
            sample_granularity="document",
            max_samples=context.effective_max_samples,
        )
    )
    units: list[_ObservedSchemaUnit] = []
    for sample in documents:
        artifact_kind = _eligible_artifact_kind(sample, context=context)
        if artifact_kind is None:
            continue
        units.append(
            _ObservedSchemaUnit(
                cluster_payload=sample,
                schema_samples=[sample],
                artifact_kind=artifact_kind,
                conversation_id=_document_conversation_id(sample, context.raw_id),
                profile_tokens=_document_profile_tokens(sample),
            )
        )
    return units


def extract_schema_units_from_payload(
    payload: object,
    *,
    provider_name: Provider,
    source_path: str | Path | None,
    raw_id: str | None,
    observed_at: str | None = None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> list[SchemaUnit]:
    """Extract clusterable schema units from one decoded payload."""
    if not is_json_value(payload):
        return []
    normalized_payload: JSONValue = payload
    context = _build_observation_context(
        provider_name=provider_name,
        source_path=source_path,
        raw_id=raw_id,
        observed_at=observed_at,
        config=config,
        max_samples=max_samples,
    )

    if config.sample_granularity == "record":
        observed = _extract_record_observation(
            normalized_payload,
            context=context,
            config=config,
        )
        if observed is None:
            return []
        return [_to_schema_unit(observed, context)]

    return [
        _to_schema_unit(observed, context)
        for observed in _extract_document_observations(
            normalized_payload,
            context=context,
        )
    ]


def _compact_schema_value(value: JSONValue) -> JSONValue:
    if isinstance(value, str):
        return value[:_SCHEMA_SAMPLE_STRING_LIMIT] if len(value) > _SCHEMA_SAMPLE_STRING_LIMIT else value
    if isinstance(value, list):
        return [_compact_schema_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _compact_schema_value(child) for key, child in value.items()}
    return value


def _compact_schema_samples(samples: list[SchemaSample]) -> list[SchemaSample]:
    compacted: list[SchemaSample] = []
    for sample in samples:
        compacted_sample = _compact_schema_value(sample)
        if isinstance(compacted_sample, dict):
            compacted.append(json_document(compacted_sample))
    return compacted


def _coarse_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _document_profile_tokens(sample: SchemaSample) -> tuple[str, ...]:
    tokens: list[str] = []
    for key, value in sorted(sample.items()):
        tokens.append(f"field:{key}")
        value_type = _coarse_type(value)
        if value_type in {"array", "object"}:
            tokens.append(f"shape:{key}:{value_type}")
        if key in {"mapping", "chat_messages", "chunkedPrompt", "chunks", "messages"}:
            tokens.append(f"anchor:{key}")
    return tuple(tokens[:96])


def _record_profile_tokens(
    samples: list[SchemaSample],
    *,
    record_type_key: str | None,
) -> tuple[str, ...]:
    bucket_keys: dict[str, set[str]] = {}
    for sample in samples[:512]:
        bucket = record_bucket_key(sample, record_type_key)
        keys = bucket_keys.setdefault(bucket, set())
        keys.update(str(key) for key in sample)

    tokens: list[str] = []
    for bucket, keys in sorted(bucket_keys.items()):
        tokens.append(f"bucket:{bucket}")
        for key in sorted(keys)[:24]:
            tokens.append(f"field:{bucket}:{key}")
    return tuple(tokens[:160])


def _document_conversation_id(sample: SchemaSample, raw_id: str | None) -> str | None:
    for key in ("conversation_id", "id", "uuid"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return raw_id
