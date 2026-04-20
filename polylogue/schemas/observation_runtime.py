"""Schema-observation payload extraction runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.json import JSONDocument, JSONValue, is_json_value, json_document
from polylogue.lib.raw_payload import extract_payload_samples, record_bucket_key
from polylogue.schemas.observation_identity import derive_bundle_scope, schema_cluster_id
from polylogue.schemas.observation_models import ProviderConfig, SchemaUnit
from polylogue.types import Provider

SchemaSample: TypeAlias = JSONDocument

_SCHEMA_SAMPLE_STRING_LIMIT = 1024


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
    provider_token = Provider.from_string(provider_name)
    effective_max_samples = max_samples if max_samples is not None else config.schema_sample_cap
    bundle_scope = derive_bundle_scope(provider_token, source_path)
    source_path_text = str(source_path) if source_path is not None else None

    if config.sample_granularity == "record":
        artifact = classify_artifact(
            normalized_payload,
            provider=provider_token,
            source_path=source_path,
        )
        if not artifact.schema_eligible:
            return []

        samples = _compact_schema_samples(
            extract_payload_samples(
                normalized_payload,
                sample_granularity="record",
                max_samples=effective_max_samples,
                record_type_key=config.record_type_key,
            )
        )
        if not samples:
            return []

        return [
            SchemaUnit(
                cluster_payload=normalized_payload,
                schema_samples=samples,
                artifact_kind=artifact.cohort,
                conversation_id=raw_id,
                raw_id=raw_id,
                source_path=source_path_text,
                bundle_scope=bundle_scope,
                observed_at=observed_at,
                exact_structure_id=schema_cluster_id(normalized_payload, artifact.cohort),
                profile_tokens=_record_profile_tokens(
                    samples,
                    record_type_key=config.record_type_key,
                ),
            )
        ]

    documents = _compact_schema_samples(
        extract_payload_samples(
            normalized_payload,
            sample_granularity="document",
            max_samples=effective_max_samples,
        )
    )
    units: list[SchemaUnit] = []
    for sample in documents:
        artifact = classify_artifact(
            sample,
            provider=provider_token,
            source_path=source_path,
        )
        if not artifact.schema_eligible:
            continue
        units.append(
            SchemaUnit(
                cluster_payload=sample,
                schema_samples=[sample],
                artifact_kind=artifact.cohort,
                conversation_id=_document_conversation_id(sample, raw_id),
                raw_id=raw_id,
                source_path=source_path_text,
                bundle_scope=bundle_scope,
                observed_at=observed_at,
                exact_structure_id=schema_cluster_id(sample, artifact.cohort),
                profile_tokens=_document_profile_tokens(sample),
            )
        )
    return units


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
