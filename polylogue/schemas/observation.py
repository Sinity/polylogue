"""Runtime-safe payload structure and schema-observation helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.raw_payload import extract_payload_samples, record_bucket_key
from polylogue.types import Provider

_SCHEMA_SAMPLE_STRING_LIMIT = 1024


@dataclass
class ProviderConfig:
    """Configuration for one provider's schema observation behavior."""

    name: Provider
    description: str
    db_provider_name: Provider | None = None
    session_dir: Path | None = None
    max_sessions: int | None = None
    sample_granularity: str = "document"
    record_type_key: str | None = None
    schema_sample_cap: int | None = None


@dataclass(frozen=True)
class SchemaUnit:
    """Clusterable schema-observation input."""

    cluster_payload: Any
    schema_samples: list[dict[str, Any]]
    artifact_kind: str
    conversation_id: str | None = None
    raw_id: str | None = None
    source_path: str | None = None
    bundle_scope: str | None = None
    observed_at: str | None = None
    exact_structure_id: str = ""
    profile_tokens: tuple[str, ...] = ()


PROVIDERS: dict[Provider, ProviderConfig] = {
    Provider.CHATGPT: ProviderConfig(
        name=Provider.CHATGPT,
        description="ChatGPT message format",
        db_provider_name=Provider.CHATGPT,
        sample_granularity="document",
    ),
    Provider.CLAUDE_CODE: ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code message format",
        db_provider_name=Provider.CLAUDE_CODE,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
    Provider.CLAUDE_AI: ProviderConfig(
        name=Provider.CLAUDE_AI,
        description="Claude AI web message format",
        db_provider_name=Provider.CLAUDE_AI,
        sample_granularity="document",
    ),
    Provider.GEMINI: ProviderConfig(
        name=Provider.GEMINI,
        description="Gemini AI Studio message format",
        db_provider_name=Provider.GEMINI,
        sample_granularity="document",
    ),
    Provider.CODEX: ProviderConfig(
        name=Provider.CODEX,
        description="OpenAI Codex CLI session format",
        db_provider_name=Provider.CODEX,
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
}


def resolve_provider_config(provider_name: str | Provider) -> ProviderConfig:
    canonical_provider = Provider.from_string(provider_name)
    if canonical_provider in PROVIDERS:
        return PROVIDERS[canonical_provider]

    config = next((c for c in PROVIDERS.values() if c.db_provider_name == canonical_provider), None)
    if config is not None:
        return config

    return ProviderConfig(
        name=canonical_provider,
        description=f"{canonical_provider} export format",
        db_provider_name=canonical_provider,
        sample_granularity="document",
    )


def fingerprint_hash(fingerprint: Any) -> str:
    """Compute a stable short hash for a structural fingerprint."""
    raw = repr(fingerprint).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def schema_cluster_id(cluster_payload: Any, artifact_kind: str) -> str:
    """Compute a stable cluster identifier for a schema unit."""
    from polylogue.schemas.shape_fingerprint import _structure_fingerprint

    return fingerprint_hash((artifact_kind, _structure_fingerprint(cluster_payload)))


def profile_cluster_id(artifact_kind: str, profile_tokens: tuple[str, ...]) -> str:
    """Compute a stable identifier for a profile-token cohort."""
    return fingerprint_hash((artifact_kind, tuple(sorted(profile_tokens))))


def profile_similarity(left: set[str], right: set[str]) -> float:
    """Similarity score tolerant to optional fields but resistant to drift."""
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return ((overlap / len(left)) + (overlap / len(right))) / 2.0


def derive_bundle_scope(
    provider_name: str | Provider,
    source_path: str | Path | None,
) -> str | None:
    """Return the provider-specific bundle scope for a raw artifact path."""
    if source_path is None:
        return None

    provider_token = Provider.from_string(provider_name)
    path = Path(str(source_path))
    normalized = str(path)

    if provider_token is Provider.CLAUDE_CODE:
        if "/subagents/" in normalized:
            return path.parent.parent.name or None
        if path.name in {"bridge-pointer.json", "sessions-index.json"}:
            return path.parent.name or None
        if path.name.startswith("agent-") and path.name.endswith(".meta.json"):
            return path.parent.parent.name or None
        if path.suffix == ".jsonl":
            return path.stem or None

    if provider_token is Provider.CODEX and path.suffix == ".jsonl":
        return path.stem or None

    return path.stem or path.name or None


def _compact_schema_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _SCHEMA_SAMPLE_STRING_LIMIT:
            return value
        return value[:_SCHEMA_SAMPLE_STRING_LIMIT]
    if isinstance(value, list):
        return [_compact_schema_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _compact_schema_value(child) for key, child in value.items()}
    return value


def _compact_schema_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for sample in samples:
        compacted_sample = _compact_schema_value(sample)
        if isinstance(compacted_sample, dict):
            compacted.append(compacted_sample)
    return compacted


def _coarse_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _document_profile_tokens(sample: dict[str, Any]) -> tuple[str, ...]:
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
    samples: list[dict[str, Any]],
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


def _document_conversation_id(sample: dict[str, Any], raw_id: str | None) -> str | None:
    for key in ("conversation_id", "id", "uuid"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return raw_id


def extract_schema_units_from_payload(
    payload: Any,
    *,
    provider_name: Provider,
    source_path: str | Path | None,
    raw_id: str | None,
    observed_at: str | None = None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> list[SchemaUnit]:
    """Extract clusterable schema units from one decoded payload."""
    provider_token = Provider.from_string(provider_name)
    effective_max_samples = max_samples
    if effective_max_samples is None:
        effective_max_samples = config.schema_sample_cap

    if config.sample_granularity == "record":
        artifact = classify_artifact(
            payload,
            provider=provider_token,
            source_path=source_path,
        )
        if not artifact.schema_eligible:
            return []

        samples = extract_payload_samples(
            payload,
            sample_granularity="record",
            max_samples=effective_max_samples,
            record_type_key=config.record_type_key,
        )
        samples = _compact_schema_samples(samples)
        if not samples:
            return []

        return [
            SchemaUnit(
                cluster_payload=payload,
                schema_samples=samples,
                artifact_kind=artifact.cohort,
                conversation_id=raw_id,
                raw_id=raw_id,
                source_path=str(source_path) if source_path is not None else None,
                bundle_scope=derive_bundle_scope(provider_token, source_path),
                observed_at=observed_at,
                exact_structure_id=schema_cluster_id(payload, artifact.cohort),
                profile_tokens=_record_profile_tokens(
                    samples,
                    record_type_key=config.record_type_key,
                ),
            )
        ]

    documents = extract_payload_samples(
        payload,
        sample_granularity="document",
        max_samples=effective_max_samples,
    )
    documents = _compact_schema_samples(documents)
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
                source_path=str(source_path) if source_path is not None else None,
                bundle_scope=derive_bundle_scope(provider_token, source_path),
                observed_at=observed_at,
                exact_structure_id=schema_cluster_id(sample, artifact.cohort),
                profile_tokens=_document_profile_tokens(sample),
            )
        )
    return units


__all__ = [
    "PROVIDERS",
    "ProviderConfig",
    "SchemaUnit",
    "derive_bundle_scope",
    "extract_schema_units_from_payload",
    "fingerprint_hash",
    "profile_cluster_id",
    "profile_similarity",
    "resolve_provider_config",
    "schema_cluster_id",
]
