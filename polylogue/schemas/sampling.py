"""Sample loading from polylogue database and session files.

Provides provider configs plus the schema-unit extraction helpers used by
version inference and validation. A schema unit is the thing we cluster into
versions:
- document providers cluster conversation documents
- record providers cluster whole raw streams, then infer a per-record schema
  union inside that cluster
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.provider_identity import (
    CORE_RUNTIME_PROVIDERS,
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.lib.raw_payload import (
    build_raw_payload_envelope,
    collect_limited_samples,
    extract_record_samples_from_raw_content,
    extract_payload_samples,
    record_bucket_key,
)
from polylogue.paths import db_path as default_db_path
from polylogue.types import Provider

_SCHEMA_SAMPLE_STRING_LIMIT = 1024


@dataclass
class ProviderConfig:
    """Configuration for a provider's schema generation."""

    name: Provider
    description: str
    db_provider_name: Provider | None = None  # Provider name in polylogue DB
    session_dir: Path | None = None  # Optional session-dir fallback
    max_sessions: int | None = None
    sample_granularity: str = "document"  # "document" | "record"
    record_type_key: str | None = None  # best-effort stratification key
    schema_sample_cap: int | None = None  # bounded per-unit sampling for inference


@dataclass(frozen=True)
class SchemaUnit:
    """Clusterable input for schema version inference."""

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


# Provider configurations
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


def _resolve_provider_config(provider_name: str | Provider) -> ProviderConfig:
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


def _sample_provider_where_clause(provider_name: str | Provider) -> tuple[str, tuple[Any, ...]]:
    provider_token = str(Provider.from_string(provider_name))
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        "payload_provider = ? "
        "OR (payload_provider IS NULL AND provider_name = ?) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (
        provider_token,
        provider_token,
        *CORE_RUNTIME_PROVIDERS,
    )
    return clause, params


def fingerprint_hash(fingerprint: Any) -> str:
    """Compute a stable short hash for a structural fingerprint."""
    raw = repr(fingerprint).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def schema_cluster_id(cluster_payload: Any, artifact_kind: str) -> str:
    """Compute a stable cluster identifier for a schema unit."""
    from polylogue.schemas.schema_generation import _structure_fingerprint

    return fingerprint_hash((artifact_kind, _structure_fingerprint(cluster_payload)))


def profile_cluster_id(artifact_kind: str, profile_tokens: tuple[str, ...]) -> str:
    """Compute a stable cluster identifier for a profile-token cohort."""
    return fingerprint_hash((artifact_kind, tuple(sorted(profile_tokens))))


def profile_similarity(left: set[str], right: set[str]) -> float:
    """Similarity score tolerant to optional fields but resistant to drift."""
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return ((overlap / len(left)) + (overlap / len(right))) / 2.0


def _document_conversation_id(sample: dict[str, Any], raw_id: str | None) -> str | None:
    for key in ("conversation_id", "id", "uuid"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return raw_id


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

    if provider_token is Provider.CODEX:
        if path.suffix == ".jsonl":
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
        return {
            str(key): _compact_schema_value(child)
            for key, child in value.items()
        }
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
        keys.update(str(key) for key in sample.keys())

    tokens: list[str] = []
    for bucket, keys in sorted(bucket_keys.items()):
        tokens.append(f"bucket:{bucket}")
        for key in sorted(keys)[:24]:
            tokens.append(f"field:{bucket}:{key}")
    return tuple(tokens[:160])


def _extract_schema_units(
    payload: Any,
    *,
    provider_name: Provider,
    source_path: str | Path | None,
    raw_id: str | None,
    observed_at: str | None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> list[SchemaUnit]:
    """Extract clusterable schema units from one decoded payload."""
    provider_name = Provider.from_string(provider_name)
    effective_max_samples = max_samples
    if effective_max_samples is None:
        effective_max_samples = config.schema_sample_cap

    if config.sample_granularity == "record":
        artifact = classify_artifact(
            payload,
            provider=provider_name,
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
                bundle_scope=derive_bundle_scope(provider_name, source_path),
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
            provider=provider_name,
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
                bundle_scope=derive_bundle_scope(provider_name, source_path),
                observed_at=observed_at,
                exact_structure_id=schema_cluster_id(sample, artifact.cohort),
                profile_tokens=_document_profile_tokens(sample),
            )
        )
    return units


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
    """Public wrapper used by registry/runtime version matching."""
    return _extract_schema_units(
        payload,
        provider_name=provider_name,
        source_path=source_path,
        raw_id=raw_id,
        observed_at=observed_at,
        config=config,
        max_samples=max_samples,
    )


def _iter_schema_units_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Any:
    """Yield clusterable schema units from raw_conversations."""
    provider_name = Provider.from_string(provider_name)
    conn = sqlite3.connect(db_path)
    try:
        query_provider = config.db_provider_name or provider_name
        where_clause, where_params = _sample_provider_where_clause(query_provider)
        cursor = conn.execute(
            f"""
            SELECT raw_content, source_path, provider_name, payload_provider, raw_id, file_mtime, acquired_at
            FROM raw_conversations
            WHERE {where_clause}
            """,
            where_params,
        )
        batch_size = 1 if config.sample_granularity == "record" else 100
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                if config.sample_granularity == "record":
                    runtime_provider = canonical_runtime_provider(
                        row[3] or row[2],
                    )
                    if canonical_schema_provider(
                        runtime_provider,
                    ) != str(provider_name):
                        continue

                    sample_limit = max_samples
                    if sample_limit is None:
                        sample_limit = config.schema_sample_cap or 128

                    try:
                        samples = extract_record_samples_from_raw_content(
                            row[0],
                            max_samples=sample_limit,
                            record_type_key=config.record_type_key,
                        )
                    except Exception:
                        samples = []

                    if samples:
                        samples = _compact_schema_samples(samples)
                        artifact = classify_artifact(
                            samples,
                            provider=provider_name,
                            source_path=row[1],
                        )
                        if artifact.schema_eligible:
                            yield SchemaUnit(
                                cluster_payload=samples,
                                schema_samples=samples,
                                artifact_kind=artifact.cohort,
                                conversation_id=row[4],
                                raw_id=row[4],
                                source_path=str(row[1]) if row[1] is not None else None,
                                bundle_scope=derive_bundle_scope(provider_name, row[1]),
                                observed_at=row[5] or row[6],
                                exact_structure_id=schema_cluster_id(samples, artifact.cohort),
                                profile_tokens=_record_profile_tokens(
                                    samples,
                                    record_type_key=config.record_type_key,
                                ),
                            )
                            continue

                try:
                    envelope = build_raw_payload_envelope(
                        row[0],
                        source_path=row[1],
                        fallback_provider=row[2],
                        payload_provider=row[3],
                        jsonl_dict_only=config.sample_granularity == "record",
                    )
                except Exception:
                    continue
                if canonical_schema_provider(
                    envelope.provider,
                ) != str(provider_name):
                    continue
                yield from _extract_schema_units(
                    envelope.payload,
                    provider_name=provider_name,
                    source_path=row[1],
                    raw_id=row[4],
                    observed_at=row[5] or row[6],
                    config=config,
                    max_samples=max_samples,
                )
    finally:
        conn.close()


def _iter_schema_units_from_sessions(
    provider_name: Provider,
    session_dir: Path,
    *,
    max_sessions: int | None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Any:
    """Yield clusterable schema units from filesystem session files."""
    provider_name = Provider.from_string(provider_name)
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = max(1, len(jsonl_files) // max_sessions)
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            records.append(parsed)
        except OSError:
            continue

        if not records:
            continue

        yield from _extract_schema_units(
            records,
            provider_name=provider_name,
            source_path=path,
            raw_id=path.stem,
            observed_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            config=config,
            max_samples=max_samples,
        )


def iter_schema_units(
    provider_name: str | Provider,
    *,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> Any:
    """Yield schema units for a provider from DB, with session fallback."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()

    config = _resolve_provider_config(provider_name)
    yielded_any = False

    if config.db_provider_name and db_path.exists():
        for unit in _iter_schema_units_from_db(
            provider_name,
            db_path=db_path,
            config=config,
            max_samples=max_samples,
        ):
            yielded_any = True
            yield unit

    if yielded_any or config.session_dir is None:
        return

    yield from _iter_schema_units_from_sessions(
        provider_name,
        config.session_dir,
        max_sessions=config.max_sessions,
        config=config,
        max_samples=max_samples,
    )


def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: bool = False,
) -> Any:
    """Yield individual sample dicts from the database."""
    provider_name = Provider.from_string(provider_name)
    for unit in _iter_schema_units_from_db(provider_name, db_path=db_path, config=config):
        for sample in unit.schema_samples:
            if with_conv_ids:
                yield sample, unit.conversation_id
            else:
                yield sample


def _iter_samples_from_sessions(
    session_dir: Path,
    *,
    max_sessions: int | None,
) -> Any:
    """Yield individual sample dicts from session files."""
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = max(1, len(jsonl_files) // max_sessions)
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            yield parsed
        except OSError:
            continue


def load_samples_from_db(
    provider_name: str | Provider,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from polylogue database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return []

    config = _resolve_provider_config(provider_name)
    if max_samples is None:
        return list(_iter_samples_from_db(provider_name, db_path=db_path, config=config))
    return collect_limited_samples(
        lambda: _iter_samples_from_db(provider_name, db_path=db_path, config=config),
        limit=max_samples,
        stratify=config.sample_granularity == "record",
        record_type_key=config.record_type_key,
    )


def load_samples_from_sessions(
    session_dir: Path,
    max_sessions: int | None = None,
    max_samples: int | None = None,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Load samples from JSONL session files."""
    if max_samples is None:
        return list(_iter_samples_from_sessions(session_dir, max_sessions=max_sessions))
    return collect_limited_samples(
        lambda: _iter_samples_from_sessions(session_dir, max_sessions=max_sessions),
        limit=max_samples,
        stratify=True,
        record_type_key=record_type_key,
    )


def get_sample_count_from_db(
    provider_name: str | Provider,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return 0

    config = _resolve_provider_config(provider_name)
    provider_tokens = [str(provider_name)]
    if config.db_provider_name and str(config.db_provider_name) not in provider_tokens:
        provider_tokens.append(str(config.db_provider_name))

    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in provider_tokens)
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name IN ({placeholders})
            """,
            provider_tokens,
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


__all__ = [
    "ProviderConfig",
    "PROVIDERS",
    "SchemaUnit",
    "_resolve_provider_config",
    "_sample_provider_where_clause",
    "_iter_samples_from_db",
    "_iter_samples_from_sessions",
    "_iter_schema_units_from_db",
    "_iter_schema_units_from_sessions",
    "derive_bundle_scope",
    "extract_schema_units_from_payload",
    "fingerprint_hash",
    "get_sample_count_from_db",
    "iter_schema_units",
    "load_samples_from_db",
    "load_samples_from_sessions",
    "profile_cluster_id",
    "profile_similarity",
    "schema_cluster_id",
]
