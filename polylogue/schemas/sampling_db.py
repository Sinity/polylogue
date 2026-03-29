"""Database-backed sample loading for schema tooling."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import (
    CORE_RUNTIME_PROVIDERS,
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.lib.raw_payload import extract_record_samples_from_raw_content
from polylogue.logging import get_logger
from polylogue.paths import db_path as default_db_path
from polylogue.storage.blob_store import get_blob_store
from polylogue.schemas.observation import (
    ProviderConfig,
    extract_schema_units_from_payload,
    resolve_provider_config,
)
from polylogue.types import Provider

logger = get_logger(__name__)


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


def _iter_schema_units_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    max_samples: int | None = None,
):
    """Yield clusterable schema units from raw_conversations."""
    provider_name = Provider.from_string(provider_name)
    blob_store = get_blob_store()
    conn = sqlite3.connect(db_path)
    try:
        query_provider = config.db_provider_name or provider_name
        where_clause, where_params = _sample_provider_where_clause(query_provider)
        cursor = conn.execute(
            f"""
            SELECT source_path, provider_name, payload_provider, raw_id, file_mtime, acquired_at
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
                # row indices: 0=source_path, 1=provider_name, 2=payload_provider,
                #              3=raw_id, 4=file_mtime, 5=acquired_at
                source_path = row[0]
                provider_name_col = row[1]
                payload_provider_col = row[2]
                raw_id = row[3]
                file_mtime = row[4]
                acquired_at = row[5]

                # Load raw content from blob store.
                raw_content: Any = blob_store.blob_path(raw_id)

                if config.sample_granularity == "record":
                    runtime_provider = canonical_runtime_provider(payload_provider_col or provider_name_col)
                    if canonical_schema_provider(runtime_provider) != str(provider_name):
                        continue

                    sample_limit = max_samples
                    if sample_limit is None:
                        sample_limit = config.schema_sample_cap or 128

                    try:
                        samples = extract_record_samples_from_raw_content(
                            raw_content,
                            max_samples=sample_limit,
                            record_type_key=config.record_type_key,
                        )
                    except Exception:
                        samples = []

                    if samples:
                        yield from extract_schema_units_from_payload(
                            samples,
                            provider_name=provider_name,
                            source_path=source_path,
                            raw_id=raw_id,
                            observed_at=file_mtime or acquired_at,
                            config=config,
                            max_samples=max_samples,
                        )
                        continue
                
                try:
                    from polylogue.schemas import sampling as sampling_root

                    envelope = sampling_root.build_raw_payload_envelope(
                        raw_content,
                        source_path=source_path,
                        fallback_provider=provider_name_col,
                        payload_provider=payload_provider_col,
                        jsonl_dict_only=config.sample_granularity == "record",
                    )
                except Exception:
                    continue
                if canonical_schema_provider(envelope.provider) != str(provider_name):
                    continue
                yield from extract_schema_units_from_payload(
                    envelope.payload,
                    provider_name=provider_name,
                    source_path=source_path,
                    raw_id=raw_id,
                    observed_at=file_mtime or acquired_at,
                    config=config,
                    max_samples=max_samples,
                )
    finally:
        conn.close()


def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: bool = False,
):
    """Yield individual sample dicts from the database."""
    provider_name = Provider.from_string(provider_name)
    for unit in _iter_schema_units_from_db(provider_name, db_path=db_path, config=config):
        for sample in unit.schema_samples:
            if with_conv_ids:
                yield sample, unit.conversation_id
            else:
                yield sample


def get_sample_count_from_db(
    provider_name: str | Provider,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in the database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return 0

    config = resolve_provider_config(provider_name)
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
    "_iter_samples_from_db",
    "_iter_schema_units_from_db",
    "_sample_provider_where_clause",
    "get_sample_count_from_db",
]
