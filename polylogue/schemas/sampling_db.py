"""Database-backed sample loading for schema tooling."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, overload

from polylogue.lib.json import JSONDocument
from polylogue.lib.provider_identity import (
    CORE_RUNTIME_PROVIDERS,
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.lib.raw_payload import extract_record_samples_from_raw_content
from polylogue.lib.raw_payload.decode import RawPayloadEnvelope
from polylogue.logging import get_logger
from polylogue.paths import db_path as archive_db_path
from polylogue.schemas.observation import (
    ProviderConfig,
    SchemaUnit,
    extract_schema_units_from_payload,
    resolve_provider_config,
)
from polylogue.storage.blob_store import get_blob_store
from polylogue.types import Provider

logger = get_logger(__name__)

SchemaSample: TypeAlias = JSONDocument


@dataclass(frozen=True)
class _RawConversationRow:
    source_path: str | None
    provider_name: str | None
    payload_provider: str | None
    raw_id: str
    file_mtime: str | None
    acquired_at: str | None

    @property
    def observed_at(self) -> str | None:
        return self.file_mtime or self.acquired_at


def _sample_provider_where_clause(provider_name: str | Provider) -> tuple[str, tuple[str, ...]]:
    provider_token = str(Provider.from_string(provider_name))
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        "payload_provider = ? "
        "OR (payload_provider IS NULL AND provider_name = ?) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[str, ...] = (
        provider_token,
        provider_token,
        *CORE_RUNTIME_PROVIDERS,
    )
    return clause, params


def _coerce_schema_row(row: sqlite3.Row) -> _RawConversationRow:
    return _RawConversationRow(
        source_path=row[0],
        provider_name=row[1],
        payload_provider=row[2],
        raw_id=row[3],
        file_mtime=row[4],
        acquired_at=row[5],
    )


def _record_sample_limit(
    *,
    config: ProviderConfig,
    max_samples: int | None,
    full_corpus: bool,
) -> int | None:
    if full_corpus:
        return None
    if max_samples is not None:
        return max_samples
    return config.schema_sample_cap or 128


def _iter_record_stream_units(
    *,
    row: _RawConversationRow,
    provider_name: Provider,
    raw_content: Path,
    config: ProviderConfig,
    max_samples: int | None,
    full_corpus: bool,
) -> Iterator[SchemaUnit]:
    runtime_provider = canonical_runtime_provider(row.payload_provider or row.provider_name)
    if canonical_schema_provider(runtime_provider) != str(provider_name):
        return

    try:
        samples = extract_record_samples_from_raw_content(
            raw_content,
            max_samples=_record_sample_limit(
                config=config,
                max_samples=max_samples,
                full_corpus=full_corpus,
            ),
            record_type_key=config.record_type_key,
        )
    except Exception:
        samples = []

    if not samples:
        return

    yield from extract_schema_units_from_payload(
        samples,
        provider_name=provider_name,
        source_path=row.source_path,
        raw_id=row.raw_id,
        observed_at=row.observed_at,
        config=config,
        max_samples=max_samples,
    )


def _build_raw_payload_envelope_for_row(
    row: _RawConversationRow,
    *,
    provider_name: Provider,
    raw_content: Path,
    config: ProviderConfig,
) -> RawPayloadEnvelope | None:
    try:
        from polylogue.schemas import sampling as sampling_root

        return sampling_root.build_raw_payload_envelope(
            raw_content,
            source_path=row.source_path,
            fallback_provider=row.provider_name or str(provider_name),
            payload_provider=row.payload_provider,
            jsonl_dict_only=config.sample_granularity == "record",
        )
    except Exception:
        return None


def _iter_schema_units_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    max_samples: int | None = None,
    full_corpus: bool = False,
) -> Iterator[SchemaUnit]:
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
            batch = tuple(_coerce_schema_row(row) for row in cursor.fetchmany(batch_size))
            if not batch:
                break
            for row in batch:
                raw_content = blob_store.blob_path(row.raw_id)

                if config.sample_granularity == "record":
                    yielded_record_units = False
                    for unit in _iter_record_stream_units(
                        row=row,
                        provider_name=provider_name,
                        raw_content=raw_content,
                        config=config,
                        max_samples=max_samples,
                        full_corpus=full_corpus,
                    ):
                        yielded_record_units = True
                        yield unit
                    if yielded_record_units:
                        continue

                envelope = _build_raw_payload_envelope_for_row(
                    row,
                    provider_name=provider_name,
                    raw_content=raw_content,
                    config=config,
                )
                if envelope is None:
                    continue
                if canonical_schema_provider(envelope.provider) != str(provider_name):
                    continue
                yield from extract_schema_units_from_payload(
                    envelope.payload,
                    provider_name=provider_name,
                    source_path=row.source_path,
                    raw_id=row.raw_id,
                    observed_at=row.observed_at,
                    config=config,
                    max_samples=max_samples,
                )
    finally:
        conn.close()


@overload
def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: Literal[False] = False,
) -> Iterator[SchemaSample]: ...


@overload
def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: Literal[True],
) -> Iterator[tuple[SchemaSample, str | None]]: ...


def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: bool = False,
) -> Iterator[SchemaSample | tuple[SchemaSample, str | None]]:
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
        db_path = archive_db_path()
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
