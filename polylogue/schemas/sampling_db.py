"""Database-backed sample loading for schema tooling."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, overload

from polylogue.archive.raw_payload import extract_record_samples_from_raw_content
from polylogue.archive.raw_payload.decode import RawPayloadEnvelope
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument
from polylogue.core.provider_identity import (
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.logging import get_logger
from polylogue.paths import db_path as index_db_path
from polylogue.schemas.observation import (
    ProviderConfig,
    SchemaUnit,
    extract_schema_units_from_payload,
    resolve_provider_config,
)
from polylogue.schemas.observation_models import ObservationTerminalRecorder, ObservationTerminalStatus
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.sqlite.connection_profile import connection_context

logger = get_logger(__name__)

SchemaSample: TypeAlias = JSONDocument


def _blob_hash_hex(blob_hash: object) -> str:
    """Return the lowercase hex digest addressing a raw payload blob.

    Native ``raw_sessions.blob_hash`` is the 32-byte SHA-256 digest stored as
    a BLOB; the blob store addresses files by hex digest.
    """
    if isinstance(blob_hash, (bytes, bytearray)):
        return bytes(blob_hash).hex()
    return str(blob_hash)


def _ms_to_iso(value: object) -> str | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        epoch_ms = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class _RawSessionRow:
    source_path: str | None
    origin: str
    raw_id: str
    blob_hash: bytes
    file_mtime_ms: int | None
    acquired_at_ms: int | None
    validation_status: str | None

    @property
    def provider_token(self) -> str:
        try:
            return provider_from_origin(Origin.from_string(self.origin)).value
        except (ValueError, KeyError):
            return Provider.UNKNOWN.value

    @property
    def observed_at(self) -> str | None:
        return _ms_to_iso(self.file_mtime_ms) or _ms_to_iso(self.acquired_at_ms)


def _sample_origins_for_provider(source_name: Provider, config: ProviderConfig) -> tuple[str, ...]:
    """Return the archive origin tokens that hold rows for this provider.

    raw rows carry a single ``origin`` token (#1743). The requested provider —
    and its configured ``db_source_name`` alias, when set — map to archive
    origins via :func:`origin_from_provider`.
    """
    origins: list[str] = [origin_from_provider(source_name).value]
    if config.db_source_name is not None:
        alias = origin_from_provider(Provider.from_string(config.db_source_name)).value
        if alias not in origins:
            origins.append(alias)
    return tuple(origins)


def _sample_provider_where_clause(source_name: str | Provider) -> tuple[str, tuple[str, ...]]:
    provider = Provider.from_string(source_name)
    origin = origin_from_provider(provider).value
    return "origin = ?", (origin,)


def _coerce_schema_row(row: sqlite3.Row) -> _RawSessionRow:
    return _RawSessionRow(
        source_path=row["source_path"],
        origin=str(row["origin"]),
        raw_id=str(row["raw_id"]),
        blob_hash=bytes(row["blob_hash"]) if row["blob_hash"] is not None else b"",
        file_mtime_ms=row["file_mtime_ms"],
        acquired_at_ms=row["acquired_at_ms"],
        validation_status=row["validation_status"],
    )


def _record_terminal(
    recorder: ObservationTerminalRecorder | None,
    row: _RawSessionRow,
    *,
    status: ObservationTerminalStatus,
    reason: str,
    artifact_kind: str | None = None,
) -> None:
    if recorder is None:
        return
    recorder(
        raw_id=row.raw_id,
        status=status,
        artifact_kind=artifact_kind,
        source_path=row.source_path,
        reason=reason,
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
    row: _RawSessionRow,
    source_name: Provider,
    raw_content: Path,
    config: ProviderConfig,
    max_samples: int | None,
    full_corpus: bool,
) -> Iterator[SchemaUnit]:
    runtime_provider = canonical_runtime_provider(row.provider_token)
    if canonical_schema_provider(runtime_provider) != str(source_name):
        return

    samples = extract_record_samples_from_raw_content(
        raw_content,
        max_samples=_record_sample_limit(
            config=config,
            max_samples=max_samples,
            full_corpus=full_corpus,
        ),
        record_type_key=config.record_type_key,
    )

    if not samples:
        return

    yield from extract_schema_units_from_payload(
        samples,
        source_name=source_name,
        source_path=row.source_path,
        raw_id=row.raw_id,
        observed_at=row.observed_at,
        config=config,
        max_samples=max_samples,
    )


def _build_raw_payload_envelope_for_row(
    row: _RawSessionRow,
    *,
    source_name: Provider,
    raw_content: Path,
    config: ProviderConfig,
) -> RawPayloadEnvelope | None:
    from polylogue.schemas import sampling as sampling_root

    return sampling_root.build_raw_payload_envelope(
        raw_content,
        source_path=row.source_path,
        fallback_provider=row.provider_token or str(source_name),
        jsonl_dict_only=config.sample_granularity == "record",
    )


def _iter_schema_units_from_db(
    source_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    max_samples: int | None = None,
    full_corpus: bool = False,
    terminal_recorder: ObservationTerminalRecorder | None = None,
) -> Iterator[SchemaUnit]:
    """Yield clusterable schema units from raw_sessions.

    Raw acquisition rows live in the ``source.db`` tier (#1743). Given an
    ``index.db`` path, the sibling ``source.db`` of the same archive root holds
    ``raw_sessions``; it is opened read-write but only read here.
    """
    source_name = Provider.from_string(source_name)
    source_db_path = db_path.parent / "source.db"
    if not source_db_path.exists():
        return
    blob_store = get_blob_store()
    query_provider = config.db_source_name or source_name
    origins = _sample_origins_for_provider(Provider.from_string(query_provider), config)
    placeholders = ",".join("?" for _ in origins)
    with connection_context(source_db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"""
            SELECT source_path, origin, raw_id, blob_hash, file_mtime_ms, acquired_at_ms,
                   validation_status
            FROM raw_sessions
            WHERE origin IN ({placeholders})
            """,
            origins,
        )
        batch_size = 1 if config.sample_granularity == "record" else 100
        while True:
            batch = tuple(_coerce_schema_row(row) for row in cursor.fetchmany(batch_size))
            if not batch:
                break
            for row in batch:
                raw_content = blob_store.blob_path(_blob_hash_hex(row.blob_hash))

                if row.validation_status == "failed":
                    _record_terminal(
                        terminal_recorder,
                        row,
                        status="quarantined",
                        reason="source_validation_failed",
                    )
                    continue

                if config.sample_granularity == "record":
                    try:
                        record_units = list(
                            _iter_record_stream_units(
                                row=row,
                                source_name=source_name,
                                raw_content=raw_content,
                                config=config,
                                max_samples=max_samples,
                                full_corpus=full_corpus,
                            )
                        )
                    except Exception:
                        logger.exception("Failed to extract record samples from raw content: %s", raw_content)
                        record_units = []
                    if record_units:
                        yield from record_units
                        artifact_kinds = {unit.artifact_kind for unit in record_units}
                        _record_terminal(
                            terminal_recorder,
                            row,
                            status="included",
                            reason="observed_schema_units",
                            artifact_kind=next(iter(artifact_kinds)) if len(artifact_kinds) == 1 else "mixed",
                        )
                        continue

                try:
                    envelope = _build_raw_payload_envelope_for_row(
                        row,
                        source_name=source_name,
                        raw_content=raw_content,
                        config=config,
                    )
                except Exception:
                    logger.exception("Failed to build raw payload envelope for %s", raw_content)
                    _record_terminal(
                        terminal_recorder,
                        row,
                        status="decode_failed",
                        reason="payload_decode_failed",
                    )
                    continue
                if envelope is None:
                    _record_terminal(
                        terminal_recorder,
                        row,
                        status="decode_failed",
                        reason="payload_envelope_missing",
                    )
                    continue
                if canonical_schema_provider(envelope.provider) != str(source_name):
                    _record_terminal(
                        terminal_recorder,
                        row,
                        status="intentionally_excluded",
                        reason="provider_mismatch",
                    )
                    continue
                units = extract_schema_units_from_payload(
                    envelope.payload,
                    source_name=source_name,
                    source_path=row.source_path,
                    raw_id=row.raw_id,
                    observed_at=row.observed_at,
                    config=config,
                    max_samples=max_samples,
                )
                if not units:
                    _record_terminal(
                        terminal_recorder,
                        row,
                        status="unsupported",
                        reason="no_schema_eligible_units",
                    )
                    continue
                yield from units
                artifact_kinds = {unit.artifact_kind for unit in units}
                _record_terminal(
                    terminal_recorder,
                    row,
                    status="included",
                    reason="observed_schema_units",
                    artifact_kind=next(iter(artifact_kinds)) if len(artifact_kinds) == 1 else "mixed",
                )


@overload
def _iter_samples_from_db(
    source_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: Literal[False] = False,
) -> Iterator[SchemaSample]: ...


@overload
def _iter_samples_from_db(
    source_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: Literal[True],
) -> Iterator[tuple[SchemaSample, str | None]]: ...


def _iter_samples_from_db(
    source_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: bool = False,
) -> Iterator[SchemaSample | tuple[SchemaSample, str | None]]:
    """Yield individual sample dicts from the database."""
    source_name = Provider.from_string(source_name)
    for unit in _iter_schema_units_from_db(source_name, db_path=db_path, config=config):
        for sample in unit.schema_samples:
            if with_conv_ids:
                yield sample, unit.session_id
            else:
                yield sample


def get_sample_count_from_db(
    source_name: str | Provider,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in the database."""
    source_name = Provider.from_string(source_name)
    if db_path is None:
        db_path = index_db_path()
    if not db_path.exists():
        return 0

    config = resolve_provider_config(source_name)
    origins = _sample_origins_for_provider(source_name, config)
    placeholders = ",".join("?" for _ in origins)

    with connection_context(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages m
            JOIN sessions c ON m.session_id = c.session_id
            WHERE c.origin IN ({placeholders})
            """,
            origins,
        ).fetchone()
        return row[0] if row else 0


__all__ = [
    "_iter_samples_from_db",
    "_iter_schema_units_from_db",
    "_sample_provider_where_clause",
    "get_sample_count_from_db",
]
