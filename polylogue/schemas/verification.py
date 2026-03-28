"""Schema verification workflow for full raw-corpus checks.

This module provides a non-mutating verification pass over ``raw_conversations``
so operators can run explicit schema gates before schema promotion/releases.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.artifact_taxonomy import ArtifactKind
from polylogue.lib.provider_identity import CORE_RUNTIME_PROVIDERS
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.paths import db_path as default_db_path
from polylogue.protocols import ProgressCallback
from polylogue.schemas.packages import SchemaElementManifest, SchemaVersionPackage
from polylogue.schemas.validator import SchemaValidator
from polylogue.storage.artifact_observations import (
    ensure_artifact_observations,
)
from polylogue.storage.artifact_observations import (
    list_artifact_cohorts as list_durable_artifact_cohorts,
)
from polylogue.storage.artifact_observations import (
    list_artifact_observations as list_durable_artifact_observations,
)
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import ArtifactCohortSummary, ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider


@dataclass
class ProviderSchemaVerification:
    """Per-provider schema verification summary."""

    provider: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    drift_records: int = 0
    skipped_no_schema: int = 0
    decode_errors: int = 0
    quarantined_records: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "drift_records": self.drift_records,
            "skipped_no_schema": self.skipped_no_schema,
            "decode_errors": self.decode_errors,
            "quarantined_records": self.quarantined_records,
        }


@dataclass
class SchemaVerificationReport:
    """Aggregate report for schema verification over raw corpus."""

    providers: dict[str, ProviderSchemaVerification]
    max_samples: int | None
    total_records: int
    record_limit: int | None = None
    record_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_samples": self.max_samples if self.max_samples is not None else "all",
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "total_records": self.total_records,
            "providers": {
                provider: stats.to_dict() for provider, stats in sorted(self.providers.items())
            },
        }


@dataclass
class ProviderArtifactProof:
    """Per-provider proof of raw artifact support and linkage."""

    provider: str
    total_records: int = 0
    contract_backed_records: int = 0
    unsupported_parseable_records: int = 0
    recognized_non_parseable_records: int = 0
    unknown_records: int = 0
    decode_errors: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)
    package_versions: dict[str, int] = field(default_factory=dict)
    element_kinds: dict[str, int] = field(default_factory=dict)
    resolution_reasons: dict[str, int] = field(default_factory=dict)
    linked_sidecars: int = 0
    orphan_sidecars: int = 0
    subagent_streams: int = 0
    streams_with_sidecars: int = 0
    sidecar_agent_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "contract_backed_records": self.contract_backed_records,
            "unsupported_parseable_records": self.unsupported_parseable_records,
            "recognized_non_parseable_records": self.recognized_non_parseable_records,
            "unknown_records": self.unknown_records,
            "decode_errors": self.decode_errors,
            "artifact_counts": dict(sorted(self.artifact_counts.items())),
            "package_versions": dict(sorted(self.package_versions.items())),
            "element_kinds": dict(sorted(self.element_kinds.items())),
            "resolution_reasons": dict(sorted(self.resolution_reasons.items())),
            "linked_sidecars": self.linked_sidecars,
            "orphan_sidecars": self.orphan_sidecars,
            "subagent_streams": self.subagent_streams,
            "streams_with_sidecars": self.streams_with_sidecars,
            "sidecar_agent_types": dict(sorted(self.sidecar_agent_types.items())),
        }


@dataclass
class ArtifactProofReport:
    """Aggregate proof report over the raw artifact corpus."""

    providers: dict[str, ProviderArtifactProof]
    total_records: int
    record_limit: int | None = None
    record_offset: int = 0

    @property
    def contract_backed_records(self) -> int:
        return sum(stats.contract_backed_records for stats in self.providers.values())

    @property
    def unsupported_parseable_records(self) -> int:
        return sum(stats.unsupported_parseable_records for stats in self.providers.values())

    @property
    def recognized_non_parseable_records(self) -> int:
        return sum(stats.recognized_non_parseable_records for stats in self.providers.values())

    @property
    def unknown_records(self) -> int:
        return sum(stats.unknown_records for stats in self.providers.values())

    @property
    def decode_errors(self) -> int:
        return sum(stats.decode_errors for stats in self.providers.values())

    @property
    def linked_sidecars(self) -> int:
        return sum(stats.linked_sidecars for stats in self.providers.values())

    @property
    def orphan_sidecars(self) -> int:
        return sum(stats.orphan_sidecars for stats in self.providers.values())

    @property
    def subagent_streams(self) -> int:
        return sum(stats.subagent_streams for stats in self.providers.values())

    @property
    def streams_with_sidecars(self) -> int:
        return sum(stats.streams_with_sidecars for stats in self.providers.values())

    @property
    def artifact_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for kind, count in stats.artifact_counts.items():
                counts[kind] = counts.get(kind, 0) + count
        return dict(sorted(counts.items()))

    @property
    def package_versions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for version, count in stats.package_versions.items():
                counts[version] = counts.get(version, 0) + count
        return dict(sorted(counts.items()))

    @property
    def element_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for element_kind, count in stats.element_kinds.items():
                counts[element_kind] = counts.get(element_kind, 0) + count
        return dict(sorted(counts.items()))

    @property
    def resolution_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for reason, count in stats.resolution_reasons.items():
                counts[reason] = counts.get(reason, 0) + count
        return dict(sorted(counts.items()))

    @property
    def is_clean(self) -> bool:
        return (
            self.unsupported_parseable_records == 0
            and self.unknown_records == 0
            and self.decode_errors == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "total_records": self.total_records,
            "summary": {
                "contract_backed_records": self.contract_backed_records,
                "unsupported_parseable_records": self.unsupported_parseable_records,
                "recognized_non_parseable_records": self.recognized_non_parseable_records,
                "unknown_records": self.unknown_records,
                "decode_errors": self.decode_errors,
                "linked_sidecars": self.linked_sidecars,
                "orphan_sidecars": self.orphan_sidecars,
                "subagent_streams": self.subagent_streams,
                "streams_with_sidecars": self.streams_with_sidecars,
                "artifact_counts": self.artifact_counts,
                "package_versions": self.package_versions,
                "element_kinds": self.element_kinds,
                "resolution_reasons": self.resolution_reasons,
                "clean": self.is_clean,
            },
            "providers": {
                provider: stats.to_dict() for provider, stats in sorted(self.providers.items())
            },
        }


def _verification_provider_clause(providers: list[str]) -> tuple[str, tuple[Any, ...]]:
    provider_placeholders = ",".join("?" for _ in providers)
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        f"payload_provider IN ({provider_placeholders}) "
        f"OR (payload_provider IS NULL AND provider_name IN ({provider_placeholders})) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (
        *providers,
        *providers,
        *CORE_RUNTIME_PROVIDERS,
    )
    return clause, params


def _bounded_window(
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int]:
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    return bounded_limit, bounded_offset


def _iter_verification_rows(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None,
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int, Iterator[sqlite3.Row]]:
    bounded_limit, bounded_offset = _bounded_window(record_limit, record_offset)
    provider_where: str = ""
    where_params: tuple[Any, ...] = ()
    if providers:
        provider_where, where_params = _verification_provider_clause(providers)

    def _rows() -> Iterator[sqlite3.Row]:
        batch_size_limit = 50  # raw_content blobs can be very large
        last_rowid = 0

        if bounded_offset > 0:
            offset_query = "SELECT rowid FROM raw_conversations "
            if provider_where:
                offset_query += f"WHERE {provider_where} "
            offset_query += "ORDER BY rowid LIMIT 1 OFFSET ?"
            row = conn.execute(offset_query, (*where_params, bounded_offset - 1)).fetchone()
            if row is None:
                return
            last_rowid = row[0]

        base_query = (
            "SELECT rowid, raw_id, provider_name, payload_provider, source_path, raw_content "
            "FROM raw_conversations "
        )
        records_fetched = 0
        while True:
            if bounded_limit is not None:
                remaining = bounded_limit - records_fetched
                if remaining <= 0:
                    break
                batch_size = min(batch_size_limit, remaining)
            else:
                batch_size = batch_size_limit

            if provider_where:
                query = base_query + f"WHERE rowid > ? AND ({provider_where}) ORDER BY rowid LIMIT ?"
                params: tuple[Any, ...] = (last_rowid, *where_params, batch_size)
            else:
                query = base_query + "WHERE rowid > ? ORDER BY rowid LIMIT ?"
                params = (last_rowid, batch_size)

            rows = conn.execute(query, params).fetchall()
            if not rows:
                break

            last_rowid = rows[-1]["rowid"]
            records_fetched += len(rows)
            for row in rows:
                yield row

    return bounded_limit, bounded_offset, _rows()


def _increment_count(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def _candidate_provider(row: sqlite3.Row) -> tuple[str, str | None]:
    raw_provider = str(row["provider_name"])
    stored_payload_provider = row["payload_provider"]
    return str(stored_payload_provider or raw_provider), stored_payload_provider


def _subagent_link_key(source_path: str | None) -> str | None:
    normalized = str(source_path or "").replace("\\", "/").lower()
    if not normalized:
        return None
    for suffix in (".meta.json", ".jsonl.txt", ".jsonl", ".ndjson"):
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)]
            if stem.rsplit("/", 1)[-1].startswith("agent-"):
                return stem
    return None


def _sidecar_agent_type(payload: Any) -> str | None:
    if isinstance(payload, dict):
        agent_type = payload.get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        agent_type = payload[0].get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    return None


def _register_resolution(
    stats: ProviderArtifactProof,
    *,
    package: SchemaVersionPackage,
    element: SchemaElementManifest,
    reason: str,
) -> None:
    stats.contract_backed_records += 1
    _increment_count(stats.package_versions, package.version)
    _increment_count(stats.element_kinds, element.element_kind)
    _increment_count(stats.resolution_reasons, reason)


def list_artifact_observation_rows(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> list[ArtifactObservationRecord]:
    """Return durable artifact observations, hydrating historical rows as needed."""
    db_path = db_path or default_db_path()
    if not db_path.exists():
        return []

    bounded_limit, bounded_offset = _bounded_window(record_limit, record_offset)
    with open_connection(db_path) as conn:
        ensure_artifact_observations(conn, providers=providers, refresh_resolutions=True)
        return list_durable_artifact_observations(
            conn,
            providers=providers,
            support_statuses=support_statuses,
            artifact_kinds=artifact_kinds,
            limit=bounded_limit,
            offset=bounded_offset,
        )


def list_artifact_cohort_rows(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> list[ArtifactCohortSummary]:
    """Return durable artifact cohort summaries, hydrating historical rows as needed."""
    db_path = db_path or default_db_path()
    if not db_path.exists():
        return []

    bounded_limit, bounded_offset = _bounded_window(record_limit, record_offset)
    with open_connection(db_path) as conn:
        ensure_artifact_observations(conn, providers=providers, refresh_resolutions=True)
        return list_durable_artifact_cohorts(
            conn,
            providers=providers,
            support_statuses=support_statuses,
            artifact_kinds=artifact_kinds,
            limit=bounded_limit,
            offset=bounded_offset,
        )


def prove_raw_artifact_coverage(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> ArtifactProofReport:
    """Report durable artifact support, unknowns, and Claude sidecar linkage."""
    db_path = db_path or default_db_path()
    bounded_limit, bounded_offset = _bounded_window(record_limit, record_offset)
    if not db_path.exists():
        return ArtifactProofReport(
            providers={},
            total_records=0,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )

    stats_by_provider: dict[str, ProviderArtifactProof] = {}
    linkage_state: dict[str, dict[str, set[str]]] = {}
    observations = list_artifact_observation_rows(
        db_path=db_path,
        providers=providers,
        record_limit=record_limit,
        record_offset=record_offset,
    )
    total_records = len(observations)

    for observation in observations:
        provider = str(observation.payload_provider or Provider.from_string(observation.provider_name))
        stats = stats_by_provider.setdefault(
            provider,
            ProviderArtifactProof(provider=provider),
        )
        stats.total_records += 1
        _increment_count(stats.artifact_counts, observation.artifact_kind)

        if observation.link_group_key is not None:
            state = linkage_state.setdefault(provider, {"sidecars": set(), "streams": set()})
            if observation.artifact_kind == ArtifactKind.AGENT_SIDECAR_META.value:
                state["sidecars"].add(observation.link_group_key)
                if observation.sidecar_agent_type is not None:
                    _increment_count(stats.sidecar_agent_types, observation.sidecar_agent_type)
            elif observation.artifact_kind == ArtifactKind.SUBAGENT_CONVERSATION_STREAM.value:
                state["streams"].add(observation.link_group_key)

        if observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE:
            stats.contract_backed_records += 1
            if observation.resolved_package_version is not None:
                _increment_count(stats.package_versions, observation.resolved_package_version)
            if observation.resolved_element_kind is not None:
                _increment_count(stats.element_kinds, observation.resolved_element_kind)
            if observation.resolution_reason is not None:
                _increment_count(stats.resolution_reasons, observation.resolution_reason)
        elif observation.support_status is ArtifactSupportStatus.UNSUPPORTED_PARSEABLE:
            stats.unsupported_parseable_records += 1
        elif observation.support_status is ArtifactSupportStatus.RECOGNIZED_UNPARSED:
            stats.recognized_non_parseable_records += 1
        elif observation.support_status is ArtifactSupportStatus.UNKNOWN:
            stats.unknown_records += 1
        elif observation.support_status is ArtifactSupportStatus.DECODE_FAILED:
            stats.decode_errors += 1

    for provider, state in linkage_state.items():
        stats = stats_by_provider.setdefault(
            provider,
            ProviderArtifactProof(provider=provider),
        )
        linked = state["sidecars"] & state["streams"]
        stats.linked_sidecars = len(linked)
        stats.orphan_sidecars = len(state["sidecars"] - state["streams"])
        stats.subagent_streams = len(state["streams"])
        stats.streams_with_sidecars = len(linked)

    return ArtifactProofReport(
        providers=stats_by_provider,
        total_records=total_records,
        record_limit=bounded_limit,
        record_offset=bounded_offset,
    )


def verify_raw_corpus(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
    quarantine_malformed: bool = False,
) -> SchemaVerificationReport:
    """Run non-mutating schema verification over ``raw_conversations``.

    Args:
        db_path: Optional SQLite path. Defaults to polylogue default DB.
        providers: Optional filter by runtime payload provider values.
        max_samples: Per-record sample count. ``None`` means validate all dict
            records from list payloads (equivalent to ``all``).
        quarantine_malformed: When true, malformed/decode-failed raw rows are
            marked as failed validation and given parse_error context.
    """
    db_path = db_path or default_db_path()
    if not db_path.exists():
        return SchemaVerificationReport(
            providers={},
            max_samples=max_samples,
            total_records=0,
            record_limit=record_limit,
            record_offset=max(0, record_offset),
        )

    stats_by_provider: dict[str, ProviderSchemaVerification] = {}
    total_records = 0
    bounded_limit, bounded_offset = _bounded_window(record_limit, record_offset)
    provider_filter = set(providers or [])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        quarantine_updates: list[tuple[str, str, str, str | None]] = []
        _ignored_limit, _ignored_offset, rows = _iter_verification_rows(
            conn,
            providers=providers,
            record_limit=record_limit,
            record_offset=record_offset,
        )
        for row in rows:
            candidate_provider, stored_payload_provider = _candidate_provider(row)
            raw_provider = str(row["provider_name"])

            try:
                envelope = build_raw_payload_envelope(
                    row["raw_content"],
                    source_path=str(row["source_path"] or ""),
                    fallback_provider=raw_provider,
                    payload_provider=stored_payload_provider,
                    jsonl_dict_only=True,
                )
                payload = envelope.payload
                malformed_lines = envelope.malformed_jsonl_lines
            except Exception as exc:
                if provider_filter and candidate_provider not in provider_filter:
                    continue
                total_records += 1
                provider_stats = stats_by_provider.setdefault(
                    candidate_provider,
                    ProviderSchemaVerification(provider=candidate_provider),
                )
                provider_stats.total_records += 1
                provider_stats.decode_errors += 1
                if quarantine_malformed:
                    raw_id = str(row["raw_id"])
                    reason = f"Unable to decode payload: {type(exc).__name__}"
                    quarantine_updates.append((raw_id, reason, candidate_provider, stored_payload_provider))
                    provider_stats.quarantined_records += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue

            actual_provider = envelope.provider
            if provider_filter and actual_provider not in provider_filter:
                continue
            total_records += 1
            provider_stats = stats_by_provider.setdefault(
                actual_provider,
                ProviderSchemaVerification(provider=actual_provider),
            )
            provider_stats.total_records += 1
            if not envelope.artifact.schema_eligible:
                provider_stats.skipped_no_schema += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue
            if malformed_lines:
                provider_stats.decode_errors += 1
                if quarantine_malformed:
                    raw_id = str(row["raw_id"])
                    reason = f"Malformed JSONL lines: {malformed_lines}"
                    quarantine_updates.append((raw_id, reason, actual_provider, actual_provider))
                    provider_stats.quarantined_records += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue

            try:
                validator = SchemaValidator.for_payload(
                    actual_provider,
                    payload,
                    source_path=str(row["source_path"] or ""),
                )
            except (FileNotFoundError, ImportError):
                provider_stats.skipped_no_schema += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue

            samples = validator.validation_samples(
                payload,
                max_samples=max_samples,
            )
            if not samples:
                provider_stats.valid_records += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue

            invalid_found = False
            drift_found = False
            for sample in samples:
                result = validator.validate(sample)
                if not result.is_valid:
                    invalid_found = True
                if result.has_drift:
                    drift_found = True

            if invalid_found:
                provider_stats.invalid_records += 1
            else:
                provider_stats.valid_records += 1
            if drift_found:
                provider_stats.drift_records += 1

            if progress_callback is not None:
                progress_callback(1)

        if quarantine_updates:
            validated_at = datetime.now(tz=timezone.utc).isoformat()
            for raw_id, reason, provider, payload_provider in quarantine_updates:
                conn.execute(
                    """
                    UPDATE raw_conversations
                    SET validation_status = 'failed',
                        validation_error = ?,
                        validation_drift_count = 0,
                        validation_provider = ?,
                        validation_mode = 'strict',
                        validated_at = ?,
                        payload_provider = COALESCE(?, payload_provider)
                    WHERE raw_id = ?
                    """,
                    (reason, provider, validated_at, payload_provider, raw_id),
                )
                conn.execute(
                    """
                    UPDATE raw_conversations
                    SET parse_error = COALESCE(parse_error, ?)
                    WHERE raw_id = ? AND parsed_at IS NULL
                    """,
                    (reason, raw_id),
                )
            conn.commit()

    finally:
        conn.close()

    return SchemaVerificationReport(
        providers=stats_by_provider,
        max_samples=max_samples,
        total_records=total_records,
        record_limit=bounded_limit,
        record_offset=bounded_offset,
    )
