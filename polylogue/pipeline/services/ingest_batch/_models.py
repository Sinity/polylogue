"""SQL constants, protocol classes, and dataclasses for batch ingest."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.schemas.drift_sentinel import SchemaDriftObservation
from polylogue.sinex.models import PublicationPayload
from polylogue.storage.raw.models import RawSessionStateUpdate

if TYPE_CHECKING:
    import aiosqlite

_DEFAULT_INGEST_WORKER_LIMIT = 16
_INGEST_SOFT_BLOB_LIMIT_BYTES = 256 * 1024 * 1024
_INGEST_HIGH_BLOB_LIMIT_BYTES = 512 * 1024 * 1024
_INGEST_EXTREME_BLOB_LIMIT_BYTES = 2048 * 1024 * 1024
_SINEX_STAGED_PAYLOAD_LIMIT_BYTES = 256 * 1024 * 1024


class _RawStateRepositoryLike(Protocol):
    @property
    def source_backend(self) -> _SourceTierBackendLike | None: ...

    async def update_raw_state(self, raw_id: str, *, state: RawSessionStateUpdate) -> object: ...


class _ParsingServiceRawStateLike(Protocol):
    @property
    def repository(self) -> _RawStateRepositoryLike: ...


class _BulkConnectionBackendLike(Protocol):
    def bulk_connection(self) -> AbstractAsyncContextManager[None]: ...


class _SourceTierBackendLike(_BulkConnectionBackendLike, Protocol):
    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    async def save_raw_failure_evidence(
        self,
        raw_id: str,
        *,
        artifact_kind: str,
        support_status: str,
        outcome_code: str,
        retryable: bool | None,
        evidence_ref: str | None,
        remediation: str | None,
        diagnostic: str | None,
    ) -> None: ...


class _ConnectionBackendLike(Protocol):
    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...


@dataclass(slots=True)
class _RawIngestOutcome:
    raw_id: str
    payload_provider: str | None
    validation_status: str
    validation_error: str | None
    parse_error: str | None
    error: str | None
    had_sessions: bool
    # Keep the worker's typed disposition intact through the batch summary so
    # the raw-state persistence boundary can retain the same evidence instead
    # of reconstructing it from free-form error text.
    outcome_code: str = "success"
    retryable: bool | None = False
    evidence_ref: str | None = None
    remediation: str | None = None
    diagnostic: str | None = None


@dataclass(slots=True)
class _IngestBatchSummary:
    outcomes: dict[str, _RawIngestOutcome] = field(default_factory=dict)
    failed_raw_ids: dict[str, str] = field(default_factory=dict)
    skipped_raw_ids: set[str] = field(default_factory=set)
    processed_ids: set[str] = field(default_factory=set)
    changed_session_ids: list[str] = field(default_factory=list)
    fts_repair_session_ids: list[str] = field(default_factory=list)
    publication_payloads_by_raw_id: dict[str, list[PublicationPayload]] = field(default_factory=dict)
    publication_payload_bytes: int = 0
    publication_deferred_raw_ids: set[str] = field(default_factory=set)
    schema_drift_observations: list[SchemaDriftObservation] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "sessions": 0,
            "messages": 0,
            "attachments": 0,
            "session_events": 0,
            "raw_links": 0,
            "skipped_sessions": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
            "skipped_session_events": 0,
            # polylogue-rujy AC4: bytes this batch pushed through the
            # content-addressed blob store for Claude Code tool-result
            # sidecar text (see _preacquire_sidecar_blobs). "new" is bytes
            # whose hash didn't already exist in the store; "dedup" is bytes
            # whose hash was already present (the write was a free no-op).
            "sidecar_blob_bytes_new": 0,
            "sidecar_blob_bytes_dedup": 0,
            "sidecar_blobs_written": 0,
            # polylogue-azf7: records whose sessions materialized without
            # provider assembly enrichment because on-demand sidecar
            # discovery raised mid-ingest (see IngestRecordResult.sessions_
            # unenriched). Surfaces silent title/enrichment degradation
            # that previously only reached a log line.
            "sessions_unenriched": 0,
        }
    )
    changed_counts: dict[str, int] = field(
        default_factory=lambda: {
            "sessions": 0,
            "messages": 0,
            "attachments": 0,
            "session_events": 0,
        }
    )
    parse_failures: int = 0
    total_msgs: int = 0
    total_convos: int = 0
    raw_record_count: int = 0
    worker_count: int = 0
    total_blob_mb: float = 0.0
    total_result_bytes: int = 0
    max_result_bytes: int = 0
    max_result_raw_id: str | None = None
    elapsed_s: float = 0.0
    setup_elapsed_s: float = 0.0
    max_current_rss_mb: float | None = None
    result_wait_s: float = 0.0
    drain_elapsed_s: float = 0.0
    write_elapsed_s: float = 0.0
    max_write_elapsed_s: float = 0.0
    flush_elapsed_s: float = 0.0
    commit_elapsed_s: float = 0.0
    teardown_elapsed_s: float = 0.0
    wal_checkpoint_mode: str = "none"
    wal_bytes_before_checkpoint: int = 0
    wal_bytes_after_checkpoint: int = 0
    wal_checkpointed_pages: int = 0
    wal_busy_pages: int = 0
    wal_checkpoint_elapsed_s: float = 0.0
    wal_checkpoint_error: str | None = None
    worker_progress_in_flight: int = 0
    worker_progress_completed: int = 0
    worker_progress_total: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _IngestWorkerRequest:
    archive_root_str: str
    blob_root_str: str
    validation_mode: str
    measure_ingest_result_size: bool


_SessionEntry = tuple[str, SessionWritePayload]

# Re-exported from canonical source polylogue/core/common.py
