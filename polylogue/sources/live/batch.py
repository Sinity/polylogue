"""In-process live batch convergence for daemon source ingestion."""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

from polylogue.archive.artifact_taxonomy import (
    ArtifactKind,
    classify_artifact_path,
    strong_path_classification,
)
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
    raw_receipt_order_sql,
)
from polylogue.archive.revision_replay import ApplicationDecision, RevisionCandidate, plan_revision_replay
from polylogue.archive.zip_admission import open_zip_entry
from polylogue.config import Source
from polylogue.core.compute import DaemonBackpressureError, DaemonOperationCancelled
from polylogue.core.compute_cancel import raise_if_operation_cancelled
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.degraded import is_fully_degraded
from polylogue.core.enums import Origin, Provider
from polylogue.core.errors import DatabaseError, SchemaVersionMismatchError
from polylogue.core.memory import release_process_memory
from polylogue.core.metrics import (
    read_cgroup_memory_current_mb,
    read_cgroup_memory_peak_mb,
    read_cgroup_memory_swap_current_mb,
    read_cgroup_path,
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)
from polylogue.core.protocols import ArchiveRootOwner
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_coordinates import (
    MemberAddressingMode,
    captured_zip_member_raw_id,
    zip_member_record_coordinate,
    zip_member_source_index,
)
from polylogue.core.raw_failure_evidence import (
    PARTIAL_TRUNCATED_TAIL,
    RAW_FAILURE_EVIDENCE_KINDS,
    RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS,
    PartialAdmission,
    RawFailureEvidenceKind,
    RetainedRawDecodeRefusalError,
)
from polylogue.core.sources import origin_from_provider
from polylogue.core.stage_admission import admit_stage_write
from polylogue.core.storage_faults import (
    ARCHIVE_SIDE_FAULTS,
    CAPACITY_FAULTS,
    StorageFaultKind,
    raise_if_storage_fault,
    storage_fault_kind,
)
from polylogue.core.timestamp_authority import timestamp_millis
from polylogue.logging import ERROR, WARNING, bind, emit, get_logger
from polylogue.pipeline.ingest_outcomes import (
    IngestAttemptDisposition,
    classify_archive_write_exception,
    corrupt_input_disposition,
    downstream_failure_disposition,
    non_session_artifact_disposition,
    success_disposition,
)
from polylogue.sources.acquisition_boundary import (
    admit_bound_bytes,
    capture_bound_path,
    captured_path_coordinate,
    open_bound_container,
    release_refused_capture,
)
from polylogue.sources.decoder_zip import (
    is_declared_artifact_path,
)
from polylogue.sources.decoders import _ZipEntryValidator
from polylogue.sources.dispatch import (
    ForeignOriginContentError,
    bound_location_provider,
    is_jsonl_source_path,
)
from polylogue.sources.live.archive_open import _open_archive_for_live_write, _source_tier_acquisition_required
from polylogue.sources.live.batch_observability import (
    record_attempt_progress,
)
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _MAX_APPEND_PLAN_PAYLOAD_BYTES,
    JsonlBoundary,
    JsonlFrontier,
    LiveRetainedRunner,
    PreAcquisitionDecision,
    _accumulate_stage_timings,
    _append_plan_group_ready,
    _AppendPlan,
    _AppendResult,
    _archive_blob_exists,
    _blob_copy_heartbeat,
    _DeferredAppend,
    _full_ingest_result_from_summary,
    _full_parse_progress_groups,
    _FullIngestHeartbeat,
    _FullIngestResult,
    _ingest_pass_exhausted,
    _path_size,
    _throttled_phase_heartbeat,
    bind_hook_carrier_baseline_revision,
    classify_pre_writer_admissions,
    claude_semantic_frontier_for_prefix,
    claude_semantic_frontier_for_prefix_with_bytes,
    cursor_prefix_hash,
    cursor_state_after_full_ingest,
    decode_claude_semantic_frontier,
    encode_claude_semantic_frontier_digests,
    encode_cursor_hash_authority,
    file_prefix_sha256,
    fingerprint_file,
    foreign_origin_exclusion,
    jsonl_complete_prefix,
    jsonl_complete_prefix_path,
    jsonl_prefix_record_count,
    last_complete_newline_from_tail,
    retryable_read_fault,
    sha256_range_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.batch_support import (
    _FULL_PARSE_PROGRESS_MAX_BYTES as _FULL_PARSE_PROGRESS_MAX_BYTES,
)
from polylogue.sources.live.batch_support import (
    _FULL_PARSE_PROGRESS_MAX_FILES as _FULL_PARSE_PROGRESS_MAX_FILES,
)
from polylogue.sources.live.convergence_debt import (
    ConvergenceDebt,
    convergence_debt_from_state,
    convergence_debt_from_states,
    debt_by_path,
)
from polylogue.sources.live.convergence_outcome import record_convergence_outcomes, settled_convergence_stages
from polylogue.sources.live.cursor import (
    ConvergenceDebtBatchEntry,
    ConvergenceDebtSettlement,
    ConvergenceDebtWrite,
    CursorPathAuthority,
    CursorRecord,
    CursorStore,
)
from polylogue.sources.live.dedup import handle_schema_version_mismatch, handle_structural_database_error
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import (
    REFUSED_CORRUPT_INPUT,
    REFUSED_DAEMON_DEGRADED,
    REFUSED_NO_SESSIONS,
    REFUSED_UNATTEMPTED,
    REFUSED_UNATTEMPTED_TIME_BUDGET,
    SETTLED_EXCLUSION_REASONS,
    LiveBatchMetrics,
    LiveFullIngestAggregate,
    split_offered_bytes,
)
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage, PreparedLiveSQLiteCapture
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.origin_specs import (
    artifact_rule_for_path,
    database_capability_for_provider,
    frontier_kind_for_origin,
    path_declaration_refuses_session,
)
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.retained_acquisition import SourceInputRecord
from polylogue.sources.revision_backfill import (
    RetainedPreparationRetryableError,
)
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    stream_preserved_zip_entry_raw_data,
    zip_acquisition_fingerprint,
)
from polylogue.sources.source_staging import bind_source_input
from polylogue.sources.sqlite_snapshot import (
    codex_state_raw_id,
    hermes_profile_raw_id,
    is_sqlite_path,
    snapshot_sqlite_to_blob,
    sqlite_snapshot_failure_as_oserror,
    sqlite_source_revision,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    archive_tier_spec,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceManifest,
    SourceItemAdmission,
    SourceItemMemberDisposition,
    acquired_zip_manifest,
    complete_source_item_enumeration,
    publish_acquired_zip_input,
    record_source_item_member_disposition,
    source_item_id,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    prepared_row_dispositions,
)
from polylogue.storage.sqlite.connection_profile import (
    attach_readonly_database,
    open_readonly_connection,
    open_source_tier_write_connection,
)
from polylogue.storage.sqlite.write_lease import UnleasedWriteError

if TYPE_CHECKING:
    from polylogue.storage.raw_retention import RawFrontierBlockedPaths

logger = get_logger(__name__)


#: Convergence-debt stage name for the recurring raw-retention owner. Raw
#: retention is one of the four derived/durable-storage domains
#: (polylogue-6kur AC5) that must each have a single ordinary owner with
#: bounded retry; this stage name is how its unfinished work is retained in
#: the shared ``convergence_debt`` ledger and found again on the next pass.
RAW_RETENTION_STAGE = "raw_retention"
#: How many superseded snapshots one pass compacts for one source path. The
#: bound keeps a pass proportional to its batch; the remainder is named in a
#: debt row rather than silently dropped.
RAW_RETENTION_LIMIT_PER_PATH = 25
#: How many retry-due backlog paths one pass additionally drains. The pass
#: stays bounded by its own input plus this constant, never by the ledger.
#: Each pass is one writer admission that resolves retention authority once
#: for all of its paths: the page amortizes that resolution while bounding
#: how long one pass holds the writer; the retry loop, not the page, drains
#: a cold build's backlog.
RAW_RETENTION_BACKLOG_PER_PASS = 64


class CursorAuthorityBlockedError(RuntimeError):
    """The canonical raw frontier proof did not authorize live source selection."""


def _file_observation(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _vanished_truncated_capture(record: RawSessionRecord) -> bool:
    """Whether a JSONL capture ends in an unterminated record and its file is gone."""
    prefix = record.complete_prefix_size
    if not record.requires_complete_record_boundary or prefix is None or not 0 <= prefix < record.blob_size:
        return False
    try:
        Path(record.source_path).stat()
    except FileNotFoundError:
        return True
    return False


def _stable_truncated_tail_admission(record: RawSessionRecord) -> PartialAdmission | None:
    """The typed partial for a stable capture whose final record is truncated."""
    prefix = record.complete_prefix_size
    if prefix is None or not 0 < prefix < record.blob_size:
        return None
    try:
        stable = _file_observation(Path(record.source_path).stat()) == record.captured_file_observation
    except OSError:
        return None
    if not stable:
        return None
    if record.complete_prefix_record_count is None:
        raise AssertionError("a stable partial JSONL admission has no off-writer record count")
    return PartialAdmission(
        reason=PARTIAL_TRUNCATED_TAIL,
        complete_record_count=record.complete_prefix_record_count,
        complete_prefix_bytes=prefix,
        source_bytes=record.blob_size,
    )


def _disposition_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    """Prepared-row dispositions recorded between two snapshots, zero terms dropped."""
    delta = {reason: after[reason] - before.get(reason, 0) for reason in after}
    return {reason: count for reason, count in sorted(delta.items()) if count}


def _is_tool_result_sidecar_path(path: Path, *, provider: Provider) -> bool:
    """Whether ``path`` is a Claude Code ``tool-results/`` sidecar by path rule."""
    classification = strong_path_classification(path, provider=provider)
    return classification is not None and classification.kind is ArtifactKind.TOOL_RESULT_SIDECAR


#: How many source paths one pinned source-tier evidence query may name.
#: SQLite's default host-parameter limit is 999; this stays well inside it
#: while still collapsing a whole cold-build page into a couple of reads.
_SOURCE_EVIDENCE_QUERY_CHUNK = 500


def _retained_raw_fingerprint(raw_id: object, blob_hash: object, *, archive_root: Path) -> str | None:
    """A ``raw_sessions`` row's fingerprint, or ``None`` when its bytes are gone.

    Shared by the single-path and pinned-page reads so both interpret a row
    identically: a raw id is evidence only while the blob it names is still
    on disk.
    """
    if isinstance(blob_hash, bytes):
        blob_hash_hex = blob_hash.hex()
    elif isinstance(blob_hash, str):
        blob_hash_hex = blob_hash.lower()
    else:
        return None
    if not _archive_blob_exists(archive_root, blob_hash_hex):
        return None
    return raw_id if isinstance(raw_id, str) and raw_id else None


LiveBatchEventEmitter = Callable[[str, dict[str, object]], None]
LiveBatchSyncRunner = Callable[..., Awaitable[Any]]
P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AppendCapabilityReceipt:
    """Production append-route capability for one resolved wire selection."""

    provider: str
    package_version: str
    element_kind: str
    status: Literal["supported", "unsupported"]
    reason: str | None
    capability_source: str = "LiveBatchProcessor.append"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "package_version": self.package_version,
            "element_kind": self.element_kind,
            "operation": "append_prefix",
            "status": self.status,
            "reason": self.reason,
            "capability_source": self.capability_source,
        }


_APPEND_CAPABLE_PROVIDER_VALUES = frozenset({Provider.CODEX.value, Provider.CLAUDE_CODE.value})


def append_capability_receipt(
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    stable_session_identity: bool,
) -> AppendCapabilityReceipt:
    """Resolve append support from the live route's identity contract."""
    if provider not in _APPEND_CAPABLE_PROVIDER_VALUES:
        return AppendCapabilityReceipt(
            provider=provider,
            package_version=package_version,
            element_kind=element_kind,
            status="unsupported",
            reason="live append route supports only Codex and Claude Code JSONL identity contracts",
        )
    if not stable_session_identity:
        return AppendCapabilityReceipt(
            provider=provider,
            package_version=package_version,
            element_kind=element_kind,
            status="unsupported",
            reason="append delta requires a stable persisted session identity",
        )
    return AppendCapabilityReceipt(
        provider=provider,
        package_version=package_version,
        element_kind=element_kind,
        status="supported",
        reason=None,
    )


_ARCHIVE_RUNTIME_TIERS = ",".join(spec.tier.value for spec in ARCHIVE_TIER_SPECS.values())
_ARCHIVE_NATIVE_WRITE_TIERS = "source,index"
_FULL_CAPTURE_PREFIX_PROOF_ATTEMPTS = 2
#: How much older than its observation a file's last change must be before
#: an unchanged ``stat`` proves unchanged bytes. Above any filesystem's
#: timestamp granularity, plus slack for a small wall-clock step.
_SETTLED_OBSERVATION_MARGIN_NS = 2_000_000_000


def _settled_observation_unchanged(
    stat: os.stat_result,
    *,
    captured_file_observation: tuple[int, int, int, int, int],
    captured_observed_at_ns: int | None,
) -> bool:
    """Whether the capture's pre-read observation still proves the source's bytes.

    The capture took ``captured_file_observation`` just before reading the
    bytes it hashed. Any write after that ``stat`` sets the inode's change
    time to the time of the write, which is later than the observation --
    so when the file's recorded change time was already *settled* (older
    than the observation by more than the timestamp granularity) and the
    current observation is identical, no write happened since, and the
    captured hash is the file's hash without reading it again. A change
    time inside the margin is the racy case (a write in the same timestamp
    tick is invisible to ``stat``) and falls back to re-hashing, as does any
    difference in device, inode, size, mtime or ctime.
    """
    if captured_observed_at_ns is None:
        return False
    if _file_observation(stat) != captured_file_observation:
        return False
    captured_ctime_ns = captured_file_observation[4]
    return captured_ctime_ns + _SETTLED_OBSERVATION_MARGIN_NS < captured_observed_at_ns


@dataclass(frozen=True)
class _FullCapturePrefixProof:
    """Result of proving a captured full-file prefix is still trustworthy."""

    outcome: Literal["verified", "deferred", "rejected"]
    stat: os.stat_result | None
    bytes_read: int


def _single_route_stage_payload(*, append_file_count: int, full_file_count: int) -> dict[str, object] | None:
    if append_file_count > 0 and full_file_count == 0:
        return {"storage_route": "archive_append"}
    if full_file_count > 0 and append_file_count == 0:
        return {
            "storage_route": "archive_full",
            "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
            "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
        }
    return None


def _iso_to_epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


#: Artifact kinds whose retained bytes other sessions are enriched from
#: (``retained_assembly``): session indexes, prompt history, export asset maps.
_ENRICHMENT_EVIDENCE_KINDS = frozenset({"session_index", "prompt_history_log", "export_asset_index", "export_asset"})


def _enrichment_evidence_first(paths: list[Path], provider: Provider) -> list[Path]:
    """Admit enrichment evidence ahead of the sessions it describes.

    A pass admits records in order, and each session is enriched (or its
    prepared carrier revalidated) against the evidence admitted before it.
    Discovery order puts ``sessions-index.json`` after the UUID-named
    transcripts beside it, so without this a transcript in the same pass as
    its index was published with the heuristic title while retained replay
    of the same bytes used the index. The sort is stable otherwise.
    """

    def evidence_rank(path: Path) -> int:
        rule = artifact_rule_for_path(provider, str(path))
        return 0 if rule is not None and rule.kind in _ENRICHMENT_EVIDENCE_KINDS else 1

    return sorted(paths, key=evidence_rank)


_FullRecordKey = tuple[str, str, int | None]


def _full_record_key(record: RawSessionRecord) -> _FullRecordKey:
    return record.raw_id, record.source_path, record.source_index


@dataclass(slots=True, init=False)
class _CapturedZipEnumeration:
    """Private denominator retained until the actual raw admissions settle."""

    manifest: FrozenSourceManifest
    coordinates: PickleSpool[str]
    dispositions: PickleSpool[SourceInputRecord]
    member_count: int | None
    file_observation: tuple[int, int, int, int, int] | None

    def __init__(
        self,
        manifest: FrozenSourceManifest,
        *,
        file_observation: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        self.manifest = manifest
        self.member_count = None
        self.file_observation = file_observation
        self.coordinates = PickleSpool()
        try:
            self.dispositions = PickleSpool()
        except BaseException:
            self.coordinates.close()
            raise

    @property
    def item_id(self) -> str:
        return source_item_id(
            source_generation_id=self.manifest.source_generation_id,
            logical_coordinate=self.manifest.inputs[0].coordinate,
            addressing_mode="physical-file-v1",
        )

    def close(self) -> None:
        try:
            self.coordinates.close()
        finally:
            self.dispositions.close()


@dataclass(slots=True)
class _ArchiveFullWriteResult:
    raw_ids: dict[_FullRecordKey, str] = field(default_factory=dict)
    # Accepted raw bytes whose selected preparation is still running or
    # awaiting worker capacity. The cursor must defer without spending its
    # finite parse-failure budget.
    preparation_deferred_raw_ids: dict[_FullRecordKey, str] = field(default_factory=dict)
    # Terminal refusals are durably retained and therefore handled by this
    # observation. Keep them separate from accepted raw ids so deferred
    # authority failures remain retryable.
    terminal_raw_ids: dict[_FullRecordKey, str] = field(default_factory=dict)
    # Accepted raws that settled to a terminal outcome with nothing
    # admissible -- no session (a recorded terminal shape outcome) or corrupt
    # input -- keyed to the settled exclusion reason. Their paths still
    # advance the cursor as successes; intake reports them excluded.
    settled_exclusions: dict[_FullRecordKey, str] = field(default_factory=dict)
    # Accepted raws admitted only in part -- a stable capture whose final
    # record is truncated admits its complete records -- with what was left
    # out. Intake reports them admitted with the partial, never a plain success.
    partial_admissions: dict[_FullRecordKey, PartialAdmission] = field(default_factory=dict)
    # A raw whose membership census does not produce an accepted session is
    # still a durably acquired, successfully parsed source observation. The
    # decision can be pending for the materialization conveyor or already
    # resolved as ambiguous/deferred. Neither state is a transient source-file
    # failure: retrying identical bytes burns the live catch-up budget without
    # supplying new authority evidence. Track it separately from ``raw_ids``
    # so the cursor records the observation as complete while the durable raw
    # membership state remains queryable and a later file change reopens it.
    deferred_raw_ids: dict[_FullRecordKey, str] = field(default_factory=dict)
    session_ids: list[str] = field(default_factory=list)
    session_count: int = 0
    message_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # The archive can forget on purpose (polylogue-27m): a record whose blob
    # hash is durably excised is a deliberate skip, not a failure -- tracked
    # separately from ordinary parse/write failures so operators can tell
    # the two apart (summed into ParseResult.excised_skips by the one-shot
    # route in operations/canonical_archive_ingest.py).
    excised_skips: int = 0
    excised_paths: set[Path] = field(default_factory=set)
    # polylogue-11cg9: raw ids never attempted this pass because the declared
    # wall-clock budget (``max_pass_seconds``) was already exceeded before
    # their turn. Not a failure and not a conveyor hand-off -- the record was
    # never opened at all, so its path must stay out of both ``raw_ids`` and
    # ``deferred_raw_ids`` (the two buckets the caller already treats as
    # "durably observed") and out of the caller's success/failure accounting
    # entirely. It remains ordinary backlog: the next catch-up scan or watch
    # tick re-discovers it via the unchanged cursor, exactly like any other
    # untouched file.
    skipped_raw_ids: set[_FullRecordKey] = field(default_factory=set)
    time_budget_exceeded: bool = False
    # polylogue-3ijaa: the writer hold was already past its declared bound when
    # this pass finished writing. The archive is committed, so this is reported
    # rather than raised -- raising after the commit would leave the session
    # written and its cursor unrecorded, and every later pass would re-parse
    # and re-write the same bytes forever. The caller records its cursors and
    # then stops taking new work.


def _snapshot_fault_kinds(exc: BaseException) -> frozenset[StorageFaultKind]:
    """The storage faults a SQLite source export may escape with.

    The export first allocates its staging file in the archive, so an
    ``OSError`` raised directly is archive-side (capacity or read-only). A
    SQLite error translated to ``OSError`` may come from opening the source
    database itself, where only a full archive is unambiguous.
    """
    if isinstance(exc.__cause__, sqlite3.Error):
        return CAPACITY_FAULTS
    return ARCHIVE_SIDE_FAULTS


def _release_unwritten_publication_receipts(source_db_path: Path, records: Sequence[RawSessionRecord]) -> None:
    """Release the blob reservations of records a storage fault left unwritten.

    Best effort against storage that just failed: the first release the
    storage refuses is reported and ends the attempt, and the fault that
    caused it is still the one propagated.
    """
    from polylogue.storage.blob_publication import release_refused_publication_receipt

    for record in records:
        try:
            release_refused_publication_receipt(
                source_db_path,
                record.blob_publication_receipt_id,
                record.blob_hash or record.raw_id,
            )
        except Exception as exc:
            emit(
                "live.ingest.publication_release_failed",
                level=ERROR,
                outcome="error",
                reason="storage_fault",
                raw_id=record.raw_id,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return


@dataclass(slots=True)
class _OpenIngestAttempt:
    """The ``ingest_attempts`` row one ``_ingest_files`` call opened, until it is finished.

    ``scope`` holds the attempt's correlation binding: every event this
    process emits inside the attempt -- across ``asyncio.to_thread`` and the
    writer handoff, which copy the context -- carries its ``attempt_id``,
    the key of ``ingest_attempts`` and ``daemon_stage_events``. The caller
    that created the holder closes the scope when the attempt returns or
    escapes.
    """

    attempt_id: str | None = None
    #: Whether the start write returned, so the row is known to exist.
    started: bool = False
    #: Whether the final close was submitted (it may complete detached).
    finishing: bool = False
    finished: bool = False
    scope: ExitStack = field(default_factory=ExitStack)

    def opened(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        self.scope.enter_context(bind(attempt_id=attempt_id))


#: Typed reason of a ZIP member whose value no archive content identity can hold.
_CONTENT_IDENTITY_REFUSED = "content_identity_refused"


def _zip_member_debt_prefix(path: Path) -> str:
    """The prefix every member debt coordinate of ``path`` starts with, and only those."""
    return f"{path}:#"


def _zip_member_debt_subject(path: Path, ordinal: int, member: str) -> str:
    """Debt coordinate for one ZIP member; the ordinal keeps duplicate names apart."""
    return f"{_zip_member_debt_prefix(path)}{ordinal}:{member}"


def _full_publication_stage_timings(timings: Mapping[str, float]) -> dict[str, float]:
    """Report a retained publication's stages as this full pass's own.

    Live full acquisition no longer parses or writes Index itself; the
    retained owner does both for the raws it acquired. Its replay-route
    segment (``revision_replay``, ``membership_replay``) names how the owner
    published, so the batch reports the stage under ``full.`` instead:
    ``revision_replay.index.session_upsert`` becomes
    ``full.index.session_upsert`` and ``provider_parse`` ``full.provider_parse``.
    """
    reported: dict[str, float] = {}
    for key, elapsed in timings.items():
        route, separator, stage = key.partition(".")
        name = stage if separator and route.endswith("_replay") else key
        reported[f"full.{name}"] = reported.get(f"full.{name}", 0.0) + float(elapsed)
    return reported


def _hook_carrier_raw(source: sqlite3.Connection, raw_id: str, path: Path) -> bool:
    """Whether a raw is a hook-event carrier, whose acquisition is its admission.

    A carrier never yields a session; its events are materialized from the
    retained bytes by the ``hook_events`` derivation. Other declared
    non-session evidence settles as an ordinary no-session exclusion.
    """
    row = source.execute("SELECT detected_provider FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    if row is None or row[0] is None:
        return False
    classification = classify_artifact_path(str(path), provider=Provider.from_string(str(row[0])))
    return classification is not None and classification.kind is ArtifactKind.HOOK_EVENT_CARRIER


def _admit_live_full_raw(
    archive: ArchiveStore,
    record: RawSessionRecord,
    payload: bytes | None,
    *,
    acquisition_provider: Provider,
    acquired_at_ms: int,
) -> tuple[str, str]:
    """Admit one captured record before retained preparation acquires its original witness."""
    blob_hash = record.blob_hash or record.raw_id
    if payload is None:
        source_raw_id = archive.write_raw_blob_ref(
            provider=acquisition_provider,
            capture_mode=record.capture_mode,
            blob_hash_hex=blob_hash,
            blob_size=record.blob_size,
            source_path=record.source_path,
            canonical_source_path=record.frozen_canonical_source_path(),
            captured_profile_key=record.captured_profile_key,
            captured_zip_coordinate=record.captured_zip_coordinate,
            source_item=record.source_item,
            addressing_mode=record.addressing_mode,
            content_identity=record.content_identity,
            source_index=record.source_index or 0,
            # A populated ``blob_hash`` field means this
            # record already has a durable blob reference.
            # SQLite snapshots and ZIP members both keep a
            # raw id distinct from that blob address: the
            # former is profile/path scoped, the latter is
            # coordinate scoped.  Preserve that identity at
            # source admission rather than collapsing either
            # kind back onto a shared content hash.
            raw_id=(record.raw_id if record.blob_hash is not None else None),
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=timestamp_millis(record.file_mtime),
            blob_publication_receipt_id=record.blob_publication_receipt_id,
            post_parse=True,
        )
        source_write_name = "full.source_raw_blob_ref_write"
    else:
        source_raw_id = archive.write_raw_payload(
            provider=acquisition_provider,
            capture_mode=record.capture_mode,
            payload=payload,
            source_path=record.source_path,
            canonical_source_path=record.frozen_canonical_source_path(),
            captured_profile_key=record.captured_profile_key,
            captured_zip_coordinate=record.captured_zip_coordinate,
            source_item=record.source_item,
            addressing_mode=record.addressing_mode,
            content_identity=record.content_identity,
            source_index=record.source_index or 0,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=timestamp_millis(record.file_mtime),
            blob_publication_receipt_id=record.blob_publication_receipt_id,
            raw_id=record.raw_id if record.captured_zip_coordinate is not None else None,
            post_parse=True,
        )
        source_write_name = "full.source_raw_write"
    path_artifact = classify_artifact_path(record.source_path, provider=acquisition_provider)
    if path_artifact is not None and path_artifact.kind is ArtifactKind.HOOK_EVENT_CARRIER:
        # A carrier's first capture is the full baseline of its physical
        # append chain; later growth binds append revisions onto it.
        bind_hook_carrier_baseline_revision(
            archive,
            source_raw_id,
            provider=acquisition_provider,
            source_path=record.source_path,
            source_revision=blob_hash,
        )
    return source_raw_id, source_write_name


class LiveBatchProcessor:
    """Run the daemon live ingest batch path without filesystem watching."""

    def __init__(
        self,
        polylogue: ArchiveRootOwner,
        sources: Iterable[Any],
        *,
        cursor: CursorStore,
        parser_fingerprint: str | Callable[[], str],
        converger: object | None = None,
        stop_requested: Callable[[], bool] | None = None,
        event_emitter: LiveBatchEventEmitter | None = None,
        sync_runner: LiveBatchSyncRunner | None = None,
        convergence_runner: LiveBatchSyncRunner | None = None,
        append_runner: Callable[[Any, list[_AppendPlan]], Awaitable[_AppendResult]] | None = None,
        retained_runner: LiveRetainedRunner | None = None,
        sqlite_capture_stage: LiveSQLiteCaptureStage | None = None,
    ) -> None:
        self._refused_paths: frozenset[Path] = frozenset()
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor
        # ZIP member coordinates refused during the current archive pass, so
        # debt for members a later revision removed can be cleared.
        self._zip_member_refusals_this_pass: dict[str, set[str]] = {}
        self._parser_fingerprint = parser_fingerprint
        self._converger = converger
        self._stop_requested = stop_requested or (lambda: False)
        self._event_emitter = event_emitter
        self._sync_runner = sync_runner
        self._convergence_runner = convergence_runner
        self._append_runner = append_runner
        self._retained_runner = retained_runner
        self._last_cursor_write_stale = False
        self._last_append_cursor_proof_bytes = 0
        # Set for the duration of one pass's cursor-commit loop by
        # ``_pinned_source_tier_evidence``; ``None`` everywhere else, which is
        # what keeps every caller outside that loop on its own read.
        self._pinned_raw_fingerprints: dict[str, str | None] | None = None
        self._pinned_history_sidecars: dict[str, bool] | None = None
        self._raw_compaction_min_acquired_at = datetime.now(UTC).isoformat()
        # One-shot channels out of ``_resynthesize_cursor_from_source``, which
        # cannot widen its own return type without rewriting every one of its
        # refusal exits. ``_append_plan`` drains both immediately after
        # calling it, so neither outlives a single planning attempt.
        self._pending_legacy_promotions: dict[Path, Callable[[], bool]] = {}
        self._resynthesized_cursors: dict[Path, CursorRecord] = {}
        # The watcher supplies a parse stage for JSON/JSONL preparation before
        # the writer hold. Direct callers may pass None for baseline parity.
        self._sqlite_capture_stage = sqlite_capture_stage

    def cursor_authority_block_reason(self) -> str | None:
        """Return the canonical frontier reason that blocks live ingestion.

        Small unit tests may exercise the batch processor before an active
        archive has been bootstrapped. The real watcher cannot write without
        these tiers, so the preflight is intentionally deferred until they
        exist. Once they do, the readiness proof is fail-closed and shared
        with raw convergence, recovery, and reindex.
        """
        if _source_tier_acquisition_required():
            # The derived tier is explicitly unavailable in this mode. Raw
            # admission establishes source authority without consulting or
            # mutating it; trying to resolve the active index pointer here
            # would defeat the acquire-only route before it can run.
            return None
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        if (
            not (archive_root / "source.db").is_file()
            or not ArchiveLocation.resolve(archive_root).active_index_path.is_file()
        ):
            return None
        from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

        return raw_frontier_source_selection_block_reason(archive_root)

    def require_cursor_authority(self, paths: Iterable[Path] | None = None) -> None:
        """Fail closed before a live batch can create attempts or write data."""
        selected = [Path(path) for path in paths] if paths is not None else None
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        if _source_tier_acquisition_required():
            return None
        source_present = (archive_root / "source.db").is_file()
        index_present = ArchiveLocation.resolve(archive_root).active_index_path.is_file()
        if not source_present and not index_present:
            from polylogue.storage.sqlite.archive_tiers.archive_plan import archive_format_marker_path

            if archive_format_marker_path(archive_root).exists():
                raise CursorAuthorityBlockedError(
                    "live watcher source-selection gate blocked: archive tiers disappeared"
                )
            # Direct construction in a small test may precede archive bootstrap.
            return None
        from polylogue.storage.frontier_existence import raw_existence_block_reason, stable_selected_authority_frame

        def require_global_existence() -> None:
            global_reason = raw_existence_block_reason(archive_root)
            if global_reason is not None:
                raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {global_reason}")

        require_global_existence()
        if selected is None:
            # The production entry points pass their exact page. Keep the
            # pathless diagnostic contract for callers without a selection.
            try:
                with stable_selected_authority_frame(archive_root):
                    reason = self.cursor_authority_block_reason()
                    if reason is not None:
                        raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {reason}")
                    require_global_existence()
            except (OSError, ValueError) as exc:
                raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {exc}") from exc
            return None
        try:
            with stable_selected_authority_frame(archive_root):
                blocked = self._blocked_source_paths(selected)
                # The selected proof runs on a separate pinned read. Consume
                # any new global journal evidence before leaving the frame.
                require_global_existence()
        except (OSError, ValueError) as exc:
            raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {exc}") from exc
        reason = blocked.unattributed_reason or "selected source frontier is violated"
        if blocked.unattributed_reason is None:
            if not blocked.source_paths:
                # Only authority gaps remain; ingesting their paths is what
                # resolves them, so nothing is refused.
                logger.debug("live.watcher: cursor authority names only resolvable gaps: %s", reason)
                return None
            # Resolve BOTH sides. A violation recorded through a symlinked
            # watch root is stored under that spelling; a restart configured
            # with the real path then selected the physically identical file
            # and matched nothing, so the per-path gate admitted a source it
            # already knows is blocked.
            blocked_spellings = set(blocked.source_paths)
            for blocked_path in blocked.source_paths:
                try:
                    blocked_spellings.add(str(Path(blocked_path).resolve()))
                except OSError:
                    continue
            refused = frozenset(
                path for path in selected if str(path) in blocked_spellings or str(path.resolve()) in blocked_spellings
            )
            if len(refused) < len(selected):
                self._refused_paths = refused
                if refused:
                    logger.warning(
                        "live.watcher: cursor authority refused %d of %d path(s) in this batch: %s",
                        len(refused),
                        len(selected),
                        reason,
                    )
                return None
        raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {reason}")

    def admit_paths(self, paths: Iterable[Path]) -> list[Path]:
        """The subset of ``paths`` the frontier proof admits, in order.

        Raises when nothing may proceed: every path is refused, or the refusal
        is one no path explains.
        """
        selected = list(paths)
        self.require_cursor_authority(selected)
        refused = self._refused_paths
        self._refused_paths = frozenset()
        return [path for path in selected if path not in refused]

    def _blocked_source_paths(self, paths: Sequence[Path]) -> RawFrontierBlockedPaths:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        from polylogue.storage.raw_retention import raw_frontier_blocked_selected_paths

        return raw_frontier_blocked_selected_paths(archive_root, paths)

    async def ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
        max_pass_seconds: float | None = None,
        whole_archive_convergence: bool = True,
        defer_convergence: bool = False,
    ) -> LiveBatchMetrics:
        """Ingest files in batch, run post-ingest convergence, and return metrics.

        ``whole_archive_convergence=False`` bounds post-ingest convergence to
        this batch's own subjects. ``defer_convergence`` retains the durable
        raw and cursor commits while leaving derived work for a later bounded
        catch-up batch. It also leaves existing convergence debt untouched;
        only an executed pass may resolve that evidence.

        Each ops-tier publication takes the daemon writer through
        :meth:`_run_sync`.  The shared ops connection remains inside the
        synchronous archive-publication worker, where its thread-local scope
        cannot leak over page planning, parsing, or convergence.
        """
        # A cold build's ops checkpoint holder spans the whole page: the
        # archive pass closes before this page's cursor, convergence and
        # attempt writes, and releasing it there made each of those
        # publications checkpoint ``ops.db`` on close again.
        from polylogue.sources.live.cold_build import active_cold_build_generation

        cold_build = active_cold_build_generation(
            Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        )
        if cold_build is not None:
            cold_build.begin_ops_page()
        attempt = _OpenIngestAttempt()
        try:
            with attempt.scope:
                try:
                    return await self._ingest_files(
                        paths,
                        queued_file_count=queued_file_count,
                        skipped_file_count=skipped_file_count,
                        emit_event=emit_event,
                        max_pass_seconds=max_pass_seconds,
                        whole_archive_convergence=whole_archive_convergence,
                        defer_convergence=defer_convergence,
                        open_attempt=attempt,
                    )
                except (asyncio.CancelledError, DaemonOperationCancelled):
                    # Cancellation is shutdown: no further ops write is admitted
                    # here (it could hold the writer past the shutdown deadline).
                    # The row stays ``running`` and the next start records it as
                    # ``interrupted``, which is what happened; this event says so
                    # now, with the attempt it names.
                    if attempt.attempt_id is not None and not attempt.finished:
                        emit(
                            "live.ingest.attempt_cancelled",
                            level=WARNING,
                            outcome="refused",
                            # Cancelled before the start write returned: the row
                            # exists only if that write was already admitted (the
                            # coordinator then finishes it detached); a queued
                            # start never commits. Say which case was observed.
                            reason=(
                                "cancelled_during_finish"
                                if attempt.finishing
                                else "cancelled"
                                if attempt.started
                                else "cancelled_before_start_confirmed"
                            ),
                            attempt_id=attempt.attempt_id,
                        )
                    raise
                except Exception as exc:
                    raise_if_operation_cancelled(exc)
                    await self._finish_escaped_attempt(attempt, exc)
                    raise_if_storage_fault(exc)
                    raise
        finally:
            if cold_build is not None:
                cold_build.end_ops_page()

    async def _finish_escaped_attempt(self, attempt: _OpenIngestAttempt, exc: Exception) -> None:
        """Close the attempt row an escaping exception left ``running``.

        Lock contention, a storage fault or an unexpected defect can leave the
        batch before its ordinary finish. Without this the row stays
        ``running`` -- status reports the page as still in flight -- until the
        next daemon start relabels it ``interrupted``, which is not what
        happened. The classification is the same one the in-batch handlers
        use. If the ops tier itself refuses the write (it may share the full
        disk), the refusal is reported and the original exception still
        propagates unchanged.
        """
        if attempt.attempt_id is None or attempt.finished or not attempt.started:
            # No row is known to exist (the start write itself failed), so
            # there is nothing to close and no attempt to name.
            return
        # A spent writer hold is a property of the pass, not of these inputs
        # (the adapter reports the page retryable); record it that way rather
        # than as the non-retryable parser-defect fallback.
        disposition = classify_archive_write_exception(exc)
        fault = storage_fault_kind(exc)
        emit(
            "live.ingest.attempt_escaped",
            level=ERROR if fault is not None else WARNING,
            outcome="error",
            reason=(f"storage_fault.{fault.value}" if fault is not None else disposition.outcome_code),
            attempt_id=attempt.attempt_id,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        try:
            closed = await self._run_ops_write(
                "attempt_finish",
                self._cursor.finish_ingest_attempt,
                attempt.attempt_id,
                status="failed",
                phase="failed",
                error=f"{type(exc).__name__}: {exc}",
                disposition=disposition,
            )
        except Exception as finish_exc:
            emit(
                "live.ingest.attempt_finish_failed",
                level=ERROR,
                outcome="error",
                reason="ops_write_refused",
                attempt_id=attempt.attempt_id,
                error_type=type(finish_exc).__name__,
                error_detail=str(finish_exc),
            )
            return
        if closed is False:
            # The bounded cursor-write retries gave up on a locked ops tier:
            # the row is still ``running``, and saying so is the point.
            emit(
                "live.ingest.attempt_finish_failed",
                level=ERROR,
                outcome="error",
                reason="ops_write_skipped",
                attempt_id=attempt.attempt_id,
            )
            return
        attempt.finished = True

    async def _ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
        max_pass_seconds: float | None = None,
        whole_archive_convergence: bool = True,
        defer_convergence: bool = False,
        open_attempt: _OpenIngestAttempt | None = None,
    ) -> LiveBatchMetrics:
        """Body of :meth:`ingest_files`, with each ops write separately admitted."""
        if is_fully_degraded():
            # The daemon has been marked structurally unable to ingest (e.g.
            # schema mismatch detected at preflight or on the first batch).
            # Do not enter the full-parse path — that is what produced the
            # IOPS storm in #1003 — nor the source-selection gate, which reads
            # the archive's existence journals on every call.
            return self._degraded_skip_metrics(paths, queued_file_count, skipped_file_count)
        self.require_cursor_authority(paths)
        refused_paths = self._refused_paths
        self._refused_paths = frozenset()
        if refused_paths:
            paths = [path for path in paths if path not in refused_paths]
            skipped_file_count += len(refused_paths)
        batch_started = time.perf_counter()
        # polylogue-11cg9: a dedicated monotonic reference for the
        # max_pass_seconds budget, separate from ``batch_started`` (used for
        # the unrelated ``total_time_s`` instrumentation) -- matches the
        # ``pass_started_monotonic`` convention already used by de2a's
        # raw-materialization checkpoint and qlae's drive-catchup checkpoint.
        pass_started_monotonic = time.monotonic()
        db_bytes_before = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        # Sizes are captured once so the offered / ingested / failed / refused
        # split below reconciles exactly, even if a file grows mid-batch.
        path_sizes = {path: _path_size(path) for path in paths}
        input_bytes = sum(path_sizes.values())
        # The key is chosen before the write: if cancellation detaches the
        # admitted start (the coordinator shields an acquired write), the row
        # can still commit, and the cancelled attempt must be able to name it.
        attempt_id = str(uuid.uuid4())
        if open_attempt is not None:
            open_attempt.opened(attempt_id)
        attempt_id = await self._run_ops_write(
            "attempt_start",
            self._cursor.begin_ingest_attempt,
            paths=paths,
            input_bytes=input_bytes,
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            attempt_id=attempt_id,
        )
        if open_attempt is not None:
            open_attempt.started = True
        await self._record_attempt_progress_admitted(
            attempt_id,
            phase="planning",
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            input_bytes=input_bytes,
            succeeded_file_count=0,
            failed_file_count=0,
            source_payload_read_bytes=0,
            cursor_fingerprint_read_bytes=0,
            parse_time_s=0.0,
            convergence_time_s=0.0,
            total_time_s=0.0,
        )
        source_payload_read_bytes = 0
        cursor_fingerprint_read_bytes = 0
        stale_cursor_write_count = 0
        stale_cursor_paths: set[str] = set()
        parse_time_s = 0.0
        convergence_time_s = 0.0
        raw_compaction_runs = 0
        stage_timings: dict[str, float] = {}
        failed_paths: list[str] = []
        excluded_by_path: dict[Path, str] = {}
        detection_fallbacks_by_path: dict[Path, str] = {}
        settled_exclusions: dict[Path, str] = {}
        partial_admissions: dict[Path, PartialAdmission] = {}
        succeeded_paths: set[Path] = set()
        # polylogue-cnu3: the most severe structural disposition this batch
        # hit, if any. Set at each terminal except-clause below by
        # classifying the real caught exception's type (never by
        # text-matching the ``error`` string). ``None`` at the end means the
        # batch completed cleanly.
        attempt_disposition: IngestAttemptDisposition | None = None
        ingest_worker_count_max = 0
        full_ingest_aggregate = LiveFullIngestAggregate()
        cursor_records = self._cursor.get_records(paths)
        append_file_count = 0
        pending_append_plans: list[_AppendPlan] = []
        full_paths: list[Path] = []
        deferred_paths: list[Path] = []
        preparation_deferred_paths: set[Path] = set()
        source_read_deferred_paths: set[Path] = set()
        # Identity-scoped session touches for this batch (polylogue-20d.13):
        # collected as (source_name, session_id) pairs so the daemon can emit
        # session.appended/session.updated/message.appended events carrying
        # real refs instead of an unscoped aggregate.
        new_session_touches: list[tuple[str, str]] = []
        updated_session_touches: list[tuple[str, str]] = []

        async def flush_append_plans() -> None:
            nonlocal convergence_time_s
            nonlocal cursor_fingerprint_read_bytes
            nonlocal ingest_worker_count_max
            nonlocal parse_time_s
            nonlocal pending_append_plans
            nonlocal stale_cursor_write_count
            nonlocal stale_cursor_paths
            if not pending_append_plans:
                return
            plans = pending_append_plans
            pending_append_plans = []
            await self._record_attempt_progress_admitted(
                attempt_id,
                phase="append_parse",
                succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                current_source=plans[0].source_name,
                current_path=plans[0].path,
                stage_payload={"storage_route": "archive_append"},
            )
            t0 = time.perf_counter()
            try:
                if self._append_runner is None:
                    raise RuntimeError("append ingestion requires the canonical prepared raw owner")
                append_result = await self._append_runner(self, plans)
            except SchemaVersionMismatchError as exc:
                handle_schema_version_mismatch(plans[0].source_name, exc)
                for plan in plans:
                    failed_paths.append(str(plan.path))
                # Use an empty result so the per-plan cleanup loop below
                # (``for plan in append_result.failed``) does NOT re-push the
                # same paths into ``failed_paths`` and does NOT call
                # ``_record_failed_cursor`` against the DB we already know is
                # structurally unusable. The by_source loop further down also
                # checks ``is_fully_degraded()`` and skips the full-parse phase.
                append_result = _AppendResult(succeeded=[], failed=[], worker_count=0)
            ingest_worker_count_max = max(ingest_worker_count_max, append_result.worker_count)
            parse_time_s += time.perf_counter() - t0
            _accumulate_stage_timings(stage_timings, append_result.stage_timings_s)
            append_stage_payload: dict[str, object] = {
                "storage_route": "archive_append",
                "append_stage_timings_s": {
                    name: round(elapsed, 6) for name, elapsed in append_result.stage_timings_s.items()
                },
            }
            release_process_memory()
            await self._record_attempt_progress_admitted(
                attempt_id,
                phase="convergence",
                succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                convergence_time_s=convergence_time_s,
                current_source=plans[0].source_name,
                current_path=plans[0].path,
                stage_payload=append_stage_payload,
            )
            convergence_debt: list[ConvergenceDebt] = []
            convergence_settlements: list[ConvergenceDebtSettlement] = []
            if not defer_convergence:
                (
                    _converged_paths,
                    elapsed,
                    timings,
                    convergence_debt,
                    convergence_settlements,
                ) = await self._run_convergence_paths(
                    "watcher.live_ingest.append_convergence",
                    [plan.path for plan in append_result.succeeded],
                    whole_archive=whole_archive_convergence,
                    session_ids=tuple(append_result.session_ids_by_path.values()),
                )
                convergence_time_s += elapsed
                release_process_memory()
                _accumulate_stage_timings(stage_timings, timings)
            debt_by_source_path = debt_by_path(convergence_debt)
            outcome_items: list[tuple[Path, Iterable[ConvergenceDebt]]] = []
            for plan in append_result.succeeded:
                outcome_items.append((plan.path, debt_by_source_path.get(plan.path, ())))
            if outcome_items and not defer_convergence:
                # Publish the retry obligation before advancing any cursor in
                # this group. If ops.db remains locked, the batch raises and
                # the unchanged source frontier will be offered again.
                await self._run_ops_write(
                    "convergence_outcomes",
                    self._record_convergence_outcomes,
                    outcome_items,
                    convergence_settlements,
                )
            for plan in append_result.succeeded:
                succeeded_paths.add(plan.path)
                if not await self._run_ops_write("cursor_append", self._record_append_cursor, plan):
                    stale_cursor_write_count += 1
                    stale_cursor_paths.add(str(plan.path))
                cursor_fingerprint_read_bytes += self._last_append_cursor_proof_bytes
                session_id = append_result.session_ids_by_path.get(plan.path)
                if session_id:
                    updated_session_touches.append((plan.source_name, session_id))
            for plan in append_result.failed:
                failed_paths.append(str(plan.path))
                cursor_fingerprint_read_bytes += await self._run_ops_write(
                    "cursor_failed", self._record_failed_cursor, plan.path
                )
            for plan in append_result.deferred:
                # polylogue-hat0: this plan's bytes are already durably
                # written and revision-bound in source.db (write_raw_payload
                # ran before the quarantined/ambiguous classification), just
                # not yet accepted into the replay chain. Mark exactly this
                # range as the pending-authority frontier so the next
                # observation of an unchanged file recognizes there is
                # nothing new to capture instead of re-mining an identical
                # duplicate raw row forever.
                cursor_fingerprint_read_bytes += await self._run_ops_write(
                    "cursor_deferred",
                    record_deferred_append_cursor,
                    self._cursor,
                    plan.path,
                    cursor=self._cursor.get_record(plan.path),
                    parser_fingerprint=self._current_parser_fingerprint(),
                    source_name=plan.source_name,
                    deferred_end_offset=plan.last_complete_newline,
                )
                deferred_paths.append(plan.path)

        for path in paths:
            if _source_tier_acquisition_required():
                # Append planning and replay both consult the active index to
                # prove lineage. In acquire-only mode the derived tier is the
                # unavailable component, so capture the complete source
                # observation through the source-only full route instead.
                full_paths.append(path)
                continue
            if is_fully_degraded():
                full_paths.append(path)
                continue
            cursor = cursor_records.get(path)
            # ``cursor_records`` is this batch's own single read of every
            # offered path, so a path missing from it has no cursor row --
            # asking ``ingest_cursor`` again per path is one more ops.db
            # connection for an answer the batch already holds.
            append_plan = (
                self.plan_append(path, cursor=cursor, cursor_is_known=True)
                if self._can_ingest_appends_directly()
                else None
            )
            # Drain unconditionally: an entry left behind would be read by a
            # later pass as though it described that pass's observation.
            # Planning may have rebuilt a cursor from durable evidence, and
            # without it the deferral below has nothing to persist and
            # re-mints the same resynthesis on every tick.
            resynthesized = self._resynthesized_cursors.pop(path, None)
            if cursor is None:
                cursor = resynthesized
            if isinstance(append_plan, _DeferredAppend):
                # No new authority-relevant append happened this pass (no
                # complete trailing newline yet, or -- polylogue-hat0 --
                # _append_plan itself recognized an already-pending deferred
                # range with no growth past it). Preserve any existing
                # pending-authority marker unchanged rather than clearing it.
                cursor_fingerprint_read_bytes += await self._run_ops_write(
                    "cursor_deferred",
                    record_deferred_append_cursor,
                    self._cursor,
                    path,
                    cursor=cursor,
                    parser_fingerprint=self._current_parser_fingerprint(),
                    source_name=self._source_name_for(path),
                    deferred_end_offset=cursor.deferred_end_offset if cursor is not None else None,
                )
                deferred_paths.append(path)
            elif append_plan is None:
                full_paths.append(path)
            else:
                pending_append_plans.append(append_plan)
                append_file_count += 1
                source_payload_read_bytes += append_plan.bytes_read
                cursor_fingerprint_read_bytes += append_plan.authority_bytes_read
                if _append_plan_group_ready(pending_append_plans):
                    await flush_append_plans()
        await flush_append_plans()

        by_source: dict[str, list[Path]] = {}
        for path in full_paths:
            by_source.setdefault(self._source_name_for(path), []).append(path)

        # polylogue-11cg9: a shared clock + "processed at least one group"
        # flag across the *whole* by-source/progress-group nesting, so the
        # declared budget bounds this entire ``ingest_files`` call the same
        # way de2a/qlae bound their own passes -- not a fresh budget re-armed
        # per source or per progress group, which would let N groups each
        # individually "in budget" sum to an unbounded total hold. The first
        # group across every source always completes regardless of budget
        # (forward-progress guarantee); only later groups are ever skipped.
        full_ingest_time_budget_exceeded = False
        processed_any_full_group = False
        for source_name, grouped_paths in by_source.items():
            if is_fully_degraded():
                # A prior source group hit a structural error this batch.
                # Don't burn IOPS on remaining groups.
                for path in grouped_paths:
                    failed_paths.append(str(path))
                continue
            if full_ingest_time_budget_exceeded:
                # Remaining sources stay ordinary backlog for the next tick;
                # nothing here was attempted, so nothing to mark failed.
                break
            # Evidence goes first across the whole source, before the list is
            # split into progress groups; a later group cannot revisit an
            # earlier one (see ``_enrichment_evidence_first``).
            ordered_paths = _enrichment_evidence_first(
                list(grouped_paths),
                Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name)),
            )
            progress_groups = list(_full_parse_progress_groups(ordered_paths))
            held_pending: list[Path] = []
            while progress_groups:
                source_paths = progress_groups.pop(0)
                if self._stop_requested():
                    break
                if is_fully_degraded():
                    break
                if processed_any_full_group and _ingest_pass_exhausted(
                    max_pass_seconds=max_pass_seconds,
                    pass_started=pass_started_monotonic,
                    checkpoint="full_parse_progress_group",
                ):
                    full_ingest_time_budget_exceeded = True
                    break
                if held_pending and source_paths is held_pending:
                    # The held group is starting; its own result accounts for it.
                    held_pending = []
                t0 = time.perf_counter()
                try:
                    await self._record_attempt_progress_admitted(
                        attempt_id,
                        phase="full_parse",
                        succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                    )
                    current_path = source_paths[0] if source_paths else None
                    full_result = await self._ingest_full_paths(
                        source_paths,
                        source_name=source_name,
                        attempt_id=attempt_id,
                        max_pass_seconds=max_pass_seconds,
                        pass_started=pass_started_monotonic,
                        heartbeat=self._full_ingest_heartbeat(
                            attempt_id,
                            source_name=source_name,
                            current_path=current_path,
                            succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                            failed_file_count=len(failed_paths),
                            source_payload_read_bytes=source_payload_read_bytes,
                            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                            parse_time_s=parse_time_s,
                            convergence_time_s=convergence_time_s,
                        ),
                    )
                    processed_any_full_group = True
                    if full_result.ordering_held:
                        # Later revisions of a session this group published
                        # go next, warmed against that publication.
                        held_pending = list(full_result.ordering_held)
                        progress_groups.insert(0, held_pending)
                    if full_result.time_budget_exceeded:
                        full_ingest_time_budget_exceeded = True
                    ingest_worker_count_max = max(ingest_worker_count_max, full_result.worker_count)
                    full_ingest_aggregate.add(full_result)
                    for session_id in full_result.changed_session_ids:
                        new_session_touches.append((source_name, session_id))
                except SchemaVersionMismatchError as exc:
                    handle_schema_version_mismatch(source_name, exc)
                    attempt_disposition = classify_archive_write_exception(exc)
                    # Account for every queued path in this source group, not
                    # only the current progress chunk — later chunks would
                    # hit the same structural error with no information gain.
                    for path in grouped_paths:
                        failed_paths.append(str(path))
                    await self._record_attempt_progress_admitted(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    # Stop processing this batch entirely — every remaining
                    # source group would hit the same structural error.
                    break
                except DatabaseError as exc:
                    handle_structural_database_error(source_name, exc)
                    attempt_disposition = classify_archive_write_exception(exc)
                    for path in grouped_paths:
                        failed_paths.append(str(path))
                    await self._record_attempt_progress_admitted(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    break
                except Exception as exc:
                    raise_if_operation_cancelled(exc)
                    if isinstance(exc, UnleasedWriteError):
                        # A missing writer is a configuration refusal, not a
                        # property of these files: never count them failed.
                        raise
                    if isinstance(exc, DaemonBackpressureError):
                        # Bounded compute admission is pass-level infrastructure
                        # pressure; poisoning each source cursor would quarantine
                        # valid input before the next pass can retry it.
                        raise
                    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
                        # Archive contention is infrastructure state, not a
                        # poison payload. Let LiveWatcher requeue the source
                        # group without advancing or excluding its cursors.
                        raise
                    # A full disk, I/O error or corrupt page fails every file
                    # in the group the same way; marking them failed would
                    # back good inputs off into quarantine.
                    raise_if_storage_fault(exc)
                    logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                    attempt_disposition = classify_archive_write_exception(exc)
                    for path in source_paths:
                        failed_paths.append(str(path))
                        cursor_fingerprint_read_bytes += await self._run_ops_write(
                            "cursor_failed", self._record_failed_cursor, path
                        )
                    await self._record_attempt_progress_admitted(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                        failed_file_count=len(failed_paths),
                        source_payload_read_bytes=source_payload_read_bytes,
                        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                        parse_time_s=parse_time_s,
                        current_source=source_name,
                        current_path=source_paths[0] if source_paths else None,
                        error=str(exc),
                    )
                    continue
                parse_elapsed = time.perf_counter() - t0
                parse_time_s += parse_elapsed
                source_payload_read_bytes += full_result.source_payload_read_bytes
                _accumulate_stage_timings(stage_timings, full_result.stage_timings_s)
                release_process_memory()
                await self._record_attempt_progress_admitted(
                    attempt_id,
                    phase="convergence",
                    succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    convergence_time_s=convergence_time_s,
                    current_source=source_name,
                    current_path=source_paths[0] if source_paths else None,
                )
                convergence_debt: list[ConvergenceDebt] = []
                convergence_settlements: list[ConvergenceDebtSettlement] = []
                # Clearing convergence debt is only honest when a convergence
                # pass actually ran: a re-observed file with no session changes
                # executes zero stages, and recording an empty outcome would
                # delete every stage's debt for the path (polylogue-zbzxs).
                convergence_ran = (
                    bool(full_result.changed_session_count or full_result.raw_deferred) and not defer_convergence
                )
                if convergence_ran:
                    (
                        _converged_paths,
                        elapsed,
                        timings,
                        convergence_debt,
                        convergence_settlements,
                    ) = await self._run_convergence_paths(
                        "watcher.live_ingest.full_convergence",
                        full_result.succeeded,
                        whole_archive=whole_archive_convergence,
                        session_ids=full_result.changed_session_ids,
                    )
                    convergence_time_s += elapsed
                    release_process_memory()
                    _accumulate_stage_timings(stage_timings, timings)
                elif full_result.succeeded:
                    logger.info(
                        "live.watcher: skipping full convergence for %d source observation(s) without session changes",
                        len(full_result.succeeded),
                    )
                debt_by_source_path = debt_by_path(convergence_debt)
                # Pin source evidence inside the cursor writer admission, not
                # across this await: another admitted writer may excise it
                # after convergence outcomes commit.
                if full_result.succeeded:
                    outcome_items = (
                        [(path, debt_by_source_path.get(path, ())) for path in full_result.succeeded]
                        if convergence_ran and not _source_tier_acquisition_required()
                        else []
                    )
                    if outcome_items:
                        # Keep each source's cursor behind its failed/deferred
                        # convergence obligation until that obligation commits.
                        await self._run_ops_write(
                            "convergence_outcomes",
                            self._record_convergence_outcomes,
                            outcome_items,
                            convergence_settlements,
                        )
                    # One admission and one ops commit for the group's
                    # cursors: per-file admissions cost a writer round trip
                    # and an ops commit each, most of the group's ops time.
                    cursor_writes = await self._run_ops_write(
                        "cursor_full",
                        self._record_full_cursors,
                        [
                            (
                                path,
                                {
                                    "raw_fingerprint": full_result.raw_fingerprints.get(path),
                                    "raw_byte_size": full_result.raw_byte_sizes.get(path),
                                    "frontier_byte_size": full_result.raw_frontier_sizes.get(path),
                                    "source_name": full_result.raw_source_names.get(path),
                                    "source_revision": full_result.raw_source_revisions.get(path),
                                    "source_fingerprint": full_result.raw_source_fingerprints.get(path),
                                    "captured_content_hash": full_result.captured_content_hashes.get(path),
                                    "canonical_source_path": full_result.captured_canonical_source_paths.get(path),
                                    "captured_profile_key": full_result.captured_profile_keys.get(path),
                                    "captured_file_observation": full_result.captured_file_observations.get(path),
                                    "captured_observed_at_ns": full_result.captured_observation_times_ns.get(path),
                                },
                            )
                            for path in full_result.succeeded
                        ],
                    )
                    for path, (read_bytes, stale) in zip(full_result.succeeded, cursor_writes, strict=True):
                        succeeded_paths.add(path)
                        cursor_fingerprint_read_bytes += read_bytes
                        if stale:
                            stale_cursor_write_count += 1
                            stale_cursor_paths.add(str(path))
                for path in full_result.failed:
                    if path in full_result.excised_paths:
                        excluded_by_path[path] = "durably_excised"
                    else:
                        failed_paths.append(str(path))
                        cursor_fingerprint_read_bytes += await self._run_ops_write(
                            "cursor_failed",
                            self._record_failed_cursor,
                            path,
                            attempted_observation=full_result.captured_file_observations.get(path),
                        )
                for path in full_result.preparation_deferred:
                    deferred_paths.append(path)
                    preparation_deferred_paths.add(path)
                    await self._run_ops_write(
                        "cursor_deferred_preparation",
                        self._defer_full_cursor_retry,
                        path,
                        source_name=source_name,
                        captured_file_observation=full_result.captured_file_observations.get(path),
                    )
                for path in full_result.source_read_deferred:
                    deferred_paths.append(path)
                    source_read_deferred_paths.add(path)
                    await self._run_ops_write(
                        "cursor_deferred_source_read",
                        self._defer_source_read_cursor_retry,
                        path,
                        source_name=source_name,
                        captured_file_observation=full_result.captured_file_observations.get(path),
                    )
                excluded_by_path.update(full_result.excluded)
                detection_fallbacks_by_path.update(full_result.detection_fallbacks)
                settled_exclusions.update(full_result.settled_exclusions)
                partial_admissions.update(full_result.partial_admissions)
                # Acquisition or a definitive current refusal discharges the
                # read obligation; neither proves unrelated derivation debt.
                read_settled = set(full_result.succeeded) | {
                    path
                    for path, reason in full_result.excluded.items()
                    if reason
                    not in {REFUSED_UNATTEMPTED, REFUSED_UNATTEMPTED_TIME_BUDGET, "dropped without a recorded outcome"}
                }
                if read_settled:
                    await self._run_ops_write(
                        "source_read_settlement",
                        self._cursor.apply_convergence_debt_batch,
                        (
                            ConvergenceDebtBatchEntry(
                                clears=tuple(
                                    ConvergenceDebtSettlement("source_path", str(path), "live_ingest_source_read")
                                    for path in read_settled
                                )
                            ),
                        ),
                    )
                emit(
                    "live.ingest.source_group",
                    source_name=source_name,
                    files=len(full_result.succeeded),
                    duration_ms=parse_elapsed * 1000,
                )
            # A held revision the loop ended before reaching was never
            # attempted: it stays retryable instead of reading as settled.
            failed_now = set(failed_paths)
            for path in held_pending:
                if str(path) in failed_now:
                    continue
                deferred_paths.append(path)
                preparation_deferred_paths.add(path)
                await self._run_ops_write(
                    "cursor_deferred_preparation",
                    self._defer_full_cursor_retry,
                    path,
                    source_name=source_name,
                )

        summary_stage_payload = _single_route_stage_payload(
            append_file_count=append_file_count,
            full_file_count=len(full_paths),
        )
        if deferred_paths:
            # polylogue-3r36h: a deferral is bounded backpressure ("no new
            # authority-relevant append this pass"), not a failure. Folding it
            # into ``failed_file_count`` is what daemon status and catch-up
            # status then report to the operator as failed files. Report the
            # count in its own unit instead; the retry projection and the
            # ``live_ingest_deferred`` convergence-debt rows are unchanged.
            summary_stage_payload = {
                **(summary_stage_payload or {}),
                "deferred_file_count": len(deferred_paths),
            }
        summary_stage_payload = {
            **(summary_stage_payload or {}),
            "excluded_file_count": len(excluded_by_path) + len(settled_exclusions),
        }
        if partial_admissions:
            partial_reasons: dict[str, int] = {}
            for partial in partial_admissions.values():
                partial_reasons[partial.reason] = partial_reasons.get(partial.reason, 0) + 1
            summary_stage_payload = {
                **(summary_stage_payload or {}),
                "partial_file_count": len(partial_admissions),
                "partial_reasons": partial_reasons,
                "partial_left_out_bytes": sum(
                    partial.source_bytes - partial.complete_prefix_bytes for partial in partial_admissions.values()
                ),
            }
        # The ingest-attempt receipt has separate units for parsed raw files
        # and materialized sessions.  Count the actual session identities
        # touched by this batch; using ``succeeded_file_count`` here would
        # silently turn a multi-session file into one materialized session and
        # would also credit failed cursor items.
        materialized_session_count = len(
            {session_id for _source_name, session_id in (*new_session_touches, *updated_session_touches)}
        )
        await self._record_attempt_progress_admitted(
            attempt_id,
            phase="cursor_update",
            succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
            failed_file_count=len(failed_paths),
            materialized_count=materialized_session_count,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
            stale_cursor_write_count=stale_cursor_write_count,
            stage_payload=summary_stage_payload,
        )

        if (
            not is_fully_degraded()
            and not _source_tier_acquisition_required()
            and (succeeded_paths or self._raw_retention_backlog_paths(exclude=set()))
        ):
            compaction_started = time.perf_counter()
            await self._run_source_writer(
                "watcher.live_ingest.raw_compaction",
                self._compact_superseded_raw_snapshots,
                sorted(succeeded_paths),
            )
            raw_compaction_runs = 1
            stage_timings["raw_compaction"] = time.perf_counter() - compaction_started

        deferred_debt_writes = tuple(
            ConvergenceDebtWrite(
                stage=(
                    "live_ingest_source_read" if deferred_path in source_read_deferred_paths else "live_ingest_deferred"
                ),
                subject_type="source_path",
                subject_id=str(deferred_path),
                error=(
                    "ingest deferred: source read unavailable"
                    if deferred_path in source_read_deferred_paths
                    else "ingest deferred: JSONL worker preparation pending"
                    if deferred_path in preparation_deferred_paths
                    else "ingest deferred: no new authority-relevant append this pass"
                ),
                deferred=True,
            )
            for deferred_path in deferred_paths
        )
        if deferred_debt_writes:
            # The attempt receipt folds these into ``failed_file_count`` and
            # ``LiveBatchMetrics`` does not count them at all, so a deferral
            # was readable neither as a failure nor as a success
            # (polylogue-3r36h). Record it as what it is: deliberate
            # bounded-backpressure debt, which lands as
            # ``convergence_debt.status = 'deferred'``.
            await self._run_ops_write(
                "convergence_debt_batch",
                self._cursor.apply_convergence_debt_batch,
                (ConvergenceDebtBatchEntry(writes=deferred_debt_writes),),
            )
        retry_paths = failed_paths + [str(path) for path in deferred_paths]
        # ``succeeded_paths`` stays the cursor-completed set this method
        # advances. Public accounting moves a path that produced no session
        # to the exclusions, matching the intake outcome for the same path.
        admitted_paths = succeeded_paths - settled_exclusions.keys()
        reported_excluded = {
            **excluded_by_path,
            **settled_exclusions,
        }
        excluded_reasons: dict[str, int] = {}
        for reason in reported_excluded.values():
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
        ingested_bytes, failed_bytes, refused_bytes_by_reason = split_offered_bytes(
            path_sizes,
            succeeded=admitted_paths,
            partial_admissions=partial_admissions,
            failed=(Path(path) for path in failed_paths),
            excluded=reported_excluded,
            deferred=deferred_paths,
            unattempted_reason=(
                REFUSED_UNATTEMPTED_TIME_BUDGET if full_ingest_time_budget_exceeded else REFUSED_UNATTEMPTED
            ),
        )
        db_bytes_after = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        metrics = LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
            failed_file_count=len(failed_paths),
            excluded_file_count=sum(excluded_reasons.values()),
            excluded_reasons=dict(excluded_reasons),
            excluded_paths={str(path): reason for path, reason in reported_excluded.items()},
            detection_fallback_paths={str(path): reason for path, reason in detection_fallbacks_by_path.items()},
            deferred_paths=tuple(str(path) for path in deferred_paths),
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=input_bytes,
            ingested_bytes=ingested_bytes,
            failed_bytes=failed_bytes,
            refused_bytes_by_reason=refused_bytes_by_reason,
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            ingest_worker_count_max=ingest_worker_count_max,
            append_file_count=append_file_count,
            full_file_count=len(full_paths),
            archive_bytes_before=db_bytes_before,
            archive_bytes_after=db_bytes_after,
            archive_write_bytes_delta=max(0, db_bytes_after - db_bytes_before),
            parse_time_s=round(parse_time_s, 6),
            convergence_time_s=round(convergence_time_s, 6),
            total_time_s=round(time.perf_counter() - batch_started, 6),
            **full_ingest_aggregate.to_metric_kwargs(),
            rss_current_mb=read_current_rss_mb(),
            rss_peak_self_mb=read_peak_rss_self_mb(),
            rss_peak_children_mb=read_peak_rss_children_mb(),
            cgroup_path=read_cgroup_path(),
            cgroup_memory_current_mb=read_cgroup_memory_current_mb(),
            cgroup_memory_peak_mb=read_cgroup_memory_peak_mb(),
            cgroup_memory_swap_current_mb=read_cgroup_memory_swap_current_mb(),
            stale_cursor_write_count=stale_cursor_write_count,
            stale_cursor_paths=tuple(sorted(stale_cursor_paths)),
            raw_compaction_runs=raw_compaction_runs,
            stage_timings_s={name: round(elapsed, 6) for name, elapsed in stage_timings.items()},
            failed_paths=retry_paths,
            succeeded_paths=tuple(sorted(admitted_paths)),
            settled_exclusion_paths={str(path): reason for path, reason in sorted(settled_exclusions.items())},
            partial_admission_paths={
                str(path): partial for path, partial in sorted(partial_admissions.items()) if path in admitted_paths
            },
            new_sessions=tuple(new_session_touches),
            updated_sessions=tuple(updated_session_touches),
            time_budget_exceeded=full_ingest_time_budget_exceeded,
        )
        if emit_event and self._event_emitter is not None:
            # The daemon's emitter appends to the ops-tier event ledger, so it
            # is an ops publication like every other one in this method and
            # takes the writer through ``_run_sync``. Called inline it ran on
            # the event loop with no lease held, and an armed single-writer
            # boundary raised ``UnleasedWriteError`` at the very end of a batch
            # that had already done its work -- the page was then reported
            # refused and its files were never re-offered.
            await self._run_sync(
                "watcher.live_ingest.ops.batch_event",
                self._event_emitter,
                "ingestion_batch",
                metrics.to_payload(),
            )
        await self._record_attempt_progress_admitted(
            attempt_id,
            phase="completed",
            status="completed",
            queued_file_count=metrics.queued_file_count,
            needed_file_count=metrics.needed_file_count,
            skipped_file_count=metrics.skipped_file_count,
            succeeded_file_count=len(succeeded_paths - settled_exclusions.keys()),
            failed_file_count=len(failed_paths),
            materialized_count=materialized_session_count,
            input_bytes=input_bytes,
            ingested_bytes=metrics.ingested_bytes,
            failed_bytes=metrics.failed_bytes,
            refused_bytes=metrics.refused_bytes,
            refused_bytes_by_reason=dict(metrics.refused_bytes_by_reason),
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            archive_write_bytes_delta=metrics.archive_write_bytes_delta,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
            total_time_s=metrics.total_time_s,
            stage_timings_s=metrics.stage_timings_s,
            stale_cursor_write_count=stale_cursor_write_count,
            stage_payload=summary_stage_payload,
        )
        if attempt_disposition is not None:
            final_disposition = attempt_disposition
        elif (
            not retry_paths
            and not full_ingest_time_budget_exceeded
            and settled_exclusions
            and set(reported_excluded) == set(paths)
            and set(reported_excluded.values()) <= SETTLED_EXCLUSION_REASONS
        ):
            # Every source settled to nothing admissible: the durable attempt
            # agrees with the intake's EXCLUDED outcome instead of SUCCESS.
            if REFUSED_CORRUPT_INPUT in reported_excluded.values():
                final_disposition = corrupt_input_disposition(
                    evidence_ref="batch:corrupt_input_sources",
                    diagnostic=(
                        f"{sum(reason == REFUSED_CORRUPT_INPUT for reason in reported_excluded.values())} "
                        "source item(s) are corrupt input"
                    ),
                )
            else:
                final_disposition = non_session_artifact_disposition(
                    evidence_ref="batch:no_session_sources",
                    diagnostic=f"{len(settled_exclusions)} source item(s) parsed to no session",
                )
        elif not retry_paths:
            final_disposition = success_disposition(
                evidence_ref="batch:partial_admission" if metrics.partial_admission_paths else None
            )
        else:
            # Per-record failures are settled by the retained owner as typed
            # refusals or retryable preparation failures, but aggregating them
            # up to this attempt-level row is deferred follow-up work
            # (polylogue-cnu3 PR body). Reporting SUCCESS here
            # would be dishonest given ``retry_paths`` is non-empty, so this
            # falls back to the explicit "not yet classified" bucket rather
            # than guessing.
            final_disposition = downstream_failure_disposition(
                evidence_ref="batch:per_item_failure_aggregate",
                diagnostic=f"{len(retry_paths)} source item(s) failed without a batch-level exception",
            )
        if open_attempt is not None:
            # A cancellation from here on may race a close the coordinator
            # finishes detached; the cancellation event says so.
            open_attempt.finishing = True
        await self._run_ops_write(
            "attempt_finish",
            self._cursor.finish_ingest_attempt,
            attempt_id,
            status="completed" if not retry_paths else "completed_with_failures",
            phase="completed",
            error="; ".join(retry_paths[:3]) if retry_paths else None,
            disposition=final_disposition,
        )
        if open_attempt is not None:
            open_attempt.finished = True
        timing_items = sorted(metrics.stage_timings_s.items(), key=lambda item: (-item[1], item[0]))
        timing_map: dict[str, float] = {}
        for name, seconds in timing_items[:12]:
            label = re.sub(r"[^a-z0-9_.]", "_", name.lower())[:48]
            if not label or not label[0].isalpha():
                label = "phase_" + label[:42]
            timing_map[label] = seconds * 1000
        emit(
            "live.ingest.chunk",
            outcome=(
                "degraded"
                if metrics.failed_file_count
                or metrics.excluded_file_count
                or metrics.deferred_paths
                or metrics.partial_admission_paths
                else "ok"
            ),
            files=metrics.needed_file_count,
            bytes=metrics.input_bytes,
            duration_ms=metrics.total_time_s * 1000,
            succeeded=metrics.succeeded_file_count,
            failed=metrics.failed_file_count,
            refused=metrics.excluded_file_count,
            deferred=len(metrics.deferred_paths),
            partial_file_count=len(metrics.partial_admission_paths),
            partial_left_out_bytes=sum(
                partial.source_bytes - partial.complete_prefix_bytes
                for partial in metrics.partial_admission_paths.values()
            ),
            stage_timings_ms=timing_map,
            stage_timings_omitted=max(0, len(timing_items) - len(timing_map)),
        )
        return metrics

    def _degraded_skip_metrics(
        self,
        paths: list[Path],
        queued_file_count: int | None,
        skipped_file_count: int,
    ) -> LiveBatchMetrics:
        """Empty-ingest metrics for the degraded short-circuit path.

        The offered bytes are reported and refused in full: a batch that was
        handed files and admitted none of them is not an idle one, and
        reporting zero offered bytes hides the refusal from every receipt.
        """
        offered_bytes = sum(_path_size(path) for path in paths)
        return LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count + len(paths),
            succeeded_file_count=0,
            failed_file_count=0,
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=offered_bytes,
            refused_bytes_by_reason=({REFUSED_DAEMON_DEGRADED: offered_bytes} if offered_bytes else {}),
            source_payload_read_bytes=0,
            cursor_fingerprint_read_bytes=0,
            ingest_worker_count_max=0,
            append_file_count=0,
            full_file_count=0,
            archive_bytes_before=0,
            archive_bytes_after=0,
            archive_write_bytes_delta=0,
            parse_time_s=0.0,
            convergence_time_s=0.0,
            total_time_s=0.0,
            rss_current_mb=read_current_rss_mb(),
            rss_peak_self_mb=read_peak_rss_self_mb(),
            rss_peak_children_mb=read_peak_rss_children_mb(),
            cgroup_path=read_cgroup_path(),
            cgroup_memory_current_mb=read_cgroup_memory_current_mb(),
            cgroup_memory_peak_mb=read_cgroup_memory_peak_mb(),
            cgroup_memory_swap_current_mb=read_cgroup_memory_swap_current_mb(),
            stale_cursor_write_count=0,
            stage_timings_s={},
            failed_paths=[],
            daemon_degraded_skip=True,
        )

    def _record_attempt_progress(self, attempt_id: str, **kwargs: Any) -> None:
        record_attempt_progress(self._cursor, attempt_id, **kwargs)

    async def _run_ops_write(
        self,
        operation: str,
        function: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Admit one ops-tier publication without holding a whole intake page.

        One publication is one unit of ops-tier bookkeeping, but it is rarely
        one statement: a full-cursor commit reads the record, upserts it and
        resets its failure counters, and a convergence outcome clears stale
        debt for the source path and every session it touched before
        recording the new rows. Each of those used to open, commit and close
        its own ``ops.db`` connection -- ~19 connection close/open pairs and
        ~10 commits per ingested file, measured as 60% of a warm chunk's wall
        clock, almost all of it in ``sqlite3.Connection.close`` (which
        checkpoints the WAL) rather than in the statements themselves.

        The publication therefore runs inside ONE ``ops_write_scope``. The
        scope shares a connection; it does not merge transactions. Every
        ``_connect_ops`` block inside still commits on success and rolls back
        on failure in the same order, so the durable state after a crash at
        any point is exactly what it was before -- only the connection churn
        between those commits is gone.

        The scope is entered inside ``_run_sync``'s worker function because it
        is thread-local: entering it on the event loop thread would not reach
        the thread that actually performs the write.
        """

        def publish(*scoped_args: P.args, **scoped_kwargs: P.kwargs) -> T:
            with self._cursor.ops_write_scope():
                return function(*scoped_args, **scoped_kwargs)

        return await self._run_sync(f"watcher.live_ingest.ops.{operation}", publish, *args, **kwargs)

    async def _record_attempt_progress_admitted(self, attempt_id: str, **kwargs: Any) -> None:
        await self._run_ops_write("attempt_progress", self._record_attempt_progress, attempt_id, **kwargs)

    def _full_ingest_heartbeat(
        self,
        attempt_id: str,
        *,
        source_name: str,
        current_path: Path | None,
        succeeded_file_count: int,
        failed_file_count: int,
        source_payload_read_bytes: int,
        cursor_fingerprint_read_bytes: int,
        parse_time_s: float,
        convergence_time_s: float,
    ) -> _FullIngestHeartbeat:
        def emit(
            phase: str,
            *,
            current_path_override: Path | None = None,
            payload_read_bytes: int | None = None,
            stage_payload: dict[str, object] | None = None,
        ) -> None:
            self._record_attempt_progress(
                attempt_id,
                phase=phase,
                succeeded_file_count=succeeded_file_count,
                failed_file_count=failed_file_count,
                source_payload_read_bytes=(
                    source_payload_read_bytes if payload_read_bytes is None else payload_read_bytes
                ),
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                convergence_time_s=convergence_time_s,
                current_source=source_name,
                current_path=current_path if current_path_override is None else current_path_override,
                stage_payload=stage_payload,
            )

        return _throttled_phase_heartbeat(emit)

    def _record_failed_cursor(
        self, path: Path, *, attempted_observation: tuple[int, int, int, int, int] | None = None
    ) -> int:
        # polylogue-awy5: an already-excluded cursor is a poison pill the
        # daemon has already given up on (5-failure cap,
        # ``_MAX_CURSOR_FAILURES_BEFORE_EXCLUDE``). Re-running the same
        # crash-looping batch against it and calling ``mark_failed`` again
        # every pass has no effect on the cursor's *state* (the lifecycle
        # table only permits EXCLUDED -> EXCLUDED here) but keeps
        # incrementing ``failure_count`` forever with no upper bound --
        # measured live at 689/864/975/1001/2018 on the five ZIPs that hit
        # this path, i.e. thousands of full acquire+parse+crash cycles
        # burned re-discovering a fact the cursor already recorded. Consult
        # ``excluded`` before touching the cursor at all so a poisoned
        # source stops being re-queued into wasted work.
        try:
            preexisting = self._cursor.get_record(path)
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc):
                raise
            preexisting = None
        if preexisting is not None and preexisting.excluded:
            # The watcher revives an excluded cursor only when the file's
            # observation differs from the one the exclusion is bound to.
            # Rebind it to the observation this attempt actually read, so a
            # changing file costs one attempt per change rather than one per
            # poll (polylogue-d8fpj). Without that observation nothing is
            # rebound: a fresh stat could name a later, unattempted revision
            # and quarantine it unread.
            if attempted_observation is not None:
                try:
                    self._cursor.mark_excluded(
                        path,
                        observation=attempted_observation,
                        parser_fingerprint=self._current_parser_fingerprint(),
                    )
                except sqlite3.OperationalError as exc:
                    if not is_transient_sqlite_lock(exc):
                        raise
            return 0
        try:
            stat = path.stat()
            authority = CursorPathAuthority.observe(path)
        except FileNotFoundError:
            try:
                self._cursor.mark_failed(path, authority=None)
            except sqlite3.OperationalError as exc:
                if not is_transient_sqlite_lock(exc):
                    raise
                logger.warning("live.watcher: skipped failed-cursor mark for missing file %s: %s", path, exc)
            return 0
        try:
            existing = self._cursor.get_record(path)
            # A failed write cannot adopt any observation from the unaccepted
            # file state. In particular, pairing the accepted offset with the
            # newer byte size makes the watcher's stable-size fast path hide a
            # retry after failure metadata is cleared.
            if existing is None:
                try:
                    tail_hash, _tail_bytes = tail_hash_from_path(path, stat.st_size)
                except FileNotFoundError:
                    self._cursor.mark_failed(path, authority=None)
                    return 0
                self._cursor.set(
                    path,
                    stat.st_size,
                    authority=authority,
                    byte_offset=0,
                    last_complete_newline=0,
                    parser_fingerprint=self._current_parser_fingerprint(),
                    content_fingerprint=None,
                    tail_hash=tail_hash,
                    source_name=self._source_name_for(path),
                    st_dev=stat.st_dev,
                    st_ino=stat.st_ino,
                    mtime_ns=stat.st_mtime_ns,
                )
            self._cursor.mark_failed(path, authority=authority, failed_stat=stat)
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc):
                raise
            logger.warning("live.watcher: skipped failed-cursor bookkeeping for %s: %s", path, exc)
        return stat.st_size

    def _record_full_cursors(self, items: Sequence[tuple[Path, dict[str, Any]]]) -> list[tuple[int, bool]]:
        """Record several full-ingest cursors in one ops transaction.

        Returns each path's fingerprint read bytes and whether its write was
        stale, in order.
        """
        results: list[tuple[int, bool]] = []
        # One bounded source read per page, after acquiring the writer. Its
        # lifetime cannot cross an admission where retained rows may change.
        with (
            self._pinned_source_tier_evidence([path for path, _kwargs in items]),
            self._cursor.ops_batch(),
        ):
            for path, kwargs in items:
                read_bytes = self._record_full_cursor(path, **kwargs)
                results.append((read_bytes, self._last_cursor_write_stale))
        return results

    def _record_full_cursor(
        self,
        path: Path,
        *,
        canonical_source_path: str | None = None,
        captured_profile_key: str | None = None,
        raw_fingerprint: str | None = None,
        raw_byte_size: int | None = None,
        frontier_byte_size: int | None = None,
        source_name: str | None = None,
        source_revision: str | None = None,
        source_fingerprint: str | None = None,
        captured_content_hash: str | None = None,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
        captured_observed_at_ns: int | None = None,
    ) -> int:
        self._last_cursor_write_stale = False
        resolved_source_name = source_name or self._source_name_for(path)
        captured_authority = (
            CursorPathAuthority(canonical_source_path, captured_profile_key)
            if canonical_source_path is not None
            else None
        )
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._last_cursor_write_stale = True
            self._invalidate_cursor_for_full_retry(
                path,
                source_name=resolved_source_name,
                captured_file_observation=captured_file_observation,
                authority=captured_authority,
            )
            return 0
        raw_fingerprint = raw_fingerprint or self._latest_raw_fingerprint(path)
        if self._archive_source_db_path().exists() and not self._source_tier_evidence_retained(
            path, raw_fingerprint=raw_fingerprint
        ):
            # A cursor commit is a claim that these bytes were consumed and
            # their evidence retained. With the source tier present and no
            # raw row for this path, nothing was retained, so advancing would
            # make the bytes unreachable: the frontier gate then reads the
            # path as a cursor absent from the source tier, and no route can
            # ever re-read it. Leave the cursor where it is with a typed,
            # retryable reason instead.
            self._last_cursor_write_stale = True
            logger.warning(
                "live.watcher: refusing to advance cursor past bytes that left no source evidence: %s",
                path,
            )
            return 0
        # SQLite-backed sources are identified by an acquisition revision,
        # not by the snapshot file's byte length. Record the live database
        # observation so a stable source does not look perpetually grown when
        # the consistent snapshot used a different page count.
        byte_size = stat.st_size if source_revision is not None or raw_byte_size is None else raw_byte_size
        prefix_proof = self._full_capture_still_matches(
            path,
            stat=stat,
            byte_size=byte_size,
            captured_content_hash=captured_content_hash,
            captured_file_observation=captured_file_observation,
            captured_observed_at_ns=captured_observed_at_ns,
        )
        bytes_read = prefix_proof.bytes_read
        if prefix_proof.outcome == "deferred":
            self._last_cursor_write_stale = True
            logger.info(
                "live.watcher: captured prefix remained busy; preserving raw for cursor reconciliation: %s",
                path,
            )
            self._defer_full_cursor_retry(
                path, source_name=resolved_source_name, stat=stat, authority=captured_authority
            )
            return bytes_read
        if prefix_proof.outcome != "verified":
            self._last_cursor_write_stale = True
            logger.warning(
                "live.watcher: source changed after full capture; cursor invalidated for full retry: %s",
                path,
            )
            self._invalidate_cursor_for_full_retry(
                path, source_name=resolved_source_name, stat=stat, authority=captured_authority
            )
            return bytes_read
        assert prefix_proof.stat is not None
        stat = prefix_proof.stat
        if source_revision is not None:
            # Ordinary append cursors use the blob-backed source revision as
            # their byte-proof identity. A retained raw failure instead binds
            # the cursor to its durable source-tier ID so the next growth can
            # find typed failure evidence and force full replay.
            fp = (
                raw_fingerprint
                if raw_fingerprint is not None and self._raw_failure_requires_full_replay(path, raw_fingerprint)
                else source_revision
            )
            last_nl = frontier_byte_size if frontier_byte_size is not None else byte_size
            if captured_content_hash is not None:
                bounded_tail_hash, tail_bytes = tail_hash_from_path(path, byte_size)
                tail_hash = encode_cursor_hash_authority(
                    captured_content_hash,
                    bounded_tail_hash,
                    ctime_ns=stat.st_ctime_ns,
                )
                bytes_read += tail_bytes
            else:
                # Only a capture without a content hash keys its cursor on the
                # bound file revision; binding opens a fresh reader process.
                tail_hash = source_fingerprint or sqlite_source_revision(path)
        else:
            fp, last_nl, tail_hash, cursor_state_bytes = cursor_state_after_full_ingest(
                path,
                byte_size,
                raw_fingerprint=raw_fingerprint,
            )
            if frontier_byte_size is not None:
                last_nl = frontier_byte_size
            # The blob acquisition digest is already the exact hash of this
            # complete prefix.  Reuse it only at EOF, after the surrounding
            # full-capture proofs have verified that the mutable path still
            # binds to that acquisition.  A deferred/incomplete JSONL tail
            # has a shorter prefix and must be rehashed independently.
            if captured_content_hash is not None and last_nl == byte_size:
                prefix_hash = captured_content_hash.lower()
                prefix_bytes = 0
            else:
                prefix_hash, prefix_bytes = sha256_range_from_path(
                    path,
                    start_offset=0,
                    end_offset=last_nl,
                )
            tail_hash = encode_cursor_hash_authority(prefix_hash, tail_hash, ctime_ns=stat.st_ctime_ns)
            bytes_read += cursor_state_bytes + prefix_bytes
        cursor_provider = Provider.from_string(resolved_source_name)
        if (
            frontier_kind_for_origin(origin_from_provider(cursor_provider)) == "claude-header-body"
            and path.suffix.lower() in {".jsonl", ".ndjson"}
            and not path_declaration_refuses_session(cursor_provider, path)
        ):
            semantic_authority = claude_semantic_frontier_for_prefix(
                path, frontier_byte_size if frontier_byte_size is not None else byte_size
            )
            if semantic_authority is not None:
                last_nl = frontier_byte_size if frontier_byte_size is not None else byte_size
                tail_hash = semantic_authority
                bytes_read += last_nl
            elif raw_fingerprint is None or not self._raw_failure_requires_full_replay(path, raw_fingerprint):
                self._last_cursor_write_stale = True
                self._invalidate_cursor_for_full_retry(
                    path, source_name=resolved_source_name, stat=stat, authority=captured_authority
                )
                return bytes_read
            # Otherwise the retained raw is settled with terminal failure
            # evidence (a malformed complete record has no semantic frontier):
            # the cursor keeps its byte-prefix authority, and that evidence
            # forces a full replay when the file changes. Refusing here left
            # the unchanged capture retried on every pass (polylogue-xf8qp).
        final_prefix_proof = self._full_capture_still_matches(
            path,
            stat=stat,
            byte_size=byte_size,
            captured_content_hash=captured_content_hash,
            captured_file_observation=captured_file_observation,
            captured_observed_at_ns=captured_observed_at_ns,
        )
        bytes_read += final_prefix_proof.bytes_read
        if final_prefix_proof.outcome == "deferred":
            self._last_cursor_write_stale = True
            logger.info(
                "live.watcher: captured prefix remained busy after cursor proof; preserving raw for reconciliation: %s",
                path,
            )
            self._defer_full_cursor_retry(
                path, source_name=resolved_source_name, stat=stat, authority=captured_authority
            )
            return bytes_read
        if final_prefix_proof.outcome != "verified":
            self._last_cursor_write_stale = True
            self._invalidate_cursor_for_full_retry(
                path,
                source_name=resolved_source_name,
                stat=final_prefix_proof.stat,
                captured_file_observation=captured_file_observation,
                authority=captured_authority,
            )
            return bytes_read
        assert final_prefix_proof.stat is not None
        final_stat = final_prefix_proof.stat
        updated = self._cursor.set(
            path,
            byte_size,
            authority=captured_authority or CursorPathAuthority.observe(path),
            byte_offset=last_nl,
            last_complete_newline=last_nl,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=fp,
            tail_hash=tail_hash,
            source_name=resolved_source_name,
            st_dev=final_stat.st_dev,
            st_ino=final_stat.st_ino,
            mtime_ns=final_stat.st_mtime_ns,
            allow_backward=final_stat.st_size <= byte_size,
            deferred_end_offset=(
                byte_size if frontier_byte_size is not None and frontier_byte_size < byte_size else None
            ),
        )
        self._last_cursor_write_stale = not updated
        if not updated:
            logger.warning(
                "live.watcher: full cursor frontier was rejected; cursor invalidated for full retry: %s",
                path,
            )
            self._invalidate_cursor_for_full_retry(
                path,
                source_name=resolved_source_name,
                stat=final_stat,
                captured_file_observation=captured_file_observation,
                authority=captured_authority,
            )
            return bytes_read
        self._cursor.reset_failures(path)
        return bytes_read

    def _full_capture_still_matches(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        byte_size: int,
        captured_content_hash: str | None,
        captured_file_observation: tuple[int, int, int, int, int] | None,
        captured_observed_at_ns: int | None = None,
    ) -> _FullCapturePrefixProof:
        if captured_file_observation is None:
            try:
                final_stat = path.stat()
            except OSError:
                return _FullCapturePrefixProof("rejected", None, 0)
            initial_outcome: Literal["verified", "rejected"] = (
                "verified" if _file_observation(final_stat) == _file_observation(stat) else "rejected"
            )
            return _FullCapturePrefixProof(initial_outcome, final_stat, 0)
        captured_dev, captured_ino, _captured_size, _captured_mtime_ns, _captured_ctime_ns = captured_file_observation
        if (stat.st_dev, stat.st_ino) != (captured_dev, captured_ino) or stat.st_size < byte_size:
            return _FullCapturePrefixProof("rejected", stat, 0)
        if captured_content_hash is None:
            try:
                final_stat = path.stat()
            except OSError:
                return _FullCapturePrefixProof("rejected", None, 0)
            legacy_outcome: Literal["verified", "rejected"] = (
                "verified"
                if _file_observation(stat) == captured_file_observation
                and _file_observation(final_stat) == _file_observation(stat)
                else "rejected"
            )
            return _FullCapturePrefixProof(legacy_outcome, final_stat, 0)
        normalized_fingerprint = captured_content_hash.lower()
        if len(normalized_fingerprint) != 64 or any(char not in "0123456789abcdef" for char in normalized_fingerprint):
            return _FullCapturePrefixProof("rejected", stat, 0)
        if _settled_observation_unchanged(
            stat,
            captured_file_observation=captured_file_observation,
            captured_observed_at_ns=captured_observed_at_ns,
        ):
            return _FullCapturePrefixProof("verified", stat, 0)

        bytes_read = 0
        latest_stat = stat
        for _attempt in range(_FULL_CAPTURE_PREFIX_PROOF_ATTEMPTS):
            try:
                proof_start = path.stat()
                if (proof_start.st_dev, proof_start.st_ino) != (
                    captured_dev,
                    captured_ino,
                ) or proof_start.st_size < byte_size:
                    return _FullCapturePrefixProof("rejected", proof_start, bytes_read)
                current_fingerprint, proof_bytes = sha256_range_from_path(
                    path,
                    start_offset=0,
                    end_offset=byte_size,
                )
                proof_end = path.stat()
            except (EOFError, OSError):
                return _FullCapturePrefixProof("rejected", None, bytes_read)
            bytes_read += proof_bytes
            latest_stat = proof_end
            if current_fingerprint != normalized_fingerprint:
                return _FullCapturePrefixProof("rejected", proof_end, bytes_read)
            if _file_observation(proof_start) == _file_observation(proof_end):
                return _FullCapturePrefixProof("verified", proof_end, bytes_read)
            if (
                (proof_end.st_dev, proof_end.st_ino) != (captured_dev, captured_ino)
                or proof_end.st_size < byte_size
                or proof_end.st_size <= proof_start.st_size
            ):
                return _FullCapturePrefixProof("rejected", proof_end, bytes_read)
        return _FullCapturePrefixProof("deferred", latest_stat, bytes_read)

    def _defer_source_read_cursor_retry(
        self,
        path: Path,
        *,
        source_name: str,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        """Schedule unread bytes without spending the permanent-failure budget."""
        existing = self._cursor.get_record(path)
        authority = CursorPathAuthority.of_record(existing) if existing is not None else None
        if authority is None:
            try:
                authority = CursorPathAuthority.observe(path)
            except OSError as exc:
                if isinstance(exc, FileNotFoundError) or retryable_read_fault(exc):
                    # An unreadable new file has no accepted coordinate to
                    # bind a cursor to. Its explicit deferred outcome and debt
                    # keep it owed; discovery/adapter retries retain the path.
                    return
                raise
        self._defer_full_cursor_retry(
            path,
            source_name=source_name,
            captured_file_observation=captured_file_observation,
            authority=authority,
        )

    def _defer_full_cursor_retry(
        self,
        path: Path,
        *,
        source_name: str,
        stat: os.stat_result | None = None,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
        authority: CursorPathAuthority | None = None,
    ) -> None:
        """Back off a busy full-prefix handoff without discarding its raw evidence."""

        self._invalidate_cursor_for_full_retry(
            path,
            source_name=source_name,
            stat=stat,
            captured_file_observation=captured_file_observation,
            authority=authority,
        )
        self._cursor.defer_full_cursor_reconciliation(path)

    def _invalidate_cursor_for_full_retry(
        self,
        path: Path,
        *,
        source_name: str,
        stat: os.stat_result | None = None,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
        authority: CursorPathAuthority | None = None,
    ) -> None:
        existing = self._cursor.get_record(path)
        if authority is None:
            # A writer with no capture of its own takes the cursor's recorded
            # authority, and observes the file only when there is none.
            authority = CursorPathAuthority.of_record(existing) if existing is not None else None
        if authority is None:
            try:
                authority = CursorPathAuthority.observe(path)
            except FileNotFoundError:
                # The file vanished before any authority was captured: a
                # typed failed cursor, retried when the path reappears.
                self._cursor.mark_failed(path, authority=None)
                return
        if stat is not None:
            observation = _file_observation(stat)
        elif captured_file_observation is not None:
            observation = captured_file_observation
        elif existing is not None:
            observation = (
                existing.st_dev or 0,
                existing.st_ino or 0,
                existing.byte_size,
                existing.mtime_ns or 0,
                0,
            )
        else:
            observation = (0, 0, 0, 0, 0)
        st_dev, st_ino, byte_size, mtime_ns, _ctime_ns = observation
        updated = self._cursor.set(
            path,
            byte_size,
            authority=authority,
            byte_offset=0,
            last_complete_newline=0,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=None,
            tail_hash=None,
            source_name=source_name,
            st_dev=st_dev or None,
            st_ino=st_ino or None,
            mtime_ns=mtime_ns or None,
            failure_count=existing.failure_count if existing is not None else 0,
            next_retry_at=existing.next_retry_at if existing is not None else None,
            # Every caller reaches here because the source changed under an
            # admitted handoff and must be fully re-ingested. Carrying the old
            # ``excluded`` flag forward alongside the NEW filesystem
            # observation made the watcher's exclusion branch see an unchanged
            # identity and skip the path forever: the current bytes were never
            # acquired and nothing recorded retryable backlog for them.
            excluded=False,
            allow_backward=True,
        )
        if not updated:
            raise sqlite3.OperationalError(f"failed to persist cursor invalidation for {path}")

    def _record_convergence_outcomes(
        self,
        outcomes: Iterable[tuple[Path, Iterable[ConvergenceDebt]]],
        settlements: Iterable[ConvergenceDebtSettlement] = (),
    ) -> None:
        record_convergence_outcomes(self._cursor, outcomes, settlements=settlements)

    def _converge_paths(
        self,
        paths: Iterable[Path],
        *,
        whole_archive: bool = True,
        session_ids: Iterable[str] = (),
    ) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt], list[ConvergenceDebtSettlement]]:
        unique_paths = tuple(sorted(dict.fromkeys(paths)))
        if not unique_paths:
            return set(), 0.0, {}, [], []
        if self._converger is None:
            return set(unique_paths), 0.0, {}, [], []

        started = time.perf_counter()
        try:
            converge_batch = getattr(self._converger, "converge_batch", None)
            if callable(converge_batch):
                # The keyword is passed only when it narrows the pass, so
                # convergers without the parameter keep their whole-archive
                # default.
                states, timings = (
                    converge_batch(unique_paths) if whole_archive else converge_batch(unique_paths, whole_archive=False)
                )
                batch_completed = {
                    path for path in unique_paths if path in states and bool(getattr(states[path], "converged", False))
                }
                debt_items = convergence_debt_from_states(unique_paths, states)
                settlements = [item for state in states.values() for item in settled_convergence_stages(state)]
                batch_stage_timings = {stage_name: float(elapsed) for stage_name, elapsed in timings.items()}
                # #1654: after convergence, check for new hook events that
                # carry paste evidence and update matching messages. Scoped to
                # this batch's sessions so the scan is bounded by the batch,
                # not by the archive's whole hook history.
                t_paste = time.perf_counter()
                paste_session_ids = tuple(dict.fromkeys(session_ids))
                try:
                    from polylogue.sources.live.hook_paste_enrichment import enrich_paste_from_hooks

                    # This pass may run with the writer released (the stage
                    # engine no longer holds it across compute), so its write
                    # takes the writer for itself rather than assuming one.
                    def enrich_and_clear_paste_debt() -> None:
                        enrich_paste_from_hooks(self._cursor._db_path, session_ids=paste_session_ids)
                        for paste_session_id in paste_session_ids:
                            self._cursor.clear_convergence_debt(
                                stage="hook_paste_enrichment",
                                subject_type="session_id",
                                subject_id=str(paste_session_id),
                            )

                    admit_stage_write("convergence.hook_paste_enrichment", enrich_and_clear_paste_debt)
                except DaemonOperationCancelled:
                    raise
                except Exception as exc:
                    # A debug line made this indistinguishable from success:
                    # the stage still recorded its elapsed time and nothing
                    # else said the sessions in it never got their paste
                    # evidence (polylogue-3r36h). Record the same
                    # convergence debt any other unconverged stage records so
                    # the sessions stay re-derivable.
                    emit(
                        "live.ingest.hook_paste_enrichment_failed",
                        level=WARNING,
                        outcome="error",
                        reason="hook_paste_enrichment_raised",
                        count=len(paste_session_ids),
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
                    # ``exc`` is unbound at the end of this except clause, so
                    # the debt writer closes over a plain local instead.
                    paste_error = str(exc)

                    def record_paste_debt() -> None:
                        for paste_session_id in paste_session_ids:
                            self._cursor.record_convergence_debt(
                                stage="hook_paste_enrichment",
                                subject_type="session_id",
                                subject_id=str(paste_session_id),
                                error=paste_error,
                            )

                    admit_stage_write("convergence.hook_paste_enrichment.debt", record_paste_debt)
                batch_stage_timings["hook_paste_enrichment"] = time.perf_counter() - t_paste
                return (
                    batch_completed,
                    time.perf_counter() - started,
                    batch_stage_timings,
                    debt_items,
                    settlements,
                )

            per_file_completed: set[Path] = set()
            stage_timings: dict[str, float] = {}
            per_file_debt_items: list[ConvergenceDebt] = []
            settlements = []
            for path in unique_paths:
                invalidate = getattr(self._converger, "invalidate_file", None)
                if callable(invalidate):
                    invalidate(path)
                state = self._converger.converge_file(path)  # type: ignore[attr-defined]
                settlements.extend(settled_convergence_stages(state))
                for stage_name, elapsed in getattr(state, "last_stage_times", {}).items():
                    stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + float(elapsed)
                if bool(getattr(state, "converged", False)):
                    per_file_completed.add(path)
                else:
                    per_file_debt_items.extend(convergence_debt_from_state(path, state))
            return per_file_completed, time.perf_counter() - started, stage_timings, per_file_debt_items, settlements
        except DaemonOperationCancelled:
            raise
        except Exception as exc:
            logger.warning("live.watcher: post-ingest converge failed: %s", exc)
            return (
                set(),
                time.perf_counter() - started,
                {},
                [ConvergenceDebt(path=path, stage="convergence", error=str(exc)) for path in unique_paths],
                [],
            )

    @contextmanager
    def _pinned_source_tier_evidence(self, paths: Sequence[Path]) -> Iterator[None]:
        """Answer one pass's source-tier evidence questions with one connection.

        ``_source_tier_evidence_retained`` asks the same two questions of
        ``source.db`` for every ``tool-results/`` sidecar the pass committed,
        and each answer opened its own read-only connection: two file opens,
        two schema loads and two single-row queries *per path*. One connection
        and one query per question serves the page, but only within its
        current writer admission: a later admission must re-read the durable
        source rather than authorize cursors from stale cached evidence.

        The read has no failure policy of its own, deliberately. An absent
        table is answered from ``sqlite_schema`` exactly as
        ``_history_sidecar_retained`` already answers it, and a genuine
        SQLite error propagates -- which is what the un-pinned route does
        too: ``_latest_archive_tiers_raw_fingerprint`` resolves a broken read
        to ``None``, and ``_source_tier_evidence_retained`` then asks
        ``_history_sidecar_retained``, whose connection is unguarded and
        whose docstring says so. A source tier that cannot be read must
        requeue the pass rather than resolve to "no evidence", and adding a
        second, softer policy here would only duplicate that decision.
        """
        sidecars = [
            path
            for path in dict.fromkeys(paths)
            if _is_tool_result_sidecar_path(path, provider=Provider.from_string(self._source_name_for(path)))
        ]
        source_db = self._archive_source_db_path()
        if not sidecars or not source_db.exists():
            yield
            return
        with closing(open_readonly_connection(source_db)) as conn:
            self._pinned_raw_fingerprints = self._pinned_latest_raw_fingerprints(
                conn, sidecars, archive_root=source_db.parent
            )
            self._pinned_history_sidecars = self._pinned_history_sidecar_rows(conn, sidecars)
        try:
            yield
        finally:
            self._pinned_raw_fingerprints = None
            self._pinned_history_sidecars = None

    def _pinned_latest_raw_fingerprints(
        self,
        conn: sqlite3.Connection,
        paths: Sequence[Path],
        *,
        archive_root: Path,
    ) -> dict[str, str | None]:
        """``_latest_archive_tiers_raw_fingerprint`` for many paths, one query per chunk.

        ``ROW_NUMBER`` reproduces the per-path ``ORDER BY acquired_at_ms
        DESC, raw_id DESC LIMIT 1`` the single-path query uses, tie-break
        included, so the pinned answer is the answer that route gives.
        """
        resolved: dict[str, str | None] = {}
        keys = [str(path) for path in paths]
        # An archive whose source tier has no ``raw_sessions`` yet answers
        # "no evidence" for every path, which is what the single-path read
        # resolves to as well. Probing the catalog states that as a fact
        # rather than as a swallowed error.
        declared = conn.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'raw_sessions'").fetchone()
        if declared is None:
            return dict.fromkeys(keys, None)
        for start in range(0, len(keys), _SOURCE_EVIDENCE_QUERY_CHUNK):
            chunk = keys[start : start + _SOURCE_EVIDENCE_QUERY_CHUNK]
            placeholders = ", ".join("?" * len(chunk))
            rows = conn.execute(
                f"""
                SELECT source_path, raw_id, blob_hash FROM (
                    SELECT
                        source_path,
                        raw_id,
                        blob_hash,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_path
                            ORDER BY {raw_receipt_order_sql("raw_sessions")} DESC, raw_id DESC
                        ) AS revision_rank
                    FROM raw_sessions
                    WHERE source_path IN ({placeholders})
                      AND COALESCE(source_index, 0) >= 0
                )
                WHERE revision_rank = 1
                """,
                chunk,
            ).fetchall()
            for source_path, raw_id, blob_hash in rows:
                resolved[str(source_path)] = _retained_raw_fingerprint(raw_id, blob_hash, archive_root=archive_root)
        for key in keys:
            resolved.setdefault(key, None)
        return resolved

    def _pinned_history_sidecar_rows(self, conn: sqlite3.Connection, paths: Sequence[Path]) -> dict[str, bool]:
        """``_history_sidecar_retained`` for many paths, one query per chunk."""
        keys = [str(path) for path in paths]
        declared = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'history_sidecars'"
        ).fetchone()
        if declared is None:
            return dict.fromkeys(keys, False)
        present: set[str] = set()
        for start in range(0, len(keys), _SOURCE_EVIDENCE_QUERY_CHUNK):
            chunk = keys[start : start + _SOURCE_EVIDENCE_QUERY_CHUNK]
            placeholders = ", ".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT DISTINCT source_path FROM history_sidecars WHERE source_path IN ({placeholders})",
                chunk,
            ).fetchall()
            present.update(str(row[0]) for row in rows)
        return {key: key in present for key in keys}

    def _latest_raw_fingerprint(self, path: Path) -> str | None:
        pinned = self._pinned_raw_fingerprints
        if pinned is not None and str(path) in pinned:
            return pinned[str(path)]
        return self._latest_archive_tiers_raw_fingerprint(path)

    def _source_tier_evidence_retained(self, path: Path, *, raw_fingerprint: str | None) -> bool:
        """Whether ``path``'s consumed bytes left evidence the archive still holds.

        An ordinary transcript's raw id comes back from the batch that wrote
        it and is corroborated afterwards by the index tier, so the id itself
        is the proof and re-reading source.db per path would only cost time.
        A Claude Code ``tool-results/`` sidecar has no index-tier trace at
        all: its one record anywhere is the source-tier row, and a batch that
        reports a raw id it did not durably retain leaves the bytes reachable
        by nothing. Confirm that row against source.db before the cursor
        claims the bytes.
        """
        if raw_fingerprint is None:
            return False
        if not _is_tool_result_sidecar_path(path, provider=Provider.from_string(self._source_name_for(path))):
            return True
        return self._latest_raw_fingerprint(path) is not None or self._history_sidecar_retained(path)

    def _history_sidecar_retained(self, path: Path) -> bool:
        """Whether ``source.db`` holds a ``history_sidecars`` row for ``path``.

        A read failure here is infrastructure state, not an answer: it
        propagates so the pass can requeue rather than resolving to "no
        evidence" and refusing a cursor the archive may well support.
        """
        pinned = self._pinned_history_sidecars
        if pinned is not None and str(path) in pinned:
            return pinned[str(path)]
        conn = open_readonly_connection(self._archive_source_db_path())
        try:
            declared = conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'history_sidecars'"
            ).fetchone()
            if declared is None:
                return False
            row = conn.execute(
                "SELECT 1 FROM history_sidecars WHERE source_path = ? LIMIT 1",
                (str(path),),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def _archive_source_db_path(self) -> Path:
        return Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent)) / "source.db"

    def _latest_archive_tiers_raw_fingerprint(self, path: Path) -> str | None:
        source_db = self._archive_source_db_path()
        if not source_db.exists():
            return None
        try:
            conn = open_readonly_connection(source_db)
            try:
                row = conn.execute(
                    f"""
                    SELECT raw_id, blob_hash
                    FROM raw_sessions
                    WHERE source_path = ?
                      AND COALESCE(source_index, 0) >= 0
                    ORDER BY {raw_receipt_order_sql("raw_sessions")} DESC, raw_id DESC
                    LIMIT 1
                    """,
                    (str(path),),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return _retained_raw_fingerprint(row[0], row[1], archive_root=source_db.parent)

    def _current_parser_fingerprint(self) -> str:
        if callable(self._parser_fingerprint):
            return self._parser_fingerprint()
        return self._parser_fingerprint

    def _source_name_for(self, path: Path) -> str:
        source = deepest_source_for_path(path, self._sources)
        if source is not None:
            return str(source.name)
        return path.parent.name

    def _can_ingest_appends_directly(self) -> bool:
        backend = getattr(self._polylogue, "backend", None)
        return isinstance(getattr(backend, "db_path", None), Path)

    async def _ingest_full_paths(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _FullIngestResult:
        """Capture mutable state before admitting Source acquisition's writer."""
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        capability = database_capability_for_provider(Provider.CODEX)
        state_paths = [
            path
            for path in paths
            if provider in (Provider.CODEX, Provider.UNKNOWN)
            and capability is not None
            and (member := capability.member(path.name)) is not None
            and member.disposition != "out-of-scope"
        ]
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        captures: dict[Path, PreparedLiveSQLiteCapture | Exception] = {}
        primary: BaseException | None = None
        try:
            if state_paths:
                stage = self._sqlite_capture_stage
                if stage is None:
                    raise RetainedPreparationRetryableError("Live SQLite capture requires its supplied compute stage")
                cancelled = threading.Event()
                preparation = asyncio.ensure_future(
                    asyncio.to_thread(
                        stage.prepare_sqlite_paths,
                        state_paths,
                        archive_root=archive_root,
                        cancelled=cancelled,
                        fallback_provider=provider,
                    )
                )
                try:
                    captures = await asyncio.shield(preparation)
                except asyncio.CancelledError as cancellation:
                    cancelled.set()
                    try:
                        captures = await preparation
                    except BaseException as failure:
                        raise BaseExceptionGroup(
                            "live cancellation and capture drain failed", [cancellation, failure]
                        ) from cancellation
                    raise
            admissions = await self._classify_pre_writer_admissions(
                [path for path in paths if path not in captures], fallback_provider=provider
            )
            return await self._ingest_full_paths_prepared(
                paths,
                source_name=source_name,
                heartbeat=heartbeat,
                attempt_id=attempt_id,
                max_pass_seconds=max_pass_seconds,
                pass_started=pass_started,
                pre_writer_admissions={**admissions, **captures},
            )
        except BaseException as failure:
            primary = failure
            raise
        finally:
            failures: list[BaseException] = []
            for capture in captures.values():
                if isinstance(capture, PreparedLiveSQLiteCapture):
                    try:
                        capture.discard()
                    except BaseException as failure:
                        failures.append(failure)
            if failures:
                if primary is not None:
                    failures.insert(0, primary)
                raise BaseExceptionGroup("live state publication and capture cleanup failed", failures) from primary

    async def _classify_pre_writer_admissions(
        self, paths: list[Path], *, fallback_provider: Provider
    ) -> dict[Path, PreAcquisitionDecision | Exception]:
        """Classify every non-state input on a worker before Source's writer.

        A JSONL admission streams the file's records; taking it under the
        writer would hold every other archive writer for that read.
        """
        if not paths:
            return {}
        cancelled = threading.Event()

        def checkpoint() -> None:
            if cancelled.is_set():
                raise DaemonOperationCancelled("pre-writer admission cancelled")

        classification = asyncio.ensure_future(
            asyncio.to_thread(
                classify_pre_writer_admissions, paths, fallback_provider=fallback_provider, checkpoint=checkpoint
            )
        )
        try:
            return await asyncio.shield(classification)
        except asyncio.CancelledError:
            cancelled.set()
            with suppress(DaemonOperationCancelled):
                await classification
            raise

    async def _ingest_full_paths_prepared(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
        pre_writer_admissions: Mapping[Path, PreparedLiveSQLiteCapture | PreAcquisitionDecision | Exception],
    ) -> _FullIngestResult:
        paths = _enrichment_evidence_first(
            paths, Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        )
        result = await self._run_source_writer(
            "watcher.live_ingest.full",
            self._ingest_full_paths_sync_in_ops_scope,
            paths,
            source_name=source_name,
            heartbeat=heartbeat,
            attempt_id=attempt_id,
            max_pass_seconds=max_pass_seconds,
            pass_started=pass_started,
            pre_writer_admissions=pre_writer_admissions,
        )
        if not result.acquired_raw_ids or _source_tier_acquisition_required():
            return result
        if self._retained_runner is None:
            raise RetainedPreparationRetryableError("Live retained publication requires its supplied owner")
        # Acquisition's writer has physically returned before the same long-lived
        # owner opens the original preparation window and publishes its outcome.
        terminal_refusals: dict[str, RetainedRawDecodeRefusalError] = {}

        def settle_terminal_refusal(_keys: tuple[str, ...], refusal: RetainedRawDecodeRefusalError) -> None:
            terminal_refusals[refusal.raw_id] = refusal

        replay = await self._retained_runner(result.acquired_raw_ids, on_terminal_refusal=settle_terminal_refusal)
        outcomes = replay.receipts
        written = tuple(dict.fromkeys(sid for outcome in outcomes for sid in outcome.written_session_ids))
        changed = tuple(dict.fromkeys(sid for outcome in outcomes for sid in outcome.changed_session_ids))
        stage_timings = dict(result.stage_timings_s)
        for outcome in outcomes:
            _accumulate_stage_timings(stage_timings, _full_publication_stage_timings(outcome.stage_timings_s))
        # A raw whose preparation failed retryably stays retained and
        # unpublished; its path fails so the cursor retries it next pass,
        # while its siblings' receipts above still count.
        failed_raw_ids = {failure.raw_id for failure in replay.failures}
        for failure in replay.failures:
            emit(
                "live.ingest.retained_preparation_failed",
                level=WARNING,
                outcome="error",
                reason="retryable_preparation",
                raw_id=failure.raw_id,
                error_type=type(failure.error).__name__,
                error_detail=str(failure.error),
            )
        retry_failed = self._retained_retryable_failures(result) | {
            path for path, raw_id in result.raw_fingerprints.items() if raw_id in failed_raw_ids
        }
        return replace(
            result,
            stage_timings_s=stage_timings,
            succeeded=[path for path in result.succeeded if path not in retry_failed],
            failed=[*result.failed, *(path for path in result.succeeded if path in retry_failed)],
            settled_exclusions={
                **result.settled_exclusions,
                **self._retained_settled_exclusions(result, terminal_refusals),
            },
            worker_count=1,
            ingested_session_count=len(written),
            ingested_message_count=sum(outcome.written_message_count for outcome in outcomes),
            changed_session_count=len(changed),
            changed_session_ids=changed,
        )

    def _retained_retryable_failures(self, result: _FullIngestResult) -> set[Path]:
        """Paths whose retained publication settled a retryable refusal.

        Publication records the refusal as typed Source evidence beside the
        failure (a membership cohort that may not move its accepted head this
        pass). The observation is complete but not admitted, so its cursor
        must retry rather than settle.
        """
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        failed: set[Path] = set()
        with closing(open_readonly_connection(archive_root / "source.db")) as source:
            for path, raw_id in result.raw_fingerprints.items():
                if path not in result.succeeded:
                    continue
                if (
                    source.execute(
                        "SELECT 1 FROM raw_sessions AS r JOIN raw_artifacts AS a ON a.raw_id = r.raw_id "
                        "WHERE r.raw_id = ? AND r.parse_error IS NOT NULL AND a.artifact_kind = ? LIMIT 1",
                        (raw_id, RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value),
                    ).fetchone()
                    is not None
                ):
                    failed.add(path)
        return failed

    def _retained_settled_exclusions(
        self, result: _FullIngestResult, terminal_refusals: Mapping[str, RetainedRawDecodeRefusalError]
    ) -> dict[Path, str]:
        """Read each acquired raw's terminal outcome after its retained publication.

        Acquisition no longer parses, so a path's exclusion comes from the
        typed Source evidence its canonical publication settled: a terminal
        decode refusal is corrupt input, and a current-parser census that
        found no session is a settled no-session observation (xf8qp).
        """
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        fingerprint = raw_authority_parser_fingerprint()
        settled: dict[Path, str] = {}
        with closing(open_readonly_connection(archive_root / "source.db")) as source:
            for path, raw_id in result.raw_fingerprints.items():
                if path not in result.succeeded:
                    continue
                corrupt = raw_id in terminal_refusals or (
                    source.execute(
                        "SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND artifact_kind = ? LIMIT 1",
                        (raw_id, RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value),
                    ).fetchone()
                    is not None
                )
                if corrupt:
                    settled[path] = REFUSED_CORRUPT_INPUT
                elif not _hook_carrier_raw(source, raw_id, path) and (
                    source.execute(
                        "SELECT 1 FROM raw_membership_census WHERE raw_id = ? AND parser_fingerprint = ? "
                        "AND status = 'non_session'",
                        (raw_id, fingerprint),
                    ).fetchone()
                    is not None
                ):
                    settled[path] = REFUSED_NO_SESSIONS
        return settled

    async def _run_source_writer(
        self,
        actor: str,
        function: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Run a body that writes the Source tier on the daemon's writer.

        Source SQL refuses any write without a held writer lease, so a body
        run on a bare worker thread would fail at its first durable statement
        (a blob reservation, a raw row) after staging bytes. Without the
        writer runner the route is refused before any work.
        """
        if self._sync_runner is None:
            raise UnleasedWriteError(f"{actor} writes the Source tier and requires the daemon writer runner")
        return cast(T, await self._sync_runner(actor, function, *args, **kwargs))

    async def _run_sync(
        self,
        actor: str,
        function: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Run blocking batch work through the daemon's exit-safe writer runner."""
        if self._sync_runner is not None:
            return cast(T, await self._sync_runner(actor, function, *args, **kwargs))
        return await asyncio.to_thread(function, *args, **kwargs)

    def _ingest_full_paths_sync_in_ops_scope(
        self,
        paths: list[Path],
        **kwargs: Any,
    ) -> _FullIngestResult:
        """Run the blocking full-ingest body under its own ``ops.db`` scope.

        ``_run_sync`` hands the body to a worker thread, and the scope is
        thread-local, so the scope entered in :meth:`ingest_files` does not
        reach it. Entering one here shares a single ``ops.db`` connection
        across this body's cursor and telemetry writes as well; it is left
        before the thread returns.
        """
        with self._cursor.ops_write_scope():
            return self._ingest_full_paths_sync(paths, **kwargs)

    def _ingest_full_paths_sync(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
        pre_writer_admissions: Mapping[Path, PreparedLiveSQLiteCapture | PreAcquisitionDecision | Exception],
    ) -> _FullIngestResult:
        """Acquire and write one source group; a storage fault takes its staged blobs with it.

        Blobs staged for earlier files in the pass are published only by the
        write that the fault prevented. Leaving their private staging copies
        behind would spend more of an already-full archive on every retry.
        """
        publishers: list[ArchiveBlobPublisher] = []
        zip_inputs: dict[Path, _CapturedZipEnumeration] = {}
        try:
            return self._ingest_full_paths_sync_staged(
                paths,
                publishers=publishers,
                zip_inputs=zip_inputs,
                source_name=source_name,
                heartbeat=heartbeat,
                attempt_id=attempt_id,
                max_pass_seconds=max_pass_seconds,
                pass_started=pass_started,
                pre_writer_admissions=pre_writer_admissions,
            )
        except Exception as exc:
            # Classify here, not by type: the publication flush raises a raw
            # SQLITE_FULL/ENOSPC that only a later handler converts.
            if storage_fault_kind(exc) is not None:
                for publisher in publishers:
                    discard_pending = getattr(publisher, "discard_pending", None)
                    if callable(discard_pending):
                        discard_pending()
            raise
        finally:
            with ExitStack() as cleanup:
                for captured_input in zip_inputs.values():
                    cleanup.callback(captured_input.close)

    def _ingest_full_paths_sync_staged(
        self,
        paths: list[Path],
        *,
        publishers: list[ArchiveBlobPublisher],
        zip_inputs: dict[Path, _CapturedZipEnumeration],
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
        pre_writer_admissions: Mapping[Path, PreparedLiveSQLiteCapture | PreAcquisitionDecision | Exception],
    ) -> _FullIngestResult:
        if not paths:
            return _FullIngestResult(succeeded=[], failed=[], source_payload_read_bytes=0)
        pass_clock_started = pass_started if pass_started is not None else time.monotonic()
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        blob_root = archive_root / "blob"
        raw_records: list[RawSessionRecord] = []
        raw_by_record: dict[_FullRecordKey, Path] = {}
        raw_byte_sizes: dict[Path, int] = {}
        raw_frontier_sizes: dict[Path, int] = {}
        raw_payloads: dict[str, bytes] = {}
        raw_source_names: dict[Path, str] = {}
        raw_source_revisions: dict[Path, str] = {}
        raw_sqlite_source_paths: dict[Path, Path] = {}
        raw_canonical_source_paths: dict[Path, str] = {}
        raw_profile_keys: dict[Path, str] = {}
        raw_profile_source_paths: dict[Path, Path] = {}
        raw_source_fingerprints: dict[Path, str] = {}
        captured_content_hashes: dict[Path, str] = {}
        captured_file_observations: dict[Path, tuple[int, int, int, int, int]] = {}
        captured_observation_times_ns: dict[Path, int] = {}
        failed: list[Path] = []
        antigravity_excised_paths: set[Path] = set()
        preparation_deferred_paths: list[Path] = []
        source_read_deferred_paths: list[Path] = []
        ingested: list[Path] = []
        source_payload_read_bytes = 0
        fallback_provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        acquisition_capture_mode = None if fallback_provider is Provider.UNKNOWN else fallback_provider
        source_db = archive_root / "source.db"
        if not source_db.is_file():
            logger.error("source-only acquisition refused because the durable source tier is missing: %s", source_db)
            return _FullIngestResult(succeeded=[], failed=list(paths), source_payload_read_bytes=0)
        from polylogue.storage.blob_publication import ArchiveBlobPublisher, require_published

        blob_store = ArchiveBlobPublisher(source_db, blob_root)
        publishers.append(blob_store)
        publishers.extend(
            capture.publisher
            for capture in pre_writer_admissions.values()
            if isinstance(capture, PreparedLiveSQLiteCapture)
        )
        archive_active = self._archive_active(archive_root)
        archive_bootstrapped = False
        if heartbeat is not None:
            heartbeat(
                "full_archive_storage_probe",
                current_path=paths[0],
                source_payload_read_bytes=0,
                stage_payload=self._archive_storage_probe_payload(
                    archive_root, archive_active=archive_active, archive_bootstrapped=archive_bootstrapped
                ),
                force=True,
            )
        excluded_paths: dict[Path, str] = {}
        detection_fallbacks: dict[Path, str] = {}
        acquisition_time_budget_exceeded = False
        reached_any_path = False

        def admit_acquisition(path: Path) -> bool:
            nonlocal acquisition_time_budget_exceeded, reached_any_path
            pass_exhausted = _ingest_pass_exhausted(
                max_pass_seconds=max_pass_seconds if reached_any_path else None,
                pass_started=pass_clock_started,
                checkpoint="full_acquisition_file",
            )
            if pass_exhausted:
                acquisition_time_budget_exceeded = True
                excluded_paths[path] = REFUSED_UNATTEMPTED_TIME_BUDGET
                return False
            reached_any_path = True
            return True

        antigravity_pairs: dict[Path, tuple[Any, ParsedSession]] = {}
        # Vendor conversion supplies the acquired export; it is not discarded
        # parser prewarming. Derived-only outage keeps its existing deferral.
        antigravity_pb_paths = [
            path
            for path in paths
            if not _source_tier_acquisition_required()
            and fallback_provider is Provider.ANTIGRAVITY
            and path.suffix.lower() == ".pb"
            and antigravity.classify_source_path(path).role is antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF
        ]
        if antigravity_pb_paths:
            from polylogue.sources.source_parsing import (
                _antigravity_source_root,
                iter_antigravity_language_server_sessions,
            )

            configured_source = deepest_source_for_path(antigravity_pb_paths[0], self._sources)
            source_root = (
                configured_source.root
                if configured_source is not None
                else _antigravity_source_root(antigravity_pb_paths[0])
            )
            source = Source(name="antigravity", path=source_root)
            for path in antigravity_pb_paths:
                try:
                    observed_at_ns = time.time_ns()
                    captured_file_observations[path] = _file_observation(path.stat())
                except OSError:
                    continue
                captured_observation_times_ns[path] = observed_at_ns
            try:
                for raw_data, session in iter_antigravity_language_server_sessions(
                    source,
                    capture_raw=True,
                    blob_root=blob_root,
                    blob_store=blob_store,
                    only_cascade_ids=frozenset(path.stem for path in antigravity_pb_paths),
                    excised=antigravity_excised_paths,
                    admit_path=admit_acquisition,
                ):
                    if raw_data is not None:
                        antigravity_pairs[Path(raw_data.source_path)] = (raw_data, session)
            except Exception as exc:
                raise_if_operation_cancelled(exc)
                raise_if_storage_fault(exc, kinds=ARCHIVE_SIDE_FAULTS)
                logger.exception("antigravity: language-server cohort conversion failed")
            for path in antigravity_pb_paths:
                if path in excluded_paths:
                    continue
                pair = antigravity_pairs.get(path)
                if pair is None:
                    failed.append(path)
                    continue
                raw_data, session = pair
                if raw_data.blob_hash is None or raw_data.blob_size is None:
                    failed.append(path)
                    continue
                if path not in captured_file_observations:
                    failed.append(path)
                    continue
                raw_id = raw_data.blob_hash
                raw_byte_sizes[path] = raw_data.blob_size
                raw_source_names[path] = Provider.ANTIGRAVITY.value
                captured_content_hashes[path] = raw_id
                if raw_data.canonical_source_path is not None:
                    raw_canonical_source_paths[path] = raw_data.canonical_source_path
                raw_records.append(
                    RawSessionRecord(
                        raw_id=raw_id,
                        payload_provider=Provider.ANTIGRAVITY,
                        capture_mode=Provider.ANTIGRAVITY,
                        source_name=Provider.ANTIGRAVITY.value,
                        source_path=str(path),
                        canonical_source_path=raw_data.canonical_source_path,
                        source_index=0,
                        blob_size=raw_data.blob_size,
                        blob_publication_receipt_id=raw_data.blob_publication_receipt_id,
                        acquired_at=datetime.now(UTC).isoformat(),
                        file_mtime=raw_data.file_mtime,
                        captured_source_revision=raw_id,
                        captured_file_observation=captured_file_observations[path],
                    )
                )
                raw_by_record[_full_record_key(raw_records[-1])] = path
                ingested.append(path)
        from polylogue.sources.sqlite_export import source_byte_page_sequence

        # One reader process serves this page of inputs; a failed capture
        # retires only its own reader. A reader per file made capture pay one
        # interpreter start per input.
        with source_byte_page_sequence() as byte_pages:
            for path in (path for path in paths if path not in antigravity_pb_paths):
                if not admit_acquisition(path):
                    continue
                blob_hash: str | None = None
                blob_publication_receipt_id: str | None = None
                prepared = pre_writer_admissions[path]
                if isinstance(prepared, Exception):
                    raise_if_operation_cancelled(prepared)
                    raise_if_storage_fault(prepared, kinds=_snapshot_fault_kinds(prepared))
                    if retryable_read_fault(prepared):
                        source_read_deferred_paths.append(path)
                    else:
                        failed.append(path)
                    continue
                captured_sqlite = prepared if isinstance(prepared, PreparedLiveSQLiteCapture) else None
                try:
                    observed_at_ns = captured_sqlite.observed_at_ns if captured_sqlite is not None else time.time_ns()
                    stat = captured_sqlite.source_stat if captured_sqlite is not None else path.stat()
                except OSError as exc:
                    if retryable_read_fault(exc):
                        source_read_deferred_paths.append(path)
                    else:
                        failed.append(path)
                    continue
                captured_file_observations[path] = _file_observation(stat)
                captured_observation_times_ns[path] = observed_at_ns
                admission = captured_sqlite.admission if captured_sqlite is not None else prepared
                assert isinstance(admission, PreAcquisitionDecision)
                if admission.refused:
                    assert admission.excluded_reason is not None
                    self._mark_refused_cursor(
                        path,
                        stat,
                        source_name=fallback_provider.value,
                        reason=admission.excluded_reason,
                        excluded=excluded_paths,
                    )
                    continue
                if admission.excluded_reason is not None:
                    if admission.detection_crash is not None:
                        detection_fallbacks[path] = admission.detection_crash
                    logger.info(
                        "live.source_candidate_not_admitted path=%s provider=%s reason=%s",
                        path,
                        fallback_provider.value,
                        admission.excluded_reason,
                    )
                    self._mark_excluded_cursor(
                        path,
                        stat,
                        source_name=(admission.detected_provider or fallback_provider).value,
                        reason=admission.excluded_reason,
                        excluded=excluded_paths,
                    )
                    continue
                hermes_database_capability = database_capability_for_provider(Provider.HERMES)
                hermes_member = (
                    hermes_database_capability.member(path.name) if hermes_database_capability is not None else None
                )
                hermes_owned_sqlite_name = fallback_provider is Provider.HERMES and (
                    (hermes_member is not None and hermes_member.disposition != "out-of-scope")
                    # Admitted by the Hermes schema signature under an
                    # undeclared filename (see classify_pre_acquisition).
                    or (is_sqlite_path(path) and admission.detected_provider is Provider.HERMES)
                )
                if heartbeat is not None:
                    heartbeat("full_file_scan", current_path=path, source_payload_read_bytes=source_payload_read_bytes)
                if path.suffix.lower() == ".zip":
                    file_mtime = datetime.fromtimestamp(stat.st_mtime_ns / 1000000000, UTC).isoformat()
                    source_only_zip = self._extract_source_only_zip_member_records(
                        path,
                        blob_store=blob_store,
                        fallback_provider=fallback_provider,
                        file_mtime=file_mtime,
                        zip_inputs=zip_inputs,
                    )
                    if source_only_zip is None:
                        failed.append(path)
                        continue
                    zip_records, zip_bytes = source_only_zip
                    captured_zip = zip_inputs.get(path)
                    if captured_zip is not None:
                        input_identity = captured_zip.manifest.inputs[0].captured_identity
                        if input_identity is None:
                            raise ValueError("ZIP input lost its captured namespace")
                        raw_canonical_source_paths[path] = input_identity.canonical_source_path
                        raw_profile_keys[path] = input_identity.profile_key
                        if captured_zip.file_observation is not None:
                            captured_file_observations[path] = captured_zip.file_observation
                    if not zip_records:
                        self._mark_excluded_cursor(
                            path,
                            stat,
                            source_name=fallback_provider.value,
                            reason="zip container held no admissible record",
                            excluded=excluded_paths,
                        )
                        continue
                    for _member_raw_id, member_record in zip_records:
                        raw_records.append(member_record)
                        raw_by_record[_full_record_key(member_record)] = path
                    source_payload_read_bytes += zip_bytes
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                        )
                    ingested.append(path)
                    raw_byte_sizes[path] = stat.st_size
                    continue
                codex_database_capability = database_capability_for_provider(Provider.CODEX)
                codex_member = (
                    codex_database_capability.member(path.name) if codex_database_capability is not None else None
                )
                codex_owned_sqlite_name = (
                    fallback_provider is Provider.CODEX
                    and codex_member is not None
                    and (codex_member.disposition != "out-of-scope")
                )
                antigravity_trajectory = fallback_provider in {
                    Provider.ANTIGRAVITY,
                    Provider.UNKNOWN,
                } and antigravity.looks_like_trajectory_db_path(path)
                if antigravity_trajectory:
                    provider = Provider.ANTIGRAVITY
                    source_name = provider.value
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                            )
                        with sqlite_snapshot_failure_as_oserror():
                            snapshot = snapshot_sqlite_to_blob(
                                path,
                                blob_store,
                                heartbeat=_blob_copy_heartbeat(
                                    heartbeat, path=path, source_payload_read_bytes=source_payload_read_bytes
                                ),
                            )
                        blob_hash, blob_size = (snapshot.blob_hash, snapshot.blob_size)
                        blob_publication_receipt_id = snapshot.blob_publication_receipt_id
                        source_path = snapshot.source_path
                        raw_sqlite_source_paths[path] = source_path
                        raw_canonical_source_paths[path] = str(snapshot.identity_path)
                        raw_id = antigravity.trajectory_raw_id(
                            source_path, snapshot.source_revision, identity_path=snapshot.identity_path
                        )
                        raw_source_revisions[path] = snapshot.source_revision
                        raw_source_fingerprints[path] = snapshot.source_fingerprint
                    except Exception as error:
                        if not antigravity._is_trajectory_storage_error(error):
                            raise
                        raise_if_storage_fault(error, kinds=_snapshot_fault_kinds(error))
                        logger.exception("antigravity: trajectory SQLite acquisition failed: %s", path)
                        if retryable_read_fault(error):
                            source_read_deferred_paths.append(path)
                        else:
                            failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                        )
                elif hermes_owned_sqlite_name:
                    provider = Provider.HERMES
                    source_name = provider.value
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                            )
                        with sqlite_snapshot_failure_as_oserror():
                            snapshot = snapshot_sqlite_to_blob(
                                path,
                                blob_store,
                                heartbeat=_blob_copy_heartbeat(
                                    heartbeat, path=path, source_payload_read_bytes=source_payload_read_bytes
                                ),
                            )
                        blob_hash, blob_size = (snapshot.blob_hash, snapshot.blob_size)
                        blob_publication_receipt_id = snapshot.blob_publication_receipt_id
                        source_path = snapshot.source_path
                        raw_sqlite_source_paths[path] = source_path
                        raw_canonical_source_paths[path] = str(snapshot.identity_path)
                        raw_profile_keys[path] = snapshot.captured_profile_key
                        raw_id = hermes_profile_raw_id(
                            source_path,
                            0,
                            snapshot.source_revision,
                            identity_path=snapshot.captured_profile_source_path,
                            profile_identity=snapshot.captured_profile_key,
                        )
                        raw_source_revisions[path] = snapshot.source_revision
                        raw_source_fingerprints[path] = snapshot.source_fingerprint
                    except OSError as exc:
                        raise_if_storage_fault(exc, kinds=_snapshot_fault_kinds(exc))
                        if retryable_read_fault(exc):
                            source_read_deferred_paths.append(path)
                        else:
                            failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                        )
                elif captured_sqlite is not None:
                    captured_snapshot = captured_sqlite.snapshot
                    if captured_snapshot is None:
                        raise RetainedPreparationRetryableError("accepted state capture has no logical export")
                    provider = Provider.CODEX
                    source_name = provider.value
                    blob_hash, blob_size = (captured_snapshot.blob_hash, captured_snapshot.blob_size)
                    blob_publication_receipt_id = captured_snapshot.blob_publication_receipt_id
                    source_path = captured_snapshot.source_path
                    raw_id = codex_state_raw_id(source_path, captured_snapshot.source_revision)
                    raw_canonical_source_paths[path] = str(captured_snapshot.identity_path)
                    raw_profile_keys[path] = captured_snapshot.captured_profile_key
                    if captured_snapshot.captured_profile_source_path is not None:
                        raw_profile_source_paths[path] = captured_snapshot.captured_profile_source_path
                    captured_file_observations[path] = _file_observation(captured_sqlite.source_stat)
                    captured_observation_times_ns[path] = captured_sqlite.observed_at_ns
                    raw_source_revisions[path] = captured_snapshot.source_revision
                    raw_source_fingerprints[path] = captured_snapshot.source_fingerprint
                    source_payload_read_bytes += blob_size
                elif codex_owned_sqlite_name or (
                    fallback_provider in (Provider.CODEX, Provider.UNKNOWN)
                    and codex_member is not None
                    and (codex_member.disposition != "out-of-scope")
                ):
                    raise RetainedPreparationRetryableError("state acquisition requires pre-writer logical capture")
                else:
                    if (
                        _source_tier_acquisition_required()
                        and _source_tier_acquisition_required()
                        and (fallback_provider is Provider.ANTIGRAVITY)
                        and path.name.endswith(".metadata.json")
                    ):
                        failed.append(path)
                        continue
                    provider = fallback_provider
                    source_name = provider.value
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                            )
                        capture = capture_bound_path(
                            blob_store,
                            path,
                            fallback_provider,
                            heartbeat=_blob_copy_heartbeat(
                                heartbeat, path=path, source_payload_read_bytes=source_payload_read_bytes
                            ),
                            byte_page=byte_pages.page(),
                        )
                        raw_id, blob_size = (capture.blob_hash, capture.blob_size)
                        raw_canonical_source_paths[path] = capture.canonical_source_path
                        captured_file_observations[path] = capture.file_observation
                        if (
                            source_name in {Provider.HERMES.value, Provider.UNKNOWN.value}
                            and capture.captured_profile_key is not None
                        ):
                            raw_profile_keys[path] = capture.captured_profile_key
                            if capture.captured_profile_source_path is not None:
                                raw_profile_source_paths[path] = Path(capture.captured_profile_source_path)
                        blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    except ForeignOriginContentError as exc:
                        self._mark_refused_cursor(
                            path,
                            stat,
                            source_name=fallback_provider.value,
                            reason=foreign_origin_exclusion(exc),
                            excluded=excluded_paths,
                        )
                        continue
                    except OSError as exc:
                        raise_if_storage_fault(exc, kinds=ARCHIVE_SIDE_FAULTS)
                        emit(
                            "live.source_capture.failed",
                            level=WARNING,
                            outcome="degraded",
                            path=str(path),
                            error_type=type(exc).__name__,
                            error_detail=str(exc),
                        )
                        failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy", current_path=path, source_payload_read_bytes=source_payload_read_bytes
                        )
                ingested.append(path)
                acquired_via_sqlite_snapshot = path in raw_source_revisions
                raw_byte_sizes[path] = stat.st_size if acquired_via_sqlite_snapshot else blob_size
                jsonl_boundary: JsonlBoundary | JsonlFrontier | None = None
                if is_jsonl_source_path(str(path)):
                    jsonl_boundary = (
                        jsonl_complete_prefix(raw_payloads[raw_id])
                        if raw_id in raw_payloads
                        else jsonl_complete_prefix_path(blob_store.blob_path(raw_id))
                    )
                if jsonl_boundary is not None:
                    raw_frontier_sizes[path] = jsonl_boundary.prefix_size
                complete_prefix_record_count: int | None = None
                if (
                    jsonl_boundary is not None
                    and jsonl_boundary.incomplete_tail
                    and (not jsonl_boundary.malformed_record)
                    and (0 < jsonl_boundary.prefix_size < blob_size)
                ):
                    if isinstance(jsonl_boundary, JsonlBoundary):
                        complete_prefix_record_count = jsonl_boundary.record_count
                    else:
                        with blob_store.open(raw_id) as prefix_handle:
                            complete_prefix_record_count = jsonl_prefix_record_count(
                                prefix_handle, jsonl_boundary.prefix_size, stop=self._stop_requested
                            )
                raw_source_names[path] = source_name
                if not acquired_via_sqlite_snapshot:
                    captured_content_hashes[path] = raw_id
                    if provider is Provider.HERMES:
                        from polylogue.core.raw_failure_evidence import MissingProfileIdentityError

                        profile_identity = raw_profile_keys.get(path)
                        profile_source_path = raw_profile_source_paths.get(path)
                        if profile_identity is None or profile_source_path is None:
                            raise MissingProfileIdentityError("Hermes capture has no bound profile namespace")
                        blob_hash = raw_id
                        raw_id = hermes_profile_raw_id(
                            str(path),
                            0,
                            blob_hash,
                            identity_path=profile_source_path,
                            profile_identity=profile_identity,
                        )
                        if blob_hash in raw_payloads:
                            raw_payloads[raw_id] = raw_payloads.pop(blob_hash)
                        raw_source_revisions[path] = blob_hash
                raw_records.append(
                    RawSessionRecord(
                        raw_id=raw_id,
                        blob_hash=blob_hash
                        if (acquired_via_sqlite_snapshot or provider is Provider.HERMES) and blob_hash is not None
                        else None,
                        payload_provider=provider,
                        capture_mode=acquisition_capture_mode,
                        source_name=source_name,
                        source_path=str(captured_sqlite.snapshot.source_path)
                        if captured_sqlite is not None and captured_sqlite.snapshot is not None
                        else str(raw_sqlite_source_paths.get(path, path)),
                        canonical_source_path=str(captured_sqlite.snapshot.identity_path)
                        if captured_sqlite is not None and captured_sqlite.snapshot is not None
                        else raw_canonical_source_paths.get(path),
                        captured_profile_key=captured_sqlite.snapshot.captured_profile_key
                        if captured_sqlite is not None and captured_sqlite.snapshot is not None
                        else raw_profile_keys.get(path),
                        source_index=0,
                        blob_size=blob_size,
                        blob_publication_receipt_id=blob_publication_receipt_id,
                        acquired_at=datetime.fromtimestamp(observed_at_ns / 1000000000, UTC).isoformat(),
                        file_mtime=datetime.fromtimestamp(stat.st_mtime_ns / 1000000000, UTC).isoformat(),
                        captured_source_revision=raw_source_revisions.get(path, raw_id),
                        requires_complete_record_boundary=is_jsonl_source_path(str(path)),
                        complete_prefix_size=jsonl_boundary.prefix_size
                        if jsonl_boundary is not None and (not jsonl_boundary.malformed_record)
                        else None,
                        complete_prefix_record_count=complete_prefix_record_count,
                        captured_file_observation=captured_file_observations.get(path),
                    )
                )
                raw_source_revisions.setdefault(path, raw_id)
                raw_by_record[_full_record_key(raw_records[-1])] = path
        archive_write: _ArchiveFullWriteResult | None = None
        raw_deferred_paths: list[Path] = []
        skipped_paths: set[Path] = set()
        time_budget_exceeded = acquisition_time_budget_exceeded
        if raw_records or zip_inputs:
            try:
                for publisher in publishers:
                    publisher.flush()
                for sqlite_capture in pre_writer_admissions.values():
                    if isinstance(sqlite_capture, PreparedLiveSQLiteCapture) and sqlite_capture.snapshot is not None:
                        if sqlite_capture.snapshot is None:
                            raise ValueError("prepared state is missing its retained acquisition")
                        require_published(
                            sqlite_capture.publisher,
                            sqlite_capture.snapshot.blob_hash,
                            source_path=str(sqlite_capture.snapshot.source_path),
                        )
            except Exception as exc:
                if storage_fault_kind(exc) is not None:
                    _release_unwritten_publication_receipts(source_db, raw_records)
                raise
            available_records = [record for record in raw_records if record.raw_id in raw_payloads]
            missing_payload_records = [record for record in raw_records if record.raw_id not in raw_payloads]
            if heartbeat is not None:
                heartbeat(
                    "full_archive_write",
                    current_path=ingested[-1] if ingested else None,
                    source_payload_read_bytes=source_payload_read_bytes,
                    stage_payload={
                        "storage_route": "archive_full",
                        "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
                        "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
                        "input_file_count": len(raw_records),
                        "payload_available_file_count": len(available_records),
                        "payload_unavailable_file_count": len(missing_payload_records),
                        "payload_replayed_from_blob_file_count": len(missing_payload_records),
                    },
                    force=True,
                )
            dispositions_before = prepared_row_dispositions()
            try:
                archive_write = self._acquire_full_records_archive(
                    raw_records,
                    raw_payloads,
                    blob_store,
                    zip_inputs=zip_inputs,
                    max_pass_seconds=max_pass_seconds,
                    pass_started=pass_clock_started,
                )
            except Exception as exc:
                if storage_fault_kind(exc) is not None:
                    _release_unwritten_publication_receipts(source_db, raw_records)
                raise
            skipped_paths = {raw_by_record[key] for key in archive_write.skipped_raw_ids if key in raw_by_record}
            preparation_deferred_paths += [
                raw_by_record[key] for key in archive_write.preparation_deferred_raw_ids if key in raw_by_record
            ]
            raw_deferred_paths = [raw_by_record[key] for key in archive_write.deferred_raw_ids if key in raw_by_record]
            failed.extend(
                raw_by_record[key]
                for key in raw_by_record
                if key not in archive_write.raw_ids
                and key not in archive_write.deferred_raw_ids
                and (key not in archive_write.terminal_raw_ids)
                and (key not in archive_write.skipped_raw_ids)
                and (key not in archive_write.preparation_deferred_raw_ids)
            )
            if heartbeat is not None:
                heartbeat(
                    "full_archive_write_completed",
                    current_path=ingested[-1] if ingested else None,
                    source_payload_read_bytes=source_payload_read_bytes,
                    stage_payload={
                        "storage_route": "archive_full",
                        "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
                        "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
                        "written_raw_count": len(archive_write.raw_ids),
                        "ingested_session_count": archive_write.session_count,
                        "ingested_message_count": archive_write.message_count,
                        "payload_unavailable_file_count": len(missing_payload_records),
                        "payload_replayed_from_blob_file_count": len(missing_payload_records),
                        "prepared_row_dispositions": _disposition_delta(
                            dispositions_before, prepared_row_dispositions()
                        ),
                    },
                    force=True,
                )
            time_budget_exceeded = time_budget_exceeded or archive_write.time_budget_exceeded
        failed_set = set(failed)
        retained_records = (
            {
                **archive_write.raw_ids,
                **archive_write.deferred_raw_ids,
                **archive_write.terminal_raw_ids,
                **archive_write.preparation_deferred_raw_ids,
            }
            if archive_write is not None
            else {}
        )
        raw_fingerprints = {
            path: retained_records[key] for key, path in raw_by_record.items() if key in retained_records
        }
        succeeded_paths = [
            path
            for path in ingested
            if path not in failed_set and path not in skipped_paths and (path not in preparation_deferred_paths)
        ]
        settled_exclusions: dict[Path, str] = {}
        if archive_write is not None and archive_write.settled_exclusions:
            keys_by_path: dict[Path, list[_FullRecordKey]] = {}
            for key, path in raw_by_record.items():
                keys_by_path.setdefault(path, []).append(key)
            for path in succeeded_paths:
                keys = keys_by_path.get(path)
                if not keys or any(key not in archive_write.settled_exclusions for key in keys):
                    continue
                reasons = {archive_write.settled_exclusions[key] for key in keys}
                settled_exclusions[path] = (
                    REFUSED_CORRUPT_INPUT if REFUSED_CORRUPT_INPUT in reasons else REFUSED_NO_SESSIONS
                )
        partial_admissions: dict[Path, PartialAdmission] = {}
        if archive_write is not None and archive_write.partial_admissions:
            for key, path in raw_by_record.items():
                partial = archive_write.partial_admissions.get(key)
                if partial is not None and path in succeeded_paths and (path not in settled_exclusions):
                    partial_admissions.setdefault(path, partial)
        for path in skipped_paths:
            excluded_paths.setdefault(path, REFUSED_UNATTEMPTED_TIME_BUDGET)
        accounted = (
            set(succeeded_paths)
            | failed_set
            | set(excluded_paths)
            | set(preparation_deferred_paths)
            | set(source_read_deferred_paths)
        )
        for path in paths:
            if path not in accounted:
                excluded_paths[path] = "dropped without a recorded outcome"
                emit(
                    "live.ingest.planned_path_unrecorded",
                    level=WARNING,
                    outcome="degraded",
                    reason="no_recorded_outcome",
                    path=str(path),
                )
        result = _full_ingest_result_from_summary(
            succeeded=succeeded_paths,
            failed=failed,
            preparation_deferred=preparation_deferred_paths,
            source_read_deferred=source_read_deferred_paths,
            raw_deferred=raw_deferred_paths if raw_records and archive_write is not None else [],
            source_payload_read_bytes=source_payload_read_bytes,
            excluded=excluded_paths,
            detection_fallbacks={
                path: reason for path, reason in detection_fallbacks.items() if path in succeeded_paths
            },
            settled_exclusions=settled_exclusions,
            partial_admissions=partial_admissions,
            raw_fingerprints=raw_fingerprints,
            raw_byte_sizes=raw_byte_sizes,
            raw_frontier_sizes=raw_frontier_sizes,
            raw_source_names=raw_source_names,
            raw_source_revisions=raw_source_revisions,
            raw_source_fingerprints=raw_source_fingerprints,
            captured_content_hashes=captured_content_hashes,
            captured_canonical_source_paths=raw_canonical_source_paths,
            captured_profile_keys={
                path: key
                for path, key in raw_profile_keys.items()
                if raw_source_names.get(path) == Provider.HERMES.value
            },
            captured_file_observations=captured_file_observations,
            captured_observation_times_ns=captured_observation_times_ns,
            archive_write=archive_write,
            excised_skips=archive_write.excised_skips if archive_write is not None else 0,
            excised_paths=(
                *(archive_write.excised_paths if archive_write is not None else ()),
                *sorted(antigravity_excised_paths),
            ),
            time_budget_exceeded=time_budget_exceeded,
        )
        result = replace(
            result, acquired_raw_ids=tuple(dict.fromkeys(archive_write.raw_ids.values())) if archive_write else ()
        )
        raw_records.clear()
        raw_by_record.clear()
        blob_store.discard_pending()
        return result

    def _archive_active(self, archive_root: Path) -> bool:
        if _source_tier_acquisition_required():
            return (archive_root / "source.db").exists() and (archive_root / "user.db").exists()
        return (
            ArchiveLocation.resolve(archive_root).active_index_path.exists() and (archive_root / "source.db").exists()
        )

    def _archive_storage_probe_payload(
        self,
        archive_root: Path,
        *,
        archive_active: bool,
        archive_bootstrapped: bool,
    ) -> dict[str, object]:
        if _source_tier_acquisition_required():
            tier_paths = {
                tier.value: archive_root / archive_tier_spec(tier).filename
                for tier in (ArchiveTier.SOURCE, ArchiveTier.USER)
            }
            storage_route = "archive_source_acquisition"
            storage_tiers = ",".join(tier_paths)
            storage_write_tiers = ArchiveTier.SOURCE.value
        else:
            tier_paths = {
                spec.tier.value: (
                    ArchiveLocation.resolve(archive_root).active_index_path
                    if spec.tier.value == "index"
                    else archive_root / spec.filename
                )
                for spec in ARCHIVE_TIER_SPECS.values()
            }
            storage_route = "archive_full"
            storage_tiers = _ARCHIVE_RUNTIME_TIERS
            storage_write_tiers = _ARCHIVE_NATIVE_WRITE_TIERS
        present = [tier for tier, path in tier_paths.items() if path.exists()]
        missing = [tier for tier, path in tier_paths.items() if not path.exists()]
        user_versions: dict[str, int | None] = {}
        for tier, path in tier_paths.items():
            if not path.exists():
                user_versions[tier] = None
                continue
            try:
                # Reports the stamped version, so it must not refuse on skew.
                conn = open_readonly_connection(path, validate_schema=False)
                try:
                    user_versions[tier] = int(conn.execute("PRAGMA user_version").fetchone()[0])
                finally:
                    conn.close()
            except sqlite3.Error:
                user_versions[tier] = -1
        return {
            "storage_route": storage_route,
            "storage_tiers": storage_tiers,
            "storage_write_tiers": storage_write_tiers,
            "archive_active": archive_active,
            "archive_bootstrapped": archive_bootstrapped,
            "archive_present_tiers": ",".join(present),
            "archive_missing_tiers": ",".join(missing),
            "archive_tier_user_versions_json": json_dumps(user_versions, sort_keys=True),
        }

    def _acquire_full_records_archive(
        self,
        records: list[RawSessionRecord],
        raw_payloads: dict[str, bytes],
        blob_store: BlobStore,
        *,
        zip_inputs: Mapping[Path, _CapturedZipEnumeration] | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _ArchiveFullWriteResult:
        """Commit acquisition before the original retained owner prepares Index work."""
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        result = _ArchiveFullWriteResult()
        pass_clock_started = pass_started if pass_started is not None else time.monotonic()
        with ArchiveStore.open_source_tier_acquisition(archive_root) as archive:
            if zip_inputs:
                from polylogue.storage.blob_publication import flush_blob_publications

                flush_blob_publications(blob_store)
                conn = archive.source_connection
                for captured_input in zip_inputs.values():
                    conn.execute("SAVEPOINT live_zip_input")
                    try:
                        observed_at_ms = _iso_to_epoch_ms(datetime.now(UTC).isoformat())
                        publish_acquired_zip_input(
                            conn,
                            captured_input.manifest,
                            observed_at_ms=observed_at_ms,
                        )
                        for disposition in captured_input.dispositions:
                            if disposition.entry_ordinal is None or disposition.member_name is None:
                                raise ValueError("ZIP disposition lacks its exact central coordinate")
                            if disposition.member_disposition is None:
                                raise ValueError("ZIP disposition lacks its terminal decision")
                            record_source_item_member_disposition(
                                conn,
                                source_generation_id=captured_input.manifest.source_generation_id,
                                source_item_id=captured_input.item_id,
                                entry_ordinal=disposition.entry_ordinal,
                                member_name=disposition.member_name,
                                disposition=SourceItemMemberDisposition(disposition.member_disposition),
                                diagnostic=disposition.diagnostic or "",
                                observed_at_ms=observed_at_ms,
                            )
                    except BaseException:
                        conn.execute("ROLLBACK TO live_zip_input")
                        conn.execute("RELEASE live_zip_input")
                        raise
                    conn.execute("RELEASE live_zip_input")
                    conn.commit()
            for record_index, record in enumerate(records):
                if record_index > 0 and _ingest_pass_exhausted(
                    max_pass_seconds=max_pass_seconds,
                    pass_started=pass_clock_started,
                    checkpoint="archive_write_record",
                ):
                    result.skipped_raw_ids.update(_full_record_key(item) for item in records[record_index:])
                    result.time_budget_exceeded = True
                    break
                try:
                    provider = record.payload_provider or Provider.from_string(record.source_name)
                    acquisition_provider = (
                        provider
                        if record.capture_mode is None or Provider.from_string(record.capture_mode) is Provider.UNKNOWN
                        else record.capture_mode
                    )
                    started = time.perf_counter()
                    source_raw_id, source_write_name = _admit_live_full_raw(
                        archive,
                        record,
                        raw_payloads.get(record.raw_id),
                        acquisition_provider=acquisition_provider,
                        acquired_at_ms=_iso_to_epoch_ms(record.acquired_at),
                    )
                    result.raw_ids[_full_record_key(record)] = source_raw_id
                    _accumulate_stage_timings(
                        result.stage_timings_s, {source_write_name: time.perf_counter() - started}
                    )
                    if _vanished_truncated_capture(record):
                        # The unterminated tail can never complete: the file
                        # is gone. The retained capture is terminal corrupt
                        # input, settled by its retained publication.
                        archive.record_raw_failure_evidence(
                            source_raw_id,
                            provider=acquisition_provider,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=_iso_to_epoch_ms(record.acquired_at),
                            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
                        )
                        archive.mark_raw_parse_failed(
                            source_raw_id,
                            provider=acquisition_provider,
                            error=ValueError("captured JSONL payload ends before a complete record boundary"),
                            preserve_existing_failure_evidence=True,
                        )
                        continue
                    partial = _stable_truncated_tail_admission(record)
                    if partial is not None:
                        # The full raw is conserved; its canonical parse admits
                        # only the proven complete records. Say so (xf8qp).
                        result.partial_admissions[_full_record_key(record)] = partial
                except ContentExcisedError as exc:
                    # The archive can forget on purpose (polylogue-27m): this
                    # record's blob hash is durably excised, so acquire
                    # refuses to re-store it via the streaming/blob-ref
                    # write route. This is deliberate, not a failure -- log
                    # at info (not the warning level used for real ingest
                    # failures below) and count it separately so operators
                    # don't mistake it for a broken file. source_raw_id is
                    # still None here (the write never completed), so the
                    # record is correctly left out of result.raw_ids and the
                    # caller's cursor bookkeeping treats it the same as any
                    # other unavailable content.
                    result.excised_skips += 1
                    # A ZIP member record is offered by its container path;
                    # normalize the durable ``container:member`` coordinate
                    # back to that offered path for caller-side accounting.
                    # A loose file may itself contain a colon: only a confirmed
                    # ZIP member maps back to its container path.
                    result.excised_paths.add(
                        Path(record.captured_zip_coordinate.declared_container)
                        if record.captured_zip_coordinate is not None
                        else Path(record.source_path)
                    )
                    # The bytes were published (staged and reserved) before the
                    # write refused them. Nothing will ever reference them, so
                    # the success path's receipt consumption never runs and the
                    # orphaned reservation keeps blob GC away from the excised
                    # hash forever -- while every repeat pass over the same
                    # unchanged file accrues another receipt. Release it here so
                    # ordinary GC reclaims the content the operator excised.
                    from polylogue.storage.blob_publication import release_refused_publication_receipt

                    released = release_refused_publication_receipt(
                        archive.source_db_path,
                        record.blob_publication_receipt_id,
                        # Same hash spelling the write used: a streaming
                        # record carries blob_hash, and a route that derived
                        # its raw_id from the content hash carries it there.
                        record.blob_hash or record.raw_id,
                    )
                    logger.info(
                        "live.watcher: skipping durably excised content for %s: %s "
                        "(publication reservation released: %s)",
                        record.source_path,
                        exc,
                        released,
                    )
            if zip_inputs:
                conn = archive.source_connection

                def check_completion_stop() -> None:
                    if self._stop_requested():
                        raise asyncio.CancelledError

                for captured_input in zip_inputs.values():
                    if captured_input.member_count is None:
                        continue  # No normal decoder EOF was observed.
                    actual_count = conn.execute(
                        "SELECT COUNT(*) FROM source_item_raw_members "
                        "WHERE source_generation_id=? AND source_item_id=? AND raw_id IS NOT NULL",
                        (captured_input.manifest.source_generation_id, captured_input.item_id),
                    ).fetchone()[0]
                    if actual_count != len(captured_input.coordinates):
                        continue  # An admission failed or was deferred before raw publication.
                    conn.execute("SAVEPOINT live_zip_completion")
                    try:
                        complete_source_item_enumeration(
                            conn,
                            source_generation_id=captured_input.manifest.source_generation_id,
                            source_item_id=captured_input.item_id,
                            record_coordinates=captured_input.coordinates,
                            member_ordinals=range(captured_input.member_count),
                            member_count=captured_input.member_count,
                            enumeration_fingerprint=captured_input.manifest.enumeration_fingerprint,
                            enumerated_at_ms=_iso_to_epoch_ms(datetime.now(UTC).isoformat()),
                            check_stop=check_completion_stop,
                        )
                    except BaseException:
                        conn.execute("ROLLBACK TO live_zip_completion")
                        conn.execute("RELEASE live_zip_completion")
                        raise
                    conn.execute("RELEASE live_zip_completion")
                    conn.commit()
        return result

    def _extract_source_only_zip_member_records(
        self,
        path: Path,
        *,
        blob_store: BlobStore,
        fallback_provider: Provider,
        file_mtime: str,
        zip_inputs: dict[Path, _CapturedZipEnumeration],
    ) -> tuple[list[tuple[str, RawSessionRecord]], int] | None:
        """Acquire admitted ZIP members without interpreting their bytes.

        A derived-tier outage does not authorize the source tier to infer a
        provider, parse JSON, or classify a member.  It does still enforce the
        ordinary ZIP admission policy before streaming every retained member
        under its exact ``<zip>:<member>`` coordinate.
        """
        source = Source(name=fallback_provider.value, path=path.parent)
        acquired_at = datetime.now(UTC).isoformat()
        records: list[tuple[str, RawSessionRecord]] = []
        total_bytes = 0
        validator = _ZipEntryValidator(fallback_provider, cursor_state=None, zip_path=path)
        try:
            with (
                bind_source_input(path) as captured,
                open_bound_container(
                    blob_store,
                    captured,
                ) as physical,
                zipfile.ZipFile(physical.stream) as zf,
            ):
                central_directory = zf.infolist()
                container_hash, _container_size, input_receipt = physical.retain()
                if input_receipt is None:
                    raise ValueError("ZIP input lacks its accepted publication receipt")
                manifest = acquired_zip_manifest(
                    blob_hash=container_hash,
                    publication_receipt_id=input_receipt,
                    captured_identity=captured.captured_identity,
                    enumeration_fingerprint=zip_acquisition_fingerprint(fallback_provider, preserved_only=True),
                    source_name=fallback_provider.value,
                )
                captured_input = _CapturedZipEnumeration(manifest, file_observation=physical.file_observation)
                previous = zip_inputs.get(path)
                if previous is not None:
                    previous.close()
                zip_inputs[path] = captured_input

                def disposition(info: zipfile.ZipInfo, reason: str, kind: str = "unselected") -> None:
                    captured_input.dispositions.append(
                        SourceInputRecord(
                            coordinate=json_dumps(["zip-member-v1", entry_ordinal], separators=(",", ":")),
                            data=None,
                            entry_ordinal=entry_ordinal,
                            member_name=info.filename,
                            member_disposition=kind,
                            diagnostic=reason,
                        )
                    )

                allowed_path = is_declared_artifact_path if fallback_provider is Provider.UNKNOWN else None
                for entry_ordinal, info in enumerate(central_directory):
                    if (
                        next(
                            iter(
                                validator.filter_entries((info,), allowed_path=allowed_path, on_unselected=disposition)
                            ),
                            None,
                        )
                        is None
                    ):
                        continue
                    if info.file_size == 0:
                        with open_zip_entry(zf, info) as empty:
                            if empty.read(1):
                                raise zipfile.BadZipFile("empty member yielded data")
                        disposition(info, "member is empty")
                        continue
                    split_index = 0
                    source_index = zip_member_source_index(
                        entry_ordinal=entry_ordinal,
                        split_index=split_index,
                    )
                    member_context = ZipEntryReadContext(
                        source=source,
                        zip_path=path,
                        entry=info,
                        file_mtime=file_mtime,
                        provider_hint=fallback_provider,
                        blob_store=blob_store,
                        bound_provider=bound_location_provider(fallback_provider),
                        captured_input_identity=captured.captured_identity,
                        container_blob_hash=physical.blob_hash,
                        decoder_fingerprint=manifest.enumeration_fingerprint,
                        entry_ordinal=entry_ordinal,
                    )
                    try:
                        # The member is preserved through the boundary, which
                        # refuses a foreign record before it is retained.
                        raw_data = stream_preserved_zip_entry_raw_data(
                            zf,
                            member_context,
                            provider_hint=fallback_provider,
                        )
                    except ForeignOriginContentError as exc:
                        disposition(info, f"{exc.code}: {exc}", "refused")
                        self._record_zip_member_refusal(
                            path,
                            entry_ordinal,
                            info.filename,
                            reason=foreign_origin_exclusion(exc),
                            error=f"{exc.code}: {exc}",
                        )
                        continue
                    except ContentIdentityRefusal as exc:
                        disposition(info, f"{_CONTENT_IDENTITY_REFUSED}: {exc}", "refused")
                        self._record_zip_member_refusal(
                            path,
                            entry_ordinal,
                            info.filename,
                            reason=_CONTENT_IDENTITY_REFUSED,
                            error=f"{_CONTENT_IDENTITY_REFUSED}: {exc}",
                        )
                        continue
                    if raw_data.blob_hash is None:
                        raise ValueError("preserved ZIP member lost its accepted bytes")
                    total_bytes += raw_data.blob_size or 0
                    coordinate = raw_data.captured_zip_coordinate
                    if coordinate is None:
                        raise ValueError("accepted ZIP member lost its captured coordinate")
                    member_raw_id = captured_zip_member_raw_id(coordinate, raw_data.blob_hash)
                    record_coordinate = zip_member_record_coordinate(
                        entry_ordinal=entry_ordinal,
                        split_index=split_index,
                        addressing_mode=coordinate.addressing_mode,
                    )
                    captured_input.coordinates.append(record_coordinate)
                    member = SourceItemAdmission(
                        manifest.source_generation_id,
                        captured_input.item_id,
                        record_coordinate,
                        entry_ordinal,
                        split_index,
                        coordinate.addressing_mode.value,
                        raw_data.content_identity,
                    )
                    records.append(
                        (
                            member_raw_id,
                            RawSessionRecord(
                                raw_id=member_raw_id,
                                blob_hash=raw_data.blob_hash,
                                payload_provider=fallback_provider,
                                capture_mode=fallback_provider,
                                source_name=fallback_provider.value,
                                source_path=raw_data.source_path,
                                canonical_source_path=raw_data.canonical_source_path,
                                captured_profile_key=raw_data.captured_profile_key,
                                captured_zip_coordinate=coordinate,
                                source_item=member,
                                source_index=source_index,
                                addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                                content_identity=raw_data.content_identity,
                                blob_size=raw_data.blob_size or 0,
                                blob_publication_receipt_id=raw_data.blob_publication_receipt_id,
                                acquired_at=acquired_at,
                                file_mtime=raw_data.file_mtime,
                            ),
                        )
                    )
                captured_input.member_count = len(central_directory)
        except (zipfile.BadZipFile, OSError) as exc:
            # An aborted scan reconciles nothing; drop its partial refusal set
            # so a later pass starts clean.
            self._zip_member_refusals_this_pass.pop(str(path), None)
            raise_if_storage_fault(exc, kinds=ARCHIVE_SIDE_FAULTS)
            for _raw_id, record in records:
                if record.blob_hash is not None:
                    release_refused_capture(blob_store, record.blob_hash, record.blob_publication_receipt_id)
            logger.warning("Failed to expand inbox ZIP %s: %s", path, exc)
            # A transport/read failure is not evidence that the archive has no
            # admissible members. Keep it distinct from a successful empty
            # extraction so the caller records retryable failure state instead
            # of permanently acknowledging this source coordinate as excluded.
            return None
        self._reconcile_zip_member_refusals(path)
        return records, total_bytes

    def _reconcile_zip_member_refusals(self, path: Path) -> None:
        """Clear every member refusal of ``path`` this completed scan did not re-record.

        Members a later archive revision removed or now admits keep no gap.
        An aborted scan never reaches here, so it reconciles nothing.
        """
        self._cursor.clear_convergence_debt_under_prefix(
            stage="live_ingest_admission",
            subject_type="source_path",
            prefix=_zip_member_debt_prefix(path),
            keep=frozenset(self._zip_member_refusals_this_pass.pop(str(path), ())),
        )

    def _mark_excluded_cursor(
        self,
        path: Path,
        stat: object,
        *,
        source_name: str,
        reason: str = "unspecified",
        excluded: dict[Path, str] | None = None,
    ) -> None:
        """Quarantine a path and record why, so the pass can account for it.

        A planned path that is neither ingested nor failed is invisible in the
        batch counters, which is indistinguishable from an idle source; the
        caller passes its ``excluded`` map so every drop is named.
        """
        if excluded is not None:
            excluded[path] = reason
        st_size = int(getattr(stat, "st_size", 0))
        try:
            authority = CursorPathAuthority.observe(path)
        except FileNotFoundError:
            # The file vanished before its exclusion was recorded: a typed
            # failed cursor, re-evaluated when the path reappears.
            self._cursor.mark_failed(path, authority=None)
            return
        self._cursor.set(
            path,
            st_size,
            authority=authority,
            byte_offset=st_size,
            last_complete_newline=st_size,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=None,
            source_name=source_name,
            st_dev=getattr(stat, "st_dev", None),
            st_ino=getattr(stat, "st_ino", None),
            mtime_ns=getattr(stat, "st_mtime_ns", None),
            excluded=True,
        )

    def _record_zip_member_refusal(self, path: Path, ordinal: int, member: str, *, reason: str, error: str) -> None:
        """Keep one durable gap for one refused ZIP member.

        The single record for every member refusal (a foreign-origin record,
        or a value no archive identity can hold): the archive cursor advances
        with its admissible siblings, so the refused member carries its own
        ``live_ingest_admission`` debt under ``<zip>:#<ordinal>:<member>``
        whose error starts with the typed ``reason``. A later completed scan
        that admits the member, or no longer holds it, clears it
        (:meth:`_reconcile_zip_member_refusals`).
        """
        member_path = _zip_member_debt_subject(path, ordinal, member)
        self._zip_member_refusals_this_pass.setdefault(str(path), set()).add(member_path)
        emit(
            "live.ingest.zip_member_refused",
            level=WARNING,
            outcome="refused",
            source_path=member_path,
            reason=reason,
        )
        self._cursor.record_convergence_debt(
            stage="live_ingest_admission",
            subject_type="source_path",
            subject_id=member_path,
            error=error,
        )

    def _mark_refused_cursor(
        self,
        path: Path,
        stat: object,
        *,
        source_name: str,
        reason: str,
        excluded: dict[Path, str] | None = None,
    ) -> None:
        """Record a typed admission refusal a later rule change must revisit.

        Unlike :meth:`_mark_excluded_cursor` this neither advances the cursor
        to EOF nor quarantines the path: the refusal is a property of the
        current admission rules, not of the bytes (polylogue-dznyt). The
        cursor keeps a counted failure with no content fingerprint, so
        ``list_retry_records`` surfaces it, and a durable ``convergence_debt``
        row names the reason. Every later pass re-evaluates the same path and
        re-records the same refusal, so declaring the provider -- or building
        a streaming route for it -- admits the file with no operator step.
        """
        if excluded is not None:
            excluded[path] = reason
        st_size = int(getattr(stat, "st_size", 0))
        emit(
            "live.ingest.file_refused",
            level=WARNING,
            outcome="refused",
            source_path=str(path),
            reason=reason,
        )
        try:
            authority = CursorPathAuthority.observe(path)
        except FileNotFoundError:
            # The file vanished before its refusal was recorded: a typed
            # failed cursor, re-evaluated when the path reappears.
            self._cursor.mark_failed(path, authority=None)
            return
        self._cursor.set(
            path,
            st_size,
            authority=authority,
            byte_offset=0,
            last_complete_newline=0,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=None,
            source_name=source_name,
            st_dev=getattr(stat, "st_dev", None),
            st_ino=getattr(stat, "st_ino", None),
            mtime_ns=getattr(stat, "st_mtime_ns", None),
            failure_count=1,
            excluded=False,
            allow_backward=True,
        )
        self._cursor.record_convergence_debt(
            stage="live_ingest_admission",
            subject_type="source_path",
            subject_id=str(path),
            error=reason,
        )

    def _resynthesize_cursor_from_source(self, path: Path) -> CursorRecord | None:
        """Reconstruct an append-eligible cursor from durable ``source.db`` evidence.

        The disposable ``ops.db`` cursor is the primary source.  This fallback
        accepts a unique byte-proven full chain, including legacy append rows
        only when retained blobs form contiguous slices of the current source
        bytes.  It recomputes their revision and offsets before replay.  Any
        missing or ambiguous evidence falls through to full capture.
        """
        source_db = self._archive_source_db_path()
        if not source_db.exists():
            return None
        try:
            conn = open_readonly_connection(source_db)
        except sqlite3.Error:
            return None
        try:
            key_rows = conn.execute(
                """
                SELECT DISTINCT logical_source_key
                FROM raw_sessions
                WHERE source_path = ?
                  AND revision_kind = 'full'
                  AND revision_authority = 'byte_proven'
                  AND logical_source_key IS NOT NULL
                """,
                (str(path),),
            ).fetchall()
            if len(key_rows) != 1:
                # No durable full head, or more than one distinct logical
                # identity has ever been captured at this path -- ambiguous.
                return None
            logical_source_key = str(key_rows[0][0])
            rows = conn.execute(
                """
                SELECT raw_id, revision_kind, source_revision, acquisition_generation,
                       revision_authority, blob_size, predecessor_raw_id, baseline_raw_id,
                       append_start_offset, append_end_offset, predecessor_source_revision,
                       lower(hex(blob_hash)) AS blob_hash_hex, source_path, source_index, acquired_at_ms,
                       parsed_at_ms, parse_error
                FROM raw_sessions
                WHERE logical_source_key = ? AND source_revision IS NOT NULL
                """,
                (logical_source_key,),
            ).fetchall()
        except sqlite3.Error:
            return None
        finally:
            conn.close()
        if not rows:
            return None
        blob_hash_by_raw_id = {str(row[0]): row[11] for row in rows}
        # Before offset recording was introduced, append observations were
        # retained with ``source_index=-1`` and no byte window.  Reconstruct
        # those windows only when the retained bytes, the source path, and a
        # unique preceding full observation prove one disjoint prefix chain.
        # This is source evidence, not an ops cursor or a size-only guess.
        path_rows = [row for row in rows if str(row[12]) == str(path) and row[11]]
        if any(
            int(row[13]) == -1
            and str(row[1]) == RawRevisionKind.UNKNOWN.value
            and (row[15] is None or row[16] is not None)
            for row in path_rows
        ):
            # A pre-offset append that never completed its original replay
            # cannot become an append frontier. Full capture preserves its
            # missing parser/index work.
            return None
        full_sizes = [
            (int(row[14]), int(row[5]), str(row[0]))
            for row in path_rows
            if int(row[13]) == 0 and str(row[1]) == RawRevisionKind.FULL.value and str(row[0])
        ]
        full_sizes = sorted(full_sizes)
        inferred: dict[str, tuple[int, int, str, str | None, str, str]] = {}
        if full_sizes:
            try:
                source_size = path.stat().st_size
            except OSError:
                source_size = None
        else:
            source_size = None

        def retained_blob_matches_source(
            blob_path: Path,
            *,
            blob_start: int,
            source_start: int,
            size: int,
        ) -> bool:
            if source_size is None or source_start + size > source_size:
                return False
            try:
                if blob_path.stat().st_size < blob_start + size:
                    return False
                with path.open("rb") as source_handle, blob_path.open("rb") as blob_handle:
                    source_handle.seek(source_start)
                    blob_handle.seek(blob_start)
                    remaining = size
                    while remaining:
                        chunk_size = min(1 << 20, remaining)
                        if source_handle.read(chunk_size) != blob_handle.read(chunk_size):
                            return False
                        remaining -= chunk_size
            except OSError:
                return False
            return True

        if source_size is not None:
            previous_end: int | None = None
            previous_raw_id: str | None = None
            previous_revision: str | None = None
            baseline_raw_id: str | None = None
            for row in sorted(path_rows, key=lambda item: (int(item[14]), str(item[0]))):
                raw_id = str(row[0])
                if str(row[1]) == RawRevisionKind.FULL.value:
                    blob_hash = blob_hash_by_raw_id.get(raw_id)
                    if not blob_hash:
                        inferred.clear()
                        previous_end = None
                        previous_raw_id = None
                        previous_revision = None
                        baseline_raw_id = None
                        continue
                    blob_path = source_db.parent / "blob" / blob_hash[:2] / blob_hash[2:]
                    if not retained_blob_matches_source(
                        blob_path,
                        blob_start=0,
                        source_start=0,
                        size=int(row[5]),
                    ):
                        inferred.clear()
                        previous_end = None
                        previous_raw_id = None
                        previous_revision = None
                        baseline_raw_id = None
                        continue
                    previous_end = int(row[5])
                    previous_raw_id = raw_id
                    previous_revision = str(row[2])
                    baseline_raw_id = raw_id
                    continue
                if row[13] != -1 or row[8] is not None or row[9] is not None:
                    continue
                if (
                    previous_end is None
                    or previous_raw_id is None
                    or previous_revision is None
                    or baseline_raw_id is None
                ):
                    continue
                blob_hash = blob_hash_by_raw_id.get(raw_id)
                if not blob_hash:
                    continue
                blob_path = source_db.parent / "blob" / blob_hash[:2] / blob_hash[2:]
                delta_start = 0
                delta_size = int(row[5])
                end = previous_end + delta_size
                if not retained_blob_matches_source(
                    blob_path,
                    blob_start=delta_start,
                    source_start=previous_end,
                    size=delta_size,
                ):
                    # Replacement, truncation, or an old incompatible payload
                    # shape invalidates the whole reconstruction.
                    inferred.clear()
                    break
                try:
                    delta_hash, _ = sha256_range_from_path(
                        blob_path,
                        start_offset=delta_start,
                        end_offset=int(row[5]),
                    )
                except (EOFError, OSError, ValueError):
                    inferred.clear()
                    break
                inferred[raw_id] = (
                    previous_end,
                    end,
                    previous_raw_id,
                    previous_revision,
                    baseline_raw_id,
                    append_source_revision(previous_revision, delta_hash),
                )
                previous_end = end
                previous_raw_id = raw_id
                previous_revision = inferred[raw_id][5]
        candidates = [
            RevisionCandidate(
                raw_id=str(row[0]),
                logical_source_key=logical_source_key,
                kind=(RawRevisionKind.APPEND if str(row[0]) in inferred else RawRevisionKind(str(row[1]))),
                source_revision=(inferred[str(row[0])][5] if str(row[0]) in inferred else str(row[2])),
                acquisition_generation=int(row[3]),
                authority=(
                    RawRevisionAuthority.BYTE_PROVEN if str(row[0]) in inferred else RawRevisionAuthority(str(row[4]))
                ),
                blob_size=int(row[5]),
                predecessor_source_revision=(
                    inferred[str(row[0])][3]
                    if str(row[0]) in inferred
                    else (str(row[10]) if row[10] is not None else None)
                ),
                predecessor_raw_id=(
                    inferred[str(row[0])][2]
                    if str(row[0]) in inferred
                    else (str(row[6]) if row[6] is not None else None)
                ),
                baseline_raw_id=(
                    inferred[str(row[0])][4]
                    if str(row[0]) in inferred
                    else (str(row[7]) if row[7] is not None else None)
                ),
                append_start_offset=(
                    inferred[str(row[0])][0]
                    if str(row[0]) in inferred
                    else (int(row[8]) if row[8] is not None else None)
                ),
                append_end_offset=(
                    inferred[str(row[0])][1]
                    if str(row[0]) in inferred
                    else (int(row[9]) if row[9] is not None else None)
                ),
            )
            for row in rows
        ]
        try:
            replay_plan = plan_revision_replay(candidates)
        except ValueError:
            return None
        if not replay_plan.accepted_chain:
            return None
        candidate_by_id = {candidate.raw_id: candidate for candidate in candidates}
        if any(
            application.decision is ApplicationDecision.AMBIGUOUS
            and candidate_by_id[application.raw_id].kind is RawRevisionKind.APPEND
            for application in replay_plan.applications
        ):
            # Durable ambiguity is a full-snapshot boundary.  In particular,
            # do not reconstruct from a baseline behind an overlapping or
            # legacy append whose byte frontier was never recorded.
            return None
        if any(
            candidate.kind is RawRevisionKind.APPEND
            and candidate.authority is RawRevisionAuthority.BYTE_PROVEN
            and (candidate.append_start_offset is None or candidate.append_end_offset is None)
            and candidate.baseline_raw_id == replay_plan.accepted_chain[0]
            for candidate in candidates
        ):
            return None
        head_raw_id = replay_plan.accepted_chain[-1]
        head = next(candidate for candidate in candidates if candidate.raw_id == head_raw_id)
        reconstructed_head = head_raw_id in inferred
        append_head = head.kind is RawRevisionKind.APPEND
        if head.kind is not RawRevisionKind.FULL and not append_head and not reconstructed_head:
            return None
        reconstructed_prefix_proof: tuple[str, os.stat_result] | None = None
        if append_head or reconstructed_head:
            if head.append_end_offset is None:
                return None

            def verified_reconstructed_prefix_hash() -> tuple[str, os.stat_result] | None:
                """Return a prefix digest only when one file observation matches every retained component."""
                components: list[tuple[Path, int, int, int]] = []
                expected_end = 0
                for raw_id in replay_plan.accepted_chain:
                    candidate = candidate_by_id[raw_id]
                    blob_hash = blob_hash_by_raw_id.get(raw_id)
                    if blob_hash is None or len(blob_hash) != 64:
                        return None
                    if candidate.kind is RawRevisionKind.FULL:
                        if expected_end:
                            return None
                        source_start = 0
                        source_end = candidate.blob_size
                        blob_start = 0
                    elif candidate.kind is RawRevisionKind.APPEND:
                        if (
                            candidate.append_start_offset != expected_end
                            or candidate.append_end_offset is None
                            or candidate.append_end_offset <= expected_end
                        ):
                            return None
                        source_start = expected_end
                        source_end = candidate.append_end_offset
                        blob_start = 0
                        if raw_id in inferred:
                            blob_start = candidate.blob_size - (source_end - source_start)
                        elif candidate.blob_size != source_end - source_start:
                            return None
                    else:
                        return None
                    if blob_start < 0:
                        return None
                    components.append(
                        (
                            source_db.parent / "blob" / blob_hash[:2] / blob_hash[2:],
                            source_start,
                            source_end,
                            blob_start,
                        )
                    )
                    expected_end = source_end
                if expected_end != head.append_end_offset:
                    return None
                try:
                    with path.open("rb") as source_handle:
                        before = os.fstat(source_handle.fileno())
                        if before.st_size < expected_end:
                            return None
                        digest = sha256()
                        for blob_path, source_start, source_end, blob_start in components:
                            with blob_path.open("rb") as blob_handle:
                                source_handle.seek(source_start)
                                blob_handle.seek(blob_start)
                                remaining = source_end - source_start
                                while remaining:
                                    chunk_size = min(1 << 20, remaining)
                                    source_chunk = source_handle.read(chunk_size)
                                    blob_chunk = blob_handle.read(chunk_size)
                                    if len(source_chunk) != chunk_size or source_chunk != blob_chunk:
                                        return None
                                    digest.update(source_chunk)
                                    remaining -= chunk_size
                        after = os.fstat(source_handle.fileno())
                except OSError:
                    return None
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    return None
                return digest.hexdigest(), after

            reconstructed_prefix_proof = verified_reconstructed_prefix_hash()
            if reconstructed_prefix_proof is None:
                return None
            reconstructed_revisions = [
                (
                    candidate.raw_id,
                    RawRevisionEnvelope(
                        logical_source_key=candidate.logical_source_key,
                        kind=candidate.kind,
                        source_revision=candidate.source_revision,
                        acquisition_generation=candidate.acquisition_generation,
                        predecessor_source_revision=candidate.predecessor_source_revision,
                        predecessor_raw_id=candidate.predecessor_raw_id,
                        baseline_raw_id=candidate.baseline_raw_id,
                        append_start_offset=candidate.append_start_offset,
                        append_end_offset=candidate.append_end_offset,
                        authority=candidate.authority,
                    ),
                )
                for candidate in candidates
                if candidate.raw_id in inferred and candidate.raw_id in replay_plan.accepted_chain
            ]

            # Promotion is a durable source.db write, so it must not run
            # while the cursor it supports is still provisional. Planning can
            # still defer or refuse after this point, and a promotion already
            # committed for a plan that never applied leaves durable state
            # attesting to an append no cursor records. Hand the caller a
            # thunk instead and let it commit once the plan is real.
            def promote_legacy_appends(
                reconstructed_revisions: list[tuple[str, RawRevisionEnvelope]] = reconstructed_revisions,
                reconstructed_prefix_proof: tuple[str, os.stat_result] = reconstructed_prefix_proof,
            ) -> bool:
                try:
                    archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
                    with _open_archive_for_live_write(archive_root) as archive:
                        source_prefix_sha256, source_stat = reconstructed_prefix_proof
                        archive.promote_reconstructed_legacy_append_revisions(
                            reconstructed_revisions,
                            source_prefix_sha256=source_prefix_sha256,
                            source_size=source_stat.st_size,
                            source_mtime_ns=source_stat.st_mtime_ns,
                            source_ctime_ns=source_stat.st_ctime_ns,
                            observed_at_ms=int(datetime.now(UTC).timestamp() * 1000),
                        )
                except (OSError, sqlite3.Error, ValueError):
                    return False
                return True

            self._pending_legacy_promotions[path] = promote_legacy_appends
        blob_hash_hex = blob_hash_by_raw_id.get(head_raw_id)
        if blob_hash_hex is None or len(blob_hash_hex) != 64:
            return None
        if append_head or reconstructed_head:
            if head.append_end_offset is None or reconstructed_prefix_proof is None:
                return None
            byte_offset = head.append_end_offset
            reconstructed_prefix_hash, _source_stat = reconstructed_prefix_proof
            tail_hash = encode_cursor_hash_authority(
                reconstructed_prefix_hash,
                reconstructed_prefix_hash,
                ctime_ns=0,
            )
        else:
            byte_offset = head.blob_size
            tail_hash = encode_cursor_hash_authority(blob_hash_hex, blob_hash_hex, ctime_ns=0)
        source_name = self._source_name_for(path)
        source_provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        if (
            not path_declaration_refuses_session(source_provider, path)
            and frontier_kind_for_origin(origin_from_provider(source_provider)) == "claude-header-body"
        ):
            # The frontier is composed from the live file, so it describes
            # whatever bytes are on disk now -- not the retained bytes this
            # cursor claims authority over. The reconstructed branch already
            # proved its prefix against every retained blob component byte by
            # byte; the full-head branch has proved nothing, so a rewrite
            # preserving `head.blob_size` would be adopted as authority.
            # Require the live prefix to reproduce the retained blob digest.
            # Both proofs read the live file through independent opens. Bind
            # them to ONE observation: between them a same-length rewrite let
            # the retained prefix verify while the frontier was composed over
            # a different body, and the returned cursor then made
            # ``_append_plan`` trust those unread bytes.
            try:
                frontier_before = path.stat()
            except OSError:
                return None
            if not append_head and not reconstructed_head and file_prefix_sha256(path, byte_offset) != blob_hash_hex:
                return None
            composed_tail_hash = claude_semantic_frontier_for_prefix(path, byte_offset)
            if composed_tail_hash is None:
                return None
            try:
                frontier_after = path.stat()
            except OSError:
                return None
            if _file_observation(frontier_before) != _file_observation(frontier_after):
                return None
            tail_hash = composed_tail_hash
        return CursorRecord(
            source_path=str(path),
            byte_size=byte_offset,
            byte_offset=byte_offset,
            last_complete_newline=byte_offset,
            record_count=0,
            updated_at=datetime.now(UTC).isoformat(),
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=head.source_revision,
            tail_hash=tail_hash,
            source_name=None,
            st_dev=None,
            st_ino=None,
            mtime_ns=None,
        )

    def _cursor_references_raw_failure_requiring_full_replay(self, path: Path, cursor: CursorRecord) -> bool:
        """Return whether a typed raw failure invalidates append-only replay.

        Typed raw-failure evidence retains a source observation that did not
        materialize a session. Whether it was terminally rejected or deferred
        while hot, an append-only parser cannot recover its missing prefix.
        When the file subsequently grows, replay the complete source so the
        parser receives its header and preceding messages intact.
        """
        if cursor.content_fingerprint is None:
            return False
        return self._raw_failure_requires_full_replay(path, cursor.content_fingerprint)

    def _raw_failure_requires_full_replay(self, path: Path, raw_id: str) -> bool:
        """Return whether one durable raw ID carries replay-blocking evidence."""
        source_db = self._archive_source_db_path()
        if not source_db.exists():
            return False
        placeholders = ", ".join("?" for _ in RAW_FAILURE_EVIDENCE_KINDS)
        support_pairs = " OR ".join(
            "(a.artifact_kind = ? AND a.support_status = ?)"
            for _ in RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS
        )
        try:
            conn = open_readonly_connection(source_db)
            try:
                return (
                    conn.execute(
                        f"""
                        SELECT 1
                        FROM raw_sessions AS r
                        JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
                        WHERE r.raw_id = ?
                          AND r.origin IS a.origin
                          AND r.source_path IS a.source_path
                          AND r.source_path = ?
                          AND r.source_index IS a.source_index
                          AND a.artifact_kind IN ({placeholders})
                          AND ({support_pairs})
                        LIMIT 1
                        """,
                        (
                            raw_id,
                            str(path),
                            *sorted(RAW_FAILURE_EVIDENCE_KINDS),
                            *[value for pair in RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair],
                        ),
                    ).fetchone()
                    is not None
                )
            finally:
                conn.close()
        except sqlite3.Error:
            return False

    def plan_append(
        self,
        path: Path,
        *,
        cursor: CursorRecord | None = None,
        cursor_is_known: bool = False,
        source_index: int = -1,
    ) -> _AppendPlan | _DeferredAppend | None:
        """Plan one append through the live route's cursor and byte proofs.

        The watcher leaves ``source_index`` at its legacy sentinel because it
        does not own historical ordering. Callers that have independently
        proven an ordering coordinate may provide it here; all validation and
        identity resolution remain in the same production planner.

        ``cursor_is_known`` distinguishes "this caller did not look" from
        "this caller looked and there is no cursor row". A batch that already
        read every path's record in one query knows the second, and re-asking
        per path only repeats a read the batch has already paid for.
        """
        return self._append_plan(path, cursor=cursor, cursor_is_known=cursor_is_known, source_index=source_index)

    def _append_plan(
        self,
        path: Path,
        *,
        cursor: CursorRecord | None = None,
        cursor_is_known: bool = False,
        source_index: int = -1,
    ) -> _AppendPlan | _DeferredAppend | None:
        # Append planning is safe only for newline-delimited record streams.
        # Watch-source names describe acquisition routes, not file semantics:
        # mutable browser snapshots can arrive through the generic inbox.
        source_name = self._source_name_for(path)
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        path_artifact = classify_artifact_path(str(path), provider=provider)
        is_hook_carrier = path_artifact is not None and path_artifact.kind is ArtifactKind.HOOK_EVENT_CARRIER
        # Hook carriers are declared raw-only because they are never session
        # transcripts, but their retained bytes still have a physical append
        # route. Apply the session-parsing refusal only to other raw-only
        # artifacts so a grown carrier can retain its exact new byte slice.
        if not is_hook_carrier and path_declaration_refuses_session(provider, path):
            return None
        if path.suffix.lower() != ".jsonl" and not (path.suffix.lower() == ".ndjson" and is_hook_carrier):
            return None
        if source_name == "hermes" and not is_hook_carrier:
            # polylogue-flxh: the Hermes watch source's only .jsonl artifact
            # class is NeMo Relay ATOF (state.db is .db, ATIF/session
            # snapshots are .json -- see default_sources()'s own docstring).
            # A real ATOF file is shared across every Hermes session on the
            # install, so a growth batch can span a session boundary; the
            # raw-revision-authority replay chain requires exactly one
            # logical session per raw revision, and incremental append on
            # such a batch silently and permanently drops the pre-existing
            # session's new event (confirmed, see the regression test this
            # commit removes the xfail from). ATOF is therefore always
            # routed through the full/bundle ingest path below instead,
            # which already handles multi-session grouping correctly.
            return None
        parser_fingerprint = self._current_parser_fingerprint()
        if cursor is None and not cursor_is_known:
            cursor = self._cursor.get_record(path)
        pending_promotion: Callable[[], bool] | None = None
        if cursor is None:
            # polylogue-aex0: the disposable ops.db cursor is gone (reset,
            # schema mismatch, or never written yet) -- try to resynthesize
            # an equivalent one from source.db's durable revision-chain
            # evidence before giving up to a full capture. A cursor that
            # *does* exist but is stale for another reason (parser upgrade,
            # exclusion, failure bookkeeping) is a deliberate invalidation,
            # not disposable-tier loss, and must keep forcing a full
            # re-ingest exactly as before -- never resynthesized over.
            cursor = self._resynthesize_cursor_from_source(path)
            # Drain both channels now so neither survives this attempt.
            # The promotion is committed only where a real plan is returned;
            # the cursor is published so a deferred outcome can still persist
            # the resynthesized state instead of discarding it.
            pending_promotion = self._pending_legacy_promotions.pop(path, None)
            if cursor is not None:
                self._resynthesized_cursors[path] = cursor
        if cursor is None or cursor.parser_fingerprint != parser_fingerprint or cursor.content_fingerprint is None:
            return None
        if self._cursor_references_raw_failure_requiring_full_replay(path, cursor):
            return None
        expected_prefix_hash = cursor_prefix_hash(cursor.tail_hash)
        source_name = self._source_name_for(path)
        source_provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        claude_session_stream = (
            not path_declaration_refuses_session(source_provider, path)
            and frontier_kind_for_origin(origin_from_provider(source_provider)) == "claude-header-body"
        )
        claude_frontier = decode_claude_semantic_frontier(cursor.tail_hash) if claude_session_stream else None
        if claude_session_stream and claude_frontier is None:
            return None
        if claude_frontier is None and expected_prefix_hash is None:
            return None
        claude_header_sha256: str | None = None
        claude_publication_body_sha256: str | None = None
        stable_hasher: Any | None = None
        try:
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                canonical_source_path = captured_path_coordinate(path, handle.fileno())
                # The cursor this plan extends already carries the file's
                # captured authority; observe only when it does not.
                captured_authority = CursorPathAuthority.of_record(cursor)
                if captured_authority is None or captured_authority.canonical_source_path != canonical_source_path:
                    captured_authority = CursorPathAuthority.observe(path)
                if claude_frontier is None and stat.st_size <= cursor.byte_offset:
                    return None
                if cursor.st_dev is not None and cursor.st_dev != stat.st_dev:
                    return None
                if cursor.st_ino is not None and cursor.st_ino != stat.st_ino:
                    return None
                if (
                    cursor.deferred_end_offset is not None
                    and stat.st_size <= cursor.deferred_end_offset
                    and cursor.mtime_ns is not None
                    and stat.st_mtime_ns == cursor.mtime_ns
                ):
                    # polylogue-hat0: an earlier pass already durably wrote
                    # and revision-bound a raw for this exact byte range
                    # (start_offset..deferred_end_offset), but its authority
                    # was quarantined/ambiguous and it is still pending
                    # resolution. The file is byte-for-byte and mtime
                    # identical to that attempt -- there is nothing new to
                    # capture. Replanning here would re-mint an identical
                    # duplicate raw row and re-defer forever, on every single
                    # watcher tick, without ever advancing. Wait for either
                    # genuine growth past the already-captured window (the
                    # ``stat.st_size <= cursor.deferred_end_offset`` guard
                    # above no longer holds) or authority resolving through
                    # another path.
                    return _DEFER_APPEND
                if claude_frontier is not None:
                    header_end = handle.readline()
                    if not header_end.endswith(b"\n"):
                        return _DEFER_APPEND
                    try:
                        json_loads(header_end)
                    except (UnicodeDecodeError, ValueError):
                        return None
                    claude_header_sha256 = sha256(header_end).hexdigest()
                    start_offset = len(header_end) + claude_frontier.body_bytes
                    handle.seek(len(header_end))
                    stable_hasher = sha256()
                    remaining = claude_frontier.body_bytes
                    while remaining > 0:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk:
                            return _DEFER_APPEND
                        stable_hasher.update(chunk)
                        remaining -= len(chunk)
                    if stable_hasher.hexdigest() != claude_frontier.body_sha256:
                        return None
                else:
                    start_offset = max(cursor.byte_offset, 0)
                if stat.st_size <= start_offset:
                    return None
                append_window = min(stat.st_size - start_offset, _MAX_APPEND_PLAN_PAYLOAD_BYTES)
                handle.seek(start_offset)
                payload = handle.read(append_window)
                newline_at = payload.rfind(b"\n")
                if newline_at < 0:
                    return _DEFER_APPEND
                complete_payload = payload[: newline_at + 1]
                if not complete_payload:
                    return _DEFER_APPEND
                last_complete_newline = start_offset + newline_at + 1
                tail_start = max(0, last_complete_newline - 64 * 1024)
                handle.seek(tail_start)
                accepted_tail = handle.read(last_complete_newline - tail_start)
                accepted_hasher = sha256()
                if claude_frontier is None:
                    handle.seek(0)
                    remaining = start_offset
                    while remaining > 0:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk:
                            return _DEFER_APPEND
                        accepted_hasher.update(chunk)
                        remaining -= len(chunk)
                    if accepted_hasher.hexdigest() != expected_prefix_hash:
                        return None
                remaining = last_complete_newline - start_offset
                accepted_prefix_hash = None
                if claude_frontier is None:
                    while remaining > 0:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk:
                            return _DEFER_APPEND
                        accepted_hasher.update(chunk)
                        remaining -= len(chunk)
                    accepted_prefix_hash = accepted_hasher.hexdigest()
                else:
                    assert stable_hasher is not None
                    publication_hasher = stable_hasher.copy()
                    publication_hasher.update(complete_payload)
                    claude_publication_body_sha256 = publication_hasher.hexdigest()
                final_stat = os.fstat(handle.fileno())
        except OSError:
            return None
        if _file_observation(final_stat) != _file_observation(stat):
            return _DEFER_APPEND
        # The append delta is retained under the location's origin exactly as
        # a full capture is, so every appended record is validated. A foreign
        # record falls back to the full route, whose captured-blob validation
        # records the typed refusal.
        append_source = self._source_name_for(path)
        try:
            admit_bound_bytes(
                complete_payload,
                str(path),
                Provider.from_string(canonical_acquisition_provider(append_source, source_name=append_source)),
            )
        except ForeignOriginContentError:
            return None
        append_result = self._append_payload_for_provider(path, self._source_name_for(path), complete_payload)
        if append_result is None:
            return None
        append_payload, native_id_hint, acquisition_native_id_hint = append_result
        tail_hash = sha256(complete_payload).hexdigest()
        # Planning succeeded, so the reconstructed legacy chain this plan
        # rests on is now worth making durable.
        if pending_promotion is not None and not pending_promotion():
            return None
        return _AppendPlan(
            path=path,
            canonical_source_path=canonical_source_path,
            captured_profile_key=captured_authority.captured_profile_key,
            source_name=self._source_name_for(path),
            start_offset=start_offset,
            last_complete_newline=last_complete_newline,
            stat_size=stat.st_size,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            payload=append_payload,
            payload_hash=tail_hash,
            cursor_fingerprint=cursor.content_fingerprint,
            bytes_read=len(payload),
            source_index=source_index,
            accepted_tail_hash=sha256(accepted_tail).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
            accepted_prefix_hash=accepted_prefix_hash,
            authority_bytes_read=last_complete_newline,
            native_id_hint=native_id_hint,
            acquisition_native_id_hint=acquisition_native_id_hint,
            accepted_claude_body_sha256=(claude_frontier.body_sha256 if claude_frontier is not None else None),
            accepted_claude_body_bytes=(claude_frontier.body_bytes if claude_frontier is not None else None),
            accepted_claude_header_sha256=claude_header_sha256,
            accepted_claude_publication_body_sha256=claude_publication_body_sha256,
            parser_fingerprint=parser_fingerprint,
        )

    def _append_payload_for_provider(
        self, path: Path, source_name: str, payload: bytes
    ) -> tuple[bytes, str | None, str | None] | None:
        """Return literal bytes plus logical and acquisition identity hints.

        polylogue-u19l: this used to prepend a synthetic ``session_meta``
        line ahead of ``payload`` for Codex before hashing/storing it, so the
        raw row could self-describe its provider session id on independent
        replay. That made the stored blob architecturally never a literal
        byte-slice of the live file, which permanently defeats a live-source
        byte-identity re-verification check (see
        ``storage/raw_retention.py``'s live-source-verification plan) even
        when the live file is completely untouched.

        Now the identity is resolved here exactly as before, but returned as
        a sidecar hint instead of being spliced into the hashed bytes.
        Callers persist the Codex acquisition hint to
        ``raw_sessions.native_id`` and pass the logical hint back as the
        parser's ``fallback_id`` at replay time
        (``revision_backfill.parse_retained_raw_sessions``), which is exactly
        equivalent for Codex: ``_parse_records`` only ever falls back to
        ``fallback_id`` when the payload carries no ``session_meta`` record
        of its own -- true for every append delta -- so the resolved
        identity still wins in precisely the same cases it used to.

        Historical rows written before this change still carry the
        synthetic header in their stored bytes; this function only affects
        NEW writes going forward, per polylogue-u19l's scope.
        """
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        path_artifact = classify_artifact_path(str(path), provider=provider)
        if path_artifact is not None and path_artifact.kind is ArtifactKind.HOOK_EVENT_CARRIER:
            return payload, None, None
        if provider in {Provider.CODEX, Provider.CLAUDE_CODE}:
            identity = self._existing_provider_session_id(
                path,
                expected_origin=origin_from_provider(provider).value,
            )
            capability = append_capability_receipt(
                provider=provider.value,
                package_version="live",
                element_kind="session_record_stream",
                stable_session_identity=identity is not None,
            )
            if capability.status != "supported":
                return None
        else:
            identity = None
        if identity is None:
            # Append acquisition binds the delta to its declared session and
            # refuses a plan without one (hook carriers excepted above). With
            # no identity to bind -- a provider without a stable session
            # identity, or a session that lives only in an unpromoted cold
            # build -- the full route re-reads the file instead.
            return None
        if provider is Provider.CODEX:
            # A Codex append-mode delta is the file's tail bytes only -- the
            # real `session_meta` header that carries native-session identity
            # was already consumed by an earlier full/append observation and
            # is not part of this delta. `identity` is recovered from durable
            # evidence (`_existing_provider_session_id`: the archived
            # session's own native id, or this file's own previously-read
            # `session_meta` line) before hashing, never guessed, and is
            # returned as a sidecar hint instead of being spliced into the
            # hashed bytes (polylogue-u19l) -- see this method's docstring.
            logger.info(
                "codex_append_identity_resolved_as_sidecar_hint",
                path=str(path),
                identity=identity,
                reason="append-mode delta lacks its own session_meta header; "
                "identity recovered from archived session / prior session_meta "
                "line and carried as native_id_hint, not spliced into hashed bytes",
            )
            assert identity is not None
            return payload, identity, identity
        if provider is Provider.CLAUDE_CODE and not self._claude_code_tail_matches_existing_identity(
            path, payload, existing_id=identity
        ):
            return None
        # Claude append raws have historically used native_id=NULL. Its own
        # records carry sessionId, so the resolved identity is needed for
        # governance but must not change deterministic acquisition identity
        # for a retry of pre-upgrade bytes.
        return payload, identity, None

    def _existing_provider_session_id(self, path: Path, *, expected_origin: str) -> str | None:
        identity = self._existing_archive_session_native_id(path, expected_origin=expected_origin)
        if identity is not None:
            return identity
        if expected_origin != Origin.CODEX_SESSION.value:
            return None
        codex_identity = self._codex_session_meta_native_id(path)
        if codex_identity is None:
            return None
        # An index-only identity recovery is valid when the source tier has no
        # row for this path.  It is not valid when the path is already owned by
        # another origin: accepting the Codex id from the index in that case
        # would turn a mixed-origin path collision into an append match.
        if self._source_path_has_conflicting_origin(path, expected_origin=expected_origin):
            return None
        if self._archive_has_native_session("codex-session", codex_identity):
            return codex_identity
        return None

    def _source_path_has_conflicting_origin(self, path: Path, *, expected_origin: str) -> bool:
        """Reject a raw path whose source or joined indexed origin disagrees."""
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        index_db = ArchiveLocation.resolve(archive_root).active_index_path
        if not source_db.exists() or not index_db.exists():
            return False
        try:
            conn = open_readonly_connection(index_db)
            try:
                attach_readonly_database(conn, source_db, alias="source_tier")
                row = conn.execute(
                    """
                    SELECT 1
                    FROM source_tier.raw_sessions AS r
                    LEFT JOIN sessions AS s ON s.raw_id = r.raw_id
                    WHERE r.source_path = ?
                      AND (r.origin <> ? OR (s.session_id IS NOT NULL AND s.origin <> ?))
                    LIMIT 1
                    """,
                    (str(path), expected_origin, expected_origin),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            # This query protects an append from adopting another session's
            # native id. An unavailable ownership view is unsafe to treat as
            # unowned, so defer instead of using the global Codex fallback.
            logger.warning(
                "live.watcher: source-path ownership view unavailable for %s; refusing Codex identity fallback: %s",
                path,
                exc,
            )
            return True
        return row is not None

    def _codex_session_meta_native_id(self, path: Path) -> str | None:
        try:
            with path.open("rb") as handle:
                line = handle.readline(1024 * 1024)
        except OSError:
            return None
        if not line:
            return None
        try:
            record = json_loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError):
            return None
        if not isinstance(record, dict) or record.get("type") != "session_meta":
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None
        value = payload.get("id")
        return value if isinstance(value, str) and value.strip() else None

    def _archive_has_native_session(self, origin: str, native_id: str) -> bool:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        index_db = ArchiveLocation.resolve(archive_root).active_index_path
        if not index_db.exists():
            return False
        try:
            conn = open_readonly_connection(index_db)
            try:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM sessions AS s
                    WHERE s.origin = ? AND s.native_id = ?
                    LIMIT 1
                    """,
                    (origin, native_id),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        return row is not None

    def _existing_archive_session_native_id(self, path: Path, *, expected_origin: str) -> str | None:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        index_db = ArchiveLocation.resolve(archive_root).active_index_path
        source_db = archive_root / "source.db"
        if not index_db.exists() or not source_db.exists():
            return None
        try:
            conn = open_readonly_connection(index_db)
            try:
                attach_readonly_database(conn, source_db, alias="source_tier")
                row = conn.execute(
                    """
                    SELECT s.native_id
                    FROM sessions AS s
                    JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                    WHERE s.origin = ? AND r.origin = ? AND r.source_path = ?
                    ORDER BY s.sort_key_ms DESC, s.created_at_ms DESC, s.session_id DESC
                    LIMIT 1
                    """,
                    (expected_origin, expected_origin, str(path)),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) and value.strip() else None

    def _claude_code_tail_matches_existing_identity(
        self, path: Path, payload: bytes, *, existing_id: str | None
    ) -> bool:
        if existing_id is None:
            return False
        session_ids: set[str] = set()
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                record = json_loads(line)
            except ValueError:
                return False
            if not isinstance(record, dict):
                return False
            session_id = record.get("sessionId")
            if isinstance(session_id, str) and session_id.strip():
                session_ids.add(session_id)
        if not session_ids:
            return existing_id == path.stem
        return any(existing_id == session_id or existing_id.startswith(f"{session_id}:") for session_id in session_ids)

    def _raw_retention_backlog_paths(self, *, exclude: set[Path]) -> list[Path]:
        """Return retry-due raw-retention backlog this pass should also drain.

        Retention is the recurring owner of superseded live raw payloads, and a
        recurring owner is only bounded-retrying if it comes back to what its
        last bounded pass did not reach. The backlog lives in the ordinary
        ``convergence_debt`` ledger under :data:`RAW_RETENTION_STAGE`, which
        already carries the attempt count and the exponential ``next_retry_at``
        backoff, so this reads that ledger rather than keeping a second one.

        An unreadable ledger is deliberately not caught here. The backlog is
        the only record that this work is still owed, so a pass that could not
        read it has not proven there is none; swallowing the error would make
        a broken ops tier look like an empty queue.
        """
        backlog = self._cursor.list_convergence_debt(
            limit=RAW_RETENTION_BACKLOG_PER_PASS,
            stage=RAW_RETENTION_STAGE,
            retry_due_only=True,
        )
        ordered: list[Path] = []
        for debt in backlog:
            if debt.subject_type != "source_path":
                continue
            path = Path(debt.subject_id)
            if path in exclude or path in ordered:
                continue
            ordered.append(path)
        return ordered

    def _record_raw_retention_outcome(
        self,
        paths: Sequence[Path],
        *,
        residual: set[Path],
        error: str | None,
        deferred: bool,
    ) -> None:
        """Retain unfinished retention work as ordinary retryable debt.

        One ``ops.db`` connection covers the whole pass: this loop runs once
        per compaction pass over every path the pass scoped, and opening,
        checkpointing and closing a connection per path made the bookkeeping
        cost more than the compaction it records. Each path still commits its
        own row, in the same order, so an interrupted pass leaves exactly the
        prefix it left before.
        """
        with self._cursor.ops_write_scope():
            for path in paths:
                if path in residual:
                    self._cursor.record_convergence_debt(
                        stage=RAW_RETENTION_STAGE,
                        subject_type="source_path",
                        subject_id=str(path),
                        error=error,
                        deferred=deferred,
                    )
                else:
                    self._cursor.clear_convergence_debt(
                        stage=RAW_RETENTION_STAGE,
                        subject_type="source_path",
                        subject_id=str(path),
                    )

    def _compact_superseded_raw_snapshots(self, paths: list[Path]) -> None:
        if _source_tier_acquisition_required():
            return
        from polylogue.storage.sqlite.write_lease import require_write_lease

        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        # The watcher invokes this through ``_run_sync`` and its shared
        # DaemonWriteCoordinator. Keep the door guarded as well so a direct
        # or test-double invocation cannot bypass process-wide enforcement.
        require_write_lease("live raw snapshot compaction", archive_root=archive_root)
        from polylogue.storage.index_generation import ActiveWriterLease
        from polylogue.storage.raw_retention import (
            RawRetentionSafetyError,
            active_raw_retention_authority,
            compact_paths_superseded_raw_snapshots,
        )

        source_db = archive_root / "source.db"
        if not source_db.exists():
            return
        # This pass's own subjects plus the retry-due backlog a previous
        # bounded pass named. Both populations go through the one authority
        # resolution below, so draining the backlog costs no extra hold.
        scoped_paths = list(dict.fromkeys(paths))
        backlog_paths = self._raw_retention_backlog_paths(exclude=set())
        scoped_paths = list(dict.fromkeys((*scoped_paths, *backlog_paths)))
        if not scoped_paths:
            return
        from polylogue.sources.live.cold_build import active_cold_build_generation

        if active_cold_build_generation(archive_root) is not None:
            # The candidate has these index rows, but active-index authority
            # still points at the old generation. Keep the exact paths owed
            # and retry them on a later watcher pass after promotion.
            self._record_raw_retention_outcome(
                scoped_paths,
                residual=set(scoped_paths),
                error="raw retention deferred until inactive index generation is promoted",
                deferred=True,
            )
            return
        lease = ActiveWriterLease(archive_root)
        lease.acquire()
        try:
            index_db = ArchiveLocation.resolve(archive_root).active_index_path
            with (
                closing(open_source_tier_write_connection(source_db, archive_root=archive_root)) as conn,
                closing(open_readonly_connection(index_db)) as index_conn,
                conn,
            ):
                conn.row_factory = sqlite3.Row
                try:
                    retention_authority = active_raw_retention_authority(
                        conn,
                        index_db_path=index_db,
                        terminal_source_paths=scoped_paths,
                        authority_source_paths=scoped_paths,
                    )
                except RawRetentionSafetyError as exc:
                    # A refusal used to be one log line and nothing else: the
                    # pass returned, the paths were never revisited, and the
                    # archive kept superseded payloads with no record that
                    # anything owed them. Record it as ordinary failed debt so
                    # the next due pass retries it with the shared backoff.
                    logger.warning("live.watcher: skipped unsafe raw snapshot compaction: %s", exc)
                    self._record_raw_retention_outcome(
                        scoped_paths,
                        residual=set(scoped_paths),
                        error=f"raw retention refused: {exc}",
                        deferred=False,
                    )
                    return
                backlog_set = set(backlog_paths)
                current_paths = [path for path in scoped_paths if path not in backlog_set]
                results = [
                    compact_paths_superseded_raw_snapshots(
                        conn,
                        selected_paths,
                        limit_per_path=RAW_RETENTION_LIMIT_PER_PATH,
                        # Only recorded backlog may predate this watcher.
                        min_acquired_at=min_acquired_at,
                        protected_raw_ids=retention_authority.protected_raw_ids,
                        eligible_raw_ids=retention_authority.eligible_raw_ids,
                        index_conn=index_conn,
                    )
                    for selected_paths, min_acquired_at in (
                        (current_paths, self._raw_compaction_min_acquired_at),
                        (backlog_paths, None),
                    )
                    if selected_paths
                ]
        finally:
            lease.close()
        errors = tuple(error for result in results for error in result.errors)
        if errors:
            # Blob-unlink errors only. Their subjects are already unreferenced,
            # so the ordinary blob-GC owner collects them; routing them into
            # retention debt would file work under the wrong owner and leave a
            # row that no retention pass can ever clear.
            emit(
                "live.watcher.raw_retention.blob_unlink_failed",
                level=WARNING,
                outcome="degraded",
                error_count=len(errors),
                error_detail="; ".join(errors[:3]),
            )
        # A bound that truncates silently reports a finished answer it did not
        # compute. Name the bound in the debt row instead.
        self._record_raw_retention_outcome(
            scoped_paths,
            residual={Path(path) for result in results for path in result.residual_source_paths},
            error=(
                f"raw retention bounded at {RAW_RETENTION_LIMIT_PER_PATH} superseded snapshots "
                "per source path per pass; backlog retained"
            ),
            deferred=True,
        )

    def _record_append_cursor(self, plan: _AppendPlan) -> bool:
        """Persist a proven append frontier against one stable observation."""
        latest_stat: os.stat_result | None = None
        proof_start: os.stat_result | None = None
        proof_end: os.stat_result | None = None
        tail_hash: str | None = None
        stored_tail_hash: str | None = None
        publication_end = plan.last_complete_newline
        self._last_append_cursor_proof_bytes = 0
        is_claude_frontier = frontier_kind_for_origin(
            origin_from_provider(Provider.from_string(plan.source_name))
        ) == "claude-header-body" and not path_declaration_refuses_session(
            Provider.from_string(plan.source_name), plan.path
        )
        disappeared_after_admission = False
        try:
            if plan.parser_fingerprint is not None and plan.parser_fingerprint != self._current_parser_fingerprint():
                raise ValueError("parser semantics changed during append admission")
            proof_start = plan.path.stat()
            latest_stat = proof_start
            if (
                proof_start.st_dev != plan.st_dev
                or proof_start.st_ino != plan.st_ino
                or (not is_claude_frontier and proof_start.st_size < plan.last_complete_newline)
            ):
                raise ValueError("source replaced or truncated")
            if is_claude_frontier:
                with plan.path.open("rb") as handle:
                    current_header = handle.readline()
                self._last_append_cursor_proof_bytes += len(current_header)
                if not current_header.endswith(b"\n"):
                    raise ValueError("Claude header is incomplete")
                current_append_start = len(current_header) + (plan.accepted_claude_body_bytes or 0)
                publication_end = current_append_start + (plan.last_complete_newline - plan.start_offset)
                if proof_start.st_size < publication_end:
                    raise ValueError("Claude source truncated")
                current_payload_hash, payload_bytes = sha256_range_from_path(
                    plan.path,
                    start_offset=current_append_start,
                    end_offset=publication_end,
                )
                self._last_append_cursor_proof_bytes += payload_bytes
                if current_payload_hash != plan.payload_hash:
                    raise ValueError("accepted Claude append bytes changed")
                stored_tail_hash, frontier_bytes = claude_semantic_frontier_for_prefix_with_bytes(
                    plan.path,
                    publication_end,
                    expected_stable_body_sha256=plan.accepted_claude_body_sha256,
                    expected_stable_body_bytes=plan.accepted_claude_body_bytes,
                )
                self._last_append_cursor_proof_bytes += frontier_bytes
                if stored_tail_hash is None:
                    raise ValueError("accepted Claude semantic body changed")
            else:
                publication_end = plan.last_complete_newline
                payload_hash, payload_bytes = sha256_range_from_path(
                    plan.path,
                    start_offset=plan.start_offset,
                    end_offset=publication_end,
                )
                self._last_append_cursor_proof_bytes += payload_bytes
                tail_hash, tail_bytes = tail_hash_from_path(plan.path, publication_end)
                self._last_append_cursor_proof_bytes += tail_bytes
                if payload_hash != plan.payload_hash:
                    raise ValueError("accepted append bytes changed")
                if plan.accepted_tail_hash is not None and tail_hash != plan.accepted_tail_hash:
                    raise ValueError("accepted append tail changed")
                if plan.accepted_prefix_hash is not None:
                    prefix_hash, prefix_bytes = sha256_range_from_path(
                        plan.path,
                        start_offset=0,
                        end_offset=publication_end,
                    )
                    self._last_append_cursor_proof_bytes += prefix_bytes
                    if prefix_hash != plan.accepted_prefix_hash:
                        raise ValueError("accepted prefix changed")
                stored_tail_hash = (
                    encode_cursor_hash_authority(
                        plan.accepted_prefix_hash,
                        tail_hash,
                        ctime_ns=plan.ctime_ns or 0,
                    )
                    if plan.accepted_prefix_hash is not None
                    else tail_hash
                )
            proof_end = plan.path.stat()
            latest_stat = proof_end
            if _file_observation(proof_end) != _file_observation(proof_start):
                raise ValueError("source changed during cursor verification")
        except FileNotFoundError:
            if proof_start is not None:
                self._invalidate_cursor_for_full_retry(
                    plan.path,
                    source_name=plan.source_name,
                    stat=latest_stat,
                    authority=CursorPathAuthority(plan.canonical_source_path, plan.captured_profile_key),
                )
                return False
            disappeared_after_admission = True
        except (EOFError, OSError, ValueError) as exc:
            logger.warning(
                "live.watcher: source changed after append persistence; cursor invalidated for full retry: %s: %s",
                plan.path,
                exc,
            )
            self._invalidate_cursor_for_full_retry(
                plan.path,
                source_name=plan.source_name,
                stat=latest_stat,
                authority=CursorPathAuthority(plan.canonical_source_path, plan.captured_profile_key),
                captured_file_observation=(
                    plan.st_dev,
                    plan.st_ino,
                    plan.stat_size,
                    plan.mtime_ns,
                    plan.ctime_ns or 0,
                ),
            )
            return False
        if disappeared_after_admission:
            logger.info("live.watcher: source disappeared after append persistence: %s", plan.path)
            publication_end = plan.last_complete_newline
            if is_claude_frontier:
                if (
                    plan.accepted_claude_header_sha256 is None
                    or plan.accepted_claude_publication_body_sha256 is None
                    or plan.accepted_claude_body_bytes is None
                ):
                    self._invalidate_cursor_for_full_retry(
                        plan.path,
                        source_name=plan.source_name,
                        stat=None,
                        authority=CursorPathAuthority(plan.canonical_source_path, plan.captured_profile_key),
                    )
                    return False
                stored_tail_hash = encode_claude_semantic_frontier_digests(
                    header_sha256=plan.accepted_claude_header_sha256,
                    body_sha256=plan.accepted_claude_publication_body_sha256,
                    body_bytes=plan.accepted_claude_body_bytes + (plan.last_complete_newline - plan.start_offset),
                )
            else:
                tail_hash = plan.accepted_tail_hash or plan.payload_hash
                stored_tail_hash = (
                    encode_cursor_hash_authority(
                        plan.accepted_prefix_hash,
                        tail_hash,
                        ctime_ns=plan.ctime_ns or 0,
                    )
                    if plan.accepted_prefix_hash is not None
                    else tail_hash
                )
            cursor_stat_size = plan.stat_size
            cursor_st_dev = plan.st_dev
            cursor_st_ino = plan.st_ino
            cursor_mtime_ns = plan.mtime_ns
        else:
            assert proof_end is not None
            # Only the plan inspected the trailing bytes. The proof checks
            # the published prefix, not any new complete records beyond it.
            # Rebase the observed size for Claude's mutable header, without
            # blessing concurrent body growth as an already-probed tail.
            planned_size = plan.stat_size + (publication_end - plan.last_complete_newline)
            cursor_stat_size = min(proof_end.st_size, planned_size)
            cursor_st_dev = proof_end.st_dev
            cursor_st_ino = proof_end.st_ino
            cursor_mtime_ns = proof_end.st_mtime_ns if proof_end.st_size == planned_size else plan.mtime_ns
        assert stored_tail_hash is not None
        content_fingerprint = append_source_revision(plan.cursor_fingerprint or "", plan.payload_hash)
        # The plan captured its file's authority at acquisition; the source
        # may have vanished since, so it is never re-observed here.
        authority = CursorPathAuthority(plan.canonical_source_path, plan.captured_profile_key)
        updated = self._cursor.set(
            plan.path,
            cursor_stat_size,
            authority=authority,
            byte_offset=publication_end,
            last_complete_newline=publication_end,
            parser_fingerprint=plan.parser_fingerprint or self._current_parser_fingerprint(),
            content_fingerprint=content_fingerprint,
            tail_hash=stored_tail_hash,
            source_name=plan.source_name,
            st_dev=cursor_st_dev,
            st_ino=cursor_st_ino,
            mtime_ns=cursor_mtime_ns,
        )
        if updated:
            self._cursor.reset_failures(plan.path)
        return updated

    async def _run_convergence_paths(
        self,
        actor: str,
        paths: Iterable[Path],
        *,
        whole_archive: bool = True,
        session_ids: Iterable[str] = (),
    ) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt], list[ConvergenceDebtSettlement]]:
        if self._converger is None:
            return self._converge_paths(paths, whole_archive=whole_archive, session_ids=session_ids)
        if self._convergence_runner is None:
            raise RuntimeError("configured convergence requires its admitted preparation owner")
        return cast(
            tuple[set[Path], float, dict[str, float], list[ConvergenceDebt], list[ConvergenceDebtSettlement]],
            await self._convergence_runner(
                actor, self._converge_paths, paths, whole_archive=whole_archive, session_ids=session_ids
            ),
        )


# fmt: off
__all__ = [
    "AppendCapabilityReceipt",
    "LiveBatchMetrics",
    "LiveBatchProcessor",
    "_FullIngestResult",
    "_FULL_PARSE_PROGRESS_MAX_BYTES",
    "_FULL_PARSE_PROGRESS_MAX_FILES",
    "_MAX_APPEND_PLAN_PAYLOAD_BYTES",
    "_full_parse_progress_groups",
    "append_capability_receipt",
    "fingerprint_file",
    "last_complete_newline_from_tail",
]
# fmt: on
