"""In-process live batch convergence for daemon source ingestion."""

from __future__ import annotations

import asyncio
import contextvars
import os
import sqlite3
import time
import zipfile
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from json import JSONDecodeError as StdlibJSONDecodeError
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.raw_payload.decode import jsonl_session_artifact
from polylogue.archive.revision_authority import (
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.archive.revision_replay import RevisionCandidate, plan_revision_replay
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.config import Source
from polylogue.core.degraded import degraded_reason, is_fully_degraded
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
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_EVIDENCE_KINDS,
    RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS,
    RawFailureEvidenceKind,
)
from polylogue.core.sources import origin_from_provider
from polylogue.logging import get_logger
from polylogue.pipeline.ids import session_revision_projection
from polylogue.pipeline.ingest_outcomes import (
    IngestAttemptDisposition,
    classify_archive_write_exception,
    legacy_unknown_disposition,
    success_disposition,
)
from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
from polylogue.sources.decoder_json import PartialJsonStreamError
from polylogue.sources.decoder_zip import ZipBombError, open_bounded_zip_entry
from polylogue.sources.decoders import JsonlDecodeError, _iter_json_stream, _ZipEntryValidator
from polylogue.sources.dispatch import (
    _detect_provider_from_raw_bytes,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.live.append_ingest import ingest_append_plans, reset_transient_raw_parse_state
from polylogue.sources.live.archive_open import _open_archive_for_live_write
from polylogue.sources.live.batch_observability import (
    record_attempt_progress,
)
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _MAX_APPEND_PLAN_PAYLOAD_BYTES,
    _STREAMING_FULL_INGEST_BYTES,
    _accumulate_stage_timings,
    _append_plan_group_ready,
    _AppendPlan,
    _AppendResult,
    _archive_blob_exists,
    _blob_copy_heartbeat,
    _DeferredAppend,
    _detect_provider_from_path_sample,
    _full_ingest_result_from_summary,
    _full_ingest_worker_count,
    _full_parse_progress_groups,
    _FullIngestHeartbeat,
    _FullIngestResult,
    _jsonl_provider_and_session_artifact,
    _parse_path_as_session_artifact,
    _parse_payload_as_session_artifact,
    _path_size,
    _throttled_phase_heartbeat,
    cursor_prefix_hash,
    cursor_state_after_full_ingest,
    encode_cursor_hash_authority,
    fingerprint_file,
    last_complete_newline_from_tail,
    sha256_range_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.batch_support import (
    _LARGE_FULL_PARSE_PROGRESS_BYTES as _LARGE_FULL_PARSE_PROGRESS_BYTES,
)
from polylogue.sources.live.batch_support import (
    _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES as _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES,
)
from polylogue.sources.live.batch_support import (
    _SMALL_FULL_PARSE_PROGRESS_MAX_FILES as _SMALL_FULL_PARSE_PROGRESS_MAX_FILES,
)
from polylogue.sources.live.convergence_debt import (
    ConvergenceDebt,
    convergence_debt_from_state,
    convergence_debt_from_states,
    debt_by_path,
)
from polylogue.sources.live.convergence_outcome import record_convergence_outcome
from polylogue.sources.live.cursor import CursorRecord, CursorStore
from polylogue.sources.live.dedup import handle_schema_version_mismatch, handle_structural_database_error
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics, LiveFullIngestAggregate
from polylogue.sources.live.parse_prefetch import LiveParseCandidate, LiveParseStage
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers import codex_state, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import (
    _declared_non_session_artifact_classification,
    parse_retained_raw_sessions,
)
from polylogue.sources.source_acquisition_components import (
    _DETECTION_PREFIX_SIZE,
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
)
from polylogue.sources.sqlite_snapshot import (
    codex_state_raw_id,
    hermes_profile_raw_id,
    original_sqlite_source_path,
    snapshot_sqlite_to_blob,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.archive import ActiveByteRevisionChainError
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root as initialize_archive_root,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)


class CursorAuthorityBlockedError(RuntimeError):
    """The canonical raw frontier proof did not authorize live source selection."""


@dataclass(slots=True)
class CursorAuthorityAuthorization:
    """Single-use, exact-use exception to the live cursor authority gate.

    This token is deliberately process-local and context-local.  It is not a
    switch, environment variable, or archive setting.  The reconciliation
    command creates one only after re-proving the selected path and frontier;
    the first gate check consumes it.
    """

    source_path_digest: str
    cursor_byte_offset: int
    accepted_frontier: int
    plan_digest: str
    force_full_ingest: bool = False
    consumed: bool = False


_CURSOR_AUTHORIZATION: contextvars.ContextVar[CursorAuthorityAuthorization | None] = contextvars.ContextVar(
    "polylogue_cursor_authority_authorization",
    default=None,
)


def cursor_authority_path_digest(path: Path) -> str:
    """Digest one resolved source path without retaining its private text."""

    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()


@contextmanager
def scoped_cursor_authority_authorization(
    *,
    source_path_digest: str,
    cursor_byte_offset: int,
    accepted_frontier: int,
    plan_digest: str,
    force_full_ingest: bool = False,
) -> Iterator[None]:
    """Install one exact-use authorization for the normal ingest route."""

    authorization = CursorAuthorityAuthorization(
        source_path_digest=source_path_digest,
        cursor_byte_offset=cursor_byte_offset,
        accepted_frontier=accepted_frontier,
        plan_digest=plan_digest,
        force_full_ingest=force_full_ingest,
    )
    marker = _CURSOR_AUTHORIZATION.set(authorization)
    try:
        yield
    finally:
        _CURSOR_AUTHORIZATION.reset(marker)


# polylogue-0jf4: known ~/.codex live SQLite state filenames, matched by name
# first (cheap, no I/O) before the structural table-shape check in
# ``codex_state.is_in_scope_codex_sqlite_path`` decides whether to acquire.
# Kept in sync with ``sources/parsers/codex_state.py``'s ``CODEX_STATE_FIDELITY``.
_CODEX_STATE_DB_NAMES = frozenset({"state_5.sqlite", "goals_1.sqlite", "memories_1.sqlite"})
_CODEX_OUT_OF_SCOPE_STATE_DB_NAMES = frozenset({"logs_2.sqlite", "codex-dev.db"})


def _file_observation(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hot_capture_prefix_is_proven(
    path: str,
    payload: bytes | None,
    *,
    blob_hash: str,
    blob_size: int,
) -> bool:
    """Prove a rejected JSONL capture is a live prefix, never merely assume it.

    A later source size alone is insufficient because a rewrite can have the
    same pathname.  The retained bytes must still be the exact current prefix
    and the source must have grown beyond them.
    """
    expected_fingerprint = sha256(payload).hexdigest() if payload is not None else blob_hash.lower()
    if len(expected_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in expected_fingerprint
    ):
        return False
    source = Path(path)
    try:
        proof_start = source.stat()
        if proof_start.st_size <= blob_size:
            return False
        fingerprint, _bytes_read = sha256_range_from_path(source, start_offset=0, end_offset=blob_size)
        proof_end = source.stat()
    except (EOFError, OSError):
        return False
    return fingerprint == expected_fingerprint and _file_observation(proof_start) == _file_observation(proof_end)


def _write_codex_thread_state_evidence(
    archive: Any,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    source_path: str,
    acquired_at_ms: int,
) -> None:
    """Attach ``threads``/``thread_spawn_edges`` evidence to EXISTING sessions.

    polylogue-0jf4 acceptance criterion 3: threads.title and
    thread_spawn_edges must reach the archive as typed evidence without ever
    minting a session or session of their own -- the same hook-event
    incident precedent as polylogue-31r1 (standalone hook-event ingestion
    once inflated the archive from 18,391 to 83,286 sessions). Reuses
    ``ArchiveStore.write_hook_event``/``raw_hook_events`` exactly as
    ``sources/hooks.py`` does: a durable, session-scoped evidence row keyed
    by ``session_native_id`` (here the Codex ``thread_id``), joined at read
    time (``ArchiveStore.hook_event_summary_for_session``) rather than
    materialized into ``index.db`` via a full session replace. No schema
    change -- ``raw_hook_events.event_type`` is unconstrained TEXT.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

    for thread in snapshot.threads:
        payload: dict[str, object] = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "cwd": thread.cwd,
            "source": thread.source,
            "model": thread.model,
            "agent_nickname": thread.agent_nickname,
            "agent_role": thread.agent_role,
            "archived": thread.archived,
        }
        encoded = json_dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-title:{thread.thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_title",
                payload=payload,
                observed_at_ms=thread.updated_at_ms or acquired_at_ms,
                native_id=f"{thread.thread_id}:codex_thread_title",
                session_native_id=thread.thread_id,
            ),
        )
    for edge in snapshot.spawn_edges:
        edge_payload: dict[str, object] = {
            "parent_thread_id": edge.parent_thread_id,
            "child_thread_id": edge.child_thread_id,
            "status": edge.status,
        }
        encoded = json_dumps(edge_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-spawn-edge:{edge.parent_thread_id}:{edge.child_thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_spawn_edge",
                payload=edge_payload,
                observed_at_ms=acquired_at_ms,
                native_id=f"{edge.parent_thread_id}:{edge.child_thread_id}:codex_thread_spawn_edge",
                session_native_id=edge.parent_thread_id,
            ),
        )


def _is_json_stream_decode_error(error: BaseException) -> bool:
    return isinstance(error, (StdlibJSONDecodeError, UnicodeDecodeError, PartialJsonStreamError, JsonlDecodeError))


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


def append_capability_receipt(
    *,
    provider: str,
    package_version: str,
    element_kind: str,
    stable_session_identity: bool,
) -> AppendCapabilityReceipt:
    """Resolve append support from the live route's identity contract."""
    if provider not in {"codex", "claude-code"}:
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


def _blob_jsonl_has_session_evidence(
    blob_store: BlobStore,
    blob_hash: str,
    *,
    provider: Provider,
    source_path: str,
) -> bool:
    if Path(source_path).suffix.lower() != ".jsonl":
        return False
    try:
        return jsonl_session_artifact(blob_store.blob_path(blob_hash), provider=provider) is not None
    except (OSError, ValueError):
        return False


def _live_parse_stage_candidates(paths: list[Path], *, fallback_provider: Provider) -> list[LiveParseCandidate]:
    """Select and read eligible files for off-writer-hold pre-parse (polylogue-wf8a).

    Deliberately narrow scope: only plain ``.jsonl`` provider-session files
    below ``_STREAMING_FULL_INGEST_BYTES`` are eligible -- exactly the branch
    at lines ~1377-1425 of ``_ingest_full_paths_sync`` that reads the whole
    payload into memory and later parses it via ``parse_payload``/
    ``parse_stream_payload``. Zip bundles, Hermes state/verification
    databases, browser-capture snapshots, and streaming-threshold files are
    left untouched (never selected here) -- they keep parsing inline exactly
    as before; this is a strict subset, not a rewrite, of the existing
    file-type dispatch. Uses the SAME detection helpers
    (``_jsonl_provider_and_session_artifact``) the writer-held pass uses, so
    provider identity can never diverge between prewarm and the real parse.
    """
    candidates: list[LiveParseCandidate] = []
    for path in paths:
        if path.suffix.lower() != ".jsonl":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
            continue
        provider, parse_as_session = _jsonl_provider_and_session_artifact(path, fallback_provider)
        if not parse_as_session:
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        source_path = str(path)
        candidates.append(
            LiveParseCandidate(
                cache_key=source_path,
                provider=provider,
                payload=payload,
                source_path=source_path,
                fallback_id=path.stem,
                is_stream=is_stream_record_provider(source_path, str(provider)),
            )
        )
    return candidates


def _captured_jsonl_ends_at_record_boundary(
    *,
    source_path: str,
    required: bool,
    payload: bytes | None,
    blob_store: BlobStore,
    blob_hash: str,
    blob_size: int,
) -> bool:
    path = Path(source_path)
    if not required or path.suffix.lower() not in {".jsonl", ".ndjson"}:
        return True
    if blob_size <= 0:
        # A zero-byte capture has zero records -- none complete, none
        # incomplete -- so it is trivially at a record boundary. This is
        # NOT the same condition as a mid-write truncation: a session file
        # the provider has created but not yet written its first line into
        # (a live-watcher race) is empty by construction, not corrupted.
        # Treating it as "incomplete" here misclassified genuinely-empty
        # raws as truncated-boundary parse failures (polylogue raw-failure
        # accounting, 2026-07-29); the correct downstream outcome for an
        # empty payload is the ordinary "produced no sessions" path below,
        # not this one.
        return True
    if payload is not None:
        tail = payload.rsplit(b"\n", 1)[-1]
    else:
        chunks: list[bytes] = []
        remaining = blob_size
        with blob_store.open(blob_hash) as handle:
            while remaining > 0:
                chunk_size = min(64 * 1024, remaining)
                remaining -= chunk_size
                handle.seek(remaining)
                chunk = handle.read(chunk_size)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    chunks.append(chunk[newline + 1 :])
                    break
                chunks.append(chunk)
        tail = b"".join(reversed(chunks))
    if not tail.strip():
        return True
    try:
        json_loads(tail)
    except (UnicodeDecodeError, ValueError):
        return False
    return True


@dataclass(slots=True)
class _ArchiveFullWriteResult:
    raw_ids: dict[str, str] = field(default_factory=dict)
    # Terminal refusals are durably retained and therefore handled by this
    # observation. Keep them separate from accepted raw ids so deferred
    # authority failures remain retryable.
    terminal_raw_ids: dict[str, str] = field(default_factory=dict)
    # A raw whose membership census does not produce an accepted session is
    # still a durably acquired, successfully parsed source observation. The
    # decision can be pending for the materialization conveyor or already
    # resolved as ambiguous/deferred. Neither state is a transient source-file
    # failure: retrying identical bytes burns the live catch-up budget without
    # supplying new authority evidence. Track it separately from ``raw_ids``
    # so the cursor records the observation as complete while the durable raw
    # membership state remains queryable and a later file change reopens it.
    deferred_raw_ids: dict[str, str] = field(default_factory=dict)
    session_ids: list[str] = field(default_factory=list)
    session_count: int = 0
    message_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # The archive can forget on purpose (polylogue-27m): a record whose blob
    # hash is durably excised is a deliberate skip, not a failure -- tracked
    # separately from ordinary parse/write failures so operators can tell
    # the two apart (mirrors ParseResult.excised_skips on the CLI import
    # path in pipeline/services/archive_ingest.py).
    excised_skips: int = 0
    # polylogue-11cg9: raw ids never attempted this pass because the declared
    # wall-clock budget (``max_pass_seconds``) was already exceeded before
    # their turn. Not a failure and not a conveyor hand-off -- the record was
    # never opened at all, so its path must stay out of both ``raw_ids`` and
    # ``deferred_raw_ids`` (the two buckets the caller already treats as
    # "durably observed") and out of the caller's success/failure accounting
    # entirely. It remains ordinary backlog: the next catch-up scan or watch
    # tick re-discovers it via the unchanged cursor, exactly like any other
    # untouched file.
    skipped_raw_ids: set[str] = field(default_factory=set)
    time_budget_exceeded: bool = False


class LiveBatchProcessor:
    """Run the daemon live ingest batch path without filesystem watching."""

    def __init__(
        self,
        polylogue: Polylogue,
        sources: Iterable[Any],
        *,
        cursor: CursorStore,
        parser_fingerprint: str | Callable[[], str],
        converger: object | None = None,
        stop_requested: Callable[[], bool] | None = None,
        event_emitter: LiveBatchEventEmitter | None = None,
        sync_runner: LiveBatchSyncRunner | None = None,
        parse_stage: LiveParseStage | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor
        self._parser_fingerprint = parser_fingerprint
        self._converger = converger
        self._stop_requested = stop_requested or (lambda: False)
        self._event_emitter = event_emitter
        self._sync_runner = sync_runner
        self._last_cursor_write_stale = False
        self._raw_compaction_min_acquired_at = datetime.now(UTC).isoformat()
        # polylogue-wf8a: when set (the watcher always sets one; a caller
        # that wants the unmodified in-hold parse path passes ``None``
        # explicitly, e.g. an equivalence test's baseline run),
        # ``_ingest_full_paths`` pre-parses eligible small JSONL candidates
        # in this stage's bounded thread pool BEFORE
        # ``_ingest_full_paths_sync`` ever requests the writer hold -- see
        # ``polylogue.sources.live.parse_prefetch`` for the full safety
        # argument (identical shape to ``DaemonParseStage``).
        self._parse_stage = parse_stage

    def cursor_authority_block_reason(self) -> str | None:
        """Return the canonical frontier reason that blocks live ingestion.

        Small unit tests may exercise the batch processor before an active
        archive has been bootstrapped. The real watcher cannot write without
        these tiers, so the preflight is intentionally deferred until they
        exist. Once they do, the readiness proof is fail-closed and shared
        with raw convergence, recovery, and reindex.
        """
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        if (
            not (archive_root / "source.db").is_file()
            or not ArchiveLocation.resolve(archive_root).active_index_path.is_file()
        ):
            return None
        from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

        return raw_frontier_source_selection_block_reason(archive_root)

    def _consume_scoped_cursor_authority(self, paths: Iterable[Path]) -> CursorAuthorityAuthorization:
        authorization = _CURSOR_AUTHORIZATION.get()
        if authorization is None:
            raise CursorAuthorityBlockedError("scoped cursor authority authorization is missing")
        if authorization.consumed:
            raise CursorAuthorityBlockedError("scoped cursor authority authorization was already consumed")
        selected_paths = tuple(path.resolve() for path in paths)
        if (
            len(selected_paths) != 1
            or cursor_authority_path_digest(selected_paths[0]) != authorization.source_path_digest
        ):
            raise CursorAuthorityBlockedError("scoped cursor authority authorization does not match the selected path")
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        from polylogue.readiness.capability import raw_frontier_integrity_projection
        from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot

        projection = raw_frontier_integrity_projection(
            archive_root,
            raw_materialization_readiness_snapshot(archive_root),
            sample_limit=100,
        )
        if (
            projection.overall_status != "violated"
            or projection.broken_head_count
            or projection.missing_source_raw_count
            or projection.cursor_ahead_count != 1
            or len(projection.cursor_ahead_samples) != 1
        ):
            raise CursorAuthorityBlockedError("scoped cursor authority no longer matches the global violation set")
        sample = projection.cursor_ahead_samples[0]
        if (
            cursor_authority_path_digest(Path(sample.source_path)) != authorization.source_path_digest
            or sample.cursor_byte_offset != authorization.cursor_byte_offset
            or sample.accepted_frontier != authorization.accepted_frontier
        ):
            raise CursorAuthorityBlockedError("scoped cursor authority frontier binding changed")
        authorization.consumed = True
        return authorization

    def require_cursor_authority(self, paths: Iterable[Path] | None = None) -> CursorAuthorityAuthorization | None:
        """Fail closed before a live batch can create attempts or write data."""
        reason = self.cursor_authority_block_reason()
        authorization = _CURSOR_AUTHORIZATION.get()
        if reason is None:
            if authorization is not None:
                raise CursorAuthorityBlockedError("scoped cursor authority authorization has no planned violation")
            return None
        if authorization is not None:
            if paths is None:
                raise CursorAuthorityBlockedError("scoped cursor authority requires an exact selected path")
            return self._consume_scoped_cursor_authority(paths)
        raise CursorAuthorityBlockedError(f"live watcher source-selection gate blocked: {reason}")

    async def ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        emit_event: bool = True,
        max_pass_seconds: float | None = None,
    ) -> LiveBatchMetrics:
        """Ingest files in batch, run post-ingest convergence, and return metrics."""
        authorization = self.require_cursor_authority(paths)
        if is_fully_degraded():
            # The daemon has been marked structurally unable to ingest (e.g.
            # schema mismatch detected at preflight or on the first batch).
            # Do not enter the full-parse path — that is what produced the
            # IOPS storm in #1003.
            return self._degraded_skip_metrics(paths, queued_file_count, skipped_file_count)
        batch_started = time.perf_counter()
        # polylogue-11cg9: a dedicated monotonic reference for the
        # max_pass_seconds budget, separate from ``batch_started`` (used for
        # the unrelated ``total_time_s`` instrumentation) -- matches the
        # ``pass_started_monotonic`` convention already used by de2a's
        # raw-materialization checkpoint and qlae's drive-catchup checkpoint.
        pass_started_monotonic = time.monotonic()
        db_bytes_before = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        input_bytes = sum(_path_size(path) for path in paths)
        attempt_id = self._cursor.begin_ingest_attempt(
            paths=paths,
            input_bytes=input_bytes,
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
        )
        self._record_attempt_progress(
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
            stage_payload=(
                {
                    "cursor_authority_plan_digest": authorization.plan_digest,
                    "cursor_authority_path_digest": authorization.source_path_digest,
                }
                if authorization is not None
                else None
            ),
        )
        source_payload_read_bytes = 0
        cursor_fingerprint_read_bytes = 0
        stale_cursor_write_count = 0
        parse_time_s = 0.0
        convergence_time_s = 0.0
        stage_timings: dict[str, float] = {}
        failed_paths: list[str] = []
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
            if not pending_append_plans:
                return
            plans = pending_append_plans
            pending_append_plans = []
            self._record_attempt_progress(
                attempt_id,
                phase="append_parse",
                succeeded_file_count=len(succeeded_paths),
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
                append_result = await self._run_sync(
                    "watcher.live_ingest.append",
                    self._ingest_append_plans,
                    plans,
                )
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
            self._record_attempt_progress(
                attempt_id,
                phase="convergence",
                succeeded_file_count=len(succeeded_paths),
                failed_file_count=len(failed_paths),
                source_payload_read_bytes=source_payload_read_bytes,
                cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                parse_time_s=parse_time_s,
                convergence_time_s=convergence_time_s,
                current_source=plans[0].source_name,
                current_path=plans[0].path,
                stage_payload=append_stage_payload,
            )
            _converged_paths, elapsed, timings, convergence_debt = await self._run_sync(
                "watcher.live_ingest.append_convergence",
                self._converge_paths,
                [plan.path for plan in append_result.succeeded],
            )
            convergence_time_s += elapsed
            release_process_memory()
            _accumulate_stage_timings(stage_timings, timings)
            debt_by_source_path = debt_by_path(convergence_debt)
            for plan in append_result.succeeded:
                succeeded_paths.add(plan.path)
                if not self._record_append_cursor(plan):
                    stale_cursor_write_count += 1
                self._record_convergence_outcome(plan.path, debt_by_source_path.get(plan.path, ()))
                session_id = append_result.session_ids_by_path.get(plan.path)
                if session_id:
                    updated_session_touches.append((plan.source_name, session_id))
            for plan in append_result.failed:
                failed_paths.append(str(plan.path))
                cursor_fingerprint_read_bytes += self._record_failed_cursor(plan.path)
            for plan in append_result.deferred:
                # polylogue-hat0: this plan's bytes are already durably
                # written and revision-bound in source.db (write_raw_payload
                # ran before the quarantined/ambiguous classification), just
                # not yet accepted into the replay chain. Mark exactly this
                # range as the pending-authority frontier so the next
                # observation of an unchanged file recognizes there is
                # nothing new to capture instead of re-mining an identical
                # duplicate raw row forever.
                cursor_fingerprint_read_bytes += record_deferred_append_cursor(
                    self._cursor,
                    plan.path,
                    cursor=self._cursor.get_record(plan.path),
                    parser_fingerprint=self._current_parser_fingerprint(),
                    source_name=plan.source_name,
                    deferred_end_offset=plan.last_complete_newline,
                )
                deferred_paths.append(plan.path)

        for path in paths:
            if authorization is not None and authorization.force_full_ingest:
                full_paths.append(path)
                continue
            if is_fully_degraded():
                full_paths.append(path)
                continue
            cursor = cursor_records.get(path)
            append_plan = self._append_plan(path, cursor=cursor) if self._can_ingest_appends_directly() else None
            if isinstance(append_plan, _DeferredAppend):
                # No new authority-relevant append happened this pass (no
                # complete trailing newline yet, or -- polylogue-hat0 --
                # _append_plan itself recognized an already-pending deferred
                # range with no growth past it). Preserve any existing
                # pending-authority marker unchanged rather than clearing it.
                cursor_fingerprint_read_bytes += record_deferred_append_cursor(
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
            for source_paths in _full_parse_progress_groups(grouped_paths):
                if self._stop_requested():
                    break
                if is_fully_degraded():
                    break
                if (
                    processed_any_full_group
                    and max_pass_seconds is not None
                    and (time.monotonic() - pass_started_monotonic) > max_pass_seconds
                ):
                    full_ingest_time_budget_exceeded = True
                    break
                t0 = time.perf_counter()
                try:
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse",
                        succeeded_file_count=len(succeeded_paths),
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
                            succeeded_file_count=len(succeeded_paths),
                            failed_file_count=len(failed_paths),
                            source_payload_read_bytes=source_payload_read_bytes,
                            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                            parse_time_s=parse_time_s,
                            convergence_time_s=convergence_time_s,
                        ),
                    )
                    processed_any_full_group = True
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
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
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
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
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
                    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
                        # Archive contention is infrastructure state, not a
                        # poison payload. Let LiveWatcher requeue the source
                        # group without advancing or excluding its cursors.
                        raise
                    logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                    attempt_disposition = classify_archive_write_exception(exc)
                    for path in source_paths:
                        failed_paths.append(str(path))
                        cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                    self._record_attempt_progress(
                        attempt_id,
                        phase="full_parse_failed",
                        succeeded_file_count=len(succeeded_paths),
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
                self._record_attempt_progress(
                    attempt_id,
                    phase="convergence",
                    succeeded_file_count=len(succeeded_paths),
                    failed_file_count=len(failed_paths),
                    source_payload_read_bytes=source_payload_read_bytes,
                    cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
                    parse_time_s=parse_time_s,
                    convergence_time_s=convergence_time_s,
                    current_source=source_name,
                    current_path=source_paths[0] if source_paths else None,
                )
                convergence_debt: list[ConvergenceDebt] = []
                if full_result.changed_session_count:
                    _converged_paths, elapsed, timings, convergence_debt = await self._run_sync(
                        "watcher.live_ingest.full_convergence",
                        self._converge_paths,
                        full_result.succeeded,
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
                for path in full_result.succeeded:
                    succeeded_paths.add(path)
                    cursor_fingerprint_read_bytes += self._record_full_cursor(
                        path,
                        raw_fingerprint=full_result.raw_fingerprints.get(path),
                        raw_byte_size=full_result.raw_byte_sizes.get(path),
                        source_name=full_result.raw_source_names.get(path),
                        source_revision=full_result.raw_source_revisions.get(path),
                        captured_content_hash=full_result.captured_content_hashes.get(path),
                        captured_file_observation=full_result.captured_file_observations.get(path),
                    )
                    if self._last_cursor_write_stale:
                        stale_cursor_write_count += 1
                    self._record_convergence_outcome(path, debt_by_source_path.get(path, ()))
                for path in full_result.failed:
                    failed_paths.append(str(path))
                    cursor_fingerprint_read_bytes += self._record_failed_cursor(path)
                logger.info(
                    "live.watcher: batch ingested %s — %d in %.1fs (%.1f/s)",
                    source_name,
                    len(full_result.succeeded),
                    parse_elapsed,
                    len(full_result.succeeded) / max(parse_elapsed, 0.01),
                )

        summary_stage_payload = _single_route_stage_payload(
            append_file_count=append_file_count,
            full_file_count=len(full_paths),
        )
        self._record_attempt_progress(
            attempt_id,
            phase="cursor_update",
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths) + len(deferred_paths),
            source_payload_read_bytes=source_payload_read_bytes,
            cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
            parse_time_s=parse_time_s,
            convergence_time_s=convergence_time_s,
            stale_cursor_write_count=stale_cursor_write_count,
            stage_payload=summary_stage_payload,
        )

        if succeeded_paths:
            await self._run_sync(
                "watcher.live_ingest.raw_compaction",
                self._compact_superseded_raw_snapshots,
                sorted(succeeded_paths),
            )

        retry_paths = failed_paths + [str(path) for path in deferred_paths]
        db_bytes_after = _path_size(self._cursor._db_path) + _path_size(self._cursor._db_path.with_suffix(".db-wal"))
        metrics = LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count,
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths),
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=input_bytes,
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
            stage_timings_s={name: round(elapsed, 6) for name, elapsed in stage_timings.items()},
            failed_paths=retry_paths,
            new_sessions=tuple(new_session_touches),
            updated_sessions=tuple(updated_session_touches),
            time_budget_exceeded=full_ingest_time_budget_exceeded,
        )
        if emit_event and self._event_emitter is not None:
            self._event_emitter("ingestion_batch", metrics.to_payload())
        self._record_attempt_progress(
            attempt_id,
            phase="completed",
            status="completed",
            queued_file_count=metrics.queued_file_count,
            needed_file_count=metrics.needed_file_count,
            skipped_file_count=metrics.skipped_file_count,
            succeeded_file_count=len(succeeded_paths),
            failed_file_count=len(failed_paths) + len(deferred_paths),
            input_bytes=input_bytes,
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
        elif not retry_paths:
            final_disposition = success_disposition()
        else:
            # Per-record failures (validation/corrupt-input/unsupported-shape)
            # are already classified where they occur -- see
            # ``ingest_worker.py``'s ``IngestRecordResult.outcome_code`` --
            # but aggregating them up to this attempt-level row is deferred
            # follow-up work (polylogue-cnu3 PR body). Reporting SUCCESS here
            # would be dishonest given ``retry_paths`` is non-empty, so this
            # falls back to the explicit "not yet classified" bucket rather
            # than guessing.
            final_disposition = legacy_unknown_disposition(
                diagnostic=f"{len(retry_paths)} path(s) failed without a batch-level exception"
            )
        self._cursor.finish_ingest_attempt(
            attempt_id,
            status="completed" if not retry_paths else "completed_with_failures",
            phase="completed",
            error="; ".join(retry_paths[:3]) if retry_paths else None,
            disposition=final_disposition,
        )
        return metrics

    def _degraded_skip_metrics(
        self,
        paths: list[Path],
        queued_file_count: int | None,
        skipped_file_count: int,
    ) -> LiveBatchMetrics:
        """Empty-ingest metrics for the degraded short-circuit path."""
        return LiveBatchMetrics(
            queued_file_count=queued_file_count if queued_file_count is not None else len(paths),
            needed_file_count=len(paths),
            skipped_file_count=skipped_file_count + len(paths),
            succeeded_file_count=0,
            failed_file_count=0,
            source_group_count=len({self._source_name_for(path) for path in paths}),
            input_bytes=0,
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
        )

    def _record_attempt_progress(self, attempt_id: str, **kwargs: Any) -> None:
        record_attempt_progress(self._cursor, attempt_id, **kwargs)

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

    def _record_failed_cursor(self, path: Path) -> int:
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
            return 0
        try:
            stat = path.stat()
        except FileNotFoundError:
            try:
                self._cursor.mark_failed(path)
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
                    self._cursor.mark_failed(path)
                    return 0
                self._cursor.set(
                    path,
                    stat.st_size,
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
            self._cursor.mark_failed(path, failed_stat=stat)
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc):
                raise
            logger.warning("live.watcher: skipped failed-cursor bookkeeping for %s: %s", path, exc)
        return stat.st_size

    def _record_full_cursor(
        self,
        path: Path,
        *,
        raw_fingerprint: str | None = None,
        raw_byte_size: int | None = None,
        source_name: str | None = None,
        source_revision: str | None = None,
        captured_content_hash: str | None = None,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
    ) -> int:
        self._last_cursor_write_stale = False
        resolved_source_name = source_name or self._source_name_for(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._last_cursor_write_stale = True
            self._invalidate_cursor_for_full_retry(
                path,
                source_name=resolved_source_name,
                captured_file_observation=captured_file_observation,
            )
            return 0
        raw_fingerprint = raw_fingerprint or self._latest_raw_fingerprint(path)
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
        )
        bytes_read = prefix_proof.bytes_read
        if prefix_proof.outcome == "deferred":
            self._last_cursor_write_stale = True
            logger.info(
                "live.watcher: captured prefix remained busy; preserving raw for cursor reconciliation: %s",
                path,
            )
            self._defer_full_cursor_retry(path, source_name=resolved_source_name, stat=stat)
            return bytes_read
        if prefix_proof.outcome != "verified":
            self._last_cursor_write_stale = True
            logger.warning(
                "live.watcher: source changed after full capture; cursor invalidated for full retry: %s",
                path,
            )
            self._invalidate_cursor_for_full_retry(path, source_name=resolved_source_name, stat=stat)
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
            last_nl = byte_size
            tail_hash = source_revision
            if captured_content_hash is not None:
                bounded_tail_hash, tail_bytes = tail_hash_from_path(path, byte_size)
                tail_hash = encode_cursor_hash_authority(
                    captured_content_hash,
                    bounded_tail_hash,
                    ctime_ns=stat.st_ctime_ns,
                )
                bytes_read += tail_bytes
        else:
            fp, last_nl, tail_hash, cursor_state_bytes = cursor_state_after_full_ingest(
                path,
                byte_size,
                raw_fingerprint=raw_fingerprint,
            )
            prefix_hash, prefix_bytes = sha256_range_from_path(
                path,
                start_offset=0,
                end_offset=last_nl,
            )
            tail_hash = encode_cursor_hash_authority(prefix_hash, tail_hash, ctime_ns=stat.st_ctime_ns)
            bytes_read += cursor_state_bytes + prefix_bytes
        final_prefix_proof = self._full_capture_still_matches(
            path,
            stat=stat,
            byte_size=byte_size,
            captured_content_hash=captured_content_hash,
            captured_file_observation=captured_file_observation,
        )
        bytes_read += final_prefix_proof.bytes_read
        if final_prefix_proof.outcome == "deferred":
            self._last_cursor_write_stale = True
            logger.info(
                "live.watcher: captured prefix remained busy after cursor proof; preserving raw for reconciliation: %s",
                path,
            )
            self._defer_full_cursor_retry(path, source_name=resolved_source_name, stat=stat)
            return bytes_read
        if final_prefix_proof.outcome != "verified":
            self._last_cursor_write_stale = True
            self._invalidate_cursor_for_full_retry(
                path,
                source_name=resolved_source_name,
                stat=final_prefix_proof.stat,
                captured_file_observation=captured_file_observation,
            )
            return bytes_read
        assert final_prefix_proof.stat is not None
        final_stat = final_prefix_proof.stat
        updated = self._cursor.set(
            path,
            byte_size,
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

    def _defer_full_cursor_retry(self, path: Path, *, source_name: str, stat: os.stat_result) -> None:
        """Back off a busy full-prefix handoff without discarding its raw evidence."""

        self._invalidate_cursor_for_full_retry(path, source_name=source_name, stat=stat)
        self._cursor.defer_full_cursor_reconciliation(path)

    def _invalidate_cursor_for_full_retry(
        self,
        path: Path,
        *,
        source_name: str,
        stat: os.stat_result | None = None,
        captured_file_observation: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        existing = self._cursor.get_record(path)
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
            excluded=bool(existing.excluded) if existing is not None else False,
            allow_backward=True,
        )
        if not updated:
            raise sqlite3.OperationalError(f"failed to persist cursor invalidation for {path}")

    def _record_convergence_outcome(self, path: Path, debts: Iterable[ConvergenceDebt]) -> None:
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        record_convergence_outcome(self._cursor, path, debts, archive_root=archive_root)

    def _converge_paths(
        self, paths: Iterable[Path]
    ) -> tuple[set[Path], float, dict[str, float], list[ConvergenceDebt]]:
        unique_paths = tuple(sorted(dict.fromkeys(paths)))
        if not unique_paths:
            return set(), 0.0, {}, []
        if self._converger is None:
            return set(unique_paths), 0.0, {}, []

        started = time.perf_counter()
        try:
            converge_batch = getattr(self._converger, "converge_batch", None)
            if callable(converge_batch):
                states, timings = converge_batch(unique_paths)
                batch_completed = {
                    path for path in unique_paths if path in states and bool(getattr(states[path], "converged", False))
                }
                debt_items = convergence_debt_from_states(unique_paths, states)
                # #1654: after convergence, check for new hook events
                # that carry paste evidence and update matching messages.
                try:
                    from polylogue.sources.live.hook_paste_enrichment import enrich_paste_from_hooks

                    enrich_paste_from_hooks(self._cursor._db_path)
                except Exception:
                    logger.debug("hook_paste: enrichment failed (non-fatal)", exc_info=True)
                return (
                    batch_completed,
                    time.perf_counter() - started,
                    {stage_name: float(elapsed) for stage_name, elapsed in timings.items()},
                    debt_items,
                )

            per_file_completed: set[Path] = set()
            stage_timings: dict[str, float] = {}
            per_file_debt_items: list[ConvergenceDebt] = []
            for path in unique_paths:
                invalidate = getattr(self._converger, "invalidate_file", None)
                if callable(invalidate):
                    invalidate(path)
                state = self._converger.converge_file(path)  # type: ignore[attr-defined]
                for stage_name, elapsed in getattr(state, "last_stage_times", {}).items():
                    stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + float(elapsed)
                if bool(getattr(state, "converged", False)):
                    per_file_completed.add(path)
                else:
                    per_file_debt_items.extend(convergence_debt_from_state(path, state))
            return per_file_completed, time.perf_counter() - started, stage_timings, per_file_debt_items
        except Exception as exc:
            logger.warning("live.watcher: post-ingest converge failed: %s", exc)
            return (
                set(),
                time.perf_counter() - started,
                {},
                [ConvergenceDebt(path=path, stage="convergence", error=str(exc)) for path in unique_paths],
            )

    def _latest_raw_fingerprint(self, path: Path) -> str | None:
        return self._latest_archive_tiers_raw_fingerprint(path)

    def _archive_source_db_path(self) -> Path:
        return Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent)) / "source.db"

    def _latest_archive_tiers_raw_fingerprint(self, path: Path) -> str | None:
        source_db = self._archive_source_db_path()
        if not source_db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT raw_id, blob_hash
                    FROM raw_sessions
                    WHERE source_path = ?
                      AND COALESCE(source_index, 0) >= 0
                    ORDER BY acquired_at_ms DESC, raw_id DESC
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
        raw_id, blob_hash = row
        if isinstance(blob_hash, bytes):
            blob_hash_hex = blob_hash.hex()
        elif isinstance(blob_hash, str):
            blob_hash_hex = blob_hash.lower()
        else:
            return None
        if not _archive_blob_exists(source_db.parent, blob_hash_hex):
            return None
        return raw_id if isinstance(raw_id, str) and raw_id else None

    def _current_parser_fingerprint(self) -> str:
        if callable(self._parser_fingerprint):
            return self._parser_fingerprint()
        return self._parser_fingerprint

    def _source_name_for(self, path: Path) -> str:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return str(source.name)
            except OSError:
                continue
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
        if self._parse_stage is not None:
            # polylogue-wf8a: pre-parse eligible candidates BEFORE ever
            # asking the write coordinator for the writer hold below --
            # identical sequencing guarantee to ``DaemonParseStage.warm``
            # (see ``polylogue.sources.live.parse_prefetch``). A warm()
            # failure here must never abort ingestion; it only means every
            # candidate falls back to being parsed inline, exactly as if the
            # flag were off.
            fallback_provider = Provider.from_string(
                canonical_acquisition_provider(source_name, source_name=source_name)
            )
            try:
                candidates = await asyncio.to_thread(
                    _live_parse_stage_candidates, paths, fallback_provider=fallback_provider
                )
                if candidates:
                    warmed = await asyncio.to_thread(self._parse_stage.warm, candidates)
                    if warmed:
                        logger.info(
                            "live.watcher: parse-stage prefetch warmed %d of %d file(s) off the writer hold",
                            warmed,
                            len(candidates),
                        )
            except Exception:
                logger.warning(
                    "live.watcher: parse-stage prefetch failed; falling back to in-hold parse",
                    exc_info=True,
                )
        return await self._run_sync(
            "watcher.live_ingest.full",
            self._ingest_full_paths_sync,
            paths,
            source_name=source_name,
            heartbeat=heartbeat,
            attempt_id=attempt_id,
            max_pass_seconds=max_pass_seconds,
            pass_started=pass_started,
        )

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

    def _ingest_full_paths_sync(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: _FullIngestHeartbeat | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _FullIngestResult:
        if not paths:
            return _FullIngestResult(succeeded=[], failed=[], source_payload_read_bytes=0)
        pass_clock_started = pass_started if pass_started is not None else time.monotonic()
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        blob_root = archive_root / "blob"
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        blob_store = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
        raw_records: list[RawSessionRecord] = []
        raw_by_id: dict[str, Path] = {}
        raw_byte_sizes: dict[Path, int] = {}
        raw_payloads: dict[str, bytes] = {}
        parsed_sessions_by_raw_id: dict[str, list[ParsedSession]] = {}
        raw_source_names: dict[Path, str] = {}
        raw_source_revisions: dict[Path, str] = {}
        captured_content_hashes: dict[Path, str] = {}
        captured_file_observations: dict[Path, tuple[int, int, int, int, int]] = {}
        failed: list[Path] = []
        ingested: list[Path] = []
        source_payload_read_bytes = 0
        fallback_provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        acquisition_capture_mode = fallback_provider

        archive_bootstrapped = not self._archive_active(archive_root)
        if archive_bootstrapped:
            initialize_archive_root(archive_root)
        archive_active = self._archive_active(archive_root)
        if heartbeat is not None:
            heartbeat(
                "full_archive_storage_probe",
                current_path=paths[0],
                source_payload_read_bytes=0,
                stage_payload=self._archive_storage_probe_payload(
                    archive_root,
                    archive_active=archive_active,
                    archive_bootstrapped=archive_bootstrapped,
                ),
                force=True,
            )

        for path in paths:
            blob_hash: str | None = None
            blob_publication_receipt_id: str | None = None
            try:
                stat = path.stat()
            except OSError:
                failed.append(path)
                continue
            captured_file_observations[path] = _file_observation(stat)
            origin_artifact_rule = artifact_rule_for_path(fallback_provider, str(path))
            if heartbeat is not None:
                heartbeat(
                    "full_file_scan",
                    current_path=path,
                    source_payload_read_bytes=source_payload_read_bytes,
                )
            if path.suffix.lower() == ".zip":
                file_mtime = datetime.fromtimestamp(stat.st_mtime_ns / 1_000_000_000, UTC).isoformat()
                zip_records, zip_bytes = self._extract_zip_member_records(
                    path,
                    blob_store=blob_store,
                    fallback_provider=fallback_provider,
                    file_mtime=file_mtime,
                )
                if not zip_records:
                    self._mark_excluded_cursor(path, stat, source_name=fallback_provider.value)
                    continue
                for member_raw_id, member_record in zip_records:
                    raw_records.append(member_record)
                    raw_by_id[member_raw_id] = path
                source_payload_read_bytes += zip_bytes
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
                ingested.append(path)
                raw_byte_sizes[path] = stat.st_size
                continue
            if hermes_state.looks_like_state_db_path(
                path
            ) or hermes_verification.looks_like_verification_evidence_db_path(path):
                provider = Provider.HERMES
                source_name = provider.value
                try:
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                    snapshot = snapshot_sqlite_to_blob(
                        path,
                        blob_store,
                        heartbeat=_blob_copy_heartbeat(
                            heartbeat,
                            path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        ),
                    )
                    blob_hash, blob_size = snapshot.blob_hash, snapshot.blob_size
                    blob_publication_receipt_id = snapshot.blob_publication_receipt_id
                    source_path = original_sqlite_source_path(path) or path
                    raw_id = hermes_profile_raw_id(source_path, 0, blob_hash)
                    raw_source_revisions[path] = snapshot.source_revision
                except OSError:
                    failed.append(path)
                    continue
                source_payload_read_bytes += blob_size
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif path.name in _CODEX_STATE_DB_NAMES and codex_state.is_in_scope_codex_sqlite_path(path):
                # polylogue-0jf4: acquire live Codex SQLite state the same
                # way Hermes acquires its state.db -- a consistent
                # backup/snapshot (never a raw read of a possibly-live-locked
                # file) into the content-addressed blob store. The filename
                # gate keeps this cheap for the vast majority of ~/.codex
                # traffic (JSONL rollouts); ``is_in_scope_codex_sqlite_path``
                # then re-confirms the table shape before trusting the name.
                provider = Provider.CODEX
                source_name = provider.value
                try:
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                    snapshot = snapshot_sqlite_to_blob(
                        path,
                        blob_store,
                        heartbeat=_blob_copy_heartbeat(
                            heartbeat,
                            path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        ),
                    )
                    blob_hash, blob_size = snapshot.blob_hash, snapshot.blob_size
                    blob_publication_receipt_id = snapshot.blob_publication_receipt_id
                    source_path = original_sqlite_source_path(path) or path
                    raw_id = codex_state_raw_id(source_path, blob_hash)
                    raw_source_revisions[path] = snapshot.source_revision
                except OSError:
                    failed.append(path)
                    continue
                source_payload_read_bytes += blob_size
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif path.name in _CODEX_STATE_DB_NAMES or path.name in _CODEX_OUT_OF_SCOPE_STATE_DB_NAMES:
                # Matched a known Codex state-db filename but either failed
                # structural verification (mid-write, corrupt, or a future
                # Codex schema change) or is a database CODEX_STATE_FIDELITY
                # (sources/parsers/codex_state.py) declares out-of-scope
                # (logs_2.sqlite's 627 MB of runtime tracing, codex-dev.db's
                # automation config) -- exclude cleanly without ever reading
                # the bytes as a generic session artifact.
                self._mark_excluded_cursor(path, stat, source_name=fallback_provider.value)
                continue
            elif origin_artifact_rule is not None and origin_artifact_rule.parse_policy != "session":
                provider = fallback_provider
                source_name = provider.value
                if stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy",
                                current_path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            )
                        raw_id, blob_size = blob_store.write_from_path(
                            path,
                            heartbeat=_blob_copy_heartbeat(
                                heartbeat,
                                path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            ),
                        )
                        blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    except OSError:
                        failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                else:
                    try:
                        payload = path.read_bytes()
                    except OSError:
                        failed.append(path)
                        continue
                    raw_id, blob_size = blob_store.write_from_bytes(payload)
                    blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    raw_payloads[raw_id] = payload
                    source_payload_read_bytes += len(payload)
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif path.suffix.lower() == ".jsonl":
                provider, parse_as_session = _jsonl_provider_and_session_artifact(path, fallback_provider)
                source_name = provider.value
                if not parse_as_session and provider is not Provider.UNKNOWN:
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                if stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
                    try:
                        if heartbeat is not None:
                            heartbeat(
                                "full_blob_copy",
                                current_path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            )
                        raw_id, blob_size = blob_store.write_from_path(
                            path,
                            heartbeat=_blob_copy_heartbeat(
                                heartbeat,
                                path=path,
                                source_payload_read_bytes=source_payload_read_bytes,
                            ),
                        )
                        blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    except OSError:
                        failed.append(path)
                        continue
                    source_payload_read_bytes += blob_size
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                else:
                    try:
                        payload = path.read_bytes()
                    except OSError:
                        failed.append(path)
                        continue
                    raw_id, blob_size = blob_store.write_from_bytes(payload)
                    blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                    raw_payloads[raw_id] = payload
                    source_payload_read_bytes += len(payload)
                    if self._parse_stage is not None:
                        # polylogue-wf8a: bridge the path-keyed prewarm cache
                        # to the raw_id keyspace the instant it becomes known
                        # -- see polylogue.sources.live.parse_prefetch for
                        # why the cache cannot be keyed on raw_id directly,
                        # and why ``pop`` re-verifies the payload bytes match
                        # exactly (a live-appending file can grow between the
                        # prewarm read and this one).
                        cached_sessions = self._parse_stage.cache.pop(str(path), payload=payload)
                        if cached_sessions is not None:
                            parsed_sessions_by_raw_id[raw_id] = cached_sessions
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
            elif source_name == "browser-capture" and path.suffix.lower() == ".json":
                try:
                    payload = path.read_bytes()
                except OSError:
                    failed.append(path)
                    continue
                provider = _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)
                source_name = provider.value
                if (
                    not _parse_payload_as_session_artifact(path, provider=provider, payload=payload)
                    and provider is not Provider.UNKNOWN
                ):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                raw_id, blob_size = blob_store.write_from_bytes(payload)
                blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                raw_payloads[raw_id] = payload
                source_payload_read_bytes += len(payload)
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            elif stat.st_size >= _STREAMING_FULL_INGEST_BYTES:
                provider = _detect_provider_from_path_sample(path, fallback_provider)
                source_name = provider.value
                if not _parse_path_as_session_artifact(path, provider=provider):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                try:
                    if heartbeat is not None:
                        heartbeat(
                            "full_blob_copy",
                            current_path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        )
                    raw_id, blob_size = blob_store.write_from_path(
                        path,
                        heartbeat=_blob_copy_heartbeat(
                            heartbeat,
                            path=path,
                            source_payload_read_bytes=source_payload_read_bytes,
                        ),
                    )
                    blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                except OSError:
                    failed.append(path)
                    continue
                source_payload_read_bytes += blob_size
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            else:
                try:
                    payload = path.read_bytes()
                except OSError:
                    failed.append(path)
                    continue
                provider = _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)
                source_name = provider.value
                if (
                    not _parse_payload_as_session_artifact(path, provider=provider, payload=payload)
                    and provider is not Provider.UNKNOWN
                ):
                    self._mark_excluded_cursor(path, stat, source_name=source_name)
                    continue
                raw_id, blob_size = blob_store.write_from_bytes(payload)
                blob_publication_receipt_id = blob_store.receipt_id(raw_id)
                raw_payloads[raw_id] = payload
                source_payload_read_bytes += len(payload)
                if heartbeat is not None:
                    heartbeat(
                        "full_blob_copy",
                        current_path=path,
                        source_payload_read_bytes=source_payload_read_bytes,
                    )
            ingested.append(path)
            # The source may grow while its blob is copied. The blob size is
            # the durable acquisition boundary; the pre-copy stat is only a
            # planning observation. SQLite snapshots are logical copies, so
            # their cursor remains tied to the source file stat instead.
            # ``path in raw_source_revisions`` is true exactly for the two
            # sqlite-snapshot acquisition branches (Hermes state/verification
            # dbs and, per polylogue-0jf4, Codex state dbs) -- both mint a
            # deterministic raw_id distinct from the blob's own content hash,
            # unlike every other branch where raw_id IS the content hash.
            acquired_via_sqlite_snapshot = path in raw_source_revisions
            raw_byte_sizes[path] = stat.st_size if acquired_via_sqlite_snapshot else blob_size
            raw_source_names[path] = source_name
            if not acquired_via_sqlite_snapshot:
                captured_content_hashes[path] = raw_id
            raw_records.append(
                RawSessionRecord(
                    raw_id=raw_id,
                    blob_hash=(blob_hash if acquired_via_sqlite_snapshot and blob_hash is not None else None),
                    payload_provider=provider,
                    capture_mode=(
                        acquisition_capture_mode if acquisition_capture_mode is not Provider.UNKNOWN else provider
                    ),
                    source_name=source_name,
                    source_path=(
                        str(original_sqlite_source_path(path) or path) if path in raw_source_revisions else str(path)
                    ),
                    source_index=0,
                    blob_size=blob_size,
                    blob_publication_receipt_id=blob_publication_receipt_id,
                    acquired_at=datetime.now(UTC).isoformat(),
                    file_mtime=datetime.fromtimestamp(stat.st_mtime_ns / 1_000_000_000, UTC).isoformat(),
                    captured_source_revision=raw_source_revisions.get(path, raw_id),
                    requires_complete_record_boundary=path.suffix.lower() in {".jsonl", ".ndjson"},
                )
            )
            raw_source_revisions.setdefault(path, raw_id)
            raw_by_id[raw_id] = path

        summary: _IngestBatchSummary | None = None
        skipped_paths: set[Path] = set()
        time_budget_exceeded = False
        if raw_records:
            blob_store.flush()
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
            archive_write = self._ingest_full_records_archive(
                raw_records,
                raw_payloads,
                blob_store,
                parsed_sessions_by_raw_id,
                max_pass_seconds=max_pass_seconds,
                pass_started=pass_clock_started,
            )
            # skipped_raw_ids (polylogue-11cg9) are records the time budget
            # never let the archive-write loop reach at all -- neither a
            # failure nor a conveyor hand-off, so they must be excluded from
            # ``failed`` the same way ``deferred_raw_ids`` already is, and
            # then also excluded from ``succeeded`` below (unlike
            # deferred_raw_ids, whose raw row IS durably written this pass).
            skipped_paths = {raw_by_id[raw_id] for raw_id in archive_write.skipped_raw_ids if raw_id in raw_by_id}
            # deferred_raw_ids is a conveyor hand-off, not a failure -- only
            # raws in neither map (a real exception was raised) count below.
            failed.extend(
                raw_by_id[raw_id]
                for raw_id in raw_by_id
                if raw_id not in archive_write.raw_ids
                and raw_id not in archive_write.deferred_raw_ids
                and raw_id not in archive_write.terminal_raw_ids
                and raw_id not in archive_write.skipped_raw_ids
            )
            raw_by_id = {
                (
                    archive_write.raw_ids.get(raw_id)
                    or archive_write.deferred_raw_ids.get(raw_id)
                    or archive_write.terminal_raw_ids.get(raw_id)
                    or raw_id
                ): path
                for raw_id, path in raw_by_id.items()
            }
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
                    },
                    force=True,
                )
            summary = _IngestBatchSummary(
                worker_count=1,
                total_convos=archive_write.session_count,
                total_msgs=archive_write.message_count,
                changed_session_ids=archive_write.session_ids,
                stage_timings_s=archive_write.stage_timings_s,
            )
            time_budget_exceeded = archive_write.time_budget_exceeded

        failed_set = set(failed)
        raw_fingerprints = {path: raw_id for raw_id, path in raw_by_id.items()}
        result = _full_ingest_result_from_summary(
            succeeded=[path for path in ingested if path not in failed_set and path not in skipped_paths],
            failed=failed,
            source_payload_read_bytes=source_payload_read_bytes,
            raw_fingerprints=raw_fingerprints,
            raw_byte_sizes=raw_byte_sizes,
            raw_source_names=raw_source_names,
            raw_source_revisions=raw_source_revisions,
            captured_content_hashes=captured_content_hashes,
            captured_file_observations=captured_file_observations,
            summary=summary,
            time_budget_exceeded=time_budget_exceeded,
        )
        raw_records.clear()
        raw_by_id.clear()
        blob_store.discard_pending()
        return result

    def _archive_active(self, archive_root: Path) -> bool:
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
        tier_paths = {
            spec.tier.value: (
                ArchiveLocation.resolve(archive_root).active_index_path
                if spec.tier.value == "index"
                else archive_root / spec.filename
            )
            for spec in ARCHIVE_TIER_SPECS.values()
        }
        present = [tier for tier, path in tier_paths.items() if path.exists()]
        missing = [tier for tier, path in tier_paths.items() if not path.exists()]
        user_versions: dict[str, int | None] = {}
        for tier, path in tier_paths.items():
            if not path.exists():
                user_versions[tier] = None
                continue
            try:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    user_versions[tier] = int(conn.execute("PRAGMA user_version").fetchone()[0])
                finally:
                    conn.close()
            except sqlite3.Error:
                user_versions[tier] = -1
        return {
            "storage_route": "archive_full",
            "storage_tiers": _ARCHIVE_RUNTIME_TIERS,
            "storage_write_tiers": _ARCHIVE_NATIVE_WRITE_TIERS,
            "archive_active": archive_active,
            "archive_bootstrapped": archive_bootstrapped,
            "archive_present_tiers": ",".join(present),
            "archive_missing_tiers": ",".join(missing),
            "archive_tier_user_versions_json": json_dumps(user_versions, sort_keys=True),
        }

    def _ingest_full_records_archive(
        self,
        records: list[RawSessionRecord],
        raw_payloads: dict[str, bytes],
        blob_store: BlobStore,
        parsed_sessions_by_raw_id: dict[str, list[ParsedSession]] | None = None,
        *,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _ArchiveFullWriteResult:

        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        result = _ArchiveFullWriteResult()
        pass_clock_started = pass_started if pass_started is not None else time.monotonic()
        with _open_archive_for_live_write(archive_root) as archive:
            for record_index, record in enumerate(records):
                # polylogue-11cg9: a single logical session write cannot be
                # split mid-transaction (it must remain atomic), so the
                # checkpoint this loop can safely offer is *between* records,
                # not inside one. The first record of every pass always
                # completes regardless of budget (forward-progress
                # guarantee, same shape as de2a's raw-materialization
                # checkpoint and qlae's drive-catchup batch checkpoint) --
                # only records after the first are ever skipped for time.
                if (
                    record_index > 0
                    and max_pass_seconds is not None
                    and (time.monotonic() - pass_clock_started) > max_pass_seconds
                ):
                    for remaining in records[record_index:]:
                        result.skipped_raw_ids.add(remaining.raw_id)
                    result.time_budget_exceeded = True
                    break
                provider: Provider | None = None
                source_raw_id: str | None = None
                acquired_at_ms = 0
                try:
                    record_timings: dict[str, float] = {}
                    t0 = time.perf_counter()
                    provider = record.payload_provider or Provider.from_string(record.source_name)
                    payload = raw_payloads.get(record.raw_id)
                    source_name = Path(record.source_path).name
                    fallback_id = Path(record.source_path).stem
                    blob_hash = record.blob_hash or record.raw_id
                    acquired_at_ms = _iso_to_epoch_ms(record.acquired_at)
                    artifact_classification = _declared_non_session_artifact_classification(
                        provider,
                        record.source_path,
                    )
                    session_evidence = (
                        _blob_jsonl_has_session_evidence(
                            blob_store,
                            blob_hash,
                            provider=provider,
                            source_path=record.source_path,
                        )
                        if payload is None
                        else _parse_payload_as_session_artifact(
                            Path(record.source_path),
                            provider=provider,
                            payload=payload,
                        )
                    )
                    if artifact_classification is not None and not session_evidence:
                        explicit_raw_id = record.raw_id if record.blob_hash is not None else None
                        if payload is None:
                            source_raw_id = archive.admit_raw_artifact_blob_ref(
                                provider=provider,
                                blob_hash_hex=blob_hash,
                                blob_size=record.blob_size,
                                source_path=record.source_path,
                                source_index=record.source_index or 0,
                                acquired_at_ms=acquired_at_ms,
                                raw_id=explicit_raw_id,
                                classification=artifact_classification,
                                blob_publication_receipt_id=record.blob_publication_receipt_id,
                            ).raw_id
                        else:
                            source_raw_id = archive.admit_raw_artifact_payload(
                                provider=provider,
                                payload=payload,
                                source_path=record.source_path,
                                source_index=record.source_index or 0,
                                acquired_at_ms=acquired_at_ms,
                                raw_id=explicit_raw_id,
                                classification=artifact_classification,
                                blob_publication_receipt_id=record.blob_publication_receipt_id,
                            ).raw_id
                        result.raw_ids[record.raw_id] = source_raw_id
                        _accumulate_stage_timings(result.stage_timings_s, record_timings)
                        continue
                    source_write_started = time.perf_counter()
                    if payload is None:
                        source_raw_id = archive.write_raw_blob_ref(
                            provider=provider,
                            capture_mode=record.capture_mode,
                            blob_hash_hex=blob_hash,
                            blob_size=record.blob_size,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            # A populated ``blob_hash`` field marks a
                            # sqlite-snapshot acquisition (Hermes or, per
                            # polylogue-0jf4, Codex state dbs), whose raw_id
                            # is a deterministic profile/path-scoped id
                            # distinct from the blob's own content hash --
                            # every other provider's raw_id already IS the
                            # content hash, so passing it again is a no-op.
                            raw_id=(record.raw_id if record.blob_hash is not None else None),
                            acquired_at_ms=acquired_at_ms,
                            blob_publication_receipt_id=record.blob_publication_receipt_id,
                            post_parse=True,
                        )
                        source_write_name = "full.source_raw_blob_ref_write"
                    else:
                        source_raw_id = archive.write_raw_payload(
                            provider=provider,
                            capture_mode=record.capture_mode,
                            payload=payload,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=acquired_at_ms,
                            blob_publication_receipt_id=record.blob_publication_receipt_id,
                            post_parse=True,
                        )
                        source_write_name = "full.source_raw_write"
                    record_timings[source_write_name] = time.perf_counter() - source_write_started
                    degraded = degraded_reason()
                    if degraded is not None and degraded.derived_only:
                        # polylogue-gbs02: the derived tier (index.db/
                        # embeddings.db) is behind the running code, but
                        # source.db just got this record durably -- stop
                        # here, before any parse/materialize/index work
                        # touches the stale derived tier. Same admit-and-skip
                        # shape as the OriginSpec fact-artifact branch below,
                        # just a different reason for stopping short of parse.
                        result.raw_ids[record.raw_id] = source_raw_id
                        _accumulate_stage_timings(result.stage_timings_s, record_timings)
                        continue
                    if not _captured_jsonl_ends_at_record_boundary(
                        source_path=record.source_path,
                        required=record.requires_complete_record_boundary,
                        payload=payload,
                        blob_store=blob_store,
                        blob_hash=blob_hash,
                        blob_size=record.blob_size,
                    ):
                        if _hot_capture_prefix_is_proven(
                            record.source_path,
                            payload,
                            blob_hash=blob_hash,
                            blob_size=record.blob_size,
                        ):
                            evidence_kind = (
                                RawFailureEvidenceKind.DEFERRED_CLAUDE_CODE_PARTIAL_JSONL
                                if provider is Provider.CLAUDE_CODE
                                else RawFailureEvidenceKind.DEFERRED_HOT_JSONL_CAPTURE
                            )
                            archive.record_raw_failure_evidence(
                                source_raw_id,
                                provider=provider,
                                source_path=record.source_path,
                                source_index=record.source_index or 0,
                                acquired_at_ms=acquired_at_ms,
                                kind=evidence_kind,
                            )
                            archive.mark_raw_parse_failed(
                                source_raw_id,
                                provider=provider,
                                error=ValueError("captured JSONL payload ends before a complete record boundary"),
                                preserve_existing_failure_evidence=True,
                            )
                            result.raw_ids[record.raw_id] = source_raw_id
                            _accumulate_stage_timings(result.stage_timings_s, record_timings)
                            continue
                        archive.record_raw_failure_evidence(
                            source_raw_id,
                            provider=provider,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=acquired_at_ms,
                            kind=RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
                        )
                        archive.mark_raw_parse_failed(
                            source_raw_id,
                            provider=provider,
                            error=ValueError("captured JSONL payload ends before a complete record boundary"),
                            preserve_existing_failure_evidence=True,
                        )
                        result.raw_ids[record.raw_id] = source_raw_id
                        _accumulate_stage_timings(result.stage_timings_s, record_timings)
                        continue
                    t0 = time.perf_counter()
                    cached_sessions = (
                        parsed_sessions_by_raw_id.pop(record.raw_id, None) if parsed_sessions_by_raw_id else None
                    )
                    if cached_sessions is not None:
                        # polylogue-wf8a: this record's decode already ran
                        # off the writer hold (``LiveParseStage.warm``,
                        # re-verified byte-identical to what
                        # ``blob_store.write_from_bytes`` just wrote --
                        # see ``LiveParsePrefetchCache.pop``). Every branch
                        # below is skipped; this is a pure shortcut of the
                        # SAME parse, never a different one.
                        sessions = cached_sessions
                    elif provider is Provider.HERMES and hermes_state.looks_like_state_db_path(
                        blob_store.blob_path(blob_hash), immutable=True
                    ):
                        sessions = hermes_state.parse_state_db(
                            blob_store.blob_path(blob_hash),
                            fallback_id=fallback_id,
                            profile_root=Path(record.source_path).parent,
                            immutable=True,
                        )
                    elif provider is Provider.HERMES and hermes_verification.looks_like_verification_evidence_db_path(
                        blob_store.blob_path(blob_hash), immutable=True
                    ):
                        sessions = hermes_verification.parse_verification_evidence_db(
                            blob_store.blob_path(blob_hash),
                            fallback_id=fallback_id,
                            profile_root=Path(record.source_path).parent,
                            immutable=True,
                        )
                    elif provider is Provider.CODEX and codex_state.is_in_scope_codex_sqlite_path(
                        blob_store.blob_path(blob_hash), immutable=True
                    ):
                        # polylogue-0jf4: Codex state dbs never become
                        # sessions of their own -- thread_state's evidence
                        # (titles, spawn edges) attaches to the EXISTING
                        # codex-session rows it describes via
                        # _write_codex_thread_state_evidence, never a full
                        # session replace. goals_1.sqlite/memories_1.sqlite
                        # are acquire-partial (CODEX_STATE_FIDELITY): the raw
                        # snapshot admitted above is already durable
                        # evidence; no derived parse is wired in this change.
                        state_path = blob_store.blob_path(blob_hash)
                        state_kind = codex_state.classify_codex_sqlite_path(state_path, immutable=True)
                        if state_kind == "thread_state":
                            state_snapshot = codex_state.parse_codex_state_db(state_path, immutable=True)
                            _write_codex_thread_state_evidence(
                                archive,
                                state_snapshot,
                                source_path=record.source_path,
                                acquired_at_ms=acquired_at_ms,
                            )
                        result.raw_ids[record.raw_id] = source_raw_id
                        _accumulate_stage_timings(result.stage_timings_s, record_timings)
                        continue
                    elif is_stream_record_provider(record.source_path, str(provider)):
                        if payload is None:
                            with blob_store.open(blob_hash) as payload_handle:
                                sessions = parse_stream_payload(
                                    provider,
                                    _iter_json_stream(
                                        payload_handle,
                                        source_name,
                                        fail_on_decode_error=provider is Provider.UNKNOWN,
                                    ),
                                    fallback_id,
                                    source_path=record.source_path,
                                )
                        else:
                            sessions = parse_stream_payload(
                                provider,
                                _iter_json_stream(
                                    BytesIO(payload),
                                    source_name,
                                    fail_on_decode_error=provider is Provider.UNKNOWN,
                                ),
                                fallback_id,
                                source_path=record.source_path,
                            )
                    else:
                        if payload is None:
                            with blob_store.open(blob_hash) as payload_handle:
                                payloads = list(
                                    _iter_json_stream(
                                        payload_handle,
                                        source_name,
                                        fail_on_decode_error=provider is Provider.UNKNOWN,
                                    )
                                )
                        else:
                            payloads = list(
                                _iter_json_stream(
                                    BytesIO(payload),
                                    source_name,
                                    fail_on_decode_error=provider is Provider.UNKNOWN,
                                )
                            )
                        sessions = parse_payload(
                            provider,
                            payloads,
                            fallback_id,
                            source_path=record.source_path,
                        )
                    # polylogue-9ykn: a session requires positive
                    # conversational evidence -- a parse that produced only
                    # zero-message sessions is treated exactly like a parse
                    # that produced none: a recorded, bounded
                    # mark_raw_parse_failed outcome below, never a silently
                    # written phantom session.
                    sessions = require_positive_conversational_evidence(
                        sessions, provider=provider, source_path=record.source_path
                    )
                    record_timings["full.provider_parse"] = record_timings.get("full.provider_parse", 0.0) + (
                        time.perf_counter() - t0
                    )
                    if not sessions:
                        archive.record_raw_failure_evidence(
                            source_raw_id,
                            provider=provider,
                            source_path=record.source_path,
                            source_index=record.source_index or 0,
                            acquired_at_ms=acquired_at_ms,
                            kind=(
                                RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION
                                if provider is Provider.UNKNOWN
                                else RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE
                            ),
                        )
                        archive.mark_raw_parse_failed(
                            source_raw_id,
                            provider=provider,
                            error=ValueError(
                                "parsed raw payload produced no sessions with positive conversational evidence"
                            ),
                            preserve_existing_failure_evidence=True,
                        )
                        result.raw_ids[record.raw_id] = source_raw_id
                        _accumulate_stage_timings(result.stage_timings_s, record_timings)
                        continue
                    record_raw_id = source_raw_id
                    record_session_ids: list[str] = []
                    record_session_count = 0
                    record_message_count = 0
                    raw_authority_complete = True
                    if len(sessions) == 1:
                        session = sessions[0]
                        logical_source_key = f"{provider.value}:{session.provider_session_id}"
                        # Receiver artifacts are complete, mutable snapshots of
                        # one browser-visible session. Their stable path does
                        # not make successive serialized JSON payloads a
                        # byte-prefix chain, so use typed membership authority
                        # from the first observation. Replacement snapshots
                        # can then advance only through strict parsed-content
                        # growth while every prior raw blob remains retained.
                        is_browser_capture_snapshot = (
                            self._source_name_for(Path(record.source_path)) == "browser-capture"
                        )
                        if is_browser_capture_snapshot or archive.raw_membership_raw_ids(logical_source_key):
                            archive.replace_raw_membership_census(
                                source_raw_id,
                                sessions,
                                parser_fingerprint=self._current_parser_fingerprint(),
                                censused_at_ms=acquired_at_ms,
                            )
                            (
                                record_session_ids,
                                record_session_count,
                                record_message_count,
                                raw_authority_complete,
                            ) = self._apply_membership_sessions(
                                archive,
                                source_raw_id,
                                sessions,
                                acquired_at_ms=acquired_at_ms,
                                stage_timings_s=record_timings,
                                # Once this logical source is governed by a
                                # membership census, the freshly classified
                                # complete single-session snapshot must join
                                # that same candidate set. Admitting only
                                # browser captures strands ordinary
                                # bundle→single transitions in quarantine.
                                allow_current_complete_raw=True,
                            )
                        else:
                            archive.bind_raw_revision(
                                source_raw_id,
                                RawRevisionEnvelope(
                                    logical_source_key=logical_source_key,
                                    kind=RawRevisionKind.FULL,
                                    source_revision=record.captured_source_revision or source_raw_id,
                                    acquisition_generation=0,
                                    authority=RawRevisionAuthority.QUARANTINED,
                                ),
                            )
                            plan = archive.classify_raw_revision_cohort_for_live_watch(logical_source_key)
                            if plan.accepted_raw_ids:
                                parsed_by_raw_id = self._parse_raw_revision_chain(
                                    archive,
                                    plan,
                                    current_raw_id=source_raw_id,
                                    current_session=session,
                                )
                                session_id, applied_raw_ids = archive.apply_raw_revision_replay(
                                    plan,
                                    parsed_by_raw_id,
                                    acquired_at_ms=acquired_at_ms,
                                    stage_timings_s=record_timings,
                                    stage_timing_prefix="full",
                                    defer_fts=True,
                                )
                                self._cursor.record_convergence_debt(
                                    stage="fts",
                                    subject_type="session_id",
                                    subject_id=session_id,
                                    error="live full ingest deferred FTS to preserve writer availability",
                                    deferred=True,
                                )
                                record_session_ids.append(session_id)
                                record_session_count = 1
                                record_message_count = sum(
                                    len(parsed_by_raw_id[raw_id].messages) for raw_id in applied_raw_ids
                                )
                            else:
                                # polylogue-hm2f (live-path half of the
                                # polylogue-52l2 guard): an empty cohort here
                                # can mean two different things.
                                #
                                # (1) This identity has NO retired sibling
                                # evidence -- a genuine, terminal
                                # rewrite/divergence conflict between raws
                                # that are all still live 'full' rows. There
                                # is nothing this tick can safely resolve;
                                # leave this raw out of both raw_ids and
                                # deferred_raw_ids so the caller's aggregation
                                # counts it as a real failure (fail-closed),
                                # with a log line so the cause is diagnosable.
                                #
                                # (2) This identity DOES have retired sibling
                                # evidence (raw_membership_retired_full_
                                # revision_siblings is non-empty): the 52l2
                                # guard is exactly what emptied this cohort.
                                # Offline backfill reunites retired siblings
                                # with a newly discovered raw via its own
                                # connected-component/membership_candidates
                                # bookkeeping across the whole census pass;
                                # the live incremental path processes one
                                # record per tick with no such durable
                                # bookkeeping, so it re-derives the sibling
                                # set from the same durable marker the guard
                                # itself reads. Fold this raw into membership
                                # governance -- the same call sequence used
                                # above for pre-existing-membership and
                                # multi-session identities -- and pass the
                                # retired siblings in explicitly so the real
                                # content-prefix classifier
                                # (classify_membership_revisions) weighs this
                                # raw against every known sibling instead of
                                # evaluating it alone. If the siblings turn
                                # out to genuinely disagree, that classifier
                                # still refuses to pick a winner (ambiguous
                                # quarantine), so this can only ever recover a
                                # real resolution, never fabricate one.
                                retired_siblings = archive.raw_membership_retired_full_revision_siblings(
                                    logical_source_key
                                )
                                if not retired_siblings:
                                    logger.warning(
                                        "live.watcher: no unique byte-revision candidate accepted for %s "
                                        "(logical_source_key=%s) -- surfacing as failed",
                                        record.source_path,
                                        logical_source_key,
                                    )
                                    continue
                                logger.info(
                                    "live.watcher: reunifying %s with %d retired sibling(s) under membership "
                                    "governance (logical_source_key=%s)",
                                    record.source_path,
                                    len(retired_siblings),
                                    logical_source_key,
                                )
                                (
                                    record_session_ids,
                                    record_session_count,
                                    record_message_count,
                                    raw_authority_complete,
                                ) = self._apply_membership_sessions(
                                    archive,
                                    source_raw_id,
                                    sessions,
                                    acquired_at_ms=acquired_at_ms,
                                    stage_timings_s=record_timings,
                                    allow_current_complete_raw=True,
                                    extra_member_raw_ids=retired_siblings,
                                )
                    else:
                        archive.replace_raw_membership_census(
                            source_raw_id,
                            sessions,
                            parser_fingerprint=self._current_parser_fingerprint(),
                            censused_at_ms=acquired_at_ms,
                        )
                        (
                            record_session_ids,
                            record_session_count,
                            record_message_count,
                            raw_authority_complete,
                        ) = self._apply_membership_sessions(
                            archive,
                            source_raw_id,
                            sessions,
                            acquired_at_ms=acquired_at_ms,
                            stage_timings_s=record_timings,
                            # This raw passed artifact taxonomy and produced a complete
                            # multi-session census; admit only this caller-owned candidate.
                            allow_current_complete_raw=True,
                        )
                    if raw_authority_complete:
                        result.raw_ids[record.raw_id] = record_raw_id
                    elif archive.raw_membership_decision_pending(source_raw_id):
                        # Census recorded, no exception, decision IS NULL -- the
                        # raw-authority protocol's own async classification (the
                        # raw-materialization conveyor) hasn't arbitrated this
                        # raw yet. Not a failure; see deferred_raw_ids.
                        result.deferred_raw_ids[record.raw_id] = record_raw_id
                    else:
                        # A decided ambiguous/deferred classification is
                        # fail-closed for materialization, but it is not a
                        # failed acquisition or parse. Re-running the exact
                        # same source path cannot resolve a content conflict;
                        # it only makes every daemon restart re-read the same
                        # historical bytes. Preserve the durable unresolved
                        # decision and make the live cursor idempotent. Any
                        # changed observation returns through full ingest and
                        # may supply the evidence needed to arbitrate it.
                        result.deferred_raw_ids[record.raw_id] = record_raw_id
                        logger.info(
                            "live.watcher: membership decision remains unresolved for %s "
                            "raw=%s; preserving durable authority debt without retrying unchanged bytes",
                            record.source_path,
                            source_raw_id,
                        )
                    result.session_ids.extend(record_session_ids)
                    result.session_count += record_session_count
                    result.message_count += record_message_count
                    _accumulate_stage_timings(result.stage_timings_s, record_timings)
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
                    logger.info(
                        "live.watcher: skipping durably excised content for %s: %s",
                        record.source_path,
                        exc,
                    )
                except Exception as exc:
                    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
                        if provider is not None and source_raw_id is not None:
                            reset_transient_raw_parse_state(
                                archive,
                                source_raw_id,
                                provider=provider,
                            )
                        raise
                    if provider is not None and source_raw_id is not None:
                        preserve_existing_failure_evidence = False
                        if provider is Provider.UNKNOWN and _is_json_stream_decode_error(exc):
                            archive.record_raw_failure_evidence(
                                source_raw_id,
                                provider=provider,
                                source_path=record.source_path,
                                source_index=record.source_index or 0,
                                acquired_at_ms=acquired_at_ms,
                                kind=RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE,
                            )
                            result.terminal_raw_ids[record.raw_id] = source_raw_id
                            preserve_existing_failure_evidence = True
                        archive.mark_raw_parse_failed(
                            source_raw_id,
                            provider=provider,
                            error=exc,
                            preserve_existing_failure_evidence=preserve_existing_failure_evidence,
                        )
                    logger.warning(
                        "live.watcher: archive full ingest failed for %s: %s: %s",
                        record.source_path,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
        return result

    def _parse_raw_revision_chain(
        self,
        archive: Any,
        plan: Any,
        *,
        current_raw_id: str | None = None,
        current_session: ParsedSession | None = None,
    ) -> dict[str, Any]:
        parsed_by_raw_id: dict[str, Any] = {}
        for raw_id in plan.accepted_raw_ids:
            sessions = (
                [current_session]
                if raw_id == current_raw_id and current_session is not None
                else self._parse_retained_raw_sessions(archive, raw_id)
            )
            if len(sessions) != 1:
                raise RuntimeError(f"raw revision {raw_id} did not replay to exactly one session")
            parsed_by_raw_id[raw_id] = sessions[0]
        return parsed_by_raw_id

    def _apply_membership_sessions(
        self,
        archive: Any,
        source_raw_id: str,
        sessions: list[Any],
        *,
        acquired_at_ms: int,
        stage_timings_s: dict[str, float] | None = None,
        allow_current_complete_raw: bool = False,
        extra_member_raw_ids: tuple[str, ...] = (),
    ) -> tuple[list[str], int, int, bool]:
        """Apply membership-governed classification for one logical identity.

        ``extra_member_raw_ids`` (polylogue-hm2f) names raws that carry known
        membership evidence for this identity but are not currently
        discoverable through ``archive.raw_membership_raw_ids`` -- concretely,
        raws previously retired from full-revision byte governance
        (``raw_membership_retired_full_revision_siblings``): their
        ``raw_session_memberships`` row survives retirement with
        ``revision_authority`` left at the default ``'quarantined'``, so the
        ordinary byte-proven-or-caller-owned query never surfaces them. Forced
        inclusion here is what lets a later-discovered raw be weighed by the
        real content-prefix classifier against siblings the 52l2 guard is
        aware of instead of being evaluated alone.
        """
        session_ids: list[str] = []
        session_count = 0
        message_count = 0
        for session in sessions:
            logical_source_key = f"{session.source_name.value}:{session.provider_session_id}"
            for revision_raw_id in archive.convertible_full_revision_raw_ids(logical_source_key):
                retained_sessions = (
                    sessions
                    if revision_raw_id == source_raw_id
                    else self._parse_retained_raw_sessions(archive, revision_raw_id)
                )
                matches = [
                    item
                    for item in retained_sessions
                    if f"{item.source_name.value}:{item.provider_session_id}" == logical_source_key
                ]
                if len(retained_sessions) != 1 or len(matches) != 1:
                    raise RuntimeError(
                        f"full revision {revision_raw_id}:{logical_source_key} no longer parses uniquely"
                    )
                try:
                    archive.replace_raw_membership_census(
                        revision_raw_id,
                        retained_sessions,
                        parser_fingerprint=self._current_parser_fingerprint(),
                        censused_at_ms=acquired_at_ms,
                        detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                        retire_full_revision_governance=True,
                    )
                except ActiveByteRevisionChainError:
                    # polylogue-lpen: another raw's predecessor_raw_id/
                    # baseline_raw_id still points at revision_raw_id, so it
                    # cannot be retired out of byte-revision governance yet.
                    # This is expected, transient sibling-discovery-ordering
                    # contention (the same class as polylogue-52l2/hm2f, on
                    # the retire leg instead of the accept-cohort leg), not a
                    # failure of the raw currently being ingested
                    # (source_raw_id). Defer this specific sibling's
                    # retirement to a later tick -- once the dependent chain
                    # resolves, convertible_full_revision_raw_ids will surface
                    # it again -- instead of letting the exception propagate
                    # up and quarantine the unrelated raw being processed.
                    logger.warning(
                        "live.watcher: deferring full-revision retirement of %s (%s): "
                        "active byte-revision chain dependent still present",
                        revision_raw_id,
                        logical_source_key,
                    )
                    continue
            member_sessions: dict[str, Any] = {}
            projections: dict[str, Any] = {}
            revisions: list[MembershipRevision] = []
            member_raw_ids = list(
                archive.raw_membership_raw_ids(
                    logical_source_key,
                    include_complete_raw_id=source_raw_id if allow_current_complete_raw else None,
                )
            )
            for extra_raw_id in extra_member_raw_ids:
                if extra_raw_id not in member_raw_ids:
                    member_raw_ids.append(extra_raw_id)
            accepted_head_raw_id = archive.raw_revision_head_raw_id(logical_source_key)
            if accepted_head_raw_id is not None and accepted_head_raw_id not in member_raw_ids:
                member_raw_ids.append(accepted_head_raw_id)
            for member_raw_id in member_raw_ids:
                retained_sessions = (
                    sessions
                    if member_raw_id == source_raw_id
                    else self._parse_retained_raw_sessions(archive, member_raw_id)
                )
                matches = [
                    item
                    for item in retained_sessions
                    if f"{item.source_name.value}:{item.provider_session_id}" == logical_source_key
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"membership {member_raw_id}:{logical_source_key} no longer parses uniquely")
                projection = session_revision_projection(matches[0])
                member_sessions[member_raw_id] = matches[0]
                projections[member_raw_id] = projection
                retained_session = matches[0]
                browser_snapshot_fidelity: Literal["dom", "native"] | None = None
                if NATIVE_BROWSER_CAPTURE_INGEST_FLAG in retained_session.ingest_flags or (
                    COMPACT_BROWSER_CAPTURE_INGEST_FLAG in retained_session.ingest_flags
                ):
                    browser_snapshot_fidelity = "native"
                elif DOM_FALLBACK_INGEST_FLAG in retained_session.ingest_flags:
                    browser_snapshot_fidelity = "dom"
                revisions.append(
                    MembershipRevision(
                        member_raw_id,
                        projection,
                        retained_session.updated_at,
                        observed_at_ms=archive.raw_revision_acquired_at_ms(member_raw_id),
                        browser_snapshot_fidelity=browser_snapshot_fidelity,
                        provider_message_ids=frozenset(
                            message.provider_message_id
                            for message in retained_session.messages
                            if message.provider_message_id is not None
                        ),
                        provider_attachment_ids=frozenset(
                            attachment.provider_attachment_id for attachment in retained_session.attachments
                        ),
                    )
                )
            classification = classify_membership_revisions(revisions, existing_accepted_raw_id=accepted_head_raw_id)
            membership_session_id = archive.apply_raw_membership_classification(
                logical_source_key,
                classification,
                member_sessions,
                projections,
                acquired_at_ms=acquired_at_ms,
                stage_timings_s=stage_timings_s,
                stage_timing_prefix="full",
                defer_fts=True,
            )
            if membership_session_id is not None:
                session_ids.append(membership_session_id)
                self._cursor.record_convergence_debt(
                    stage="fts",
                    subject_type="session_id",
                    subject_id=membership_session_id,
                    error="live membership ingest deferred FTS to preserve writer availability",
                    deferred=True,
                )
                session_count += 1
                message_count += len(member_sessions[classification.accepted_raw_ids[-1]].messages)
        return (
            session_ids,
            session_count,
            message_count,
            archive.raw_membership_authority_complete(source_raw_id),
        )

    @staticmethod
    def _parse_retained_raw_sessions(archive: Any, raw_id: str) -> list[Any]:
        return parse_retained_raw_sessions(archive, raw_id)

    def _extract_zip_member_records(
        self,
        path: Path,
        *,
        blob_store: BlobStore,
        fallback_provider: Provider,
        file_mtime: str,
    ) -> tuple[list[tuple[str, RawSessionRecord]], int]:
        """Expand an inbox ZIP into one raw record per session member.

        Inbox ZIPs (account exports dropped into the watched inbox) were
        previously routed into the byte-level provider-detection branch, where
        ``orjson.loads`` over the ZIP container raised and the entire archive
        was silently marked excluded (#1683). This reuses the maintenance
        acquisition path's member extraction so inbox ZIPs ingest the same way
        as ``polylogue run`` source ZIPs: each member is provider-detected,
        multi-session members are split, and grouped/metadata entries preserve
        their original bytes.

        Returns ``([], 0)`` (caller marks the path excluded) only when the ZIP
        is unreadable or contains no session-bearing members.
        """
        source = Source(name=fallback_provider.value, path=path.parent)
        acquired_at = datetime.now(UTC).isoformat()
        records: list[tuple[str, RawSessionRecord]] = []
        total_bytes = 0
        validator = _ZipEntryValidator(
            fallback_provider,
            cursor_state=None,
            zip_path=path,
        )
        try:
            with zipfile.ZipFile(path) as zf:
                entries = list(validator.filter_entries(zf.infolist()))
                # A GDPR/Takeout export ZIP dropped into a provider-agnostic
                # inbox (``fallback_provider is Provider.UNKNOWN``) still has
                # a real dominant provider -- it just isn't visible from any
                # *one* member in isolation. Establish it once from whichever
                # member detects cleanly (typically ``conversations.json``)
                # and seed every member's detection with it, rather than
                # each low-signal sibling (``user.json``,
                # ``message_feedback.json``, ``shared_conversations.json``,
                # attachment ``file_*.json`` blobs, ...) independently
                # falling back to ``Provider.UNKNOWN`` -> ``unknown-export``
                # (polylogue-hs3y). A source that already resolved a
                # provider (a per-provider watched directory) is left alone.
                zip_provider_hint = fallback_provider
                if fallback_provider is Provider.UNKNOWN:
                    zip_provider_hint = self._sniff_zip_provider(zf, entries) or fallback_provider
                for info in entries:
                    if info.file_size == 0:
                        continue
                    try:
                        for raw_data in iter_zip_entry_raw_data(
                            zf,
                            ZipEntryReadContext(
                                source=source,
                                zip_path=path,
                                entry=info,
                                file_mtime=file_mtime,
                                provider_hint=zip_provider_hint,
                                blob_store=blob_store,
                            ),
                        ):
                            if raw_data.blob_hash is None:
                                continue
                            member_provider = raw_data.provider_hint or fallback_provider
                            member_size = raw_data.blob_size or 0
                            total_bytes += member_size
                            records.append(
                                (
                                    raw_data.blob_hash,
                                    RawSessionRecord(
                                        raw_id=raw_data.blob_hash,
                                        payload_provider=member_provider,
                                        capture_mode=(
                                            fallback_provider
                                            if fallback_provider is not Provider.UNKNOWN
                                            else member_provider
                                        ),
                                        source_name=member_provider.value,
                                        source_path=raw_data.source_path,
                                        source_index=raw_data.source_index or 0,
                                        blob_size=member_size,
                                        blob_publication_receipt_id=raw_data.blob_publication_receipt_id,
                                        acquired_at=acquired_at,
                                        file_mtime=raw_data.file_mtime,
                                    ),
                                )
                            )
                    except ZipBombError as exc:
                        logger.warning("Skipping ZIP member %s in %s: %s", info.filename, path, exc)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning("Failed to expand inbox ZIP %s: %s", path, exc)
            return [], 0
        return records, total_bytes

    @staticmethod
    def _sniff_zip_provider(
        zf: zipfile.ZipFile,
        entries: list[zipfile.ZipInfo],
    ) -> Provider | None:
        """Detect a ZIP's dominant provider from whichever member detects cleanly.

        Reads only a small prefix (``_DETECTION_PREFIX_SIZE``, matching the
        equivalent whole-file detection budget in
        ``source_acquisition_components.read_plain_source_file``) of each
        JSON/JSONL member in order until one yields a positive, non-unknown
        ``detect_provider`` result. Returns ``None`` if no member detects
        (e.g. a genuinely mixed or non-conversation ZIP), leaving the caller
        to keep the original ``Provider.UNKNOWN`` fallback.
        """
        for info in entries:
            name_lower = info.filename.lower()
            if not name_lower.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                continue
            try:
                with open_bounded_zip_entry(zf, info) as handle:
                    prefix = handle.read(_DETECTION_PREFIX_SIZE)
            except (zipfile.BadZipFile, OSError, ZipBombError):
                continue
            if not prefix:
                continue
            detected = _detect_provider_from_raw_bytes(
                prefix,
                info.filename,
                Provider.UNKNOWN,
                truncated_tail_ok=True,
            )
            if detected is not Provider.UNKNOWN:
                return detected
        return None

    def _mark_excluded_cursor(self, path: Path, stat: object, *, source_name: str) -> None:
        st_size = int(getattr(stat, "st_size", 0))
        self._cursor.set(
            path,
            st_size,
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

    def _resynthesize_cursor_from_source(self, path: Path) -> CursorRecord | None:
        """Reconstruct an append-eligible cursor from durable ``source.db`` evidence.

        ``ingest_cursor`` (``CursorStore``) lives in the **disposable** ``ops.db``
        tier: every reset (index rebuild, schema mismatch, ``polylogue ops
        reset``) wipes it, forcing the next observation of every still-growing
        file back onto the full-capture path even though the file itself
        hasn't changed shape at all (polylogue-aex0, see
        ``docs/design/prefix-blob-reclamation.md``'s "forward-fix sibling"
        section). This is a *secondary* lookup, tried only when
        :meth:`_append_plan`'s primary ``ops.db`` cursor is absent or
        unusable; it never weakens ``RawRevisionAuthority`` -- the exact same
        ``plan_revision_replay`` the durable classifier already uses is the
        sole source of truth for which raw is the accepted head.

        Only a ``revision_kind='full'`` accepted head is resynthesized. An
        append-kind head's stored raw payload is not reliably byte-identical
        to the live file at that offset: historical Codex append rows (from
        before polylogue-u19l) had a synthetic ``session_meta`` line
        injected ahead of the real delta by ``_append_payload_for_provider``,
        and this code path cannot cheaply tell an old row from a new one, so
        it cannot stand in for a verified file-byte prefix -- and reusing a
        *stale* full baseline
        behind an already-accepted append chain would create a second
        sibling append candidate at the same offset, making
        ``plan_revision_replay`` mark the whole chain ambiguous. Declining
        whenever the head isn't 'full' avoids both hazards. This still
        covers the dominant real-world case: the very next observation of a
        file after an ops.db reset takes the full-capture path and writes a
        fresh byte-proven 'full' revision, which this fallback can
        immediately use so the *following* observation resumes appending
        instead of full-recapturing forever.

        Returns ``None`` whenever the durable evidence is ambiguous, absent,
        or not a 'full' head -- callers fall through to the existing
        full-capture path exactly as before this fallback existed.
        """
        source_db = self._archive_source_db_path()
        if not source_db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
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
                       lower(hex(blob_hash)) AS blob_hash_hex
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
        candidates = [
            RevisionCandidate(
                raw_id=str(row[0]),
                logical_source_key=logical_source_key,
                kind=RawRevisionKind(str(row[1])),
                source_revision=str(row[2]),
                acquisition_generation=int(row[3]),
                authority=RawRevisionAuthority(str(row[4])),
                blob_size=int(row[5]),
                predecessor_source_revision=str(row[10]) if row[10] is not None else None,
                predecessor_raw_id=str(row[6]) if row[6] is not None else None,
                baseline_raw_id=str(row[7]) if row[7] is not None else None,
                append_start_offset=int(row[8]) if row[8] is not None else None,
                append_end_offset=int(row[9]) if row[9] is not None else None,
            )
            for row in rows
        ]
        try:
            replay_plan = plan_revision_replay(candidates)
        except ValueError:
            return None
        if not replay_plan.accepted_chain:
            return None
        head_raw_id = replay_plan.accepted_chain[-1]
        head = next(candidate for candidate in candidates if candidate.raw_id == head_raw_id)
        if head.kind is not RawRevisionKind.FULL:
            return None
        blob_hash_hex = blob_hash_by_raw_id.get(head_raw_id)
        if blob_hash_hex is None or len(blob_hash_hex) != 64:
            return None
        byte_offset = head.blob_size
        return CursorRecord(
            source_path=str(path),
            byte_size=byte_offset,
            byte_offset=byte_offset,
            last_complete_newline=byte_offset,
            record_count=0,
            updated_at=datetime.now(UTC).isoformat(),
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=head.source_revision,
            # The prefix hash IS the 'full' raw's own blob hash (its content is
            # exactly bytes[0:byte_offset] of the source at capture time); the
            # tail-hash component is never read by ``_append_plan`` (only the
            # prefix half of the encoded authority is consulted there), so it
            # is filled with the same digest rather than re-reading the blob.
            tail_hash=encode_cursor_hash_authority(blob_hash_hex, blob_hash_hex, ctime_ns=0),
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
            conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
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
                          AND r.parse_error IS NOT NULL
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

    def _append_plan(self, path: Path, *, cursor: CursorRecord | None = None) -> _AppendPlan | _DeferredAppend | None:
        # Append planning is safe only for newline-delimited record streams.
        # Watch-source names describe acquisition routes, not file semantics:
        # mutable browser snapshots can arrive through the generic inbox.
        if path.suffix.lower() != ".jsonl":
            return None
        if self._source_name_for(path) == "hermes":
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
        cursor = cursor or self._cursor.get_record(path)
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
        if (
            cursor is None
            or cursor.parser_fingerprint != self._current_parser_fingerprint()
            or cursor.content_fingerprint is None
        ):
            return None
        if self._cursor_references_raw_failure_requiring_full_replay(path, cursor):
            return None
        expected_prefix_hash = cursor_prefix_hash(cursor.tail_hash)
        if expected_prefix_hash is None:
            return None
        try:
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                if stat.st_size <= cursor.byte_offset:
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
                start_offset = max(cursor.byte_offset, 0)
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
                handle.seek(0)
                accepted_hasher = sha256()
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
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        return _DEFER_APPEND
                    accepted_hasher.update(chunk)
                    remaining -= len(chunk)
                accepted_prefix_hash = accepted_hasher.hexdigest()
                final_stat = os.fstat(handle.fileno())
        except OSError:
            return None
        if _file_observation(final_stat) != _file_observation(stat):
            return _DEFER_APPEND
        append_result = self._append_payload_for_provider(path, self._source_name_for(path), complete_payload)
        if append_result is None:
            return None
        append_payload, native_id_hint = append_result
        tail_hash = sha256(complete_payload).hexdigest()
        return _AppendPlan(
            path=path,
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
            accepted_tail_hash=sha256(accepted_tail).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
            accepted_prefix_hash=accepted_prefix_hash,
            authority_bytes_read=last_complete_newline,
            native_id_hint=native_id_hint,
        )

    def _append_payload_for_provider(
        self, path: Path, source_name: str, payload: bytes
    ) -> tuple[bytes, str | None] | None:
        """Return the literal append payload plus an optional identity hint.

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
        Callers persist it to ``raw_sessions.native_id`` (``_AppendPlan.
        native_id_hint`` -> ``append_ingest.py``) and pass it back as the
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
            return payload, identity
        if provider is Provider.CLAUDE_CODE and not self._claude_code_tail_matches_existing_identity(
            path, payload, existing_id=identity
        ):
            return None
        return payload, None

    def _existing_provider_session_id(self, path: Path, *, expected_origin: str) -> str | None:
        identity = self._existing_archive_session_native_id(path, expected_origin=expected_origin)
        if identity is not None:
            return identity
        if expected_origin != Origin.CODEX_SESSION.value:
            return None
        codex_identity = self._codex_session_meta_native_id(path)
        if codex_identity is None:
            return None
        if self._archive_has_native_session("codex-session", codex_identity):
            return codex_identity
        return None

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
        source_db = archive_root / "source.db"
        if not index_db.exists() or not source_db.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
            try:
                conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
                row = conn.execute(
                    """
                    SELECT 1
                    FROM sessions AS s
                    JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                    WHERE s.origin = ? AND r.origin = ? AND s.native_id = ?
                    LIMIT 1
                    """,
                    (origin, origin, native_id),
                ).fetchone()
                conn.execute("DETACH DATABASE source_tier")
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
            conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
            try:
                conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
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
                conn.execute("DETACH DATABASE source_tier")
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

    def _ingest_append_plans(self, plans: list[_AppendPlan]) -> _AppendResult:
        return ingest_append_plans(self, plans)

    def _compact_superseded_raw_snapshots(self, paths: list[Path]) -> None:
        if not paths:
            return
        from polylogue.storage.raw_retention import (
            RawRetentionSafetyError,
            active_raw_retention_authority,
            compact_paths_superseded_raw_snapshots,
        )

        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        index_db = ArchiveLocation.resolve(archive_root).active_index_path
        if not source_db.exists():
            return
        with closing(sqlite3.connect(source_db)) as conn, conn:
            conn.row_factory = sqlite3.Row
            try:
                retention_authority = active_raw_retention_authority(conn, index_db_path=index_db)
            except RawRetentionSafetyError as exc:
                logger.warning("live.watcher: skipped unsafe raw snapshot compaction: %s", exc)
                return
            result = compact_paths_superseded_raw_snapshots(
                conn,
                paths,
                limit_per_path=25,
                min_acquired_at=self._raw_compaction_min_acquired_at,
                protected_raw_ids=retention_authority.protected_raw_ids,
                eligible_raw_ids=retention_authority.eligible_raw_ids,
            )
        if result.errors:
            logger.warning("live.watcher: raw snapshot compaction errors: %s", "; ".join(result.errors[:3]))

    def _record_append_cursor(self, plan: _AppendPlan) -> bool:
        """Persist a proven append frontier without mistaking later growth for rewrite.

        ``plan`` carries a prefix witness produced before append persistence.
        A live JSONL writer may append another record before this method runs;
        that does not invalidate the accepted range. Keep the plan's original
        observation in the cursor so a same-size replacement is still forced
        through the full route on the next batch.
        """
        latest_stat: os.stat_result | None = None
        expected_observation = (
            plan.st_dev,
            plan.st_ino,
            plan.stat_size,
            plan.mtime_ns,
            plan.ctime_ns,
        )
        try:
            stat = plan.path.stat()
            latest_stat = stat
            if stat.st_dev != plan.st_dev or stat.st_ino != plan.st_ino or stat.st_size < plan.last_complete_newline:
                raise ValueError("source replaced or truncated")
            payload_hash, _payload_bytes = sha256_range_from_path(
                plan.path,
                start_offset=plan.start_offset,
                end_offset=plan.last_complete_newline,
            )
            tail_hash, _tail_bytes = tail_hash_from_path(plan.path, plan.last_complete_newline)
            final_stat = plan.path.stat()
            latest_stat = final_stat
            if (
                final_stat.st_dev != plan.st_dev
                or final_stat.st_ino != plan.st_ino
                or final_stat.st_size < plan.last_complete_newline
            ):
                raise ValueError("source replaced or truncated during cursor verification")
            if payload_hash != plan.payload_hash:
                raise ValueError("accepted append bytes changed")
            if plan.accepted_tail_hash is not None and tail_hash != plan.accepted_tail_hash:
                raise ValueError("accepted append tail changed")
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
                captured_file_observation=(
                    plan.st_dev,
                    plan.st_ino,
                    plan.stat_size,
                    plan.mtime_ns,
                    plan.ctime_ns or 0,
                ),
            )
            return False
        if _file_observation(final_stat) != expected_observation:
            logger.info(
                "live.watcher: source changed after append persistence; retained proven append frontier: %s",
                plan.path,
            )
        content_fingerprint = append_source_revision(plan.cursor_fingerprint or "", plan.payload_hash)
        stored_tail_hash = (
            encode_cursor_hash_authority(
                plan.accepted_prefix_hash,
                tail_hash,
                ctime_ns=plan.ctime_ns or 0,
            )
            if plan.accepted_prefix_hash is not None
            else tail_hash
        )
        updated = self._cursor.set(
            plan.path,
            plan.stat_size,
            byte_offset=plan.last_complete_newline,
            last_complete_newline=plan.last_complete_newline,
            parser_fingerprint=self._current_parser_fingerprint(),
            content_fingerprint=content_fingerprint,
            tail_hash=stored_tail_hash,
            source_name=plan.source_name,
            st_dev=plan.st_dev,
            st_ino=plan.st_ino,
            mtime_ns=plan.mtime_ns,
        )
        if updated:
            self._cursor.reset_failures(plan.path)
        return updated


# fmt: off
__all__ = [
    "AppendCapabilityReceipt",
    "LiveBatchMetrics",
    "LiveBatchProcessor",
    "_FullIngestResult",
    "_LARGE_FULL_PARSE_PROGRESS_BYTES",
    "_MAX_APPEND_PLAN_PAYLOAD_BYTES",
    "_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES",
    "_SMALL_FULL_PARSE_PROGRESS_MAX_FILES",
    "_STREAMING_FULL_INGEST_BYTES",
    "_full_ingest_worker_count",
    "_full_parse_progress_groups",
    "append_capability_receipt",
    "fingerprint_file",
    "last_complete_newline_from_tail",
]
# fmt: on
