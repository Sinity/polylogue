"""Small archive-root façade over archive source/index/user tiers.

Writer module: index.

Raw revision/membership governance (the twin-write ``raw-membership-classification``
contract spanning index and source) moved to
``polylogue.storage.sqlite.archive_tiers.revision_governance`` (polylogue-1r9c);
``delete_sessions`` is this module's remaining direct writer, index-only
(user-tier overlays are deliberately left in place, see its docstring).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, BinaryIO, Literal, NoReturn, TypedDict, cast

from polylogue.annotations.batch import AnnotationBatch
from polylogue.annotations.schema import AnnotationSchema
from polylogue.archive.artifact_taxonomy import ArtifactClassification
from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns
from polylogue.archive.query.predicate import (
    QueryPredicate,
)
from polylogue.archive.revision_authority import (
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.archive.revision_replay import (
    RevisionCandidate,
    RevisionReplayPlan,
)
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostEstimateStatus,
    CostModelBreakdown,
    CostUnavailableReason,
    CostUsagePayload,
    _normalize_model,
)
from polylogue.archive.semantic.subscription_pricing import compute_credit_cost
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.archive.stats import ArchiveStats
from polylogue.archive.topology.edge import topology_status_composes_sql
from polylogue.core.enums import Origin, Provider
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.core.json import require_json_value
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import SessionId
from polylogue.insights.affordance_usage import (
    clean_patterns as _clean_affordance_patterns,
)
from polylogue.insights.affordance_usage import (
    evidence_kind_for_row as _affordance_evidence_kind,
)
from polylogue.insights.affordance_usage import (
    family_for_text as _affordance_family_for_text,
)
from polylogue.insights.affordance_usage import (
    like_param as _affordance_like_param,
)
from polylogue.insights.affordance_usage import (
    matched_by_row as _affordance_matched_by,
)
from polylogue.insights.affordance_usage import (
    normalized_tool_name_for_row as _affordance_normalized_tool_name,
)
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveDebtInsight,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfileInsight,
    SessionLatencyProfilePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionWorkEventInsight,
    ThreadInsight,
    UsageTimelineInsight,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.archive_models import ThreadMemberEvidencePayload, ThreadPayload
from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport, _audit_one
from polylogue.insights.confidence import ConfidenceBand
from polylogue.insights.confidence import from_score as confidence_from_score
from polylogue.insights.feedback import LearningCorrection, parse_correction_kind
from polylogue.insights.objective_posture import structural_objective_posture
from polylogue.insights.readiness import (
    InsightOriginCoverage,
    InsightReadinessEntry,
    InsightReadinessQuery,
    InsightReadinessReport,
    InsightReadinessVerdict,
    InsightStorageArtifact,
    InsightVersionCoverage,
    known_insight_readiness_names,
    normalize_insight_readiness_name,
)
from polylogue.insights.rigor import list_rigor_contracts
from polylogue.insights.session_label import session_structural_label_for_session
from polylogue.insights.temporal_source import time_confidence_for_source
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.pipeline.ids import SessionRevisionProjection
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import Heartbeat, PreparedBlob
from polylogue.storage.fts.sql import (
    FTS_BULK_SESSION_WRITE_GUARD,
    delete_session_identity_rows_sql,
    delete_session_rows_sql,
    trigram_delete_session_rows_sql,
)
from polylogue.storage.insights.session.records import SessionProfileRecord
from polylogue.storage.insights.session.runtime import (
    SESSION_INSIGHT_MATERIALIZATION_TYPES,
    SessionInsightStatusSnapshot,
)
from polylogue.storage.insights.session.status import session_insight_status_sync
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.archive_tiers import archive_query_reads as _archive_query_reads
from polylogue.storage.sqlite.archive_tiers.archive_query_reads import (
    ArchiveActionQueryRow,
    ArchiveAssertionQueryRow,
    ArchiveBlockQueryRow,
    ArchiveContextSnapshotQueryRow,
    ArchiveDelegationAncestryRow,
    ArchiveDelegationCard,
    ArchiveDelegationQueryRow,
    ArchiveDelegationSubtreeRow,
    ArchiveFileQueryRow,
    ArchiveMessageQueryRow,
    ArchiveObservedEventQueryRow,
    ArchiveQueryUnitAggregateRow,
    ArchiveQueryUnitMultiAggregatePage,
    ArchiveRunQueryRow,
    _action_command_expression,
    _session_filter_clause,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    archive_tier_spec,
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.raw_admission import RawAdmissionResult
from polylogue.storage.sqlite.archive_tiers.read_insights import ArchiveReadInsights
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    FullSnapshotFoldAuthorization,
)
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    ActiveByteRevisionChainError,
    ArchiveRawParsedWriteResult,
    MembershipReplayConflictError,
    _authorize_full_snapshot_fold,
    _flush_pending_raw_parse_states,
    _index_parsed_for_retained_raw,
    _promote_contiguous_append_evidence,
    _raw_parse_failure_state,
    _raw_parse_success_state,
    _raw_revision_authority,
    _raw_revision_candidates,
    _raw_revision_matches_segments,
    _raw_revision_payload_digest_and_size,
    _raw_revision_source_path_has_divergent_evidence,
    _write_parsed_precedence_result,
    admit_raw_and_parsed_result,
    admit_raw_artifact_blob_ref,
    admit_raw_artifact_payload,
    apply_raw_membership_classification,
    apply_raw_revision_replay,
    bind_raw_revision,
    blob_path_for_hash,
    classify_raw_revision_cohort_for_frozen_candidate,
    classify_raw_revision_cohort_for_live_watch,
    classify_raw_revision_cohort_for_rebuild_repair,
    classify_untyped_full_revision_groups,
    convertible_full_revision_raw_ids,
    defer_raw_revision_adoption,
    expand_raw_membership_selection,
    expand_raw_membership_selection_sync,
    finalize_raw_parse_state,
    mark_raw_parse_failed,
    mark_raw_parse_succeeded,
    membership_decisions_for_classification,
    open_raw_revision_material,
    pending_raw_revision_logical_keys,
    raw_append_revision_parent,
    raw_full_revision_generation,
    raw_membership_authority_complete,
    raw_membership_census_rows,
    raw_membership_decision_pending,
    raw_membership_raw_ids,
    raw_membership_rebuild_raw_ids,
    raw_membership_retired_full_revision_siblings,
    raw_membership_selection_components,
    raw_membership_selection_components_sync,
    raw_native_id,
    raw_payload_sizes,
    raw_revision_acquired_at_ms,
    raw_revision_descriptor,
    raw_revision_file_mtime,
    raw_revision_head_raw_id,
    raw_revision_material,
    raw_revision_observation_order,
    raw_revision_observed_at_ms,
    raw_revision_rebuild_selection,
    raw_revision_replay_adoptable,
    raw_revision_replay_plan,
    record_raw_failure_evidence,
    release_provisional_full_revisions,
    replace_raw_membership_census,
    require_frozen_membership_authority,
    unclassified_raw_revision_rows,
    write_parsed_for_retained_raw,
    write_parsed_for_retained_raw_result,
    write_raw_and_parsed,
    write_raw_and_parsed_result,
    write_raw_blob_ref,
    write_raw_payload,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    ArchiveSourceBlobRef,
    delete_source_hook_event,
    deterministic_blob_hash,
    deterministic_raw_session_id,
    list_hook_events,
    record_raw_container_coordinate,
    write_source_hook_event,
)
from polylogue.storage.sqlite.archive_tiers.types import (
    ArchiveTier,
)
from polylogue.storage.sqlite.archive_tiers.user_annotations import (
    DurableAnnotationSchema,
    list_durable_annotation_schemas,
    persist_annotation_batch,
    persist_annotation_schema,
    read_annotation_batch,
    read_durable_annotation_schema,
)
from polylogue.storage.sqlite.archive_tiers.user_annotations import (
    list_annotation_batches as _list_annotation_batches,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    ArchiveBlackboardNoteEnvelope,
    AssertionKind,
    assertion_id_for_annotation,
    assertion_id_for_correction,
    assertion_id_for_mark,
    assertion_id_for_recall_pack,
    assertion_id_for_saved_view,
    assertion_id_for_session_metadata,
    assertion_id_for_session_tag,
    assertion_id_for_workspace,
    correction_id_for,
    list_archive_blackboard_note_envelopes,
    list_assertions_by_kind,
    list_assertions_for_target,
    mark_assertion_status,
    read_assertion_envelope,
    upsert_annotation,
    upsert_blackboard_note,
    upsert_correction,
    upsert_mark,
    upsert_recall_pack,
    upsert_saved_view,
    upsert_session_metadata_assertion,
    upsert_session_tag_assertion,
    upsert_workspace,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveInsightMaterialization,
    ArchiveSessionEnvelope,
    ArchiveSessionPhase,
    ArchiveSessionWorkEvent,
    ArchiveWriteOutcome,
    PreparedSessionRows,
    read_archive_session_envelope,
    read_archive_session_page,
    read_insight_materialization,
    read_session_phases,
    read_session_work_events,
    rebuild_archive_messages_fts,
    search_archive_blocks,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection_profile import (
    BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS,
    READ_CONNECTION_PRAGMA_STATEMENTS,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
    open_connection,
    open_readonly_connection,
)
from polylogue.storage.sqlite.queries.sessions_identity import session_id_prefix_bounds
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync
from polylogue.storage.usage import SessionUsageCost, session_usage_costs_for_connection


@dataclass(slots=True)
class _UsageTimelineAccumulator:
    bucket: str
    source_name: str | None
    model_name: str | None
    event_count: int = 0
    event_session_count: int = 0
    usage: CostUsagePayload = field(default_factory=CostUsagePayload)
    reasoning_output_tokens: int = 0
    stored_cost_usd: float = 0.0
    subscription_credits: float = 0.0
    cost_provenance_counts: dict[str, int] = field(default_factory=dict)
    source_sort_key: float | None = None

    def note_sort_key(self, value: object) -> None:
        if isinstance(value, int | float) and (self.source_sort_key is None or float(value) > self.source_sort_key):
            self.source_sort_key = float(value)


@dataclass(slots=True)
class _CostRollupAccumulator:
    source_name: str
    model_name: str | None
    normalized_model: str | None
    session_count: int = 0
    priced_session_count: int = 0
    unavailable_session_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    basis: CostBasisPayload = field(default_factory=CostBasisPayload)
    usage: CostUsagePayload = field(default_factory=CostUsagePayload)
    total_usd: float = 0.0
    confidence_total: float = 0.0
    source_updated_at_ms: int | None = None
    source_sort_key: float | None = None
    per_model: dict[tuple[str | None, str | None], CostModelBreakdown] = field(default_factory=dict)

    def note_source_updated_at(self, value: object) -> None:
        if isinstance(value, int) and (self.source_updated_at_ms is None or value > self.source_updated_at_ms):
            self.source_updated_at_ms = value

    def note_sort_key(self, value: object) -> None:
        if isinstance(value, int | float) and (self.source_sort_key is None or float(value) > self.source_sort_key):
            self.source_sort_key = float(value)


class IndexStatus(TypedDict):
    """block-FTS index existence and indexed-document count."""

    exists: bool
    count: int


@dataclass(frozen=True, slots=True)
class ArchiveSessionSummary:
    """archive summary projection over archive sessions."""

    session_id: str
    native_id: str
    origin: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    word_count: int
    tags: tuple[str, ...]
    parent_id: str | None = None
    session_kind: str = "standard"
    reported_duration_ms: int | None = None
    tool_use_count: int = 0
    thinking_count: int = 0
    paste_count: int = 0
    user_message_count: int = 0
    authored_user_message_count: int = 0
    assistant_message_count: int = 0
    system_message_count: int = 0
    tool_message_count: int = 0
    user_word_count: int = 0
    authored_user_word_count: int = 0
    assistant_word_count: int = 0
    working_directories: tuple[str, ...] = ()
    title_source: str | None = None
    title_ref: str | None = None
    title_confidence: float | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    # See ``Session.display_name`` / ``SessionSummary.display_name``
    # (polylogue-cgfy): a provider-assigned name (e.g. Claude Code's slug)
    # distinct from the (possibly derived) title.
    display_name: str | None = None
    # Read-time projection over current structural evidence. Never persisted.
    display_label: str | None = None
    terminal_state: str | None = None
    total_cost_usd: float | None = None
    cost_provenance: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSessionSearchHit:
    """Search hit projection over archive block FTS."""

    rank: int
    session_id: str
    block_id: str
    message_id: str
    origin: str
    title: str | None
    snippet: str
    lane_ranks: dict[str, int | None] | None = None


# polylogue-qsb4: both traversals recurse natively over the `delegations`
# view (backed by `delegation_facts`, the richer typed source of truth). One
# SQLite recursive CTE each -- no N+1, no client-side stitching.
#
# Quarantined edges (session_links' TopologyEdgeStatus cycle-break
# precedent, reused verbatim: delegation_facts_source's `mapping_state =
# 'quarantined'` rows already mark exactly this) and authority-contradicted
# rows are excluded from traversal by construction. The latter is not a
# topological cycle, but it is an authoritative verdict that the inferred
# parent edge is false; composing it would reintroduce rejected topology. A
# defensive visited-path guard
# (the `path`/`instr()` machinery below) is still carried on every step
# despite that exclusion: quarantine is asserted by the topology resolver
# over `session_links` alone, while `delegations` unions edges resolved by
# an independent mechanism (`content_pairs`' provider-asserted content-
# identity match, `delegation_facts_source` in index.py) that the topology
# resolver's cycle detector never inspects. Two content-identity-matched
# edges could in principle compose into a cycle the quarantine pass never
# saw; the guard makes that structurally unreachable rather than assumed
# absent.


# polylogue-a7xr.16: table-driving the SELECT side. ``query_messages`` and
# ``query_session_messages`` used to hand-duplicate an identical "fetch every
# block for a set of message_ids, hydrate ArchiveBlockRow" block (column
# list, WHERE/ORDER BY shape, and the twelve-line per-field accessor loop),
# so a column added to this projection had to be added twice, by hand, in
# lockstep. ``_ARCHIVE_BLOCK_QUERY_COLUMNS`` is now the single source of
# truth for that projection's column list -- both the SELECT clause and the
# hydration loop derive from it -- and ``_fetch_blocks_for_messages`` is the
# one place either query method calls. Unlike the INSERT side (write.py),
# column order here carries no correctness risk on its own (sqlite3.Row is
# accessed by name, not position); the risk this eliminates is the same
# projection existing in two places that can silently drift out of sync.


# polylogue-a7xr.16: the same drift hazard, for ``ArchiveFileQueryRow``.
# ``query_files`` and ``query_session_files`` derive their affected-file
# aggregate from two different inner subqueries (actions vs. blocks), but the
# OUTER SELECT projecting the aggregate's columns (aliased ``f.`` for the
# inner subquery, ``s.`` for the joined session) is byte-identical between
# the two methods, as was the twelve-line hydration loop consuming it.
# ``_ARCHIVE_FILE_QUERY_COLUMNS`` (output name, source expression) is the one
# place that outer projection is named; ``_ARCHIVE_FILE_QUERY_SELECT_SQL``
# derives the shared SELECT fragment from it and ``_hydrate_archive_file_query_row``
# derives the shared hydration from the same output names, so the two can
# never drift out of sync with each other again.


# polylogue-aif4: the same drift hazard, for ``ArchiveActionQueryRow``.
# ``query_actions`` and ``query_session_actions`` both join the actions view
# (aliased ``a``) to ``sessions``/``messages`` and project the identical
# sixteen-column action shape -- hand-duplicated byte-for-byte before this.
# ``query_session_action_occurrences`` is deliberately NOT included: it
# selects from raw ``blocks`` (aliased ``u``/``r``, no follow-up relation) to
# stay cheap on very large sessions, so its column *sources* genuinely
# differ even though the output shape rhymes -- forcing it onto this same
# fragment would either lose that cost tradeoff or fake follow-up columns
# that were never computed.


# polylogue-aif4: ``query_unit_counts`` and ``query_unit_multi_counts`` each
# hand-maintained an identical copy of the unit -> row-alias map and the unit
# -> FROM-clause map used to dispatch a terminal aggregate query across the
# seven SQL-backed query units. Both dicts were byte-identical between the
# two methods (the only per-call variation is the ``action`` entry's derived
# relation name, now a parameter). One source of truth here means a new
# query unit is wired into aggregate counts by editing one place, not two
# that can silently drift apart.


class _SourceTierOnlyIndexConnection:
    """Loud placeholder for an archive mode that must not open ``index.db``.

    Acquire-only ingestion (polylogue-gbs02) deliberately never opens
    ``index.db`` — a derived tier awaiting rebuild may be at an older schema
    version, and the safest way to guarantee no index write can occur is to
    hold no index handle at all. Any code path that reaches for the index
    connection in this mode is a bug; it must fail immediately and loudly
    instead of writing through a stale-schema handle.
    """

    def __init__(self, mode: str = "source-tier acquisition") -> None:
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        if name == "close":
            return lambda: None
        raise RuntimeError(
            f"index tier is unavailable in {self._mode} mode "
            f"(attempted connection attribute {name!r}); only raw source-tier "
            "access is permitted while the derived tier is unavailable"
        )


class InactiveCandidateDurableWriteError(RuntimeError):
    """An inactive generation attempted to mutate read-through durable state."""


class ReadOnlyArchiveError(RuntimeError):
    """A read-only archive evidence store received a mutation request."""


class _InactiveCandidateBlobPublisher(ArchiveBlobPublisher):
    """Read frozen blob bytes while refusing candidate publication attempts."""

    @staticmethod
    def _refuse() -> NoReturn:
        raise InactiveCandidateDurableWriteError(
            "inactive candidate generations may read frozen blobs but may not publish or replace blob bytes"
        )

    def _queue(self, prepared: PreparedBlob) -> NoReturn:
        del prepared
        self._refuse()

    def prepare_from_path(self, source: Path, *, heartbeat: Heartbeat | None = None) -> NoReturn:
        del source, heartbeat
        self._refuse()

    def prepare_from_fileobj(self, source: IO[bytes], *, heartbeat: Heartbeat | None = None) -> NoReturn:
        del source, heartbeat
        self._refuse()

    def prepare_from_bytes(self, data: bytes) -> NoReturn:
        del data
        self._refuse()

    def allocate_staging_path(self, *, prefix: str, suffix: str = "") -> NoReturn:
        del prefix, suffix
        self._refuse()

    def discard_staging_path(
        self,
        staged_path: Path,
        *,
        companion_suffixes: Iterable[str] = (),
    ) -> NoReturn:
        del staged_path, companion_suffixes
        self._refuse()

    def publish_prepared(self, prepared: PreparedBlob) -> NoReturn:
        del prepared
        self._refuse()

    def publish_many(self, prepared: Iterable[PreparedBlob]) -> NoReturn:
        del prepared
        self._refuse()

    @staticmethod
    def discard_prepared(prepared: PreparedBlob) -> NoReturn:
        del prepared
        _InactiveCandidateBlobPublisher._refuse()

    def write_from_path(self, source: Path, *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        del source, heartbeat
        return self._refuse()

    def write_from_fileobj(self, source: IO[bytes], *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        del source, heartbeat
        return self._refuse()

    def write_from_bytes(self, data: bytes) -> tuple[str, int]:
        del data
        return self._refuse()

    def flush(self) -> tuple[()]:
        return ()

    def discard_pending(self) -> None:
        return None


class ArchiveStore:
    """Minimal archive-root façade for archive source/index/user tiers."""

    def __init__(
        self,
        archive_root: Path,
        *,
        initialize: bool = True,
        read_only: bool = False,
        read_timeout: float = 5.0,
        owned_inactive_generation: tuple[str, str] | None = None,
        source_tier_acquisition: bool = False,
        frozen_source_validation: bool = False,
        frozen_index_path: Path | None = None,
        opened_index_fd: int | None = None,
    ) -> None:
        if source_tier_acquisition and read_only:
            raise ValueError("source_tier_acquisition mode is a writer mode; read_only must be False")
        if frozen_source_validation and (not read_only or owned_inactive_generation is not None):
            raise ValueError("frozen source validation requires a read-only active archive")
        if frozen_index_path is not None and not read_only:
            raise ValueError("a pinned index path is valid only for read-only archive access")
        if opened_index_fd is not None and not read_only:
            raise ValueError("an opened index descriptor is valid only for read-only archive access")
        self._source_tier_acquisition = source_tier_acquisition
        self._owned_inactive_generation = owned_inactive_generation
        self._frozen_source_validation = frozen_source_validation
        self._frozen_index_path = frozen_index_path
        self._opened_index_fd = opened_index_fd
        self._pinned_read = frozen_index_path is not None
        self._inactive_candidate_durable_read_only = owned_inactive_generation is not None or frozen_source_validation
        self._active_writer_lease = None
        if not read_only:
            from polylogue.paths import archive_root as configured_archive_root
            from polylogue.storage.archive_identity import assert_writable_archive_identity

            if owned_inactive_generation is None:
                from polylogue.storage.index_generation import ActiveWriterLease

                self._active_writer_lease = ActiveWriterLease(archive_root)
                self._active_writer_lease.acquire()
                if not source_tier_acquisition:
                    try:
                        assert_writable_archive_identity(
                            configured_root=configured_archive_root(),
                            active_root=archive_root,
                        )
                    except Exception:
                        self._active_writer_lease.close()
                        self._active_writer_lease = None
                        raise
            else:
                from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore

                generation_id, owner_id = owned_inactive_generation
                # An inactive generation is opened from its generation root,
                # while the configured root may intentionally point at a
                # different live archive. Read the candidate's declared root,
                # then require the store anchored at that root to return the
                # exact same metadata. Deriving the root from ``../..`` fails
                # for supported split-index layouts, where generations live
                # beside the external active index rather than below the
                # durable archive root.
                generation = IndexGeneration(
                    **json.loads((archive_root / "generation.json").read_text(encoding="utf-8"))
                )
                declared_archive_root = Path(generation.archive_root).resolve(strict=True)
                authoritative_generation = IndexGenerationStore.for_archive_root(
                    declared_archive_root,
                    repair_anchor=False,
                ).load(generation_id)
                if (
                    generation != authoritative_generation
                    or generation.owner_id != owner_id
                    or generation.state != "inactive"
                    or Path(generation.index_path).parent.resolve(strict=True) != archive_root.resolve(strict=True)
                ):
                    raise RuntimeError("inactive index generation ownership validation failed")
                for filename in ("source.db", "user.db", "embeddings.db", "ops.db", "blob"):
                    expected = declared_archive_root / filename
                    candidate = archive_root / filename
                    if expected.exists() or expected.is_symlink():
                        if not candidate.is_symlink() or candidate.resolve(strict=True) != expected.resolve(
                            strict=True
                        ):
                            raise RuntimeError(
                                f"inactive index generation has an invalid read-through target: {filename}"
                            )
                    elif candidate.exists() or candidate.is_symlink():
                        raise RuntimeError(f"inactive index generation invented a read-through target: {filename}")
        try:
            self._initialize_store(
                archive_root,
                # The generation store already initialized the candidate's
                # sole owned tier, index.db. Active-root initialization would
                # reinterpret deliberate source/user read-through symlinks as
                # writable durable tiers.
                initialize=initialize and not source_tier_acquisition and owned_inactive_generation is None,
                read_only=read_only,
                read_timeout=read_timeout,
                opened_index_fd=opened_index_fd,
                # polylogue-623q: only ever True for a write connection against
                # an OWNED INACTIVE generation -- never read until promoted,
                # discarded wholesale on any failure -- so it is safe to open
                # with a far more aggressive durability/speed tradeoff than
                # the live single-writer profile. See
                # BULK_BUILD_WRITE_CONNECTION_PROFILE's docstring.
                bulk_build_profile=owned_inactive_generation is not None,
            )
        except Exception:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
            if self._active_writer_lease is not None:
                self._active_writer_lease.close()
                self._active_writer_lease = None
            raise

    def _initialize_store(
        self,
        archive_root: Path,
        *,
        initialize: bool,
        read_only: bool,
        read_timeout: float,
        bulk_build_profile: bool = False,
        opened_index_fd: int | None = None,
    ) -> None:
        self.archive_root = archive_root
        self.source_db_path = archive_root / "source.db"
        self.embeddings_db_path = archive_root / "embeddings.db"
        self.user_db_path = archive_root / "user.db"
        self.ops_db_path = archive_root / "ops.db"
        self._read_only = read_only
        if opened_index_fd is not None and not read_only:
            raise ValueError("an opened index descriptor is valid only for read-only archive access")
        # Attribute type declarations shared by every open mode (the
        # source-tier acquisition branch below returns early, so inference
        # from a single assignment site would otherwise mistype these).
        self._source_conn: sqlite3.Connection | None = None
        self._blob_publisher: ArchiveBlobPublisher | None = None
        self._pending_index_blob_receipts: list[tuple[str, bytes]] = []
        self._pending_raw_parse_states: list[tuple[str, RawSessionStateUpdate]] = []
        if getattr(self, "_source_tier_acquisition", False):
            # polylogue-gbs02: acquire-only mode validates the DURABLE tiers it
            # will write and never touches derived tiers. A stale or missing
            # durable tier is a hard refusal (writing through it would corrupt
            # irreplaceable evidence); a stale index/embeddings tier is exactly
            # the situation this mode exists for and is not checked here.
            for tier in (ArchiveTier.SOURCE, ArchiveTier.USER):
                spec = archive_tier_spec(tier)
                path = archive_root / spec.filename
                if not path.exists():
                    raise RuntimeError(f"source-tier acquisition refused: durable tier {spec.filename} is missing")
                with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=read_timeout)) as vconn:
                    current = int(vconn.execute("PRAGMA user_version").fetchone()[0])
                if current != spec.version:
                    raise RuntimeError(
                        f"source-tier acquisition refused: durable tier {spec.filename} "
                        f"user_version {current} != expected {spec.version}"
                    )
            self._conn = cast(sqlite3.Connection, _SourceTierOnlyIndexConnection())
            self._user_tier_attached = False
            self._tags_relation = "session_tags"
            self._blob_publisher = ArchiveBlobPublisher(self.source_db_path, self.archive_root / "blob")
            return
        if self._frozen_index_path is not None:
            self.index_db_path = self._frozen_index_path
        else:
            # The configured root owns the durable tiers, while an active
            # generation can keep index.db elsewhere. A writable open must
            # follow the same pointer as readiness and live ingest instead of
            # silently mutating a stale conventional root/index.db shadow.
            from polylogue.storage.archive_identity import resolve_active_index_path

            self.index_db_path = resolve_active_index_path(archive_root)
        if self._frozen_source_validation:
            # Candidate admission derives every decision from source.db and
            # frozen blob bytes. Requiring an index handle here would make the
            # derived tier being rebuilt a prerequisite for its own rebuild.
            self._conn = cast(
                sqlite3.Connection,
                _SourceTierOnlyIndexConnection("frozen source validation"),
            )
            self._user_tier_attached = False
            self._tags_relation = "session_tags"
            self._blob_publisher = _InactiveCandidateBlobPublisher(
                self.source_db_path,
                self.archive_root / "blob",
            )
            return
        if initialize:
            initialize_active_archive_root(archive_root)
        if read_only:
            self._conn = open_readonly_connection(
                self.index_db_path,
                timeout=read_timeout,
                opened_main_fd=opened_index_fd,
            )
            pragma_statements = READ_CONNECTION_PRAGMA_STATEMENTS
        else:
            self._conn = (
                sqlite3.connect(f"file:{self.index_db_path}?mode=rw", uri=True)
                if self._inactive_candidate_durable_read_only
                else sqlite3.connect(self.index_db_path)
            )
            pragma_statements = (
                BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS
                if bulk_build_profile
                else WRITE_CONNECTION_PRAGMA_STATEMENTS
            )
        self._conn.row_factory = sqlite3.Row
        for statement in pragma_statements:
            self._conn.execute(statement)
        if read_only:
            self._conn.execute(f"PRAGMA busy_timeout = {max(0, int(read_timeout * 1000))}")
        elif not self._pinned_read:
            # Fresh-bootstrap and same-version reopen both skip runtime-index
            # ensure elsewhere (initialize_archive_tier only replays DDL once,
            # at current_version==0. Owned inactive generations (bulk
            # rebuilds, revision backfill) open exactly this write connection
            # and nothing else, so without this call a generation could run
            # its whole lifetime — including prefix-tail dependent rewrites —
            # against an index.db missing performance indexes such as
            # idx_session_events_source_message /
            # idx_session_agent_policies_source_message (polylogue-crd8).
            ensure_runtime_indexes_sync(self._conn)
        self._user_tier_attached = False
        self._tags_relation = "session_tags"
        if not read_only:
            publisher_type = (
                _InactiveCandidateBlobPublisher if self._inactive_candidate_durable_read_only else ArchiveBlobPublisher
            )
            self._blob_publisher = publisher_type(self.source_db_path, self.archive_root / "blob")
        self._attach_user_tier_if_present()

    def _require_writable(self, operation: str) -> None:
        """Reject mutations before they can open or use a writable tier."""
        if self._read_only:
            raise ReadOnlyArchiveError(f"read-only archive evidence cannot {operation}")

    @classmethod
    def open_existing(
        cls,
        archive_root: Path,
        *,
        read_only: bool = True,
        read_timeout: float = 5.0,
        index_path: Path | None = None,
        opened_main_fd: int | None = None,
    ) -> ArchiveStore:
        """Open archive tier files.

        Read-only opens never bootstrap missing tiers; read/status surfaces must
        not create an empty archive and then report it as usable. Writers opt
        into bootstrap by passing ``read_only=False``. Read-only evidence tools
        may pass an already-resolved ``index_path`` to remain pinned to one
        physical generation across an active-pointer promotion. An opened main
        descriptor can additionally bind the index connection to that inode.
        """
        if index_path is not None and not read_only:
            raise ValueError("index_path is valid only for read-only archive access")
        initialize = not read_only
        return cls(
            archive_root,
            initialize=initialize,
            read_only=read_only,
            read_timeout=read_timeout,
            frozen_index_path=index_path,
            opened_index_fd=opened_main_fd,
        )

    @classmethod
    def open_source_tier_acquisition(cls, archive_root: Path) -> ArchiveStore:
        """Open a writer restricted to raw source-tier admission (polylogue-gbs02).

        Used by acquire-only degraded mode: durable tiers (source/user) must be
        current; derived tiers (index/embeddings) are never opened, so their
        schema state is irrelevant and cannot be corrupted. Only raw admission
        surfaces (``write_raw_payload``/``write_raw_blob_ref``/
        ``write_hook_event``) are usable; anything touching the index
        connection raises immediately.
        """
        return cls(archive_root, initialize=False, read_only=False, source_tier_acquisition=True)

    @classmethod
    def open_frozen_source_validation(
        cls,
        archive_root: Path,
        *,
        active_index_path: Path | None = None,
    ) -> ArchiveStore:
        """Open the live tiers without repairing or mutating any durable or pointer state."""
        return cls(
            archive_root,
            initialize=False,
            read_only=True,
            frozen_source_validation=True,
            frozen_index_path=active_index_path,
        )

    @classmethod
    def open_owned_inactive_generation(cls, archive_root: Path, *, generation_id: str, owner_id: str) -> ArchiveStore:
        """Open a typed inactive generation without weakening normal identity checks."""
        return cls(
            archive_root,
            initialize=True,
            read_only=False,
            owned_inactive_generation=(generation_id, owner_id),
        )

    @staticmethod
    def _needs_tier_bootstrap(archive_root: Path) -> bool:
        return any(
            not (archive_root / filename).exists()
            for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")
        )

    def set_read_progress_guard(self, guard: Callable[[], int], *, n_opcodes: int = 2000) -> None:
        """Install a SQLite progress handler on the index read connection.

        ``guard`` returning nonzero aborts the active statement with
        ``sqlite3.OperationalError: interrupted``. Execution control
        (polylogue-z9gh.1) uses this for cancellation/deadline enforcement;
        the store deliberately exposes only this narrow hook rather than the
        raw connection.
        """
        self._conn.set_progress_handler(guard, n_opcodes)

    def clear_read_progress_guard(self) -> None:
        """Remove the index connection's progress handler before ownership ends."""

        self._conn.set_progress_handler(None, 0)

    def begin_read_snapshot(self) -> None:
        """Begin the owned read transaction used by one controlled query call."""

        self._conn.execute("BEGIN")

    def end_read_snapshot(self) -> None:
        """Release the owned read snapshot without ever committing read work."""

        if self._conn.in_transaction:
            self._conn.rollback()

    def interrupt_reads(self) -> None:
        """Interrupt any statement active on the index read connection.

        Safe to call from another thread (``sqlite3.Connection.interrupt``
        is explicitly cross-thread callable).
        """
        self._conn.interrupt()

    def _optional_source_conn(self) -> sqlite3.Connection | None:
        """Return the source.db handle for evidence reads, or ``None``.

        Topology hook-evidence consultation is strictly additive: a store whose
        source tier is absent or unopenable (index-only harnesses, a read-only
        candidate without the durable tier staged) must behave exactly as it did
        before hook authority existed rather than fail a session write.
        """
        try:
            return self._ensure_source_conn()
        except sqlite3.Error:
            return None

    def _ensure_source_conn(self) -> sqlite3.Connection:
        """Return the persistent source.db connection, opening it lazily."""
        if self._source_conn is None:
            if self._read_only or self._inactive_candidate_durable_read_only:
                conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
                conn.execute("PRAGMA query_only = ON")
            else:
                conn = sqlite3.connect(self.source_db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            self._source_conn = conn
        return self._source_conn

    def _open_user_write_connection(self, *, initialize: bool = False) -> sqlite3.Connection:
        """Open user.db for mutation unless this store is an inactive candidate."""
        self._require_writable("mutate user.db")
        if self._inactive_candidate_durable_read_only:
            raise InactiveCandidateDurableWriteError(
                "inactive candidate generations may read frozen user assertions but may not mutate user.db"
            )
        if initialize:
            initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        return open_connection(self.user_db_path)

    def commit(self) -> None:
        """Commit index.db and any source transaction left by other callers.

        Raw ingest writes commit source references promptly to consume
        publication receipts; bulk cadence applies to the derived index.
        """
        self._require_writable("commit archive writes")
        if self._source_tier_acquisition:
            if self._source_conn is not None:
                self._source_conn.commit()
            return
        self._conn.commit()
        self._consume_index_blob_receipts()
        self._flush_pending_raw_parse_states()
        if self._source_conn is not None:
            self._source_conn.commit()

    def rollback(self) -> None:
        """Roll back the index.db and (if open) source.db write connections.

        Used by a bulk caller to discard an uncommitted, half-applied batch when
        a write raises, before propagating the error.
        """
        if self._source_tier_acquisition:
            if self._source_conn is not None:
                self._source_conn.rollback()
            return
        self._conn.rollback()
        self._pending_index_blob_receipts.clear()
        self._pending_raw_parse_states.clear()
        if self._source_conn is not None:
            self._source_conn.rollback()

    def close(self) -> None:
        if self._blob_publisher is not None:
            self._blob_publisher.discard_pending()
        if self._source_conn is not None:
            self._source_conn.close()
            self._source_conn = None
        self._conn.close()
        if self._active_writer_lease is not None:
            self._active_writer_lease.close()
            self._active_writer_lease = None

    def write_parsed(self, session: ParsedSession, *, content_hash: str | None = None) -> str:
        """Write a parsed session to index.db."""
        self._require_writable("write index.db")
        acquired, refs = self._preacquire_attachment_blobs(
            session,
            source_path=f"session:{session.provider_session_id}",
            acquired_at_ms=int(time.time() * 1000),
        )
        if self._blob_publisher is not None:
            self._blob_publisher.flush()
        session_id = write_parsed_session_to_archive(
            self._conn,
            session,
            content_hash=content_hash,
            preacquired_attachment_blobs=acquired,
            source_conn=self._optional_source_conn(),
        )
        self._pending_index_blob_receipts.extend(
            (ref.publication_receipt_id, ref.blob_hash) for ref in refs if ref.publication_receipt_id is not None
        )
        self._consume_index_blob_receipts()
        return session_id

    def write_parsed_result(self, session: ParsedSession, *, content_hash: str | None = None) -> dict[str, int]:
        """Write a parsed session and report whether precedence skipped it."""
        self._require_writable("write index.db")
        acquired, refs = self._preacquire_attachment_blobs(
            session,
            source_path=f"session:{session.provider_session_id}",
            acquired_at_ms=int(time.time() * 1000),
        )
        if self._blob_publisher is not None:
            self._blob_publisher.flush()
        outcomes: list[ArchiveWriteOutcome] = []
        write_parsed_session_to_archive(
            self._conn,
            session,
            content_hash=content_hash,
            preacquired_attachment_blobs=acquired,
            source_conn=self._optional_source_conn(),
            write_outcome=outcomes,
        )
        self._pending_index_blob_receipts.extend(
            (ref.publication_receipt_id, ref.blob_hash) for ref in refs if ref.publication_receipt_id is not None
        )
        self._consume_index_blob_receipts()
        stale_skipped = bool(outcomes and outcomes[0].stale_skipped)
        counts = self._skipped_counts(session) if stale_skipped else self._write_counts(session)
        counts["stale_skipped"] = int(stale_skipped)
        return counts

    def _consume_index_blob_receipts(self) -> None:
        """Consume receipts only after index attachment rows are committed."""
        if not self._pending_index_blob_receipts:
            return
        referenced: list[tuple[str, bytes]] = []
        retained: list[tuple[str, bytes]] = []
        for publication_id, blob_hash in self._pending_index_blob_receipts:
            row = self._conn.execute(
                "SELECT 1 FROM attachments WHERE blob_hash = ? LIMIT 1",
                (blob_hash,),
            ).fetchone()
            (referenced if row is not None else retained).append((publication_id, blob_hash))
        if referenced:
            source_conn = self._ensure_source_conn()
            with source_conn:
                from polylogue.storage.blob_publication import consume_blob_publication_receipt

                for publication_id, blob_hash in referenced:
                    consume_blob_publication_receipt(source_conn, publication_id, blob_hash)
        self._pending_index_blob_receipts = retained

    @staticmethod
    def _write_counts(session: ParsedSession) -> dict[str, int]:
        return {
            "sessions": 1,
            "messages": len(session.messages),
            "attachments": len(session.attachments),
            "session_events": len(session.session_events),
            "skipped_sessions": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
            "skipped_session_events": 0,
            "raw_links": 0,
        }

    @staticmethod
    def _skipped_counts(session: ParsedSession, *, session_events: int = 0) -> dict[str, int]:
        return {
            "sessions": 0,
            "messages": 0,
            "attachments": 0,
            "session_events": session_events,
            "skipped_sessions": 1,
            "skipped_messages": len(session.messages),
            "skipped_attachments": len(session.attachments),
            "skipped_session_events": len(session.session_events),
            "raw_links": 0,
        }

    def _preacquire_attachment_blobs(
        self,
        session: ParsedSession,
        *,
        source_path: str,
        acquired_at_ms: int,
    ) -> tuple[
        dict[int, tuple[bytes | None, int, str]],
        tuple[ArchiveSourceBlobRef, ...],
    ]:
        """Prepare inline attachment bytes before their durable transaction."""
        if self._blob_publisher is None:
            return {}, ()
        acquired: dict[int, tuple[bytes | None, int, str]] = {}
        refs: list[ArchiveSourceBlobRef] = []
        for attachment in session.attachments:
            if self._inactive_candidate_durable_read_only:
                if attachment.inline_bytes is not None:
                    hash_hex = hashlib.sha256(attachment.inline_bytes).hexdigest()
                    size = len(attachment.inline_bytes)
                elif attachment.precomputed_blob is not None:
                    hash_hex, size = attachment.precomputed_blob
                else:
                    continue
                blob_path = self._blob_publisher.blob_path(hash_hex)
                if not blob_path.is_file() or blob_path.stat().st_size != size:
                    raise InactiveCandidateDurableWriteError(
                        "inactive candidate requires attachment bytes to be present in the frozen blob namespace: "
                        f"{hash_hex}"
                    )
                with blob_path.open("rb") as handle:
                    stored_hash = hashlib.file_digest(handle, "sha256").hexdigest()
                if stored_hash != hash_hex:
                    raise InactiveCandidateDurableWriteError(
                        "inactive candidate found attachment bytes that do not match the frozen blob identity: "
                        f"{hash_hex}"
                    )
                acquired[id(attachment)] = (bytes.fromhex(hash_hex), size, "acquired")
                continue
            if attachment.inline_bytes is None:
                continue
            hash_hex, size = self._blob_publisher.write_from_bytes(attachment.inline_bytes)
            blob_hash = bytes.fromhex(hash_hex)
            acquired[id(attachment)] = (blob_hash, size, "acquired")
            refs.append(
                ArchiveSourceBlobRef(
                    blob_hash=blob_hash,
                    ref_type="attachment",
                    source_path=source_path,
                    size_bytes=size,
                    acquired_at_ms=acquired_at_ms,
                    publication_receipt_id=self._blob_publisher.receipt_id(hash_hex),
                )
            )
        return acquired, tuple(refs)

    def _write_parsed_precedence_result(
        self,
        session: ParsedSession,
        *,
        raw_id: str,
        source_index: int,
        stage_timings_s: dict[str, float] | None,
        stage_timing_prefix: str,
        manage_transaction: bool,
        preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]] | None = None,
        revision_authoritative: bool = False,
        bulk_fts: bool = False,
        bulk_build: bool = False,
        defer_fts_rebuild: bool = False,
    ) -> ArchiveRawParsedWriteResult:
        return _write_parsed_precedence_result(
            self,
            session,
            raw_id=raw_id,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            preacquired_attachment_blobs=preacquired_attachment_blobs,
            revision_authoritative=revision_authoritative,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            defer_fts_rebuild=defer_fts_rebuild,
        )

    def write_raw_and_parsed(
        self,
        session: ParsedSession,
        *,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        source_index: int = 0,
        raw_id: str | None = None,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
        blob_publication_receipt_id: str | None = None,
        finalize_raw_parse: bool = True,
    ) -> tuple[str, str]:
        self._require_writable("write source.db and index.db")
        return write_raw_and_parsed(
            self,
            session,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            blob_publication_receipt_id=blob_publication_receipt_id,
            finalize_raw_parse=finalize_raw_parse,
        )

    def write_raw_payload(
        self,
        *,
        provider: Provider,
        capture_mode: Provider | None = None,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        source_index: int = 0,
        raw_id: str | None = None,
        native_id: str | None = None,
        blob_publication_receipt_id: str | None = None,
        revision: RawRevisionEnvelope | None = None,
        post_parse: bool = False,
    ) -> str:
        self._require_writable("write source.db raw evidence")
        return write_raw_payload(
            self,
            provider=provider,
            capture_mode=capture_mode,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            native_id=native_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            revision=revision,
            post_parse=post_parse,
        )

    def raw_native_id(self, raw_id: str) -> str | None:
        return raw_native_id(self, raw_id)

    def write_hook_event(
        self,
        *,
        provider: Provider,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        hook_event: ArchiveHookEvent,
        source_index: int = 0,
        carrier_source_id: str | None = None,
        carrier_relative_path: str | None = None,
        carrier_role: str = "primary-writable",
    ) -> str:
        """Persist a hook event as session-linked evidence, not a session.

        Publishes the raw bytes (durable blob) and writes a ``raw_hook_events``
        row keyed to its parent session via ``hook_event.session_native_id`` —
        but NO ``raw_sessions`` row, so a hook can never materialize into a
        standalone empty session (polylogue-31r1).
        """
        self._require_writable("write source.db hook evidence")
        if self._blob_publisher is None:
            raise RuntimeError("raw archive writes require a writable archive publisher")
        raw_hash, _raw_size = self._blob_publisher.write_from_bytes(payload)
        receipt_id = self._blob_publisher.receipt_id(raw_hash)
        self._blob_publisher.flush()
        origin = origin_from_provider(provider)
        raw_id = deterministic_raw_session_id(
            origin,
            source_path,
            source_index,
            deterministic_blob_hash(payload),
            hook_event.native_id,
        )
        return write_source_hook_event(
            self._ensure_source_conn(),
            origin=origin,
            source_path=source_path,
            payload=payload,
            acquired_at_ms=acquired_at_ms,
            raw_id=raw_id,
            hook_event=hook_event,
            blob_publication_receipt_id=receipt_id,
            carrier_source_id=carrier_source_id or "representative-hook-source",
            carrier_relative_path=carrier_relative_path or source_path,
            carrier_role=carrier_role,
            manage_transaction=True,
        )

    def delete_hook_event(self, hook_event_id: str) -> bool:
        """Delete a hook event and its source-tier payload reference."""
        self._require_writable("delete source.db hook evidence")
        return delete_source_hook_event(self._ensure_source_conn(), hook_event_id)

    def write_raw_blob_ref(
        self,
        *,
        provider: Provider,
        capture_mode: Provider | None = None,
        blob_hash_hex: str,
        blob_size: int,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        source_index: int = 0,
        raw_id: str | None = None,
        blob_publication_receipt_id: str | None = None,
        revision: RawRevisionEnvelope | None = None,
        post_parse: bool = False,
    ) -> str:
        self._require_writable("write source.db blob reference")
        return write_raw_blob_ref(
            self,
            provider=provider,
            capture_mode=capture_mode,
            blob_hash_hex=blob_hash_hex,
            blob_size=blob_size,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
            revision=revision,
            post_parse=post_parse,
        )

    def record_raw_container_coordinate(
        self,
        raw_id: str,
        *,
        coordinate_format: Literal["zip-v2"],
        entry_ordinal: int,
        split_index: int,
    ) -> None:
        self._require_writable("record source.db container coordinate")
        record_raw_container_coordinate(
            self._ensure_source_conn(),
            raw_id,
            coordinate_format=coordinate_format,
            entry_ordinal=entry_ordinal,
            split_index=split_index,
        )

    def admit_raw_artifact_payload(
        self,
        *,
        provider: Provider,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        classification: ArtifactClassification,
        source_index: int = 0,
        raw_id: str | None = None,
        blob_publication_receipt_id: str | None = None,
    ) -> RawAdmissionResult:
        """Route a non-conversational artifact payload through the raw-admission chokepoint.

        See :func:`polylogue.storage.sqlite.archive_tiers.revision_governance.admit_raw_artifact_payload`.
        """
        self._require_writable("admit source.db artifact")
        return admit_raw_artifact_payload(
            self,
            provider=provider,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            classification=classification,
            source_index=source_index,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
        )

    def admit_raw_artifact_blob_ref(
        self,
        *,
        provider: Provider,
        blob_hash_hex: str,
        blob_size: int,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        classification: ArtifactClassification,
        source_index: int = 0,
        raw_id: str | None = None,
        blob_publication_receipt_id: str | None = None,
    ) -> RawAdmissionResult:
        """Route a prepublished non-conversational blob through typed admission."""
        self._require_writable("admit source.db artifact blob reference")
        return admit_raw_artifact_blob_ref(
            self,
            provider=provider,
            blob_hash_hex=blob_hash_hex,
            blob_size=blob_size,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            classification=classification,
            source_index=source_index,
            raw_id=raw_id,
            blob_publication_receipt_id=blob_publication_receipt_id,
        )

    def write_parsed_for_retained_raw(
        self,
        session: ParsedSession,
        *,
        raw_id: str,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
        finalize_raw_parse: bool = True,
        revision_authoritative: bool = False,
    ) -> tuple[str, str]:
        self._require_writable("write retained source.db and index.db evidence")
        return write_parsed_for_retained_raw(
            self,
            session,
            raw_id=raw_id,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            finalize_raw_parse=finalize_raw_parse,
            revision_authoritative=revision_authoritative,
        )

    def write_parsed_for_retained_raw_result(
        self,
        session: ParsedSession,
        *,
        raw_id: str,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
        finalize_raw_parse: bool = True,
        revision_authoritative: bool = False,
    ) -> ArchiveRawParsedWriteResult:
        self._require_writable("write retained source.db and index.db evidence")
        return write_parsed_for_retained_raw_result(
            self,
            session,
            raw_id=raw_id,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            finalize_raw_parse=finalize_raw_parse,
            revision_authoritative=revision_authoritative,
        )

    def bind_raw_revision(self, raw_id: str, revision: RawRevisionEnvelope, *, manage_transaction: bool = True) -> None:
        self._require_writable("bind source.db revision")
        return bind_raw_revision(self, raw_id, revision, manage_transaction=manage_transaction)

    def release_provisional_full_revisions(self, raw_ids: Sequence[str]) -> None:
        self._require_writable("release source.db revisions")
        return release_provisional_full_revisions(self, raw_ids)

    def raw_full_revision_generation(self, logical_source_key: str) -> int:
        return raw_full_revision_generation(self, logical_source_key)

    def raw_append_revision_parent(
        self,
        logical_source_key: str,
        start_offset: int,
        predecessor_revision: str | None,
    ) -> tuple[str, str, int] | None:
        return raw_append_revision_parent(self, logical_source_key, start_offset, predecessor_revision)

    def raw_membership_retired_full_revision_siblings(self, logical_source_key: str) -> tuple[str, ...]:
        return raw_membership_retired_full_revision_siblings(self, logical_source_key)

    def _raw_revision_source_path_has_divergent_evidence(self, logical_source_key: str) -> bool:
        return _raw_revision_source_path_has_divergent_evidence(self, logical_source_key)

    def classify_raw_revision_cohort_for_rebuild_repair(
        self,
        logical_source_key: str,
        *,
        manage_transaction: bool = True,
    ) -> RevisionReplayPlan:
        self._require_writable("classify source.db revision authority")
        return classify_raw_revision_cohort_for_rebuild_repair(
            self,
            logical_source_key,
            manage_transaction=manage_transaction,
        )

    def classify_raw_revision_cohort_for_frozen_candidate(self, logical_source_key: str) -> RevisionReplayPlan:
        return classify_raw_revision_cohort_for_frozen_candidate(self, logical_source_key)

    def require_frozen_membership_authority(
        self,
        logical_source_key: str,
        classification: MembershipClassification,
    ) -> None:
        require_frozen_membership_authority(
            self,
            logical_source_key,
            classification,
            membership_decisions_for_classification(classification),
        )

    def classify_raw_revision_cohort_for_live_watch(
        self,
        logical_source_key: str,
        *,
        manage_transaction: bool = True,
    ) -> RevisionReplayPlan:
        self._require_writable("classify source.db revision authority")
        return classify_raw_revision_cohort_for_live_watch(
            self,
            logical_source_key,
            manage_transaction=manage_transaction,
        )

    def classify_untyped_full_revision_groups(self, raw_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
        return classify_untyped_full_revision_groups(self, raw_ids)

    @staticmethod
    def _promote_contiguous_append_evidence(conn: sqlite3.Connection, logical_source_key: str) -> None:
        return _promote_contiguous_append_evidence(conn, logical_source_key)

    def _raw_revision_authority(self, raw_id: str) -> str | None:
        return _raw_revision_authority(self, raw_id)

    def raw_revision_replay_plan(self, logical_source_key: str) -> RevisionReplayPlan:
        return raw_revision_replay_plan(self, logical_source_key)

    def _raw_revision_candidates(self, logical_source_key: str) -> list[RevisionCandidate]:
        return _raw_revision_candidates(self, logical_source_key)

    def _authorize_full_snapshot_fold(
        self,
        *,
        existing_head: tuple[object, ...],
        full_candidate: RevisionCandidate,
        candidates: Mapping[str, RevisionCandidate],
    ) -> FullSnapshotFoldAuthorization | None:
        return _authorize_full_snapshot_fold(
            self, existing_head=existing_head, full_candidate=full_candidate, candidates=candidates
        )

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
        return raw_revision_descriptor(self, raw_id)

    @contextmanager
    def open_raw_revision_material(self, raw_id: str) -> Iterator[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
        with open_raw_revision_material(self, raw_id) as _governance_result:
            yield _governance_result

    def raw_revision_material(self, raw_id: str) -> tuple[Provider, bytes, str, RawRevisionKind]:
        return raw_revision_material(self, raw_id)

    def blob_path_for_hash(self, blob_hash: str) -> Path | None:
        return blob_path_for_hash(self, blob_hash)

    def _raw_revision_payload_digest_and_size(self, raw_id: str) -> tuple[str, int]:
        return _raw_revision_payload_digest_and_size(self, raw_id)

    def _raw_revision_matches_segments(self, full_raw_id: str, segment_raw_ids: Sequence[str]) -> bool:
        return _raw_revision_matches_segments(self, full_raw_id, segment_raw_ids)

    def unclassified_raw_revision_rows(self) -> tuple[tuple[str, int], ...]:
        return unclassified_raw_revision_rows(self)

    def pending_raw_revision_logical_keys(self) -> tuple[str, ...]:
        return pending_raw_revision_logical_keys(self)

    def raw_revision_rebuild_selection(
        self,
        raw_ids: list[str] | None,
    ) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
        return raw_revision_rebuild_selection(self, raw_ids)

    def raw_membership_census_rows(
        self, raw_ids: Sequence[str] | None = None
    ) -> tuple[tuple[str, int, bool, int], ...]:
        return raw_membership_census_rows(self, raw_ids)

    def raw_payload_sizes(self, raw_ids: Sequence[str]) -> dict[str, int]:
        return raw_payload_sizes(self, raw_ids)

    def replace_raw_membership_census(
        self,
        raw_id: str,
        sessions: list[ParsedSession] | None,
        *,
        parser_fingerprint: str,
        censused_at_ms: int,
        detail: str = "",
        retire_full_revision_governance: bool = False,
        manage_transaction: bool = True,
    ) -> None:
        self._require_writable("replace source.db membership census")
        return replace_raw_membership_census(
            self,
            raw_id,
            sessions,
            parser_fingerprint=parser_fingerprint,
            censused_at_ms=censused_at_ms,
            detail=detail,
            retire_full_revision_governance=retire_full_revision_governance,
            manage_transaction=manage_transaction,
        )

    def convertible_full_revision_raw_ids(self, logical_source_key: str) -> tuple[str, ...]:
        return convertible_full_revision_raw_ids(self, logical_source_key)

    def expand_raw_membership_selection(self, raw_ids: list[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return expand_raw_membership_selection(self, raw_ids)

    @staticmethod
    def raw_membership_selection_components_sync(
        conn: sqlite3.Connection,
        raw_ids: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        return raw_membership_selection_components_sync(conn, raw_ids)

    def raw_membership_selection_components(self, raw_ids: list[str]) -> tuple[tuple[str, ...], ...]:
        return raw_membership_selection_components(self, raw_ids)

    @staticmethod
    def expand_raw_membership_selection_sync(
        conn: sqlite3.Connection,
        raw_ids: list[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return expand_raw_membership_selection_sync(conn, raw_ids)

    def raw_membership_raw_ids(
        self,
        logical_source_key: str,
        *,
        include_complete_raw_id: str | None = None,
    ) -> tuple[str, ...]:
        return raw_membership_raw_ids(self, logical_source_key, include_complete_raw_id=include_complete_raw_id)

    def raw_revision_acquired_at_ms(self, raw_id: str) -> int:
        return raw_revision_acquired_at_ms(self, raw_id)

    def raw_revision_file_mtime(self, raw_id: str) -> str | None:
        return raw_revision_file_mtime(self, raw_id)

    def raw_revision_observed_at_ms(self, raw_id: str) -> int:
        return raw_revision_observed_at_ms(self, raw_id)

    def raw_revision_observation_order(self, raw_id: str) -> tuple[int, int]:
        return raw_revision_observation_order(self, raw_id)

    def raw_membership_rebuild_raw_ids(self, logical_source_key: str) -> tuple[str, ...]:
        return raw_membership_rebuild_raw_ids(self, logical_source_key)

    def raw_revision_head_raw_id(self, logical_source_key: str) -> str | None:
        return raw_revision_head_raw_id(self, logical_source_key)

    def raw_membership_authority_complete(self, raw_id: str) -> bool:
        return raw_membership_authority_complete(self, raw_id)

    def raw_membership_decision_pending(self, raw_id: str) -> bool:
        return raw_membership_decision_pending(self, raw_id)

    def raw_revision_replay_adoptable(self, sessions: Sequence[ParsedSession]) -> bool:
        return raw_revision_replay_adoptable(self, sessions)

    def defer_raw_revision_adoption(
        self,
        logical_source_key: str,
        raw_ids: Sequence[str],
        sessions: Sequence[ParsedSession],
    ) -> None:
        self._require_writable("defer source.db revision adoption")
        return defer_raw_revision_adoption(self, logical_source_key, raw_ids, sessions)

    def apply_raw_revision_replay(
        self,
        plan: RevisionReplayPlan,
        parsed_by_raw_id: dict[str, ParsedSession],
        *,
        acquired_at_ms: int,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "revision_replay",
        manage_transaction: bool = True,
        bulk_fts: bool = False,
        bulk_build: bool = False,
        defer_fts: bool = False,
        skip_already_applied: bool = False,
        prepared_by_raw_id: dict[str, PreparedSessionRows | Future[PreparedSessionRows]] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        self._require_writable("apply source.db revision replay")
        return apply_raw_revision_replay(
            self,
            plan,
            parsed_by_raw_id,
            acquired_at_ms=acquired_at_ms,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            defer_fts=defer_fts,
            skip_already_applied=skip_already_applied,
            prepared_by_raw_id=prepared_by_raw_id,
        )

    def apply_raw_membership_classification(
        self,
        logical_source_key: str,
        classification: MembershipClassification,
        parsed_by_raw_id: dict[str, ParsedSession],
        projections_by_raw_id: dict[str, SessionRevisionProjection],
        *,
        acquired_at_ms: int,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "membership_replay",
        manage_transaction: bool = True,
        bulk_fts: bool = False,
        bulk_build: bool = False,
        defer_fts: bool = False,
    ) -> str | None:
        self._require_writable("apply source.db membership classification")
        return apply_raw_membership_classification(
            self,
            logical_source_key,
            classification,
            parsed_by_raw_id,
            projections_by_raw_id,
            acquired_at_ms=acquired_at_ms,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            defer_fts=defer_fts,
        )

    def finalize_raw_parse_state(self, raw_id: str, *, state: RawSessionStateUpdate) -> None:
        self._require_writable("finalize source.db parse state")
        return finalize_raw_parse_state(self, raw_id, state=state)

    def mark_raw_parse_failed(
        self,
        raw_id: str,
        *,
        provider: Provider,
        error: BaseException,
        preserve_existing_failure_evidence: bool = False,
    ) -> None:
        self._require_writable("mark source.db parse failure")
        return mark_raw_parse_failed(
            self,
            raw_id,
            provider=provider,
            error=error,
            preserve_existing_failure_evidence=preserve_existing_failure_evidence,
        )

    def record_raw_failure_evidence(
        self,
        raw_id: str,
        *,
        provider: Provider,
        source_path: str,
        source_index: int,
        acquired_at_ms: int,
        kind: RawFailureEvidenceKind,
    ) -> None:
        self._require_writable("record source.db failure evidence")
        return record_raw_failure_evidence(
            self,
            raw_id,
            provider=provider,
            source_path=source_path,
            source_index=source_index,
            acquired_at_ms=acquired_at_ms,
            kind=kind,
        )

    def mark_raw_parse_succeeded(self, raw_id: str, *, provider: Provider) -> None:
        self._require_writable("mark source.db parse success")
        return mark_raw_parse_succeeded(self, raw_id, provider=provider)

    def _flush_pending_raw_parse_states(self) -> None:
        return _flush_pending_raw_parse_states(self)

    def _index_parsed_for_retained_raw(
        self,
        session: ParsedSession,
        *,
        raw_id: str,
        source_index: int,
        stage_timings_s: dict[str, float] | None,
        stage_timing_prefix: str,
        manage_transaction: bool,
        preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]],
        finalize_raw_parse: bool,
        revision_authoritative: bool = False,
        bulk_fts: bool = False,
        bulk_build: bool = False,
        defer_fts_rebuild: bool = False,
    ) -> ArchiveRawParsedWriteResult:
        return _index_parsed_for_retained_raw(
            self,
            session,
            raw_id=raw_id,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            preacquired_attachment_blobs=preacquired_attachment_blobs,
            finalize_raw_parse=finalize_raw_parse,
            revision_authoritative=revision_authoritative,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
            defer_fts_rebuild=defer_fts_rebuild,
        )

    @staticmethod
    def _raw_parse_success_state(provider: Provider) -> RawSessionStateUpdate:
        return _raw_parse_success_state(provider)

    @staticmethod
    def _raw_parse_failure_state(provider: Provider, exc: BaseException) -> RawSessionStateUpdate:
        return _raw_parse_failure_state(provider, exc)

    def write_raw_and_parsed_result(
        self,
        session: ParsedSession,
        *,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        source_index: int = 0,
        raw_id: str | None = None,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
        blob_publication_receipt_id: str | None = None,
        finalize_raw_parse: bool = True,
    ) -> ArchiveRawParsedWriteResult:
        self._require_writable("write source.db and index.db evidence")
        return write_raw_and_parsed_result(
            self,
            session,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            source_index=source_index,
            raw_id=raw_id,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            blob_publication_receipt_id=blob_publication_receipt_id,
            finalize_raw_parse=finalize_raw_parse,
        )

    def admit_raw_and_parsed_result(
        self,
        session: ParsedSession,
        *,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        file_mtime_ms: int | None = None,
        logical_source_key: str,
        source_index: int = 0,
        raw_id: str | None = None,
        shared_raw: bool = False,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "append",
        manage_transaction: bool = True,
        blob_publication_receipt_id: str | None = None,
        finalize_raw_parse: bool = True,
    ) -> ArchiveRawParsedWriteResult:
        """Write raw bytes through the raw-admission chokepoint, then index.

        See :func:`polylogue.storage.sqlite.archive_tiers.revision_governance.admit_raw_and_parsed_result`.
        Restricted to first-observation callers (no prior head exists for
        ``logical_source_key``); use :meth:`write_raw_and_parsed_result` for
        callers with revision-chain/dedup semantics of their own.
        """
        self._require_writable("admit source.db and index.db evidence")
        return admit_raw_and_parsed_result(
            self,
            session,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            file_mtime_ms=file_mtime_ms,
            logical_source_key=logical_source_key,
            source_index=source_index,
            raw_id=raw_id,
            shared_raw=shared_raw,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            blob_publication_receipt_id=blob_publication_receipt_id,
            finalize_raw_parse=finalize_raw_parse,
        )

    def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
        """Read a session envelope from index.db."""
        return read_archive_session_envelope(self._conn, session_id)

    def read_session_page(self, session_id: str, *, limit: int, offset: int) -> ArchiveSessionEnvelope:
        """Read a bounded ``[offset, offset + limit)`` page of a session's transcript.

        See ``read_archive_session_page`` for the bounding contract (ordinary
        sessions compose only the requested window; prefix-sharing lineage
        children fall back to full composition, matching the DB-backed
        reader's own established constraint).
        """
        return read_archive_session_page(self._conn, session_id, limit=limit, offset=offset)

    def has_prefix_lineage(self, session_id: str) -> bool:
        """Return whether a session's logical transcript inherits a prefix."""
        row = self._conn.execute(
            f"""
            SELECT 1
            FROM session_links
            WHERE src_session_id = ?
              AND inheritance = 'prefix-sharing'
              AND resolved_dst_session_id IS NOT NULL
              AND {topology_status_composes_sql()}
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return row is not None

    def session_lineage_edges(self, session_ids: Sequence[str]) -> dict[str, tuple[str | None, tuple[str, ...]]]:
        """Return ``(parent_session_id, child_session_ids)`` per requested id.

        Reuses the same ``sessions.parent_session_id`` column the
        ``lineage:id:`` predicate already filters sessions by (shared
        ``root_session_id``) to materialize the direct parent/child edges for
        one already-selected page of a lineage family, rather than performing
        a second unbounded recursive graph traversal (#z9gh.3). Only direct
        (one-hop) edges are returned; children outside ``session_ids`` are
        still discovered (the child query is unscoped by the input set), but
        parents outside ``session_ids`` are reported by id only, not hydrated.
        """
        if not session_ids:
            return {}
        ids = tuple(dict.fromkeys(session_ids))
        placeholders = ",".join("?" for _ in ids)
        parent_rows = self._conn.execute(
            f"SELECT session_id, parent_session_id FROM sessions WHERE session_id IN ({placeholders})",
            ids,
        ).fetchall()
        parent_by_id: dict[str, str | None] = {
            str(row["session_id"]): (str(row["parent_session_id"]) if row["parent_session_id"] else None)
            for row in parent_rows
        }
        child_rows = self._conn.execute(
            f"SELECT session_id, parent_session_id FROM sessions WHERE parent_session_id IN ({placeholders})",
            ids,
        ).fetchall()
        children_by_parent: dict[str, list[str]] = {}
        for row in child_rows:
            parent_id = str(row["parent_session_id"])
            children_by_parent.setdefault(parent_id, []).append(str(row["session_id"]))
        return {
            session_id: (
                parent_by_id.get(session_id),
                tuple(children_by_parent.get(session_id, ())),
            )
            for session_id in ids
        }

    def get_session_tree(self, session_id: str) -> list[ArchiveSessionEnvelope]:
        """Return the rooted archive session tree containing ``session_id``."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        root_session_id = self._root_session_id_for_tree(resolved_session_id)
        rows = self._conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id = ?
               OR root_session_id = ?
            ORDER BY
                CASE WHEN session_id = ? THEN 0 ELSE 1 END,
                COALESCE(sort_key_ms, created_at_ms, updated_at_ms),
                session_id
            """,
            (root_session_id, root_session_id, root_session_id),
        ).fetchall()
        return [read_archive_session_envelope(self._conn, str(row["session_id"])) for row in rows]

    def _root_session_id_for_tree(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT root_session_id, parent_session_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row["root_session_id"]:
            return str(row["root_session_id"])

        current_id = session_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            parent_row = self._conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (current_id,),
            ).fetchone()
            if parent_row is None or not parent_row["parent_session_id"]:
                return current_id
            current_id = str(parent_row["parent_session_id"])
        return session_id

    def raw_artifacts_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return raw acquisition surface rows for one archive session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return [], 0
        raw_row = self._conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (resolved_session_id,),
        ).fetchone()
        if raw_row is None or raw_row["raw_id"] is None or not self.source_db_path.exists():
            return [], 0
        raw_id = str(raw_row["raw_id"])
        source_conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            total = int(
                source_conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            )
            rows = source_conn.execute(
                """
                SELECT raw_id, origin, capture_mode, source_path, blob_size, acquired_at_ms,
                       parsed_at_ms, validation_status
                FROM raw_sessions
                WHERE raw_id = ?
                ORDER BY acquired_at_ms DESC, raw_id
                LIMIT ? OFFSET ?
                """,
                (raw_id, max(limit, 0), max(offset, 0)),
            ).fetchall()
        finally:
            source_conn.close()
        return [
            {
                "raw_id": str(row["raw_id"]),
                "origin": str(row["origin"]),
                "source_path": str(row["source_path"]),
                "blob_size": int(row["blob_size"] or 0),
                "acquired_at": _iso_from_ms(row["acquired_at_ms"]),
                "parsed_at": _iso_from_ms(row["parsed_at_ms"]),
                "validation_status": row["validation_status"],
            }
            for row in rows
        ], total

    def hook_event_summary_for_session(self, session_id: str) -> dict[str, object] | None:
        """Return per-event-type hook counts and timestamps for one session.

        Hook events (``PreToolUse``/``PostToolUse``/``UserPromptSubmit``/
        ``SessionStart``/...) are persisted as ``raw_hook_events`` rows keyed
        by ``(origin, session_native_id)`` -- never as a session of their own
        (polylogue-31r1). This is the read-model that surfaces them as
        evidence attached to the session that produced them, joining on the
        session's own ``origin``/``native_id`` rather than its ``raw_id``
        (hook rows have no ``raw_id`` link -- they attach to the *session*
        identity, not to any one raw acquisition record).
        """
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        row = self._conn.execute(
            "SELECT origin, native_id FROM sessions WHERE session_id = ?",
            (resolved_session_id,),
        ).fetchone()
        if row is None or not self.source_db_path.exists():
            return None
        origin = str(row["origin"])
        native_id = str(row["native_id"])
        source_conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            events = list_hook_events(source_conn, origin=origin, session_native_id=native_id)
        finally:
            source_conn.close()
        by_event_type: dict[str, int] = {}
        for event in events:
            by_event_type[event.event_type] = by_event_type.get(event.event_type, 0) + 1
        observed_ms = [event.observed_at_ms for event in events]
        return {
            "session_id": resolved_session_id,
            "total": len(events),
            "by_event_type": dict(sorted(by_event_type.items())),
            "first_observed_at": _iso_from_ms(min(observed_ms)) if observed_ms else None,
            "last_observed_at": _iso_from_ms(max(observed_ms)) if observed_ms else None,
        }

    def get_session_work_event_insights(self, session_id: str) -> list[SessionWorkEventInsight]:
        """Read archive work-event insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_work_event_insights(session_id=resolved_session_id)

    def list_session_work_event_insights(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        heuristic_label: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionWorkEventInsight]:
        """List archive work-event insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("we.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if heuristic_label is not None:
            where.append("we.work_event_type = ?")
            params.append(heuristic_label)
        # A work event with no reliable timestamp anywhere in its fallback
        # chain (COALESCE(...) IS NULL) is not evidence it falls outside a
        # since/until window -- include it rather than let SQL's NULL
        # propagation silently exclude it (polylogue-2seq, sort_key_ms
        # COALESCE audit).
        if since_ms is not None:
            where.append(
                "(COALESCE(we.started_at_ms, s.sort_key_ms) IS NULL OR COALESCE(we.started_at_ms, s.sort_key_ms) >= ?)"
            )
            params.append(since_ms)
        if until_ms is not None:
            where.append(
                "(COALESCE(we.started_at_ms, s.sort_key_ms) IS NULL OR COALESCE(we.started_at_ms, s.sort_key_ms) <= ?)"
            )
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT we.session_id, we.position
            FROM session_work_events we
            JOIN sessions s ON s.session_id = we.session_id
            {clause}
            ORDER BY COALESCE(we.started_at_ms, s.sort_key_ms) DESC, we.session_id, we.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        events_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionWorkEventInsight] = {}
        for event_session_id in events_by_session:
            materialization = _read_archive_materialization(self._conn, "work_events", event_session_id)
            session_origin = _session_origin(self._conn, event_session_id)
            for event in read_session_work_events(self._conn, session_id=event_session_id).values():
                if heuristic_label is None or event.work_event_type == heuristic_label:
                    indexed[(event.session_id, event.position)] = _work_event_insight_from_archive_row(
                        event,
                        origin=session_origin,
                        materialization=materialization,
                    )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_session_phase_insights(self, session_id: str) -> list[SessionPhaseInsight]:
        """Read archive phase insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_phase_insights(session_id=resolved_session_id)

    def list_session_phase_insights(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseInsight]:
        """List archive phase insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("sp.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        # A phase with no reliable timestamp anywhere in its fallback chain
        # (COALESCE(...) IS NULL) is not evidence it falls outside a
        # since/until window -- include it rather than let SQL's NULL
        # propagation silently exclude it (polylogue-2seq, sort_key_ms
        # COALESCE audit).
        if since_ms is not None:
            where.append(
                "(COALESCE(sp.started_at_ms, s.sort_key_ms) IS NULL OR COALESCE(sp.started_at_ms, s.sort_key_ms) >= ?)"
            )
            params.append(since_ms)
        if until_ms is not None:
            where.append(
                "(COALESCE(sp.started_at_ms, s.sort_key_ms) IS NULL OR COALESCE(sp.started_at_ms, s.sort_key_ms) <= ?)"
            )
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT sp.session_id, sp.position
            FROM session_phases sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY COALESCE(sp.started_at_ms, s.sort_key_ms) DESC, sp.session_id, sp.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        phases_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionPhaseInsight] = {}
        for phase_session_id in phases_by_session:
            materialization = _read_archive_materialization(self._conn, "phases", phase_session_id)
            session_origin = _session_origin(self._conn, phase_session_id)
            for phase in read_session_phases(self._conn, session_id=phase_session_id).values():
                indexed[(phase.session_id, phase.position)] = _phase_insight_from_archive_row(
                    phase,
                    origin=session_origin,
                    materialization=materialization,
                )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_thread_insight(self, thread_id: str) -> ThreadInsight | None:
        """Read one archive thread projection as a public thread insight."""
        row = self._conn.execute(
            "SELECT thread_id FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return self._thread_insight_from_id(str(row["thread_id"]))

    def list_thread_insights(
        self,
        *,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ThreadInsight]:
        """List threads as public thread insights."""
        where: list[str] = []
        params: list[object] = []
        if query:
            like = f"%{query.strip().lower()}%"
            where.append(
                """
                (
                    lower(t.thread_id) LIKE ?
                    OR EXISTS (
                        SELECT 1
                        FROM thread_sessions qts
                        JOIN sessions qs ON qs.session_id = qts.session_id
                        WHERE qts.thread_id = t.thread_id
                          AND (
                            lower(qs.session_id) LIKE ?
                            OR lower(COALESCE(qs.title, '')) LIKE ?
                            OR lower(COALESCE(qs.git_repository_url, '')) LIKE ?
                            OR lower(COALESCE(qs.git_branch, '')) LIKE ?
                          )
                    )
                )
                """.strip()
            )
            params.extend([like, like, like, like, like])
        if since_ms is not None:
            where.append("t.created_at_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("t.created_at_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT t.thread_id
            FROM threads t
            {clause}
            ORDER BY t.created_at_ms DESC, t.thread_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [insight for row in rows if (insight := self._thread_insight_from_id(str(row["thread_id"]))) is not None]

    def _thread_insight_from_id(self, thread_id: str) -> ThreadInsight | None:
        row = self._conn.execute(
            """
            SELECT thread_id, created_at_ms, session_count
            FROM threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        session_rows = self._conn.execute(
            """
            SELECT s.session_id, s.parent_session_id, s.origin, s.title,
                   s.message_count, s.word_count, s.tool_use_count,
                   s.created_at_ms, s.updated_at_ms, s.git_repository_url,
                   s.git_branch, sp.first_message_at, sp.last_message_at,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd)
                      FROM session_model_usage u WHERE u.session_id = s.session_id)
                     AS profile_total_cost_usd
            FROM thread_sessions ts
            JOIN sessions s ON s.session_id = ts.session_id
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            WHERE ts.thread_id = ?
            ORDER BY ts.position, s.sort_key_ms, s.session_id
            """,
            (thread_id,),
        ).fetchall()
        session_ids = tuple(str(session["session_id"]) for session in session_rows)
        origin_breakdown: dict[str, int] = {}
        for session in session_rows:
            session_origin = str(session["origin"])
            origin_breakdown[session_origin] = origin_breakdown.get(session_origin, 0) + 1
        start_ms = min(
            (
                timestamp_ms
                for session in session_rows
                if (
                    timestamp_ms := _profile_or_session_timestamp_ms(
                        session,
                        profile_column="first_message_at",
                        session_column="created_at_ms",
                    )
                )
                is not None
            ),
            default=None,
        )
        end_ms = max(
            (
                timestamp_ms
                for session in session_rows
                if (
                    timestamp_ms := _profile_or_session_timestamp_ms(
                        session,
                        profile_column="last_message_at",
                        session_column="updated_at_ms",
                    )
                )
                is not None
            ),
            default=None,
        )
        dominant_repo = _dominant_repo(session_rows)
        member_evidence = tuple(
            ThreadMemberEvidencePayload(
                session_id=str(session["session_id"]),
                parent_id=str(session["parent_session_id"]) if session["parent_session_id"] else None,
                role=_archive_thread_member_role(session, str(row["thread_id"])),
                depth=_thread_member_depth(session_rows, str(session["session_id"])),
                confidence=1.0,
                support_signals=_archive_thread_member_support_signals(session),
                evidence=_archive_thread_member_evidence(session, str(row["thread_id"]), index),
            )
            for index, session in enumerate(session_rows)
        )
        lineage_signals: tuple[str, ...] = ("archive_threads", "archive_thread_sessions")
        if any(session["parent_session_id"] is not None for session in session_rows):
            lineage_signals = (*lineage_signals, "explicit_lineage")
        payload = ThreadPayload(
            start_time=_iso_from_ms(start_ms),
            end_time=_iso_from_ms(end_ms),
            dominant_repo=dominant_repo,
            session_ids=session_ids,
            session_count=len(session_ids),
            depth=max((member.depth for member in member_evidence), default=0),
            branch_count=sum(1 for session in session_rows if session["parent_session_id"] is not None),
            total_messages=sum(int(session["message_count"] or 0) for session in session_rows),
            total_cost_usd=sum(float(session["profile_total_cost_usd"] or 0.0) for session in session_rows),
            wall_duration_ms=max(end_ms - start_ms, 0) if start_ms is not None and end_ms is not None else 0,
            origin_breakdown=origin_breakdown,
            confidence=1.0 if session_rows else 0.0,
            support_level=ConfidenceBand.STRONG if len(session_rows) > 1 else ConfidenceBand.MODERATE,
            support_signals=lineage_signals,
            member_evidence=member_evidence,
        )
        materialization = _read_archive_materialization(self._conn, "thread", thread_id)
        return ThreadInsight(
            thread_id=str(row["thread_id"]),
            root_id=str(row["thread_id"]),
            dominant_repo=dominant_repo,
            provenance=_archive_provenance(materialization),
            thread=payload,
        )

    def list_session_cost_insights(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionCostInsight]:
        """List archive session cost insights from sessions plus session_profiles."""
        if model is not None:
            return []
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            try:
                resolved_session_id = self.resolve_session_id(session_id)
            except KeyError:
                # Unknown session id: no cost insight exists. Returning [] lets
                # the daemon cost endpoint run its existence check and answer
                # 404 instead of surfacing this as an opaque 500.
                return []
            where.append("s.session_id = ?")
            params.append(resolved_session_id)
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.sort_key_ms,
                   (SELECT SUM(u.cost_credits) FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_credits,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_usd,
                   (SELECT CASE WHEN COUNT(u.model_name) = 0 THEN NULL WHEN COUNT(u.cost_usd) = COUNT(u.model_name) THEN 0 ELSE 1 END FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_is_estimated,
                   COALESCE((SELECT MAX(u.cost_provenance) FROM session_model_usage u WHERE u.session_id = s.session_id), CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END) AS cost_provenance,
                   (
                       SELECT smu.model_name
                       FROM session_model_usage smu
                       WHERE smu.session_id = s.session_id
                       ORDER BY smu.input_tokens + smu.output_tokens DESC, smu.model_name
                       LIMIT 1
                   ) AS model_name
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        canonical = session_usage_costs_for_connection(self._conn, [str(row["session_id"]) for row in rows])
        insights = [
            _session_cost_insight_from_archive_row(self._conn, row, canonical.get(str(row["session_id"])))
            for row in rows
        ]
        if status is not None:
            insights = [insight for insight in insights if insight.estimate.status == status]
        return insights

    def list_cost_rollup_insights(
        self,
        *,
        origin: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CostRollupInsight]:
        """Aggregate archive model-usage rows into public cost rollups."""
        origin = _origin_value(origin)
        where = ["s.sort_key_ms > 0"]
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)

        rows = self._conn.execute(
            f"""
            SELECT s.origin AS source_name,
                   u.model_name AS model_name,
                   COUNT(DISTINCT u.session_id) AS session_count,
                   COALESCE(SUM(u.cost_usd), CASE WHEN COUNT(DISTINCT u.model_name) = 1 THEN MAX(s.reported_cost_usd) ELSE 0.0 END, 0.0) AS stored_cost_usd,
                   COALESCE(SUM(u.cost_credits), 0.0) AS stored_credits,
                   COALESCE(SUM(u.input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(u.output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(u.cache_read_tokens), 0) AS cache_read_tokens,
                   COALESCE(SUM(u.cache_write_tokens), 0) AS cache_write_tokens,
                   COALESCE(SUM(
                       u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens
                   ), 0) AS total_tokens,
                   COALESCE(
                       COALESCE(u.cost_provenance, CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END),
                       'unknown'
                   ) AS cost_provenance,
                   MAX(s.updated_at_ms) AS source_updated_at,
                   MAX(s.sort_key_ms) AS source_sort_key
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            WHERE {" AND ".join(where)}
            GROUP BY s.origin,
                     u.model_name,
                     COALESCE(u.cost_provenance, CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' ELSE 'unknown' END)
            """,
            tuple(params),
        ).fetchall()
        no_usage_where = where + ["u.session_id IS NULL"]
        no_usage_rows = self._conn.execute(
            f"""
            SELECT s.origin AS source_name,
                   NULL AS model_name,
                   COUNT(DISTINCT s.session_id) AS session_count,
                   COALESCE(SUM(s.reported_cost_usd), 0.0) AS stored_cost_usd,
                   0.0 AS stored_credits,
                   0 AS input_tokens,
                   0 AS output_tokens,
                   0 AS cache_read_tokens,
                   0 AS cache_write_tokens,
                   0 AS total_tokens,
                   CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' ELSE 'unknown' END AS cost_provenance,
                   MAX(s.updated_at_ms) AS source_updated_at,
                   MAX(s.sort_key_ms) AS source_sort_key
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            LEFT JOIN session_model_usage u ON u.session_id = s.session_id
            WHERE {" AND ".join(no_usage_where)}
            GROUP BY s.origin, CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' ELSE 'unknown' END
            """,
            tuple(params),
        ).fetchall()

        grouped: dict[tuple[str, str | None], _CostRollupAccumulator] = {}
        materialized_at = datetime.now(UTC).isoformat()
        for row in [*rows, *no_usage_rows]:
            source_origin = str(row["source_name"] or "unknown")
            source_name = source_origin
            model_name = str(row["model_name"]) if row["model_name"] is not None else None
            normalized_model = _normalize_model(model_name) if model_name is not None else None
            if model is not None and model not in {model_name, normalized_model}:
                continue
            key = (source_name, normalized_model or model_name)
            session_count = int(row["session_count"] or 0)
            stored_cost_usd = float(row["stored_cost_usd"] or 0.0)
            stored_credits = float(row["stored_credits"] or 0.0)
            input_tokens = int(row["input_tokens"] or 0)
            output_tokens = int(row["output_tokens"] or 0)
            cache_read_tokens = int(row["cache_read_tokens"] or 0)
            cache_write_tokens = int(row["cache_write_tokens"] or 0)
            total_tokens = int(row["total_tokens"] or 0)
            provenance = str(row["cost_provenance"] or "unknown")

            usage = CostUsagePayload(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                total_tokens=total_tokens,
            )
            subscription_credits = stored_credits or float(
                compute_credit_cost(
                    normalized_model or "",
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                )
            )
            basis = CostBasisPayload(
                provider_reported_usd=stored_cost_usd if provenance in {"exact", "origin_reported"} else 0.0,
                catalog_priced_usd=stored_cost_usd if provenance in {"priced", "estimated"} else 0.0,
                subscription_equivalent_usd=subscription_credits,
            )

            entry = grouped.setdefault(
                key,
                _CostRollupAccumulator(
                    source_name=source_name,
                    model_name=model_name,
                    normalized_model=normalized_model,
                ),
            )
            entry.session_count += session_count
            if stored_cost_usd > 0 and provenance in {"exact", "origin_reported"}:
                status = "exact"
                confidence = 1.0
            elif stored_cost_usd > 0:
                status = "priced"
                confidence = 0.7 if provenance == "estimated" else 0.9
            else:
                status = "unavailable"
                confidence = 0.0
            entry.status_counts[status] = entry.status_counts.get(status, 0) + session_count
            if stored_cost_usd > 0:
                entry.priced_session_count += session_count
                entry.confidence_total += session_count * confidence
            else:
                entry.unavailable_session_count += session_count
            entry.basis = entry.basis.plus(basis)
            entry.usage = entry.usage.plus(usage)
            entry.total_usd += stored_cost_usd
            entry.note_source_updated_at(row["source_updated_at"])
            entry.note_sort_key(row["source_sort_key"])
            per_model_key = (model_name, normalized_model)
            prior_breakdown = entry.per_model.get(per_model_key)
            if prior_breakdown is None:
                entry.per_model[per_model_key] = CostModelBreakdown(
                    model_name=model_name,
                    normalized_model=normalized_model,
                    usage=usage,
                    basis=basis,
                    total_usd=stored_cost_usd,
                    session_count=session_count,
                )
            else:
                entry.per_model[per_model_key] = CostModelBreakdown(
                    model_name=model_name,
                    normalized_model=normalized_model,
                    usage=prior_breakdown.usage.plus(usage),
                    basis=prior_breakdown.basis.plus(basis),
                    total_usd=prior_breakdown.total_usd + stored_cost_usd,
                    session_count=prior_breakdown.session_count + session_count,
                )

        rollups: list[CostRollupInsight] = []
        for entry in grouped.values():
            rollups.append(
                CostRollupInsight(
                    origin=entry.source_name,
                    model_name=entry.model_name,
                    normalized_model=entry.normalized_model,
                    session_count=entry.session_count,
                    priced_session_count=entry.priced_session_count,
                    unavailable_session_count=entry.unavailable_session_count,
                    status_counts=dict(sorted(entry.status_counts.items())),
                    total_usd=entry.total_usd,
                    basis=entry.basis,
                    unavailable_reason_counts=(
                        {"no_tokens": entry.unavailable_session_count} if entry.unavailable_session_count else {}
                    ),
                    per_model_breakdown=tuple(
                        sorted(entry.per_model.values(), key=lambda item: item.total_usd, reverse=True)
                    ),
                    usage=entry.usage,
                    confidence=(
                        entry.confidence_total / entry.priced_session_count if entry.priced_session_count else None
                    ),
                    provenance=ArchiveInsightProvenance(
                        materializer_version=0,
                        materialized_at=materialized_at,
                        source_updated_at=_iso_from_ms(entry.source_updated_at_ms),
                        source_sort_key=entry.source_sort_key,
                    ),
                )
            )
        rollups.sort(key=lambda insight: insight.total_usd, reverse=True)
        if offset:
            rollups = rollups[offset:]
        if limit is not None:
            rollups = rollups[: max(int(limit), 0)]
        return rollups

    def list_usage_timeline_insights(
        self,
        *,
        origin: str | None = None,
        model: str | None = None,
        group_by: str = "month-origin-model",
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[UsageTimelineInsight]:
        """Aggregate provider usage and cost evidence by session-month buckets."""

        origin = _origin_value(origin)
        include_origin = group_by in {"month-origin", "month-origin-model"}
        include_model = group_by in {"month-model", "month-origin-model"}
        buckets: dict[tuple[str, str | None, str | None], _UsageTimelineAccumulator] = {}

        def key_for(bucket: str, source_name: str | None, model_name: str | None) -> tuple[str, str | None, str | None]:
            return (bucket, source_name if include_origin else None, model_name if include_model else None)

        event_scan_cutoff_ms: int | None = None
        skip_event_scan = False
        # The first-page cutoff optimization below reasons about events sorted
        # by a real occurred_at_ms/sort_key_ms timestamp; a genuinely timeless
        # event (both NULL, landing in the "unknown" bucket) doesn't fit that
        # ordering at all, and the heuristic's own cost_page probe excludes
        # timeless sessions, so it cannot see one coming. If it fired anyway,
        # the caller's event_scan_cutoff_ms branch below adds an unconditional
        # "e.occurred_at_ms IS NOT NULL" filter, silently dropping the very
        # "unknown" bucket rows this fix exists to preserve. Skip the
        # optimization entirely whenever a timeless event exists rather than
        # risk that.
        has_timeless_event = (
            limit is not None
            and offset == 0
            and limit > 0
            and bool(
                self._conn.execute(
                    """
                    SELECT 1 FROM session_provider_usage_events e
                    JOIN sessions s ON s.session_id = e.session_id
                    WHERE e.occurred_at_ms IS NULL AND s.sort_key_ms IS NULL
                    LIMIT 1
                    """
                ).fetchone()
            )
        )
        if limit is not None and offset == 0 and limit > 0 and not has_timeless_event:
            event_scan_cutoff_ms, skip_event_scan = self._usage_timeline_event_scan_cutoff_ms(
                origin=origin,
                model=model,
                group_by=group_by,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
            )

        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if model is not None:
            where.append("e.model_name = ?")
            params.append(model)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        if event_scan_cutoff_ms is not None:
            where.append("e.occurred_at_ms IS NOT NULL")
            where.append("e.occurred_at_ms < ?")
            params.append(event_scan_cutoff_ms)
        event_rows = []
        if not skip_event_scan:
            where_clause = " AND ".join(where) if where else "1=1"
            event_rows = self._conn.execute(
                f"""
                SELECT CASE WHEN COALESCE(e.occurred_at_ms, s.sort_key_ms) IS NULL THEN 'unknown'
                            ELSE strftime('%Y-%m', COALESCE(e.occurred_at_ms, s.sort_key_ms)/1000, 'unixepoch')
                       END AS bucket,
                       s.origin AS source_name,
                       COALESCE(e.model_name, '') AS model_name,
                       COUNT(*) AS event_count,
                       COUNT(DISTINCT e.session_id) AS session_count,
                       COALESCE(SUM(e.last_input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(e.last_output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(e.last_cached_input_tokens), 0) AS cache_read_tokens,
                       COALESCE(SUM(e.last_cache_write_tokens), 0) AS cache_write_tokens,
                       COALESCE(SUM(e.last_total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(e.last_reasoning_output_tokens), 0) AS reasoning_output_tokens,
                       MAX(COALESCE(e.occurred_at_ms, s.sort_key_ms)) AS source_sort_key
                FROM session_provider_usage_events e
                JOIN sessions s ON s.session_id = e.session_id
                WHERE {where_clause}
                GROUP BY bucket, s.origin, model_name
                """,
                tuple(params),
            ).fetchall()

        for row in event_rows:
            bucket = str(row["bucket"])
            source_name = str(row["source_name"] or "unknown")
            model_name = str(row["model_name"] or "unknown")
            item = buckets.setdefault(
                key_for(bucket, source_name, model_name),
                _UsageTimelineAccumulator(
                    bucket=bucket,
                    source_name=source_name if include_origin else None,
                    model_name=model_name if include_model else None,
                ),
            )
            item.event_count += int(row["event_count"] or 0)
            item.event_session_count += int(row["session_count"] or 0)
            item.usage = item.usage.plus(
                CostUsagePayload(
                    input_tokens=int(row["input_tokens"] or 0),
                    output_tokens=int(row["output_tokens"] or 0),
                    cache_read_tokens=int(row["cache_read_tokens"] or 0),
                    cache_write_tokens=int(row["cache_write_tokens"] or 0),
                    total_tokens=int(row["total_tokens"] or 0),
                )
            )
            item.reasoning_output_tokens += int(row["reasoning_output_tokens"] or 0)
            item.note_sort_key(row["source_sort_key"])

        # No longer excludes timeless sessions (was "s.sort_key_ms > 0", which
        # silently dropped their cost/usage from every bucket forever, not
        # just under a since/until window -- polylogue-rvtu). The bucket
        # expression below routes such rows to an explicit "unknown" bucket.
        cost_where: list[str] = []
        cost_params: list[object] = []
        if origin is not None:
            cost_where.append("s.origin = ?")
            cost_params.append(origin)
        if model is not None:
            cost_where.append("u.model_name = ?")
            cost_params.append(model)
        if since_ms is not None:
            cost_where.append("s.sort_key_ms >= ?")
            cost_params.append(since_ms)
        if until_ms is not None:
            cost_where.append("s.sort_key_ms <= ?")
            cost_params.append(until_ms)
        where_clause = " AND ".join(cost_where) if cost_where else "1=1"
        cost_rows = self._conn.execute(
            f"""
            SELECT CASE WHEN s.sort_key_ms IS NULL THEN 'unknown'
                        ELSE strftime('%Y-%m', s.sort_key_ms/1000, 'unixepoch')
                   END AS bucket,
                   s.origin AS source_name,
                   COALESCE(u.model_name, '') AS model_name,
                   COUNT(DISTINCT u.session_id) AS session_count,
                   COALESCE(SUM(u.cost_usd), 0.0) AS stored_cost_usd,
                   COALESCE(SUM(u.cost_credits), 0.0) AS stored_credits,
                   COALESCE(SUM(u.input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(u.output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(u.cache_write_tokens), 0) AS cache_write_tokens,
                   COALESCE(u.cost_provenance, 'unknown') AS cost_provenance,
                   MAX(s.sort_key_ms) AS source_sort_key
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            WHERE {where_clause}
            GROUP BY bucket, s.origin, model_name, cost_provenance
            """,
            tuple(cost_params),
        ).fetchall()
        for row in cost_rows:
            bucket = str(row["bucket"])
            source_name = str(row["source_name"] or "unknown")
            model_name = str(row["model_name"] or "unknown")
            item = buckets.setdefault(
                key_for(bucket, source_name, model_name),
                _UsageTimelineAccumulator(
                    bucket=bucket,
                    source_name=source_name if include_origin else None,
                    model_name=model_name if include_model else None,
                ),
            )
            item.stored_cost_usd += float(row["stored_cost_usd"] or 0.0)
            item.subscription_credits += float(row["stored_credits"] or 0.0)
            if not float(row["stored_credits"] or 0.0):
                item.subscription_credits += compute_credit_cost(
                    _normalize_model(str(row["model_name"] or "")),
                    int(row["input_tokens"] or 0),
                    int(row["output_tokens"] or 0),
                    0,
                    int(row["cache_write_tokens"] or 0),
                )
            provenance = str(row["cost_provenance"] or "unknown")
            item.cost_provenance_counts[provenance] = item.cost_provenance_counts.get(provenance, 0) + int(
                row["session_count"] or 0
            )
            item.note_sort_key(row["source_sort_key"])

        materialized_at = datetime.now(UTC).isoformat()
        rows: list[UsageTimelineInsight] = []
        for item in buckets.values():
            timeline_model_name: str | None = item.model_name
            cost_session_count = sum(item.cost_provenance_counts.values())
            rows.append(
                UsageTimelineInsight(
                    group_by=group_by,
                    bucket=item.bucket,
                    origin=item.source_name,
                    model_name=timeline_model_name,
                    normalized_model=_normalize_model(timeline_model_name) if timeline_model_name else None,
                    session_count=max(cost_session_count, item.event_session_count),
                    event_count=item.event_count,
                    usage=item.usage,
                    reasoning_output_tokens=item.reasoning_output_tokens,
                    stored_cost_usd=item.stored_cost_usd,
                    subscription_credits=item.subscription_credits,
                    cost_provenance_counts=dict(sorted(item.cost_provenance_counts.items())),
                    provenance=ArchiveInsightProvenance(
                        materializer_version=0,
                        materialized_at=materialized_at,
                        source_updated_at=None,
                        source_sort_key=item.source_sort_key,
                    ),
                )
            )
        rows.sort(key=lambda insight: (insight.bucket, insight.origin or "", insight.normalized_model or ""))
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[: max(int(limit), 0)]
        return rows

    def _usage_timeline_event_scan_cutoff_ms(
        self,
        *,
        origin: str | None,
        model: str | None,
        group_by: str,
        since_ms: int | None,
        until_ms: int | None,
        limit: int,
    ) -> tuple[int | None, bool]:
        """Return an event scan upper bound for first-page timeline reads.

        The timeline is sorted ascending by bucket/origin/model. When the first
        page is fully determined by cheap session_model_usage rows that all sort
        before the first provider usage event, scanning the multi-million-row
        provider event table is avoidable. If provider events may affect the
        first page, return a bucket-end cutoff so the event leg can still use
        the occurred_at_ms runtime index instead of scanning the whole table.
        """

        include_origin = group_by in {"month-origin", "month-origin-model"}
        include_model = group_by in {"month-model", "month-origin-model"}
        group_columns = ["bucket"]
        if include_origin:
            group_columns.append("source_name")
        if include_model:
            group_columns.append("model_name")
        where = ["s.sort_key_ms > 0"]
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if model is not None:
            where.append("u.model_name = ?")
            params.append(model)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        cost_page = self._conn.execute(
            f"""
            SELECT strftime('%Y-%m', s.sort_key_ms/1000, 'unixepoch') AS bucket,
                   s.origin AS source_name,
                   COALESCE(u.model_name, '') AS model_name
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            WHERE {" AND ".join(where)}
            GROUP BY {", ".join(group_columns)}
            ORDER BY bucket, source_name, model_name
            LIMIT ?
            """,
            (*params, max(int(limit), 0)),
        ).fetchall()
        if len(cost_page) < limit:
            return None, False

        last_bucket = str(cost_page[-1]["bucket"])
        cutoff_ms = _month_bucket_end_ms(last_bucket)
        event_where = ["e.occurred_at_ms IS NOT NULL"]
        event_params: list[object] = []
        if origin is not None:
            event_where.append("s.origin = ?")
            event_params.append(origin)
        if model is not None:
            event_where.append("e.model_name = ?")
            event_params.append(model)
        if since_ms is not None:
            event_where.append("COALESCE(e.occurred_at_ms, s.sort_key_ms) >= ?")
            event_params.append(since_ms)
        if until_ms is not None:
            event_where.append("COALESCE(e.occurred_at_ms, s.sort_key_ms) <= ?")
            event_params.append(until_ms)
        first_event = self._conn.execute(
            f"""
            SELECT e.occurred_at_ms
            FROM session_provider_usage_events e
            JOIN sessions s ON s.session_id = e.session_id
            WHERE {" AND ".join(event_where)}
            ORDER BY e.occurred_at_ms
            LIMIT 1
            """,
            tuple(event_params),
        ).fetchone()
        if first_event is None:
            return cutoff_ms, True
        first_event_ms = int(first_event["occurred_at_ms"] or 0)
        if first_event_ms >= cutoff_ms:
            return cutoff_ms, True
        return cutoff_ms, False

    def list_archive_debt_insights(
        self,
        *,
        category: str | None = None,
        only_actionable: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveDebtInsight]:
        """Report consistency debt."""
        insights = [
            _archive_messages_fts_debt(self._conn),
            _archive_profile_rows_debt(self._conn),
            _archive_profile_counts_debt(self._conn),
            _archive_materialization_debt(self._conn),
            _archive_source_raw_link_debt(self.index_db_path, self.source_db_path),
            _archive_user_overlay_debt(self.index_db_path, self.user_db_path),
        ]
        insights.sort(key=lambda insight: (insight.category, insight.debt_name))
        if category is not None:
            insights = [insight for insight in insights if insight.category == category]
        if only_actionable:
            insights = [insight for insight in insights if not insight.healthy]
        if offset:
            insights = insights[offset:]
        if limit is not None:
            insights = insights[: max(int(limit), 0)]
        return insights

    def get_session_latency_profile_insight(self, session_id: str) -> SessionLatencyProfileInsight | None:
        """Project one latency profile from timestamped messages."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        row = self._conn.execute(
            """
            SELECT session_id, origin, title, sort_key_ms
            FROM sessions
            WHERE session_id = ?
            """,
            (resolved_session_id,),
        ).fetchone()
        return None if row is None else _session_latency_profile_from_archive_row(self._conn, row)

    def list_session_latency_profile_insights(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        only_stuck: bool = False,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionLatencyProfileInsight]:
        """Project archive latency profiles from sessions plus timestamped messages."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("s.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.sort_key_ms
            FROM sessions s
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        insights = [_session_latency_profile_from_archive_row(self._conn, row) for row in rows]
        if only_stuck:
            insights = [insight for insight in insights if insight.latency.stuck_tool_count > 0]
        return insights

    def find_stuck_session_latency_profile_insights(
        self,
        *,
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
    ) -> list[SessionLatencyProfileInsight]:
        """Return archive latency profiles with stuck tools.

        currently lacks session event start/end pairs, so stuck
        tool detection remains conservative and this returns only profiles
        whose projected stuck count is non-zero.
        """
        return self.list_session_latency_profile_insights(
            origin=origin,
            only_stuck=True,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=0,
        )

    def _fetch_session_profile_row(self, session_id: str) -> sqlite3.Row | None:
        """Resolve *session_id* and fetch its joined session/profile row, or None."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        rows = self._conn.execute(
            """
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_method,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_usd,
                   (SELECT CASE WHEN COUNT(u.model_name) = 0 THEN NULL WHEN COUNT(u.cost_usd) = COUNT(u.model_name) THEN 0 ELSE 1 END FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_is_estimated,
                   COALESCE((SELECT MAX(u.cost_provenance) FROM session_model_usage u WHERE u.session_id = s.session_id), CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END) AS cost_provenance,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            WHERE sp.session_id = ?
            """,
            (resolved_session_id,),
        ).fetchall()
        return rows[0] if rows else None

    def get_session_profile_insight(self, session_id: str, *, tier: str = "merged") -> SessionProfileInsight | None:
        """Read one archive session profile insight."""
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_insight_from_archive_row(self._conn, row, tier=tier)

    def get_session_profile_record(self, session_id: str) -> SessionProfileRecord | None:
        """Read one archive session profile as a domain :class:`SessionProfileRecord`.

        Mirrors :meth:`get_session_profile_insight` but rehydrates the full
        record needed by ``hydrate_session_profile`` (domain ``SessionProfile``)
        and the provenance-based staleness check. The materialization HWM
        provenance is pulled from ``read_insight_materialization`` so the
        downstream ``is_stale`` comparison is grounded in the same source the
        daemon's ``/insights`` profile panel consumes.

        Returns ``None`` when the session id does not resolve or has no
        materialized profile.
        """
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_record_from_archive_row(self._conn, row)

    def list_session_profile_insights(
        self,
        *,
        origin: str | None = None,
        workflow_shape: str | None = None,
        terminal_state: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        min_wallclock_seconds: float | None = None,
        max_wallclock_seconds: float | None = None,
        sort: str | None = None,
    ) -> list[SessionProfileInsight]:
        """List archive session profile insights.

        ``min_wallclock_seconds`` / ``max_wallclock_seconds`` filter on the
        session's message-timestamp span (last minus first message), and
        ``sort='wallclock'`` orders by that span descending.
        """
        # Wallclock span = newest minus oldest message timestamp for the session.
        wall_expr = (
            "(SELECT MAX(m.occurred_at_ms) - MIN(m.occurred_at_ms) "
            "FROM messages m WHERE m.session_id = s.session_id AND m.occurred_at_ms IS NOT NULL)"
        )
        where: list[str] = []
        params: list[object] = []
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if workflow_shape is not None:
            where.append("sp.workflow_shape = ?")
            params.append(workflow_shape)
        if terminal_state is not None:
            where.append("sp.terminal_state = ?")
            params.append(terminal_state)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        if first_message_since is not None:
            where.append("sp.first_message_at >= ?")
            params.append(first_message_since)
        if first_message_until is not None:
            where.append("sp.first_message_at <= ?")
            params.append(first_message_until)
        if session_date_since is not None:
            where.append("sp.canonical_session_date >= date(?)")
            params.append(session_date_since)
        if session_date_until is not None:
            where.append("sp.canonical_session_date <= date(?)")
            params.append(session_date_until)
        if min_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) >= ?")
            params.append(int(min_wallclock_seconds * 1000))
        if max_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) <= ?")
            params.append(int(max_wallclock_seconds * 1000))
        clause = "WHERE " + " AND ".join(where) if where else ""
        order_by = f"{wall_expr} DESC, s.session_id" if sort == "wallclock" else "s.sort_key_ms DESC, s.session_id"
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_method,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_usd,
                   (SELECT CASE WHEN COUNT(u.model_name) = 0 THEN NULL WHEN COUNT(u.cost_usd) = COUNT(u.model_name) THEN 0 ELSE 1 END FROM session_model_usage u WHERE u.session_id = s.session_id) AS cost_is_estimated,
                   COALESCE((SELECT MAX(u.cost_provenance) FROM session_model_usage u WHERE u.session_id = s.session_id), CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END) AS cost_provenance,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY {order_by}
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_session_profile_insight_from_archive_row(self._conn, row, tier=tier) for row in rows]

    def read_summary(self, session_id: str) -> ArchiveSessionSummary:
        """Read one session summary by exact session id."""
        row = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.parent_session_id,
                   s.session_kind,
                   s.message_count, s.word_count, s.reported_duration_ms,
                   s.tool_use_count, s.thinking_count, s.paste_count,
                   s.user_message_count, s.authored_user_message_count,
                   s.assistant_message_count, s.system_message_count,
                   s.tool_message_count, s.user_word_count, s.authored_user_word_count,
                   s.assistant_word_count,
                   s.title_source, s.title_ref, s.title_confidence, s.git_branch, s.git_repository_url, s.provider_project_ref,
                   s.display_name,
                   sp.terminal_state,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS total_cost_usd,
                   COALESCE((SELECT MAX(u.cost_provenance) FROM session_model_usage u WHERE u.session_id = s.session_id), CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END) AS cost_provenance,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            WHERE s.session_id = ?
            GROUP BY s.session_id
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _summary_from_row(row, self._conn)

    def resolve_session_id(self, token: str) -> str:
        """Resolve an exact or prefix session id token."""
        exact = self._conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (token,),
        ).fetchone()
        if exact is not None:
            return str(exact["session_id"])
        if ":" in token:
            provider_token, native_id = token.split(":", 1)
            origin_id = f"{origin_from_provider(Provider.from_string(provider_token)).value}:{native_id}"
            exact = self._conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (origin_id,),
            ).fetchone()
            if exact is not None:
                return str(exact["session_id"])
        lower_bound, upper_bound = session_id_prefix_bounds(token)
        where = "session_id >= ?"
        params: list[str] = [lower_bound]
        if upper_bound is not None:
            where = f"{where} AND session_id < ?"
            params.append(upper_bound)
        rows = self._conn.execute(
            f"""
            SELECT session_id
            FROM sessions
            WHERE {where}
            ORDER BY session_id
            LIMIT 2
            """,
            tuple(params),
        ).fetchall()
        if not rows:
            # Suffix fallback: a bare native id (e.g. the UUID that appears as
            # the session's source filename, ``1944721d-...``), full or a
            # prefix of it, resolves to the stored ``<origin>:<native_id>``.
            # Try the EXACT native id first (no trailing ``%``): in an
            # archive with sibling native ids where one is a prefix of
            # another (``abc`` and ``abcd``), an exact lookup for ``abc``
            # must still return only the ``abc`` row, not raise ambiguous
            # just because ``abcd`` also matches the widened prefix pattern
            # below (#2626 review). Only fall through to the prefix-widened
            # (trailing ``%``) match -- which allows a truncated prefix to
            # resolve, #7q16 -- when the exact lookup finds nothing. The
            # leading ``:`` anchors both patterns to right after the origin
            # separator so neither can match mid-native-id. Provider native
            # ids are globally unique, so a single match is unambiguous;
            # multiple matches raise just like the prefix path rather than
            # guessing.
            if ":" not in token:
                like_token = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                exact_suffix_rows = self._conn.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE session_id LIKE '%:' || ? ESCAPE '\\'
                    ORDER BY session_id
                    LIMIT 2
                    """,
                    (like_token,),
                ).fetchall()
                if len(exact_suffix_rows) == 1:
                    return str(exact_suffix_rows[0]["session_id"])
                if len(exact_suffix_rows) > 1:
                    raise ValueError(f"session id suffix {token!r} is ambiguous")
                suffix_rows = self._conn.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE session_id LIKE '%:' || ? || '%' ESCAPE '\\'
                    ORDER BY session_id
                    LIMIT 2
                    """,
                    (like_token,),
                ).fetchall()
                if len(suffix_rows) == 1:
                    return str(suffix_rows[0]["session_id"])
                if len(suffix_rows) > 1:
                    raise ValueError(f"session id prefix {token!r} is ambiguous")
            raise KeyError(token)
        if len(rows) > 1:
            raise ValueError(f"session id prefix {token!r} is ambiguous")
        return str(rows[0]["session_id"])

    def resolve_exact_session_ids(
        self,
        session_ids: Sequence[str],
        *,
        page_size: int = 256,
    ) -> dict[str, str]:
        """Resolve canonical session IDs in bounded set-oriented SQLite reads.

        This intentionally handles only exact stored IDs. Callers that accept
        abbreviated or provider-alias IDs retain ``resolve_session_id`` as the
        compatibility fallback for the unresolved subset.
        """
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        resolved: dict[str, str] = {}
        for offset in range(0, len(session_ids), page_size):
            page = session_ids[offset : offset + page_size]
            if not page:
                continue
            placeholders = ", ".join("?" for _ in page)
            rows = self._conn.execute(
                f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})",
                tuple(page),
            ).fetchall()
            resolved.update({str(row["session_id"]): str(row["session_id"]) for row in rows})
        return resolved

    def search_blocks(self, query: str) -> list[str]:
        """Search indexed block text and return block ids."""
        return search_archive_blocks(self._conn, query)

    def rebuild_index(self) -> int:
        """Rebuild the block FTS index from index.db blocks."""
        self._require_writable("rebuild index.db")
        rebuilt_rows = rebuild_archive_messages_fts(self._conn)
        self._conn.commit()
        return rebuilt_rows

    def index_status(self) -> IndexStatus:
        """Return ``{exists, count}`` for the archive block FTS index.

        The block FTS index (``messages_fts`` over ``blocks``) is trigger-maintained, so a
        missing table means it was never built and the count is the
        indexed-block total.
        """
        if not _table_exists(self._conn, "messages_fts"):
            return IndexStatus(exists=False, count=0)
        return IndexStatus(exists=True, count=_count_scalar(self._conn, "SELECT COUNT(*) FROM messages_fts"))

    def add_user_tags(
        self,
        session_ids: tuple[str, ...],
        tags: tuple[str, ...],
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> int:
        """Add user tag assertions to archive user.db and return changed count."""
        changed = 0
        user_conn = self._open_user_write_connection(initialize=True)
        user_conn.row_factory = sqlite3.Row
        try:
            with user_conn:
                for session_id in tuple(
                    dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)
                ):
                    for tag in tags:
                        normalized_tag = tag.strip().lower()
                        if not normalized_tag:
                            raise ValueError("tag cannot be empty")
                        existing = read_assertion_envelope(
                            user_conn,
                            assertion_id_for_session_tag(session_id, normalized_tag, "user"),
                        )
                        if existing is not None and existing.status != "deleted":
                            continue
                        changed += 1
                        upsert_session_tag_assertion(
                            user_conn,
                            session_id=session_id,
                            tag=normalized_tag,
                            tag_source="user",
                            method="cli",
                            author_ref=author_ref,
                            author_kind=author_kind,
                            evidence={"source": "archive_query"},
                        )
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return changed

    def remove_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
        """Mark user tag assertions deleted and return deleted row count."""
        self._require_writable("delete user.db tags")
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids or not self.user_db_path.exists():
            return 0
        removed = 0
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                for session_id in resolved_session_ids:
                    for tag in tags:
                        normalized_tag = tag.strip().lower()
                        if not normalized_tag:
                            raise ValueError("tag cannot be empty")
                        assertion_id = assertion_id_for_session_tag(session_id, normalized_tag, "user")
                        assertion = read_assertion_envelope(user_conn, assertion_id)
                        if assertion is None or assertion.status == "deleted":
                            continue
                        if mark_assertion_status(user_conn, assertion_id, "deleted"):
                            removed += 1
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return removed

    def list_user_tags(self, *, origin: str | None = None) -> dict[str, int]:
        """Return user tag counts over archive sessions."""
        where = "WHERE st.tag_source = 'user'"
        params: list[object] = []
        if origin is not None:
            where += " AND s.origin = ?"
            params.append(origin)
        rows = self._conn.execute(
            f"""
            SELECT st.tag, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, st.tag
            """,
            tuple(params),
        ).fetchall()
        return {str(row["tag"]): int(row["count"] or 0) for row in rows}

    def list_session_tag_rollup_insights(
        self,
        *,
        origin: str | None = None,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[SessionTagRollupInsight]:
        """Aggregate archive session tags into public tag-rollup insights."""
        where: list[str] = []
        params: list[object] = []
        origin = _origin_value(origin)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if query:
            where.append("lower(st.tag) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        filter_params = tuple(params)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT st.tag,
                   COUNT(DISTINCT s.session_id) AS session_count,
                   COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'user' THEN s.session_id END) AS explicit_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'auto' THEN s.session_id END) AS auto_count,
                   MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            JOIN {self._tags_relation} st ON st.session_id = s.session_id
            {clause}
            GROUP BY st.tag
            ORDER BY session_count DESC, st.tag
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            SessionTagRollupInsight(
                tag=str(row["tag"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                explicit_count=int(row["explicit_count"] or 0),
                auto_count=int(row["auto_count"] or 0),
                origin_breakdown=_tag_origin_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                repo_breakdown=_tag_repo_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=None,
                    source_updated_at=_iso_from_ms(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                    input_high_water_mark=_iso_from_ms(row["source_sort_key_ms"]),
                    input_high_water_mark_source="sort_key" if row["source_sort_key_ms"] is not None else None,
                    time_confidence="estimated" if row["source_sort_key_ms"] is not None else "unknown",
                ),
            )
            for row in rows
        ]

    def list_tool_usage_insights(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageInsight]:
        """Aggregate tool-usage insights from action rows."""
        return self._read_insights().list_tool_usage_insights(query)

    def _read_insights(self) -> ArchiveReadInsights:
        """Bind insight SQL to this store's caller-owned read snapshot."""
        return ArchiveReadInsights(
            self._conn,
            normalize_origin=_origin_value,
            iso_from_milliseconds=_iso_from_ms,
        )

    def list_tool_call_count_rows(self, query: ToolUsageInsightQuery | None = None) -> list[dict[str, object]]:
        """Fast call-count-only tool rollups from tool-use blocks."""
        request = query or ToolUsageInsightQuery()
        where = ["b.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.origin)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(b.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(b.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        if request.session_id:
            where.append("b.session_id = ?")
            params.append(request.session_id)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                COALESCE(NULLIF(b.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count
            FROM blocks b
            JOIN sessions s ON s.session_id = b.session_id
            WHERE {" AND ".join(where)}
            GROUP BY s.origin, normalized_tool_name, action_kind
            ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "tool_use"),
                "call_count": int(row["call_count"] or 0),
            }
            for row in rows
        ]

    def list_tool_observed_event_count_rows(
        self, query: ToolUsageInsightQuery | None = None
    ) -> list[dict[str, object]]:
        """Tool outcome rollups from canonical tool-use/result block evidence."""
        request = query or ToolUsageInsightQuery()
        where = ["u.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.origin)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
        handler_expr = (
            "CASE "
            f"WHEN {tool_expr} >= 'mcp__' AND {tool_expr} < 'mcp__\U0010ffff' THEN 'mcp' "
            f"WHEN NULLIF({_action_command_expression('u')}, '') IS NOT NULL THEN 'shell' "
            "ELSE COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') "
            "END"
        )
        status_expr = (
            "CASE "
            "WHEN r.tool_result_exit_code IS NOT NULL "
            "THEN CASE WHEN r.tool_result_exit_code = 0 THEN 'ok' ELSE 'failed' END "
            "WHEN r.tool_result_is_error IS NOT NULL "
            "THEN CASE WHEN r.tool_result_is_error = 1 THEN 'failed' ELSE 'ok' END "
            "ELSE 'unknown' "
            "END"
        )
        where.append("r.rowid IS NOT NULL")
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        if request.mcp_server:
            mcp_prefix = f"mcp__{request.mcp_server.lower()}__"
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(mcp_prefix)
            params.append(f"{mcp_prefix}\U0010ffff")
        if request.action_kind:
            where.append(f"{handler_expr} = ?")
            params.append(request.action_kind)
        if request.session_id:
            where.append("u.session_id = ?")
            params.append(request.session_id)
        if request.since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(request.since_ms)
        if request.limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            params.extend((request.limit, request.offset))
        elif request.offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            params.append(request.offset)
        else:
            limit_clause = ""
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin AS origin,
                {tool_expr} AS normalized_tool_name,
                {handler_expr} AS action_kind,
                {status_expr} AS status,
                COUNT(*) AS event_count
            FROM blocks u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE {" AND ".join(where)}
            GROUP BY s.origin, normalized_tool_name, action_kind, status
            ORDER BY event_count DESC, s.origin ASC, normalized_tool_name ASC, status ASC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "unknown"),
                "status": str(row["status"] or "unknown"),
                "event_count": int(row["event_count"] or 0),
            }
            for row in rows
        ]

    def list_tool_action_evidence_count_rows(
        self,
        query: ToolUsageInsightQuery | None = None,
        *,
        detail_patterns: tuple[str, ...] = (),
        since_ms: int | None = None,
    ) -> list[dict[str, object]]:
        """Tool/affordance rollups from the canonical ``actions`` projection.

        Unlike raw tool-use block counts, this basis can match command/path/input
        details and then normalize generic shell rows into families such as
        ``codebase-memory/command-detail``. The normalized grouping is a read
        projection; raw tool names remain folded into the evidence kind and
        matched-by fields rather than replacing source evidence.
        """

        request = query or ToolUsageInsightQuery()
        where: list[str] = ["u.block_type = 'tool_use'"]
        params: list[object] = []
        origin = _origin_for_tool_usage_filter(request.origin)
        if origin:
            where.append("s.origin = ?")
            params.append(origin)
        tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
        if request.tool:
            where.append(f"{tool_expr} = LOWER(?)")
            params.append(request.tool)
        tool_patterns: tuple[str, ...] = ()
        if request.mcp_server:
            tool_patterns = (f"mcp__{request.mcp_server.lower()}__",)
            where.append(f"{tool_expr} >= ?")
            where.append(f"{tool_expr} < ?")
            params.append(tool_patterns[0])
            params.append(f"{tool_patterns[0]}\U0010ffff")
        if request.action_kind:
            where.append("COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') = ?")
            params.append(request.action_kind)
        if request.session_id:
            where.append("u.session_id = ?")
            params.append(request.session_id)
        effective_since_ms = since_ms if since_ms is not None else request.since_ms
        if effective_since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(effective_since_ms)
        cleaned_details = _clean_affordance_patterns(detail_patterns)
        fts_queries = tuple(
            fts_query for pattern in cleaned_details if (fts_query := normalize_fts5_query(pattern)) is not None
        )
        if cleaned_details:
            if not fts_queries:
                return []
            where.append(
                "("
                + " OR ".join("u.rowid IN (SELECT rowid FROM messages_fts WHERE text MATCH ?)" for _ in fts_queries)
                + ")"
            )
            params.extend(fts_queries)
            if (
                not request.tool
                and not request.mcp_server
                and not request.action_kind
                and (family := _affordance_family_for_text(" ".join(cleaned_details))) is not None
            ):
                return self._list_tool_action_detail_evidence_count_rows(
                    where=where,
                    params=tuple(params),
                    family=family,
                    detail_patterns=cleaned_details,
                    limit=request.limit,
                    offset=request.offset,
                )

        def fetch_rows() -> list[sqlite3.Row]:
            return list(
                self._conn.execute(
                    f"""
            SELECT
                s.origin AS origin,
                u.tool_name AS tool_name,
                COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') AS action_kind,
                u.session_id AS session_id,
                u.message_id AS message_id,
                COALESCE(u.tool_command, '') || ' ' ||
                    COALESCE(u.tool_path, '') || ' ' ||
                    COALESCE(u.tool_input, '') AS match_detail,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code
            FROM blocks u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            {"WHERE " + " AND ".join(where) if where else ""}
            """,
                    tuple(params),
                ).fetchall()
            )

        rows = fetch_rows()

        buckets: dict[tuple[str, str, str, str], dict[str, object]] = {}
        sessions: dict[tuple[str, str, str, str], set[str]] = {}
        for row in rows:
            origin = str(row["origin"] or "unknown-export")
            public_row = {
                "tool_name": str(row["tool_name"] or ""),
                "match_detail": str(row["match_detail"] or ""),
            }
            if cleaned_details and not any(
                pattern in str(public_row["match_detail"]).lower() for pattern in cleaned_details
            ):
                continue
            normalized_tool_name = _affordance_normalized_tool_name(public_row)
            evidence_kind = _affordance_evidence_kind(public_row)
            matched_by = _affordance_matched_by(
                public_row,
                tool_patterns=tool_patterns,
                detail_patterns=cleaned_details,
            )
            key = (
                origin,
                normalized_tool_name,
                str(row["action_kind"] or "tool_use"),
                evidence_kind,
            )
            bucket = buckets.setdefault(
                key,
                {
                    "origin": origin,
                    "normalized_tool_name": normalized_tool_name,
                    "action_kind": str(row["action_kind"] or "tool_use"),
                    "evidence_kind": evidence_kind,
                    "matched_by": matched_by,
                    "call_count": 0,
                    "session_count": 0,
                    "error_count": 0,
                    "nonzero_exit_count": 0,
                },
            )
            sessions.setdefault(key, set()).add(str(row["session_id"]))
            bucket["call_count"] = int(str(bucket["call_count"])) + 1
            bucket["error_count"] = int(str(bucket["error_count"])) + (1 if int(row["is_error"] or 0) == 1 else 0)
            bucket["nonzero_exit_count"] = int(str(bucket["nonzero_exit_count"])) + (
                1 if row["exit_code"] is not None and int(row["exit_code"] or 0) != 0 else 0
            )
        for key, bucket in buckets.items():
            bucket["session_count"] = len(sessions.get(key, set()))
        ordered = sorted(
            buckets.values(),
            key=lambda item: (
                -int(str(item["call_count"])),
                str(item["origin"]),
                str(item["normalized_tool_name"]),
                str(item["evidence_kind"]),
            ),
        )
        offset = request.offset or 0
        if request.limit is not None:
            ordered = ordered[offset : offset + request.limit]
        elif offset:
            ordered = ordered[offset:]
        return ordered

    def _list_tool_action_detail_evidence_count_rows(
        self,
        *,
        where: list[str],
        params: tuple[object, ...],
        family: str,
        detail_patterns: tuple[str, ...],
        limit: int | None,
        offset: int,
    ) -> list[dict[str, object]]:
        """Fast grouped action-evidence rows for generic command detail matches."""

        generic_tools = ("exec_command", "functions", "functions.exec_command", "bash", "shell", "client")
        tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
        detail_expr = (
            "LOWER(COALESCE(u.tool_command, '') || ' ' || "
            "COALESCE(u.tool_path, '') || ' ' || COALESCE(u.tool_input, ''))"
        )
        detail_clauses = " OR ".join(f"{detail_expr} LIKE ? ESCAPE '\\'" for _ in detail_patterns)
        all_where = [*where, f"({detail_clauses})", f"{tool_expr} IN ({', '.join('?' for _ in generic_tools)})"]
        all_params: list[object] = [*params]
        all_params.extend(_affordance_like_param(pattern) for pattern in detail_patterns)
        all_params.extend(generic_tools)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ? OFFSET ?"
            all_params.extend((limit, offset))
        elif offset:
            limit_clause = "LIMIT -1 OFFSET ?"
            all_params.append(offset)
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin AS origin,
                COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count,
                COUNT(DISTINCT u.session_id) AS session_count,
                SUM(CASE WHEN r.tool_result_is_error = 1 THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0 THEN 1 ELSE 0 END)
                    AS nonzero_exit_count
            FROM blocks u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN blocks r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE {" AND ".join(all_where)}
            GROUP BY s.origin, action_kind
            ORDER BY call_count DESC, s.origin ASC, action_kind ASC
            {limit_clause}
            """,
            tuple(all_params),
        ).fetchall()
        return [
            {
                "origin": str(row["origin"] or "unknown-export"),
                "normalized_tool_name": f"{family}/command-detail",
                "action_kind": str(row["action_kind"] or "tool_use"),
                "evidence_kind": "command_detail",
                "matched_by": "detail",
                "call_count": int(row["call_count"] or 0),
                "session_count": int(row["session_count"] or 0),
                "error_count": int(row["error_count"] or 0),
                "nonzero_exit_count": int(row["nonzero_exit_count"] or 0),
            }
            for row in rows
        ]

    def list_archive_coverage_insights(
        self,
        *,
        group_by: str = "origin",
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveCoverageInsight]:
        return self._read_insights().list_archive_coverage_insights(
            group_by=group_by,
            origin=origin,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=offset,
        )

    def set_user_metadata(self, session_ids: tuple[str, ...], pairs: tuple[tuple[str, object], ...]) -> int:
        """Set human-owned metadata as archive user.db assertions."""
        user_conn = self._open_user_write_connection(initialize=True)
        user_conn.row_factory = sqlite3.Row
        try:
            changed = 0
            with user_conn:
                for session_id in tuple(
                    dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)
                ):
                    for key, value in pairs:
                        normalized_key = key.strip()
                        if not normalized_key:
                            raise ValueError("metadata key cannot be empty")
                        existing = read_assertion_envelope(
                            user_conn,
                            assertion_id_for_session_metadata(session_id, normalized_key),
                        )
                        if (
                            existing is not None
                            and existing.status != "deleted"
                            and _canonical_json_text(existing.value) == _canonical_json_text(value)
                        ):
                            continue
                        upsert_session_metadata_assertion(
                            user_conn,
                            session_id=session_id,
                            key=normalized_key,
                            value=value,
                        )
                        changed += 1
        finally:
            user_conn.close()
        return changed

    def read_user_metadata(self, session_id: str) -> dict[str, object]:
        """Read human-owned metadata assertions for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return {}
        user_conn = open_readonly_connection(self.user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            rows = list_assertions_for_target(user_conn, f"session:{resolved_session_id}", kind=AssertionKind.METADATA)
        finally:
            user_conn.close()
        decoded: dict[str, object] = {}
        for assertion in rows:
            if assertion.status == "deleted" or assertion.key is None:
                continue
            decoded[str(assertion.key)] = assertion.value
        return decoded

    def delete_user_metadata(self, session_id: str, key: str) -> int:
        """Mark one user metadata assertion deleted."""
        resolved_session_id = self.resolve_session_id(session_id)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("metadata key cannot be empty")
        if not self.user_db_path.exists():
            return 0
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                assertion_id = assertion_id_for_session_metadata(resolved_session_id, normalized_key)
                assertion = read_assertion_envelope(user_conn, assertion_id)
                if assertion is None or assertion.status == "deleted":
                    return 0
                return 1 if mark_assertion_status(user_conn, assertion_id, "deleted") else 0
        finally:
            user_conn.close()

    def add_mark(
        self,
        target_type: str,
        target_id: str,
        mark_type: str,
        *,
        owner_session_id: str | None = None,
    ) -> bool:
        """Add one user mark to archive user.db."""
        if target_type == "message" and not owner_session_id:
            raise ValueError("message marks require a resolved owner_session_id")
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_mark(target_type, target_id, mark_type))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_mark(
                    user_conn,
                    target_type,
                    target_id,
                    mark_type,
                    owner_session_id=owner_session_id,
                )
            return not exists
        finally:
            user_conn.close()

    def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove one user mark from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                return mark_assertion_status(
                    user_conn,
                    assertion_id_for_mark(target_type, target_id, mark_type),
                    "deleted",
                )
        finally:
            user_conn.close()

    def list_marks(
        self,
        *,
        mark_type: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List user marks from archive user.db."""
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.MARK)
        finally:
            user_conn.close()
        selected: list[tuple[ArchiveAssertionEnvelope, str, str]] = []
        for assertion in assertions:
            found_target_type, found_target_id = _split_user_target_ref(assertion.target_ref)
            if mark_type and assertion.key != mark_type:
                continue
            if target_type and found_target_type != target_type:
                continue
            if target_id and found_target_id != target_id:
                continue
            selected.append((assertion, found_target_type, found_target_id))
        owners = _user_state_session_ids((item[0] for item in selected), index_conn=self._conn)
        out: list[dict[str, str]] = []
        for assertion, found_target_type, found_target_id in selected:
            owner_session_id = owners[assertion.assertion_id]
            if session_id and target_id is None and owner_session_id != session_id:
                continue
            out.append(
                {
                    "target_type": found_target_type,
                    "target_id": found_target_id,
                    "session_id": owner_session_id,
                    "message_id": found_target_id if found_target_type == "message" else "",
                    "mark_type": str(assertion.key or ""),
                    "created_at": str(assertion.created_at_ms),
                }
            )
        return out

    def save_annotation(
        self,
        annotation_id: str,
        target_type: str,
        target_id: str,
        note_text: str,
        *,
        owner_session_id: str | None = None,
    ) -> bool:
        """Create or update one annotation in archive user.db."""
        if target_type == "message" and not owner_session_id:
            raise ValueError("message annotations require a resolved owner_session_id")
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_annotation(annotation_id))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_annotation(
                    user_conn,
                    target_type,
                    target_id,
                    note_text,
                    owner_session_id=owner_session_id,
                    annotation_id=annotation_id,
                )
            return not exists
        finally:
            user_conn.close()

    def save_annotation_schema(
        self,
        schema: AnnotationSchema,
        *,
        registered_at_ms: int | None = None,
    ) -> DurableAnnotationSchema:
        """Persist an immutable annotation schema definition in ``user.db``."""

        user_conn = self._open_user_write_connection(initialize=True)
        user_conn.row_factory = sqlite3.Row
        try:
            with user_conn:
                return persist_annotation_schema(
                    user_conn,
                    schema,
                    registered_at_ms=registered_at_ms if registered_at_ms is not None else int(time.time() * 1000),
                )
        finally:
            user_conn.close()

    def get_annotation_schema(
        self,
        schema_id: str,
        version: int | None = None,
    ) -> DurableAnnotationSchema | None:
        """Resolve one durable schema definition, defaulting to its latest version."""

        if not self.user_db_path.exists():
            return None
        user_conn = open_readonly_connection(self.user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            return read_durable_annotation_schema(user_conn, schema_id, version)
        finally:
            user_conn.close()

    def list_annotation_schemas(self) -> tuple[DurableAnnotationSchema, ...]:
        """List durable annotation schema definitions in identity order."""

        if not self.user_db_path.exists():
            return ()
        user_conn = open_readonly_connection(self.user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            return list_durable_annotation_schemas(user_conn)
        finally:
            user_conn.close()

    def save_annotation_batch(self, batch: AnnotationBatch) -> AnnotationBatch:
        """Persist one immutable annotation-batch provenance container."""

        user_conn = self._open_user_write_connection(initialize=True)
        user_conn.row_factory = sqlite3.Row
        try:
            with user_conn:
                return persist_annotation_batch(user_conn, batch)
        finally:
            user_conn.close()

    def get_annotation_batch(self, batch_id: str) -> AnnotationBatch | None:
        """Read one durable annotation batch by id."""

        if not self.user_db_path.exists():
            return None
        user_conn = open_readonly_connection(self.user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            return read_annotation_batch(user_conn, batch_id)
        finally:
            user_conn.close()

    def list_annotation_batches(
        self,
        *,
        schema_id: str | None = None,
        schema_version: int | None = None,
        target_ref: str | None = None,
        limit: int | None = None,
    ) -> tuple[AnnotationBatch, ...]:
        """List durable batch metadata with focused schema/target filters."""

        if not self.user_db_path.exists():
            return ()
        user_conn = open_readonly_connection(self.user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            return _list_annotation_batches(
                user_conn,
                schema_id=schema_id,
                schema_version=schema_version,
                target_ref=target_ref,
                limit=limit,
            )
        finally:
            user_conn.close()

    def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Read one annotation from archive user.db."""
        rows = self.list_annotations(annotation_id=annotation_id)
        return rows[0] if rows else None

    def list_annotations(
        self,
        *,
        annotation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations from archive user.db.

        When ``session_id`` is supplied (and no explicit target filter),
        the result includes both the session-target annotation and every
        message-target annotation whose durable owner scope or exact indexed
        message ownership matches that session. Legacy message assertions
        without either authority are intentionally excluded.
        """
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.ANNOTATION)
        finally:
            user_conn.close()
        selected: list[tuple[ArchiveAssertionEnvelope, str, str]] = []
        for assertion in assertions:
            found_annotation_id = str(assertion.key or "")
            found_target_type, found_target_id = _split_user_target_ref(assertion.target_ref)
            if annotation_id and found_annotation_id != annotation_id:
                continue
            if target_type and found_target_type != target_type:
                continue
            if target_id and found_target_id != target_id:
                continue
            selected.append((assertion, found_target_type, found_target_id))
        owners = _user_state_session_ids((item[0] for item in selected), index_conn=self._conn)
        out: list[dict[str, str]] = []
        for assertion, found_target_type, found_target_id in selected:
            owner_session_id = owners[assertion.assertion_id]
            if session_id and target_id is None and owner_session_id != session_id:
                continue
            out.append(
                {
                    "annotation_id": str(assertion.key or ""),
                    "target_type": found_target_type,
                    "target_id": found_target_id,
                    "session_id": owner_session_id,
                    "message_id": found_target_id if found_target_type == "message" else "",
                    "note_text": assertion.body_text or "",
                    "created_at": str(assertion.created_at_ms),
                    "updated_at": str(assertion.updated_at_ms),
                }
            )
        return out

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete one annotation from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_annotation(annotation_id), "deleted")
        finally:
            user_conn.close()

    def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Create or update one saved view in archive user.db."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must not be empty")
        query = json.loads(query_json)
        if not isinstance(query, dict):
            raise ValueError("query_json must encode an object")
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            assertion_id = assertion_id_for_saved_view(view_id)
            assertion = read_assertion_envelope(user_conn, assertion_id)
            name_assertion = _active_assertion_by_kind_key(user_conn, AssertionKind.SAVED_QUERY, normalized_name)
            exists = (assertion is not None and assertion.status != "deleted") or name_assertion is not None
            with user_conn:
                if name_assertion is not None and name_assertion.assertion_id != assertion_id:
                    mark_assertion_status(user_conn, name_assertion.assertion_id, "deleted")
                upsert_saved_view(user_conn, normalized_name, query, view_id=view_id)
            return not exists
        finally:
            user_conn.close()

    def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get one saved view by id from archive user.db."""
        return next((row for row in self.list_views() if row["view_id"] == view_id), None)

    def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get one saved view by name from archive user.db."""
        return next((row for row in self.list_views() if row["name"] == name), None)

    def list_views(self) -> list[dict[str, str]]:
        """List saved views from archive user.db."""
        return self._list_views()

    def _list_views(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.SAVED_QUERY)
        finally:
            user_conn.close()
        return [
            {
                "view_id": _id_from_target_ref(assertion.target_ref, "saved_view:"),
                "name": str(assertion.key or ""),
                "query_json": json.dumps(
                    assertion.value if isinstance(assertion.value, dict) else {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "created_at": str(assertion.created_at_ms),
            }
            for assertion in assertions
        ]

    def delete_view(self, view_id: str) -> bool:
        """Delete one saved view from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_saved_view(view_id), "deleted")
        finally:
            user_conn.close()

    def save_recall_pack(
        self,
        pack_id: str,
        label: str,
        session_ids_json: str,
        payload_json: str,
    ) -> bool:
        """Create or update one recall pack in archive user.db."""
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("payload_json must encode an object")
        payload = dict(payload)
        payload["session_ids_json"] = session_ids_json
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            assertion = read_assertion_envelope(user_conn, assertion_id_for_recall_pack(pack_id))
            exists = assertion is not None and assertion.status != "deleted"
            with user_conn:
                upsert_recall_pack(user_conn, label, payload, recall_pack_id=pack_id)
            return not exists
        finally:
            user_conn.close()

    def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get one recall pack by id from archive user.db."""
        return next((row for row in self.list_recall_packs() if row["pack_id"] == pack_id), None)

    def list_recall_packs(self) -> list[dict[str, str]]:
        """List recall packs from archive user.db."""
        return self._list_recall_packs()

    def _list_recall_packs(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.RECALL_PACK)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            payload = assertion.value if isinstance(assertion.value, dict) else {}
            if not isinstance(payload, dict):
                payload = {}
            session_ids_json = payload.pop("session_ids_json", "[]")
            out.append(
                {
                    "pack_id": _id_from_target_ref(assertion.target_ref, "recall_pack:"),
                    "label": str(assertion.key or ""),
                    "session_ids_json": str(session_ids_json),
                    "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    "created_at": str(assertion.created_at_ms),
                }
            )
        return out

    def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete one recall pack from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_recall_pack(pack_id), "deleted")
        finally:
            user_conn.close()

    def save_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        """Create or update one reader workspace in archive user.db."""
        settings: dict[str, object] = {
            "mode": mode,
            "open_targets_json": open_targets_json,
            "layout_json": layout_json,
            "active_target_json": active_target_json,
        }
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            assertion_id = assertion_id_for_workspace(workspace_id)
            assertion = read_assertion_envelope(user_conn, assertion_id)
            name_assertion = _active_assertion_by_kind_key(user_conn, AssertionKind.WORKSPACE_NOTE, name)
            exists = (assertion is not None and assertion.status != "deleted") or name_assertion is not None
            with user_conn:
                if name_assertion is not None and name_assertion.assertion_id != assertion_id:
                    mark_assertion_status(user_conn, name_assertion.assertion_id, "deleted")
                upsert_workspace(user_conn, name, settings, workspace_id=workspace_id)
            return not exists
        finally:
            user_conn.close()

    def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get one workspace by id from archive user.db."""
        return next((row for row in self.list_workspaces() if row["workspace_id"] == workspace_id), None)

    def get_workspace_by_name(self, name: str) -> dict[str, str] | None:
        """Get one workspace by name from archive user.db."""
        return next((row for row in self.list_workspaces() if row["name"] == name), None)

    def list_workspaces(self) -> list[dict[str, str]]:
        """List workspaces from archive user.db."""
        return self._list_workspaces()

    def _list_workspaces(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        del where, params
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.WORKSPACE_NOTE)
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for assertion in assertions:
            settings = assertion.value if isinstance(assertion.value, dict) else {}
            if not isinstance(settings, dict):
                settings = {}
            out.append(
                {
                    "workspace_id": _id_from_target_ref(assertion.target_ref, "workspace:"),
                    "name": str(assertion.key or ""),
                    "mode": str(settings.get("mode") or ""),
                    "open_targets_json": str(settings.get("open_targets_json") or "[]"),
                    "layout_json": str(settings.get("layout_json") or "{}"),
                    "active_target_json": str(settings.get("active_target_json") or "{}"),
                    "created_at": str(assertion.created_at_ms),
                    "updated_at": str(assertion.updated_at_ms),
                }
            )
        return out

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete one workspace from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                return mark_assertion_status(user_conn, assertion_id_for_workspace(workspace_id), "deleted")
        finally:
            user_conn.close()

    def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> LearningCorrection:
        """Record one learning correction in archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        stored_payload: dict[str, object] = {"payload": dict(payload), "note": note}
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            with user_conn:
                upsert_correction(
                    user_conn,
                    "insight",
                    resolved_session_id,
                    correction_kind.value,
                    stored_payload,
                    author_ref=author_ref,
                    author_kind=author_kind,
                )
        finally:
            user_conn.close()
        listed = self.list_corrections(session_id=resolved_session_id, kind=correction_kind.value)
        if not listed:
            raise KeyError((resolved_session_id, correction_kind.value))
        return listed[0]

    def list_corrections(self, *, session_id: str | None = None, kind: str | None = None) -> list[LearningCorrection]:
        """List learning corrections from archive user.db."""
        if not self.user_db_path.exists():
            return []
        resolved_session_id = self.resolve_session_id(session_id) if session_id else None
        correction_kind = parse_correction_kind(kind).value if kind is not None else None
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            assertions = list_assertions_by_kind(user_conn, AssertionKind.CORRECTION)
        finally:
            user_conn.close()
        out: list[LearningCorrection] = []
        for assertion in assertions:
            target_type, target_id = _split_user_target_ref(assertion.target_ref)
            if target_type != "insight":
                continue
            if resolved_session_id is not None and target_id != resolved_session_id:
                continue
            if correction_kind is not None and assertion.key != correction_kind:
                continue
            payload_json = json.dumps(assertion.value if isinstance(assertion.value, dict) else {}, sort_keys=True)
            out.append(
                _learning_correction_from_archive_row(
                    (target_id, str(assertion.key or ""), payload_json, assertion.updated_at_ms)
                )
            )
        return out

    def delete_correction(self, session_id: str, kind: str) -> bool:
        """Delete one learning correction from archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        if not self.user_db_path.exists():
            return False
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                correction_id = correction_id_for("insight", resolved_session_id, correction_kind.value)
                return mark_assertion_status(user_conn, assertion_id_for_correction(correction_id), "deleted")
        finally:
            user_conn.close()

    def clear_corrections(self, session_id: str) -> int:
        """Delete all learning corrections for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return 0
        user_conn = self._open_user_write_connection()
        try:
            with user_conn:
                deleted_count = 0
                for assertion in list_assertions_by_kind(user_conn, AssertionKind.CORRECTION):
                    target_type, target_id = _split_user_target_ref(assertion.target_ref)
                    if (
                        target_type == "insight"
                        and target_id == resolved_session_id
                        and mark_assertion_status(user_conn, assertion.assertion_id, "deleted")
                    ):
                        deleted_count += 1
                return deleted_count
        finally:
            user_conn.close()

    def post_blackboard_note(
        self,
        body: str,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
        note_id: str | None = None,
        author_ref: str | None = None,
        author_kind: str = "user",
        evidence_refs: tuple[str, ...] = (),
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> ArchiveBlackboardNoteEnvelope:
        """Insert-or-update one blackboard note in archive user.db."""
        user_conn = self._open_user_write_connection(initialize=True)
        try:
            envelope = upsert_blackboard_note(
                user_conn,
                body,
                target_type=target_type,
                target_id=target_id,
                note_id=note_id,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=evidence_refs,
                staleness=staleness,
                context_policy=context_policy,
            )
            user_conn.commit()
            return envelope
        finally:
            user_conn.close()

    def list_blackboard_notes(self, *, limit: int | None = None) -> list[ArchiveBlackboardNoteEnvelope]:
        """List blackboard notes from archive user.db, newest first.

        Assertion rows own note ids, targets, body text, and timestamps.
        Structured-field decoding (kind/title/scope) is a presentation concern
        handled by ``polylogue.archive.blackboard``.
        """
        if not self.user_db_path.exists():
            return []
        user_conn = open_readonly_connection(self.user_db_path)
        try:
            return list_archive_blackboard_note_envelopes(user_conn, limit=limit)
        finally:
            user_conn.close()

    def _trigram_trigger_is_guarded(self, conn: sqlite3.Connection) -> bool:
        """Whether the live ``blocks_command_trigram_ad`` trigger honors the
        bulk-write guard (CodeRabbit #3263 P1).

        ``CREATE TRIGGER IF NOT EXISTS`` (index.py's additive-DDL convention)
        never replaces an already-created same-name trigger, so an archive
        last rebuilt before the guard clause was added to this trigger (#3259)
        keeps its old, ungated body forever -- reopening it does not upgrade
        it. Setting ``FTS_BULK_SESSION_WRITE_GUARD`` and relying on the guard
        to suppress that trigger during the block cascade below would be a
        silent no-op on such an archive: the old trigger fires anyway and
        reissues an FTS5 ``'delete'`` command for a trigram row this method's
        own explicit pre-delete already removed, which raises "database disk
        image is malformed" and rolls back the whole batch. Checked once per
        call (cheap: one indexed sqlite_master lookup) rather than assumed
        from ``INDEX_SCHEMA_VERSION`` alone, since the guard clause landed
        without its own version bump (additive/inert until this method).
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'blocks_command_trigram_ad'"
        ).fetchone()
        return row is not None and row["sql"] is not None and "derived_refresh_guard" in row["sql"]

    def delete_sessions(self, session_ids: tuple[str, ...]) -> int:
        """Delete rebuildable archive sessions by id.

        User-tier overlays are intentionally left in ``user.db``; the user
        overlay orphan checker owns follow-up visibility for those durable rows.

        polylogue-meoz: a plain per-session ``DELETE FROM sessions`` detonates
        the per-row derived-refresh triggers -- ``blocks_action_pairs_ad``
        (gated on the ``'session-write'`` guard, see ``index.py``) fires once
        per deleted ``blocks`` row (both the explicit block cascade and the FK
        ``ON DELETE CASCADE`` from ``messages``/``sessions`` still invoke AFTER
        DELETE triggers per row) and, on each firing, deletes+rebuilds the
        *entire session's* ``action_pairs`` via two window-function scans and
        re-derives ``delegation_facts``. Live incident 2026-07-21: deleting 91
        sessions ran 3h, 375GB reads, zero commit, before being killed.
        ``messages_fts``/``blocks_command_trigram`` are external-content FTS5
        tables with no FK cascade at all, so those triggers would fire and
        maintain per-row postings unless suppressed the same way.

        This mirrors ``_bulk_fts_session_guard``'s delete-then-guard-then-
        mutate shape (``write.py``): explicit session-scoped FTS/identity/
        trigram deletes run first (while ``blocks`` rows still exist -- the
        trigram table needs the OLD text to locate its postings), then BOTH
        ``derived_refresh_guard`` rows are set for the whole batch --
        ``'session-write'`` (suppresses ``blocks_action_pairs_a{i,d,u}`` and
        the ``delegation_facts`` triggers) and ``'fts-bulk-session-write'``
        (suppresses the ``messages_fts``/``blocks_command_trigram`` trigger
        BODIES, redundant here since those rows are already gone, but kept for
        parity with the guard contract and defense-in-depth against any block
        row that slips past the explicit pre-delete). ``action_pairs``/
        ``delegation_facts`` are also explicitly cleared by session id --
        belt-and-suspenders alongside the ``ON DELETE CASCADE`` FKs those
        tables already carry against ``sessions``/``blocks``, and it keeps the
        post-condition legible without depending on cascade timing. The
        physical ``sessions``/``messages``/``blocks`` rows are then removed by
        one ``DELETE FROM sessions`` per id, relying on indexed FK cascades
        (not per-row trigger logic) for the tree removal; guard rows are
        cleared and the transaction commits once for the whole batch.

        The per-row ``query_unit_frame_{sessions,messages,blocks}_delete``
        epoch-bump triggers (``index.py``) are NOT gated by either guard --
        they still fire once per deleted row. Each firing is a single
        primary-key ``UPDATE query_unit_frame_state SET epoch = epoch + 1
        WHERE singleton = 1`` against a one-row table, not a window-function
        rebuild, so it is O(1) per row rather than O(session_size) per row;
        left ungated deliberately rather than folding a third guard in for a
        cost that was never implicated by the incident.
        """
        self._require_writable("delete index.db sessions")
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids:
            return 0
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        trigram_guarded = self._trigram_trigger_is_guarded(conn)
        deleted = 0
        try:
            with conn:
                for session_id in resolved_session_ids:
                    # Must run before the blocks rows are removed below --
                    # external-content FTS5 deletion needs the OLD text/rowid
                    # mapping to locate the postings to remove. Only safe to
                    # do explicitly when the live trigger will honor the
                    # guard below; on an archive whose trigger predates it,
                    # skip this and let that same (ungated) trigger clean up
                    # trigram rows per-block during the cascade delete, same
                    # as before this method existed.
                    conn.execute(delete_session_rows_sql(1), (session_id,))
                    conn.execute(delete_session_identity_rows_sql(1), (session_id,))
                    if trigram_guarded:
                        conn.execute(trigram_delete_session_rows_sql(), (session_id,))
                conn.execute("INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES ('session-write')")
                if trigram_guarded:
                    conn.execute(
                        "INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES (?)",
                        (FTS_BULK_SESSION_WRITE_GUARD,),
                    )
                try:
                    for session_id in resolved_session_ids:
                        conn.execute("DELETE FROM action_pairs WHERE session_id = ?", (session_id,))
                        conn.execute("DELETE FROM delegation_facts WHERE parent_session_id = ?", (session_id,))
                        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                        deleted += max(int(cursor.rowcount), 0)
                finally:
                    conn.execute("DELETE FROM derived_refresh_guard WHERE guard_name = 'session-write'")
                    if trigram_guarded:
                        conn.execute(
                            "DELETE FROM derived_refresh_guard WHERE guard_name = ?",
                            (FTS_BULK_SESSION_WRITE_GUARD,),
                        )
        finally:
            conn.close()
        return deleted

    def _attach_user_tier_if_present(self) -> None:
        if self._user_tier_attached or not self.user_db_path.exists():
            return
        user_db_uri = (
            f"file:{self.user_db_path}?mode=ro"
            if self._read_only or self._inactive_candidate_durable_read_only
            else str(self.user_db_path)
        )
        try:
            self._conn.execute("ATTACH DATABASE ? AS user_tier", (user_db_uri,))
        except Exception as exc:
            raise self._user_tier_unavailable(reason=f"cannot open SQLite database ({exc})") from exc
        self._user_tier_attached = True
        self._tags_relation = _all_session_tags_sql()

    def _user_tier_unavailable(self, *, reason: str) -> ArchiveTierUnavailableError:
        return ArchiveTierUnavailableError(
            tier="user.db",
            path=str(self.user_db_path.resolve(strict=False)),
            reason=reason,
            guidance="restore or initialize the durable user tier at this path, then retry the query; "
            "the reader will not create a replacement or search another archive root",
        )

    def require_user_tier(self) -> None:
        """Validate the durable user tier before an assertion read executes SQL."""
        if not self.user_db_path.exists():
            raise self._user_tier_unavailable(reason="the file does not exist")
        if not self.user_db_path.is_file():
            raise self._user_tier_unavailable(reason="the path is not a regular file")
        try:
            self._attach_user_tier_if_present()
            version = int(self._conn.execute("PRAGMA user_tier.user_version").fetchone()[0])
            expected = archive_tier_spec(ArchiveTier.USER).version
            if version != expected:
                raise self._user_tier_unavailable(reason=f"schema version {version} does not match expected {expected}")
            if not _table_exists(self._conn, "assertions", schema="user_tier"):
                raise self._user_tier_unavailable(reason="the assertions table is missing")
        except ArchiveTierUnavailableError:
            raise
        except Exception as exc:
            raise self._user_tier_unavailable(reason=f"cannot validate SQLite schema ({exc})") from exc

    def count_sessions(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
    ) -> int:
        """Count sessions in the archive index."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        if session_id is not None:
            try:
                resolved_id = self.resolve_session_id(session_id)
            except KeyError:
                return 0
            where = f"{where} AND s.session_id = ?" if where else "WHERE s.session_id = ?"
            params.append(resolved_id)
        return int(self._conn.execute(f"SELECT COUNT(*) FROM sessions s {where}", params).fetchone()[0])

    def session_insight_status(self) -> SessionInsightStatusSnapshot:
        """Return readiness for session insight tables."""
        return session_insight_status_sync(self._conn)

    def insight_readiness_report(self, query: InsightReadinessQuery | None = None) -> InsightReadinessReport:
        """Return public insight readiness from tables."""
        request = query or InsightReadinessQuery()
        selected = (
            tuple(normalize_insight_readiness_name(insight) for insight in request.insights)
            if request.insights
            else known_insight_readiness_names()
        )
        status = self.session_insight_status()
        origin_filter = _origin_value(request.origin)
        since_ms = _epoch_ms_from_iso(request.since)
        until_ms = _epoch_ms_from_iso(request.until)
        total_sessions = self.count_sessions(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        coverage = self._archive_session_origin_coverage(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        entries = tuple(
            entry
            for name in selected
            if (
                entry := self._insight_readiness_entry(
                    name,
                    status=status,
                    total_sessions=total_sessions,
                    origin_coverage=coverage,
                    origin=origin_filter,
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
            )
            is not None
        )
        return InsightReadinessReport(
            checked_at=datetime.now(UTC).isoformat(),
            aggregate_verdict=_insight_readiness_aggregate_verdict(entries),
            total_sessions=total_sessions,
            origin=request.origin,
            since=request.since,
            until=request.until,
            insights=entries,
        )

    def audit_insight_rigor(self, query: InsightRigorAuditQuery | None = None) -> InsightRigorAuditReport:
        """Audit insight rigor over read models."""
        request = query or InsightRigorAuditQuery()
        targeted = set(request.insights) if request.insights else None
        entries = []
        for contract in list_rigor_contracts():
            if targeted is not None and contract.insight_name not in targeted:
                continue
            rows = self._rigor_audit_rows(contract.insight_name, limit=max(request.sample_limit, 0))
            entries.append(_audit_one(rows, contract))
        return InsightRigorAuditReport(sample_limit=request.sample_limit, entries=tuple(entries))

    def _rigor_audit_rows(self, insight_name: str, *, limit: int) -> list[object]:
        if insight_name == "session_profiles":
            return list(self.list_session_profile_insights(limit=limit))
        if insight_name == "session_work_events":
            return list(self.list_session_work_event_insights(limit=limit))
        if insight_name == "session_phases":
            return list(self.list_session_phase_insights(limit=limit))
        if insight_name == "threads":
            return list(self.list_thread_insights(limit=limit))
        if insight_name == "session_tag_rollups":
            return list(self.list_session_tag_rollup_insights(limit=limit))
        return []

    def _archive_session_origin_coverage(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[InsightOriginCoverage, ...]:
        """Per-provider session distribution for insight readiness coverage."""
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(
            f"SELECT origin, COUNT(*) AS n, MIN(created_at_ms) AS lo, MAX(updated_at_ms) AS hi "
            f"FROM sessions {clause} GROUP BY origin ORDER BY n DESC, origin",
            tuple(params),
        ).fetchall()
        return tuple(
            InsightOriginCoverage(
                origin=str(row["origin"]),
                row_count=int(row["n"]),
                min_time=_iso_from_ms(row["lo"]),
                max_time=_iso_from_ms(row["hi"]),
            )
            for row in rows
        )

    def _readiness_session_filter(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[str, list[object]]:
        """Build a ``WHERE`` fragment over the joined ``sessions`` (alias ``s``)."""
        clauses: list[str] = []
        params: list[object] = []
        if origin is not None:
            clauses.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) <= ?")
            params.append(until_ms)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _archive_materialization_signals(
        self,
        insight_type: str,
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[tuple[InsightVersionCoverage, ...], int, int]:
        """Derive version coverage, incompatible count, and native staleness.

        Reads the ``insight_materialization`` high-water marks for ``insight_type``
        joined to ``sessions``. A row is *incompatible* (legacy) when its
        ``materializer_version`` is below ``SESSION_INSIGHT_MATERIALIZER_VERSION``;
        it is *stale* when its captured ``source_sort_key_ms`` no longer matches the
        live session ``sort_key_ms`` (the native source high-water mark). The
        ``session_profiles.materializer_version``/``source_sort_key`` columns are not
        used here: they are not reliably populated by the canonical rebuild path,
        so the materialization ledger is the authoritative provenance source.
        """
        if not _table_exists(self._conn, "insight_materialization"):
            return ((), 0, 0)
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        version_rows = self._conn.execute(
            "SELECT im.materializer_version AS version, COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "GROUP BY im.materializer_version ORDER BY im.materializer_version",
            (insight_type, *params),
        ).fetchall()
        versions = {str(int(row["version"])): int(row["n"]) for row in version_rows}
        incompatible_count = sum(
            count for version, count in versions.items() if int(version) < SESSION_INSIGHT_MATERIALIZER_VERSION
        )
        version_coverage = (
            (
                InsightVersionCoverage(
                    field="materializer_version",
                    current_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                    versions=versions,
                    incompatible_count=incompatible_count,
                ),
            )
            if versions
            else ()
        )
        stale_row = self._conn.execute(
            "SELECT COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "AND COALESCE(im.source_sort_key_ms, -1) != COALESCE(s.sort_key_ms, -1)",
            (insight_type, *params),
        ).fetchone()
        stale_count = int(stale_row["n"]) if stale_row is not None else 0
        return (version_coverage, incompatible_count, stale_count)

    def _archive_fallback_coverage(
        self,
        table_name: str,
        column_paths: tuple[tuple[str, str], ...],
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[int, dict[str, int]]:
        """Count rows whose enrichment provenance carries fallback reasons.

        Each insight row stores its fallback markers as JSON arrays under
        ``$.fallback_reasons`` inside one or more payload columns (e.g.
        ``inference_payload_json`` and ``enrichment_payload_json`` on
        ``session_profiles``). A row is *degraded* when any declared
        ``(column, path)`` holds a non-empty array; the row is counted at most
        once regardless of how many columns flag it. ``reason_totals`` sums
        occurrences per reason across every inspected column.
        """
        if not _table_exists(self._conn, table_name):
            return (0, {})
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        any_terms = " OR ".join(
            f"json_array_length(COALESCE(json_extract(t.{column}, '{path}'), '[]')) > 0"
            for column, path in column_paths
        )
        degraded_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table_name} AS t "
            "JOIN sessions AS s ON s.session_id = t.session_id "
            f"WHERE ({any_terms}){clause}",
            tuple(params),
        ).fetchone()
        degraded_count = int(degraded_row["n"]) if degraded_row is not None else 0
        reason_totals: dict[str, int] = {}
        for column, path in column_paths:
            rows = self._conn.execute(
                "SELECT value AS reason, COUNT(*) AS occurrences "
                f"FROM {table_name} AS t "
                "JOIN sessions AS s ON s.session_id = t.session_id, "
                f"json_each(COALESCE(json_extract(t.{column}, '{path}'), '[]')) "
                f"WHERE 1=1{clause} GROUP BY value",
                tuple(params),
            ).fetchall()
            for row in rows:
                reason = str(row["reason"])
                reason_totals[reason] = reason_totals.get(reason, 0) + int(row["occurrences"])
        return (degraded_count, dict(sorted(reason_totals.items())))

    def _insight_readiness_entry(
        self,
        name: str,
        *,
        status: SessionInsightStatusSnapshot,
        total_sessions: int,
        origin_coverage: tuple[InsightOriginCoverage, ...] = (),
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> InsightReadinessEntry | None:
        specs = {
            "session_profiles": (
                "Session Profiles",
                "session_profiles",
                status.profile_row_count,
                total_sessions,
                status.missing_profile_row_count,
                0,
                status.orphan_profile_row_count,
                {"profile_rows_ready": status.profile_rows_ready},
                ("session_profiles",),
            ),
            "session_work_events": (
                "Work Events",
                "session_work_events",
                status.work_event_inference_count,
                status.expected_work_event_inference_count,
                0,
                status.stale_work_event_inference_count,
                status.orphan_work_event_inference_count,
                {"work_event_inference_rows_ready": status.work_event_inference_rows_ready},
                ("session_work_events",),
            ),
            "session_phases": (
                "Session Phases",
                "session_phases",
                status.phase_count,
                status.expected_phase_count,
                0,
                status.stale_phase_count,
                status.orphan_phase_count,
                {"phase_rows_ready": status.phase_rows_ready},
                ("session_phases",),
            ),
            "threads": (
                "Threads",
                "threads",
                status.thread_count,
                status.root_threads,
                0,
                status.stale_thread_count,
                status.orphan_thread_count,
                {"threads_ready": status.threads_ready},
                ("threads", "thread_sessions"),
            ),
            "session_tag_rollups": (
                "Session Tag Rollups",
                "session_tags",
                status.tag_rollup_count,
                status.expected_tag_rollup_count,
                0,
                status.stale_tag_rollup_count,
                0,
                {"tag_rollups_ready": status.tag_rollups_ready},
                ("session_tags",),
            ),
            "archive_coverage": (
                "Archive Coverage",
                "sessions",
                total_sessions,
                total_sessions,
                0,
                0,
                0,
                {},
                ("sessions",),
            ),
        }
        spec = specs.get(name)
        if spec is None:
            return None
        (
            display_name,
            table_name,
            row_count,
            expected_row_count,
            missing_count,
            stale_count,
            orphan_count,
            ready_flags,
            artifact_names,
        ) = spec
        table_present = _table_exists(self._conn, table_name)
        artifacts = tuple(
            InsightStorageArtifact(
                name=artifact,
                present=_table_exists(self._conn, artifact),
                ready=ready_flags[next(iter(ready_flags))] if len(ready_flags) == 1 else None,
            )
            for artifact in artifact_names
        )
        # Provenance-backed insights (profiles, work events, phases) carry their
        # materializer version and source high-water mark in the
        # ``insight_materialization`` ledger; the #1278 fallback taxonomy lives in
        # each session profile's ``provenance_json``. Threads/tags/coverage have no
        # such ledger entry and keep the status-derived staleness only.
        version_coverage: tuple[InsightVersionCoverage, ...] = ()
        incompatible_count = 0
        materialization_type = _INSIGHT_MATERIALIZATION_TYPE.get(name)
        if materialization_type is not None and table_present:
            version_coverage, incompatible_count, native_stale = self._archive_materialization_signals(
                materialization_type, origin=origin, since_ms=since_ms, until_ms=until_ms
            )
            stale_count = native_stale
        degraded_count = 0
        fallback_reason_counts: dict[str, int] = {}
        fallback = _INSIGHT_FALLBACK_PAYLOAD.get(name)
        if fallback is not None and table_present:
            fallback_table, fallback_column_paths = fallback
            degraded_count, fallback_reason_counts = self._archive_fallback_coverage(
                fallback_table,
                fallback_column_paths,
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        verdict = _archive_insight_readiness_verdict(
            table_present=table_present,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            ready_flags=ready_flags,
            total_sessions=total_sessions,
        )
        return InsightReadinessEntry(
            insight_name=name,
            display_name=display_name,
            verdict=verdict,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            fallback_reason_counts=fallback_reason_counts,
            storage_artifacts=artifacts,
            ready_flags=ready_flags,
            origin_coverage=origin_coverage,
            version_coverage=version_coverage,
            evidence=_archive_insight_readiness_evidence(
                row_count=row_count,
                expected_row_count=expected_row_count,
                missing_count=missing_count,
                stale_count=stale_count,
                orphan_count=orphan_count,
                incompatible_count=incompatible_count,
                degraded_count=degraded_count,
                fallback_reason_counts=fallback_reason_counts,
                ready_flags=ready_flags,
            ),
        )

    def list_summaries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
        sample: bool = False,
        sort: str | None = None,
        reverse: bool = False,
    ) -> list[ArchiveSessionSummary]:
        """List session summaries ordered like the normal archive recency view."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        if session_id is not None:
            try:
                resolved_id = self.resolve_session_id(session_id)
            except KeyError:
                return []
            where = f"{where} AND s.session_id = ?" if where else "WHERE s.session_id = ?"
            params.append(resolved_id)
        order_by = _summary_order_by(sample=sample, sort=sort, reverse=reverse)
        params.extend([limit, 0 if sample else offset])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.parent_session_id,
                   s.session_kind,
                   s.message_count, s.word_count, s.reported_duration_ms,
                   s.tool_use_count, s.thinking_count, s.paste_count,
                   s.user_message_count, s.authored_user_message_count,
                   s.assistant_message_count, s.system_message_count,
                   s.tool_message_count, s.user_word_count, s.authored_user_word_count,
                   s.assistant_word_count,
                   s.title_source, s.title_ref, s.title_confidence, s.git_branch, s.git_repository_url, s.provider_project_ref,
                   s.display_name,
                   sp.terminal_state,
                   (SELECT COALESCE(SUM(u.cost_usd), s.reported_cost_usd) FROM session_model_usage u WHERE u.session_id = s.session_id) AS total_cost_usd,
                   COALESCE((SELECT MAX(u.cost_provenance) FROM session_model_usage u WHERE u.session_id = s.session_id), CASE WHEN s.reported_cost_usd IS NOT NULL THEN 'origin_reported' END) AS cost_provenance,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            {where}
            GROUP BY s.session_id
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [_summary_from_row(row, self._conn) for row in rows]

    def search_summaries(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        reverse: bool = False,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Search archive block text and return session-level hits with snippets."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            # Empty / whitespace / asterisk-only query: no FTS expression to
            # run. Mirror the read model lexical path and return no hits rather
            # than raising ``fts5: syntax error``.
            return []
        # A real query needs the block FTS index. Surface a degraded index as a
        # sanitized DatabaseError (→ 503 "Search index") instead of a raw
        # ``no such table`` 500 or a misleading empty-result 200.
        _ensure_messages_fts_ready(self._conn)
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            filter_params.append(session_id)
        order_by = _search_order_by(sort=sort, reverse=reverse)
        params: list[object] = [match_query, *filter_params]
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"""
            SELECT b.block_id, b.message_id, b.session_id, s.origin, s.native_id, s.title,
                   b.search_text AS fallback_text,
                   snippet(messages_fts, 4, '[', ']', '...', 12) AS snippet,
                   rank
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            JOIN sessions s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
            {where}
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [
            ArchiveSessionSearchHit(
                rank=index,
                session_id=str(row["session_id"]),
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                snippet=_highlight_search_snippet(
                    str(row["snippet"] or ""),
                    fallback=str(row["fallback_text"] or ""),
                    query=match_query,
                ),
            )
            for index, row in enumerate(rows, start=offset + 1)
        ]

    def count_search_sessions(
        self,
        query: str,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
    ) -> int:
        """Count distinct sessions matching the archive block FTS search."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            return 0
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            # Resolve against blocks, not sessions: b.session_id already carries
            # the value, so this predicate must not be what forces the join.
            where = f"{where} AND b.session_id = ?"
            filter_params.append(session_id)
        row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT b.session_id)
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            {_sessions_join_if_filtered(where)}
            WHERE messages_fts MATCH ?
            {where}
            """,
            [match_query, *filter_params],
        ).fetchone()
        return int(row[0] if row is not None else 0)

    def search_session_ids(
        self,
        query: str,
        *,
        limit: int | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
    ) -> tuple[str, ...]:
        """Return distinct sessions matching the archive block FTS search."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            return ()
        _ensure_messages_fts_ready(self._conn)
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        limit_clause = "" if limit is None else "LIMIT ?"
        params: list[object] = [match_query, *filter_params]
        if limit is not None:
            params.append(max(int(limit), 0))
        rows = self._conn.execute(
            f"""
            SELECT b.session_id, MIN(rank) AS best_rank
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            {_sessions_join_if_filtered(where)}
            WHERE messages_fts MATCH ?
            {where}
            GROUP BY b.session_id
            ORDER BY best_rank, b.session_id
            {limit_clause}
            """,
            params,
        ).fetchall()
        return tuple(str(row["session_id"]) for row in rows)

    def semantic_summaries(
        self,
        scored_message_ids: list[tuple[str, float]],
        *,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        root: bool | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Resolve vector-ranked message ids into filtered session-level hits."""
        if not scored_message_ids:
            return []
        message_ids = tuple(message_id for message_id, _score in scored_message_ids)
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            root=root,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        placeholders = ", ".join("?" for _ in message_ids)
        where = f"{where} AND m.message_id IN ({placeholders})" if where else f"WHERE m.message_id IN ({placeholders})"
        params.extend(message_ids)
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            params.append(session_id)
        rows = self._conn.execute(
            f"""
            SELECT m.message_id, m.session_id, s.origin, s.native_id, s.title,
                   b.block_id, b.text
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            LEFT JOIN blocks b
              ON b.message_id = m.message_id
             AND b.position = (
                 SELECT MIN(position)
                 FROM blocks
                 WHERE message_id = m.message_id
                   AND text IS NOT NULL
             )
            {where}
            """,
            params,
        ).fetchall()
        rows_by_message_id = {str(row["message_id"]): row for row in rows}
        deduped: list[ArchiveSessionSearchHit] = []
        seen_sessions: set[str] = set()
        for message_id, _score in scored_message_ids:
            row = rows_by_message_id.get(message_id)
            if row is None:
                continue
            session_id = str(row["session_id"])
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            text = str(row["text"] or "")
            deduped.append(
                ArchiveSessionSearchHit(
                    rank=len(deduped) + 1,
                    session_id=session_id,
                    block_id=str(row["block_id"] or message_id),
                    message_id=message_id,
                    origin=str(row["origin"]),
                    title=str(row["title"]) if row["title"] is not None else None,
                    snippet=text[:160],
                )
            )
        page = deduped[offset : offset + limit]
        return [replace(hit, rank=offset + index) for index, hit in enumerate(page, start=1)]

    def query_messages(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveMessageQueryRow]:
        return _archive_query_reads.query_messages(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_session_messages(
        self,
        session_ids: Sequence[str],
        *,
        limit: int = 50,
        offset: int = 0,
        sort_direction: Literal["asc", "desc"] = "asc",
        roles: Sequence[str] = (),
        message_type: str | None = None,
        material_origins: Sequence[str] = (),
    ) -> list[ArchiveMessageQueryRow]:
        return _archive_query_reads.query_session_messages(
            self,
            session_ids,
            limit=limit,
            offset=offset,
            sort_direction=sort_direction,
            roles=roles,
            message_type=message_type,
            material_origins=material_origins,
        )

    def count_session_messages(
        self,
        session_ids: Sequence[str],
        *,
        roles: Sequence[str] = (),
        message_type: str | None = None,
        material_origins: Sequence[str] = (),
    ) -> int:
        return _archive_query_reads.count_session_messages(
            self, session_ids, roles=roles, message_type=message_type, material_origins=material_origins
        )

    def query_unit_counts(
        self,
        unit: str,
        predicate: QueryPredicate,
        *,
        group_by: str | None = None,
        sort: Literal["count", "key"] | None = None,
        sort_direction: Literal["asc", "desc"] = "desc",
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
    ) -> list[ArchiveQueryUnitAggregateRow]:
        return _archive_query_reads.query_unit_counts(
            self,
            unit,
            predicate,
            group_by=group_by,
            sort=sort,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
        )

    def query_unit_multi_counts(
        self,
        unit: str,
        predicate: QueryPredicate,
        *,
        group_by: Sequence[str],
        sort: Literal["count", "key"] | None = None,
        sort_direction: Literal["asc", "desc"] = "desc",
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
    ) -> ArchiveQueryUnitMultiAggregatePage:
        return _archive_query_reads.query_unit_multi_counts(
            self,
            unit,
            predicate,
            group_by=group_by,
            sort=sort,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
        )

    def query_actions(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveActionQueryRow]:
        return _archive_query_reads.query_actions(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_session_actions(
        self,
        session_ids: Sequence[str],
        *,
        limit: int = 50,
        offset: int = 0,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveActionQueryRow]:
        return _archive_query_reads.query_session_actions(
            self, session_ids, limit=limit, offset=offset, sort_direction=sort_direction
        )

    def query_session_action_occurrences(
        self,
        session_ids: Sequence[str],
        *,
        limit: int = 50,
        offset: int = 0,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveActionQueryRow]:
        return _archive_query_reads.query_session_action_occurrences(
            self, session_ids, limit=limit, offset=offset, sort_direction=sort_direction
        )

    def get_delegation_attempt(
        self,
        *,
        instruction_tool_use_block_id: str | None = None,
        parent_session_id: str | None = None,
        child_session_id: str | None = None,
    ) -> ArchiveDelegationQueryRow | None:
        return _archive_query_reads.get_delegation_attempt(
            self,
            instruction_tool_use_block_id=instruction_tool_use_block_id,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
        )

    def get_delegation_card(
        self,
        *,
        instruction_tool_use_block_id: str | None = None,
        parent_session_id: str | None = None,
        child_session_id: str | None = None,
    ) -> ArchiveDelegationCard | None:
        return _archive_query_reads.get_delegation_card(
            self,
            instruction_tool_use_block_id=instruction_tool_use_block_id,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
        )

    def query_delegations(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveDelegationQueryRow]:
        return _archive_query_reads.query_delegations(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def get_delegation_ancestry(self, session_id: str) -> list[ArchiveDelegationAncestryRow]:
        return _archive_query_reads.get_delegation_ancestry(self, session_id)

    def get_delegation_subtree(self, session_id: str) -> list[ArchiveDelegationSubtreeRow]:
        return _archive_query_reads.get_delegation_subtree(self, session_id)

    def query_files(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveFileQueryRow]:
        return _archive_query_reads.query_files(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_session_files(
        self,
        session_ids: Sequence[str],
        *,
        limit: int = 50,
        offset: int = 0,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveFileQueryRow]:
        return _archive_query_reads.query_session_files(
            self, session_ids, limit=limit, offset=offset, sort_direction=sort_direction
        )

    def _query_file_counts(
        self,
        predicate: QueryPredicate,
        *,
        group_by: str | None,
        sort: Literal["count", "key"] | None,
        sort_direction: Literal["asc", "desc"],
        limit: int,
        offset: int,
        session_filters: Mapping[str, object] | None,
    ) -> list[ArchiveQueryUnitAggregateRow]:
        return _archive_query_reads._query_file_counts(
            self,
            predicate,
            group_by=group_by,
            sort=sort,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
        )

    def query_blocks(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveBlockQueryRow]:
        return _archive_query_reads.query_blocks(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_assertions(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveAssertionQueryRow]:
        return _archive_query_reads.query_assertions(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_runs(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveRunQueryRow]:
        return _archive_query_reads.query_runs(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_observed_events(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveObservedEventQueryRow]:
        return _archive_query_reads.query_observed_events(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def query_context_snapshots(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
        session_filters: Mapping[str, object] | None = None,
        sort: Literal["time"] | None = None,
        sort_direction: Literal["asc", "desc"] = "asc",
    ) -> list[ArchiveContextSnapshotQueryRow]:
        return _archive_query_reads.query_context_snapshots(
            self,
            predicate,
            limit=limit,
            offset=offset,
            session_filters=session_filters,
            sort=sort,
            sort_direction=sort_direction,
        )

    def stats(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
        root: bool | None = None,
    ) -> ArchiveStats:
        """Return archive-level stats from filtered archive index sessions."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            root=root,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS total_sessions,
                   COALESCE(SUM(s.message_count), 0) AS total_messages
            FROM sessions s
            {where}
            """,
            params,
        ).fetchone()
        provider_rows = self._conn.execute(
            f"""
            SELECT s.origin, COUNT(*) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, s.origin
            """,
            params,
        ).fetchall()
        attachment_row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT ar.attachment_id) AS total_attachments
            FROM sessions s
            JOIN attachment_refs ar ON ar.session_id = s.session_id
            {where}
            """,
            params,
        ).fetchone()
        role_row = self._conn.execute(
            f"""
            SELECT COALESCE(SUM(s.user_message_count), 0) AS user_count,
                   COALESCE(SUM(s.assistant_message_count), 0) AS assistant_count,
                   COALESCE(SUM(s.system_message_count), 0) AS system_count,
                   COALESCE(SUM(s.tool_message_count), 0) AS tool_count,
                   COALESCE(SUM(s.message_count), 0)
                     - COALESCE(SUM(s.user_message_count), 0)
                     - COALESCE(SUM(s.assistant_message_count), 0)
                     - COALESCE(SUM(s.system_message_count), 0)
                     - COALESCE(SUM(s.tool_message_count), 0) AS unknown_count
            FROM sessions s
            {where}
            """,
            params,
        ).fetchone()
        if where:
            message_type_rows = self._conn.execute(
                f"""
                SELECT m.message_type AS group_key,
                       COUNT(*) AS count
                FROM sessions s
                JOIN messages m ON m.session_id = s.session_id
                {where}
                GROUP BY m.message_type
                ORDER BY count DESC, m.message_type
                """,
                params,
            ).fetchall()
            material_origin_rows = self._conn.execute(
                f"""
                SELECT m.material_origin AS group_key,
                       COUNT(*) AS count
                FROM sessions s
                JOIN messages m ON m.session_id = s.session_id
                {where}
                GROUP BY m.material_origin
                ORDER BY count DESC, m.material_origin
                """,
                params,
            ).fetchall()
        else:
            message_type_rows = self._conn.execute(
                """
                SELECT m.message_type AS group_key,
                       COUNT(*) AS count
                FROM messages m
                GROUP BY m.message_type
                ORDER BY count DESC, m.message_type
                """
            ).fetchall()
            material_origin_rows = self._conn.execute(
                """
                SELECT m.material_origin AS group_key,
                       COUNT(*) AS count
                FROM messages m
                GROUP BY m.material_origin
                ORDER BY count DESC, m.material_origin
                """
            ).fetchall()
        return ArchiveStats(
            total_sessions=int(row["total_sessions"] or 0) if row is not None else 0,
            total_messages=int(row["total_messages"] or 0) if row is not None else 0,
            total_attachments=int(attachment_row["total_attachments"] or 0) if attachment_row is not None else 0,
            origins={str(provider_row["origin"]): int(provider_row["count"] or 0) for provider_row in provider_rows},
            role_counts={
                key: count
                for key, count in (
                    ("tool", int(role_row["tool_count"] or 0) if role_row is not None else 0),
                    ("assistant", int(role_row["assistant_count"] or 0) if role_row is not None else 0),
                    ("user", int(role_row["user_count"] or 0) if role_row is not None else 0),
                    ("system", int(role_row["system_count"] or 0) if role_row is not None else 0),
                    ("unknown", int(role_row["unknown_count"] or 0) if role_row is not None else 0),
                )
                if count > 0
            },
            message_types={str(item["group_key"] or "unknown"): int(item["count"] or 0) for item in message_type_rows},
            material_origins={
                str(item["group_key"] or "unknown"): int(item["count"] or 0) for item in material_origin_rows
            },
            db_size_bytes=self.index_db_path.stat().st_size if self.index_db_path.exists() else 0,
        )

    def stats_by(
        self,
        group_by: str,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        project_refs: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
        root: bool | None = None,
    ) -> dict[str, int]:
        """Return filtered session counts grouped by a archive dimension."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            root=root,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        rows = self._conn.execute(_stats_by_sql(group_by, where, tags_relation=self._tags_relation), params).fetchall()
        results = {str(row["group_key"]): int(row["count"] or 0) for row in rows if row["group_key"] is not None}
        return results

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _summary_from_row(row: sqlite3.Row, conn: sqlite3.Connection) -> ArchiveSessionSummary:
    import json

    def row_int(key: str) -> int:
        try:
            value = row[key]
        except IndexError:
            return 0
        return int(value or 0)

    raw_tags = json.loads(str(row["tags_json"] or "[]"))
    tags = tuple(str(tag) for tag in raw_tags if tag is not None)
    raw_working_dirs = json.loads(str(row["working_directories_json"] or "[]"))
    working_directories = tuple(str(path) for path in raw_working_dirs if path)
    origin = str(row["origin"])
    session_id = str(row["session_id"])
    message_count = int(row["message_count"] or 0)
    raw_title = str(row["title"]) if row["title"] is not None else None
    raw_title_source = str(row["title_source"]) if row["title_source"] is not None else None
    # A non-blank ``sessions.title`` is only a genuine provider title when
    # ``title_source`` says so. ``title_source='unknown'`` rows still
    # carry a NON-NULL title -- the writer's pre-cijx.4 fallback stores the
    # raw native id there (a bare UUID, or "<uuid>:agent-<hash>" for a
    # subagent), which is exactly the "worse than the UUID it replaces" case
    # decision 3 exists to fix. Measured live: 7,501 of 15,401 root sessions
    # (48.7%) carry title_source='unknown' -- checking only "is title
    # non-blank" (the pre-fix condition) made the structural-label fallback
    # dead code for all of them. ``title_source='path'`` is the structural
    # label's own prior output; treating it as "not a real title" keeps this
    # idempotent on rebuild instead of freezing a stale message count.
    # polylogue-5dfu: ``TitleSource.UNKNOWN``/``TitleSource.USER`` were
    # deleted from the enum (UNKNOWN was a redundant second "no evidence"
    # spelling of NULL; USER had zero producers) but this still checks the
    # bare *strings* -- an un-rebuilt archive can carry either value as
    # stale on-disk data from before this change until its next full
    # reparse, and both must keep failing this membership test exactly as
    # they did before.
    has_real_title = bool(raw_title and raw_title.strip()) and raw_title_source in {"origin", "heuristic"}
    provider_title = raw_title if has_real_title else None
    try:
        raw_display_name = row["display_name"]
    except IndexError:
        # Not every caller's SELECT projects display_name; treat absence as
        # unknown rather than raising (matching parent_id's guard below).
        display_name: str | None = None
    else:
        display_name = str(raw_display_name).strip() or None if raw_display_name is not None else None
    session_date = _iso_from_ms(row["created_at_ms"]) or _iso_from_ms(row["updated_at_ms"])
    structural_label = session_structural_label_for_session(
        conn,
        session_id,
        message_count=message_count,
        provider_title=None,
        session_date=session_date,
    )
    display_label = provider_title or display_name or structural_label
    title = provider_title
    title_source = raw_title_source if provider_title is not None else None
    parent_id: str | None
    try:
        raw_parent_id = row["parent_session_id"]
    except IndexError:
        # Not every caller's SELECT projects parent_session_id (e.g. rows built
        # for contexts that never need root/child filtering); treat absence as
        # unknown rather than raising, matching row_int's IndexError handling
        # above for other optional columns.
        parent_id = None
    else:
        parent_id = str(raw_parent_id) if raw_parent_id else None
    return ArchiveSessionSummary(
        session_id=session_id,
        native_id=str(row["native_id"]),
        origin=origin,
        title=title,
        display_label=display_label,
        title_source=title_source,
        parent_id=parent_id,
        title_ref=str(row["title_ref"]) if row["title_ref"] is not None else None,
        title_confidence=(float(row["title_confidence"]) if row["title_confidence"] is not None else None),
        session_kind=str(row["session_kind"] or "standard"),
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        message_count=message_count,
        word_count=int(row["word_count"] or 0),
        tags=tags,
        reported_duration_ms=(int(row["reported_duration_ms"]) if row["reported_duration_ms"] is not None else None),
        tool_use_count=row_int("tool_use_count"),
        thinking_count=row_int("thinking_count"),
        paste_count=row_int("paste_count"),
        user_message_count=row_int("user_message_count"),
        authored_user_message_count=row_int("authored_user_message_count"),
        assistant_message_count=row_int("assistant_message_count"),
        system_message_count=row_int("system_message_count"),
        tool_message_count=row_int("tool_message_count"),
        user_word_count=row_int("user_word_count"),
        authored_user_word_count=row_int("authored_user_word_count"),
        assistant_word_count=row_int("assistant_word_count"),
        working_directories=working_directories,
        git_branch=str(row["git_branch"]) if row["git_branch"] is not None else None,
        git_repository_url=str(row["git_repository_url"]) if row["git_repository_url"] is not None else None,
        provider_project_ref=(str(row["provider_project_ref"]) if row["provider_project_ref"] is not None else None),
        display_name=display_name,
        terminal_state=(str(row["terminal_state"]) if row["terminal_state"] is not None else None),
        total_cost_usd=(float(row["total_cost_usd"]) if row["total_cost_usd"] is not None else None),
        cost_provenance=(str(row["cost_provenance"]) if row["cost_provenance"] is not None else None),
    )


def _highlight_search_snippet(snippet: str, *, fallback: str, query: str) -> str:
    """Return bracket-highlighted text when contentless FTS omits markers."""
    import re

    text = snippet or fallback
    if "[" in text and "]" in text:
        return text
    terms = [term.strip('"') for term in re.findall(r'"[^"]+"|[\w.-]+', query) if term.strip('"')]
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(text):
            return str(pattern.sub(lambda match: f"[{match.group(0)}]", text, count=1))
    return text


def _summary_order_by(*, sample: bool, sort: str | None, reverse: bool) -> str:
    if sample or sort == "random":
        return "ORDER BY RANDOM()"
    direction = "ASC" if reverse else "DESC"
    if sort in {None, "date"}:
        return f"ORDER BY s.sort_key_ms IS NULL, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "messages":
        return f"ORDER BY s.message_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "words":
        return f"ORDER BY s.word_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "longest":
        return f"""
            ORDER BY (
                SELECT COALESCE(MAX(m.word_count), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    if sort == "tokens":
        return f"""
            ORDER BY (
                SELECT COALESCE(SUM(m.input_tokens + m.output_tokens + m.cache_read_tokens + m.cache_write_tokens), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    raise ValueError("archive root query sort must be one of date, messages, words, longest, tokens, random.")


def _search_order_by(*, sort: str | None, reverse: bool) -> str:
    if sort is None:
        return "ORDER BY rank DESC" if reverse else "ORDER BY rank"
    return _summary_order_by(sample=False, sort=sort, reverse=reverse)


def _with_session_id_filter(
    where: str,
    params: list[object],
    table_alias: str,
    *,
    session_ids: tuple[str, ...],
) -> tuple[str, list[object]]:
    if not session_ids:
        return where, params
    placeholders = ", ".join("?" for _ in session_ids)
    clause = f"{table_alias}.session_id IN ({placeholders})"
    merged_params = [*params, *session_ids]
    if where:
        return f"{where} AND {clause}", merged_params
    return f"WHERE {clause}", merged_params


def _sessions_join_if_filtered(where: str) -> str:
    """Join ``sessions`` only when the WHERE clause actually references it.

    The FTS search queries carry an optional session-level filter built with
    table alias ``s``. When no such filter is supplied -- the common case, e.g.
    a bare ``polylogue find "docker"`` -- the join resolved nothing: neither
    query projects an ``s.*`` column, and ``blocks.session_id`` already carries
    the value. Every matched block paid a sessions-PK probe for nothing.

    Measured on the live archive (18,871 sessions / 4.9M messages): dropping
    the join roughly halves random reads per matched block (2.50 -> 1.29
    pread64 syscalls, controlled A/B on two never-queried terms of comparable
    match volume) and cuts warm wall time ~18%. Cold, the exact-count query
    for a common term was ~1.15s. This is not a missing index -- EXPLAIN
    already showed correct index use on both sides -- it is a join that should
    not have been there.

    Keyed on the rendered WHERE text so the join can never be elided while
    something still references the alias: if a filter is present the join is
    emitted verbatim, and callers that filter by a bare session id resolve it
    against ``b.session_id`` instead so that predicate alone does not drag the
    join back in.
    """
    return "JOIN sessions s ON s.session_id = b.session_id" if "s." in where else ""


def _with_since_session_filter(
    conn: sqlite3.Connection,
    where: str,
    params: list[object],
    table_alias: str,
    *,
    since_session_id: str | None,
    prefix: str = "WHERE",
) -> tuple[str, list[object]]:
    if since_session_id is None:
        return where, params
    reference = _since_session_reference(conn, since_session_id)
    if reference is None:
        clause = "0 = 1"
        if where:
            return f"{where} AND {clause}", params
        return f"{prefix} {clause}", params
    ref_session_id, ref_sort_key_ms, ref_paths = reference
    clauses = [f"{table_alias}.session_id != ?"]
    merged_params: list[object] = [*params, ref_session_id]
    if ref_sort_key_ms is not None:
        clauses.append(f"{table_alias}.sort_key_ms > ?")
        merged_params.append(ref_sort_key_ms)
    if ref_paths:
        path_clauses: list[str] = []
        for ref_path in ref_paths:
            exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(ref_path)
            path_clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM session_working_dirs since_cwd
                    WHERE since_cwd.session_id = {table_alias}.session_id
                      AND (
                        REPLACE(since_cwd.path, char(92), '/') = ?
                        OR REPLACE(since_cwd.path, char(92), '/') LIKE ? ESCAPE '\\'
                      )
                )
                """.strip()
            )
            merged_params.extend([exact_prefix, child_prefix])
        clauses.append("(" + " OR ".join(path_clauses) + ")")
    clause = " AND ".join(clauses)
    if where:
        return f"{where} AND {clause}", merged_params
    return f"{prefix} {clause}", merged_params


def _since_session_reference(
    conn: sqlite3.Connection,
    token: str,
) -> tuple[str, int | None, tuple[str, ...]] | None:
    lower_bound, upper_bound = session_id_prefix_bounds(token)
    prefix_clause = "s.session_id >= ?"
    prefix_params: list[str] = [lower_bound]
    if upper_bound is not None:
        prefix_clause = f"{prefix_clause} AND s.session_id < ?"
        prefix_params.append(upper_bound)
    rows = conn.execute(
        f"""
        SELECT s.session_id,
               COALESCE(
                   (SELECT MAX(m.occurred_at_ms) FROM messages m WHERE m.session_id = s.session_id),
                   s.sort_key_ms
               ) AS anchor_ms
        FROM sessions s
        WHERE s.session_id = ? OR ({prefix_clause})
        ORDER BY CASE WHEN s.session_id = ? THEN 0 ELSE 1 END, s.session_id
        LIMIT 2
        """,
        (token, *prefix_params, token),
    ).fetchall()
    if not rows:
        return None
    row = rows[0]
    session_id = str(row["session_id"])
    path_rows = conn.execute(
        """
        SELECT path
        FROM session_working_dirs
        WHERE session_id = ?
        ORDER BY position, path
        """,
        (session_id,),
    ).fetchall()
    paths = tuple(str(path_row["path"]) for path_row in path_rows if path_row["path"])
    anchor_value = row["anchor_ms"]
    return session_id, int(anchor_value) if anchor_value is not None else None, paths


def _all_session_tags_sql() -> str:
    return """
        (
            SELECT session_id, tag, tag_source, method, confidence, evidence_json
            FROM session_tags
            WHERE tag_source = 'auto'
            UNION ALL
            SELECT
                substr(target_ref, 9) AS session_id,
                COALESCE(key, body_text) AS tag,
                'user' AS tag_source,
                json_extract(value_json, '$.method') AS method,
                confidence,
                json_extract(value_json, '$.evidence') AS evidence_json
            FROM user_tier.assertions
            WHERE kind = 'tag'
              AND target_ref LIKE 'session:%'
              AND COALESCE(status, 'active') != 'deleted'
              AND COALESCE(key, body_text) IS NOT NULL
        )
    """


def _split_user_target_ref(target_ref: str) -> tuple[str, str]:
    target_type, sep, target_id = target_ref.partition(":")
    if not sep:
        return "", target_ref
    return target_type, target_id


def _user_state_session_ids(
    assertions: Iterable[ArchiveAssertionEnvelope],
    *,
    index_conn: sqlite3.Connection | None = None,
) -> dict[str, str]:
    """Resolve assertion owners with one bounded indexed-message lookup batch."""
    owners: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    for assertion in assertions:
        target_type, target_id = _split_user_target_ref(assertion.target_ref)
        if target_type == "session":
            owners[assertion.assertion_id] = target_id
            continue
        if target_type != "message":
            owners[assertion.assertion_id] = ""
            continue
        if assertion.scope_ref is not None and assertion.scope_ref.startswith("session:"):
            durable_owner = assertion.scope_ref[len("session:") :]
            if durable_owner:
                owners[assertion.assertion_id] = durable_owner
                continue
        unresolved.append((assertion.assertion_id, target_id))

    indexed_owners: dict[str, str] = {}
    if index_conn is not None:
        for offset in range(0, len(unresolved), 500):
            batch = sorted({target_id for _assertion_id, target_id in unresolved[offset : offset + 500]})
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = index_conn.execute(
                f"SELECT message_id, session_id FROM messages WHERE message_id IN ({placeholders})",
                batch,
            ).fetchall()
            indexed_owners.update({str(row[0]): str(row[1]) for row in rows})
    for assertion_id, target_id in unresolved:
        owners[assertion_id] = indexed_owners.get(target_id, "")
    return owners


def _id_from_target_ref(target_ref: str, prefix: str) -> str:
    return target_ref[len(prefix) :] if target_ref.startswith(prefix) else target_ref


def _active_assertion_by_kind_key(
    conn: sqlite3.Connection,
    kind: str,
    key: str,
) -> ArchiveAssertionEnvelope | None:
    for assertion in list_assertions_by_kind(conn, kind):
        if assertion.key == key:
            return assertion
    return None


def _learning_correction_from_archive_row(row: sqlite3.Row | tuple[object, ...]) -> LearningCorrection:
    session_id = str(row[0])
    kind = parse_correction_kind(str(row[1]))
    try:
        stored = json.loads(str(row[2]))
    except json.JSONDecodeError:
        stored = {}
    if isinstance(stored, dict) and isinstance(stored.get("payload"), dict):
        payload = {str(key): str(value) for key, value in dict(stored["payload"]).items()}
        note_raw = stored.get("note")
        note = str(note_raw) if note_raw is not None else None
    elif isinstance(stored, dict):
        payload = {str(key): str(value) for key, value in stored.items()}
        note = None
    else:
        payload = {}
        note = None
    raw_updated_at_ms = row[3]
    updated_at_ms = int(str(raw_updated_at_ms or 0))
    return LearningCorrection(
        session_id=session_id,
        kind=kind,
        payload=payload,
        note=note,
        created_at=datetime.fromtimestamp(updated_at_ms / 1000.0, tz=UTC),
    )


def _origin_value(origin: str | None) -> str | None:
    if origin is None:
        return None
    if origin == "":
        return Origin.UNKNOWN_EXPORT.value
    return Origin(origin).value


def _origin_for_tool_usage_filter(origin: str | None) -> str | None:
    return _origin_value(origin)


def _session_origin(conn: sqlite3.Connection, session_id: str) -> str:
    row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return str(row["origin"]) if row is not None else "unknown-export"


def _read_archive_materialization(
    conn: sqlite3.Connection,
    insight_type: str,
    session_id: str,
) -> ArchiveInsightMaterialization | None:
    try:
        return read_insight_materialization(conn, insight_type, session_id)
    except KeyError:
        return None


def _archive_provenance(
    materialization: ArchiveInsightMaterialization | None,
    *,
    input_high_water_mark: str | None = None,
    input_high_water_mark_source: str | None = None,
) -> ArchiveInsightProvenance:
    if materialization is None:
        return ArchiveInsightProvenance(
            materializer_version=1,
            materialized_at=None,
            input_high_water_mark=input_high_water_mark,
            input_high_water_mark_source=input_high_water_mark_source,
            time_confidence=time_confidence_for_source(input_high_water_mark_source),
        )
    resolved_hwm = (
        input_high_water_mark
        if input_high_water_mark is not None
        else _iso_from_ms(materialization.input_high_water_mark_ms)
    )
    resolved_source = (
        input_high_water_mark_source
        if input_high_water_mark_source is not None
        else materialization.input_high_water_mark_source
    )
    return ArchiveInsightProvenance(
        materializer_version=materialization.materializer_version,
        materialized_at=_iso_from_ms(materialization.materialized_at_ms),
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms),
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0 if materialization.source_sort_key_ms is not None else None
        ),
        input_high_water_mark=resolved_hwm,
        input_high_water_mark_source=resolved_source,
        time_confidence=time_confidence_for_source(resolved_source),
    )


def _archive_inference_provenance(
    materialization: ArchiveInsightMaterialization | None,
    *,
    input_high_water_mark: str | None = None,
    input_high_water_mark_source: str | None = None,
) -> ArchiveInferenceProvenance:
    base = _archive_provenance(
        materialization,
        input_high_water_mark=input_high_water_mark,
        input_high_water_mark_source=input_high_water_mark_source,
    )
    return ArchiveInferenceProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        input_high_water_mark=base.input_high_water_mark,
        input_high_water_mark_source=base.input_high_water_mark_source,
        time_confidence=base.time_confidence,
        inference_version=base.materializer_version,
        inference_family="archive",
    )


def _archive_enrichment_provenance(
    materialization: ArchiveInsightMaterialization | None,
) -> ArchiveEnrichmentProvenance:
    base = _archive_provenance(materialization)
    return ArchiveEnrichmentProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        input_high_water_mark=base.input_high_water_mark,
        input_high_water_mark_source=base.input_high_water_mark_source,
        time_confidence=base.time_confidence,
        enrichment_version=base.materializer_version,
        enrichment_family="archive",
    )


def _work_event_insight_from_archive_row(
    event: ArchiveSessionWorkEvent,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization | None,
) -> SessionWorkEventInsight:
    evidence_payload = {
        **event.evidence,
        "start_index": event.start_index,
        "end_index": event.end_index,
        "start_time": _iso_from_ms(event.started_at_ms),
        "end_time": _iso_from_ms(event.ended_at_ms),
        "duration_ms": event.duration_ms,
        "file_paths": event.file_paths,
        "tools_used": event.tools_used,
    }
    inference_payload = {
        **event.inference,
        "heuristic_label": event.work_event_type,
        "summary": event.summary,
        "confidence": event.confidence,
        "support_level": confidence_from_score(event.confidence),
    }
    return SessionWorkEventInsight(
        event_id=event.event_id,
        session_id=event.session_id,
        origin=origin,
        event_index=event.position,
        provenance=_archive_provenance(
            materialization,
            input_high_water_mark=event.input_high_water_mark,
            input_high_water_mark_source=event.input_high_water_mark_source,
        ),
        inference_provenance=_archive_inference_provenance(
            materialization,
            input_high_water_mark=event.input_high_water_mark,
            input_high_water_mark_source=event.input_high_water_mark_source,
        ),
        evidence=WorkEventEvidencePayload.model_validate(evidence_payload),
        inference=WorkEventInferencePayload.model_validate(inference_payload),
    )


def _phase_insight_from_archive_row(
    phase: ArchiveSessionPhase,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization | None,
) -> SessionPhaseInsight:
    evidence_payload = {
        **phase.evidence,
        "start_time": _iso_from_ms(phase.started_at_ms),
        "end_time": _iso_from_ms(phase.ended_at_ms),
        "message_range": (phase.start_index, phase.end_index),
        "duration_ms": phase.duration_ms,
        "tool_counts": phase.tool_counts,
        "word_count": phase.word_count,
    }
    return SessionPhaseInsight(
        phase_id=phase.phase_id,
        session_id=phase.session_id,
        origin=origin,
        phase_index=phase.position,
        provenance=_archive_provenance(
            materialization,
            input_high_water_mark=phase.input_high_water_mark,
            input_high_water_mark_source=phase.input_high_water_mark_source,
        ),
        evidence=SessionPhaseEvidencePayload.model_validate(evidence_payload),
    )


@dataclass(frozen=True)
class _SessionProfileComponents:
    """Extracted session-profile payloads shared by the insight and record builders."""

    materialization: ArchiveInsightMaterialization | None
    evidence: SessionEvidencePayload
    inference: SessionInferencePayload
    enrichment: SessionEnrichmentPayload | None


def _session_profile_components_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> _SessionProfileComponents:
    """Build the evidence/inference/enrichment payloads from a session profile row.

    This is the shared extraction used by both
    :func:`_session_profile_insight_from_archive_row` (tier-gated insight projection)
    and :meth:`ArchiveStore.get_session_profile_record` (full domain-record
    hydration). All three payloads are always materialized here; the insight
    builder applies tier gating on top.

    Reads the typed *_payload_json columns written by the canonical
    session-profile writer (replace_session_profiles_bulk_sync).  The legacy
    provenance_json column has been dropped from the DDL.
    """
    from polylogue.storage.sqlite.queries.mappers_insight_fallback import parse_payload_model

    session_id = str(row["session_id"])
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    workflow_shape = str(row["workflow_shape"] or "unknown")
    workflow_confidence = float(row["workflow_shape_confidence"] or 0.0)
    terminal_state = str(row["terminal_state"] or "unknown")
    terminal_method = str(row["terminal_state_method"] or "unknown")
    terminal_confidence = float(row["terminal_state_confidence"] or 0.0)

    evidence = parse_payload_model(row, "evidence_payload_json", record_id=session_id, model=SessionEvidencePayload)
    if evidence is None:
        # Fallback for rows written before the typed-column migration: build
        # a minimal payload from the direct session/profile row columns.
        evidence = SessionEvidencePayload.model_validate(
            {
                "created_at": _iso_from_ms(row["created_at_ms"]),
                "updated_at": _iso_from_ms(row["updated_at_ms"]),
                "message_count": int(row["message_count"] or 0),
                "substantive_count": int(row["substantive_count"] or 0),
                "attachment_count": int(row["attachment_count"] or 0),
                "tool_use_count": int(row["tool_use_count"] or 0),
                "thinking_count": int(row["thinking_count"] or 0),
                "word_count": int(row["word_count"] or 0),
                "total_cost_usd": float(row["total_cost_usd"] or row["cost_usd"] or 0.0),
                "total_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
                "cost_is_estimated": bool(row["cost_is_estimated"]),
                "cost_provenance": str(row["cost_provenance"] or "unknown"),
                "logical_session_id": str(row["root_session_id"] or session_id),
                "tool_calls_per_minute": float(row["tool_calls_per_minute"] or 0.0),
            }
        )

    inference = parse_payload_model(row, "inference_payload_json", record_id=session_id, model=SessionInferencePayload)
    if inference is None:
        inference = SessionInferencePayload.model_validate(
            {
                "work_event_count": int(row["work_event_count"] or 0),
                "phase_count": int(row["phase_count"] or 0),
                "engaged_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "engaged_minutes": float(row["total_duration_ms"] or row["duration_ms"] or 0) / 60000.0,
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_method": terminal_method,
                "terminal_state_confidence": terminal_confidence,
                "support_level": confidence_from_score(max(workflow_confidence, terminal_confidence)),
            }
        )
    else:
        # The denormalized native session_profiles columns are the authoritative
        # ranking signals; reconcile the JSON-derived payload onto them so resume
        # ranking and aggregation read the queryable native columns rather than a
        # divergent payload copy.
        inference = inference.model_copy(
            update={
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_method": terminal_method,
                "terminal_state_confidence": terminal_confidence,
            }
        )

    enrichment = parse_payload_model(
        row, "enrichment_payload_json", record_id=session_id, model=SessionEnrichmentPayload
    )
    if enrichment is not None:
        # polylogue-37t.23: same reconciliation as `inference.terminal_state`
        # above -- the structural_inference tier of `objective_posture` is a
        # pure function of terminal_state, so it is recomputed onto the
        # authoritative native columns rather than trusted from the
        # enrichment JSON (which may have been materialized before a
        # native-column repair/backfill).
        enrichment = enrichment.model_copy(
            update={
                "objective_posture": structural_objective_posture(
                    terminal_state=inference.terminal_state,
                    terminal_state_confidence=inference.terminal_state_confidence,
                    terminal_state_evidence=dict(evidence.terminal_state_evidence),
                    as_of=_iso_from_ms(materialization.source_updated_at_ms) if materialization is not None else None,
                )
            }
        )
    return _SessionProfileComponents(
        materialization=materialization,
        evidence=evidence,
        inference=inference,
        enrichment=enrichment,
    )


def _session_profile_insight_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    tier: str,
) -> SessionProfileInsight:
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    include_evidence = tier in {"merged", "evidence"}
    include_inference = tier in {"merged", "inference"}
    include_enrichment = tier == "merged"
    evidence = components.evidence if include_evidence else None
    inference = components.inference if include_inference else None
    enrichment = None
    enrichment_provenance = None
    if include_enrichment and components.enrichment is not None:
        enrichment = components.enrichment
        enrichment_provenance = _archive_enrichment_provenance(materialization)
    return SessionProfileInsight(
        semantic_tier=tier,
        session_id=session_id,
        logical_session_id=str(row["root_session_id"] or session_id),
        origin=str(row["origin"]),
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        evidence=evidence,
        inference_provenance=_archive_inference_provenance(materialization) if include_inference else None,
        inference=inference,
        enrichment_provenance=enrichment_provenance,
        enrichment=enrichment,
    )


def _session_profile_record_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> SessionProfileRecord:
    """Build the full domain :class:`SessionProfileRecord` from a session profile row.

    Reuses the same payload extraction as the insight projection and pulls the
    materialization HWM provenance from ``read_insight_materialization`` so the
    record carries the fields ``hydrate_session_profile`` and the
    ``is_stale`` staleness check expect. The FTS-only ``*_search_text`` fields
    are not stored in the archive ``session_profiles`` row and are not read by
    ``hydrate_session_profile``; they are synthesized as non-empty strings here
    purely to satisfy the model's required-non-empty validators.
    """
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    evidence = components.evidence
    inference = components.inference
    enrichment = components.enrichment if components.enrichment is not None else SessionEnrichmentPayload()
    logical_session_id = str(row["root_session_id"] or session_id)
    source_name = str(row["origin"])
    title = str(row["title"]) if row["title"] is not None else None
    workflow_shape = str(row["workflow_shape"] or "unknown")
    materialized_at = _iso_from_ms(materialization.materialized_at_ms) if materialization is not None else None
    # search_text* are FTS-only and not consumed by hydrate_session_profile;
    # synthesize a stable non-empty string so the record validates.
    search_text = title or workflow_shape or session_id
    return SessionProfileRecord(
        session_id=SessionId(session_id),
        logical_session_id=SessionId(logical_session_id),
        materializer_version=materialization.materializer_version if materialization is not None else 1,
        materialized_at=materialized_at,
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms) if materialization is not None else None,
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0
            if materialization is not None and materialization.source_sort_key_ms is not None
            else None
        ),
        input_high_water_mark=(
            _iso_from_ms(materialization.input_high_water_mark_ms) if materialization is not None else None
        ),
        input_high_water_mark_source=(
            materialization.input_high_water_mark_source if materialization is not None else None
        ),
        input_row_count=materialization.input_row_count if materialization is not None else 0,
        source_name=source_name,
        title=title,
        first_message_at=evidence.first_message_at,
        last_message_at=evidence.last_message_at,
        canonical_session_date=evidence.canonical_session_date,
        repo_paths=evidence.repo_paths,
        repo_names=inference.repo_names,
        tags=evidence.tags,
        auto_tags=inference.auto_tags,
        message_count=int(row["message_count"] or 0),
        substantive_count=int(row["substantive_count"] or 0),
        attachment_count=int(row["attachment_count"] or 0),
        work_event_count=int(row["work_event_count"] or 0),
        phase_count=int(row["phase_count"] or 0),
        word_count=int(row["word_count"] or 0),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_cost_usd=evidence.total_cost_usd,
        total_duration_ms=evidence.total_duration_ms,
        engaged_duration_ms=inference.engaged_duration_ms,
        tool_active_duration_ms=evidence.tool_active_duration_ms,
        wall_duration_ms=evidence.wall_duration_ms,
        workflow_shape=workflow_shape,
        workflow_shape_confidence=float(row["workflow_shape_confidence"] or 0.0),
        terminal_state=str(row["terminal_state"] or "unknown"),
        terminal_state_method=str(row["terminal_state_method"] or "unknown"),
        terminal_state_confidence=float(row["terminal_state_confidence"] or 0.0),
        cost_is_estimated=bool(row["cost_is_estimated"]),
        thinking_duration_ms=evidence.thinking_duration_ms,
        output_duration_ms=evidence.output_duration_ms,
        tool_duration_ms=evidence.tool_duration_ms,
        tool_calls_per_minute=float(row["tool_calls_per_minute"] or 0.0),
        timing_provenance=evidence.timing_provenance,
        total_input_tokens=evidence.total_input_tokens,
        total_output_tokens=evidence.total_output_tokens,
        total_cache_read_tokens=evidence.total_cache_read_tokens,
        total_cache_write_tokens=evidence.total_cache_write_tokens,
        total_credit_cost=evidence.total_credit_cost,
        cost_provenance=str(row["cost_provenance"] or "unknown"),
        evidence_payload=evidence,
        inference_payload=inference,
        search_text=search_text,
        evidence_search_text=search_text,
        inference_search_text=search_text,
        enrichment_payload=enrichment,
        enrichment_search_text=search_text,
    )


def _session_cost_insight_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    canonical: SessionUsageCost | None = None,
) -> SessionCostInsight:
    session_id = str(row["session_id"])
    source_name = str(row["origin"])
    total_usd = float(
        (canonical.total_usd if canonical and canonical.total_usd is not None else row["cost_usd"]) or 0.0
    )
    cost_provenance = canonical.provenance if canonical is not None else str(row["cost_provenance"] or "")
    try:
        raw_model_name = row["model_name"]
    except (IndexError, KeyError):
        raw_model_name = None
    model_name = str(raw_model_name) if raw_model_name is not None else None
    normalized_model = _normalize_model(model_name) if model_name else None
    status: CostEstimateStatus
    unavailable_reason: CostUnavailableReason | None
    provenance: tuple[str, ...]
    missing_reasons: tuple[str, ...]
    # A materialized profile cost (or a canonical priced/provider/known-zero
    # verdict) outranks a usage-evidence absence: sessions whose cost was
    # priced and stored must not report unavailable just because they carry
    # no per-model usage rows.
    if total_usd > 0 or (
        canonical is not None and canonical.availability in {"priced", "provider_money", "known_zero"}
    ):
        status = "exact" if cost_provenance in {"exact", "origin_reported"} else "priced"
        confidence = 1.0 if status == "exact" else (0.7 if row["cost_is_estimated"] else 0.9)
        basis = (
            CostBasisPayload(provider_reported_usd=total_usd)
            if status == "exact"
            else CostBasisPayload(catalog_priced_usd=total_usd)
        )
        missing_reasons = ()
        unavailable_reason = None
        provenance = ("session_usage_cost", cost_provenance or status)
    elif canonical is not None and canonical.availability == "unpriced":
        status = "unavailable"
        confidence = 0.0
        basis = CostBasisPayload()
        missing_reasons = ("no_price",)
        unavailable_reason = "no_price"
        provenance = ("session_usage_cost", canonical.availability)
    elif canonical is not None and canonical.availability == "no_tokens":
        status = "unavailable"
        confidence = 0.0
        basis = CostBasisPayload()
        missing_reasons = ("no_tokens",)
        unavailable_reason = "no_tokens"
        provenance = ("session_usage_cost", canonical.availability)
    else:
        status = "unavailable"
        confidence = 0.0
        basis = CostBasisPayload()
        missing_reasons = ("archive_profile_no_cost",)
        unavailable_reason = "no_tokens"
        provenance = ("session_usage_cost",)
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    return SessionCostInsight(
        session_id=session_id,
        origin=source_name,
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        estimate=CostEstimatePayload(
            origin=source_name,
            session_id=session_id,
            model_name=model_name,
            normalized_model=normalized_model,
            status=status,
            confidence=confidence,
            total_usd=total_usd,
            basis=basis,
            missing_reasons=missing_reasons,
            unavailable_reason=unavailable_reason,
            provenance=provenance,
        ),
        provenance=_archive_provenance(materialization),
    )


def _canonical_json_text(value: object) -> str:
    return json.dumps(require_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stats_by_sql(group_by: str, where: str, *, tags_relation: str = "session_tags") -> str:
    if group_by in {"provider", "origin"}:
        return f"""
            SELECT s.origin AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, group_key
        """
    if group_by in {"day", "month", "year"}:
        formats = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}
        return f"""
            SELECT strftime('{formats[group_by]}', s.sort_key_ms / 1000, 'unixepoch') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY group_key DESC
        """
    if group_by == "tag":
        return f"""
            SELECT st.tag AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {tags_relation} st ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, group_key
        """
    if group_by == "role":
        return f"""
            SELECT COALESCE(NULLIF(m.role, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "message_type":
        return f"""
            SELECT COALESCE(NULLIF(m.message_type, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "material_origin":
        return f"""
            SELECT COALESCE(NULLIF(m.material_origin, ''), 'unknown') AS group_key,
                   COUNT(*) AS count
            FROM sessions s
            JOIN messages m ON m.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "repo":
        return f"""
            SELECT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_repos sr ON sr.session_id = s.session_id
            JOIN repos r ON r.repo_id = sr.repo_id
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY count DESC, group_key
        """
    if group_by == "tool":
        return f"""
            SELECT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "action":
        return f"""
            SELECT COALESCE(NULLIF(a.semantic_type, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "work-kind":
        return f"""
            SELECT COALESCE(NULLIF(sp.workflow_shape, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_profiles sp ON sp.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    raise ValueError(
        "Unknown group_by "
        f"{group_by!r}; expected one of: provider, origin, day, month, year, tag, role, "
        "message_type, material_origin, repo, tool, action, work-kind"
    )


def _count_scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _ensure_messages_fts_ready(conn: sqlite3.Connection) -> None:
    """Raise ``DatabaseError`` unless message FTS is built and complete.

    Mirrors the archive FTS readiness contract for the split-
    file archive: a missing ``messages_fts`` virtual table means the search index
    was never built, and an FTS row count below the text-bearing block count
    means a bulk write suspended the triggers and never restored them. Both are
    reported as a sanitized ``DatabaseError`` so the reader degrades to a 503
    "Search index" response instead of surfacing a raw ``no such table`` /
    empty-result 200.
    """
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_sync

    check_fts_readiness(message_fts_search_readiness_sync(conn))


def _epoch_ms_from_iso(value: object) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


# Insights whose provenance is tracked in the ``insight_materialization`` ledger
# (materializer version + source high-water mark). Threads use a separate version
# namespace and are intentionally excluded from the version-compatibility check.
_INSIGHT_MATERIALIZATION_TYPE: dict[str, str] = {
    "session_profiles": "session_profile",
    "session_work_events": "work_events",
    "session_phases": "phases",
}

# Insights whose #1278 fallback markers are stored as JSON arrays inside payload
# columns: (table_name, ((column, json_path), ...)). Session profiles carry the
# inference and enrichment fallback reasons under ``$.fallback_reasons`` in their
# respective ``inference_payload_json`` / ``enrichment_payload_json`` columns.
_INSIGHT_FALLBACK_PAYLOAD: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "session_profiles": (
        "session_profiles",
        (
            ("inference_payload_json", "$.fallback_reasons"),
            ("enrichment_payload_json", "$.fallback_reasons"),
        ),
    ),
}


def _archive_insight_readiness_verdict(
    *,
    table_present: bool,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    ready_flags: dict[str, bool],
    total_sessions: int,
) -> InsightReadinessVerdict:
    if not table_present:
        return "missing"
    if incompatible_count:
        return "incompatible"
    if stale_count or orphan_count:
        return "stale"
    if missing_count or (expected_row_count is not None and row_count < expected_row_count):
        return "partial"
    if row_count == 0:
        # An empty archive (no sessions at all) reports every surface as empty.
        # In a populated archive a surface with 0 expected rows is vacuously
        # ready (e.g. no tags to roll up); a surface that should hold rows was
        # already caught by the partial branch above.
        if total_sessions > 0 and expected_row_count == 0:
            return "ready"
        return "empty"
    if degraded_count:
        return "degraded"
    if ready_flags and all(ready_flags.values()):
        return "ready"
    if not ready_flags:
        return "ready"
    return "unknown"


def _insight_readiness_aggregate_verdict(entries: tuple[InsightReadinessEntry, ...]) -> InsightReadinessVerdict:
    verdicts = {entry.verdict for entry in entries}
    for verdict in ("incompatible", "stale", "partial", "missing", "degraded", "unknown", "empty"):
        if verdict in verdicts:
            return verdict
    return "ready"


def _archive_insight_readiness_evidence(
    *,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    fallback_reason_counts: dict[str, int],
    ready_flags: dict[str, bool],
) -> tuple[str, ...]:
    values = [f"rows={row_count}"]
    if expected_row_count is not None:
        values.append(f"expected={expected_row_count}")
    if missing_count:
        values.append(f"missing={missing_count}")
    if stale_count:
        values.append(f"stale={stale_count}")
    if orphan_count:
        values.append(f"orphan={orphan_count}")
    if incompatible_count:
        values.append(f"incompatible={incompatible_count}")
    if degraded_count:
        values.append(f"degraded={degraded_count}")
    values.extend(f"fallback_reason={reason}={count}" for reason, count in fallback_reason_counts.items())
    values.extend(f"{key}={value}" for key, value in sorted(ready_flags.items()))
    return tuple(values)


def _archive_debt(
    *,
    name: str,
    category: str,
    issue_count: int,
    detail: str,
    destructive: bool = False,
) -> ArchiveDebtInsight:
    return ArchiveDebtInsight(
        debt_name=name,
        category=category,
        maintenance_target=name,
        destructive=destructive,
        issue_count=issue_count,
        healthy=issue_count == 0,
        detail=detail,
    )


def _archive_messages_fts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    text_blocks = _count_scalar(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
    fts_rows = _count_scalar(conn, "SELECT COUNT(*) FROM messages_fts")
    issue_count = abs(text_blocks - fts_rows)
    detail = "archive message FTS synchronized" if issue_count == 0 else f"{issue_count:,} message FTS row mismatch"
    return _archive_debt(
        name="archive_messages_fts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_rows_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
        )
        """,
    )
    orphaned = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE NOT EXISTS (
            SELECT 1 FROM sessions AS s WHERE s.session_id = p.session_id
        )
        """,
    )
    issue_count = missing + orphaned
    detail = (
        "archive session profile rows complete"
        if issue_count == 0
        else f"{missing:,} missing and {orphaned:,} orphaned archive session profile rows"
    )
    return _archive_debt(
        name="archive_session_profile_rows",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_counts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    work_event_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.work_event_count != (
            SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
        )
        """,
    )
    phase_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.phase_count != (
            SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
        )
        """,
    )
    issue_count = work_event_mismatch + phase_mismatch
    detail = (
        "archive profile derived counts match timeline rows"
        if issue_count == 0
        else f"{work_event_mismatch:,} work-event and {phase_mismatch:,} phase count mismatches"
    )
    return _archive_debt(
        name="archive_profile_counts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_materialization_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _archive_missing_materialization_counts(conn)
    issue_count = sum(missing.values())
    detail = (
        "archive insight materialization rows complete"
        if issue_count == 0
        else "missing archive materialization rows: "
        + ", ".join(f"{key}={value}" for key, value in sorted(missing.items()) if value)
    )
    return _archive_debt(
        name="archive_insight_materialization",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_source_raw_link_debt(index_db_path: Path, source_db_path: Path) -> ArchiveDebtInsight:
    # Cross-tier debt reads use a dedicated connection.  The long-lived
    # ArchiveStore connection may already own a transaction; attaching and
    # detaching on that connection can fail with "database ... is locked".
    # Closing this short-lived connection releases the attached schema and all
    # statement cursors as one lifetime, so no DETACH race is possible.
    with closing(open_readonly_connection(index_db_path)) as conn:
        raw_links = _count_scalar(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
        if not source_db_path.exists():
            issue_count = raw_links
            detail = (
                "archive sessions have no source raw links"
                if raw_links == 0
                else f"source.db missing while {raw_links:,} sessions carry raw_id links"
            )
            return _archive_debt(
                name="archive_source_raw_links",
                category="source_ingest",
                issue_count=issue_count,
                detail=detail,
            )
        conn.execute("ATTACH DATABASE ? AS source_debt", (f"file:{source_db_path}?mode=ro",))
        missing = _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE s.raw_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM source_debt.raw_sessions AS r WHERE r.raw_id = s.raw_id
              )
            """,
        )
    detail = "archive source raw links resolve" if missing == 0 else f"{missing:,} sessions reference missing raw rows"
    return _archive_debt(
        name="archive_source_raw_links",
        category="source_ingest",
        issue_count=missing,
        detail=detail,
    )


def _archive_user_overlay_debt(index_db_path: Path, user_db_path: Path) -> ArchiveDebtInsight:
    if not user_db_path.exists():
        return _archive_debt(
            name="archive_user_overlay_orphans",
            category="archive_cleanup",
            issue_count=0,
            detail="archive user tier absent; no overlay orphan check needed",
        )
    # See _archive_source_raw_link_debt: a dedicated short-lived connection
    # avoids ATTACH/DETACH racing the long-lived ArchiveStore connection's
    # own transaction.
    with closing(open_readonly_connection(index_db_path)) as conn:
        conn.execute("ATTACH DATABASE ? AS user_debt", (f"file:{user_db_path}?mode=ro",))
        checks = (
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.target_ref LIKE 'session:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = substr(u.target_ref, 9))",
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.kind IN ('mark', 'annotation', 'note', 'suppression') "
            "AND u.target_ref LIKE 'message:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM sessions s "
            "  WHERE substr(u.target_ref, 9) = s.session_id "
            "     OR (substr(substr(u.target_ref, 9), 1, length(s.session_id)) = s.session_id "
            "         AND substr(substr(u.target_ref, 9), length(s.session_id) + 1, 1) = ':')"
            ")",
            "SELECT COUNT(*) FROM user_debt.assertions u "
            "WHERE u.kind = 'correction' "
            "AND u.target_ref LIKE 'insight:%' "
            "AND COALESCE(u.status, '') != 'deleted' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = substr(u.target_ref, 9))",
        )
        issue_count = sum(_count_scalar(conn, sql) for sql in checks)
    detail = (
        "archive user overlays resolve to index sessions"
        if issue_count == 0
        else f"{issue_count:,} archive user overlay rows reference missing sessions"
    )
    return _archive_debt(
        name="archive_user_overlay_orphans",
        category="archive_cleanup",
        issue_count=issue_count,
        detail=detail,
    )


def _session_latency_profile_from_archive_row(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> SessionLatencyProfileInsight:
    session_id = str(row["session_id"])
    response_rows = conn.execute(
        """
        SELECT role, occurred_at_ms
        FROM messages
        WHERE session_id = ?
          AND occurred_at_ms IS NOT NULL
          AND role IN ('user', 'assistant')
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    agent_response_ms: list[int] = []
    user_response_ms: list[int] = []
    previous_role: str | None = None
    previous_at: int | None = None
    for message in response_rows:
        role = str(message["role"])
        occurred_at = int(message["occurred_at_ms"])
        if previous_role is not None and previous_at is not None:
            delta_ms = max(occurred_at - previous_at, 0)
            if previous_role == "user" and role == "assistant":
                agent_response_ms.append(delta_ms)
            elif previous_role == "assistant" and role == "user" and delta_ms <= 1_800_000:
                user_response_ms.append(delta_ms)
        previous_role = role
        previous_at = occurred_at
    tool_counts = _latency_tool_category_counts(conn, session_id)
    materialization = _read_archive_materialization(conn, "latency", session_id)
    return SessionLatencyProfileInsight(
        session_id=session_id,
        origin=str(row["origin"]),
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        latency=SessionLatencyProfilePayload(
            median_tool_call_ms=0,
            p90_tool_call_ms=0,
            max_tool_call_ms=0,
            stuck_tool_count=0,
            median_agent_response_ms=_median_ms(agent_response_ms),
            median_user_response_ms=_median_ms(user_response_ms),
            tool_call_count_by_category=tool_counts,
        ),
    )


def _latency_tool_category_counts(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(semantic_type, ''), 'unknown') AS category, COUNT(*) AS count
        FROM actions
        WHERE session_id = ?
        GROUP BY category
        ORDER BY count DESC, category
        """,
        (session_id,),
    ).fetchall()
    return {str(row["category"]): int(row["count"] or 0) for row in rows}


def _median_ms(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return int((ordered[middle - 1] + ordered[middle]) / 2)


def _archive_missing_materialization_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        insight_type: _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM insight_materialization AS m
                WHERE m.insight_type = ? AND m.session_id = s.session_id
            )
            """,
            (insight_type,),
        )
        for insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
    }


def _dominant_repo(rows: list[sqlite3.Row]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        repo = row["git_repository_url"]
        if not isinstance(repo, str) or not repo:
            continue
        counts[repo] = counts.get(repo, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _thread_member_depth(rows: list[sqlite3.Row], session_id: str) -> int:
    parents = {
        str(row["session_id"]): str(row["parent_session_id"]) for row in rows if row["parent_session_id"] is not None
    }
    depth = 0
    current = session_id
    seen: set[str] = set()
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
        depth += 1
    return depth


def _archive_thread_member_role(row: sqlite3.Row, thread_id: str) -> str:
    if str(row["session_id"]) == thread_id:
        return "root"
    if row["parent_session_id"] is not None:
        return "parent_continuation"
    return "member"


def _archive_thread_member_support_signals(row: sqlite3.Row) -> tuple[str, ...]:
    signals = ["archive_thread_sessions"]
    if row["parent_session_id"] is not None:
        signals.append("parent_session_id")
    return tuple(signals)


def _archive_thread_member_evidence(row: sqlite3.Row, thread_id: str, position: int) -> tuple[str, ...]:
    evidence = [f"position={position}"]
    if row["parent_session_id"] is not None:
        evidence.append(f"parent_id={row['parent_session_id']}")
        evidence.append(f"root_id={thread_id}")
    return tuple(evidence)


def _profile_or_session_timestamp_ms(row: sqlite3.Row, *, profile_column: str, session_column: str) -> int | None:
    profile_timestamp = row[profile_column]
    if isinstance(profile_timestamp, str) and profile_timestamp.strip():
        parsed = _epoch_ms_from_iso(profile_timestamp)
        if parsed is not None:
            return parsed
    session_timestamp = row[session_column]
    return int(session_timestamp) if isinstance(session_timestamp, int) else None


def _tag_origin_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        tag_params,
    ).fetchall()
    return {str(row["origin"]): int(row["count"] or 0) for row in rows}


def _tag_repo_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.git_repository_url AS repo, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
          AND s.git_repository_url IS NOT NULL
          AND s.git_repository_url != ''
        GROUP BY s.git_repository_url
        ORDER BY count DESC, s.git_repository_url
        """,
        tag_params,
    ).fetchall()
    return {str(row["repo"]): int(row["count"] or 0) for row in rows}


def _with_exact_tag_filter(clause: str, params: tuple[object, ...], tag: str) -> tuple[str, tuple[object, ...]]:
    if clause:
        return f"{clause} AND st.tag = ?", (*params, tag)
    return "WHERE st.tag = ?", (tag,)


def _iso_from_ms(value: object) -> str | None:
    if not isinstance(value, int):
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _month_bucket_end_ms(bucket: str) -> int:
    year_text, month_text = bucket.split("-", 1)
    year = int(year_text)
    month = int(month_text)
    end = datetime(year + 1, 1, 1, tzinfo=UTC) if month == 12 else datetime(year, month + 1, 1, tzinfo=UTC)
    return int(end.timestamp() * 1000)


__all__ = [
    "ActiveByteRevisionChainError",
    "ArchiveActionQueryRow",
    "ArchiveAssertionQueryRow",
    "ArchiveBlockQueryRow",
    "ArchiveContextSnapshotQueryRow",
    "ArchiveDelegationAncestryRow",
    "ArchiveDelegationCard",
    "ArchiveDelegationQueryRow",
    "ArchiveDelegationSubtreeRow",
    "ArchiveFileQueryRow",
    "ArchiveMessageQueryRow",
    "ArchiveObservedEventQueryRow",
    "ArchiveQueryUnitAggregateRow",
    "ArchiveQueryUnitMultiAggregatePage",
    "ArchiveRawParsedWriteResult",
    "ArchiveRunQueryRow",
    "ArchiveSessionSearchHit",
    "ArchiveSessionSummary",
    "ArchiveStore",
    "MembershipReplayConflictError",
]
