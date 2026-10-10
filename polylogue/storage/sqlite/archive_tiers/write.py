"""Minimal archive index parsed-session writer/read helpers.

Writer module: index.

Session tag CRUD (the former ``user``-tier twin-write
contract here) moved to ``session_annotations_write.py`` — see that
module's docstring for the current writer-module declaration.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from builtins import BaseExceptionGroup
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from datetime import time as datetime_time
from decimal import Decimal
from enum import Enum
from itertools import chain, islice, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol, cast, overload
from urllib.parse import urlparse

import ijson

from polylogue.archive.attachment.availability import AttachmentAvailability, resolve_attachment_availability
from polylogue.archive.message.types import MessageType
from polylogue.archive.revision_authority import RawRevisionKind, is_work_event_raw_id
from polylogue.archive.semantic.pricing import catalog_cost_for_tokens
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.repo_identity import normalize_repo_name, normalize_repo_path
from polylogue.archive.topology.edge import (
    HOOK_AUTHORITATIVE_LINK_METHOD,
    HOOK_CONTRADICTED_LINK_METHOD,
    HOOK_DERIVED_LINK_METHODS,
    HOOK_SUPERSEDED_LINK_METHOD,
    INVALIDATED_PREFIX_EVIDENCE_KEY,
    DispatchResolutionReason,
    TopologyEdgeStatus,
    TopologyEdgeType,
    branch_type_to_edge_type,
    invalidated_prefix_sql,
    topology_status_composes_sql,
)
from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import (
    BlockType,
    LinkType,
    Origin,
    PasteBoundary,
    Provider,
    SessionKind,
    ToolOutcome,
    admitted_session_kind,
)
from polylogue.core.hook_payload import payload_key_spellings
from polylogue.core.identity_law import attachment_content_id, attachment_native_identity, attachment_reference_id
from polylogue.core.identity_law import block_id as archive_block_id
from polylogue.core.identity_law import message_id as archive_message_id
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.message_native_identity import (
    message_native_key,
    native_id_from_storage,
    normalized_message_native_id,
    source_native_id_from_json,
    source_native_id_json,
    sqlite_message_native_id,
)
from polylogue.core.message_owner import MessageOwnerAmbiguityError
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.core.sql_settlement import current_native_sql_lifetimes, retain_native_sql_lifetimes
from polylogue.core.timestamp_authority import producer_timestamp_flags, session_evidence_timestamps
from polylogue.core.timestamps import parse_timestamp, to_epoch_ms
from polylogue.core.types import AttachmentDirection, LineageInheritance, require_literal
from polylogue.logging import WARNING, emit, get_logger
from polylogue.pipeline.ids import (
    SIDECAR_BLOB_EVENT_TYPES,
    SIDECAR_LOCATOR_KEYS,
    MessageContentIdentity,
    MessageOwnerResolution,
    attachment_message_owner_key,
    block_content_identities,
    bound_session_content_hash,
    disk_message_content_identities,
    disk_message_owner_resolution,
    event_message_owner_key,
    message_content_identities,
    message_content_identity,
    message_owner_resolution,
    message_semantic_content_address,
)
from polylogue.sources.assembly import SidecarData
from polylogue.sources.origin_specs import lowering_fingerprint, origin_specs, parser_fingerprint_for_origin
from polylogue.sources.parsers.base import (
    ParseAccounting,
    ParsedAttachment,
    ParsedContentBlock,
    ParsedDispatchObservation,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.parsers.base_support import derive_attachment_provenance
from polylogue.sources.parsers.claude.orchestration import (
    parse_claude_orchestration_artifact,
)
from polylogue.sources.parsers.hermes_identity import split_qualified_session_id
from polylogue.sources.prepared_message_sink import SqliteMessageSink, _prepared_reader, normalize_active_branch
from polylogue.sources.sidecar_evidence import SidecarResolver
from polylogue.sources.tool_outcomes import derive_tool_outcomes as _derive_tool_outcomes
from polylogue.storage.archive_identity import archive_root_for_index_path
from polylogue.storage.attachment_reasons import AttachmentOwnerResolutionReason
from polylogue.storage.blob_store import BlobStore, blob_store_for_connection
from polylogue.storage.derived.session.summary import SESSION_SUMMARY_MEASURES, refresh_session_summary
from polylogue.storage.fts.fts_lifecycle import message_fts_triggers_present_sync
from polylogue.storage.fts.pl_fold import pl_fold_sql_expr
from polylogue.storage.fts.sql import (
    FTS_BULK_SESSION_WRITE_GUARD,
    delete_session_identity_rows_sql,
    delete_session_rows_sql,
    insert_session_identity_rows_sql,
    insert_session_rows_sql,
)
from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor
from polylogue.storage.runtime import (
    LINEAGE_TRUNCATION_CYCLE,
    LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT,
    LineageTruncationReason,
)
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.action_pairs import refresh_action_pairs
from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
from polylogue.storage.sqlite.archive_tiers.common import require_vocabulary
from polylogue.storage.sqlite.archive_tiers.ingest_precedence import should_skip_stale_replace
from polylogue.storage.sqlite.archive_tiers.session_annotations_write import (
    ArchiveSessionTag,
    read_session_tags,
    upsert_session_tag,
)
from polylogue.storage.sqlite.archive_tiers.session_suppression import (
    record_suppression_refusal,
    session_write_is_suppressed,
)
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    open_scratch_connection,
    retained_native_sql_owners_for_lifetime,
)
from polylogue.storage.sqlite.delegation_facts import refresh_delegation_facts_for_sessions
from polylogue.storage.sqlite.reference_seal import (
    IndexMutationScope,
    PreparedIndexMutation,
    ReferenceSealError,
    current_index_mutation_scope,
    index_path_for_connection,
    note_current_deleted_message_ids,
    note_current_deleted_session,
    note_current_lineage_change,
    note_current_session_namespace_change,
)
from polylogue.storage.sqlite.session_shard import (
    SessionShard,
    SessionShardBuilder,
    build_session_shard,
    open_session_shard,
    shard_owner_reader_window,
)
from polylogue.storage.usage import (
    UsageProjectionModel,
    project_provider_usage_events,
    provider_usage_event_identity,
    provider_usage_request_events,
    provider_usage_required_lane_indices,
)


@dataclass(frozen=True, slots=True)
class CatalogCost:
    value: float


@dataclass(frozen=True, slots=True)
class ProviderCost:
    value: float


def _write_provider_cost(
    conn: sqlite3.Connection, session_id: str, model_names: Sequence[str], cost: ProviderCost
) -> None:
    """Pass a provider dollar total through to a session's model rows.

    The provider prices the session, not the model, so the one exact total is
    split across the rows the session declares in proportion to the evidence
    that is genuinely per-model: catalog dollars, else billable tokens, else an
    equal share. Every declared row therefore carries a share and every other
    row for the session is cleared, so ``SUM(provider_cost_usd)`` over the
    session is the reported total exactly once and no row of a
    provider-priced session falls through to the catalog fallback in
    ``COALESCE(SUM(provider), SUM(catalog))``. A merge-append that switches
    models carries the total to the incoming rows.
    """
    if not isinstance(cost, ProviderCost):
        raise TypeError("provider cost writes require ProviderCost")
    names = list(dict.fromkeys(model_names))
    if not names:
        return
    placeholders = ", ".join("?" for _ in names)
    weight_rows = conn.execute(
        f"""SELECT model_name,
                   COALESCE(catalog_cost_usd, 0.0) AS catalog,
                   COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)
                   + COALESCE(cache_read_tokens, 0) + COALESCE(cache_write_tokens, 0) AS tokens
            FROM session_model_usage
            WHERE session_id = ? AND model_name IN ({placeholders})""",
        (session_id, *names),
    ).fetchall()
    catalog = {str(row[0]): float(row[1] or 0.0) for row in weight_rows}
    tokens = {str(row[0]): float(row[2] or 0) for row in weight_rows}
    weights = [catalog.get(name, 0.0) for name in names]
    if sum(weights) <= 0.0:
        weights = [tokens.get(name, 0.0) for name in names]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(names)
    total_weight = sum(weights)
    # The residue lands on the final row so the shares re-sum to the exact
    # reported total rather than a rounded one.
    shares = [cost.value * weight / total_weight for weight in weights[:-1]]
    shares.append(cost.value - sum(shares))
    for name, share in zip(names, shares, strict=True):
        conn.execute(
            """UPDATE session_model_usage SET provider_cost_usd = ?
               WHERE session_id = ? AND model_name = ?""",
            (share, session_id, name),
        )
    conn.execute(
        f"""UPDATE session_model_usage SET provider_cost_usd = NULL
            WHERE session_id = ? AND model_name NOT IN ({placeholders})""",
        (session_id, *names),
    )


logger = get_logger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_ACOMPACT_PARENT_MEMBERSHIP_THRESHOLD = 0.90


@dataclass(frozen=True, slots=True)
class ArchiveBlockRow:
    block_id: str
    message_id: str
    block_type: str
    text: str | None
    content_identity: str
    content_occurrence: int
    content_hash: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    semantic_type: str | None = None
    tool_input: str | None = None
    metadata: str | None = None
    media_type: str | None = None
    signature: str | None = None
    language: str | None = None
    name: str | None = None
    # Legacy structural fields retained for compatibility. tool_outcome is the
    # canonical closed outcome vocabulary for admitted tool blocks.
    tool_result_is_error: int | None = None
    tool_result_exit_code: int | None = None
    tool_outcome: ToolOutcome | None = None
    # Why the keystone outcome above is unresolved (schema v46). NULL when the
    # outcome IS known, never "unknown of unknown" (polylogue-cuxz.8).
    tool_result_outcome_unknown_reason: str | None = None


# The compact block read model's own fields name its projection; BLOCKS_SPEC
# decides which of them are stored columns, so a renamed or dropped column
# cannot leave a SELECT silently stale. ``metadata`` has no blocks column and
# is therefore never projected.
ARCHIVE_BLOCK_ROW_COLUMNS: tuple[str, ...] = tuple(
    column.name
    for column in archive_tiers_specs.BLOCKS_SPEC.all_columns
    if column.name in {field.name for field in fields(ArchiveBlockRow)}
)


def archive_block_row_select_sql(columns: tuple[str, ...] = ARCHIVE_BLOCK_ROW_COLUMNS) -> str:
    """Render the block projection ``archive_block_row`` consumes."""
    return ", ".join(columns)


def archive_block_row(row: sqlite3.Row, columns: tuple[str, ...] = ARCHIVE_BLOCK_ROW_COLUMNS) -> ArchiveBlockRow:
    """Hydrate the compact block read model from a row selected over ``columns``."""
    values: dict[str, Any] = {name: row[name] for name in columns}
    outcome = values.get("tool_outcome")
    if outcome is not None:
        values["tool_outcome"] = ToolOutcome(cast(str, outcome))
    return ArchiveBlockRow(**values)


@dataclass(frozen=True, slots=True)
class ArchiveAttachmentRow:
    attachment_id: str
    message_id: str | None
    display_name: str | None = None
    media_type: str | None = None
    byte_count: int = 0
    upload_origin: str | None = None
    direction: str | None = None
    producer_ref: str | None = None
    source_url: str | None = None
    caption: str | None = None
    blob_hash: bytes | None = None
    acquisition_status: str | None = None
    generation_id: str | None = None
    availability: AttachmentAvailability | None = None


def _attachment_availability(
    blob_store: BlobStore | None,
    blob_hash: bytes | None,
    acquisition_status: str | None,
    generation_id: str | None = None,
) -> AttachmentAvailability | None:
    """Resolve an attachment's bytes against the blob store of the archive being read.

    The store is the read archive's own CAS, never the process-configured one:
    an ``ArchiveStore`` opened on another root must not report its bytes
    missing, or borrow another archive's copy of the same hash. A reader that
    supplies no store does not resolve availability at all (``None``).
    """
    if blob_store is None:
        return None
    return resolve_attachment_availability(
        blob_hash=blob_hash,
        acquisition_status=acquisition_status,
        verify=blob_store.verify_for_read,
        exists=blob_store.exists,
        generation_id=generation_id,
    )


@dataclass(frozen=True, slots=True)
class ArchiveMessageRow:
    message_id: str
    native_id: str | None
    role: str
    position: int
    variant_index: int
    is_active_path: bool
    is_active_leaf: bool
    blocks: tuple[ArchiveBlockRow, ...]
    identity_source: str = "content"
    message_type: str = "message"
    material_origin: str = "unknown"
    word_count: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    paste_boundary_state: str | None = None
    occurred_at: str | None = None
    duration_ms: int = 0
    parent_message_id: str | None = None
    attachments: tuple[ArchiveAttachmentRow, ...] = ()
    # Exact composition provenance for semantic transcript rendering. Parent
    # rows retain their original source session when a prefix-sharing child is
    # composed, so inherited evidence is not guessed from position.
    source_session_id: str | None = None
    # Provider-reported terminal signal for this turn (schema v46, Anthropic
    # ``message.stop_reason``). None means unreported/not-applicable, never a
    # guessed happy-path default (polylogue-cuxz.8).
    stop_reason: str | None = None
    # Per-message usage as stored. ``None`` is "the provider reported no
    # counter at this grain", distinct from a measured zero.
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


# The compact envelope projection is a subset of the canonical messages row.
# Its order follows ``MESSAGES_SPEC``; the only read-model alias is the
# historical ``paste_boundary_state`` name.
ARCHIVE_MESSAGE_ROW_COLUMNS: tuple[str, ...] = tuple(
    column.name
    for column in archive_tiers_specs.MESSAGES_SPEC.all_columns
    if column.name in {field.name for field in fields(ArchiveMessageRow)}
    or column.name in {"paste_boundary", "occurred_at_ms"}
)


def archive_message_row_select_sql() -> str:
    """Render the compact message projection from the canonical declaration."""
    return ", ".join(
        "paste_boundary AS paste_boundary_state" if name == "paste_boundary" else name
        for name in ARCHIVE_MESSAGE_ROW_COLUMNS
    )


@dataclass(frozen=True, slots=True)
class ArchiveAgentPolicy:
    policy_id: str
    session_id: str
    position: int
    approval_policy: str | None
    sandbox_policy: str | None
    network_policy: str | None
    observed_at_ms: int | None
    source_message_id: str | None


@dataclass(frozen=True, slots=True)
class ArchiveSessionEnvelope:
    session_id: str
    native_id: str
    origin: str
    title: str | None
    active_leaf_message_id: str | None
    messages: tuple[ArchiveMessageRow, ...]
    session_kind: str = SessionKind.STANDARD.value
    parent_session_id: str | None = None
    root_session_id: str | None = None
    branch_type: str | None = None
    title_source: str | None = None
    title_ref: str | None = None
    # See ``ArchiveSessionSummary.display_name`` (polylogue-cgfy): the full-read
    # envelope carries it too, so a detail read and a summary read cannot
    # disagree about a session's provider-assigned name.
    display_name: str | None = None
    instructions_text: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    # polylogue-gt1z (v49): exact provider-reported session cost total.
    reported_cost_usd: float | None = None
    orphan_attachments: tuple[ArchiveAttachmentRow, ...] = ()
    # 4ts.6: whether ``messages`` is the FULL logical transcript, or a
    # silently truncated one -- a prefix-sharing child composition can drop
    # ancestors past a recursion depth limit, or return only its own
    # divergent tail when the parent's branch point was hard-deleted.
    lineage_complete: bool = True
    lineage_truncation_reason: LineageTruncationReason | None = None
    # Bounded composition facts established during this exact archive read.
    # ``none`` is explicit rather than NULL so callers can distinguish a root
    # or ordinary child from an unavailable composition signal.
    lineage_inheritance: str = "none"
    lineage_branch_point_message_id: str | None = None
    # The TRUE composed transcript length. ``None`` (the default) means
    # ``messages`` already holds every composed message, i.e. an ordinary
    # unbounded read via ``read_archive_session_envelope``. A bounded page
    # read (``read_archive_session_page``) sets this to the full count while
    # ``messages`` holds only the requested window, so callers can page
    # without materializing the whole transcript.
    total_message_count: int | None = None


ARCHIVE_SESSION_ENVELOPE_COLUMNS: tuple[str, ...] = tuple(
    column.name
    for column in archive_tiers_specs.SESSIONS_SPEC.all_columns
    if column.name in {field.name for field in fields(ArchiveSessionEnvelope)}
    or column.name in {"created_at_ms", "updated_at_ms"}
)


def archive_session_envelope_select_sql() -> str:
    """Render the compact session projection in canonical declaration order."""
    return ", ".join(ARCHIVE_SESSION_ENVELOPE_COLUMNS)


def archive_message_display_text(blocks: Iterable[ArchiveBlockRow]) -> str:
    """Flatten an archive-tier message's blocks into its display ``text``.

    Single source of truth for the ``message.text`` field the daemon's two
    session-detail routes each used to compute independently (polylogue-6o9b):
    the DB-backed route (``archive/hydration.py:archive_message_to_domain``,
    reached via ``Polylogue.get_session()``) and the archive-backed route
    (``daemon/http.py:_archive_message_payload``) both read the same
    ``ArchiveMessageRow.blocks`` and must produce byte-identical text for
    the same message, since the browser reader's client-side
    rendering heuristic (``renderMessageBlocks``) dispatches off this single
    flattened field. Joins every non-empty block's text in block order,
    blank-line separated -- this intentionally includes
    THINKING/TOOL_USE/TOOL_RESULT/CODE block text, not just prose TEXT
    blocks, matching both routes' prior (independently duplicated) behavior.
    Narrowing this to prose-only content is a separate follow-up (see
    ``investigations/rendering-path-divergence.md``), not this fix's scope.
    """
    return "\n\n".join(block.text for block in blocks if block.text)


@dataclass(frozen=True, slots=True)
class SessionEventWriteResult:
    wrote_provider_usage_events: bool = False


@dataclass(frozen=True, slots=True)
class ArchiveWriteOutcome:
    """Truthful result of the archive writer's precedence decision."""

    session_id: str
    wrote: bool
    stale_skipped: bool = False
    #: The operator tombstoned this session (durable ``user.db`` suppression)
    #: and the write was refused rather than resurrecting it. Counted and
    #: logged by ``session_suppression``; never a silent drop.
    suppression_skipped: bool = False
    # An attachment whose parser evidence does not identify one safe message
    # owner is deliberately not given a guessed attachment_refs edge. Keep
    # that decision in the production write receipt so acquisition/replay can
    # account for every parsed attachment without treating it as silent loss.
    unresolved_attachment_owners: tuple[tuple[str, AttachmentOwnerResolutionReason], ...] = ()


class PreparedSessionWriteRefusedError(RuntimeError):
    """A pinned prepared write no longer describes the admitted archive state."""


class PreparedSessionSettlementError(RuntimeError):
    """All carrier closes were attempted; their original owners can retry."""

    code = "native_sql_unsettled"
    retryable = True

    def __init__(self, failures: Sequence[BaseException]) -> None:
        super().__init__("prepared session artifact cleanup remains unsettled")
        self.failures = tuple(failures)


class LineageSignatureCache:
    """Bounded, batch-local cache for canonical lineage signatures.

    A lineage child is parsed by the provider parser before this writer sees
    it, but the writer still has to reconstruct the parent's composed prefix
    and hash every message before it can decide where the child diverges. The
    old cache memoized only each session's *own* signatures in an unbounded
    ``dict``; composed signatures were rebuilt for every sibling. This cache
    carries both layers and evicts by weighted recency, so a large parent
    cannot turn a batch into another unbounded resident corpus.

    Values retain canonical ``(message_id, content_signature)`` pairs.
    ``message_id`` is deliberately kept alongside the signature: it is the
    archive's identity/provenance witness for the eventual branch point, not a
    cache-invented positional id. The cache is only a hint. ``enabled=False``
    is the explicit anti-vacuity/control path and has exactly the same output
    contract as a miss on every lookup.

    The cache lives for one ordered ingest drain. ``pop`` removes the rewritten
    session and any composed entries that depended on it, retaining unrelated
    parent work while preventing stale branch identities after replacement.
    Each composed entry records its whole ancestor closure, not just the
    sessions its own walk visited, because a walk that stopped at a cached
    ancestor never saw that ancestor's parents and the ancestor's entry can be
    evicted before one of them is rewritten.
    """

    _ENTRY_OVERHEAD_BYTES = 96

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024, enabled: bool = True) -> None:
        if max_bytes < 0:
            raise ValueError("lineage signature cache max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self.enabled = enabled
        self._entries: OrderedDict[tuple[str, str], tuple[list[tuple[str, str]], int]] = OrderedDict()
        self._dependencies: dict[str, frozenset[str]] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _weight(
        session_id: str, signatures: list[tuple[str, str]], *, dependencies: frozenset[str] = frozenset()
    ) -> int:
        return (
            LineageSignatureCache._ENTRY_OVERHEAD_BYTES
            + len(session_id)
            + sum(len(message_id) + len(signature) + 16 for message_id, signature in signatures)
            # Ancestor closures are retained evidence too. Charge their set
            # slots and strings against the same budget.
            + (216 + sum(96 + 4 * len(dependency) for dependency in dependencies) if dependencies else 0)
        )

    def _get(self, kind: str, session_id: str) -> list[tuple[str, str]] | None:
        if not self.enabled:
            self.misses += 1
            return None
        key = (kind, session_id)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def _put(
        self,
        kind: str,
        session_id: str,
        signatures: list[tuple[str, str]],
        *,
        dependencies: frozenset[str] = frozenset(),
    ) -> None:
        if not self.enabled or self.max_bytes == 0:
            return
        key = (kind, session_id)
        weight = self._weight(session_id, signatures, dependencies=dependencies)
        if weight > self.max_bytes:
            # A whale must not evict the whole useful cache just to remain a
            # one-entry cache. It is a normal miss on the next descendant.
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if kind == "composed":
                self._dependencies.pop(session_id, None)
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        self._entries[key] = (signatures, weight)
        self._bytes += weight
        if kind == "composed":
            self._dependencies[session_id] = dependencies
        while self._bytes > self.max_bytes and self._entries:
            oldest_key, (_value, oldest_weight) = self._entries.popitem(last=False)
            self._bytes -= oldest_weight
            if oldest_key[0] == "composed":
                self._dependencies.pop(oldest_key[1], None)
            self.evictions += 1

    def get(self, session_id: str, default: list[tuple[str, str]] | None = None) -> list[tuple[str, str]] | None:
        """Mapping-compatible lookup for an own-signature entry."""
        value = self._get("own", session_id)
        return default if value is None else value

    def __setitem__(self, session_id: str, signatures: list[tuple[str, str]]) -> None:
        self._put("own", session_id, signatures)

    def get_composed(self, session_id: str) -> list[tuple[str, str]] | None:
        return self._get("composed", session_id)

    def composed_dependencies(self, session_id: str) -> frozenset[str] | None:
        """The closure a resident composed entry depends on, or ``None`` when absent."""
        if ("composed", session_id) not in self._entries:
            return None
        return self._dependencies.get(session_id)

    def set_composed(
        self,
        session_id: str,
        signatures: list[tuple[str, str]],
        *,
        dependencies: frozenset[str],
    ) -> None:
        self._put("composed", session_id, signatures, dependencies=dependencies)

    def pop(self, session_id: str, default: object = None) -> object:
        """Invalidate one session and every composed descendant depending on it."""
        if not self.enabled:
            return default
        # Every composed entry carries its full ancestor closure, so one pass
        # finds each dependent without relying on intermediate entries that
        # may already have been evicted.
        impacted = {session_id}
        impacted.update(
            composed_id for composed_id, dependencies in self._dependencies.items() if session_id in dependencies
        )
        removed: object = default
        for key in tuple(self._entries):
            if key[1] not in impacted:
                continue
            value, weight = self._entries.pop(key)
            self._bytes -= weight
            if key[0] == "composed":
                self._dependencies.pop(key[1], None)
            if key == ("own", session_id):
                removed = value
        return removed

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        return self._bytes


_SignatureCacheLike = dict[str, list[tuple[str, str]]] | LineageSignatureCache


def _retain_stale_session_observations(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    fallback_timestamp: str | None = None,
) -> None:
    """Retain monotonic observations from a stale snapshot without replacing rows.

    Freshness/content governance may reject the snapshot, but its earlier
    creation evidence and repository observations are still legitimate evidence.
    Keeping this at the low-level writer boundary prevents direct API and
    revision-governance callers from silently losing observations retained by
    batch ingest.
    """
    scope = current_index_mutation_scope()
    if scope is None:
        raise RuntimeError("stale observations require the caller's Index mutation scope")
    scope.require_new_work(conn)
    scope.note_lineage_change(session_id)
    candidate_created_at_ms, _candidate_updated_at_ms = session_evidence_timestamps(
        session,
        fallback_timestamp=fallback_timestamp,
    )
    if candidate_created_at_ms is not None:
        conn.execute(
            """
            UPDATE sessions
            SET created_at_ms = CASE
                WHEN created_at_ms IS NULL THEN ?
                ELSE MIN(created_at_ms, ?)
            END
            WHERE session_id = ?
            """,
            (candidate_created_at_ms, candidate_created_at_ms, session_id),
        )
    _write_repo_edges(conn, session_id, session, update_session_observations=False)


@dataclass(frozen=True, slots=True)
class PreparedSessionRows:
    """Sealed row and identity evidence produced before writer admission.

    The content hash covers the full normalized input. Canonical preparation
    resolves lineage and append offsets against its own read snapshot, then
    captures exact owner resolution alongside the published row range. The
    writer validates the carrier and predecessor; it never lowers inline.
    """

    session_id: str
    session_content_hash: bytes
    message_rows: Sequence[tuple[object, ...]]
    block_rows: Sequence[tuple[object, ...]]
    #: Content-derived fallback identities resolved while preparing the rows.
    #: The writer may consume these only after validating that they still
    #: match the carried row tuples and the input content hash.
    content_identities: Sequence[MessageContentIdentity]
    owner_resolution: MessageOwnerResolution
    position_offset: int = 0
    #: Per-digest content-occurrence counts already stored for this session,
    #: the append-side analogue of ``position_offset``. Empty for a
    #: full-replace write, where the session's rows are the only rows.
    content_occurrence_offsets: tuple[tuple[str, int], ...] = ()
    scratch: tempfile.TemporaryDirectory[str] | None = None


class _ShardRowSequence(Sequence[tuple[object, ...]]):
    """Read one prepared table range without retaining bound row tuples."""

    def __init__(self, path: Path, table: str, lo: int, hi: int, *, lifetime: object | None = None) -> None:
        self._lifetime = lifetime
        self.path = path
        self.table = table
        self.lo = lo
        self.hi = hi

    def __len__(self) -> int:
        return max(0, self.hi - self.lo + 1)

    @overload
    def __getitem__(self, index: int) -> tuple[object, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[object, ...]]: ...

    def __getitem__(self, index: int | slice) -> tuple[object, ...] | list[tuple[object, ...]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = index + len(self) if index < 0 else index
        if ordinal < 0 or ordinal >= len(self):
            raise IndexError(index)
        with (
            retain_native_sql_lifetimes(*(() if self._lifetime is None else (self._lifetime,))),
            _prepared_reader(self.path) as conn,
        ):
            row = conn.execute(f"SELECT * FROM {self.table} WHERE rowid = ?", (self.lo + ordinal,)).fetchone()
        if row is None:
            raise PreparedSessionWriteRefusedError("prepared row disappeared")
        return tuple(row)

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        after = self.lo - 1
        while after < self.hi:
            with (
                retain_native_sql_lifetimes(*(() if self._lifetime is None else (self._lifetime,))),
                _prepared_reader(self.path) as conn,
            ):
                rows = conn.execute(
                    f"SELECT rowid, * FROM {self.table} WHERE rowid > ? AND rowid <= ? ORDER BY rowid LIMIT 512",
                    (after, self.hi),
                ).fetchall()
            if not rows:
                return
            after = int(rows[-1][0])
            yield from (tuple(row[1:]) for row in rows)


@dataclass(frozen=True, slots=True)
class _IdentityScope:
    """How a materialized child's stored message IDs were computed.

    A child that inherited a prefix stored only its tail, identified over the
    tail alone (duplicate native IDs and content occurrences counted within
    it). When a parent rewrite materializes the prefix into the child, the
    copies take native IDs except those listed in ``content_copies``: per
    content identity, the prefix-relative occurrence ordinals of the copies
    that took a content ID, numbered after the tail's occurrences. A replay of
    the now spawned-fresh child reproduces every stored ID from this record,
    so no ID moves (polylogue-5gg3u).
    """

    inherited_messages: int
    content_copies: Mapping[str, tuple[int, ...]]
    #: The copied prefix's content identities, as a digest of their sequence
    #: and the first of them, so a replay finds the copied messages by content
    #: even when the transcript gained a message before them.
    prefix_digest: str = ""
    first_identity: str = ""
    #: Per content identity, the occurrence the first content copy took: the
    #: tail's occurrences of it at materialization. A row outside the prefix
    #: from that point on (a later append) numbers past the copies, as the
    #: append allocated it, so a replay never hands it a copy's ID.
    copy_bases: Mapping[str, int] = field(default_factory=dict)
    #: The tail as materialization found it: its length and the digest of
    #: its content identities in order. A replay whose first ``tail_length``
    #: rows after the prefix still hash to it has only appended since.
    tail_length: int = 0
    tail_digest: str = ""
    #: Per copied prefix row, ``n`` when it was stored under its native ID
    #: (found by that ID) or ``c`` when by content (found by its content).
    prefix_kinds: str = ""
    #: The positions materialization gave the copied rows when it renumbered
    #: them (keyed by source session and position, which a replay cannot see),
    #: as ``(first, length)`` runs of consecutive positions; empty when the
    #: source coordinates were kept.
    prefix_positions: tuple[tuple[int, int], ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "inherited_messages": self.inherited_messages,
                "content_copies": {key: list(value) for key, value in sorted(self.content_copies.items())},
                "prefix_digest": self.prefix_digest,
                "first_identity": self.first_identity,
                "copy_bases": dict(sorted(self.copy_bases.items())),
                "tail_length": self.tail_length,
                "tail_digest": self.tail_digest,
                "prefix_kinds": self.prefix_kinds,
                "prefix_positions": [list(run) for run in self.prefix_positions],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, encoded: str) -> _IdentityScope | None:
        try:
            raw = json.loads(encoded)
            count = int(raw["inherited_messages"])
            copies = {
                str(key): tuple(sorted(int(item) for item in value))
                for key, value in dict(raw.get("content_copies") or {}).items()
            }
            bases = {str(key): int(value) for key, value in dict(raw.get("copy_bases") or {}).items()}
            tail_length = int(raw.get("tail_length") or 0)
            runs = tuple((int(run[0]), int(run[1])) for run in raw.get("prefix_positions") or ())
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        return (
            cls(
                count,
                copies,
                str(raw.get("prefix_digest") or ""),
                str(raw.get("first_identity") or ""),
                bases,
                tail_length,
                str(raw.get("tail_digest") or ""),
                str(raw.get("prefix_kinds") or ""),
                runs,
            )
            if count > 0
            else None
        )


def _record_identity_scope(conn: sqlite3.Connection, session_id: str, scope: _IdentityScope) -> None:
    conn.execute(
        """INSERT INTO session_identity_scopes (session_id, scope_json) VALUES (?, ?)
           ON CONFLICT(session_id) DO UPDATE SET scope_json = excluded.scope_json""",
        (session_id, scope.to_json()),
    )


def _prefix_key(native_id: str | None, content_identity: str | None) -> str | None:
    """The key a copied prefix row is found by: its native ID when it has one.

    A native row stays the same message across a content edit, so it is found
    by that ID; only an ID-less row is found by its content.
    """
    if native_id is not None:
        from polylogue.core.message_native_identity import message_native_key

        return message_native_key(native_id)
    return None if content_identity is None else f"c:{content_identity}"


def _position_runs(positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """``positions`` as ``(first, length)`` runs of consecutive values."""
    runs: list[list[int]] = []
    for position in positions:
        if runs and runs[-1][0] + runs[-1][1] == position:
            runs[-1][1] += 1
        else:
            runs.append([position, 1])
    return tuple((first, length) for first, length in runs)


def _expand_position_runs(runs: Iterable[tuple[int, int]]) -> list[int]:
    return [first + offset for first, length in runs for offset in range(length)]


def _identity_sequence_frame(identity: str) -> bytes:
    """Frame each exact key so opaque native names cannot move a boundary."""
    encoded = identity.encode("utf-8", "surrogatepass")
    return len(encoded).to_bytes(8, "big") + encoded


def _identity_sequence_digest(identities: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(_identity_sequence_frame(identity))
    return digest.hexdigest()


def _materialized_identity_scope(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> _IdentityScope | None:
    """The identity scope this child's stored IDs follow, if its prefix was materialized.

    Kept per session, not on an edge: a later revision that no longer declares
    the parent deletes the edge, and the IDs must still not move.
    """
    if before_input is not None:
        before_input(
            "session_identity_scopes",
            ("scope_json",),
            "SELECT rowid FROM session_identity_scopes WHERE session_id=?",
            (session_id,),
        )
    with connection_cursor(
        conn, "SELECT scope_json FROM session_identity_scopes WHERE session_id = ?", (session_id,)
    ) as _input_cursor:
        row = _input_cursor.fetchone()
    return _IdentityScope.from_json(str(row[0])) if row is not None else None


@dataclass(frozen=True, slots=True)
class PreparedMessageContext:
    """The writer-visible result of normalizing and slicing one input session.

    This deliberately retains the complete-input duplicate set separately from
    the surviving-tail duplicate set.  Session events refer to the former,
    while rows and coordinates use the latter.
    """

    effective_session: ParsedSession
    messages: Sequence[ParsedMessage]
    event_duplicate_native_ids: frozenset[str]
    duplicate_native_ids: frozenset[str]
    effective_session_kind: SessionKind
    hook_parent_native_id: str | None
    parent_session_id: str | None
    branch_point_message_id: str | None
    branch_point_content_address: bytes | None
    lineage_inheritance: str | None
    lineage_prefix_digest: bytes | None
    inherited_source_message_ids: Mapping[str, str]
    #: The retained source path of the raw this write publishes, which places
    #: a Codex child in its install root. A raw's source path never changes,
    #: so the write that consumes this context reads the same root.
    child_source_path: str | None
    #: The parent rows the inherited prefix resolves to, by prefix ordinal:
    #: ``messages`` is a ``_MessageTail`` whose ``start`` is this length.
    #: Evidence an inherited message owns (attachments) resolves through it.
    inherited_prefix_message_ids: Sequence[str] = ()
    #: The identity scope a materialized prefix recorded on this child's edge
    #: (``_IdentityScope``) and the content identities it assigns, index-aligned
    #: with ``messages``. ``None`` for every other write.
    identity_scope: _IdentityScope | None = None
    content_identities: tuple[MessageContentIdentity, ...] | None = None

    def close(self) -> None:
        """Attempt every owned carrier's cleanup before reporting unsettled SQL."""
        artifacts: list[_DiskDuplicateNativeIds | _DiskSourceMessageIds | _DiskSignatureSequence] = []
        for duplicates in (self.event_duplicate_native_ids, self.duplicate_native_ids):
            if isinstance(duplicates, _DiskDuplicateNativeIds):
                artifacts.append(duplicates)
        if isinstance(self.inherited_source_message_ids, _DiskSourceMessageIds):
            artifacts.append(self.inherited_source_message_ids)
        if isinstance(self.inherited_prefix_message_ids, _PrefixMessageIds):
            composed = self.inherited_prefix_message_ids._composed
            if isinstance(composed, _DiskSignatureSequence):
                artifacts.append(composed)
        failures: list[BaseException] = []
        for artifact in artifacts:
            try:
                artifact.close()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            raise PreparedSessionSettlementError(failures)


class _MessageTail(Sequence[ParsedMessage]):
    """Ordinal view of a prepared sequence after a shared lineage prefix."""

    def __init__(self, messages: Sequence[ParsedMessage], start: int) -> None:
        self.messages = messages
        self.start = start
        self.path = getattr(messages, "path", None)

    def __len__(self) -> int:
        return len(self.messages) - self.start

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | list[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = index + len(self) if index < 0 else index
        if ordinal < 0 or ordinal >= len(self):
            raise IndexError(index)
        return self.messages[self.start + ordinal]

    def __iter__(self) -> Iterator[ParsedMessage]:
        if isinstance(self.messages, SqliteMessageSink):
            yield from self.messages.iter_from(self.start)
            return
        for ordinal, message in enumerate(self.messages):
            if ordinal >= self.start:
                yield message


class _MessagePrefix(Sequence[ParsedMessage]):
    """Ordinal view of the leading ``stop`` messages of a prepared sequence.

    The inherited prefix a ``_MessageTail`` slices away, kept addressable so
    evidence an inherited message owns resolves against the prefix ordinals.
    """

    def __init__(self, messages: Sequence[ParsedMessage], stop: int) -> None:
        self.messages = messages
        self.stop = max(0, min(stop, len(messages)))
        self.path = getattr(messages, "path", None)

    def __len__(self) -> int:
        return self.stop

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | list[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = index + len(self) if index < 0 else index
        if ordinal < 0 or ordinal >= len(self):
            raise IndexError(index)
        return self.messages[ordinal]

    def __iter__(self) -> Iterator[ParsedMessage]:
        for ordinal, message in enumerate(self.messages):
            if ordinal >= self.stop:
                return
            yield message


class _PrefixMessageIds(Sequence[str]):
    """The composed parent row each inherited prefix ordinal resolves to.

    A view over the parent's composed ``(message_id, signature)`` sequence
    rather than a copy, so a disk-backed composition stays on disk.
    """

    def __init__(self, composed: Sequence[tuple[str, str]], count: int) -> None:
        self._composed = composed
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return self._composed[ordinal][0]


@contextmanager
def _prefix_attachment_owner_ordinals(
    messages: Sequence[ParsedMessage],
    prefix_count: int,
    attachments: Sequence[ParsedAttachment],
) -> Iterator[tuple[dict[object, int], frozenset[object]]]:
    """Resolve attachments to the inherited prefix message that owns them.

    Owner keys are resolved over the first ``prefix_count`` of ``messages``
    with the same private contract ``_write_attachments`` applies to a tail,
    so an inherited message owns exactly the attachments it would own as a
    stored message. Yields ``(prefix ordinal by acquisition key, acquisition
    keys whose prefix owner is ambiguous)``; an attachment no prefix message
    owns is in neither.
    """
    prefix = _MessagePrefix(messages, prefix_count)
    if not attachments or not len(prefix):
        yield {}, frozenset()
        return
    scope: AbstractContextManager[MessageOwnerResolution] = (
        disk_message_owner_resolution(prefix)
        if isinstance(messages, SqliteMessageSink)
        else nullcontext(message_owner_resolution(list(prefix)))
    )
    with scope as resolution:
        wanted: defaultdict[str, list[object]] = defaultdict(list)
        ambiguous: set[object] = set()
        for attachment in attachments:
            try:
                owner_key = attachment_message_owner_key(attachment, resolution)
            except MessageOwnerAmbiguityError:
                ambiguous.add(attachment.acquisition_key)
                continue
            if owner_key is None or owner_key in resolution.ambiguous_keys:
                continue
            wanted[owner_key].append(attachment.acquisition_key)
        ordinals: dict[object, int] = {}
        if wanted:
            for ordinal, owner_key in enumerate(resolution.keys):
                for acquisition_key in wanted.pop(owner_key, ()):
                    ordinals[acquisition_key] = ordinal
                if not wanted:
                    break
        yield ordinals, frozenset(ambiguous)


def _message_references_attachment(conn: sqlite3.Connection, message_id: str, attachment_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM attachment_refs WHERE message_id = ? AND attachment_id = ? LIMIT 1",
            (message_id, attachment_id),
        ).fetchone()
        is not None
    )


def _attachment_shared_prefix_limit(
    conn: sqlite3.Connection,
    messages: Sequence[ParsedMessage],
    parent_composed: Sequence[tuple[str, str]],
    shared: int,
    attachments: Sequence[ParsedAttachment],
) -> int:
    """End a signature-aligned prefix before an inherited message the parent cannot represent.

    A message signature covers the role and blocks, not the attachments, so a
    replayed message can carry an attachment its parent row does not
    reference. Inheriting that message would leave the attachment without an
    owner any composed read reaches; ending the shared prefix before it keeps
    the message, and its attachment, in the child's own tail. An attachment the
    parent row already references is the parent's observation and inherits
    with its message.
    """
    if not attachments or shared <= 0:
        return shared
    limit = shared
    with _prefix_attachment_owner_ordinals(messages, shared, attachments) as (ordinals, _ambiguous):
        for attachment in attachments:
            ordinal = ordinals.get(attachment.acquisition_key)
            if ordinal is None or ordinal >= limit:
                continue
            if not _message_references_attachment(conn, parent_composed[ordinal][0], _attachment_id("", attachment)):
                limit = ordinal
    return limit


def _file_edit_shared_prefix_limit(messages: Sequence[ParsedMessage], shared: int) -> int:
    """Keep a divergent edit result and its exact invocation in the same owner.

    File-edit rows are keyed by invocation block. A child cannot inherit that
    block and publish a different edit result without replacing parent evidence.
    """
    if shared <= 0 or not any(
        block.file_edit is not None and _block_type(block) is BlockType.TOOL_RESULT
        for message in messages
        for block in _message_blocks(message)
    ):
        return shared
    from polylogue.sources.tool_outcomes import iter_tool_result_owners

    limit = shared
    with closing(iter_tool_result_owners(messages)) as associations:
        for result_message, result_block, use_message, _use_block in associations:
            if (
                min(result_message, use_message) < limit <= max(result_message, use_message)
                and _message_blocks(messages[result_message])[result_block].file_edit is not None
            ):
                limit = min(result_message, use_message)
    return limit


def _stored_file_edit_shared_prefix_limit(
    conn: sqlite3.Connection, child_session_id: str, child_composed: Sequence[tuple[str, str]], shared: int
) -> int:
    """The same ownership boundary for a child stored before its parent."""
    if (
        shared <= 0
        or conn.execute("SELECT 1 FROM file_edits WHERE session_id=? LIMIT 1", (child_session_id,)).fetchone() is None
    ):
        return shared
    table = f"temp.{_GUARD_PREFIX}file_edit_prefix"
    conn.execute(f"CREATE TEMP TABLE {table} (message_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL)")
    try:
        conn.executemany(
            f"INSERT INTO {table} VALUES (?, ?)",
            ((message_id, ordinal) for ordinal, (message_id, _digest) in enumerate(islice(child_composed, shared))),
        )
        row = conn.execute(
            f"""WITH RECURSIVE intervals(lo,hi) AS (
                SELECT MIN(COALESCE(invocation.ordinal,:shared),COALESCE(result.ordinal,:shared)),
                       MAX(COALESCE(invocation.ordinal,:shared),COALESCE(result.ordinal,:shared))
                FROM file_edits AS f JOIN blocks AS b ON b.block_id=f.tool_use_block_id
                LEFT JOIN {table} AS invocation ON invocation.message_id=b.message_id
                LEFT JOIN {table} AS result ON result.message_id=f.message_id
                WHERE f.session_id=:child
            ), boundary(ordinal) AS (
                SELECT :shared UNION
                SELECT i.lo FROM intervals AS i JOIN boundary AS b ON i.lo < b.ordinal AND i.hi >= b.ordinal
            ) SELECT MIN(ordinal) FROM boundary""",
            {"child": child_session_id, "shared": shared},
        ).fetchone()
        return shared if row[0] is None else min(shared, int(row[0]))
    finally:
        conn.execute(f"DROP TABLE {table}")


def _stored_attachment_shared_prefix_limit(
    conn: sqlite3.Connection,
    child_composed: Sequence[tuple[str, str]],
    parent_composed: Sequence[tuple[str, str]],
    shared: int,
) -> int:
    """``_attachment_shared_prefix_limit`` for a child whose rows are already stored.

    A child written before its parent owns its whole transcript, including the
    attachment references of the messages its parent later turns out to share.
    Those rows are deleted when the prefix is extracted, so the prefix must end
    before the first one holding a reference its parent row lacks -- the same
    boundary the child would have taken had the parent been written first.
    """
    limit = shared
    for start in range(0, shared, 500):
        # The composed sequences may be disk-backed. Keep only one SQL batch
        # resident, and stop once the earliest attachment boundary is known.
        ordinal_by_message = {child_composed[ordinal][0]: ordinal for ordinal in range(start, min(start + 500, shared))}
        batch = tuple(ordinal_by_message)
        placeholders = ",".join("?" for _ in batch)
        for message_id, attachment_id in conn.execute(
            f"SELECT message_id, attachment_id FROM attachment_refs WHERE message_id IN ({placeholders})",
            batch,
        ):
            ordinal = ordinal_by_message[str(message_id)]
            if ordinal < limit and not _message_references_attachment(
                conn, parent_composed[ordinal][0], str(attachment_id)
            ):
                limit = ordinal
        if limit < shared:
            return limit
    return limit


def _scoped_identities(
    messages: Sequence[ParsedMessage], scope: _IdentityScope
) -> tuple[_ScopedMessages, tuple[MessageContentIdentity, ...]]:
    """A view and content identities that reproduce a materialized child's IDs.

    The tail keeps the identity it had while inheriting: duplicates and
    occurrences over the tail alone. A prefix message takes its native ID
    unless its prefix-relative occurrence is listed in the scope, or it has
    none; a listed copy takes the content occurrence after the tail's. The
    decision is carried by the message itself (an ID-less view of a content
    copy), so the rows need no duplicate set. Positions are placed as
    materialization placed them: prefix below tail, renumbered densely when
    the parsed coordinates collide (``_plan_prefix_positions``).

    Messages are streamed, never listed, so a disk-backed session stays on
    disk; only the per-message identity pairs are held.
    """
    count = min(scope.inherited_messages, len(messages))
    digests = [message_content_identity(message) for message in messages]
    natives_seen = [_normalized_message_native_id(message) for message in messages]
    start = _copied_prefix_start(natives_seen, digests, scope, count)
    tail_counts: Counter[str] = Counter()
    tail_natives: Counter[str] = Counter()
    prefix_keys: list[tuple[int, int]] = []
    before_positions: list[int] = []
    natives: list[str | None] = []
    min_tail_position: int | None = None
    for ordinal, message in enumerate(messages):
        digest = digests[ordinal]
        position = message.position if message.position is not None else ordinal
        native = _normalized_message_native_id(message)
        natives.append(native)
        if start <= ordinal < start + count:
            prefix_keys.append((position, message.variant_index or 0))
            continue
        tail_counts[digest] += 1
        if native is not None:
            tail_natives[native] += 1
        if ordinal < start:
            before_positions.append(position)
        elif min_tail_position is None or position < min_tail_position:
            min_tail_position = position
    positions = [position for position, _variant in prefix_keys]
    fits = (
        len(set(prefix_keys)) == len(prefix_keys)
        and positions == sorted(positions)
        and (min_tail_position is None or not positions or positions[-1] < min_tail_position)
        and (not before_positions or not positions or max(before_positions) < positions[0])
    )
    recorded = _expand_position_runs(scope.prefix_positions)
    if len(recorded) != count:
        recorded = []
    prefix_positions: dict[int, int] = {}
    prefix_ordinal_positions: dict[int, int] = {}
    before_positions_map: dict[int, int] = {}
    tail_shift = 0
    if recorded or not fits:
        # Rows gained before the copied prefix take the lowest positions,
        # then the prefix, then the tail above both -- the order
        # materialization composed, with room for what was added since. The
        # prefix takes the positions materialization recorded, row by row,
        # since it numbered them by source session and position.
        for position in sorted(set(before_positions)):
            before_positions_map[position] = len(before_positions_map)
        base = len(before_positions_map)
        if recorded:
            for offset, (position, placed) in enumerate(zip(positions, recorded, strict=True)):
                prefix_ordinal_positions[offset] = base + placed
                prefix_positions.setdefault(position, base + placed)
            floor = base + max(recorded) + 1
        else:
            for position in positions:
                prefix_positions.setdefault(position, base + len(prefix_positions))
            floor = base + len(prefix_positions)
        if min_tail_position is not None and min_tail_position < floor:
            tail_shift = floor - min_tail_position
    # Occurrences are decided in three passes, then listed by ordinal: rows
    # after the prefix keep the numbering they had (the tail's at
    # materialization, then past the copies for later appends), recorded
    # copies take theirs, and rows the materialization never saw (gained
    # before the prefix, or an unrecorded ID-less prefix row) take fresh
    # occurrences past every one of those.
    after = digests[start + count :]
    tail_changed = bool(scope.tail_digest) and (
        len(after) < scope.tail_length or _identity_sequence_digest(after[: scope.tail_length]) != scope.tail_digest
    )
    if tail_changed:
        after_counts = Counter(after)
        for digest in scope.content_copies:
            if after_counts[digest] > scope.copy_bases.get(digest, tail_counts[digest]):
                # A message identical to a copy was added among the tail, not
                # only appended after it: which rows hold the stored
                # occurrences is no longer evident. Refused, never guessed.
                raise InheritedPrefixMaterializationError(
                    "a message identical to a copied prefix row was added inside the tail; its stored IDs are ambiguous"
                )
    occurrences: dict[int, int] = {}
    cleared: set[int] = set()
    taken: dict[str, set[int]] = defaultdict(set)
    after_seen: Counter[str] = Counter()
    for ordinal in range(start + count, len(digests)):
        digest = digests[ordinal]
        seen = after_seen[digest]
        after_seen[digest] += 1
        copies = scope.content_copies.get(digest, ())
        base = scope.copy_bases.get(digest, tail_counts[digest]) if copies else seen + 1
        occurrences[ordinal] = seen if seen < base else seen + len(copies)
        taken[digest].add(occurrences[ordinal])
    prefix_seen: Counter[str] = Counter()
    unrecorded: list[int] = []
    for ordinal in range(start, start + count):
        digest = digests[ordinal]
        occurrence = prefix_seen[digest]
        prefix_seen[digest] += 1
        copies = scope.content_copies.get(digest, ())
        if occurrence in copies:
            cleared.add(ordinal)
            occurrences[ordinal] = scope.copy_bases.get(digest, tail_counts[digest]) + copies.index(occurrence)
            taken[digest].add(occurrences[ordinal])
        else:
            # Native, or an ID-less message the scope did not record: the
            # occurrence stays unique and past every recorded one.
            unrecorded.append(ordinal)
    for ordinal in [*range(0, start), *unrecorded]:
        digest = digests[ordinal]
        fresh = max(taken[digest], default=-1) + 1
        occurrences[ordinal] = fresh
        taken[digest].add(fresh)
    identities = [(digest, occurrences[ordinal]) for ordinal, digest in enumerate(digests)]
    # A native repeated outside the prefix, or shared with a copied prefix row
    # that keeps it, is a content identity for the non-prefix row: the stored
    # prefix row keeps its ID.
    prefix_natives = {
        native
        for ordinal, native in enumerate(natives)
        if native is not None and start <= ordinal < start + count and ordinal not in cleared
    }
    tail_duplicates = frozenset(native for native, seen in tail_natives.items() if seen > 1 or native in prefix_natives)
    view = _ScopedMessages(
        messages,
        start=start,
        before_positions=before_positions_map,
        count=count,
        cleared=frozenset(cleared),
        tail_duplicates=tail_duplicates,
        prefix_positions=prefix_positions,
        prefix_ordinal_positions=prefix_ordinal_positions,
        tail_shift=tail_shift,
    )
    return view, tuple(identities)


def _copied_prefix_start(
    natives: Sequence[str | None], digests: Sequence[str], scope: _IdentityScope, count: int
) -> int:
    """Where the materialized prefix sits in this transcript, found by content.

    The copied messages are the run of ``count`` whose content identities
    hash to the recorded sequence; a message gained before them does not move
    which messages they are. Without a match the copied prefix itself
    changed, and no stable evidence says which messages hold its stored IDs:
    refused by name rather than guessed from the leading ones.
    """
    if not scope.prefix_digest:
        return 0
    kinds = scope.prefix_kinds or "c" * count

    def key(ordinal: int, kind: str) -> str:
        # A row stored under its native ID is found by that ID, so a content
        # edit does not lose it; a row stored by content is found by content.
        native = natives[ordinal]
        return native if kind == "n" and native is not None else f"c:{digests[ordinal]}" if kind == "c" else ""

    matches = [
        start
        for start in range(0, len(digests) - count + 1)
        if key(start, kinds[0]) == scope.first_identity
        and _identity_sequence_digest(key(start + offset, kinds[offset]) for offset in range(count))
        == scope.prefix_digest
    ]
    if len(matches) > 1:
        # Two runs hold the copied content: choosing one could hand a new
        # message a stored ID. Refused by name rather than guessed.
        raise InheritedPrefixMaterializationError(
            f"the materialized prefix appears {len(matches)} times in this transcript; its stored IDs are ambiguous"
        )
    if not matches:
        raise InheritedPrefixMaterializationError(
            "the materialized prefix no longer appears in this transcript; its stored IDs cannot be placed"
        )
    return matches[0]


class _ScopedMessages(_MessageTail):
    """A materialized child's messages as its stored rows identify them.

    Wraps the parsed sequence (a disk-backed sink stays one, so the writer's
    streaming route still takes it) and applies the identity scope per
    message: a content copy or a tail duplicate is presented without its
    native ID, and positions are placed as materialization placed them.
    """

    def __init__(
        self,
        messages: Sequence[ParsedMessage],
        *,
        start: int,
        before_positions: Mapping[int, int],
        count: int,
        cleared: frozenset[int],
        tail_duplicates: frozenset[str],
        prefix_positions: Mapping[int, int],
        tail_shift: int,
        prefix_ordinal_positions: Mapping[int, int] | None = None,
    ) -> None:
        super().__init__(messages, 0)
        #: Prefix-relative ordinal -> the position materialization recorded.
        self._prefix_ordinal_positions = prefix_ordinal_positions or {}
        self._prefix_start = start
        self._before_positions = before_positions
        self._count = count
        self._cleared = cleared
        self._tail_duplicates = tail_duplicates
        self._prefix_positions = prefix_positions
        self._tail_shift = tail_shift

    def remap_position(self, position: int | None, *, region: str) -> int | None:
        """``position`` placed as materialization placed its region's rows."""
        if position is None:
            return None
        if region == "prefix":
            return self._prefix_positions.get(position, position)
        if region == "before":
            return self._before_positions.get(position, position)
        return position + self._tail_shift

    def _region(self, ordinal: int) -> str:
        if ordinal < self._prefix_start:
            return "before"
        return "prefix" if ordinal < self._prefix_start + self._count else "tail"

    def _scoped(self, ordinal: int, message: ParsedMessage) -> ParsedMessage:
        update: dict[str, object] = {}
        region = self._region(ordinal)
        if ordinal in self._cleared or (
            region != "prefix" and _normalized_message_native_id(message) in self._tail_duplicates
        ):
            update["provider_message_id"] = None
        placed = self._prefix_ordinal_positions.get(ordinal - self._prefix_start) if region == "prefix" else None
        position = placed if placed is not None else self.remap_position(message.position, region=region)
        if position != message.position:
            update["position"] = position
        return message.model_copy(update=update) if update else message

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | list[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = index + len(self) if index < 0 else index
        return self._scoped(ordinal, super().__getitem__(ordinal))

    def __iter__(self) -> Iterator[ParsedMessage]:
        for ordinal, message in enumerate(super().__iter__()):
            yield self._scoped(ordinal, message)

    def remap_events(self, events: Sequence[ParsedSessionEvent]) -> list[ParsedSessionEvent] | None:
        """Boundary positions moved with the rows they address, or ``None`` if none move."""
        if not self._prefix_positions and not self._tail_shift and not self._before_positions:
            return None
        min_tail = min(
            (
                message.position
                for ordinal, message in enumerate(super().__iter__())
                if ordinal >= self._prefix_start + self._count and message.position is not None
            ),
            default=None,
        )

        def moved(position: int | None) -> int | None:
            if position is None:
                return None
            if min_tail is not None and position >= min_tail:
                return self.remap_position(position, region="tail")
            if position in self._prefix_positions:
                return self.remap_position(position, region="prefix")
            return self.remap_position(position, region="before")

        return [
            event.model_copy(
                update={
                    "boundary_start_position": moved(event.boundary_start_position),
                    "boundary_end_position": moved(event.boundary_end_position),
                }
            )
            for event in events
        ]


@dataclass(frozen=True, slots=True)
class PreparedSessionWrite:
    """An exact prepared lowering for one normalized pending replay write.

    ``input_content_hash`` and ``rows.session_content_hash`` cover the
    timestamp-normalized pending input before lineage slicing.  The aggregate
    chain hash is intentionally outside this carrier.
    """

    session_id: str
    input_content_hash: bytes
    merge_append: bool
    fallback_timestamp: str | None
    context: PreparedMessageContext
    rows: PreparedSessionRows
    predecessor: tuple[object, ...] | None
    reparse_receipt: RevisionApplicationReceipt | None
    enrichment_binding: tuple[str | None, str | None] | None
    cross_acquisition_union: _PreparedCrossAcquisitionUnion | None = None

    def close(self) -> None:
        """Close every carrier before retiring its actual scratch directory."""
        failures: list[BaseException] = []
        if self.cross_acquisition_union is not None:
            scratch = self.cross_acquisition_union.carry_forward.scratch
            if scratch is not None:
                try:
                    scratch.close()
                except BaseException as failure:
                    failures.append(failure)
        try:
            self.context.close()
        except BaseException as failure:
            failures.append(failure)
        if failures:
            raise PreparedSessionSettlementError(failures)
        if self.rows.scratch is not None:
            _cleanup_prepared_scratch(self.rows.scratch)


def _cleanup_prepared_scratch(scratch: tempfile.TemporaryDirectory[str]) -> None:
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

    pending = retained_native_sql_owners_for_lifetime(scratch)
    if pending:
        raise NativeConnectionSettlementError(
            pending[0], RuntimeError("prepared write scratch retains physical SQL custody")
        )
    scratch.cleanup()


@dataclass(frozen=True, slots=True)
class _PreparedCrossAcquisitionUnion:
    """Merged rows and old projections pinned to one predecessor revision."""

    predecessor: tuple[object, ...]
    prefix_sharing_parent: bool
    rows: PreparedSessionRows
    carry_forward: _ProjectionCarryForward


#: Publication counts accepted canonical preparations. Refused carriers fail
#: before mutation; no writer-side lowering fallback exists. The counter is
#: diagnostic and does not decide admission.
_PREPARED_DISPOSITIONS: Counter[str] = Counter()
_PREPARED_DISPOSITIONS_LOCK = threading.Lock()

#: Reasons that mean the writer used the prepared work. Everything else is a
#: decline, and the split is what a measurement reads.
PREPARED_ACCEPTED_DISPOSITIONS = frozenset({"prepared_write"})


def _record_prepared_disposition(reason: str) -> None:
    with _PREPARED_DISPOSITIONS_LOCK:
        _PREPARED_DISPOSITIONS[reason] += 1


def prepared_row_dispositions() -> dict[str, int]:
    """Snapshot of how session writes resolved the prepared-rows question."""
    with _PREPARED_DISPOSITIONS_LOCK:
        return dict(_PREPARED_DISPOSITIONS)


def reset_prepared_row_dispositions() -> None:
    """Clear the counters so a caller can measure one bounded interval."""
    with _PREPARED_DISPOSITIONS_LOCK:
        _PREPARED_DISPOSITIONS.clear()


def _validated_prepared_content_identities(
    prepared: PreparedSessionRows,
    messages: Sequence[ParsedMessage],
) -> Sequence[MessageContentIdentity]:
    """Validate and return the parse-side identity carrier without hashing.

    The tuple is source-bound by the prepared content hash admission gate. For
    tuple rows, also check the two identity columns that will be inserted; a
    changed or corrupted carrier must refuse rather than silently regenerate
    identity on the writer.
    """
    identities = prepared.content_identities
    if len(identities) != len(messages):
        raise PreparedSessionWriteRefusedError("prepared identity carrier does not cover the parsed messages")
    for identity, occurrence in identities:
        identity_value: object = identity
        occurrence_value: object = occurrence
        if (
            not isinstance(identity_value, str)
            or not identity_value
            or not isinstance(occurrence_value, int)
            or occurrence_value < 0
        ):
            raise PreparedSessionWriteRefusedError("prepared identity carrier has an invalid value")
    if isinstance(prepared, PreparedSessionRows):
        insert_columns = tuple(column.name for column in archive_tiers_specs.MESSAGES_SPEC.insert_columns)
        try:
            identity_index = insert_columns.index("content_identity")
            occurrence_index = insert_columns.index("content_occurrence")
        except ValueError as exc:
            raise PreparedSessionWriteRefusedError("message row identity columns are missing") from exc
        if len(prepared.message_rows) != len(identities) or any(
            (row[identity_index], row[occurrence_index]) != identity
            for row, identity in zip(prepared.message_rows, identities, strict=True)
        ):
            raise PreparedSessionWriteRefusedError("prepared identity carrier disagrees with message rows")
    return identities


def _prepared_message_context(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    origin: Origin,
    session_id: str,
    native_id: str,
    merge_append: bool,
    signature_cache: _SignatureCacheLike | None,
    source_read: SessionSourceRead | None,
    child_source_path: str | None,
    before_input: BeforeIndexInput | None = None,
) -> PreparedMessageContext:
    """Canonical normalization-before-lineage-slicing context for one write."""
    if isinstance(session.messages, SqliteMessageSink):
        # Borrow the canonical operand sealed beside the original parser fields.
        messages: Sequence[ParsedMessage] = session.messages.normalized_messages(session.session_events, origin=origin)
    else:
        messages = _derive_tool_outcomes(
            normalize_active_branch(session.messages), session.session_events, origin=origin
        )
    event_duplicate_native_ids = _duplicate_message_native_ids(messages)
    effective_session = session
    hook_parent_claim = _authoritative_parent_claim(
        conn,
        source_read,
        origin=origin.value,
        child_session_id=session_id,
        child_native_id=native_id,
        child_provider_values=_child_provider_values(session),
        parent_candidate=session.parent_session_provider_id,
        child_source_path=child_source_path,
        before_input=before_input,
    )
    hook_parent_provider_id = hook_parent_claim.parent_native_id if hook_parent_claim is not None else None
    effective_session_kind = session.session_kind
    if hook_parent_provider_id is not None and session.parent_session_provider_id is None:
        effective_session_kind = SessionKind.SUBAGENT
    parent_session_id: str | None = None
    branch_point_message_id: str | None = None
    branch_point_content_address: bytes | None = None
    lineage_inheritance: str | None = None
    lineage_prefix_digest: bytes | None = None
    inherited_source_message_ids: Mapping[str, str] = {}
    inherited_prefix_message_ids: Sequence[str] = ()
    # A child whose prefix was materialized owns its whole transcript and the
    # IDs recorded for it; a replay keeps it spawned-fresh rather than slicing
    # it against whatever the parent holds now, which would move those IDs.
    # Only a full write re-applies it to IDs; an append keeps stored IDs.
    identity_scope = _materialized_identity_scope(conn, session_id, before_input)
    parent_composed: Sequence[tuple[str, str]] | None = None
    if not merge_append:
        lineage_session = session
        if hook_parent_provider_id is not None:
            lineage_session = session.model_copy(update={"parent_session_provider_id": hook_parent_provider_id})
        parent_session_id = _existing_parent_session_id(conn, lineage_session, origin.value, before_input)
        acompact = _is_claude_code_acompact_session(session)
        force_spawned_fresh = False
        if parent_session_id is not None and messages:
            cycle_walk = _would_create_cycle(
                conn, child_id=session_id, proposed_parent_id=parent_session_id, before_input=before_input
            )
            force_spawned_fresh = cycle_walk.outcome != "acyclic"
            if acompact:
                parent_composed = (
                    _disk_composed_db_signatures(
                        conn, parent_session_id, messages.path.parent, before_input=before_input
                    )
                    if isinstance(messages, SqliteMessageSink)
                    else _composed_db_signatures(
                        conn, parent_session_id, cache=signature_cache, before_input=before_input
                    )
                )
                membership = _acompact_content_membership_ratio(
                    parent_composed,
                    _iter_parsed_acompact_prefix_signatures(messages)
                    if isinstance(messages, SqliteMessageSink)
                    else _parsed_acompact_prefix_signatures(messages),
                )
                if membership is not None:
                    if membership < _ACOMPACT_PARENT_MEMBERSHIP_THRESHOLD:
                        effective_session = session.model_copy(update={"branch_type": BranchType.SIDECHAIN})
                        lineage_inheritance = "spawned-fresh"
                        force_spawned_fresh = True
                    elif session.branch_type is BranchType.SIDECHAIN:
                        effective_session = session.model_copy(update={"branch_type": BranchType.CONTINUATION})
                elif session.branch_type is BranchType.SIDECHAIN:
                    lineage_inheritance = "spawned-fresh"
                    force_spawned_fresh = True
            # Drive's ``branchParent.promptId`` is a source-asserted
            # cross-session relation without a local branch-message id.  Its
            # chunks are not proof that this child replays the parent's
            # prefix; content alignment would fabricate a branch point from
            # coincidental text.  Keep the topology edge and leave its
            # branch_point_message_id explicitly unresolved.
            drive_prompt_parent = origin is Origin.AISTUDIO_DRIVE and bool(session.parent_session_provider_id)
            if identity_scope is not None:
                lineage_inheritance = "spawned-fresh"
                force_spawned_fresh = True
            if not force_spawned_fresh and not drive_prompt_parent:
                (
                    branch_point_message_id,
                    lineage_inheritance,
                    messages,
                    inherited_source_message_ids,
                    lineage_prefix_digest,
                    inherited_prefix_message_ids,
                ) = _extract_prefix_tail(
                    conn,
                    parent_session_id,
                    messages,
                    cache=signature_cache,
                    parent_composed=parent_composed,
                    attachments=session.attachments,
                    before_input=before_input,
                )
            if branch_point_message_id is not None:
                branch_point_content_address = _message_content_address_for_id(
                    conn, branch_point_message_id, before_input
                )
    scoped_identities: tuple[MessageContentIdentity, ...] | None = None
    if identity_scope is not None and not merge_append:
        scoped_view, scoped_identities = _scoped_identities(messages, identity_scope)
        messages = scoped_view
        remapped_events = scoped_view.remap_events(list(effective_session.session_events))
        if remapped_events is not None:
            effective_session = effective_session.model_copy(update={"session_events": remapped_events})
    if isinstance(parent_composed, _DiskSignatureSequence) and not (
        isinstance(inherited_prefix_message_ids, _PrefixMessageIds)
        and inherited_prefix_message_ids._composed is parent_composed
    ):
        parent_composed.close()
    return PreparedMessageContext(
        effective_session=effective_session,
        messages=messages if isinstance(messages, (SqliteMessageSink, _MessageTail)) else tuple(messages),
        event_duplicate_native_ids=event_duplicate_native_ids,
        duplicate_native_ids=frozenset() if scoped_identities is not None else _duplicate_message_native_ids(messages),
        identity_scope=identity_scope,
        content_identities=scoped_identities,
        effective_session_kind=effective_session_kind,
        hook_parent_native_id=hook_parent_provider_id,
        parent_session_id=parent_session_id,
        branch_point_message_id=branch_point_message_id,
        branch_point_content_address=branch_point_content_address,
        lineage_inheritance=lineage_inheritance,
        lineage_prefix_digest=lineage_prefix_digest,
        inherited_source_message_ids=inherited_source_message_ids,
        inherited_prefix_message_ids=inherited_prefix_message_ids,
        child_source_path=child_source_path,
    )


def prepared_lineage_bindings(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    source_read: SessionSourceRead | None = None,
    child_source_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Return the hook and resolved-parent claims a prepared write depends on.

    ``child_source_path`` is the retained source path of the raw the write
    publishes (:func:`raw_source_path`).
    """
    origin = origin_from_provider(session.source_name)
    native_id = _stored_session_native_id(session.provider_session_id)
    session_id = archive_session_id(origin.value, native_id)
    claim = _authoritative_parent_claim(
        conn,
        source_read,
        origin=origin.value,
        child_session_id=session_id,
        child_native_id=native_id,
        child_provider_values=_child_provider_values(session),
        parent_candidate=session.parent_session_provider_id,
        child_source_path=child_source_path,
    )
    hook_parent_native_id = claim.parent_native_id if claim is not None else None
    lineage_session = (
        session.model_copy(update={"parent_session_provider_id": hook_parent_native_id})
        if hook_parent_native_id is not None
        else session
    )
    return hook_parent_native_id, _existing_parent_session_id(conn, lineage_session, origin.value)


def validate_prepared_session_lineage(
    conn: sqlite3.Connection,
    input_session: ParsedSession,
    prepared_write: PreparedSessionWrite,
    *,
    source_read: SessionSourceRead | None,
) -> None:
    """Revalidate the original prepared parent and inherited-byte witnesses."""
    context = prepared_write.context
    hook_parent, parent = prepared_lineage_bindings(
        conn, input_session, source_read=source_read, child_source_path=context.child_source_path
    )
    if hook_parent != context.hook_parent_native_id or (
        not prepared_write.merge_append and parent != context.parent_session_id
    ):
        raise PreparedSessionWriteRefusedError("prepared replay lineage evidence changed")
    if context.lineage_prefix_digest is None:
        return
    if parent is None:
        raise PreparedSessionWriteRefusedError("prepared replay lineage parent disappeared")
    inherited_count = len(input_session.messages) - len(context.messages)
    source = input_session.messages
    if isinstance(source, SqliteMessageSink):
        signatures: Sequence[tuple[str, str]] = _disk_composed_db_signatures(
            conn, parent, source.path.parent, prefix_length=inherited_count
        )
    else:
        with closing(_iter_composed_rows(conn, parent)) as rows:
            signatures = [(message_id, digest) for message_id, digest, _owner in islice(rows, inherited_count)]
    primary: BaseException | None = None
    try:
        if (
            len(signatures) < inherited_count
            or _lineage_prefix_digest(islice(signatures, inherited_count)) != context.lineage_prefix_digest
        ):
            raise PreparedSessionWriteRefusedError("prepared replay lineage prefix changed")
        if (
            isinstance(context.messages, _MessageTail)
            and _attachment_shared_prefix_limit(
                conn, context.messages.messages, signatures, inherited_count, context.effective_session.attachments
            )
            < inherited_count
        ):
            raise PreparedSessionWriteRefusedError("prepared replay lineage prefix attachments changed")
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if isinstance(signatures, _DiskSignatureSequence):
            try:
                signatures.close()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "prepared lineage and native settlement failed", [primary, cleanup]
                    ) from None
                raise


def prepared_session_storage_lineage_matches(conn: sqlite3.Connection, prepared_write: PreparedSessionWrite) -> bool:
    """Whether unchanged content already has this exact prefix representation."""
    context = prepared_write.context
    edge = _prefix_sharing_edge_row(conn, prepared_write.session_id)
    if context.lineage_prefix_digest is None:
        return edge is None
    return (
        context.lineage_inheritance == "prefix-sharing"
        and edge is not None
        and tuple(edge)
        == (
            context.parent_session_id,
            context.branch_point_message_id,
            context.branch_point_content_address,
        )
    )


def prepare_session_write(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    merge_append: bool,
    fallback_timestamp: str | None = None,
    source_read: SessionSourceRead | None = None,
    signature_cache: _SignatureCacheLike | None = None,
    raw_id: str | None = None,
    force_replace: bool = False,
    prepared_rows: PreparedSessionRows | None = None,
    before_input: BeforeIndexInput | None = None,
) -> PreparedSessionWrite:
    """Prepare the canonical pending write while its lineage evidence is pinned."""
    from polylogue.core.timestamp_authority import normalize_session_timestamps
    from polylogue.storage.sqlite.archive_tiers.revision_governance import prepare_accepted_head_reparse_receipt

    if raw_id is not None and source_read is None:
        raise PreparedSessionWriteRefusedError(
            "retained session preparation requires its original Source read capability"
        )

    normalized = normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp)
    origin = origin_from_provider(normalized.source_name)
    native_id = _stored_session_native_id(normalized.provider_session_id)
    session_id = archive_session_id(origin.value, native_id)
    context = _prepared_message_context(
        conn,
        normalized,
        origin=origin,
        session_id=session_id,
        native_id=native_id,
        merge_append=merge_append,
        signature_cache=signature_cache,
        source_read=source_read,
        child_source_path=raw_source_path(source_read, raw_id),
        before_input=before_input,
    )
    scratch: tempfile.TemporaryDirectory[str] | None = None
    prepared_union: _PreparedCrossAcquisitionUnion | None = None
    try:
        if before_input is not None:
            before_input(
                "sessions",
                ("content_hash", "raw_id", "updated_at_ms"),
                "SELECT rowid FROM sessions WHERE session_id=?",
                (session_id,),
            )
        with connection_cursor(
            conn, "SELECT content_hash, raw_id, updated_at_ms FROM sessions WHERE session_id=?", (session_id,)
        ) as cursor:
            predecessor_row = cursor.fetchone()
        predecessor = tuple(predecessor_row) if predecessor_row is not None else None
        position_offset = _next_message_position(conn, session_id) if merge_append else 0
        content_occurrence_offsets = (
            _stored_content_occurrences(conn, session_id, before_input) if merge_append and context.messages else {}
        )
        if (
            prepared_rows is not None
            and not merge_append
            and context.content_identities is None
            and len(context.messages) == len(normalized.messages)
            and prepared_rows.session_id == session_id
            and prepared_rows.session_content_hash == _prepared_session_content_hash(normalized)
        ):
            _validated_prepared_content_identities(prepared_rows, context.messages)
            rows = prepared_rows
        elif isinstance(context.messages, SqliteMessageSink) or (
            isinstance(context.messages, _MessageTail) and isinstance(context.messages.messages, SqliteMessageSink)
        ):
            source_path = cast(Path, context.messages.path)
            scratch = tempfile.TemporaryDirectory(prefix="polylogue-prepared-write-", dir=source_path.parent)
            with ExitStack() as preparation_stack:
                preparation_stack.enter_context(retain_native_sql_lifetimes(scratch))
                for duplicates in (context.event_duplicate_native_ids, context.duplicate_native_ids):
                    if isinstance(duplicates, _DiskDuplicateNativeIds):
                        preparation_stack.enter_context(duplicates.reader())
                builder = SessionShardBuilder(Path(scratch.name) / "rows.db")
                try:
                    with (
                        (
                            nullcontext(context.content_identities)
                            if context.content_identities is not None
                            else disk_message_content_identities(
                                context.messages, occurrence_offsets=content_occurrence_offsets
                            )
                        ) as identities,
                        disk_message_owner_resolution(
                            context.messages.messages
                            if isinstance(context.messages, _MessageTail)
                            else context.messages
                        ) as owners,
                    ):
                        builder.add_streamed(
                            session_id=session_id,
                            owner_resolution=owners,
                            session_content_hash=_prepared_session_content_hash(normalized),
                            message_rows=_iter_message_rows(
                                session_id,
                                context.messages,
                                position_offset=position_offset,
                                duplicate_native_ids=context.duplicate_native_ids,
                                content_identities=identities,
                            ),
                            block_rows=_iter_block_rows(
                                session_id,
                                context.messages,
                                position_offset=position_offset,
                                duplicate_native_ids=context.duplicate_native_ids,
                                content_identities=identities,
                            ),
                        )
                    shard = open_session_shard(builder.seal().path)
                except BaseException as primary:
                    try:
                        builder.abandon()
                    except BaseException as cleanup:
                        raise cleanup from primary
                    raise
            entry = shard.sessions[0]
            rows = PreparedSessionRows(
                session_id=session_id,
                session_content_hash=entry.session_content_hash,
                message_rows=_ShardRowSequence(
                    shard.path, "messages", entry.message_lo, entry.message_hi, lifetime=scratch
                ),
                block_rows=_ShardRowSequence(shard.path, "blocks", entry.block_lo, entry.block_hi, lifetime=scratch),
                content_identities=entry.content_identities,
                owner_resolution=entry.owner_resolution,
                position_offset=position_offset,
                content_occurrence_offsets=tuple(sorted(content_occurrence_offsets.items())),
                scratch=scratch,
            )
        else:
            content_identities = (
                context.content_identities
                if context.content_identities is not None
                else message_content_identities(context.messages, occurrence_offsets=content_occurrence_offsets)
            )
            rows = PreparedSessionRows(
                session_id=session_id,
                session_content_hash=_prepared_session_content_hash(normalized),
                message_rows=tuple(
                    _build_message_rows(
                        session_id,
                        context.messages,
                        position_offset=position_offset,
                        duplicate_native_ids=context.duplicate_native_ids,
                        content_identities=content_identities,
                    )
                ),
                block_rows=tuple(
                    _build_block_rows(
                        session_id,
                        context.messages,
                        position_offset=position_offset,
                        duplicate_native_ids=context.duplicate_native_ids,
                        content_identities=content_identities,
                    )
                ),
                content_identities=tuple(content_identities),
                owner_resolution=message_owner_resolution(
                    list(context.messages.messages if isinstance(context.messages, _MessageTail) else context.messages)
                ),
                position_offset=position_offset,
                content_occurrence_offsets=tuple(sorted(content_occurrence_offsets.items())),
            )
        prepared_union = None
        if raw_id is not None and not merge_append and not force_replace:
            union_source_path = getattr(context.messages, "path", None)
            directory = union_source_path.parent if isinstance(union_source_path, Path) else Path(tempfile.gettempdir())
            prepared_union = _prepare_cross_acquisition_union(
                conn,
                session_id,
                rows,
                raw_id=raw_id,
                directory=directory,
                before_input=before_input,
            )
        return PreparedSessionWrite(
            session_id=session_id,
            input_content_hash=rows.session_content_hash,
            merge_append=merge_append,
            fallback_timestamp=fallback_timestamp,
            context=context,
            rows=rows,
            cross_acquisition_union=prepared_union,
            predecessor=predecessor,
            reparse_receipt=(
                prepare_accepted_head_reparse_receipt(
                    conn,
                    source_read,
                    raw_id=raw_id,
                    session_id=session_id,
                    content_hash=rows.session_content_hash,
                    before_input=before_input,
                )
                if raw_id is not None and source_read is not None
                else None
            ),
            enrichment_binding=(
                (
                    normalized.enrichment_evidence_key,
                    source_read.session_enrichment_key(
                        conn,
                        provider=normalized.source_name,
                        source_path=context.child_source_path,
                        native_id=native_id,
                    ),
                )
                if raw_id is not None and source_read is not None
                else None
            ),
        )
    except BaseException as primary:
        failures: list[BaseException] = []
        if prepared_union is not None and prepared_union.carry_forward.scratch is not None:
            try:
                prepared_union.carry_forward.scratch.close()
            except BaseException as failure:
                failures.append(failure)
        if scratch is not None:
            try:
                pending = retained_native_sql_owners_for_lifetime(scratch)
                if pending:
                    raise NativeConnectionSettlementError(
                        pending[0], RuntimeError("prepared row construction remains owned")
                    )
                scratch.cleanup()
            except BaseException as failure:
                failures.append(failure)
        try:
            context.close()
        except BaseException as failure:
            failures.append(failure)
        if failures:
            raise PreparedSessionSettlementError(failures) from primary
        raise


def prepared_session_rows_from_shard(shard_path: Path, session_id: str) -> PreparedSessionRows:
    """Expose a sealed artifact's existing rows to read-only preparation."""
    shard = open_session_shard(shard_path)
    try:
        entry = shard.by_session_id()[session_id]
    except KeyError as exc:
        raise PreparedSessionWriteRefusedError("sealed shard lacks one exact session row range") from exc
    return PreparedSessionRows(
        session_id=entry.session_id,
        session_content_hash=entry.session_content_hash,
        message_rows=_ShardRowSequence(shard.path, "messages", entry.message_lo, entry.message_hi),
        block_rows=_ShardRowSequence(shard.path, "blocks", entry.block_lo, entry.block_hi),
        content_identities=entry.content_identities,
        owner_resolution=entry.owner_resolution,
    )


def prepare_session_rows(
    session: ParsedSession,
    *,
    position_offset: int = 0,
    content_occurrence_offsets: Mapping[str, int] | None = None,
) -> PreparedSessionRows:
    """Build ``PreparedSessionRows`` for ``session``'s full-replace write.

    Normalizes messages exactly as ``write_parsed_session_to_archive`` does for a non-merge-append, non-lineage-sliced write (see
    ``normalize_active_branch``), then reuses the same row-tuple builders the
    writer itself calls (``_build_message_rows``/``_build_block_rows``) at
    ``position_offset=0`` -- the offset every full-replace write uses. A
    byte-replay preparation can supply its pinned append offset, so the
    admitted writer need not rebuild per-message/block tuples for a composed
    tail. List-backed input stays in memory; sealed message input uses
    creator-owned indexed scratch during preparation. No native handle is
    carried into publication.
    """
    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    messages = _derive_tool_outcomes(normalize_active_branch(session.messages), session.session_events, origin=origin)
    duplicate_native_ids = _duplicate_message_native_ids(messages)
    try:
        with (
            duplicate_native_ids.reader()
            if isinstance(duplicate_native_ids, _DiskDuplicateNativeIds)
            else nullcontext()
        ):
            content_identities = message_content_identities(messages, occurrence_offsets=content_occurrence_offsets)
            message_rows = _build_message_rows(
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
                content_identities=content_identities,
            )
            block_rows = _build_block_rows(
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
                content_identities=content_identities,
            )
            result = PreparedSessionRows(
                session_id=session_id,
                session_content_hash=_prepared_session_content_hash(session),
                message_rows=tuple(message_rows),
                block_rows=tuple(block_rows),
                content_identities=tuple(content_identities),
                owner_resolution=message_owner_resolution(list(messages)),
                position_offset=position_offset,
                content_occurrence_offsets=tuple(sorted((content_occurrence_offsets or {}).items())),
            )

    finally:
        if isinstance(duplicate_native_ids, _DiskDuplicateNativeIds):
            duplicate_native_ids.close()
    return result


def _prepared_session_content_hash(session: ParsedSession) -> bytes:
    """Use the parse-bound digest, falling back for direct pure callers."""
    bound = bound_session_content_hash(session)
    if bound is not None:
        return bytes.fromhex(bound)
    from polylogue.pipeline.ids import session_content_hash

    return bytes.fromhex(session_content_hash(session))


def prepare_session_shard(directory: Path, sessions: Sequence[ParsedSession]) -> SessionShard:
    """Build one sealed shard holding every session in ``sessions``.

    The stage-A half of the shard transport: pure computation over parsed
    sessions plus one scratch file under ``directory``. No archive
    connection, so it runs on a parse worker well outside any writer hold.
    """
    if not any(isinstance(session.messages, SqliteMessageSink) for session in sessions):
        return build_session_shard(directory, tuple(prepare_session_rows(session) for session in sessions))
    directory.mkdir(parents=True, exist_ok=True)
    builder = SessionShardBuilder(directory / f"shard-{uuid.uuid4().hex}.db")
    try:
        for session in sessions:
            append_session_to_shard(builder, session)
        return open_session_shard(builder.seal().path)
    except BaseException as primary:
        try:
            builder.abandon()
        except BaseException as cleanup:
            raise cleanup from primary
        raise


def append_session_to_shard(builder: SessionShardBuilder, session: ParsedSession) -> None:
    """Append one parsed session using the same row and identity lowering as a cohort."""
    if not isinstance(session.messages, SqliteMessageSink):
        builder.add(prepare_session_rows(session))
        return
    sink_messages = session.messages
    origin = origin_from_provider(session.source_name)
    with ExitStack() as held:
        # The parsed session is complete: its walks below reuse one decode.
        held.enter_context(sink_messages.held_walks())
        messages: Sequence[ParsedMessage] = sink_messages.normalized_messages(session.session_events, origin=origin)
        if isinstance(messages, SqliteMessageSink):
            held.enter_context(messages.held_walks())
        _append_held_session_to_shard(builder, session, messages, origin=origin)


def _append_held_session_to_shard(
    builder: SessionShardBuilder, session: ParsedSession, messages: Sequence[ParsedMessage], *, origin: Origin
) -> None:
    session_id = archive_session_id(origin.value, session.provider_session_id)
    duplicates = _duplicate_message_native_ids(messages)
    try:
        with (
            disk_message_content_identities(messages) as identities,
            disk_message_owner_resolution(messages) as owners,
            # Every message asks whether its native id is ambiguous; one
            # reader serves the whole append instead of one per question.
            duplicates.reader() if isinstance(duplicates, _DiskDuplicateNativeIds) else nullcontext(),
        ):
            builder.add_streamed(
                session_id=session_id,
                session_content_hash=_prepared_session_content_hash(session),
                owner_resolution=owners,
                message_rows=_iter_message_rows(
                    session_id,
                    messages,
                    duplicate_native_ids=duplicates,
                    content_identities=identities,
                ),
                block_rows=_iter_block_rows(
                    session_id,
                    messages,
                    duplicate_native_ids=duplicates,
                    content_identities=identities,
                ),
            )
    finally:
        if isinstance(duplicates, _DiskDuplicateNativeIds):
            duplicates.close()


@contextmanager
def _index_write_scope(
    conn: sqlite3.Connection,
    *,
    archive_root: Path | None,
    mutation_scope: IndexMutationScope | None,
    manage_transaction: bool,
) -> Iterator[IndexMutationScope]:
    """Admit both stale-observation and content writes through one owner seam."""
    if mutation_scope is not None:
        mutation_scope.require_new_work(conn)
        if not conn.in_transaction:
            raise ReferenceSealError("a borrowed Index scope requires its active transaction")
        yield mutation_scope
        return
    if archive_root is None:
        raise ReferenceSealError("Index writes require their matching transaction scope or active archive root")
    if not manage_transaction or conn.in_transaction:
        raise ReferenceSealError("an outer Index transaction requires its explicit matching scope")
    with (
        PreparedIndexMutation(index_path_for_connection(conn), archive_root=archive_root) as seal,
        seal.mutation_scope(conn) as scope,
    ):
        yield scope


def write_parsed_session_to_archive(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
    pending_input_content_hash: str | None = None,
    raw_id: str | None = None,
    fallback_timestamp: str | None = None,
    merge_append: bool = False,
    force_replace: bool = False,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    signature_cache: _SignatureCacheLike | None = None,
    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None,
    sidecar_blob_locators: Mapping[str, Mapping[str, str]] | None = None,
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    fresh_build: bool = False,
    fresh_build_batch: set[str] | None = None,
    defer_fts_rebuild: bool = False,
    prepared_write: PreparedSessionWrite,
    source_read: SessionSourceRead | None = None,
    child_source_path: str | None = None,
    write_outcome: list[ArchiveWriteOutcome] | None = None,
    unit_accounting: ParseAccounting | None = None,
    archive_root: Path | None = None,
    mutation_scope: IndexMutationScope | None = None,
) -> str:
    """Write one parsed session into an initialized archive index DB.

    ``source_read`` (optional) is the durable ``source.db`` handle. It is used
    only to consult acquired hook evidence when writing this session's
    topology edge; passing ``None`` leaves that evidence unconsulted.

    ``child_source_path`` is the retained source path of ``raw_id``
    (:func:`raw_source_path`), which places a Codex child in the install root
    whose projected spawn parent it reads; ``None`` leaves the root unknown.
    A ``prepared_write`` carries the path it was prepared with instead.

    ``prepared_write`` carries the canonical off-writer lowering. Publication
    validates its captured predecessor, lineage and append frontier before
    changing rows; changed evidence requires new preparation.

    ``pending_input_content_hash`` (polylogue-3hfl7, default ``None``) names
    the digest of the rows THIS CALL publishes, for the callers where that is
    not the digest stored on ``sessions.content_hash``. Exactly one caller
    needs the distinction: an append passes the DELTA in ``session`` while
    ``content_hash`` must remain the MERGED session's digest, because that is
    the value a later re-ingest compares its own full-session hash against to
    decide the content is unchanged. Admitting a ``prepared`` carrier against
    the merged digest would admit rows covering a different message count and
    then refuse inside ``_validated_prepared_content_identities`` -- a hard
    failure, not a slow path. Default ``None`` means "the two coincide" and
    leaves every non-append caller exactly as before.

    By default the whole write runs in its own transaction (``with conn:``)
    committed on success. A bulk caller that wants many sessions in one
    transaction — to amortize the per-commit fsync and WAL page churn that
    dominate re-ingest I/O — passes ``manage_transaction=False`` and owns the
    surrounding commit and any rollback-on-error itself.

    ``bulk_fts`` (polylogue-crd8, default ``False`` so ordinary daemon ingest
    is byte-for-byte unchanged) enables the guard-gated bulk FTS mode for the
    cascading prefix-tail re-extraction this write can trigger on *other*
    (child) sessions via ``_resolve_session_graph`` -- see
    ``_bulk_fts_session_guard``. Only the offline rebuild/backfill replay path
    turns this on.

    ``bulk_build`` (polylogue-v6i3, default ``False``) is the broader
    bulk-generation-build lifecycle this session write may be part of. FTS
    surfaces remain explicitly rebuilt by the offline path; action and
    delegation relations are query-time views over canonical rows.

    ``defer_fts_rebuild`` is for an authoritative raw-revision replay that
    owns one targeted repair and exactness proof after its writes. It avoids
    rebuilding the same session's FTS surfaces twice in the same transaction.
    Direct callers retain the immediate-ready default.

    ``fresh_build`` is restricted to a from-empty index generation.  It keeps
    identity and content hashing, but skips compare/replace preparation after
    proving that this session id has not been written in the generation.  A
    repeated id is an assertion failure rather than an implicit duplicate or
    overwrite; live ingest never enables this mode.

    ``sidecar_blob_locators`` maps a sidecar event's ``tool_use_id`` to the
    publication metadata stored beside it (see ``_stored_event_payload``).
    It is outside the session object, so the session written is the one
    whose identity was bound, on the prepared and unprepared routes alike.
    """
    # A work-event raw carries one event and no session header. Writing it as
    # an ordinary session would upsert default header values over the stored
    # session (and, on a same-raw full replay, replace its transcript), so it
    # is an event-only append that keeps every session-owned field, including
    # the transcript's ``raw_id`` and ``content_hash``: the accepted revision
    # head and later re-ingest compare against those, and an annotation does
    # not change which raw authored the session. The rule keys on the
    # retained raw identity, so every route replays an event the same way.
    if mutation_scope is not None:
        mutation_scope.require_new_work(conn)
    elif archive_root is None:
        raise ReferenceSealError("Index writes require their matching transaction scope or active archive root")
    elif not manage_transaction or conn.in_transaction:
        raise ReferenceSealError("an outer Index transaction requires its explicit matching scope")
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes

    dependencies = () if prepared_write.rows.scratch is None else (prepared_write.rows.scratch,)
    with retain_native_sql_lifetimes(*dependencies):
        stored_header = (
            _stored_session_header(
                conn,
                archive_session_id(
                    origin_from_provider(session.source_name).value,
                    _stored_session_native_id(session.provider_session_id),
                ),
            )
            if is_work_event_raw_id(raw_id) and not session.messages and not session.attachments
            else None
        )
        event_only = stored_header is not None
        if event_only:
            # The session already exists in this generation, so even a cold build
            # appends to it rather than asserting a fresh, absent session.
            merge_append = True
            force_replace = False
            fresh_build = False
        if fresh_build and (merge_append or force_replace):
            raise ValueError("fresh_build is only valid for an untouched full-replace session")
        t0 = time.perf_counter()

        admission = unit_accounting or session.unit_accounting
        if admission is not None:
            try:
                admission.assert_conserved()
            except ValueError as exc:
                raise ValueError(f"parse admission conservation refused: {exc}") from exc

        def add_timing(name: str, started_at: float) -> None:
            _add_stage_timing(
                stage_timings_s,
                stage_timing_prefix=stage_timing_prefix,
                name=name,
                started_at=started_at,
            )

        conn.execute("PRAGMA foreign_keys = ON")
        origin = origin_from_provider(session.source_name)
        native_id = _stored_session_native_id(session.provider_session_id)
        session_id = archive_session_id(origin.value, native_id)
        # Durable non-resurrection, checked once for every write route.
        # An identity-preserving reset keeps the raw evidence in source.db and
        # records the operator's deletion as a suppression assertion in user.db,
        # so a later replay/rebuild of that retained raw row would otherwise
        # recreate the session the operator deleted. This is the shared choke
        # point for live ingest and full replay/reindex, so the refusal lives
        # here rather than in each replay caller. It is counted and logged --
        # never a silent drop (see ``session_suppression``).
        if session_write_is_suppressed(conn, session_id):
            record_suppression_refusal(session_id, route="write_parsed_session_to_archive")
            if write_outcome is not None:
                write_outcome.append(ArchiveWriteOutcome(session_id=session_id, wrote=False, suppression_skipped=True))
            return session_id
        parser_semantic_fingerprint = parser_fingerprint_for_origin(origin)
        lowering_semantic_fingerprint = lowering_fingerprint()
        # This session's own rows are about to be rewritten; drop any stale memoized
        # own-signatures so the batch cache never serves pre-write rows for it.
        if signature_cache is not None:
            signature_cache.pop(session_id, None)
        # polylogue-m3p9: providers that carry no session-level created_at/updated_at
        # (Codex, many Claude Code sessions, ...) previously left
        # sessions.created_at_ms/updated_at_ms permanently NULL for 79% of the live
        # archive, silently excluding those sessions from `since:` filters, recency
        # ordering, and --by year/month histograms. Fall back to message evidence
        # (min/max message ``occurred_at_ms``) computed over THIS write's full
        # parsed message set, i.e. before any prefix-tail slicing below: a
        # prefix-sharing child's derived created_at_ms should reflect the whole
        # conversation's start, not just its divergent tail. The derived max is
        # correct either way -- the newest message always survives slicing into
        # the tail. Provider-supplied session timestamps always win; this is
        # fallback only, applied identically on merge-append (where ``messages``
        # is just the newly appended tail, so the derived max naturally advances
        # updated_at_ms with each append and the ON CONFLICT COALESCE below keeps
        # the already-set created_at_ms untouched).
        # Keep the writer's effective freshness identical to ingest/replay
        # normalization: producer fields, then authored messages, then semantic
        # session events, then explicitly supplied acquisition evidence. Never use
        # the ingest wall clock as a substitute for missing source evidence.
        session_created_at_ms, session_updated_at_ms = session_evidence_timestamps(
            session,
            fallback_timestamp=fallback_timestamp,
        )
        producer_created, producer_updated = producer_timestamp_flags(session)
        # incoming_freshness_ms now reflects the same fallback: previously a
        # provider that omitted both session timestamps produced
        # incoming_freshness_ms=None, which unconditionally bypassed the
        # skip-stale-replace check below (freshness "unknown"). With derivation,
        # these sessions get a real freshness signal from their own message
        # evidence, so a genuinely older/stale replay of such a session is now
        # correctly skipped instead of always winning.
        incoming_freshness_ms = session_updated_at_ms or session_created_at_ms
        if not fresh_build and not force_replace and not merge_append and incoming_freshness_ms is not None:
            row = conn.execute(
                "SELECT updated_at_ms FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            existing_updated_at_ms = int(row[0]) if row is not None and row[0] is not None else None
            if should_skip_stale_replace(
                incoming_freshness_ms=incoming_freshness_ms,
                existing_updated_at_ms=existing_updated_at_ms,
            ):
                # The stale path returns before the normal write transaction below;
                # own a short transaction here so direct callers cannot lose repairs.
                with _index_write_scope(
                    conn,
                    archive_root=archive_root,
                    mutation_scope=mutation_scope,
                    manage_transaction=manage_transaction,
                ) as scope:
                    scope.note_lineage_change(session_id)
                    _retain_stale_session_observations(conn, session_id, session)
                add_timing("index.skip_stale_replace", t0)
                if write_outcome is not None:
                    write_outcome.append(ArchiveWriteOutcome(session_id=session_id, wrote=False, stale_skipped=True))
                return session_id
        bound_hash = bound_session_content_hash(session)
        input_content_hash = (
            bytes.fromhex(content_hash)
            if content_hash is not None
            else (
                bytes.fromhex(bound_hash) if bound_hash is not None else _hash_bytes("session", origin.value, native_id)
            )
        )
        # polylogue-3hfl7: ``input_content_hash`` is the digest STORED on
        # ``sessions.content_hash`` -- for an append that is the merged session,
        # not the delta in ``session``. ``pending_content_hash`` is the digest of
        # the rows this call actually publishes, and it is the only one a
        # ``prepared`` carrier may be admitted against. They coincide for every
        # caller that does not pass ``pending_input_content_hash``.
        pending_content_hash = (
            bytes.fromhex(pending_input_content_hash) if pending_input_content_hash is not None else input_content_hash
        )
        if prepared_write is None:
            raise PreparedSessionWriteRefusedError("session publication requires its canonical off-writer preparation")
        if (
            prepared_write.session_id != session_id
            or prepared_write.input_content_hash != pending_content_hash
            or prepared_write.merge_append != merge_append
            or prepared_write.rows.session_content_hash != pending_content_hash
        ):
            raise PreparedSessionWriteRefusedError("prepared write is stale or has a different pending input")
        predecessor_row = conn.execute(
            "SELECT content_hash, raw_id, updated_at_ms FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        predecessor = tuple(predecessor_row) if predecessor_row is not None else None
        if predecessor != prepared_write.predecessor:
            raise PreparedSessionWriteRefusedError("session predecessor changed after off-writer preparation")
        context = prepared_write.context
        input_session = session
        session = context.effective_session
        messages = context.messages
        event_duplicate_message_native_ids = context.event_duplicate_native_ids
        duplicate_message_native_ids = context.duplicate_native_ids
        effective_session_kind = context.effective_session_kind
        branch_point_message_id = context.branch_point_message_id
        branch_point_content_address = context.branch_point_content_address
        lineage_inheritance = context.lineage_inheritance
        inherited_source_message_ids = context.inherited_source_message_ids
        # The value published to ``sessions.content_hash``. A later re-ingest
        # compares its own FULL-session digest against this row, so an append
        # must store the merged digest even though it writes only the delta
        # (polylogue-3hfl7) -- carrier admission uses ``pending_content_hash``.
        session_content_hash = input_content_hash
        prepared_rows_to_use = prepared_write.rows
        _record_prepared_disposition("prepared_write")
        content_identities = _validated_prepared_content_identities(prepared_rows_to_use, messages)
        active_leaf_message_id = _active_leaf_message_id(
            session_id,
            messages,
            session.active_leaf_message_provider_id,
            duplicate_native_ids=duplicate_message_native_ids,
            content_identities=content_identities,
        )
        add_timing("index.prepare", t0)
        # When the caller owns the transaction (bulk batching) we must not commit
        # per session; nullcontext leaves BEGIN/COMMIT to the caller.
        invalidated_identity_children: set[str] = set()
        prefix_guard: _InheritedPrefixGuard | None = None
        try:
            with ExitStack() as mutation_stack:
                for duplicates in (context.event_duplicate_native_ids, context.duplicate_native_ids):
                    if isinstance(duplicates, _DiskDuplicateNativeIds):
                        mutation_stack.enter_context(duplicates.reader())
                if isinstance(context.inherited_source_message_ids, _DiskSourceMessageIds):
                    mutation_stack.enter_context(context.inherited_source_message_ids.reader())
                if isinstance(context.inherited_prefix_message_ids, _PrefixMessageIds):
                    composed = context.inherited_prefix_message_ids._composed
                    if isinstance(composed, _DiskSignatureSequence):
                        mutation_stack.enter_context(composed.reader())
                mutation_stack.enter_context(shard_owner_reader_window(prepared_write.rows.owner_resolution))
                mutation_stack.enter_context(
                    _index_write_scope(
                        conn,
                        archive_root=archive_root,
                        mutation_scope=mutation_scope,
                        manage_transaction=manage_transaction,
                    )
                )
                prepared_union = prepared_write.cross_acquisition_union
                if prepared_union is not None:
                    if prepared_union.carry_forward.scratch is not None:
                        mutation_stack.enter_context(prepared_union.carry_forward.scratch.access(write=True))
                    current_predecessor = conn.execute(
                        "SELECT raw_id, content_hash, parser_fingerprint, lowering_fingerprint, "
                        "parent_session_id, active_leaf_message_id FROM sessions WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if (
                        raw_id is None
                        or current_predecessor is None
                        or tuple(current_predecessor) != prepared_union.predecessor
                        or current_predecessor[0] == raw_id
                    ):
                        raise PreparedSessionWriteRefusedError("prepared field union predecessor changed")
                    current_parent_guard = (
                        conn.execute(
                            "SELECT 1 FROM session_links WHERE resolved_dst_session_id = ? "
                            "AND inheritance = 'prefix-sharing' "
                            f"AND {topology_status_composes_sql()} LIMIT 1",
                            (session_id,),
                        ).fetchone()
                        is not None
                    )
                    if current_parent_guard != prepared_union.prefix_sharing_parent:
                        raise PreparedSessionWriteRefusedError("prepared field union branch membership changed")
                validate_prepared_session_lineage(conn, input_session, prepared_write, source_read=source_read)
                conn.execute("INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES ('session-write')")
                if bulk_build:
                    # polylogue-v6i3: gate the messages_fts trigger
                    # BODIES for this session's *entire* write (block inserts in the
                    # ordinary merge/full-replace paths, not just the prefix-tail
                    # reextract cascade -- see _bulk_fts_session_guard, which detects
                    # this outer guard and becomes a no-op rather than double-managing
                    # the same row). Cleared alongside the 'session-write' guard below.
                    conn.execute(
                        "INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES (?)",
                        (FTS_BULK_SESSION_WRITE_GUARD,),
                    )
                # polylogue-geop: capture whichever raw acquisition is CURRENTLY
                # stored before the upsert below overwrites sessions.raw_id with
                # this write's own value -- _union_with_existing_rows needs the
                # PRIOR raw_id to tell "same acquisition re-parsed" (replace)
                # from "different acquisition" (union) apart. Reading it after
                # the upsert would always see this write's own raw_id and could
                # never observe a difference.
                existing_session_raw_id: str | None = None
                # A session row can exist with a NULL ``raw_id`` (ambiguous with
                # "no row at all" for the union-precedence check), but existence
                # of the row itself is unambiguous from ``fetchone()``.  A damaged
                # derived index can instead have the inverse shape: its session
                # row was lost while its message membership remains.  Capture both
                # facts before the upsert, so the full replacement clears only this
                # session's retained membership rather than colliding with it.
                session_row_existed = False
                session_membership_existed = False
                if not merge_append and not fresh_build:
                    existing_raw_id_row = conn.execute(
                        "SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)
                    ).fetchone()
                    if existing_raw_id_row is not None:
                        session_row_existed = True
                        existing_session_raw_id = existing_raw_id_row[0]
                    session_membership_existed = session_row_existed or (
                        conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
                        is not None
                    )
                elif fresh_build and not merge_append:
                    # Fresh mode is a correctness contract, not a hint.  Keep the
                    # absence check even when the caller batches transactions so a
                    # duplicate session can never silently replace rows.
                    if (
                        conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                        is not None
                    ):
                        raise AssertionError(f"fresh_build requires an absent session_id: {session_id}")
                    if (fresh_build_batch is None or not fresh_build_batch) and conn.execute(
                        "SELECT 1 FROM sessions LIMIT 1"
                    ).fetchone() is not None:
                        raise AssertionError("fresh_build requires an empty archive generation")
                    if fresh_build_batch is not None:
                        fresh_build_batch.add(session_id)
                # Whether any row this save replaces can already exist. Every
                # session-owned projection cascades from ``sessions`` or
                # ``messages``, so a session with neither (every session of a
                # from-empty build) has nothing for a replace prelude to delete.
                prior_session_rows = merge_append or session_membership_existed
                t0 = time.perf_counter()
                session_row_values = {
                    "native_id": native_id,
                    "origin": origin.value,
                    "raw_id": raw_id,
                    "parser_fingerprint": parser_semantic_fingerprint,
                    "lowering_fingerprint": lowering_semantic_fingerprint,
                    "branch_type": _enum_value(session.branch_type),
                    "active_leaf_message_id": active_leaf_message_id,
                    "title": _sqlite_text(session.title),
                    "session_kind": admitted_session_kind(
                        effective_session_kind,
                        branch_type=session.branch_type,
                    ).value,
                    "title_source": _enum_value(session.title_source),
                    "title_ref": _sqlite_text(session.title_ref),
                    "display_name": _sqlite_text(session.display_name),
                    "pending_drafts_json": _json_dumps(session.pending_drafts) if session.pending_drafts else None,
                    "git_branch": _sqlite_text(session.git_branch),
                    "git_repository_url": _sqlite_text(session.git_repository_url),
                    "commit_hash": _sqlite_text(session.git_commit_hash),
                    "instructions_text": _sqlite_text(session.instructions_text),
                    "reported_duration_ms": session.reported_duration_ms,
                    "reported_cost_usd": session.reported_cost_usd,
                    "provider_project_ref": _sqlite_text(session.provider_project_ref),
                    "content_hash": session_content_hash,
                    "created_at_ms": session_created_at_ms,
                    "updated_at_ms": session_updated_at_ms,
                    # Messages are inserted later in this transaction.  The one
                    # authoritative replacement immediately after that insert
                    # publishes the declared counter projection; fresh rows begin
                    # at the schema's zero value rather than carrying a second
                    # parser-side tally.
                    **{measure.column: 0 for measure in SESSION_SUMMARY_MEASURES},
                }
                if stored_header is not None:
                    session_row_values.update(stored_header)
                sessions_spec = archive_tiers_specs.SESSIONS_SPEC
                note_current_session_namespace_change(conn)
                note_current_lineage_change(conn, session_id)
                conn.execute(
                    f"""
                    INSERT INTO sessions (
                        {sessions_spec.insert_column_names}
                    ) VALUES ({sessions_spec.insert_placeholder_string})
                    ON CONFLICT(origin, native_id) DO UPDATE SET
                        {sessions_spec.conflict_update_sql(" " * 20)}
                    """,
                    (
                        *sessions_spec.extract_tuple(session_row_values),
                        *sessions_spec.conflict_update_tuple(
                            {
                                "producer_created": producer_created,
                                "producer_updated": producer_updated,
                                "force_replace": force_replace,
                                "producer_updated_or_merge_append": producer_updated or merge_append,
                            }
                        ),
                    ),
                )
                add_timing("index.session_upsert", t0)
                if not event_only:
                    invalidated_identity_children = _write_session_identity_claims(
                        conn, session_id, origin.value, session, prior_claims=prior_session_rows
                    )
                position_offset = 0
                stale_attachment_ids: set[str] = set()
                projection_carry_forward: _ProjectionCarryForward | None = None
                append_has_new_messages = False
                t0 = time.perf_counter()
                if merge_append:
                    position_offset = _next_message_position(conn, session_id)
                    _assert_unique_message_coordinates(session_id, messages, position_offset=position_offset)
                    append_has_new_messages = _append_has_new_native_messages(
                        conn, session_id, prepared_rows_to_use.message_rows
                    )
                    if not event_only and append_has_new_messages:
                        conn.execute(
                            """
                            UPDATE messages
                            SET is_active_leaf = 0
                            WHERE session_id = ?
                              AND is_active_path = 1
                              AND is_active_leaf = 1
                            """,
                            (session_id,),
                        )
                        active_leaf_message_id = _active_leaf_message_id(
                            session_id,
                            messages,
                            session.active_leaf_message_provider_id,
                            content_identities=content_identities,
                            duplicate_native_ids=duplicate_message_native_ids,
                        )
                        conn.execute(
                            "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?",
                            (active_leaf_message_id, session_id),
                        )
                    add_timing("index.merge_prepare", t0)
                    # The append frontier has two coordinates now: the next
                    # position, and the next per-digest occurrence above the
                    # stored ordinals. Prepared rows pinned against either
                    # stale value would generate ids that collide with, or skip
                    # past, what is stored (polylogue-eqsri).
                    stored_content_occurrences = (
                        tuple(sorted(_stored_content_occurrences(conn, session_id).items())) if messages else ()
                    )
                    if (
                        prepared_write.rows.position_offset != position_offset
                        or prepared_write.rows.content_occurrence_offsets != stored_content_occurrences
                    ):
                        raise PreparedSessionWriteRefusedError(
                            "prepared replay append lowering no longer matches its pinned frontier"
                        )
                else:
                    # The preparer sealed a cross-acquisition union from its own
                    # read, but this writer's precedence decision (force replace,
                    # same acquisition, or no prior membership) is authoritative.
                    # When it says the incoming rows replace wholesale, the union
                    # is dropped here, before the attachment bookkeeping, so the
                    # replaced messages' attachments are refreshed like any other
                    # replacement. Refusing instead deferred the path and the next
                    # pass re-prepared the same union: a livelock that left an
                    # interrupted append's tail unmaterialized (b8of0).
                    applicable_union = (
                        prepared_write.cross_acquisition_union
                        if _cross_acquisition_union_applies(
                            session_membership_existed=session_membership_existed,
                            force_replace=force_replace,
                            raw_id=raw_id,
                            existing_raw_id=existing_session_raw_id,
                        )
                        else None
                    )
                    stale_attachment_ids = (
                        set() if applicable_union is not None else session_attachment_ids(conn, session_id)
                    )
                    prefix_guard = _capture_inherited_prefixes(conn, session_id) if session_membership_existed else None
                    projection_carry_forward = _replace_full_session_messages_and_blocks(
                        conn,
                        session,
                        messages,
                        duplicate_native_ids=duplicate_message_native_ids,
                        raw_id=raw_id,
                        existing_raw_id=existing_session_raw_id,
                        session_membership_existed=session_membership_existed,
                        force_replace=force_replace,
                        stage_timings_s=stage_timings_s,
                        stage_timing_prefix=stage_timing_prefix,
                        bulk_build=bulk_build,
                        defer_fts_rebuild=defer_fts_rebuild,
                        prepared=prepared_rows_to_use,
                        prepared_union=applicable_union,
                        content_identities=content_identities,
                    )
                    add_timing("index.full_replace", t0)
                if merge_append:
                    t0 = time.perf_counter()
                    _write_append_messages_and_blocks(
                        conn,
                        session_id,
                        messages,
                        message_rows=prepared_rows_to_use.message_rows,
                        block_rows=prepared_rows_to_use.block_rows,
                    )
                    add_timing("index.messages_and_blocks", t0)
                    t0 = time.perf_counter()
                    _write_file_edits(
                        conn,
                        session_id,
                        messages,
                        position_offset=position_offset,
                        duplicate_native_ids=duplicate_message_native_ids,
                        content_identities=content_identities,
                        message_id_remap=projection_carry_forward.incoming_message_id_remap
                        if projection_carry_forward is not None
                        else None,
                    )
                    add_timing("index.file_edits", t0)
                    t0 = time.perf_counter()
                    _write_web_constructs(
                        conn,
                        session,
                        messages,
                        position_offset=position_offset,
                        duplicate_native_ids=duplicate_message_native_ids,
                        replace_session=False,
                        content_identities=content_identities,
                        message_id_remap=projection_carry_forward.incoming_message_id_remap
                        if projection_carry_forward is not None
                        else None,
                    )
                    add_timing("index.web_constructs", t0)
                t0 = time.perf_counter()
                # polylogue-geop: an attachment_id that projection carry-forward
                # is about to restore an attachment_refs row for must NOT be
                # swept here just because it looks unreferenced right now --
                # its old attachment_refs row was already cascade-deleted by the
                # full-replace's message DELETE and the replacement hasn't been
                # (re)inserted yet (_restore_captured_projection_rows runs after
                # this call). Passing it through refresh_attachment_ids would
                # zero its ref_count and delete the attachments row outright,
                # so the later restore's FK to attachments(attachment_id) fails.
                carried_forward_attachment_ids = (
                    {
                        cast(str, row[0])
                        for row in projection_carry_forward.captured.attachment_refs
                        if row[2] in projection_carry_forward.live_message_ids
                    }
                    if projection_carry_forward is not None and projection_carry_forward.scratch is None
                    else set()
                )
                refresh_attachment_ids: Iterable[str] = stale_attachment_ids - carried_forward_attachment_ids
                if projection_carry_forward is not None and projection_carry_forward.scratch is not None:
                    refresh_attachment_ids = _UnionSet(
                        projection_carry_forward.scratch, "refresh_attachment", "attachment_id"
                    )
                unresolved_attachment_owners = _write_attachments(
                    conn,
                    session_id,
                    messages,
                    session.attachments,
                    supplying_raw_id=raw_id,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                    refresh_attachment_ids=refresh_attachment_ids,
                    preacquired_blobs=preacquired_attachment_blobs,
                    content_identities=content_identities,
                    message_id_remap=projection_carry_forward.incoming_message_id_remap
                    if projection_carry_forward is not None
                    else None,
                    inherited_prefix_message_ids=context.inherited_prefix_message_ids,
                    owner_resolution=prepared_write.rows.owner_resolution,
                    replace_owner_gaps=not merge_append,
                )
                add_timing("index.attachments", t0)
                t0 = time.perf_counter()
                _write_paste_spans(
                    conn,
                    session_id,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                    content_identities=content_identities,
                    message_id_remap=projection_carry_forward.incoming_message_id_remap
                    if projection_carry_forward is not None
                    else None,
                )
                add_timing("index.paste_spans", t0)
                if projection_carry_forward is not None:
                    # polylogue-geop: all four evidence-dependent projection
                    # tables (attachment_refs/paste_spans just above, file_edits/
                    # web_content_constructs inside _replace_full_session_
                    # messages_and_blocks) have now been rebuilt from the
                    # incoming ParsedSession alone -- restore any pre-delete row
                    # a reinjected/reconciled message or block owned that the
                    # rebuild didn't recreate.
                    t0 = time.perf_counter()
                    _restore_captured_projection_rows(conn, projection_carry_forward)
                    # Recount every previously retained row after native-reference
                    # restoration; a replaced payload descriptor may have lost
                    # its last reference while omitted native objects survive.
                    post_restore_attachment_ids: Iterable[str] = carried_forward_attachment_ids
                    if projection_carry_forward.scratch is not None:
                        post_restore_attachment_ids = _UnionSet(
                            projection_carry_forward.scratch, "carried_attachment", "attachment_id"
                        )
                    refresh_and_sweep_attachment_rows(conn, post_restore_attachment_ids)
                    add_timing("index.restore_projections", t0)
                t0 = time.perf_counter()
                _write_parent_links(
                    conn,
                    session_id,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                    content_identities=content_identities,
                    message_id_remap=projection_carry_forward.incoming_message_id_remap
                    if projection_carry_forward is not None
                    else None,
                    inherited_message_ids=inherited_source_message_ids,
                    stored_parents=merge_append,
                )
                add_timing("index.parent_links", t0)
                t0 = time.perf_counter()
                if _reconcile_tool_use_outcomes(conn, session_id):
                    _rehash_session_messages(conn, session_id)
                add_timing("index.tool_outcomes", t0)
                t0 = time.perf_counter()
                if not bulk_build:
                    # A first save holds no pairs: it issues no prelude delete
                    # (polylogue-ctha5).
                    refresh_action_pairs(conn, session_id, prior_rows=prior_session_rows)
                add_timing("index.action_pairs", t0)
                t0 = time.perf_counter()
                _write_session_link(
                    conn,
                    session_id,
                    session,
                    branch_point_message_id=branch_point_message_id,
                    branch_point_content_address=branch_point_content_address,
                    inheritance=lineage_inheritance,
                    source_read=source_read,
                    prior_links=prior_session_rows,
                    child_source_path=context.child_source_path,
                    preserve_prefix_loss=merge_append,
                )
                add_timing("index.session_link", t0)
                t0 = time.perf_counter()
                event_position_offset = _next_session_event_position(conn, session_id)
                session_event_result = _write_session_events(
                    conn,
                    session_id,
                    messages,
                    session.session_events,
                    position_offset=position_offset,
                    event_position_offset=event_position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                    inherited_source_message_ids=inherited_source_message_ids,
                    ambiguous_source_provider_ids=event_duplicate_message_native_ids,
                    content_identities=content_identities,
                    message_id_remap=projection_carry_forward.incoming_message_id_remap
                    if projection_carry_forward is not None
                    else None,
                    sidecar_blob_locators=sidecar_blob_locators,
                    owner_resolution=prepared_write.rows.owner_resolution,
                )
                add_timing("index.session_events", t0)
                if projection_carry_forward is not None:
                    t0 = time.perf_counter()
                    _restore_captured_provider_usage_rows(conn, projection_carry_forward)
                    add_timing("index.restore_provider_usage", t0)
                if not event_only:
                    t0 = time.perf_counter()
                    _write_working_dirs(conn, session_id, session.working_directories)
                    add_timing("index.working_dirs", t0)
                    t0 = time.perf_counter()
                    _write_session_refs(conn, session_id, session)
                    add_timing("index.session_refs", t0)
                    t0 = time.perf_counter()
                    _write_repo_edges(conn, session_id, session)
                    add_timing("index.repo_edges", t0)
                t0 = time.perf_counter()
                _seed_session_model_usage_rows(
                    conn,
                    session_id,
                    session,
                    replace_existing_model_rows=not merge_append,
                    aggregate_message_tokens=not merge_append or _messages_have_token_counts(messages),
                )
                add_timing("index.model_usage_seed", t0)
                if merge_append and session_event_result.wrote_provider_usage_events:
                    t0 = time.perf_counter()
                    if origin is Origin.CLAUDE_CODE_SESSION and _has_appended_claude_message_usage(
                        conn,
                        session_id,
                        start_position=event_position_offset,
                    ):
                        _reconcile_session_model_usage_from_persisted_evidence(conn, session_id)
                    else:
                        _aggregate_appended_provider_usage_into_model_usage(
                            conn,
                            session_id,
                            start_position=event_position_offset,
                        )
                    add_timing("index.provider_usage_rollup", t0)
                elif not merge_append:
                    t0 = time.perf_counter()
                    _aggregate_provider_usage_into_model_usage(conn, session_id)
                    add_timing("index.provider_usage_rollup", t0)
                t0 = time.perf_counter()
                # The summary is one authoritative projection of stored messages.
                # Append has no typed disjoint-insert proof, so it takes the same
                # replacement path as full writes and lineage re-extraction.
                refresh_session_summary(conn, session_id)
                add_timing("index.session_counts", t0)
                t0 = time.perf_counter()
                graph_kwargs: dict[str, Any] = {
                    "cache": signature_cache,
                    "add_timing": add_timing,
                    "bulk_fts": bulk_fts,
                    "bulk_build": bulk_build,
                }
                if invalidated_identity_children:
                    graph_kwargs["invalidated_session_ids"] = invalidated_identity_children
                if source_read is not None:
                    graph_kwargs["source_read"] = source_read
                _project_manual_continuations(conn, session_id)
                graph_changed_ids = _resolve_session_graph(conn, session_id, native_id, origin.value, **graph_kwargs)
                add_timing("index.graph_resolve", t0)
                materialized_ids: set[str] = set()
                if prefix_guard is not None:
                    t0 = time.perf_counter()
                    materialized_ids = _settle_inherited_prefixes(
                        conn, prefix_guard, cache=signature_cache, bulk_fts=bulk_fts, bulk_build=bulk_build
                    )
                    add_timing("index.inherited_prefix_guard", t0)
                t0 = time.perf_counter()
                if not bulk_build:
                    # The session-write guard suppresses the block and link
                    # triggers that would refresh these cohorts, and graph
                    # resolution and prefix settlement change other sessions' rows
                    # and edges: a late parent deletes each child's inherited
                    # prefix, a replace copies one into a child.
                    refresh_delegation_facts_for_sessions(conn, {session_id, *graph_changed_ids, *materialized_ids})
                add_timing("index.delegation_facts", t0)
                conn.execute("DELETE FROM derived_refresh_guard WHERE guard_name = 'session-write'")
                if bulk_build:
                    conn.execute(
                        "DELETE FROM derived_refresh_guard WHERE guard_name = ?",
                        (FTS_BULK_SESSION_WRITE_GUARD,),
                    )
                if merge_append and session.ingest_flags:
                    t0 = time.perf_counter()
                    _write_ingest_flag_tags(conn, session_id, session.ingest_flags)
                    add_timing("index.ingest_flags", t0)
                elif not merge_append:
                    t0 = time.perf_counter()
                    if prior_session_rows:
                        _replace_ingest_flag_tags(conn, session_id, session.ingest_flags)
                    else:
                        _write_ingest_flag_tags(conn, session_id, session.ingest_flags)
                    add_timing("index.ingest_flags", t0)
        except sqlite3.IntegrityError as exc:
            raise sqlite3.IntegrityError(
                f"FOREIGN KEY constraint failed writing session_id={session_id!r} "
                f"origin={origin.value!r} native_id={native_id!r}: {exc}"
            ) from exc
        # The lineage columns of every child invalidated above are now NULL, so the
        # child reads as a complete root while its recomposed prefix is gone. The
        # loss is named as ordinary retryable convergence debt (ops tier) rather
        # than left silent until someone orders a full rebuild (polylogue-e0xan).
        _record_identity_invalidation_debt(conn, invalidated_identity_children)
        if write_outcome is not None:
            write_outcome.append(
                ArchiveWriteOutcome(
                    session_id=session_id,
                    wrote=(not merge_append or append_has_new_messages or _append_has_auxiliary_input(session)),
                    unresolved_attachment_owners=unresolved_attachment_owners,
                )
            )
        return session_id


def _add_stage_timing(
    stage_timings_s: dict[str, float] | None,
    *,
    stage_timing_prefix: str,
    name: str,
    started_at: float,
) -> None:
    if stage_timings_s is None:
        return
    key = f"{stage_timing_prefix}.{name}"
    stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)


def _write_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Write parser-level ingest flags as auto-tags in the same transaction.

    Each flag is lowercased and written as ``(session_id, flag, 'auto')`` with
    ``method='parser'``.  Duplicate flags on re-ingest are silently skipped
    (``ON CONFLICT DO NOTHING``) so repeated ingest of the same session is
    idempotent.  Called from inside the ``with conn:`` block of
    ``write_parsed_session_to_archive`` so the tag rows are committed atomically
    with the session row they reference.
    """
    for raw_flag in flags:
        normalized = raw_flag.strip().lower()
        if not normalized:
            continue
        conn.execute(
            """
            INSERT INTO session_tags (session_id, tag, tag_source, method)
            VALUES (?, ?, 'auto', 'parser')
            ON CONFLICT(session_id, tag, tag_source) DO NOTHING
            """,
            (session_id, normalized),
        )


def _replace_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Synchronize parser-owned flags to the accepted full replacement."""
    conn.execute(
        "DELETE FROM session_tags WHERE session_id = ? AND tag_source = 'auto' AND method = 'parser'",
        (session_id,),
    )
    _write_ingest_flag_tags(conn, session_id, flags)


def upsert_parser_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Upsert parser-owned ingest flag tags for an already-materialized session."""
    _write_ingest_flag_tags(conn, session_id, flags)


def replace_parser_ingest_flag_tags(conn: sqlite3.Connection, session_id: str, flags: list[str]) -> None:
    """Replace parser-owned ingest flags for an accepted current owner."""
    _replace_ingest_flag_tags(conn, session_id, flags)


def _clear_session_projection_rows(conn: sqlite3.Connection, session_id: str) -> None:
    """Clear rows owned by parsed-session replacement before rewriting it."""
    conn.execute(
        """
        UPDATE messages
        SET parent_message_id = NULL
        WHERE parent_message_id IN (
            SELECT message_id FROM messages WHERE session_id = ?
        )
        """,
        (session_id,),
    )
    _purge_session_message_fts_when_delete_trigger_missing(conn, session_id)
    # Bulk rebuilds disable foreign keys, so mirror the refs' child cascade
    # while their session membership can still identify the dependent rows.
    conn.execute(
        "DELETE FROM attachment_native_ids WHERE ref_id IN (SELECT ref_id FROM attachment_refs WHERE session_id = ?)",
        (session_id,),
    )
    for table in (
        "blocks",
        "attachment_refs",
        "paste_spans",
        "session_provider_usage_events",
        "session_agent_policies",
        "session_working_dirs",
        "session_repos",
        "session_commits",
        "session_model_usage",
        "session_refs",
    ):
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    # capture_gap rows are archive-generated ingest evidence, not projections
    # owned by whichever parser payload currently wins source precedence.
    conn.execute(
        "DELETE FROM session_events WHERE session_id = ? AND event_type != 'capture_gap'",
        (session_id,),
    )
    # Hook-derived edges survive a full replace for the same reason capture_gap
    # events just did: they are acquired durable evidence (polylogue-foee's
    # codex_thread_spawn_edge spool), not projections owned by whichever parser
    # payload currently wins source precedence. Without this exemption a plain
    # re-parse DELETEs the authoritative edge and, when the caller has no
    # source handle to re-derive it from, silently reinstates the inferred
    # parent -- the "later inference overwrites the authoritative result" hole
    # in its most destructive form. The contradicted marker is preserved with
    # it so composition stays deterministic rather than falling back to an
    # observed_at_ms race between two unqualified edges.
    conn.execute(
        """
        DELETE FROM session_links
        WHERE src_session_id = ? AND COALESCE(method, '') NOT IN (?, ?)
        """,
        (session_id, HOOK_AUTHORITATIVE_LINK_METHOD, HOOK_CONTRADICTED_LINK_METHOD),
    )


def _purge_session_message_fts_when_delete_trigger_missing(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete current session FTS rows before block deletion when triggers are suspended."""
    trigger_row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'messages_fts_ad'",
    ).fetchone()
    if trigger_row is not None:
        return
    table_rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'virtual table')
          AND name IN ('messages_fts', 'messages_fts_docsize')
        """,
    ).fetchall()
    if {str(row[0]) for row in table_rows} != {"messages_fts", "messages_fts_docsize"}:
        return
    from polylogue.storage.fts.sql import (
        delete_session_identity_rows_sql,
        delete_session_rows_sql,
    )

    conn.execute(delete_session_rows_sql(1), (session_id,))
    # polylogue-miwv: pair the messages_fts delete with its identity-ledger
    # companion -- the blocks this session owns are about to be deleted too
    # (see the caller, ``_clear_session_projection_rows``), so an unpaired
    # delete here would leave orphaned ``messages_fts_identity`` rows (the
    # "left-over ledger row" failure class ``message_identity_mismatch_sql``
    # detects).
    conn.execute(delete_session_identity_rows_sql(1), (session_id,))


@dataclass(frozen=True, slots=True)
class _TranscriptSegment:
    """One contiguous run of a single session's own rows in a composed transcript.

    ``upto_position``/``upto_variant_index`` bound the run at an inherited
    branch point; ``None`` means the session contributes every row it owns.
    ``message_count`` is that run's length, counted in SQL rather than by
    materializing the rows -- which is the whole point of planning a
    composition before fetching it.
    """

    session_id: str
    upto_position: int | None
    upto_variant_index: int | None
    message_count: int


@dataclass(frozen=True, slots=True)
class _ComposedTranscriptPlan:
    """The segment layout of a composed transcript, decided without reading rows.

    #2467 stores a prefix-sharing child as its divergent tail and recomposes
    the ancestral prefix on read. This is that recomposition decided once, in
    SQL work proportional to the chain depth, so a caller can either
    materialize every segment (``read_archive_session_envelope``) or fetch
    exactly one ``[offset, offset + limit)`` window across the segments
    (``read_archive_session_page``) from the same composition rules.

    Two materializers over one plan is also what keeps a page read and a full
    read from ever disagreeing about the transcript they describe: the
    ordering, the branch-point cut and the truncation verdict are computed in
    one place, not re-derived per surface.
    """

    segments: tuple[_TranscriptSegment, ...]
    total_message_count: int
    lineage_complete: bool
    lineage_truncation_reason: LineageTruncationReason | None
    lineage_inheritance: str
    lineage_branch_point_message_id: str | None


def _count_session_messages(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    upto_position: int | None = None,
    upto_variant_index: int | None = None,
) -> int:
    """Count a session's own message rows, optionally through a branch point."""
    upto_clause = ""
    upto_params: tuple[int, int] | tuple[()] = ()
    if upto_position is not None and upto_variant_index is not None:
        upto_clause = " AND (position, variant_index) <= (?, ?)"
        upto_params = (upto_position, upto_variant_index)
    with connection_cursor(
        conn, f"SELECT COUNT(*) FROM messages WHERE session_id = ?{upto_clause}", (session_id, *upto_params)
    ) as cursor:
        row = cursor.fetchone()
    return int(row[0])


BeforeIndexInput = Callable[[str, tuple[str, ...], str, tuple[object, ...]], None]


def _message_coordinates(
    conn: sqlite3.Connection, message_id: str, before_input: BeforeIndexInput | None = None
) -> tuple[str, int, int] | None:
    """Return ``(session_id, position, variant_index)`` for one stored message.

    The owning session is part of the answer because a composed transcript
    holds rows from several sessions: both the branch-point cut and the
    composed-index locate have to know which segment a message belongs to
    before they can place it.
    """
    if before_input is not None:
        before_input(
            "messages",
            ("session_id", "position", "variant_index"),
            "SELECT rowid FROM messages WHERE message_id=?",
            (message_id,),
        )
    with connection_cursor(
        conn, "SELECT session_id, position, variant_index FROM messages WHERE message_id = ?", (message_id,)
    ) as cursor:
        row = cursor.fetchone()
    if row is None:
        return None
    return (str(row["session_id"]), int(row["position"]), int(row["variant_index"]))


class _SegmentList:
    """A composed transcript's segments, cut and extended down a lineage chain.

    One list is cut at each branch point and extended by each tail, with an
    index of each session's segment and a running total, so a deep chain
    composes in time linear in its depth instead of rescanning and copying
    every intermediate segment tuple. Sessions on a chain are distinct (the
    walk's visited set), so each owns at most one segment.
    """

    def __init__(
        self, conn: sqlite3.Connection, base: Sequence[_TranscriptSegment], before_input: BeforeIndexInput | None = None
    ) -> None:
        self._conn = conn
        self._before_input = before_input
        self.segments = list(base)
        self._index = {segment.session_id: position for position, segment in enumerate(self.segments)}
        self.total = sum(segment.message_count for segment in self.segments)

    def cut_at(self, branch_point_message_id: str) -> bool:
        """Cut after the branch point; ``False`` when it is not in the segments.

        ``False`` is the dangling branch point: the parent message was
        hard-deleted, or it is not part of the parent's composed transcript.
        """
        located = _message_coordinates(self._conn, branch_point_message_id, self._before_input)
        if located is None:
            return False
        owner_session_id, position, variant_index = located
        at = self._index.get(owner_session_id)
        if at is None:
            return False
        segment = self.segments[at]
        if (
            segment.upto_position is not None
            and segment.upto_variant_index is not None
            and (position, variant_index) > (segment.upto_position, segment.upto_variant_index)
        ):
            return False
        for dropped in self.segments[at:]:
            self.total -= dropped.message_count
            del self._index[dropped.session_id]
        del self.segments[at:]
        self.append(
            _TranscriptSegment(
                session_id=owner_session_id,
                upto_position=position,
                upto_variant_index=variant_index,
                message_count=_count_session_messages(
                    self._conn, owner_session_id, upto_position=position, upto_variant_index=variant_index
                ),
            )
        )
        return True

    def clear(self) -> None:
        self.segments, self._index, self.total = [], {}, 0

    def append(self, segment: _TranscriptSegment) -> None:
        self._index[segment.session_id] = len(self.segments)
        self.segments.append(segment)
        self.total += segment.message_count


def _composed_transcript_plan(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> _ComposedTranscriptPlan:
    """Plan a session's composed transcript: the parent's prefix, then its own tail.

    4ts.6: two paths yield an INCOMPLETE transcript -- a cycle, and a dangling
    branch point (the parent message was hard-deleted, so only the child's own
    tail remains, starting mid-conversation). Both are recorded on the plan
    rather than served as if whole, and a parent's own incompleteness
    propagates down, since a child's composed view contains that parent's
    transcript. The walk is iterative and bounded by its visited set alone:
    every step adds a new session, so no depth cap drops a valid ancestor.
    """

    def own_segment(target_session_id: str) -> _TranscriptSegment:
        return _TranscriptSegment(
            session_id=target_session_id,
            upto_position=None,
            upto_variant_index=None,
            message_count=_count_session_messages(conn, target_session_id),
        )

    chain: list[tuple[str, str, str]] = []
    visited = {session_id}
    cursor_session_id = session_id
    plan: _ComposedTranscriptPlan
    while True:
        prefix_lost = _has_incomplete_prefix(conn, cursor_session_id, before_input)
        edge = _prefix_sharing_edge_sync(conn, cursor_session_id, before_input)
        if edge is None:
            own = own_segment(cursor_session_id)
            plan = _ComposedTranscriptPlan(
                segments=(own,),
                total_message_count=own.message_count,
                lineage_complete=not prefix_lost,
                lineage_truncation_reason=LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT if prefix_lost else None,
                lineage_inheritance="none",
                lineage_branch_point_message_id=None,
            )
            break
        parent_session_id, branch_point_message_id = edge
        if parent_session_id == cursor_session_id or parent_session_id in visited:
            own = own_segment(cursor_session_id)
            plan = _ComposedTranscriptPlan(
                segments=(own,),
                total_message_count=own.message_count,
                lineage_complete=False,
                lineage_truncation_reason=LINEAGE_TRUNCATION_CYCLE,
                lineage_inheritance="prefix-sharing",
                lineage_branch_point_message_id=branch_point_message_id,
            )
            break
        chain.append((cursor_session_id, parent_session_id, branch_point_message_id))
        visited.add(parent_session_id)
        cursor_session_id = parent_session_id

    if not chain:
        return plan
    composed = _SegmentList(conn, plan.segments, before_input)
    complete, reason = plan.lineage_complete, plan.lineage_truncation_reason
    for child_session_id, parent_session_id, branch_point_message_id in reversed(chain):
        if _has_invalidated_prefix(conn, child_session_id, before_input) and complete:
            complete, reason = False, LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT
        cut = _branch_point_content_address_matches(
            conn, child_session_id, parent_session_id, branch_point_message_id, before_input
        ) and composed.cut_at(branch_point_message_id)
        # The parent's own incompleteness is checked first: a branch point
        # missing from a truncated parent is a symptom of that truncation, not
        # an independent dangling-branch-point condition.
        if not cut:
            composed.clear()
            if complete:
                complete, reason = False, LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT
        composed.append(own_segment(child_session_id))
    return _ComposedTranscriptPlan(
        segments=tuple(composed.segments),
        total_message_count=composed.total,
        lineage_complete=complete,
        lineage_truncation_reason=reason,
        lineage_inheritance="prefix-sharing",
        # The requested session's own edge: the chain is leaf-first.
        lineage_branch_point_message_id=chain[0][2],
    )


def _read_session_header_row(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    """Read the declared envelope projection for one session, or raise ``KeyError``."""
    session = conn.execute(
        f"""
        SELECT {archive_session_envelope_select_sql()}
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        raise KeyError(session_id)
    if not isinstance(session, sqlite3.Row):
        raise TypeError(f"expected sqlite3.Row for session {session_id!r}, got {type(session).__name__}")
    return session


def _read_session_working_directories(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    return tuple(
        str(row["path"])
        for row in conn.execute(
            """
            SELECT path
            FROM session_working_dirs
            WHERE session_id = ?
            ORDER BY position, path
            """,
            (session_id,),
        ).fetchall()
    )


def _read_orphan_attachments(
    conn: sqlite3.Connection, session_id: str, *, blob_store: BlobStore | None
) -> tuple[ArchiveAttachmentRow, ...]:
    """Read a session's message-less attachment refs.

    Same construction as the per-message rows -- including ``blob_hash`` and
    the resolved ``availability``. A page read and a full read of one session
    must not disagree about whether an attachment's bytes are present
    (the narrower-page failure class polylogue-blpir named for the session
    projection).
    """
    rows = conn.execute(
        """
        SELECT a.attachment_id AS attachment_id, a.display_name AS display_name, a.media_type AS media_type,
               a.byte_count AS byte_count, a.blob_hash AS blob_hash, a.acquisition_status AS acquisition_status,
               r.upload_origin AS upload_origin, r.direction AS direction,
               r.producer_ref AS producer_ref, r.source_url AS source_url,
               r.caption AS caption
        FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ? AND r.message_id IS NULL
        ORDER BY a.attachment_id
        """,
        (session_id,),
    ).fetchall()
    return tuple(_archive_attachment_row(row, message_id=None, blob_store=blob_store) for row in rows)


def _archive_attachment_row(
    row: sqlite3.Row, *, message_id: str | None, blob_store: BlobStore | None
) -> ArchiveAttachmentRow:
    blob_hash = bytes(row["blob_hash"]) if row["blob_hash"] is not None else None
    return ArchiveAttachmentRow(
        attachment_id=row["attachment_id"],
        message_id=message_id,
        display_name=row["display_name"],
        media_type=row["media_type"],
        byte_count=int(row["byte_count"] or 0),
        upload_origin=row["upload_origin"],
        direction=row["direction"],
        producer_ref=row["producer_ref"],
        source_url=row["source_url"],
        caption=row["caption"],
        blob_hash=blob_hash,
        acquisition_status=row["acquisition_status"],
        availability=_attachment_availability(blob_store, blob_hash, row["acquisition_status"]),
    )


def _composed_session_envelope(
    session: sqlite3.Row,
    *,
    plan: _ComposedTranscriptPlan,
    messages: tuple[ArchiveMessageRow, ...],
    working_directories: tuple[str, ...],
    orphan_attachments: tuple[ArchiveAttachmentRow, ...],
    total_message_count: int | None,
) -> ArchiveSessionEnvelope:
    """Assemble one envelope from a session header and a materialized plan."""
    return ArchiveSessionEnvelope(
        session_id=session["session_id"],
        native_id=session["native_id"],
        origin=session["origin"],
        title=session["title"],
        session_kind=session["session_kind"],
        active_leaf_message_id=session["active_leaf_message_id"],
        messages=messages,
        lineage_complete=plan.lineage_complete,
        lineage_truncation_reason=plan.lineage_truncation_reason,
        lineage_inheritance=plan.lineage_inheritance,
        lineage_branch_point_message_id=plan.lineage_branch_point_message_id,
        parent_session_id=session["parent_session_id"],
        root_session_id=session["root_session_id"],
        branch_type=session["branch_type"],
        title_source=session["title_source"],
        title_ref=session["title_ref"],
        display_name=session["display_name"],
        instructions_text=session["instructions_text"],
        created_at=_iso_from_ms(session["created_at_ms"]),
        updated_at=_iso_from_ms(session["updated_at_ms"]),
        working_directories=working_directories,
        git_branch=session["git_branch"],
        git_repository_url=session["git_repository_url"],
        provider_project_ref=session["provider_project_ref"],
        reported_cost_usd=session["reported_cost_usd"],
        orphan_attachments=orphan_attachments,
        total_message_count=total_message_count,
    )


def read_archive_session_envelope(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    blob_store: BlobStore | None = None,
) -> ArchiveSessionEnvelope:
    """Read a compact archive envelope, holding one read snapshot across composition.

    ``blob_store`` is the CAS of the archive ``conn`` belongs to; attachment
    availability is resolved against it and left ``None`` without one.

    For a prefix-sharing lineage child (#2467) the inherited prefix is not stored
    under this session; the returned ``messages`` compose the parent's transcript
    up to the branch point followed by this session's own messages, so reads see
    the full logical transcript while storage holds each message once.

    Composition issues multiple autocommit SELECTs across a recursive parent
    walk (own read -> edge read -> recursive parent read). Without a held
    transaction, a concurrent parent re-ingest between those reads can yield a
    torn transcript (4ts.4). If ``conn`` is not already inside a transaction
    (e.g. a caller-held write transaction), this wraps the whole composition
    in one deferred read transaction so every SELECT sees the same snapshot;
    the recursive call sees ``conn.in_transaction`` already true and skips
    re-wrapping.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN DEFERRED")
        try:
            return read_archive_session_envelope(conn, session_id, blob_store=blob_store)
        finally:
            conn.execute("ROLLBACK")
    conn.row_factory = sqlite3.Row
    session = _read_session_header_row(conn, session_id)
    plan = _composed_transcript_plan(conn, session_id)
    messages: list[ArchiveMessageRow] = []
    for segment in plan.segments:
        messages.extend(
            _fetch_session_rows(
                conn,
                segment.session_id,
                upto_position=segment.upto_position,
                upto_variant_index=segment.upto_variant_index,
                blob_store=blob_store,
            )
        )
    return _composed_session_envelope(
        session,
        plan=plan,
        messages=tuple(messages),
        working_directories=_read_session_working_directories(conn, session_id),
        orphan_attachments=_read_orphan_attachments(conn, session_id, blob_store=blob_store),
        # An unbounded read already holds every composed message, so the
        # bounded-page count stays absent (see ``ArchiveSessionEnvelope``).
        total_message_count=None,
    )


def _fetch_session_rows(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    upto_position: int | None = None,
    upto_variant_index: int | None = None,
    blob_store: BlobStore | None,
) -> list[ArchiveMessageRow]:
    """Every row one session contributes to a composed transcript, in order.

    The unbounded materializer for one ``_TranscriptSegment``. Related rows
    are hydrated in bind-safe batches by ``_hydrate_archive_message_rows``.
    """
    upto_clause = ""
    upto_params: tuple[int, int] | tuple[()] = ()
    if upto_position is not None and upto_variant_index is not None:
        upto_clause = " AND (position, variant_index) <= (?, ?)"
        upto_params = (upto_position, upto_variant_index)
    message_rows = conn.execute(
        f"""
        SELECT {archive_message_row_select_sql()}
        FROM messages
        WHERE session_id = ?{upto_clause}
        ORDER BY position, variant_index
        """,
        (session_id, *upto_params),
    ).fetchall()
    return _hydrate_archive_message_rows(conn, message_rows, session_id=session_id, blob_store=blob_store)


def _hydrate_archive_message_rows(
    conn: sqlite3.Connection,
    message_rows: Sequence[sqlite3.Row],
    *,
    session_id: str,
    blob_store: BlobStore | None,
) -> list[ArchiveMessageRow]:
    """Hydrate ordered message rows with blocks and attachments in safe batches.

    Both whole-session composition and bounded transcript pages use this
    relation reader. A modest batch ceiling keeps each ``IN`` query compact;
    the connection's actual variable limit may reduce it further. Batching
    limits statement size only and never limits how many rows are returned.
    """
    if not message_rows:
        return []
    variable_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    if variable_limit < 1:
        raise ValueError("SQLite connection cannot bind a message identifier")
    batch_size = min(variable_limit, 500)
    rows: list[ArchiveMessageRow] = []
    for start in range(0, len(message_rows), batch_size):
        batch = message_rows[start : start + batch_size]
        message_ids = [str(row["message_id"]) for row in batch]
        placeholders = ",".join("?" for _ in message_ids)
        block_rows = conn.execute(
            f"""
            SELECT {archive_block_row_select_sql()}
            FROM blocks
            WHERE message_id IN ({placeholders})
            ORDER BY message_id, position
            """,
            message_ids,
        ).fetchall()
        blocks_by_message: dict[str, list[ArchiveBlockRow]] = {}
        for block in block_rows:
            blocks_by_message.setdefault(str(block["message_id"]), []).append(archive_block_row(block))

        attachment_rows = conn.execute(
            f"""
            SELECT r.message_id AS message_id, a.attachment_id AS attachment_id,
                   a.display_name AS display_name, a.media_type AS media_type, a.byte_count AS byte_count,
                   a.blob_hash AS blob_hash, a.acquisition_status AS acquisition_status,
                   r.upload_origin AS upload_origin, r.direction AS direction, r.producer_ref AS producer_ref,
                   r.source_url AS source_url, r.caption AS caption
            FROM attachment_refs r
            JOIN attachments a ON a.attachment_id = r.attachment_id
            WHERE r.message_id IN ({placeholders})
            ORDER BY r.message_id, a.attachment_id
            """,
            message_ids,
        ).fetchall()
        attachments_by_message: dict[str, list[ArchiveAttachmentRow]] = {}
        for attachment in attachment_rows:
            message_id = str(attachment["message_id"])
            attachments_by_message.setdefault(message_id, []).append(
                _archive_attachment_row(attachment, message_id=message_id, blob_store=blob_store)
            )

        rows.extend(
            _row_to_archive_message(
                message,
                session_id,
                blocks=tuple(blocks_by_message.get(str(message["message_id"]), ())),
                attachments=tuple(attachments_by_message.get(str(message["message_id"]), ())),
            )
            for message in batch
        )
    return rows


def _row_to_archive_message(
    row: sqlite3.Row,
    session_id: str,
    *,
    blocks: tuple[ArchiveBlockRow, ...],
    attachments: tuple[ArchiveAttachmentRow, ...],
) -> ArchiveMessageRow:
    return ArchiveMessageRow(
        message_id=row["message_id"],
        native_id=row["native_id"],
        identity_source=row["identity_source"],
        role=row["role"],
        position=row["position"],
        variant_index=row["variant_index"],
        is_active_path=bool(row["is_active_path"]),
        is_active_leaf=bool(row["is_active_leaf"]),
        blocks=blocks,
        message_type=row["message_type"],
        material_origin=row["material_origin"],
        word_count=int(row["word_count"] or 0),
        has_tool_use=bool(row["has_tool_use"]),
        has_thinking=bool(row["has_thinking"]),
        has_paste=bool(row["has_paste"]),
        paste_boundary_state=row["paste_boundary_state"],
        occurred_at=_iso_from_ms(row["occurred_at_ms"]),
        duration_ms=int(row["duration_ms"] or 0),
        parent_message_id=row["parent_message_id"],
        attachments=attachments,
        source_session_id=session_id,
        stop_reason=row["stop_reason"],
        model_name=row["model_name"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cache_read_tokens=row["cache_read_tokens"],
        cache_write_tokens=row["cache_write_tokens"],
    )


def _fetch_message_window(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    offset: int,
    limit: int,
    blob_store: BlobStore | None,
) -> list[ArchiveMessageRow]:
    """Bounded ``[offset, offset + limit)`` window of a session's OWN rows.

    Batches block/attachment reads for exactly the returned messages through
    the same bind-safe hydrator used by the full-session reader.

    It takes no branch-point bound: a ``_TranscriptSegment`` already counted
    the rows it contributes, and since a bounded segment is a prefix of this
    session's own ordering, a window that stays inside that count can never
    reach a row past the branch point.
    """
    if limit <= 0:
        return []
    message_rows = conn.execute(
        f"""
        SELECT {archive_message_row_select_sql()}
        FROM messages
        WHERE session_id = ?
        ORDER BY position, variant_index
        LIMIT ? OFFSET ?
        """,
        (session_id, max(limit, 0), max(offset, 0)),
    ).fetchall()
    return _hydrate_archive_message_rows(conn, message_rows, session_id=session_id, blob_store=blob_store)


def _fetch_planned_window(
    conn: sqlite3.Connection,
    segments: tuple[_TranscriptSegment, ...],
    *,
    offset: int,
    limit: int,
    blob_store: BlobStore | None,
) -> list[ArchiveMessageRow]:
    """Fetch ``[offset, offset + limit)`` of a composed transcript.

    The plan already knows each segment's length, so the window is mapped onto
    the one or two segments it actually intersects and every other segment is
    skipped without reading a row. SQL work is proportional to ``limit`` and
    the chain depth, never to the composed transcript length.
    """
    if limit <= 0:
        return []
    remaining_offset = max(offset, 0)
    remaining = limit
    window: list[ArchiveMessageRow] = []
    for segment in segments:
        if remaining <= 0:
            break
        if remaining_offset >= segment.message_count:
            remaining_offset -= segment.message_count
            continue
        take = min(remaining, segment.message_count - remaining_offset)
        window.extend(
            _fetch_message_window(conn, segment.session_id, offset=remaining_offset, limit=take, blob_store=blob_store)
        )
        remaining -= take
        remaining_offset = 0
    return window


def read_archive_session_page(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    limit: int,
    offset: int,
    blob_store: BlobStore | None = None,
) -> ArchiveSessionEnvelope:
    """Read a bounded ``[offset, offset + limit)`` PAGE of a session's transcript.

    The page composes only the requested window at the SQL layer -- header,
    that window's messages, their blocks/attachments, and orphan attachments
    -- so first paint is bounded by the page size rather than the session's
    total message count (polylogue-07g6).

    That bound holds for a prefix-sharing lineage child too (polylogue-0le8d).
    The child's own rows are only its divergent tail, but
    ``_composed_transcript_plan`` resolves the ancestral prefix into segment
    lengths without reading a row, so the window is fetched from the one or
    two segments it intersects. A deep chain costs one plan per ancestor, not
    one full ancestral transcript.

    ``total_message_count`` on the returned envelope always carries the TRUE
    composed transcript length; ``messages`` holds only the requested window.
    A negative ``offset`` is clamped to zero rather than wrapping a window
    onto the end of the transcript. ``blob_store`` resolves attachment
    availability exactly as ``read_archive_session_envelope`` does.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN DEFERRED")
        try:
            return read_archive_session_page(conn, session_id, limit=limit, offset=offset, blob_store=blob_store)
        finally:
            conn.execute("ROLLBACK")
    conn.row_factory = sqlite3.Row
    session = _read_session_header_row(conn, session_id)
    plan = _composed_transcript_plan(conn, session_id)
    return _composed_session_envelope(
        session,
        plan=plan,
        messages=tuple(_fetch_planned_window(conn, plan.segments, offset=offset, limit=limit, blob_store=blob_store)),
        working_directories=_read_session_working_directories(conn, session_id),
        orphan_attachments=_read_orphan_attachments(conn, session_id, blob_store=blob_store),
        total_message_count=plan.total_message_count,
    )


def _count_session_messages_before(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    position: int,
    variant_index: int,
) -> int:
    """Count a session's own rows that precede ``(position, variant_index)``."""
    with connection_cursor(
        conn,
        """
        SELECT COUNT(*)
        FROM messages
        WHERE session_id = ?
          AND (position, variant_index) < (?, ?)
        """,
        (session_id, position, variant_index),
    ) as cursor:
        row = cursor.fetchone()
    return int(row[0])


def locate_composed_message(
    conn: sqlite3.Connection, session_id: str, message_id: str, *, before_input: BeforeIndexInput | None = None
) -> int | None:
    """Return a message's index in a session's COMPOSED transcript, or ``None``.

    ``None`` means this session's composed transcript does not contain that
    message -- it is a refusal for the caller to name, never index zero.

    Bounded exactly the way ``read_archive_session_page`` is (polylogue-2go3o):
    the composition is planned first, so the ancestral prefix of a
    prefix-sharing child resolves into segment lengths without reading a row,
    the owning segment is found by coordinate, and the rank inside it is one
    indexed COUNT. Numbering a message by composing the transcript -- the only
    way to answer this before the plan existed -- costs the whole ancestral
    chain for a deep child, which is the cost a deep link was supposed to
    remove.

    The index is the ``(position, variant_index)`` composition order every
    read here windows with, so a located index and the page holding it can
    never disagree.
    """
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            return locate_composed_message(conn, session_id, message_id, before_input=before_input)
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass
    conn.row_factory = sqlite3.Row
    located = _message_coordinates(conn, message_id, before_input)
    if located is None:
        return None
    owner_session_id, position, variant_index = located
    plan = _composed_transcript_plan(conn, session_id, before_input)
    preceding = 0
    for segment in plan.segments:
        if segment.session_id == owner_session_id and not (
            segment.upto_position is not None
            and segment.upto_variant_index is not None
            and (position, variant_index) > (segment.upto_position, segment.upto_variant_index)
        ):
            return preceding + _count_session_messages_before(
                conn,
                owner_session_id,
                position=position,
                variant_index=variant_index,
            )
        preceding += segment.message_count
    return None


def read_session_agent_policies(conn: sqlite3.Connection, session_id: str) -> list[ArchiveAgentPolicy]:
    """Read all agent-policy rows for a session, ordered by position."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT policy_id, session_id, position, approval_policy,
               sandbox_policy, network_policy, observed_at_ms, source_message_id
        FROM session_agent_policies
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return [
        ArchiveAgentPolicy(
            policy_id=str(row["policy_id"]),
            session_id=str(row["session_id"]),
            position=int(row["position"]),
            approval_policy=row["approval_policy"],
            sandbox_policy=row["sandbox_policy"],
            network_policy=row["network_policy"],
            observed_at_ms=row["observed_at_ms"],
            source_message_id=row["source_message_id"],
        )
        for row in rows
    ]


def search_archive_blocks(conn: sqlite3.Connection, query: str) -> list[str]:
    """Return block ids matched by the archive contentless FTS table.

    A read open admits an index whose message FTS surface is absent
    (``MESSAGE_FTS_DEGRADABLE_OBJECTS``) because search reports that state
    itself. Honour that here rather than reaching SQL and raising
    ``no such table: messages_fts``. Presence, not freshness, is the check:
    rebuild and differential routes read this surface mid-convergence, when
    it is legitimately behind ``blocks``.
    """
    from polylogue.core.errors import DatabaseError
    from polylogue.core.sqlite_introspection import table_exists
    from polylogue.storage.fts.fts_lifecycle import MESSAGE_SEARCH_REPAIR_HINT

    match_query = normalize_fts5_query(query)
    if match_query is None:
        return []
    if not table_exists(conn, "messages_fts"):
        raise DatabaseError(f"Search index not built. {MESSAGE_SEARCH_REPAIR_HINT}")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT b.block_id
        FROM messages_fts f
        JOIN blocks b ON b.rowid = f.rowid
        WHERE f.text MATCH ?
        ORDER BY rank
        """,
        (match_query,),
    ).fetchall()
    return [row["block_id"] for row in rows]


def rebuild_archive_messages_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the archive message FTS index from canonical ``blocks`` rows."""
    conn.execute("DELETE FROM messages_fts")
    conn.execute(
        f"""
        INSERT INTO messages_fts(rowid, text)
        SELECT rowid, {pl_fold_sql_expr("search_text")}
        FROM blocks
        WHERE search_text != ''
        """
    )
    row = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()
    return int(row[0] if row is not None else 0)


def _build_message_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> list[tuple[object, ...]]:
    """Pure row-tuple builder for the ``messages`` table (no DB access).

    Extracted from ``_write_messages`` (polylogue-623q) so the exact same
    row-construction logic can run off the writer thread (see
    ``prepare_session_rows``) and be reused verbatim by the writer via
    ``executemany`` -- byte-identical output either way. This function
    decides only *what* the rows are, never *whether* to write them.
    """
    return list(
        _iter_message_rows(
            session_id,
            messages,
            content_identities=content_identities,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
    )


def _iter_message_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> Iterator[tuple[object, ...]]:
    for fallback_position, message in enumerate(messages):
        position = position_offset + (message.position if message.position is not None else fallback_position)
        variant_index = message.variant_index if message.variant_index is not None else 0
        values: dict[str, object] = {
            "session_id": session_id,
            "native_id": _stored_message_native_id(message, duplicate_native_ids),
            "source_native_id_json": (source_native_id_json(message.provider_message_id)),
            "position": position,
            "role": _enum_value(message.role),
            "message_type": _enum_value(message.message_type),
            "material_origin": _enum_value(message.material_origin),
            "model_name": _sqlite_text(message.model_name),
            "model_effort": _sqlite_text(message.model_effort),
            "sender_name": _sqlite_text(message.sender_name),
            "recipient": _sqlite_text(message.recipient),
            "delivery_status": _sqlite_text(message.delivery_status),
            "end_turn": None if message.end_turn is None else int(message.end_turn),
            "user_context_text": _sqlite_text(message.user_context_text),
            "has_tool_use": _has_block(message, BlockType.TOOL_USE),
            "has_thinking": _has_block(message, BlockType.THINKING),
            "has_paste": _has_paste(message),
            "paste_boundary": _paste_boundary(message),
            "variant_index": variant_index,
            "is_active_path": 1 if message.is_active_path is not False else 0,
            "is_active_leaf": 1 if message.is_active_leaf else 0,
            "word_count": _word_count(message.text),
            "input_tokens": message.input_tokens,
            "output_tokens": message.output_tokens,
            "cache_read_tokens": message.cache_read_tokens,
            "cache_write_tokens": message.cache_write_tokens,
            "duration_ms": message.duration_ms,
            "content_address": _message_content_address(message),
            "content_hash": _message_content_hash(session_id, message, position=position, variant_index=variant_index),
            "fields_digest": _message_fields_digest(message),
            "occurred_at_ms": message.occurred_at_ms
            if message.occurred_at_ms is not None
            else to_epoch_ms(message.timestamp, numeric_unit="seconds"),
            "stop_reason": _enum_value(message.stop_reason),
        }
        content_identity, content_occurrence = content_identities[fallback_position]
        values["content_identity"] = content_identity
        values["content_occurrence"] = content_occurrence
        values["identity_source"] = "native" if values["native_id"] is not None else "content"
        yield archive_tiers_specs.MESSAGES_SPEC.extract_tuple(values)


def _messages_insert_sql() -> str:
    spec = archive_tiers_specs.MESSAGES_SPEC
    return f"""
        INSERT INTO messages (
            {spec.insert_column_names}
        ) VALUES ({spec.insert_placeholder_string})
        """


def _write_append_messages_and_blocks(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    message_rows: Iterable[tuple[object, ...]],
    block_rows: Iterable[tuple[object, ...]],
) -> None:
    """Append only new normalized native IDs while keeping each message's blocks aligned.

    The existence check must happen before inserting the message row. Doing
    the two writes together avoids retaining a potentially large set of IDs
    between message and block insertion, and still allows session metadata,
    events, and attachments to follow their ordinary append paths.
    """
    message_columns = [
        column.name
        for column in archive_tiers_specs.MESSAGES_SPEC.writable_columns
        if column.extract_placeholder == "?"
    ]
    native_index = message_columns.index("native_id")
    carrier_index = message_columns.index("source_native_id_json")
    message_iter = iter(message_rows)
    block_iter = iter(block_rows)
    pending_messages: list[tuple[object, ...]] = []
    pending_blocks: list[tuple[object, ...]] = []

    def flush_messages() -> None:
        if pending_messages:
            conn.executemany(_messages_insert_sql(), pending_messages)
            pending_messages.clear()

    def flush_blocks() -> None:
        if pending_blocks:
            conn.executemany(_blocks_insert_sql(), pending_blocks)
            pending_blocks.clear()

    for message, message_row in zip(messages, message_iter, strict=True):
        message_blocks = _message_blocks(message)
        aligned_blocks = islice(block_iter, len(message_blocks))
        if _stored_native_id_exists(conn, session_id, message_row[native_index], carrier=message_row[carrier_index]):
            for _ in aligned_blocks:
                pass
            continue
        pending_messages.append(message_row)
        if len(pending_messages) >= 256:
            flush_messages()
        for block_row in aligned_blocks:
            pending_blocks.append(block_row)
            if len(pending_blocks) >= 4096:
                # The owning message row must exist before its first block,
                # while blocks can flush independently of message batches.
                flush_messages()
                flush_blocks()
    if next(block_iter, None) is not None or next(message_iter, None) is not None:
        raise RuntimeError("prepared append rows exceeded their parsed message owners")
    flush_messages()
    flush_blocks()


def _append_has_new_native_messages(
    conn: sqlite3.Connection, session_id: str, rows: Iterable[tuple[object, ...]]
) -> bool:
    columns = [
        column.name
        for column in archive_tiers_specs.MESSAGES_SPEC.writable_columns
        if column.extract_placeholder == "?"
    ]
    native_index = columns.index("native_id")
    carrier_index = columns.index("source_native_id_json")
    return any(
        not _stored_native_id_exists(conn, session_id, row[native_index], carrier=row[carrier_index]) for row in rows
    )


def _append_has_auxiliary_input(session: ParsedSession) -> bool:
    """Whether an append carries product data beyond its message delta."""
    return bool(
        session.attachments
        or session.session_events
        or session.provider_session_aliases
        or session.session_kind.value != "standard"
        or session.parent_session_provider_id
        or session.branch_point_provider_message_id
        or session.branch_type
        or session.title
        or session.title_source
        or session.title_ref
        or session.instructions_text
        or session.reported_duration_ms is not None
        or session.reported_cost_usd is not None
        or session.models_used
        or session.working_directories
        or session.git_branch
        or session.git_repository_url
        or session.provider_project_ref
        or session.team_name
        or session.git_commit_hash
        or session.ingest_flags
        or session.display_name
        or session.pending_drafts
        or session.session_refs
    )


def _write_messages(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    rows: Iterable[tuple[object, ...]] | None = None,
) -> None:
    """Write message rows using table-driven column specification.

    The messages table column spec (archive_tiers_specs.MESSAGES_SPEC) defines:
      - writable_columns: the ordered list of columns to INSERT
      - The column names and placeholders are generated from the spec
      - The tuple order is derived from the spec's writable_columns order

    This consolidates the three hand-aligned duplicates (column list in INSERT,
    placeholder string, tuple order) into a single source of truth.

    ``rows`` (polylogue-623q), when provided, is used verbatim instead of
    rebuilding from ``messages`` -- the caller (``_replace_full_session_
    messages_and_blocks``) supplies this when it has an already-validated
    ``PreparedSessionRows`` computed off the writer thread. ``messages`` is
    still required in that case (every call site passes it regardless) so
    this function's signature and every other caller stay unchanged.
    """
    if rows is None:
        rows = _iter_message_rows(
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
            content_identities=content_identities,
        )
    conn.executemany(_messages_insert_sql(), rows)


def _message_content_hash(
    session_id: str,
    message: ParsedMessage,
    *,
    position: int,
    variant_index: int,
) -> bytes:
    """Digest the stored message content, not just its identity.

    This hash is deliberately IDENTITY-INCLUSIVE (session_id, position,
    variant_index, provider_message_id) -- it drives row-level re-ingest/
    dedup change detection for the ``messages`` table itself. It is no
    longer what embedding freshness is gated on: since polylogue-q88p,
    embeddings are keyed by ``vector_derivation_hash`` (identity-FREE --
    ``storage/embeddings/identity.py``), computed straight from the
    embedder's input text, so a rebuild or lineage-normalization shift that
    changes this hash without changing the actual text no longer forces a
    wasted re-embed.

    The message's own fields enter through their stored digest
    (``_message_fields_digest``, the ``fields_digest`` column), so a row moved
    or copied outside this parse is rehashed exactly from its stored columns.
    """
    return _message_row_hash(
        session_id,
        message.provider_message_id,
        position,
        variant_index,
        _message_fields_digest(message),
        _parsed_block_hash_parts(message),
    )


def _row_digest_text(value: str | None) -> str:
    """Frame optional stored text: absent is empty, every present value starts with ``=``.

    The tag makes NULL, an empty string and literal tag strings disjoint.
    SQLite's surrogate replacement is shared by parsed and stored-row routes.
    """
    text = _sqlite_text(value)
    return "" if text is None else "=" + text


def _message_fields_digest(message: ParsedMessage) -> bytes:
    """The message's own content fields, apart from its identity and its blocks."""
    return _hash_bytes(
        "message-fields",
        _row_digest_text(_enum_value(message.role)),
        _row_digest_text(_enum_value(message.message_type)),
        _row_digest_text(_enum_value(message.material_origin)),
        _row_digest_text(message.text),
        _row_digest_text(message.user_context_text),
        _row_digest_text(_enum_value(message.stop_reason)),
        _row_digest_text(message.model_name),
        _row_digest_text(message.model_effort),
        _row_digest_text(message.sender_name),
        _row_digest_text(message.recipient),
        _row_digest_text(message.delivery_status),
        "" if message.end_turn is None else str(int(message.end_turn)),
        "" if message.occurred_at_ms is None else str(message.occurred_at_ms),
    )


def _parsed_block_hash_parts(message: ParsedMessage) -> Iterator[str]:
    for block in _message_blocks(message):
        yield from (
            _block_type(block).value,
            _row_digest_text(block.text),
            _row_digest_text(block.tool_name),
            _row_digest_text(block.tool_id),
            _row_digest_text(_json_dumps(block.tool_input) if block.tool_input is not None else None),
            _row_digest_text(_semantic_type(block)),
            _row_digest_text(block.media_type),
            _row_digest_text(_block_language(block)),
            "" if block.is_error is None else str(int(block.is_error)),
            "" if block.exit_code is None else str(block.exit_code),
            _row_digest_text(_enum_value(block.tool_outcome)),
        )


def _stored_block_hash_parts(block_rows: Iterable[Sequence[object]], b_idx: Mapping[str, int]) -> Iterator[str]:
    """``_parsed_block_hash_parts`` of stored block rows, in the same order."""
    for row in block_rows:
        is_error = row[b_idx["tool_result_is_error"]]
        exit_code = row[b_idx["tool_result_exit_code"]]
        yield from (
            cast(str, row[b_idx["block_type"]]),
            _row_digest_text(cast("str | None", row[b_idx["text"]])),
            _row_digest_text(cast("str | None", row[b_idx["tool_name"]])),
            _row_digest_text(cast("str | None", row[b_idx["tool_id"]])),
            _row_digest_text(cast("str | None", row[b_idx["tool_input"]])),
            _row_digest_text(cast("str | None", row[b_idx["semantic_type"]])),
            _row_digest_text(cast("str | None", row[b_idx["media_type"]])),
            _row_digest_text(cast("str | None", row[b_idx["language"]])),
            "" if is_error is None else str(int(cast(int, is_error))),
            "" if exit_code is None else str(cast(int, exit_code)),
            _row_digest_text(cast("str | None", row[b_idx["tool_outcome"]])),
        )


def _message_row_hash(
    session_id: str,
    native_id: str | None,
    position: int,
    variant_index: int,
    fields_digest: bytes,
    block_parts: Iterable[str],
) -> bytes:
    """``content_hash``'s framing (``_hash_bytes``), streamed over the block parts."""
    digest = hashlib.sha256()
    for part in ("message", session_id, native_id or "", str(position), str(variant_index), fields_digest.hex()):
        encoded = part.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for part in block_parts:
        encoded = part.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _row_fields_digest(row: Sequence[object], m_idx: Mapping[str, int]) -> bytes:
    """A stored message row's ``fields_digest``.

    A row written before the column existed has none; its digest is then
    rebuilt from the stored columns, which lack only the message text.
    """
    stored = row[m_idx["fields_digest"]]
    if isinstance(stored, bytes | bytearray | memoryview):
        return bytes(stored)
    return _hash_bytes(
        "message-fields",
        *(
            _row_digest_text(None if row[m_idx[name]] is None else str(row[m_idx[name]]))
            for name in ("role", "message_type", "material_origin")
        ),
        "",
        *(
            _row_digest_text(None if row[m_idx[name]] is None else str(row[m_idx[name]]))
            for name in (
                "user_context_text",
                "stop_reason",
                "model_name",
                "model_effort",
                "sender_name",
                "recipient",
                "delivery_status",
            )
        ),
        *("" if row[m_idx[name]] is None else str(row[m_idx[name]]) for name in ("end_turn", "occurred_at_ms")),
    )


def _message_content_address(message: ParsedMessage) -> bytes:
    """Return the complete current semantic witness used at branch points.

    This delegates to the content-hash projection so newly hashed message or
    block fields cannot be omitted from branch verification.
    """
    return message_semantic_content_address(message)


def _block_content_hash(
    *,
    block_type: str,
    text: str | None,
    tool_name: str | None,
    tool_input_json: str | None,
    semantic_type: str | None,
    media_type: str | None,
    language: str | None,
    is_error: bool | None,
    exit_code: int | None,
    tool_outcome: ToolOutcome | str | None = None,
    outcome_unknown_reason: str | None = None,
    semantic_extra_json: str | None = None,
) -> bytes:
    """Digest a block's canonical EVIDENCE, deliberately excluding identity (svfj).

    Excludes session_id/message_id/position/tool_id on purpose: those shift
    on fork-position replay, re-ingest renumbering, and provider tool-id
    regeneration, but the block's actual evidence content does not. This is
    the anchor atom multiple programs stand on (webui citations, finding
    evidence refs, drift detection, compaction loss anchors, export
    citations) -- a citation keyed on this hash survives all three shifts.
    """

    return _hash_bytes(
        "block",
        block_type,
        _row_digest_text(text),
        _row_digest_text(tool_name),
        _row_digest_text(tool_input_json),
        _row_digest_text(semantic_type),
        _row_digest_text(media_type),
        _row_digest_text(language),
        "" if is_error is None else str(int(is_error)),
        "" if exit_code is None else str(exit_code),
        _row_digest_text(_enum_value(tool_outcome)),
        _row_digest_text(outcome_unknown_reason),
        _row_digest_text(semantic_extra_json),
    )


#: ``semantic_extra_json`` of a block with no metadata, file edit or web
#: constructs -- ``_json_dumps`` of that value, spelled out because the
#: module's JSON helper is defined below. ``content_hash`` still digests it;
#: only the stored column is NULL, so an ordinary block carries no copy.
_EMPTY_SEMANTIC_EXTRA_JSON = '{"file_edit":null,"metadata":null,"web_constructs":[]}'


def _block_semantic_extra_json(block: ParsedContentBlock) -> str:
    """The semantic extras ``_block_content_hash`` digests for a parsed block."""
    return _json_dumps(
        {
            "metadata": block.metadata,
            "file_edit": block.file_edit.model_dump(mode="json") if block.file_edit else None,
            "web_constructs": [item.model_dump(mode="json") for item in block.web_constructs],
        }
    )


def _stored_semantic_extra_json(semantic_extra_json: str) -> str | None:
    """The ``blocks.semantic_extra_json`` value for a block's digested extras."""
    return None if semantic_extra_json == _EMPTY_SEMANTIC_EXTRA_JSON else semantic_extra_json


def _hashed_semantic_extra_json(stored: object) -> str:
    """The extras a stored block row's ``content_hash`` digests."""
    return _EMPTY_SEMANTIC_EXTRA_JSON if stored is None else cast(str, stored)


def _build_block_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> list[tuple[object, ...]]:
    """Pure row-tuple builder for the ``blocks`` table (no DB access).

    Extracted from ``_write_blocks`` (polylogue-623q); see
    ``_build_message_rows`` for why this split exists.
    """
    return list(
        _iter_block_rows(
            session_id,
            messages,
            content_identities=content_identities,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
    )


def _iter_block_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> Iterator[tuple[object, ...]]:
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            duplicate_native_ids=duplicate_native_ids,
        )
        blocks = _message_blocks(message)
        for position, (block, identity) in enumerate(zip(blocks, block_content_identities(blocks), strict=True)):
            block_type = _block_type(block)
            tool_input_json = _json_dumps(block.tool_input) if block.tool_input is not None else None
            semantic_type = _semantic_type(block)
            language = _block_language(block)
            is_error = getattr(block, "is_error", None)
            exit_code = getattr(block, "exit_code", None)
            tool_outcome = getattr(block, "tool_outcome", None)
            outcome_unknown_reason = _enum_value(block.outcome_unknown_reason)
            signature = getattr(block, "signature", None)
            semantic_extra_json = _block_semantic_extra_json(block)
            values: dict[str, object] = {
                "message_id": message_id,
                "session_id": session_id,
                "position": position,
                "content_identity": identity.content_identity,
                "content_occurrence": identity.content_occurrence,
                "block_type": block_type.value,
                "text": _sqlite_text(block.text),
                "tool_name": _sqlite_text(block.tool_name),
                "tool_id": _sqlite_text(block.tool_id),
                "tool_input": tool_input_json,
                "semantic_type": _sqlite_text(semantic_type),
                "media_type": _sqlite_text(block.media_type),
                "language": _sqlite_text(language),
                "name": _sqlite_text(_block_name(block)),
                "tool_result_is_error": _sqlite_bool(is_error),
                "tool_result_exit_code": exit_code,
                "tool_outcome": getattr(block, "tool_outcome", None),
                "tool_result_outcome_unknown_reason": outcome_unknown_reason,
                "signature": _sqlite_text(signature),
                "semantic_extra_json": _stored_semantic_extra_json(semantic_extra_json),
                "content_hash": _block_content_hash(
                    block_type=block_type.value,
                    text=block.text,
                    tool_name=block.tool_name,
                    tool_input_json=tool_input_json,
                    semantic_type=semantic_type,
                    media_type=block.media_type,
                    language=language,
                    is_error=is_error,
                    exit_code=exit_code,
                    tool_outcome=tool_outcome,
                    outcome_unknown_reason=outcome_unknown_reason,
                    semantic_extra_json=semantic_extra_json,
                ),
            }
            yield archive_tiers_specs.BLOCKS_SPEC.extract_tuple(values)


def _blocks_insert_sql() -> str:
    spec = archive_tiers_specs.BLOCKS_SPEC
    return f"""
        INSERT OR REPLACE INTO blocks (
            {spec.insert_column_names}
        ) VALUES ({spec.insert_placeholder_string})
        """


def _write_blocks(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    rows: Iterable[tuple[object, ...]] | None = None,
) -> None:
    """Write block rows using table-driven column specification.

    The blocks table column spec (archive_tiers_specs.BLOCKS_SPEC) defines:
      - writable_columns: the ordered list of columns to INSERT
      - The column names and placeholders are generated from the spec
      - The tuple order is derived from the spec's writable_columns order

    This consolidates the hand-aligned duplicates into a single source of truth.

    ``rows`` (polylogue-623q): see ``_write_messages`` for the contract --
    used verbatim when provided, otherwise built fresh from ``messages``.
    """
    if rows is None:
        rows = _iter_block_rows(
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
            content_identities=content_identities,
        )
    conn.executemany(_blocks_insert_sql(), rows)


def _reconcile_tool_use_outcomes(conn: sqlite3.Connection, session_id: str) -> bool:
    """Apply the shared association after original stored parent links settle."""
    from polylogue.storage.sqlite.action_pairs import action_pairs_select_sql

    association = action_pairs_select_sql(use_bound=" AND u.session_id=?", result_bound=" AND r.session_id=?")
    rows = conn.execute(
        f"""WITH associated AS ({association})
        SELECT b.*,a.tool_outcome AS associated_outcome FROM associated a
        JOIN blocks b ON b.block_id=a.tool_use_block_id""",
        (session_id, session_id, session_id),
    ).fetchall()
    changed = False
    for use in rows:
        outcome = use["associated_outcome"]
        # Result absence does not retract independent execution-sidecar facts.
        if outcome == ToolOutcome.NO_RESULT.value and use["tool_outcome"] is not None:
            continue
        if outcome == use["tool_outcome"]:
            continue
        content_hash = _block_content_hash(
            block_type=use["block_type"],
            text=use["text"],
            tool_name=use["tool_name"],
            tool_input_json=use["tool_input"],
            semantic_type=use["semantic_type"],
            media_type=use["media_type"],
            language=use["language"],
            is_error=use["tool_result_is_error"],
            exit_code=use["tool_result_exit_code"],
            tool_outcome=outcome,
            outcome_unknown_reason=use["tool_result_outcome_unknown_reason"],
            semantic_extra_json=_hashed_semantic_extra_json(use["semantic_extra_json"]),
        )
        conn.execute(
            "UPDATE blocks SET tool_outcome = ?, content_hash = ? WHERE block_id = ?",
            (outcome, content_hash, use["block_id"]),
        )
        changed = True
    return changed


def _build_file_edit_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> list[tuple[object, ...]]:
    """Pure row-tuple builder for ``file_edits`` (polylogue-2qx.4).

    ``ParsedFileEdit`` arrives attached to the TOOL_RESULT block that reports
    the edit outcome (matching where the provider's own structuredPatch/
    originalFile fields live on the wire), but ``file_edits`` is keyed by the
    TOOL_USE block that made the call -- resolved here via the shared
    ``tool_id``, exactly as the ``actions`` view pairs tool_use<->tool_result.
    The shared invocation association resolves each result occurrence, so
    reused tool IDs cannot redirect an edit to a later call. An edit without
    a unique TOOL_USE in this write refuses instead of guessing a key.
    """
    return list(
        _iter_file_edit_rows(
            session_id,
            messages,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
    )


def _iter_file_edit_rows(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> Generator[tuple[object, ...], None, None]:
    from polylogue.sources.tool_outcomes import iter_tool_result_owners

    source = messages.messages if isinstance(messages, _MessageTail) else messages
    disk_index = _DiskSourceMessageIds(source.path.parent) if isinstance(source, SqliteMessageSink) else None
    tool_use_block_ids: dict[str, str] | _DiskSourceMessageIds = disk_index if disk_index is not None else {}
    try:
        has_edits = False
        for fallback_position, message in enumerate(messages):
            message_id = _message_id(
                session_id,
                message,
                fallback_position,
                content_identities=content_identities,
                message_id_remap=message_id_remap,
                duplicate_native_ids=duplicate_native_ids,
            )
            blocks = _message_blocks(message)
            for block_ordinal, (block, identity) in enumerate(
                zip(blocks, block_content_identities(blocks), strict=True)
            ):
                has_edits |= block.file_edit is not None and bool(block.tool_id)
                if _block_type(block) is BlockType.TOOL_USE and block.tool_id:
                    tool_use_block_ids[f"use:{fallback_position}:{block_ordinal}"] = archive_block_id(
                        message_id,
                        content_identity=identity.content_identity,
                        content_occurrence=identity.content_occurrence,
                    )
        if not has_edits:
            return
        with closing(iter_tool_result_owners(messages)) as associations:
            for result_message, result_block, use_message, use_block in associations:
                tool_use_block_ids[f"result:{result_message}:{result_block}"] = tool_use_block_ids[
                    f"use:{use_message}:{use_block}"
                ]
        if disk_index is not None:
            disk_index.finish()

        def pending_edits() -> Iterator[tuple[str, str, ParsedMessage, ParsedContentBlock]]:
            for fallback_position, message in enumerate(messages):
                message_id = _message_id(
                    session_id,
                    message,
                    fallback_position,
                    content_identities=content_identities,
                    message_id_remap=message_id_remap,
                    duplicate_native_ids=duplicate_native_ids,
                )
                for block_ordinal, block in enumerate(_message_blocks(message)):
                    if block.file_edit is not None and block.tool_id:
                        side = "use" if _block_type(block) is BlockType.TOOL_USE else "result"
                        yield f"{side}:{fallback_position}:{block_ordinal}", message_id, message, block

        pending = pending_edits()
        while batch := list(islice(pending, 512)):
            rows: list[tuple[object, ...]] = []
            with disk_index.reader() if disk_index is not None else nullcontext():
                for owner_key, message_id, message, block in batch:
                    file_edit = block.file_edit
                    assert file_edit is not None and block.tool_id is not None
                    tool_use_block_id = tool_use_block_ids.get(owner_key)
                    if tool_use_block_id is None:
                        raise MessageOwnerAmbiguityError("file edit has no unique tool-use occurrence")
                    rows.append(
                        (
                            tool_use_block_id,
                            session_id,
                            message_id,
                            _sqlite_text(file_edit.file_path),
                            _json_dumps(file_edit.structured_patch) if file_edit.structured_patch is not None else None,
                            _sqlite_text(file_edit.original_file),
                            _sqlite_text(file_edit.old_string),
                            _sqlite_text(file_edit.new_string),
                            _sqlite_bool(file_edit.replace_all),
                            _sqlite_bool(file_edit.user_modified),
                            message.occurred_at_ms
                            if message.occurred_at_ms is not None
                            else to_epoch_ms(message.timestamp, numeric_unit="seconds"),
                        )
                    )
            # No generator-owned SQLite handle remains live at this boundary.
            yield from rows
    finally:
        if disk_index is not None:
            disk_index.close()


def _write_file_edits(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    rows = _iter_file_edit_rows(
        session_id,
        messages,
        position_offset=position_offset,
        duplicate_native_ids=duplicate_native_ids,
        content_identities=content_identities,
        message_id_remap=message_id_remap,
    )
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO file_edits (
                tool_use_block_id, session_id, message_id, file_path,
                structured_patch_json, original_file, old_string, new_string,
                replace_all, user_modified, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    finally:
        rows.close()


def _write_session_refs(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    """Write tracker-agnostic session references (polylogue-2qx.4).

    Full-replace-only (mirrors ``_write_working_dirs``/``_write_repo_edges``):
    ``session_refs`` rows are re-derived from ``session.session_refs`` on
    every write, not appended incrementally, since the parser always emits
    the complete current set for a session.
    """
    observed_at_ms = to_epoch_ms(session.updated_at, numeric_unit="seconds") or to_epoch_ms(
        session.created_at, numeric_unit="seconds"
    )
    for position, ref in enumerate(session.session_refs):
        conn.execute(
            """
            INSERT OR REPLACE INTO session_refs (
                session_id, position, kind, repo, ref_number, url, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                position,
                _enum_value(ref.kind),
                _sqlite_text(ref.repo),
                ref.number,
                ref.url,
                observed_at_ms,
            ),
        )


def _write_web_constructs(
    conn: sqlite3.Connection,
    session: ParsedSession,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    replace_session: bool = True,
    prior_rows: bool = True,
) -> None:
    """Write web constructs, replacing the session's (or, merging, each block's) prior rows.

    ``prior_rows`` is false when every message is new: there is nothing to
    replace, so neither the session-wide nor the per-block delete runs.
    """
    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    provider = _enum_value(session.source_name)
    rows: list[tuple[object, ...]] = []
    block_ids: list[str] = []
    if replace_session and prior_rows:
        conn.execute("DELETE FROM web_content_constructs WHERE session_id = ?", (session_id,))

    def _flush_rows() -> None:
        if block_ids:
            conn.executemany(
                "DELETE FROM web_content_constructs WHERE block_id = ?", ((block_id,) for block_id in block_ids)
            )
            block_ids.clear()
        if rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO web_content_constructs (
                    session_id, message_id, block_id, position, provider, construct_type,
                    provider_key, title, url, text, source_id, group_id, group_title,
                    query, asset_pointer, mime_type, status, task_id, task_type,
                    rank, start_index, end_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            rows.clear()

    # Iterate the (possibly lineage-sliced) tail messages, not session.messages —
    # a web construct on an inherited-prefix message would FK-violate against rows
    # that were never written under this session (#2467 audit).
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            duplicate_native_ids=duplicate_native_ids,
        )
        blocks = _message_blocks(message)
        for block, identity in zip(blocks, block_content_identities(blocks), strict=True):
            block_id = archive_block_id(
                message_id,
                content_identity=identity.content_identity,
                content_occurrence=identity.content_occurrence,
            )
            if not replace_session and prior_rows:
                block_ids.append(block_id)
            for construct_position, construct in enumerate(block.web_constructs):
                rows.append(
                    (
                        session_id,
                        message_id,
                        block_id,
                        construct_position,
                        provider,
                        _enum_value(construct.construct_type),
                        _sqlite_text(construct.provider_key),
                        _sqlite_text(construct.title),
                        _sqlite_text(construct.url),
                        _sqlite_text(construct.text),
                        _sqlite_text(construct.source_id),
                        _sqlite_text(construct.group_id),
                        _sqlite_text(construct.group_title),
                        _sqlite_text(construct.query),
                        _sqlite_text(construct.asset_pointer),
                        _sqlite_text(construct.mime_type),
                        _sqlite_text(construct.status),
                        _sqlite_text(construct.task_id),
                        _sqlite_text(construct.task_type),
                        construct.rank,
                        construct.start_index,
                        construct.end_index,
                    )
                )
            if max(len(rows), len(block_ids)) >= 128:
                _flush_rows()
    _flush_rows()


def _merge_json_value(new: object, old: object, *, context: str = "") -> object:
    """Field-path union of two JSON-decoded values (polylogue-geop).

    An acquisition is a PARTIAL OBSERVATION: measured across 44,171 chatgpt
    messages present in both an April and a July export, 453,956 field
    observations were April-only, zero were July-only, and the 2,479
    "conflicts" were the newer export simply carrying fewer keys one level
    deeper (dropped ``start_idx``/``end_idx``/``matched_text``/... from
    citation records) -- never a genuine value disagreement. So the rule is:
    for each path, take whichever side has a non-null value; when both do,
    prefer the newer (``new``) side but log loudly if they disagree, since
    that case is not expected to occur for any provider observed so far.
    """
    if new is None:
        return old
    if old is None:
        return new
    if isinstance(new, dict) and isinstance(old, dict):
        merged: dict[object, object] = {}
        for key in dict.fromkeys((*old.keys(), *new.keys())):
            merged[key] = _merge_json_value(new.get(key), old.get(key), context=context)
        return merged
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        return [_merge_json_value(n, o, context=context) for n, o in zip(new, old, strict=True)]
    if new != old:
        logger.warning(
            "field-path union conflict at %s: newer=%r older=%r -- keeping newer (polylogue-geop)",
            context,
            new,
            old,
        )
    return new


def _coalesce_scalar(new: object, old: object) -> object:
    return new if new is not None else old


_SPLICE_FRONT: object = object()


def _splice_merge_keys(old_keys: Sequence[object], new_keys: Sequence[object]) -> list[tuple[str, object]]:
    """Merge two ordered key sequences, preserving the relative order of both.

    A key present in ``old_keys`` but not ``new_keys`` is spliced back in
    immediately after the nearest PRECEDING key common to both sequences (or
    at the very front if none precedes it) -- not appended after everything.
    This is what lets a reinjected message/block land back in its correct
    relative transcript position instead of scrambling order (polylogue-geop
    PR review): naive append turns old ``[user, tool, assistant]`` + new
    ``[user, assistant]`` (tool dropped) into stored ``[user, assistant,
    tool]``; this produces ``[user, tool, assistant]``.

    Returns ``(source, key)`` pairs in final order. ``source`` is ``"new"``
    for any key present in ``new_keys`` (including common ones -- the merge
    step still uses the incoming row as the coalesce base for those) and
    ``"old"`` for a key present only in ``old_keys``.
    """
    common = set(old_keys) & set(new_keys)
    vanished_after: dict[object, list[object]] = {}
    last_common: object = _SPLICE_FRONT
    for key in old_keys:
        if key in common:
            last_common = key
        else:
            vanished_after.setdefault(last_common, []).append(key)
    result: list[tuple[str, object]] = [("old", k) for k in vanished_after.get(_SPLICE_FRONT, [])]
    for key in new_keys:
        result.append(("new", key))
        if key in common:
            result.extend(("old", k) for k in vanished_after.get(key, []))
    return result


def _message_row_id(session_id: str, row: tuple[object, ...], idx: Mapping[str, int]) -> str:
    return archive_message_id(
        session_id,
        native_id_from_storage(
            cast("str | None", row[idx["native_id"]]),
            cast("str | None", row[idx["source_native_id_json"]]),
        ),
        content_identity=cast("str | None", row[idx["content_identity"]]),
        content_occurrence=cast(int, row[idx["content_occurrence"]]),
    )


def _message_semantic_key(row: tuple[object, ...], idx: Mapping[str, int]) -> tuple[object, object, object]:
    # Original names distinguish separate Source occurrences with identical
    # content; they are evidence, never a substitute for the semantic digest.
    return row[idx["content_identity"]], row[idx["content_occurrence"]], row[idx["source_native_id_json"]]


def _retain_message_identity(
    new: tuple[object, ...], old: tuple[object, ...], idx: Mapping[str, int]
) -> tuple[object, ...]:
    values = [_coalesce_scalar(n, o) for n, o in zip(new, old, strict=True)]
    # Reconciliation preserves the predecessor's stable owner even when
    # normalization of duplicate Source names changes the incoming identity.
    for name in ("native_id", "content_identity", "content_occurrence", "identity_source"):
        values[idx[name]] = old[idx[name]]
    if old[idx["source_native_id_json"]] is not None:
        values[idx["source_native_id_json"]] = old[idx["source_native_id_json"]]
    return tuple(values)


def _block_identity_key(row: tuple[object, ...], b_idx: dict[str, int]) -> tuple[object, object]:
    """The immutable Source identity used by both reconciliation implementations."""
    return row[b_idx["content_identity"]], row[b_idx["content_occurrence"]]


def _block_identity_keys(rows: list[tuple[object, ...]], b_idx: dict[str, int]) -> list[object]:
    """Match the same Source content, never a type occurrence or display slot."""
    return [_block_identity_key(row, b_idx) for row in rows]


#: One tool result's outcome, spread across a canonical column and its legacy
#: compatibility fields. They describe a single verdict, so a merge takes them
#: from one row together: coalescing them independently can pair a fresh
#: ``tool_outcome`` with a stale ``is_error``/``exit_code`` that contradicts it.
_TOOL_VERDICT_COLUMNS = (
    "tool_outcome",
    "tool_result_is_error",
    "tool_result_exit_code",
    "tool_result_outcome_unknown_reason",
)


def _expresses_tool_verdict(row: tuple[object, ...], b_idx: dict[str, int]) -> bool:
    """Report whether a row states a tool outcome at all."""
    return row[b_idx["tool_outcome"]] is not None


def _apply_tool_verdict(
    merged_values: list[object],
    new_row: tuple[object, ...],
    old_row: tuple[object, ...],
    b_idx: dict[str, int],
) -> None:
    """Copy one row's whole tool verdict into the merged row."""
    source = new_row if _expresses_tool_verdict(new_row, b_idx) else old_row
    for col_name in _TOOL_VERDICT_COLUMNS:
        idx = b_idx.get(col_name)
        if idx is not None:
            merged_values[idx] = source[idx]


def _coalesce_block_row(
    new_row: tuple[object, ...],
    old_row: tuple[object, ...],
    b_idx: dict[str, int],
    *,
    message_id: str,
    position: int,
) -> tuple[object, ...]:
    """Field-path coalesce of one matched block pair (shared by both the
    direct block-loop and the structural-identity reconciliation loop)."""
    merged_values: list[object] = list(new_row)
    conflict_context = f"blocks.tool_input message_id={message_id} position={position}"
    tool_input_idx = b_idx["tool_input"]
    for col_name, idx in b_idx.items():
        if col_name in ("message_id", "session_id", "position", "content_hash"):
            continue
        if col_name in _TOOL_VERDICT_COLUMNS:
            continue  # taken as one verdict below
        if col_name == "tool_input":
            new_raw = cast("str | None", new_row[idx])
            old_raw = cast("str | None", old_row[idx])
            new_json = json.loads(new_raw) if new_raw is not None else None
            old_json = json.loads(old_raw) if old_raw is not None else None
            merged_json = _merge_json_value(new_json, old_json, context=conflict_context)
            merged_values[idx] = _json_dumps(merged_json) if merged_json is not None else None
        else:
            merged_values[idx] = _coalesce_scalar(new_row[idx], old_row[idx])
    _apply_tool_verdict(merged_values, new_row, old_row, b_idx)
    is_error_value = merged_values[b_idx["tool_result_is_error"]]
    merged_values[b_idx["content_hash"]] = _block_content_hash(
        block_type=cast(str, merged_values[b_idx["block_type"]]),
        text=cast("str | None", merged_values[b_idx["text"]]),
        tool_name=cast("str | None", merged_values[b_idx["tool_name"]]),
        tool_input_json=cast("str | None", merged_values[tool_input_idx]),
        semantic_type=cast("str | None", merged_values[b_idx["semantic_type"]]),
        media_type=cast("str | None", merged_values[b_idx["media_type"]]),
        language=cast("str | None", merged_values[b_idx["language"]]),
        is_error=None if is_error_value is None else bool(is_error_value),
        exit_code=cast("int | None", merged_values[b_idx["tool_result_exit_code"]]),
        tool_outcome=cast("str | None", merged_values[b_idx["tool_outcome"]]),
        outcome_unknown_reason=cast("str | None", merged_values[b_idx["tool_result_outcome_unknown_reason"]]),
        semantic_extra_json=_hashed_semantic_extra_json(merged_values[b_idx["semantic_extra_json"]]),
    )
    return tuple(merged_values)


@dataclass(frozen=True, slots=True)
class _CapturedProjections:
    """Pre-delete snapshot of a session's evidence-dependent projection rows
    (polylogue-geop PR review P1: ``attachment_refs``/``paste_spans``/
    ``file_edits``/``web_content_constructs`` are rebuilt SOLELY from the
    incoming ``ParsedSession``'s domain objects on every full replace, never
    from the union'd/reinjected row tuples -- so a message or block the
    field-path union restores still silently loses its attachment, paste
    span, file-edit, or citation metadata unless that metadata is captured
    here before the delete and re-inserted after the incoming rebuild)."""

    attachment_refs: Sequence[tuple[object, ...]]
    attachment_native_ids: Sequence[tuple[object, ...]]
    paste_spans: Sequence[tuple[object, ...]]
    file_edits: Sequence[tuple[object, ...]]
    web_content_constructs: Sequence[tuple[object, ...]]
    provider_usage_events: Sequence[tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class _ProjectionCarryForward:
    captured: _CapturedProjections
    live_message_ids: Set[str]
    message_id_remap: Mapping[str, str | None]
    scratch: _UnionScratch | None = None
    incoming_message_id_remap: Mapping[str, str] = field(default_factory=dict)


class _UnionScratch:
    """Private indexed rows retained until the merged projections are restored."""

    def __init__(self, directory: Path) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-field-union-", dir=directory)
        self._closed = False
        self._lock = threading.RLock()
        self._path = Path(self._scratch.name) / "union.db"
        self._reader_owner: NativeSQLCustodyOwner | None = None
        self._owner: NativeSQLCustodyOwner | None = None
        self._conn: sqlite3.Connection | None = None
        try:
            self._owner = open_scratch_connection(self._path, lifetime_dependencies=(self,))
        except NativeConnectionSettlementError as failure:
            failure.owner.scratch_directory = self._scratch
            raise
        assert self._owner.connection is not None
        self._conn = self._owner.connection
        try:
            for name in ("old_message", "new_message", "old_block", "new_block", "merged_message", "merged_block"):
                self.conn.execute(
                    f"CREATE TABLE {name} (ordinal INTEGER PRIMARY KEY, key TEXT, owner TEXT, position INTEGER, row_blob BLOB NOT NULL)"
                )
                self.conn.execute(f"CREATE INDEX {name}_key ON {name}(key)")
                self.conn.execute(f"CREATE INDEX {name}_owner ON {name}(owner, position)")
            self.conn.execute("CREATE TABLE live_message (message_id TEXT PRIMARY KEY) WITHOUT ROWID")
            self.conn.execute("CREATE TABLE message_remap (old_id TEXT PRIMARY KEY, new_id TEXT) WITHOUT ROWID")
            self.conn.execute(
                "CREATE TABLE message_owner_map (key TEXT PRIMARY KEY, message_id TEXT NOT NULL) WITHOUT ROWID"
            )
            self.conn.execute(
                "CREATE TABLE old_only (anchor TEXT, old_ordinal INTEGER, native_id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            self.conn.execute("CREATE INDEX old_only_anchor ON old_only(anchor, old_ordinal)")
            for name in (
                "attachment_refs",
                "attachment_native_ids",
                "paste_spans",
                "file_edits",
                "web_content_constructs",
                "provider_usage_events",
            ):
                self.conn.execute(f"CREATE TABLE captured_{name} (ordinal INTEGER PRIMARY KEY, row_blob BLOB NOT NULL)")
            self.conn.execute("CREATE TABLE restored_attachment_ref (ref_id TEXT PRIMARY KEY) WITHOUT ROWID")
            self.conn.execute(
                "CREATE TABLE captured_attachment_owner (attachment_id TEXT NOT NULL, message_id TEXT NOT NULL)"
            )
            self.conn.execute("CREATE INDEX captured_attachment_owner_id ON captured_attachment_owner(attachment_id)")
            self.conn.execute("CREATE TABLE refresh_attachment (attachment_id TEXT PRIMARY KEY) WITHOUT ROWID")
            self.conn.execute("CREATE TABLE carried_attachment (attachment_id TEXT PRIMARY KEY) WITHOUT ROWID")
        except BaseException as primary:
            self._owner.scratch_directory = self._scratch
            _close_failed_native_construction(self._owner, primary)
            self.close()
            raise

    def put(
        self,
        table: str,
        ordinal: int,
        row: tuple[object, ...],
        *,
        key: str | None = None,
        owner: str | None = None,
        position: int | None = None,
    ) -> None:
        self.conn.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?)",
            (ordinal, key, owner, position, pickle.dumps(row, protocol=5)),
        )

    def row(self, table: str, *, key: str | None = None, ordinal: int | None = None) -> tuple[object, ...] | None:
        if key is not None:
            result = self.conn.execute(
                f"SELECT row_blob FROM {table} WHERE key = ? ORDER BY ordinal DESC LIMIT 1", (key,)
            ).fetchone()
        else:
            result = self.conn.execute(f"SELECT row_blob FROM {table} WHERE ordinal = ?", (ordinal,)).fetchone()
        return pickle.loads(result[0]) if result is not None else None

    def rows(self, table: str) -> _PickledRowSequence:
        return _PickledRowSequence(self, table)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("union SQL requires its creator-owned access scope")
        owner = self._owner if self._owner is not None else self._reader_owner
        if owner is None:
            raise RuntimeError("union access has no native owner")
        return owner.require_connection()

    def finish(self) -> None:
        """Close preparation SQL before transporting the artifact."""
        self.conn.commit()
        assert self._owner is not None
        self._owner.close()
        self._owner = None
        self._conn = None

    @contextmanager
    def access(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("union artifact is closed")
            if self._conn is not None:
                yield self.conn
                return
            uri = self._path.as_uri() + ("?mode=rw" if write else "?mode=ro")
            conn = connect_measured(uri, uri=True)
            owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=(*current_native_sql_lifetimes(), self))
            self._reader_owner = owner
            self._conn = conn
            try:
                yield owner.require_connection()
                if write:
                    conn.commit()
            finally:
                self._conn = None
                self._reader_owner = None
                owner.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._owner is not None:
                if self._owner in retained_native_sql_owners_for_lifetime(self):
                    self._owner.close()
                self._owner = None
                self._conn = None
            pending = retained_native_sql_owners_for_lifetime(self)
            if pending:
                raise NativeConnectionSettlementError(pending[0], RuntimeError("union artifact remains owned"))
            self._scratch.cleanup()
            self._closed = True

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


class _PickledRowSequence(Sequence[tuple[object, ...]]):
    def __init__(self, scratch: _UnionScratch, table: str) -> None:
        self.scratch, self.table = scratch, table

    def __len__(self) -> int:
        with self.scratch.access() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

    @overload
    def __getitem__(self, index: int) -> tuple[object, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[object, ...]]: ...

    def __getitem__(self, index: int | slice) -> tuple[object, ...] | list[tuple[object, ...]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = index + len(self) if index < 0 else index
        with self.scratch.access() as conn:
            row = conn.execute(f"SELECT row_blob FROM {self.table} WHERE ordinal = ?", (ordinal,)).fetchone()
        if row is None:
            raise IndexError(index)
        return cast(tuple[object, ...], pickle.loads(row[0]))

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        after = -1
        while True:
            with self.scratch.access() as conn:
                rows = conn.execute(
                    f"SELECT ordinal, row_blob FROM {self.table} WHERE ordinal > ? ORDER BY ordinal LIMIT 512", (after,)
                ).fetchall()
            if not rows:
                return
            after = int(rows[-1][0])
            yield from (pickle.loads(row[1]) for row in rows)


class _UnionMap(Mapping[str, str | None]):
    def __init__(self, scratch: _UnionScratch, table: str, key_col: str, value_col: str) -> None:
        self.scratch, self.table, self.key_col, self.value_col = scratch, table, key_col, value_col

    def __getitem__(self, key: str) -> str | None:
        with self.scratch.access() as conn:
            row = conn.execute(f"SELECT {self.value_col} FROM {self.table} WHERE {self.key_col} = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0]) if row[0] is not None else None

    def __iter__(self) -> Iterator[str]:
        after = ""
        first = True
        while True:
            with self.scratch.access() as conn:
                rows = conn.execute(
                    f"SELECT {self.key_col} FROM {self.table} WHERE ? OR {self.key_col} > ? "
                    f"ORDER BY {self.key_col} LIMIT 512",
                    (first, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            first = False
            yield from (str(row[0]) for row in rows)

    def __len__(self) -> int:
        with self.scratch.access() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])


class _UnionSet(Set[str]):
    def __init__(self, scratch: _UnionScratch, table: str, column: str) -> None:
        self.scratch, self.table, self.column = scratch, table, column

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self.scratch.access() as conn:
            return conn.execute(f"SELECT 1 FROM {self.table} WHERE {self.column} = ?", (key,)).fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        yield from _UnionMap(self.scratch, self.table, self.column, self.column)

    def __len__(self) -> int:
        with self.scratch.access() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])


def _capture_session_projection_rows(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    scratch: _UnionScratch | None = None,
    before_input: BeforeIndexInput | None = None,
) -> _CapturedProjections:
    if before_input is not None:
        fields = {
            "attachment_refs": (
                "attachment_id",
                "session_id",
                "message_id",
                "position",
                "upload_origin",
                "direction",
                "producer_ref",
                "source_url",
                "caption",
                "supplying_raw_id",
                "native_identity",
            ),
            "paste_spans": (
                "message_id",
                "session_id",
                "position",
                "start_offset",
                "end_offset",
                "boundary_state",
                "source_event_id",
                "source_marker",
                "content_hash",
                "observed_at_ms",
            ),
            "file_edits": (
                "tool_use_block_id",
                "session_id",
                "message_id",
                "file_path",
                "structured_patch_json",
                "original_file",
                "old_string",
                "new_string",
                "replace_all",
                "user_modified",
                "observed_at_ms",
            ),
            "web_content_constructs": (
                "session_id",
                "message_id",
                "block_id",
                "position",
                "provider",
                "construct_type",
                "provider_key",
                "title",
                "url",
                "text",
                "source_id",
                "group_id",
                "group_title",
                "query",
                "asset_pointer",
                "mime_type",
                "status",
                "task_id",
                "task_type",
                "rank",
                "start_index",
                "end_index",
            ),
            "session_provider_usage_events": _PROVIDER_USAGE_EVENT_COLUMNS,
        }
        for table, columns in fields.items():
            before_input(table, tuple(columns), f"SELECT rowid FROM {table} WHERE session_id=?", (session_id,))
        before_input(
            "attachment_native_ids",
            ("ref_id", "id_kind", "native_id"),
            "SELECT ani.rowid FROM attachment_native_ids ani JOIN attachment_refs ar "
            "ON ani.ref_id=ar.ref_id WHERE ar.session_id=?",
            (session_id,),
        )
    if scratch is not None:
        selections = {
            "attachment_refs": "SELECT attachment_id, session_id, message_id, position, upload_origin, direction, producer_ref, source_url, caption, supplying_raw_id, native_identity FROM attachment_refs WHERE session_id = ?",
            "paste_spans": "SELECT message_id, session_id, position, start_offset, end_offset, boundary_state, source_event_id, source_marker, content_hash, observed_at_ms FROM paste_spans WHERE session_id = ?",
            "file_edits": "SELECT tool_use_block_id, session_id, message_id, file_path, structured_patch_json, original_file, old_string, new_string, replace_all, user_modified, observed_at_ms FROM file_edits WHERE session_id = ?",
            "web_content_constructs": "SELECT session_id, message_id, block_id, position, provider, construct_type, provider_key, title, url, text, source_id, group_id, group_title, query, asset_pointer, mime_type, status, task_id, task_type, rank, start_index, end_index FROM web_content_constructs WHERE session_id = ?",
            "provider_usage_events": "SELECT "
            + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
            + " FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
        }
        for name, sql in selections.items():
            with connection_cursor(conn, sql, (session_id,)) as _input_cursor:
                for ordinal, row in enumerate(_input_cursor):
                    scratch.conn.execute(
                        f"INSERT INTO captured_{name} VALUES (?, ?)",
                        (ordinal, pickle.dumps(tuple(row), protocol=5)),
                    )
                    if name == "attachment_refs":
                        scratch.conn.execute("INSERT INTO captured_attachment_owner VALUES (?, ?)", (row[0], row[2]))
        with connection_cursor(
            conn,
            "SELECT ani.ref_id, ani.id_kind, ani.native_id FROM attachment_native_ids ani "
            "JOIN attachment_refs ar ON ani.ref_id = ar.ref_id "
            "WHERE ar.session_id = ?",
            (session_id,),
        ) as _input_cursor:
            for ordinal, row in enumerate(_input_cursor):
                scratch.conn.execute(
                    "INSERT INTO captured_attachment_native_ids VALUES (?, ?)",
                    (ordinal, pickle.dumps(tuple(row), protocol=5)),
                )
        return _CapturedProjections(
            *(
                scratch.rows(f"captured_{name}")
                for name in (
                    "attachment_refs",
                    "attachment_native_ids",
                    "paste_spans",
                    "file_edits",
                    "web_content_constructs",
                    "provider_usage_events",
                )
            )
        )
    with connection_cursor(
        conn,
        "SELECT attachment_id, session_id, message_id, position, upload_origin, direction, producer_ref, source_url, "
        "caption, supplying_raw_id, native_identity FROM attachment_refs WHERE session_id = ?",
        (session_id,),
    ) as _input_cursor:
        attachment_refs = _input_cursor.fetchall()
    ref_ids = [attachment_reference_id(str(row[2]), str(row[10])) for row in attachment_refs]
    attachment_native_ids: list[sqlite3.Row] = []
    if ref_ids:
        placeholders = ",".join("?" for _ in ref_ids)
        with connection_cursor(
            conn,
            f"SELECT ref_id, id_kind, native_id FROM attachment_native_ids WHERE ref_id IN ({placeholders})",
            ref_ids,
        ) as _input_cursor:
            attachment_native_ids = _input_cursor.fetchall()
    with connection_cursor(
        conn,
        "SELECT message_id, session_id, position, start_offset, end_offset, boundary_state, "
        "source_event_id, source_marker, content_hash, observed_at_ms FROM paste_spans WHERE session_id = ?",
        (session_id,),
    ) as _input_cursor:
        paste_spans = _input_cursor.fetchall()
    with connection_cursor(
        conn,
        "SELECT tool_use_block_id, session_id, message_id, file_path, structured_patch_json, "
        "original_file, old_string, new_string, replace_all, user_modified, observed_at_ms "
        "FROM file_edits WHERE session_id = ?",
        (session_id,),
    ) as _input_cursor:
        file_edits = _input_cursor.fetchall()
    with connection_cursor(
        conn,
        "SELECT session_id, message_id, block_id, position, provider, construct_type, provider_key, "
        "title, url, text, source_id, group_id, group_title, query, asset_pointer, mime_type, status, "
        "task_id, task_type, rank, start_index, end_index FROM web_content_constructs WHERE session_id = ?",
        (session_id,),
    ) as _input_cursor:
        web_content_constructs = _input_cursor.fetchall()
    with connection_cursor(
        conn,
        "SELECT "
        + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
        + " FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
        (session_id,),
    ) as _input_cursor:
        provider_usage_events = _input_cursor.fetchall()
    return _CapturedProjections(
        attachment_refs=[tuple(r) for r in attachment_refs],
        attachment_native_ids=[tuple(r) for r in attachment_native_ids],
        paste_spans=[tuple(r) for r in paste_spans],
        file_edits=[tuple(r) for r in file_edits],
        web_content_constructs=[tuple(r) for r in web_content_constructs],
        provider_usage_events=[tuple(r) for r in provider_usage_events],
    )


def _restore_captured_projection_rows(
    conn: sqlite3.Connection,
    carry_forward: _ProjectionCarryForward,
) -> None:
    """Re-insert a captured pre-delete projection row whose slot wasn't
    reclaimed by the incoming acquisition's own rebuild. Only rows whose
    owning message survived the merge (``live_message_ids``) are eligible --
    a message the field-path union deliberately did not reinject (the
    prefix-sharing-parent guard) must not have its sidecar evidence restored
    either. Block references keep immutable Source identities when display
    positions change. Attachment references retain exact native identity under
    their owning message; display position is not identity. Paste-span positions
    are independent per-message ordinals, unrelated to transcript order.
    """
    captured = carry_forward.captured
    live_message_ids = carry_forward.live_message_ids

    restored_attachment_ref_ids: set[str] = set()
    scratch = carry_forward.scratch
    if scratch is not None:
        scratch.conn.execute("DELETE FROM restored_attachment_ref")
    for row in captured.attachment_refs:
        message_id = cast(str, row[2])
        if message_id not in live_message_ids:
            continue
        exists = conn.execute(
            "SELECT 1 FROM attachment_refs WHERE message_id = ? AND native_identity = ?", (message_id, row[10])
        ).fetchone()
        if exists is None:
            # ``supplying_raw_id`` travels with the row: the reference comes
            # from the earlier acquisition that held it, not from this one.
            conn.execute(
                "INSERT OR IGNORE INTO attachment_refs "
                "(attachment_id, session_id, message_id, position, upload_origin, direction, producer_ref, source_url, "
                "caption, supplying_raw_id, native_identity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            ref_id = attachment_reference_id(message_id, str(row[10]))
            if scratch is None:
                restored_attachment_ref_ids.add(ref_id)
            else:
                scratch.conn.execute("INSERT OR IGNORE INTO restored_attachment_ref VALUES (?)", (ref_id,))

    for row in captured.attachment_native_ids:
        if row[0] in restored_attachment_ref_ids or (
            scratch is not None
            and scratch.conn.execute("SELECT 1 FROM restored_attachment_ref WHERE ref_id = ?", (row[0],)).fetchone()
            is not None
        ):
            conn.execute(
                "INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id) VALUES (?, ?, ?)",
                row,
            )

    for row in captured.paste_spans:
        message_id = cast(str, row[0])
        position = row[2]
        if message_id not in live_message_ids:
            continue
        exists = conn.execute(
            "SELECT 1 FROM paste_spans WHERE message_id = ? AND position = ?", (message_id, position)
        ).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO paste_spans "
                "(message_id, session_id, position, start_offset, end_offset, boundary_state, "
                "source_event_id, source_marker, content_hash, observed_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )

    for row in captured.file_edits:
        message_id = cast(str, row[2])
        if message_id not in live_message_ids:
            continue
        block_id = cast(str, row[0])
        exists = conn.execute("SELECT 1 FROM file_edits WHERE tool_use_block_id = ?", (block_id,)).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO file_edits "
                "(tool_use_block_id, session_id, message_id, file_path, structured_patch_json, "
                "original_file, old_string, new_string, replace_all, user_modified, observed_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (block_id, *row[1:]),
            )

    for row in captured.web_content_constructs:
        message_id = cast(str, row[1])
        if message_id not in live_message_ids:
            continue
        block_id = cast(str, row[2])
        position = row[3]
        exists = conn.execute(
            "SELECT 1 FROM web_content_constructs WHERE block_id = ? AND position = ?", (block_id, position)
        ).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO web_content_constructs ("
                "session_id, message_id, block_id, position, provider, construct_type, provider_key, "
                "title, url, text, source_id, group_id, group_title, query, asset_pointer, mime_type, status, "
                "task_id, task_type, rank, start_index, end_index"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row[0], message_id, block_id, *row[3:]),
            )


_PROVIDER_USAGE_EVENT_COLUMNS = (
    "session_id",
    "source_message_id",
    "position",
    "provider_event_type",
    "model_name",
    "last_input_tokens",
    "last_output_tokens",
    "last_cached_input_tokens",
    "last_cache_write_tokens",
    "last_reasoning_output_tokens",
    "last_total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_input_tokens",
    "total_cache_write_tokens",
    "total_reasoning_output_tokens",
    "total_tokens",
    "occurred_at_ms",
    # polylogue-1pzmq: this list is both the capture SELECT and the restore
    # INSERT, and the restore DELETEs the session's rows first -- so a column
    # missing here is not merely unmerged, it is erased from rows that already
    # held it. ``request_id`` was exactly that: written by the event writer,
    # then nulled by the first carry-forward reconciliation.
    "request_id",
    "source_message_provider_id",
    "source_message_resolution",
    "finish_reason",
    "api_block_index",
    "quota_limits_json",
)


def _provider_usage_event_key(
    row: tuple[object, ...],
    occurrence_by_base: dict[tuple[str, ...], int],
) -> tuple[str, ...]:
    values = dict(zip(_PROVIDER_USAGE_EVENT_COLUMNS, row, strict=True))
    stable = provider_usage_event_identity(values)
    if stable is not None:
        base: tuple[str, ...] = stable
    else:
        # An unanchored row has no provider-issued identity. Its event type,
        # model, and ordinal are the narrowest bounded reconciliation rule;
        # unmatched rows are retained rather than inferred into a duplicate.
        base = (
            "ambiguous",
            str(values.get("provider_event_type") or ""),
            str(values.get("model_name") or "").strip(),
        )
    occurrence = occurrence_by_base.get(base, 0)
    occurrence_by_base[base] = occurrence + 1
    return (*base, str(occurrence))


_USAGE_SOURCE_MESSAGE_INDEX = _PROVIDER_USAGE_EVENT_COLUMNS.index("source_message_id")
_USAGE_RESOLUTION_INDEX = _PROVIDER_USAGE_EVENT_COLUMNS.index("source_message_resolution")


def _carried_usage_source_message_id(
    conn: sqlite3.Connection,
    carry_forward: _ProjectionCarryForward,
    source_message_id: object,
) -> tuple[bool, str | None]:
    """Decide whether an older usage row survives the union, and its message.

    Returns ``(retained, message_id)``. A row with no message is session-scoped
    and always survives. A row whose message the union dropped -- remapped to
    nothing, or absent from the rows this write stored -- describes usage that
    is no longer part of the session, so it does not survive; keeping it with
    a NULL message would still count its tokens.
    """
    if source_message_id is None:
        return True, None
    message_id = carry_forward.message_id_remap.get(cast(str, source_message_id), cast(str, source_message_id))
    if message_id is None:
        return False, None
    if conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (message_id,)).fetchone() is None:
        return False, None
    return True, message_id


def _merge_provider_usage_event_rows(
    incoming: tuple[object, ...],
    existing: tuple[object, ...],
    *,
    carried_source_message_id: str | None,
) -> tuple[object, ...]:
    """Keep the richer observation for one reconciled usage-event identity.

    ``carried_source_message_id`` is the older row's message as it stands after
    the union (see ``_carried_usage_source_message_id``), never its raw stored
    id, which may name a message this write removed.
    """
    merged = list(incoming)
    # Nullable text/timestamp lanes: an acquisition that simply did not report
    # one keeps the older observation. ``source_message_resolution`` is NOT
    # NULL and states how the stored row is attributed: it is this write's,
    # unless the message comes from the older row, which resolved it.
    if merged[_USAGE_SOURCE_MESSAGE_INDEX] is None and carried_source_message_id is not None:
        merged[_USAGE_SOURCE_MESSAGE_INDEX] = carried_source_message_id
        merged[_USAGE_RESOLUTION_INDEX] = "resolved"
    for index in (4, 17, 18, 19, 21, 22, 23):
        if merged[index] is None:
            merged[index] = existing[index]
    # NULL means unreported. Across proven-distinct acquisitions retain an
    # earlier observation when the new acquisition omits it; an explicit zero
    # remains a measurement and must not become an earlier positive value.
    for index in range(5, 17):
        new_value = merged[index]
        old_value = existing[index]
        if new_value is None:
            merged[index] = old_value
    return tuple(merged)


def _restore_captured_provider_usage_rows(
    conn: sqlite3.Connection,
    carry_forward: _ProjectionCarryForward,
) -> None:
    """Union provider usage evidence after the incoming event write.

    Usage rows are a sibling typed projection, not part of the message/block
    union. Reconcile anchored rows by provider message/type/model and retain
    unmatched older rows at a fresh position, unless the union dropped the
    message an older row describes. This preserves richer evidence when a
    poorer acquisition omits it or reports zero, without turning a cumulative
    observation or model switch into an additive delta.
    """
    captured = carry_forward.captured.provider_usage_events
    if not captured:
        return
    if carry_forward.scratch is not None:
        _restore_captured_provider_usage_rows_disk(conn, carry_forward)
        return
    session_id = cast(str, captured[0][0])
    incoming_rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT "
            + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
            + " FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
    ]
    incoming_by_key: dict[tuple[str, ...], tuple[object, ...]] = {}
    incoming_key_order: list[tuple[str, ...]] = []
    incoming_occurrences: dict[tuple[str, ...], int] = {}
    for row in incoming_rows:
        key = _provider_usage_event_key(row, incoming_occurrences)
        incoming_by_key[key] = row
        incoming_key_order.append(key)
    captured_by_key: dict[tuple[str, ...], tuple[object, ...]] = {}
    captured_key_order: list[tuple[str, ...]] = []
    captured_occurrences: dict[tuple[str, ...], int] = {}
    for old_row in captured:
        key = _provider_usage_event_key(old_row, captured_occurrences)
        captured_by_key[key] = old_row
        captured_key_order.append(key)

    # Rebuild the tiny typed projection in merged event order.  Appending an
    # unmatched old cumulative row would make it look newer than an incoming
    # model-switch observation and would misattribute the session high-water.
    merged_rows: list[tuple[object, ...]] = []
    ordered_keys = _splice_merge_keys(
        cast(list[object], captured_key_order),
        cast(list[object], incoming_key_order),
    )
    for source, raw_key in ordered_keys:
        key = cast(tuple[str, ...], raw_key)
        if source == "new":
            row = incoming_by_key[key]
            matched_old_row = captured_by_key.get(key)
            if matched_old_row is not None:
                _retained, carried_message_id = _carried_usage_source_message_id(
                    conn, carry_forward, matched_old_row[_USAGE_SOURCE_MESSAGE_INDEX]
                )
                row = _merge_provider_usage_event_rows(
                    row, matched_old_row, carried_source_message_id=carried_message_id
                )
        else:
            row_list = list(captured_by_key[key])
            retained, carried_message_id = _carried_usage_source_message_id(
                conn, carry_forward, row_list[_USAGE_SOURCE_MESSAGE_INDEX]
            )
            if not retained:
                continue
            row_list[_USAGE_SOURCE_MESSAGE_INDEX] = carried_message_id
            row = tuple(row_list)
        row_list = list(row)
        row_list[0] = session_id
        row_list[2] = len(merged_rows)
        merged_rows.append(tuple(row_list))

    conn.execute("DELETE FROM session_provider_usage_events WHERE session_id = ?", (session_id,))
    conn.executemany(
        "INSERT INTO session_provider_usage_events ("
        + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" for _ in _PROVIDER_USAGE_EVENT_COLUMNS)
        + ")",
        merged_rows,
    )


def _restore_captured_provider_usage_rows_disk(
    conn: sqlite3.Connection, carry_forward: _ProjectionCarryForward
) -> None:
    """Use indexed scratch for the typed usage projection's splice order."""
    scratch = carry_forward.scratch
    assert scratch is not None
    db = scratch.conn
    captured = carry_forward.captured.provider_usage_events
    session_id = cast(str, captured[0][0])
    for name in ("old_usage", "new_usage"):
        db.execute(f"DROP TABLE IF EXISTS {name}")
        db.execute(f"CREATE TABLE {name} (ordinal INTEGER PRIMARY KEY, key TEXT UNIQUE, row_blob BLOB NOT NULL)")
    db.execute("DROP TABLE IF EXISTS usage_count")
    db.execute(
        "CREATE TABLE usage_count (side TEXT, base TEXT, next_ordinal INTEGER, PRIMARY KEY(side, base)) WITHOUT ROWID"
    )
    db.execute("DROP TABLE IF EXISTS old_usage_only")
    db.execute("CREATE TABLE old_usage_only (anchor TEXT, ordinal INTEGER PRIMARY KEY)")
    db.execute("CREATE INDEX old_usage_only_anchor ON old_usage_only(anchor, ordinal)")
    # Decided before the session's rows are deleted and while the merged
    # messages are readable: whether each older row survives, and its message.
    db.execute("DROP TABLE IF EXISTS old_usage_carry")
    db.execute("CREATE TABLE old_usage_carry (ordinal INTEGER PRIMARY KEY, retained INTEGER NOT NULL, message_id TEXT)")

    def spool(side: str, rows: Iterable[tuple[object, ...]]) -> None:
        for ordinal, row in enumerate(rows):
            if side == "old":
                retained, carried_message_id = _carried_usage_source_message_id(
                    conn, carry_forward, row[_USAGE_SOURCE_MESSAGE_INDEX]
                )
                db.execute(
                    "INSERT INTO old_usage_carry VALUES (?, ?, ?)",
                    (ordinal, int(retained), carried_message_id),
                )
            values = dict(zip(_PROVIDER_USAGE_EVENT_COLUMNS, row, strict=True))
            stable = provider_usage_event_identity(values)
            base = (
                stable
                if stable is not None
                else (
                    "ambiguous",
                    str(values.get("provider_event_type") or ""),
                    str(values.get("model_name") or "").strip(),
                )
            )
            base_text = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
            count_row = db.execute(
                "SELECT next_ordinal FROM usage_count WHERE side = ? AND base = ?", (side, base_text)
            ).fetchone()
            occurrence = int(count_row[0]) if count_row is not None else 0
            db.execute(
                "INSERT INTO usage_count VALUES (?, ?, ?) ON CONFLICT(side, base) "
                "DO UPDATE SET next_ordinal = excluded.next_ordinal",
                (side, base_text, occurrence + 1),
            )
            key = json.dumps((*base, str(occurrence)), ensure_ascii=False, separators=(",", ":"))
            db.execute(
                f"INSERT INTO {side}_usage VALUES (?, ?, ?)",
                (ordinal, key, pickle.dumps(row, protocol=5)),
            )

    spool("old", iter(captured))
    spool(
        "new",
        (
            tuple(row)
            for row in conn.execute(
                "SELECT "
                + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
                + " FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
                (session_id,),
            )
        ),
    )
    anchor: str | None = None
    for ordinal, key in db.execute("SELECT ordinal, key FROM old_usage ORDER BY ordinal"):
        if db.execute("SELECT 1 FROM new_usage WHERE key = ?", (key,)).fetchone() is not None:
            anchor = cast(str, key)
        else:
            db.execute("INSERT INTO old_usage_only VALUES (?, ?)", (anchor, ordinal))

    def ordered_rows() -> Iterator[tuple[object, ...]]:
        position = 0

        def emit_old(anchor_key: str | None) -> Iterator[tuple[object, ...]]:
            nonlocal position
            for blob, carried_message_id in db.execute(
                "SELECT old_usage.row_blob, old_usage_carry.message_id FROM old_usage_only "
                "JOIN old_usage USING (ordinal) JOIN old_usage_carry USING (ordinal) "
                "WHERE anchor IS ? AND old_usage_carry.retained = 1 ORDER BY ordinal",
                (anchor_key,),
            ):
                values = list(pickle.loads(blob))
                values[_USAGE_SOURCE_MESSAGE_INDEX] = carried_message_id
                values[0], values[2] = session_id, position
                position += 1
                yield tuple(values)

        yield from emit_old(None)
        for key, blob in db.execute("SELECT key, row_blob FROM new_usage ORDER BY ordinal"):
            values = pickle.loads(blob)
            old = db.execute(
                "SELECT old_usage.row_blob, old_usage_carry.message_id FROM old_usage "
                "JOIN old_usage_carry USING (ordinal) WHERE old_usage.key = ?",
                (key,),
            ).fetchone()
            if old is not None:
                values = _merge_provider_usage_event_rows(
                    values, pickle.loads(old[0]), carried_source_message_id=old[1]
                )
            row = list(values)
            row[0], row[2] = session_id, position
            position += 1
            yield tuple(row)
            if old is not None:
                yield from emit_old(cast(str, key))

    conn.execute("DELETE FROM session_provider_usage_events WHERE session_id = ?", (session_id,))
    conn.executemany(
        "INSERT INTO session_provider_usage_events ("
        + ", ".join(_PROVIDER_USAGE_EVENT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" for _ in _PROVIDER_USAGE_EVENT_COLUMNS)
        + ")",
        ordered_rows(),
    )


def _union_with_existing_rows(
    conn: sqlite3.Connection,
    session_id: str,
    message_rows: list[tuple[object, ...]],
    block_rows: list[tuple[object, ...]],
    *,
    raw_id: str | None,
    existing_raw_id: str | None,
    force_replace: bool = False,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], _ProjectionCarryForward | None]:
    """Field-path union coalesce of a fresh full-replace against what is stored.

    polylogue-geop: newer provider exports are not supersets of older ones
    (measured: a 2026-07 chatgpt export dropped the entire tool/system role
    layer and 20K+ code blocks present in the 2026-04 export of the same
    conversations). A full-session replace must therefore never let a
    newer, poorer acquisition delete a field -- or a whole message/block --
    that an older acquisition already supplied. This runs *before* the
    session's old ``messages``/``blocks`` rows are deleted so it can read
    them, then returns the merged row tuples for ``_write_messages``/
    ``_write_blocks`` to insert in place of the plain freshly-built rows.

    THIS ONLY APPLIES ACROSS TWO DIFFERENT ACQUISITIONS, never within one.
    A full-session replace has two structurally different causes that this
    function must not conflate:

      - Two DIFFERENT acquisitions of the same logical session (a second
        export download, a later browser capture) are independent partial
        observations of one underlying reality -- neither is authoritative,
        so they union. This is the measured chatgpt case above.
      - A RE-PARSE of the SAME acquisition (unchanged raw bytes, a parser
        bugfix or corrected message-boundary logic) is not a second
        observation -- it is a better reading of the same evidence, and
        must be able to REPLACE a value the old parse got wrong. Unioning
        here would make every historical mis-parse immortal and defeat the
        entire point of a `reprocess` re-run.

    ``raw_id`` is the discriminator: it is content-addressed (SHA-256 of the
    acquired bytes, `polylogue/pipeline/services/acquisition_records.py`),
    so the SAME raw file re-parsed via `reprocess` (parse-only, no
    re-acquire) always carries the identical `raw_id` its original ingest
    did, while a genuinely different acquisition (different export
    generation, different capture) gets a different one. When both the
    incoming `raw_id` and the session's currently-stored `sessions.raw_id`
    are known and equal, this is a same-acquisition re-parse: return the
    rows unchanged (ordinary replace, exactly as before this change) with no
    field union and no reinjection. Union only fires when both are known and
    differ -- proven different acquisitions. When either side is unknown
    (`None` -- e.g. an index-only fixture writer that never threads a raw_id
    through), there is no positive evidence of a different
    acquisition, so this also falls back to plain replace rather than
    guessing; approximating "unknown" as "different" would let a corrected
    re-parse's retraction be silently defeated by union whenever a caller
    doesn't happen to supply provenance.

    Every Source message has stable semantic digest and occurrence evidence.
    Match canonical native identities first, then exact semantic occurrences
    where duplicate-name normalization changed native versus content identity.

    Four things beyond plain field coalescing (PR #3413 review):

      1. Reinjecting a vanished message/block preserves its RELATIVE
         transcript order via ``_splice_merge_keys`` -- appending it after
         the incoming maximum would silently reorder the conversation
         (worse than dropping it, since it's invisible).
      2. Blocks are matched by content-addressed structural identity
         (``_block_identity_keys`` -- exact Source digest and occurrence), not raw position, since dropping a non-trailing block
         shifts everything after it.
      3. Sidecar projection rows (``attachment_refs``/``paste_spans``/
         ``file_edits``/``web_content_constructs``) tied to reinjected or
         reconciled evidence are captured here and restored by the caller
         after its own incoming-only rebuild -- see ``_ProjectionCarryForward``.
      4. A matched message's block-derived flags (``has_tool_use``/
         ``has_thinking``) and ``content_hash`` are recomputed from its
         FINAL block set, not left describing only the incoming
         acquisition's blocks.
    """
    # ``existing_raw_id`` must be captured by the CALLER before its own
    # sessions upsert overwrites ``sessions.raw_id`` -- by the time this
    # function runs, a fresh re-query here would always see this write's own
    # value and could never detect a difference. See
    # ``_replace_full_session_messages_and_blocks``'s docstring.
    if force_replace or raw_id is None or existing_raw_id is None or existing_raw_id == raw_id:
        # Same acquisition re-parsed, provenance unknown on either side, or
        # the caller already made its own authoritative precedence call
        # (force_replace): ordinary replace, no union -- a corrected
        # re-parse (or a caller-decided supersession) must be able to
        # retract content the old parse wrongly produced.
        return message_rows, block_rows, None

    m_spec = archive_tiers_specs.MESSAGES_SPEC
    b_spec = archive_tiers_specs.BLOCKS_SPEC
    m_cols = [col.name for col in m_spec.writable_columns if col.extract_placeholder == "?"]
    b_cols = [col.name for col in b_spec.writable_columns if col.extract_placeholder == "?"]
    m_idx = {name: i for i, name in enumerate(m_cols)}
    b_idx = {name: i for i, name in enumerate(b_cols)}

    existing_message_rows = conn.execute(
        f"SELECT {', '.join(m_cols)} FROM messages WHERE session_id = ?", (session_id,)
    ).fetchall()
    existing_by_key = {_message_row_id(session_id, tuple(row), m_idx): tuple(row) for row in existing_message_rows}
    if not existing_by_key:
        return message_rows, block_rows, None
    old_native = {key: key for key, row in existing_by_key.items() if row[m_idx["native_id"]] is not None}
    old_semantic = {_message_semantic_key(row, m_idx): key for key, row in existing_by_key.items()}
    new_message_keys: list[str] = []
    incoming_remap: dict[str, str] = {}
    for row in message_rows:
        incoming_id = _message_row_id(session_id, row, m_idx)
        key = old_native.get(incoming_id)
        semantic_key = old_semantic.get(_message_semantic_key(row, m_idx))
        if (
            key is None
            and semantic_key is not None
            and (row[m_idx["native_id"]] is None or existing_by_key[semantic_key][m_idx["native_id"]] is None)
        ):
            key = semantic_key
        key = key or incoming_id
        new_message_keys.append(key)
        incoming_remap[incoming_id] = key

    existing_block_rows = conn.execute(
        f"SELECT {', '.join(b_cols)} FROM blocks WHERE session_id = ?", (session_id,)
    ).fetchall()
    existing_blocks_by_message: dict[str, list[tuple[object, ...]]] = {}
    for erow in existing_block_rows:
        row_message_id = cast(str, erow[b_idx["message_id"]])
        existing_blocks_by_message.setdefault(row_message_id, []).append(tuple(erow))
    for rows in existing_blocks_by_message.values():
        rows.sort(key=lambda r: cast(int, r[b_idx["position"]]))

    native_idx = m_idx["native_id"]
    position_idx = m_idx["position"]
    variant_idx = m_idx["variant_index"]
    has_tool_use_idx = m_idx["has_tool_use"]
    has_thinking_idx = m_idx["has_thinking"]
    message_content_hash_idx = m_idx["content_hash"]
    block_position_idx = b_idx["position"]

    # A session that is itself the resolved parent of a prefix-sharing child
    # has a load-bearing message set: the child's stored
    # `branch_point_message_id` names a specific message this session must
    # still contain, or composition falls back to "dangling" (by design --
    # see session_links' docstring and _extract_prefix_tail). Reinjecting a
    # message this acquisition intentionally dropped (e.g. a corrected
    # variant cut) would silently resurrect a branch point and fabricate a
    # prefix the operator's own re-ingest just removed. Whole-message
    # reinjection is skipped for such parents; field-path union of messages
    # and blocks BOTH acquisitions still assert is unaffected (it never
    # changes which messages exist).
    is_prefix_sharing_parent = (
        conn.execute(
            "SELECT 1 FROM session_links WHERE resolved_dst_session_id = ? AND inheritance = 'prefix-sharing' "
            f"AND {topology_status_composes_sql()} LIMIT 1",
            (session_id,),
        ).fetchone()
        is not None
    )

    # --- Message-level reconciliation: preserve relative transcript order ---
    # (PR review P1 write.py:2359) A vanished message must be spliced back
    # into its correct relative position, not appended after the incoming
    # maximum -- see `_splice_merge_keys`.
    old_message_order = sorted(existing_by_key, key=lambda key: cast(int, existing_by_key[key][position_idx]))
    incoming_row_by_key = dict(zip(new_message_keys, message_rows, strict=True))
    matched_message_keys = existing_by_key.keys() & incoming_row_by_key.keys()
    splice_old_keys: list[object] = [] if is_prefix_sharing_parent else list(old_message_order)
    merged_message_rows: list[tuple[object, ...]] = []
    merged_message_ids: dict[str, str] = {}
    for new_pos, (source, raw_key) in enumerate(_splice_merge_keys(splice_old_keys, new_message_keys)):
        key = cast(str, raw_key)
        if source == "new":
            row = incoming_row_by_key[key]
            if key in existing_by_key:
                row = _retain_message_identity(row, existing_by_key[key], m_idx)
        else:
            row = existing_by_key[key]
        values = list(row)
        values[position_idx] = new_pos
        merged_message_rows.append(tuple(values))
        merged_message_ids[key] = key
    live_message_ids = frozenset(merged_message_ids.values())

    # --- Block-level reconciliation, per still-live message ---
    # (PR review P1 write.py:2383) Matching by raw position mismatches once a
    # non-trailing block is dropped -- match by content-addressed structural
    # identity instead (`_block_identity_keys`), then splice-merge exactly
    # like messages above.
    incoming_blocks_by_message: dict[str, list[tuple[object, ...]]] = {}
    for row in block_rows:
        values = list(row)
        owner = cast(str, row[b_idx["message_id"]])
        values[b_idx["message_id"]] = incoming_remap.get(owner, owner)
        incoming_blocks_by_message.setdefault(cast(str, values[b_idx["message_id"]]), []).append(tuple(values))
    for rows in incoming_blocks_by_message.values():
        rows.sort(key=lambda r: cast(int, r[block_position_idx]))

    matched_message_ids = incoming_blocks_by_message.keys() & existing_blocks_by_message.keys()
    merged_block_rows: list[tuple[object, ...]] = []
    final_blocks_by_message: dict[str, list[tuple[object, ...]]] = {}

    for message_id, new_rows in incoming_blocks_by_message.items():
        if message_id not in matched_message_ids:
            merged_block_rows.extend(new_rows)
            final_blocks_by_message[message_id] = new_rows
            continue
        old_rows = existing_blocks_by_message[message_id]
        new_keys = _block_identity_keys(new_rows, b_idx)
        old_keys = _block_identity_keys(old_rows, b_idx)
        new_row_by_key = dict(zip(new_keys, new_rows, strict=True))
        old_row_by_key = dict(zip(old_keys, old_rows, strict=True))

        final_rows: list[tuple[object, ...]] = []
        for new_block_pos, (source, block_key) in enumerate(_splice_merge_keys(old_keys, new_keys)):
            if source == "new":
                row = new_row_by_key[block_key]
                if block_key in old_row_by_key:
                    row = _coalesce_block_row(
                        row, old_row_by_key[block_key], b_idx, message_id=message_id, position=new_block_pos
                    )
            else:
                row = old_row_by_key[block_key]
                logger.info(
                    "field-path union (polylogue-geop): reinjecting block message_id=%s "
                    "dropped by newer acquisition, restored at relative position %s",
                    message_id,
                    new_block_pos,
                )
            row_list = list(row)
            row_list[block_position_idx] = new_block_pos
            final_rows.append(tuple(row_list))
        merged_block_rows.extend(final_rows)
        final_blocks_by_message[message_id] = final_rows

    # Blocks for a wholesale-reinjected message (old-only, no incoming block
    # sequence to reconcile against) carry over verbatim, unchanged position.
    for message_id, old_rows in existing_blocks_by_message.items():
        if message_id in matched_message_ids:
            continue
        if message_id not in live_message_ids:
            continue  # is_prefix_sharing_parent skip: message truly not reinjected
        merged_block_rows.extend(old_rows)
        final_blocks_by_message[message_id] = old_rows

    # --- Recompute block-derived message metadata for reconciled messages ---
    # (PR review P2 write.py:2334) A matched message's has_tool_use/
    # has_thinking flags and content_hash must reflect its FINAL block set,
    # not whatever the incoming acquisition alone reported.
    recomputed_message_rows: list[tuple[object, ...]] = []
    for row in merged_message_rows:
        nid = native_id_from_storage(
            cast("str | None", row[native_idx]), cast("str | None", row[m_idx["source_native_id_json"]])
        )
        message_id = _message_row_id(session_id, row, m_idx)
        if message_id not in matched_message_keys:
            recomputed_message_rows.append(row)
            continue
        final_blocks = final_blocks_by_message.get(message_id, [])
        row_list = list(row)
        row_list[has_tool_use_idx] = (
            1 if any(b[b_idx["block_type"]] == BlockType.TOOL_USE.value for b in final_blocks) else 0
        )
        row_list[has_thinking_idx] = (
            1 if any(b[b_idx["block_type"]] == BlockType.THINKING.value for b in final_blocks) else 0
        )
        row_list[message_content_hash_idx] = _message_row_hash(
            session_id,
            nid,
            cast(int, row_list[position_idx]),
            cast(int, row_list[variant_idx] or 0),
            _row_fields_digest(row_list, m_idx),
            _stored_block_hash_parts(final_blocks, b_idx),
        )
        recomputed_message_rows.append(tuple(row_list))

    # --- Capture sidecar projection rows for restoration after the incoming
    # rebuild (PR review P1 write.py:2433) ---
    message_id_remap = {key: merged_message_ids.get(key) for key in existing_by_key}
    carry_forward = _ProjectionCarryForward(
        captured=_capture_session_projection_rows(conn, session_id),
        live_message_ids=live_message_ids,
        message_id_remap=message_id_remap,
        incoming_message_id_remap=incoming_remap,
    )

    return recomputed_message_rows, merged_block_rows, carry_forward


def _prepare_cross_acquisition_union(
    conn: sqlite3.Connection,
    session_id: str,
    incoming: PreparedSessionRows,
    *,
    raw_id: str,
    directory: Path,
    before_input: BeforeIndexInput | None = None,
) -> _PreparedCrossAcquisitionUnion | None:
    """Reconcile a different acquisition on a read-only index snapshot.

    The predecessor's canonical rows and sidecars live in private indexed
    scratch. Only one message's block sequence is decoded at a time. The
    returned predecessor binding is checked again by the publisher before it
    deletes any row; a stale snapshot takes the normal prepared retry route.
    """
    if before_input is not None:
        before_input(
            "sessions",
            (
                "raw_id",
                "content_hash",
                "parser_fingerprint",
                "lowering_fingerprint",
                "parent_session_id",
                "active_leaf_message_id",
            ),
            "SELECT rowid FROM sessions WHERE session_id=?",
            (session_id,),
        )
    with connection_cursor(
        conn,
        "SELECT raw_id, content_hash, parser_fingerprint, lowering_fingerprint, "
        "parent_session_id, active_leaf_message_id FROM sessions WHERE session_id = ?",
        (session_id,),
    ) as _input_cursor:
        predecessor_row = _input_cursor.fetchone()
    if predecessor_row is None or predecessor_row[0] is None or predecessor_row[0] == raw_id:
        return None
    predecessor = tuple(predecessor_row)
    m_cols = [col.name for col in archive_tiers_specs.MESSAGES_SPEC.writable_columns if col.extract_placeholder == "?"]
    b_cols = [col.name for col in archive_tiers_specs.BLOCKS_SPEC.writable_columns if col.extract_placeholder == "?"]
    mi = {name: index for index, name in enumerate(m_cols)}
    bi = {name: index for index, name in enumerate(b_cols)}
    if before_input is not None:
        before_input(
            "messages",
            tuple(m_cols),
            "SELECT rowid FROM messages WHERE session_id=? ORDER BY position,variant_index",
            (session_id,),
        )
        before_input(
            "blocks",
            tuple(b_cols),
            "SELECT rowid FROM blocks WHERE session_id=? ORDER BY message_id,position",
            (session_id,),
        )
    scratch = _UnionScratch(directory)
    try:
        scratch.conn.execute(
            "CREATE TABLE old_semantic (digest TEXT, occurrence INTEGER, key TEXT PRIMARY KEY, native TEXT, source_name TEXT) WITHOUT ROWID"
        )
        scratch.conn.execute("CREATE INDEX old_semantic_native ON old_semantic(native)")
        scratch.conn.execute("CREATE INDEX old_semantic_key ON old_semantic(digest, occurrence, source_name)")
        scratch.conn.execute(
            "CREATE TABLE incoming_message_remap (old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL) WITHOUT ROWID"
        )
        with connection_cursor(
            conn,
            f"SELECT {', '.join(m_cols)} FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (session_id,),
        ) as _input_cursor:
            for ordinal, raw_row in enumerate(_input_cursor):
                row = tuple(raw_row)
                key = _message_row_id(session_id, row, mi)
                scratch.put("old_message", ordinal, row, key=key, position=cast(int, row[mi["position"]]))
                scratch.conn.execute(
                    "INSERT INTO old_semantic VALUES (?, ?, ?, ?, ?)",
                    (
                        row[mi["content_identity"]],
                        row[mi["content_occurrence"]],
                        key,
                        key if row[mi["native_id"]] is not None else None,
                        row[mi["source_native_id_json"]],
                    ),
                )
        if not len(scratch.rows("old_message")):
            scratch.close()
            return None
        for ordinal, row in enumerate(incoming.message_rows):
            incoming_id = _message_row_id(session_id, row, mi)
            native = row[mi["native_id"]]
            matched = (
                scratch.conn.execute("SELECT key FROM old_semantic WHERE native = ?", (incoming_id,)).fetchone()
                if native is not None
                else None
            )
            if matched is None:
                matched = scratch.conn.execute(
                    "SELECT key FROM old_semantic WHERE digest = ? AND occurrence = ? AND source_name IS ? AND (native IS NULL OR ? IS NULL)",
                    (*_message_semantic_key(row, mi), native),
                ).fetchone()
            key = str(matched[0]) if matched is not None else incoming_id
            scratch.put("new_message", ordinal, tuple(row), key=key, position=ordinal)
            scratch.conn.execute("INSERT INTO incoming_message_remap VALUES (?, ?)", (incoming_id, key))
        with connection_cursor(
            conn,
            f"SELECT {', '.join(b_cols)} FROM blocks WHERE session_id = ? ORDER BY message_id, position",
            (session_id,),
        ) as _input_cursor:
            for ordinal, row in enumerate(_input_cursor):
                scratch.put(
                    "old_block",
                    ordinal,
                    tuple(row),
                    owner=cast(str, row[bi["message_id"]]),
                    position=cast(int, row[bi["position"]]),
                )
        for ordinal, row in enumerate(incoming.block_rows):
            values = list(row)
            owner = cast(str, row[bi["message_id"]])
            remapped = scratch.conn.execute(
                "SELECT new_id FROM incoming_message_remap WHERE old_id = ?", (owner,)
            ).fetchone()
            values[bi["message_id"]] = remapped[0] if remapped is not None else owner
            scratch.put(
                "new_block",
                ordinal,
                tuple(values),
                owner=cast(str, values[bi["message_id"]]),
                position=cast(int, row[bi["position"]]),
            )
        with connection_cursor(
            conn,
            "SELECT 1 FROM session_links WHERE resolved_dst_session_id = ? AND inheritance = 'prefix-sharing' "
            f"AND {topology_status_composes_sql()} LIMIT 1",
            (session_id,),
        ) as cursor:
            parent_guard = cursor.fetchone() is not None

        # The old-only rows retain their old relative order, anchored after
        # the nearest preceding exact message key present in the new evidence.
        anchor: str | None = None
        if not parent_guard:
            old_native_rows = scratch.conn.execute(
                "SELECT key, ordinal FROM old_message WHERE key IS NOT NULL "
                "AND ordinal IN (SELECT MAX(ordinal) FROM old_message GROUP BY key) ORDER BY position"
            )
            for native, old_ordinal in old_native_rows:
                present = (
                    scratch.conn.execute("SELECT 1 FROM new_message WHERE key = ? LIMIT 1", (native,)).fetchone()
                    is not None
                )
                if present:
                    anchor = cast(str, native)
                else:
                    scratch.conn.execute("INSERT INTO old_only VALUES (?, ?, ?)", (anchor, old_ordinal, native))

        next_message = 0

        def add_message(row: tuple[object, ...], key: str) -> None:
            nonlocal next_message
            values = list(row)
            values[mi["position"]] = next_message
            message_id = _message_row_id(session_id, tuple(values), mi)
            scratch.put("merged_message", next_message, tuple(values), key=key, owner=message_id, position=next_message)
            scratch.conn.execute("INSERT OR IGNORE INTO live_message VALUES (?)", (message_id,))
            scratch.conn.execute("INSERT OR REPLACE INTO message_owner_map VALUES (?, ?)", (key, message_id))
            next_message += 1

        def add_old_after(anchor_key: str | None) -> None:
            for (old_ordinal,) in scratch.conn.execute(
                "SELECT old_ordinal FROM old_only WHERE anchor IS ? ORDER BY old_ordinal", (anchor_key,)
            ):
                old_row = scratch.row("old_message", ordinal=old_ordinal)
                assert old_row is not None
                add_message(old_row, _message_row_id(session_id, old_row, mi))

        add_old_after(None)
        for ordinal, row in enumerate(incoming.message_rows):
            key = str(scratch.conn.execute("SELECT key FROM new_message WHERE ordinal = ?", (ordinal,)).fetchone()[0])
            old_row = scratch.row("old_message", key=key)
            add_message(_retain_message_identity(tuple(row), old_row, mi) if old_row is not None else tuple(row), key)
            if old_row is not None:
                add_old_after(key)

        # Structural keys and splice anchors stay indexed even when one
        # message contains an exceptionally long block sequence.
        next_block = 0

        def owner_rows(table: str, owner: str) -> Iterator[tuple[object, ...]]:
            for (blob,) in scratch.conn.execute(
                f"SELECT row_blob FROM {table} WHERE owner = ? ORDER BY position", (owner,)
            ):
                yield pickle.loads(blob)

        def add_block(row: tuple[object, ...]) -> None:
            nonlocal next_block
            scratch.put(
                "merged_block",
                next_block,
                row,
                owner=cast(str, row[bi["message_id"]]),
                position=cast(int, row[bi["position"]]),
            )
            next_block += 1

        for side in ("old", "new"):
            scratch.conn.execute(
                f"CREATE TABLE {side}_block_structural (ordinal INTEGER PRIMARY KEY, key BLOB NOT NULL, position INTEGER NOT NULL)"
            )
            scratch.conn.execute(f"CREATE INDEX {side}_block_structural_key ON {side}_block_structural(key)")
        scratch.conn.execute("CREATE TABLE old_block_only (ordinal INTEGER PRIMARY KEY, anchor BLOB)")
        scratch.conn.execute("CREATE INDEX old_block_only_anchor ON old_block_only(anchor, ordinal)")

        def indexed_block_keys(side: str, owner: str) -> None:
            for ordinal, position, blob in scratch.conn.execute(
                f"SELECT ordinal, position, row_blob FROM {side}_block WHERE owner = ? ORDER BY position", (owner,)
            ):
                row = pickle.loads(blob)
                key = _block_identity_key(row, bi)
                scratch.conn.execute(
                    f"INSERT INTO {side}_block_structural VALUES (?, ?, ?)",
                    (ordinal, pickle.dumps(key, protocol=5), position),
                )

        def keyed_block(side: str, key: bytes) -> tuple[tuple[object, ...], int] | None:
            found = scratch.conn.execute(
                f"SELECT ordinal, position FROM {side}_block_structural WHERE key = ? ORDER BY ordinal DESC LIMIT 1",
                (key,),
            ).fetchone()
            if found is None:
                return None
            row = scratch.row(f"{side}_block", ordinal=found[0])
            assert row is not None
            return row, cast(int, found[1])

        for (message_id,) in scratch.conn.execute("SELECT owner FROM new_block GROUP BY owner ORDER BY MIN(ordinal)"):
            old_exists = (
                scratch.conn.execute("SELECT 1 FROM old_block WHERE owner = ? LIMIT 1", (message_id,)).fetchone()
                is not None
            )
            if not old_exists:
                for row in owner_rows("new_block", message_id):
                    add_block(row)
                continue
            scratch.conn.execute("DELETE FROM old_block_structural")
            scratch.conn.execute("DELETE FROM new_block_structural")
            scratch.conn.execute("DELETE FROM old_block_only")
            indexed_block_keys("old", message_id)
            indexed_block_keys("new", message_id)
            anchor_key: bytes | None = None
            for ordinal, key in scratch.conn.execute("SELECT ordinal, key FROM old_block_structural ORDER BY ordinal"):
                if keyed_block("new", key) is not None:
                    anchor_key = key
                else:
                    scratch.conn.execute("INSERT INTO old_block_only VALUES (?, ?)", (ordinal, anchor_key))

            block_position = 0

            def emit_old(anchor: bytes | None, *, owner: str = message_id) -> None:
                nonlocal block_position
                for (ordinal,) in scratch.conn.execute(
                    "SELECT ordinal FROM old_block_only WHERE anchor IS ? ORDER BY ordinal", (anchor,)
                ):
                    old_key = scratch.conn.execute(
                        "SELECT key FROM old_block_structural WHERE ordinal = ?", (ordinal,)
                    ).fetchone()[0]
                    previous_row = keyed_block("old", old_key)
                    assert previous_row is not None
                    row, _previous = previous_row
                    values = list(row)
                    values[bi["position"]] = block_position
                    add_block(tuple(values))
                    block_position += 1

            emit_old(None)
            for (key,) in scratch.conn.execute("SELECT key FROM new_block_structural ORDER BY ordinal"):
                new_pair = keyed_block("new", key)
                assert new_pair is not None
                row = new_pair[0]
                old_pair = keyed_block("old", key)
                if old_pair is not None:
                    row = _coalesce_block_row(row, old_pair[0], bi, message_id=message_id, position=block_position)
                values = list(row)
                values[bi["position"]] = block_position
                add_block(tuple(values))
                block_position += 1
                if old_pair is not None:
                    emit_old(key)
        for (message_id,) in scratch.conn.execute("SELECT owner FROM old_block GROUP BY owner ORDER BY MIN(ordinal)"):
            if scratch.conn.execute("SELECT 1 FROM new_block WHERE owner = ? LIMIT 1", (message_id,)).fetchone():
                continue
            if not scratch.conn.execute("SELECT 1 FROM live_message WHERE message_id = ?", (message_id,)).fetchone():
                continue
            for row in owner_rows("old_block", message_id):
                add_block(row)

        for ordinal, native, message_id, blob in scratch.conn.execute(
            "SELECT ordinal, key, owner, row_blob FROM merged_message WHERE key IS NOT NULL"
        ):
            if (
                not scratch.conn.execute("SELECT 1 FROM old_message WHERE key = ?", (native,)).fetchone()
                or not scratch.conn.execute("SELECT 1 FROM new_message WHERE key = ?", (native,)).fetchone()
            ):
                continue
            rehashed_row = list(cast(tuple[object, ...], pickle.loads(blob)))
            rehashed_row[mi["has_tool_use"]] = int(
                any(
                    block[bi["block_type"]] == BlockType.TOOL_USE.value
                    for block in owner_rows("merged_block", message_id)
                )
            )
            rehashed_row[mi["has_thinking"]] = int(
                any(
                    block[bi["block_type"]] == BlockType.THINKING.value
                    for block in owner_rows("merged_block", message_id)
                )
            )
            rehashed_row[mi["content_hash"]] = _message_row_hash(
                session_id,
                native_id_from_storage(
                    cast("str | None", rehashed_row[mi["native_id"]]),
                    cast("str | None", rehashed_row[mi["source_native_id_json"]]),
                ),
                cast(int, rehashed_row[mi["position"]]),
                cast(int, rehashed_row[mi["variant_index"]] or 0),
                _row_fields_digest(rehashed_row, mi),
                _stored_block_hash_parts(owner_rows("merged_block", message_id), bi),
            )
            scratch.conn.execute(
                "UPDATE merged_message SET row_blob = ? WHERE ordinal = ?",
                (pickle.dumps(tuple(rehashed_row), protocol=5), ordinal),
            )
        for (native,) in scratch.conn.execute("SELECT key FROM old_message WHERE key IS NOT NULL GROUP BY key"):
            old_id = native
            live = scratch.conn.execute("SELECT message_id FROM message_owner_map WHERE key = ?", (native,)).fetchone()
            scratch.conn.execute(
                "INSERT OR REPLACE INTO message_remap VALUES (?, ?)", (old_id, live[0] if live is not None else None)
            )
        captured = _capture_session_projection_rows(conn, session_id, scratch=scratch, before_input=before_input)
        scratch.conn.execute(
            "INSERT INTO carried_attachment SELECT DISTINCT owner.attachment_id "
            "FROM captured_attachment_owner owner "
            "JOIN live_message live ON live.message_id = owner.message_id"
        )
        scratch.conn.execute(
            "INSERT INTO refresh_attachment SELECT DISTINCT owner.attachment_id "
            "FROM captured_attachment_owner owner WHERE NOT EXISTS ("
            "SELECT 1 FROM captured_attachment_owner retained "
            "JOIN live_message live ON live.message_id = retained.message_id "
            "WHERE retained.attachment_id = owner.attachment_id)"
        )
        carry = _ProjectionCarryForward(
            captured,
            _UnionSet(scratch, "live_message", "message_id"),
            _UnionMap(scratch, "message_remap", "old_id", "new_id"),
            scratch,
            cast(Mapping[str, str], _UnionMap(scratch, "incoming_message_remap", "old_id", "new_id")),
        )
        with retain_native_sql_lifetimes(scratch):
            builder = SessionShardBuilder(Path(scratch._scratch.name) / "merged.db")
        try:
            builder.add_streamed(
                session_id=session_id,
                session_content_hash=incoming.session_content_hash,
                owner_resolution=incoming.owner_resolution,
                message_rows=iter(scratch.rows("merged_message")),
                block_rows=iter(scratch.rows("merged_block")),
            )
            with retain_native_sql_lifetimes(scratch):
                shard = open_session_shard(builder.seal().path)
        except BaseException as primary:
            try:
                builder.abandon()
            except BaseException as cleanup:
                raise cleanup from primary
            raise
        entry = shard.sessions[0]
        rows = PreparedSessionRows(
            session_id,
            entry.session_content_hash,
            _ShardRowSequence(shard.path, "messages", entry.message_lo, entry.message_hi, lifetime=scratch),
            _ShardRowSequence(shard.path, "blocks", entry.block_lo, entry.block_hi, lifetime=scratch),
            entry.content_identities,
            entry.owner_resolution,
        )
        scratch.finish()
        return _PreparedCrossAcquisitionUnion(predecessor, parent_guard, rows, carry)
    except BaseException as primary:
        try:
            scratch.close()
        except BaseException as cleanup:
            raise cleanup from primary
        raise


def _cross_acquisition_union_applies(
    *,
    session_membership_existed: bool,
    force_replace: bool,
    raw_id: str | None,
    existing_raw_id: str | None,
) -> bool:
    """Whether a full replacement must union with another acquisition's rows."""
    return (
        session_membership_existed
        and not force_replace
        and raw_id is not None
        and existing_raw_id is not None
        and raw_id != existing_raw_id
    )


def _replace_full_session_messages_and_blocks(
    conn: sqlite3.Connection,
    session: ParsedSession,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    duplicate_native_ids: frozenset[str],
    raw_id: str | None = None,
    existing_raw_id: str | None = None,
    session_membership_existed: bool = True,
    force_replace: bool = False,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    bulk_build: bool = False,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
    prepared_union: _PreparedCrossAcquisitionUnion | None = None,
) -> _ProjectionCarryForward | None:
    """Replace one session's messages/blocks wholesale.

    Returns the ``_ProjectionCarryForward`` payload computed by
    ``_union_with_existing_rows`` (or ``None`` when nothing was unioned), so
    the caller (``write_parsed_session_to_archive``) can restore
    ``attachment_refs``/``paste_spans`` after ITS OWN later rebuild of those
    two tables -- they are written outside this function, in the shared
    merge_append/full-replace tail, so this function cannot restore them
    itself without running before they even exist.

    ``bulk_build`` (polylogue-v6i3, default ``False``) skips the write-side
    refresh work owned by this function. Action and delegation relations are
    query-time views; ordinary full-session replacement keeps their results
    current through the canonical rows.

    ``prepared`` (polylogue-623q), when given, is an already-validated row
    set -- the caller (``write_parsed_session_to_archive``) has already
    confirmed it was computed from THIS session's content hash and that no
    lineage tail-slicing changed ``messages`` since. The message/block
    row-building loops (the CPU-bound part of this function -- per-item
    hashing, JSON encoding, enum lookups) are then skipped entirely.
    ``PreparedSessionRows`` streams its sealed rows into the ``executemany``
    on this (writer) thread. A different
    acquisition with prior rows requires ``prepared_union``: its reconciled
    rows were sealed outside the writer and its predecessor is rechecked.
    ``None`` retains the direct list-backed route.

    ``raw_id`` (polylogue-geop) identifies which raw acquisition this write's
    ``messages`` were parsed from; ``existing_raw_id`` is whatever
    ``sessions.raw_id`` held for this session BEFORE this write's caller
    upserted its own value over it (the caller must capture this ahead of
    that upsert -- by the time this function runs, the row already reflects
    the new write). See ``_union_with_existing_rows`` for how the two are
    compared to discriminate "different acquisition, union" from "same
    acquisition re-parsed, replace".

    ``force_replace`` also disables the union outright: it is the caller's
    OWN authoritative precedence decision (e.g. `ingest_precedence.
    browser_capture_precedence()` deciding a fuller native capture supersedes
    a weaker DOM-fallback one) that this write must win wholesale, not merge
    with -- unioning against a session the caller has explicitly ruled
    inferior would silently partially undo that decision.

    ``session_membership_existed`` (default ``True`` so callers that omit it
    preserve the established replacement behavior) records whether this
    session had either its session row or message membership before the
    caller's upsert. The latter covers recovery from a partial derived-output
    loss where a removed session row leaves child messages behind. When
    ``False``, the session has neither row nor messages, so the scoped delete
    cascade is a no-op and can be skipped on a fresh build. The query is
    scoped to this ``session_id``; rows for any other retained raw cohort are
    never considered or deleted. Same-connection reads include uncommitted
    earlier writes, so a session revisited in one bulk transaction correctly
    runs the replacement lifecycle on its second occurrence.
    """

    def add_timing(name: str, started_at: float) -> None:
        _add_stage_timing(
            stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            name=f"index.full_replace.{name}",
            started_at=started_at,
        )

    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    _assert_unique_message_coordinates(session_id, messages)
    t0 = time.perf_counter()
    # An input shard contains only the incoming acquisition. Across
    # acquisitions the read-only preparer must seal reconciled rows before
    # this writer reaches its replacement transaction.
    needs_union = _cross_acquisition_union_applies(
        session_membership_existed=session_membership_existed,
        force_replace=force_replace,
        raw_id=raw_id,
        existing_raw_id=existing_raw_id,
    )
    if (
        needs_union
        and prepared_union is None
        and (
            isinstance(messages, SqliteMessageSink)
            or isinstance(messages, _MessageTail)
            and isinstance(messages.messages, SqliteMessageSink)
        )
    ):
        raise PreparedSessionWriteRefusedError("disk-backed cross-acquisition write requires prepared field union")
    tuple_rows = prepared_union.rows if prepared_union is not None else prepared
    # polylogue-geop: compute the field-path union against whatever is
    # currently stored *before* any delete below removes it. Must run ahead
    # of the FTS/base-table deletes -- both messages and blocks are read here.
    unioned_message_rows: Iterable[tuple[object, ...]]
    unioned_block_rows: Iterable[tuple[object, ...]]
    if prepared_union is not None:
        if not needs_union:
            # The caller passes only a union its own precedence applies.
            raise PreparedSessionWriteRefusedError("prepared field union no longer applies")
        unioned_message_rows = prepared_union.rows.message_rows
        unioned_block_rows = prepared_union.rows.block_rows
        carry_forward = prepared_union.carry_forward
    elif not needs_union:
        # A first write cannot have prior rows to reconcile.  The caller's
        # session PK lookup already proved this session_id is absent, and all
        # message/block rows are created together with that session row.  Skip
        # the union helper (including its acquisition checks and potential
        # session-scoped reads) on the from-empty rebuild path.  This is the
        # same invariant used below to skip the delete cascade, but it also
        # avoids paying field-path-union overhead for every new session.
        unioned_message_rows = (
            tuple_rows.message_rows
            if tuple_rows is not None
            else _iter_message_rows(
                session_id, messages, duplicate_native_ids=duplicate_native_ids, content_identities=content_identities
            )
        )
        unioned_block_rows = (
            tuple_rows.block_rows
            if tuple_rows is not None
            else _iter_block_rows(
                session_id, messages, duplicate_native_ids=duplicate_native_ids, content_identities=content_identities
            )
        )
        carry_forward = None
    else:
        unioned_message_rows, unioned_block_rows, carry_forward = _union_with_existing_rows(
            conn,
            session_id,
            list(tuple_rows.message_rows)
            if tuple_rows is not None
            else _build_message_rows(
                session_id, messages, duplicate_native_ids=duplicate_native_ids, content_identities=content_identities
            ),
            list(tuple_rows.block_rows)
            if tuple_rows is not None
            else _build_block_rows(
                session_id, messages, duplicate_native_ids=duplicate_native_ids, content_identities=content_identities
            ),
            raw_id=raw_id,
            existing_raw_id=existing_raw_id,
            force_replace=force_replace,
        )
    add_timing("field_path_union", t0)
    t0 = time.perf_counter()
    use_scoped_fts_rebuild = not bulk_build and message_fts_triggers_present_sync(conn)
    add_timing("fts_probe", t0)
    if session_membership_existed:
        note_current_lineage_change(conn, session_id)
        note_current_deleted_message_ids(
            conn,
            (
                str(row[0])
                for row in conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,))
            ),
        )
    if use_scoped_fts_rebuild:
        # These deletes are keyed by session_id exactly like the base-table
        # cascade below. With neither a prior session row nor messages, they
        # are no-ops. Orphaned messages take the replacement path even when
        # the session row was lost.
        if session_membership_existed:
            t0 = time.perf_counter()
            conn.execute(delete_session_rows_sql(1), (session_id,))
            add_timing("fts_messages_delete", t0)
            # polylogue-miwv: identity-ledger companion, same chunk params as the
            # messages_fts delete above -- see message_identity_mismatch_sql's
            # docstring for why this non-bulk full-session-replace fast path must
            # keep messages_fts_identity paired with messages_fts.
            t0 = time.perf_counter()
            conn.execute(delete_session_identity_rows_sql(1), (session_id,))
            add_timing("fts_identity_delete", t0)
        t0 = time.perf_counter()
        # Keep the canonical triggers structurally present and gate the
        # message bodies for the whole replacement.  This is the
        # same protocol used for guarded lineage rewrites, but here it covers
        # the session's own delete and insert as well.
        conn.execute(
            "INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES (?)",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        )
        add_timing("fts_guard", t0)
    replacement_complete = False
    try:
        # The scoped cascade is unnecessary only when this session has neither
        # a prior row nor retained messages. A partial session-row loss leaves
        # the latter behind, so it must follow the same atomic cleanup path as
        # an ordinary replacement.
        if session_membership_existed:
            t0 = time.perf_counter()
            _clear_session_projection_rows(conn, session_id)
            add_timing("clear_projection_rows", t0)
            t0 = time.perf_counter()
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            add_timing("delete_messages", t0)
        t0 = time.perf_counter()
        _write_messages(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
            rows=unioned_message_rows,
            content_identities=content_identities,
        )
        add_timing("messages", t0)
        if carry_forward is not None:
            leaf = conn.execute(
                "SELECT active_leaf_message_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if leaf is not None and leaf[0] is not None:
                mapped_leaf = carry_forward.incoming_message_id_remap.get(str(leaf[0]), str(leaf[0]))
                conn.execute(
                    "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?", (mapped_leaf, session_id)
                )
        t0 = time.perf_counter()
        _write_blocks(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
            rows=unioned_block_rows,
            content_identities=content_identities,
        )
        add_timing("blocks", t0)
        t0 = time.perf_counter()
        _write_file_edits(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
            content_identities=content_identities,
            message_id_remap=carry_forward.incoming_message_id_remap if carry_forward is not None else None,
        )
        add_timing("file_edits", t0)
        t0 = time.perf_counter()
        _write_web_constructs(
            conn,
            session,
            messages,
            duplicate_native_ids=duplicate_native_ids,
            prior_rows=session_membership_existed,
            content_identities=content_identities,
            message_id_remap=carry_forward.incoming_message_id_remap if carry_forward is not None else None,
        )
        add_timing("web_constructs", t0)
        replacement_complete = True
    finally:
        if use_scoped_fts_rebuild:
            t0 = time.perf_counter()
            if replacement_complete and not defer_fts_rebuild:
                conn.execute(insert_session_rows_sql(1), (session_id,))
                # polylogue-miwv: identity-ledger companion, same chunk params
                # as the messages_fts insert above.
                conn.execute(insert_session_identity_rows_sql(1), (session_id,))
                add_timing("fts_insert", t0)
            conn.execute(
                "DELETE FROM derived_refresh_guard WHERE guard_name = ?",
                (FTS_BULK_SESSION_WRITE_GUARD,),
            )
            add_timing("fts_guard_clear", t0)

    # NOTE: restoration of captured projection rows (attachment_refs/
    # paste_spans/file_edits/web_content_constructs) deliberately does NOT
    # happen here. attachment_refs/paste_spans are rebuilt by the CALLER
    # after this function returns (they live in the shared merge_append/
    # full-replace tail in ``write_parsed_session_to_archive``, not in this
    # function), so restoring now -- before that rebuild has even run --
    # would restore into empty tables and then risk being clobbered if that
    # later rebuild does its own session-scoped delete. The caller restores
    # once, after all four tables are rebuilt. See this function's docstring.
    return carry_forward


def _messages_have_token_counts(messages: Sequence[ParsedMessage]) -> bool:
    return any(
        value is not None
        for message in messages
        for value in (
            message.input_tokens,
            message.output_tokens,
            message.cache_read_tokens,
            message.cache_write_tokens,
        )
    )


def _attachment_provenance(
    attachment: ParsedAttachment,
    owning_message: ParsedMessage | None,
    *,
    resolved_message_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the direction and producer this attachment is persisted with.

    Attachments carry their own provenance once a parser has derived it.
    Records reconstructed from evidence written before the field existed carry
    none, so the owning turn's role supplies it here through the same shared
    derivation the parsers use -- the stored column is never null.

    A model turn carrying no provider-assigned id yields a ``model_output``
    direction with no producer, because the parser has no identity to name
    at parse time. That combination is rejected below and would fail the
    whole session's write. ``resolved_message_id`` is the stored identity
    this attachment is being written against, which is exactly the producer
    the parser could not yet name, so it supplies the producer for both the
    parser-declared and the role-derived path.
    """
    producer_fallback = f"message:{resolved_message_id}" if resolved_message_id else None
    if attachment.direction is not None:
        producer_ref = attachment.producer_ref
        if attachment.direction == "model_output" and not producer_ref:
            producer_ref = producer_fallback
        return attachment.direction, producer_ref
    if owning_message is None:
        return None, None
    derived_direction, derived_producer = derive_attachment_provenance(
        owning_message.role, owning_message.provider_message_id
    )
    producer_ref = attachment.producer_ref or derived_producer
    if derived_direction == "model_output" and not producer_ref:
        producer_ref = producer_fallback
    return derived_direction, producer_ref


def _attachment_message_id_maps(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] | None = None,
    owner_resolution: MessageOwnerResolution,
    wanted_owner_keys: set[str] | None = None,
) -> tuple[MessageOwnerResolution, dict[str, str], dict[str, ParsedMessage]]:
    """Build the authoritative attachment-owner lookup maps.

    The first result is the shared private owner-resolution contract. The
    second maps its resolved owner keys to stored message ids, including the
    full ``(position, variant_index)`` coordinate and any reorder-stable
    parser evidence. The third maps those stored message ids back to the
    owning parsed message, which is what lets the write boundary derive an
    attachment's direction from its owning turn. Keep this shared with
    repair/relink paths so they cannot invent a weaker ownership rule than
    the production write.
    """
    duplicates = duplicate_native_ids if duplicate_native_ids is not None else _duplicate_message_native_ids(messages)
    resolution = owner_resolution
    by_owner_key: dict[str, str] = {}
    by_message_id: dict[str, ParsedMessage] = {}
    keys = iter(resolution.keys)
    if isinstance(messages, _MessageTail):
        for _ordinal in range(messages.start):
            next(keys)
    for fallback_position, (message, owner_key) in enumerate(zip(messages, keys, strict=True)):
        if owner_key in resolution.ambiguous_keys or (
            wanted_owner_keys is not None and owner_key not in wanted_owner_keys
        ):
            continue
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            duplicate_native_ids=duplicates,
        )
        by_owner_key[owner_key] = message_id
        by_message_id.setdefault(message_id, message)
    return resolution, by_owner_key, by_message_id


def _next_message_position(conn: sqlite3.Connection, session_id: str) -> int:
    """Return the position offset used when appending messages to a session."""
    row = conn.execute(
        "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _stored_content_occurrences(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> dict[str, int]:
    """Return the next content occurrence for each digest stored in a session.

    The append-side analogue of ``_next_message_position``: an appended
    message whose declared semantics match one already written continues that
    digest's numbering instead of restarting at zero and colliding with the
    stored row's ``message_id``. Materialized-prefix identity scopes preserve
    their existing ordinals across replacement, so retained rows can have gaps.
    """
    if before_input is not None:
        before_input(
            "messages",
            ("content_identity", "content_occurrence"),
            "SELECT rowid FROM messages WHERE session_id=? AND content_identity IS NOT NULL",
            (session_id,),
        )
    with connection_cursor(
        conn,
        """
        SELECT content_identity, MAX(content_occurrence) + 1
        FROM messages
        WHERE session_id = ? AND content_identity IS NOT NULL
        GROUP BY content_identity
        """,
        (session_id,),
    ) as _input_cursor:
        rows = _input_cursor.fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _write_attachments(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    attachments: Iterable[ParsedAttachment],
    *,
    supplying_raw_id: str | None,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    refresh_attachment_ids: Iterable[str] | None = None,
    preacquired_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None,
    owner_resolution: MessageOwnerResolution,
    inherited_prefix_message_ids: Sequence[str] = (),
    replace_owner_gaps: bool = True,
) -> tuple[tuple[str, AttachmentOwnerResolutionReason], ...]:
    """Write attachment rows and their message references.

    The returned owner gaps are also recorded in ``attachment_owner_gaps``:
    a full write replaces the session's gaps, an append write replaces only
    those of the attachments it carries.

    An attachment the tail does not own may belong to an inherited prefix
    message (``inherited_prefix_message_ids`` names those parent rows by
    ordinal). The parent row is then its owner: when it references the
    attachment, that reference is what the composed child reads, so the child
    adds none -- its bytes can still complete the shared ``attachments`` row.
    A prefix owner that does not reference it is a typed
    ``INHERITED_OWNER_UNREFERENCED``, never an orphaned metadata row.
    """
    attachments = tuple(attachments)

    def identity(attachment: ParsedAttachment) -> str:
        acquired = (preacquired_blobs or {}).get(attachment.acquisition_key)
        return _attachment_id(session_id, attachment, blob_hash=acquired[0] if acquired is not None else None)

    if not attachments:
        refresh_and_sweep_attachment_rows(conn, refresh_attachment_ids or ())
        if replace_owner_gaps:
            _record_attachment_owner_gaps(conn, session_id, (), replace_session=True)
        return ()
    wanted_owner_keys: set[str] | None = None
    if owner_resolution is not None:
        wanted_owner_keys = set()
        for attachment in attachments:
            try:
                owner_key = attachment_message_owner_key(attachment, owner_resolution)
            except MessageOwnerAmbiguityError:
                continue
            if owner_key is not None:
                wanted_owner_keys.add(owner_key)
    owner_resolution, by_owner_key, owning_messages = _attachment_message_id_maps(
        session_id,
        messages,
        position_offset=position_offset,
        duplicate_native_ids=duplicate_native_ids,
        content_identities=content_identities,
        message_id_remap=message_id_remap,
        owner_resolution=owner_resolution,
        wanted_owner_keys=wanted_owner_keys,
    )
    attachment_positions: dict[object, int] = {}
    resolved_message_ids: dict[object, str] = {}
    attachments_by_message: defaultdict[str, list[ParsedAttachment]] = defaultdict(list)
    unresolved: dict[str, AttachmentOwnerResolutionReason] = {}
    tail_unowned: list[ParsedAttachment] = []
    for attachment in attachments:
        try:
            owner_key = attachment_message_owner_key(attachment, owner_resolution)
        except MessageOwnerAmbiguityError:
            # The attachment remains represented by the session hash and raw
            # evidence, but no message owner is safe to guess.
            unresolved[identity(attachment)] = AttachmentOwnerResolutionReason.OWNER_AMBIGUOUS
            continue
        message_id = by_owner_key.get(owner_key) if owner_key is not None else None
        if message_id is not None:
            resolved_message_ids[attachment.acquisition_key] = message_id
            attachments_by_message[message_id].append(attachment)
        else:
            tail_unowned.append(attachment)
            # A named owner that no written message carries is a lost owner;
            # an attachment the provider never linked has no owner to lose.
            unresolved[identity(attachment)] = (
                AttachmentOwnerResolutionReason.PROVIDER_NEVER_LINKED
                if owner_key is None
                else AttachmentOwnerResolutionReason.MESSAGE_MISSING
            )
    # Acquisition keys whose owner is an inherited parent row: referenced by
    # it (the parent owns the observation), or not (typed, nothing written).
    inherited_owned: set[object] = set()
    inherited_unreferenced: set[object] = set()
    if tail_unowned and isinstance(messages, _MessageTail) and len(inherited_prefix_message_ids):
        with _prefix_attachment_owner_ordinals(messages.messages, messages.start, tail_unowned) as (
            prefix_ordinals,
            prefix_ambiguous,
        ):
            for attachment in tail_unowned:
                attachment_id = identity(attachment)
                if attachment.acquisition_key in prefix_ambiguous:
                    unresolved[attachment_id] = AttachmentOwnerResolutionReason.OWNER_AMBIGUOUS
                    continue
                ordinal = prefix_ordinals.get(attachment.acquisition_key)
                if ordinal is None:
                    continue
                if _message_references_attachment(conn, inherited_prefix_message_ids[ordinal], attachment_id):
                    inherited_owned.add(attachment.acquisition_key)
                    unresolved.pop(attachment_id, None)
                else:
                    inherited_unreferenced.add(attachment.acquisition_key)
                    unresolved[attachment_id] = AttachmentOwnerResolutionReason.INHERITED_OWNER_UNREFERENCED
    for message_group in attachments_by_message.values():
        versions: dict[str, str] = {}
        for attachment in message_group:
            native_identity = _attachment_reference_native_identity(attachment)
            attachment_id = identity(attachment)
            previous = versions.setdefault(native_identity, attachment_id)
            if previous != attachment_id:
                raise AttachmentReferenceAmbiguityError("one native attachment identity has competing objects")
        native_identities = sorted({_attachment_reference_native_identity(item) for item in message_group})
        positions = {native_identity: position for position, native_identity in enumerate(native_identities)}
        for attachment in message_group:
            attachment_positions[attachment.acquisition_key] = positions[
                _attachment_reference_native_identity(attachment)
            ]
    touched_attachment_ids: set[str] = set()
    for attachment in attachments:
        attachment_id = identity(attachment)
        message_id = resolved_message_ids.get(attachment.acquisition_key)
        if message_id is None and attachment.acquisition_key in inherited_owned:
            # The inherited parent row references this attachment; the child
            # composes it through that row. The child's copy can still carry
            # bytes the shared row lacks, and nothing here adds a reference.
            _write_attachment_row(conn, attachment_id, attachment, preacquired_blobs)
            continue
        if message_id is None and attachment.acquisition_key in inherited_unreferenced:
            continue
        if message_id is None:
            # An attachment carrying inline or precomputed bytes whose owner
            # is absent from this ingest cannot be represented by a reachable
            # ref. Do not acquire or persist those bytes; a prior row is swept
            # through ``refresh_attachment_ids`` below when its last ref was
            # dropped. Metadata-only records remain as unfetched evidence.
            if attachment.inline_bytes is not None or attachment.precomputed_blob is not None:
                continue
            _write_attachment_row(conn, attachment_id, attachment, preacquired_blobs)
            continue
        direction, producer_ref = _attachment_provenance(
            attachment, owning_messages.get(message_id), resolved_message_id=message_id
        )
        require_literal(direction, AttachmentDirection, name="attachment direction")
        if direction == "model_output" and not producer_ref:
            raise ValueError(
                "model_output attachment requires producer provenance: "
                f"attachment_id={attachment.provider_attachment_id!r}"
            )
        touched_attachment_ids.add(attachment_id)
        _write_attachment_row(conn, attachment_id, attachment, preacquired_blobs)
        ref_position = attachment_positions[attachment.acquisition_key]
        native_identity = _attachment_reference_native_identity(attachment)
        ref_id = attachment_reference_id(message_id, native_identity)
        # Bulk rebuilds may suspend FK enforcement. Mirror REPLACE's cascade
        # explicitly so identifiers from an older projection cannot survive.
        existing_ref = conn.execute(
            "SELECT attachment_id FROM attachment_refs WHERE message_id = ? AND native_identity = ?",
            (message_id, native_identity),
        ).fetchone()
        if existing_ref is not None:
            touched_attachment_ids.add(str(existing_ref[0]))
        conn.execute("DELETE FROM attachment_native_ids WHERE ref_id = ?", (ref_id,))
        conn.execute(
            """
            INSERT INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin, direction, producer_ref, source_url,
                caption, supplying_raw_id, native_identity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, native_identity) DO UPDATE SET
                attachment_id = excluded.attachment_id,
                session_id = excluded.session_id,
                position = excluded.position,
                upload_origin = excluded.upload_origin,
                direction = excluded.direction,
                producer_ref = excluded.producer_ref,
                source_url = excluded.source_url,
                caption = excluded.caption,
                -- This acquisition holds the reference too. A write with no
                -- raw identity keeps the supplier already known to hold it.
                supplying_raw_id = COALESCE(excluded.supplying_raw_id, attachment_refs.supplying_raw_id)
            """,
            (
                attachment_id,
                session_id,
                message_id,
                ref_position,
                _sqlite_text(attachment.upload_origin),
                direction,
                _sqlite_text(producer_ref),
                _sqlite_text(_attachment_source_url(attachment)),
                _sqlite_text(_attachment_caption(attachment)),
                supplying_raw_id,
                native_identity,
            ),
        )
        _write_attachment_native_ids(conn, ref_id, attachment)
    affected_attachment_ids = chain(touched_attachment_ids, refresh_attachment_ids or ())
    # polylogue-w06b: a full-replace re-ingest (or a re-ingest whose attachment
    # can no longer be matched to a message via the shared owner key, e.g.
    # the owning message became a duplicate-native-id exclusion or dropped
    # out of this ingest's message set) drops a previously-written
    # attachment_refs row for `refresh_attachment_ids` without this
    # function ever writing a replacement ref. The refresh below reflects
    # that as ref_count 0 and sweeps the now ref-less `attachments` row --
    # it would otherwise survive, unreachable from any session/message read
    # path (get_attachments/get_attachments_batch both INNER JOIN
    # attachment_refs), while still reporting acquisition_status='acquired'
    # and real fetched bytes.
    refresh_and_sweep_attachment_rows(conn, affected_attachment_ids)
    gaps = tuple(sorted(unresolved.items()))
    _record_attachment_owner_gaps(
        conn,
        session_id,
        gaps,
        replace_session=replace_owner_gaps,
        written_attachment_ids={identity(attachment) for attachment in attachments},
    )
    return gaps


def _record_attachment_owner_gaps(
    conn: sqlite3.Connection,
    session_id: str,
    gaps: Sequence[tuple[str, AttachmentOwnerResolutionReason]],
    *,
    replace_session: bool,
    written_attachment_ids: Iterable[str] = (),
) -> None:
    """Record why this write left attachments without an owner ref.

    ``replace_session`` drops every earlier gap of the session (a full
    write); otherwise only the gaps of ``written_attachment_ids`` are
    replaced, so an append keeps the gaps its earlier writes recorded.
    """
    if replace_session:
        conn.execute("DELETE FROM attachment_owner_gaps WHERE session_id = ?", (session_id,))
    else:
        written = iter(sorted(set(written_attachment_ids)))
        while batch := tuple(islice(written, 128)):
            conn.execute(
                "DELETE FROM attachment_owner_gaps WHERE session_id = ? AND attachment_id IN ({})".format(
                    ",".join("?" for _ in batch)
                ),
                (session_id, *batch),
            )
    conn.executemany(
        "INSERT OR REPLACE INTO attachment_owner_gaps(session_id, attachment_id, reason) VALUES (?, ?, ?)",
        [
            (
                session_id,
                attachment_id,
                require_vocabulary(reason, AttachmentOwnerResolutionReason, field="attachment_owner_gaps.reason"),
            )
            for attachment_id, reason in gaps
        ],
    )


def recorded_attachment_owner_gaps(
    conn: sqlite3.Connection, session_id: str
) -> tuple[tuple[str, AttachmentOwnerResolutionReason], ...]:
    """The owner gaps the session's latest write recorded, for a replay that does not rewrite it."""
    return tuple(
        (str(attachment_id), AttachmentOwnerResolutionReason(str(reason)))
        for attachment_id, reason in conn.execute(
            "SELECT attachment_id, reason FROM attachment_owner_gaps WHERE session_id = ? ORDER BY attachment_id",
            (session_id,),
        )
    )


def _write_attachment_row(
    conn: sqlite3.Connection,
    attachment_id: str,
    attachment: ParsedAttachment,
    preacquired_blobs: Mapping[object, tuple[bytes | None, int, str]] | None,
) -> None:
    """Upsert the attachment's identity and bytes, leaving refs to the caller."""
    acquired_blob = (preacquired_blobs or {}).get(attachment.acquisition_key)
    blob_hash, byte_count, acquisition_status = (
        acquired_blob if acquired_blob is not None else _acquire_attachment_blob(conn, attachment)
    )
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status, ref_count
        ) VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(attachment_id) DO UPDATE SET
            display_name = COALESCE(excluded.display_name, attachments.display_name),
            media_type = COALESCE(excluded.media_type, attachments.media_type),
            byte_count = CASE
                WHEN excluded.blob_hash IS NULL AND attachments.blob_hash IS NOT NULL
                THEN attachments.byte_count ELSE excluded.byte_count END,
            blob_hash = COALESCE(excluded.blob_hash, attachments.blob_hash),
            acquisition_status =
                CASE WHEN excluded.acquisition_status = 'acquired'
                     THEN 'acquired' ELSE attachments.acquisition_status END
        """,
        (
            attachment_id,
            _sqlite_text(attachment.name),
            _sqlite_text(attachment.mime_type),
            byte_count,
            blob_hash,
            acquisition_status,
        ),
    )


def refresh_and_sweep_attachment_rows(conn: sqlite3.Connection, attachment_ids: Iterable[str]) -> None:
    """Recompute ``attachments.ref_count`` from live refs and sweep zero-ref rows.

    Every path that removes ``attachment_refs`` rows must call this with the
    affected attachment ids -- including an FK cascade, which removes them
    with no Python code observing it. Otherwise an acquired ``attachments``
    row survives with a stale ref_count and no canonical ref, which archive
    verification and ``blob-reference-closure`` report as archive
    verification errors.

    A caller re-ingesting content must exclude the ids it is carrying
    forward: those rows are about to be re-referenced, and sweeping them
    would break the restore's FK to ``attachments(attachment_id)``. That
    exemption is why this stays in Python rather than becoming a trigger,
    which could not see it.
    """
    source = iter(attachment_ids)
    while batch := tuple(islice(source, 128)):
        placeholders = ",".join("?" for _ in batch)
        conn.execute(
            f"""
            UPDATE attachments
            SET ref_count = (
                SELECT COUNT(*) FROM attachment_refs WHERE attachment_refs.attachment_id = attachments.attachment_id
            )
            WHERE attachment_id IN ({placeholders})
            """,
            batch,
        )
        conn.execute(
            f"DELETE FROM attachments WHERE ref_count <= 0 AND attachment_id IN ({placeholders})",
            batch,
        )


def session_attachment_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT attachment_id FROM attachment_refs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _write_paste_spans(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    for fallback_position, message in enumerate(messages):
        if not _has_paste(message):
            continue
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            duplicate_native_ids=duplicate_native_ids,
        )
        text = message.text or ""
        for evidence in message.paste_spans:
            boundary = PasteBoundary(evidence.boundary_state)
            start_offset = evidence.start_offset if evidence.start_offset is not None else 0
            end_offset = evidence.end_offset if evidence.end_offset is not None else len(text)
            conn.execute(
                """
                INSERT OR REPLACE INTO paste_spans (
                    message_id, session_id, position, start_offset, end_offset, boundary_state,
                    source_event_id, source_marker, content_hash, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    evidence.position,
                    start_offset,
                    end_offset,
                    boundary.value,
                    _sqlite_text(evidence.source_event_id),
                    _sqlite_text(evidence.source_marker),
                    evidence.content_hash or _hash_bytes("paste", message_id, str(evidence.position), text),
                    evidence.observed_at_ms
                    if evidence.observed_at_ms is not None
                    else message.occurred_at_ms
                    if message.occurred_at_ms is not None
                    else to_epoch_ms(message.timestamp, numeric_unit="seconds"),
                ),
            )


class _StoredSessionParents(Mapping[str, str]):
    """Native parent ids resolved to this session's stored native rows.

    Only a parent id that names exactly one stored native row resolves: a
    stored duplicate has a content-derived id and so no ``:n:`` row, and a
    value the appended batch itself repeats stays ambiguous.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        inherited: Mapping[str, str],
        duplicate_native_ids: frozenset[str],
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._inherited = inherited
        self._duplicates = duplicate_native_ids

    def __getitem__(self, key: str) -> str:
        inherited = self._inherited.get(key)
        if inherited is not None:
            return inherited
        native_id = normalized_message_native_id(key)
        if not native_id or message_native_key(native_id) in self._duplicates:
            raise KeyError(key)
        candidate = archive_message_id(self._session_id, native_id)
        if self._conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (candidate,)).fetchone() is None:
            raise KeyError(key)
        return candidate

    def __iter__(self) -> Iterator[str]:
        raise TypeError("stored session parents are resolved by key only")

    def __len__(self) -> int:
        raise TypeError("stored session parents are resolved by key only")


def _write_parent_links(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    inherited_message_ids: Mapping[str, str] | None = None,
    stored_parents: bool = False,
) -> None:
    """Resolve each written message's declared parent to a stored row.

    A prefix-sharing child writes only its tail, so a tail message whose
    parent is the last inherited message resolves through
    ``inherited_message_ids`` (provider id -> the inherited row), keeping the
    tree connected across the boundary instead of leaving it NULL.

    A merge-append (``stored_parents``) writes only the new records of the
    same session, so a new record whose declared native parent is an
    earlier stored record resolves to that row. Without the edge, a later
    tool reply chained under an earlier one cannot be paired with the
    invocation both answer.
    """
    source = messages.messages if isinstance(messages, _MessageTail) else messages
    updates = _ParentLinkUpdates(conn)
    batch_inherited = inherited_message_ids or {}
    inherited: Mapping[str, str] = (
        _StoredSessionParents(conn, session_id, batch_inherited, duplicate_native_ids)
        if stored_parents
        else batch_inherited
    )
    if isinstance(source, SqliteMessageSink):
        disk_index = _DiskMessageEventIndex(source.path.parent)
        try:
            for fallback_position, message in enumerate(messages):
                message_id = _message_id(
                    session_id,
                    message,
                    fallback_position,
                    content_identities=content_identities,
                    message_id_remap=message_id_remap,
                    duplicate_native_ids=duplicate_native_ids,
                )
                if message.provider_message_id and _normalized_message_native_id(message) not in duplicate_native_ids:
                    disk_index[message.provider_message_id] = message_id
                if message.position is not None:
                    disk_index.replace_boundary(message.position, message_id)
            for fallback_position, message in enumerate(messages):
                parent_message_id = (
                    disk_index.get(message.parent_message_provider_id) if message.parent_message_provider_id else None
                )
                if parent_message_id is None and message.parent_message_provider_id:
                    parent_message_id = inherited.get(message.parent_message_provider_id)
                if parent_message_id is None and message.parent_message_position is not None:
                    parent_message_id = disk_index.boundary_message_id(message.parent_message_position)
                if parent_message_id is not None:
                    updates.add(
                        parent_message_id,
                        _message_id(
                            session_id,
                            message,
                            fallback_position,
                            content_identities=content_identities,
                            message_id_remap=message_id_remap,
                            duplicate_native_ids=duplicate_native_ids,
                        ),
                    )
            updates.flush()
        finally:
            disk_index.close()
        return
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and _normalized_message_native_id(message) not in duplicate_native_ids
    }
    by_message_position = {
        message.position: _message_id(
            session_id,
            message,
            fallback_position,
            content_identities=content_identities,
            message_id_remap=message_id_remap,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.position is not None
    }
    for fallback_position, message in enumerate(messages):
        parent_message_id = (
            by_native_id.get(message.parent_message_provider_id) if message.parent_message_provider_id else None
        )
        if parent_message_id is None and message.parent_message_provider_id:
            parent_message_id = inherited.get(message.parent_message_provider_id)
        if parent_message_id is None and message.parent_message_position is not None:
            parent_message_id = by_message_position.get(message.parent_message_position)
        if parent_message_id is None:
            continue
        updates.add(
            parent_message_id,
            _message_id(
                session_id,
                message,
                fallback_position,
                content_identities=content_identities,
                message_id_remap=message_id_remap,
                duplicate_native_ids=duplicate_native_ids,
            ),
        )
    updates.flush()


class _ParentLinkUpdates:
    """Parent-link updates applied in bounded ``executemany`` batches.

    One statement per message was a Python-to-SQLite round trip per row on
    the writer; each message's update is independent of every other's, so
    batching changes only the cost, and the batch bound keeps memory flat
    for a whale session.
    """

    _BATCH = 4096

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._rows: list[tuple[str, str]] = []

    def add(self, parent_message_id: str, message_id: str) -> None:
        self._rows.append((parent_message_id, message_id))
        if len(self._rows) >= self._BATCH:
            self.flush()

    def flush(self) -> None:
        if self._rows:
            self._conn.executemany("UPDATE messages SET parent_message_id = ? WHERE message_id = ?", self._rows)
            self._rows = []


@dataclass(frozen=True, slots=True)
class _HookParentClaim:
    """One durable hook assertion of a child's parent, with its own evidence.

    ``evidence`` is merged into ``session_links.evidence_json`` so the row
    itself records which hook fields decided it.

    ``parent_native_id`` is ``None`` when the hook evidence is present but
    self-contradictory. That is not silence: no parent is adopted, and the
    evidence explaining the refusal is still retained on the row.
    """

    parent_native_id: str | None
    evidence: Mapping[str, object]


#: Claude Code writes a dispatched child's transcript to
#: ``<session>/subagents/agent-<agentId>.jsonl`` and the child parser claims
#: that stem as a provider alias, so a hook payload's bare ``agent_id`` maps
#: onto the child by stripping this prefix.
_SUBAGENT_STEM_PREFIX = "agent-"

#: Hook event types whose payload carries the dispatched agent's identity.
_AGENT_BEARING_HOOK_EVENTS = ("PreToolUse", "PostToolUse")

#: The canonical hook keys one dispatch claim is built from, in extraction order.
_AGENT_DISPATCH_HOOK_KEYS = ("agent_id", "agent_type", "tool_use_id")

#: SQLite parameter batch for the child's ``tool_id`` membership check.
_HOOK_TOOL_ID_CHUNK = 500


def _hook_payload_json_paths(key: str) -> tuple[str, ...]:
    """Every JSON path one canonical hook key can occupy in ``payload_json``.

    A spool-drained row stores the producer envelope (the harness payload
    nested under ``$.payload``); evidence written directly stores the bare
    payload. Both generations of every spelling come from
    :func:`polylogue.core.hook_payload.payload_key_spellings`, so a reader
    keyed on one spelling cannot see the other as a field nothing ever sent.
    Envelope before payload, matching ``hook_record_field``'s precedence.
    """
    spellings = payload_key_spellings(key)
    return tuple([f"$.{spelling}" for spelling in spellings] + [f"$.payload.{spelling}" for spelling in spellings])


def _hook_field_slices(keys: Sequence[str]) -> dict[str, slice]:
    """Where each key's values sit in the concatenated multi-path result."""
    slices: dict[str, slice] = {}
    offset = 0
    for key in keys:
        width = len(_hook_payload_json_paths(key))
        slices[key] = slice(offset, offset + width)
        offset += width
    return slices


def _hook_extracted_fields(extracted: object, field_slices: Mapping[str, slice]) -> dict[str, str]:
    """Decode one multi-path ``json_extract`` array into canonical string fields.

    First non-empty spelling wins, matching ``hook_payload_field``; a key whose
    every spelling is absent or empty is simply missing from the result, never
    an empty string a caller could mistake for a sent value.
    """
    if not isinstance(extracted, str):
        return {}
    try:
        values = json.loads(extracted)
    except json.JSONDecodeError:
        return {}
    if not isinstance(values, list):
        return {}
    fields: dict[str, str] = {}
    for key, span in field_slices.items():
        for value in values[span]:
            if isinstance(value, str) and value:
                fields[key] = value
                break
    return fields


class MembershipHeadSourceRead(Protocol):
    """Selected Source evidence for the canonical membership head decision."""

    def raw_revision_authority(self, raw_id: str) -> str | None: ...

    def membership_head_revision(self, raw_id: str) -> tuple[object, ...] | None: ...

    def membership_has_dangling_append(
        self,
        logical_source_key: str,
        existing_raw_id: str,
        classified_raw_ids: frozenset[str],
    ) -> bool: ...


_MEMBERSHIP_HEAD_REVISION_SQL = (
    "SELECT source_revision, acquisition_generation, append_end_offset, blob_size FROM raw_sessions WHERE raw_id=?"
)


def _membership_dangling_append_predicate(classified_raw_ids: frozenset[str]) -> str:
    placeholders = ", ".join("?" for _ in classified_raw_ids) or "NULL"
    return (
        "child.logical_source_key=? AND child.raw_id!=? "
        f"AND child.raw_id NOT IN ({placeholders}) "
        "AND child.predecessor_source_revision IS NOT NULL "
        "AND child.predecessor_source_revision=(SELECT source_revision FROM raw_sessions WHERE raw_id=?)"
    )


class SessionSourceRead(Protocol):
    """Scoped durable evidence used by canonical session preparation."""

    @property
    def blob_store(self) -> BlobStore: ...

    def raw_export_order(self, raw_id: str) -> int | None: ...

    def raw_revision_observation_order(self, raw_id: str) -> tuple[int, int]: ...

    def raw_revision_lineage(self, raw_id: str) -> tuple[str | None, str | None] | None: ...

    def session_enrichment_key(
        self,
        index: sqlite3.Connection,
        *,
        provider: Provider,
        source_path: str | None,
        native_id: str,
    ) -> str | None: ...

    def raw_path(self, raw_id: str) -> str | None: ...

    def hook_spool_present(self) -> bool: ...

    def raw_sessions_present(self) -> bool: ...

    def codex_parent_rows(self, child_native_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...

    def claude_dispatch_rows(self, parent_native_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...

    def acquisition_rows(self, origin: str, native_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...

    def sidecar_page(
        self,
        source_path: str,
        after: tuple[int, str] | None,
    ) -> list[tuple[str, bytes, str, str, int]]: ...

    def open_sidecar_payload(self, raw_id: str, blob_hash: bytes) -> AbstractContextManager[BinaryIO]: ...


_SOURCE_RAW_PATH_SQL = "SELECT source_path FROM raw_sessions WHERE raw_id = ?"
_SOURCE_RAW_REVISION_LINEAGE_SQL = "SELECT baseline_raw_id,predecessor_raw_id FROM raw_sessions WHERE raw_id=?"


def _raw_revision_lineage_from_row(row: sqlite3.Row | None) -> tuple[str | None, str | None] | None:
    if row is None:
        return None
    baseline, predecessor = row
    return (None if baseline is None else str(baseline), None if predecessor is None else str(predecessor))


_SOURCE_CODEX_PARENT_SQL = """
    SELECT json_extract(payload_json, '$.parent_thread_id')
    FROM raw_hook_events
    WHERE event_type = 'codex_thread_spawn_edge'
      AND json_extract(payload_json, '$.child_thread_id') = ?
    ORDER BY observed_at_ms DESC
    LIMIT 1
"""
_SOURCE_ACQUISITION_SQL = "SELECT DISTINCT source_path FROM raw_sessions WHERE origin = ? AND native_id = ?"
_SOURCE_SIDECAR_SQL = """
    SELECT source_path, blob_hash, origin, raw_id, source_index
    FROM raw_sessions WHERE source_path=?
      AND (? IS NULL OR source_index>? OR (source_index=? AND raw_id>?))
    ORDER BY source_index, raw_id LIMIT 256
"""


def _source_sidecar_page_parameters(
    source_path: str,
    after: tuple[int, str] | None,
) -> tuple[object, ...]:
    index, raw_id = after if after is not None else (None, None)
    return source_path, index, index, index, raw_id


def _source_sidecar_page(rows: Iterable[sqlite3.Row | tuple[Any, ...]]) -> list[tuple[str, bytes, str, str, int]]:
    return [(str(row[0]), bytes(row[1]), str(row[2]), str(row[3]), int(row[4])) for row in rows]


def _source_claude_dispatch_sql() -> str:
    paths = [path for key in _AGENT_DISPATCH_HOOK_KEYS for path in _hook_payload_json_paths(key)]
    path_sql = ", ".join(f"'{path}'" for path in paths)
    events = ", ".join("?" * len(_AGENT_BEARING_HOOK_EVENTS))
    return (
        f"SELECT DISTINCT json_extract(payload_json, {path_sql}) FROM raw_hook_events "
        f"WHERE origin = ? AND session_native_id = ? AND event_type IN ({events})"
    )


if TYPE_CHECKING:
    from polylogue.archive.revision_replay import RevisionReplayPlan
    from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
    from polylogue.storage.sqlite.agent_thread_state import ThreadParentRowRead
    from polylogue.storage.sqlite.archive_tiers.revision_application import RevisionApplicationReceipt
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        PreparedRevisionAdoption,
        RawMembershipSelectionFamily,
    )


_RAW_PENDING_ENVELOPE_SQL = "SELECT logical_source_key FROM raw_sessions WHERE raw_id=?"


def _raw_pending_envelope_from_row(row: sqlite3.Row | tuple[object, ...] | None) -> bool:
    from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX

    return row is not None and str(row[0] or "").startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)


class ConnectionSessionSourceRead:
    """Borrow the ordinary Source connection and close each evidence cursor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @property
    def blob_store(self) -> BlobStore:
        # Metadata-only predicates do not resolve or open a Blob capability.
        # Actual payload access still derives it from this same connection.
        return blob_store_for_connection(self._connection)

    @contextmanager
    def _statement(self, sql: str, parameters: tuple[object, ...] = ()) -> Generator[sqlite3.Cursor, None, None]:
        with connection_cursor(self._connection, sql, parameters) as cursor:
            yield cursor

    def publication_source_path(self) -> Path:
        from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

        return ConnectionBlobPublicationRead(self._connection).publication_source_path()

    def publication_blob_is_excised(self, blob_hash: bytes) -> bool:
        from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

        return ConnectionBlobPublicationRead(self._connection).publication_blob_is_excised(blob_hash)

    def publication_reservation(self, publication_id: str) -> tuple[bytes, int, str] | None:
        from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

        return ConnectionBlobPublicationRead(self._connection).publication_reservation(publication_id)

    def retained_attachment_reference(
        self, raw_id: str, raw_blob_hash: bytes, coordinate: str, blob_hash: bytes, size_bytes: int
    ) -> None:
        from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

        ConnectionBlobPublicationRead(self._connection).retained_attachment_reference(
            raw_id, raw_blob_hash, coordinate, blob_hash, size_bytes
        )

    def raw_selection_values(
        self,
        family: RawMembershipSelectionFamily,
        operands: Sequence[str],
    ) -> set[str]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _raw_selection_queries

        found: set[str] = set()
        if not operands:
            return found
        for table, column, predicate, parameters in _raw_selection_queries(family, tuple(operands)):
            with self._statement(f"SELECT {column} FROM {table} WHERE {predicate}", parameters) as rows:
                for row in rows:
                    check_compute_cancelled()
                    found.add(str(row[0]))
        return found

    def pending_raw_envelope_has_membership_authority(self, logical_source_key: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            pending_raw_envelope_has_membership_authority,
        )

        return pending_raw_envelope_has_membership_authority(self._connection, logical_source_key)

    def raw_revision_rebuild_logical_keys(self, raw_ids: Sequence[str]) -> tuple[str, ...]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _raw_revision_rebuild_logical_keys

        return _raw_revision_rebuild_logical_keys(self, raw_ids)

    def replay_representative_rows(self, keys: Sequence[str]) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _raw_replay_representative_query

        return self._statement(*_raw_replay_representative_query(keys))

    def completed_source_item_rows(self, raw_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.sqlite.archive_tiers.source_items import _COMPLETED_SOURCE_ITEM_SQL

        return self._statement(_COMPLETED_SOURCE_ITEM_SQL, (raw_id,))

    def retained_artifact_page(
        self,
        origin: str,
        artifact_kind: str,
        coordinate: str,
        *,
        prefix: bool,
        after: str | None,
    ) -> list[tuple[str, str, str, int, int | None, int | None, str | None]]:
        from polylogue.sources.retained_assembly import _retained_artifact_page_from_rows, _retained_artifact_page_sql

        with self._statement(
            _retained_artifact_page_sql(prefix),
            (origin, coordinate, artifact_kind, after, after),
        ) as rows:
            return _retained_artifact_page_from_rows(rows)

    def retained_group_member(
        self,
        generation: str,
        item: str,
        source_path: str,
        origin: str,
        artifact_kind: str,
    ) -> tuple[str, str, int, str | None] | None:
        from polylogue.sources.retained_assembly import _RETAINED_GROUP_MEMBER_SQL, _retained_group_member_from_row

        with self._statement(
            _RETAINED_GROUP_MEMBER_SQL,
            (generation, item, source_path, origin, artifact_kind),
        ) as rows:
            return _retained_group_member_from_row(rows.fetchone())

    def raw_export_order(self, raw_id: str) -> int | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _RAW_EXPORT_ORDER_SQL,
            _raw_export_order_from_row,
        )

        with self._statement(_RAW_EXPORT_ORDER_SQL, (raw_id,)) as rows:
            return _raw_export_order_from_row(rows.fetchone())

    def raw_revision_observation_order(self, raw_id: str) -> tuple[int, int]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _RAW_ACQUISITION_TIME_SQL,
            _RAW_OBSERVATION_RECEIPT_SQL,
            _raw_acquisition_observation,
        )

        with self._statement(_RAW_OBSERVATION_RECEIPT_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        if row is not None:
            return int(row[0]), int(row[1])
        with self._statement(_RAW_ACQUISITION_TIME_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return _raw_acquisition_observation(row, raw_id)

    def raw_revision_lineage(self, raw_id: str) -> tuple[str | None, str | None] | None:
        with self._statement(_SOURCE_RAW_REVISION_LINEAGE_SQL, (raw_id,)) as rows:
            return _raw_revision_lineage_from_row(rows.fetchone())

    def session_enrichment_key(
        self,
        index: sqlite3.Connection,
        *,
        provider: Provider,
        source_path: str | None,
        native_id: str,
    ) -> str | None:
        from polylogue.sources.revision_backfill import provider_binds_enrichment, session_enrichment_evidence_key

        if not provider_binds_enrichment(provider) or not source_path or not native_id:
            return None

        return session_enrichment_evidence_key(
            provider=provider,
            source_path=source_path,
            native_id=native_id,
            index_conn=index,
            source_conn=self._connection,
            blob_root=self.blob_store.root,
        )

    def raw_has_pending_envelope(self, raw_id: str) -> bool:
        with self._statement(_RAW_PENDING_ENVELOPE_SQL, (raw_id,)) as rows:
            return _raw_pending_envelope_from_row(rows.fetchone())

    def raw_revision_authority(self, raw_id: str) -> str | None:
        with self._statement("SELECT revision_authority FROM raw_sessions WHERE raw_id=?", (raw_id,)) as rows:
            row = rows.fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def membership_head_revision(self, raw_id: str) -> tuple[object, ...] | None:
        with self._statement(_MEMBERSHIP_HEAD_REVISION_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return None if row is None else tuple(row)

    def membership_has_dangling_append(
        self,
        logical_source_key: str,
        existing_raw_id: str,
        classified_raw_ids: frozenset[str],
    ) -> bool:
        parameters = (logical_source_key, existing_raw_id, *sorted(classified_raw_ids), existing_raw_id)
        predicate = _membership_dangling_append_predicate(classified_raw_ids)
        with self._statement(f"SELECT 1 FROM raw_sessions child WHERE {predicate} LIMIT 1", parameters) as rows:
            return rows.fetchone() is not None

    def raw_path(self, raw_id: str) -> str | None:
        with self._statement(_SOURCE_RAW_PATH_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return str(row[0]) if row is not None and row[0] else None

    def hook_spool_present(self) -> bool:
        with self._statement("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_hook_events'") as rows:
            return rows.fetchone() is not None

    def raw_sessions_present(self) -> bool:
        with self._statement("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'") as rows:
            return rows.fetchone() is not None

    def codex_parent_rows(self, child_native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(_SOURCE_CODEX_PARENT_SQL, (child_native_id,))

    def claude_dispatch_rows(self, parent_native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(
            _source_claude_dispatch_sql(),
            (Origin.CLAUDE_CODE_SESSION.value, parent_native_id, *_AGENT_BEARING_HOOK_EVENTS),
        )

    def acquisition_rows(self, origin: str, native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return self._statement(_SOURCE_ACQUISITION_SQL, (origin, native_id))

    def sidecar_page(
        self,
        source_path: str,
        after: tuple[int, str] | None,
    ) -> list[tuple[str, bytes, str, str, int]]:
        with self._statement(_SOURCE_SIDECAR_SQL, _source_sidecar_page_parameters(source_path, after)) as rows:
            return _source_sidecar_page(rows)

    def open_sidecar_payload(self, raw_id: str, blob_hash: bytes) -> AbstractContextManager[BinaryIO]:
        return self.blob_store.open(blob_hash.hex())


class PreparedSessionSourceRead:
    """Read selected Source state through the original witness's pinned inputs.

    The caller owns the original read window and Source producer context.
    Exact predicates hydrate original matches; touched coordinates remain
    suppressed, while canonical queries also include newly staged matches.
    """

    def __init__(self, seal: PreparedIndexMutation, *, blob_store: BlobStore) -> None:
        if blob_store.root.resolve() != (seal.archive_root / "blob").resolve():
            raise ReferenceSealError("prepared Source reads require their original archive's blob capability")
        self._seal = seal
        self.blob_store = blob_store

    def raw_revision_lineage(self, raw_id: str) -> tuple[str | None, str | None] | None:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(_SOURCE_RAW_REVISION_LINEAGE_SQL, (raw_id,)) as rows:
            return _raw_revision_lineage_from_row(rows.fetchone())

    def session_enrichment_key(
        self,
        index: sqlite3.Connection,
        *,
        provider: Provider,
        source_path: str | None,
        native_id: str,
    ) -> str | None:
        if index is not self._seal.observer("index"):
            raise ReferenceSealError("enrichment preparation requires this Source reader's original Index observer")
        from polylogue.sources.revision_backfill import session_enrichment_evidence_key_from_reader

        return session_enrichment_evidence_key_from_reader(
            provider=provider,
            source_path=source_path,
            native_id=native_id,
            evidence_reader=self,
        )

    def hook_tool_responses(
        self,
        *,
        origin: str,
        session_native_ids: Iterable[str],
        tool_use_ids: Iterable[str],
    ) -> Mapping[str, object]:
        """Read matching hook payloads inside this prepared Source fence.

        Candidate rows are selected by provider-native session and tool-call
        identity. Original rows are hydrated a page at a time into the sealed
        Source view; decoding then streams from that view in bounded batches.
        """
        from polylogue.sources.live.hook_tool_response import hook_tool_responses_from_rows

        natives = tuple(dict.fromkeys(value for value in session_native_ids if value))
        tools = tuple(dict.fromkeys(value for value in tool_use_ids if value))
        if not natives or not tools:
            return {}
        native_marks = ", ".join("?" for _ in natives)
        result: dict[str, object] = {}
        for start in range(0, len(tools), 128):
            batch = tools[start : start + 128]
            tool_marks = ", ".join("?" for _ in batch)
            predicate = (
                f"origin=? AND session_native_id IN ({native_marks}) AND event_type='PostToolUse' "
                "AND json_valid(payload_json) "
                "AND json_extract(payload_json, '$.payload.tool_use_id') IN (" + tool_marks + ")"
            )
            base_parameters = (origin, *natives, *batch)
            after_observed: int | None = None
            after_event: str | None = None
            while True:
                check_compute_cancelled()
                with self._seal.original_rows(
                    "source",
                    f"SELECT rowid, observed_at_ms, hook_event_id FROM raw_hook_events WHERE {predicate} "
                    "AND (? IS NULL OR observed_at_ms>? OR (observed_at_ms=? AND hook_event_id>?)) "
                    "ORDER BY observed_at_ms, hook_event_id LIMIT 256",
                    (*base_parameters, after_observed, after_observed, after_observed, after_event),
                ) as rows:
                    page = tuple(rows.fetchall())
                if not page:
                    break
                for _rowid, _observed, event_id in page:
                    check_compute_cancelled()
                    self._load_matches(
                        "raw_hook_events",
                        "SELECT rowid FROM raw_hook_events WHERE hook_event_id=?",
                        (event_id,),
                    )
                _rowid, after_observed, after_event = page[-1]

            with self._seal.source_rows(
                f"SELECT hook_event_id, payload_json FROM raw_hook_events WHERE {predicate} "
                "ORDER BY observed_at_ms, hook_event_id",
                base_parameters,
            ) as rows:
                while event_page := rows.fetchmany(256):
                    check_compute_cancelled()
                    result.update(hook_tool_responses_from_rows(event_page, tool_use_ids=batch))
        return result

    def hook_tool_response_evidence_digest(
        self,
        *,
        origin: str,
        session_native_ids: Iterable[str],
    ) -> str | None:
        """Digest every matching PostToolUse row within the prepared Source fence."""
        from polylogue.sources.live.hook_tool_response import hook_tool_response_evidence_digest

        natives = tuple(dict.fromkeys(value for value in session_native_ids if value))
        if not natives:
            return None
        native_marks = ", ".join("?" for _ in natives)
        predicate = f"origin=? AND session_native_id IN ({native_marks}) AND event_type='PostToolUse'"
        parameters = (origin, *natives)
        after_rowid = 0
        while True:
            check_compute_cancelled()
            with self._seal.original_rows(
                "source",
                f"SELECT rowid FROM raw_hook_events WHERE {predicate} AND rowid>? ORDER BY rowid LIMIT 256",
                (*parameters, after_rowid),
            ) as rows:
                page = tuple(int(row[0]) for row in rows.fetchall())
            if not page:
                break
            for rowid in page:
                self._load_matches("raw_hook_events", "SELECT rowid FROM raw_hook_events WHERE rowid=?", (rowid,))
            after_rowid = page[-1]
        with self._seal.source_rows(
            f"SELECT hook_event_id, payload_json FROM raw_hook_events WHERE {predicate} ORDER BY hook_event_id",
            parameters,
        ) as rows:
            return hook_tool_response_evidence_digest(rows)

    def _load_matches(self, table: str, sql: str, parameters: tuple[object, ...]) -> None:

        with self._seal.original_rows("source", sql, parameters) as rows:
            for (rowid,) in rows:
                check_compute_cancelled()
                with self._seal.verified_namespace():
                    if self._seal.source_row_is_loaded(table, rowid):
                        continue
                    image = self._seal.retain_tier_row("source", table, rowid)
                    if image is not None:
                        self._seal.load_source_row(image)

    @property
    def archive_root(self) -> Path:
        return self._seal.archive_root

    def retained_state_titles(self, thread_ids: Iterable[str], source_path: str | None) -> Mapping[str, str] | None:
        """Spool canonical titles from the same original enrolled Index owner."""
        from polylogue.sources.codex_state_projection import iter_thread_title_candidates
        from polylogue.sources.retained_title_index import RetainedTitleIndex

        if not self._seal.has_tier_capability("index"):
            return None
        with closing(
            iter_thread_title_candidates(
                self._seal.observer("index"),
                thread_ids=thread_ids,
                source_path=source_path,
                before_input=self._seal.before_index_input,
            )
        ) as rows:
            return RetainedTitleIndex(rows)

    def retained_assembly_evidence(
        self,
        sidecar_data: SidecarData,
        *,
        provider: Provider,
        source_path: str,
        captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    ) -> SidecarData:
        from polylogue.sources.retained_assembly import with_retained_assembly_evidence

        return with_retained_assembly_evidence(
            sidecar_data,
            provider=provider,
            source_read=self,
            blob_store=self.blob_store,
            source_path=source_path,
            captured_zip_coordinate=captured_zip_coordinate,
        )

    def completed_source_item_rows(self, raw_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.sqlite.archive_tiers.source_items import _COMPLETED_SOURCE_ITEM_SQL

        # Membership hydration includes its exact raw/item/generation FK
        # dependencies. The native owner retains staged versions and includes
        # new matches; an omitted original member never proves absence.
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        self._load_matches(
            "source_item_raw_members",
            "SELECT rowid FROM source_item_raw_members WHERE raw_id=?",
            (raw_id,),
        )
        return self._seal.source_rows(_COMPLETED_SOURCE_ITEM_SQL, (raw_id,))

    def _load_assembly_raw_dependencies(self, raw_ids_sql: str, parameters: tuple[object, ...]) -> None:

        after: str | None = None
        while True:
            check_compute_cancelled()
            with self._seal.source_rows(
                f"SELECT raw_id FROM ({raw_ids_sql}) WHERE (? IS NULL OR raw_id>?) "
                "GROUP BY raw_id ORDER BY raw_id LIMIT 256",
                (*parameters, after, after),
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                return
            for raw_id in page:
                check_compute_cancelled()
                self._load_matches(
                    "blob_refs",
                    "SELECT rowid FROM blob_refs WHERE ref_id=? AND ref_type='raw_payload'",
                    (raw_id,),
                )
                self._load_matches(
                    "raw_container_coordinates",
                    "SELECT rowid FROM raw_container_coordinates WHERE raw_id=?",
                    (raw_id,),
                )
                self._load_matches("raw_artifacts", "SELECT rowid FROM raw_artifacts WHERE raw_id=?", (raw_id,))
            after = page[-1]

    def retained_artifact_page(
        self,
        origin: str,
        artifact_kind: str,
        coordinate: str,
        *,
        prefix: bool,
        after: str | None,
    ) -> list[tuple[str, str, str, int, int | None, int | None, str | None]]:
        from polylogue.sources.retained_assembly import (
            _retained_artifact_page_from_rows,
            _retained_artifact_page_sql,
            _retained_artifact_predicate,
        )

        predicate = f"a.origin=? AND ({_retained_artifact_predicate(prefix)}) AND a.artifact_kind=?"
        parameters = (origin, coordinate, artifact_kind)
        if after is None:
            # The exact original predicate hydrates all matching coordinates,
            # while the Native owner suppresses touched original versions.
            # Canonical ranking then includes every new/staged matching row.
            self._load_matches("raw_artifacts", f"SELECT a.rowid FROM raw_artifacts a WHERE {predicate}", parameters)
            self._load_assembly_raw_dependencies(
                f"SELECT a.raw_id FROM raw_artifacts a WHERE {predicate}",
                parameters,
            )
        with self._seal.source_rows(
            _retained_artifact_page_sql(prefix),
            (*parameters, after, after),
        ) as rows:
            return _retained_artifact_page_from_rows(rows)

    def retained_group_member(
        self,
        generation: str,
        item: str,
        source_path: str,
        origin: str,
        artifact_kind: str,
    ) -> tuple[str, str, int, str | None] | None:
        from polylogue.sources.retained_assembly import _RETAINED_GROUP_MEMBER_SQL, _retained_group_member_from_row

        artifact_parameters = (source_path, origin, artifact_kind)
        artifact_predicate = "source_path=? AND origin=? AND artifact_kind=?"
        self._load_matches(
            "raw_artifacts",
            f"SELECT rowid FROM raw_artifacts WHERE {artifact_predicate}",
            artifact_parameters,
        )
        self._load_assembly_raw_dependencies(
            f"SELECT raw_id FROM raw_artifacts WHERE {artifact_predicate}",
            artifact_parameters,
        )
        # A staged artifact can newly match this path even though its original
        # version did not. Select its actual current raw coordinate first, then
        # hydrate the exact original membership for that raw/group. No whole
        # acquisition group is rescanned for each artifact path.
        after: str | None = None
        while True:
            check_compute_cancelled()
            with self._seal.source_rows(
                f"SELECT raw_id FROM raw_artifacts WHERE {artifact_predicate} AND (? IS NULL OR raw_id>?) "
                "GROUP BY raw_id ORDER BY raw_id LIMIT 256",
                (*artifact_parameters, after, after),
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                break
            for raw_id in page:
                self._load_matches(
                    "source_item_raw_members",
                    "SELECT rowid FROM source_item_raw_members "
                    "WHERE source_generation_id=? AND source_item_id=? AND raw_id=?",
                    (generation, item, raw_id),
                )
            after = page[-1]
        with self._seal.source_rows(
            _RETAINED_GROUP_MEMBER_SQL,
            (generation, item, source_path, origin, artifact_kind),
        ) as rows:
            return _retained_group_member_from_row(rows.fetchone())

    def retained_attachment_reference(
        self, raw_id: str, raw_blob_hash: bytes, coordinate: str, blob_hash: bytes, size_bytes: int
    ) -> None:
        from polylogue.storage.blob_publication import (
            _RETAINED_ATTACHMENT_REFERENCE_SQL,
            _require_retained_attachment_reference,
        )

        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        self._load_matches(
            "blob_refs",
            "SELECT rowid FROM blob_refs WHERE ref_id=? AND ref_type='attachment' "
            "AND coalesce(source_path,'')=? AND blob_hash=?",
            (raw_id, coordinate, blob_hash),
        )
        with self._seal.source_rows(
            _RETAINED_ATTACHMENT_REFERENCE_SQL, (raw_id, raw_blob_hash, coordinate, blob_hash, size_bytes)
        ) as rows:
            _require_retained_attachment_reference(rows.fetchone())

    def publication_source_path(self) -> Path:
        # The original observer's actual main identity belongs to this archive.
        # The selected witness main is private preparation, never a Source target.
        with self._seal.original_rows("source", "PRAGMA database_list") as rows:
            database_path = next((str(row[2]) for row in rows if row[1] == "main"), "")
        if not database_path:
            raise PreparedSessionWriteRefusedError("published claim has no original Source target")
        return Path(database_path).resolve()

    def publication_blob_is_excised(self, blob_hash: bytes) -> bool:
        self._load_matches(
            "excised_content",
            "SELECT rowid FROM excised_content WHERE removed_hash=? AND hash_kind='blob_hash'",
            (blob_hash,),
        )
        with self._seal.source_rows(
            "SELECT 1 FROM excised_content WHERE removed_hash=? AND hash_kind='blob_hash' LIMIT 1",
            (blob_hash,),
        ) as rows:
            return rows.fetchone() is not None

    def publication_reservation(self, publication_id: str) -> tuple[bytes, int, str] | None:
        from polylogue.storage.blob_publication import _PUBLICATION_RESERVATION_SQL, _publication_reservation_from_row

        self._load_matches(
            "blob_publication_reservations",
            "SELECT rowid FROM blob_publication_reservations WHERE publication_id=?",
            (publication_id,),
        )
        with self._seal.source_rows(_PUBLICATION_RESERVATION_SQL, (publication_id,)) as rows:
            return _publication_reservation_from_row(rows.fetchone())

    def raw_selection_values(
        self,
        family: RawMembershipSelectionFamily,
        operands: Sequence[str],
    ) -> set[str]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _raw_selection_queries

        found: set[str] = set()
        if not operands:
            return found
        for table, column, predicate, parameters in _raw_selection_queries(family, tuple(operands)):
            self._load_matches(table, f"SELECT rowid FROM {table} WHERE {predicate}", parameters)
            with self._seal.source_rows(f"SELECT {column} FROM {table} WHERE {predicate}", parameters) as rows:
                for row in rows:
                    check_compute_cancelled()
                    found.add(str(row[0]))
        return found

    def expand_raw_membership_selection(self, raw_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _expand_raw_membership_selection

        return _expand_raw_membership_selection(self, raw_ids)

    def raw_membership_census_rows(self, raw_ids: Sequence[str]) -> tuple[tuple[str, int, bool, int], ...]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _load_parser_census_source_inputs,
            _raw_membership_census_query,
            _raw_membership_census_rows_from_rows,
        )

        for raw_id in raw_ids:
            _load_parser_census_source_inputs(self._seal, raw_id)
        with self._seal.source_rows(*_raw_membership_census_query(raw_ids)) as rows:
            return _raw_membership_census_rows_from_rows(rows)

    def membership_key_has_pending_envelope_member(self, logical_source_key: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _MEMBERSHIP_PENDING_ENVELOPE_SQL,
            _load_membership_selector_inputs,
        )
        from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX

        _load_membership_selector_inputs(self._seal, logical_source_key, None)
        with self._seal.source_rows(
            _MEMBERSHIP_PENDING_ENVELOPE_SQL,
            (logical_source_key, len(PENDING_RAW_LOGICAL_SOURCE_PREFIX), PENDING_RAW_LOGICAL_SOURCE_PREFIX),
        ) as rows:
            return rows.fetchone() is not None

    def raw_revision_replay_plan(self, logical_source_key: str) -> RevisionReplayPlan:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_revision_replay_plan

        return prepared_raw_revision_replay_plan(self._seal, logical_source_key)

    def pending_raw_envelope_has_membership_authority(self, logical_source_key: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _PENDING_ENVELOPE_MEMBERSHIP_SQL,
            _load_parser_census_source_inputs,
        )
        from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX

        if not logical_source_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX):
            return False
        self._load_matches(
            "raw_sessions",
            "SELECT rowid FROM raw_sessions WHERE logical_source_key=?",
            (logical_source_key,),
        )
        after: str | None = None
        while True:
            with self._seal.source_rows(
                "SELECT raw_id FROM raw_sessions WHERE logical_source_key=? "
                "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                (logical_source_key, after, after),
            ) as rows:
                page = rows.fetchall()
            if not page:
                break
            for (raw_id,) in page:
                check_compute_cancelled()
                _load_parser_census_source_inputs(self._seal, str(raw_id))
            after = str(page[-1][0])
        with self._seal.source_rows(_PENDING_ENVELOPE_MEMBERSHIP_SQL, (logical_source_key,)) as rows:
            return rows.fetchone() is not None

    def raw_has_membership_authority(self, raw_id: str) -> bool:
        self._load_matches(
            "raw_session_memberships",
            "SELECT rowid FROM raw_session_memberships WHERE raw_id=?",
            (raw_id,),
        )
        with self._seal.source_rows(
            "SELECT 1 FROM raw_session_memberships WHERE raw_id=? LIMIT 1",
            (raw_id,),
        ) as rows:
            return rows.fetchone() is not None

    def raw_has_membership_governed_pending_envelope(self, raw_id: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _RAW_PENDING_MEMBERSHIP_SQL,
            _load_parser_census_source_inputs,
        )
        from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX

        _load_parser_census_source_inputs(self._seal, raw_id)
        with self._seal.source_rows(
            _RAW_PENDING_MEMBERSHIP_SQL,
            (raw_id, len(PENDING_RAW_LOGICAL_SOURCE_PREFIX), PENDING_RAW_LOGICAL_SOURCE_PREFIX),
        ) as rows:
            return rows.fetchone() is not None

    def raw_revision_rebuild_logical_keys(self, raw_ids: Sequence[str]) -> tuple[str, ...]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _raw_revision_rebuild_logical_keys

        return _raw_revision_rebuild_logical_keys(self, raw_ids)

    def replay_representative_rows(self, keys: Sequence[str]) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _load_raw_observation_inputs,
            _raw_replay_representative_query,
        )

        marks = ",".join("?" for _ in keys)
        parameters = tuple(keys)
        self._load_matches(
            "raw_sessions",
            f"SELECT rowid FROM raw_sessions WHERE logical_source_key IN ({marks})",
            parameters,
        )
        self._load_matches(
            "raw_session_memberships",
            f"SELECT rowid FROM raw_session_memberships WHERE logical_source_key IN ({marks})",
            parameters,
        )
        after: str | None = None
        while True:
            with self._seal.source_rows(
                f"SELECT raw_id FROM (SELECT raw_id FROM raw_sessions WHERE logical_source_key IN ({marks}) "
                f"UNION SELECT raw_id FROM raw_session_memberships WHERE logical_source_key IN ({marks})) "
                "WHERE (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                (*parameters, *parameters, after, after),
            ) as rows:
                page = rows.fetchall()
            if not page:
                break
            for (raw_id,) in page:
                _load_raw_observation_inputs(self._seal, str(raw_id))
            after = str(page[-1][0])
        return self._seal.source_rows(*_raw_replay_representative_query(keys))

    def raw_membership_rebuild_raw_ids(self, logical_source_key: str) -> tuple[str, ...]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_membership_rebuild_raw_ids

        return prepared_raw_membership_rebuild_raw_ids(self._seal, logical_source_key)

    def raw_membership_logical_raw_ids(self, logical_source_key: str) -> tuple[str, ...]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _MEMBERSHIP_LOGICAL_RAW_IDS_SQL

        self._load_matches(
            "raw_session_memberships",
            "SELECT rowid FROM raw_session_memberships WHERE logical_source_key=?",
            (logical_source_key,),
        )
        with self._seal.source_rows(_MEMBERSHIP_LOGICAL_RAW_IDS_SQL, (logical_source_key,)) as rows:
            return tuple(str(row[0]) for row in rows)

    def raw_revision_head_raw_id(self, logical_source_key: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _RAW_REVISION_HEAD_SQL,
            _raw_revision_head_from_row,
        )

        if not self._seal.has_tier_capability("index"):
            return None
        parameters = (logical_source_key,)
        self._seal.before_index_input(
            "raw_revision_heads",
            ("accepted_raw_id",),
            "SELECT rowid FROM raw_revision_heads WHERE logical_source_key=?",
            parameters,
        )
        with self._seal.original_rows("index", _RAW_REVISION_HEAD_SQL, parameters) as rows:
            return _raw_revision_head_from_row(rows.fetchone())

    def adoption_session_hash(self, session_id: str) -> tuple[object, ...] | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _REPLAY_ADOPTION_SESSION_SQL

        parameters = (session_id,)
        self._seal.before_index_input(
            "sessions",
            ("content_hash",),
            "SELECT rowid FROM sessions WHERE session_id=?",
            parameters,
        )
        with self._seal.original_rows("index", _REPLAY_ADOPTION_SESSION_SQL, parameters) as rows:
            row = rows.fetchone()
        return None if row is None else tuple(row)

    def adoption_session_is_governed(self, session_id: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _REPLAY_ADOPTION_HEAD_SQL

        parameters = (session_id,)
        self._seal.before_index_input(
            "raw_revision_heads",
            ("session_id",),
            "SELECT rowid FROM raw_revision_heads WHERE session_id=?",
            parameters,
        )
        with self._seal.original_rows("index", _REPLAY_ADOPTION_HEAD_SQL, parameters) as rows:
            return rows.fetchone() is not None

    def adoption_raw_revision(self, logical_source_key: str, raw_id: str) -> tuple[str, int]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
            _REPLAY_ADOPTION_RAW_REVISION_SQL,
            _load_raw_session_input,
        )

        _load_raw_session_input(self._seal, raw_id)
        parameters = (raw_id, logical_source_key)
        with self._seal.original_rows(
            "source",
            "SELECT rowid FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
            parameters,
        ) as rows:
            for (rowid,) in rows:
                check_compute_cancelled()
                if self._seal.source_row_is_loaded("raw_session_memberships", rowid):
                    continue
                image = self._seal.retain_tier_row("source", "raw_session_memberships", rowid)
                if image is not None:
                    self._seal.load_source_row(image)
        with self._seal.source_rows(_REPLAY_ADOPTION_RAW_REVISION_SQL, (logical_source_key, raw_id)) as rows:
            row = rows.fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"deferred raw revision lacks source evidence: {raw_id}")
        return str(row[0]), int(row[1])

    def prepare_raw_revision_replay_adoption(
        self,
        sessions: Sequence[ParsedSession],
        *,
        logical_source_key: str,
        raw_ids: Sequence[str],
    ) -> PreparedRevisionAdoption:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepare_raw_revision_replay_adoption

        return prepare_raw_revision_replay_adoption(
            self, sessions, logical_source_key=logical_source_key, raw_ids=raw_ids
        )

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_revision_descriptor

        return prepared_raw_revision_descriptor(self._seal, raw_id)

    def raw_validation_mode(self, raw_id: str) -> str | None:
        """Read the persisted validation policy bound to this raw revision."""
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows("SELECT validation_mode FROM raw_sessions WHERE raw_id=?", (raw_id,)) as rows:
            row = rows.fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def raw_schema_eligible(self, raw_id: str) -> bool:
        """Whether this typed artifact participates in schema validation.

        Missing or mixed artifact evidence is treated as eligible. Explicit
        captured native grammar and non-session classifications may declare
        that JSON schema validation does not apply.
        """
        self._load_matches("raw_artifacts", "SELECT rowid FROM raw_artifacts WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(
            "SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM raw_artifacts WHERE raw_id=?) "
            "OR EXISTS (SELECT 1 FROM raw_artifacts WHERE raw_id=? "
            "AND schema_eligible IS NOT 0) THEN 1 ELSE 0 END",
            (raw_id, raw_id),
        ) as rows:
            row = rows.fetchone()
        return row is not None and bool(row[0])

    def raw_parser_confirmed_non_session(self, raw_id: str) -> bool:
        """Whether current parser authority terminally classifies this raw as non-session."""
        from polylogue.archive.revision_authority import raw_authority_parser_fingerprint

        self._load_matches("raw_membership_census", "SELECT rowid FROM raw_membership_census WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(
            "SELECT status, parser_fingerprint FROM raw_membership_census WHERE raw_id=?", (raw_id,)
        ) as rows:
            row = rows.fetchone()
        return bool(row is not None and row[0] == "non_session" and row[1] == raw_authority_parser_fingerprint())

    def raw_revision_file_mtime(self, raw_id: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_revision_file_mtime

        return prepared_raw_revision_file_mtime(self._seal, raw_id)

    def raw_native_id(self, raw_id: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_native_id

        return prepared_raw_native_id(self._seal, raw_id)

    def uncensused_identity_opaque_raw_ids(self, raw_ids: Sequence[str]) -> tuple[str, ...]:
        """Selected raws with no native id or typed key whose parser census is not current.

        Only its parsed content assigns such an envelope to a logical cohort,
        so another envelope's replay cannot know whether it shares that cohort.
        """
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_parser_census_is_current
        from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX

        opaque: list[str] = []
        for start in range(0, len(raw_ids), 256):
            check_compute_cancelled()
            chunk = tuple(raw_ids[start : start + 256])
            marks = ",".join("?" for _ in chunk)
            self._load_matches("raw_sessions", f"SELECT rowid FROM raw_sessions WHERE raw_id IN ({marks})", chunk)
            with self._seal.source_rows(
                f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({marks}) AND native_id IS NULL "
                "AND (logical_source_key IS NULL OR substr(logical_source_key,1,?)=?) ORDER BY raw_id",
                (*chunk, len(PENDING_RAW_LOGICAL_SOURCE_PREFIX), PENDING_RAW_LOGICAL_SOURCE_PREFIX),
            ) as rows:
                envelopes = tuple(str(row[0]) for row in rows)
            opaque.extend(raw_id for raw_id in envelopes if not prepared_parser_census_is_current(self._seal, raw_id))
        return tuple(opaque)

    def raw_ids_for_native_session(self, origin: str, native_id: str) -> tuple[str, ...]:
        """Retained raws acquired as one origin's native session, before or after census."""
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _load_raw_observation_inputs

        self._load_matches(
            "raw_sessions",
            "SELECT rowid FROM raw_sessions WHERE origin=? AND native_id=?",
            (origin, native_id),
        )
        with self._seal.source_rows(
            "SELECT raw_id FROM raw_sessions WHERE origin=? AND native_id=? ORDER BY raw_id",
            (origin, native_id),
        ) as rows:
            raw_ids = tuple(str(row[0]) for row in rows)
        for raw_id in raw_ids:
            _load_raw_observation_inputs(self._seal, raw_id)
        return raw_ids

    @contextmanager
    def open_raw_revision_material(
        self,
        raw_id: str,
    ) -> Generator[tuple[Provider, BinaryIO, str, RawRevisionKind], None, None]:
        provider, blob_hash, source_path, kind, _size = self.raw_revision_descriptor(raw_id)
        self.raw_revision_blob_path(raw_id)
        with self.blob_store.open(blob_hash) as payload:
            yield provider, payload, source_path, kind

    def raw_revision_material(self, raw_id: str) -> tuple[Provider, bytes, str, RawRevisionKind]:
        with self.open_raw_revision_material(raw_id) as (provider, payload, source_path, kind):
            return provider, payload.read(), source_path, kind

    @contextmanager
    def open_raw_container_material(self, raw_id: str) -> Generator[BinaryIO | None, None, None]:
        """Open the accepted physical container a ZIP member raw was read from.

        The container identity comes from the member's pinned coordinate
        receipt; its bytes are charged as an original input before the read.
        """
        coordinate = self.raw_captured_zip_coordinate(raw_id)
        if coordinate is None:
            raise ValueError(f"raw has no captured ZIP container: {raw_id}")
        path = self.blob_store.blob_path(coordinate.container_blob_hash)
        if not path.exists():
            yield None
            return
        self._seal.retain_original_container_input(bytes.fromhex(coordinate.container_blob_hash), path.stat().st_size)
        with self.blob_store.open(coordinate.container_blob_hash) as container:
            yield container

    def raw_revision_blob_path(self, raw_id: str) -> Path | None:
        """Enroll this original retained acquisition before any payload access.

        The Native input ledger validates the same pinned raw descriptor and
        charges its current reservation before a parser opens the file. This
        capability cannot pretend a newly staged acquisition was original.
        """
        _provider, blob_hash, _source_path, _kind, size = self.raw_revision_descriptor(raw_id)
        retained_hash, retained_size = self._seal.retain_original_blob_input(raw_id)
        if retained_hash.hex() != blob_hash or retained_size != size:
            raise ReferenceSealError("retained parser input changed its original acquisition bytes")
        path = self.blob_store.blob_path(blob_hash)
        return path if path.exists() else None

    def raw_append_logical_key(self, raw_id: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.source_write import (
            _RAW_APPEND_LOGICAL_KEY_SQL,
            _raw_append_logical_key_from_row,
        )

        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(_RAW_APPEND_LOGICAL_KEY_SQL, (raw_id,)) as rows:
            return _raw_append_logical_key_from_row(rows.fetchone())

    def raw_profile_identity(self, raw_id: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.source_write import (
            _RAW_PROFILE_IDENTITY_SQL,
            _raw_profile_identity_from_row,
        )

        self._load_matches(
            "raw_profile_identity_receipts",
            "SELECT rowid FROM raw_profile_identity_receipts WHERE raw_id=?",
            (raw_id,),
        )
        with self._seal.source_rows(_RAW_PROFILE_IDENTITY_SQL, (raw_id,)) as rows:
            return _raw_profile_identity_from_row(rows.fetchone())

    def raw_captured_zip_coordinate(self, raw_id: str) -> CapturedZipMemberCoordinate | None:
        from polylogue.storage.sqlite.archive_tiers.source_write import (
            _RAW_CAPTURED_ZIP_COORDINATE_SQL,
            _raw_captured_zip_coordinate_from_row,
        )

        self._load_matches(
            "raw_container_coordinates",
            "SELECT rowid FROM raw_container_coordinates WHERE raw_id=?",
            (raw_id,),
        )
        with self._seal.source_rows(_RAW_CAPTURED_ZIP_COORDINATE_SQL, (raw_id,)) as rows:
            return _raw_captured_zip_coordinate_from_row(rows.fetchone())

    def raw_export_order(self, raw_id: str) -> int | None:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_export_order

        return prepared_raw_export_order(self._seal, raw_id)

    def raw_revision_observation_order(self, raw_id: str) -> tuple[int, int]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_revision_observation_order

        return prepared_raw_revision_observation_order(self._seal, raw_id)

    def raw_has_pending_envelope(self, raw_id: str) -> bool:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(_RAW_PENDING_ENVELOPE_SQL, (raw_id,)) as rows:
            return _raw_pending_envelope_from_row(rows.fetchone())

    def raw_revision_authority(self, raw_id: str) -> str | None:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows("SELECT revision_authority FROM raw_sessions WHERE raw_id=?", (raw_id,)) as rows:
            row = rows.fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def membership_head_revision(self, raw_id: str) -> tuple[object, ...] | None:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(_MEMBERSHIP_HEAD_REVISION_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return None if row is None else tuple(row)

    def membership_has_dangling_append(
        self,
        logical_source_key: str,
        existing_raw_id: str,
        classified_raw_ids: frozenset[str],
    ) -> bool:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (existing_raw_id,))
        # Load the whole exact logical-key predicate before the canonical
        # negative read. A newly staged parent revision can change which
        # original child matches, and touched originals stay suppressed.
        self._load_matches(
            "raw_sessions", "SELECT rowid FROM raw_sessions WHERE logical_source_key=?", (logical_source_key,)
        )
        parameters = (logical_source_key, existing_raw_id, *sorted(classified_raw_ids), existing_raw_id)
        predicate = _membership_dangling_append_predicate(classified_raw_ids)
        with self._seal.source_rows(f"SELECT 1 FROM raw_sessions child WHERE {predicate} LIMIT 1", parameters) as rows:
            return rows.fetchone() is not None

    def raw_path(self, raw_id: str) -> str | None:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(_SOURCE_RAW_PATH_SQL, (raw_id,)) as rows:
            row = rows.fetchone()
        return str(row[0]) if row is not None and row[0] else None

    def hook_spool_present(self) -> bool:
        with self._seal.original_rows(
            "source", "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_hook_events'"
        ) as rows:
            return rows.fetchone() is not None

    def raw_sessions_present(self) -> bool:
        with self._seal.original_rows(
            "source", "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'"
        ) as rows:
            return rows.fetchone() is not None

    def codex_parent_rows(self, child_native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_matches(
            "raw_hook_events",
            "SELECT rowid FROM raw_hook_events WHERE event_type='codex_thread_spawn_edge' "
            "AND json_extract(payload_json, '$.child_thread_id')=?",
            (child_native_id,),
        )
        return self._seal.source_rows(_SOURCE_CODEX_PARENT_SQL, (child_native_id,))

    def claude_dispatch_rows(self, parent_native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        events = ", ".join("?" * len(_AGENT_BEARING_HOOK_EVENTS))
        parameters = (Origin.CLAUDE_CODE_SESSION.value, parent_native_id, *_AGENT_BEARING_HOOK_EVENTS)
        self._load_matches(
            "raw_hook_events",
            f"SELECT rowid FROM raw_hook_events WHERE origin=? AND session_native_id=? AND event_type IN ({events})",
            parameters,
        )
        return self._seal.source_rows(_source_claude_dispatch_sql(), parameters)

    def acquisition_rows(self, origin: str, native_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        self._load_matches(
            "raw_sessions", "SELECT rowid FROM raw_sessions WHERE origin=? AND native_id=?", (origin, native_id)
        )
        return self._seal.source_rows(_SOURCE_ACQUISITION_SQL, (origin, native_id))

    def _load_retained_scope(self, predicate: str, parameters: tuple[object, ...]) -> None:

        self._load_matches("raw_sessions", f"SELECT rowid FROM raw_sessions WHERE {predicate}", parameters)
        after: str | None = None
        while True:
            check_compute_cancelled()
            with self._seal.source_rows(
                f"SELECT raw_id FROM raw_sessions WHERE ({predicate}) "
                "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                (*parameters, after, after),
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                return
            for raw_id in page:
                check_compute_cancelled()
                self._load_matches(
                    "blob_refs",
                    "SELECT rowid FROM blob_refs WHERE ref_id=? AND ref_type='raw_payload'",
                    (raw_id,),
                )
            after = page[-1]

    def retained_children_rows(self, low: str, high: str) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.sources.live.sidecar_resolution import _RETAINED_CHILDREN_SQL

        self._load_retained_scope("source_path>=? AND source_path<?", (low, high))
        return self._seal.source_rows(_RETAINED_CHILDREN_SQL, (low, high))

    def retained_sibling_rows(
        self,
        root_path: str,
        low: str,
        high: str,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.sources.live.sidecar_resolution import _RETAINED_SIBLINGS_SQL

        parameters = (root_path, low, high)
        self._load_retained_scope("source_path=? OR (source_path>=? AND source_path<?)", parameters)
        return self._seal.source_rows(_RETAINED_SIBLINGS_SQL, parameters)

    def retained_sidecar_resolver(self) -> SidecarResolver:
        from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

        return RetainedSidecarResolver(self.archive_root, blob_root=self.blob_store.root, source_read=self)

    def sidecar_page(
        self,
        source_path: str,
        after: tuple[int, str] | None,
    ) -> list[tuple[str, bytes, str, str, int]]:
        self._load_matches("raw_sessions", "SELECT rowid FROM raw_sessions WHERE source_path=?", (source_path,))
        with self._seal.source_rows(_SOURCE_SIDECAR_SQL, _source_sidecar_page_parameters(source_path, after)) as rows:
            return _source_sidecar_page(rows)

    def open_sidecar_payload(self, raw_id: str, blob_hash: bytes) -> AbstractContextManager[BinaryIO]:
        retained_hash, _size = self._seal.retain_original_blob_input(raw_id)
        if retained_hash != blob_hash:
            raise ReferenceSealError("selected sidecar changed its original acquisition bytes")
        return self.blob_store.open(blob_hash.hex())

    def raw_terminal_decode_refusal(self, raw_id: str) -> RetainedRawDecodeRefusalError | None:
        """Read current refusal through this preparation's original Source inputs."""
        from polylogue.archive.revision_authority import raw_authority_parser_fingerprint
        from polylogue.core.raw_failure_evidence import (
            RAW_FAILURE_VALIDATION_FAILURE_KINDS,
            retained_raw_decode_refusal_from_row,
        )
        from polylogue.storage.sqlite.queries.raw_state import retained_raw_decode_refusal_sql

        for table in ("raw_sessions", "raw_authority_parser_census", "raw_artifacts"):
            self._load_matches(table, f"SELECT rowid FROM {table} WHERE raw_id=?", (raw_id,))
        with self._seal.source_rows(
            retained_raw_decode_refusal_sql(),
            (raw_id, raw_authority_parser_fingerprint(), *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS)),
        ) as rows:
            return retained_raw_decode_refusal_from_row(raw_id, rows.fetchone())

    def raw_parser_census_is_current(self, raw_id: str) -> bool:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_parser_census_is_current

        return prepared_parser_census_is_current(self._seal, raw_id)


def _hook_spool_present(source_read: SessionSourceRead) -> bool:
    """Whether the original Source tier carries the declared hook spool."""
    return source_read.hook_spool_present()


def raw_source_path(source_read: SessionSourceRead | None, raw_id: str | None) -> str | None:
    """The retained path of one raw in the caller's selected Source state."""
    return source_read.raw_path(raw_id) if source_read is not None and raw_id else None


def _codex_spawn_edge_parent_claim(
    conn: sqlite3.Connection,
    source_read: SessionSourceRead | None,
    *,
    child_native_id: str,
    child_source_path: str | None,
    before_input: BeforeIndexInput | None = None,
    row_read: ThreadParentRowRead | None = None,
) -> _HookParentClaim | None:
    """Return projected or spooled Codex parent evidence for a child.

    The retained state export is projected into the index tier and is the
    primary source. A thread id is one install root's own name, so the
    projection is read in the root of the rollout that produced the child
    (``child_source_path``); a child whose rollout path the source tier
    cannot supply reads the parent every root agrees on. Older source tiers
    may carry the same evidence in the durable hook spool, which remains a
    compatible fallback.
    """
    if not child_native_id:
        return None
    try:
        from polylogue.sources.codex_state_projection import read_parent_thread_id

        projected_parent = read_parent_thread_id(
            conn, child_native_id, source_path=child_source_path, before_input=before_input, row_read=row_read
        )
    except (ImportError, sqlite3.Error) as exc:
        # Silence here archives a child as a root with no parent edge and no
        # trace that the projection was ever consulted (polylogue-3r36h). The
        # spool fallback below may still supply the edge; when it does not,
        # this line is the only evidence the lineage was lost to a failure
        # rather than to absent evidence.
        emit(
            "storage.codex_spawn_edge.parent_projection_unavailable",
            level=WARNING,
            outcome="degraded",
            reason="the child is archived as a root because its parent projection could not be read",
            session_id=child_native_id,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        projected_parent = None
    if projected_parent is not None:
        return _HookParentClaim(projected_parent, {"codex_thread_spawn_edge_parent": projected_parent})
    if source_read is None or not _hook_spool_present(source_read):
        return None
    with source_read.codex_parent_rows(child_native_id) as lineage_rows:
        row = lineage_rows.fetchone()
    if row is None or row[0] is None:
        return None
    parent = str(row[0]).strip()
    if not parent:
        return None
    return _HookParentClaim(parent, {"codex_thread_spawn_edge_parent": parent})


def _child_tool_use_id_matches(
    conn: sqlite3.Connection,
    child_session_id: str,
    tool_use_ids: Sequence[str],
    before_input: BeforeIndexInput | None = None,
) -> int:
    """Count the hook-attributed tool calls archived as this child's own blocks."""
    matched = 0
    for start in range(0, len(tool_use_ids), _HOOK_TOOL_ID_CHUNK):
        chunk = tool_use_ids[start : start + _HOOK_TOOL_ID_CHUNK]
        placeholders = ", ".join("?" * len(chunk))
        if before_input is not None:
            before_input(
                "blocks",
                ("tool_id",),
                f"SELECT rowid FROM blocks WHERE session_id=? AND block_type=? AND tool_id IN ({placeholders})",
                (child_session_id, BlockType.TOOL_USE.value, *chunk),
            )
        with connection_cursor(
            conn,
            f"""
            SELECT COUNT(DISTINCT tool_id) FROM blocks
            WHERE session_id = ? AND block_type = ? AND tool_id IN ({placeholders})
            """,
            (child_session_id, BlockType.TOOL_USE.value, *chunk),
        ) as cursor:
            matched += int(cursor.fetchone()[0])
    return matched


def _claude_agent_dispatch_parent_claim(
    conn: sqlite3.Connection,
    source_read: SessionSourceRead,
    *,
    child_session_id: str,
    child_provider_values: Iterable[str],
    parent_candidate: str,
    before_input: BeforeIndexInput | None = None,
) -> _HookParentClaim | None:
    """Return the hook-asserted dispatch of one Claude Code subagent child.

    Claude Code stamps ``agent_id`` (the agent instance) and ``agent_type``
    (the agent definition) on the ``PreToolUse``/``PostToolUse`` payloads of
    the calls a dispatched agent itself made, in the DISPATCHING session's hook
    journal. The dispatching ``Agent`` call carries no such pair, so the pair
    identifies the child rather than the dispatch block: it is the runtime's
    own per-call assertion of which agent instance ran under which session,
    which transcript shape cannot reconstruct.

    The lookup is keyed on ``session_native_id``, the spool's only indexed
    access path, so it costs the named parent's own hook events rather than a
    scan of the spool. It therefore confirms a parent the parser named and
    cannot discover one; hook silence stays silence.

    A claim is admitted only when a ``tool_use_id`` the hook attributed to the
    agent instance is archived as a ``tool_use`` block in this child. That join
    is what makes the assertion about this session rather than about a name
    that merely matches.

    All three fields are lifted by ONE multi-path ``json_extract`` per row.
    Hook payloads carry whole tool inputs and responses, so a per-field
    extraction would re-parse megabytes of JSON for every subagent written.
    """
    agent_ids = sorted(
        {
            value[len(_SUBAGENT_STEM_PREFIX) :]
            for value in child_provider_values
            if value.startswith(_SUBAGENT_STEM_PREFIX) and len(value) > len(_SUBAGENT_STEM_PREFIX)
        }
    )
    if not agent_ids or not _hook_spool_present(source_read):
        return None
    field_slices = _hook_field_slices(_AGENT_DISPATCH_HOOK_KEYS)
    wanted = set(agent_ids)
    tool_use_ids: dict[str, set[str]] = defaultdict(set)
    agent_types: dict[str, str] = {}
    with source_read.claude_dispatch_rows(parent_candidate) as lineage_rows:
        for (extracted,) in lineage_rows:
            fields = _hook_extracted_fields(extracted, field_slices)
            agent_id = fields.get("agent_id")
            tool_use_id = fields.get("tool_use_id")
            if agent_id is None or tool_use_id is None or agent_id not in wanted:
                continue
            tool_use_ids[agent_id].add(tool_use_id)
            agent_type = fields.get("agent_type")
            if agent_type is not None and agent_id not in agent_types:
                agent_types[agent_id] = agent_type
    # One tool_use_id belongs to exactly one agent instance. Two hook records
    # attributing the same id to different agents is a contradiction in the
    # evidence itself, and the sorted iteration below would otherwise resolve
    # it by arbitrary agent-id order. Flag the ambiguity instead: no parent is
    # adopted, and the contested ids are retained as the reason.
    claimants_by_tool_use_id: dict[str, set[str]] = defaultdict(set)
    for agent_id, ids in tool_use_ids.items():
        for tool_use_id in ids:
            claimants_by_tool_use_id[tool_use_id].add(agent_id)
    contested = {tool_use_id for tool_use_id, owners in claimants_by_tool_use_id.items() if len(owners) > 1}
    for agent_id in agent_ids:
        matches = _child_tool_use_id_matches(
            conn, child_session_id, sorted(tool_use_ids.get(agent_id, ())), before_input
        )
        if not matches:
            continue
        contested_ids = sorted(tool_use_ids.get(agent_id, set()) & contested)
        contested_matches = (
            contested_ids if _child_tool_use_id_matches(conn, child_session_id, contested_ids, before_input) else []
        )
        if contested_matches:
            return _HookParentClaim(
                None,
                {
                    "claude_hook_parent_ambiguity": "one tool_use_id is claimed by more than one agent instance",
                    "claude_hook_contested_tool_use_ids": contested_matches,
                    "claude_hook_contested_agent_ids": sorted(
                        {owner for tool_use_id in contested_matches for owner in claimants_by_tool_use_id[tool_use_id]}
                    ),
                },
            )
        return _HookParentClaim(
            parent_candidate,
            {
                "claude_hook_agent_id": agent_id,
                "claude_hook_agent_type": agent_types.get(agent_id),
                "claude_hook_tool_use_id_matches": matches,
            },
        )
    return None


def _child_provider_values(session: ParsedSession) -> tuple[str, ...]:
    """The provider names this child claims for itself, its own id first."""
    values = [
        (session.provider_session_id or "").strip(),
        *(str(alias).strip() for alias in session.provider_session_aliases),
    ]
    return tuple(dict.fromkeys(value for value in values if value))


def _authoritative_parent_claim(
    conn: sqlite3.Connection,
    source_read: SessionSourceRead | None,
    *,
    origin: str,
    child_session_id: str,
    child_native_id: str,
    child_provider_values: Iterable[str],
    parent_candidate: str | None,
    child_source_path: str | None,
    before_input: BeforeIndexInput | None = None,
) -> _HookParentClaim | None:
    """Return the hook-asserted parent for this child, or ``None`` for silence.

    ``None`` means "hook evidence is silent about this child", which is not the
    same as "hook evidence disagrees" -- only the latter is a conflict.

    A Claude Code claim can only be confirmed under a named parent (the spool
    is keyed by the dispatching session), so the parser's candidate is asked
    first and then every parent a preserved authoritative edge of this child
    names. When the parser moves the child from A to B and only A's journal
    attributes the agent's calls, A's durable claim still decides the edge;
    asking the candidate alone would read B's silence as no evidence and leave
    two composing parents.

    ``child_source_path`` is the retained source path of the raw this write
    publishes, which places a Codex child in its install root.
    """
    if not child_native_id:
        return None
    if origin == Origin.CODEX_SESSION.value:
        return _codex_spawn_edge_parent_claim(
            conn,
            source_read,
            child_native_id=child_native_id,
            child_source_path=child_source_path,
            before_input=before_input,
        )
    if origin != Origin.CLAUDE_CODE_SESSION.value:
        return None
    provider_values = tuple(child_provider_values)
    if before_input is not None:
        before_input(
            "session_links",
            ("dst_native_id", "evidence_json"),
            "SELECT rowid FROM session_links WHERE src_session_id=? AND dst_origin=? "
            "AND method=? AND status IS NULL ORDER BY dst_native_id",
            (child_session_id, origin, HOOK_AUTHORITATIVE_LINK_METHOD),
        )
    with connection_cursor(
        conn,
        """
        SELECT dst_native_id, evidence_json FROM session_links
        WHERE src_session_id = ? AND dst_origin = ? AND method = ? AND status IS NULL
        ORDER BY dst_native_id
        """,
        (child_session_id, origin, HOOK_AUTHORITATIVE_LINK_METHOD),
    ) as _input_cursor:
        preserved = _input_cursor.fetchall()
    candidates = dict.fromkeys(
        candidate for candidate in (parent_candidate, *(str(row[0]) for row in preserved)) if candidate
    )
    if source_read is not None:
        for candidate in candidates:
            claim = _claude_agent_dispatch_parent_claim(
                conn,
                source_read,
                child_session_id=child_session_id,
                child_provider_values=provider_values,
                parent_candidate=candidate,
                before_input=before_input,
            )
            if claim is not None:
                return claim
    # A verified, still-active claim does not turn into hook silence merely
    # because this write omitted the source handle or no longer repeats the
    # child's old tool call. A fresh claim above may supersede it; silence may
    # not. Preserve the existing evidence, not a second authority/cache.
    if len(preserved) > 1:
        raise ValueError(f"ambiguous preserved hook parents for {child_session_id}")
    if preserved:
        return _HookParentClaim(str(preserved[0][0]), json.loads(preserved[0][1] or "{}"))
    return None


@dataclass(frozen=True, slots=True)
class CodexProjectionSessionInput:
    physical_rowid: int
    raw_id: str | None
    created_at_ms: int | None
    updated_at_ms: int | None
    parent_session_id: str | None
    root_session_id: str | None
    branch_type: str | None
    session_kind: str | None
    source_path: str | None


@dataclass(frozen=True, slots=True)
class CodexProjectionLinkInput:
    physical_rowid: int
    src_session_id: str
    dst_origin: str
    dst_native_id: str
    link_type: str
    method: str | None
    status: str | None
    observed_at_ms: int | None
    resolved_dst_session_id: str | None
    resolved_at_ms: int | None
    branch_point_message_id: str | None
    branch_point_content_address: bytes | None
    inheritance: str | None
    parent_tool_use_block_id: str | None
    evidence_json: str | None


@dataclass(frozen=True, slots=True)
class PreparedCodexLinkInputs:
    """Reachable original link operands on the graph carrier's same native owner."""

    owner: NativeSQLCustodyOwner
    reference_seal: PreparedIndexMutation

    def session_record(self, session_id: str) -> CodexProjectionSessionInput | None:
        with connection_cursor(
            self.owner.require_connection(),
            "SELECT physical_rowid,raw_id,created_at_ms,updated_at_ms,parent_session_id,"
            "root_session_id,branch_type,session_kind,source_path "
            "FROM codex_link_input_sessions WHERE session_id=?",
            (session_id,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Codex projection requested an unprepared original session coordinate")
        return None if row[0] is None else CodexProjectionSessionInput(*tuple(row))

    def link_record(self, physical_rowid: int) -> CodexProjectionLinkInput:
        with connection_cursor(
            self.owner.require_connection(),
            "SELECT physical_rowid,"
            + ",".join(_CODEX_LINK_FIELDS)
            + " FROM codex_link_input_links WHERE physical_rowid=?",
            (physical_rowid,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Codex projection requested an unprepared original link coordinate")
        return CodexProjectionLinkInput(*tuple(row))


class PreparedCodexSpawnParentLinks:
    """Serve retained original operands and explicitly written Index cells.

    Native selectors retain the writer's sequential row ordering. Only a
    column recorded by this producer as written may be fetched from its live
    postimage; every other value must exist in the original domain carrier.
    """

    def __init__(self, inputs: PreparedCodexLinkInputs, connection: sqlite3.Connection) -> None:
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError, current_index_mutation_scope

        scope = current_index_mutation_scope()
        if scope is None or scope.seal is not inputs.reference_seal:
            raise ReferenceSealError("prepared Codex links require their original admitted Index scope")
        scope.require_new_work(connection)
        self.inputs = inputs
        self.connection = connection
        with connection_cursor(
            inputs.owner.require_connection(),
            "CREATE TABLE codex_link_written_cells(table_name TEXT,physical_rowid INTEGER,"
            "column_name TEXT,PRIMARY KEY(table_name,physical_rowid,column_name))",
        ):
            pass

    def record_written(self, table: str, physical_rowid: int, columns: Iterable[str]) -> None:
        allowed = (
            _CODEX_LINK_FIELDS
            if table == "session_links"
            else _CODEX_LINK_SESSION_FIELDS
            if table == "sessions"
            else ()
        )
        for column in columns:
            if column not in allowed:
                raise RuntimeError("Codex link producer cannot record an undeclared written cell")
            with connection_cursor(
                self.inputs.owner.require_connection(),
                "INSERT OR IGNORE INTO codex_link_written_cells VALUES(?,?,?)",
                (table, physical_rowid, column),
            ):
                pass

    def _written(self, table: str, physical_rowid: int, column: str) -> bool:
        with connection_cursor(
            self.inputs.owner.require_connection(),
            "SELECT 1 FROM codex_link_written_cells WHERE table_name=? AND physical_rowid=? AND column_name=?",
            (table, physical_rowid, column),
        ) as cursor:
            return cursor.fetchone() is not None

    def _cell(self, table: str, physical_rowid: int, column: str, original: object) -> object:
        if not self._written(table, physical_rowid, column):
            return original
        # Table and column are finite domain operands, never caller SQL.
        with connection_cursor(
            self.connection,
            f"SELECT {column} FROM {table} WHERE rowid=?",
            (physical_rowid,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Codex producer's written coordinate disappeared")
        return row[0]

    def _link_cell(self, physical_rowid: int, column: str) -> object:
        if self._written("session_links", physical_rowid, column):
            return self._cell("session_links", physical_rowid, column, None)
        return getattr(self.inputs.link_record(physical_rowid), column)

    def _link_rowids(self, predicate: str, parameters: tuple[object, ...], order: str = "") -> list[int]:
        with connection_cursor(
            self.connection,
            "SELECT rowid FROM session_links WHERE " + predicate + order,
            parameters,
        ) as cursor:
            return [int(row[0]) for row in cursor]

    def session_header(self, session_id: str) -> tuple[object, object, object] | None:
        row = self.inputs.session_record(session_id)
        if row is None:
            return None
        return (
            self._cell("sessions", row.physical_rowid, "root_session_id", row.root_session_id),
            self._cell("sessions", row.physical_rowid, "branch_type", row.branch_type),
            self._cell("sessions", row.physical_rowid, "session_kind", row.session_kind),
        )

    def composing_parent_link(self, session_id: str) -> tuple[object, object] | None:
        rowids = self._link_rowids(
            "src_session_id=? AND resolved_dst_session_id IS NOT NULL AND " + topology_status_composes_sql(),
            (session_id,),
            " ORDER BY observed_at_ms IS NULL,observed_at_ms,dst_origin,dst_native_id,link_type LIMIT 1",
        )
        if not rowids:
            return None
        return (self._link_cell(rowids[0], "resolved_dst_session_id"), self._link_cell(rowids[0], "link_type"))

    def first_link_type(self, session_id: str) -> tuple[object] | None:
        rowids = self._link_rowids(
            "src_session_id=?",
            (session_id,),
            " ORDER BY observed_at_ms IS NULL,observed_at_ms,dst_origin,dst_native_id,link_type LIMIT 1",
        )
        return None if not rowids else (self._link_cell(rowids[0], "link_type"),)

    def parent_of(self, session_id: str) -> str | None:
        composing = self.composing_parent_link(session_id)
        if composing is not None:
            return None if composing[0] is None else str(composing[0])
        row = self.inputs.session_record(session_id)
        if row is None:
            return None
        parent = self._cell("sessions", row.physical_rowid, "parent_session_id", row.parent_session_id)
        return None if parent is None else str(parent)

    def existing_method(self, source: str, origin: str, native_id: str, link_type: str) -> object | None:
        rowids = self._link_rowids(
            "src_session_id=? AND dst_origin=? AND dst_native_id=? AND link_type=?",
            (source, origin, native_id, link_type),
        )
        return None if not rowids else self._link_cell(rowids[0], "method")

    def stale_authoritative_destinations(self, source: str, link_type: str, winner: str) -> list[str]:
        rowids = self._link_rowids(
            "src_session_id=? AND link_type=? AND method=? AND dst_native_id!=?",
            (source, link_type, HOOK_AUTHORITATIVE_LINK_METHOD, winner),
        )
        return [str(self._link_cell(rowid, "dst_native_id")) for rowid in rowids]

    def resolution_candidates(self, session_id: str) -> list[tuple[str, str, str, str]]:
        candidates: list[tuple[str, str, str, str]] = []
        with connection_cursor(
            self.connection,
            "SELECT session_links.rowid,claims.rowid " + _LINK_RESOLUTION_CLAIM_FROM_SQL,
            (session_id,),
        ) as cursor:
            selected = [(int(row[0]), int(row[1])) for row in cursor]
        for link_rowid, claim_rowid in selected:
            with connection_cursor(
                self.inputs.owner.require_connection(),
                "SELECT claimant_session_id FROM codex_link_input_claims WHERE physical_rowid=?",
                (claim_rowid,),
            ) as cursor:
                claim = cursor.fetchone()
            if claim is None:
                raise RuntimeError("Codex link resolution selected an unprepared original identity claimant")
            candidates.append(
                (
                    str(self._link_cell(link_rowid, "dst_origin")),
                    str(self._link_cell(link_rowid, "dst_native_id")),
                    str(self._link_cell(link_rowid, "link_type")),
                    str(claim[0]),
                )
            )
        with connection_cursor(
            self.connection,
            "SELECT links.rowid,sessions.rowid " + _LINK_RESOLUTION_FALLBACK_FROM_SQL,
            (session_id,),
        ) as cursor:
            selected = [(int(row[0]), int(row[1])) for row in cursor]
        for link_rowid, session_rowid in selected:
            with connection_cursor(
                self.inputs.owner.require_connection(),
                "SELECT session_id FROM codex_link_input_sessions WHERE physical_rowid=?",
                (session_rowid,),
            ) as cursor:
                target = cursor.fetchone()
            if target is None:
                raise RuntimeError("Codex link resolution selected an unprepared original fallback session")
            candidates.append(
                (
                    str(self._link_cell(link_rowid, "dst_origin")),
                    str(self._link_cell(link_rowid, "dst_native_id")),
                    str(self._link_cell(link_rowid, "link_type")),
                    str(target[0]),
                )
            )
        return candidates

    def resolution_written(self, physical_rowid: int, columns: tuple[str, ...]) -> None:
        self.record_written("session_links", physical_rowid, columns)

    def session_written(self, session_id: str, columns: tuple[str, ...]) -> None:
        row = self.inputs.session_record(session_id)
        if row is None:
            raise RuntimeError("Codex projection wrote a session absent from its original carrier")
        self.record_written("sessions", row.physical_rowid, columns)

    def descendant_root_written(self, physical_rowid: int) -> None:
        with connection_cursor(
            self.inputs.owner.require_connection(),
            "SELECT 1 FROM codex_link_input_sessions WHERE physical_rowid=?",
            (physical_rowid,),
        ) as cursor:
            if cursor.fetchone() is None:
                raise RuntimeError("Codex root propagation reached an unprepared original descendant")
        self.record_written("sessions", physical_rowid, ("root_session_id",))

    def inserted_link(self, physical_rowid: int) -> None:
        self.record_written("session_links", physical_rowid, _CODEX_LINK_FIELDS)

    def superseded_link(self, source: str, link_type: str, native_id: str) -> None:
        for rowid in self._link_rowids(
            "src_session_id=? AND link_type=? AND dst_native_id=?",
            (source, link_type, native_id),
        ):
            self.record_written(
                "session_links",
                rowid,
                (
                    "status",
                    "method",
                    "evidence_json",
                    "resolved_dst_session_id",
                    "observed_at_ms",
                ),
            )

    def existing_session_for_native(self, origin: str, native_id: str) -> str | None:
        with connection_cursor(
            self.inputs.owner.require_connection(),
            "SELECT resolved_session_id FROM codex_link_input_targets WHERE origin=? AND native_id=? AND processed=1",
            (origin, native_id),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Codex link projection requested an unprepared original target")
        return None if row[0] is None else str(row[0])

    def parser_rows(self, session_id: str, origin: str) -> list[tuple[object, ...]]:
        rowids = self._link_rowids(
            "src_session_id=? AND dst_origin=?",
            (session_id, origin),
            " ORDER BY observed_at_ms IS NULL,observed_at_ms,dst_native_id,link_type",
        )
        columns = (
            "dst_native_id",
            "link_type",
            "method",
            "status",
            "branch_point_message_id",
            "branch_point_content_address",
            "inheritance",
            "parent_tool_use_block_id",
            "evidence_json",
        )
        return [tuple(self._link_cell(rowid, column) for column in columns) for rowid in rowids]


def prepare_codex_link_inputs(
    seal: PreparedIndexMutation,
    *,
    owner: NativeSQLCustodyOwner,
    child_native_ids: Iterable[str],
    parent_native_ids: Iterable[str],
    source_read: SessionSourceRead,
) -> PreparedCodexLinkInputs:
    """Retain the canonical reachable original session/link/identity closure.

    The caller supplies every possible graph parent, including superseded
    fallbacks across scopes. All original reads belong to its same pinned
    witness window. The worklists and operands use the existing graph child's
    native scratch, and this function grants no writer or reference authority.
    """
    scratch = owner.require_connection()
    original = seal.observer("index")
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_link_input_session_queue(session_id TEXT PRIMARY KEY, processed INTEGER NOT NULL DEFAULT 0)",
    ):
        pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_link_input_targets(origin TEXT, native_id TEXT, direct_session_id TEXT, resolved_session_id TEXT, processed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(origin,native_id))",
    ):
        pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_link_input_sessions(session_id TEXT PRIMARY KEY,physical_rowid INTEGER UNIQUE,"
        "raw_id TEXT,created_at_ms INTEGER,updated_at_ms INTEGER,parent_session_id TEXT,root_session_id TEXT,"
        "branch_type TEXT,session_kind TEXT,source_path TEXT)",
    ):
        pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_link_input_claims(physical_rowid INTEGER PRIMARY KEY,origin TEXT,"
        "provider_value TEXT,claimant_session_id TEXT)",
    ):
        pass
    with connection_cursor(
        scratch,
        "CREATE TABLE codex_link_input_links(physical_rowid INTEGER PRIMARY KEY,src_session_id TEXT,"
        "dst_origin TEXT,dst_native_id TEXT,link_type TEXT,method TEXT,status TEXT,observed_at_ms INTEGER,"
        "resolved_dst_session_id TEXT,resolved_at_ms INTEGER,branch_point_message_id TEXT,"
        "branch_point_content_address BLOB,inheritance TEXT,parent_tool_use_block_id TEXT,evidence_json TEXT)",
    ):
        pass

    def enqueue_session(session_id: str | None) -> None:
        if session_id is not None:
            with connection_cursor(
                scratch, "INSERT OR IGNORE INTO codex_link_input_session_queue(session_id) VALUES(?)", (session_id,)
            ):
                pass

    def enqueue_target(origin: str, native_id: str) -> None:
        with connection_cursor(
            scratch,
            "INSERT OR IGNORE INTO codex_link_input_targets(origin,native_id) VALUES(?,?)",
            (origin, native_id),
        ):
            pass

    for native_id in chain(child_native_ids, parent_native_ids):
        check_compute_cancelled()
        if native_id and native_id.strip():
            enqueue_target(Origin.CODEX_SESSION.value, native_id.strip())

    while True:
        check_compute_cancelled()
        with connection_cursor(
            scratch,
            "SELECT origin,native_id FROM codex_link_input_targets WHERE processed=0 ORDER BY origin,native_id LIMIT 64",
        ) as cursor:
            targets = cursor.fetchall()
        for origin, native_id in targets:
            parameters = (origin, native_id)
            resolved = _existing_session_id_for_native(original, origin, native_id, seal.before_index_input)
            direct = archive_session_id(origin, native_id)
            with connection_cursor(original, "SELECT rowid FROM sessions WHERE session_id=?", (direct,)) as cursor:
                direct_exists = cursor.fetchone() is not None
            if direct_exists:
                enqueue_session(direct)
            enqueue_session(resolved)
            with connection_cursor(
                original,
                "SELECT rowid FROM session_identity_claims WHERE origin=? "
                "AND identity_namespace='provider-session' AND provider_value=? ORDER BY claimant_session_id",
                parameters,
            ) as identities:
                for (rowid,) in identities:
                    check_compute_cancelled()
                    seal.before_index_input(
                        "session_identity_claims",
                        ("origin", "identity_namespace", "provider_value", "claimant_session_id"),
                        "SELECT rowid FROM session_identity_claims WHERE rowid=?",
                        (rowid,),
                    )
                    with connection_cursor(
                        original,
                        "SELECT origin,provider_value,claimant_session_id FROM session_identity_claims WHERE rowid=?",
                        (rowid,),
                    ) as cursor:
                        row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError("selected Codex identity claimant disappeared inside its original window")
                    with connection_cursor(
                        scratch,
                        "INSERT OR IGNORE INTO codex_link_input_claims VALUES(?,?,?,?)",
                        (rowid, *tuple(row)),
                    ):
                        pass
                    enqueue_session(str(row[2]))
            with connection_cursor(
                scratch,
                "UPDATE codex_link_input_targets SET direct_session_id=?,resolved_session_id=?,processed=1 "
                "WHERE origin=? AND native_id=?",
                (direct if direct_exists else None, resolved, *parameters),
            ):
                pass
        with connection_cursor(
            scratch,
            "SELECT session_id FROM codex_link_input_session_queue WHERE processed=0 ORDER BY session_id LIMIT 64",
        ) as cursor:
            sessions = cursor.fetchall()
        for (session_id,) in sessions:
            check_compute_cancelled()
            with connection_cursor(original, "SELECT rowid FROM sessions WHERE session_id=?", (session_id,)) as cursor:
                selected = cursor.fetchone()
            if selected is None:
                with connection_cursor(
                    scratch, "INSERT INTO codex_link_input_sessions(session_id) VALUES(?)", (session_id,)
                ):
                    pass
            else:
                rowid = int(selected[0])
                seal.before_index_input(
                    "sessions", _CODEX_LINK_SESSION_FIELDS, "SELECT rowid FROM sessions WHERE rowid=?", (rowid,)
                )
                with connection_cursor(
                    original,
                    "SELECT " + ",".join(_CODEX_LINK_SESSION_FIELDS) + " FROM sessions WHERE rowid=?",
                    (rowid,),
                ) as cursor:
                    header = cursor.fetchone()
                if header is None:
                    raise RuntimeError("selected Codex session disappeared inside its original window")
                source_path = raw_source_path(source_read, header[1])
                with connection_cursor(
                    scratch,
                    "INSERT INTO codex_link_input_sessions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (header[0], rowid, *tuple(header[1:]), source_path),
                ):
                    pass
                enqueue_session(None if header[4] is None else str(header[4]))
                enqueue_session(None if header[5] is None else str(header[5]))
                with connection_cursor(
                    original,
                    "SELECT rowid FROM session_links WHERE src_session_id=? "
                    "ORDER BY observed_at_ms IS NULL,observed_at_ms,dst_origin,dst_native_id,link_type",
                    (session_id,),
                ) as identities:
                    for (link_rowid,) in identities:
                        check_compute_cancelled()
                        seal.before_index_input(
                            "session_links",
                            _CODEX_LINK_FIELDS,
                            "SELECT rowid FROM session_links WHERE rowid=?",
                            (link_rowid,),
                        )
                        with connection_cursor(
                            original,
                            "SELECT " + ",".join(_CODEX_LINK_FIELDS) + " FROM session_links WHERE rowid=?",
                            (link_rowid,),
                        ) as cursor:
                            link = cursor.fetchone()
                        if link is None:
                            raise RuntimeError("selected Codex link disappeared inside its original window")
                        with connection_cursor(
                            scratch,
                            "INSERT INTO codex_link_input_links VALUES(" + ",".join("?" for _ in range(15)) + ")",
                            (link_rowid, *tuple(link)),
                        ):
                            pass
                        enqueue_target(str(link[1]), str(link[2]))
                        enqueue_session(None if link[7] is None else str(link[7]))
                with connection_cursor(
                    original, "SELECT rowid FROM sessions WHERE parent_session_id=? ORDER BY rowid", (session_id,)
                ) as descendants:
                    for (child_rowid,) in descendants:
                        check_compute_cancelled()
                        seal.before_index_input(
                            "sessions", ("session_id",), "SELECT rowid FROM sessions WHERE rowid=?", (child_rowid,)
                        )
                        with connection_cursor(
                            original, "SELECT session_id FROM sessions WHERE rowid=?", (child_rowid,)
                        ) as cursor:
                            child = cursor.fetchone()
                        if child is None:
                            raise RuntimeError("selected Codex descendant disappeared inside its original window")
                        enqueue_session(str(child[0]))
            with connection_cursor(
                scratch, "UPDATE codex_link_input_session_queue SET processed=1 WHERE session_id=?", (session_id,)
            ):
                pass
        if not targets and not sessions:
            break
    scratch.commit()
    return PreparedCodexLinkInputs(owner, seal)


_CODEX_LINK_FIELDS = (
    "src_session_id",
    "dst_origin",
    "dst_native_id",
    "link_type",
    "method",
    "status",
    "observed_at_ms",
    "resolved_dst_session_id",
    "resolved_at_ms",
    "branch_point_message_id",
    "branch_point_content_address",
    "inheritance",
    "parent_tool_use_block_id",
    "evidence_json",
)


_CODEX_LINK_SESSION_FIELDS = (
    "session_id",
    "raw_id",
    "created_at_ms",
    "updated_at_ms",
    "parent_session_id",
    "root_session_id",
    "branch_type",
    "session_kind",
)


_LINK_RESOLUTION_CLAIM_FROM_SQL = "FROM session_links\n            JOIN session_identity_claims claims\n              ON claims.origin = session_links.dst_origin\n             AND claims.identity_namespace = 'provider-session'\n             AND claims.provider_value = session_links.dst_native_id\n            WHERE session_links.src_session_id = ?\n              AND session_links.resolved_dst_session_id IS NULL\n              AND session_links.status IS NULL\n              AND NOT EXISTS (\n                    SELECT 1 FROM session_identity_claims conflicts\n                    WHERE conflicts.origin = claims.origin\n                      AND conflicts.identity_namespace = claims.identity_namespace\n                      AND conflicts.provider_value = claims.provider_value\n                      AND conflicts.claimant_session_id != claims.claimant_session_id\n                  )\n            "


_LINK_RESOLUTION_FALLBACK_FROM_SQL = "FROM session_links AS links\n            JOIN sessions\n              ON sessions.session_id = links.dst_origin || ':' || links.dst_native_id\n            WHERE links.src_session_id = ?\n              AND links.resolved_dst_session_id IS NULL\n              AND links.status IS NULL\n              AND NOT EXISTS (\n                    SELECT 1 FROM session_identity_claims claims\n                    WHERE claims.origin = links.dst_origin\n                      AND claims.identity_namespace = 'provider-session'\n                      AND claims.provider_value = links.dst_native_id\n              )\n            "


class SessionLinkMutationRead(Protocol):
    """Authority operands for the canonical hook link mutation bodies."""

    def existing_method(self, source: str, origin: str, native_id: str, link_type: str) -> object | None: ...
    def stale_authoritative_destinations(self, source: str, link_type: str, winner: str) -> list[str]: ...
    def inserted_link(self, physical_rowid: int) -> None: ...
    def superseded_link(self, source: str, link_type: str, native_id: str) -> None: ...


class ConnectionSessionLinkMutationRead:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def inserted_link(self, physical_rowid: int) -> None:
        pass

    def superseded_link(self, source: str, link_type: str, native_id: str) -> None:
        pass

    def existing_method(self, source: str, origin: str, native_id: str, link_type: str) -> object | None:
        with connection_cursor(
            self._connection,
            "SELECT method FROM session_links WHERE src_session_id=? AND dst_origin=? "
            "AND dst_native_id=? AND link_type=?",
            (source, origin, native_id, link_type),
        ) as cursor:
            row = cursor.fetchone()
        return None if row is None else row[0]

    def stale_authoritative_destinations(self, source: str, link_type: str, winner: str) -> list[str]:
        with connection_cursor(
            self._connection,
            "SELECT dst_native_id FROM session_links WHERE src_session_id=? AND link_type=? "
            "AND method=? AND dst_native_id!=?",
            (source, link_type, HOOK_AUTHORITATIVE_LINK_METHOD, winner),
        ) as cursor:
            return [str(row[0]) for row in cursor]


class SessionLinkResolutionRead(Protocol):
    """Exact original/postimage operands of sequential link resolution."""

    def resolution_candidates(self, session_id: str) -> list[tuple[str, str, str, str]]: ...
    def parent_of(self, session_id: str) -> str | None: ...
    def resolution_written(self, physical_rowid: int, columns: tuple[str, ...]) -> None: ...


class ConnectionSessionLinkResolutionRead:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def parent_of(self, session_id: str) -> str | None:
        return _walk_parent_of(self._connection, session_id)

    def resolution_written(self, physical_rowid: int, columns: tuple[str, ...]) -> None:
        # Ordinary reads already observe their connection's current postimage.
        pass

    def resolution_candidates(self, session_id: str) -> list[tuple[str, str, str, str]]:
        with connection_cursor(
            self._connection,
            "SELECT session_links.dst_origin, session_links.dst_native_id, session_links.link_type,\n                   claims.claimant_session_id "
            + _LINK_RESOLUTION_CLAIM_FROM_SQL,
            (session_id,),
        ) as cursor:
            candidates = [(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in cursor]
        # The canonical fallback applies only when no exact identity claim exists.
        with connection_cursor(
            self._connection,
            "SELECT links.dst_origin, links.dst_native_id, links.link_type, sessions.session_id "
            + _LINK_RESOLUTION_FALLBACK_FROM_SQL,
            (session_id,),
        ) as cursor:
            candidates.extend((str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in cursor)
        return candidates


class SessionLineageProjectionRead(Protocol):
    """Original or produced lineage operands for the shared projection body."""

    def session_header(self, session_id: str) -> tuple[object, object, object] | None: ...
    def composing_parent_link(self, session_id: str) -> tuple[object, object] | None: ...
    def first_link_type(self, session_id: str) -> tuple[object] | None: ...
    def session_written(self, session_id: str, columns: tuple[str, ...]) -> None: ...
    def descendant_root_written(self, physical_rowid: int) -> None: ...


class ConnectionSessionLineageProjectionRead:
    """Borrow ordinary Index operands and settle each actual read cursor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def session_written(self, session_id: str, columns: tuple[str, ...]) -> None:
        pass

    def descendant_root_written(self, physical_rowid: int) -> None:
        pass

    def session_header(self, session_id: str) -> tuple[object, object, object] | None:
        with connection_cursor(
            self._connection,
            "SELECT root_session_id,branch_type,session_kind FROM sessions WHERE session_id=?",
            (session_id,),
        ) as cursor:
            row = cursor.fetchone()
        return None if row is None else (row[0], row[1], row[2])

    def composing_parent_link(self, session_id: str) -> tuple[object, object] | None:
        with connection_cursor(
            self._connection,
            f"""
            SELECT resolved_dst_session_id, link_type
            FROM session_links
            WHERE src_session_id = ? AND resolved_dst_session_id IS NOT NULL
              AND {topology_status_composes_sql()}
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = cursor.fetchone()
        return None if row is None else (row[0], row[1])

    def first_link_type(self, session_id: str) -> tuple[object] | None:
        with connection_cursor(
            self._connection,
            "SELECT link_type FROM session_links WHERE src_session_id=? "
            "ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type LIMIT 1",
            (session_id,),
        ) as cursor:
            row = cursor.fetchone()
        return None if row is None else (row[0],)


def _supersede_stale_authoritative_links(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    link_type: str,
    winning_dst_native_id: str,
    observed_at_ms: int,
    read: SessionLinkMutationRead | None = None,
) -> None:
    """Demote any authoritative edge the newest hook claim disagrees with.

    ``_authoritative_parent_claim`` already returns the NEWEST spawn-edge row
    (``ORDER BY observed_at_ms DESC``), so newest-hook-wins is the rule. Without
    this, a revised hook claim naming a different parent lands at a DIFFERENT
    primary key, and the previous authoritative edge can be neither purged (it
    is exempt from the projection purge) nor overwritten (the downgrade guard
    refuses) -- leaving two permanent authoritative edges for one child and
    handing composition an arrival-order choice between them. That is precisely
    the defect this whole mechanism exists to remove, so it must not be
    reintroduced by the mechanism's own durability rules.

    The loser is re-marked, never deleted: both claims stay auditable, and the
    typed state is the same ``AUTHORITY_CONTRADICTED`` an inferred loser gets,
    because the outcome for composition is identical.
    """
    if read is None:
        read = ConnectionSessionLinkMutationRead(conn)
    stale = read.stale_authoritative_destinations(src_session_id, link_type, winning_dst_native_id)
    for stale_native_id in stale:
        with connection_cursor(
            conn,
            """
            UPDATE session_links
               SET status = ?, method = ?, evidence_json = ?, resolved_dst_session_id = NULL,
                   observed_at_ms = ?
             WHERE src_session_id = ? AND link_type = ? AND dst_native_id = ?
            """,
            (
                TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value,
                HOOK_SUPERSEDED_LINK_METHOD,
                _json_dumps(
                    {
                        "superseded_by_hook_parent": winning_dst_native_id,
                        "superseded_hook_parent": stale_native_id,
                    }
                ),
                observed_at_ms,
                src_session_id,
                link_type,
                stale_native_id,
            ),
        ):
            pass
        read.superseded_link(src_session_id, link_type, stale_native_id)


MANUAL_CONTINUATION_VALUE_SCHEMA = "polylogue.manual-continuation.v1"


class ManualContinuationAuthorityError(ReferenceSealError):
    """Durable manual continuation authority cannot be read or projected."""


def _project_manual_continuations(conn: sqlite3.Connection, session_id: str) -> None:
    """Derive operator continuation edges from this scope's declared User tier.

    Prose handoff assertions have no topology semantics. Only the explicit
    value discriminator written by the manual continuation product is read.
    Missing parents stay unresolved until the canonical graph pass sees them.
    """
    scope = current_index_mutation_scope()
    if scope is None:
        raise ManualContinuationAuthorityError("manual continuation projection requires its Index mutation scope")
    scope.require_new_work(conn)
    reader = scope.user_reader()
    if reader is None:
        # A declared standalone Index has no durable assertions to project.
        return
    scope.note_lineage_change(session_id)
    # Edges can predate this session's first row, so test their own ownership.
    # An empty projection has nothing to retire, even when a User reader exists.
    if (
        conn.execute(
            "SELECT 1 FROM session_links WHERE src_session_id=? AND method='manual-continuation' LIMIT 1",
            (session_id,),
        ).fetchone()
        is not None
    ):
        conn.execute("DELETE FROM session_links WHERE src_session_id=? AND method='manual-continuation'", (session_id,))
    try:
        with connection_cursor(
            reader,
            "SELECT assertion_id, value_json, updated_at_ms FROM assertions "
            "WHERE target_ref=? AND kind='handoff' AND COALESCE(status, '')!='deleted' "
            "ORDER BY assertion_id",
            (f"session:{session_id}",),
        ) as rows:
            for assertion_id, value_json, observed_at_ms in rows:
                value = None if value_json is None else json.loads(value_json)
                if not isinstance(value, dict) or value.get("_schema") != MANUAL_CONTINUATION_VALUE_SCHEMA:
                    continue
                parent = value.get("parent_session_id")
                if not isinstance(parent, str) or ":" not in parent or not all(parent.split(":", 1)):
                    raise ManualContinuationAuthorityError(
                        f"manual continuation assertion {assertion_id!r} lacks an exact parent session id"
                    )
                parent_origin, parent_native = parent.split(":", 1)
                # Explicit operator authority retains the manual product's
                # replacement semantics for this exact edge coordinate.
                conn.execute(
                    "INSERT OR REPLACE INTO session_links "
                    "(src_session_id,dst_origin,dst_native_id,link_type,inheritance,method,confidence,evidence_json,observed_at_ms) "
                    "VALUES (?, ?, ?, 'continuation', 'spawned-fresh', 'manual-continuation', 1.0, '[]', ?)",
                    (session_id, parent_origin, parent_native, observed_at_ms),
                )
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise ManualContinuationAuthorityError("cannot read durable manual continuation assertions") from exc


def _upsert_session_link(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    link_type: str,
    branch_point_message_id: str | None,
    inheritance: str | None,
    branch_point_content_address: bytes | None = None,
    status: str | None,
    parent_tool_use_block_id: str | None,
    method: str,
    confidence: float,
    evidence_json: str,
    observed_at_ms: int,
    read: SessionLinkMutationRead | None = None,
) -> int | None:
    """Write one edge without letting inference downgrade hook authority.

    This replaces a bare ``INSERT OR REPLACE``. That statement rewrote every
    column of an existing row, so an ordinary re-parse of a child silently
    reset an ``authoritative-hook-evidence`` edge back to ``parser-parent``
    with ``status = NULL`` -- the "later inference overwrites the
    authoritative result" hole this guard closes. The idiom mirrors
    ``revision_authority_refuses_write``: a lower-authority writer is refused
    rather than allowed to win by last-writer-wins.
    """
    if inheritance is not None:
        require_literal(inheritance, LineageInheritance, name="lineage inheritance")
    if read is None:
        read = ConnectionSessionLinkMutationRead(conn)
    existing_method = read.existing_method(src_session_id, dst_origin, dst_native_id, link_type)
    if str(existing_method or "") in HOOK_DERIVED_LINK_METHODS and method not in HOOK_DERIVED_LINK_METHODS:
        return None
    with connection_cursor(
        conn,
        """
        INSERT OR REPLACE INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type,
            branch_point_message_id, branch_point_content_address, inheritance,
            status, parent_tool_use_block_id, method, confidence, evidence_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            src_session_id,
            dst_origin,
            dst_native_id,
            link_type,
            branch_point_message_id,
            branch_point_content_address,
            inheritance,
            status,
            parent_tool_use_block_id,
            method,
            confidence,
            evidence_json,
            observed_at_ms,
        ),
    ) as cursor:
        rowid = cursor.lastrowid
    if rowid is None:
        raise RuntimeError("session link replacement did not report its physical row identity")
    read.inserted_link(int(rowid))
    return int(rowid)


#: ``evidence_json`` key holding the provider-native id of the parent message
#: a parser asserted as this child's divergence point
#: (``ParsedSession.branch_point_provider_message_id``). It is kept after the
#: id binds so the claim stays auditable and re-binds idempotently against a
#: parent that arrives, or is replaced, later.
ASSERTED_BRANCH_POINT_EVIDENCE_KEY = "asserted_branch_point_native_id"


class AssertedBranchPointAmbiguousError(RuntimeError):
    """A provider branch-point name identifies several composed parent rows."""

    code = "asserted_branch_point_ambiguous"


def _bind_asserted_branch_point(
    conn: sqlite3.Connection,
    parent_session_id: str | None,
    native_id: str | None,
) -> str | None:
    """Return the unique composed parent row named by an asserted branch point.

    ``None`` when the parent or that exact message is not in the archive yet:
    ``session_links.branch_point_message_id`` has no FK, so an unbacked id
    would be a dangling lineage reference rather than a placeholder --
    ``_refill_inbound_asserted_branch_points`` binds it when the parent lands.
    """
    if not parent_session_id or not native_id:
        return None
    # Normalization can move a parent's prefix into its ancestors. The provider
    # still names that original message, including when the child never replays
    # the prefix. Resolve only inside the parent's actual composed segments.
    source_name = source_native_id_json(native_id)
    bound: str | None = None
    plan = _composed_transcript_plan(conn, parent_session_id)
    for segment in plan.segments:
        sql = "SELECT message_id FROM messages WHERE session_id = ? AND source_native_id_json = ?"
        parameters: tuple[object, ...] = (segment.session_id, source_name)
        if segment.upto_position is not None and segment.upto_variant_index is not None:
            sql += " AND (position, variant_index) <= (?, ?)"
            parameters += (segment.upto_position, segment.upto_variant_index)
        # Two admitted occurrences settle ambiguity; unrelated suffix rows do
        # not participate. Content IDs never masquerade as native evidence.
        for (candidate,) in conn.execute(sql + " LIMIT 2", parameters):
            if bound is not None and bound != candidate:
                raise AssertedBranchPointAmbiguousError(
                    f"branch point {native_id!r} names several rows in parent {parent_session_id!r}"
                )
            bound = str(candidate)
    return bound


def _refill_inbound_asserted_branch_points(conn: sqlite3.Connection, parent_session_id: str) -> set[str]:
    """Bind children's asserted branch points now that this parent exists.

    The mirror of ``_refill_inbound_dispatch_block_ids`` for the branch point:
    a child parsed before its parent recorded the claim in ``evidence_json``
    but had no message to point at. Runs after inbound identity resolution so
    edges resolved on this same write are covered.
    """
    rows = conn.execute(
        f"""
        SELECT src_session_id, dst_origin, dst_native_id, link_type,
               evidence_json -> '$.{ASSERTED_BRANCH_POINT_EVIDENCE_KEY}'
          FROM session_links
         WHERE resolved_dst_session_id = ?
           AND branch_point_message_id IS NULL
           AND json_valid(evidence_json)
           AND json_type(evidence_json) = 'object'
           AND COALESCE(json_type(evidence_json, '$.{ASSERTED_BRANCH_POINT_EVIDENCE_KEY}'), 'null') != 'null'
           -- A materialized child owns its prefix: re-binding its assertion
           -- would compose the parent's prefix in front of its own copy.
           AND src_session_id NOT IN (SELECT session_id FROM session_identity_scopes)
        """,
        (parent_session_id,),
    ).fetchall()
    changed: set[str] = set()
    for src_session_id, dst_origin, dst_native_id, link_type, asserted_json in rows:
        # Read in its JSON spelling and decoded here: a stored lone-surrogate
        # escape would make ``json_extract`` materialize text that is not UTF-8.
        asserted_native_id = source_native_id_from_json(asserted_json)
        bound = _bind_asserted_branch_point(conn, parent_session_id, asserted_native_id)
        if bound is None:
            continue
        conn.execute(
            """
            UPDATE session_links
               SET branch_point_message_id = ?,
                   branch_point_content_address = ?,
                   inheritance = 'prefix-sharing'
             WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
            """,
            (
                bound,
                _message_content_address_for_id(conn, bound),
                src_session_id,
                dst_origin,
                dst_native_id,
                link_type,
            ),
        )

        changed.add(str(src_session_id))
    return changed


def _write_session_link(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    branch_point_message_id: str | None = None,
    branch_point_content_address: bytes | None = None,
    inheritance: str | None = None,
    source_read: SessionSourceRead | None = None,
    prior_links: bool = True,
    child_source_path: str | None = None,
    preserve_prefix_loss: bool = False,
) -> None:
    """Write this child's outbound parent edge, honouring hook authority.

    ``source_read`` is the durable ``source.db`` handle carrying the acquired
    hook evidence -- Codex ``codex_thread_spawn_edge`` rows and Claude Code's
    ``agent_id``/``agent_type`` tool-call stamps. It is optional exactly as it
    is on ``revision_authority_refuses_write``: an index-only harness passes
    ``None`` and every behaviour below collapses to the pre-existing
    parser-only path.

    Why this lives on the write path rather than in a post-ingest
    reconciliation pass: ``session_links`` is in the REBUILDABLE index tier
    while hook evidence is in the DURABLE source tier, so every reindex
    reconstructs these rows from scratch. Only a derivation that runs inside
    ``write_parsed_session_to_archive`` -- the single choke point shared by
    live incremental ingest and full raw replay -- survives a rebuild by
    construction. A convergence-stage applier would silently lose the
    authoritative marking on the next reindex.
    """
    origin = origin_from_provider(session.source_name).value
    observed_at_ms = (
        to_epoch_ms(session.updated_at, numeric_unit="seconds")
        or to_epoch_ms(session.created_at, numeric_unit="seconds")
        or 0
    )
    # Match exact stored provider identities and parser-emitted aliases after
    # the same normalization used for session native ids.
    parent_native_id = _sqlite_text((session.parent_session_provider_id or "").strip()) or None
    if origin == Origin.HERMES_SESSION.value and parent_native_id is not None:
        parent_native_id = _resolved_hermes_parent_native_id(conn, origin, parent_native_id)
        session = session.model_copy(update={"parent_session_provider_id": parent_native_id})
        if parent_native_id is None:
            return
    if prior_links:
        # A session saved for the first time has no earlier claims to retire.
        _retire_stale_parser_assertions(conn, session_id, parent_native_id)
    hook_claim = _authoritative_parent_claim(
        conn,
        source_read,
        origin=origin,
        child_session_id=session_id,
        child_native_id=(session.provider_session_id or "").strip(),
        child_provider_values=_child_provider_values(session),
        parent_candidate=parent_native_id,
        child_source_path=child_source_path,
    )
    hook_parent = hook_claim.parent_native_id if hook_claim is not None else None
    hook_evidence: Mapping[str, object] = hook_claim.evidence if hook_claim is not None else {}

    if not session.parent_session_provider_id:
        # Hook evidence can know a parent transcript inference never found.
        # Without this the authoritative edge would simply not exist.
        if hook_parent is not None:
            _supersede_stale_authoritative_links(
                conn,
                src_session_id=session_id,
                link_type=LinkType.SUBAGENT.value,
                winning_dst_native_id=hook_parent,
                observed_at_ms=observed_at_ms,
            )
            _upsert_session_link(
                conn,
                src_session_id=session_id,
                dst_origin=origin,
                dst_native_id=hook_parent,
                link_type=LinkType.SUBAGENT.value,
                branch_point_message_id=branch_point_message_id,
                branch_point_content_address=branch_point_content_address,
                inheritance=inheritance,
                status=None,
                parent_tool_use_block_id=None,
                method=HOOK_AUTHORITATIVE_LINK_METHOD,
                confidence=1.0,
                evidence_json=_json_dumps({**hook_evidence, "parser_parent": None}),
                observed_at_ms=observed_at_ms,
            )
        return
    dst_native_id = parent_native_id
    if not dst_native_id:
        return
    link_type = branch_type_to_edge_type(session.branch_type, default=TopologyEdgeType.BRANCH).value
    dispatch = _resolve_parent_tool_use_block(conn, session, source_read=source_read)
    parent_tool_use_block_id = dispatch.block_id
    method = dispatch.method or "parser-parent"
    evidence: dict[str, object] = {"parent_session_provider_id": session.parent_session_provider_id}
    if preserve_prefix_loss:
        previous = conn.execute(
            "SELECT evidence_json FROM session_links WHERE src_session_id = ? "
            "AND dst_origin = ? AND dst_native_id = ? AND link_type = ?",
            (session_id, origin, dst_native_id, link_type),
        ).fetchone()
        if previous is not None:
            prior_evidence = json.loads(str(previous[0]))
            if isinstance(prior_evidence, dict) and INVALIDATED_PREFIX_EVIDENCE_KEY in prior_evidence:
                evidence[INVALIDATED_PREFIX_EVIDENCE_KEY] = prior_evidence[INVALIDATED_PREFIX_EVIDENCE_KEY]
    if hook_claim is not None and hook_parent is None:
        # Hook evidence spoke and contradicted itself. Retain it on the row
        # rather than degrading to silence, so the refusal is inspectable.
        evidence.update(hook_evidence)
    if origin == Origin.AISTUDIO_DRIVE.value and branch_point_message_id is None:
        evidence["branch_point_resolution"] = "unresolved-source-no-local-message-id"
    # A parser-asserted branch point is the only branch point available when
    # the child does not physically replay the parent's prefix, so prefix
    # alignment produced none. Alignment wins where both exist: it is measured
    # against stored content, the assertion is a provider claim.
    asserted_branch_point = session.branch_point_provider_message_id or ""
    if asserted_branch_point:
        evidence[ASSERTED_BRANCH_POINT_EVIDENCE_KEY] = json.loads(
            cast(str, source_native_id_json(asserted_branch_point))
        )
        if branch_point_message_id is None:
            branch_point_message_id = _bind_asserted_branch_point(
                conn,
                _existing_parent_session_id(conn, session, origin),
                asserted_branch_point,
            )
            if branch_point_message_id is not None:
                # The staleness witness records which parent content this edge
                # was bound against. That is as true of an assertion as of a
                # measured alignment: the composed read splices the parent's
                # prefix in either case, so an edge with no witness is one whose
                # prefix can silently drift underneath it.
                branch_point_content_address = _message_content_address_for_id(conn, branch_point_message_id)
        # ``fork-context-ref`` children do not replay the parent's prefix, but
        # their effective context is still the parent's transcript through the
        # provider-asserted branch point. Mark a successfully bound assertion
        # as prefix-sharing so every public read composes that context back in.
        if branch_point_message_id is not None:
            inheritance = "prefix-sharing"
    identity_reason = _session_target_resolution_reason(conn, origin, dst_native_id)
    if identity_reason is not None:
        evidence["resolution_reason"] = identity_reason
    if dispatch.reason is not None:
        evidence["dispatch_reason"] = dispatch.reason
    status: str | None = None

    # A conflict is scoped to (child, link_type), NOT to the primary key.
    # Because the PK carries dst_native_id, two contradictory parents would
    # otherwise land as two coexisting, independently resolvable rows with
    # nothing recording that they compete -- and _refresh_session_projection
    # would then pick one by observed_at_ms order.
    agreeing = hook_parent is not None and hook_parent == dst_native_id
    contradicted = hook_parent is not None and hook_parent != dst_native_id
    if agreeing:
        method = HOOK_AUTHORITATIVE_LINK_METHOD
        evidence.update(hook_evidence)
    elif contradicted:
        method = HOOK_CONTRADICTED_LINK_METHOD
        status = TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value
        evidence.update(hook_evidence)
        evidence["contradiction"] = "authoritative hook evidence names a different parent"
        evidence["resolution_reason"] = "identity-contradiction"

    _upsert_session_link(
        conn,
        src_session_id=session_id,
        dst_origin=origin,
        dst_native_id=dst_native_id,
        link_type=link_type,
        branch_point_message_id=branch_point_message_id,
        branch_point_content_address=branch_point_content_address,
        inheritance=inheritance,
        status=status,
        parent_tool_use_block_id=parent_tool_use_block_id,
        method=method,
        confidence=1.0,
        evidence_json=_json_dumps(evidence),
        observed_at_ms=observed_at_ms,
    )

    if agreeing and hook_parent is not None:
        _supersede_stale_authoritative_links(
            conn,
            src_session_id=session_id,
            link_type=link_type,
            winning_dst_native_id=dst_native_id,
            observed_at_ms=observed_at_ms,
        )

    if contradicted and hook_parent is not None:
        _supersede_stale_authoritative_links(
            conn,
            src_session_id=session_id,
            link_type=link_type,
            winning_dst_native_id=hook_parent,
            observed_at_ms=observed_at_ms,
        )
        # The dispatch block above was resolved against the parser's parent,
        # which the hook just contradicted. A block the hook parent's own edge
        # already names is a call in that parent and stays bound.
        hook_edge_block = conn.execute(
            """
            SELECT parent_tool_use_block_id FROM session_links
            WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
            """,
            (session_id, origin, hook_parent, link_type),
        ).fetchone()
        _upsert_session_link(
            conn,
            src_session_id=session_id,
            dst_origin=origin,
            dst_native_id=hook_parent,
            link_type=link_type,
            branch_point_message_id=branch_point_message_id,
            branch_point_content_address=branch_point_content_address,
            inheritance=inheritance,
            status=None,
            parent_tool_use_block_id=(
                hook_edge_block[0]
                if hook_edge_block is not None and hook_edge_block[0] is not None
                else parent_tool_use_block_id
            ),
            method=HOOK_AUTHORITATIVE_LINK_METHOD,
            confidence=1.0,
            evidence_json=_json_dumps({**hook_evidence, "superseded_parser_parent": dst_native_id}),
            observed_at_ms=observed_at_ms,
        )


def _retire_stale_parser_assertions(conn: sqlite3.Connection, session_id: str, parser_parent: str | None) -> None:
    """Drop parser claims of a parent the child's current parse no longer asserts.

    A full replace keeps hook-derived edges, and a contradicted edge is one of
    them, so without this a parser revision A -> B under a contradicting hook
    leaves the retired A claim beside the current B claim.
    ``rederive_codex_spawn_parent_links`` recovers the parser's parent from
    these rows, so a surviving A would come back, with its link type,
    inheritance and branch point, the next time the hook parent moves. A
    contradicted edge carries nothing but the parser's claim, so it goes; an
    authoritative edge stays as hook evidence and loses only the claim.
    """
    conn.execute(
        "DELETE FROM session_links WHERE src_session_id = ? AND method = ? AND dst_native_id IS NOT ?",
        (session_id, HOOK_CONTRADICTED_LINK_METHOD, parser_parent),
    )
    conn.execute(
        """
        UPDATE session_links
           SET evidence_json = json_remove(evidence_json, '$.parent_session_provider_id')
         WHERE src_session_id = ? AND method = ? AND dst_native_id IS NOT ?
           AND json_extract(evidence_json, '$.parent_session_provider_id') IS NOT NULL
        """,
        (session_id, HOOK_AUTHORITATIVE_LINK_METHOD, parser_parent),
    )


def _link_evidence(raw: object) -> dict[str, object]:
    try:
        evidence = json.loads(str(raw)) if raw is not None else {}
    except json.JSONDecodeError:
        return {}
    return evidence if isinstance(evidence, dict) else {}


#: Evidence keys a hook decision adds to a parser edge; stripped before the
#: edge is re-decided against a revised spawn-edge projection.
_HOOK_DECISION_EVIDENCE_KEYS = ("codex_thread_spawn_edge_parent", "contradiction")


def rederive_codex_spawn_parent_links(
    conn: sqlite3.Connection,
    child_native_ids: Iterable[str],
    *,
    source_read: SessionSourceRead | None,
    prepared: PreparedCodexSpawnParentLinks | None = None,
    graph_read: ThreadParentRowRead | None = None,
) -> list[str]:
    """Re-decide archived Codex children's parent edges from the current projection.

    ``_write_session_link`` consults the spawn-edge projection only when the
    child is saved. A child saved before the state export that names its
    parent -- raw replay order, or a live child whose spawn edge the runtime
    records after the transcript -- keeps the inferred parent or none, and no
    later save of the unchanged transcript revisits it. The projection writer
    calls this for every child whose projected parent changed, so topology
    converges to the same edges whichever of the two arrives first.

    The parser's claim is recovered from the child's stored edge (its
    ``parent_session_provider_id`` evidence); the hook decision is then the
    same one ``_write_session_link`` makes. A child the projection is silent
    about is left as its save wrote it, exactly as a save would. Returns the
    session ids whose edges were rewritten. ``source_read`` supplies each
    child's retained rollout path, so its parent is read in its own install
    root exactly as its save read it.

    Each rewritten child's parent pointer is set as soon as its edges resolve,
    because the next child's cycle check reads it; each rewritten child's
    projection is then refreshed once, and a moved root reaches its
    descendants through the same projection step.
    """
    origin = Origin.CODEX_SESSION.value
    rewritten: list[str] = []
    for child_native_id in sorted({value.strip() for value in child_native_ids if value and value.strip()}):
        child_session_id = (
            _existing_session_id_for_native(conn, origin, child_native_id)
            if prepared is None
            else prepared.existing_session_for_native(origin, child_native_id)
        )
        if child_session_id is None:
            continue
        prepared_observed_at_ms: int | None = None
        if prepared is None:
            with connection_cursor(
                conn, "SELECT raw_id FROM sessions WHERE session_id=?", (child_session_id,)
            ) as cursor:
                stored_raw = cursor.fetchone()
            child_source_path = raw_source_path(source_read, stored_raw[0] if stored_raw is not None else None)
        else:
            stored = prepared.inputs.session_record(child_session_id)
            if stored is None:
                raise RuntimeError("prepared Codex target has no original child session")
            child_source_path = stored.source_path
            prepared_observed_at_ms = (
                stored.updated_at_ms if stored.updated_at_ms is not None else stored.created_at_ms or 0
            )
        hook_claim = _codex_spawn_edge_parent_claim(
            conn,
            None,
            child_native_id=child_native_id,
            child_source_path=child_source_path,
            row_read=graph_read,
        )
        if hook_claim is None or hook_claim.parent_native_id is None:
            continue
        hook_parent = hook_claim.parent_native_id
        if prepared is None:
            with connection_cursor(
                conn,
                """
            SELECT dst_native_id, link_type, method, status, branch_point_message_id,
                   branch_point_content_address, inheritance, parent_tool_use_block_id, evidence_json
            FROM session_links
            WHERE src_session_id = ? AND dst_origin = ?
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_native_id, link_type
            """,
                (child_session_id, origin),
            ) as cursor:
                rows = cursor.fetchall()
        else:
            rows = prepared.parser_rows(child_session_id, origin)
        authoritative = [str(row[0]) for row in rows if row[2] == HOOK_AUTHORITATIVE_LINK_METHOD and row[3] is None]
        if authoritative == [hook_parent]:
            continue
        parser_row = next(
            (
                row
                for row in rows
                if row[2] not in HOOK_DERIVED_LINK_METHODS
                or row[2] == HOOK_CONTRADICTED_LINK_METHOD
                or "parent_session_provider_id" in _link_evidence(row[8])
            ),
            None,
        )
        if prepared is None:
            with connection_cursor(
                conn,
                "SELECT COALESCE(updated_at_ms, created_at_ms, 0) FROM sessions WHERE session_id=?",
                (child_session_id,),
            ) as cursor:
                timing = cursor.fetchone()
            observed_at_ms = int(timing[0]) if timing is not None and timing[0] is not None else 0
        else:
            if prepared_observed_at_ms is None:
                raise RuntimeError("prepared Codex target lacks its original timestamp")
            observed_at_ms = prepared_observed_at_ms
        if parser_row is None:
            _supersede_stale_authoritative_links(
                conn,
                src_session_id=child_session_id,
                link_type=LinkType.SUBAGENT.value,
                winning_dst_native_id=hook_parent,
                observed_at_ms=observed_at_ms,
                read=prepared,
            )
            _upsert_session_link(
                conn,
                src_session_id=child_session_id,
                dst_origin=origin,
                dst_native_id=hook_parent,
                link_type=LinkType.SUBAGENT.value,
                branch_point_message_id=None,
                inheritance=None,
                status=None,
                parent_tool_use_block_id=None,
                method=HOOK_AUTHORITATIVE_LINK_METHOD,
                confidence=1.0,
                evidence_json=_json_dumps({**hook_claim.evidence, "parser_parent": None}),
                observed_at_ms=observed_at_ms,
                read=prepared,
            )
        else:
            (
                parser_parent,
                link_type,
                _method,
                _status,
                branch_point_message_id,
                branch_point_content_address,
                inheritance,
                parent_tool_use_block_id,
                raw_evidence,
            ) = parser_row
            evidence = {
                key: value
                for key, value in _link_evidence(raw_evidence).items()
                if key not in _HOOK_DECISION_EVIDENCE_KEYS
                and not key.startswith("superseded_")
                and not (key == "resolution_reason" and value == "identity-contradiction")
            }
            evidence.update(hook_claim.evidence)
            agreeing = parser_parent == hook_parent
            if not agreeing:
                evidence["contradiction"] = "authoritative hook evidence names a different parent"
                evidence["resolution_reason"] = "identity-contradiction"
            _supersede_stale_authoritative_links(
                conn,
                src_session_id=child_session_id,
                link_type=link_type,
                winning_dst_native_id=hook_parent,
                observed_at_ms=observed_at_ms,
                read=prepared,
            )
            _upsert_session_link(
                conn,
                src_session_id=child_session_id,
                dst_origin=origin,
                dst_native_id=parser_parent,
                link_type=link_type,
                branch_point_message_id=branch_point_message_id,
                branch_point_content_address=branch_point_content_address,
                inheritance=inheritance,
                status=None if agreeing else TopologyEdgeStatus.AUTHORITY_CONTRADICTED.value,
                parent_tool_use_block_id=parent_tool_use_block_id,
                method=HOOK_AUTHORITATIVE_LINK_METHOD if agreeing else HOOK_CONTRADICTED_LINK_METHOD,
                confidence=1.0,
                evidence_json=_json_dumps(evidence),
                observed_at_ms=observed_at_ms,
                read=prepared,
            )
            if not agreeing:
                _upsert_session_link(
                    conn,
                    src_session_id=child_session_id,
                    dst_origin=origin,
                    dst_native_id=hook_parent,
                    link_type=link_type,
                    branch_point_message_id=branch_point_message_id,
                    branch_point_content_address=branch_point_content_address,
                    inheritance=inheritance,
                    status=None,
                    parent_tool_use_block_id=parent_tool_use_block_id,
                    method=HOOK_AUTHORITATIVE_LINK_METHOD,
                    confidence=1.0,
                    evidence_json=_json_dumps({**hook_claim.evidence, "superseded_parser_parent": parser_parent}),
                    observed_at_ms=observed_at_ms,
                    read=prepared,
                )
        _resolve_outbound_session_links(conn, child_session_id, origin, read=prepared)
        parent_link = (
            _composing_parent_link(conn, child_session_id)
            if prepared is None
            else prepared.composing_parent_link(child_session_id)
        )
        with connection_cursor(
            conn,
            "UPDATE sessions SET parent_session_id = ? WHERE session_id = ?",
            (str(parent_link[0]) if parent_link is not None else None, child_session_id),
        ):
            pass
        if prepared is not None:
            prepared.session_written(child_session_id, ("parent_session_id",))
        rewritten.append(child_session_id)
    # One seen set means each rewritten child, and each ancestor the refresh
    # climbs to, is projected once; a child whose root moves carries its
    # descendants along (``_propagate_root_to_descendants``).
    seen: set[str] = set()
    for session_id in rewritten:
        _refresh_session_projection(conn, session_id, seen=seen, read=prepared)
    return rewritten


def _session_target_resolution_reason(conn: sqlite3.Connection, origin: str, provider_value: str) -> str | None:
    """Return the typed reason an exact session target is not resolvable."""
    claim_rows = conn.execute(
        """SELECT DISTINCT claimant_session_id
           FROM session_identity_claims
           WHERE origin = ? AND identity_namespace = 'provider-session'
             AND provider_value = ?""",
        (origin, provider_value),
    ).fetchall()
    if len(claim_rows) > 1:
        return "identity-contradiction"
    if claim_rows:
        return None
    canonical_id = archive_session_id(origin, provider_value)
    if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (canonical_id,)).fetchone() is not None:
        return None
    return "target-not-yet-observed"


def _resolve_parent_tool_use_block(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    source_read: SessionSourceRead | None = None,
) -> _DispatchResolution:
    """Resolve parent-side dispatch evidence to the exact TOOL_USE block.

    A supplied parent identity scopes the lookup to that exact resolved
    session. A parent not yet in the archive is pending, not refused.
    """
    parent_id = getattr(session, "parent_session_provider_id", None)
    if not parent_id:
        return _DISPATCH_PENDING
    origin = origin_from_provider(session.source_name).value
    parent_session_id = _existing_parent_session_id(conn, session, origin)
    if parent_session_id is None:
        return _DISPATCH_PENDING
    child_session_id = archive_session_id(origin, session.provider_session_id.strip())
    return _resolve_parent_dispatch_block(conn, parent_session_id, child_session_id, source_read=source_read)


def _branch_type_from_link_type(link_type: object) -> str | None:
    try:
        return BranchType(str(link_type)).value
    except ValueError:
        return None


# polylogue-4ts.10: cycle detection + quarantine, ported from the dead
# async engine at storage/sqlite/queries/session_links.py (zero production
# callers -- test-only) into the sole live writer of session_links rows.
# Before this, both live resolution entry points (_resolve_outbound_session_links
# below, and the inbound-parent loop in _resolve_session_graph) resolved
# every matching edge unconditionally; a real cycle in the parent chain was
# only ever caught by _refresh_session_projection's seen-set short-circuit
# and _composed_db_signatures' visited-set truncation, which silently pick
# an arbitrary root/branch point rather than persisting evidence of the
# rejected edge -- session_links.status stayed NULL/empty on every row.


@dataclass(frozen=True, slots=True)
class _CycleWalkResult:
    outcome: Literal["acyclic", "cycle", "budget_exhausted"]
    path: tuple[str, ...]


def _walk_parent_of(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> str | None:
    # Select the same composing edge before copying its potentially large ID.
    # The session projection supplies the original COALESCE fallback only
    # when this canonical edge predicate has no match.
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            return _walk_parent_of(conn, session_id, before_input)
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass
    with connection_cursor(
        conn,
        "SELECT rowid FROM session_links AS links WHERE links.src_session_id=? "
        "AND links.resolved_dst_session_id IS NOT NULL "
        f"AND {topology_status_composes_sql('links.status')} "
        "ORDER BY links.observed_at_ms IS NULL,links.observed_at_ms,"
        "links.dst_origin,links.dst_native_id,links.link_type LIMIT 1",
        (session_id,),
    ) as cursor:
        selected = cursor.fetchone()
    if selected is not None:
        if before_input is not None:
            before_input(
                "session_links",
                ("resolved_dst_session_id",),
                "SELECT rowid FROM session_links WHERE rowid=?",
                (selected[0],),
            )
        with connection_cursor(
            conn, "SELECT resolved_dst_session_id FROM session_links WHERE rowid=?", (selected[0],)
        ) as cursor:
            row = cursor.fetchone()
    else:
        if before_input is not None:
            before_input(
                "sessions", ("parent_session_id",), "SELECT rowid FROM sessions WHERE session_id=?", (session_id,)
            )
        with connection_cursor(
            conn, "SELECT parent_session_id FROM sessions WHERE session_id=?", (session_id,)
        ) as cursor:
            row = cursor.fetchone()
    return None if row is None or row[0] is None else str(row[0])


def _would_create_cycle(
    conn: sqlite3.Connection,
    *,
    child_id: str,
    proposed_parent_id: str,
    before_input: BeforeIndexInput | None = None,
    read: SessionLinkResolutionRead | None = None,
) -> _CycleWalkResult:
    """Classify the proposed edge without conflating exhaustion with a cycle.

    Walks the resolved parent chain upward from ``proposed_parent_id`` (see
    ``_walk_parent_of`` for which edge that is). The walk is full-chain, not
    single-hop, and a visited set alone terminates it, so a valid lineage of
    any depth is admitted. A loop that does not contain ``child_id`` cannot be
    produced through the two guarded resolution routes; if one is ever
    hand-written into the tier the walk stops there, which is indeterminate
    (``budget_exhausted``, the steps walked as its budget) and stays
    quarantined, but is not evidence that the proposed edge closes a cycle.
    """
    if proposed_parent_id == child_id:
        return _CycleWalkResult("cycle", (child_id, child_id))
    path: list[str] = [child_id, proposed_parent_id]
    visited = {proposed_parent_id}
    current = proposed_parent_id
    while True:
        next_parent = _walk_parent_of(conn, current, before_input) if read is None else read.parent_of(current)
        if next_parent is None:
            return _CycleWalkResult("acyclic", tuple(path))
        if next_parent == child_id:
            path.append(child_id)
            return _CycleWalkResult("cycle", tuple(path))
        if next_parent in visited:
            return _CycleWalkResult("budget_exhausted", tuple(path))
        path.append(next_parent)
        visited.add(next_parent)
        current = next_parent


def _quarantine_session_link(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    link_type: str,
    cycle_walk: _CycleWalkResult,
    observed_at_ms: int,
) -> tuple[int, ...]:
    """Mark one unsafe edge quarantined with accurately typed evidence."""
    if cycle_walk.outcome == "cycle":
        evidence_payload: dict[str, JSONValue] = {
            "reason": "cycle_rejected",
            "cycle_path": list(cycle_walk.path),
            "detected_at_ms": observed_at_ms,
        }
    elif cycle_walk.outcome == "budget_exhausted":
        evidence_payload = {
            "reason": "cycle_walk_budget_exhausted",
            "walk_path": list(cycle_walk.path),
            # The steps walked before the loop stopped it.
            "walk_budget": len(cycle_walk.path) - 2,
            "detected_at_ms": observed_at_ms,
        }
    else:
        raise ValueError("acyclic session link cannot be quarantined as a cycle risk")
    evidence_payload["resolution_reason"] = "cycle-quarantine"
    evidence = _json_dumps(evidence_payload)
    with connection_cursor(
        conn,
        """
        UPDATE session_links
           SET status = ?,
               evidence_json = ?,
               resolved_at_ms = ?
         WHERE src_session_id = ?
           AND dst_origin = ?
           AND dst_native_id = ?
           AND link_type = ?
        RETURNING rowid
        """,
        (
            TopologyEdgeStatus.QUARANTINED.value,
            evidence,
            observed_at_ms,
            src_session_id,
            dst_origin,
            dst_native_id,
            link_type,
        ),
    ) as cursor:
        return tuple(int(row[0]) for row in cursor)


def _resolve_session_graph(
    conn: sqlite3.Connection,
    session_id: str,
    _native_id: str,
    origin: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    add_timing: Callable[[str, float], None] | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    invalidated_session_ids: set[str] | None = None,
    source_read: SessionSourceRead | None = None,
) -> set[str]:
    """Resolve this session's lineage edges and re-anchor what its write moved.

    A branch point this write relocated onto identical content elsewhere in the
    lineage is re-anchored here. Whatever cannot be re-anchored is kept intact
    by :func:`_settle_inherited_prefixes`, which the caller runs next
    (polylogue-gy2yu).

    Returns the sessions whose rows or edges this resolution may have changed,
    ``session_id`` included, so the caller can refresh the derived relations
    the session-write guard kept their triggers from refreshing.
    """

    def record_substage(name: str, started_at: float) -> None:
        if add_timing is not None:
            add_timing(f"index.graph_resolve.{name}", started_at)

    t0 = time.perf_counter()
    conn.execute(
        """
        UPDATE sessions
        SET root_session_id = session_id
        WHERE session_id = ? AND root_session_id IS NULL
        """,
        (session_id,),
    )
    record_substage("root_init", t0)
    t0 = time.perf_counter()
    _resolve_outbound_session_links(conn, session_id, origin)
    record_substage("outbound_links", t0)
    t0 = time.perf_counter()
    _refill_inbound_dispatch_block_ids(conn, session_id, source_read=source_read)
    record_substage("inbound_dispatch_blocks", t0)
    t0 = time.perf_counter()
    has_outbound_link = (
        conn.execute("SELECT 1 FROM session_links WHERE src_session_id = ? LIMIT 1", (session_id,)).fetchone()
        is not None
    )
    inbound_rows = conn.execute(
        """
        SELECT links.src_session_id, links.link_type, links.dst_native_id
        FROM session_links links
        JOIN session_identity_claims claims
          ON claims.origin = links.dst_origin
         AND claims.identity_namespace = 'provider-session'
         AND claims.provider_value = links.dst_native_id
         AND claims.claimant_session_id = ?
        WHERE NOT EXISTS (
                SELECT 1 FROM session_identity_claims conflicts
                WHERE conflicts.origin = claims.origin
                  AND conflicts.identity_namespace = claims.identity_namespace
                  AND conflicts.provider_value = claims.provider_value
                  AND conflicts.claimant_session_id != claims.claimant_session_id
              )
          AND links.resolved_dst_session_id IS NULL
          AND links.status IS NULL
          AND links.dst_origin = ?
        """,
        (session_id, origin),
    ).fetchall()
    record_substage("inbound_lookup", t0)
    t0 = time.perf_counter()
    # polylogue-gy2yu: a replaced parent whose branch-point message moved to an
    # ancestor has no outbound link, no *unresolved* inbound edge and a current
    # root projection, so without this the write takes the fast path and never
    # re-anchors the child onto the relocated row. The lookup is an indexed
    # range probe over ``idx_session_links_branch_point``, not a scan.
    anchored_dangling_ids = branch_points_anchored_in_session(conn, session_id)
    asserted_child_ids = _refill_inbound_asserted_branch_points(conn, session_id)
    record_substage("anchored_branch_points", t0)
    t0 = time.perf_counter()
    if (
        not has_outbound_link
        and not inbound_rows
        and not invalidated_session_ids
        and not anchored_dangling_ids
        and not asserted_child_ids
        and _root_projection_current(conn, session_id)
    ):
        record_substage("root_current_check", t0)
        return {session_id}
    record_substage("root_current_check", t0)
    composed_cache: dict[str, list[tuple[str, str]]] = {}
    t0 = time.perf_counter()
    resolved_child_ids: list[str] = []
    reextract_invalidated_ids: set[str] = set()
    for row in inbound_rows:
        child_id, link_type, dst_native_id = str(row[0]), str(row[1]), str(row[2])
        # polylogue-4ts.10: session_id is about to become child_id's parent --
        # refuse (quarantine, with evidence) rather than silently resolve if
        # that would close a cycle or cannot be decided within the walk budget.
        cycle_walk = _would_create_cycle(conn, child_id=child_id, proposed_parent_id=session_id)
        if cycle_walk.outcome != "acyclic":
            _quarantine_session_link(
                conn,
                src_session_id=child_id,
                dst_origin=origin,
                dst_native_id=dst_native_id,
                link_type=link_type,
                cycle_walk=cycle_walk,
                observed_at_ms=int(time.time() * 1000),
            )
            continue
        dispatch = _resolve_parent_dispatch_block(conn, session_id, child_id, source_read=source_read)
        conn.execute(
            f"""
            UPDATE session_links
            SET resolved_dst_session_id = ?,
                resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms),
                parent_tool_use_block_id = COALESCE(parent_tool_use_block_id, ?),
                evidence_json = json_remove({_DISPATCH_REASON_EVIDENCE_SQL}, '$.resolution_reason'),
                method = CASE
                    WHEN parent_tool_use_block_id IS NULL AND ? IS NOT NULL THEN 'parent-tool-use-id'
                    ELSE method
                END
            WHERE src_session_id = ?
              AND dst_native_id = ?
              AND link_type = ?
              AND resolved_dst_session_id IS NULL
              AND status IS NULL
            """,
            (
                session_id,
                dispatch.block_id,
                dispatch.reason,
                dispatch.reason,
                dispatch.block_id,
                child_id,
                dst_native_id,
                link_type,
            ),
        )
        _canonicalize_session_link_evidence(
            conn,
            src_session_id=child_id,
            dst_origin=origin,
            dst_native_id=dst_native_id,
            link_type=link_type,
        )
        resolved_child_ids.append(child_id)
        # Deferred tail extraction (#2467): a child ingested before its parent was
        # stored whole (the inherited prefix could not be aligned yet). Now that
        # the parent exists, normalize the child the same way the parent-known
        # write path does — drop the inherited prefix rows and record the edge.
        reextract_invalidated_ids |= _reextract_prefix_tail_db(
            conn,
            child_id,
            session_id,
            cache=cache,
            composed_cache=composed_cache,
            add_timing=add_timing,
            bulk_fts=bulk_fts,
            bulk_build=bulk_build,
        )
    record_substage("reextract_prefix_tails", t0)

    t0 = time.perf_counter()
    asserted_child_ids |= _refill_inbound_asserted_branch_points(conn, session_id)
    record_substage("inbound_asserted_branch_points", t0)

    # polylogue-7xrv5: ``reextract_invalidated_ids`` carries the sessions whose
    # branch points were invalidated by the re-extraction above. They are the
    # *source* of the stale edge, so neither ``session_id`` nor
    # ``resolved_child_ids`` names them, and ``invalidated_session_ids`` is
    # populated only by identity-claim invalidation. Without them the in-write
    # repair skips exactly the generation it exists to fix and the archive's
    # composed content becomes ingest-order dependent.
    impacted_session_ids = {
        session_id,
        *resolved_child_ids,
        *asserted_child_ids,
        *reextract_invalidated_ids,
        *anchored_dangling_ids,
        *(invalidated_session_ids or set()),
    }
    t0 = time.perf_counter()
    _repair_stale_prefix_branch_points_db(conn, impacted_session_ids, cache=cache, composed_cache=composed_cache)
    record_substage("repair_stale_branch_points", t0)
    t0 = time.perf_counter()
    projection_seen: set[str] = set()
    for impacted_session_id in impacted_session_ids:
        _refresh_session_projection(conn, impacted_session_id, seen=projection_seen)
    record_substage("projection_refresh", t0)
    return impacted_session_ids


def _refill_inbound_dispatch_block_ids(
    conn: sqlite3.Connection,
    parent_session_id: str,
    *,
    source_read: SessionSourceRead | None = None,
) -> None:
    """Rebind resolved children to this parent's dispatch blocks.

    Writing a parent replaces its messages and blocks, and
    ``session_links.parent_tool_use_block_id`` is ``ON DELETE SET NULL``, so
    every inbound child edge loses the join key while the replacement
    reinserts the same deterministic block ids. Identity resolution revisits
    only unresolved edges, so an already-resolved child is repaired here or
    not at all. This is also where a child written before its parent carried
    dispatch evidence converges: parent-first and child-first ingest reach
    the same edge. A refusal is re-typed each time so the recorded reason
    reflects the parent as it stands now.
    """
    rows = conn.execute(
        """SELECT src_session_id, dst_origin, dst_native_id, link_type
           FROM session_links
           WHERE resolved_dst_session_id = ?
             AND parent_tool_use_block_id IS NULL
             AND status IS NULL""",
        (parent_session_id,),
    ).fetchall()
    for src_session_id, dst_origin, dst_native_id, link_type in rows:
        dispatch = _resolve_parent_dispatch_block(conn, parent_session_id, str(src_session_id), source_read=source_read)
        conn.execute(
            f"""UPDATE session_links
               SET parent_tool_use_block_id = ?,
                   evidence_json = {_DISPATCH_REASON_EVIDENCE_SQL},
                   method = CASE WHEN method = 'parser-parent' AND ? IS NOT NULL THEN 'parent-tool-use-id' ELSE method END
               WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
                 AND parent_tool_use_block_id IS NULL""",
            (
                dispatch.block_id,
                dispatch.reason,
                dispatch.reason,
                dispatch.block_id,
                src_session_id,
                dst_origin,
                dst_native_id,
                link_type,
            ),
        )
        _canonicalize_session_link_evidence(
            conn,
            src_session_id=str(src_session_id),
            dst_origin=str(dst_origin),
            dst_native_id=str(dst_native_id),
            link_type=str(link_type),
        )


def _root_projection_current(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        """
        SELECT root_session_id, parent_session_id, created_at_ms, updated_at_ms
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return row is not None and row[0] == session_id and row[1] is None


def _resolve_outbound_session_links(
    conn: sqlite3.Connection,
    session_id: str,
    origin: str,
    *,
    read: SessionLinkResolutionRead | None = None,
) -> None:
    """Resolve ``session_id``'s own unresolved outbound edges (it is the child).

    polylogue-4ts.10: candidates are evaluated one at a time (rather than a
    single blanket UPDATE) so each can be checked against
    ``sessions.parent_session_id`` before being resolved. A candidate whose
    resolution would close a loop or exhaust the walk budget is quarantined.

    ``status IS NULL`` is the OPERATIVE exclusion gate, deliberately kept
    stricter than ``topology_status_composes_sql()``. Any non-NULL status means
    an intervention already happened for this edge, and an edge under
    intervention must not silently acquire a resolved parent -- so the resolver
    excludes every exceptional marker, including ``REPAIRED``, not just the
    composition-excluded ones. Aligning it to the generated predicate would
    LOOSEN it (``REPAIRED`` would become resolvable), which is a real semantic
    change to a member that has no producer anywhere in the tree and therefore
    could not be verified; that is why this reads ``status IS NULL`` rather than
    the shared predicate.

    Consequence worth stating plainly, because it changes how the exclusion set
    should be read: a contradicted edge never resolves here, so composition can
    never traverse it regardless of
    ``COMPOSITION_EXCLUDED_TOPOLOGY_STATUSES``. That set is defense-in-depth for
    this member -- it covers an edge that resolved BEFORE acquiring a status
    (the cycle-quarantine path, which marks an already-resolved edge) and any
    future writer that sets a status post-resolution. It is not the mechanism
    that keeps contradicted edges out of lineage today.
    """
    if read is None:
        read = ConnectionSessionLinkResolutionRead(conn)
    candidates = read.resolution_candidates(session_id)
    for dst_origin, dst_native_id, link_type, proposed_parent_id in candidates:
        cycle_walk = _would_create_cycle(conn, child_id=session_id, proposed_parent_id=proposed_parent_id, read=read)
        if cycle_walk.outcome != "acyclic":
            written_rowids = _quarantine_session_link(
                conn,
                src_session_id=session_id,
                dst_origin=dst_origin,
                dst_native_id=dst_native_id,
                link_type=link_type,
                cycle_walk=cycle_walk,
                observed_at_ms=int(time.time() * 1000),
            )
            for rowid in written_rowids:
                read.resolution_written(rowid, ("status", "evidence_json", "resolved_at_ms"))
            continue
        with connection_cursor(
            conn,
            """
            UPDATE session_links
               SET resolved_dst_session_id = ?,
                   resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms),
                   evidence_json = json_remove(evidence_json, '$.resolution_reason')
             WHERE src_session_id = ?
               AND dst_origin = ?
               AND dst_native_id = ?
               AND link_type = ?
               AND resolved_dst_session_id IS NULL
               AND status IS NULL
            RETURNING rowid
            """,
            (proposed_parent_id, session_id, dst_origin, dst_native_id, link_type),
        ) as cursor:
            for (rowid,) in cursor:
                read.resolution_written(int(rowid), ("resolved_dst_session_id", "resolved_at_ms", "evidence_json"))


def _projected_session_kind(
    read: SessionLineageProjectionRead,
    session_id: str,
    branch_type: object,
) -> str | None:
    row = read.session_header(session_id)
    if row is None:
        return None
    kind = row[2]
    if kind is not None and not isinstance(kind, (str, SessionKind)):
        raise ValueError("projected session has an invalid canonical kind")
    return admitted_session_kind(kind, branch_type=cast("str | None", branch_type)).value


def _composing_parent_link(conn: sqlite3.Connection, session_id: str) -> tuple[object, object] | None:
    return ConnectionSessionLineageProjectionRead(conn).composing_parent_link(session_id)


def _refresh_session_projection(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    seen: set[str],
    read: SessionLineageProjectionRead | None = None,
) -> None:
    """Refresh the projection of ``session_id`` and every unrefreshed ancestor.

    Walks up to the first session already refreshed (or a root), then projects
    top-down, so a lineage of any depth never recurses; ``seen`` stops a cycle.
    """
    if read is None:
        read = ConnectionSessionLineageProjectionRead(conn)
    pending: list[tuple[str, str, object]] = []
    current = session_id
    while current not in seen:
        seen.add(current)
        parent_link = read.composing_parent_link(current)
        if parent_link is None:
            _project_lineage_root(conn, current, read=read)
            break
        pending.append((current, str(parent_link[0]), parent_link[1]))
        current = str(parent_link[0])
    for child_session_id, parent_session_id, link_type in reversed(pending):
        _project_lineage_child(conn, child_session_id, parent_session_id, link_type, read=read)


def _project_lineage_root(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    read: SessionLineageProjectionRead,
) -> None:
    unresolved_link = read.first_link_type(session_id)
    header = read.session_header(session_id)
    branch_type: str | None
    if unresolved_link is not None:
        branch_type = _branch_type_from_link_type(unresolved_link[0])
    else:
        branch_type = str(header[1]) if header is not None and header[1] else None
    previous_root_id = _stored_root_session_id(read, session_id)
    with connection_cursor(
        conn,
        """
        UPDATE sessions
        SET parent_session_id = NULL,
            root_session_id = session_id,
            branch_type = ?,
            session_kind = ?
        WHERE session_id = ?
        """,
        (branch_type, _projected_session_kind(read, session_id, branch_type), session_id),
    ):
        pass
    read.session_written(session_id, ("parent_session_id", "root_session_id", "branch_type", "session_kind"))
    if previous_root_id != session_id:
        _propagate_root_to_descendants(conn, session_id, session_id, read=read)


def _project_lineage_child(
    conn: sqlite3.Connection,
    session_id: str,
    parent_session_id: str,
    link_type: object,
    *,
    read: SessionLineageProjectionRead,
) -> None:
    parent_header = read.session_header(parent_session_id)
    parent_root_id = (
        str(parent_header[0]) if parent_header is not None and parent_header[0] is not None else parent_session_id
    )
    projected_branch_type = _branch_type_from_link_type(link_type)
    previous_root_id = _stored_root_session_id(read, session_id)
    with connection_cursor(
        conn,
        """
        UPDATE sessions
        SET parent_session_id = ?,
            root_session_id = ?,
            branch_type = ?,
            session_kind = ?
        WHERE session_id = ?
        """,
        (
            parent_session_id,
            parent_root_id,
            projected_branch_type,
            _projected_session_kind(read, session_id, projected_branch_type),
            session_id,
        ),
    ):
        pass
    read.session_written(session_id, ("parent_session_id", "root_session_id", "branch_type", "session_kind"))
    if previous_root_id != parent_root_id:
        _propagate_root_to_descendants(conn, session_id, parent_root_id, read=read)


def _stored_root_session_id(read: SessionLineageProjectionRead, session_id: str) -> str | None:
    row = read.session_header(session_id)
    return None if row is None or row[0] is None else str(row[0])


def _propagate_root_to_descendants(
    conn: sqlite3.Connection,
    session_id: str,
    root_session_id: str,
    *,
    read: SessionLineageProjectionRead,
) -> None:
    """Give every projected descendant of ``session_id`` its new root.

    A descendant's root is its parent's root, so when a late parent or a moved
    edge changes ``session_id``'s root, the whole subtree below it changes with
    it. The projection refresh only walks upward, and the ``threads`` view
    groups on the stored root, so a grandchild left on the old root would
    surface as a second thread. The walk follows the indexed
    ``parent_session_id`` projection, touches only this subtree, and ``UNION``
    terminates it on a cycle.
    """
    with connection_cursor(
        conn,
        """
        WITH RECURSIVE below(session_id) AS (
            SELECT session_id FROM sessions WHERE parent_session_id = :session_id
            UNION
            SELECT s.session_id FROM sessions AS s JOIN below ON s.parent_session_id = below.session_id
        )
        UPDATE sessions
           SET root_session_id = :root_session_id
         WHERE session_id IN (SELECT session_id FROM below)
           AND root_session_id IS NOT :root_session_id
        RETURNING rowid
        """,
        {"session_id": session_id, "root_session_id": root_session_id},
    ) as cursor:
        for (rowid,) in cursor:
            read.descendant_root_written(int(rowid))


def _refresh_thread(conn: sqlite3.Connection, root_session_id: str) -> None:
    # Thread membership and summary are query-time views over sessions.
    del conn, root_session_id


def _next_session_event_position(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


# Event types lowered into sibling typed tables or represented by dialogue
# messages. Independent wire evidence must have its own retained event:
#
# - ``token_count`` / ``message_usage``: numeric usage is retained in
#   ``session_provider_usage_events`` (the cost model's read path,
#   ``storage/usage.py``). This table does not retain arbitrary wire fields.
#   Codex quota windows are preserved by a separate ``rate_limits`` event;
#   historical rows written before that parser change do not contain them.
# - ``agent_policy``: fully re-derivable from ``session_agent_policies``
#   (dedicated typed table, identical fields, sole confirmed reader
#   ``read_session_agent_policies``).
# - ``agent_message``: the payload never carries text (Codex never populates
#   it there); the real text is guaranteed to exist as a ``ParsedMessage``
#   via ``_codex_event_message`` -- this is a pure existence marker with a
#   message-shaped twin already present.
# - ``agent_reasoning`` (polylogue-fuky, 2026-08-02, the pending
#   evidence-doctrine call this comment used to defer): confirmed DUPLICATE
#   by reading raw wire records across three live Codex sessions --
#   ``agent_reasoning.text`` is the same live-streamed reasoning-summary
#   bullet text already carried in full by the paired ``reasoning`` record's
#   ``summary[].text`` (one session: 156/156 identical values; a second:
#   262/262 identical set; a third: 1,846 reasoning bullets vs. 1,859
#   agent_reasoning ticks, >99% overlap, the residual being minor
#   text-normalization on the same underlying bullets). ``reasoning``
#   records are already materialized as a THINKING-block ``ParsedMessage``
#   via ``_codex_reasoning_message`` (index v50) -- ``agent_reasoning`` is a
#   live per-tick echo of that same content with nothing incremental to
#   offer, matching this file's own ``agent_message`` rationale above (a
#   twin already exists) plus the "streaming ticks superseded by the final
#   record" pattern documented for Claude Code's ``progress`` subtypes in
#   ``claude/code_parser.py``. See ``sources/parsers/codex.py``'s
#   ``_CODEX_KNOWN_RESPONSE_ITEM_TYPES`` comment for the full classification
#   this filtering decision is one piece of.
#
# ``reasoning``/``turn_context`` remain deliberately excluded from this set
# (still need their own operator evidence-doctrine call -- out of scope for
# the polylogue-fuky audit that resolved ``agent_reasoning``);
# ``function_call``/``function_call_output`` payload-slimming is a separate,
# not-yet-decided change. Parsers keep emitting all of these events
# unchanged -- only this writer materialization step filters them.
_SESSION_EVENTS_REDUNDANT_TYPES = frozenset(
    {"token_count", "message_usage", "agent_policy", "agent_message", "agent_reasoning"}
)
# Usage events carry a typed ``source_message_resolution``: an id naming
# several occurrences is recorded as ``ambiguous`` (polylogue-1pzmq) instead of
# refusing the session. Other events have no typed slot for that ambiguity.
_TYPED_USAGE_EVENT_TYPES = frozenset({"token_count", "message_usage"})


def _last_agent_policy_values(
    conn: sqlite3.Connection, session_id: str
) -> tuple[str | None, str | None, str | None] | None:
    """Return the policy tuple of the session's highest-position stored row.

    polylogue-cuxz.11: ``session_agent_policies`` records a value-change
    interval, so an append must compare against what is already stored rather
    than starting a fresh run. A full replace clears the session's rows first
    (``_clear_session_projection_rows``), so this correctly returns ``None``
    there and the first observation is always retained.
    """
    row = conn.execute(
        """
        SELECT approval_policy, sandbox_policy, network_policy
        FROM session_agent_policies
        WHERE session_id = ?
        ORDER BY position DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return (row[0], row[1], row[2])


_SessionEventInsertRow = tuple[
    str,
    str | None,
    str | None,
    int,
    str | None,
    str,
    int | None,
    int | None,
    int | None,
    str | None,
    str | None,
]


def _stored_event_payload(
    event: ParsedSessionEvent, sidecar_blob_locators: Mapping[str, Mapping[str, str]] | None
) -> Mapping[str, object]:
    """The event payload as stored, with its sidecar blob locator beside it.

    The locator (``blob_hash``, or the typed ``blob_refusal``) is publication
    metadata the ingest batch learns when it publishes the sidecar's bytes,
    after the session's identity was bound over its parsed events
    (polylogue-bgnxh). It is added to the stored row only; the parsed event,
    and so the session's content hash, never carry it.
    """
    if not sidecar_blob_locators or event.event_type not in SIDECAR_BLOB_EVENT_TYPES:
        return event.payload
    tool_use_id = event.payload.get("tool_use_id")
    locator = sidecar_blob_locators.get(tool_use_id) if isinstance(tool_use_id, str) else None
    if locator is None:
        return event.payload
    undeclared = set(locator) - SIDECAR_LOCATOR_KEYS
    if undeclared:
        # A key outside the declared locator set would enter the stored row
        # without the hash partition excluding it.
        raise ValueError(f"sidecar locator carries undeclared keys: {sorted(undeclared)}")
    return {**event.payload, **locator}


def _write_session_events(
    conn: sqlite3.Connection,
    session_id: str,
    messages: Sequence[ParsedMessage],
    events: Iterable[ParsedSessionEvent],
    *,
    owner_resolution: MessageOwnerResolution,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    position_offset: int = 0,
    event_position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    inherited_source_message_ids: Mapping[str, str] | None = None,
    ambiguous_source_provider_ids: frozenset[str] = frozenset(),
    sidecar_blob_locators: Mapping[str, Mapping[str, str]] | None = None,
) -> SessionEventWriteResult:
    source = messages.messages if isinstance(messages, _MessageTail) else messages
    disk_index = _DiskMessageEventIndex(source.path.parent) if isinstance(source, SqliteMessageSink) else None
    try:
        by_native_id: dict[str, str] | _DiskMessageEventIndex = disk_index if disk_index is not None else {}
        by_owner_key: dict[str, str | None] = {}
        owner_keys = iter(owner_resolution.keys)
        if isinstance(messages, _MessageTail):
            for _ordinal in range(messages.start):
                owner_key = next(owner_keys)
                if owner_key in owner_resolution.ambiguous_keys:
                    continue
                if disk_index is not None:
                    disk_index.add_occurrence(owner_key, None)
                else:
                    by_owner_key[owner_key] = None
        for fallback_position, (message, owner_key) in enumerate(zip(messages, owner_keys, strict=True)):
            message_id = _message_id(
                session_id,
                message,
                fallback_position,
                content_identities=content_identities,
                message_id_remap=message_id_remap,
                duplicate_native_ids=duplicate_native_ids,
            )
            if owner_key not in owner_resolution.ambiguous_keys:
                if disk_index is not None:
                    disk_index.add_occurrence(owner_key, message_id)
                else:
                    by_owner_key[owner_key] = message_id
            if disk_index is not None:
                effective_position = message.position if message.position is not None else fallback_position
                disk_index.add_boundary(effective_position, message_id)
            if (
                message.provider_message_id
                and _normalized_message_native_id(message) not in duplicate_native_ids
                and _normalized_message_native_id(message) not in ambiguous_source_provider_ids
            ):
                by_native_id[message.provider_message_id] = message_id
        wrote_provider_usage_events = False
        position = event_position_offset
        session_event_rows: list[_SessionEventInsertRow] = []
        agent_policy_rows: list[tuple[object, ...]] = []
        # polylogue-cuxz.11: the Codex wire restates the whole policy on every
        # turn_context, so a row per observation made this table a change-log of
        # non-changes -- 402,869 rows carrying 3,053 distinct facts across 3,031
        # sessions, 99.3% of which never changed policy at all. Only a genuine
        # value change is retained; `position` still marks where the retained
        # value took effect, so a row is the START of an interval that runs until
        # the next row.
        last_agent_policy = _last_agent_policy_values(conn, session_id)
        provider_usage_rows: list[tuple[object, ...]] = []

        def _flush_rows() -> None:
            if session_event_rows:
                conn.executemany(
                    "DELETE FROM session_event_array_items WHERE session_id = ? AND event_position = ?",
                    ((str(row[0]), int(row[3])) for row in session_event_rows),
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO session_events (session_id, source_message_id, "
                    "source_message_provider_id, position, event_type, payload_json, occurred_at_ms, "
                    "boundary_start_position, boundary_end_position, boundary_message_id, "
                    "source_message_provider_id_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    session_event_rows,
                )
                session_event_rows.clear()
            if agent_policy_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO session_agent_policies (session_id, source_message_id, "
                    "position, approval_policy, sandbox_policy, network_policy, observed_at_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    agent_policy_rows,
                )
                agent_policy_rows.clear()
            if provider_usage_rows:
                conn.executemany(_PROVIDER_USAGE_EVENT_INSERT_SQL, provider_usage_rows)
                provider_usage_rows.clear()

        def _insert_streamed_event(row: _SessionEventInsertRow, arrays: Mapping[str, object]) -> None:
            conn.execute(
                "DELETE FROM session_event_array_items WHERE session_id = ? AND event_position = ?",
                (str(row[0]), int(row[3])),
            )
            conn.execute(
                "INSERT OR REPLACE INTO session_events (session_id, source_message_id, "
                "source_message_provider_id, position, event_type, payload_json, occurred_at_ms, "
                "boundary_start_position, boundary_end_position, boundary_message_id, "
                "source_message_provider_id_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            for payload_key, array in arrays.items():
                from polylogue.sources.streamed_event_payload import StreamedJsonArray

                if not isinstance(array, StreamedJsonArray):
                    raise TypeError("session event streamed payload marker changed after preparation")
                conn.executemany(
                    "INSERT INTO session_event_array_items "
                    "(session_id, event_position, payload_key, item_ordinal, value_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        (session_id, int(row[3]), payload_key, ordinal, _json_dumps(value))
                        for ordinal, value in enumerate(array.iter_values())
                    ),
                )

        source_message_id: str | None
        for event in events:
            source_message_provider_id = event.source_message_provider_id
            if (
                event.owner_coordinate is None
                and inherited_source_message_ids is not None
                and source_message_provider_id is not None
                and source_message_provider_id in inherited_source_message_ids
            ):
                continue
            if event.owner_coordinate is not None:
                event_owner_key = event_message_owner_key(event, owner_resolution)
                source_message_id = (
                    disk_index.occurrence_message_id(event_owner_key)
                    if disk_index is not None and event_owner_key is not None
                    else by_owner_key.get(event_owner_key or "")
                )
                if source_message_id is None:
                    inherited = (
                        disk_index.has_occurrence(event_owner_key)
                        if disk_index is not None and event_owner_key is not None
                        else event_owner_key in by_owner_key
                    )
                    if inherited:
                        continue
                    raise MessageOwnerAmbiguityError("event occurrence was not published")
            else:
                source_message_id = by_native_id.get(source_message_provider_id or "")
                if (
                    source_message_id is None
                    and source_message_provider_id
                    and event.event_type not in _TYPED_USAGE_EVENT_TYPES
                    and (
                        message_native_key(source_message_provider_id) in duplicate_native_ids
                        or message_native_key(source_message_provider_id) in ambiguous_source_provider_ids
                    )
                ):
                    raise MessageOwnerAmbiguityError("event native message ID requires exact occurrence evidence")
            if source_message_id is None and inherited_source_message_ids is not None:
                source_message_id = inherited_source_message_ids.get(source_message_provider_id or "")
            if event.event_type not in _SESSION_EVENTS_REDUNDANT_TYPES:
                boundary_message_id = None
                if event.boundary_message_position is not None:
                    if disk_index is not None:
                        boundary_message_id = disk_index.boundary_message_id(event.boundary_message_position)
                    else:
                        for fallback_position, message in enumerate(messages):
                            # Parsers may leave ``position`` unset; the ordinal is then
                            # the message's position, exactly as ``_message_id`` derives it.
                            effective_position = message.position if message.position is not None else fallback_position
                            if effective_position == event.boundary_message_position:
                                boundary_message_id = _message_id(
                                    session_id,
                                    message,
                                    fallback_position,
                                    content_identities=content_identities,
                                    message_id_remap=message_id_remap,
                                    duplicate_native_ids=duplicate_native_ids,
                                )
                                break
                stored_payload = _stored_event_payload(event, sidecar_blob_locators)
                from polylogue.sources.streamed_event_payload import StreamedJsonArray

                streamed_arrays = {
                    key: value for key, value in stored_payload.items() if isinstance(value, StreamedJsonArray)
                }
                ordinary_payload: dict[str, object] = {}
                for key, value in stored_payload.items():
                    if isinstance(value, StreamedJsonArray):
                        if len(value) == 0:
                            ordinary_payload[key] = []
                    else:
                        ordinary_payload[key] = value
                row: _SessionEventInsertRow = (
                    session_id,
                    source_message_id,
                    sqlite_message_native_id(source_message_provider_id),
                    position,
                    _sqlite_text(event.event_type),
                    _json_dumps(ordinary_payload),
                    to_epoch_ms(event.timestamp, numeric_unit="seconds"),
                    event.boundary_start_position + position_offset
                    if event.boundary_start_position is not None
                    else None,
                    event.boundary_end_position + position_offset if event.boundary_end_position is not None else None,
                    boundary_message_id,
                    source_native_id_json(source_message_provider_id),
                )
                if streamed_arrays:
                    _flush_rows()
                    _insert_streamed_event(row, streamed_arrays)
                else:
                    session_event_rows.append(row)
            if event.event_type == "agent_policy":
                # Every field keeps its named producer: approval/sandbox/network
                # all come from this payload, and ``source_message_id`` is the
                # resolved message the observation was attached to (its partial
                # index and prefix-delete clear path live in
                # ``clear_session_agent_policies_source_message_sql``). Nothing is
                # dropped for having been constant in one census; what is dropped
                # is the restatement of an unchanged value.
                policy_values = (
                    _sqlite_text(_payload_string(event.payload, "approval", "approval_policy")),
                    _sqlite_text(_payload_string(event.payload, "sandbox", "sandbox_policy")),
                    _sqlite_text(_payload_string(event.payload, "network", "network_policy")),
                )
                if policy_values != last_agent_policy:
                    last_agent_policy = policy_values
                    agent_policy_rows.append(
                        (
                            session_id,
                            source_message_id,
                            position,
                            *policy_values,
                            to_epoch_ms(event.timestamp, numeric_unit="seconds"),
                        ),
                    )
            elif event.event_type in _TYPED_USAGE_EVENT_TYPES:
                # polylogue-1pzmq: an event whose declared provider message id
                # resolves to no row here used to be dropped outright. The
                # commonest cause is not a malformed export but a deliberate
                # writer decision: a provider id duplicated within the session is
                # excluded from ``by_native_id`` because those messages get
                # content-derived ids, so every usage event attached to them
                # vanished. The usage is still real evidence about this session;
                # record it with its declared provider id and a typed statement of
                # what the attribution actually is.
                declared_provider_id = normalized_message_native_id(event.source_message_provider_id)
                declared_provider_id = declared_provider_id or None
                resolution = _provider_usage_source_resolution(
                    declared_provider_id,
                    source_message_id=source_message_id,
                    ambiguous_source_provider_ids=ambiguous_source_provider_ids,
                    duplicate_native_ids=duplicate_native_ids,
                )
                usage_row = _provider_usage_event_row(
                    session_id,
                    source_message_id,
                    position,
                    event,
                    source_message_provider_id=declared_provider_id,
                    source_message_resolution=resolution,
                )
                if _provider_usage_event_has_evidence(event, usage_row):
                    provider_usage_rows.append(usage_row)
                    wrote_provider_usage_events = True
            position += 1
            if max(len(session_event_rows), len(agent_policy_rows), len(provider_usage_rows)) >= 128:
                _flush_rows()
        _flush_rows()
        return SessionEventWriteResult(wrote_provider_usage_events=wrote_provider_usage_events)

    finally:
        if disk_index is not None:
            disk_index.close()


class _DiskMessageEventIndex(Mapping[str, str]):
    """Indexed source and boundary lookups for a disk-backed session."""

    def __init__(self, directory: Path) -> None:
        from polylogue.storage.sqlite.connection_profile import open_scratch_connection

        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-event-owners-", dir=directory)
        self._sql_owner = open_scratch_connection(
            Path(self._scratch.name) / "owners.db", scratch_directory=self._scratch, lifetime_dependencies=(self,)
        )
        assert self._sql_owner.connection is not None
        try:
            self._conn.execute(
                "CREATE TABLE owner (provider_id TEXT PRIMARY KEY, message_id TEXT NOT NULL) WITHOUT ROWID"
            )
            self._conn.execute("CREATE TABLE boundary (position INTEGER PRIMARY KEY, message_id TEXT NOT NULL)")
            self._conn.execute("CREATE TABLE occurrence (owner_key TEXT PRIMARY KEY, message_id TEXT) WITHOUT ROWID")
        except BaseException as primary:
            _close_failed_native_construction(self._sql_owner, primary)
            self.close()
            raise

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._sql_owner.require_connection()

    def __setitem__(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO owner VALUES (?, ?)", (message_native_key(key), value))

    def add_boundary(self, position: int, message_id: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO boundary VALUES (?, ?)", (position, message_id))

    def replace_boundary(self, position: int, message_id: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO boundary VALUES (?, ?)", (position, message_id))

    def boundary_message_id(self, position: int) -> str | None:
        row = self._conn.execute("SELECT message_id FROM boundary WHERE position = ?", (position,)).fetchone()
        return str(row[0]) if row is not None else None

    def __getitem__(self, key: str) -> str:
        row = self._conn.execute(
            "SELECT message_id FROM owner WHERE provider_id = ?", (message_native_key(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        after = ""
        first = True
        while True:
            rows = self._conn.execute(
                "SELECT provider_id FROM owner WHERE ? OR provider_id > ? ORDER BY provider_id LIMIT 512",
                (first, after),
            ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            first = False
            yield from (str(row[0]) for row in rows)

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM owner").fetchone()[0])

    def close(self) -> None:
        if self._sql_owner in retained_native_sql_owners_for_lifetime(self):
            self._sql_owner.close()

    def __del__(self) -> None:
        if getattr(self, "_sql_owner", None) is not None:
            self.close()

    def add_occurrence(self, owner_key: str, message_id: str | None) -> None:
        self._conn.execute("INSERT INTO occurrence VALUES (?, ?)", (owner_key, message_id))

    def has_occurrence(self, owner_key: str) -> bool:
        return self._conn.execute("SELECT 1 FROM occurrence WHERE owner_key = ?", (owner_key,)).fetchone() is not None

    def occurrence_message_id(self, owner_key: str) -> str | None:
        row = self._conn.execute("SELECT message_id FROM occurrence WHERE owner_key = ?", (owner_key,)).fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None


_PROVIDER_USAGE_EVENT_INSERT_SQL = """
    INSERT OR REPLACE INTO session_provider_usage_events (
        session_id, source_message_id, position, provider_event_type, model_name,
        last_input_tokens, last_output_tokens, last_cached_input_tokens,
        last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
        total_input_tokens, total_output_tokens, total_cached_input_tokens,
        total_cache_write_tokens, total_reasoning_output_tokens, total_tokens,
        occurred_at_ms, request_id,
        source_message_provider_id, source_message_resolution, finish_reason,
        api_block_index, quota_limits_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _provider_usage_source_resolution(
    declared_provider_id: str | None,
    *,
    source_message_id: str | None,
    ambiguous_source_provider_ids: frozenset[str],
    duplicate_native_ids: frozenset[str],
) -> str:
    """Classify how this usage event's provider message id resolved.

    ``session`` is the absence of a claim (Codex's session-global
    ``token_count``); ``ambiguous`` and ``unresolved`` are two different
    failures to honour one -- the first because the id names more than one
    message in this session, the second because it names none.
    """
    if declared_provider_id is None:
        return "session"
    if source_message_id is not None:
        return "resolved"
    if (
        message_native_key(declared_provider_id) in ambiguous_source_provider_ids
        or message_native_key(declared_provider_id) in duplicate_native_ids
    ):
        return "ambiguous"
    return "unresolved"


def _provider_usage_event_row(
    session_id: str,
    source_message_id: str | None,
    position: int,
    event: ParsedSessionEvent,
    *,
    source_message_provider_id: str | None = None,
    source_message_resolution: str = "session",
) -> tuple[object, ...]:
    """Build one ``session_provider_usage_events`` row from a parsed event.

    The ``total_*`` lanes are stored exactly as the provider reported them,
    including on a prefix-sharing child. Every origin that reaches these
    columns (``codex-session`` and ``hermes-session`` are the only two that
    emit ``total_token_usage``) scopes its cumulative counter to the physical
    session, not to the lineage chain: a resumed Codex thread restarts its
    ``total_token_usage`` at one context window, and a Hermes child row's
    counters move up and down relative to its parent's. Rebasing a child
    against its parent therefore subtracts tokens the child never counted
    (polylogue-uoq3x).
    """
    last_usage = _payload_mapping(event.payload, "last_token_usage")
    total_usage = _payload_mapping(event.payload, "total_token_usage")
    total_input = _payload_optional_int(total_usage, "input_tokens")
    total_output = _payload_optional_int(total_usage, "output_tokens")
    total_cache_read = _payload_optional_int(total_usage, "cached_input_tokens")
    total_cache_write = _payload_optional_int(total_usage, "cache_write_tokens")
    total_reasoning = _payload_optional_int(total_usage, "reasoning_output_tokens")
    last_total_tokens = _payload_optional_int(last_usage, "total_tokens")
    total_tokens = _payload_optional_int(total_usage, "total_tokens")
    return (
        session_id,
        source_message_id,
        position,
        _sqlite_text(event.event_type),
        _sqlite_text(_payload_string(event.payload, "model", "model_name")),
        _payload_optional_int(last_usage, "input_tokens"),
        _payload_optional_int(last_usage, "output_tokens"),
        _payload_optional_int(last_usage, "cached_input_tokens"),
        _payload_optional_int(last_usage, "cache_write_tokens"),
        _payload_optional_int(last_usage, "reasoning_output_tokens"),
        last_total_tokens,
        total_input,
        total_output,
        total_cache_read,
        total_cache_write,
        total_reasoning,
        total_tokens,
        to_epoch_ms(event.timestamp, numeric_unit="seconds"),
        _sqlite_text(_payload_string(event.payload, "request_id")),
        _sqlite_text(source_message_provider_id),
        source_message_resolution,
        _sqlite_text(_payload_string(event.payload, "finish_reason", "stop_reason")),
        _payload_optional_int(event.payload, "api_block_index"),
        (_json_dumps(quota_limits) if (quota_limits := _payload_mapping(event.payload, "quota_limits")) else None),
    )


def _provider_usage_event_has_evidence(event: ParsedSessionEvent, row: tuple[object, ...]) -> bool:
    """Return whether this event carries any fact worth a ``session_provider_usage_events`` row.

    ``message_usage``/``token_count`` are in ``_SESSION_EVENTS_REDUNDANT_TYPES``
    on the premise that this typed row carries the whole payload, so anything
    the payload states and this predicate does not recognise is dropped on the
    floor -- which is how a Drive chunk reporting only ``finishReason``, and a
    Claude turn reporting ``stop_reason`` beside an all-zero ``usage``, used to
    disappear (polylogue-1pzmq).

    Evidence is decided over the facts this row can actually HOLD, not over
    the payload as a whole: writing a row for a fact with no column stores
    nothing and only makes the loss harder to see. ``polylogue-664l`` dropped
    the eight Hermes billing-provenance columns after a zero-reader audit, so
    a billing-only payload still writes no row here.

    A present token counter, including measured zero, is evidence. Two other
    facts have columns and count on their own:

    - the provider correlation id (``request_id``);
    - the provider's terminal signal (``finish_reason``/``stop_reason``) --
      the fact this predicate used to destroy, because a Drive chunk that
      reports only ``finishReason``, and a Claude turn that reports
      ``stop_reason`` beside an all-zero ``usage``, both leave every token
      field at zero.

    ``model`` deliberately does not count: it names the subject of an
    observation rather than being one.

    Parsers omit counters they cannot observe, so NULL remains distinct from
    an explicit zero through this row.
    """
    if any(value is not None for value in row[5:17]):
        return True
    return (
        bool(_payload_string(event.payload, "request_id"))
        or bool(_payload_string(event.payload, "finish_reason", "stop_reason"))
        or _payload_optional_int(event.payload, "api_block_index") is not None
        or bool(_payload_mapping(event.payload, "quota_limits"))
    )


def _aggregate_provider_usage_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Persist the canonical complete-session provider event projection."""
    for projection in _provider_usage_projections(conn, session_id):
        if projection.model_name is None:
            continue  # The unresolved event remains durable evidence, without guessed attribution.
        _upsert_provider_usage_model_rollup(conn, projection)


def _aggregate_appended_provider_usage_into_model_usage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    start_position: int,
) -> None:
    """Fold the append window canonically and retain the original persistence law."""
    for projection in _provider_usage_projections(conn, session_id, start_position=start_position):
        model = projection.model_name
        if model is None:
            continue
        if projection.cumulative:
            _upsert_provider_usage_model_rollup(conn, projection)
            _clear_stale_cumulative_rollups(conn, session_id, keep_model=model)
        elif not _provider_usage_has_cumulative_total(conn, session_id, model):
            _increment_provider_usage_model_rollup(conn, projection)
        elif "missing_token_lanes" in projection.missing_reasons:
            conn.execute(
                "UPDATE session_model_usage SET catalog_cost_usd = NULL, provider_lanes_complete = 0 "
                "WHERE session_id = ? AND model_name = ?",
                (session_id, model),
            )


def _has_appended_claude_message_usage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    start_position: int,
) -> bool:
    return (
        conn.execute(
            """
        SELECT 1 FROM session_provider_usage_events
        WHERE session_id = ? AND provider_event_type = 'message_usage' AND position >= ?
        LIMIT 1
        """,
            (session_id, start_position),
        ).fetchone()
        is not None
    )


def _reconcile_session_model_usage_from_persisted_evidence(
    conn: sqlite3.Connection,
    session_id: str,
) -> int:
    """Rebuild model usage from stored messages and provider events in this transaction."""
    _reconcile_session_model_usage_rows(conn, session_id)
    _aggregate_message_tokens_into_model_usage(conn, session_id)
    _reprice_model_usage_rows(conn, session_id)
    _aggregate_provider_usage_into_model_usage(conn, session_id)
    reported_cost_row = conn.execute(
        "SELECT reported_cost_usd FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    model_names = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        )
    )
    if reported_cost_row is not None and reported_cost_row[0] is not None and model_names:
        _write_provider_cost(conn, session_id, model_names, ProviderCost(float(reported_cost_row[0])))
    row = conn.execute(
        "SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _provider_usage_existing_models(conn: sqlite3.Connection, session_id: str) -> list[str]:
    return [
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        ).fetchall()
        if row[0] and str(row[0]).strip()
    ]


def _provider_usage_has_cumulative_total(conn: sqlite3.Connection, session_id: str, model_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
          AND model_name = ?
          AND (
            total_input_tokens IS NOT NULL
            OR total_output_tokens IS NOT NULL
            OR total_cached_input_tokens IS NOT NULL
            OR total_cache_write_tokens IS NOT NULL
          )
        LIMIT 1
        """,
        (session_id, model_name),
    ).fetchone()
    return row is not None


def _clear_stale_cumulative_rollups(conn: sqlite3.Connection, session_id: str, *, keep_model: str) -> None:
    """Return every model except ``keep_model`` to its per-message token totals.

    The Codex cumulative total is session-global, so exactly one rollup row
    should carry it. When an append window's latest cumulative is attributed to
    a different model than a previous window, the earlier model's rollup still
    holds a (now-subsumed) cumulative; left in place it would be summed back in
    on read (#2472).

    Each other row is set to exactly what a full write leaves it holding once
    the cumulative goes to ``keep_model``: the sum of its own messages' token
    columns, zero when it has none. Whether a message reported a counter is
    not the test -- a message that reported an explicit zero is a measurement
    of zero, and exempting its model kept a whole stale cumulative on the row.

    Rows with tokens or unmappable provider evidence are compared with their
    messages. The new cumulative supersedes that earlier incomplete evidence
    even when its additive lanes were zero.
    """
    stored_rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, provider_lanes_complete, provider_usage_observed
        FROM session_model_usage
        WHERE session_id = ? AND model_name != ?
          AND (input_tokens + output_tokens + cache_read_tokens + cache_write_tokens > 0
               OR provider_lanes_complete = 0 OR provider_usage_observed)
        """,
        (session_id, keep_model),
    ).fetchall()
    if not stored_rows:
        return
    candidate_models = [str(row[0]) for row in stored_rows]
    observed_sql, complete_sql = _message_usage_presence_sql(conn, session_id)
    message_totals = {
        str(row[0]): (
            int(row[1] or 0),
            int(row[2] or 0),
            int(row[3] or 0),
            int(row[4] or 0),
            bool(row[5]),
            bool(row[6]),
        )
        for row in conn.execute(
            f"""
            SELECT model_name,
                   SUM(input_tokens), SUM(output_tokens),
                   SUM(cache_read_tokens), SUM(cache_write_tokens),
                   {observed_sql}, {complete_sql}
            FROM messages
            WHERE session_id = ?
              AND model_name IN ({", ".join("?" for _ in candidate_models)})
            GROUP BY model_name
            """,
            (session_id, *candidate_models),
        )
    }
    for row in stored_rows:
        model_name = str(row[0])
        totals = message_totals.get(model_name, (0, 0, 0, 0, False, True))
        if (
            (int(row[1] or 0), int(row[2] or 0), int(row[3] or 0), int(row[4] or 0)) == totals[:4]
            and bool(row[5]) == totals[5]
            and bool(row[6]) == totals[4]
        ):
            continue
        catalog_cost = _price_provider_usage_tokens(
            conn,
            model_name,
            input_tokens=totals[0],
            output_tokens=totals[1],
            cache_read_tokens=totals[2],
            cache_write_tokens=totals[3],
        )
        if not totals[5]:
            catalog_cost = None
        elif totals[4] and not any(totals[:4]):
            zero_cost, _ = catalog_cost_for_tokens(model_name, 0, 0, 0, 0)
            catalog_cost = None if zero_cost is None else CatalogCost(zero_cost)
        conn.execute(
            """
            UPDATE session_model_usage
            SET input_tokens = ?,
                output_tokens = ?,
                cache_read_tokens = ?,
                cache_write_tokens = ?,
                catalog_cost_usd = ?,
                provider_lanes_complete = ?,
                provider_usage_observed = ?
            WHERE session_id = ? AND model_name = ?
            """,
            (
                *totals[:4],
                None if catalog_cost is None else catalog_cost.value,
                int(totals[5]),
                int(totals[4]),
                session_id,
                model_name,
            ),
        )


def _price_provider_usage_tokens(
    conn: sqlite3.Connection,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> CatalogCost | None:
    """Catalog-price disjoint-lane token totals for a provider-usage rollup row.

    Return a catalog-computed cost. Provider-reported dollars use a separate
    ``ProviderCost`` write path and can never enter this function.
    """
    if input_tokens + output_tokens + cache_read_tokens + cache_write_tokens <= 0:
        return None  # This scalar-only caller has no observation proving measured zero.
    cost_usd, _ = catalog_cost_for_tokens(
        model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    )
    return None if cost_usd is None else CatalogCost(cost_usd)


def _reprice_model_usage_rows(conn: sqlite3.Connection, session_id: str) -> int:
    """Refresh catalog costs for persisted token totals during a rebuild.

    ``session_model_usage`` rows can outlive the provider-event or message
    evidence that originally produced them (for example, an index generated
    before catalog pricing was wired in). Rebuilds must still reprice those
    canonical token totals so catalog-covered models are not left with a
    misleading NULL cost. Provider-reported dollars remain untouched.
    """
    rows = conn.execute(
        """
        SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, provider_lanes_complete, provider_usage_observed
        FROM session_model_usage
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()
    changed = 0
    for row in rows:
        if not row[5]:
            continue
        model_name = str(row[0] or "").strip()
        if not model_name:
            continue
        catalog_cost = _price_provider_usage_tokens(
            conn,
            model_name,
            input_tokens=int(row[1] or 0),
            output_tokens=int(row[2] or 0),
            cache_read_tokens=int(row[3] or 0),
            cache_write_tokens=int(row[4] or 0),
        )
        if row[6] and not any(row[index] for index in range(1, 5)):
            zero_cost, _ = catalog_cost_for_tokens(model_name, 0, 0, 0, 0)
            catalog_cost = None if zero_cost is None else CatalogCost(zero_cost)
        value = None if catalog_cost is None else catalog_cost.value
        changed += conn.execute(
            """
            UPDATE session_model_usage
            SET catalog_cost_usd = ?
            WHERE session_id = ? AND model_name = ?
            """,
            (value, session_id, model_name),
        ).rowcount
    return changed


def _upsert_provider_usage_model_rollup(conn: sqlite3.Connection, projection: UsageProjectionModel) -> None:
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name, input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens, catalog_cost_usd, provider_lanes_complete, provider_usage_observed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            catalog_cost_usd = excluded.catalog_cost_usd,
            provider_lanes_complete = excluded.provider_lanes_complete,
            provider_usage_observed = excluded.provider_usage_observed
        """,
        (
            projection.session_id,
            projection.model_name,
            projection.input_tokens,
            projection.output_tokens,
            projection.cache_read_tokens,
            projection.cache_write_tokens,
            projection.cost_usd,
            int(projection.provider_lanes_complete),
            int(projection.provider_usage_observed),
        ),
    )


def _increment_provider_usage_model_rollup(
    conn: sqlite3.Connection,
    projection: UsageProjectionModel,
) -> None:
    existing = conn.execute(
        """SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, provider_lanes_complete, provider_usage_observed
           FROM session_model_usage WHERE session_id = ? AND model_name = ?""",
        (projection.session_id, projection.model_name),
    ).fetchone()
    lanes = tuple(
        int((existing[index] if existing else 0) or 0) + value
        for index, value in enumerate(
            (
                projection.input_tokens,
                projection.output_tokens,
                projection.cache_read_tokens,
                projection.cache_write_tokens,
            )
        )
    )
    cost, reasons = catalog_cost_for_tokens(projection.model_name, *lanes)
    provider_lanes_complete = projection.provider_lanes_complete and (existing is None or bool(existing[4]))
    if not provider_lanes_complete:
        cost = None
        reasons = (*reasons, "missing_token_lanes")
    _upsert_provider_usage_model_rollup(
        conn,
        UsageProjectionModel(
            session_id=projection.session_id,
            model_name=projection.model_name,
            input_tokens=lanes[0],
            output_tokens=lanes[1],
            cache_read_tokens=lanes[2],
            cache_write_tokens=lanes[3],
            cost_usd=cost,
            state="incomplete" if cost is None else "complete",
            missing_reasons=reasons,
            provider_lanes_complete=provider_lanes_complete,
            provider_usage_observed=projection.provider_usage_observed or (existing is not None and bool(existing[5])),
        ),
    )


#: Session-owned columns an event-only write keeps from the stored row.
_EVENT_ONLY_PRESERVED_COLUMNS: tuple[str, ...] = (
    "raw_id",
    "content_hash",
    "branch_type",
    "active_leaf_message_id",
    "title",
    "session_kind",
    "title_source",
    "title_ref",
    "display_name",
    "pending_drafts_json",
    "git_branch",
    "git_repository_url",
    "commit_hash",
    "instructions_text",
    "reported_duration_ms",
    "reported_cost_usd",
    "provider_project_ref",
    "created_at_ms",
    "updated_at_ms",
)


def _stored_session_header(
    conn: sqlite3.Connection, session_id: str
) -> dict[str, bytes | str | int | float | None] | None:
    """The stored session-owned header, or ``None`` when the session is absent."""
    row = conn.execute(
        f"SELECT {', '.join(_EVENT_ONLY_PRESERVED_COLUMNS)} FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return None if row is None else dict(zip(_EVENT_ONLY_PRESERVED_COLUMNS, tuple(row), strict=True))


def _write_working_dirs(conn: sqlite3.Connection, session_id: str, working_directories: Iterable[str]) -> None:
    for position, path in enumerate(working_directories):
        conn.execute(
            """
            INSERT OR REPLACE INTO session_working_dirs (session_id, path, position)
            VALUES (?, ?, ?)
            """,
            (session_id, _sqlite_text(path), position),
        )


def _seed_session_model_usage_rows(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    replace_existing_model_rows: bool = True,
    aggregate_message_tokens: bool = True,
) -> None:
    """Seed skeleton ``session_model_usage`` rows for a session's known models.

    Formerly also wrote ``session.reported_cost_usd`` into a
    ``session_reported_costs`` table; that write path was removed
    (polylogue-v2mg) as a zero-consumer table -- nothing ever read it back.
    ``session.reported_cost_usd`` is instead written onto ``sessions.
    reported_cost_usd`` by the main session INSERT above (polylogue-gt1z,
    v49) -- a session-level column, not a per-model one, matching the shape
    of the value (one exact dollar total per session, not per model). That
    column is what feeds ``_session_level_estimate``'s real ``status ==
    "exact"`` cost path; nothing here writes it a second time.
    """
    declared_names = {model_name.strip() for model_name in session.models_used if model_name.strip()}
    model_names = set(declared_names)
    model_names.update(message.model_name.strip() for message in session.messages if message.model_name)
    # NULL, not 'origin_reported': this is a skeleton placeholder for a
    # session's declared model before any pricing pass has run (typically
    # overwritten within this same call by _aggregate_message_tokens_into_
    # model_usage below, or later by a provider-usage-event rollup). It makes
    # no cost claim yet, so it must not carry a provenance string that
    # asserts one -- 'origin_reported' now means a genuine provider-reported
    # dollar figure (polylogue-shnc/polylogue-gt1z), which this row does not
    # have. ``declared`` records the parser's model declaration, the one
    # fact re-derivation cannot recover from messages or usage events; an
    # append only ever adds a declaration.
    model_usage_sql = (
        """
        INSERT OR REPLACE INTO session_model_usage (
            session_id, model_name, declared
        ) VALUES (?, ?, ?)
        """
        if replace_existing_model_rows
        else """
        INSERT INTO session_model_usage (
            session_id, model_name, declared
        ) VALUES (?, ?, ?)
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            declared = MAX(session_model_usage.declared, excluded.declared)
        """
    )
    stored_model_names = [cast(str, _sqlite_text(model_name)) for model_name in sorted(model_names)]
    for model_name, stored_model_name in zip(sorted(model_names), stored_model_names, strict=True):
        conn.execute(model_usage_sql, (session_id, stored_model_name, int(model_name in declared_names)))
    if aggregate_message_tokens:
        _aggregate_message_tokens_into_model_usage(conn, session_id)
    # After aggregation: the catalog dollars that weight the provider total's
    # split across this session's models are written by the pass above.
    if session.reported_cost_usd is not None:
        _write_provider_cost(
            conn,
            session_id,
            stored_model_names,
            ProviderCost(session.reported_cost_usd),
        )


def _message_usage_presence_sql(conn: sqlite3.Connection, session_id: str) -> tuple[str, str]:
    origin_row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    origin = str(origin_row[0]) if origin_row is not None else ""
    lane_names = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    present = " OR ".join(f"{name} IS NOT NULL" for name in lane_names)
    missing = " OR ".join(f"{lane_names[index]} IS NULL" for index in provider_usage_required_lane_indices(origin))
    # Identity-only messages do not assert usage. A partial measurement does.
    observed_sql = f"MAX(CASE WHEN {present} THEN 1 ELSE 0 END)"
    complete_sql = f"MIN(CASE WHEN ({present}) AND ({missing}) THEN 0 ELSE 1 END)"
    return observed_sql, complete_sql


def _aggregate_message_tokens_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Aggregate per-message token counts into session_model_usage and compute cost_usd.

    Called after messages are written (and after skeleton model-usage rows exist).
    Handles both full-write and merge-append paths: it always reads ALL messages
    currently in the DB for the session, so the token sums stay consistent with
    the full message set regardless of append ordering.

    Models with no messages carrying token data keep DEFAULT 0 token counts.
    Identity-only rows and incomplete measurements have no complete catalog
    price. A fully measured zero receives a zero catalog price when its model
    is priceable, while a model without a catalog entry remains unpriced.

    Empty or NULL model_name values in the messages table are excluded from
    aggregation (the model is unknown so pricing is impossible).

    The UPSERT only overwrites an existing row when the new message-walked
    token total is >= what is already stored (monotonic-safe), so a
    provider-usage-event cumulative rollup (``_upsert_provider_usage_model_rollup``, typically far larger
    for Codex since messages rarely carry its per-message usage) is never
    clobbered by a smaller/zero message-walk result on a later unrelated
    write. Before polylogue-shnc this was scoped by ``cost_provenance =
    'origin_reported'``, which stopped discriminating once provider-usage
    rollups started sharing the 'priced' label with real message-derived
    pricing (see ``_price_provider_usage_tokens``).
    """
    observed_sql, complete_sql = _message_usage_presence_sql(conn, session_id)
    # Aggregate token counts from the messages table for all known models.
    token_rows = conn.execute(
        f"""
        SELECT model_name,
               SUM(input_tokens)        AS sum_input,
               SUM(output_tokens)       AS sum_output,
               SUM(cache_read_tokens)   AS sum_cache_read,
               SUM(cache_write_tokens)  AS sum_cache_write,
               COUNT(*)                 AS msg_count,
               {observed_sql} AS provider_usage_observed,
               {complete_sql} AS provider_lanes_complete
        FROM messages
        WHERE session_id = ?
          AND model_name IS NOT NULL
          AND model_name != ''
        GROUP BY model_name
        """,
        (session_id,),
    ).fetchall()

    if not token_rows:
        return

    for row in token_rows:
        model_name: str = str(row[0])
        sum_input: int = int(row[1] or 0)
        sum_output: int = int(row[2] or 0)
        sum_cache_read: int = int(row[3] or 0)
        sum_cache_write: int = int(row[4] or 0)
        msg_count: int = int(row[5] or 0)
        observed = bool(row[6])
        complete = bool(row[7])

        catalog_cost = _price_provider_usage_tokens(
            conn,
            model_name,
            input_tokens=sum_input,
            output_tokens=sum_output,
            cache_read_tokens=sum_cache_read,
            cache_write_tokens=sum_cache_write,
        )

        if not complete:
            catalog_cost = None
        elif observed and sum_input + sum_output + sum_cache_read + sum_cache_write == 0:
            zero_cost, _ = catalog_cost_for_tokens(model_name, 0, 0, 0, 0)
            catalog_cost = None if zero_cost is None else CatalogCost(zero_cost)

        # UPSERT: the skeleton row was created by _seed_session_model_usage_rows above.
        # For models that somehow landed in messages but not in models_used/
        # session.messages (edge case with merge_append + partial data), we
        # INSERT a fresh row.  For normal cases this is an UPDATE on the
        # existing skeleton row.
        conn.execute(
            """
            INSERT INTO session_model_usage (
                session_id, model_name,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                message_count,
                catalog_cost_usd, provider_usage_observed, provider_lanes_complete
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, model_name) DO UPDATE SET
                input_tokens       = excluded.input_tokens,
                output_tokens      = excluded.output_tokens,
                cache_read_tokens  = excluded.cache_read_tokens,
                cache_write_tokens = excluded.cache_write_tokens,
                message_count      = excluded.message_count,
                catalog_cost_usd    = excluded.catalog_cost_usd,
                provider_usage_observed = excluded.provider_usage_observed,
                provider_lanes_complete = excluded.provider_lanes_complete
            WHERE (
                COALESCE(session_model_usage.input_tokens, 0)
                + COALESCE(session_model_usage.output_tokens, 0)
                + COALESCE(session_model_usage.cache_read_tokens, 0)
                + COALESCE(session_model_usage.cache_write_tokens, 0)
            ) <= (excluded.input_tokens + excluded.output_tokens + excluded.cache_read_tokens + excluded.cache_write_tokens)
            """,
            (
                session_id,
                model_name,
                sum_input,
                sum_output,
                sum_cache_read,
                sum_cache_write,
                msg_count,
                None if catalog_cost is None else catalog_cost.value,
                int(observed),
                int(complete),
            ),
        )


def _reconcile_session_model_usage_rows(conn: sqlite3.Connection, session_id: str) -> int:
    """Reset a session's usage rows to the evidence the aggregates re-derive from.

    Every row's measured values are cleared, so the message and provider-event
    aggregates that follow rebuild them from stored evidence alone: a
    session-global cumulative the latest event no longer attributes to a model
    cannot survive on that model's row. A row is then kept only when evidence
    still names its model: a message, a provider usage event, or the parser's
    declaration. A usage/cost correction that renames a message in place
    leaves the old message-only row without a source, so it goes. A declared
    row stays even with no message naming it -- it is the row an unnamed
    ``token_count`` is attributed to when it is the session's only model, and
    deleting it first would drop that event's tokens.

    ``declared`` is written by the session write that also writes this rollup,
    so it never moves without the rollup being rewritten in the same
    transaction. Returns the number of rows deleted.
    """
    conn.execute(
        """
        UPDATE session_model_usage
        SET input_tokens = 0,
            output_tokens = 0,
            cache_read_tokens = 0,
            cache_write_tokens = 0,
            message_count = 0,
            provider_cost_usd = NULL,
            catalog_cost_usd = NULL,
            provider_lanes_complete = 1,
            provider_usage_observed = 0
        WHERE session_id = ?
        """,
        (session_id,),
    )
    return conn.execute(
        """
        DELETE FROM session_model_usage
        WHERE session_id = ?
          AND declared = 0
          AND NOT EXISTS (
              SELECT 1
              FROM messages AS m
              WHERE m.session_id = session_model_usage.session_id
                AND m.model_name = session_model_usage.model_name
          )
          AND NOT EXISTS (
              SELECT 1
              FROM session_provider_usage_events AS e
              WHERE e.session_id = session_model_usage.session_id
                AND TRIM(COALESCE(e.model_name, '')) = session_model_usage.model_name
          )
        """,
        (session_id,),
    ).rowcount


def _write_repo_edges(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    update_session_observations: bool = True,
) -> None:
    observed_at_ms = to_epoch_ms(session.updated_at, numeric_unit="seconds") or to_epoch_ms(
        session.created_at, numeric_unit="seconds"
    )
    raw_root_paths = tuple(path for path in session.working_directories if path)
    origin_url = (session.git_repository_url or "").strip()
    # polylogue-cijx.4 decision 1: resolve each raw cwd to its git root before
    # deduplicating, so multiple cwds inside the same checkout (or a cwd
    # that is an agent-worktree subdirectory) collapse to one row instead of
    # one row per distinct raw path.
    #
    # polylogue-cijx.2 AC4: a session with no git evidence resolves to a
    # directory, not a repository. A raw cwd that resolves to no discoverable
    # git root is dropped here -- not kept as a "dir:<raw_path>" fallback --
    # unless an explicit remote (``origin_url``) is known, in which case
    # identity comes from the remote and the raw cwd is retained purely as a
    # representative checkout-root value. Without this, every session whose
    # cwd happens to be, say, ``/home/sinity`` would synthesize a "sinity"
    # repository from a bare directory with zero git evidence.
    resolved_root_paths: list[str] = []
    for raw_path in raw_root_paths:
        discovered_root = _discovered_repo_root_path(raw_path)
        if discovered_root is not None:
            resolved_root_paths.append(discovered_root)
        elif origin_url:
            resolved_root_paths.append(raw_path)
    root_paths = tuple(dict.fromkeys(resolved_root_paths))
    if not root_paths and not origin_url:
        # polylogue-1pzmq: no repo identity resolves -- but the parser may
        # still have been handed an explicit commit and branch. That is the
        # documented Codex shape on a checkout this machine no longer has (or
        # never had): branch and commit, no remote, and a cwd that resolves to
        # no git root here. Returning outright threw away commit evidence the
        # export stated. Record it with a NULL ``repo_id``: the column is
        # nullable exactly so a commit can be known without a repository, and
        # keying a synthetic repo on the commit hash alone would collide
        # across repositories that share it (a fork, a cherry-pick, an
        # imported subtree).
        if update_session_observations and session.git_commit_hash:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_commits (
                    session_id, commit_sha, repo_id, detection_type, method,
                    confidence, evidence_json, created_at_ms
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _sqlite_text(session.git_commit_hash),
                    "explicit_ref",
                    "parser-git-meta-unresolved-repo",
                    1.0,
                    _json_dumps(
                        {
                            "git_repository_url": None,
                            "root_path": None,
                            "git_branch": session.git_branch,
                            # The cwds the session declared, kept verbatim: they
                            # are the only handle on which checkout this was, and
                            # a later acquisition on a machine that has the repo
                            # can resolve them.
                            "unresolved_working_directories": list(raw_root_paths),
                        }
                    ),
                    observed_at_ms or 0,
                ),
            )
        return
    for root_path in root_paths or ("",):
        repo_name = _repo_name(origin_url, root_path)
        # polylogue-cijx.4 decision 1: identity is the canonicalized remote
        # (when known), NOT origin_url+root_path -- two worktree checkouts of
        # the same remote must upsert the SAME repos row. See
        # `repo_identity_key` for the full rationale.
        repo_id = repo_identity_key(origin_url, root_path)
        conn.execute(
            """
            INSERT INTO repos (repo_id, origin_url, root_path, repo_name, first_seen_at_ms, last_seen_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_id) DO UPDATE SET
                origin_url = COALESCE(NULLIF(repos.origin_url, ''), excluded.origin_url),
                root_path = COALESCE(NULLIF(repos.root_path, ''), excluded.root_path),
                repo_name = COALESCE(NULLIF(repos.repo_name, ''), excluded.repo_name),
                first_seen_at_ms = MIN(repos.first_seen_at_ms, excluded.first_seen_at_ms),
                last_seen_at_ms = MAX(repos.last_seen_at_ms, excluded.last_seen_at_ms)
            """,
            # repos.origin_url/root_path are now representative display
            # values, not identity (repo_id is) -- first-seen wins on
            # conflict rather than being overwritten by a later checkout, so
            # they stay stable once set. repos.repo_name is NOT NULL DEFAULT
            # ''. _repo_name() returns None when no name can be derived
            # (e.g. a session whose cwd is "/" or "."): insert the schema's
            # empty-string sentinel instead of NULL so the session is not
            # dropped, while the NULLIF above keeps a later re-ingest from
            # clobbering a previously-derived name with ''.
            (
                repo_id,
                _sqlite_text(origin_url),
                _sqlite_text(root_path),
                _sqlite_text(repo_name or ""),
                observed_at_ms or 0,
                observed_at_ms or 0,
            ),
        )
        conn.execute(
            """
            INSERT INTO repo_checkouts (repo_id, root_path, first_seen_at_ms, last_seen_at_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repo_id, root_path) DO UPDATE SET
                first_seen_at_ms = MIN(repo_checkouts.first_seen_at_ms, excluded.first_seen_at_ms),
                last_seen_at_ms = MAX(repo_checkouts.last_seen_at_ms, excluded.last_seen_at_ms)
            """,
            (repo_id, _sqlite_text(root_path), observed_at_ms or 0, observed_at_ms or 0),
        )
        if update_session_observations:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_repos (
                    session_id, repo_id, root_path, branch_name, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    repo_id,
                    _sqlite_text(root_path),
                    _sqlite_text(session.git_branch or ""),
                    observed_at_ms or 0,
                ),
            )
        if update_session_observations and session.git_commit_hash:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_commits (
                    session_id, commit_sha, repo_id, detection_type, method,
                    confidence, evidence_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _sqlite_text(session.git_commit_hash),
                    repo_id,
                    "explicit_ref",
                    "parser-git-meta",
                    1.0,
                    _json_dumps(
                        {
                            "git_repository_url": origin_url or None,
                            "root_path": root_path or None,
                            "git_branch": session.git_branch,
                        }
                    ),
                    observed_at_ms or 0,
                ),
            )


def _duplicate_message_coordinates(
    messages: Sequence[ParsedMessage],
    *,
    position_offset: int = 0,
) -> dict[tuple[int, int], list[str]]:
    """Group native message ids by effective (position, variant_index).

    Mirrors the exact position/variant_index fallback ``_write_messages``
    uses to build INSERT rows (``position_offset + (message.position or
    fallback_position)``, ``message.variant_index or 0``), so this reports
    precisely the coordinate pairs that would collide against the messages
    table's ``PRIMARY KEY(session_id, position, variant_index)``. Only
    coordinates shared by two or more messages are returned.
    """
    coordinates: dict[tuple[int, int], list[str]] = defaultdict(list)
    for fallback_position, message in enumerate(messages):
        position = position_offset + (message.position if message.position is not None else fallback_position)
        variant_index = message.variant_index if message.variant_index is not None else 0
        coordinates[(position, variant_index)].append(message.provider_message_id or "<no native id>")
    return {key: native_ids for key, native_ids in coordinates.items() if len(native_ids) > 1}


def _assert_unique_message_coordinates(
    session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    position_offset: int = 0,
) -> None:
    """Fail loudly, before any row is written, on a (position, variant_index) collision.

    ``messages`` is unique on exactly ``(session_id, position, variant_index)``
    (see ``storage/sqlite/archive_tiers/index.py``). If a parser bug assigns
    that same pair to two distinct native message ids, ``INSERT OR REPLACE``
    silently drops one message row while ``_write_blocks`` still computes
    distinct Python-side ``message_id`` values for both -- the dropped
    message's blocks then fail their foreign key days later, far from the
    actual bug. Raising here turns that into an immediate, loud error naming
    the session and the exact colliding native ids instead.
    """
    if isinstance(messages, (SqliteMessageSink, _MessageTail)):
        source = messages.messages if isinstance(messages, _MessageTail) else messages
        if isinstance(source, SqliteMessageSink):
            from polylogue.storage.sqlite.connection_profile import scratch_connection_context

            with scratch_connection_context(
                prefix="polylogue-coordinates-", filename="coordinates.db", directory=source.path.parent
            ) as index:
                index.execute(
                    "CREATE TABLE coordinate (position INTEGER NOT NULL, variant_index INTEGER NOT NULL, "
                    "native_id TEXT NOT NULL, PRIMARY KEY(position, variant_index)) WITHOUT ROWID"
                )
                for fallback_position, message in enumerate(messages):
                    position = position_offset + (
                        message.position if message.position is not None else fallback_position
                    )
                    variant_index = message.variant_index if message.variant_index is not None else 0
                    native_id = message.provider_message_id or "<no native id>"
                    try:
                        index.execute(
                            "INSERT INTO coordinate VALUES (?, ?, ?)",
                            (position, variant_index, source_native_id_json(native_id)),
                        )
                    except sqlite3.IntegrityError:
                        previous = index.execute(
                            "SELECT native_id FROM coordinate WHERE position = ? AND variant_index = ?",
                            (position, variant_index),
                        ).fetchone()
                        raise ValueError(
                            f"duplicate message coordinates in session {session_id!r}: "
                            f"(position={position}, variant_index={variant_index}) <- "
                            f"{[source_native_id_from_json(previous[0]), native_id]!r}. A parser assigned the same "
                            "(position, variant_index) pair to distinct native message ids; "
                            "writing this batch would silently drop one message."
                        ) from None
            return
    duplicates = _duplicate_message_coordinates(messages, position_offset=position_offset)
    if not duplicates:
        return
    detail = "; ".join(
        f"(position={position}, variant_index={variant_index}) <- {native_ids!r}"
        for (position, variant_index), native_ids in sorted(duplicates.items())
    )
    raise ValueError(
        f"duplicate message coordinates in session {session_id!r}: {detail}. "
        "A parser assigned the same (position, variant_index) pair to distinct "
        "native message ids; writing this batch would silently drop one "
        "message (and orphan its blocks) via INSERT OR REPLACE."
    )


def _message_blocks(message: ParsedMessage) -> Sequence[ParsedContentBlock]:
    if message.blocks:
        return message.blocks
    if message.text:
        return (ParsedContentBlock(type=BlockType.TEXT, text=message.text),)
    return ()


# --- Lineage normalization (#2467): prefix-inheritance tail extraction ---------
#
# A fork / resume / spawned subagent / auto-compaction child rollout physically
# copies the parent's context as a leading prefix. We store only the child's
# divergent tail plus a lineage edge with a branch point, so each real message is
# stored exactly once. The branch point is found by conservative contiguous
# prefix-alignment against the parent's *composed* transcript, using a per-message
# content signature (role + ordered block content). A message is treated as
# inherited only inside the matching leading run, so a genuinely-new block that
# happens to equal a parent block is never dropped.


def _prefix_alignment_signature(role: str, block_hashes: Iterable[bytes]) -> str:
    """Digest what a replayed prefix copies: the role and each block's evidence.

    A child that replays its parent's prefix (a fork, a resume, a compaction
    continuation) re-emits those messages under new provider ids, timestamps,
    parent links and provenance classification, all of which the complete
    semantic address hashes. Alignment compares the role and each block's
    identity-free evidence digest (``blocks.content_hash``: text, tool call,
    outcome, error and exit evidence), so a divergent answer still ends the
    shared prefix. The branch point itself stays guarded by the parent's
    stored complete content address (``branch_point_content_address``).
    """
    digest = hashlib.sha256(b"polylogue-prefix-alignment-v2\0")
    encoded_role = role.encode("utf-8", "surrogatepass")
    digest.update(len(encoded_role).to_bytes(8, "big"))
    digest.update(encoded_role)
    for block_hash in block_hashes:
        digest.update(len(block_hash).to_bytes(8, "big"))
        digest.update(block_hash)
    return digest.hexdigest()


def _parsed_block_evidence_hash(block: ParsedContentBlock) -> bytes:
    """The stored ``blocks.content_hash`` this block will be written with."""
    return _block_content_hash(
        block_type=_block_type(block).value,
        text=block.text,
        tool_name=block.tool_name,
        tool_input_json=_json_dumps(block.tool_input) if block.tool_input is not None else None,
        semantic_type=_semantic_type(block),
        media_type=block.media_type,
        language=_block_language(block),
        is_error=getattr(block, "is_error", None),
        exit_code=getattr(block, "exit_code", None),
        tool_outcome=getattr(block, "tool_outcome", None),
        outcome_unknown_reason=_enum_value(block.outcome_unknown_reason),
        semantic_extra_json=_block_semantic_extra_json(block),
    )


def _parsed_message_signature(message: ParsedMessage) -> str:
    """The prefix alignment signature of a parsed message (see above)."""
    return _prefix_alignment_signature(
        _enum_value(message.role) or "",
        (_parsed_block_evidence_hash(block) for block in _message_blocks(message)),
    )


def _signatures_from_block_rows(rows: Iterable[Sequence[object]]) -> Iterator[tuple[str, str]]:
    """Group ``(message_id, role, block content_hash)`` rows per message."""
    current_id: str | None = None
    current_role = ""
    hashes: list[bytes] = []
    for message_id, role, block_hash in rows:
        if message_id != current_id:
            if current_id is not None:
                yield current_id, _prefix_alignment_signature(current_role, hashes)
            current_id, current_role, hashes = str(message_id), str(role or ""), []
        if block_hash is not None:
            hashes.append(bytes(cast(bytes, block_hash)))
    if current_id is not None:
        yield current_id, _prefix_alignment_signature(current_role, hashes)


def _is_acompact_native_id(native_id: str) -> bool:
    return native_id.rsplit(":", 1)[-1].startswith("agent-acompact-")


def _is_claude_code_acompact_session(session: ParsedSession) -> bool:
    return session.source_name is Provider.CLAUDE_CODE and _is_acompact_native_id(session.provider_session_id)


def _parsed_acompact_prefix_signatures(messages: Sequence[ParsedMessage]) -> list[str]:
    """Return content signatures before the compaction summary boundary.

    The summary is expected to be unique output even for a true main-session
    compactor, so it cannot count against parent membership.  Later records are
    outside the copied prefix and are likewise excluded once a summary appears.
    """
    return list(_iter_parsed_acompact_prefix_signatures(messages))


def _iter_parsed_acompact_prefix_signatures(messages: Sequence[ParsedMessage]) -> Iterator[str]:
    for message in messages:
        if message.message_type is MessageType.SUMMARY:
            break
        yield _parsed_message_signature(message)


def _acompact_content_membership_ratio(
    parent_composed: Sequence[tuple[str, str]],
    child_prefix_signatures: Iterable[str],
) -> float | None:
    """Return multiset content membership of an acompact prefix in its parent.

    Classification uses membership rather than contiguous alignment: the former
    answers whether this artifact belongs to the asserted parent at all, while
    `_extract_prefix_tail` remains the stricter loss-prevention gate for deleting
    inherited rows.  Duplicate signatures are bounded by parent multiplicity so
    repeated boilerplate cannot manufacture overlap.
    """
    if isinstance(parent_composed, _DiskSignatureSequence):
        with parent_composed.reader() as counts:
            counts.execute(
                "CREATE TEMP TABLE IF NOT EXISTS membership (digest TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID"
            )
            counts.execute("DELETE FROM membership")
            for _message_id, signature in parent_composed:
                counts.execute(
                    "INSERT INTO membership VALUES (?, 1) ON CONFLICT(digest) DO UPDATE SET n = n + 1",
                    (signature,),
                )
            matching_count = 0
            total_count = 0
            for signature in child_prefix_signatures:
                total_count += 1
                row = counts.execute("SELECT n FROM membership WHERE digest = ?", (signature,)).fetchone()
                if row is not None and int(row[0]) > 0:
                    matching_count += 1
                    counts.execute("UPDATE membership SET n = n - 1 WHERE digest = ?", (signature,))
            return matching_count / total_count if total_count else None
    prefix = tuple(child_prefix_signatures)
    if not prefix:
        return None
    if not parent_composed:
        return 0.0
    parent_counts = Counter(signature for _message_id, signature in parent_composed)
    matching_count = 0
    for signature in prefix:
        if parent_counts[signature] <= 0:
            continue
        parent_counts[signature] -= 1
        matching_count += 1
    return matching_count / len(prefix)


def _db_acompact_prefix_signatures(
    conn: sqlite3.Connection,
    session_id: str,
    child_composed: Sequence[tuple[str, str]],
) -> list[str]:
    summary_message_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND message_type = 'summary'",
            (session_id,),
        ).fetchall()
    }
    signatures: list[str] = []
    for message_id, signature in child_composed:
        if message_id in summary_message_ids:
            break
        signatures.append(signature)
    return signatures


def _db_claude_acompact_branch_type(conn: sqlite3.Connection, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT origin, native_id, branch_type FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    origin, native_id, branch_type = row
    if origin != origin_from_provider(Provider.CLAUDE_CODE).value or not _is_acompact_native_id(str(native_id)):
        return None
    return str(branch_type) if branch_type is not None else ""


def _own_db_signatures(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> list[tuple[str, str]]:
    """Return prefix alignment signatures for this session's own stored rows."""
    if before_input is not None:
        before_input(
            "messages",
            ("message_id", "role", "position", "variant_index"),
            "SELECT rowid FROM messages WHERE session_id=? ORDER BY position,variant_index",
            (session_id,),
        )
        before_input(
            "blocks",
            ("message_id", "position", "content_hash"),
            "SELECT rowid FROM blocks WHERE session_id=? ORDER BY message_id,position",
            (session_id,),
        )
    with connection_cursor(
        conn,
        """
        SELECT m.message_id, m.role, b.content_hash
        FROM messages m
        LEFT JOIN blocks b ON b.session_id = m.session_id AND b.message_id = m.message_id
        WHERE m.session_id = ?
        ORDER BY m.position, m.variant_index, b.position
        """,
        (session_id,),
    ) as cursor:
        return list(_signatures_from_block_rows(cursor))


def _iter_own_db_signatures(
    conn: sqlite3.Connection,
    segment: _TranscriptSegment,
    before_input: BeforeIndexInput | None = None,
) -> Generator[tuple[str, str], None, None]:
    segment_parameters = (
        segment.session_id,
        segment.upto_position,
        segment.upto_position,
        segment.upto_position,
        segment.upto_variant_index,
    )
    if before_input is not None:
        before_input(
            "messages",
            ("message_id", "role", "position", "variant_index"),
            "SELECT rowid FROM messages WHERE session_id=? "
            "AND (? IS NULL OR position < ? OR (position = ? AND variant_index <= ?)) "
            "ORDER BY position,variant_index",
            segment_parameters,
        )
        before_input(
            "blocks",
            ("message_id", "position", "content_hash"),
            "SELECT rowid FROM blocks WHERE session_id=? ORDER BY message_id,position",
            (segment.session_id,),
        )
    with connection_cursor(
        conn,
        """
        SELECT m.message_id, m.role, b.content_hash
        FROM messages m
        LEFT JOIN blocks b ON b.session_id = m.session_id AND b.message_id = m.message_id
        WHERE m.session_id = ?
          AND (? IS NULL OR m.position < ? OR (m.position = ? AND m.variant_index <= ?))
        ORDER BY m.position, m.variant_index, b.position
        """,
        segment_parameters,
    ) as cursor:
        yield from _signatures_from_block_rows(cursor)


class _DiskSignatureSequence(Sequence[tuple[str, str]]):
    def __init__(self, directory: Path) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-lineage-signatures-", dir=directory)
        self._lock = threading.RLock()
        self._closed = False
        self._path = Path(self._scratch.name) / "signatures.db"
        self._reader_owner: NativeSQLCustodyOwner | None = None
        self._writer: NativeSQLCustodyOwner | None = None
        try:
            self._writer = open_scratch_connection(self._path, lifetime_dependencies=(self,))
        except NativeConnectionSettlementError as failure:
            failure.owner.scratch_directory = self._scratch
            raise
        assert self._writer.connection is not None
        self._conn: sqlite3.Connection | None = self._writer.connection
        try:
            self._connection().execute(
                "CREATE TABLE signature (ordinal INTEGER PRIMARY KEY, message_id TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            self._count = 0
        except BaseException as primary:
            self._writer.scratch_directory = self._scratch
            _close_failed_native_construction(self._writer, primary)
            self.close()
            raise

    def append(self, message_id: str, digest: str) -> None:
        self._connection().execute("INSERT INTO signature VALUES (?, ?, ?)", (self._count, message_id, digest))
        self._count += 1

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> tuple[str, str]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[str, str]]: ...

    def __getitem__(self, index: int | slice) -> tuple[str, str] | list[tuple[str, str]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        with self.reader() as conn:
            row = conn.execute("SELECT message_id, digest FROM signature WHERE ordinal = ?", (ordinal,)).fetchone()
        if row is None:
            raise PreparedSessionWriteRefusedError("lineage signature disappeared")
        return str(row[0]), str(row[1])

    def __iter__(self) -> Iterator[tuple[str, str]]:
        after = -1
        while True:
            with self.reader() as conn:
                rows = conn.execute(
                    "SELECT ordinal, message_id, digest FROM signature WHERE ordinal > ? ORDER BY ordinal LIMIT 512",
                    (after,),
                ).fetchall()
            if not rows:
                return
            after = int(rows[-1][0])
            yield from ((str(row[1]), str(row[2])) for row in rows)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("signature SQL requires its creator-owned reader")
        owner = self._writer if self._writer is not None else self._reader_owner
        if owner is None:
            raise RuntimeError("artifact reader has no native owner")
        return owner.require_connection()

    def finish(self) -> None:
        self._connection().commit()
        assert self._writer is not None
        self._writer.close()
        self._writer = None
        self._conn = None

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("signature artifact is closed")
            if self._conn is not None:
                yield self._connection()
                return
            conn = connect_measured(f"{self._path.as_uri()}?mode=ro", uri=True)
            owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=(*current_native_sql_lifetimes(), self))
            self._reader_owner = owner
            self._conn = conn
            try:
                yield owner.require_connection()
            finally:
                self._conn = None
                self._reader_owner = None
                owner.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._writer is not None:
                if self._writer in retained_native_sql_owners_for_lifetime(self):
                    self._writer.close()
                self._writer = None
                self._conn = None
            pending = retained_native_sql_owners_for_lifetime(self)
            if pending:
                raise NativeConnectionSettlementError(pending[0], RuntimeError("signature artifact remains owned"))
            self._scratch.cleanup()
            self._closed = True

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


def _iter_composed_rows(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> Generator[tuple[str, str, str], None, None]:
    """``(message_id, signature, owner)`` of a composed transcript, streamed segment by segment.

    The composition rules are ``_composed_db_signatures``': the first
    composing prefix-sharing edge, the branch-point witness, the visited set
    and the dangling-branch-point rule. Only the segment plan is held.
    """
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            with closing(_iter_composed_rows(conn, session_id, before_input)) as rows:
                yield from rows
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass
        return
    chain: list[tuple[str, str]] = []
    visited = {session_id}
    cursor_session_id = session_id
    # ``visited`` bounds the walk: every step adds a new session.
    while True:
        edge = _prefix_sharing_edge_row(conn, cursor_session_id, before_input)
        if edge is None:
            break
        parent_id, branch_point = str(edge[0]), str(edge[1])
        witness = bytes(edge[2]) if edge[2] is not None else None
        if witness is not None and _message_content_address_for_id(conn, branch_point, before_input) != witness:
            break
        if parent_id in visited:
            break
        chain.append((cursor_session_id, branch_point))
        visited.add(parent_id)
        cursor_session_id = parent_id
    composed = _SegmentList(
        conn,
        (_TranscriptSegment(cursor_session_id, None, None, _count_session_messages(conn, cursor_session_id)),),
        before_input,
    )
    for child_id, branch_point in reversed(chain):
        if not composed.cut_at(branch_point):
            composed.clear()
        composed.append(_TranscriptSegment(child_id, None, None, _count_session_messages(conn, child_id)))
    for segment in list(composed.segments):
        with closing(_iter_own_db_signatures(conn, segment, before_input)) as signatures:
            for message_id, digest in signatures:
                yield message_id, digest, segment.session_id


def _disk_composed_db_signatures(
    conn: sqlite3.Connection,
    session_id: str,
    directory: Path,
    *,
    before_input: BeforeIndexInput | None = None,
    prefix_length: int | None = None,
) -> _DiskSignatureSequence:
    """Compose signatures, optionally only the declared inherited prefix.

    The current composition plan and branch witnesses are still resolved in
    full. Only signature consumption and scratch rows stop at the prefix.
    """
    opened_snapshot = not conn.in_transaction
    if opened_snapshot:
        with connection_cursor(conn, "BEGIN DEFERRED") as _input_cursor:
            pass
    result: _DiskSignatureSequence | None = None
    try:
        result = _DiskSignatureSequence(directory)
        with closing(_iter_composed_rows(conn, session_id, before_input)) as rows:
            for message_id, digest, _owner in islice(rows, prefix_length):
                result.append(message_id, digest)
        result.finish()
        return result
    except BaseException as primary:
        if result is not None:
            try:
                result.close()
            except BaseException as cleanup:
                raise cleanup from primary
        raise
    finally:
        if opened_snapshot:
            with connection_cursor(conn, "ROLLBACK") as _input_cursor:
                pass


def _signature_cache_get(
    cache: _SignatureCacheLike | None,
    session_id: str,
) -> list[tuple[str, str]] | None:
    if cache is None:
        return None
    return cache.get(session_id)


def _signature_cache_set(
    cache: _SignatureCacheLike | None,
    session_id: str,
    signatures: list[tuple[str, str]],
) -> None:
    if cache is not None:
        cache[session_id] = signatures


def _signature_cache_get_composed(
    cache: _SignatureCacheLike | None,
    session_id: str,
) -> list[tuple[str, str]] | None:
    if isinstance(cache, LineageSignatureCache):
        return cache.get_composed(session_id)
    return None


def _signature_cache_set_composed(
    cache: _SignatureCacheLike | None,
    session_id: str,
    signatures: list[tuple[str, str]],
    *,
    dependencies: frozenset[str],
) -> None:
    if isinstance(cache, LineageSignatureCache):
        cache.set_composed(session_id, signatures, dependencies=dependencies)


def _prefix_lineage_closure(
    conn: sqlite3.Connection,
    session_id: str,
    cache: _SignatureCacheLike | None,
    before_input: BeforeIndexInput | None = None,
) -> frozenset[str]:
    """``session_id`` and every ancestor its composed transcript can draw from.

    A resident composed entry already records this closure. Otherwise the
    prefix-sharing edges are walked without reading any signatures; the walk
    ignores staleness witnesses, so it may name an ancestor the composition
    stopped short of, which only widens invalidation.
    """
    if isinstance(cache, LineageSignatureCache):
        known = cache.composed_dependencies(session_id)
        if known is not None:
            return known
    closure = {session_id}
    cursor = session_id
    while (edge := _prefix_sharing_edge_sync(conn, cursor, before_input)) is not None and edge[0] not in closure:
        cursor = edge[0]
        closure.add(cursor)
    return frozenset(closure)


def _composed_db_signatures(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    cache: _SignatureCacheLike | None = None,
    composed_cache: dict[str, list[tuple[str, str]]] | None = None,
    before_input: BeforeIndexInput | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(message_id, signature), ...]`` for ``session_id``'s composed
    transcript (its inherited prefix + own tail). Walk the lineage iteratively
    and compose root-to-leaf, mirroring the async read path without inheriting
    the synchronous envelope reader's Python-stack limit.

    When ``cache`` is supplied, each session's OWN signatures are memoized by
    ``session_id`` for the life of one ingest batch. A
    :class:`LineageSignatureCache` additionally carries composed
    (prefix+own) results with weighted recency and dependency invalidation.
    ``composed_cache`` remains the narrow operation-local compatibility path
    used by graph repair callers that pass a plain dict.

    Callers typically already hold a write transaction (single-writer ingest),
    but if ``conn`` is not already inside one this wraps the whole recursive
    composition in one deferred read transaction (4ts.4), matching
    ``read_archive_session_envelope``'s guard against a torn read.
    """
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED") as _input_cursor:
            pass
        try:
            return _composed_db_signatures(
                conn, session_id, cache=cache, composed_cache=composed_cache, before_input=before_input
            )
        finally:
            with connection_cursor(conn, "ROLLBACK") as _input_cursor:
                pass

    def own_signatures(target_session_id: str) -> list[tuple[str, str]]:
        own = _signature_cache_get(cache, target_session_id)
        if own is None:
            own = _own_db_signatures(conn, target_session_id, before_input)
            _signature_cache_set(cache, target_session_id, own)
        return own

    # Only a LineageSignatureCache keeps dependencies, so only it pays for
    # the closure walks below.
    tracks_dependencies = isinstance(cache, LineageSignatureCache)

    # Collect (child, branch point, child-owned rows) leaf-first, then compose
    # from the oldest reached ancestor down. The visited set is the cycle guard
    # and bounds the walk: every step adds a new session. ``dependencies`` ends
    # as the requested session's full ancestor closure: the sessions walked
    # plus the closure of wherever the walk stopped.
    chain: list[tuple[str, str, list[tuple[str, str]]]] = []
    visited = {session_id}
    dependencies = {session_id}
    cursor_session_id = session_id
    composed: list[tuple[str, str]]
    while True:
        cached_composed = composed_cache.get(cursor_session_id) if composed_cache is not None else None
        if cached_composed is None:
            cached_composed = _signature_cache_get_composed(cache, cursor_session_id)
        if cached_composed is not None:
            composed = cached_composed
            if tracks_dependencies:
                dependencies |= _prefix_lineage_closure(conn, cursor_session_id, cache, before_input)
            break
        own = own_signatures(cursor_session_id)
        edge = _prefix_sharing_edge_row(conn, cursor_session_id, before_input)
        if edge is None:
            composed = own
            if composed_cache is not None:
                composed_cache[cursor_session_id] = composed
            _signature_cache_set_composed(cache, cursor_session_id, composed, dependencies=frozenset(dependencies))
            break
        parent_id, branch_point_message_id = str(edge[0]), str(edge[1])
        witness = None if edge[2] is None else bytes(edge[2])
        if witness is not None:
            current = _message_content_address_for_id(conn, branch_point_message_id, before_input)
            if current is None or current != witness:
                composed = own
                # The refusal holds only while the branch point's content
                # differs from the witness, and that row belongs to the parent
                # or one of its ancestors: a rewrite of any of them can restore
                # the match.
                if tracks_dependencies:
                    dependencies |= _prefix_lineage_closure(conn, parent_id, cache, before_input)
                if composed_cache is not None:
                    composed_cache[cursor_session_id] = composed
                _signature_cache_set_composed(cache, cursor_session_id, composed, dependencies=frozenset(dependencies))
                break
        if parent_id in visited:
            composed = own
            if composed_cache is not None:
                composed_cache[cursor_session_id] = composed
            _signature_cache_set_composed(cache, cursor_session_id, composed, dependencies=frozenset(dependencies))
            break
        chain.append((cursor_session_id, branch_point_message_id, own))
        visited.add(parent_id)
        dependencies.add(parent_id)
        cursor_session_id = parent_id
    if not chain:
        return composed
    # One list is cut at each branch point and extended by each tail, with a
    # first-position index, so a chain composes in time linear in the rows it
    # touches; only the requested session's result is cached, since keeping
    # every intermediate transcript would be quadratic. The start may be a
    # cached result, so it is copied, never cut in place.
    composed = list(composed)
    position: dict[str, int] = {}
    for index, (message_id, _signature) in enumerate(composed):
        position.setdefault(message_id, index)
    for _child_session_id, branch_point_message_id, own in reversed(chain):
        at = position.get(branch_point_message_id)
        if at is None:
            # A genuinely missing branch point is not a license to inherit a
            # nearby or entire prefix. This matches the async reader's rule.
            composed, position = [], {}
        else:
            for message_id, _signature in composed[at + 1 :]:
                if position.get(message_id, -1) > at:
                    del position[message_id]
            del composed[at + 1 :]
        for entry in own:
            position.setdefault(entry[0], len(composed))
            composed.append(entry)
    if composed_cache is not None:
        composed_cache[session_id] = composed
    _signature_cache_set_composed(cache, session_id, composed, dependencies=frozenset(dependencies))
    return composed


@contextmanager
def _bulk_fts_session_guard(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    enabled: bool,
    bulk_build: bool = False,
) -> Iterator[None]:
    """Suspend per-row ``messages_fts`` trigger maintenance for one session.

    polylogue-crd8: whale prefix-sharing lineage sessions (fork/resume/
    auto-compaction) can carry 10K+ tool blocks in the inherited prefix that
    ``_reextract_prefix_tail_db`` deletes wholesale once the parent is known.
    Deleting those blocks one row at a time fires ``messages_fts_ad`` once per
    row (contentless-FTS posting-list maintenance), which measured as a
    25+ minute single-DELETE stall on the live rebuild.

    While ``enabled``, this sets a **dedicated** ``derived_refresh_guard`` row
    (``fts-bulk-session-write``) that gates the ``messages_fts_{ai,ad,au}``
    trigger BODIES (see ``polylogue.storage.fts.sql``) -- so that surface
    gets the explicit session-scoped delete/re-insert bracketing below; the
    triggers stay
    structurally present in ``sqlite_master`` throughout; only their WHEN
    clause short-circuits. This is deliberately a *different* guard name from
    the existing ``session-write`` guard: that guard is set unconditionally
    for every session write (see ``write_parsed_session_to_archive``), so
    gating FTS maintenance on it would silently stop FTS indexing for
    ordinary daemon ingest, not just whale bulk replays.

    The caller's block deletion is bracketed by one explicit session-scoped
    FTS delete (covering both the doomed prefix rows and the surviving tail
    rows) and, in the ``finally``, one explicit session-scoped FTS re-insert
    (repopulating whatever blocks remain for ``session_id`` -- the surviving
    tail after the caller's mutation). This mirrors the same delete-then-
    insert shape ``_replace_full_session_messages_and_blocks`` already uses
    for a session's own full replace, just gated by a guard row instead of a
    raw ``DROP TRIGGER``/``CREATE TRIGGER`` pair, so the trigger-presence half
    of ``assert_session_fts_exact_sync`` never observes a trigger-less window.

    ``bulk_build`` (polylogue-v6i3, default ``False``): when the caller is
    inside a bulk-generation-build session write, ``write_parsed_session_to_
    archive`` already set the same dedicated guard row for the whole
    transaction (every block mutation across the entire session write, not
    just this dependent-delete) and owns clearing it at the end. This
    context manager becomes a plain no-op in that case -- it must not
    re-insert/re-delete the same row (that would prematurely clear the
    guard for the remainder of the outer write) and must not perform the
    explicit delete-then-reinsert either: the bulk-build lifecycle leaves
    ``messages_fts`` empty throughout replay and repopulates it archive-wide
    exactly once at readiness, so any per-session insert here would just be
    redone work.
    """
    if bulk_build:
        yield
        return
    if not enabled:
        yield
        return
    conn.execute(delete_session_rows_sql(1), (session_id,))
    # polylogue-miwv: identity-ledger companion, same chunk params as the
    # messages_fts delete above.
    conn.execute(delete_session_identity_rows_sql(1), (session_id,))
    conn.execute(
        "INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES (?)",
        (FTS_BULK_SESSION_WRITE_GUARD,),
    )
    try:
        yield
    finally:
        conn.execute(insert_session_rows_sql(1), (session_id,))
        # polylogue-miwv: identity-ledger companion, same chunk params as the
        # messages_fts insert above.
        conn.execute(insert_session_identity_rows_sql(1), (session_id,))
        conn.execute(
            "DELETE FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        )


_REEXTRACTED_PREFIX_BLOCK_SINK: ContextVar[Callable[[str, int, str], None] | None] = ContextVar(
    "polylogue_reextracted_prefix_block_sink", default=None
)


@contextmanager
def report_reextracted_prefix_blocks(sink: Callable[[str, int, str], None]) -> Iterator[None]:
    """Stream text blocks a late parent removes from an earlier child's rows to ``sink``.

    A child written before its parent was stored whole, so its accepted marker
    carrier already holds candidates for the replayed prefix under the
    child's own message ids. When this write re-extracts the child to its
    tail, those blocks' canonical owner becomes the parent. Each removed
    ``(message_id, position, text)`` row is handed to ``sink`` as it is read,
    so the caller retains only what it derives, never the prefix text.
    """
    previous = _REEXTRACTED_PREFIX_BLOCK_SINK.set(sink)
    try:
        yield
    finally:
        _REEXTRACTED_PREFIX_BLOCK_SINK.reset(previous)


def _reextract_prefix_tail_db(
    conn: sqlite3.Connection,
    child_session_id: str,
    parent_session_id: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    composed_cache: dict[str, list[tuple[str, str]]] | None = None,
    add_timing: Callable[[str, float], None] | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
) -> set[str]:
    """Normalize a child that was stored whole because its parent was ingested
    later (#2467). Aligns the child's already-stored messages against the parent's
    composed transcript, deletes the inherited-prefix rows, and records the edge.
    Only runs while the lineage edge is still un-extracted (``inheritance`` NULL).

    Returns the set of *other* sessions whose prefix-sharing edges named one of
    the deleted inherited-prefix rows as their branch point (polylogue-7xrv5).
    In a three-generation lineage ``P -> B -> A`` visited as ``A, B, P``, A's
    edge to B binds to a B-owned message while B is still stored whole; this
    call deletes exactly those rows, so A's branch point dangles and A composes
    to its own tail. The caller must add these ids to the repair scope --
    neither ``session_id`` nor ``resolved_child_ids`` contains A.
    """

    def record_substage(name: str, started_at: float) -> None:
        if add_timing is not None:
            add_timing(f"index.graph_resolve.reextract_prefix_tails.{name}", started_at)

    t0 = time.perf_counter()
    edge = conn.execute(
        """
        SELECT dst_origin, dst_native_id, link_type
        FROM session_links
        WHERE src_session_id = ?
          AND resolved_dst_session_id = ?
          AND inheritance IS NULL
        LIMIT 1
        """,
        (child_session_id, parent_session_id),
    ).fetchone()
    record_substage("edge_lookup", t0)
    if edge is None:
        return set()
    dst_origin, dst_native_id, link_type = edge
    # This child is a stored tail, not the full replay that deferred extraction
    # expects. Its recorded loss remains owned by retained child replay.
    if _has_invalidated_prefix(conn, child_session_id):
        return set()
    # A Drive ``branchParent.promptId`` is a source-asserted session edge, not
    # evidence that the child replays the parent's message prefix.  The child
    # may have arrived before its parent, so the normal deferred extraction
    # path must not manufacture a branch point from coincidental matching
    # content once the parent becomes available.  The parser-side writer
    # records this typed unresolved state in the edge evidence; preserve it
    # across parent-first and child-first replay alike.
    unresolved_drive_branch = conn.execute(
        """
        SELECT 1
        FROM session_links
        WHERE src_session_id = ?
          AND dst_origin = ?
          AND dst_native_id = ?
          AND link_type = ?
          AND json_valid(evidence_json)
          AND json_extract(evidence_json, '$.branch_point_resolution') =
              'unresolved-source-no-local-message-id'
        LIMIT 1
        """,
        (child_session_id, dst_origin, dst_native_id, link_type),
    ).fetchone()
    if unresolved_drive_branch is not None:
        record_substage("deferred_source_branch_unresolved", time.perf_counter())
        return set()
    t0 = time.perf_counter()
    parent_composed = _composed_db_signatures(
        conn,
        parent_session_id,
        cache=cache,
        composed_cache=composed_cache,
    )
    record_substage("parent_composed", t0)
    t0 = time.perf_counter()
    child_composed = _composed_db_signatures(
        conn,
        child_session_id,
        cache=cache,
        composed_cache=composed_cache,
    )
    record_substage("child_composed", t0)

    def _set_edge(
        branch_point_message_id: str | None,
        inheritance: str,
        *,
        branch_point_content_address: bytes | None = None,
        next_link_type: str | None = None,
    ) -> None:
        current_link_type = str(link_type)
        target_link_type = next_link_type or current_link_type
        if target_link_type != current_link_type:
            # One parsed parent assertion should yield one edge. Remove a stale
            # duplicate of the target natural key before changing the PK lane so
            # a rebuild remains convergent even after interrupted older repairs.
            conn.execute(
                """
                DELETE FROM session_links
                WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
                """,
                (child_session_id, dst_origin, dst_native_id, target_link_type),
            )
        conn.execute(
            """
            UPDATE session_links
            SET link_type = ?, branch_point_message_id = ?, inheritance = ?
                , branch_point_content_address = ?
            WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
            """,
            (
                target_link_type,
                branch_point_message_id,
                inheritance,
                branch_point_content_address,
                child_session_id,
                dst_origin,
                dst_native_id,
                current_link_type,
            ),
        )

    t0 = time.perf_counter()
    acompact_branch_type = _db_claude_acompact_branch_type(conn, child_session_id)
    force_spawned_fresh = False
    resolved_link_type: str | None = None
    if acompact_branch_type is not None:
        membership = _acompact_content_membership_ratio(
            parent_composed,
            _db_acompact_prefix_signatures(conn, child_session_id, child_composed),
        )
        if membership is not None:
            if membership < _ACOMPACT_PARENT_MEMBERSHIP_THRESHOLD:
                force_spawned_fresh = True
            else:
                # The parser's fresh-head signal is intentionally conservative.
                # Once the parent exists, content membership is authoritative.
                resolved_link_type = TopologyEdgeType.CONTINUATION.value
                conn.execute(
                    "UPDATE sessions SET branch_type = ? WHERE session_id = ?",
                    (BranchType.CONTINUATION.value, child_session_id),
                )
        elif acompact_branch_type == BranchType.SIDECHAIN.value:
            force_spawned_fresh = True
    record_substage("acompact_membership", t0)
    if force_spawned_fresh:
        t0 = time.perf_counter()
        _set_edge(None, "spawned-fresh", next_link_type=TopologyEdgeType.SIDECHAIN.value)
        conn.execute(
            "UPDATE sessions SET branch_type = ? WHERE session_id = ?",
            (BranchType.SIDECHAIN.value, child_session_id),
        )
        record_substage("acompact_reclassify", t0)
        return set()

    t0 = time.perf_counter()
    k = 0
    limit = min(len(parent_composed), len(child_composed))
    while k < limit and parent_composed[k][1] == child_composed[k][1]:
        k += 1
    k = _stored_attachment_shared_prefix_limit(conn, child_composed, parent_composed, k)
    k = _stored_file_edit_shared_prefix_limit(conn, child_session_id, child_composed, k)
    record_substage("signature_compare", t0)

    if k == 0:
        t0 = time.perf_counter()
        _set_edge(None, "spawned-fresh", next_link_type=resolved_link_type)
        record_substage("edge_update", t0)
        return set()
    prefix_message_ids = [child_composed[i][0] for i in range(k)]
    # polylogue-7xrv5: capture the third generation before its branch points are
    # deleted. Any prefix-sharing edge naming one of these soon-to-be-deleted
    # rows belongs to a grandchild that resolved against this child while the
    # child was still stored whole; its branch point dangles the moment the rows
    # go, and only the returned ids put it inside the repair scope.
    t0 = time.perf_counter()
    inbound_prefix_bp_placeholders = ",".join("?" for _ in prefix_message_ids)
    invalidated_branch_point_sources = {
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT DISTINCT src_session_id
            FROM session_links
            WHERE inheritance = 'prefix-sharing'
              AND branch_point_message_id IN ({inbound_prefix_bp_placeholders})
              AND src_session_id <> ?
            """,
            (*prefix_message_ids, child_session_id),
        ).fetchall()
    }
    record_substage("inbound_branch_point_scan", t0)
    t0 = time.perf_counter()
    _remap_session_event_prefix_refs(
        conn,
        child_session_id,
        tuple((prefix_message_ids[index], parent_composed[index][0]) for index in range(k)),
    )
    record_substage("session_event_ref_remap", t0)
    t0 = time.perf_counter()
    _reextract_provider_usage_tail_db(
        conn,
        child_session_id,
        prefix_message_ids=prefix_message_ids,
    )
    record_substage("provider_usage_tail", t0)
    retired_block_sink = _REEXTRACTED_PREFIX_BLOCK_SINK.get()
    if retired_block_sink is not None:
        retired_placeholders = ",".join("?" for _ in prefix_message_ids)
        for row in conn.execute(
            f"SELECT message_id, position, text FROM blocks "
            f"WHERE message_id IN ({retired_placeholders}) AND text IS NOT NULL",
            tuple(prefix_message_ids),
        ):
            retired_block_sink(str(row[0]), int(row[1]), str(row[2]))
    t0 = time.perf_counter()
    with _bulk_fts_session_guard(conn, child_session_id, enabled=bulk_fts, bulk_build=bulk_build):
        if k == len(child_composed):
            _delete_all_session_message_dependents(conn, child_session_id, prefix_message_ids)
            record_substage("dependent_delete", t0)
        else:
            placeholders = ",".join("?" for _ in prefix_message_ids)
            _delete_prefix_message_dependents(conn, prefix_message_ids)
            record_substage("dependent_delete", t0)
            t0 = time.perf_counter()
            conn.execute(
                f"DELETE FROM messages WHERE message_id IN ({placeholders})",
                tuple(prefix_message_ids),
            )
            record_substage("message_delete", t0)
    # The child's own rows just changed (inherited prefix deleted); drop its
    # memoized own-signatures so any later compose in this batch recomputes them.
    t0 = time.perf_counter()
    if cache is not None:
        cache.pop(child_session_id, None)
    if composed_cache is not None:
        composed_cache.pop(child_session_id, None)
    _set_edge(
        parent_composed[k - 1][0],
        "prefix-sharing",
        branch_point_content_address=_message_content_address_for_id(conn, parent_composed[k - 1][0]),
        next_link_type=resolved_link_type,
    )
    record_substage("edge_update", t0)
    t0 = time.perf_counter()
    refresh_session_summary(conn, child_session_id)
    # Late-parent resolution mutates an already-materialized child outside its
    # own write path, so usage must be rebuilt here: it is aggregated at write
    # time and nothing else revisits it. Derived session rows converge on their
    # own -- the refreshed counts move the child's high-water mark, which is
    # what the staleness comparison reads. The reset keeps parser-declared
    # model rows, which neither aggregate below can recreate.
    _reconcile_session_model_usage_rows(conn, child_session_id)
    _aggregate_message_tokens_into_model_usage(conn, child_session_id)
    _aggregate_provider_usage_into_model_usage(conn, child_session_id)
    # Derived session products cache the pre-extraction message set. Their
    # staleness predicate compares the session's sort key, updated-at and
    # content hash, none of which re-extraction moves, so the rows are dropped
    # outright: a missing profile is what makes the child a convergence
    # candidate again.
    conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (child_session_id,))
    conn.execute("DELETE FROM session_latency_profiles WHERE session_id = ?", (child_session_id,))
    record_substage("count_refresh", t0)
    if not bulk_build:
        # The session-write guard kept the blocks delete trigger from
        # re-deriving the child's action pairs; its delegation cohort is
        # refreshed by the write that owns this resolution, after the edges
        # settle.
        t0 = time.perf_counter()
        _reconcile_tool_use_outcomes(conn, child_session_id)
        _rehash_session_messages(conn, child_session_id)
        refresh_action_pairs(conn, child_session_id)
        record_substage("action_pairs", t0)
    return invalidated_branch_point_sources


def _suffix_after_session_id(message_id: str, session_id: str) -> str | None:
    prefix = f"{session_id}:"
    if not message_id.startswith(prefix):
        return None
    return message_id[len(prefix) :]


_NATIVE_MSG_SUFFIX_RE = re.compile(r"^msg-(\d+)$")


def _native_msg_ordinal(native_id: str) -> int | None:
    match = _NATIVE_MSG_SUFFIX_RE.match(native_id)
    if match is None:
        return None
    return int(match.group(1))


def _replacement_for_stale_prefix_branch_point(
    parent_composed: Sequence[tuple[str, str]],
    stale_suffix: str,
) -> str | None:
    exact_candidates = [message_id for message_id, _sig in parent_composed if message_id.endswith(f":{stale_suffix}")]
    if len(exact_candidates) == 1:
        return exact_candidates[0]
    if exact_candidates:
        return None

    stale_ordinal = _native_msg_ordinal(stale_suffix)
    if stale_ordinal is None:
        return None

    predecessor: tuple[int, str] | None = None
    ambiguous = False
    for message_id, _sig in parent_composed:
        native_suffix = message_id.rsplit(":", 1)[-1]
        ordinal = _native_msg_ordinal(native_suffix)
        if ordinal is None or ordinal >= stale_ordinal:
            continue
        if predecessor is None or ordinal > predecessor[0]:
            predecessor = (ordinal, message_id)
            ambiguous = False
        elif ordinal == predecessor[0]:
            ambiguous = True
    if predecessor is None or ambiguous:
        return None
    return predecessor[1]


def dangling_prefix_branch_point_sql(alias: str = "l") -> str:
    """Return the predicate selecting composing prefix-sharing edges whose
    branch point names a message row that no longer exists.

    A dangling branch point makes the composed reader bail to the child's own
    tail, so every such edge is a session that reads short. Shared by the
    in-write/archive-wide repair and by the readiness census
    (``polylogue.operations.daemon_workload_probe``) so the measured condition
    and the repaired condition cannot drift apart (polylogue-7xrv5).
    """
    return f"""
        {alias}.inheritance = 'prefix-sharing'
          AND {alias}.resolved_dst_session_id IS NOT NULL
          AND {alias}.branch_point_message_id IS NOT NULL
          AND {topology_status_composes_sql(f"{alias}.status")}
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = {alias}.branch_point_message_id
          )
    """


#: Upper exclusive bound of the ``<session_id>:`` message-id namespace. Every
#: ``messages.message_id`` is ``session_id || ':n:' || native_id`` or
#: ``session_id || ':c:' || content_identity || '.' || occurrence``, so the
#: half-open BINARY range ``[sid || ':', sid || ';')`` -- ``';'`` is the byte
#: after ``':'`` -- selects exactly the ids a session owns. The range form (not
#: ``LIKE``/``GLOB``/``substr``) is what lets SQLite drive
#: ``idx_session_links_branch_point`` instead of scanning ``session_links``.
_MESSAGE_ID_NAMESPACE_UPPER_BOUND = ";"


def branch_points_anchored_in_session(conn: sqlite3.Connection, session_id: str) -> set[str]:
    """Sessions whose prefix-sharing branch point names a *missing* row of *session_id*.

    polylogue-gy2yu. A full replace deletes every message of ``session_id``
    before reinserting the new transcript, so a branch point naming a row that
    moved elsewhere in the lineage now dangles until it is re-anchored.
    Identity resolution only ever revisits *unresolved* edges, so without this
    lookup an already-resolved child is in none of ``_resolve_session_graph``'s
    impacted sets and the in-write re-anchoring skips it.

    The anchor, not the edge's parent, is the right key: a child that branched
    inside its parent's *inherited* prefix carries a branch point owned by an
    ancestor, whose ``resolved_dst_session_id`` never names the replaced
    session.
    """
    rows = conn.execute(
        f"""
        SELECT DISTINCT l.src_session_id
        FROM session_links l
        WHERE l.branch_point_message_id >= :low
          AND l.branch_point_message_id < :high
          AND {dangling_prefix_branch_point_sql()}
        """,
        {
            "low": f"{session_id}:",
            "high": f"{session_id}{_MESSAGE_ID_NAMESPACE_UPPER_BOUND}",
        },
    ).fetchall()
    return {str(row[0]) for row in rows}


def count_dangling_prefix_branch_points(conn: sqlite3.Connection) -> tuple[int, int]:
    """Count archive-wide dangling prefix-sharing branch points.

    Returns ``(edge_count, session_count)`` -- the number of stale edges and the
    number of distinct sessions that therefore compose to their own tail only.
    """
    row = conn.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT l.src_session_id)
        FROM session_links l
        WHERE {dangling_prefix_branch_point_sql()}
        """
    ).fetchone()
    if row is None:
        return (0, 0)
    return (int(row[0]), int(row[1]))


# --- Inherited-prefix preservation (polylogue-gy2yu) ---------------------------
#
# Invariant: a session write never changes the composed transcript of a child
# that inherits a prefix from it. A child's inherited prefix is a *view* of its
# parent's rows, but its content is evidence from the child's own bytes -- the
# child's raw physically replayed that prefix. When a parent full replace drops,
# rewrites or relocates a message some child inherits, the same write keeps the
# child's composed transcript intact: the in-write repair re-anchors a
# relocated branch point onto identical content, and anything it cannot
# re-anchor is materialized into the child's own rows, making the child
# self-contained. No child is ever left composing to a truncated tail, so no
# out-of-band re-derivation exists for this loss.

#: Tables whose rows hang off one message and travel with it. ``action_pairs``
#: is omitted: it is derived from ``blocks`` by :func:`refresh_action_pairs`.
_MESSAGE_DEPENDENT_TABLES: tuple[str, ...] = (
    "blocks",
    "attachment_refs",
    "paste_spans",
    "web_content_constructs",
    "file_edits",
)

#: Child-owned rows that reference a message by ``source_message_id``. A parent
#: delete with foreign keys on sets these to NULL; the guard puts them back.
_SOURCE_MESSAGE_REF_TABLES: tuple[str, ...] = (
    "session_events",
    "session_provider_usage_events",
    "session_agent_policies",
)

_GUARD_PREFIX = "polylogue_prefix_guard_"

#: SQLite parameter chunk for the guard's keyed lookups.
_GUARD_ID_CHUNK = 500

#: Every stored column that names a message by id, as ``(table, column)``.
#: A parent's delete nulls the foreign-key ones and leaves the plain-text
#: ``boundary_message_id`` naming a vanished row; the guard puts each back.
_MESSAGE_REF_COLUMNS: tuple[tuple[str, str], ...] = (
    *((table, "source_message_id") for table in _SOURCE_MESSAGE_REF_TABLES),
    ("session_events", "boundary_message_id"),
    ("messages", "parent_message_id"),
)


@dataclass(slots=True)
class _InheritedPrefixGuard:
    """What sessions inheriting from a rewritten session held before its write."""

    rewritten_session_id: str
    #: inheriting session -> the session it resolves its prefix through. Each
    #: session's pre-write composed transcript is in the guard's TEMP
    #: ``before`` table, never in Python: a whale child's prefix, times every
    #: child of a whale parent, would not fit the writer.
    parents: dict[str, str]
    #: ``(src, dst_origin, dst_native_id, link_type, block_id, message_id)`` for
    #: every edge dispatched from a block of a snapshotted message.
    dispatch_refs: tuple[tuple[str, str, str, str, str, str], ...]


class InheritedPrefixMaterializationError(RuntimeError):
    """An inherited prefix row has no identity it can take inside the child."""


def _insertable_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA main.table_xinfo('{table}')") if int(row[6]) == 0]


def _snapshot_table(table: str) -> str:
    return f"temp.{_GUARD_PREFIX}{table}"


def _drop_prefix_guard_tables(conn: sqlite3.Connection) -> None:
    for table in (
        "messages",
        *_MESSAGE_DEPENDENT_TABLES,
        "attachment_native_ids",
        "attachments",
        "session_provider_usage_events",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {_snapshot_table(table)}")
    conn.execute(f"DROP TABLE IF EXISTS temp.{_GUARD_PREFIX}ids")
    conn.execute(f"DROP TABLE IF EXISTS temp.{_GUARD_PREFIX}plan")
    for table in (
        "refs",
        "positions",
        "before",
        "now",
        "reanchors",
        "copies",
        "sources",
        "placed",
        "taken",
        "occurrences",
    ):
        conn.execute(f"DROP TABLE IF EXISTS temp.{_GUARD_PREFIX}{table}")


def _capture_inherited_prefixes(conn: sqlite3.Connection, session_id: str) -> _InheritedPrefixGuard | None:
    """Record what every session inheriting from ``session_id`` holds, before a replace.

    Two edge sets can lose their prefix: direct prefix-sharing children, and
    any deeper descendant whose branch point is a row of ``session_id`` (it
    branched inside an intermediate session's inherited prefix, so its
    resolved parent never names ``session_id``). Only ``session_id``'s own rows
    can disappear in its write, so only those are copied aside, into
    connection-private TEMP tables; ancestor rows stay readable in place. The
    common case -- no inheriting session -- costs two indexed probes.
    """
    candidates = conn.execute(
        f"""
        SELECT DISTINCT src_session_id FROM session_links
        WHERE inheritance = 'prefix-sharing'
          AND branch_point_message_id IS NOT NULL
          AND resolved_dst_session_id IS NOT NULL
          AND src_session_id <> resolved_dst_session_id
          AND {topology_status_composes_sql()}
          AND (
              resolved_dst_session_id = :session
              OR (branch_point_message_id >= :low AND branch_point_message_id < :high)
          )
        ORDER BY src_session_id
        """,
        {
            "session": session_id,
            "low": f"{session_id}:",
            "high": f"{session_id}{_MESSAGE_ID_NAMESPACE_UPPER_BOUND}",
        },
    ).fetchall()
    # A child is affected only through the edge readers compose (the first by
    # ``link_type, dst_origin, dst_native_id``): another affected edge of the
    # same child is not the one its transcript follows.
    parents: dict[str, str] = {}
    for (candidate,) in candidates:
        selected = _prefix_sharing_edge_sync(conn, str(candidate))
        if selected is None or selected[0] == str(candidate):
            continue
        parent_id, branch_point = selected
        if parent_id == session_id or (
            f"{session_id}:" <= branch_point < f"{session_id}{_MESSAGE_ID_NAMESPACE_UPPER_BOUND}"
        ):
            parents[str(candidate)] = parent_id
    if not parents:
        return None
    _drop_prefix_guard_tables(conn)
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}refs (
               table_name TEXT NOT NULL, column_name TEXT NOT NULL, row_id INTEGER NOT NULL,
               session_id TEXT NOT NULL, old_id TEXT NOT NULL,
               PRIMARY KEY (table_name, column_name, row_id))"""
    )
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}before (
               child TEXT NOT NULL, ordinal INTEGER NOT NULL, message_id TEXT NOT NULL,
               signature TEXT NOT NULL, owner TEXT NOT NULL, PRIMARY KEY (child, ordinal))"""
    )
    conn.execute(f"CREATE INDEX temp.{_GUARD_PREFIX}before_id ON {_GUARD_PREFIX}before (child, message_id)")
    conn.execute(f"CREATE INDEX temp.{_GUARD_PREFIX}before_owner ON {_GUARD_PREFIX}before (owner, message_id)")
    before = f"temp.{_GUARD_PREFIX}before"
    for child in parents:
        # The pre-write composed transcript, streamed into the guard table.
        for ordinal, (message_id, signature, owner) in enumerate(_iter_composed_rows(conn, child)):
            conn.execute(f"INSERT INTO {before} VALUES (?, ?, ?, ?, ?)", (child, ordinal, message_id, signature, owner))
        # Every inherited row, not only the rewritten session's: a
        # materialized child owns copies of its whole prefix, and a reference
        # left on an ancestor row would name a message outside its transcript.
        for table, column in _MESSAGE_REF_COLUMNS:
            conn.execute(
                f"""INSERT OR IGNORE INTO temp.{_GUARD_PREFIX}refs
                    SELECT ?, ?, rowid, session_id, {column} FROM main.{table}
                    WHERE session_id = ? AND {column} IN (
                        SELECT message_id FROM {before} WHERE child = ? AND owner <> ?)""",
                (table, column, child, child, child),
            )
    conn.execute(f"CREATE TEMP TABLE {_GUARD_PREFIX}ids (message_id TEXT PRIMARY KEY)")
    conn.execute(
        f"INSERT OR IGNORE INTO temp.{_GUARD_PREFIX}ids SELECT message_id FROM {before} WHERE owner = ? AND owner <> child",
        (session_id,),
    )
    ids = f"SELECT message_id FROM temp.{_GUARD_PREFIX}ids"
    # Any other session's reference into a row this write deletes -- a
    # deeper descendant composing through a child that will materialize --
    # is nulled (or left dangling) before a copy exists to remap it onto.
    for table, column in _MESSAGE_REF_COLUMNS:
        conn.execute(
            f"""INSERT OR IGNORE INTO temp.{_GUARD_PREFIX}refs
                SELECT ?, ?, rowid, session_id, {column} FROM main.{table}
                WHERE {column} IN ({ids}) AND session_id <> ?""",
            (table, column, session_id),
        )
    conn.execute(
        f"CREATE TEMP TABLE {_GUARD_PREFIX}messages AS SELECT * FROM main.messages WHERE message_id IN ({ids})"
    )
    for table in _MESSAGE_DEPENDENT_TABLES:
        conn.execute(
            f"CREATE TEMP TABLE {_GUARD_PREFIX}{table} AS SELECT * FROM main.{table} WHERE message_id IN ({ids})"
        )
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}attachment_native_ids AS
            SELECT n.* FROM main.attachment_native_ids AS n
            WHERE n.ref_id IN (SELECT ref_id FROM {_snapshot_table("attachment_refs")})"""
    )
    # The replace sweeps an attachment whose last reference was a dropped
    # message, so the copied refs need their attachment rows back.
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}attachments AS
            SELECT a.* FROM main.attachments AS a
            WHERE a.attachment_id IN (SELECT attachment_id FROM {_snapshot_table("attachment_refs")})"""
    )
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}session_provider_usage_events AS
            SELECT * FROM main.session_provider_usage_events WHERE source_message_id IN ({ids})"""
    )
    # A dispatch pointer into a snapshotted block is nulled by the delete
    # (ON DELETE SET NULL) and must follow the block to wherever it lives next.
    dispatch_refs = tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]))
        for row in conn.execute(
            f"""SELECT l.src_session_id, l.dst_origin, l.dst_native_id, l.link_type,
                       l.parent_tool_use_block_id, b.message_id
                FROM session_links AS l JOIN {_snapshot_table("blocks")} AS b
                  ON b.block_id = l.parent_tool_use_block_id"""
        )
    )
    return _InheritedPrefixGuard(
        rewritten_session_id=session_id,
        parents=parents,
        dispatch_refs=dispatch_refs,
    )


def _settle_inherited_prefixes(
    conn: sqlite3.Connection,
    guard: _InheritedPrefixGuard,
    *,
    cache: _SignatureCacheLike | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
) -> set[str]:
    """Keep every captured session's composed transcript whole after the write.

    A session whose branch point still resolves -- in place, or re-anchored
    onto identical content -- keeps inheriting: it sees its parent's current
    prefix up to that point, including in-place edits of inherited messages,
    and its own references into re-anchored rows follow them. A session whose
    branch point this write removed, or any of whose pre-write inherited rows
    neither survives nor re-anchors, would otherwise compose to a shortened
    transcript; its pre-write inherited prefix is materialized into its own
    rows instead.
    Shallower sessions settle first, so a descendant anchored in rows its
    parent just materialized follows them rather than being copied again.
    Returns the materialized sessions.

    A session whose edge this write invalidated (an identity contradiction) is
    not this guard's: that loss has its own named route.
    """
    materialized: set[str] = set()
    now = f"temp.{_GUARD_PREFIX}now"
    #: pre-write id -> the row a composable session now inherits in its place.
    conn.execute(f"CREATE TEMP TABLE {_GUARD_PREFIX}reanchors (old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL)")
    #: materialized session -> its copy of each pre-write inherited row.
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}copies (
               child TEXT NOT NULL, old_id TEXT NOT NULL, new_id TEXT NOT NULL, PRIMARY KEY (child, old_id))"""
    )
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}now (
               ordinal INTEGER PRIMARY KEY, message_id TEXT NOT NULL, signature TEXT NOT NULL, owner TEXT NOT NULL)"""
    )
    conn.execute(f"CREATE INDEX temp.{_GUARD_PREFIX}now_id ON {_GUARD_PREFIX}now (message_id)")
    conn.execute(f"CREATE INDEX temp.{_GUARD_PREFIX}now_signature ON {_GUARD_PREFIX}now (signature, ordinal)")

    def depth(session: str) -> int:
        hops, seen = 0, {session}
        while (parent := guard.parents.get(session)) is not None and parent not in seen:
            hops, session = hops + 1, parent
            seen.add(parent)
        return hops

    try:
        for child in sorted(guard.parents, key=lambda item: (depth(item), item)):
            parent = guard.parents[child]
            edge = conn.execute(
                """SELECT 1 FROM session_links
                   WHERE src_session_id = ? AND resolved_dst_session_id = ?
                     AND inheritance = 'prefix-sharing' AND branch_point_message_id IS NOT NULL
                   LIMIT 1""",
                (child, parent),
            ).fetchone()
            if edge is None:
                continue
            conn.execute(f"DELETE FROM {now}")
            for ordinal, (message_id, signature, owner) in enumerate(_iter_composed_rows(conn, child)):
                conn.execute(f"INSERT INTO {now} VALUES (?, ?, ?, ?)", (ordinal, message_id, signature, owner))
            if conn.execute(f"SELECT 1 FROM {now} WHERE owner <> ? LIMIT 1", (child,)).fetchone() is not None:
                # Every pre-write inherited row must still be composed, in
                # place or re-anchored; a surviving branch point does not prove
                # the rows before it survived. One lost row materializes the
                # whole pre-write prefix instead of shortening the transcript.
                reanchored = _reanchor_inherited_rows(conn, child)
                if reanchored is not None:
                    # Message identity alone does not preserve its materials.
                    # A replacement can keep every message while dropping the
                    # attachment refs a child previously inherited. Compare the
                    # existing guard snapshot before deciding to keep sharing;
                    # its normal materialization below restores lost refs/bytes.
                    attachments_survive = all(
                        _message_references_attachment(
                            conn, reanchored.get(str(message_id), str(message_id)), str(attachment_id)
                        )
                        for message_id, attachment_id in conn.execute(
                            f"""SELECT a.message_id, a.attachment_id
                                FROM {_snapshot_table("attachment_refs")} AS a
                                WHERE a.message_id IN (
                                    SELECT message_id FROM temp.{_GUARD_PREFIX}before
                                    WHERE child = ? AND owner <> ?
                                )""",
                            (child, child),
                        )
                    )
                    if attachments_survive:
                        conn.executemany(
                            f"INSERT OR REPLACE INTO temp.{_GUARD_PREFIX}reanchors VALUES (?, ?)", reanchored.items()
                        )
                        continue
            _materialize_inherited_prefix(
                conn,
                child,
                guard.rewritten_session_id,
                parent,
                bulk_fts=bulk_fts,
                bulk_build=bulk_build,
            )
            if cache is not None:
                cache.pop(child, None)
            materialized.add(child)
            if not _composition_matches_before(conn, child):
                raise InheritedPrefixMaterializationError(
                    f"materializing the inherited prefix of {child!r} did not reproduce its composed transcript"
                )
        _restore_message_refs(conn)
        _restore_dispatch_refs(conn, guard.dispatch_refs, bulk_build=bulk_build)
    finally:
        _drop_prefix_guard_tables(conn)
    return materialized


def _reanchor_inherited_rows(conn: sqlite3.Connection, child: str) -> dict[str, str] | None:
    """Map each pre-write inherited id of ``child`` onto the row now composed in its place.

    ``None`` when a pre-write inherited row is neither still composed nor
    re-anchored. A row still composed under its own id keeps it (an in-place
    edit needs no entry), so a surviving message is never claimed by another
    one's content. Only the ids that left the composition are then matched,
    in order, by content signature against the rows no surviving id holds, so
    a message inserted or removed elsewhere in the prefix does not break the
    mapping of unchanged rows around it, and a removed duplicate never takes
    the row of an identical message that survived. Read from the guard's
    ``before``/``now`` tables; the mapping holds only rows that moved.
    """
    before = f"temp.{_GUARD_PREFIX}before"
    now = f"temp.{_GUARD_PREFIX}now"
    mapping: dict[str, str] = {}
    cursor = 0
    lost = conn.execute(
        f"""SELECT b.message_id, b.signature FROM {before} AS b
            WHERE b.child = ? AND b.owner <> ?
              AND NOT EXISTS (SELECT 1 FROM {now} AS n WHERE n.message_id = b.message_id AND n.owner <> ?)
            ORDER BY b.ordinal""",
        (child, child, child),
    ).fetchall()
    for old, signature in lost:
        match = conn.execute(
            f"""SELECT n.ordinal, n.message_id FROM {now} AS n
                WHERE n.signature = ? AND n.ordinal >= ? AND n.owner <> ?
                  AND NOT EXISTS (SELECT 1 FROM {before} AS b WHERE b.child = ? AND b.message_id = n.message_id)
                ORDER BY n.ordinal LIMIT 1""",
            (signature, cursor, child, child),
        ).fetchone()
        if match is None:
            return None
        mapping[str(old)] = str(match[1])
        cursor = int(match[0]) + 1
    return mapping


def _composition_matches_before(conn: sqlite3.Connection, child: str) -> bool:
    """Whether ``child``'s composed signatures equal the guard's pre-write ones, streamed."""
    expected = conn.execute(
        f"SELECT signature FROM temp.{_GUARD_PREFIX}before WHERE child = ? ORDER BY ordinal", (child,)
    )
    composed = (signature for _message_id, signature, _owner in _iter_composed_rows(conn, child))
    wanted = (str(row[0]) for row in expected)
    return all(have == want for have, want in zip_longest(composed, wanted))


def _copy_of(conn: sqlite3.Connection, session_id: str, old_id: str) -> str | None:
    row = conn.execute(
        f"SELECT new_id FROM temp.{_GUARD_PREFIX}copies WHERE child = ? AND old_id = ?", (session_id, old_id)
    ).fetchone()
    return None if row is None else str(row[0])


def _reanchor_of(conn: sqlite3.Connection, old_id: str) -> str | None:
    row = conn.execute(f"SELECT new_id FROM temp.{_GUARD_PREFIX}reanchors WHERE old_id = ?", (old_id,)).fetchone()
    return None if row is None else str(row[0])


def _restore_message_refs(conn: sqlite3.Connection) -> None:
    """Point every captured message reference at the row its session now composes.

    A reference follows the copy its own lineage holds (itself or its nearest
    materialized ancestor), else a re-anchored row, else it keeps the old id
    when that row survived. Streamed from the guard's TEMP table.
    """
    captured = conn.execute(
        f"""SELECT table_name, column_name, row_id, session_id, old_id FROM temp.{_GUARD_PREFIX}refs
            ORDER BY session_id, old_id"""
    )
    resolved: tuple[str, str, str] | None = None
    for table, column, row_id, session, old_id in captured:
        if resolved is None or resolved[:2] != (str(session), str(old_id)):
            copied = _copy_in_lineage(conn, str(session), str(old_id))
            target = copied[1] if copied is not None else _reanchor_of(conn, str(old_id)) or str(old_id)
            resolved = (str(session), str(old_id), target)
        target = resolved[2]
        # With foreign keys on, the parent's delete nulled a reference; with
        # them suspended (bulk rebuild), or for the plain-text boundary id, it
        # still names the old row.
        conn.execute(
            f"""UPDATE main.{table} SET {column} = ?
                WHERE rowid = ? AND ({column} IS NULL OR {column} = ?)
                  AND EXISTS (SELECT 1 FROM messages WHERE message_id = ?)""",
            (target, row_id, old_id, target),
        )


def _restore_dispatch_refs(
    conn: sqlite3.Connection,
    refs: Sequence[tuple[str, str, str, str, str, str]],
    *,
    bulk_build: bool,
) -> None:
    """Point each captured dispatch edge at its block's current home.

    The block survives in place when its message did; otherwise it now lives
    in the session that materialized the message -- preferring the edge's own
    resolved parent, the session that dispatched it.
    """
    refreshed: set[str] = set()
    for src, dst_origin, dst_native_id, link_type, block_id, message_id in refs:
        target: str | None = None
        resolved = conn.execute(
            """SELECT resolved_dst_session_id FROM session_links
               WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?""",
            (src, dst_origin, dst_native_id, link_type),
        ).fetchone()
        dispatcher = None if resolved is None or resolved[0] is None else str(resolved[0])
        # The dispatcher saw the call through its composed transcript: when
        # its lineage now holds a copy, that copy is the call. Otherwise the
        # block survived in place only if it is the same call by its evidence
        # -- a rewrite can reuse the generated id for a different one.
        copied = _copy_in_lineage(conn, dispatcher, message_id)
        if copied is not None:
            target = copied[1] + block_id[len(message_id) :]
            refreshed.add(copied[0])
        elif (
            conn.execute(
                f"""SELECT 1 FROM main.blocks AS b JOIN {_snapshot_table("blocks")} AS old
                      ON old.block_id = b.block_id
                    WHERE b.block_id = ? AND b.block_type IS old.block_type AND b.tool_id IS old.tool_id
                      AND b.tool_name IS old.tool_name AND b.tool_input IS old.tool_input""",
                (block_id,),
            ).fetchone()
            is not None
        ):
            target = block_id
        else:
            reanchored = _reanchor_of(conn, message_id)
            if reanchored is not None:
                # The message survived elsewhere in the lineage (re-anchored).
                candidate = reanchored + block_id[len(message_id) :]
                if conn.execute("SELECT 1 FROM blocks WHERE block_id = ?", (candidate,)).fetchone() is not None:
                    target = candidate
        if target is None:
            continue
        conn.execute(
            """UPDATE session_links SET parent_tool_use_block_id = ?
               WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
                 AND (parent_tool_use_block_id IS NULL OR parent_tool_use_block_id = ?)""",
            (target, src, dst_origin, dst_native_id, link_type, block_id),
        )
        # The session-write guard suppresses the link refresh trigger, and the
        # delegation row belongs to the edge's resolved dispatcher.
        if dispatcher is not None:
            refreshed.add(dispatcher)
    if not bulk_build:
        refresh_delegation_facts_for_sessions(conn, refreshed)


def _copy_in_lineage(conn: sqlite3.Connection, dispatcher: str | None, message_id: str) -> tuple[str, str] | None:
    """``(owner, copy id)`` of the copy of ``message_id`` on the dispatcher's own lineage.

    The dispatcher saw the row through its composed transcript, so the copy it
    now sees is the one its nearest materialized ancestor (or itself) holds.
    A copy held by an unrelated sibling is never chosen.
    """
    seen: set[str] = set()
    cursor = dispatcher
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        copied = _copy_of(conn, cursor, message_id)
        if copied is not None:
            return cursor, copied
        row = conn.execute(
            f"""SELECT resolved_dst_session_id FROM session_links
               WHERE src_session_id = ? AND inheritance = 'prefix-sharing'
                 AND resolved_dst_session_id IS NOT NULL AND branch_point_message_id IS NOT NULL
                 AND {topology_status_composes_sql()}
               ORDER BY link_type, dst_origin, dst_native_id LIMIT 1""",
            (cursor,),
        ).fetchone()
        cursor = None if row is None else str(row[0])
    return None


def _plan_prefix_positions(conn: sqlite3.Connection, child: str) -> int:
    """Place the guard's ``sources`` rows; return the slots they need below the tail.

    The source coordinates are kept when they are unique, ordered and sit below
    the child's own rows; otherwise the rows are renumbered densely by source
    ``(session, position)`` in order of first appearance, so sibling variants
    stay one turn. The result is the TEMP ``placed`` table.
    """
    sources = f"temp.{_GUARD_PREFIX}sources"
    own_min = conn.execute("SELECT MIN(position) FROM messages WHERE session_id = ?", (child,)).fetchone()[0]
    duplicate = conn.execute(
        f"SELECT 1 FROM {sources} GROUP BY position, variant_index HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    unordered = conn.execute(
        f"""SELECT 1 FROM (SELECT position, LAG(position) OVER (ORDER BY ordinal) AS previous FROM {sources})
            WHERE previous IS NOT NULL AND position < previous LIMIT 1"""
    ).fetchone()
    highest = conn.execute(f"SELECT MAX(position) FROM {sources}").fetchone()[0]
    fits = duplicate is None and unordered is None and (own_min is None or highest is None or highest < int(own_min))
    conn.execute(
        f"CREATE TEMP TABLE {_GUARD_PREFIX}placed (ordinal INTEGER PRIMARY KEY, new_position INTEGER NOT NULL)"
    )
    if fits:
        conn.execute(f"INSERT INTO temp.{_GUARD_PREFIX}placed SELECT ordinal, position FROM {sources}")
        return 0
    conn.execute(
        f"""INSERT INTO temp.{_GUARD_PREFIX}placed
            SELECT s.ordinal, d.slot FROM {sources} AS s JOIN (
                SELECT session_id, position, ROW_NUMBER() OVER (ORDER BY MIN(ordinal)) - 1 AS slot
                FROM {sources} GROUP BY session_id, position
            ) AS d ON d.session_id = s.session_id AND d.position = s.position"""
    )
    return int(conn.execute(f"SELECT COUNT(DISTINCT new_position) FROM temp.{_GUARD_PREFIX}placed").fetchone()[0])


def _materialize_inherited_prefix(
    conn: sqlite3.Connection,
    child: str,
    rewritten_session_id: str,
    parent_session_id: str,
    *,
    bulk_fts: bool,
    bulk_build: bool,
) -> None:
    """Copy ``child``'s pre-write inherited rows into its own rows and stop its inheritance.

    The inherited rows are the guard's ``before`` rows ``child`` does not own.
    Rows the rewritten session owned come from the pre-write TEMP snapshot,
    every other ancestor row from the live tables. Each copy keeps its native
    identity when the child has no row of that native id; otherwise it takes
    the next free content occurrence, so no stored child id -- and no durable
    reference to one -- moves. The plan streams through TEMP tables, so a whale
    prefix is never held in Python; each copy is recorded in the guard's
    ``copies`` table.
    """
    del parent_session_id  # Every prefix-sharing edge of the child is flipped below.
    snapshot = _snapshot_table("messages")
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}sources (
               ordinal INTEGER PRIMARY KEY, old_id TEXT NOT NULL, session_id TEXT, position INTEGER,
               variant_index INTEGER, native_id TEXT, content_identity TEXT, content_occurrence INTEGER,
               source_native_id_json TEXT)"""
    )
    conn.execute(
        f"""INSERT INTO temp.{_GUARD_PREFIX}sources
            SELECT b.ordinal, b.message_id,
                   COALESCE(s.session_id, m.session_id), COALESCE(s.position, m.position),
                   COALESCE(s.variant_index, m.variant_index),
                   CASE WHEN s.message_id IS NOT NULL THEN s.native_id ELSE m.native_id END,
                   CASE WHEN s.message_id IS NOT NULL THEN s.content_identity ELSE m.content_identity END,
                   CASE WHEN s.message_id IS NOT NULL THEN s.content_occurrence ELSE m.content_occurrence END,
                   CASE WHEN s.message_id IS NOT NULL THEN s.source_native_id_json ELSE m.source_native_id_json END
            FROM temp.{_GUARD_PREFIX}before AS b
            LEFT JOIN {snapshot} AS s ON s.message_id = b.message_id
            LEFT JOIN main.messages AS m ON m.message_id = b.message_id AND s.message_id IS NULL
            WHERE b.child = ? AND b.owner <> ?""",
        (child, child),
    )
    missing = conn.execute(
        f"SELECT old_id FROM temp.{_GUARD_PREFIX}sources WHERE session_id IS NULL ORDER BY ordinal LIMIT 1"
    ).fetchone()
    if missing is not None:
        raise InheritedPrefixMaterializationError(f"inherited row {missing[0]!r} of {child!r} is not retained")
    inherited_count = int(conn.execute(f"SELECT COUNT(*) FROM temp.{_GUARD_PREFIX}sources").fetchone()[0])
    slots = _plan_prefix_positions(conn, child)
    # Identity decisions, one row at a time: natives already taken by the
    # child (or by an earlier copy) and each content identity's next free
    # occurrence live in TEMP tables, never in Python sets.
    conn.execute(f"CREATE TEMP TABLE {_GUARD_PREFIX}taken (native_id TEXT PRIMARY KEY)")
    from polylogue.core.message_native_identity import message_native_key

    for stored_native, source_carrier in conn.execute(
        "SELECT native_id, source_native_id_json FROM messages WHERE session_id = ? AND native_id IS NOT NULL",
        (child,),
    ):
        native_key = message_native_key(native_id_from_storage(str(stored_native), source_carrier))
        conn.execute(f"INSERT OR IGNORE INTO temp.{_GUARD_PREFIX}taken VALUES (?)", (native_key,))
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}occurrences (
               content_identity TEXT PRIMARY KEY, next_occurrence INTEGER NOT NULL, prefix_seen INTEGER NOT NULL)"""
    )
    conn.execute(
        f"""INSERT INTO temp.{_GUARD_PREFIX}occurrences
            SELECT content_identity, MAX(content_occurrence) + 1, 0 FROM messages
            WHERE session_id = ? AND content_identity IS NOT NULL GROUP BY content_identity""",
        (child,),
    )
    conn.execute(
        f"""CREATE TEMP TABLE {_GUARD_PREFIX}plan (
               ordinal INTEGER PRIMARY KEY, old_id TEXT NOT NULL UNIQUE, new_id TEXT NOT NULL UNIQUE,
               native_id TEXT, content_occurrence INTEGER, identity_source TEXT NOT NULL, content_hash BLOB NOT NULL,
               position INTEGER NOT NULL)"""
    )
    # Record which copies took a content ID, by their occurrence among the
    # prefix's messages of the same content identity, so a replay of the
    # spawned-fresh child reproduces these IDs (``_IdentityScope``).
    content_copies: dict[str, list[int]] = defaultdict(list)
    copy_bases: dict[str, int] = {}
    keys_digest = hashlib.sha256()
    kinds: list[str] = []
    first_key = ""
    keyed = True
    source_rows = conn.execute(
        f"""SELECT s.ordinal, s.old_id, s.variant_index, s.native_id, s.content_identity, s.content_occurrence,
                   p.new_position, s.source_native_id_json
            FROM temp.{_GUARD_PREFIX}sources AS s JOIN temp.{_GUARD_PREFIX}placed AS p ON p.ordinal = s.ordinal
            ORDER BY s.ordinal"""
    )
    for (
        ordinal,
        old_id,
        variant_index,
        native_id,
        content_identity,
        stored_occurrence,
        position,
        native_carrier,
    ) in source_rows:
        source_native = native_id_from_storage(
            None if native_id is None else str(native_id), None if native_carrier is None else str(native_carrier)
        )
        key = _prefix_key(source_native, None if content_identity is None else str(content_identity))
        if key is None:
            keyed = False
        else:
            keys_digest.update(_identity_sequence_frame(key))
            kinds.append("n" if source_native is not None else "c")
            first_key = first_key or key
        identity = None if content_identity is None else str(content_identity)
        ordinal_in_prefix = 0
        if identity is not None:
            seen = conn.execute(
                f"SELECT prefix_seen FROM temp.{_GUARD_PREFIX}occurrences WHERE content_identity = ?", (identity,)
            ).fetchone()
            ordinal_in_prefix = 0 if seen is None else int(seen[0])
            conn.execute(
                f"""INSERT INTO temp.{_GUARD_PREFIX}occurrences VALUES (?, 0, 1)
                    ON CONFLICT(content_identity) DO UPDATE SET prefix_seen = prefix_seen + 1""",
                (identity,),
            )
        if (
            native_id is not None
            and conn.execute(
                f"INSERT OR IGNORE INTO temp.{_GUARD_PREFIX}taken VALUES (?)", (message_native_key(source_native),)
            ).rowcount
            == 1
        ):
            new_native: str | None = str(native_id)
            content_occurrence = None if stored_occurrence is None else int(stored_occurrence)
            new_id = archive_message_id(child, source_native)
            identity_source = "native"
        elif identity is not None:
            next_row = conn.execute(
                f"SELECT next_occurrence FROM temp.{_GUARD_PREFIX}occurrences WHERE content_identity = ?", (identity,)
            ).fetchone()
            content_occurrence = int(next_row[0])
            conn.execute(
                f"UPDATE temp.{_GUARD_PREFIX}occurrences SET next_occurrence = next_occurrence + 1 WHERE content_identity = ?",
                (identity,),
            )
            new_native = None
            new_id = f"{child}:c:{identity}.{content_occurrence}"
            identity_source = "content"
            content_copies[identity].append(ordinal_in_prefix)
            copy_bases.setdefault(identity, content_occurrence)
        else:
            raise InheritedPrefixMaterializationError(
                f"inherited row {old_id!r} collides with a native id of {child!r} and has no content identity"
            )
        # A placeholder the copy's canonical hash replaces once its blocks are
        # in place (``_rehash_session_messages``); unique per copied row.
        placeholder = _hash_bytes("inherited-prefix-copy", new_id, str(position), str(variant_index))
        conn.execute(
            f"INSERT INTO temp.{_GUARD_PREFIX}plan VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ordinal, old_id, new_id, new_native, content_occurrence, identity_source, placeholder, position),
        )
    tail_digest = hashlib.sha256()
    tail_length = 0
    for (content_identity,) in conn.execute(
        "SELECT content_identity FROM messages WHERE session_id = ? ORDER BY position, variant_index", (child,)
    ):
        tail_digest.update(_identity_sequence_frame(str(content_identity or "")))
        tail_length += 1
    identity_scope = _IdentityScope(
        inherited_count,
        {identity: tuple(ordinals) for identity, ordinals in content_copies.items()},
        keys_digest.hexdigest() if keyed else "",
        first_key if keyed else "",
        copy_bases,
        tail_length,
        tail_digest.hexdigest(),
        "".join(kinds) if keyed else "",
        _position_runs(
            int(row[0]) for row in conn.execute(f"SELECT position FROM temp.{_GUARD_PREFIX}plan ORDER BY ordinal")
        )
        if slots
        else (),
    )
    tail_start = conn.execute("SELECT MIN(position) FROM messages WHERE session_id = ?", (child,)).fetchone()[0]
    shifted = _make_room_below(conn, "messages", child, slots)
    if shifted or slots:
        # A compaction boundary endpoint names a transcript position. One on a
        # row of the child's own tail moves with the tail; one on an inherited
        # row follows that row to its copied position (renumbering can make a
        # source position collide with a tail position, so the tail's own
        # rows decide first); a gap above the tail's start moves with it.
        conn.execute(
            f"""CREATE TEMP TABLE {_GUARD_PREFIX}positions (
                   source_position INTEGER PRIMARY KEY, first_position INTEGER NOT NULL,
                   last_position INTEGER NOT NULL)"""
        )
        conn.execute(
            f"""INSERT INTO temp.{_GUARD_PREFIX}positions
                SELECT s.position, MIN(p.new_position), MAX(p.new_position)
                FROM temp.{_GUARD_PREFIX}sources AS s JOIN temp.{_GUARD_PREFIX}placed AS p ON p.ordinal = s.ordinal
                GROUP BY s.position"""
        )

        def moved(endpoint: str, mapped: str) -> str:
            # The copies are not inserted yet: the child's rows are its tail,
            # already shifted.
            return f"""CASE
                WHEN EXISTS (SELECT 1 FROM messages WHERE session_id = :child AND position = {endpoint} + :shift)
                    THEN {endpoint} + :shift
                WHEN (SELECT {mapped} FROM temp.{_GUARD_PREFIX}positions WHERE source_position = {endpoint}) IS NOT NULL
                    THEN (SELECT {mapped} FROM temp.{_GUARD_PREFIX}positions WHERE source_position = {endpoint})
                WHEN {endpoint} >= :tail THEN {endpoint} + :shift
                ELSE {endpoint} END"""

        conn.execute(
            f"""UPDATE session_events
               SET boundary_start_position = {moved("boundary_start_position", "first_position")},
                   boundary_end_position = {moved("boundary_end_position", "last_position")}
               WHERE session_id = :child
                 AND (boundary_start_position IS NOT NULL OR boundary_end_position IS NOT NULL)""",
            {"shift": shifted, "child": child, "tail": tail_start if tail_start is not None else 0},
        )
        conn.execute(f"DROP TABLE temp.{_GUARD_PREFIX}positions")

    overrides = {
        "session_id": ":child",
        "native_id": "p.native_id",
        "content_occurrence": "p.content_occurrence",
        "identity_source": "p.identity_source",
        "position": "p.position",
        "content_hash": "p.content_hash",
        "is_active_leaf": "0",
        "parent_message_id": f"(SELECT q.new_id FROM temp.{_GUARD_PREFIX}plan AS q WHERE q.old_id = s.parent_message_id)",
    }
    params = {"child": child, "parent": rewritten_session_id}
    attachment_columns = ", ".join(_insertable_columns(conn, "attachments"))
    conn.execute(
        f"""INSERT OR IGNORE INTO main.attachments ({attachment_columns})
            SELECT {attachment_columns} FROM {_snapshot_table("attachments")}"""
    )
    with _bulk_fts_session_guard(conn, child, enabled=bulk_fts, bulk_build=bulk_build):
        _copy_planned_rows(conn, "messages", overrides, params)
        for table in _MESSAGE_DEPENDENT_TABLES:
            _copy_planned_rows(conn, table, _DEPENDENT_OVERRIDES[table], params)
        _copy_attachment_native_ids(conn, params)
    _copy_prefix_usage_events(conn, params)
    # Every attachment the copy references gained a ref, including ones an
    # ancestor still owns; the snapshotted ones may also need to be swept back.
    refresh_and_sweep_attachment_rows(
        conn,
        {str(row[0]) for row in conn.execute(f"SELECT attachment_id FROM {_snapshot_table('attachments')}")}
        | {
            str(row[0])
            for row in conn.execute(
                f"""SELECT r.attachment_id FROM attachment_refs AS r
                    JOIN temp.{_GUARD_PREFIX}plan AS p ON p.new_id = r.message_id"""
            )
        },
    )
    _keep_an_active_leaf(conn, child)
    stored = int(
        conn.execute(
            f"SELECT COUNT(*) FROM messages AS m JOIN temp.{_GUARD_PREFIX}plan AS p ON p.new_id = m.message_id"
        ).fetchone()[0]
    )
    if stored != inherited_count:
        raise InheritedPrefixMaterializationError(
            f"materialized {stored} of {inherited_count} inherited rows of {child!r}; the planned ids disagree "
            "with the stored identity"
        )
    conn.execute(
        f"INSERT INTO temp.{_GUARD_PREFIX}copies SELECT ?, old_id, new_id FROM temp.{_GUARD_PREFIX}plan", (child,)
    )
    # Descendants that branched inside this child's inherited prefix follow
    # the rows to their new owner -- at any depth, since a grandchild's
    # composition reaches the copies through its own parent. Their own
    # references into those rows follow too: the old rows are no longer in
    # their composed transcript.
    descendants = _composing_descendants(conn, child)
    for descendant in descendants:
        conn.execute(
            f"""UPDATE session_links
                SET branch_point_message_id = (
                    SELECT p.new_id FROM temp.{_GUARD_PREFIX}plan AS p
                    WHERE p.old_id = session_links.branch_point_message_id
                )
                WHERE src_session_id = ? AND inheritance = 'prefix-sharing'
                  AND branch_point_message_id IN (SELECT old_id FROM temp.{_GUARD_PREFIX}plan)""",
            (descendant,),
        )
    for session in (child, *descendants):
        for table, column in _MESSAGE_REF_COLUMNS:
            conn.execute(
                f"""UPDATE main.{table}
                    SET {column} = (
                        SELECT p.new_id FROM temp.{_GUARD_PREFIX}plan AS p WHERE p.old_id = {table}.{column}
                    )
                    WHERE session_id = ? AND {column} IN (SELECT old_id FROM temp.{_GUARD_PREFIX}plan)""",
                (session,),
            )
    # A dispatch pointer into a copied block that is still live (an ancestor
    # outside the rewritten session owns it) follows the copy for every
    # dispatcher that now composes through this child. Pointers into the
    # rewritten session's deleted blocks are the guard's captured edges.
    dispatchers = [child, *descendants]
    placeholders = ",".join("?" for _ in dispatchers)
    moved_blocks = f"""SELECT b.block_id, p.new_id || substr(b.block_id, length(p.old_id) + 1) AS new_block_id
                       FROM main.blocks AS b JOIN temp.{_GUARD_PREFIX}plan AS p ON p.old_id = b.message_id
                       WHERE b.session_id <> :parent"""
    redispatched = conn.execute(
        f"""SELECT DISTINCT resolved_dst_session_id FROM session_links
            WHERE resolved_dst_session_id IN ({placeholders})
              AND parent_tool_use_block_id IN (SELECT block_id FROM ({moved_blocks.replace(":parent", "?")}))""",
        (*dispatchers, rewritten_session_id),
    ).fetchall()
    if redispatched:
        conn.execute(
            f"""UPDATE session_links
                SET parent_tool_use_block_id = (
                    SELECT m.new_block_id FROM ({moved_blocks.replace(":parent", "?")}) AS m
                    WHERE m.block_id = session_links.parent_tool_use_block_id
                )
                WHERE resolved_dst_session_id IN ({placeholders})
                  AND parent_tool_use_block_id IN (SELECT block_id FROM ({moved_blocks.replace(":parent", "?")}))""",
            (rewritten_session_id, *dispatchers, rewritten_session_id),
        )
        if not bulk_build:
            refresh_delegation_facts_for_sessions(conn, {str(row[0]) for row in redispatched})
    # Every prefix-sharing edge of the child stops inheriting, not only the
    # one to this parent: the child now owns its whole transcript, and a
    # second composing edge would put another prefix in front of the copy.
    conn.execute(
        """UPDATE session_links
           SET inheritance = 'spawned-fresh', branch_point_message_id = NULL, branch_point_content_address = NULL,
               evidence_json = json_set(
                   CASE WHEN json_type(evidence_json) = 'object' THEN evidence_json ELSE '{}' END,
                   '$.inherited_prefix', 'materialized-after-parent-rewrite')
           WHERE src_session_id = ? AND inheritance = 'prefix-sharing'""",
        (child,),
    )
    _record_identity_scope(conn, child, identity_scope)
    # The copied tool uses now pair with results in the child's own tail, and
    # every row copied or moved is rehashed from its stored columns, exactly
    # as a replay of the child computes it.
    _reconcile_tool_use_outcomes(conn, child)
    _rehash_session_messages(conn, child)
    if not bulk_build:
        refresh_action_pairs(conn, child)
    refresh_session_summary(conn, child)
    _reconcile_session_model_usage_rows(conn, child)
    _aggregate_message_tokens_into_model_usage(conn, child)
    _aggregate_provider_usage_into_model_usage(conn, child)
    conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (child,))
    conn.execute("DELETE FROM session_latency_profiles WHERE session_id = ?", (child,))
    for table in ("plan", "sources", "placed", "taken", "occurrences"):
        conn.execute(f"DROP TABLE temp.{_GUARD_PREFIX}{table}")


def _composing_descendants(conn: sqlite3.Connection, session_id: str) -> list[str]:
    """Every session whose composed transcript passes through ``session_id``."""
    found: list[str] = []
    seen = {session_id}
    frontier = [session_id]
    while frontier:
        placeholders = ",".join("?" for _ in frontier)
        rows = conn.execute(
            f"""SELECT DISTINCT src_session_id FROM session_links
                WHERE resolved_dst_session_id IN ({placeholders})
                  AND inheritance = 'prefix-sharing' AND branch_point_message_id IS NOT NULL
                  AND {topology_status_composes_sql()}""",
            tuple(frontier),
        ).fetchall()
        frontier = [str(row[0]) for row in rows if str(row[0]) not in seen]
        seen.update(frontier)
        found.extend(frontier)
    return found


def _keep_an_active_leaf(conn: sqlite3.Connection, child: str) -> None:
    """Give a now self-contained child an active leaf when its own tail had none.

    A child that replayed its parent completely stored no tail, so its session
    pointer names no row; the last materialized message is its leaf.
    """
    pointer = conn.execute(
        """SELECT s.active_leaf_message_id FROM sessions AS s
           WHERE s.session_id = ?
             AND EXISTS (SELECT 1 FROM messages AS m WHERE m.message_id = s.active_leaf_message_id)""",
        (child,),
    ).fetchone()
    if pointer is not None:
        return
    leaf = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position DESC, variant_index DESC LIMIT 1",
        (child,),
    ).fetchone()
    if leaf is None:
        return
    conn.execute("UPDATE messages SET is_active_leaf = (message_id = ?) WHERE session_id = ?", (leaf[0], child))
    conn.execute("UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?", (leaf[0], child))


def _moved_id(column: str) -> str:
    """``column`` re-rooted from the old message id onto the planned one."""
    return f"p.new_id || substr(s.{column}, length(p.old_id) + 1)"


_DEPENDENT_OVERRIDES: dict[str, dict[str, str]] = {
    "blocks": {"session_id": ":child", "message_id": "p.new_id"},
    "attachment_refs": {
        "session_id": ":child",
        "message_id": "p.new_id",
        # A generated ``message:<old_id>`` producer ref names the very row
        # being re-rooted; a parser-supplied provider reference (anything
        # else) is left untouched.
        "producer_ref": (
            "CASE WHEN s.producer_ref = 'message:' || s.message_id THEN 'message:' || p.new_id ELSE s.producer_ref END"
        ),
    },
    "paste_spans": {"session_id": ":child", "message_id": "p.new_id"},
    "web_content_constructs": {"session_id": ":child", "message_id": "p.new_id", "block_id": _moved_id("block_id")},
    "file_edits": {
        "session_id": ":child",
        "message_id": "p.new_id",
        "tool_use_block_id": (
            f"(SELECT q.new_id || substr(s.tool_use_block_id, length(q.old_id) + 1) "
            f"FROM temp.{_GUARD_PREFIX}plan AS q "
            f"WHERE q.old_id = COALESCE((SELECT message_id FROM {_snapshot_table('blocks')} "
            "WHERE block_id=s.tool_use_block_id), "
            "(SELECT message_id FROM main.blocks WHERE block_id=s.tool_use_block_id)))"
        ),
    },
}


def _planned_source(table: str, key: str) -> str:
    """Rows of ``table`` bound to a planned message: parent-owned ones from the
    pre-write snapshot, ancestor-owned ones from the live table."""
    plan = f"temp.{_GUARD_PREFIX}plan"
    return (
        f"SELECT * FROM {_snapshot_table(table)} "
        f"UNION ALL SELECT * FROM main.{table} "
        f"WHERE {key} IN (SELECT old_id FROM {plan}) AND session_id <> :parent"
    )


def _copy_planned_rows(
    conn: sqlite3.Connection, table: str, overrides: Mapping[str, str], params: Mapping[str, str]
) -> None:
    columns = _insertable_columns(conn, table)
    select = ", ".join(overrides.get(column, f"s.{column}") for column in columns)
    key = "message_id"
    conn.execute(
        f"""INSERT INTO main.{table} ({", ".join(columns)})
            SELECT {select} FROM ({_planned_source(table, key)}) AS s
            JOIN temp.{_GUARD_PREFIX}plan AS p ON p.old_id = s.{key}
            ORDER BY p.ordinal""",
        params,
    )


def _copy_attachment_native_ids(conn: sqlite3.Connection, params: Mapping[str, str]) -> None:
    plan = f"temp.{_GUARD_PREFIX}plan"
    columns = _insertable_columns(conn, "attachment_native_ids")
    select = ", ".join(_moved_id("ref_id") if column == "ref_id" else f"s.{column}" for column in columns)
    conn.execute(
        f"""INSERT INTO main.attachment_native_ids ({", ".join(columns)})
            SELECT {select} FROM (
                SELECT * FROM {_snapshot_table("attachment_native_ids")}
                UNION ALL SELECT n.* FROM main.attachment_native_ids AS n
                JOIN main.attachment_refs AS r ON r.ref_id = n.ref_id
                WHERE r.message_id IN (SELECT old_id FROM {plan}) AND r.session_id <> :parent
            ) AS s
            JOIN {plan} AS p ON substr(s.ref_id, 1, length(p.old_id) + 12) = p.old_id || ':attachment:'""",
        params,
    )


def _copy_prefix_usage_events(conn: sqlite3.Connection, params: Mapping[str, str]) -> None:
    """Carry the provider usage observed on the inherited rows into the child.

    The child's own copies of these events were dropped when its prefix was
    first extracted, as duplicates of the parent's observation. Once the child
    stops inheriting it owns those messages again, and their usage with them.
    """
    plan = f"temp.{_GUARD_PREFIX}plan"
    source = (
        f"SELECT * FROM {_snapshot_table('session_provider_usage_events')} "
        f"UNION ALL SELECT * FROM main.session_provider_usage_events "
        f"WHERE source_message_id IN (SELECT old_id FROM {plan}) AND session_id <> :parent"
    )
    count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM ({source}) AS s JOIN {plan} AS p ON p.old_id = s.source_message_id", params
        ).fetchone()[0]
    )
    if count == 0:
        return
    _make_room_below(conn, "session_provider_usage_events", params["child"], count)
    overrides = {
        "session_id": ":child",
        "source_message_id": "p.new_id",
        "position": "ROW_NUMBER() OVER (ORDER BY p.ordinal, s.position) - 1",
    }
    columns = _insertable_columns(conn, "session_provider_usage_events")
    select = ", ".join(overrides.get(column, f"s.{column}") for column in columns)
    conn.execute(
        f"""INSERT INTO main.session_provider_usage_events ({", ".join(columns)})
            SELECT {select} FROM ({source}) AS s JOIN {plan} AS p ON p.old_id = s.source_message_id""",
        params,
    )


def _make_room_below(conn: sqlite3.Connection, table: str, session_id: str, count: int) -> int:
    """Shift ``session_id``'s rows in ``table`` so positions ``0..count-1`` are free.

    Returns the shift applied (0 when the rows already sit high enough). Two
    steps through a disjoint range, because a one-step shift would collide
    with the session's own unique ``(session_id, position)`` key mid-update.
    """
    row = conn.execute(f"SELECT MIN(position) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()
    if count <= 0 or row is None or row[0] is None or int(row[0]) >= count:
        return 0
    delta = count - int(row[0])
    staging = 1 << 40
    conn.execute(f"UPDATE {table} SET position = position + ? WHERE session_id = ?", (staging, session_id))
    conn.execute(f"UPDATE {table} SET position = position - ? WHERE session_id = ?", (staging - delta, session_id))
    return delta


_REHASH_BLOCK_COLUMNS: tuple[str, ...] = (
    "block_type",
    "text",
    "tool_name",
    "tool_id",
    "tool_input",
    "semantic_type",
    "media_type",
    "language",
    "tool_result_is_error",
    "tool_result_exit_code",
    "tool_outcome",
)
_REHASH_MESSAGE_COLUMNS: tuple[str, ...] = (
    "message_id",
    "native_id",
    "source_native_id_json",
    "position",
    "variant_index",
    "fields_digest",
    "role",
    "message_type",
    "material_origin",
    "user_context_text",
    "stop_reason",
    "model_name",
    "model_effort",
    "sender_name",
    "recipient",
    "delivery_status",
    "end_turn",
    "occurred_at_ms",
)


def _rehash_session_messages(conn: sqlite3.Connection, session_id: str) -> None:
    """Recompute each stored message hash of ``session_id`` from its stored rows.

    ``content_hash`` covers the row's identity, coordinates, its own fields
    (``fields_digest``) and its blocks, so a row moved, copied or re-paired
    outside the parse that produced it is given exactly the hash a replay of
    the same message computes. Streamed one message at a time.
    """
    m_idx = {name: index for index, name in enumerate(_REHASH_MESSAGE_COLUMNS)}
    b_idx = {name: index for index, name in enumerate(_REHASH_BLOCK_COLUMNS)}
    messages = conn.execute(
        f"SELECT {', '.join(_REHASH_MESSAGE_COLUMNS)} FROM messages WHERE session_id = ? ORDER BY position, variant_index",
        (session_id,),
    )
    for row in messages:
        blocks = conn.execute(
            f"SELECT {', '.join(_REHASH_BLOCK_COLUMNS)} FROM blocks WHERE message_id = ? ORDER BY position",
            (row[m_idx["message_id"]],),
        )
        content_hash = _message_row_hash(
            session_id,
            native_id_from_storage(
                cast("str | None", row[m_idx["native_id"]]),
                cast("str | None", row[m_idx["source_native_id_json"]]),
            ),
            int(row[m_idx["position"]]),
            int(row[m_idx["variant_index"]] or 0),
            _row_fields_digest(row, m_idx),
            _stored_block_hash_parts(blocks, b_idx),
        )
        conn.execute(
            "UPDATE messages SET content_hash = ? WHERE message_id = ? AND content_hash IS NOT ?",
            (content_hash, row[m_idx["message_id"]], content_hash),
        )


def _repair_stale_prefix_branch_points_db(
    conn: sqlite3.Connection,
    session_ids: set[str] | tuple[str, ...] | list[str],
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    composed_cache: dict[str, list[tuple[str, str]]] | None = None,
) -> int:
    """Refine stale immediate-parent branch-point IDs for the sessions this write touched.

    Older lineage rows can name a branch point as ``<immediate-parent>:<suffix>``
    even after that parent has itself been normalized to tail-only storage. The
    composed reader can only find physical ancestor message IDs, so these stale
    rows make the child bail to its own tail. If the suffix maps to exactly one
    message in the resolved parent's composed transcript, update the edge to the
    composed message id. Ambiguous or unmappable rows stay visible to validation.

    ``session_ids`` is required and is the write's own impacted set. This is a
    producer-side refinement inside the write transaction that created the
    stale row, never a sweep: an unscoped archive-wide variant would silently
    correct whatever the producer got wrong on some *later* pass, and the
    defect would never surface (polylogue-6kur AC4). A dangling edge that
    survives this call is reported by
    :func:`count_dangling_prefix_branch_points`, not repaired out of band.
    """
    scoped = sorted(session_ids)
    if not scoped:
        return 0
    placeholders = ",".join("?" for _ in scoped)
    scope_clause = f"AND l.src_session_id IN ({placeholders})"
    params: list[object] = list(scoped)
    rows = conn.execute(
        f"""
        SELECT l.src_session_id, l.resolved_dst_session_id, l.branch_point_message_id,
               l.branch_point_content_address
        FROM session_links l
        WHERE {dangling_prefix_branch_point_sql()}
          {scope_clause}
        ORDER BY l.src_session_id
        """,
        tuple(params),
    ).fetchall()
    repaired = 0
    local_composed_cache: dict[str, list[tuple[str, str]]] = composed_cache if composed_cache is not None else {}
    for src_session_id, parent_session_id, branch_point_message_id, witness in rows:
        parent_id = str(parent_session_id)
        stale_branch_point = str(branch_point_message_id)
        suffix = _suffix_after_session_id(stale_branch_point, parent_id)
        if suffix is None:
            continue
        parent_composed = _composed_db_signatures(
            conn,
            parent_id,
            cache=cache,
            composed_cache=local_composed_cache,
        )
        replacement = _replacement_for_stale_prefix_branch_point(parent_composed, suffix)
        if replacement is None:
            continue
        if replacement == stale_branch_point:
            continue
        # The replacement must still be the message the child branched at.
        # ``_replacement_for_stale_prefix_branch_point`` will fall back to the
        # greatest ordinal *predecessor*, which is a different message with
        # different content -- and the reader checks the stored witness, so it
        # rejects that edge and serves the child's bare tail anyway. Rewriting
        # the id regardless only removed the edge from the missing-id census,
        # so the archive read short while the census reported it clean
        # (PR #5376). A witness-disagreeing reanchor is therefore refused, and
        # ``_settle_inherited_prefixes`` keeps the child intact instead.
        if witness is not None and _message_content_address_for_id(conn, replacement) != bytes(witness):
            continue
        conn.execute(
            """
            UPDATE session_links
            SET branch_point_message_id = ?
            WHERE src_session_id = ?
              AND resolved_dst_session_id = ?
              AND branch_point_message_id = ?
              AND inheritance = 'prefix-sharing'
            """,
            (replacement, str(src_session_id), parent_id, stale_branch_point),
        )
        repaired += 1
    return repaired


def clear_messages_parent_sql(placeholders: str) -> str:
    """Return the prefix-delete ``messages.parent_message_id`` clear statement.

    Shared by :func:`_delete_all_session_message_dependents` and
    :func:`_delete_prefix_message_dependents` so tests can assert its query
    plan against the exact production SQL rather than a hand-copied string.
    Covered by ``idx_messages_parent`` (leading column ``parent_message_id``).
    """
    return f"""
        UPDATE messages
        SET parent_message_id = NULL
        WHERE parent_message_id IN ({placeholders})
        """


def clear_session_events_source_message_sql(placeholders: str) -> str:
    """Return the prefix-delete ``session_events.source_message_id`` clear statement.

    Covered by the partial index ``idx_session_events_source_message ...
    WHERE source_message_id IS NOT NULL`` (polylogue-crd8) — the planner can
    use a partial index for an ``IN (<non-null literals>)`` predicate because
    every value in the list necessarily satisfies ``IS NOT NULL``.
    """
    return f"""
        UPDATE session_events
        SET source_message_id = NULL
        WHERE source_message_id IN ({placeholders})
        """


def clear_session_agent_policies_source_message_sql(placeholders: str) -> str:
    """Return the prefix-delete ``session_agent_policies.source_message_id`` clear statement.

    Covered by the partial index ``idx_session_agent_policies_source_message
    ... WHERE source_message_id IS NOT NULL`` (polylogue-crd8), same
    partial-index-usability reasoning as
    :func:`clear_session_events_source_message_sql`.
    """
    return f"""
        UPDATE session_agent_policies
        SET source_message_id = NULL
        WHERE source_message_id IN ({placeholders})
        """


def _clear_prefix_message_id_references(
    conn: sqlite3.Connection,
    placeholders: str,
    params: tuple[str, ...],
) -> None:
    """Null out every ``messages(message_id)``-keyed back-reference to a deleted prefix.

    Shared by the two prefix/full dependent-delete helpers below so the three
    statements (and their index coverage) can't silently drift apart between
    the partial-tail and whole-session deletion paths.
    """
    conn.execute(clear_messages_parent_sql(placeholders), params)
    conn.execute(clear_session_events_source_message_sql(placeholders), params)
    conn.execute(clear_session_agent_policies_source_message_sql(placeholders), params)


def _delete_all_session_message_dependents(
    conn: sqlite3.Connection,
    session_id: str,
    prefix_message_ids: Sequence[str],
) -> None:
    """Delete a child whose entire stored transcript was inherited.

    Partial re-extraction deletes by message id because a divergent tail must
    survive. Empty-tail re-extraction can use the existing session indexes
    instead, avoiding huge ``IN (...)`` cleanup for replayed long sessions.
    """
    note_current_deleted_session(conn, session_id)
    note_current_lineage_change(conn, session_id)
    if not prefix_message_ids:
        return
    placeholders = ",".join("?" for _ in prefix_message_ids)
    params = tuple(prefix_message_ids)
    _clear_prefix_message_id_references(conn, placeholders, params)
    conn.execute("DELETE FROM web_content_constructs WHERE session_id = ?", (session_id,))
    conn.execute(
        """
        DELETE FROM attachment_native_ids
        WHERE ref_id IN (SELECT ref_id FROM attachment_refs WHERE session_id = ?)
        """,
        (session_id,),
    )
    orphaned_attachment_ids = session_attachment_ids(conn, session_id)
    conn.execute("DELETE FROM attachment_refs WHERE session_id = ?", (session_id,))
    refresh_and_sweep_attachment_rows(conn, orphaned_attachment_ids)
    conn.execute("DELETE FROM paste_spans WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM blocks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def _delete_prefix_message_dependents(conn: sqlite3.Connection, prefix_message_ids: Sequence[str]) -> None:
    """Mirror message FK side effects when bulk ingest has foreign keys off."""
    if not prefix_message_ids:
        return
    note_current_deleted_message_ids(conn, (str(message_id) for message_id in prefix_message_ids))
    placeholders = ",".join("?" for _ in prefix_message_ids)
    params = tuple(prefix_message_ids)
    _clear_prefix_message_id_references(conn, placeholders, params)
    conn.execute(
        f"""
        DELETE FROM attachment_native_ids
        WHERE ref_id IN (
            SELECT ref_id FROM attachment_refs WHERE message_id IN ({placeholders})
        )
        """,
        params,
    )
    orphaned_attachment_ids = {
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT attachment_id FROM attachment_refs WHERE message_id IN ({placeholders})",
            params,
        )
    }
    for table in ("web_content_constructs", "attachment_refs", "paste_spans", "blocks"):
        conn.execute(
            f"DELETE FROM {table} WHERE message_id IN ({placeholders})",
            params,
        )
    refresh_and_sweep_attachment_rows(conn, orphaned_attachment_ids)


def _remap_session_event_prefix_refs(
    conn: sqlite3.Connection,
    child_session_id: str,
    message_id_pairs: Sequence[tuple[str, str]],
) -> None:
    """Point child events at the canonical messages that own replayed content.

    The provider-local reference remains unchanged in its dedicated column;
    only the rebuildable canonical resolution moves from the soon-to-be-deleted
    child replay row to the corresponding parent/ancestor row.
    """
    conn.executemany(
        """
        UPDATE session_events
        SET source_message_id = ?
        WHERE session_id = ? AND source_message_id = ?
        """,
        (
            (parent_message_id, child_session_id, child_message_id)
            for child_message_id, parent_message_id in message_id_pairs
            if child_message_id != parent_message_id
        ),
    )


def _reextract_provider_usage_tail_db(
    conn: sqlite3.Connection,
    child_session_id: str,
    *,
    prefix_message_ids: Sequence[str],
) -> None:
    """Re-slice a late-resolved child's provider usage onto its stored tail.

    A usage event *bound to a replayed prefix message* is a physical duplicate
    of the parent's own event: the child no longer stores that message, so the
    row is deleted and the parent's copy remains the single observation.

    The surviving events keep their reported ``total_*`` counters verbatim.
    They are cumulative *within the physical session*, never across a lineage
    chain -- see :func:`_provider_usage_event_row`. polylogue-uoq3x: subtracting
    the parent's branch-point cumulative here was doubly wrong. It removed
    tokens the child genuinely spent re-sending the replayed context, and it
    read the parent's *stored* row, which on a chain of depth >= 2 had itself
    already been rebased, so the error compounded into a sawtooth
    (reported 10/13/16/19/22/25 stored as 10/3/13/6/16/9). Where every link
    reported the same counters the subtraction clamped them to zero and a
    companion "drop all-zero rows" delete then destroyed the evidence outright
    -- including the ``request_id``/``finish_reason``-only rows that
    :func:`_provider_usage_event_has_evidence` deliberately admits.

    The model rollup is not touched here: the caller re-derives it once the
    prefix messages are gone, from the surviving events and messages, and a
    partial clear before that would delete a declared model row it keeps.
    """
    if not prefix_message_ids:
        return
    placeholders = ",".join("?" for _ in prefix_message_ids)
    conn.execute(
        f"""
        DELETE FROM session_provider_usage_events
        WHERE session_id = ?
          AND source_message_id IN ({placeholders})
        """,
        (child_session_id, *prefix_message_ids),
    )


def _extract_prefix_tail(
    conn: sqlite3.Connection,
    parent_session_id: str,
    messages: Sequence[ParsedMessage],
    *,
    cache: _SignatureCacheLike | None = None,
    parent_composed: Sequence[tuple[str, str]] | None = None,
    attachments: Sequence[ParsedAttachment] = (),
    before_input: BeforeIndexInput | None = None,
) -> tuple[str | None, str | None, Sequence[ParsedMessage], Mapping[str, str], bytes | None, Sequence[str]]:
    """Align ``messages`` (the child's full parsed messages, which replay the
    parent's prefix) against the parent's composed transcript. Returns
    ``(branch_point_message_id, inheritance, tail_messages, inherited_refs,
    prefix_digest, inherited_prefix_message_ids)``.
    ``inherited_refs`` maps unambiguous provider-local child message ids to the
    canonical parent message rows that physically own the replayed prefix;
    ``inherited_prefix_message_ids`` names those rows by prefix ordinal. The
    shared prefix also ends before a message carrying one of ``attachments``
    its parent row does not reference (``_attachment_shared_prefix_limit``).
    """
    owns_parent = parent_composed is None
    if parent_composed is None:
        source = messages.messages if isinstance(messages, _MessageTail) else messages
        parent_composed = (
            _disk_composed_db_signatures(conn, parent_session_id, source.path.parent, before_input=before_input)
            if isinstance(source, SqliteMessageSink)
            else _composed_db_signatures(conn, parent_session_id, cache=cache, before_input=before_input)
        )
    if not parent_composed:
        if owns_parent and isinstance(parent_composed, _DiskSignatureSequence):
            parent_composed.close()
        return (None, "spawned-fresh", messages, {}, None, ())
    k = 0
    for message, (_, parent_signature) in zip(messages, parent_composed, strict=False):
        if parent_signature != _parsed_message_signature(message):
            break
        k += 1
    k = _attachment_shared_prefix_limit(
        conn,
        messages.messages if isinstance(messages, _MessageTail) else messages,
        parent_composed,
        k,
        attachments,
    )
    k = _file_edit_shared_prefix_limit(messages.messages if isinstance(messages, _MessageTail) else messages, k)
    if k == 0:
        if owns_parent and isinstance(parent_composed, _DiskSignatureSequence):
            parent_composed.close()
        return (None, "spawned-fresh", messages, {}, None, ())
    branch_point_message_id = parent_composed[k - 1][0]
    duplicate_native_ids = _duplicate_message_native_ids(messages)
    source = messages.messages if isinstance(messages, _MessageTail) else messages
    inherited_refs: dict[str, str] | _DiskSourceMessageIds = (
        _DiskSourceMessageIds(source.path.parent) if isinstance(source, SqliteMessageSink) else {}
    )
    try:
        with ExitStack() as prefix_stack:
            if isinstance(duplicate_native_ids, _DiskDuplicateNativeIds):
                prefix_stack.enter_context(duplicate_native_ids.reader())
            if isinstance(parent_composed, _DiskSignatureSequence):
                prefix_stack.enter_context(parent_composed.reader())
            for index, (message, (parent_message_id, _signature)) in enumerate(
                zip(messages, parent_composed, strict=False)
            ):
                if index >= k:
                    break
                provider_id = message.provider_message_id
                if provider_id and _normalized_message_native_id(message) not in duplicate_native_ids:
                    inherited_refs[provider_id] = parent_message_id
        if isinstance(inherited_refs, _DiskSourceMessageIds):
            inherited_refs.finish()
        if isinstance(duplicate_native_ids, _DiskDuplicateNativeIds):
            duplicate_native_ids.close()
    except BaseException as primary:
        failures: list[BaseException] = []
        for artifact in (inherited_refs, duplicate_native_ids):
            if isinstance(artifact, (_DiskSourceMessageIds, _DiskDuplicateNativeIds)):
                try:
                    artifact.close()
                except BaseException as cleanup:
                    failures.append(cleanup)
        if failures:
            raise PreparedSessionSettlementError(failures) from primary
        raise
    return (
        branch_point_message_id,
        "prefix-sharing",
        _MessageTail(messages, k),
        inherited_refs,
        _lineage_prefix_digest(islice(parent_composed, k)),
        _PrefixMessageIds(parent_composed, k),
    )


class _DiskSourceMessageIds(Mapping[str, str]):
    """Provider-local prefix references used while writing session events."""

    def __init__(self, directory: Path) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-prefix-refs-", dir=directory)
        self._lock = threading.RLock()
        self._closed = False
        self._path = Path(self._scratch.name) / "refs.db"
        self._reader_owner: NativeSQLCustodyOwner | None = None
        self._writer: NativeSQLCustodyOwner | None = None
        try:
            self._writer = open_scratch_connection(self._path, lifetime_dependencies=(self,))
        except NativeConnectionSettlementError as failure:
            failure.owner.scratch_directory = self._scratch
            raise
        assert self._writer.connection is not None
        self._conn: sqlite3.Connection | None = self._writer.connection
        try:
            self._connection().execute(
                "CREATE TABLE refs (provider_id TEXT PRIMARY KEY, message_id TEXT NOT NULL) WITHOUT ROWID"
            )
        except BaseException as primary:
            self._writer.scratch_directory = self._scratch
            _close_failed_native_construction(self._writer, primary)
            self.close()
            raise

    def __setitem__(self, key: str, value: str) -> None:
        self._connection().execute("INSERT OR REPLACE INTO refs VALUES (?, ?)", (key, value))

    def __getitem__(self, key: str) -> str:
        with self.reader() as conn:
            row = conn.execute("SELECT message_id FROM refs WHERE provider_id = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        after = ""
        first = True
        while True:
            with self.reader() as conn:
                rows = conn.execute(
                    "SELECT provider_id FROM refs WHERE ? OR provider_id > ? ORDER BY provider_id LIMIT 512",
                    (first, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            first = False
            yield from (str(row[0]) for row in rows)

    def __len__(self) -> int:
        with self.reader() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0])

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("prefix SQL requires its creator-owned reader")
        owner = self._writer if self._writer is not None else self._reader_owner
        if owner is None:
            raise RuntimeError("artifact reader has no native owner")
        return owner.require_connection()

    def finish(self) -> None:
        self._connection().commit()
        assert self._writer is not None
        self._writer.close()
        self._writer = None
        self._conn = None

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("prefix artifact is closed")
            if self._conn is not None:
                yield self._connection()
                return
            conn = connect_measured(f"{self._path.as_uri()}?mode=ro", uri=True)
            owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=(*current_native_sql_lifetimes(), self))
            self._reader_owner = owner
            self._conn = conn
            try:
                yield owner.require_connection()
            finally:
                self._conn = None
                self._reader_owner = None
                owner.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._writer is not None:
                if self._writer in retained_native_sql_owners_for_lifetime(self):
                    self._writer.close()
                self._writer = None
                self._conn = None
            pending = retained_native_sql_owners_for_lifetime(self)
            if pending:
                raise NativeConnectionSettlementError(pending[0], RuntimeError("prefix artifact remains owned"))
            self._scratch.cleanup()
            self._closed = True

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


def _lineage_prefix_digest(signatures: Iterable[tuple[str, str]]) -> bytes:
    digest = hashlib.sha256()
    for message_id, signature in signatures:
        for value in (message_id, signature):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.digest()


def _prefix_sharing_edge_row(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> sqlite3.Row | None:
    """Return the canonical first composing prefix-sharing edge."""
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            return _prefix_sharing_edge_row(conn, session_id, before_input)
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass
    with connection_cursor(
        conn,
        "SELECT rowid FROM session_links WHERE src_session_id=? AND inheritance='prefix-sharing' "
        "AND resolved_dst_session_id IS NOT NULL AND branch_point_message_id IS NOT NULL "
        f"AND {topology_status_composes_sql()} ORDER BY link_type,dst_origin,dst_native_id LIMIT 1",
        (session_id,),
    ) as cursor:
        selected = cursor.fetchone()
    if selected is None:
        return None
    if before_input is not None:
        before_input(
            "session_links",
            ("resolved_dst_session_id", "branch_point_message_id", "branch_point_content_address"),
            "SELECT rowid FROM session_links WHERE rowid=?",
            (selected[0],),
        )
    with connection_cursor(
        conn,
        "SELECT resolved_dst_session_id,branch_point_message_id,branch_point_content_address FROM session_links WHERE rowid=?",
        (selected[0],),
    ) as cursor:
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("selected lineage edge disappeared inside its owned read")
    return cast(sqlite3.Row, row)


def _prefix_sharing_edge_sync(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> tuple[str, str] | None:
    row = _prefix_sharing_edge_row(conn, session_id, before_input)
    return None if row is None else (str(row[0]), str(row[1]))


def _branch_point_content_address_matches(
    conn: sqlite3.Connection,
    child_session_id: str,
    parent_session_id: str,
    branch_point_message_id: str,
    before_input: BeforeIndexInput | None = None,
) -> bool:
    with connection_cursor(
        conn,
        "SELECT l.rowid,m.rowid FROM session_links AS l "
        "LEFT JOIN messages AS m ON m.message_id=l.branch_point_message_id "
        "WHERE l.src_session_id=? AND l.resolved_dst_session_id=? AND l.branch_point_message_id=? "
        "AND l.inheritance='prefix-sharing' LIMIT 1",
        (child_session_id, parent_session_id, branch_point_message_id),
    ) as cursor:
        selected = cursor.fetchone()
    if selected is None:
        return True
    if before_input is not None:
        before_input(
            "session_links",
            ("branch_point_content_address",),
            "SELECT rowid FROM session_links WHERE rowid=?",
            (selected[0],),
        )
        if selected[1] is not None:
            before_input("messages", ("content_address",), "SELECT rowid FROM messages WHERE rowid=?", (selected[1],))
    with connection_cursor(
        conn,
        "SELECT l.branch_point_content_address,m.content_address FROM session_links AS l "
        "LEFT JOIN messages AS m ON m.message_id=l.branch_point_message_id WHERE l.rowid=?",
        (selected[0],),
    ) as cursor:
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("selected branch-point input disappeared inside its owned read")
    if row[0] is None:
        return True
    return row[1] is not None and bytes(row[0]) == bytes(row[1])


#: Artifact stems a Hermes observer batch can carry that are file names rather
#: than session ids. The shared ``events.jsonl`` ATOF stream is the measured
#: case: its own stem reached ``session_links`` as a parent raw id, asserting
#: an edge to a session that cannot exist.
_HERMES_NON_SESSION_PARENT_RAW_IDS = frozenset({"events"})


def _resolved_hermes_parent_native_id(conn: sqlite3.Connection, origin_value: str, parent_native_id: str) -> str | None:
    """Rebind a Hermes observer's parent key onto the conversational session.

    Every ATIF/ATOF observer session composes its parent key from its OWN
    artifact's profile root. On an install where the runtime spans and the
    state db were acquired from different profile directories, that key names
    a session that does not exist and the edge can never resolve. The raw
    Hermes session id is the real join key; the profile qualifier is a
    tie-break, the same shape
    ``context/hermes_lifecycle_reconciliation.py`` already uses.

    Returns ``None`` when nothing may be asserted, preserving the fail-closed
    rule in ``hermes_spans.py``'s docstring: no edge is safer than a wrong one.
    """
    raw_id, _profile_key = split_qualified_session_id(parent_native_id)
    if not raw_id or raw_id in _HERMES_NON_SESSION_PARENT_RAW_IDS:
        return None
    exact = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
        (archive_session_id(origin_value, parent_native_id),),
    ).fetchone()
    if exact is not None:
        return parent_native_id
    rows = conn.execute(
        """SELECT native_id FROM sessions
           WHERE origin = ? AND (native_id = ? OR native_id LIKE ? || '@profile-%')
           ORDER BY native_id""",
        (origin_value, raw_id, raw_id),
    ).fetchall()
    # Exactly one conversational session carries this raw id, so the qualifier
    # mismatch was an acquisition-path artifact, not a real ambiguity. Two or
    # more is the genuine cross-install collision the qualifier exists to keep
    # apart: leave the parser's own key, which stays visibly unresolved.
    return str(rows[0][0]) if len(rows) == 1 else parent_native_id


def _existing_parent_session_id(
    conn: sqlite3.Connection, session: ParsedSession, origin_value: str, before_input: BeforeIndexInput | None = None
) -> str | None:
    parent_provider_id = session.parent_session_provider_id
    if not parent_provider_id:
        return None
    return _existing_session_id_for_native(conn, origin_value, parent_provider_id, before_input)


def _existing_session_id_for_native(
    conn: sqlite3.Connection, origin_value: str, provider_id: str, before_input: BeforeIndexInput | None = None
) -> str | None:
    """Return the archived session one exact provider session id names, if unambiguous."""
    session_id = archive_session_id(origin_value, provider_id.strip())
    with connection_cursor(conn, "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1", (session_id,)) as _input_cursor:
        row = _input_cursor.fetchone()
    if row is not None:
        return session_id
    if before_input is not None:
        before_input(
            "session_identity_claims",
            ("claimant_session_id",),
            "SELECT rowid FROM session_identity_claims WHERE origin=? "
            "AND identity_namespace='provider-session' AND provider_value=? ORDER BY claimant_session_id",
            (origin_value, provider_id.strip()),
        )
    with connection_cursor(
        conn,
        """SELECT claimant_session_id FROM session_identity_claims
           WHERE origin = ? AND identity_namespace = 'provider-session'
             AND provider_value = ? ORDER BY claimant_session_id""",
        (origin_value, provider_id.strip()),
    ) as _input_cursor:
        row = _input_cursor.fetchall()
    return str(row[0][0]) if len(row) == 1 else None


def _dispatch_child_identity_values(observation: ParsedDispatchObservation, payload: Mapping[str, object]) -> set[str]:
    """Every exact provider name this dispatch observation offers for its child."""
    values = {observation.child_provider_id.strip()} if observation.child_provider_id else set()
    candidates = payload.get("child_provider_ids")
    if isinstance(candidates, list):
        values |= {str(value).strip() for value in candidates if str(value).strip()}
    return values


def _canonical_identity_session_ids(conn: sqlite3.Connection, origin: str, values: set[str]) -> set[str] | None:
    """Resolve exact provider names to the sessions that claim them.

    ``None`` means at least one name is not resolvable to exactly one session
    -- an unclaimed or contested name could still turn out to be a different
    session, so the caller must refuse rather than assume agreement.
    """
    resolved: set[str] = set()
    for value in values:
        claimants = {
            str(row[0])
            for row in conn.execute(
                """SELECT DISTINCT claimant_session_id FROM session_identity_claims
                   WHERE origin = ? AND identity_namespace = 'provider-session'
                     AND provider_value = ?""",
                (origin, value),
            ).fetchall()
        }
        if len(claimants) > 1:
            return None
        if claimants:
            resolved |= claimants
            continue
        canonical_id = archive_session_id(origin, value)
        if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (canonical_id,)).fetchone() is None:
            return None
        resolved.add(canonical_id)
    return resolved


@dataclass(frozen=True, slots=True)
class _DispatchResolution:
    """Outcome of binding a child edge to its parent's dispatching block.

    Exactly one of ``block_id`` / ``reason`` is set: a refusal always names
    why, and a bound block never carries a stale reason.
    """

    block_id: str | None
    reason: DispatchResolutionReason | None

    @property
    def method(self) -> str | None:
        return "parent-tool-use-id" if self.block_id is not None else None


_DISPATCH_PENDING = _DispatchResolution(None, None)
"""The parent is not in the archive yet; nothing can be said either way."""


_ORIGIN_SPECS_BY_VALUE: dict[str, Any] | None = None


def _origin_carries_dispatch_identity(origin: str) -> bool:
    global _ORIGIN_SPECS_BY_VALUE
    if _ORIGIN_SPECS_BY_VALUE is None:
        _ORIGIN_SPECS_BY_VALUE = {spec.origin.value: spec for spec in origin_specs()}
    spec = _ORIGIN_SPECS_BY_VALUE.get(origin)
    if spec is None:
        return False
    return bool(spec.topology_capabilities.parent_dispatch.state != "structurally-absent")


def _session_provider_values(conn: sqlite3.Connection, session_id: str) -> set[str]:
    values = {
        str(row[0])
        for row in conn.execute(
            """SELECT provider_value FROM session_identity_claims
               WHERE claimant_session_id = ? AND identity_namespace = 'provider-session'""",
            (session_id,),
        ).fetchall()
    }
    row = conn.execute("SELECT native_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is not None and row[0]:
        values.add(str(row[0]))
    return values


def _session_acquisition_paths(
    conn: sqlite3.Connection, source_read: SessionSourceRead, session_id: str
) -> Generator[str, None, None]:
    """Indexed acquisition paths bound to this session, including retained copies.

    A new winning raw must not make a prior acquisition's sidecar disappear.
    Current raw_id is exact even before native identity is enriched; retained
    acquisitions use the existing (origin, native_id) index and only identities
    with no conflicting canonical claimant. No archive-wide source scan.
    """
    with closing(
        conn.execute("SELECT raw_id, origin, native_id FROM sessions WHERE session_id = ?", (session_id,))
    ) as lineage_rows:
        row = lineage_rows.fetchone()
    if row is None:
        return
    raw_id, origin, native_id = row
    if raw_id is not None:
        raw_path = source_read.raw_path(str(raw_id))
        if raw_path is not None:
            yield raw_path
    for value in _session_provider_values(conn, session_id) | {str(native_id)}:
        with closing(
            conn.execute(
                "SELECT 1 FROM session_identity_claims WHERE origin = ? AND identity_namespace = 'provider-session' "
                "AND provider_value = ? AND claimant_session_id != ? LIMIT 1",
                (origin, value, session_id),
            )
        ) as lineage_rows:
            conflicting = lineage_rows.fetchone()
        if conflicting is not None:
            continue
        with source_read.acquisition_rows(str(origin), value) as acquisition_paths:
            for acquisition in acquisition_paths:
                yield str(acquisition[0])


def _split_source_path(path: str) -> tuple[str, str, str]:
    """Split a source path into its directory prefix, file name and separator.

    The prefix keeps its trailing separator and the path's own separator
    style, so a derived sibling path compares equal to the stored one.
    """
    cut = max(path.rfind("/"), path.rfind("\\"))
    separator = path[cut] if cut >= 0 else "/"
    return path[: cut + 1], path[cut + 1 :], separator


def _sidecar_candidate_paths(
    conn: sqlite3.Connection,
    source_read: SessionSourceRead,
    *,
    parent_session_id: str,
    child_session_id: str,
    parent_values: set[str],
    stems: set[str],
) -> Generator[str, None, None]:
    """Exact source paths where the child's dispatch sidecar can be stored.

    Claude Code writes ``<dir>/<parent>.jsonl``, and the child's transcript
    and sidecar side by side under ``<dir>/<parent>/subagents/``. Both
    sessions' own acquisition paths therefore name the sidecar exactly: it is
    the child transcript's sibling, and it sits in the parent transcript's
    ``subagents`` directory. A session with no recorded acquisition path
    contributes no candidate.
    """
    with closing(_session_acquisition_paths(conn, source_read, child_session_id)) as acquisition_paths:
        for child_path in acquisition_paths:
            directory, _name, _separator = _split_source_path(child_path)
            yield from (f"{directory}{stem}.meta.json" for stem in stems)
    with closing(_session_acquisition_paths(conn, source_read, parent_session_id)) as acquisition_paths:
        for parent_path in acquisition_paths:
            directory, name, separator = _split_source_path(parent_path)
            parent_stem = name.removesuffix(".jsonl").removesuffix(".ndjson")
            if parent_stem in parent_values:
                yield from (
                    f"{directory}{parent_stem}{separator}subagents{separator}{stem}.meta.json" for stem in stems
                )


def _sidecar_dispatch_tool_ids(
    conn: sqlite3.Connection,
    source_read: SessionSourceRead | None,
    *,
    origin: str,
    parent_session_id: str,
    child_session_id: str,
    parent_values: set[str],
    child_values: set[str],
) -> set[str]:
    """Tool ids the child's ``agent-*.meta.json`` sidecar names, bound to this parent.

    The sidecar lives at ``<parent>/subagents/<child stem>.meta.json`` in the
    durable source tier. Its candidate paths come from the two sessions' own
    acquisitions (``_sidecar_candidate_paths``), and each is probed by exact
    ``source_path``, which ``idx_raw_sessions_source_path`` serves: resolving
    one edge costs a few index lookups however many Claude raws the archive
    holds, where a leading-wildcard pattern scanned every one of them per
    edge on the single writer.
    """
    if source_read is None or origin != Origin.CLAUDE_CODE_SESSION.value:
        return set()
    stems = {value for value in child_values if value.startswith("agent-") and ":" not in value}
    if not stems:
        return set()
    if not source_read.raw_sessions_present():
        return set()
    with closing(
        _sidecar_candidate_paths(
            conn,
            source_read,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            parent_values=parent_values,
            stems=stems,
        )
    ) as sidecar_paths:
        return _sidecar_paths_dispatch_tool_ids(
            source_read,
            origin=origin,
            sidecar_paths=sidecar_paths,
            parent_values=parent_values,
        )


def _sidecar_paths_dispatch_tool_ids(
    source_read: SessionSourceRead,
    *,
    origin: str,
    sidecar_paths: Iterable[str],
    parent_values: set[str],
) -> set[str]:
    """Read the dispatch tool ids the sidecars stored at ``sidecar_paths`` name.

    The parent directory must be one of the parent's own provider names, so a
    sidecar can never bind to a different session that happens to share a
    child stem. The bytes are read through the Source host's bound archive CAS
    capability.
    """
    tool_ids: set[str] = set()
    for sidecar_path in sidecar_paths:
        # Only ``source_path`` is constrained in SQL, so the planner can serve
        # the probe from ``idx_raw_sessions_source_path`` alone; an ``origin``
        # term would let it walk every Claude raw through the origin index.
        after: tuple[int, str] | None = None
        while page := source_read.sidecar_page(sidecar_path, after):
            after = (page[-1][4], page[-1][3])
            for source_path, blob_hash, row_origin, raw_id, _source_index in page:
                if row_origin != origin:
                    continue
                parts = str(source_path).replace("\\", "/").split("/")
                if len(parts) < 3 or parts[-3] not in parent_values:
                    continue
                # The evidence cursor is closed before byte hydration. ZIP
                # admission permits very large members. Dispatch identity is a root field, so the
                # sidecar is streamed to the root fields the artifact parser reads,
                # never read whole; the tool_use id itself is kept complete, since
                # it is the exact join key to the parent block.
                try:
                    from polylogue.sources.decoder_json import DecodedRecordSequence

                    with (
                        source_read.open_sidecar_payload(raw_id, blob_hash) as handle,
                        closing(DecodedRecordSequence.from_raw_document(handle)) as document,
                    ):
                        payload = document[0]
                        artifact = (
                            parse_claude_orchestration_artifact(str(source_path), payload)
                            if isinstance(payload, dict)
                            else None
                        )
                # RecursionError is a RuntimeError, not a ValueError: a deeply
                # nested sidecar would otherwise escape this handler and abort the
                # whole session write, and because the raw row persists it would
                # abort it again on every later replay of the same lineage.
                except (OSError, ValueError, ArithmeticError, RecursionError, ijson.JSONError) as exc:
                    emit(
                        "storage.dispatch_sidecar.refused",
                        level=WARNING,
                        outcome="refused",
                        reason="sidecar_unreadable",
                        source_path=source_path,
                        error_type=type(exc).__name__,
                    )
                    continue
                if artifact is None:
                    continue
                tool_ids.update(fact.tool_use_id for fact in artifact.facts if fact.tool_use_id)
    return tool_ids


def _resolve_parent_dispatch_block(
    conn: sqlite3.Connection,
    parent_session_id: str,
    child_session_id: str,
    *,
    source_read: SessionSourceRead | None = None,
) -> _DispatchResolution:
    """Bind a child edge to the exact parent tool_use block that dispatched it.

    Two exact witnesses are joined: the parent's own dispatch observations
    (``claude_delegation_progress`` events whose child identity resolves to
    ``child_session_id``) and the child's dispatch sidecar in the source tier.
    Provider names are compared as the sessions they resolve to: several
    exact names for one child are one identity, and only names resolving to
    different sessions contradict each other. Every refusal is typed; there
    is no ordinal, count, timestamp, or nearest-call fallback.
    """
    origin_row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (parent_session_id,)).fetchone()
    if origin_row is None:
        return _DISPATCH_PENDING
    origin = str(origin_row[0])
    if not _origin_carries_dispatch_identity(origin):
        return _DispatchResolution(None, "origin-no-dispatch-identity")
    rows = conn.execute(
        """SELECT source_message_provider_id, source_message_provider_id_json, payload_json
           FROM session_events
           WHERE session_id = ? AND event_type = 'claude_delegation_progress'""",
        (parent_session_id,),
    ).fetchall()
    tool_ids: set[str] = set()
    contradicted = False
    for stored_source_id, source_id_carrier, payload_json in rows:
        source_id = native_id_from_storage(stored_source_id, source_id_carrier)
        try:
            payload = json.loads(str(payload_json))
        except (TypeError, ValueError):
            continue
        try:
            observation_payload = dict(payload) if isinstance(payload, dict) else {}
            if source_id and "provider_tool_id" not in observation_payload:
                observation_payload["provider_tool_id"] = str(source_id)
            observation = ParsedDispatchObservation.model_validate(observation_payload)
        except (TypeError, ValueError):
            continue
        # A parser-side contradiction is provisional: it compares raw names
        # without an archive to resolve them against. Every other refusal
        # reason stands.
        if observation.resolution_reason is not None and observation.resolution_reason != "identity-contradiction":
            continue
        identity_values = _dispatch_child_identity_values(observation, observation_payload)
        if not identity_values:
            continue
        resolved = _canonical_identity_session_ids(conn, origin, identity_values)
        if resolved is None or child_session_id not in resolved:
            continue
        if len(resolved) > 1:
            contradicted = True
            continue
        tool_ids.add(observation.provider_tool_id)
    tool_ids |= _sidecar_dispatch_tool_ids(
        conn,
        source_read,
        origin=origin,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        parent_values=_session_provider_values(conn, parent_session_id),
        child_values=_session_provider_values(conn, child_session_id),
    )
    if contradicted or len(tool_ids) > 1:
        return _DispatchResolution(None, "dispatch-identity-contradiction")
    if not tool_ids:
        return _DispatchResolution(None, "dispatch-evidence-absent")
    (tool_id,) = tool_ids
    block_rows = conn.execute(
        """SELECT b.block_id FROM blocks b
           JOIN messages m ON m.message_id = b.message_id
           WHERE b.tool_id = ? AND b.block_type = 'tool_use' AND m.session_id = ?
           ORDER BY b.block_id""",
        (tool_id, parent_session_id),
    ).fetchall()
    if not block_rows:
        return _DispatchResolution(None, "dispatch-block-missing")
    if len(block_rows) > 1:
        return _DispatchResolution(None, "dispatch-tool-id-duplicate")
    return _DispatchResolution(str(block_rows[0][0]), None)


# ``evidence_json`` update fragment: bind the reason, or clear it once a block
# binds. Bound as (reason, reason). Older rows default to a JSON array, so the
# object form is established before json_set.
_DISPATCH_REASON_EVIDENCE_SQL = """
    CASE WHEN ? IS NULL
         THEN json_remove(evidence_json, '$.dispatch_reason')
         ELSE json_set(
                  CASE WHEN json_type(evidence_json) = 'object' THEN evidence_json ELSE '{}' END,
                  '$.dispatch_reason', ?)
    END
"""


def _canonicalize_session_link_evidence(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    link_type: str,
) -> None:
    """Restore canonical JSON after SQLite JSON mutation changes key order."""
    row = conn.execute(
        """SELECT evidence_json FROM session_links
           WHERE src_session_id = ? AND dst_origin = ?
             AND dst_native_id = ? AND link_type = ?""",
        (src_session_id, dst_origin, dst_native_id, link_type),
    ).fetchone()
    if row is None:
        return
    try:
        evidence = json.loads(str(row[0]))
    except (TypeError, json.JSONDecodeError):
        return
    conn.execute(
        """UPDATE session_links SET evidence_json = ?
           WHERE src_session_id = ? AND dst_origin = ?
             AND dst_native_id = ? AND link_type = ?""",
        (
            _json_dumps(evidence),
            src_session_id,
            dst_origin,
            dst_native_id,
            link_type,
        ),
    )


#: Convergence-debt stage naming a child whose recomposed lineage prefix was
#: dropped by a provider-session identity contradiction. It is its own stage
#: (not the generic ``convergence`` row) so the daemon's retry drain cannot
#: clear it by running unrelated stages that never re-derive the lost prefix.
IDENTITY_INVALIDATION_DEBT_STAGE = "lineage_prefix_recompose"

_IDENTITY_INVALIDATION_DEBT_ERROR = (
    "lineage link invalidated by a provider-session identity contradiction; "
    "the child's recomposed prefix must be re-derived from source evidence"
)


def _main_database_path(conn: sqlite3.Connection) -> Path | None:
    """Return the file backing the connection's ``main`` schema, if any."""
    for _sequence, name, filename in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "main" and str(filename or ""):
            return Path(str(filename))
    return None


def _record_identity_invalidation_debt(conn: sqlite3.Connection, session_ids: set[str]) -> None:
    """Record retryable convergence debt for lineage-invalidated children.

    The invalidation itself lands in the derived index; the debt belongs to the
    ops tier, which owns ``convergence_debt``. It is written through the same
    ``CursorStore`` writer every other debt producer uses -- no parallel ledger
    -- on the archive's own ``ops.db`` sibling. An index connection with no
    ops tier beside it (in-memory index, bare fixture) records nothing rather
    than bootstrapping a disposable tier from the write path.
    """
    _record_lineage_prefix_debt(conn, session_ids, error=_IDENTITY_INVALIDATION_DEBT_ERROR)


def _record_lineage_prefix_debt(conn: sqlite3.Connection, session_ids: set[str], *, error: str) -> None:
    """Write one ``lineage_prefix_recompose`` debt row per lost-prefix child.

    ``ops.db`` is resolved from the archive root, not from the active index's
    own directory. SQLite reports the physical file behind ``main``, so on any
    archive with a promoted index generation ``PRAGMA database_list`` answers
    ``<root>/.index-generations/<gen>/index.db`` -- and ``ops.db`` named beside
    *that* does not exist, so an ordinary daemon archive dropped the retryable
    debt without a word (PR #5376).
    """
    if not session_ids:
        return
    index_path = _main_database_path(conn)
    if index_path is None:
        return
    ops_db_path = archive_root_for_index_path(index_path) / "ops.db"
    if not ops_db_path.exists():
        return
    from polylogue.sources.live.cursor import CursorStore

    store = CursorStore(index_path, initialize=False, ops_db_path=ops_db_path)
    for session_id in sorted(session_ids):
        store.record_convergence_debt(
            stage=IDENTITY_INVALIDATION_DEBT_STAGE,
            subject_type="session_id",
            subject_id=session_id,
            error=error,
        )


def _write_session_identity_claims(
    conn: sqlite3.Connection, session_id: str, origin: str, session: ParsedSession, *, prior_claims: bool = True
) -> set[str]:
    """Persist exact parser-emitted names that can address this session.

    ``prior_claims`` is false when the session is saved for the first time:
    it can hold no claims yet, so there are none to read back or delete.
    """
    previous_values = (
        {
            str(row[0])
            for row in conn.execute(
                """SELECT provider_value FROM session_identity_claims
                   WHERE claimant_session_id = ? AND identity_namespace = 'provider-session'""",
                (session_id,),
            ).fetchall()
        }
        if prior_claims
        else set()
    )
    values = {
        str(value).strip()
        for value in [session.provider_session_id, *session.provider_session_aliases]
        if str(value).strip()
    }
    retired_values = previous_values - values
    if prior_claims:
        conn.execute("DELETE FROM session_identity_claims WHERE claimant_session_id = ?", (session_id,))
    canonical_value = str(session.provider_session_id).strip()
    for value in sorted(values):
        conn.execute(
            """INSERT OR REPLACE INTO session_identity_claims
               (origin, identity_namespace, provider_value, claimant_session_id, claim_kind, evidence_json)
               VALUES (?, 'provider-session', ?, ?, ?, '{}')""",
            (origin, value, session_id, "canonical" if value == canonical_value else "alias"),
        )
    placeholders = ", ".join("?" for _ in values)
    ambiguous_values = {
        str(row[0])
        for row in conn.execute(
            f"""SELECT provider_value FROM session_identity_claims
                WHERE origin = ? AND identity_namespace = 'provider-session'
                  AND provider_value IN ({placeholders})
                GROUP BY provider_value HAVING COUNT(*) > 1""",
            (origin, *sorted(values)),
        ).fetchall()
    }
    invalidated_children: set[str] = set()

    def _invalidate_links(values_to_invalidate: set[str], *, claimant_only: bool) -> None:
        if not values_to_invalidate:
            return
        value_placeholders = ", ".join("?" for _ in values_to_invalidate)
        claimant_clause = "AND resolved_dst_session_id = ?" if claimant_only else ""
        params: tuple[object, ...] = (origin, *sorted(values_to_invalidate))
        if claimant_only:
            params = (*params, session_id)
        rows = conn.execute(
            f"""SELECT DISTINCT src_session_id FROM session_links
                WHERE dst_origin = ? AND dst_native_id IN ({value_placeholders})
                  AND resolved_dst_session_id IS NOT NULL {claimant_clause}""",
            params,
        ).fetchall()
        invalidated_children.update(str(row[0]) for row in rows)
        conn.execute(
            f"""UPDATE session_links
                SET evidence_json = CASE
                        WHEN inheritance = 'prefix-sharing' AND branch_point_message_id IS NOT NULL
                        THEN json_set(
                            CASE WHEN json_type(evidence_json) = 'object' THEN evidence_json ELSE '{{}}' END,
                            '$.{INVALIDATED_PREFIX_EVIDENCE_KEY}',
                            json_object('parent_session_id', resolved_dst_session_id,
                                        'branch_point_message_id', branch_point_message_id,
                                        'branch_point_content_address', hex(branch_point_content_address)))
                        ELSE evidence_json END,
                    resolved_dst_session_id = NULL,
                    resolved_at_ms = NULL,
                    branch_point_message_id = NULL,
                    branch_point_content_address = NULL,
                    inheritance = NULL,
                    parent_tool_use_block_id = NULL
                WHERE dst_origin = ? AND dst_native_id IN ({value_placeholders})
                  AND resolved_dst_session_id IS NOT NULL {claimant_clause}""",
            params,
        )

    _invalidate_links(retired_values, claimant_only=True)
    _invalidate_links(ambiguous_values, claimant_only=False)
    if ambiguous_values:
        value_placeholders = ", ".join("?" for _ in ambiguous_values)
        conn.execute(
            f"""UPDATE session_links
                SET evidence_json = CASE
                    WHEN json_type(evidence_json) = 'object'
                    THEN json_set(evidence_json, '$.resolution_reason', 'identity-contradiction')
                    ELSE '{{\"resolution_reason\":\"identity-contradiction\"}}'
                END
                WHERE dst_origin = ? AND dst_native_id IN ({value_placeholders})
                  AND resolved_dst_session_id IS NULL""",
            (origin, *sorted(ambiguous_values)),
        )
    return invalidated_children


def _has_incomplete_prefix(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> bool:
    """An unresolved asserted anchor is incomplete without a guessed row ID."""
    if before_input is not None:
        before_input(
            "session_links",
            ("evidence_json", "branch_point_message_id", "status"),
            "SELECT rowid FROM session_links WHERE src_session_id = ?",
            (session_id,),
        )
        before_input(
            "session_identity_scopes",
            ("session_id",),
            "SELECT rowid FROM session_identity_scopes WHERE session_id = ?",
            (session_id,),
        )
    with connection_cursor(
        conn,
        f"""
        SELECT 1 FROM session_links WHERE src_session_id = ? AND (
            {invalidated_prefix_sql()} OR (
                branch_point_message_id IS NULL
                AND ({topology_status_composes_sql()} OR status = 'unresolved')
                AND json_valid(evidence_json)
                AND json_type(evidence_json, '$.{ASSERTED_BRANCH_POINT_EVIDENCE_KEY}') IN ('text', 'object')
                AND evidence_json -> '$.{ASSERTED_BRANCH_POINT_EVIDENCE_KEY}' != '""'
                AND src_session_id NOT IN (SELECT session_id FROM session_identity_scopes)
            )
        ) LIMIT 1""",
        (session_id,),
    ) as cursor:
        return cursor.fetchone() is not None


def _has_invalidated_prefix(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeIndexInput | None = None
) -> bool:
    """A full child replay must settle loss; resolving a parent cannot do so."""
    if before_input is not None:
        before_input(
            "session_links",
            ("evidence_json",),
            "SELECT rowid FROM session_links WHERE src_session_id = ?",
            (session_id,),
        )
    with connection_cursor(
        conn,
        f"SELECT 1 FROM session_links WHERE src_session_id = ? AND {invalidated_prefix_sql()} LIMIT 1",
        (session_id,),
    ) as cursor:
        return cursor.fetchone() is not None


def _message_content_address_for_id(
    conn: sqlite3.Connection, message_id: str, before_input: BeforeIndexInput | None = None
) -> bytes | None:
    if before_input is not None:
        before_input("messages", ("content_address",), "SELECT rowid FROM messages WHERE message_id=?", (message_id,))
    with connection_cursor(conn, "SELECT content_address FROM messages WHERE message_id = ?", (message_id,)) as cursor:
        row = cursor.fetchone()
    return None if row is None or row[0] is None else bytes(row[0])


def _active_leaf_message_id(
    session_id: str,
    messages: Sequence[ParsedMessage],
    explicit_native_id: str | None,
    *,
    content_identities: Sequence[MessageContentIdentity],
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str | None:
    if explicit_native_id:
        first_match: tuple[int, ParsedMessage] | None = None
        for fallback_position, message in enumerate(messages):
            if message.provider_message_id != explicit_native_id:
                continue
            if first_match is None:
                first_match = (fallback_position, message)
            if message.is_active_leaf:
                return _message_id(
                    session_id,
                    message,
                    fallback_position,
                    content_identities=content_identities,
                    duplicate_native_ids=duplicate_native_ids,
                )
        if first_match is not None:
            fallback_position, message = first_match
            return _message_id(
                session_id,
                message,
                fallback_position,
                content_identities=content_identities,
                duplicate_native_ids=duplicate_native_ids,
            )
    for fallback_position, message in enumerate(messages):
        if message.is_active_leaf:
            return _message_id(
                session_id,
                message,
                fallback_position,
                content_identities=content_identities,
                duplicate_native_ids=duplicate_native_ids,
            )
    return (
        _message_id(
            session_id,
            messages[-1],
            len(messages) - 1,
            content_identities=content_identities,
            duplicate_native_ids=duplicate_native_ids,
        )
        if messages
        else None
    )


def _message_id(
    session_id: str,
    message: ParsedMessage,
    fallback_position: int,
    *,
    content_identities: Sequence[MessageContentIdentity],
    message_id_remap: Mapping[str, str] | None = None,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str:
    """Resolve one parsed message's stored ``message_id``.

    ``content_identities`` is the batch's resolved fallback identities, index-
    aligned with the same ``messages`` list ``fallback_position`` indexes, and
    is required: a positional fallback derived here would disagree with the
    ``messages.message_id`` generated column and would carry the renumbering
    defect the content identity exists to remove (polylogue-eqsri).
    """
    content_identity, content_occurrence = content_identities[fallback_position]
    stored_id = archive_message_id(
        session_id,
        _effective_message_native_id(message, duplicate_native_ids),
        content_identity=content_identity,
        content_occurrence=content_occurrence,
    )
    return message_id_remap.get(stored_id, stored_id) if message_id_remap is not None else stored_id


def _duplicate_message_native_ids(messages: Iterable[ParsedMessage]) -> frozenset[str]:
    """Native ids that collide after the same normalization ``messages.native_id`` stores.

    Counts by the SQLite-storable form, preserving all whitespace. Exact
    duplicates use semantic identity and occurrence instead of overwriting
    another message through the generated unique identity.
    """
    source = messages.messages if isinstance(messages, _MessageTail) else messages
    if isinstance(source, SqliteMessageSink):
        return _DiskDuplicateNativeIds(messages, source.path.parent)
    counts = Counter(
        normalized for message in messages if (normalized := _normalized_message_native_id(message)) is not None
    )
    return frozenset(native_id for native_id, count in counts.items() if count > 1)


class _DiskDuplicateNativeIds(frozenset[str]):
    """Sealed ambiguous-id file; each native access belongs to its creator."""

    def __new__(cls, messages: Iterable[ParsedMessage], directory: Path) -> _DiskDuplicateNativeIds:
        return super().__new__(cls)

    def __init__(self, messages: Iterable[ParsedMessage], directory: Path) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        self._reader_owner: NativeSQLCustodyOwner | None = None
        self._writer: NativeSQLCustodyOwner | None = None
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-duplicate-ids-", dir=directory)
        self._path = Path(self._scratch.name) / "native-ids.db"
        try:
            self._writer = open_scratch_connection(self._path, lifetime_dependencies=(self,))
        except NativeConnectionSettlementError as failure:
            failure.owner.scratch_directory = self._scratch
            raise
        assert self._writer.connection is not None
        try:
            conn = self._writer.connection
            conn.execute("CREATE TABLE ids (native_id TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID")
            for message in messages:
                native_id = _normalized_message_native_id(message)
                if native_id is not None:
                    self._writer.require_connection().execute(
                        "INSERT INTO ids VALUES (?, 1) ON CONFLICT(native_id) DO UPDATE SET n = n + 1",
                        (native_id,),
                    )
            conn.commit()
        except BaseException as primary:
            self._writer.scratch_directory = self._scratch
            _close_failed_native_construction(self._writer, primary)
            self.close()
            raise
        self._writer.close()
        self._writer = None

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """Reuse one creator-owned reader for a complete publication window."""
        with self._lock:
            if self._closed:
                raise RuntimeError("duplicate native-id index was closed")
            if self._conn is not None:
                if self._reader_owner is None:
                    raise RuntimeError("duplicate-id reader has no native owner")
                yield self._reader_owner.require_connection()
                return
            conn = connect_measured(f"{self._path.as_uri()}?mode=ro", uri=True)
            owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=(*current_native_sql_lifetimes(), self))
            self._reader_owner = owner
            self._conn = conn
            try:
                yield owner.require_connection()
            finally:
                # Clear only the borrowing slot; the strong native census
                # retains the actual handle and artifact after failed close.
                self._conn = None
                self._reader_owner = None
                owner.close()

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        with self.reader() as conn:
            row = conn.execute("SELECT n FROM ids WHERE native_id = ?", (value,)).fetchone()
        return row is not None and int(row[0]) > 1

    def __iter__(self) -> Iterator[str]:
        after = ""
        first = True
        while True:
            with self.reader() as conn:
                rows = conn.execute(
                    "SELECT native_id FROM ids WHERE n > 1 AND (? OR native_id > ?) ORDER BY native_id LIMIT 512",
                    (first, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            first = False
            yield from (str(row[0]) for row in rows)

    def __len__(self) -> int:
        with self.reader() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM ids WHERE n > 1").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._writer is not None:
                if self._writer in retained_native_sql_owners_for_lifetime(self):
                    self._writer.close()
                self._writer = None
            pending = retained_native_sql_owners_for_lifetime(self)
            if pending:
                raise NativeConnectionSettlementError(pending[0], RuntimeError("duplicate-id artifact remains owned"))
            self._scratch.cleanup()
            self._closed = True

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


def _normalized_message_native_id(message: ParsedMessage) -> str | None:
    from polylogue.core.message_native_identity import message_native_key

    return message_native_key(message.provider_message_id)


def _effective_message_native_id(message: ParsedMessage, duplicate_native_ids: frozenset[str]) -> str | None:
    """Return the storage-normalized native id, or ``None`` if ambiguous.

    ``duplicate_native_ids`` (from ``_duplicate_message_native_ids``) is keyed
    by the same exact, surrogate-substituted form computed here, so
    membership is always compared apples-to-apples.
    """
    from polylogue.core.message_native_identity import stored_message_native_id

    return stored_message_native_id(message.provider_message_id, duplicate_native_ids)


def _stored_session_native_id(native_id: str) -> str:
    """Return the exact value ``sessions.native_id`` stores for this session.

    Single source of truth for session identity (mirrors the message-level
    ab5bad1f FK-failure fix via ``_stored_message_native_id`` below, never
    given a session-level sibling until polylogue-lyr2). The session
    INSERT bind and every call to ``core.identity_law.session_id`` (which the
    generated ``sessions.session_id`` column reimplements in SQL as
    ``origin || ':' || native_id``) MUST route through this helper, or the
    two computations can diverge: a provider-native session id carrying
    leading/trailing whitespace stores truthy verbatim in a raw INSERT bind
    while ``identity_law.session_id``'s ``_required_text`` strips it --
    producing two different spellings of what should be one row's identity.
    Raises ``ValueError`` on an empty/whitespace-only native id, matching
    ``identity_law._required_text``'s own emptiness check -- there is no
    position/variant fallback at the session level the way there is for
    messages, so an unidentifiable session must fail loudly rather than
    silently write a self-mismatched row.
    """
    stripped = native_id.strip()
    if not stripped:
        raise ValueError("session native_id cannot be empty")
    if _SURROGATE_RE.search(stripped):
        # A lone surrogate cannot be bound as SQLite text, and substituting it
        # would merge distinct provider sessions into one row: refused by name.
        raise ValueError("session native_id holds a UTF-16 surrogate code unit and cannot be stored")
    return stripped


def _stored_message_native_id(message: ParsedMessage, duplicate_native_ids: frozenset[str]) -> str | None:
    """Return the exact value ``messages.native_id`` stores for this message.

    This is the single source of truth for message identity (polylogue
    rebuild ab5bad1f FK-failure fix): both the ``_write_messages`` INSERT and
    ``_message_id`` (which feeds ``core.identity_law.message_id``, the
    reference the generated ``messages.message_id`` column reimplements in
    SQL) MUST route through this helper, or the two computations can diverge
    and a later ``blocks`` insert can reference a ``message_id`` that was
    never written.

    Native IDs are opaque. Only literal empty is absent; duplicate exact
    IDs use content identity and occurrence. Whitespace stays significant.
    """
    return sqlite_message_native_id(_effective_message_native_id(message, duplicate_native_ids))


def _stored_native_id_exists(conn: sqlite3.Connection, session_id: str, native_id: object, *, carrier: object) -> bool:
    """Whether a normalized incoming ID already has a row in this session."""
    # Every writer, including prefix materialization, stores this generated
    # identity. Seek its unique key rather than scanning the session's native
    # IDs. Keep exact stored bytes and the original equality predicates.
    return (
        isinstance(native_id, str)
        and bool(native_id)
        and conn.execute(
            "SELECT 1 FROM messages WHERE message_id = ? AND session_id = ? AND native_id = ? LIMIT 1",
            (
                archive_message_id(
                    session_id, native_id_from_storage(native_id, carrier if isinstance(carrier, str) else None)
                ),
                session_id,
                native_id,
            ),
        ).fetchone()
        is not None
    )


def _block_type(block: ParsedContentBlock) -> BlockType:
    value = _enum_value(block.type)
    if value == "thinking":
        return BlockType.THINKING
    if value == "tool_use":
        return BlockType.TOOL_USE
    if value == "tool_result":
        return BlockType.TOOL_RESULT
    if value == "image":
        return BlockType.IMAGE
    if value == "code":
        return BlockType.CODE
    if value == "document":
        return BlockType.DOCUMENT
    return BlockType.TEXT


def _block_language(block: ParsedContentBlock) -> str | None:
    metadata = block.metadata or {}
    value = metadata.get("language")
    return str(value) if value is not None else None


def _block_name(block: ParsedContentBlock) -> str | None:
    """``metadata["name"]`` of a media block is the sole input to ``blocks.name``."""
    if _block_type(block) not in (BlockType.DOCUMENT, BlockType.IMAGE):
        return None
    value = (block.metadata or {}).get("name")
    return value if isinstance(value, str) and value else None


def _semantic_type(block: ParsedContentBlock) -> str | None:
    if _block_type(block) is not BlockType.TOOL_USE or not block.tool_name:
        return None
    tool_input = cast("Mapping[str, JSONValue]", block.tool_input or {})
    category = classify_tool(block.tool_name, tool_input)
    return None if category is ToolCategory.OTHER else category.value


def _has_block(message: ParsedMessage, block_type: BlockType) -> int:
    return int(any(_enum_value(block.type) == block_type.value for block in message.blocks))


def _has_paste(message: ParsedMessage) -> int:
    return int(bool(message.paste_spans))


def _paste_boundary(message: ParsedMessage) -> str | None:
    """Message-level paste boundary state, taken from the first detected span."""
    if not message.paste_spans:
        return None
    return PasteBoundary(message.paste_spans[0].boundary_state).value


def _word_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _payload_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _payload_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _payload_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip():
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return None
    return None


def _payload_optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _repo_name(repository_url: str, root_path: str) -> str | None:
    """Derive a stable repo display name (polylogue-cijx.4 decision 1).

    Prefers ``normalize_repo_name`` -- the same remote-URL/git-root
    normalization the attribution pipeline
    (``storage/derived/session/repo_observations.py``) already uses -- so a
    session whose cwd is a deep subdirectory or an agent worktree resolves to
    the real repository name, not a raw path-basename echo (the
    "agent-<hash>" bug: a session's cwd under
    ``.claude/worktrees/agent-<hash>/`` used to yield that hash as the repo
    name because the old fallback was ``Path(root_path).name`` verbatim).
    Falls back to the previous naive derivation only when normalization
    cannot resolve anything (e.g. a git root that no longer exists on this
    filesystem).
    """
    normalized = normalize_repo_name(repository_url) if repository_url else normalize_repo_name(root_path)
    if normalized:
        return normalized
    candidate = repository_url.rstrip("/").rsplit("/", maxsplit=1)[-1] if repository_url else Path(root_path).name
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    return candidate or None


def _discovered_repo_root_path(root_path: str) -> str | None:
    """Resolve ``root_path`` to its git root when one is discoverable on disk.

    Without this, two sessions whose cwd differs only by subdirectory (or by
    worktree path under one checkout) would write two distinct checkout
    roots for what is really one working tree. Normalizing the *value*
    written into ``root_path`` to the resolved git root collapses
    same-checkout subdirectory variance and keeps this writer's rows
    consistent with the attribution-based writer
    (``storage/derived/session/repo_observations.py``), which already
    normalizes this way. As of the ``repo_identity_key`` schema fix below,
    ``root_path`` is no longer part of ``repos.repo_id`` identity when a
    remote is known -- it still matters as the value recorded in
    ``repo_checkouts``/``session_repos.root_path`` (decision 2's
    repo-relative path stripping), and as the sole identity fallback for a
    repository with no remote.

    Returns ``None`` -- not the raw ``root_path`` -- when no git root is
    discoverable (polylogue-cijx.2 AC4: "a session with no git evidence
    resolves to a directory, not a repository"). Callers with no other git
    evidence (no known remote) must not synthesize a repository identity
    from an unresolved bare cwd -- a session in ``/home/sinity`` is honestly
    a directory, not the "sinity" repo.
    """
    return normalize_repo_path(root_path)


_SCP_LIKE_REMOTE_RE = re.compile(r"^[\w.-]+@([\w.-]+):(.+)$")


def _canonicalize_repo_remote(origin_url: str) -> str:
    """Canonicalize a git remote URL to a stable identity token.

    polylogue-cijx.4 decision 1: "a repository is keyed on its normalized
    remote -- all spellings of one remote are one repo". ``git@host:owner/repo``
    (SCP-like), ``https://host/owner/repo.git``, and ``ssh://host/owner/repo``
    must collapse to the same identity. This strips scheme/userinfo,
    lowercases the host, and strips a trailing ``.git`` suffix and slash.

    Deliberately a small heuristic, not a full git-remote parser: it only
    needs to be *consistent* for the common hosting-provider URL shapes this
    archive actually observes (GitHub/GitLab/etc SSH and HTTPS remotes),
    because it feeds a stored identity key, not a validated remote. Returns
    ``""`` when ``origin_url`` is blank or unrecognizable, signaling the
    caller to fall back to the directory identity.
    """
    raw = origin_url.strip()
    if not raw:
        return ""
    if "://" in raw:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path
    else:
        match = _SCP_LIKE_REMOTE_RE.match(raw)
        if match:
            host, path = match.group(1).lower(), match.group(2)
        else:
            host, path = "", raw
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not path:
        return ""
    canonical = f"{host}/{path}" if host else path
    return canonical.lower()


def repo_identity_key(origin_url: str, root_path: str) -> str:
    """Compute the canonical ``repos.repo_id`` (polylogue-cijx.4 decision 1).

    A repository is keyed on its normalized remote when one is known --
    every worktree checkout of the same remote collapses to a single
    ``repos`` row (``remote:<host>/<path>``). Only when no remote is known
    does identity fall back to the resolved checkout root
    (``dir:<root_path>``, decision 1's "where no remote exists, the
    outermost git root"); ``root_path`` is expected to already be resolved
    to that outermost root by ``_discovered_repo_root_path`` before this is
    called -- a ``root_path`` with no discoverable git root and no known
    remote must not reach this function at all (see ``_write_repo_edges``).

    This used to be a SQLite ``GENERATED ALWAYS`` column computed as
    ``origin_url || root_path`` -- so two worktree checkouts of the exact
    same remote were two different ``repos`` rows purely because their
    checkout paths differed (measured: 3.5% session-label collision from
    this, and repo counts like "polylogue holds 106 distinct repo_ids"
    that were really ~1 repo checked out 106 places). ``repo_id`` is now a
    plain Python-computed column instead of a SQL generated expression,
    because the remote-URL canonicalization above needs real string logic
    (scheme/userinfo stripping, SCP-vs-URL unification, case folding) a SQL
    expression cannot express without duplicating this function in SQL.
    """
    canonical_remote = _canonicalize_repo_remote(origin_url)
    if canonical_remote:
        return f"remote:{canonical_remote}"
    return f"dir:{root_path}"


def _attachment_id(_session_id: str, attachment: ParsedAttachment, *, blob_hash: bytes | None = None) -> str:
    """Separate a descriptor from each measured content revision.

    Metadata-only records never borrow acquired bytes from another capture
    merely because a native file id, name and size agree. Acquired rows share
    only when their descriptor and exact payload hash agree.
    """
    if blob_hash is None:
        if attachment.inline_bytes is not None:
            blob_hash = hashlib.sha256(attachment.inline_bytes).digest()
        elif attachment.precomputed_blob is not None:
            blob_hash = bytes.fromhex(attachment.precomputed_blob[0])
    return attachment_content_id(
        attachment.provider_attachment_id,
        file_id=attachment.provider_file_id,
        drive_id=attachment.provider_drive_id,
        path=attachment.path,
        name=attachment.name,
        media_type=attachment.mime_type,
        declared_size=attachment.size_bytes,
        blob_hash=blob_hash,
    )


class AttachmentReferenceAmbiguityError(ValueError):
    """One acquisition declares competing objects under one native reference."""


def _attachment_reference_native_identity(attachment: ParsedAttachment) -> str:
    """Use the exact Source native ID; presentation and bytes do not rename it."""
    return attachment_native_identity(attachment.provider_attachment_id)


def _acquire_attachment_blob(
    conn: sqlite3.Connection,
    attachment: ParsedAttachment,
) -> tuple[bytes | None, int, str]:
    """Describe an unfetched attachment without publishing bytes.

    Inline bytes must be published before this low-level index writer is called,
    through an archive-owned publisher whose receipt spans the index commit.
    """
    del conn
    if attachment.inline_bytes is not None:
        raise ValueError("inline attachment bytes require preacquired_attachment_blobs from an archive-owned publisher")
    if attachment.precomputed_blob is not None:
        raise ValueError("a precomputed attachment blob requires preacquired_attachment_blobs to record it")
    return (None, attachment.size_bytes or 0, "unfetched")


def _attachment_source_url(attachment: ParsedAttachment) -> str | None:
    return attachment.source_url


def _attachment_caption(attachment: ParsedAttachment) -> str | None:
    return attachment.caption


def _attachment_native_id_values(attachment: ParsedAttachment) -> tuple[tuple[str, str], ...]:
    """Return SQLite-sanitized typed native identities for one attachment."""
    native_values = (
        ("attachment", attachment.provider_attachment_id),
        ("file", attachment.provider_file_id),
        ("drive", attachment.provider_drive_id),
        ("url", _attachment_source_url(attachment)),
    )
    return tuple(
        (id_kind, sanitized)
        for id_kind, native_id in native_values
        if native_id is not None
        for sanitized in (_sqlite_text(native_id),)
        if sanitized
    )


def _write_attachment_native_ids(conn: sqlite3.Connection, ref_id: str, attachment: ParsedAttachment) -> None:
    for id_kind, native_id in _attachment_native_id_values(attachment):
        conn.execute(
            """
            INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id)
            VALUES (?, ?, ?)
            """,
            (ref_id, id_kind, _sqlite_text(native_id)),
        )


def _hash_bytes(*parts: str) -> bytes:
    """Digest an ordered field tuple under an unambiguous framing.

    Each part is length-prefixed rather than NUL-separated. Imported message
    and tool content legitimately contains U+0000, so a single-byte separator
    lets two different field splits serialize to the same byte stream and so
    to the same content-identity hash -- a dedup collision an attacker or an
    ordinary provider export can both produce. A fixed 8-byte big-endian
    length makes the encoding injective over any byte content.
    """
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _json_dumps(value: object) -> str:
    """Canonical JSON text for a column; a lone surrogate stays a ``\\uXXXX`` escape.

    Escaping, unlike replacing it with U+FFFD, keeps the stored payload equal
    to the value the session's content hash was computed from.
    """
    encoded = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if encoded.isascii() or not _SURROGATE_RE.search(encoded):
        return encoded
    return _SURROGATE_RE.sub(lambda match: f"\\u{ord(match.group()):04x}", encoded)


def _json_value(value: object) -> object:
    """``value`` with every non-JSON type lowered to its stored spelling.

    Strings are kept exactly, a lone surrogate included; ``_json_dumps``
    escapes it.
    """
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        lowered = [_json_value(item) for item in value]
        return sorted(lowered, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if Decimal(as_float) == value else str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).hex()
    if isinstance(value, datetime | date | datetime_time):
        return value.isoformat()
    return value


def _sqlite_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not _SURROGATE_RE.search(value):
        return value
    return _SURROGATE_RE.sub("\ufffd", value)


def _sqlite_bool(value: bool | None) -> int | None:
    """Map an optional bool to SQLite 0/1, preserving None (unknown)."""
    if value is None:
        return None
    return 1 if value else 0


def _json_loads(raw_json: str | bytes) -> dict[str, object]:
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    loaded = json.loads(raw_json or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _json_tuple(raw_json: str | bytes) -> tuple[str, ...]:
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    loaded = json.loads(raw_json or "[]")
    return tuple(str(item) for item in loaded) if isinstance(loaded, list) else ()


def _json_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        return int(value)
    return 0


def _iso_from_ms(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    parsed = parse_timestamp(int(value) / 1000)
    return parsed.isoformat() if parsed is not None else None


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


__all__ = [
    "PreparedSessionRows",
    "prepare_session_shard",
    "ArchiveAgentPolicy",
    "ARCHIVE_BLOCK_ROW_COLUMNS",
    "ARCHIVE_MESSAGE_ROW_COLUMNS",
    "ARCHIVE_SESSION_ENVELOPE_COLUMNS",
    "ArchiveBlockRow",
    "ArchiveMessageRow",
    "ArchiveSessionTag",
    "ArchiveSessionEnvelope",
    "archive_block_row",
    "archive_block_row_select_sql",
    "archive_message_row_select_sql",
    "archive_session_envelope_select_sql",
    "read_session_agent_policies",
    "read_session_tags",
    "rebuild_archive_messages_fts",
    "replace_parser_ingest_flag_tags",
    "repo_identity_key",
    "upsert_parser_ingest_flag_tags",
    "upsert_session_tag",
    "raw_source_path",
    "read_archive_session_envelope",
    "rederive_codex_spawn_parent_links",
    "search_archive_blocks",
    "write_parsed_session_to_archive",
]


def session_revision_row_values(raw_id: str | None, content_hash: bytes) -> dict[str, object]:
    """The canonical session row's raw pointer and stored revision digest."""
    return {"raw_id": raw_id, "content_hash": content_hash}


if TYPE_CHECKING:
    from polylogue.archive.revision_replay import RevisionReplayPlan
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        PreparedRevisionAdoption,
        RawMembershipSelectionFamily,
    )


def _provider_usage_projections(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    start_position: int | None = None,
) -> tuple[UsageProjectionModel, ...]:
    """Stream the selected event authority through the canonical domain fold.

    Claude Code can write several physical assistant records for one API
    response. Its message-usage counters are snapshots at request grain, so
    fold each request's latest present lane before the provider-neutral sum.
    Every source event remains stored against its original fragment message.
    """
    models = _provider_usage_existing_models(conn, session_id)
    sole_model = models[0] if len(models) == 1 else None

    def events(selected: Iterator[dict[str, object]]) -> Iterator[dict[str, object]]:
        for values in selected:
            values["model_name"] = str(values["model_name"] or "").strip() or sole_model
            if values["model_name"] is not None:
                yield values

    origin_row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    origin = str(origin_row[0]) if origin_row is not None else ""
    with closing(provider_usage_request_events(conn, session_id, start_position=start_position)) as selected:
        return project_provider_usage_events(events(selected), origin=origin)
