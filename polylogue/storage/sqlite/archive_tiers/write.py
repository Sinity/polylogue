"""Minimal archive index parsed-session writer/read helpers.

Writer module: index.

Session tag/work-event/phase CRUD (the former ``user``-tier twin-write
contract here) moved to ``session_annotations_write.py`` — see that
module's docstring for the current writer-module declaration.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.repo_identity import normalize_repo_name, normalize_repo_path
from polylogue.archive.topology.edge import TopologyEdgeStatus, TopologyEdgeType, branch_type_to_edge_type
from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.enums import BlockType, PasteBoundary, Provider, SessionKind
from polylogue.core.identity_law import message_id as archive_message_id
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamps import parse_timestamp
from polylogue.logging import get_logger
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.fts.fts_lifecycle import message_fts_triggers_present_sync
from polylogue.storage.fts.pl_fold import pl_fold_sql_expr
from polylogue.storage.fts.sql import (
    FTS_BULK_SESSION_WRITE_GUARD,
    delete_session_identity_rows_sql,
    delete_session_rows_sql,
    insert_session_identity_rows_sql,
    insert_session_rows_sql,
    trigram_delete_session_rows_sql,
    trigram_insert_session_rows_sql,
)
from polylogue.storage.runtime import (
    LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT,
    LINEAGE_TRUNCATION_DEPTH_LIMIT,
    LineageTruncationReason,
)
from polylogue.storage.runtime.store_constants import LINEAGE_ITERATIVE_DEPTH_LIMIT
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.action_pairs import refresh_action_pairs
from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
from polylogue.storage.sqlite.archive_tiers.ingest_precedence import should_skip_stale_replace
from polylogue.storage.sqlite.archive_tiers.session_annotations_write import (
    ArchiveSessionPhase,
    ArchiveSessionTag,
    ArchiveSessionWorkEvent,
    read_session_phases,
    read_session_tags,
    read_session_work_events,
    upsert_session_phase,
    upsert_session_tag,
    upsert_session_work_event,
)
from polylogue.storage.sqlite.delegation_facts import refresh_delegation_facts_for_session

logger = get_logger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_ACOMPACT_PARENT_MEMBERSHIP_THRESHOLD = 0.90


@dataclass(frozen=True, slots=True)
class ArchiveBlockRow:
    block_id: str
    message_id: str
    block_type: str
    text: str | None
    tool_name: str | None = None
    tool_id: str | None = None
    semantic_type: str | None = None
    tool_input: str | None = None
    metadata: str | None = None
    language: str | None = None
    # Keystone structured tool-result outcome (schema v16). NULL = unknown.
    tool_result_is_error: int | None = None
    tool_result_exit_code: int | None = None
    # Why the keystone outcome above is unresolved (schema v46). NULL when the
    # outcome IS known, never "unknown of unknown" (polylogue-cuxz.8).
    tool_result_outcome_unknown_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveAttachmentRow:
    attachment_id: str
    message_id: str | None
    display_name: str | None = None
    media_type: str | None = None
    byte_count: int = 0
    upload_origin: str | None = None
    source_url: str | None = None
    caption: str | None = None


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
    title_confidence: float | None = None
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


def archive_message_display_text(blocks: Iterable[ArchiveBlockRow]) -> str:
    """Flatten an archive-tier message's blocks into its display ``text``.

    Single source of truth for the ``message.text`` field the daemon's two
    session-detail routes each used to compute independently (polylogue-6o9b):
    the DB-backed route (``api/archive.py:_archive_message_to_domain``,
    reached via ``Polylogue.get_session()``) and the archive-backed route
    (``daemon/http.py:_archive_message_payload``) both read the same
    ``ArchiveMessageRow.blocks`` and must produce byte-identical text for
    the same message, since ``daemon/web_shell_reader.py``'s client-side
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
class ArchiveInsightMaterialization:
    insight_type: str
    session_id: str
    materializer_version: int
    materialized_at_ms: int
    source_updated_at_ms: int | None
    source_sort_key_ms: int | None
    input_high_water_mark_ms: int | None
    input_row_count: int
    input_high_water_mark_source: str | None = None


@dataclass(frozen=True, slots=True)
class SessionEventWriteResult:
    wrote_provider_usage_events: bool = False


@dataclass(frozen=True, slots=True)
class PreparedSessionRows:
    """Pure, off-writer-thread-computable row tuples for one session's
    full-replace write (polylogue-623q).

    Row PREPARATION -- converting a ``ParsedSession`` tree into the SQL row
    tuples ``_replace_full_session_messages_and_blocks`` inserts -- was
    measured happening entirely inside the writer hold (34-40s per 2,000
    normal raws, 165-287s on whale pages) while parse-side warm workers sat
    idle. ``prepare_session_rows`` builds this dataclass from nothing but a
    ``ParsedSession`` (no DB connection, no archive state), so it can run on
    a parse-prefetch worker thread; the writer then only needs to run
    ``executemany`` against already-built tuples.

    ``session_content_hash`` is ``pipeline.ids.session_content_hash(session)``
    hex-decoded -- computed from the ORIGINAL (pre-lineage-slice) session, the
    same value ordinary callers already pass as ``write_parsed_session_to_
    archive(content_hash=...)``. The writer compares this against its own
    ``content_hash`` argument before ever using the prepared rows: a session
    whose content changed (or whose caller didn't supply a content_hash --
    identity-only hashes never match) always falls back to preparing inline,
    which reproduces the exact unmodified write path. A session that turns
    out to need lineage tail-slicing (prefix-sharing composition against an
    already-archived parent -- resolved by ``_extract_prefix_tail``, which
    requires a live DB read the prefetch worker never had) is a SEPARATE
    rejection condition the writer checks independently, because slicing
    changes which messages are written without changing the session's own
    content hash.
    """

    session_id: str
    session_content_hash: bytes
    message_rows: tuple[tuple[object, ...], ...]
    block_rows: tuple[tuple[object, ...], ...]


def prepare_session_rows(session: ParsedSession) -> PreparedSessionRows:
    """Build ``PreparedSessionRows`` for ``session``'s full-replace write.

    Pure function: normalizes messages exactly as ``write_parsed_session_to_
    archive`` does for a non-merge-append, non-lineage-sliced write (see
    ``_normalized_messages``), then reuses the same row-tuple builders the
    writer itself calls (``_build_message_rows``/``_build_block_rows``) at
    ``position_offset=0`` -- the offset every full-replace write uses. No
    SQLite connection, network call, or filesystem access; safe to call from
    any thread, including a parse-prefetch worker running well before (and
    concurrently with) any writer hold.
    """
    from polylogue.pipeline.ids import session_content_hash as _compute_session_content_hash

    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    messages = _normalized_messages(session.messages)
    duplicate_native_ids = _duplicate_message_native_ids(messages)
    message_rows = _build_message_rows(session_id, messages, duplicate_native_ids=duplicate_native_ids)
    block_rows = _build_block_rows(session_id, messages, duplicate_native_ids=duplicate_native_ids)
    return PreparedSessionRows(
        session_id=session_id,
        session_content_hash=bytes.fromhex(_compute_session_content_hash(session)),
        message_rows=tuple(message_rows),
        block_rows=tuple(block_rows),
    )


def write_parsed_session_to_archive(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
    raw_id: str | None = None,
    merge_append: bool = False,
    force_replace: bool = False,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    signature_cache: dict[str, list[tuple[str, str]]] | None = None,
    preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]] | None = None,
    manage_transaction: bool = True,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
) -> str:
    """Write one parsed session into an initialized archive index DB.

    ``prepared`` (polylogue-623q, default ``None``) is an optional
    ``PreparedSessionRows`` computed off this thread (typically by the daemon
    parse-prefetch worker via ``prepare_session_rows``). It is used ONLY when
    ALL of the following hold, checked right before the full-replace write:
    not ``merge_append`` (prepared rows are built for a full replace's
    ``position_offset=0``, never an append's positive offset); lineage
    resolution did not slice ``messages`` (a prefix-sharing composition
    against an already-archived parent changes which messages get written,
    and the prefetch worker never had DB access to know about that parent);
    and ``prepared.session_content_hash`` matches this call's own
    ``content_hash``. Any other case silently falls back to preparing rows
    inline -- identical to ``prepared=None`` -- so passing a stale/irrelevant
    ``prepared`` is always safe, never incorrect.

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
    bulk-generation-build lifecycle this session write may be part of: a
    full source-to-index replay that always finishes with exactly one
    archive-wide repopulate of ``messages_fts``/``blocks_command_trigram``/
    ``action_pairs``/``delegation_facts`` before readiness (see
    ``maintenance/rebuild_index.py``). When ``True``, this write skips every
    per-session refresh of those four derived surfaces entirely (not just the
    guard-gated bulk delete+insert ``bulk_fts`` performs) -- the final
    repopulate covers every session regardless, so per-session work here is
    pure waste. Only the offline rebuild call passes ``True``; ordinary
    daemon writes (and ``bulk_fts=True`` used alone, e.g. in
    ``tests/unit/storage/test_bulk_fts_prefix_reextract.py``) keep those
    surfaces exactly in sync after every write, unchanged.

    ``defer_fts_rebuild`` is for an authoritative raw-revision replay that
    owns one targeted repair and exactness proof after its writes. It avoids
    rebuilding the same session's FTS surfaces twice in the same transaction.
    Direct callers retain the immediate-ready default.
    """
    t0 = time.perf_counter()

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
    parser_semantic_fingerprint = parser_fingerprint_for_origin(origin)
    lowering_semantic_fingerprint = lowering_fingerprint()
    # This session's own rows are about to be rewritten; drop any stale memoized
    # own-signatures so the batch cache never serves pre-write rows for it.
    if signature_cache is not None:
        signature_cache.pop(session_id, None)
    messages = _normalized_messages(session.messages)
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
    derived_created_at_ms, derived_updated_at_ms = _derive_session_timestamps_from_messages(messages)
    session_created_at_ms = _timestamp_ms(session.created_at)
    if session_created_at_ms is None:
        session_created_at_ms = derived_created_at_ms
    session_updated_at_ms = _timestamp_ms(session.updated_at)
    if session_updated_at_ms is None:
        session_updated_at_ms = derived_updated_at_ms
    # incoming_freshness_ms now reflects the same fallback: previously a
    # provider that omitted both session timestamps produced
    # incoming_freshness_ms=None, which unconditionally bypassed the
    # skip-stale-replace check below (freshness "unknown"). With derivation,
    # these sessions get a real freshness signal from their own message
    # evidence, so a genuinely older/stale replay of such a session is now
    # correctly skipped instead of always winning.
    incoming_freshness_ms = session_updated_at_ms or session_created_at_ms
    if not force_replace and not merge_append and incoming_freshness_ms is not None:
        row = conn.execute(
            "SELECT updated_at_ms FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        existing_updated_at_ms = int(row[0]) if row is not None and row[0] is not None else None
        if should_skip_stale_replace(
            incoming_freshness_ms=incoming_freshness_ms,
            existing_updated_at_ms=existing_updated_at_ms,
        ):
            add_timing("index.skip_stale_replace", t0)
            return session_id
    event_duplicate_message_native_ids = _duplicate_message_native_ids(messages)
    # Lineage normalization (#2467): when this is a prefix-sharing child whose
    # parent is already in the archive, drop the inherited prefix and keep only
    # the divergent tail. All downstream writes (messages, blocks, counts,
    # attachments, events) then operate on the tail, so each real message is
    # stored exactly once. Only applies to full-replace writes; merge-append is
    # an incremental extend of the same session.
    branch_point_message_id: str | None = None
    lineage_inheritance: str | None = None
    parent_session_id: str | None = None
    inherited_source_message_ids: dict[str, str] = {}
    if not merge_append:
        parent_session_id = _existing_parent_session_id(conn, session, origin.value)
        acompact = _is_claude_code_acompact_session(session)
        force_spawned_fresh = False
        if parent_session_id is not None and messages:
            parent_composed: list[tuple[str, str]] | None = None
            if acompact:
                parent_composed = _composed_db_signatures(conn, parent_session_id, cache=signature_cache)
                membership = _acompact_content_membership_ratio(
                    parent_composed,
                    _parsed_acompact_prefix_signatures(messages),
                )
                if membership is not None:
                    if membership < _ACOMPACT_PARENT_MEMBERSHIP_THRESHOLD:
                        session = session.model_copy(update={"branch_type": BranchType.SIDECHAIN})
                        lineage_inheritance = "spawned-fresh"
                        force_spawned_fresh = True
                    elif session.branch_type is BranchType.SIDECHAIN:
                        # Parent content is authoritative over a conservative
                        # fresh-head parser hint when both are available.
                        session = session.model_copy(update={"branch_type": BranchType.CONTINUATION})
                elif session.branch_type is BranchType.SIDECHAIN:
                    lineage_inheritance = "spawned-fresh"
                    force_spawned_fresh = True
            if not force_spawned_fresh:
                (
                    branch_point_message_id,
                    lineage_inheritance,
                    messages,
                    inherited_source_message_ids,
                ) = _extract_prefix_tail(
                    conn,
                    parent_session_id,
                    messages,
                    cache=signature_cache,
                    parent_composed=parent_composed,
                )
    duplicate_message_native_ids = _duplicate_message_native_ids(messages)
    active_leaf_message_id = _active_leaf_message_id(
        session_id,
        messages,
        session.active_leaf_message_provider_id,
        duplicate_native_ids=duplicate_message_native_ids,
    )
    session_content_hash = (
        bytes.fromhex(content_hash) if content_hash is not None else _hash_bytes("session", origin.value, native_id)
    )
    # polylogue-623q: only reuse rows prepared off this thread when NONE of
    # the conditions that would make them wrong hold -- see ``prepared``'s
    # docstring above. ``lineage_inheritance == "prefix-sharing"`` is the
    # single signal that ``_extract_prefix_tail`` sliced ``messages`` away
    # from what ``prepare_session_rows`` saw (it returns ``messages``
    # unchanged in every other case, including "spawned-fresh" and no-parent).
    prepared_rows_to_use: PreparedSessionRows | None = None
    if (
        prepared is not None
        and not merge_append
        and lineage_inheritance != "prefix-sharing"
        and prepared.session_content_hash == session_content_hash
    ):
        prepared_rows_to_use = prepared
    add_timing("index.prepare", t0)
    session_counts = _session_count_values(messages)

    # When the caller owns the transaction (bulk batching) we must not commit
    # per session; nullcontext leaves BEGIN/COMMIT to the caller.
    transaction = conn if manage_transaction else nullcontext()
    try:
        with transaction:
            conn.execute("INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES ('session-write')")
            if bulk_build:
                # polylogue-v6i3: gate messages_fts/blocks_command_trigram trigger
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
            # polylogue-cs86: distinct from ``existing_session_raw_id`` above --
            # a session row can exist with a NULL ``raw_id`` (ambiguous with
            # "no row at all" for the union-precedence check), but existence
            # of the row itself is unambiguous from ``fetchone()`` and is
            # exactly what ``_replace_full_session_messages_and_blocks`` needs
            # to know whether its ~14-table point-DELETE cascade has anything
            # to do at all.
            session_row_existed = False
            if not merge_append:
                existing_raw_id_row = conn.execute(
                    "SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if existing_raw_id_row is not None:
                    session_row_existed = True
                    existing_session_raw_id = existing_raw_id_row[0]
            t0 = time.perf_counter()
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint,
                    branch_type, active_leaf_message_id,
                    title, session_kind, title_source, title_ref, title_confidence,
                    display_name, run_settings_json, pending_drafts_json,
                    git_branch, git_repository_url, commit_hash,
                    instructions_text, reported_duration_ms, reported_cost_usd, provider_project_ref,
                    message_count, word_count, tool_use_count, thinking_count,
                    paste_count, user_message_count, authored_user_message_count,
                    assistant_message_count, system_message_count,
                    tool_message_count, user_word_count, authored_user_word_count, assistant_word_count,
                    content_hash, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(origin, native_id) DO UPDATE SET
                    raw_id = excluded.raw_id,
                    parser_fingerprint = excluded.parser_fingerprint,
                    lowering_fingerprint = excluded.lowering_fingerprint,
                    branch_type = excluded.branch_type,
                    active_leaf_message_id = excluded.active_leaf_message_id,
                    title = COALESCE(excluded.title, sessions.title),
                    session_kind = excluded.session_kind,
                    -- Plain overwrite, NOT COALESCE (polylogue-0cn3): title_source/
                    -- title_ref/title_confidence are a purely derived provenance
                    -- triple -- every parser branch that sets one of the three sets
                    -- all three together (see e.g. code_parser.py's title heuristic
                    -- chain), and no durable/user-authored path ever writes them
                    -- (no rename verb exists on any surface). A COALESCE here was a
                    -- ratchet: once a weaker parser stored title_source='unknown'
                    -- (a definite, non-NULL verdict), a later run of an *improved*
                    -- parser over the same content-hash-unchanged session would
                    -- never reach this UPDATE at all (the ingest batch's
                    -- content-hash idempotency check skips re-write entirely), and
                    -- even a batch that DID reach this UPDATE would have its better
                    -- verdict silently discarded by COALESCE picking the stale
                    -- 'unknown' only if excluded happened to be NULL -- which the
                    -- classification triple never is once cijx.4 made UNKNOWN an
                    -- explicit member rather than a Python None. Recomputing the
                    -- whole triple on every write is what actually lets a rebuild
                    -- (`polylogue ops reset --index && polylogued run`, the
                    -- documented remedy for a SEMANTIC_REPARSE-class parser fix)
                    -- re-evaluate every session instead of some subset staying
                    -- frozen at their first-ever verdict.
                    title_source = excluded.title_source,
                    title_ref = excluded.title_ref,
                    title_confidence = excluded.title_confidence,
                    display_name = COALESCE(excluded.display_name, sessions.display_name),
                    run_settings_json = COALESCE(excluded.run_settings_json, sessions.run_settings_json),
                    -- Plain overwrite, NOT COALESCE like run_settings_json above:
                    -- a draft is current mutable state, so a reprocess that finds
                    -- no non-blank pendingInputs (submitted, or cleared) must
                    -- actually clear the stored value rather than preserving a
                    -- now-stale draft forever (polylogue-o4j2).
                    pending_drafts_json = excluded.pending_drafts_json,
                    git_branch = excluded.git_branch,
                    git_repository_url = excluded.git_repository_url,
                    commit_hash = excluded.commit_hash,
                    provider_project_ref = excluded.provider_project_ref,
                    -- title/display_name/run_settings_json/instructions_text keep
                    -- their COALESCE (polylogue-0cn3 sibling audit): each is
                    -- sometimes genuinely omitted on a given write (e.g. an
                    -- append-only delta batch, or an origin whose parser doesn't
                    -- populate that field for this content) without that omission
                    -- meaning "clear the prior value" -- unlike the classification
                    -- triple above, no producer treats a bare Python None as an
                    -- authoritative "recomputed and confirmed absent" verdict for
                    -- these columns, so preserving the last real value on a NULL
                    -- write is correct, not a stale-verdict ratchet.
                    instructions_text = COALESCE(excluded.instructions_text, sessions.instructions_text),
                    reported_duration_ms = excluded.reported_duration_ms,
                    reported_cost_usd = excluded.reported_cost_usd,
                    content_hash = excluded.content_hash,
                    -- Reversed COALESCE (existing wins over excluded), unlike every
                    -- other column above: created_at_ms is a durable observed fact
                    -- (the session's real creation time), not a re-derivable
                    -- classification, so it is correctly ratcheted once set --
                    -- included in the polylogue-0cn3 sibling audit as the one
                    -- column that legitimately never moves.
                    created_at_ms = COALESCE(sessions.created_at_ms, excluded.created_at_ms),
                    updated_at_ms = CASE
                        WHEN ? THEN excluded.updated_at_ms
                        ELSE MAX(COALESCE(sessions.updated_at_ms, 0), COALESCE(excluded.updated_at_ms, 0))
                    END
                """,
                (
                    native_id,
                    origin.value,
                    raw_id,
                    parser_semantic_fingerprint,
                    lowering_semantic_fingerprint,
                    _enum_value(session.branch_type),
                    active_leaf_message_id,
                    _sqlite_text(session.title),
                    _enum_value(session.session_kind) or SessionKind.STANDARD.value,
                    _enum_value(session.title_source),
                    _sqlite_text(session.title_ref),
                    session.title_confidence,
                    _sqlite_text(session.display_name),
                    _json_dumps(session.run_settings) if session.run_settings else None,
                    _json_dumps(session.pending_drafts) if session.pending_drafts else None,
                    _sqlite_text(session.git_branch),
                    _sqlite_text(session.git_repository_url),
                    _sqlite_text(session.git_commit_hash),
                    _sqlite_text(session.instructions_text),
                    session.reported_duration_ms,
                    session.reported_cost_usd,
                    _sqlite_text(session.provider_project_ref),
                    session_counts["message_count"],
                    session_counts["word_count"],
                    session_counts["tool_use_count"],
                    session_counts["thinking_count"],
                    session_counts["paste_count"],
                    session_counts["user_message_count"],
                    session_counts["authored_user_message_count"],
                    session_counts["assistant_message_count"],
                    session_counts["system_message_count"],
                    session_counts["tool_message_count"],
                    session_counts["user_word_count"],
                    session_counts["authored_user_word_count"],
                    session_counts["assistant_word_count"],
                    session_content_hash,
                    session_created_at_ms,
                    session_updated_at_ms,
                    force_replace,
                ),
            )
            add_timing("index.session_upsert", t0)
            position_offset = 0
            stale_attachment_ids: set[str] = set()
            projection_carry_forward: _ProjectionCarryForward | None = None
            t0 = time.perf_counter()
            if merge_append:
                row = conn.execute(
                    "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                position_offset = int(row[0] or 0) if row is not None else 0
                _assert_unique_message_coordinates(session_id, messages, position_offset=position_offset)
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
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                )
                conn.execute(
                    "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = ?",
                    (active_leaf_message_id, session_id),
                )
                add_timing("index.merge_prepare", t0)
            else:
                stale_attachment_ids = _session_attachment_ids(conn, session_id)
                projection_carry_forward = _replace_full_session_messages_and_blocks(
                    conn,
                    session,
                    messages,
                    duplicate_native_ids=duplicate_message_native_ids,
                    raw_id=raw_id,
                    existing_raw_id=existing_session_raw_id,
                    session_row_existed=session_row_existed,
                    force_replace=force_replace,
                    stage_timings_s=stage_timings_s,
                    stage_timing_prefix=stage_timing_prefix,
                    bulk_build=bulk_build,
                    defer_fts_rebuild=defer_fts_rebuild,
                    prepared=prepared_rows_to_use,
                )
                add_timing("index.full_replace", t0)
            if merge_append:
                t0 = time.perf_counter()
                _write_messages(
                    conn,
                    session_id,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                )
                add_timing("index.messages", t0)
                t0 = time.perf_counter()
                _write_blocks(
                    conn,
                    session_id,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                )
                add_timing("index.blocks", t0)
                t0 = time.perf_counter()
                _write_file_edits(
                    conn,
                    session_id,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                )
                add_timing("index.file_edits", t0)
                t0 = time.perf_counter()
                if not bulk_build:
                    refresh_action_pairs(conn, session_id)
                add_timing("index.action_pairs", t0)
                t0 = time.perf_counter()
                _write_web_constructs(
                    conn,
                    session,
                    messages,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_message_native_ids,
                    replace_session=False,
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
                if projection_carry_forward is not None
                else set()
            )
            _write_attachments(
                conn,
                session_id,
                messages,
                session.attachments,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
                refresh_attachment_ids=stale_attachment_ids - carried_forward_attachment_ids,
                preacquired_blobs=preacquired_attachment_blobs,
            )
            add_timing("index.attachments", t0)
            t0 = time.perf_counter()
            _write_paste_spans(
                conn,
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
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
                add_timing("index.restore_projections", t0)
            t0 = time.perf_counter()
            _write_parent_links(
                conn,
                session_id,
                messages,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
            )
            add_timing("index.parent_links", t0)
            t0 = time.perf_counter()
            _write_session_link(
                conn,
                session_id,
                session,
                branch_point_message_id=branch_point_message_id,
                inheritance=lineage_inheritance,
            )
            add_timing("index.session_link", t0)
            t0 = time.perf_counter()
            event_position_offset = _next_session_event_position(conn, session_id)
            provider_usage_baseline = (
                _provider_usage_cumulative_baseline(conn, parent_session_id, branch_point_message_id)
                if parent_session_id is not None
                and branch_point_message_id is not None
                and lineage_inheritance == "prefix-sharing"
                else None
            )
            session_event_result = _write_session_events(
                conn,
                session_id,
                messages,
                session.session_events,
                position_offset=position_offset,
                event_position_offset=event_position_offset,
                duplicate_native_ids=duplicate_message_native_ids,
                provider_usage_baseline=provider_usage_baseline,
                inherited_source_message_ids=inherited_source_message_ids,
                ambiguous_source_provider_ids=event_duplicate_message_native_ids,
            )
            add_timing("index.session_events", t0)
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
            if merge_append:
                _increment_session_counts_for_append(conn, session_id, session_counts)
            else:
                _refresh_session_counts(conn, session_id)
            add_timing("index.session_counts", t0)
            t0 = time.perf_counter()
            _resolve_session_graph(
                conn,
                session_id,
                native_id,
                origin.value,
                cache=signature_cache,
                add_timing=add_timing,
                bulk_fts=bulk_fts,
                bulk_build=bulk_build,
            )
            add_timing("index.graph_resolve", t0)
            t0 = time.perf_counter()
            if not bulk_build:
                refresh_delegation_facts_for_session(conn, session_id)
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
                _replace_ingest_flag_tags(conn, session_id, session.ingest_flags)
                add_timing("index.ingest_flags", t0)
    except sqlite3.IntegrityError as exc:
        raise sqlite3.IntegrityError(
            f"FOREIGN KEY constraint failed writing session_id={session_id!r} "
            f"origin={origin.value!r} native_id={native_id!r}: {exc}"
        ) from exc
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
    for table in (
        "action_pairs",
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
    conn.execute("DELETE FROM session_links WHERE src_session_id = ?", (session_id,))


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


def upsert_session_profile_costs(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    cost_credits: float | None = None,
    cost_usd: float | None = None,
    cost_is_estimated: bool = False,
    cost_provenance: str | None = None,
    priced_with: str | None = None,
    priced_at_ms: int | None = None,
) -> None:
    """Upsert a minimal cost/pricing slice for an existing profile row."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, cost_credits, cost_usd, cost_is_estimated, cost_provenance, priced_with, priced_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cost_credits = excluded.cost_credits,
                cost_usd = excluded.cost_usd,
                cost_is_estimated = excluded.cost_is_estimated,
                cost_provenance = excluded.cost_provenance,
                priced_with = excluded.priced_with,
                priced_at_ms = excluded.priced_at_ms
            """,
            (
                session_id,
                cost_credits,
                cost_usd,
                1 if cost_is_estimated else 0,
                cost_provenance,
                priced_with,
                priced_at_ms,
            ),
        )


def apply_insight_materialization(
    conn: sqlite3.Connection,
    *,
    insight_type: str,
    session_id: str,
    materializer_version: int,
    materialized_at_ms: int,
    source_updated_at_ms: int | None = None,
    source_sort_key_ms: int | None = None,
    input_high_water_mark_ms: int | None = None,
    input_high_water_mark_source: str | None = None,
    input_row_count: int = 0,
) -> None:
    """Stamp one session-insight materialization row without committing.

    The bulk insight rebuild (``rebuild_session_insights_sync``) materializes
    every insight table inside one transaction so a SIGKILL mid-rebuild rolls
    the WAL back to the prior insights. It therefore stamps materialization
    through this no-commit primitive; the committing ``upsert_*`` wrapper below
    is for callers that stamp a single insight as its own unit of work.
    """
    conn.execute(
        """
        INSERT INTO insight_materialization (
            insight_type, session_id, materializer_version, materialized_at_ms,
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms,
            input_high_water_mark_source, input_row_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(insight_type, session_id) DO UPDATE SET
            materializer_version = excluded.materializer_version,
            materialized_at_ms = excluded.materialized_at_ms,
            source_updated_at_ms = excluded.source_updated_at_ms,
            source_sort_key_ms = excluded.source_sort_key_ms,
            input_high_water_mark_ms = excluded.input_high_water_mark_ms,
            input_high_water_mark_source = excluded.input_high_water_mark_source,
            input_row_count = excluded.input_row_count
        """,
        (
            insight_type,
            session_id,
            materializer_version,
            materialized_at_ms,
            source_updated_at_ms,
            source_sort_key_ms,
            input_high_water_mark_ms,
            input_high_water_mark_source,
            input_row_count,
        ),
    )


def upsert_insight_materialization(
    conn: sqlite3.Connection,
    *,
    insight_type: str,
    session_id: str,
    materializer_version: int,
    materialized_at_ms: int,
    source_updated_at_ms: int | None = None,
    source_sort_key_ms: int | None = None,
    input_high_water_mark_ms: int | None = None,
    input_high_water_mark_source: str | None = None,
    input_row_count: int = 0,
) -> ArchiveInsightMaterialization:
    """Upsert the shared materialization state for one session insight."""
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        apply_insight_materialization(
            conn,
            insight_type=insight_type,
            session_id=session_id,
            materializer_version=materializer_version,
            materialized_at_ms=materialized_at_ms,
            source_updated_at_ms=source_updated_at_ms,
            source_sort_key_ms=source_sort_key_ms,
            input_high_water_mark_ms=input_high_water_mark_ms,
            input_high_water_mark_source=input_high_water_mark_source,
            input_row_count=input_row_count,
        )
    return read_insight_materialization(conn, insight_type, session_id)


def read_insight_materialization(
    conn: sqlite3.Connection,
    insight_type: str,
    session_id: str,
) -> ArchiveInsightMaterialization:
    """Read the shared materialization state for one session insight."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT insight_type, session_id, materializer_version, materialized_at_ms,
            source_updated_at_ms, source_sort_key_ms, input_high_water_mark_ms,
            input_high_water_mark_source, input_row_count
        FROM insight_materialization
        WHERE insight_type = ? AND session_id = ?
        """,
        (insight_type, session_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"{insight_type}:{session_id}")
    return ArchiveInsightMaterialization(
        insight_type=row["insight_type"],
        session_id=row["session_id"],
        materializer_version=row["materializer_version"],
        materialized_at_ms=row["materialized_at_ms"],
        source_updated_at_ms=row["source_updated_at_ms"],
        source_sort_key_ms=row["source_sort_key_ms"],
        input_high_water_mark_ms=row["input_high_water_mark_ms"],
        input_high_water_mark_source=row["input_high_water_mark_source"],
        input_row_count=row["input_row_count"],
    )


def read_archive_session_envelope(
    conn: sqlite3.Connection, session_id: str, *, _depth: int = 0
) -> ArchiveSessionEnvelope:
    """Read a compact archive envelope, holding one read snapshot across composition.

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
    recursive calls see ``conn.in_transaction`` already true and skip
    re-wrapping.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN DEFERRED")
        try:
            return read_archive_session_envelope(conn, session_id, _depth=_depth)
        finally:
            conn.execute("ROLLBACK")
    conn.row_factory = sqlite3.Row
    session = conn.execute(
        """
        SELECT session_id, native_id, origin, title, session_kind, active_leaf_message_id,
               parent_session_id, root_session_id, branch_type,
               title_source, title_ref, title_confidence, instructions_text,
               created_at_ms, updated_at_ms, git_branch, git_repository_url, provider_project_ref,
               reported_cost_usd
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        raise KeyError(session_id)
    working_directories = tuple(
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

    attachment_rows = conn.execute(
        """
        SELECT r.message_id AS message_id, a.attachment_id AS attachment_id,
               a.display_name AS display_name, a.media_type AS media_type, a.byte_count AS byte_count,
               r.upload_origin AS upload_origin, r.source_url AS source_url, r.caption AS caption
        FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ?
        ORDER BY r.message_id, a.attachment_id
        """,
        (session_id,),
    ).fetchall()
    attachments_by_message: dict[str | None, list[ArchiveAttachmentRow]] = {}
    for attachment in attachment_rows:
        attachments_by_message.setdefault(attachment["message_id"], []).append(
            ArchiveAttachmentRow(
                attachment_id=attachment["attachment_id"],
                message_id=attachment["message_id"],
                display_name=attachment["display_name"],
                media_type=attachment["media_type"],
                byte_count=int(attachment["byte_count"] or 0),
                upload_origin=attachment["upload_origin"],
                source_url=attachment["source_url"],
                caption=attachment["caption"],
            )
        )

    message_rows = conn.execute(
        """
        SELECT message_id, native_id, role, position, variant_index, is_active_path, is_active_leaf,
               message_type, material_origin, word_count, has_tool_use, has_thinking, has_paste, occurred_at_ms,
               paste_boundary AS paste_boundary_state, duration_ms, parent_message_id, stop_reason
        FROM messages
        WHERE session_id = ?
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    messages: list[ArchiveMessageRow] = []
    for message in message_rows:
        block_rows = conn.execute(
            """
            SELECT block_id, message_id, block_type, text, tool_name, tool_id, semantic_type,
                   tool_input, language, tool_result_is_error, tool_result_exit_code,
                   tool_result_outcome_unknown_reason
            FROM blocks
            WHERE message_id = ?
            ORDER BY position
            """,
            (message["message_id"],),
        ).fetchall()
        messages.append(
            ArchiveMessageRow(
                message_id=message["message_id"],
                native_id=message["native_id"],
                role=message["role"],
                position=message["position"],
                variant_index=message["variant_index"],
                is_active_path=bool(message["is_active_path"]),
                is_active_leaf=bool(message["is_active_leaf"]),
                blocks=tuple(
                    ArchiveBlockRow(
                        block_id=block["block_id"],
                        message_id=block["message_id"],
                        block_type=block["block_type"],
                        text=block["text"],
                        tool_name=block["tool_name"],
                        tool_id=block["tool_id"],
                        semantic_type=block["semantic_type"],
                        tool_input=block["tool_input"],
                        language=block["language"],
                        tool_result_is_error=block["tool_result_is_error"],
                        tool_result_exit_code=block["tool_result_exit_code"],
                        tool_result_outcome_unknown_reason=block["tool_result_outcome_unknown_reason"],
                    )
                    for block in block_rows
                ),
                message_type=message["message_type"],
                material_origin=message["material_origin"],
                word_count=int(message["word_count"] or 0),
                has_tool_use=bool(message["has_tool_use"]),
                has_thinking=bool(message["has_thinking"]),
                has_paste=bool(message["has_paste"]),
                paste_boundary_state=message["paste_boundary_state"],
                occurred_at=_iso_from_ms(message["occurred_at_ms"]),
                duration_ms=int(message["duration_ms"] or 0),
                parent_message_id=message["parent_message_id"],
                attachments=tuple(attachments_by_message.get(message["message_id"], ())),
                source_session_id=str(session["session_id"]),
                stop_reason=message["stop_reason"],
            )
        )

    # Lineage composition (#2467): prepend the parent's composed transcript up to
    # and including the branch point. The parent envelope is itself composed via
    # this same recursion, so nested lineages resolve correctly.
    #
    # 4ts.6: two paths silently return an INCOMPLETE transcript with no signal
    # -- a depth-limit cutoff (a chain deeper than _MAX_LINEAGE_DEPTH) and a
    # dangling branch point (the parent message was hard-deleted, so the
    # child's own tail is returned starting mid-conversation). Track both and
    # surface them on the envelope rather than silently serving a partial
    # transcript as if it were whole. A parent's own incompleteness (from a
    # DEEPER recursion level) also propagates up, since this session's
    # composed view includes that parent's transcript.
    lineage_complete = True
    lineage_truncation_reason: LineageTruncationReason | None = None
    lineage_inheritance = "none"
    lineage_branch_point_message_id: str | None = None
    edge = _prefix_sharing_edge_sync(conn, str(session["session_id"]))
    if edge is not None:
        lineage_inheritance = "prefix-sharing"
        parent_session_id, lineage_branch_point_message_id = edge
        if _depth >= _MAX_LINEAGE_DEPTH:
            lineage_complete = False
            lineage_truncation_reason = LINEAGE_TRUNCATION_DEPTH_LIMIT
            logger.warning(
                "lineage composition hit depth limit (%d) for session %s; ancestors beyond this depth are dropped",
                _MAX_LINEAGE_DEPTH,
                session["session_id"],
            )
        else:
            parent_envelope = read_archive_session_envelope(conn, parent_session_id, _depth=_depth + 1)
            parent_messages = parent_envelope.messages
            prefix: list[ArchiveMessageRow] = []
            found = False
            for parent_message in parent_messages:
                prefix.append(parent_message)
                if parent_message.message_id == lineage_branch_point_message_id:
                    found = True
                    break
            # Dangling branch point (parent message hard-deleted): keep this
            # session's own tail rather than splice the entire parent (#2467 audit).
            if found:
                messages = prefix + messages
            # Check the parent's OWN incompleteness first: if the parent was
            # already truncated (e.g. by the depth limit), its composed
            # messages may not include its own inherited prefix, so a
            # not-found branch point here is a SYMPTOM of that truncation,
            # not an independent dangling-branch-point condition. Surfacing
            # the parent's real reason (not a locally-derived one) avoids
            # masking the root cause one level up.
            if not parent_envelope.lineage_complete:
                lineage_complete = False
                lineage_truncation_reason = parent_envelope.lineage_truncation_reason
            elif not found:
                lineage_complete = False
                lineage_truncation_reason = LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT

    return ArchiveSessionEnvelope(
        session_id=session["session_id"],
        native_id=session["native_id"],
        origin=session["origin"],
        title=session["title"],
        session_kind=session["session_kind"],
        active_leaf_message_id=session["active_leaf_message_id"],
        messages=tuple(messages),
        lineage_complete=lineage_complete,
        lineage_truncation_reason=lineage_truncation_reason,
        lineage_inheritance=lineage_inheritance,
        lineage_branch_point_message_id=lineage_branch_point_message_id,
        parent_session_id=session["parent_session_id"],
        root_session_id=session["root_session_id"],
        branch_type=session["branch_type"],
        title_source=session["title_source"],
        title_ref=session["title_ref"],
        title_confidence=session["title_confidence"],
        instructions_text=session["instructions_text"],
        created_at=_iso_from_ms(session["created_at_ms"]),
        updated_at=_iso_from_ms(session["updated_at_ms"]),
        working_directories=working_directories,
        git_branch=session["git_branch"],
        git_repository_url=session["git_repository_url"],
        provider_project_ref=session["provider_project_ref"],
        reported_cost_usd=session["reported_cost_usd"],
        orphan_attachments=tuple(attachments_by_message.get(None, ())),
    )


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
    )


def _fetch_message_window(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    offset: int,
    limit: int,
    upto_position: int | None = None,
    upto_variant_index: int | None = None,
) -> list[ArchiveMessageRow]:
    """Bounded ``[offset, offset + limit)`` window of a session's OWN rows.

    Batches block/attachment reads for exactly the returned messages instead
    of the full session's N+1-per-message pattern in
    ``read_archive_session_envelope``, since the window is small by
    construction (``read_archive_session_page``'s whole point).
    """
    if limit <= 0:
        return []
    upto_clause = ""
    upto_params: tuple[int, int] | tuple[()] = ()
    if upto_position is not None and upto_variant_index is not None:
        upto_clause = " AND (position, variant_index) <= (?, ?)"
        upto_params = (upto_position, upto_variant_index)
    message_rows = conn.execute(
        f"""
        SELECT message_id, native_id, role, position, variant_index, is_active_path, is_active_leaf,
               message_type, material_origin, word_count, has_tool_use, has_thinking, has_paste, occurred_at_ms,
               paste_boundary AS paste_boundary_state, duration_ms, parent_message_id, stop_reason
        FROM messages
        WHERE session_id = ?{upto_clause}
        ORDER BY position, variant_index
        LIMIT ? OFFSET ?
        """,
        (session_id, *upto_params, max(limit, 0), max(offset, 0)),
    ).fetchall()
    if not message_rows:
        return []
    message_ids = [row["message_id"] for row in message_rows]
    placeholders = ",".join("?" for _ in message_ids)
    block_rows = conn.execute(
        f"""
        SELECT block_id, message_id, block_type, text, tool_name, tool_id, semantic_type,
               tool_input, language, tool_result_is_error, tool_result_exit_code,
               tool_result_outcome_unknown_reason
        FROM blocks
        WHERE message_id IN ({placeholders})
        ORDER BY message_id, position
        """,
        message_ids,
    ).fetchall()
    blocks_by_message: dict[str, list[ArchiveBlockRow]] = {}
    for block in block_rows:
        blocks_by_message.setdefault(block["message_id"], []).append(
            ArchiveBlockRow(
                block_id=block["block_id"],
                message_id=block["message_id"],
                block_type=block["block_type"],
                text=block["text"],
                tool_name=block["tool_name"],
                tool_id=block["tool_id"],
                semantic_type=block["semantic_type"],
                tool_input=block["tool_input"],
                language=block["language"],
                tool_result_is_error=block["tool_result_is_error"],
                tool_result_exit_code=block["tool_result_exit_code"],
                tool_result_outcome_unknown_reason=block["tool_result_outcome_unknown_reason"],
            )
        )
    attachment_rows = conn.execute(
        f"""
        SELECT r.message_id AS message_id, a.attachment_id AS attachment_id,
               a.display_name AS display_name, a.media_type AS media_type, a.byte_count AS byte_count,
               r.upload_origin AS upload_origin, r.source_url AS source_url, r.caption AS caption
        FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.message_id IN ({placeholders})
        ORDER BY r.message_id, a.attachment_id
        """,
        message_ids,
    ).fetchall()
    attachments_by_message: dict[str, list[ArchiveAttachmentRow]] = {}
    for attachment in attachment_rows:
        attachments_by_message.setdefault(attachment["message_id"], []).append(
            ArchiveAttachmentRow(
                attachment_id=attachment["attachment_id"],
                message_id=attachment["message_id"],
                display_name=attachment["display_name"],
                media_type=attachment["media_type"],
                byte_count=int(attachment["byte_count"] or 0),
                upload_origin=attachment["upload_origin"],
                source_url=attachment["source_url"],
                caption=attachment["caption"],
            )
        )
    return [
        _row_to_archive_message(
            row,
            session_id,
            blocks=tuple(blocks_by_message.get(row["message_id"], ())),
            attachments=tuple(attachments_by_message.get(row["message_id"], ())),
        )
        for row in message_rows
    ]


def read_archive_session_page(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> ArchiveSessionEnvelope:
    """Read a bounded ``[offset, offset + limit)`` PAGE of a session's transcript.

    For an ordinary (non-lineage) session this composes only the requested
    message window at the SQL layer -- header, that window's messages, their
    blocks/attachments, and orphan attachments -- so first paint of a large
    session is bounded by the page size, not the session's total message
    count (polylogue-07g6).

    A prefix-sharing lineage child (#2467) still requires the full composed
    parent-prefix + own-tail transcript to slice a display window correctly,
    since the child's own rows are only its divergent tail -- the exact same
    constraint ``get_messages_paginated`` already documents and accepts for
    the DB-backed reader (#2470). This mirrors that established fallback
    (full composition, sliced in Python) rather than inventing a different
    contract for the archive-backed reader; genuinely bounding the lineage
    case is tracked as a follow-up.

    ``total_message_count`` on the returned envelope always carries the TRUE
    composed transcript length; ``messages`` holds only the requested
    window.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN DEFERRED")
        try:
            return read_archive_session_page(conn, session_id, limit=limit, offset=offset)
        finally:
            conn.execute("ROLLBACK")
    conn.row_factory = sqlite3.Row

    if _prefix_sharing_edge_sync(conn, session_id) is not None:
        full = read_archive_session_envelope(conn, session_id)
        window = full.messages[offset : offset + limit] if limit > 0 else ()
        return replace(full, messages=window, total_message_count=len(full.messages))

    session = conn.execute(
        """
        SELECT session_id, native_id, origin, title, session_kind, active_leaf_message_id,
               parent_session_id, root_session_id, branch_type,
               title_source, title_ref, title_confidence, instructions_text,
               created_at_ms, updated_at_ms, git_branch, git_repository_url, provider_project_ref,
               reported_cost_usd
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        raise KeyError(session_id)
    working_directories = tuple(
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
    total_message_count = int(
        conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    )
    messages = _fetch_message_window(conn, session_id, offset=offset, limit=limit)
    orphan_attachment_rows = conn.execute(
        """
        SELECT a.attachment_id AS attachment_id, a.display_name AS display_name, a.media_type AS media_type,
               a.byte_count AS byte_count, r.upload_origin AS upload_origin, r.source_url AS source_url,
               r.caption AS caption
        FROM attachment_refs r
        JOIN attachments a ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ? AND r.message_id IS NULL
        ORDER BY a.attachment_id
        """,
        (session_id,),
    ).fetchall()
    orphan_attachments = tuple(
        ArchiveAttachmentRow(
            attachment_id=row["attachment_id"],
            message_id=None,
            display_name=row["display_name"],
            media_type=row["media_type"],
            byte_count=int(row["byte_count"] or 0),
            upload_origin=row["upload_origin"],
            source_url=row["source_url"],
            caption=row["caption"],
        )
        for row in orphan_attachment_rows
    )
    return ArchiveSessionEnvelope(
        session_id=session["session_id"],
        native_id=session["native_id"],
        origin=session["origin"],
        title=session["title"],
        session_kind=session["session_kind"],
        active_leaf_message_id=session["active_leaf_message_id"],
        messages=tuple(messages),
        lineage_complete=True,
        lineage_truncation_reason=None,
        lineage_inheritance="none",
        lineage_branch_point_message_id=None,
        parent_session_id=session["parent_session_id"],
        root_session_id=session["root_session_id"],
        branch_type=session["branch_type"],
        title_source=session["title_source"],
        title_ref=session["title_ref"],
        title_confidence=session["title_confidence"],
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
    """Return block ids matched by the archive contentless FTS table."""
    match_query = normalize_fts5_query(query)
    if match_query is None:
        return []
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
        INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, {pl_fold_sql_expr("search_text")}
        FROM blocks
        WHERE search_text != ''
        """
    )
    row = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()
    return int(row[0] if row is not None else 0)


def _build_message_rows(
    session_id: str,
    messages: list[ParsedMessage],
    *,
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
    rows: list[tuple[object, ...]] = []
    for fallback_position, message in enumerate(messages):
        position = position_offset + (message.position if message.position is not None else fallback_position)
        variant_index = message.variant_index if message.variant_index is not None else 0
        values: dict[str, object] = {
            "session_id": session_id,
            "native_id": _stored_message_native_id(message, duplicate_native_ids),
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
            "content_hash": _message_content_hash(session_id, message, position=position, variant_index=variant_index),
            "occurred_at_ms": message.occurred_at_ms
            if message.occurred_at_ms is not None
            else _timestamp_ms(message.timestamp),
            "stop_reason": _enum_value(message.stop_reason),
        }
        rows.append(archive_tiers_specs.MESSAGES_SPEC.extract_tuple(values))
    return rows


def _messages_insert_sql() -> str:
    spec = archive_tiers_specs.MESSAGES_SPEC
    return f"""
        INSERT OR REPLACE INTO messages (
            {spec.insert_column_names}
        ) VALUES ({spec.insert_placeholder_string})
        """


def _write_messages(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    rows: list[tuple[object, ...]] | None = None,
) -> None:
    """Write message rows using table-driven column specification.

    The messages table column spec (archive_tiers_specs.MESSAGES_SPEC) defines:
      - writable_columns: the ordered list of columns to INSERT (29 total)
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
        rows = _build_message_rows(
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
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
    embeddings are keyed by ``embedding_input_hash`` (identity-FREE --
    ``storage/embeddings/identity.py``), computed straight from the
    embedder's input text, so a rebuild or lineage-normalization shift that
    changes this hash without changing the actual text no longer forces a
    wasted re-embed.
    """

    block_parts: list[str] = []
    for block in _message_blocks(message):
        block_parts.extend(
            (
                _block_type(block).value,
                _sqlite_text(block.text) or "",
                _sqlite_text(block.tool_name) or "",
                _sqlite_text(block.tool_id) or "",
                _json_dumps(block.tool_input) if block.tool_input is not None else "",
                _sqlite_text(_semantic_type(block)) or "",
                _sqlite_text(block.media_type) or "",
                _sqlite_text(_block_language(block)) or "",
                "" if block.is_error is None else str(int(block.is_error)),
                "" if block.exit_code is None else str(block.exit_code),
            )
        )
    return _hash_bytes(
        "message",
        session_id,
        message.provider_message_id or "",
        str(position),
        str(variant_index),
        _enum_value(message.role) or "",
        _enum_value(message.message_type) or "",
        _enum_value(message.material_origin) or "",
        _sqlite_text(message.text) or "",
        _sqlite_text(message.user_context_text) or "",
        _enum_value(message.stop_reason) or "",
        *block_parts,
    )


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
    outcome_unknown_reason: str | None = None,
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
        _sqlite_text(text) or "",
        _sqlite_text(tool_name) or "",
        tool_input_json or "",
        _sqlite_text(semantic_type) or "",
        _sqlite_text(media_type) or "",
        _sqlite_text(language) or "",
        "" if is_error is None else str(int(is_error)),
        "" if exit_code is None else str(exit_code),
        outcome_unknown_reason or "",
    )


def _build_block_rows(
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> list[tuple[object, ...]]:
    """Pure row-tuple builder for the ``blocks`` table (no DB access).

    Extracted from ``_write_blocks`` (polylogue-623q); see
    ``_build_message_rows`` for why this split exists.
    """
    rows: list[tuple[object, ...]] = []
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        blocks = _message_blocks(message)
        for position, block in enumerate(blocks):
            block_type = _block_type(block)
            tool_input_json = _json_dumps(block.tool_input) if block.tool_input is not None else None
            semantic_type = _semantic_type(block)
            language = _block_language(block)
            is_error = getattr(block, "is_error", None)
            exit_code = getattr(block, "exit_code", None)
            outcome_unknown_reason = _enum_value(block.outcome_unknown_reason)
            signature = getattr(block, "signature", None)
            values: dict[str, object] = {
                "message_id": message_id,
                "session_id": session_id,
                "position": position,
                "block_type": block_type.value,
                "text": _sqlite_text(block.text),
                "tool_name": _sqlite_text(block.tool_name),
                "tool_id": _sqlite_text(block.tool_id),
                "tool_input": tool_input_json,
                "semantic_type": _sqlite_text(semantic_type),
                "media_type": _sqlite_text(block.media_type),
                "language": _sqlite_text(language),
                "tool_result_is_error": _sqlite_bool(is_error),
                "tool_result_exit_code": exit_code,
                "tool_result_outcome_unknown_reason": outcome_unknown_reason,
                "signature": _sqlite_text(signature),
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
                    outcome_unknown_reason=outcome_unknown_reason,
                ),
            }
            rows.append(archive_tiers_specs.BLOCKS_SPEC.extract_tuple(values))
    return rows


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
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    rows: list[tuple[object, ...]] | None = None,
) -> None:
    """Write block rows using table-driven column specification.

    The blocks table column spec (archive_tiers_specs.BLOCKS_SPEC) defines:
      - writable_columns: the ordered list of columns to INSERT (16 total)
      - The column names and placeholders are generated from the spec
      - The tuple order is derived from the spec's writable_columns order

    This consolidates the hand-aligned duplicates into a single source of truth.

    ``rows`` (polylogue-623q): see ``_write_messages`` for the contract --
    used verbatim when provided, otherwise built fresh from ``messages``.
    """
    if rows is None:
        rows = _build_block_rows(
            session_id,
            messages,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
    conn.executemany(_blocks_insert_sql(), rows)


def _build_file_edit_rows(
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> list[tuple[object, ...]]:
    """Pure row-tuple builder for ``file_edits`` (polylogue-2qx.4).

    ``ParsedFileEdit`` arrives attached to the TOOL_RESULT block that reports
    the edit outcome (matching where the provider's own structuredPatch/
    originalFile fields live on the wire), but ``file_edits`` is keyed by the
    TOOL_USE block that made the call -- resolved here via the shared
    ``tool_id``, exactly as the ``actions`` view pairs tool_use<->tool_result.
    A TOOL_RESULT carrying ``file_edit`` with no matching TOOL_USE in this
    same write (should not happen for a well-formed transcript) is dropped
    rather than guessing a key.
    """
    tool_use_block_id_by_tool_id: dict[str, str] = {}
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for position, block in enumerate(_message_blocks(message)):
            if _block_type(block) is BlockType.TOOL_USE and block.tool_id:
                tool_use_block_id_by_tool_id[block.tool_id] = f"{message_id}:{position}"

    rows: list[tuple[object, ...]] = []
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for block in _message_blocks(message):
            file_edit = getattr(block, "file_edit", None)
            if file_edit is None or not block.tool_id:
                continue
            tool_use_block_id = tool_use_block_id_by_tool_id.get(block.tool_id)
            if tool_use_block_id is None:
                continue
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
                    message.occurred_at_ms if message.occurred_at_ms is not None else _timestamp_ms(message.timestamp),
                )
            )
    return rows


def _write_file_edits(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    rows = _build_file_edit_rows(
        session_id,
        messages,
        position_offset=position_offset,
        duplicate_native_ids=duplicate_native_ids,
    )
    if rows:
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


def _write_session_refs(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    """Write tracker-agnostic session references (polylogue-2qx.4).

    Full-replace-only (mirrors ``_write_working_dirs``/``_write_repo_edges``):
    ``session_refs`` rows are re-derived from ``session.session_refs`` on
    every write, not appended incrementally, since the parser always emits
    the complete current set for a session.
    """
    observed_at_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
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
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    replace_session: bool = True,
) -> None:
    origin = origin_from_provider(session.source_name)
    session_id = archive_session_id(origin.value, session.provider_session_id)
    provider = _enum_value(session.source_name)
    rows: list[tuple[object, ...]] = []
    block_ids: list[str] = []
    # Iterate the (possibly lineage-sliced) tail messages, not session.messages —
    # a web construct on an inherited-prefix message would FK-violate against rows
    # that were never written under this session (#2467 audit).
    for fallback_position, message in enumerate(messages):
        message_id = _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        blocks = _message_blocks(message)
        for block_position, block in enumerate(blocks):
            block_id = f"{message_id}:{block_position}"
            if not replace_session:
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

    if replace_session:
        conn.execute("DELETE FROM web_content_constructs WHERE session_id = ?", (session_id,))
    else:
        conn.executemany(
            "DELETE FROM web_content_constructs WHERE block_id = ?", ((block_id,) for block_id in block_ids)
        )

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


def _splice_merge_keys(old_keys: list[object], new_keys: list[object]) -> list[tuple[str, object]]:
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


def _block_structural_keys(rows: list[tuple[object, ...]], b_idx: dict[str, int]) -> list[object]:
    """Content-addressed identity for a message's block sequence, NOT position.

    polylogue-geop PR review: position shifts when a non-trailing block is
    dropped (``[text, code, text]`` -> ``[text, text]`` shifts the trailing
    text from position 2 to position 1), so matching by raw position treats
    the shifted survivor as the omitted block's old occupant and corrupts
    both. ``svfj`` (the block evidence hash) already excludes position and
    ``tool_id`` from a single block's identity for the same reason a block
    can shift -- this extends that to sequence-level matching: prefer
    ``tool_id`` (a provider-assigned id, stable across independent export
    downloads of the same conversation) when present, else fall back to
    "the Nth block of this type seen so far in this message", which is
    exactly what distinguishes the two ``text`` blocks in the example above.
    """
    keys: list[object] = []
    occurrence: dict[object, int] = {}
    for row in rows:
        block_type = row[b_idx["block_type"]]
        tool_id = row[b_idx["tool_id"]]
        if tool_id:
            keys.append(("tool_id", block_type, tool_id))
        else:
            n = occurrence.get(block_type, 0)
            occurrence[block_type] = n + 1
            keys.append(("occurrence", block_type, n))
    return keys


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
        if col_name == "tool_input":
            new_raw = cast("str | None", new_row[idx])
            old_raw = cast("str | None", old_row[idx])
            new_json = json.loads(new_raw) if new_raw is not None else None
            old_json = json.loads(old_raw) if old_raw is not None else None
            merged_json = _merge_json_value(new_json, old_json, context=conflict_context)
            merged_values[idx] = _json_dumps(merged_json) if merged_json is not None else None
        else:
            merged_values[idx] = _coalesce_scalar(new_row[idx], old_row[idx])
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
        outcome_unknown_reason=cast("str | None", merged_values[b_idx["tool_result_outcome_unknown_reason"]]),
    )
    return tuple(merged_values)


def _message_content_hash_from_rows(
    session_id: str,
    native_id: str,
    position: int,
    variant_index: int,
    role: str | None,
    message_type: str | None,
    material_origin: str | None,
    user_context_text: str | None,
    stop_reason: str | None,
    block_rows: list[tuple[object, ...]],
    b_idx: dict[str, int],
) -> bytes:
    """Row-tuple analog of ``_message_content_hash`` for a message whose
    final block set was changed by field-path union (polylogue-geop PR
    review P2): a merged message must not keep the incoming row's hash and
    zero/tool-use/thinking flags once its blocks were reconciled against
    what an older acquisition supplied, or the stored hash and flags
    describe content that no longer matches the stored blocks.

    ``message.text`` (the ``ParsedMessage``'s own free-text field) has no
    stored column and cannot be recovered at this row-tuple layer -- this
    treats it as empty, consistently across every row this function
    computes. That makes the result NOT bit-identical to what a normal
    parse-time write would hash for the same final text, but the docstring
    on ``_message_content_hash`` already notes embedding freshness is keyed
    off ``embedding_input_hash`` instead: this remains a row-level
    change-detection signal, not a content-integrity guarantee, and this is
    a bounded, understood narrowing of it -- not silent staleness.
    """
    block_parts: list[str] = []
    for row in block_rows:
        is_error = row[b_idx["tool_result_is_error"]]
        exit_code = row[b_idx["tool_result_exit_code"]]
        block_parts.extend(
            (
                cast(str, row[b_idx["block_type"]]),
                cast("str | None", row[b_idx["text"]]) or "",
                cast("str | None", row[b_idx["tool_name"]]) or "",
                cast("str | None", row[b_idx["tool_id"]]) or "",
                cast("str | None", row[b_idx["tool_input"]]) or "",
                cast("str | None", row[b_idx["semantic_type"]]) or "",
                cast("str | None", row[b_idx["media_type"]]) or "",
                cast("str | None", row[b_idx["language"]]) or "",
                "" if is_error is None else str(int(cast(int, is_error))),
                "" if exit_code is None else str(cast(int, exit_code)),
            )
        )
    return _hash_bytes(
        "message",
        session_id,
        native_id or "",
        str(position),
        str(variant_index),
        role or "",
        message_type or "",
        material_origin or "",
        "",  # message.text unavailable at this layer -- see docstring
        user_context_text or "",
        stop_reason or "",
        *block_parts,
    )


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

    attachment_refs: list[tuple[object, ...]]
    attachment_native_ids: list[tuple[object, ...]]
    paste_spans: list[tuple[object, ...]]
    file_edits: list[tuple[object, ...]]
    web_content_constructs: list[tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class _ProjectionCarryForward:
    captured: _CapturedProjections
    block_id_remap: dict[str, str]
    live_message_ids: frozenset[str]


def _capture_session_projection_rows(conn: sqlite3.Connection, session_id: str) -> _CapturedProjections:
    attachment_refs = conn.execute(
        "SELECT attachment_id, session_id, message_id, position, upload_origin, source_url, caption "
        "FROM attachment_refs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    ref_ids = [f"{row[2]}:attachment:{row[3]}" for row in attachment_refs]
    attachment_native_ids: list[sqlite3.Row] = []
    if ref_ids:
        placeholders = ",".join("?" for _ in ref_ids)
        attachment_native_ids = conn.execute(
            f"SELECT ref_id, id_kind, native_id FROM attachment_native_ids WHERE ref_id IN ({placeholders})",
            ref_ids,
        ).fetchall()
    paste_spans = conn.execute(
        "SELECT message_id, session_id, position, start_offset, end_offset, boundary_state, "
        "source_event_id, source_marker, content_hash, observed_at_ms FROM paste_spans WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    file_edits = conn.execute(
        "SELECT tool_use_block_id, session_id, message_id, file_path, structured_patch_json, "
        "original_file, old_string, new_string, replace_all, user_modified, observed_at_ms "
        "FROM file_edits WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    web_content_constructs = conn.execute(
        "SELECT session_id, message_id, block_id, position, provider, construct_type, provider_key, "
        "title, url, text, source_id, group_id, group_title, query, asset_pointer, mime_type, status, "
        "task_id, task_type, rank, start_index, end_index FROM web_content_constructs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    return _CapturedProjections(
        attachment_refs=[tuple(r) for r in attachment_refs],
        attachment_native_ids=[tuple(r) for r in attachment_native_ids],
        paste_spans=[tuple(r) for r in paste_spans],
        file_edits=[tuple(r) for r in file_edits],
        web_content_constructs=[tuple(r) for r in web_content_constructs],
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
    either. ``block_id`` references are remapped through
    ``block_id_remap`` when the owning block's position shifted during
    reconciliation (``block_id`` embeds position); ``attachment_refs``/
    ``paste_spans`` need no remap -- their own PK ``position`` column is an
    independent per-message ordinal, not the message's transcript position.
    """
    captured = carry_forward.captured
    live_message_ids = carry_forward.live_message_ids
    block_id_remap = carry_forward.block_id_remap

    restored_attachment_ref_ids: set[str] = set()
    for row in captured.attachment_refs:
        message_id = cast(str, row[2])
        position = row[3]
        if message_id not in live_message_ids:
            continue
        exists = conn.execute(
            "SELECT 1 FROM attachment_refs WHERE message_id = ? AND position = ?", (message_id, position)
        ).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO attachment_refs "
                "(attachment_id, session_id, message_id, position, upload_origin, source_url, caption) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            restored_attachment_ref_ids.add(f"{message_id}:attachment:{position}")

    for row in captured.attachment_native_ids:
        if row[0] in restored_attachment_ref_ids:
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
        old_block_id = cast(str, row[0])
        new_block_id = block_id_remap.get(old_block_id, old_block_id)
        exists = conn.execute("SELECT 1 FROM file_edits WHERE tool_use_block_id = ?", (new_block_id,)).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO file_edits "
                "(tool_use_block_id, session_id, message_id, file_path, structured_patch_json, "
                "original_file, old_string, new_string, replace_all, user_modified, observed_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_block_id, *row[1:]),
            )

    for row in captured.web_content_constructs:
        message_id = cast(str, row[1])
        if message_id not in live_message_ids:
            continue
        old_block_id = cast(str, row[2])
        new_block_id = block_id_remap.get(old_block_id, old_block_id)
        position = row[3]
        exists = conn.execute(
            "SELECT 1 FROM web_content_constructs WHERE block_id = ? AND position = ?", (new_block_id, position)
        ).fetchone()
        if exists is None:
            conn.execute(
                "INSERT OR IGNORE INTO web_content_constructs ("
                "session_id, message_id, block_id, position, provider, construct_type, provider_key, "
                "title, url, text, source_id, group_id, group_title, query, asset_pointer, mime_type, status, "
                "task_id, task_type, rank, start_index, end_index"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row[0], message_id, new_block_id, *row[3:]),
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
    (`None` -- e.g. a caller that writes directly via `ArchiveStore.
    write_parsed()`, used by demo seeding and unit tests, never threads a
    raw_id through), there is no positive evidence of a different
    acquisition, so this also falls back to plain replace rather than
    guessing; approximating "unknown" as "different" would let a corrected
    re-parse's retraction be silently defeated by union whenever a caller
    doesn't happen to supply provenance.

    Only messages carrying a stable provider ``native_id`` participate --
    a message with no native id has no cross-acquisition identity to union
    against (its archive identity is a position/variant fallback that is
    not guaranteed stable across independent acquisitions), so those keep
    the prior whole-row-replace behavior unchanged, exactly as before this
    change.

    Four things beyond plain field coalescing (PR #3413 review):

      1. Reinjecting a vanished message/block preserves its RELATIVE
         transcript order via ``_splice_merge_keys`` -- appending it after
         the incoming maximum would silently reorder the conversation
         (worse than dropping it, since it's invisible).
      2. Blocks are matched by content-addressed structural identity
         (``_block_structural_keys`` -- ``tool_id`` or "Nth block of this
         type"), not raw position, since dropping a non-trailing block
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
    existing_by_native_id: dict[str, tuple[object, ...]] = {}
    for erow in existing_message_rows:
        row_native_id = erow[m_idx["native_id"]]
        if row_native_id is not None:
            existing_by_native_id[cast(str, row_native_id)] = tuple(erow)

    if not existing_by_native_id:
        # Nothing previously stored under a stable identity -- ordinary
        # first-write path, nothing to union against.
        return message_rows, block_rows, None

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
    role_idx = m_idx["role"]
    message_type_idx = m_idx["message_type"]
    material_origin_idx = m_idx["material_origin"]
    user_context_text_idx = m_idx["user_context_text"]
    stop_reason_idx = m_idx["stop_reason"]
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
            "SELECT 1 FROM session_links WHERE resolved_dst_session_id = ? AND inheritance = 'prefix-sharing' LIMIT 1",
            (session_id,),
        ).fetchone()
        is not None
    )

    # --- Message-level reconciliation: preserve relative transcript order ---
    # (PR review P1 write.py:2359) A vanished message must be spliced back
    # into its correct relative position, not appended after the incoming
    # maximum -- see `_splice_merge_keys`.
    old_message_order = sorted(
        existing_by_native_id.keys(), key=lambda nid: cast(int, existing_by_native_id[nid][position_idx])
    )
    incoming_native_ids = {cast(str, row[native_idx]) for row in message_rows if row[native_idx] is not None}
    matched_native_ids = set(existing_by_native_id.keys()) & incoming_native_ids
    new_message_keys: list[object] = [
        (row[native_idx] if row[native_idx] is not None else object()) for row in message_rows
    ]
    incoming_row_by_key = dict(zip(new_message_keys, message_rows, strict=True))
    splice_old_keys: list[object] = [] if is_prefix_sharing_parent else list(old_message_order)

    merged_message_rows: list[tuple[object, ...]] = []
    merged_message_ids: dict[str, str] = {}  # native_id -> merged message_id
    for new_pos, (source, key) in enumerate(_splice_merge_keys(splice_old_keys, new_message_keys)):
        if source == "new":
            row = incoming_row_by_key[key]
            nid = row[native_idx]
            if nid is not None and nid in existing_by_native_id:
                existing_row = existing_by_native_id[nid]
                row = tuple(_coalesce_scalar(nv, ov) for nv, ov in zip(row, existing_row, strict=True))
        else:
            nid = cast(str, key)
            row = existing_by_native_id[nid]
            logger.info(
                "field-path union (polylogue-geop): reinjecting message native_id=%s session_id=%s "
                "dropped by newer acquisition, restored at relative position %s",
                nid,
                session_id,
                new_pos,
            )
        row_list = list(row)
        row_list[position_idx] = new_pos
        row = tuple(row_list)
        merged_message_rows.append(row)
        if isinstance(key, str):
            merged_message_ids[key] = archive_message_id(
                session_id, key, position=new_pos, variant_index=cast(int, row_list[variant_idx] or 0)
            )

    live_message_ids = frozenset(merged_message_ids.values())

    # --- Block-level reconciliation, per still-live message ---
    # (PR review P1 write.py:2383) Matching by raw position mismatches once a
    # non-trailing block is dropped -- match by content-addressed structural
    # identity instead (`_block_structural_keys`), then splice-merge exactly
    # like messages above.
    incoming_blocks_by_message: dict[str, list[tuple[object, ...]]] = {}
    for row in block_rows:
        incoming_blocks_by_message.setdefault(cast(str, row[b_idx["message_id"]]), []).append(row)
    for rows in incoming_blocks_by_message.values():
        rows.sort(key=lambda r: cast(int, r[block_position_idx]))

    matched_message_ids = incoming_blocks_by_message.keys() & existing_blocks_by_message.keys()
    merged_block_rows: list[tuple[object, ...]] = []
    block_id_remap: dict[str, str] = {}
    final_blocks_by_message: dict[str, list[tuple[object, ...]]] = {}

    for message_id, new_rows in incoming_blocks_by_message.items():
        if message_id not in matched_message_ids:
            merged_block_rows.extend(new_rows)
            final_blocks_by_message[message_id] = new_rows
            continue
        old_rows = existing_blocks_by_message[message_id]
        new_keys = _block_structural_keys(new_rows, b_idx)
        old_keys = _block_structural_keys(old_rows, b_idx)
        new_row_by_key = dict(zip(new_keys, new_rows, strict=True))
        old_row_by_key = dict(zip(old_keys, old_rows, strict=True))
        old_position_by_key = {k: cast(int, r[block_position_idx]) for k, r in zip(old_keys, old_rows, strict=True)}

        final_rows: list[tuple[object, ...]] = []
        for new_block_pos, (source, key) in enumerate(_splice_merge_keys(old_keys, new_keys)):
            if source == "new":
                row = new_row_by_key[key]
                if key in old_row_by_key:
                    row = _coalesce_block_row(
                        row, old_row_by_key[key], b_idx, message_id=message_id, position=new_block_pos
                    )
                old_pos = old_position_by_key.get(key)
            else:
                row = old_row_by_key[key]
                old_pos = old_position_by_key[key]
                logger.info(
                    "field-path union (polylogue-geop): reinjecting block message_id=%s "
                    "dropped by newer acquisition, restored at relative position %s",
                    message_id,
                    new_block_pos,
                )
            if old_pos is not None and old_pos != new_block_pos:
                block_id_remap[f"{message_id}:{old_pos}"] = f"{message_id}:{new_block_pos}"
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
        nid = row[native_idx]
        if nid is None or nid not in matched_native_ids:
            recomputed_message_rows.append(row)
            continue
        message_id = merged_message_ids[nid]
        final_blocks = final_blocks_by_message.get(message_id, [])
        row_list = list(row)
        row_list[has_tool_use_idx] = (
            1 if any(b[b_idx["block_type"]] == BlockType.TOOL_USE.value for b in final_blocks) else 0
        )
        row_list[has_thinking_idx] = (
            1 if any(b[b_idx["block_type"]] == BlockType.THINKING.value for b in final_blocks) else 0
        )
        row_list[message_content_hash_idx] = _message_content_hash_from_rows(
            session_id,
            nid,
            cast(int, row_list[position_idx]),
            cast(int, row_list[variant_idx] or 0),
            cast("str | None", row_list[role_idx]),
            cast("str | None", row_list[message_type_idx]),
            cast("str | None", row_list[material_origin_idx]),
            cast("str | None", row_list[user_context_text_idx]),
            cast("str | None", row_list[stop_reason_idx]),
            final_blocks,
            b_idx,
        )
        recomputed_message_rows.append(tuple(row_list))

    # --- Capture sidecar projection rows for restoration after the incoming
    # rebuild (PR review P1 write.py:2433) ---
    carry_forward = _ProjectionCarryForward(
        captured=_capture_session_projection_rows(conn, session_id),
        block_id_remap=block_id_remap,
        live_message_ids=live_message_ids,
    )

    return recomputed_message_rows, merged_block_rows, carry_forward


def _replace_full_session_messages_and_blocks(
    conn: sqlite3.Connection,
    session: ParsedSession,
    messages: list[ParsedMessage],
    *,
    duplicate_native_ids: frozenset[str],
    raw_id: str | None = None,
    existing_raw_id: str | None = None,
    session_row_existed: bool = True,
    force_replace: bool = False,
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "append",
    bulk_build: bool = False,
    defer_fts_rebuild: bool = False,
    prepared: PreparedSessionRows | None = None,
) -> _ProjectionCarryForward | None:
    """Replace one session's messages/blocks wholesale.

    Returns the ``_ProjectionCarryForward`` payload computed by
    ``_union_with_existing_rows`` (or ``None`` when nothing was unioned), so
    the caller (``write_parsed_session_to_archive``) can restore
    ``attachment_refs``/``paste_spans`` after ITS OWN later rebuild of those
    two tables -- they are written outside this function, in the shared
    merge_append/full-replace tail, so this function cannot restore them
    itself without running before they even exist.

    ``bulk_build`` (polylogue-v6i3, default ``False``) skips this function's
    own scoped derived-index refresh and its ``action_pairs`` refresh entirely -- see
    ``write_parsed_session_to_archive``'s docstring for why: a bulk-build
    caller always repopulates both surfaces archive-wide exactly once at
    readiness, so per-session maintenance here is pure waste. Ordinary
    (non-bulk-build) full-session replaces are byte-for-byte unchanged.

    ``prepared`` (polylogue-623q), when given, is an already-validated
    ``PreparedSessionRows`` -- the caller (``write_parsed_session_to_archive``)
    has already confirmed it was computed from THIS session's content hash
    and that no lineage tail-slicing changed ``messages`` since. The message/
    block row-building loops (the CPU-bound part of this function -- per-item
    hashing, JSON encoding, enum lookups) are then skipped entirely in favor
    of the precomputed tuples; only the ``executemany`` against SQLite still
    runs on this (writer) thread. ``None`` (the default) reproduces the exact
    prior behavior byte-for-byte.

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

    ``session_row_existed`` (polylogue-cs86, default ``True`` -- so any
    caller that does not thread it reproduces the exact prior behavior)
    tells this function whether ``sessions`` already had a row for this
    ``session_id`` immediately before the caller's own upsert. When
    ``False``, the caller has PROVEN (via a single PK lookup on
    ``sessions``, already paid for regardless -- see
    ``write_parsed_session_to_archive``'s ``existing_session_raw_id``
    capture) that this session_id has never been written before, under
    this exact archive connection/transaction. Every row this function's
    delete cascade would otherwise remove (``_clear_session_projection_rows``'s
    ~14 tables plus the bare ``DELETE FROM messages``) is written ONLY
    together with (or after) that same ``sessions`` row, by this same
    function or by ``merge_append`` -- which itself requires the session to
    already exist (it reads ``MAX(position) FROM messages WHERE
    session_id = ?``). So "no prior ``sessions`` row" implies "no prior rows
    anywhere in the cascade" unconditionally, not merely for a fresh
    bulk-build generation -- skipping the cascade is provably a no-op, not a
    heuristic. This turns each of those ~14 point-DELETEs (and the
    parent_message_id-nulling UPDATE) from a real statement into dead work
    on the hot bulk-rebuild path, where every incoming raw's session_id is
    new on its first pass through an empty generation (measured: this
    cascade was ~20% of ``apply_s`` on a from-empty bulk build even with
    every index present, since a DELETE against zero matching rows still
    pays a full statement-execution + index-descent cost per table).
    Same-connection reads see the same transaction's own uncommitted writes,
    so a session_id revisited later in one multi-raw bulk transaction (e.g.
    a second accepted revision in the same raw-revision chain) correctly
    observes ``session_row_existed=True`` on its second occurrence and still
    runs the cascade -- this is not scoped to "first write in this process",
    it is scoped to "first write in this session_id's history in this
    archive".
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
    # polylogue-geop: compute the field-path union against whatever is
    # currently stored *before* any delete below removes it. Must run ahead
    # of the FTS/base-table deletes -- both messages and blocks are read here.
    unioned_message_rows, unioned_block_rows, carry_forward = _union_with_existing_rows(
        conn,
        session_id,
        list(prepared.message_rows)
        if prepared is not None
        else _build_message_rows(session_id, messages, duplicate_native_ids=duplicate_native_ids),
        list(prepared.block_rows)
        if prepared is not None
        else _build_block_rows(session_id, messages, duplicate_native_ids=duplicate_native_ids),
        raw_id=raw_id,
        existing_raw_id=existing_raw_id,
        force_replace=force_replace,
    )
    add_timing("field_path_union", t0)
    t0 = time.perf_counter()
    use_scoped_fts_rebuild = not bulk_build and message_fts_triggers_present_sync(conn)
    add_timing("fts_probe", t0)
    if use_scoped_fts_rebuild:
        # polylogue-cs86: these three deletes are keyed by session_id exactly
        # like the base-table cascade below, and the same proof applies -- a
        # session_id with no prior ``sessions`` row can have no prior FTS/
        # trigram rows either (they are only ever written alongside base
        # rows this same function or a prior call already wrote). Skipping
        # them when ``session_row_existed`` is False is a no-op removal, not
        # a heuristic.
        if session_row_existed:
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
            # Full replacement also deletes and recreates tool-use blocks.  The
            # trigram external-content index must be cleared while their old text
            # is still available, before the guard below suppresses its per-row
            # triggers.  Leaving it trigger-maintained made a live 18 MB Codex
            # transcript spend minutes performing thousands of individual FTS5
            # updates under the sole writer lock.
            t0 = time.perf_counter()
            conn.execute(trigram_delete_session_rows_sql(), (session_id,))
            add_timing("fts_trigram_delete", t0)
        t0 = time.perf_counter()
        # Keep the canonical triggers structurally present and gate both the
        # message and trigram bodies for the whole replacement.  This is the
        # same protocol used for guarded lineage rewrites, but here it covers
        # the session's own delete and insert as well.
        conn.execute(
            "INSERT OR REPLACE INTO derived_refresh_guard(guard_name) VALUES (?)",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        )
        add_timing("fts_guard", t0)
    replacement_complete = False
    try:
        # polylogue-cs86: ``_clear_session_projection_rows`` (~14 point-DELETEs
        # across messages/blocks/action_pairs/session_events/session_links/...)
        # plus the bare messages delete below are a no-op cascade whenever this
        # session_id has never had a ``sessions`` row before -- see
        # ``session_row_existed``'s docstring for the invariant proof. Every
        # DELETE against zero matching rows still pays a full statement
        # execution + index descent per table, measured at ~20% of apply_s on
        # a from-empty bulk-build pass even with every index present -- this
        # is where that cost was going. Skipping is exact, not approximate:
        # the statements below are guaranteed to delete zero rows in this
        # branch, so omitting them changes no observable state.
        if session_row_existed:
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
        )
        add_timing("messages", t0)
        t0 = time.perf_counter()
        _write_blocks(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
            rows=unioned_block_rows,
        )
        add_timing("blocks", t0)
        t0 = time.perf_counter()
        _write_file_edits(
            conn,
            session_id,
            messages,
            duplicate_native_ids=duplicate_native_ids,
        )
        add_timing("file_edits", t0)
        t0 = time.perf_counter()
        if not bulk_build:
            refresh_action_pairs(conn, session_id)
        add_timing("action_pairs", t0)
        t0 = time.perf_counter()
        _write_web_constructs(
            conn,
            session,
            messages,
            duplicate_native_ids=duplicate_native_ids,
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
                conn.execute(trigram_insert_session_rows_sql(), (session_id,))
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


def _refresh_session_counts(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET message_count = (SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id),
            word_count = COALESCE((SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id), 0),
            tool_use_count = COALESCE((SELECT SUM(has_tool_use) FROM messages WHERE session_id = sessions.session_id), 0),
            thinking_count = COALESCE((SELECT SUM(has_thinking) FROM messages WHERE session_id = sessions.session_id), 0),
            paste_count = COALESCE((SELECT SUM(has_paste) FROM messages WHERE session_id = sessions.session_id), 0),
            user_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'user'
            ),
            authored_user_message_count = (
                SELECT COUNT(*) FROM messages
                WHERE session_id = sessions.session_id AND material_origin = 'human_authored'
            ),
            assistant_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'assistant'
            ),
            system_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'system'
            ),
            tool_message_count = (
                SELECT COUNT(*) FROM messages WHERE session_id = sessions.session_id AND role = 'tool'
            ),
            user_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id AND role = 'user'
            ), 0),
            authored_user_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages
                WHERE session_id = sessions.session_id AND material_origin = 'human_authored'
            ), 0),
            assistant_word_count = COALESCE((
                SELECT SUM(word_count) FROM messages WHERE session_id = sessions.session_id AND role = 'assistant'
            ), 0)
        WHERE session_id = ?
        """,
        (session_id,),
    )


def _session_count_values(messages: list[ParsedMessage]) -> dict[str, int]:
    counts = {
        "message_count": 0,
        "word_count": 0,
        "tool_use_count": 0,
        "thinking_count": 0,
        "paste_count": 0,
        "user_message_count": 0,
        "authored_user_message_count": 0,
        "assistant_message_count": 0,
        "system_message_count": 0,
        "tool_message_count": 0,
        "user_word_count": 0,
        "authored_user_word_count": 0,
        "assistant_word_count": 0,
    }
    for message in messages:
        role = _enum_value(message.role)
        material_origin = _enum_value(message.material_origin)
        word_count = _word_count(message.text)
        counts["message_count"] += 1
        counts["word_count"] += word_count
        counts["tool_use_count"] += _has_block(message, BlockType.TOOL_USE)
        counts["thinking_count"] += _has_block(message, BlockType.THINKING)
        counts["paste_count"] += _has_paste(message)
        if role == "user":
            counts["user_message_count"] += 1
            counts["user_word_count"] += word_count
        elif role == "assistant":
            counts["assistant_message_count"] += 1
            counts["assistant_word_count"] += word_count
        elif role == "system":
            counts["system_message_count"] += 1
        elif role == "tool":
            counts["tool_message_count"] += 1
        if material_origin == "human_authored":
            counts["authored_user_message_count"] += 1
            counts["authored_user_word_count"] += word_count
    return counts


def _messages_have_token_counts(messages: Sequence[ParsedMessage]) -> bool:
    return any(
        message.input_tokens or message.output_tokens or message.cache_read_tokens or message.cache_write_tokens
        for message in messages
    )


def _increment_session_counts_for_append(
    conn: sqlite3.Connection,
    session_id: str,
    counts: dict[str, int],
) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET message_count = COALESCE(message_count, 0) + ?,
            word_count = COALESCE(word_count, 0) + ?,
            tool_use_count = COALESCE(tool_use_count, 0) + ?,
            thinking_count = COALESCE(thinking_count, 0) + ?,
            paste_count = COALESCE(paste_count, 0) + ?,
            user_message_count = COALESCE(user_message_count, 0) + ?,
            authored_user_message_count = COALESCE(authored_user_message_count, 0) + ?,
            assistant_message_count = COALESCE(assistant_message_count, 0) + ?,
            system_message_count = COALESCE(system_message_count, 0) + ?,
            tool_message_count = COALESCE(tool_message_count, 0) + ?,
            user_word_count = COALESCE(user_word_count, 0) + ?,
            authored_user_word_count = COALESCE(authored_user_word_count, 0) + ?,
            assistant_word_count = COALESCE(assistant_word_count, 0) + ?
        WHERE session_id = ?
        """,
        (
            counts["message_count"],
            counts["word_count"],
            counts["tool_use_count"],
            counts["thinking_count"],
            counts["paste_count"],
            counts["user_message_count"],
            counts["authored_user_message_count"],
            counts["assistant_message_count"],
            counts["system_message_count"],
            counts["tool_message_count"],
            counts["user_word_count"],
            counts["authored_user_word_count"],
            counts["assistant_word_count"],
            session_id,
        ),
    )


def _write_attachments(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    attachments: Iterable[ParsedAttachment],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    refresh_attachment_ids: set[str] | None = None,
    preacquired_blobs: dict[int, tuple[bytes | None, int, str]] | None = None,
) -> None:
    by_native_message_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and message.provider_message_id not in duplicate_native_ids
    }
    by_message_position = {
        message.position: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.position is not None
    }
    touched_attachment_ids: set[str] = set()
    for attachment in attachments:
        attachment_id = _attachment_id(session_id, attachment)
        message_id = (
            by_native_message_id.get(attachment.message_provider_id) if attachment.message_provider_id else None
        )
        if message_id is None and attachment.message_position is not None:
            message_id = by_message_position.get(attachment.message_position)
        if message_id is None:
            continue
        touched_attachment_ids.add(attachment_id)
        acquired_blob = (preacquired_blobs or {}).get(id(attachment))
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
                byte_count = excluded.byte_count,
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
        ref_position = _attachment_position(attachment)
        ref_id = f"{message_id}:attachment:{ref_position}"
        # Bulk rebuilds may suspend FK enforcement. Mirror REPLACE's cascade
        # explicitly so identifiers from an older projection cannot survive.
        conn.execute("DELETE FROM attachment_native_ids WHERE ref_id = ?", (ref_id,))
        conn.execute(
            """
            INSERT OR REPLACE INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin, source_url, caption
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_id,
                session_id,
                message_id,
                ref_position,
                _sqlite_text(attachment.upload_origin),
                _sqlite_text(_attachment_source_url(attachment)),
                _sqlite_text(_attachment_caption(attachment)),
            ),
        )
        _write_attachment_native_ids(conn, ref_id, attachment)
    affected_attachment_ids = touched_attachment_ids | (refresh_attachment_ids or set())
    if not affected_attachment_ids:
        return
    placeholders = ",".join("?" for _ in affected_attachment_ids)
    conn.execute(
        f"""
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*) FROM attachment_refs WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        WHERE attachment_id IN ({placeholders})
        """,
        tuple(sorted(affected_attachment_ids)),
    )
    # polylogue-w06b: a full-replace re-ingest (or a re-ingest whose attachment
    # can no longer be matched to a message via `by_native_message_id`, e.g.
    # the owning message became a duplicate-native-id exclusion or dropped
    # out of this ingest's message set) drops a previously-written
    # attachment_refs row for `refresh_attachment_ids` without this
    # function ever writing a replacement ref. The ref_count UPDATE above
    # correctly reflects that as 0, but nothing previously swept the now
    # ref-less `attachments` row -- it survived, unreachable from any
    # session/message read path (get_attachments/get_attachments_batch both
    # INNER JOIN attachment_refs), while still reporting
    # acquisition_status='acquired' and real fetched bytes. Sweep it here,
    # mirroring the identical cleanup `prune_attachments` and
    # `delete_session_sql` already perform after their own ref deletions.
    conn.execute(
        f"DELETE FROM attachments WHERE ref_count <= 0 AND attachment_id IN ({placeholders})",
        tuple(sorted(affected_attachment_ids)),
    )


def _session_attachment_ids(conn: sqlite3.Connection, session_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT attachment_id FROM attachment_refs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _write_paste_spans(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
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
            position_offset=position_offset,
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
                    else _timestamp_ms(message.timestamp),
                ),
            )


def _write_parent_links(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> None:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id and message.provider_message_id not in duplicate_native_ids
    }
    by_message_position = {
        message.position: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.position is not None
    }
    for fallback_position, message in enumerate(messages):
        parent_message_id = (
            by_native_id.get(message.parent_message_provider_id) if message.parent_message_provider_id else None
        )
        if parent_message_id is None and message.parent_message_position is not None:
            parent_message_id = by_message_position.get(message.parent_message_position)
        if parent_message_id is None:
            continue
        conn.execute(
            """
            UPDATE messages
            SET parent_message_id = ?
            WHERE message_id = ?
            """,
            (
                parent_message_id,
                _message_id(
                    session_id,
                    message,
                    fallback_position,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_native_ids,
                ),
            ),
        )


def _write_session_link(
    conn: sqlite3.Connection,
    session_id: str,
    session: ParsedSession,
    *,
    branch_point_message_id: str | None = None,
    inheritance: str | None = None,
) -> None:
    if not session.parent_session_provider_id:
        return
    # polylogue-lyr2: normalize the same way ``_stored_session_native_id``
    # normalizes ``sessions.native_id`` -- the resolver below matches this
    # column against ``sessions.native_id`` by exact string equality
    # (``_resolve_outbound_session_links``), so a raw (unstripped) parent
    # reference here would silently stop matching a parent whose own
    # ``native_id`` was written through the stripping helper.
    dst_native_id = _sqlite_text(session.parent_session_provider_id.strip())
    if not dst_native_id:
        return
    link_type = branch_type_to_edge_type(session.branch_type, default=TopologyEdgeType.BRANCH).value
    parent_tool_use_block_id = _resolve_parent_tool_use_block_id(conn, session)
    method = "parser-parent" if parent_tool_use_block_id is None else "parent-tool-use-id"
    conn.execute(
        """
        INSERT OR REPLACE INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type,
            branch_point_message_id, inheritance,
            status, parent_tool_use_block_id, method, confidence, evidence_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            origin_from_provider(session.source_name).value,
            dst_native_id,
            link_type,
            branch_point_message_id,
            inheritance,
            parent_tool_use_block_id,
            method,
            1.0,
            _json_dumps({"parent_session_provider_id": session.parent_session_provider_id}),
            _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at) or 0,
        ),
    )


def _resolve_parent_tool_use_block_id(conn: sqlite3.Connection, session: ParsedSession) -> str | None:
    """Resolve ``parentToolUseID`` (a provider tool_id) to its block_id.

    polylogue-2qx.4: ``tool_id`` values are provider-generated and globally
    unique across a session tree, so this is a plain lookup against whatever
    session already wrote that TOOL_USE block -- no cross-session identity
    resolution needed. ``None`` (not found / not supplied) leaves the column
    NULL, never a guess.
    """
    tool_use_provider_id = getattr(session, "parent_tool_use_provider_id", None)
    if not tool_use_provider_id:
        return None
    row = conn.execute(
        "SELECT block_id FROM blocks WHERE tool_id = ? AND block_type = 'tool_use' LIMIT 1",
        (tool_use_provider_id,),
    ).fetchone()
    return str(row[0]) if row is not None else None


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
_CYCLE_WALK_BUDGET = 1024


def _would_create_cycle(
    conn: sqlite3.Connection,
    *,
    child_id: str,
    proposed_parent_id: str,
) -> list[str] | None:
    """Return the cycle path if resolving child->proposed_parent would close a loop.

    Walks ``sessions.parent_session_id`` upward from ``proposed_parent_id``.
    Returns ``None`` for a legitimate (acyclic, or not-yet-resolvable) shape.
    """
    if proposed_parent_id == child_id:
        return [child_id, child_id]
    path: list[str] = [child_id, proposed_parent_id]
    current = proposed_parent_id
    steps = 0
    while True:
        if steps >= _CYCLE_WALK_BUDGET:
            path.append("...budget-exceeded")
            return path
        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            (current,),
        ).fetchone()
        if row is None:
            return None
        next_parent = row[0]
        if next_parent is None:
            return None
        if next_parent == child_id:
            path.append(child_id)
            return path
        path.append(next_parent)
        current = next_parent
        steps += 1


def _quarantine_session_link(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    link_type: str,
    cycle_path: list[str],
    observed_at_ms: int,
) -> None:
    """Mark one edge quarantined instead of resolving it, with evidence."""
    evidence = _json_dumps(
        {
            "reason": "cycle_rejected",
            "cycle_path": cycle_path,
            "detected_at_ms": observed_at_ms,
        }
    )
    conn.execute(
        """
        UPDATE session_links
           SET status = ?,
               evidence_json = ?,
               resolved_at_ms = ?
         WHERE src_session_id = ?
           AND dst_origin = ?
           AND dst_native_id = ?
           AND link_type = ?
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
    )


def _resolve_session_graph(
    conn: sqlite3.Connection,
    session_id: str,
    native_id: str,
    origin: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    add_timing: Callable[[str, float], None] | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
) -> None:
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
    has_outbound_link = (
        conn.execute("SELECT 1 FROM session_links WHERE src_session_id = ? LIMIT 1", (session_id,)).fetchone()
        is not None
    )
    inbound_rows = conn.execute(
        """
        SELECT links.src_session_id, links.link_type
        FROM session_links links
        WHERE links.dst_native_id = ?
          AND links.resolved_dst_session_id IS NULL
          AND links.status IS NULL
          AND links.dst_origin = ?
        """,
        (native_id, origin),
    ).fetchall()
    record_substage("inbound_lookup", t0)
    t0 = time.perf_counter()
    if not has_outbound_link and not inbound_rows and _root_projection_current(conn, session_id):
        record_substage("root_current_check", t0)
        return
    record_substage("root_current_check", t0)
    composed_cache: dict[str, list[tuple[str, str]]] = {}
    t0 = time.perf_counter()
    resolved_child_ids: list[str] = []
    for row in inbound_rows:
        child_id, link_type = str(row[0]), str(row[1])
        # polylogue-4ts.10: session_id is about to become child_id's parent --
        # refuse (quarantine, with evidence) rather than silently resolve if
        # that would close a cycle in sessions.parent_session_id.
        cycle_path = _would_create_cycle(conn, child_id=child_id, proposed_parent_id=session_id)
        if cycle_path is not None:
            _quarantine_session_link(
                conn,
                src_session_id=child_id,
                dst_origin=origin,
                dst_native_id=native_id,
                link_type=link_type,
                cycle_path=cycle_path,
                observed_at_ms=int(time.time() * 1000),
            )
            continue
        conn.execute(
            """
            UPDATE session_links
            SET resolved_dst_session_id = ?,
                resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms)
            WHERE src_session_id = ?
              AND dst_native_id = ?
              AND link_type = ?
              AND resolved_dst_session_id IS NULL
              AND status IS NULL
            """,
            (session_id, child_id, native_id, link_type),
        )
        resolved_child_ids.append(child_id)
        # Deferred tail extraction (#2467): a child ingested before its parent was
        # stored whole (the inherited prefix could not be aligned yet). Now that
        # the parent exists, normalize the child the same way the parent-known
        # write path does — drop the inherited prefix rows and record the edge.
        _reextract_prefix_tail_db(
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

    impacted_session_ids = {session_id, *resolved_child_ids}
    t0 = time.perf_counter()
    _repair_stale_prefix_branch_points_db(conn, impacted_session_ids, cache=cache, composed_cache=composed_cache)
    record_substage("repair_stale_branch_points", t0)
    t0 = time.perf_counter()
    old_root_ids = _root_ids(conn, impacted_session_ids)
    projection_seen: set[str] = set()
    for impacted_session_id in impacted_session_ids:
        _refresh_session_projection(conn, impacted_session_id, seen=projection_seen)
    root_ids_to_refresh = old_root_ids | _root_ids(conn, impacted_session_ids)
    record_substage("projection_refresh", t0)
    t0 = time.perf_counter()
    for root_session_id in root_ids_to_refresh:
        _refresh_thread(conn, root_session_id)
    record_substage("thread_refresh", t0)


def _root_projection_current(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        """
        SELECT root_session_id, parent_session_id, created_at_ms, updated_at_ms
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None or row[0] != session_id or row[1] is not None:
        return False
    thread_row = conn.execute(
        """
        SELECT created_at_ms, session_count, depth
        FROM threads
        WHERE thread_id = ?
        """,
        (session_id,),
    ).fetchone()
    if thread_row is None:
        return False
    thread_sessions = conn.execute(
        """
        SELECT session_id
        FROM thread_sessions
        WHERE thread_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return (
        int(thread_row[0] or 0) == int(row[2] or row[3] or 0)
        and int(thread_row[1] or 0) == 1
        and int(thread_row[2] or 0) == 0
        and [str(thread_session[0]) for thread_session in thread_sessions] == [session_id]
    )


def _resolve_outbound_session_links(conn: sqlite3.Connection, session_id: str, origin: str) -> None:
    """Resolve ``session_id``'s own unresolved outbound edges (it is the child).

    polylogue-4ts.10: candidates are evaluated one at a time (rather than a
    single blanket UPDATE) so each can be cycle-checked against
    ``sessions.parent_session_id`` before being resolved -- a candidate whose
    resolution would close a loop is quarantined instead, never resolved.
    """
    candidates = conn.execute(
        """
        SELECT session_links.dst_origin, session_links.dst_native_id, session_links.link_type, dst.session_id
        FROM session_links
        JOIN sessions dst
          ON dst.native_id = session_links.dst_native_id
         AND dst.origin = session_links.dst_origin
        WHERE session_links.src_session_id = ?
          AND session_links.resolved_dst_session_id IS NULL
          AND session_links.status IS NULL
        """,
        (session_id,),
    ).fetchall()
    for dst_origin, dst_native_id, link_type, proposed_parent_id in candidates:
        cycle_path = _would_create_cycle(conn, child_id=session_id, proposed_parent_id=proposed_parent_id)
        if cycle_path is not None:
            _quarantine_session_link(
                conn,
                src_session_id=session_id,
                dst_origin=dst_origin,
                dst_native_id=dst_native_id,
                link_type=link_type,
                cycle_path=cycle_path,
                observed_at_ms=int(time.time() * 1000),
            )
            continue
        conn.execute(
            """
            UPDATE session_links
               SET resolved_dst_session_id = ?,
                   resolved_at_ms = COALESCE(resolved_at_ms, observed_at_ms)
             WHERE src_session_id = ?
               AND dst_origin = ?
               AND dst_native_id = ?
               AND link_type = ?
               AND resolved_dst_session_id IS NULL
               AND status IS NULL
            """,
            (proposed_parent_id, session_id, dst_origin, dst_native_id, link_type),
        )


def _refresh_session_projection(conn: sqlite3.Connection, session_id: str, *, seen: set[str]) -> None:
    if session_id in seen:
        return
    seen.add(session_id)
    parent_link = conn.execute(
        """
        SELECT resolved_dst_session_id, link_type
        FROM session_links
        WHERE src_session_id = ? AND resolved_dst_session_id IS NOT NULL
        ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if parent_link is None:
        unresolved_link = conn.execute(
            """
            SELECT link_type
            FROM session_links
            WHERE src_session_id = ?
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_origin, dst_native_id, link_type
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        branch_type: str | None
        if unresolved_link is not None:
            branch_type = _branch_type_from_link_type(unresolved_link[0])
        else:
            existing_branch = conn.execute(
                "SELECT branch_type FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            branch_type = str(existing_branch[0]) if existing_branch is not None and existing_branch[0] else None
        conn.execute(
            """
            UPDATE sessions
            SET parent_session_id = NULL,
                root_session_id = session_id,
                branch_type = ?
            WHERE session_id = ?
            """,
            (branch_type, session_id),
        )
        return

    parent_session_id = str(parent_link[0])
    _refresh_session_projection(conn, parent_session_id, seen=seen)
    parent_root_row = conn.execute(
        """
        SELECT COALESCE(root_session_id, session_id)
        FROM sessions
        WHERE session_id = ?
        """,
        (parent_session_id,),
    ).fetchone()
    parent_root_id = str(parent_root_row[0]) if parent_root_row is not None else parent_session_id
    conn.execute(
        """
        UPDATE sessions
        SET parent_session_id = ?,
            root_session_id = ?,
            branch_type = ?
        WHERE session_id = ?
        """,
        (parent_session_id, parent_root_id, _branch_type_from_link_type(parent_link[1]), session_id),
    )


def _refresh_thread(conn: sqlite3.Connection, root_session_id: str) -> None:
    root = conn.execute(
        """
        SELECT session_id, origin, created_at_ms, updated_at_ms, COALESCE(root_session_id, session_id) AS actual_root_id
        FROM sessions
        WHERE session_id = ?
        """,
        (root_session_id,),
    ).fetchone()
    if root is None:
        return
    if root[4] != root_session_id:
        conn.execute("DELETE FROM thread_sessions WHERE thread_id = ?", (root_session_id,))
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (root_session_id,))
        return
    conn.execute(
        """
        INSERT INTO threads (thread_id, created_at_ms, session_count, depth)
        VALUES (?, ?, 0, 0)
        ON CONFLICT(thread_id) DO UPDATE SET
            created_at_ms = excluded.created_at_ms
        """,
        (root_session_id, root[2] or root[3] or 0),
    )
    session_rows = conn.execute(
        """
        SELECT session_id
        FROM sessions
        WHERE root_session_id = ? OR session_id = ?
        ORDER BY session_id != ?, sort_key_ms IS NULL, sort_key_ms, session_id
        """,
        (root_session_id, root_session_id, root_session_id),
    ).fetchall()
    desired_session_ids = [str(row[0]) for row in session_rows]
    existing_thread = conn.execute(
        """
        SELECT created_at_ms, session_count, depth
        FROM threads
        WHERE thread_id = ?
        """,
        (root_session_id,),
    ).fetchone()
    if existing_thread is not None:
        existing_session_ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT session_id
                FROM thread_sessions
                WHERE thread_id = ?
                ORDER BY position
                """,
                (root_session_id,),
            ).fetchall()
        ]
        if (
            int(existing_thread[0] or 0) == int(root[2] or root[3] or 0)
            and int(existing_thread[1] or 0) == len(desired_session_ids)
            and int(existing_thread[2] or 0) == max(len(desired_session_ids) - 1, 0)
            and existing_session_ids == desired_session_ids
        ):
            return
        if existing_session_ids == desired_session_ids[: len(existing_session_ids)]:
            new_rows = [
                (root_session_id, row[0], position)
                for position, row in enumerate(
                    session_rows[len(existing_session_ids) :], start=len(existing_session_ids)
                )
            ]
            if new_rows:
                conn.executemany(
                    """
                    INSERT INTO thread_sessions (thread_id, session_id, position)
                    VALUES (?, ?, ?)
                    """,
                    new_rows,
                )
            conn.execute(
                """
                UPDATE threads
                SET session_count = ?,
                    depth = ?
                WHERE thread_id = ?
                """,
                (len(session_rows), max(len(session_rows) - 1, 0), root_session_id),
            )
            return
        if len(existing_session_ids) == len(desired_session_ids):
            # Same membership, different order (a thread member's sort key
            # moved past siblings without joining/leaving the thread). Since
            # both lists have equal length, index i names the same numeric
            # ``position`` in both orderings, so the common leading/trailing
            # run that already agrees can be left untouched -- only the
            # differing middle span needs a delete+reinsert. Without this, a
            # giant long-lived thread (thousands of resumed/forked Codex
            # sessions) pays a full O(thread_size) rebuild on every reorder,
            # even when only a handful of rows actually moved
            # (polylogue-6wnh).
            n = len(existing_session_ids)
            common_prefix_len = 0
            while (
                common_prefix_len < n
                and existing_session_ids[common_prefix_len] == desired_session_ids[common_prefix_len]
            ):
                common_prefix_len += 1
            common_suffix_len = 0
            max_suffix = n - common_prefix_len
            while (
                common_suffix_len < max_suffix
                and existing_session_ids[n - 1 - common_suffix_len] == desired_session_ids[n - 1 - common_suffix_len]
            ):
                common_suffix_len += 1
            span_end = n - common_suffix_len
            if span_end > common_prefix_len:
                conn.execute(
                    """
                    DELETE FROM thread_sessions
                    WHERE thread_id = ? AND position >= ? AND position < ?
                    """,
                    (root_session_id, common_prefix_len, span_end),
                )
                conn.executemany(
                    """
                    INSERT INTO thread_sessions (thread_id, session_id, position)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (root_session_id, session_id, position)
                        for position, session_id in enumerate(
                            desired_session_ids[common_prefix_len:span_end], start=common_prefix_len
                        )
                    ],
                )
            conn.execute(
                """
                UPDATE threads
                SET session_count = ?,
                    depth = ?
                WHERE thread_id = ?
                """,
                (n, max(n - 1, 0), root_session_id),
            )
            return
    conn.execute("DELETE FROM thread_sessions WHERE thread_id = ?", (root_session_id,))
    if session_rows:
        conn.executemany(
            """
            INSERT INTO thread_sessions (thread_id, session_id, position)
            VALUES (?, ?, ?)
            """,
            [(root_session_id, row[0], position) for position, row in enumerate(session_rows)],
        )
    conn.execute(
        """
        UPDATE threads
        SET session_count = ?,
            depth = ?
        WHERE thread_id = ?
        """,
        (len(session_rows), max(len(session_rows) - 1, 0), root_session_id),
    )


def _root_ids(conn: sqlite3.Connection, session_ids: set[str]) -> set[str]:
    root_ids: set[str] = set()
    for session_id in session_ids:
        row = conn.execute(
            """
            SELECT COALESCE(root_session_id, session_id)
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is not None and row[0]:
            root_ids.add(str(row[0]))
    return root_ids


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


# Event types whose full evidence already lives durably in a sibling typed
# table -- materializing a second copy into ``session_events`` is a pure,
# zero-evidence-loss duplication (polylogue-bo9n consumer audit, 2026-07-19):
#
# - ``token_count`` / ``message_usage``: fully re-derivable from
#   ``session_provider_usage_events`` (the cost model's sole read path,
#   ``storage/usage.py``); every field the writer would otherwise copy into
#   ``session_events.payload_json`` is already unpacked into that table's
#   typed columns.
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


def _write_session_events(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ParsedMessage],
    events: Iterable[ParsedSessionEvent],
    *,
    position_offset: int = 0,
    event_position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
    provider_usage_baseline: Mapping[str, int] | None = None,
    inherited_source_message_ids: Mapping[str, str] | None = None,
    ambiguous_source_provider_ids: frozenset[str] = frozenset(),
) -> SessionEventWriteResult:
    by_native_id = {
        message.provider_message_id: _message_id(
            session_id,
            message,
            fallback_position,
            position_offset=position_offset,
            duplicate_native_ids=duplicate_native_ids,
        )
        for fallback_position, message in enumerate(messages)
        if message.provider_message_id
        and message.provider_message_id not in duplicate_native_ids
        and message.provider_message_id not in ambiguous_source_provider_ids
    }
    wrote_provider_usage_events = False
    position = event_position_offset
    session_event_rows: list[tuple[object, ...]] = []
    agent_policy_rows: list[tuple[object, ...]] = []
    provider_usage_rows: list[tuple[object, ...]] = []
    for event in events:
        source_message_provider_id = event.source_message_provider_id
        source_message_id = by_native_id.get(source_message_provider_id or "")
        if source_message_id is None and inherited_source_message_ids is not None:
            source_message_id = inherited_source_message_ids.get(source_message_provider_id or "")
        if event.event_type not in _SESSION_EVENTS_REDUNDANT_TYPES:
            session_event_rows.append(
                (
                    session_id,
                    source_message_id,
                    _sqlite_text(source_message_provider_id),
                    position,
                    _sqlite_text(event.event_type),
                    _sqlite_text(_event_summary(event) or ""),
                    _json_dumps(event.payload),
                    _timestamp_ms(event.timestamp),
                ),
            )
        if event.event_type == "agent_policy":
            agent_policy_rows.append(
                (
                    session_id,
                    source_message_id,
                    position,
                    _sqlite_text(_payload_string(event.payload, "approval", "approval_policy")),
                    _sqlite_text(_payload_string(event.payload, "sandbox", "sandbox_policy")),
                    _sqlite_text(_payload_string(event.payload, "network", "network_policy")),
                    _timestamp_ms(event.timestamp),
                ),
            )
        elif event.event_type in {"token_count", "message_usage"} and (
            not event.source_message_provider_id or source_message_id is not None
        ):
            row = _provider_usage_event_row(
                session_id,
                source_message_id,
                position,
                event,
                provider_usage_baseline=provider_usage_baseline,
            )
            if _provider_usage_event_row_has_evidence(row):
                provider_usage_rows.append(row)
                wrote_provider_usage_events = True
        position += 1
    if session_event_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO session_events (
                session_id, source_message_id, source_message_provider_id,
                position, event_type, summary, payload_json, occurred_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            session_event_rows,
        )
    if agent_policy_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO session_agent_policies (
                session_id, source_message_id, position, approval_policy,
                sandbox_policy, network_policy, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            agent_policy_rows,
        )
    if provider_usage_rows:
        conn.executemany(
            _PROVIDER_USAGE_EVENT_INSERT_SQL,
            provider_usage_rows,
        )
    return SessionEventWriteResult(wrote_provider_usage_events=wrote_provider_usage_events)


_PROVIDER_USAGE_EVENT_INSERT_SQL = """
    INSERT OR REPLACE INTO session_provider_usage_events (
        session_id, source_message_id, position, provider_event_type, model_name,
        last_input_tokens, last_output_tokens, last_cached_input_tokens,
        last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
        total_input_tokens, total_output_tokens, total_cached_input_tokens,
        total_cache_write_tokens, total_reasoning_output_tokens, total_tokens,
        occurred_at_ms
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _provider_usage_event_row(
    session_id: str,
    source_message_id: str | None,
    position: int,
    event: ParsedSessionEvent,
    *,
    provider_usage_baseline: Mapping[str, int] | None = None,
) -> tuple[object, ...]:
    last_usage = _payload_mapping(event.payload, "last_token_usage")
    total_usage = _payload_mapping(event.payload, "total_token_usage")
    total_input = _payload_int(total_usage, "input_tokens")
    total_output = _payload_int(total_usage, "output_tokens")
    total_cache_read = _payload_int(total_usage, "cached_input_tokens")
    total_cache_write = _payload_int(total_usage, "cache_write_tokens")
    total_reasoning = _payload_int(total_usage, "reasoning_output_tokens")
    total_tokens = _payload_int(total_usage, "total_tokens")
    if provider_usage_baseline is not None and event.event_type == "token_count":
        total_input = max(total_input - provider_usage_baseline.get("total_input_tokens", 0), 0)
        total_output = max(total_output - provider_usage_baseline.get("total_output_tokens", 0), 0)
        total_cache_read = max(total_cache_read - provider_usage_baseline.get("total_cached_input_tokens", 0), 0)
        total_cache_write = max(total_cache_write - provider_usage_baseline.get("total_cache_write_tokens", 0), 0)
        total_reasoning = max(total_reasoning - provider_usage_baseline.get("total_reasoning_output_tokens", 0), 0)
        total_tokens = max(total_tokens - provider_usage_baseline.get("total_tokens", 0), 0)
    return (
        session_id,
        source_message_id,
        position,
        _sqlite_text(event.event_type),
        _sqlite_text(_payload_string(event.payload, "model", "model_name")),
        _payload_int(last_usage, "input_tokens"),
        _payload_int(last_usage, "output_tokens"),
        _payload_int(last_usage, "cached_input_tokens"),
        _payload_int(last_usage, "cache_write_tokens"),
        _payload_int(last_usage, "reasoning_output_tokens"),
        _payload_int(last_usage, "total_tokens"),
        total_input,
        total_output,
        total_cache_read,
        total_cache_write,
        total_reasoning,
        total_tokens,
        _timestamp_ms(event.timestamp),
    )


def _provider_usage_event_row_has_evidence(row: tuple[object, ...]) -> bool:
    return any(isinstance(value, int) and value for value in row[5:17])


def _provider_usage_cumulative_baseline(
    conn: sqlite3.Connection,
    parent_session_id: str,
    branch_point_message_id: str,
) -> dict[str, int]:
    branch_row = conn.execute(
        "SELECT session_id, position FROM messages WHERE message_id = ?",
        (branch_point_message_id,),
    ).fetchone()
    if branch_row is None:
        return {}
    baseline_session_id = str(branch_row[0] or parent_session_id)
    branch_position = int(branch_row[1] or 0)
    row = conn.execute(
        """
        SELECT
          e.total_input_tokens,
          e.total_output_tokens,
          e.total_cached_input_tokens,
          e.total_cache_write_tokens,
          e.total_reasoning_output_tokens,
          e.total_tokens
        FROM session_provider_usage_events AS e
        LEFT JOIN messages AS m ON m.message_id = e.source_message_id
        WHERE e.session_id = ?
          AND e.provider_event_type = 'token_count'
          AND (
            m.position <= ?
            OR (e.source_message_id IS NULL AND e.position <= ?)
          )
          AND (
            e.total_input_tokens != 0
            OR e.total_output_tokens != 0
            OR e.total_cached_input_tokens != 0
            OR e.total_cache_write_tokens != 0
            OR e.total_reasoning_output_tokens != 0
            OR e.total_tokens != 0
          )
        ORDER BY e.position DESC
        LIMIT 1
        """,
        (baseline_session_id, branch_position, branch_position),
    ).fetchone()
    if row is None:
        return {}
    return {
        "total_input_tokens": int(row[0] or 0),
        "total_output_tokens": int(row[1] or 0),
        "total_cached_input_tokens": int(row[2] or 0),
        "total_cache_write_tokens": int(row[3] or 0),
        "total_reasoning_output_tokens": int(row[4] or 0),
        "total_tokens": int(row[5] or 0),
    }


def _provider_usage_disjoint_lanes(
    input_with_cached: int,
    output_with_reasoning: int,
    cache_read: int,
    cache_write: int,
) -> tuple[int, int, int, int]:
    """Map Codex ``token_count`` totals onto disjoint billing lanes.

    Codex (OpenAI) reports ``input_tokens`` *inclusive* of
    ``cached_input_tokens`` and ``output_tokens`` *inclusive* of
    ``reasoning_output_tokens``. Verified across the full real corpus
    (1.84M token_count events): ``cached <= input`` on 100% of rows, and
    ``total == input + output`` on 98.9% (reasoning is a subset of output,
    not an additional term).

    The cost model (`archive/semantic/pricing.py:_cost_components`) bills
    ``input`` and ``cache_read`` as *separate additive lanes* — the Anthropic
    convention where ``input`` means fresh/uncached input. So the cached
    portion must be subtracted out of ``input`` or it is billed twice: once at
    the full input rate and again at the discounted cache-read rate. On the
    real archive cached is ~96% of Codex input, so the double-count inflated
    Codex input cost by roughly 8x. Likewise ``reasoning`` is already inside
    ``output``; adding it again over-counts output.

    Returns ``(fresh_input, output, cache_read, cache_write)`` with fresh input
    clamped at zero (defensive; ``input >= cached`` holds on every observed row).
    """
    fresh_input = max(input_with_cached - cache_read, 0)
    return fresh_input, output_with_reasoning, cache_read, cache_write


def _provider_usage_row_has_lane_totals(
    total_input: int,
    total_output: int,
    total_cache_read: int,
    total_cache_write: int,
    total_reasoning: int,
) -> bool:
    """Return true when a cumulative row can be mapped to additive lanes.

    Codex reports reasoning as part of output. A row that carries only
    ``total_reasoning_output_tokens`` is useful evidence, but it cannot replace
    the latest cumulative input/output/cache lanes without zeroing the rollup.
    """

    _ = total_reasoning
    return bool(total_input or total_output or total_cache_read or total_cache_write)


def _aggregate_provider_usage_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Fold provider-reported token-count totals into model usage rows.

    Codex ``token_count`` rows carry a *session-global* cumulative running total
    in their ``total_*`` columns — the counter spans the whole session, not a
    single model. So the cumulative is taken as one session-wide latest value
    (the highest-position ``token_count`` row that carries any ``total_*``),
    attributed to the model named on that row, and written as a single rollup.
    Partitioning the cumulative by model and summing would double-count, because
    each model's "latest cumulative" already includes every prior model's
    tokens (#2472).

    Older/simple token-count rows only expose request-scoped ``last_token_usage``
    (Claude-style per-message per-model deltas); when no cumulative ``total_*``
    appears at all, those are summed per model. Unknown-model events only fall
    back to a session model when exactly one model row exists, keeping
    multi-model sessions auditable rather than guessed.
    """

    rows = conn.execute(
        """
        SELECT provider_event_type, model_name, position,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        return

    existing_models = [
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        ).fetchall()
        if row[0] and str(row[0]).strip()
    ]

    # The cumulative is session-global, so we keep a single latest cumulative
    # for the whole session (rows are ordered by position, so the last row that
    # carries any total_* wins = highest position) attributed to the model named
    # on that row. summed_last_* stays per-model for Claude-style per-message
    # reporting, and is only used when no cumulative total appears at all.
    latest_total: tuple[int, int, int, int, int, int] | None = None
    latest_total_model = ""
    summed_last_by_model: dict[str, list[int]] = {}

    for row in rows:
        model_name = str(row[1]).strip() if row[1] else ""
        if not model_name:
            model_name = existing_models[0] if len(existing_models) == 1 else ""
        if not model_name:
            continue

        last_input = int(row[3] or 0)
        last_output = int(row[4] or 0)
        last_cache_read = int(row[5] or 0)
        last_cache_write = int(row[6] or 0)
        last_reasoning = int(row[7] or 0)
        last_total = int(row[8] or 0)
        total_input = int(row[9] or 0)
        total_output = int(row[10] or 0)
        total_cache_read = int(row[11] or 0)
        total_cache_write = int(row[12] or 0)
        total_reasoning = int(row[13] or 0)
        total_tokens = int(row[14] or 0)

        if _provider_usage_row_has_lane_totals(
            total_input, total_output, total_cache_read, total_cache_write, total_reasoning
        ):
            latest_total = (
                total_input,
                total_output,
                total_cache_read,
                total_cache_write,
                total_reasoning,
                total_tokens,
            )
            latest_total_model = model_name
            continue

        if last_input or last_output or last_cache_read or last_cache_write or last_reasoning or last_total:
            bucket = summed_last_by_model.setdefault(model_name, [0, 0, 0, 0, 0])
            bucket[0] += last_input
            bucket[1] += last_output
            bucket[2] += last_cache_read
            bucket[3] += last_cache_write
            bucket[4] += last_reasoning

    if latest_total is not None:
        # Session-global cumulative: one rollup for the latest model. The
        # cumulative already subsumes every per-request last_*, so summed_last
        # rows are intentionally not written (writing them too double-counts).
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            latest_total[0], latest_total[1], latest_total[2], latest_total[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            latest_total_model,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )
        return

    for model_name, summed_totals in summed_last_by_model.items():
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            summed_totals[0], summed_totals[1], summed_totals[2], summed_totals[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            model_name,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )


def _aggregate_appended_provider_usage_into_model_usage(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    start_position: int,
) -> None:
    """Fold only newly appended provider usage events into model usage rows."""

    rows = conn.execute(
        """
        SELECT model_name, position,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
          AND position >= ?
        ORDER BY position
        """,
        (session_id, start_position),
    ).fetchall()
    if not rows:
        return

    existing_models = _provider_usage_existing_models(conn, session_id)
    # The cumulative total_* is session-global (see the full-write aggregator).
    # The highest-position appended row that carries any total_* therefore holds
    # the authoritative running total for the *whole* session, including rows
    # before start_position, so we keep one session-wide latest cumulative
    # rather than partitioning it per model (#2472).
    latest_total: tuple[int, int, int, int, int, int] | None = None
    latest_total_model = ""
    summed_last_by_model: dict[str, list[int]] = {}

    for row in rows:
        model_name = _provider_usage_model_name(row[0], existing_models)
        if not model_name:
            continue

        last_input = int(row[2] or 0)
        last_output = int(row[3] or 0)
        last_cache_read = int(row[4] or 0)
        last_cache_write = int(row[5] or 0)
        last_reasoning = int(row[6] or 0)
        last_total = int(row[7] or 0)
        total_input = int(row[8] or 0)
        total_output = int(row[9] or 0)
        total_cache_read = int(row[10] or 0)
        total_cache_write = int(row[11] or 0)
        total_reasoning = int(row[12] or 0)
        total_tokens = int(row[13] or 0)

        if _provider_usage_row_has_lane_totals(
            total_input, total_output, total_cache_read, total_cache_write, total_reasoning
        ):
            latest_total = (
                total_input,
                total_output,
                total_cache_read,
                total_cache_write,
                total_reasoning,
                total_tokens,
            )
            latest_total_model = model_name
            continue

        if last_input or last_output or last_cache_read or last_cache_write or last_reasoning or last_total:
            bucket = summed_last_by_model.setdefault(model_name, [0, 0, 0, 0, 0])
            bucket[0] += last_input
            bucket[1] += last_output
            bucket[2] += last_cache_read
            bucket[3] += last_cache_write
            bucket[4] += last_reasoning

    if latest_total is not None:
        # Overwrite the single session-global cumulative rollup. If the model
        # switched since a prior append window, the earlier model's cumulative
        # rollup is now stale (the new cumulative already subsumes it); clear
        # those stale origin_reported cumulative rows so they are not summed
        # back in alongside the new latest.
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            latest_total[0], latest_total[1], latest_total[2], latest_total[3]
        )
        _upsert_provider_usage_model_rollup(
            conn,
            session_id,
            latest_total_model,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )
        _clear_stale_cumulative_rollups(conn, session_id, keep_model=latest_total_model)
        return

    for model_name, summed_totals in summed_last_by_model.items():
        if _provider_usage_has_cumulative_total(conn, session_id, model_name):
            continue
        lane_input, lane_output, lane_cache_read, lane_cache_write = _provider_usage_disjoint_lanes(
            summed_totals[0], summed_totals[1], summed_totals[2], summed_totals[3]
        )
        _increment_provider_usage_model_rollup(
            conn,
            session_id,
            model_name,
            input_tokens=lane_input,
            output_tokens=lane_output,
            cache_read_tokens=lane_cache_read,
            cache_write_tokens=lane_cache_write,
        )


def _provider_usage_existing_models(conn: sqlite3.Connection, session_id: str) -> list[str]:
    return [
        str(row[0]).strip()
        for row in conn.execute(
            "SELECT model_name FROM session_model_usage WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        ).fetchall()
        if row[0] and str(row[0]).strip()
    ]


def _provider_usage_model_name(model_name: object, existing_models: Sequence[str]) -> str:
    resolved = str(model_name).strip() if model_name else ""
    if resolved:
        return resolved
    return existing_models[0] if len(existing_models) == 1 else ""


def _provider_usage_has_cumulative_total(conn: sqlite3.Connection, session_id: str, model_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_provider_usage_events
        WHERE session_id = ?
          AND provider_event_type = 'token_count'
          AND model_name = ?
          AND (
            total_input_tokens != 0
            OR total_output_tokens != 0
            OR total_cached_input_tokens != 0
            OR total_cache_write_tokens != 0
            OR total_reasoning_output_tokens != 0
            OR total_tokens != 0
          )
        LIMIT 1
        """,
        (session_id, model_name),
    ).fetchone()
    return row is not None


def _clear_stale_cumulative_rollups(conn: sqlite3.Connection, session_id: str, *, keep_model: str) -> None:
    """Zero stale cumulative-rollup token totals for all models except ``keep_model``.

    The Codex cumulative total is session-global, so exactly one rollup row
    should carry it. When an append window's latest cumulative is attributed to
    a different model than a previous window, the earlier model's rollup still
    holds a (now-subsumed) cumulative; left in place it would be summed back in
    on read. This resets those stale token counts to zero while keeping the
    model row itself (#2472).

    Scoped to models with no genuine per-message token evidence (``NOT
    EXISTS`` in ``messages``): a row's tokens can only have come from the
    (now-stale) provider-usage-event cumulative mechanism this function is
    cleaning up after, never from ``_aggregate_message_tokens_into_model_usage``.
    Before polylogue-shnc this was scoped by ``cost_provenance =
    'origin_reported'``, which worked only because that label was, at the
    time, written exclusively by the cumulative-rollup path; it no longer
    discriminates cleanly once provider-usage-token rollups are catalog-priced
    onto the same ``'priced'`` label real per-message pricing uses (see
    ``_price_provider_usage_tokens``), so this checks the real per-message
    evidence directly instead of a provenance string that used to be a proxy
    for it.
    """
    conn.execute(
        """
        UPDATE session_model_usage
        SET input_tokens = 0,
            output_tokens = 0,
            cache_read_tokens = 0,
            cache_write_tokens = 0,
            cost_usd = NULL,
            cost_provenance = NULL
        WHERE session_id = ?
          AND model_name != ?
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.session_id = session_model_usage.session_id
                AND m.model_name = session_model_usage.model_name
                AND (
                    COALESCE(m.input_tokens, 0) != 0
                    OR COALESCE(m.output_tokens, 0) != 0
                    OR COALESCE(m.cache_read_tokens, 0) != 0
                    OR COALESCE(m.cache_write_tokens, 0) != 0
                )
          )
        """,
        (session_id, keep_model),
    )


def _price_provider_usage_tokens(
    conn: sqlite3.Connection,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> tuple[str | None, float | None]:
    """Catalog-price disjoint-lane token totals for a provider-usage rollup row.

    Returns ``(cost_provenance, cost_usd)``. Mirrors
    ``_aggregate_message_tokens_into_model_usage``'s pricing (same catalog,
    same "no fabrication" contract): a catalog hit with billable tokens > 0
    yields ``'priced'`` plus a real ``cost_usd``; anything else -- unknown
    model, no catalog entry, zero billable tokens -- yields
    ``cost_provenance = None`` (no claim), never a 'priced'/'origin_reported'
    label with a NULL cost (polylogue-shnc: 5,016 rows previously asserted
    ``cost_provenance = 'priced'`` with NULL cost and NULL catalog).

    ``cost_provenance`` is deliberately NOT ``'origin_reported'`` here: that
    label is reserved for a genuine provider-reported DOLLAR total
    (``sessions.reported_cost_usd``, polylogue-gt1z) -- these rows carry
    provider-reported TOKEN counts priced against the catalog, which is a
    different evidentiary claim. Conflating the two previously made
    ``list_cost_rollup_insights`` read a catalog estimate as if OpenAI itself
    had reported that dollar figure (``archive.py``'s
    ``provider_reported_usd=... if provenance in {"exact","origin_reported"}``).
    """
    from polylogue.archive.semantic.pricing import PRICING, _normalize_model, estimate_cost

    normalized = _normalize_model(model_name)
    billable = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
    if normalized not in PRICING or billable <= 0:
        return None, None
    cost_usd = estimate_cost(input_tokens, output_tokens, model_name, cache_read_tokens, cache_write_tokens)
    return "priced", cost_usd


def _upsert_provider_usage_model_rollup(
    conn: sqlite3.Connection,
    session_id: str,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> None:
    input_tokens = max(int(input_tokens), 0)
    output_tokens = max(int(output_tokens), 0)
    cache_read_tokens = max(int(cache_read_tokens), 0)
    cache_write_tokens = max(int(cache_write_tokens), 0)
    cost_provenance, cost_usd = _price_provider_usage_tokens(
        conn,
        model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            cost_provenance, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            input_tokens       = excluded.input_tokens,
            output_tokens      = excluded.output_tokens,
            cache_read_tokens  = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            cost_provenance    = excluded.cost_provenance,
            cost_usd           = excluded.cost_usd
        """,
        (
            session_id,
            model_name,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            cost_provenance,
            cost_usd,
        ),
    )


def _increment_provider_usage_model_rollup(
    conn: sqlite3.Connection,
    session_id: str,
    model_name: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> None:
    input_tokens = max(int(input_tokens), 0)
    output_tokens = max(int(output_tokens), 0)
    cache_read_tokens = max(int(cache_read_tokens), 0)
    cache_write_tokens = max(int(cache_write_tokens), 0)
    existing = conn.execute(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        FROM session_model_usage
        WHERE session_id = ? AND model_name = ?
        """,
        (session_id, model_name),
    ).fetchone()
    total_input = int((existing[0] if existing else 0) or 0) + input_tokens
    total_output = int((existing[1] if existing else 0) or 0) + output_tokens
    total_cache_read = int((existing[2] if existing else 0) or 0) + cache_read_tokens
    total_cache_write = int((existing[3] if existing else 0) or 0) + cache_write_tokens
    cost_provenance, cost_usd = _price_provider_usage_tokens(
        conn,
        model_name,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_read_tokens=total_cache_read,
        cache_write_tokens=total_cache_write,
    )
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            cost_provenance, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, model_name) DO UPDATE SET
            input_tokens       = excluded.input_tokens,
            output_tokens      = excluded.output_tokens,
            cache_read_tokens  = excluded.cache_read_tokens,
            cache_write_tokens = excluded.cache_write_tokens,
            cost_provenance    = excluded.cost_provenance,
            cost_usd           = excluded.cost_usd
        """,
        (
            session_id,
            model_name,
            total_input,
            total_output,
            total_cache_read,
            total_cache_write,
            cost_provenance,
            cost_usd,
        ),
    )


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
    model_names = {model_name.strip() for model_name in session.models_used if model_name.strip()}
    model_names.update(message.model_name.strip() for message in session.messages if message.model_name)
    # NULL, not 'origin_reported': this is a skeleton placeholder for a
    # session's declared model before any pricing pass has run (typically
    # overwritten within this same call by _aggregate_message_tokens_into_
    # model_usage below, or later by a provider-usage-event rollup). It makes
    # no cost claim yet, so it must not carry a provenance string that
    # asserts one -- 'origin_reported' now means a genuine provider-reported
    # dollar figure (polylogue-shnc/polylogue-gt1z), which this row does not
    # have.
    model_usage_sql = (
        """
        INSERT OR REPLACE INTO session_model_usage (
            session_id, model_name, cost_provenance
        ) VALUES (?, ?, NULL)
        """
        if replace_existing_model_rows
        else """
        INSERT INTO session_model_usage (
            session_id, model_name, cost_provenance
        ) VALUES (?, ?, NULL)
        ON CONFLICT(session_id, model_name) DO NOTHING
        """
    )
    for model_name in sorted(model_names):
        conn.execute(model_usage_sql, (session_id, _sqlite_text(model_name)))
    if aggregate_message_tokens:
        _aggregate_message_tokens_into_model_usage(conn, session_id)


def _aggregate_message_tokens_into_model_usage(conn: sqlite3.Connection, session_id: str) -> None:
    """Aggregate per-message token counts into session_model_usage and compute cost_usd.

    Called after messages are written (and after skeleton model-usage rows exist).
    Handles both full-write and merge-append paths: it always reads ALL messages
    currently in the DB for the session, so the token sums stay consistent with
    the full message set regardless of append ordering.

    Models with no messages carrying token data keep DEFAULT 0 token counts.
    Models with no catalog price entry, or zero billable tokens, get
    cost_provenance = NULL / cost_usd = NULL together -- never 'priced' with
    a NULL cost (polylogue-shnc: that self-contradiction was live on 5,016
    rows).

    Empty or NULL model_name values in the messages table are excluded from
    aggregation (the model is unknown so pricing is impossible).

    The UPSERT only overwrites an existing row when the new message-walked
    token total is >= what is already stored (monotonic-safe), so a
    provider-usage-event cumulative rollup (``_upsert_provider_usage_model_
    rollup``/``_increment_provider_usage_model_rollup``, typically far larger
    for Codex since messages rarely carry its per-message usage) is never
    clobbered by a smaller/zero message-walk result on a later unrelated
    write. Before polylogue-shnc this was scoped by ``cost_provenance =
    'origin_reported'``, which stopped discriminating once provider-usage
    rollups started sharing the 'priced' label with real message-derived
    pricing (see ``_price_provider_usage_tokens``).
    """
    from polylogue.archive.semantic.pricing import PRICING, _normalize_model, estimate_cost

    # Aggregate token counts from the messages table for all known models.
    token_rows = conn.execute(
        """
        SELECT model_name,
               SUM(input_tokens)        AS sum_input,
               SUM(output_tokens)       AS sum_output,
               SUM(cache_read_tokens)   AS sum_cache_read,
               SUM(cache_write_tokens)  AS sum_cache_write,
               COUNT(*)                 AS msg_count
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

        # Compute cost_usd from the curated catalog when a price entry exists.
        # estimate_cost() reads the in-memory PRICING dict directly -- there
        # is no DB-backed rate mirror to keep in sync (polylogue-v2mg dropped
        # model_prices, and polylogue-resk dropped the price_catalogs
        # catalog-identity table, as zero-consumer).
        normalized = _normalize_model(model_name)
        billable = sum_input + sum_output + sum_cache_read + sum_cache_write
        if normalized in PRICING and billable > 0:
            cost_usd: float | None = estimate_cost(sum_input, sum_output, model_name, sum_cache_read, sum_cache_write)
            row_provenance: str | None = "priced"
        else:
            # polylogue-shnc: no catalog price (or no billable tokens) means no
            # claim at all -- NOT 'priced' with a NULL cost_usd, which is the
            # exact self-contradiction the forensic audit found on 5,016 live
            # rows.
            cost_usd = None
            row_provenance = None

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
                cost_usd,
                cost_provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, model_name) DO UPDATE SET
                input_tokens       = excluded.input_tokens,
                output_tokens      = excluded.output_tokens,
                cache_read_tokens  = excluded.cache_read_tokens,
                cache_write_tokens = excluded.cache_write_tokens,
                message_count      = excluded.message_count,
                cost_usd           = excluded.cost_usd,
                cost_provenance    = excluded.cost_provenance
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
                cost_usd,
                row_provenance,
            ),
        )


def _write_repo_edges(conn: sqlite3.Connection, session_id: str, session: ParsedSession) -> None:
    observed_at_ms = _timestamp_ms(session.updated_at) or _timestamp_ms(session.created_at)
    raw_root_paths = tuple(path.strip() for path in session.working_directories if path.strip())
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
        if session.git_commit_hash:
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


def _normalized_messages(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    active_leaf_count = sum(1 for message in messages if message.is_active_leaf)
    if active_leaf_count == 1 or not messages:
        return messages
    return [
        message.model_copy(update={"is_active_leaf": position == len(messages) - 1})
        for position, message in enumerate(messages)
    ]


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

_SIG_FIELD_SEP = "\x1f"
_SIG_BLOCK_SEP = "\x1e"
# The synchronous envelope reader below is recursive, so retain its conservative
# Python-stack guard. Writer-side signature composition is iterative and shares
# the async reader's much larger runaway backstop.
_MAX_LINEAGE_DEPTH = 64
_MAX_WRITER_LINEAGE_DEPTH = LINEAGE_ITERATIVE_DEPTH_LIMIT


def _canonical_json(value: object) -> str:
    """Stable JSON for signature comparison; accepts a value or a JSON string."""
    if value is None:
        return "null"
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(parsed)


def _message_signature_from_blocks(role: str, block_fields: list[tuple[str, str, str, str]]) -> str:
    parts = [role]
    for block_type, text, tool_name, tool_input in block_fields:
        parts.append(_SIG_FIELD_SEP.join((block_type, text, tool_name, tool_input)))
    return hashlib.sha256(_SIG_BLOCK_SEP.join(parts).encode("utf-8", "surrogatepass")).hexdigest()


def _parsed_message_signature(message: ParsedMessage) -> str:
    role = _enum_value(message.role) or ""
    fields: list[tuple[str, str, str, str]] = []
    for block in _message_blocks(message):
        # Serialize tool_input through the same `_json_dumps` the writer uses to
        # store it, then canonicalize — so the parsed-side signature matches the
        # DB-side signature (which canonicalizes the stored JSON string). Calling
        # `_canonical_json` on the raw value would mis-handle scalar strings, which
        # it treats as JSON to re-parse (#2467 audit M6).
        tool_input = _canonical_json(_json_dumps(block.tool_input)) if block.tool_input is not None else "null"
        fields.append(
            (
                _block_type(block).value,
                block.text or "",
                block.tool_name or "",
                tool_input,
            )
        )
    return _message_signature_from_blocks(role, fields)


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
    signatures: list[str] = []
    for message in messages:
        if message.message_type is MessageType.SUMMARY:
            break
        signatures.append(_parsed_message_signature(message))
    return signatures


def _acompact_content_membership_ratio(
    parent_composed: Sequence[tuple[str, str]],
    child_prefix_signatures: Sequence[str],
) -> float | None:
    """Return multiset content membership of an acompact prefix in its parent.

    Classification uses membership rather than contiguous alignment: the former
    answers whether this artifact belongs to the asserted parent at all, while
    `_extract_prefix_tail` remains the stricter loss-prevention gate for deleting
    inherited rows.  Duplicate signatures are bounded by parent multiplicity so
    repeated boilerplate cannot manufacture overlap.
    """
    if not child_prefix_signatures:
        return None
    if not parent_composed:
        return 0.0
    parent_counts = Counter(signature for _message_id, signature in parent_composed)
    matching_count = 0
    for signature in child_prefix_signatures:
        if parent_counts[signature] <= 0:
            continue
        parent_counts[signature] -= 1
        matching_count += 1
    return matching_count / len(child_prefix_signatures)


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


def _own_db_signatures(conn: sqlite3.Connection, session_id: str) -> list[tuple[str, str]]:
    """Return ``[(message_id, signature), ...]`` for ``session_id``'s OWN stored
    message rows (no inherited prefix). This is the expensive SQL+SHA-256 leg of
    composition; it depends only on the session's own rows, so it can be memoized
    per ingest batch and invalidated whenever those rows change."""
    own_rows = conn.execute(
        """
        SELECT m.message_id, m.position, m.variant_index, m.role,
               b.block_type, b.text, b.tool_name, b.tool_input
        FROM messages m
        LEFT JOIN blocks b ON b.session_id = m.session_id AND b.message_id = m.message_id
        WHERE m.session_id = ?
        ORDER BY m.position, m.variant_index, b.position
        """,
        (session_id,),
    ).fetchall()
    own: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_role = ""
    cur_blocks: list[tuple[str, str, str, str]] = []

    def flush() -> None:
        if cur_id is not None:
            own.append((cur_id, _message_signature_from_blocks(cur_role, cur_blocks)))

    for message_id, _position, _variant_index, role, block_type, text, tool_name, tool_input in own_rows:
        if message_id != cur_id:
            flush()
            cur_id = message_id
            cur_role = role or ""
            cur_blocks = []
        if block_type is not None:
            cur_blocks.append(
                (
                    block_type,
                    text or "",
                    tool_name or "",
                    _canonical_json(tool_input) if tool_input is not None else "null",
                )
            )
    flush()
    return own


def _composed_db_signatures(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    composed_cache: dict[str, list[tuple[str, str]]] | None = None,
    _depth: int = 0,
) -> list[tuple[str, str]]:
    """Return ``[(message_id, signature), ...]`` for ``session_id``'s composed
    transcript (its inherited prefix + own tail). Walk the lineage iteratively
    and compose root-to-leaf, mirroring the async read path without inheriting
    the synchronous envelope reader's Python-stack limit.

    When ``cache`` is supplied, each session's OWN signatures are memoized by
    ``session_id`` for the life of one ingest batch. When ``composed_cache`` is
    supplied, composed (prefix+own) results are memoized only for one graph
    resolution operation; that avoids cross-write stale ancestors while letting
    sibling delayed-tail repairs share the same parent composition.

    Callers typically already hold a write transaction (single-writer ingest),
    but if ``conn`` is not already inside one this wraps the whole recursive
    composition in one deferred read transaction (4ts.4), matching
    ``read_archive_session_envelope``'s guard against a torn read.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN DEFERRED")
        try:
            return _composed_db_signatures(conn, session_id, cache=cache, composed_cache=composed_cache, _depth=_depth)
        finally:
            conn.execute("ROLLBACK")

    def own_signatures(target_session_id: str) -> list[tuple[str, str]]:
        own = cache.get(target_session_id) if cache is not None else None
        if own is None:
            own = _own_db_signatures(conn, target_session_id)
            if cache is not None:
                cache[target_session_id] = own
        return own

    # Collect (child, branch point, child-owned rows) leaf-first, then compose
    # from the oldest reached ancestor down. A visited set is the real cycle
    # guard; the shared depth limit only bounds malformed acyclic chains.
    chain: list[tuple[str, str, list[tuple[str, str]]]] = []
    visited = {session_id}
    cursor_session_id = session_id
    composed: list[tuple[str, str]] | None = None
    remaining_depth = max(0, _MAX_WRITER_LINEAGE_DEPTH - _depth)
    for _ in range(remaining_depth):
        if composed_cache is not None:
            cached_composed = composed_cache.get(cursor_session_id)
            if cached_composed is not None:
                composed = cached_composed
                break
        own = own_signatures(cursor_session_id)
        edge = conn.execute(
            """
            SELECT resolved_dst_session_id, branch_point_message_id
            FROM session_links
            WHERE src_session_id = ?
              AND inheritance = 'prefix-sharing'
              AND resolved_dst_session_id IS NOT NULL
              AND branch_point_message_id IS NOT NULL
            LIMIT 1
            """,
            (cursor_session_id,),
        ).fetchone()
        if edge is None:
            composed = own
            if composed_cache is not None:
                composed_cache[cursor_session_id] = composed
            break
        parent_id, branch_point_message_id = str(edge[0]), str(edge[1])
        if parent_id in visited:
            composed = own
            if composed_cache is not None:
                composed_cache[cursor_session_id] = composed
            break
        chain.append((cursor_session_id, branch_point_message_id, own))
        visited.add(parent_id)
        cursor_session_id = parent_id
    if composed is None:
        # The runaway guard was exhausted. Match the async reader: start with
        # the oldest reached session's own rows, then compose the retained
        # descendant chain. Ancestors beyond the common cutoff are omitted.
        composed = own_signatures(cursor_session_id)
        if composed_cache is not None:
            composed_cache[cursor_session_id] = composed

    for child_session_id, branch_point_message_id, own in reversed(chain):
        prefix: list[tuple[str, str]] = []
        found = False
        for entry in composed:
            prefix.append(entry)
            if entry[0] == branch_point_message_id:
                found = True
                break
        # A genuinely missing branch point is not a license to inherit a nearby
        # or entire prefix. This matches the async reader's dangling-edge rule.
        composed = (prefix if found else []) + own
        if composed_cache is not None:
            composed_cache[child_session_id] = composed
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
    trigger BODIES (see ``polylogue.storage.fts.sql``) and, since
    polylogue-v6i3, the ``blocks_command_trigram_{ai,ad,au}`` bodies as well
    (see ``archive_tiers/index.py``) -- so both surfaces get the explicit
    session-scoped delete/re-insert bracketing below; the triggers stay
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
    ``messages_fts``/``blocks_command_trigram`` empty throughout replay and
    repopulates both archive-wide exactly once at readiness, so any
    per-session insert here would just be redone work.
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
    # polylogue-v6i3 gated blocks_command_trigram's trigger bodies on this same
    # guard row, so the trigram index needs the same explicit session-scoped
    # delete-then-reinsert bracketing or the guarded block mutations leave
    # stale external-content postings (later LIKE queries then raise
    # "fts5: missing row N from content table"). The delete MUST run here,
    # while the doomed prefix blocks still exist -- external-content FTS5
    # deletion needs the OLD text to locate postings.
    conn.execute(trigram_delete_session_rows_sql(), (session_id,))
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
        # Trigram companion: repopulate from whatever tool_use blocks survive.
        conn.execute(trigram_insert_session_rows_sql(), (session_id,))
        conn.execute(
            "DELETE FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        )


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
) -> None:
    """Normalize a child that was stored whole because its parent was ingested
    later (#2467). Aligns the child's already-stored messages against the parent's
    composed transcript, deletes the inherited-prefix rows, and records the edge.
    Only runs while the lineage edge is still un-extracted (``inheritance`` NULL).
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
        return
    dst_origin, dst_native_id, link_type = edge
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
            WHERE src_session_id = ? AND dst_origin = ? AND dst_native_id = ? AND link_type = ?
            """,
            (
                target_link_type,
                branch_point_message_id,
                inheritance,
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
        return

    t0 = time.perf_counter()
    k = 0
    limit = min(len(parent_composed), len(child_composed))
    while k < limit and parent_composed[k][1] == child_composed[k][1]:
        k += 1
    record_substage("signature_compare", t0)

    if k == 0:
        t0 = time.perf_counter()
        _set_edge(None, "spawned-fresh", next_link_type=resolved_link_type)
        record_substage("edge_update", t0)
        return
    prefix_message_ids = [child_composed[i][0] for i in range(k)]
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
        parent_session_id,
        parent_composed[k - 1][0],
        prefix_message_ids=prefix_message_ids,
    )
    record_substage("provider_usage_tail", t0)
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
        next_link_type=resolved_link_type,
    )
    record_substage("edge_update", t0)
    t0 = time.perf_counter()
    _refresh_session_counts(conn, child_session_id)
    record_substage("count_refresh", t0)


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


def _repair_stale_prefix_branch_points_db(
    conn: sqlite3.Connection,
    session_ids: set[str] | tuple[str, ...] | list[str] | None = None,
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    composed_cache: dict[str, list[tuple[str, str]]] | None = None,
    limit: int | None = None,
) -> int:
    """Repair stale immediate-parent branch-point IDs in prefix-sharing edges.

    Older lineage rows can name a branch point as ``<immediate-parent>:<suffix>``
    even after that parent has itself been normalized to tail-only storage. The
    composed reader can only find physical ancestor message IDs, so these stale
    rows make the child bail to its own tail. If the suffix maps to exactly one
    message in the resolved parent's composed transcript, update the edge to the
    composed message id. Ambiguous or unmappable rows stay visible to validation.
    """
    params: list[object] = []
    scope_clause = ""
    if session_ids is not None:
        scoped = sorted(session_ids)
        if not scoped:
            return 0
        placeholders = ",".join("?" for _ in scoped)
        scope_clause = f"AND l.src_session_id IN ({placeholders})"
        params.extend(scoped)
    limit_clause = ""
    if limit is not None:
        if limit < 1:
            return 0
        limit_clause = "LIMIT ?"
        params.append(limit)
    rows = conn.execute(
        f"""
        SELECT l.src_session_id, l.resolved_dst_session_id, l.branch_point_message_id
        FROM session_links l
        WHERE l.inheritance = 'prefix-sharing'
          AND l.resolved_dst_session_id IS NOT NULL
          AND l.branch_point_message_id IS NOT NULL
          {scope_clause}
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = l.branch_point_message_id
          )
        ORDER BY l.src_session_id
        {limit_clause}
        """,
        tuple(params),
    ).fetchall()
    repaired = 0
    local_composed_cache: dict[str, list[tuple[str, str]]] = composed_cache if composed_cache is not None else {}
    for src_session_id, parent_session_id, branch_point_message_id in rows:
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


def repair_stale_prefix_branch_points(conn: sqlite3.Connection, *, limit: int | None = None) -> int:
    """Repair stale prefix-sharing branch points across the current index tier."""
    return _repair_stale_prefix_branch_points_db(conn, limit=limit)


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
    conn.execute("DELETE FROM attachment_refs WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM paste_spans WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM blocks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def _delete_prefix_message_dependents(conn: sqlite3.Connection, prefix_message_ids: Sequence[str]) -> None:
    """Mirror message FK side effects when bulk ingest has foreign keys off."""
    if not prefix_message_ids:
        return
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
    for table in ("web_content_constructs", "attachment_refs", "paste_spans", "blocks"):
        conn.execute(
            f"DELETE FROM {table} WHERE message_id IN ({placeholders})",
            params,
        )


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
    parent_session_id: str,
    branch_point_message_id: str,
    *,
    prefix_message_ids: Sequence[str],
) -> None:
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
    baseline = _provider_usage_cumulative_baseline(conn, parent_session_id, branch_point_message_id)
    if baseline:
        conn.execute(
            """
            UPDATE session_provider_usage_events
            SET total_input_tokens = MAX(total_input_tokens - ?, 0),
                total_output_tokens = MAX(total_output_tokens - ?, 0),
                total_cached_input_tokens = MAX(total_cached_input_tokens - ?, 0),
                total_cache_write_tokens = MAX(total_cache_write_tokens - ?, 0),
                total_reasoning_output_tokens = MAX(total_reasoning_output_tokens - ?, 0),
                total_tokens = MAX(total_tokens - ?, 0)
            WHERE session_id = ?
              AND provider_event_type = 'token_count'
            """,
            (
                baseline.get("total_input_tokens", 0),
                baseline.get("total_output_tokens", 0),
                baseline.get("total_cached_input_tokens", 0),
                baseline.get("total_cache_write_tokens", 0),
                baseline.get("total_reasoning_output_tokens", 0),
                baseline.get("total_tokens", 0),
                child_session_id,
            ),
        )
    conn.execute(
        """
        DELETE FROM session_provider_usage_events
        WHERE session_id = ?
          AND last_input_tokens = 0
          AND last_output_tokens = 0
          AND last_cached_input_tokens = 0
          AND last_cache_write_tokens = 0
          AND last_reasoning_output_tokens = 0
          AND last_total_tokens = 0
          AND total_input_tokens = 0
          AND total_output_tokens = 0
          AND total_cached_input_tokens = 0
          AND total_cache_write_tokens = 0
          AND total_reasoning_output_tokens = 0
          AND total_tokens = 0
        """,
        (child_session_id,),
    )
    # Clear rows populated by the (now stale) provider-usage-event rollup
    # before re-deriving them below, scoped the same way
    # _clear_stale_cumulative_rollups is (polylogue-shnc): a model with no
    # genuine per-message token evidence can only hold provider-usage-rollup
    # tokens, never real message-derived pricing, so it is always safe to
    # clear and re-derive. Before polylogue-shnc this was scoped by
    # ``cost_provenance = 'origin_reported'``, which stopped discriminating
    # once provider-usage rollups started sharing the 'priced' label with
    # real message-derived pricing.
    conn.execute(
        """
        DELETE FROM session_model_usage
        WHERE session_id = ?
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.session_id = session_model_usage.session_id
                AND m.model_name = session_model_usage.model_name
                AND (
                    COALESCE(m.input_tokens, 0) != 0
                    OR COALESCE(m.output_tokens, 0) != 0
                    OR COALESCE(m.cache_read_tokens, 0) != 0
                    OR COALESCE(m.cache_write_tokens, 0) != 0
                )
          )
        """,
        (child_session_id,),
    )
    _aggregate_provider_usage_into_model_usage(conn, child_session_id)


def _extract_prefix_tail(
    conn: sqlite3.Connection,
    parent_session_id: str,
    messages: list[ParsedMessage],
    *,
    cache: dict[str, list[tuple[str, str]]] | None = None,
    parent_composed: list[tuple[str, str]] | None = None,
) -> tuple[str | None, str | None, list[ParsedMessage], dict[str, str]]:
    """Align ``messages`` (the child's full parsed messages, which replay the
    parent's prefix) against the parent's composed transcript. Returns
    ``(branch_point_message_id, inheritance, tail_messages, inherited_refs)``.
    ``inherited_refs`` maps unambiguous provider-local child message ids to the
    canonical parent message rows that physically own the replayed prefix.
    """
    if parent_composed is None:
        parent_composed = _composed_db_signatures(conn, parent_session_id, cache=cache)
    if not parent_composed:
        return (None, "spawned-fresh", messages, {})
    child_sigs = [_parsed_message_signature(m) for m in messages]
    k = 0
    limit = min(len(parent_composed), len(child_sigs))
    while k < limit and parent_composed[k][1] == child_sigs[k]:
        k += 1
    if k == 0:
        return (None, "spawned-fresh", messages, {})
    branch_point_message_id = parent_composed[k - 1][0]
    duplicate_native_ids = _duplicate_message_native_ids(messages)
    inherited_refs: dict[str, str] = {}
    for index, message in enumerate(messages[:k]):
        provider_id = message.provider_message_id
        if provider_id and provider_id not in duplicate_native_ids:
            inherited_refs[provider_id] = parent_composed[index][0]
    return (branch_point_message_id, "prefix-sharing", messages[k:], inherited_refs)


def _prefix_sharing_edge_sync(conn: sqlite3.Connection, session_id: str) -> tuple[str, str] | None:
    """Return ``(parent_session_id, branch_point_message_id)`` for a resolved
    prefix-sharing lineage edge, else ``None``. Mirrors the async reader."""
    row = conn.execute(
        """
        SELECT resolved_dst_session_id, branch_point_message_id
        FROM session_links
        WHERE src_session_id = ?
          AND inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NOT NULL
          AND branch_point_message_id IS NOT NULL
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return (str(row[0]), str(row[1]))


def _existing_parent_session_id(conn: sqlite3.Connection, session: ParsedSession, origin_value: str) -> str | None:
    parent_provider_id = session.parent_session_provider_id
    if not parent_provider_id:
        return None
    parent_session_id = archive_session_id(origin_value, parent_provider_id)
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
        (parent_session_id,),
    ).fetchone()
    return parent_session_id if row is not None else None


def _active_leaf_message_id(
    session_id: str,
    messages: list[ParsedMessage],
    explicit_native_id: str | None,
    *,
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str | None:
    if explicit_native_id:
        matching_messages = [
            (fallback_position, message)
            for fallback_position, message in enumerate(messages)
            if message.provider_message_id == explicit_native_id
        ]
        for fallback_position, message in matching_messages:
            if message.is_active_leaf:
                return _message_id(
                    session_id,
                    message,
                    fallback_position,
                    position_offset=position_offset,
                    duplicate_native_ids=duplicate_native_ids,
                )
        if matching_messages:
            fallback_position, message = matching_messages[0]
            return _message_id(
                session_id,
                message,
                fallback_position,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
            )
    for fallback_position, message in enumerate(messages):
        if message.is_active_leaf:
            return _message_id(
                session_id,
                message,
                fallback_position,
                position_offset=position_offset,
                duplicate_native_ids=duplicate_native_ids,
            )
    return (
        _message_id(
            session_id,
            messages[-1],
            len(messages) - 1,
            position_offset=position_offset,
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
    position_offset: int = 0,
    duplicate_native_ids: frozenset[str] = frozenset(),
) -> str:
    position = position_offset + (message.position if message.position is not None else 0)
    variant_index = message.variant_index if message.variant_index is not None else 0
    return archive_message_id(
        session_id,
        _stored_message_native_id(message, duplicate_native_ids),
        position=position if message.position is not None else position_offset + fallback_position,
        variant_index=variant_index,
    )


def _duplicate_message_native_ids(messages: Iterable[ParsedMessage]) -> frozenset[str]:
    """Native ids that collide once normalized the same way ``messages.native_id`` stores them.

    Counts by the surrogate-substituted (``_sqlite_text``) form, not the raw
    provider string, so two distinct raw ids that collapse onto the same
    U+FFFD-substituted text are treated as ambiguous too -- otherwise the
    ``messages`` UNIQUE generated ``message_id`` column would silently
    resolve the collision via ``INSERT OR REPLACE`` (one message vanishes)
    while Python-side code still believed both had distinct identities.
    """
    counts = Counter(
        normalized for message in messages if (normalized := _sqlite_text(message.provider_message_id)) is not None
    )
    return frozenset(native_id for native_id, count in counts.items() if count > 1)


def _effective_message_native_id(message: ParsedMessage, duplicate_native_ids: frozenset[str]) -> str | None:
    """Return the surrogate-normalized native id, or ``None`` if ambiguous.

    ``duplicate_native_ids`` (from ``_duplicate_message_native_ids``) is keyed
    by the same surrogate-substituted form computed here, so membership is
    always compared apples-to-apples.
    """
    native_id = _sqlite_text(message.provider_message_id)
    if native_id in duplicate_native_ids:
        return None
    return native_id


def _stored_session_native_id(native_id: str) -> str:
    """Return the exact value ``sessions.native_id`` stores for this session.

    Single source of truth for session identity (mirrors the message-level
    ab5bad1f FK-failure fix via ``_stored_message_native_id`` below, never
    given a session-level sibling until polylogue-lyr2). ``_write_session``'s
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

    Beyond duplicate suppression and surrogate substitution
    (``_effective_message_native_id``), this maps an empty or
    whitespace-only native id to ``None`` -- matching
    ``identity_law.message_local_id``'s ``native_id.strip()`` truthiness
    check, which falls back to the ``position.variant_index`` component --
    and strips a non-empty native id, matching
    ``identity_law._required_text``'s own ``.strip()``. Without this, a
    provider-native id of ``"  "`` stores truthy in SQLite (survives the bare
    ``or None`` the DB write used before this helper existed) while
    ``identity_law`` strips it to falsy and falls back to the position/variant
    id -- producing two different message ids for the same message.
    """
    native_id = _effective_message_native_id(message, duplicate_native_ids)
    if native_id is None:
        return None
    stripped = native_id.strip()
    return stripped or None


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


def _timestamp_ms(value: str | None) -> int | None:
    parsed = parse_timestamp(value) if value else None
    return int(parsed.timestamp() * 1000) if parsed is not None else None


def _derive_session_timestamps_from_messages(
    messages: Sequence[ParsedMessage],
) -> tuple[int | None, int | None]:
    """Fallback (created_at_ms, updated_at_ms) from message evidence (#m3p9).

    Called only as a fallback when the provider payload carries no
    session-level ``created_at``/``updated_at`` (or they fail to parse) --
    see the call site in ``write_parsed_session_to_archive``. Returns the
    min/max of ``ParsedMessage.occurred_at_ms`` across ``messages``, or
    ``(None, None)`` when no message carries a timestamp either (a
    genuinely undatable session stays NULL, it is not backdated to the
    ingest wall clock).
    """
    occurred_at_ms_values = [m.occurred_at_ms for m in messages if m.occurred_at_ms is not None]
    if not occurred_at_ms_values:
        return None, None
    return min(occurred_at_ms_values), max(occurred_at_ms_values)


def _event_summary(event: ParsedSessionEvent) -> str | None:
    summary = event.payload.get("summary") or event.payload.get("text")
    return str(summary) if summary is not None else None


def _payload_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _payload_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _payload_int(payload: Mapping[str, object], key: str) -> int:
    return _payload_optional_int(payload, key) or 0


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
    (``storage/insights/session/repo_observations.py``) already uses -- so a
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
    (``storage/insights/session/repo_observations.py``), which already
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


def _attachment_id(_session_id: str, attachment: ParsedAttachment) -> str:
    return _hash_bytes(
        "attachment",
        attachment.provider_attachment_id,
        attachment.provider_file_id or "",
        attachment.provider_drive_id or "",
        attachment.path or "",
        attachment.name or "",
        attachment.mime_type or "",
        str(attachment.size_bytes or 0),
    ).hex()


def _attachment_position(attachment: ParsedAttachment) -> int:
    digest = hashlib.sha256()
    digest.update(attachment.provider_attachment_id.encode("utf-8", errors="surrogatepass"))
    return int.from_bytes(digest.digest()[:4], "big")


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


def _write_attachment_native_ids(conn: sqlite3.Connection, ref_id: str, attachment: ParsedAttachment) -> None:
    native_values = (
        ("attachment", attachment.provider_attachment_id),
        ("file", attachment.provider_file_id),
        ("drive", attachment.provider_drive_id),
        ("url", _attachment_source_url(attachment)),
    )
    for id_kind, native_id in native_values:
        if native_id:
            conn.execute(
                """
                INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id)
                VALUES (?, ?, ?)
                """,
                (ref_id, id_kind, _sqlite_text(native_id)),
            )


def _hash_bytes(*parts: str) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.digest()


def _json_dumps(value: object) -> str:
    return json.dumps(_sqlite_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _sqlite_json_value(value: object) -> object:
    if isinstance(value, str):
        return _sqlite_text(value)
    if isinstance(value, list):
        return [_sqlite_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sqlite_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(_sqlite_text(str(key))): _sqlite_json_value(item) for key, item in value.items()}
    return value


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


def _refresh_session_profile_count(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    table: str,
    column: str,
) -> None:
    count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
    conn.execute(
        f"""
        INSERT INTO session_profiles (session_id, {column})
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            {column} = excluded.{column}
        """,
        (session_id, count),
    )


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


__all__ = [
    "ArchiveAgentPolicy",
    "ArchiveBlockRow",
    "ArchiveInsightMaterialization",
    "ArchiveMessageRow",
    "ArchiveSessionPhase",
    "ArchiveSessionTag",
    "ArchiveSessionEnvelope",
    "ArchiveSessionWorkEvent",
    "read_insight_materialization",
    "read_session_agent_policies",
    "read_session_phases",
    "read_session_tags",
    "read_session_work_events",
    "rebuild_archive_messages_fts",
    "replace_parser_ingest_flag_tags",
    "repo_identity_key",
    "upsert_session_profile_costs",
    "apply_insight_materialization",
    "upsert_insight_materialization",
    "upsert_session_phase",
    "upsert_parser_ingest_flag_tags",
    "upsert_session_tag",
    "upsert_session_work_event",
    "read_archive_session_envelope",
    "search_archive_blocks",
    "write_parsed_session_to_archive",
]
