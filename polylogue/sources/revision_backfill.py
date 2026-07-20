"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Literal

from polylogue import logging as _polylogue_logging
from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.pipeline.services.process_pool import parallel_threads_effective
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import is_stream_record_provider, parse_payload, parse_stream_payload
from polylogue.sources.parsers import hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.sqlite_snapshot import looks_like_sqlite_bytes
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_LOGGER = _polylogue_logging.get_logger(__name__)


def _browser_snapshot_fidelity(ingest_flags: Sequence[str]) -> Literal["dom", "native"] | None:
    """Derive membership-classification browser fidelity from parser ingest flags.

    ``session_revision_membership.classify_membership_revisions`` special-cases
    dom-fallback vs. native browser captures (and, since polylogue-z1c6, a
    genuine non-browser-capture revision outranking any browser capture) --
    but only when ``MembershipRevision.browser_snapshot_fidelity`` is actually
    populated. A plain provider export carries neither flag and is
    ``None`` (not a browser capture at all).
    """
    flags = set(ingest_flags)
    if NATIVE_BROWSER_CAPTURE_INGEST_FLAG in flags or COMPACT_BROWSER_CAPTURE_INGEST_FLAG in flags:
        return "native"
    if DOM_FALLBACK_INGEST_FLAG in flags:
        return "dom"
    return None


@dataclass(frozen=True, slots=True)
class RevisionBackfillResult:
    scanned: int
    classified_full: int
    replayed_logical_sources: int
    quarantined: int
    adoption_deferred: int = 0


@dataclass(frozen=True, slots=True)
class RevisionCensusResult:
    scanned: int
    classified_full: int
    quarantined: int
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]


@dataclass(slots=True)
class _RevisionCensusState:
    scanned: int
    classified: int
    quarantined: int
    censused: set[str]
    membership_candidates: dict[str, set[str]]
    provisional_full_raw_ids: dict[str, set[str]]


@dataclass(slots=True)
class _PrefetchedParse:
    sessions: list[ParsedSession]
    payload_bytes: int
    revision_kind: RawRevisionKind


class RawParsePrefetchCache:
    """Bounded, thread-safe store of parse results computed off the writer hold.

    polylogue-m6tp phase (a): the daemon's parse-stage warmer
    (``polylogue.daemon.parse_prefetch.DaemonParseStage``) populates this
    cache from a bounded ``ThreadPoolExecutor`` BEFORE the raw-materialization
    conveyor's writer-hold pass runs. ``_parse_retained_raws`` below consults
    it first and only falls back to its normal (writer-hold-resident) parse
    on a miss.

    A miss is always safe: it reproduces the exact unmodified parse path, so
    an empty, absent, or partially-warmed cache degrades to identical
    behavior -- never incorrect behavior. This is what makes the cache purely
    additive and lets every existing caller default to ``prefetch_cache=None``
    with zero change in outcome.

    Admission is capped by ``max_inflight_bytes`` (an explicit whale-memory
    budget): a payload that would exceed the remaining budget is silently NOT
    cached and is parsed normally, in the writer hold, when its turn comes.
    """

    def __init__(self, *, max_inflight_bytes: int) -> None:
        if max_inflight_bytes < 1:
            raise ValueError("max_inflight_bytes must be positive")
        self._max_inflight_bytes = max_inflight_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, _PrefetchedParse] = {}
        self._inflight_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, raw_id: str) -> bool:
        with self._lock:
            return raw_id in self._entries

    def try_admit(
        self,
        raw_id: str,
        sessions: list[ParsedSession],
        *,
        payload_bytes: int,
        revision_kind: RawRevisionKind,
    ) -> bool:
        """Admit one already-parsed raw's output. False means the cache
        already held ``raw_id`` or admitting it would exceed the budget --
        either way the caller's parse output is simply discarded, not an
        error: the writer-held pass reparses that raw normally."""
        with self._lock:
            if raw_id in self._entries:
                return False
            if self._inflight_bytes + payload_bytes > self._max_inflight_bytes:
                return False
            self._entries[raw_id] = _PrefetchedParse(sessions, payload_bytes, revision_kind)
            self._inflight_bytes += payload_bytes
            return True

    def pop(self, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind] | None:
        """Remove and return one cached parse result, releasing its budget share."""
        with self._lock:
            entry = self._entries.pop(raw_id, None)
            if entry is None:
                return None
            self._inflight_bytes -= entry.payload_bytes
            return entry.sessions, entry.payload_bytes, entry.revision_kind


class RawRevisionReplayResourceBlockedError(RuntimeError):
    def __init__(self, raw_ids: list[str], limit_bytes: int, total_bytes: int) -> None:
        self.raw_ids = tuple(raw_ids)
        self.limit_bytes = limit_bytes
        self.total_bytes = total_bytes
        super().__init__(f"{len(raw_ids)} raw revision(s) total {total_bytes} bytes exceed replay limit {limit_bytes}")


def _resource_blocked_parser_fingerprint(max_payload_bytes: int) -> str:
    """Return the durable admission identity for one bounded census envelope."""
    return f"revision-membership-v1:resource-blocked:{max_payload_bytes}"


def uncensused_historical_revision_raw_ids(
    archive_root: Path,
    raw_ids: list[str],
    *,
    max_payload_bytes: int | None = None,
) -> tuple[str, ...]:
    """Return inputs whose current parser identity has not been persisted.

    The dedicated receipt proves that the current parser actually observed
    every relevant raw. Durable revision or membership rows alone may have
    been produced by an older parser and therefore cannot establish current
    quiescence.
    """
    if not raw_ids:
        return ()
    placeholders = ",".join("?" for _ in raw_ids)
    resource_blocked_fingerprint = (
        _resource_blocked_parser_fingerprint(max_payload_bytes) if max_payload_bytes is not None else None
    )
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"""
            SELECT r.raw_id
            FROM raw_sessions AS r
            LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
            WHERE r.raw_id IN ({placeholders})
              AND NOT COALESCE(
                  c.parser_fingerprint = 'revision-membership-v1'
                  AND c.status = 'complete',
                  0
              )
              AND NOT COALESCE(
                  c.parser_fingerprint = ?
                  AND c.status = 'failed',
                  0
              )
            ORDER BY r.raw_id
            """,
            [*raw_ids, resource_blocked_fingerprint],
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def record_resource_blocked_revision_census(
    archive_root: Path,
    raw_ids: tuple[str, ...],
    *,
    max_payload_bytes: int,
    total_payload_bytes: int,
) -> None:
    """Persist a non-terminal no-retry receipt for immutable oversized bytes.

    ``failed`` is deliberately truthful: the current parser has not inspected
    the payload.  The fingerprint binds that fact to the exact admission
    envelope, so increasing the envelope (or changing the parser identity)
    re-admits the raw without a timer-driven retry storm.
    """
    if not raw_ids:
        return
    fingerprint = _resource_blocked_parser_fingerprint(max_payload_bytes)
    detail = (
        "current parser census deferred before blob open: "
        f"component payload {total_payload_bytes} exceeds envelope {max_payload_bytes}"
    )
    with sqlite3.connect(archive_root / "source.db") as conn, conn:
        for raw_id in raw_ids:
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json,
                    detail, censused_at_ms
                ) VALUES (?, ?, 'failed', '[]', ?, 0)
                ON CONFLICT(raw_id) DO UPDATE SET
                    parser_fingerprint = excluded.parser_fingerprint,
                    status = excluded.status,
                    logical_keys_json = excluded.logical_keys_json,
                    detail = excluded.detail,
                    censused_at_ms = excluded.censused_at_ms
                """,
                (raw_id, fingerprint, detail),
            )


def _record_raw_authority_parser_census(archive_root: Path, raw_ids: tuple[str, ...]) -> None:
    """Persist per-raw current-parser completion without changing governance."""
    if not raw_ids:
        return
    with sqlite3.connect(archive_root / "source.db") as conn, conn:
        for raw_id in raw_ids:
            raw = conn.execute(
                """
                SELECT logical_source_key, revision_kind
                FROM raw_sessions WHERE raw_id = ?
                """,
                (raw_id,),
            ).fetchone()
            membership_census = conn.execute(
                """
                SELECT status, detail FROM raw_membership_census
                WHERE raw_id = ? AND parser_fingerprint = 'revision-membership-v1'
                """,
                (raw_id,),
            ).fetchone()
            membership_keys = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT logical_source_key FROM raw_session_memberships
                    WHERE raw_id = ? ORDER BY logical_source_key
                    """,
                    (raw_id,),
                )
            ]
            typed_key = (
                str(raw[0])
                if raw is not None and raw[0] is not None and str(raw[1]) != RawRevisionKind.UNKNOWN.value
                else None
            )
            complete = typed_key is not None or (
                membership_census is not None and str(membership_census[0]) in {"complete", "non_session"}
            )
            logical_keys = sorted(set(membership_keys) | ({typed_key} if typed_key is not None else set()))
            detail = (
                "current parser established durable authority identity"
                if complete
                else (
                    str(membership_census[1])
                    if membership_census is not None
                    else "current parser produced no durable authority identity"
                )
            )
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json,
                    detail, censused_at_ms
                ) VALUES (?, 'revision-membership-v1', ?, ?, ?, 0)
                ON CONFLICT(raw_id) DO UPDATE SET
                    parser_fingerprint = excluded.parser_fingerprint,
                    status = excluded.status,
                    logical_keys_json = excluded.logical_keys_json,
                    detail = excluded.detail,
                    censused_at_ms = excluded.censused_at_ms
                """,
                (raw_id, "complete" if complete else "failed", json.dumps(logical_keys), detail),
            )


def _census_historical_revision_evidence(
    archive: ArchiveStore,
    spill: _ParsedSessionSpill,
    *,
    selected_raw_ids: list[str] | None,
    max_payload_bytes: int | None,
    ingest_workers: int = 1,
    commit_batch_size: int | None = None,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> _RevisionCensusState:
    """Persist a complete bounded parser census without mutating index.db.

    ``prefetch_cache`` (polylogue-m6tp phase (a)), when supplied, is threaded
    to ``_parse_retained_raws`` so a raw already parsed off the writer hold
    is applied directly instead of reparsed here. ``None`` (every existing
    caller) reproduces the exact unmodified parse path.

    ``commit_batch_size`` (polylogue-amg1): when set to a positive integer,
    ``replace_raw_membership_census``/``bind_raw_revision`` writes for up to
    that many raws share one source.db commit instead of one commit per raw
    (``sqlite3.Connection.__exit__`` -- fsync -- measured at 42.6% of wall
    time on an independent-raw corpus). This only defers WHEN bytes already
    proven durable by ``write_raw_payload`` become visible as census rows; a
    crash mid-batch loses at most one batch's progress, which the caller
    re-derives identically on retry (census is idempotent and re-run from
    durable raw bytes) -- it never leaves a raw half-written or duplicates
    an outcome. Every batch is committed (or the whole batch is discarded on
    an exception, via the caller's ``rollback()``) before this function
    returns or propagates; a crash further downstream (replay phase) cannot
    observe a partially-committed census. Default ``None`` preserves the
    original per-raw-commit behavior for every existing caller.

    Untyped raws sharing one ``source_path`` are first checked for a proven
    byte-growth chain (``ArchiveStore.classify_untyped_full_revision_groups``,
    polylogue-nh44) before any of them is parsed. Only the newest member of a
    proven chain is actually parsed; every older member is bound to the same
    learned identity without independently parsing bytes that byte comparison
    already proved are a strict prefix of the newest capture. This is a
    census-time parse-cost optimization only: it never grants authority --
    ``classify_raw_revision_cohort`` (called later, during replay) still
    independently re-derives byte-provenness from raw bytes for every raw.
    """
    state = _RevisionCensusState(0, 0, 0, set(), {}, {})
    batch_size = commit_batch_size if commit_batch_size is not None and commit_batch_size > 0 else None
    batched = batch_size is not None
    pending_commits = 0

    def commit_unit() -> None:
        nonlocal pending_commits
        pending_commits += 1
        if batch_size is not None and pending_commits >= batch_size:
            archive.commit()
            pending_commits = 0

    def apply_outcome(
        raw_id: str,
        source_index: int,
        outcomes: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception],
    ) -> None:
        state.scanned += 1
        state.censused.add(raw_id)
        if source_index < 0:
            archive.replace_raw_membership_census(
                raw_id,
                None,
                parser_fingerprint="revision-membership-v1",
                censused_at_ms=0,
                detail=BYTE_AUTHORITY_CENSUS_DETAIL,
                manage_transaction=not batched,
            )
            state.quarantined += 1
            commit_unit()
            return
        outcome = outcomes[raw_id]
        if isinstance(outcome, Exception):
            archive.replace_raw_membership_census(
                raw_id,
                None,
                parser_fingerprint="revision-membership-v1",
                censused_at_ms=0,
                detail=str(outcome),
                manage_transaction=not batched,
            )
            state.quarantined += 1
            commit_unit()
            return
        sessions, payload_bytes, revision_kind = outcome
        state.classified += int(len(sessions) == 1)
        spill.add(raw_id, sessions, payload_bytes=payload_bytes)
        if len(sessions) == 1 and revision_kind is RawRevisionKind.UNKNOWN:
            session = sessions[0]
            logical_key = f"{session.source_name.value}:{session.provider_session_id}"
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    logical_source_key=logical_key,
                    kind=RawRevisionKind.FULL,
                    source_revision=raw_id,
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
                manage_transaction=not batched,
            )
            state.provisional_full_raw_ids.setdefault(logical_key, set()).add(raw_id)
            commit_unit()
        elif revision_kind is RawRevisionKind.UNKNOWN:
            archive.replace_raw_membership_census(
                raw_id,
                sessions,
                parser_fingerprint="revision-membership-v1",
                censused_at_ms=0,
                manage_transaction=not batched,
            )
            for session in sessions:
                logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
            commit_unit()
        # else: raw_id was already typed by an earlier run; this parse was a
        # wasted (but harmless, no-op) reparse -- existing retry behavior.

    def bind_byte_proven_older_member(raw_id: str, logical_key: str) -> None:
        """Bind an older chain member to the head's learned key without parsing it.

        Its own bytes were never independently opened here; identity is
        established by construction (byte-prefix of the parsed head), not by
        inspecting this raw's content.
        """
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key=logical_key,
                kind=RawRevisionKind.FULL,
                source_revision=raw_id,
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
            manage_transaction=not batched,
        )
        state.scanned += 1
        state.censused.add(raw_id)
        state.classified += 1
        state.provisional_full_raw_ids.setdefault(logical_key, set()).add(raw_id)
        commit_unit()

    census_selections: tuple[tuple[str, ...] | None, ...]
    if selected_raw_ids is None:
        census_selections = (None,)
    else:
        census_selections = archive.raw_membership_selection_components(selected_raw_ids)
    try:
        for initial_selection in census_selections:
            census_selection = initial_selection
            while True:
                rows = archive.raw_membership_census_rows(census_selection)
                pending_rows = [(raw_id, source_index) for raw_id, source_index in rows if raw_id not in state.censused]
                if max_payload_bytes is not None:
                    payload_sizes = archive.raw_payload_sizes([raw_id for raw_id, _index in rows])
                    total_payload_bytes = sum(payload_sizes.values())
                    oversized = [raw_id for raw_id, size in payload_sizes.items() if size > max_payload_bytes]
                    if oversized or total_payload_bytes > max_payload_bytes:
                        blocked_ids = oversized or list(payload_sizes)
                        raise RawRevisionReplayResourceBlockedError(
                            sorted(blocked_ids), max_payload_bytes, total_payload_bytes
                        )
                # Parse is read-only blob->ParsedSession decode and authority-neutral;
                # spread it across a process pool when there is more than one raw to
                # parse. Archive writes below stay in fixed `pending_rows` order
                # regardless of worker completion order, so parallel and sequential
                # runs remain byte-identical.
                parseable_raw_ids = [raw_id for raw_id, source_index in pending_rows if source_index >= 0]
                chain_older_by_head = archive.classify_untyped_full_revision_groups(parseable_raw_ids)
                head_by_older = {
                    older_raw_id: head_raw_id
                    for head_raw_id, older_raw_ids in chain_older_by_head.items()
                    for older_raw_id in older_raw_ids
                }
                dispatch_raw_ids = [raw_id for raw_id in parseable_raw_ids if raw_id not in head_by_older]
                parsed_outcomes = _parse_retained_raws(
                    archive, dispatch_raw_ids, ingest_workers=ingest_workers, prefetch_cache=prefetch_cache
                )
                for raw_id, source_index in pending_rows:
                    if raw_id in head_by_older:
                        continue
                    apply_outcome(raw_id, source_index, parsed_outcomes)
                if head_by_older:
                    head_to_key = {
                        raw_id: key for key, raw_ids in state.provisional_full_raw_ids.items() for raw_id in raw_ids
                    }
                    unresolved = [
                        older_raw_id
                        for older_raw_id, head_raw_id in head_by_older.items()
                        if head_raw_id not in head_to_key
                    ]
                    # The head's parse did not yield a clean single-session bind
                    # (e.g. a coincidental byte-prefix among multi-session
                    # bundles) -- fall back to parsing every deferred member
                    # individually, exactly as if no chain had been proven.
                    fallback_outcomes = (
                        _parse_retained_raws(
                            archive, unresolved, ingest_workers=ingest_workers, prefetch_cache=prefetch_cache
                        )
                        if unresolved
                        else {}
                    )
                    for older_raw_id, head_raw_id in head_by_older.items():
                        resolved_key = head_to_key.get(head_raw_id)
                        if resolved_key is not None:
                            bind_byte_proven_older_member(older_raw_id, resolved_key)
                        else:
                            apply_outcome(older_raw_id, 0, fallback_outcomes)
                if census_selection is None:
                    break
                expanded, _keys = archive.expand_raw_membership_selection(list(census_selection))
                if set(expanded) == set(census_selection):
                    break
                census_selection = expanded
    except BaseException:
        if batched and pending_commits > 0:
            archive.rollback()
        raise
    if batched and pending_commits > 0:
        archive.commit()
    return state


def census_historical_revision_evidence(
    archive_root: Path,
    *,
    selected_raw_ids: list[str] | None = None,
    max_payload_bytes: int | None = None,
    ingest_workers: int = 1,
    commit_batch_size: int | None = None,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> RevisionCensusResult:
    """Complete the source-tier census stage without applying index changes.

    ``prefetch_cache`` (polylogue-m6tp phase (a), default ``None``) lets a
    caller (the daemon conveyor) substitute already-parsed output computed
    off the writer hold for any raw it warmed ahead of time. See
    ``RawParsePrefetchCache`` for the equivalence guarantee.
    """
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        _ParsedSessionSpill(archive_root, max_cached_payload_bytes=max_payload_bytes) as spill,
    ):
        state = _census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=selected_raw_ids,
            max_payload_bytes=max_payload_bytes,
            ingest_workers=ingest_workers,
            commit_batch_size=commit_batch_size,
            prefetch_cache=prefetch_cache,
        )
        expanded, logical_keys = archive.expand_raw_membership_selection(selected_raw_ids)
    _record_raw_authority_parser_census(archive_root, tuple(expanded))
    return RevisionCensusResult(
        state.scanned,
        state.classified,
        state.quarantined,
        expanded,
        logical_keys,
    )


def backfill_historical_revision_evidence(
    archive_root: Path,
    *,
    selected_raw_ids: list[str] | None = None,
    owned_inactive_generation: tuple[str, str] | None = None,
    retention_observer: Callable[[int, int], None] | None = None,
    max_payload_bytes: int | None = None,
    max_cached_payload_bytes: int | None = None,
    ingest_workers: int = 1,
    commit_batch_size: int | None = None,
    replay_commit_batch_size: int | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> RevisionBackfillResult:
    """Census every retained raw, then replay byte and bundle authority cohorts.

    Parser output is spilled beside the target archive during the census and
    loaded one logical authority cohort at a time. Peak retained session trees
    therefore follow the largest raw/cohort, not the archive-wide raw count.

    ``max_cached_payload_bytes`` bounds the spill cache independently of
    ``max_payload_bytes``: the latter is a component resource-envelope block
    (``None`` means unbounded, e.g. a one-shot full-archive rebuild) while the
    former only avoids doubling I/O for census-then-replay reparse. It
    defaults to ``max_payload_bytes`` so every existing bounded-envelope
    caller keeps its current exactly-sized cache; pass it explicitly to cache
    parse output for an unbounded (``max_payload_bytes=None``) census without
    also activating envelope blocking.

    ``commit_batch_size`` (polylogue-amg1, extended to the replay phase by
    polylogue-oikv) batches commits in BOTH phases: the CENSUS phase's
    per-raw source.db commits (see ``_census_historical_revision_evidence``),
    and the REPLAY phase's per-cohort index.db writes plus terminal source.db
    parse-state markers (see ``apply_raw_revision_replay`` /
    ``apply_raw_membership_classification``), across up to
    ``commit_batch_size`` cohorts sharing one commit window.

    ``None`` (the default) preserves the exact original per-unit commit
    granularity for every existing caller in both phases. When set, the
    "index commits, then source terminal marker commits" ordering invariant
    pinned by
    ``test_backfill_resumes_after_index_receipt_commits_before_source_terminal``
    et al. still holds for the replay phase -- just at BATCH granularity
    instead of per-cohort: a crash inside an open batch discards every
    cohort in that batch (its index writes and terminal markers together,
    since neither has committed), never leaves index ahead of source across
    a batch boundary, and a resume reprocesses every lost cohort from
    scratch with zero duplication (proof:
    ``test_backfill_resumes_after_replay_batch_crash_discards_whole_batch_cleanly``).

    ``bulk_fts`` (polylogue-crd8, default ``False``) is threaded to both
    ``apply_raw_revision_replay`` and ``apply_raw_membership_classification``
    to enable the guard-gated bulk FTS mode for whale prefix-sharing lineage
    cascades. Offline rebuild callers (``maintenance/rebuild_index.py`` via
    ``maintenance/replay.py``) pass ``True``; other callers leave it off.

    ``bulk_build`` (polylogue-v6i3, default ``False``) mirrors ``bulk_fts``'s
    threading to the same two apply calls, enabling the broader
    bulk-generation-build lifecycle: every per-session
    messages_fts/blocks_command_trigram/action_pairs/delegation_facts refresh
    is skipped during replay, deferred to one archive-wide repopulate at
    readiness. Only the offline rebuild caller passes ``True``.

    ``prefetch_cache`` (polylogue-gd6v, default ``None``) is threaded to the
    census phase exactly like ``census_historical_revision_evidence``'s own
    parameter: a raw already parsed off the writer hold (the daemon's
    ``DaemonParseStage``, warmed ahead of a bounded bulk-rebuild pass) is
    consumed directly instead of reparsed. A prefetch hit still flows through
    ``apply_outcome``'s ``spill.add(...)`` exactly like a freshly-parsed
    outcome, so the REPLAY phase's own ``spill.for_raw`` lookups (which do
    all of the actual cohort writes) see identical warmed content -- this is
    what makes prefetching the census phase alone enough to also skip
    replay-phase reparsing for the same raws. ``None`` (every existing
    caller) reproduces the exact unmodified parse path.
    """
    adoption_deferred = 0
    quarantined = 0
    stage_timings: dict[str, float] = {}
    logical_keys: set[str] = set()
    # The REPLAY phase's batch size is separately tunable
    # (``replay_commit_batch_size``; ``None`` inherits ``commit_batch_size``):
    # each replayed cohort may flush blob-publication receipts on a SEPARATE
    # source.db connection (a deliberate GC-safety design, see
    # storage/blob_publication.py), and that connection waits at BEGIN
    # IMMEDIATE behind the batch's held write lock -- a long replay batch
    # window therefore deadlocks into 'database is locked' once the 30s busy
    # timeout expires. Callers that batch aggressively (the full rebuild)
    # pass ``replay_commit_batch_size=1`` to keep replay at per-cohort
    # commits while still batching the census phase, which has no separate-
    # connection writers inside its window.
    effective_replay_batch = replay_commit_batch_size if replay_commit_batch_size is not None else commit_batch_size
    replay_batch_size = (
        effective_replay_batch if effective_replay_batch is not None and effective_replay_batch > 0 else None
    )
    replay_batched = replay_batch_size is not None and replay_batch_size > 1
    archive_context = (
        ArchiveStore.open_owned_inactive_generation(
            archive_root,
            generation_id=owned_inactive_generation[0],
            owner_id=owned_inactive_generation[1],
        )
        if owned_inactive_generation is not None
        else ArchiveStore.open_existing(archive_root, read_only=False)
    )
    spill_cache_bytes = max_cached_payload_bytes if max_cached_payload_bytes is not None else max_payload_bytes
    with (
        archive_context as archive,
        _ParsedSessionSpill(archive_root, max_cached_payload_bytes=spill_cache_bytes) as spill,
    ):
        census_started = time.perf_counter()
        census = _census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=selected_raw_ids,
            max_payload_bytes=max_payload_bytes,
            ingest_workers=ingest_workers,
            commit_batch_size=commit_batch_size,
            prefetch_cache=prefetch_cache,
        )
        stage_timings["census"] = time.perf_counter() - census_started
        receipt_started = time.perf_counter()
        censused_raw_ids, _censused_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        # The direct backfill entry point must publish the same current-parser
        # receipt as the census-only entry point before it assigns or applies
        # any index plan. Commit the source census first so the separate
        # durable receipt writer observes one complete source snapshot.
        archive.commit()
        _record_raw_authority_parser_census(archive_root, tuple(censused_raw_ids))
        stage_timings["census_receipt"] = time.perf_counter() - receipt_started
        membership_candidates = census.membership_candidates
        provisional_full_raw_ids = census.provisional_full_raw_ids

        _unclassified, selected_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        logical_keys.update(selected_keys)
        _selected_membership_raws, selected_membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        membership_keys = set(selected_membership_keys)

        pending_replay_commits = 0

        def commit_replay_unit() -> None:
            nonlocal pending_replay_commits
            pending_replay_commits += 1
            if replay_batch_size is not None and pending_replay_commits >= replay_batch_size:
                archive.commit()
                pending_replay_commits = 0

        replayed = 0
        byte_replayed_keys: set[str] = set()
        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort(logical_key)
            if not plan.accepted_raw_ids:
                # Complete snapshots that are not a unique byte-prefix chain
                # still carry semantic evidence. Move only that full-only
                # cohort to membership governance and let parsed-content
                # prefix rules decide it; append chains remain byte-governed.
                for raw_id in archive.convertible_full_revision_raw_ids(logical_key):
                    spill_started = time.perf_counter()
                    sessions, _payload_bytes = spill.for_raw(archive, raw_id)
                    stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                        time.perf_counter() - spill_started
                    )
                    if len(sessions) != 1:
                        raise RuntimeError(f"full revision {raw_id} no longer parses to one session")
                    archive.replace_raw_membership_census(
                        raw_id,
                        sessions,
                        parser_fingerprint="revision-membership-v1",
                        censused_at_ms=0,
                        detail="historical non-prefix full revision governance",
                        retire_full_revision_governance=True,
                    )
                    membership_candidates.setdefault(logical_key, set()).add(raw_id)
                membership_keys.add(logical_key)
                continue
            parsed_by_raw_id: dict[str, ParsedSession] = {}
            retained_bytes = 0
            for raw_id in plan.accepted_raw_ids:
                spill_started = time.perf_counter()
                sessions, payload_bytes = spill.for_raw(archive, raw_id)
                stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                    time.perf_counter() - spill_started
                )
                if len(sessions) != 1:
                    raise RuntimeError(f"classified raw revision {raw_id} no longer parses to one session")
                parsed_by_raw_id[raw_id] = sessions[0]
                retained_bytes += payload_bytes
            if retention_observer is not None:
                retention_observer(len(parsed_by_raw_id), retained_bytes)
            accepted_sessions = [parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids]
            if not archive.raw_revision_replay_adoptable(accepted_sessions):
                archive.defer_raw_revision_adoption(plan.logical_source_key, plan.accepted_raw_ids, accepted_sessions)
                provisional_raw_ids = provisional_full_raw_ids.get(logical_key, set())
                plan_raw_ids = {application.raw_id for application in plan.applications}
                if plan_raw_ids and plan_raw_ids <= provisional_raw_ids:
                    archive.release_provisional_full_revisions(sorted(plan_raw_ids))
                adoption_deferred += len(plan.accepted_raw_ids)
                continue
            try:
                archive.apply_raw_revision_replay(
                    plan,
                    parsed_by_raw_id,
                    acquired_at_ms=0,
                    stage_timings_s=stage_timings,
                    manage_transaction=not replay_batched,
                    bulk_fts=bulk_fts,
                    bulk_build=bulk_build,
                )
            except sqlite3.IntegrityError as exc:
                raise sqlite3.IntegrityError(
                    f"backfill_historical_revision_evidence: byte-proven replay failed for "
                    f"logical_key={plan.logical_source_key!r}: {exc}"
                ) from exc
            replayed += 1
            byte_replayed_keys.add(logical_key)
            if replay_batched:
                commit_replay_unit()

        for logical_key in sorted(membership_keys):
            if logical_key in byte_replayed_keys:
                continue
            member_sessions: dict[str, ParsedSession] = {}
            revisions: list[MembershipRevision] = []
            projections = {}
            retained_bytes = 0
            candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
            candidate_raw_ids.update(membership_candidates.get(logical_key, ()))
            # Cohort absorption: candidate selection is page-dependent, so a
            # head written by an EARLIER page's membership cohort for this key
            # may not be in this page's candidate set -- membership replay
            # would then refuse to retire an "unrelated" quarantined head and
            # kill the walk. Absorb the current quarantined head raw into the
            # cohort so the real prefix classifier ranks it against the new
            # members instead of any scalar comparison. Chain-governed
            # (non-quarantined) heads are deliberately NOT absorbed --
            # apply_raw_membership_classification yields to those.
            head_raw_id = archive.raw_revision_head_raw_id(logical_key)
            if head_raw_id is not None and archive._raw_revision_authority(head_raw_id) == "quarantined":
                candidate_raw_ids.add(head_raw_id)
            for raw_id in sorted(candidate_raw_ids):
                spill_started = time.perf_counter()
                sessions, payload_bytes = spill.for_raw(archive, raw_id)
                stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                    time.perf_counter() - spill_started
                )
                for session in sessions:
                    session_logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                    if session_logical_key != logical_key:
                        continue
                    projection = session_revision_projection(session)
                    member_sessions[raw_id] = session
                    projections[raw_id] = projection
                    revisions.append(
                        MembershipRevision(
                            raw_id,
                            projection,
                            session.updated_at,
                            browser_snapshot_fidelity=_browser_snapshot_fidelity(session.ingest_flags),
                            provider_message_ids=frozenset(message.provider_message_id for message in session.messages),
                            provider_attachment_ids=frozenset(
                                attachment.provider_attachment_id for attachment in session.attachments
                            ),
                        )
                    )
                    retained_bytes += payload_bytes
            if retention_observer is not None:
                retention_observer(len(member_sessions), retained_bytes)
            classification = classify_membership_revisions(revisions)
            if classification.ambiguous_raw_ids:
                quarantined += len(classification.ambiguous_raw_ids)
            accepted_sessions = [member_sessions[raw_id] for raw_id in classification.accepted_raw_ids]
            if accepted_sessions and not archive.raw_revision_replay_adoptable(accepted_sessions):
                archive.defer_raw_revision_adoption(
                    logical_key,
                    classification.accepted_raw_ids,
                    accepted_sessions,
                )
                adoption_deferred += len(classification.accepted_raw_ids)
                continue
            try:
                archive.apply_raw_membership_classification(
                    logical_key,
                    classification,
                    member_sessions,
                    projections,
                    acquired_at_ms=0,
                    stage_timings_s=stage_timings,
                    manage_transaction=not replay_batched,
                    bulk_fts=bulk_fts,
                    bulk_build=bulk_build,
                )
            except sqlite3.IntegrityError as exc:
                raise sqlite3.IntegrityError(
                    f"backfill_historical_revision_evidence: membership replay failed for "
                    f"logical_key={logical_key!r}: {exc}"
                ) from exc
            if classification.accepted_raw_ids:
                replayed += 1
            if replay_batched:
                commit_replay_unit()
        if replay_batched:
            archive.commit()
        if stage_timings:
            stage_timings["total"] = time.perf_counter() - census_started
            _LOGGER.info(
                "backfill stage timings: %s",
                " ".join(f"{key}={value:.1f}s" for key, value in sorted(stage_timings.items(), key=lambda kv: -kv[1])),
            )
    return RevisionBackfillResult(
        census.scanned,
        census.classified,
        replayed,
        census.quarantined + quarantined,
        adoption_deferred,
    )


def _parse_retained_raw(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
    provider, _blob_hash, source_path, kind, payload_size = archive.raw_revision_descriptor(raw_id)
    return parse_retained_raw_sessions(archive, raw_id), payload_size, kind


def census_parse_worker(
    raw_id: str,
    provider_token: str,
    blob_hash: str,
    source_path: str,
    is_stream: bool,
    blob_root_str: str,
    source_db_path_str: str,
) -> tuple[str, list[ParsedSession] | None, str | None]:
    """Parse one retained raw's already-published blob bytes.

    Pure read-only blob->ParsedSession decode; the caller already knows this
    raw's payload size and revision kind from its own source-tier descriptor
    lookup, so only primitive strings cross into this function -- no shared
    ``ArchiveStore`` (and thus no thread-affine sqlite connection, see
    ``_parse_unique_retained_raws_via_threads``) and no pickled object graph
    to construct one. Errors are returned rather than raised so the caller
    can apply the exact same per-raw quarantine handling as the sequential
    path.

    Dispatched onto a ``ProcessPoolExecutor`` (GIL-build fallback, see
    ``_parse_unique_retained_raws``), a ``ThreadPoolExecutor`` (real
    free-threading, see ``_parse_unique_retained_raws_via_threads``), and the
    daemon's own off-writer-hold pre-parse ``ThreadPoolExecutor``
    (``polylogue.daemon.parse_prefetch.DaemonParseStage``, polylogue-m6tp
    phase (a)) -- the function is identical every time; only the executor
    and the recreated ``ArchiveBlobPublisher``'s process/thread affinity
    differ. Public (not module-private) precisely so the daemon's warmer can
    import and dispatch it without duplicating this parse logic.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    provider = Provider(provider_token)
    publisher = ArchiveBlobPublisher(Path(source_db_path_str), Path(blob_root_str))
    try:
        if is_stream:
            with publisher.open(blob_hash) as payload:
                sessions = _parse_stream(provider, payload, source_path)
        else:
            payload_path = None
            if provider is Provider.HERMES:
                candidate_path = publisher.blob_path(blob_hash)
                payload_path = candidate_path if candidate_path.exists() else None
            sessions = _parse_one(
                provider,
                publisher.read_all(blob_hash),
                source_path,
                payload_path=payload_path,
                archive_root=Path(blob_root_str).parent,
            )
        return raw_id, sessions, None
    except Exception as exc:
        return raw_id, None, str(exc)


_DEFAULT_PARSE_DISPATCH_MAX_BYTES = 262_144  # 256 KiB


def _parse_dispatch_max_bytes() -> int:
    """Payload-size ceiling for pool dispatch to still be a net win (polylogue-amg1).

    polylogue-amg1's own measurement: 200 raws at ~50KB average with 8
    workers measured 1.22x (net win); 80 raws at ~1.7MB average measured
    0.63x (net LOSS, slower than sequential) on the same machine. The
    process-pool round trip pickles the returned ``ParsedSession`` list back
    across the process boundary -- for large payloads that pickle cost
    exceeds the parse time saved by running concurrently. Raws at or above
    this size parse sequentially in-process (no IPC); raws below it dispatch
    to the pool, where aggregate parse time genuinely dominates transfer
    cost. Override with POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES.
    """
    raw = os.environ.get("POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES")
    if raw is None:
        return _DEFAULT_PARSE_DISPATCH_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_PARSE_DISPATCH_MAX_BYTES


def _partition_raws_by_dispatch_size(
    raw_ids: list[str],
    payload_sizes: dict[str, int],
    *,
    dispatch_max_bytes: int,
) -> tuple[list[str], list[str]]:
    """Split raw ids into (pool-eligible, sequential) by payload size.

    Preserves ``raw_ids`` input order within each bucket so callers stay
    deterministic. See ``_parse_dispatch_max_bytes`` for why size, not count,
    decides pool eligibility (polylogue-amg1).
    """
    pool_raw_ids = [raw_id for raw_id in raw_ids if payload_sizes[raw_id] < dispatch_max_bytes]
    sequential_raw_ids = [raw_id for raw_id in raw_ids if payload_sizes[raw_id] >= dispatch_max_bytes]
    return pool_raw_ids, sequential_raw_ids


_DEFAULT_PARSE_POOL_MIN_AGGREGATE_BYTES = 48 * 1024 * 1024  # 48 MiB


def _parse_pool_min_aggregate_bytes() -> int:
    """Aggregate-payload floor below which pool dispatch cannot amortize.

    Each spawned worker pays the full interpreter + polylogue import before
    its first task (~1.5-2.0s measured live 2026-07-19: a 25s py-spy capture
    of the bulk rebuild census showed 20 short-lived workers spending ~95%
    of their lifetime inside importlib, because per-cohort census batches
    dispatch 1-2 sub-256KiB raws at a time and the executor is created per
    call). Sequential small-payload parse runs at roughly 20MB/s, so with 8
    workers the pool only beats sequential once the pool-eligible aggregate
    exceeds ~45MB; below that, worker spawn dominates and "parallel" is
    strictly slower. Override with
    ``POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES`` (polylogue-crd8 follow-up).
    """
    raw = os.environ.get("POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES")
    if raw is None:
        return _DEFAULT_PARSE_POOL_MIN_AGGREGATE_BYTES
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_PARSE_POOL_MIN_AGGREGATE_BYTES


def _pool_dispatch_amortizes(pool_raw_ids: list[str], payload_sizes: dict[str, int]) -> bool:
    """Decide whether a pool-eligible batch is worth spawning workers for."""
    if len(pool_raw_ids) <= 1:
        return False
    total = sum(payload_sizes[raw_id] for raw_id in pool_raw_ids)
    return total >= _parse_pool_min_aggregate_bytes()


def _parse_retained_raws(
    archive: ArchiveStore,
    raw_ids: list[str],
    *,
    ingest_workers: int,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
    """Parse a batch of retained raws, deduplicating byte-identical inputs.

    Returns each outcome keyed by raw_id: either the parsed
    ``(sessions, payload_bytes, revision_kind)`` tuple or the caught
    exception. Rows sharing the same ``(blob_hash, source_path)`` are parsed
    exactly once and the outcome fanned out: identical bytes at an identical
    path decode deterministically identically, so re-parsing them is pure
    waste (measured live 2026-07-19: 17% of newest-only bytes — e.g. one
    442MB codex rollout stored under 8 raw rows — were byte-identical
    duplicates each paying a full parse). ``source_path`` stays in the key
    because some parsers derive identity from the path (e.g. beads workspace
    ids), so cross-path duplicates are deliberately NOT deduplicated.
    Per-row ``revision_kind`` is re-attached from each row's own descriptor.

    ``prefetch_cache`` (polylogue-m6tp phase (a)) is consulted BEFORE any of
    the above: a raw_id already popped from the cache is used directly and
    excluded from dedup/dispatch entirely, so it costs neither a parse nor a
    process/thread-pool round trip here. Every raw_id NOT found in the cache
    (including all of them, when ``prefetch_cache`` is ``None`` -- the
    default for every existing caller) is parsed exactly as before.
    """
    descriptors = {raw_id: archive.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
    results: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception] = {}
    remaining_raw_ids = raw_ids
    if prefetch_cache is not None and raw_ids:
        remaining_raw_ids = []
        for raw_id in raw_ids:
            cached = prefetch_cache.pop(raw_id)
            if cached is None:
                remaining_raw_ids.append(raw_id)
            else:
                results[raw_id] = cached
    grouped: dict[tuple[str, str], list[str]] = {}
    for raw_id in remaining_raw_ids:
        _provider, blob_hash, source_path, _kind, _size = descriptors[raw_id]
        grouped.setdefault((blob_hash, source_path), []).append(raw_id)
    representatives = [members[0] for members in grouped.values()]
    unique = _parse_unique_retained_raws(
        archive, representatives, descriptors=descriptors, ingest_workers=ingest_workers
    )
    for members in grouped.values():
        outcome = unique[members[0]]
        for raw_id in members:
            if isinstance(outcome, Exception):
                results[raw_id] = outcome
            else:
                sessions, _rep_size, _rep_kind = outcome
                _provider, _blob_hash, _source_path, kind, size = descriptors[raw_id]
                results[raw_id] = (sessions, size, kind)
    return results


def _parse_unique_retained_raws_via_threads(
    archive: ArchiveStore,
    raw_ids: list[str],
    *,
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int]],
    ingest_workers: int,
) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
    """Thread-parallel parse, reachable only when ``parallel_threads_effective()``.

    Under a real free-threaded (no-GIL) interpreter, a plain
    ``ThreadPoolExecutor`` shares parsed ``ParsedSession`` object graphs by
    reference between threads, so neither of the process pool's two
    amortization costs applies: no pickle-back of the return value (#3136
    measured 0.63x/net-loss above 256KiB) and no per-worker spawn+import tax
    (#3149's ~1.5-2s floor -- threads share the one already-imported
    interpreter, they never re-pay ``import polylogue``). Both
    ``_partition_raws_by_dispatch_size`` and ``_pool_dispatch_amortizes``
    exist solely to protect against those two process-pool-specific costs
    (#3136/#3149), so this path applies NEITHER: every raw in ``raw_ids``
    dispatches to the thread pool regardless of payload size or aggregate
    bytes.

    Dispatches the same ``census_parse_worker`` function the process-pool
    path uses, deliberately -- NOT ``_parse_retained_raw(archive, raw_id)``
    directly. ``ArchiveStore`` lazily opens ``_source_conn`` as a plain
    ``sqlite3.Connection`` with the default ``check_same_thread=True``
    (``storage/sqlite/archive_tiers/archive.py:_ensure_source_conn``); the
    caller of this function (``_parse_unique_retained_raws``) already
    resolved every raw's descriptor sequentially on the calling thread
    before dispatch, so calling ``archive.raw_revision_descriptor`` again
    from a worker thread (as ``_parse_retained_raw`` does) raises
    ``sqlite3.ProgrammingError: SQLite objects created in a thread can only
    be used in that same thread`` -- confirmed empirically, not
    theoretical. ``census_parse_worker`` sidesteps this entirely: it never
    touches the shared ``ArchiveStore`` or its connections, only a fresh,
    stateless ``ArchiveBlobPublisher`` built from primitive strings
    (blob root + source.db path), whose blob reads are plain filesystem I/O.

    Result assembly is keyed by raw_id exactly like the process-pool path,
    so completion order never affects the archive state callers build from
    these results.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception] = {}
    blob_root_str = str(archive.archive_root / "blob")
    source_db_path_str = str(archive.source_db_path)
    with ThreadPoolExecutor(max_workers=min(ingest_workers, len(raw_ids))) as pool:
        future_to_raw_id = {}
        for raw_id in raw_ids:
            provider, blob_hash, source_path, _kind, _payload_size = descriptors[raw_id]
            future = pool.submit(
                census_parse_worker,
                raw_id,
                provider.value,
                blob_hash,
                source_path,
                is_stream_record_provider(source_path, str(provider)),
                blob_root_str,
                source_db_path_str,
            )
            future_to_raw_id[future] = raw_id
        for future in as_completed(future_to_raw_id):
            raw_id = future_to_raw_id[future]
            try:
                _raw_id, sessions, error = future.result()
            except Exception as exc:
                results[raw_id] = exc
                continue
            if error is not None:
                results[raw_id] = RuntimeError(error)
                continue
            _provider, _blob_hash, _source_path, kind, payload_size = descriptors[raw_id]
            results[raw_id] = (sessions or [], payload_size, kind)
    return results


def _parse_unique_retained_raws(
    archive: ArchiveStore,
    raw_ids: list[str],
    *,
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int]],
    ingest_workers: int,
) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
    """Parse already-deduplicated raws, optionally in parallel.

    Read-only blob->ParsedSession decode is authority-neutral and
    embarrassingly parallel; callers apply archive writes afterwards in a
    fixed deterministic order independent of completion order here, so
    parallel and sequential execution produce byte-identical archive state
    regardless of which raws take a parallel path versus the sequential
    path.

    Two parallel strategies, mutually exclusive per call:
    ``parallel_threads_effective()`` (real free-threading, e.g. 3.14t) routes
    to ``_parse_unique_retained_raws_via_threads`` with no size partition or
    amortization floor. Otherwise (the standard GIL build -- today's default
    and the only build the daemon deploys) falls through to the
    ``ProcessPoolExecutor`` path below, unchanged: the polylogue-7mtf
    control-run measurement proved GIL-build threads give zero parse
    speedup (0.93x-0.96x) and inflate a concurrent writer thread's commit
    latency ~5000x, so threads must never engage without a genuinely
    disabled GIL.
    """
    results: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception] = {}
    if ingest_workers <= 1 or len(raw_ids) <= 1:
        for raw_id in raw_ids:
            try:
                results[raw_id] = _parse_retained_raw(archive, raw_id)
            except Exception as exc:
                results[raw_id] = exc
        return results

    if parallel_threads_effective():
        return _parse_unique_retained_raws_via_threads(
            archive, raw_ids, descriptors=descriptors, ingest_workers=ingest_workers
        )

    payload_sizes = {raw_id: descriptors[raw_id][4] for raw_id in raw_ids}
    pool_raw_ids, sequential_raw_ids = _partition_raws_by_dispatch_size(
        raw_ids, payload_sizes, dispatch_max_bytes=_parse_dispatch_max_bytes()
    )

    for raw_id in sequential_raw_ids:
        try:
            results[raw_id] = _parse_retained_raw(archive, raw_id)
        except Exception as exc:
            results[raw_id] = exc

    if not _pool_dispatch_amortizes(pool_raw_ids, payload_sizes):
        for raw_id in pool_raw_ids:
            try:
                results[raw_id] = _parse_retained_raw(archive, raw_id)
            except Exception as exc:
                results[raw_id] = exc
        return results

    from concurrent.futures import as_completed

    from polylogue.pipeline.services.process_pool import process_pool_executor

    blob_root_str = str(archive.archive_root / "blob")
    source_db_path_str = str(archive.source_db_path)
    with process_pool_executor(max_workers=min(ingest_workers, len(pool_raw_ids))) as pool:
        future_to_raw_id = {}
        for raw_id in pool_raw_ids:
            provider, blob_hash, source_path, _kind, _payload_size = descriptors[raw_id]
            future = pool.submit(
                census_parse_worker,
                raw_id,
                provider.value,
                blob_hash,
                source_path,
                is_stream_record_provider(source_path, str(provider)),
                blob_root_str,
                source_db_path_str,
            )
            future_to_raw_id[future] = raw_id
        for future in as_completed(future_to_raw_id):
            raw_id = future_to_raw_id[future]
            try:
                _raw_id, sessions, error = future.result()
            except Exception as exc:
                results[raw_id] = exc
                continue
            if error is not None:
                results[raw_id] = RuntimeError(error)
                continue
            _provider, _blob_hash, _source_path, kind, payload_size = descriptors[raw_id]
            results[raw_id] = (sessions or [], payload_size, kind)
    return results


def parse_retained_raw_sessions(archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
    """Parse retained raw evidence without eagerly loading stream records.

    Raw-revision replay is shared by historical repair and the live full and
    append routes.  Keeping the provider-shape decision here prevents a
    seemingly harmless live replay helper from reintroducing ``read_all()``
    for Codex/Claude JSONL evidence.
    """
    provider, blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    if is_stream_record_provider(source_path, str(provider)):
        with archive.open_raw_revision_material(raw_id) as (stream_provider, payload, stream_path, _stream_kind):
            return _parse_stream(stream_provider, payload, stream_path)
    _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
    payload_path = archive.blob_path_for_hash(blob_hash) if provider is Provider.HERMES else None
    return _parse_one(
        provider,
        eager_payload,
        source_path,
        payload_path=payload_path,
        archive_root=archive.archive_root,
    )


class _ParsedSessionSpill:
    """Bounded parsed-session cache; durable raw bytes remain the replay source.

    A census may span an archive-wide set of raw rows.  Its parser output must
    not become an archive-wide second materialization.  Entries that do not
    fit the caller's existing component envelope are deliberately not cached;
    replay reparses them from durable source evidence.  This trades bounded
    I/O for completeness and makes no raw cohort silently disappear.
    """

    #: Decoded-session RAM layer budget (payload-equivalent bytes). Replay
    #: consumes a cohort almost immediately after census parses it, so a
    #: small hot layer turns the common for_raw() into a dict hit instead of
    #: a pickle.loads round-trip. Bounded independently of the sqlite layer.
    _DECODED_CACHE_PAYLOAD_BYTES: Final[int] = 256 * 1024 * 1024

    def __init__(self, archive_root: Path, *, max_cached_payload_bytes: int | None) -> None:
        # Place the spill beside the RESOLVED index tier, not the archive
        # root: on deployments where the .db files are symlinks (e.g. root
        # SSD config dir -> NVMe data disk), a spill in archive_root would
        # put census churn on the wear-limited disk the symlinks exist to
        # protect.
        index_path = archive_root / "index.db"
        spill_dir = index_path.resolve().parent if index_path.exists() else archive_root
        fd, name = tempfile.mkstemp(prefix=".revision-census-", suffix=".sqlite", dir=spill_dir)
        os.close(fd)
        self.path = Path(name)
        self.conn = sqlite3.connect(self.path)
        # Disposable single-connection cache: durability is meaningless (the
        # fallback is reparsing durable source evidence), so skip the
        # journal and every fsync -- the per-add commit previously paid a
        # synchronous journal cycle per censused raw.
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            """
            CREATE TABLE parsed_sessions (
                raw_id TEXT NOT NULL,
                logical_key TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                parsed BLOB NOT NULL,
                PRIMARY KEY(raw_id, logical_key)
            ) STRICT
            """
        )
        self.conn.execute("CREATE INDEX parsed_sessions_logical ON parsed_sessions(logical_key, raw_id)")
        self.max_cached_payload_bytes = max_cached_payload_bytes
        self.cached_payload_bytes = 0
        self._decoded: dict[str, tuple[list[ParsedSession], int]] = {}
        self._decoded_payload_bytes = 0

    def __enter__(self) -> _ParsedSessionSpill:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.conn.close()
        self.path.unlink(missing_ok=True)

    def add(self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int) -> None:
        if self.max_cached_payload_bytes is None or payload_bytes > self.max_cached_payload_bytes:
            return
        if self.cached_payload_bytes + payload_bytes > self.max_cached_payload_bytes:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT INTO parsed_sessions(raw_id, logical_key, payload_bytes, parsed) VALUES (?, ?, ?, ?)",
                (
                    (
                        raw_id,
                        f"{session.source_name.value}:{session.provider_session_id}",
                        payload_bytes,
                        pickle.dumps(session, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    for session in sessions
                ),
            )
        self.cached_payload_bytes += payload_bytes
        self._retain_decoded(raw_id, sessions, payload_bytes=payload_bytes)

    def _retain_decoded(self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int) -> None:
        if payload_bytes > self._DECODED_CACHE_PAYLOAD_BYTES:
            return
        while self._decoded and self._decoded_payload_bytes + payload_bytes > self._DECODED_CACHE_PAYLOAD_BYTES:
            oldest_raw = next(iter(self._decoded))
            _evicted_sessions, evicted_bytes = self._decoded.pop(oldest_raw)
            self._decoded_payload_bytes -= evicted_bytes
        self._decoded[raw_id] = (sessions, payload_bytes)
        self._decoded_payload_bytes += payload_bytes

    def for_raw(self, archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int]:
        decoded = self._decoded.get(raw_id)
        if decoded is not None:
            return decoded
        rows = self.conn.execute(
            "SELECT parsed, payload_bytes FROM parsed_sessions WHERE raw_id = ? ORDER BY logical_key", (raw_id,)
        ).fetchall()
        if rows:
            return [pickle.loads(bytes(row[0])) for row in rows], int(rows[0][1])
        sessions, payload_bytes, _kind = _parse_retained_raw(archive, raw_id)
        self.add(raw_id, sessions, payload_bytes=payload_bytes)
        return sessions, payload_bytes


def _parse_one(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = Path(source_path).stem
    if is_stream_record_provider(source_path, str(provider)):
        return parse_stream_payload(
            provider,
            _iter_json_stream(BytesIO(payload), source_name),
            fallback_id,
            source_path=source_path,
        )
    if provider is Provider.HERMES and looks_like_sqlite_bytes(payload):
        with _sqlite_payload_path(payload, payload_path, archive_root) as sqlite_path:
            if hermes_state.looks_like_state_db_path(sqlite_path):
                return hermes_state.parse_state_db(
                    sqlite_path,
                    fallback_id=fallback_id,
                    profile_root=Path(source_path).parent,
                )
            if hermes_verification.looks_like_verification_evidence_db_path(sqlite_path):
                return hermes_verification.parse_verification_evidence_db(sqlite_path, fallback_id=fallback_id)
    return parse_payload(
        provider,
        list(_iter_json_stream(BytesIO(payload), source_name)),
        fallback_id,
        source_path=source_path,
    )


@contextmanager
def _sqlite_payload_path(
    payload: bytes,
    payload_path: Path | None,
    archive_root: Path | None,
) -> Iterator[Path]:
    """Yield a real filesystem path for SQLite-shaped raw revision bytes.

    ``sqlite3.connect`` cannot open in-memory bytes. Prefer the already-
    materialized blob path (no copy); only spill to a bounded temp file when
    no real path is available (e.g. the blob is not yet flushed to disk).
    """
    if payload_path is not None:
        yield payload_path
        return
    scratch_dir = archive_root if archive_root is not None else Path(tempfile.gettempdir())
    fd, name = tempfile.mkstemp(prefix=".revision-sqlite-spill-", suffix=".sqlite", dir=scratch_dir)
    os.close(fd)
    temp_path = Path(name)
    try:
        temp_path.write_bytes(payload)
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_stream(provider: Provider, payload: BinaryIO, source_path: str) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = Path(source_path).stem
    return parse_stream_payload(
        provider,
        _iter_json_stream(payload, source_name),
        fallback_id,
        source_path=source_path,
    )


__all__ = [
    "RawParsePrefetchCache",
    "RawRevisionReplayResourceBlockedError",
    "RevisionBackfillResult",
    "RevisionCensusResult",
    "backfill_historical_revision_evidence",
    "census_historical_revision_evidence",
    "census_parse_worker",
    "record_resource_blocked_revision_census",
    "uncensused_historical_revision_raw_ids",
    "parse_retained_raw_sessions",
]
