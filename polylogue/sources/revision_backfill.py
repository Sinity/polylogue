"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

import os
import pickle
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Literal, cast

from polylogue import logging as _polylogue_logging
from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind
from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    durable_authority_logical_keys,
)
from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.pipeline.ids import session_revision_projection
from polylogue.pipeline.parsed_tree_size import effective_physical_memory_bytes, estimate_parsed_tree_bytes
from polylogue.pipeline.services.process_pool import (
    PoolKind,
    parallel_threads_effective,
    resolve_revision_backfill_census_dispatch,
)
from polylogue.sources.codex_state_evidence import write_codex_thread_state_evidence
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    detect_provider_from_raw_bytes_evidence,
    is_jsonl_source_path,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers import antigravity, codex_state, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.sqlite_snapshot import looks_like_sqlite_bytes
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_authority import (
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    SUPERSEDED_MEMBERSHIP_FINGERPRINTS,
    parser_census_logical_keys,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    FrozenSourceRemediationRequiredError,
    _raw_parse_success_state,
    record_current_parser_source_census,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    apply_source_raw_state_update,
    upsert_raw_artifact,
)
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionRows, prepare_session_rows

_LOGGER = _polylogue_logging.get_logger(__name__)
_REPLAY_PROVIDER_DETECTION_PREFIX_BYTES: Final[int] = 8192


def _detect_unknown_retained_provider(
    payload: BinaryIO,
    source_path: str,
) -> tuple[Provider, str]:
    """Detect retained UNKNOWN bytes without eagerly materializing JSONL.

    A byte prefix can end inside the first physical JSONL record.  For an
    oversized record stream that makes a prefix-only detector inconclusive
    even when a later bounded record identifies a streaming provider.  Scan
    complete records across the stream instead: each record is capped at the
    same detection bound, oversized records are consumed in bounded chunks,
    and the scan continues until positive provider evidence or EOF.

    Non-JSONL documents retain prefix detection here and their existing
    complete-document retry at the caller.  That eager retry is required for
    document providers whose first complete value exceeds the prefix.
    """
    stream_name = Path(source_path).name
    if not is_jsonl_source_path(source_path):
        return detect_provider_from_raw_bytes_evidence(
            payload.read(_REPLAY_PROVIDER_DETECTION_PREFIX_BYTES),
            stream_name,
            Provider.UNKNOWN,
            truncated_tail_ok=True,
        )

    # Local import avoids a source-dispatch import cycle during module load;
    # this is the same bounded physical-record iterator used by raw-payload
    # artifact inspection.
    from polylogue.archive.raw_payload.decode import _bounded_raw_lines

    last_evidence = "no bounded JSONL record identified a provider; used fallback_provider"
    for raw_line, oversized in _bounded_raw_lines(
        payload,
        max_record_bytes=_REPLAY_PROVIDER_DETECTION_PREFIX_BYTES,
    ):
        if oversized or raw_line is None:
            continue
        record_bytes = raw_line.encode("utf-8", errors="surrogatepass") if isinstance(raw_line, str) else raw_line
        provider, last_evidence = detect_provider_from_raw_bytes_evidence(
            record_bytes,
            stream_name,
            Provider.UNKNOWN,
        )
        if provider is not Provider.UNKNOWN:
            return provider, last_evidence
    return Provider.UNKNOWN, last_evidence


def _canonical_authority_logical_key(logical_key: str) -> str:
    """Normalize transitional provider and public-origin authority prefixes."""
    prefix, separator, native_id = logical_key.partition(":")
    if not separator or not native_id:
        raise ValueError(f"invalid logical source key: {logical_key!r}")
    try:
        origin = Origin(prefix)
    except ValueError:
        try:
            origin = origin_from_provider(Provider(prefix))
        except ValueError as exc:
            raise ValueError(f"unknown logical source key prefix: {prefix!r}") from exc
    return f"{origin.value}:{native_id}"


def _expand_frozen_revision_link_selection(archive_root: Path, raw_ids: Sequence[str]) -> tuple[str, ...]:
    """Include every predecessor and baseline needed to validate selected APPEND authority."""
    expanded = set(raw_ids)
    pending = set(raw_ids)
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
        while pending:
            current = tuple(sorted(pending))
            pending.clear()
            for offset in range(0, len(current), 500):
                chunk = current[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = source_conn.execute(
                    f"""
                    SELECT predecessor_raw_id, baseline_raw_id
                    FROM raw_sessions WHERE raw_id IN ({placeholders})
                    """,
                    chunk,
                )
                for predecessor_raw_id, baseline_raw_id in rows:
                    for linked_raw_id in (predecessor_raw_id, baseline_raw_id):
                        if linked_raw_id is not None and str(linked_raw_id) not in expanded:
                            expanded.add(str(linked_raw_id))
                            pending.add(str(linked_raw_id))
    return tuple(sorted(expanded))


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
    #: Wall-clock seconds per named stage of this backfill call, keyed by the
    #: SAME names logged as ``"backfill stage timings: ..."`` below --
    #: ``census``/``census_receipt``/``spill_load`` (decode-side, computed
    #: here) plus ``revision_replay.*``/``membership_replay.*`` (writer-side,
    #: recorded by ``ArchiveStore.apply_raw_revision_replay`` /
    #: ``apply_raw_membership_classification`` through the SAME dict passed
    #: in as ``stage_timings_s``) and ``total``. This is the parse-vs-apply
    #: split callers need: it was already being computed and logged, just
    #: discarded at this function's return boundary -- see
    #: ``split_parse_and_apply_seconds`` for the two-bucket rollup.
    #: ``compare=False`` -- wall-clock timings are never equal across two
    #: independent runs by construction, and existing callers (e.g.
    #: ``test_thread_parse_matches_sequential_archive_state``) compare two
    #: ``RevisionBackfillResult``s for LOGICAL equality (same counts across
    #: execution modes), not identical timing. Diagnostic metadata must not
    #: change that contract.
    stage_timings_s: dict[str, float] = field(default_factory=dict, compare=False)


#: Stage-timing keys that are decode work (read-only blob->ParsedSession
#: parse), as opposed to everything else recorded in the same dict, which is
#: SQLite writer work (index/FTS/projection writes under
#: ``revision_replay.*`` / ``membership_replay.*`` prefixes, plus the small
#: ``census_receipt`` parser-fingerprint commit). Used by
#: ``split_parse_and_apply_seconds`` to answer "is decode or the single
#: writer the bottleneck?" without guessing at every current and future
#: writer-stage key name.
_PARSE_STAGE_TIMING_KEYS: Final[frozenset[str]] = frozenset({"census", "spill_load"})


def split_parse_and_apply_seconds(stage_timings_s: dict[str, float]) -> tuple[float, float]:
    """Roll up a backfill's per-stage timings into (parse_s, apply_s).

    ``parse_s`` is decode work (``census`` + ``spill_load``): read-only,
    embarrassingly parallel, scales with ``parse_workers``. ``apply_s`` is
    everything else charged against ``total`` (writer-side index/FTS/
    projection writes plus the small census receipt commit): serialized
    through the single SQLite writer and does not scale with worker count.
    Returns ``(0.0, 0.0)`` if ``total`` was never recorded (e.g. an empty
    backfill that returned before any stage ran).
    """
    total_s = stage_timings_s.get("total", 0.0)
    parse_s = sum(stage_timings_s.get(key, 0.0) for key in _PARSE_STAGE_TIMING_KEYS)
    apply_s = max(0.0, total_s - parse_s)
    return parse_s, apply_s


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


#: Content-cache dedup key: ``(provider, blob_hash, dedup_path, native_id)``,
#: identical in shape to the per-batch grouping key ``_parse_retained_raws``
#: already uses (``dedup_path`` is ``""`` for
#: :data:`_PATH_INDEPENDENT_PARSE_PROVIDERS`, else the raw's own
#: ``source_path``) -- see :func:`_parse_retained_raws`'s docstring for why
#: that key shape is safe to reuse across rows. ``native_id`` (polylogue-
#: 6lyh1) is the fallback-identity hint an APPEND-kind raw recovers at
#: replay time (``None`` for every FULL raw and every APPEND raw with no
#: recorded hint); without it in the key, two byte-identical APPEND payloads
#: recorded under different recovered native ids -- content-hash dedup alone
#: cannot see that divergence, since path-independent providers deliberately
#: ignore ``source_path`` too -- would incorrectly fan the SAME parsed
#: session identity out to both raw_ids.
ContentCacheKey = tuple[Provider, str, str, str | None]

#: Default budget for :class:`RawParsePrefetchCache`'s cross-page content
#: cache (polylogue-oab7). Deliberately a small, fixed, conservative default
#: independent of ``max_inflight_bytes`` (the daemon's adaptive 64MiB-2GiB
#: single-tick budget for its OWN raw_id-keyed warm-ahead entries) rather than
#: reusing that number: the two caches are separate dicts inside the same
#: object and can both be resident at once, so summing an adaptive multi-GiB
#: budget with itself risks doubling the daemon's already-tuned whale-memory
#: ceiling. 256 MiB mirrors this file's own ``_DECODED_CACHE_MIN_TREE_BYTES``
#: and ``daemon/parse_prefetch.py``'s ``_MIN_MAX_CACHED_TREE_BYTES`` floor --
#: small enough to be noise against either budget, still large enough to hold
#: several typical (non-whale) parsed sessions resident across page boundaries.
_DEFAULT_CONTENT_CACHE_BYTES: Final[int] = 256 * 1024 * 1024


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

    polylogue-oab7: this class ALSO carries a second, independent store --
    the "content cache" (``get_content``/``put_content``) -- keyed by
    :data:`ContentCacheKey` rather than ``raw_id``. Unlike the raw_id-keyed
    entries above (single-pop, meant to be consumed exactly once by the
    specific raw the daemon's warmer pre-parsed), content-cache entries are
    LRU-retained until evicted, so a raw whose bytes were already parsed on
    an EARLIER page of the SAME long-lived cache instance (the daemon's
    ``DaemonParseStage.cache`` is a process-lifetime singleton -- see
    ``daemon/cli.py``'s ``_daemon_bulk_rebuild_parse_stage()``) is served
    from cache on a LATER page instead of reparsed, closing the one real gap
    left by polylogue-869u's existing dedup (which only reuses a parse
    WITHIN one bounded ``_parse_retained_raws`` batch/page, never across the
    many pages one archive-wide rebuild is split into). A miss here degrades
    identically to a miss on the raw_id-keyed store: parsed normally, nothing
    lost, only possibly reparsed.
    """

    def __init__(self, *, max_inflight_bytes: int, max_content_cache_bytes: int | None = None) -> None:
        if max_inflight_bytes < 1:
            raise ValueError("max_inflight_bytes must be positive")
        self._max_inflight_bytes = max_inflight_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, _PrefetchedParse] = {}
        self._inflight_bytes = 0
        self._max_content_cache_bytes = (
            max_content_cache_bytes if max_content_cache_bytes is not None else _DEFAULT_CONTENT_CACHE_BYTES
        )
        if self._max_content_cache_bytes < 1:
            raise ValueError("max_content_cache_bytes must be positive")
        self._content_entries: OrderedDict[ContentCacheKey, _PrefetchedParse] = OrderedDict()
        self._content_bytes = 0

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

    def peek_logical_keys(self) -> dict[str, str]:
        """Non-destructive ``raw_id -> "{origin}:{provider_session_id}"`` map.

        polylogue-pzxm: the sharded from-empty rebuild
        (``maintenance/sharded_rebuild.py``) partitions raw ids by revision-
        cohort BEFORE any shard consumes this cache with ``pop``, using the
        exact same logical-key derivation ``_parse_retained_raws`` uses when
        it writes ``membership_candidates``/``provisional_full_raw_ids``
        (``f"{session.source_name.value}:{session.provider_session_id}"``),
        so a byte-growth chain member or an ambiguous-identity pair never
        gets scheduled onto two different shards, each of which would see
        only a partial candidate set and reach a wrong (or crashing)
        classification. Unlike ``pop``, this never removes an entry or
        touches budget accounting -- reading it does not consume the cache
        for whichever shard's own replay pops these raw ids afterwards.
        Sessions with more than one parsed logical unit (bundle raws) are
        skipped: sharding falls back to that raw id's own ``source_path`` for
        those, matching pre-classification behavior for any raw this cache
        does not cover.
        """
        with self._lock:
            keys: dict[str, str] = {}
            for raw_id, entry in self._entries.items():
                if len(entry.sessions) == 1:
                    session = entry.sessions[0]
                    keys[raw_id] = f"{session.source_name.value}:{session.provider_session_id}"
            return keys

    def content_len(self) -> int:
        """Number of distinct content-cache entries currently resident."""
        with self._lock:
            return len(self._content_entries)

    def get_content(self, key: ContentCacheKey) -> tuple[list[ParsedSession], int, RawRevisionKind] | None:
        """Peek a content-cache entry without consuming it (unlike ``pop``).

        Multiple later raw_ids sharing ``key`` may each hit the same entry;
        touching it here marks it most-recently-used for the LRU eviction
        order in :meth:`put_content`.
        """
        with self._lock:
            entry = self._content_entries.get(key)
            if entry is None:
                return None
            self._content_entries.move_to_end(key)
            return entry.sessions, entry.payload_bytes, entry.revision_kind

    def put_content(
        self,
        key: ContentCacheKey,
        sessions: list[ParsedSession],
        *,
        payload_bytes: int,
        revision_kind: RawRevisionKind,
    ) -> bool:
        """Admit one freshly-parsed representative's output into the content cache.

        Returns ``False`` (a pure no-op) when ``key`` is already resident or
        ``payload_bytes`` alone exceeds the whole budget -- a single whale
        entry must never be admitted only to immediately evict every other
        entry and then still not fit itself. Otherwise admits and evicts
        least-recently-used entries (oldest ``get_content``/``put_content``
        touch first) until back under budget.
        """
        with self._lock:
            if key in self._content_entries:
                return False
            if payload_bytes > self._max_content_cache_bytes:
                return False
            self._content_entries[key] = _PrefetchedParse(sessions, payload_bytes, revision_kind)
            self._content_entries.move_to_end(key)
            self._content_bytes += payload_bytes
            while self._content_bytes > self._max_content_cache_bytes and self._content_entries:
                _evicted_key, evicted_entry = self._content_entries.popitem(last=False)
                self._content_bytes -= evicted_entry.payload_bytes
            return True


class RawRevisionReplayResourceBlockedError(RuntimeError):
    def __init__(self, raw_ids: list[str], limit_bytes: int, total_bytes: int) -> None:
        self.raw_ids = tuple(raw_ids)
        self.limit_bytes = limit_bytes
        self.total_bytes = total_bytes
        super().__init__(f"{len(raw_ids)} raw revision(s) total {total_bytes} bytes exceed replay limit {limit_bytes}")


class RebuildDeadlineExceededError(RuntimeError):
    """Raised by an injected ``deadline_check`` callback to stop replay mid-page.

    polylogue-uhgm: ``backfill_historical_revision_evidence``'s REPLAY phase
    (the byte-cohort and membership-cohort loops below) calls an optional
    caller-supplied ``deadline_check`` once at the top of every cohort
    iteration -- i.e. *between* cohorts, never mid-cohort-apply. A cohort
    already durably committed (or accepted into the currently open,
    not-yet-committed replay batch) when this fires stands; the cohort about
    to start does not begin. The caller (``maintenance/rebuild_index.py``)
    catches this and checkpoints its resumable transaction WITHOUT advancing
    the cursor, so the next pass re-derives from the exact same source-order
    position -- safe by the existing content-hash idempotency invariant
    (re-applying an already-committed cohort is a no-op upsert, never a
    duplicate), matching the crash-recovery contract an open replay batch
    already has (``test_backfill_resumes_after_replay_batch_crash_discards_
    whole_batch_cleanly``). This closes the gap where a bounded pass's
    deadline was previously observed only after the WHOLE requested page
    replayed to completion.
    """


def _resource_blocked_parser_fingerprint(max_payload_bytes: int) -> str:
    """Return the durable admission identity for one bounded census envelope."""
    return f"{RAW_AUTHORITY_PARSER_FINGERPRINT}:resource-blocked:{max_payload_bytes}"


def uncensused_historical_revision_raw_ids(
    archive_root: Path,
    raw_ids: list[str],
    *,
    max_payload_bytes: int | None = None,
) -> tuple[str, ...]:
    """Return inputs whose current parser identity has not been persisted.

    The dedicated receipt proves that *some* parser version whose semantics
    are still known to this codebase actually observed every relevant raw.
    Durable revision or membership rows alone may have been produced by an
    older parser and therefore cannot establish current quiescence.

    This deliberately accepts any *known* fingerprint (the current one, or
    one listed in ``SUPERSEDED_MEMBERSHIP_FINGERPRINTS``), not only the
    current one (polylogue-9dxn): the census answers "was this raw ever
    observed by a real parser?", which a fingerprint bump alone does not
    change -- only ``classify_membership_revisions`` semantics changing (a
    superseded fingerprint) can make a *verdict* stale, which is a separate
    question the terminal-decision check in ``storage/repair.py`` answers.
    Treating a bump as forcing full re-census here would mean every
    fingerprint bump re-parses the entire archive just to re-confirm facts
    that did not change.
    """
    if not raw_ids:
        return ()
    resource_blocked_fingerprint = (
        _resource_blocked_parser_fingerprint(max_payload_bytes) if max_payload_bytes is not None else None
    )
    known_fingerprints = [RAW_AUTHORITY_PARSER_FINGERPRINT, *sorted(SUPERSEDED_MEMBERSHIP_FINGERPRINTS)]
    known_placeholders = ",".join("?" for _ in known_fingerprints)
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        uncensused: list[str] = []
        for offset in range(0, len(raw_ids), 500):
            raw_id_chunk = raw_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in raw_id_chunk)
            rows = conn.execute(
                f"""
                SELECT r.raw_id
                FROM raw_sessions AS r
                LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                WHERE r.raw_id IN ({placeholders})
                  AND NOT COALESCE(
                      c.parser_fingerprint IN ({known_placeholders})
                      AND c.status = 'complete'
                      AND c.detail LIKE 'parser-observed:%',
                      0
                  )
                  AND NOT COALESCE(
                      c.parser_fingerprint = ?
                      AND c.status = 'failed',
                      0
                  )
                ORDER BY r.raw_id
                """,
                [*raw_id_chunk, *known_fingerprints, resource_blocked_fingerprint],
            )
            uncensused.extend(str(row[0]) for row in rows)
    return tuple(sorted(uncensused))


def record_resource_blocked_revision_census(
    archive_root: Path,
    raw_ids: tuple[str, ...],
    *,
    max_payload_bytes: int,
    total_payload_bytes: int,
    stream_safe: bool | None = None,
) -> None:
    """Persist a non-terminal no-retry receipt for immutable oversized bytes.

    ``failed`` is deliberately truthful: the current parser has not inspected
    the payload.  The fingerprint binds that fact to the exact admission
    envelope, so increasing the envelope (or changing the parser identity)
    re-admits the raw without a timer-driven retry storm.

    ``stream_safe`` (polylogue-t93b, default ``None`` for callers that have
    not classified it) records whether every member of the blocked
    component is stream-record-safe -- the daemon's escalation-tier whale
    pass only ever selects
    stream-safe components, so this distinguishes "waiting for a bounded
    whale pass" from "genuinely cannot converge automatically" directly in
    the durable census detail, without changing the admission fingerprint
    (which stays keyed on the envelope alone, preserving the existing
    re-admission-on-wider-envelope invariant).
    """
    if not raw_ids:
        return
    fingerprint = _resource_blocked_parser_fingerprint(max_payload_bytes)
    escalation_note = (
        "; escalation-eligible: stream-safe, awaiting a bounded daemon whale pass"
        if stream_safe
        else "; escalation-blocked: non-stream-safe, requires manual/offline convergence"
        if stream_safe is False
        else ""
    )
    detail = (
        "current parser census deferred before blob open: "
        f"component payload {total_payload_bytes} exceeds envelope {max_payload_bytes}{escalation_note}"
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
    ``classify_raw_revision_cohort_for_rebuild_repair`` (called later, during
    replay) still independently re-derives byte-provenness from raw bytes for
    every raw.
    """
    state = _RevisionCensusState(0, 0, 0, set(), {}, {})
    batch_size = commit_batch_size if commit_batch_size is not None and commit_batch_size > 0 else None
    batched = batch_size is not None
    pending_commits = 0

    def commit_unit() -> None:
        nonlocal pending_commits
        pending_commits += 1
        if batch_size is None:
            # ``bind_raw_revision(manage_transaction=True)`` commits its
            # revision write before the parser receipt is recorded below.
            # Commit again at the unit boundary so that trailing receipt is
            # durable before this archive wrapper closes.
            archive.commit()
            pending_commits = 0
        elif pending_commits >= batch_size:
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
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
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
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                detail=str(outcome),
                manage_transaction=not batched,
            )
            state.quarantined += 1
            commit_unit()
            return
        sessions, payload_bytes, revision_kind = outcome
        stored_provider, _blob_hash, _source_path, _stored_kind, _stored_size = archive.raw_revision_descriptor(raw_id)
        if stored_provider is Provider.UNKNOWN and sessions:
            # Acquisition deliberately did not decode an UNKNOWN source-only
            # member.  A successful replay now has durable shape evidence for
            # its provider, so retain that result independently of the later
            # index promotion outcome.
            apply_source_raw_state_update(
                archive._ensure_source_conn(),
                raw_id,
                state=RawSessionStateUpdate(payload_provider=Provider.from_string(sessions[0].source_name)),
                manage_transaction=not batched,
            )
        if not sessions:
            provider = _detected_provider_for_empty_replay(
                archive,
                raw_id,
                stored_provider=stored_provider,
                source_path=_source_path,
            )
            # A terminal artifact makes this raw ineligible for future census
            # work. Its artifact carrier, parse state, and both census receipts
            # must therefore become durable as one source-tier transaction.
            # Batches retain that transaction until their existing commit
            # boundary instead of forcing one SQLite commit per empty raw.
            transaction = nullcontext() if batched else archive._ensure_source_conn()
            with transaction:
                if stored_provider is Provider.UNKNOWN and provider is not Provider.UNKNOWN:
                    apply_source_raw_state_update(
                        archive._ensure_source_conn(),
                        raw_id,
                        state=RawSessionStateUpdate(payload_provider=provider),
                        manage_transaction=False,
                    )
                terminalized = _persist_terminal_non_session_artifact(
                    archive,
                    raw_id,
                    provider=provider,
                    source_path=_source_path,
                    source_index=source_index,
                    manage_transaction=False,
                )
                if provider is not Provider.UNKNOWN:
                    archive.replace_raw_membership_census(
                        raw_id,
                        [],
                        parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                        censused_at_ms=0,
                        retire_full_revision_governance=revision_kind is not RawRevisionKind.UNKNOWN,
                        manage_transaction=False,
                    )
                    if not terminalized:
                        apply_source_raw_state_update(
                            archive._ensure_source_conn(),
                            raw_id,
                            state=_raw_parse_success_state(provider),
                            manage_transaction=False,
                        )
            if provider is not Provider.UNKNOWN:
                commit_unit()
                return
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
            record_current_parser_source_census(archive._ensure_source_conn(), raw_id, parser_sessions=sessions)
            state.provisional_full_raw_ids.setdefault(logical_key, set()).add(raw_id)
            commit_unit()
        elif revision_kind is RawRevisionKind.UNKNOWN:
            archive.replace_raw_membership_census(
                raw_id,
                sessions,
                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                censused_at_ms=0,
                manage_transaction=not batched,
            )
            for session in sessions:
                logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
            commit_unit()
        else:
            record_current_parser_source_census(archive._ensure_source_conn(), raw_id, parser_sessions=sessions)
            commit_unit()

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
        record_current_parser_source_census(
            archive._ensure_source_conn(),
            raw_id,
            inherited_logical_keys=(logical_key,),
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
                for raw_id, _source_index, _terminal_non_session, _raw_rowid in sorted(
                    rows,
                    key=lambda row: archive.raw_revision_observation_order(row[0]),
                ):
                    if raw_id in state.censused or not _replay_retained_codex_state_evidence(archive, raw_id):
                        continue
                    state.scanned += 1
                    state.censused.add(raw_id)
                    commit_unit()
                terminal_raw_ids = {
                    raw_id for raw_id, _source_index, terminal_non_session, _raw_rowid in rows if terminal_non_session
                }
                for raw_id in terminal_raw_ids - state.censused:
                    state.scanned += 1
                    state.censused.add(raw_id)
                pending_rows = [
                    (raw_id, source_index)
                    for raw_id, source_index, terminal_non_session, _raw_rowid in rows
                    if raw_id not in state.censused and not terminal_non_session
                ]
                if max_payload_bytes is not None:
                    payload_sizes = archive.raw_payload_sizes([raw_id for raw_id, _index in pending_rows])
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
        if batched:
            archive.rollback()
        raise
    if batched and pending_commits > 0:
        archive.commit()
    return state


def _load_frozen_revision_evidence(
    archive: ArchiveStore,
    spill: _ParsedSessionSpill,
    *,
    selected_raw_ids: list[str] | None,
    max_payload_bytes: int | None,
    ingest_workers: int,
    prefetch_cache: RawParsePrefetchCache | None,
) -> _RevisionCensusState:
    """Parse a phase-2 source snapshot without changing its durable ledger."""
    expanded_raw_ids, _logical_keys = archive.expand_raw_membership_selection(selected_raw_ids)
    if selected_raw_ids is not None:
        expanded_raw_ids = _expand_frozen_revision_link_selection(archive.archive_root, expanded_raw_ids)
    recorded_logical_keys = require_current_parser_source_census(
        archive.archive_root,
        selected_raw_ids=expanded_raw_ids if selected_raw_ids is not None else None,
    )
    rows = archive.raw_membership_census_rows(expanded_raw_ids if selected_raw_ids is not None else None)
    if max_payload_bytes is not None:
        payload_sizes = archive.raw_payload_sizes(
            [raw_id for raw_id, _source_index, terminal_non_session, _raw_rowid in rows if not terminal_non_session]
        )
        total_payload_bytes = sum(payload_sizes.values())
        oversized = [raw_id for raw_id, size in payload_sizes.items() if size > max_payload_bytes]
        if oversized or total_payload_bytes > max_payload_bytes:
            raise RawRevisionReplayResourceBlockedError(
                sorted(oversized or payload_sizes), max_payload_bytes, total_payload_bytes
            )
    parseable_raw_ids = [
        raw_id
        for raw_id, source_index, terminal_non_session, _raw_rowid in rows
        if source_index >= 0 and not terminal_non_session
    ]
    parsed_outcomes = _parse_retained_raws(
        archive,
        parseable_raw_ids,
        ingest_workers=ingest_workers,
        prefetch_cache=prefetch_cache,
    )
    state = _RevisionCensusState(0, 0, 0, set(), {}, {})
    for raw_id, source_index, terminal_non_session, _raw_rowid in rows:
        state.scanned += 1
        state.censused.add(raw_id)
        if terminal_non_session:
            continue
        if source_index < 0:
            state.quarantined += 1
            continue
        outcome = parsed_outcomes[raw_id]
        if isinstance(outcome, Exception):
            raise FrozenSourceRemediationRequiredError(
                f"inactive candidate could not parse frozen raw {raw_id}: {type(outcome).__name__}: {outcome}"
            ) from outcome
        sessions, payload_bytes, revision_kind = outcome
        parsed_logical_keys = tuple(
            sorted(
                {
                    # Parser output uses Provider internally, while the
                    # persisted census is normalized to public Origin keys.
                    f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
                    for session in sessions
                }
            )
        )
        if recorded_logical_keys[raw_id] != parsed_logical_keys:
            raise FrozenSourceRemediationRequiredError(
                "inactive candidate re-derived different current-parser logical keys for frozen raw "
                f"{raw_id}: recorded={recorded_logical_keys[raw_id]!r}, parsed={parsed_logical_keys!r}"
            )
        spill.add(raw_id, sessions, payload_bytes=payload_bytes)
        state.classified += int(len(sessions) == 1)
        if revision_kind is RawRevisionKind.UNKNOWN:
            for session in sessions:
                # Membership rows intentionally retain their provider-wire
                # identity until durable authority comparison normalizes it.
                logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
    return state


def require_current_parser_source_census(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Require phase-2 parser receipts before allocating an index candidate."""
    stale_raw_ids: list[str] = []
    recorded_logical_keys: dict[str, tuple[str, ...]] = {}
    selections: tuple[tuple[str, ...] | None, ...]
    if selected_raw_ids is None:
        selections = (None,)
    else:
        selections = tuple(
            tuple(selected_raw_ids[offset : offset + 500]) for offset in range(0, len(selected_raw_ids), 500)
        )
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
        for selection in selections:
            where = "" if selection is None else f"WHERE r.raw_id IN ({','.join('?' for _ in selection)})"
            params: tuple[object, ...] = () if selection is None else selection
            rows = source_conn.execute(
                f"""
                SELECT r.raw_id, c.parser_fingerprint, c.status, c.logical_keys_json
                FROM raw_sessions AS r
                LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                {where}
                ORDER BY r.raw_id
                """,
                params,
            )
            for raw_id_value, fingerprint, status, logical_keys_json in rows:
                raw_id = str(raw_id_value)
                if fingerprint != RAW_AUTHORITY_PARSER_FINGERPRINT or status != "complete":
                    stale_raw_ids.append(raw_id)
                    continue
                normalized_keys = parser_census_logical_keys(logical_keys_json)
                if normalized_keys is None:
                    stale_raw_ids.append(raw_id)
                    continue
                recorded_logical_keys[raw_id] = normalized_keys
    if stale_raw_ids:
        sample = ", ".join(stale_raw_ids[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate requires a complete current-parser source census; "
            f"{len(stale_raw_ids)} raw(s) are stale or incomplete (sample: {sample})"
        )

    durable_bindings: dict[str, tuple[object, object, list[object], bool]] = {
        raw_id: (None, RawRevisionKind.UNKNOWN.value, [], False) for raw_id in recorded_logical_keys
    }
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
        for selection in selections:
            where = "" if selection is None else f"WHERE r.raw_id IN ({','.join('?' for _ in selection)})"
            params = () if selection is None else selection
            rows = source_conn.execute(
                f"""
                SELECT r.raw_id, r.logical_source_key, r.revision_kind, m.logical_source_key,
                       EXISTS(SELECT 1 FROM raw_artifacts AS a WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0)
                FROM raw_sessions AS r
                LEFT JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
                {where}
                ORDER BY r.raw_id, m.logical_source_key
                """,
                params,
            )
            for raw_id_value, typed_key, revision_kind, membership_key, typed_non_session in rows:
                raw_id = str(raw_id_value)
                existing_typed, existing_kind, memberships, existing_non_session = durable_bindings.get(
                    raw_id, (typed_key, revision_kind, [], bool(typed_non_session))
                )
                if membership_key is not None:
                    memberships.append(membership_key)
                durable_bindings[raw_id] = (
                    typed_key if existing_typed is None else existing_typed,
                    revision_kind if existing_kind == RawRevisionKind.UNKNOWN.value else existing_kind,
                    memberships,
                    bool(typed_non_session) or existing_non_session,
                )

    invalid_durable_bindings: set[str] = set()
    durable_logical_keys: dict[str, tuple[str, ...]] = {}
    for raw_id, (typed_key, revision_kind, membership_keys, typed_non_session) in durable_bindings.items():
        durable_keys = (
            ()
            if typed_non_session
            else durable_authority_logical_keys(
                raw_logical_key=typed_key,
                revision_kind=revision_kind,
                membership_logical_keys=membership_keys,
            )
        )
        if durable_keys is None:
            invalid_durable_bindings.add(raw_id)
        else:
            durable_logical_keys[raw_id] = durable_keys

    authority_binding_drift = sorted(
        invalid_durable_bindings
        | {
            raw_id
            for raw_id, census_keys in recorded_logical_keys.items()
            if durable_logical_keys.get(raw_id, ()) != census_keys
        }
    )
    if authority_binding_drift:
        sample = ", ".join(authority_binding_drift[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate current-parser census differs from frozen durable authority bindings; "
            f"{len(authority_binding_drift)} raw(s) require source remediation (sample: {sample})"
        )

    authority_rows: dict[str, tuple[str | None, str, str, int, str | None, str | None]] = {}
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
        for selection in selections:
            where = "" if selection is None else f"WHERE raw_id IN ({','.join('?' for _ in selection)})"
            params = () if selection is None else selection
            rows = source_conn.execute(
                f"""
                SELECT raw_id, logical_source_key, revision_kind, revision_authority,
                       source_index, predecessor_raw_id, baseline_raw_id
                FROM raw_sessions {where}
                ORDER BY raw_id
                """,
                params,
            )
            for raw_id_value, logical_key, revision_kind, authority, source_index, predecessor_id, baseline_id in rows:
                authority_rows[str(raw_id_value)] = (
                    str(logical_key) if logical_key is not None else None,
                    str(revision_kind),
                    str(authority),
                    int(source_index),
                    str(predecessor_id) if predecessor_id is not None else None,
                    str(baseline_id) if baseline_id is not None else None,
                )

    append_identity_drift: set[str] = set()
    for raw_id, (
        append_key,
        revision_kind,
        authority,
        _source_index,
        predecessor_id,
        baseline_id,
    ) in authority_rows.items():
        if revision_kind != RawRevisionKind.APPEND.value:
            continue
        predecessor = authority_rows.get(predecessor_id or "")
        baseline = authority_rows.get(baseline_id or "")
        if (
            append_key is None
            or authority != RawRevisionAuthority.BYTE_PROVEN.value
            or predecessor_id is None
            or baseline_id is None
            or predecessor_id == raw_id
            or baseline_id == raw_id
            or predecessor is None
            or baseline is None
            or predecessor[2] != RawRevisionAuthority.BYTE_PROVEN.value
            or baseline[1] != RawRevisionKind.FULL.value
            or baseline[2] != RawRevisionAuthority.BYTE_PROVEN.value
            or baseline[3] < 0
        ):
            append_identity_drift.add(raw_id)
            continue
        try:
            canonical_append_key = _canonical_authority_logical_key(append_key)
            canonical_predecessor_key = _canonical_authority_logical_key(predecessor[0] or "")
            canonical_baseline_key = _canonical_authority_logical_key(baseline[0] or "")
        except ValueError:
            append_identity_drift.add(raw_id)
            continue
        if {canonical_predecessor_key, canonical_baseline_key} != {canonical_append_key}:
            append_identity_drift.add(raw_id)
            continue

        seen = {raw_id}
        cursor_id = predecessor_id
        while True:
            if cursor_id in seen:
                append_identity_drift.add(raw_id)
                break
            seen.add(cursor_id)
            cursor = authority_rows.get(cursor_id)
            if cursor is None:
                append_identity_drift.add(raw_id)
                break
            if cursor[1] == RawRevisionKind.FULL.value:
                if cursor_id != baseline_id:
                    append_identity_drift.add(raw_id)
                break
            if cursor[1] != RawRevisionKind.APPEND.value or cursor[4] is None:
                append_identity_drift.add(raw_id)
                break
            try:
                if _canonical_authority_logical_key(cursor[0] or "") != canonical_append_key:
                    append_identity_drift.add(raw_id)
                    break
            except ValueError:
                append_identity_drift.add(raw_id)
                break
            cursor_id = cursor[4]
    if append_identity_drift:
        sample = ", ".join(sorted(append_identity_drift)[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate typed continuation identity differs from linked byte authority; "
            f"{len(append_identity_drift)} raw(s) require source remediation (sample: {sample})"
        )

    unresolved_raw_ids: list[str] = []
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
        for selection in selections:
            authority_where = "" if selection is None else f"AND r.raw_id IN ({','.join('?' for _ in selection)})"
            authority_params: tuple[object, ...] = () if selection is None else selection
            unresolved_raw_ids.extend(
                str(row[0])
                for row in source_conn.execute(
                    f"""
                    SELECT DISTINCT r.raw_id
                    FROM raw_sessions AS r
                    LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                    LEFT JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
                    WHERE r.revision_authority = 'quarantined'
                      {authority_where}
                      AND NOT EXISTS (
                          SELECT 1 FROM raw_artifacts AS a
                          WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0
                      )
                      AND (
                          c.raw_id IS NULL OR c.status NOT IN ('complete', 'non_session')
                          OR (
                              c.status = 'complete'
                              AND (
                                  m.raw_id IS NULL OR m.decision IS NULL
                                  OR m.decision IN ('ambiguous', 'deferred')
                              )
                          )
                      )
                    ORDER BY r.raw_id
                    """,
                    authority_params,
                )
            )
    if unresolved_raw_ids:
        sample = ", ".join(unresolved_raw_ids[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate requires complete frozen source authority; "
            f"{len(unresolved_raw_ids)} raw(s) remain quarantined or undecided (sample: {sample})"
        )
    return recorded_logical_keys


def validate_frozen_source_authority(
    archive_root: Path,
    *,
    active_index_path: Path | None = None,
    selected_raw_ids: list[str] | None = None,
    max_payload_bytes: int | None = None,
    ingest_workers: int = 1,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> None:
    """Re-derive every selected source decision before allocating a candidate."""
    with (
        ArchiveStore.open_frozen_source_validation(
            archive_root,
            active_index_path=active_index_path,
        ) as archive,
        _ParsedSessionSpill(
            archive_root,
            index_path=active_index_path,
            max_cached_payload_bytes=max_payload_bytes,
        ) as spill,
    ):
        census = _load_frozen_revision_evidence(
            archive,
            spill,
            selected_raw_ids=selected_raw_ids,
            max_payload_bytes=max_payload_bytes,
            ingest_workers=ingest_workers,
            prefetch_cache=prefetch_cache,
        )
        _unclassified, logical_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        _membership_raw_ids, persisted_membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        membership_keys = {*persisted_membership_keys, *census.membership_candidates}
        byte_replayed_keys: set[str] = set()

        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort_for_frozen_candidate(logical_key)
            if not plan.accepted_raw_ids:
                convertible = archive.convertible_full_revision_raw_ids(logical_key)
                if convertible:
                    raise FrozenSourceRemediationRequiredError(
                        "inactive candidate found a full-revision cohort that still requires membership "
                        f"remediation in frozen source: {logical_key}"
                    )
                continue
            byte_replayed_keys.add(logical_key)

        for logical_key in sorted(membership_keys - byte_replayed_keys):
            candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
            candidate_raw_ids.update(census.membership_candidates.get(logical_key, ()))
            revisions: list[MembershipRevision] = []
            for raw_id in sorted(candidate_raw_ids):
                sessions, _payload_bytes = spill.for_raw(archive, raw_id)
                for session in sessions:
                    session_logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                    if session_logical_key != logical_key:
                        continue
                    projection = session_revision_projection(session)
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
            classification = classify_membership_revisions(revisions, existing_accepted_raw_id=None)
            archive.require_frozen_membership_authority(logical_key, classification)


def census_historical_revision_evidence(
    archive_root: Path,
    *,
    active_index_path: Path | None = None,
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
        _ParsedSessionSpill(
            archive_root,
            index_path=active_index_path,
            max_cached_payload_bytes=max_payload_bytes,
        ) as spill,
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
    return RevisionCensusResult(
        state.scanned,
        state.classified,
        state.quarantined,
        expanded,
        logical_keys,
    )


def _lineage_aware_replay_order(
    logical_keys: set[str],
    archive: ArchiveStore,
    spill: _ParsedSessionSpill,
    archive_root: Path,
) -> list[str]:
    """Order one rebuild's byte-typed logical keys so a parent's cohort
    replays before any of its children's (polylogue-5q2u).

    Replaying a child before its parent forces ``_resolve_session_graph`` to
    store the child's shared prefix WHOLE, then re-walk and normalize it
    (delete the duplicate prefix rows, remap ``session_events`` refs, delete
    prefix-scoped dependents) once the parent finally arrives -- the
    #2467 deferred-tail path, O(orphaned_children * shared_prefix_size) real
    row-mutation work. The previous ``sorted(logical_keys)`` lexicographic
    order has zero relationship to parent/child lineage, so it triggers this
    expensive path roughly as often as not during a cold/full rebuild.
    Visiting roots first (and each child only after its parent) minimizes
    how often it triggers.

    This is deliberately scheduling-only: it must never change WHAT gets
    replayed or adopted, only the order this module's own replay loop visits
    logical keys in. A key whose parent cannot be resolved here -- no
    ``parent_session_provider_id``, a parent outside this rebuild's
    ``logical_keys`` (missing/external/cross-batch parent), or a lineage
    cycle -- degrades to the original lexicographic position among the
    unresolved remainder. Nothing is ever skipped.
    """
    sorted_keys = sorted(logical_keys)
    if len(sorted_keys) <= 1:
        return sorted_keys

    placeholders = ",".join("?" for _ in sorted_keys)
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"""
            SELECT logical_source_key, raw_id
            FROM raw_sessions
            WHERE logical_source_key IN ({placeholders})
            ORDER BY logical_source_key, acquired_at_ms DESC
            """,
            sorted_keys,
        ).fetchall()
    representative_raw_id: dict[str, str] = {}
    for logical_source_key, raw_id in rows:
        representative_raw_id.setdefault(str(logical_source_key), str(raw_id))

    parent_of: dict[str, str | None] = {}
    for key in sorted_keys:
        parent_key: str | None = None
        raw_id = representative_raw_id.get(key)
        if raw_id is not None:
            try:
                sessions, _payload_bytes = spill.for_raw(archive, raw_id)
            except Exception:
                # Lineage ordering is a scheduling optimization only -- any
                # failure here degrades to "treat as unresolved", never to a
                # replay/adoption failure.
                sessions = []
            if sessions:
                session = sessions[0]
                parent_provider_id = session.parent_session_provider_id
                if parent_provider_id:
                    parent_key = f"{session.source_name.value}:{parent_provider_id}"
        parent_of[key] = parent_key

    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for key in sorted_keys:
        parent_key = parent_of[key]
        if parent_key is not None and parent_key in logical_keys and parent_key != key:
            children.setdefault(parent_key, []).append(key)
        else:
            roots.append(key)

    ordered: list[str] = []
    seen: set[str] = set()

    def visit(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)
        for child in children.get(key, ()):
            visit(child)

    for key in roots:
        visit(key)
    # Cycles: every remaining member has a not-yet-visited parent inside the
    # set. Fall back to lexicographic order for the unresolved remainder --
    # ``visit`` still walks each one's children once reached, so nothing is
    # skipped or duplicated.
    for key in sorted_keys:
        if key not in seen:
            visit(key)
    return ordered


def backfill_historical_revision_evidence(
    archive_root: Path,
    *,
    active_index_path: Path | None = None,
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
    pipeline_decode: bool | None = None,
    deadline_check: Callable[[], None] | None = None,
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

    ``pipeline_decode`` (Lever A, parse ∥ apply) engages a
    :class:`_ReplaySpillPrefetcher` that decodes upcoming replay cohorts'
    parsed sessions on ONE background thread while the single writer applies
    the current cohort. ``None`` (the default) auto-resolves to
    ``parallel_threads_effective()``: pipelining only under a genuinely
    free-threaded interpreter, exactly like every other CPU-bound parse
    thread here; ``False`` forces the exact serial decode path; ``True``
    forces pipelining even under a GIL (tests only). All archive writes
    remain on the calling thread in unchanged order regardless of the value
    -- see the prefetcher's docstring for the equivalence argument and the
    buffered-memory bound.

    ``deadline_check`` (polylogue-uhgm, default ``None``) is called with no
    arguments at the top of every REPLAY-phase cohort iteration (both the
    byte-cohort and membership-cohort loops), i.e. between cohorts rather
    than only after this whole function returns. It is expected to raise
    :class:`RebuildDeadlineExceededError` to interrupt; ``None`` (every caller
    except the resumable offline rebuild pass) reproduces the exact
    unmodified, uninterruptible replay path.
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
    # polylogue-fpid: one background worker that builds the NEXT byte-proven
    # cohort's PreparedSessionRows (message/block row tuples -- pure CPU,
    # per-item hashing/JSON-encoding/enum lookups, no DB connection) while
    # THIS cohort's apply_raw_revision_replay() does its own preamble
    # (attachment preacquisition, blob flush, raw_revision_heads lookup) on
    # the calling thread. Only real parallelism under a free-threaded
    # interpreter (see ``parallel_threads_effective``'s docstring for why a
    # GIL build must not engage worker threads here either) -- a GIL build
    # gets ``prepare_pool=None`` and every write falls back to building rows
    # inline, byte-identical to before this change.
    prepare_pool = ThreadPoolExecutor(max_workers=1) if parallel_threads_effective() else None
    with (
        archive_context as archive,
        _ParsedSessionSpill(
            archive_root,
            index_path=active_index_path,
            max_cached_payload_bytes=spill_cache_bytes,
        ) as spill,
        prepare_pool if prepare_pool is not None else nullcontext(),
    ):
        census_started = time.perf_counter()
        if owned_inactive_generation is not None:
            census = _load_frozen_revision_evidence(
                archive,
                spill,
                selected_raw_ids=selected_raw_ids,
                max_payload_bytes=max_payload_bytes,
                ingest_workers=ingest_workers,
                prefetch_cache=prefetch_cache,
            )
        else:
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
        # ``_census_historical_revision_evidence`` records each receipt at the
        # point where it still holds that raw's parsed identities. Do not
        # reconstruct a second receipt from durable bindings here.
        archive.commit()
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
                commit_started = time.perf_counter()
                archive.commit()
                stage_timings["replay.commit"] = stage_timings.get("replay.commit", 0.0) + (
                    time.perf_counter() - commit_started
                )
                pending_replay_commits = 0

        replayed = 0
        byte_replayed_keys: set[str] = set()
        # Lever A (parse ∥ apply): decode upcoming cohorts' parsed sessions
        # off the writer thread while the single writer applies the current
        # cohort. Gated exactly like ``prepare_pool`` above -- GIL builds
        # keep the byte-identical serial decode path. The auto default also
        # requires enough cohorts to amortize the worker's setup (thread
        # spawn, two read connections, a plan scan over raw_sessions/
        # raw_session_memberships): the live raw-materialization path
        # (storage/repair.py) replays ONE authority component per call,
        # where a prefetcher could never get ahead of the writer anyway.
        effective_pipeline_decode = (
            pipeline_decode
            if pipeline_decode is not None
            else (
                parallel_threads_effective()
                and len(logical_keys) + len(membership_keys) >= _PIPELINE_DECODE_MIN_COHORTS
            )
        )
        # polylogue-5q2u: replay in lineage order (roots, then children after
        # their parent) instead of lexicographic order -- see
        # ``_lineage_aware_replay_order``'s docstring. Scheduling-only: the
        # SET of keys replayed and the plan/adoption outcome for each is
        # unaffected, only wall-clock and how often the deferred-tail path
        # (#2467) triggers. Both the pipeline-decode prefetcher and the
        # writer's own replay loop consume this SAME order so the
        # prefetcher's lookahead actually matches what the writer visits
        # next.
        ordered_logical_keys = _lineage_aware_replay_order(logical_keys, archive, spill, archive_root)
        decode_prefetcher: _ReplaySpillPrefetcher | None = None
        if effective_pipeline_decode:
            decode_prefetcher = _ReplaySpillPrefetcher(spill, archive_root=archive_root)
            spill.attach_prefetcher(decode_prefetcher)
            decode_prefetcher.start_phase(ordered_logical_keys, provisional_full_raw_ids)
        try:
            for logical_key in ordered_logical_keys:
                if deadline_check is not None:
                    deadline_check()
                if decode_prefetcher is not None:
                    decode_prefetcher.enter_key(logical_key)
                # polylogue-eqnv: the offline backfill/rebuild path is the one
                # where a stale pre-fix parser identity can split one physical
                # document's re-acquisitions across two logical_source_keys (see
                # ArchiveStore.classify_raw_revision_cohort_for_rebuild_repair's
                # docstring) -- use that entry point here, which always applies
                # the source_path cross-key guard. The live watcher
                # (sources/live/batch.py) uses
                # classify_raw_revision_cohort_for_live_watch instead: a
                # watched path can be legitimately, atomically replaced with a
                # different session's content, which must not be quarantined
                # as "divergent evidence".
                classify_started = time.perf_counter()
                plan = (
                    archive.classify_raw_revision_cohort_for_frozen_candidate(logical_key)
                    if owned_inactive_generation is not None
                    else archive.classify_raw_revision_cohort_for_rebuild_repair(
                        logical_key,
                        # Batched replay defers the classification's source.db
                        # authority updates into the same batch window as the replay
                        # writes (idempotent, re-derived on resume -- see
                        # classify_raw_revision_cohort_for_rebuild_repair's docstring).
                        manage_transaction=not replay_batched,
                    )
                )
                stage_timings["replay.classify_cohort"] = stage_timings.get("replay.classify_cohort", 0.0) + (
                    time.perf_counter() - classify_started
                )
                if not plan.accepted_raw_ids:
                    # Complete snapshots that are not a unique byte-prefix chain
                    # still carry semantic evidence. Move only that full-only
                    # cohort to membership governance and let parsed-content
                    # prefix rules decide it; append chains remain byte-governed.
                    convertible = archive.convertible_full_revision_raw_ids(logical_key)
                    if owned_inactive_generation is not None and convertible:
                        raise FrozenSourceRemediationRequiredError(
                            "inactive candidate found a full-revision cohort that still requires membership "
                            f"remediation in frozen source: {logical_key}"
                        )
                    for raw_id in convertible:
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
                            parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                            censused_at_ms=0,
                            detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                            retire_full_revision_governance=True,
                        )
                        # polylogue-eqnv: bucket by the identity the retirement
                        # reparse just recomputed (what actually lands in
                        # raw_session_memberships.logical_source_key inside
                        # replace_raw_membership_census), NOT the stale outer-
                        # loop ``logical_key`` this raw was originally censused
                        # under. Two same-document raws retired here from
                        # DIFFERENT stale keys (e.g. one carrying a since-fixed
                        # parser identity bug) re-derive the SAME fresh key on
                        # reparse; bucketing by the stale key would keep them in
                        # separate membership cohorts below and let each be
                        # accepted as an independent membership singleton --
                        # reproducing the exact fidelity-downgrade defect this
                        # retirement path exists to prevent, one layer down.
                        fresh_session = sessions[0]
                        fresh_key = f"{fresh_session.source_name.value}:{fresh_session.provider_session_id}"
                        membership_candidates.setdefault(fresh_key, set()).add(raw_id)
                        membership_keys.add(fresh_key)
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
                # polylogue-fpid: kick off row-tuple construction for the chain's
                # sole full-replace position (position 0 -- see apply_raw_
                # revision_replay's docstring) on the background pool now, so it
                # runs concurrently with the adoptability check below and this
                # cohort's own apply_raw_revision_replay() preamble. A GIL build
                # (prepare_pool is None) leaves prepared_by_raw_id empty and this
                # write falls back to building rows inline, unchanged.
                prepared_by_raw_id: dict[str, PreparedSessionRows | Future[PreparedSessionRows]] = {}
                if prepare_pool is not None:
                    position0_raw_id = plan.accepted_raw_ids[0]
                    prepared_by_raw_id[position0_raw_id] = prepare_pool.submit(
                        prepare_session_rows, parsed_by_raw_id[position0_raw_id]
                    )
                accepted_sessions = [parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids]
                adoptable_started = time.perf_counter()
                adoptable = archive.raw_revision_replay_adoptable(accepted_sessions)
                stage_timings["replay.adoptable_check"] = stage_timings.get("replay.adoptable_check", 0.0) + (
                    time.perf_counter() - adoptable_started
                )
                if not adoptable:
                    archive.defer_raw_revision_adoption(
                        plan.logical_source_key, plan.accepted_raw_ids, accepted_sessions
                    )
                    provisional_raw_ids = provisional_full_raw_ids.get(logical_key, set())
                    plan_raw_ids = {application.raw_id for application in plan.applications}
                    if plan_raw_ids and plan_raw_ids <= provisional_raw_ids:
                        archive.release_provisional_full_revisions(sorted(plan_raw_ids))
                    adoption_deferred += len(plan.accepted_raw_ids)
                    # The submitted future is discarded unused here (this cohort
                    # was deferred, not written) -- harmless: its result is a
                    # pure, side-effect-free row-tuple build with no DB
                    # connection, so an unconsumed Future just gets garbage
                    # collected once the pool shuts down.
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
                        prepared_by_raw_id=prepared_by_raw_id or None,
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

            if decode_prefetcher is not None:
                # Second phase: the final membership key set (including
                # retirement-added keys) is only known now. Retirement
                # bindings may still sit uncommitted in the open replay
                # batch, so the in-memory candidates map rides along.
                decode_prefetcher.start_phase(
                    [key for key in sorted(membership_keys) if key not in byte_replayed_keys],
                    membership_candidates,
                )
            for logical_key in sorted(membership_keys):
                if logical_key in byte_replayed_keys:
                    continue
                if deadline_check is not None:
                    deadline_check()
                if decode_prefetcher is not None:
                    decode_prefetcher.enter_key(logical_key)
                member_sessions: dict[str, ParsedSession] = {}
                revisions: list[MembershipRevision] = []
                projections = {}
                retained_bytes = 0
                candidates_started = time.perf_counter()
                candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
                candidate_raw_ids.update(membership_candidates.get(logical_key, ()))
                stage_timings["membership.candidates"] = stage_timings.get("membership.candidates", 0.0) + (
                    time.perf_counter() - candidates_started
                )
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
                        projection_started = time.perf_counter()
                        projection = session_revision_projection(session)
                        stage_timings["membership.project"] = stage_timings.get("membership.project", 0.0) + (
                            time.perf_counter() - projection_started
                        )
                        member_sessions[raw_id] = session
                        projections[raw_id] = projection
                        revisions.append(
                            MembershipRevision(
                                raw_id,
                                projection,
                                session.updated_at,
                                browser_snapshot_fidelity=_browser_snapshot_fidelity(session.ingest_flags),
                                provider_message_ids=frozenset(
                                    message.provider_message_id for message in session.messages
                                ),
                                provider_attachment_ids=frozenset(
                                    attachment.provider_attachment_id for attachment in session.attachments
                                ),
                            )
                        )
                        retained_bytes += payload_bytes
                if retention_observer is not None:
                    retention_observer(len(member_sessions), retained_bytes)
                membership_classify_started = time.perf_counter()
                # existing_accepted_raw_id is passed unconditionally (not
                # just when head_raw_id's own authority is 'quarantined',
                # unlike the narrower absorption-into-candidates condition
                # above) -- the presence-guarantee-fallback guard inside
                # classify_membership_revisions needs to know about a
                # chain-governed (non-quarantined) existing head too, to
                # correctly refuse ever retiring it, even though such a head
                # is deliberately never absorbed into the comparison cohort
                # itself.
                classification = classify_membership_revisions(revisions, existing_accepted_raw_id=head_raw_id)
                stage_timings["membership.classify"] = stage_timings.get("membership.classify", 0.0) + (
                    time.perf_counter() - membership_classify_started
                )
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
        finally:
            if decode_prefetcher is not None:
                stage_timings.update(decode_prefetcher.close())
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
        stage_timings_s=stage_timings,
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
    kind_token: str,
    native_id: str | None,
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

    ``kind_token``/``native_id`` (polylogue-6lyh1) mirror
    ``parse_retained_raw_sessions``'s own fallback-identity recovery exactly:
    an APPEND-kind raw's own record stream may carry no self-describing
    identity of its own (e.g. a Codex append delta has no ``session_meta``
    record), so the identity hint recorded at write time
    (``sources/live/batch.py``'s ``_append_payload_for_provider`` /
    ``write_raw_payload``'s ``native_id``) must be used as the parser's
    fallback_id instead of the bare filename stem -- otherwise a parallel
    dispatch path silently falls back to ``Path(source_path).stem`` and
    diverges from the sequential path whenever the source path's stem isn't
    the provider session id (polylogue-u19l fixed this for the sequential
    path only; every dispatcher of this worker must apply the identical
    fallback for parallel and sequential replay to stay byte-identical).
    Every caller resolves both values on its own calling thread (never
    inside a worker) for the same thread-affine-connection reason
    ``_parse_unique_retained_raws_via_threads`` avoids touching a shared
    ``ArchiveStore`` at all.

    Dispatched onto a ``ThreadPoolExecutor`` (real free-threading, see
    ``_parse_unique_retained_raws_via_threads``) and the daemon's own
    off-writer-hold pre-parse ``ThreadPoolExecutor``
    (``polylogue.daemon.parse_prefetch.DaemonParseStage``, polylogue-m6tp
    phase (a)), plus the pipelined replay prefetcher's reparse fallback
    (``_ReplaySpillPrefetcher._decode``) -- the function is identical every
    time; only the executor and the recreated ``ArchiveBlobPublisher``'s
    process/thread affinity differ. Public (not module-private) precisely so
    the daemon's warmer can import and dispatch it without duplicating this
    parse logic.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    provider = Provider(provider_token)
    kind = RawRevisionKind(kind_token)
    fallback_id_override = native_id if kind is RawRevisionKind.APPEND else None
    publisher = ArchiveBlobPublisher(Path(source_db_path_str), Path(blob_root_str))
    try:
        if provider is Provider.UNKNOWN:
            with publisher.open(blob_hash) as detection_payload:
                provider, _evidence = _detect_unknown_retained_provider(detection_payload, source_path)
            if is_stream_record_provider(source_path, str(provider)):
                with publisher.open(blob_hash) as stream_payload:
                    sessions = _parse_stream(
                        provider, stream_payload, source_path, fallback_id_override=fallback_id_override
                    )
                return raw_id, sessions, None
            payload = publisher.read_all(blob_hash)
            if provider is Provider.UNKNOWN:
                provider, _evidence = detect_provider_from_raw_bytes_evidence(
                    payload,
                    Path(source_path).name,
                    provider,
                )
            payload_path = None
            if provider is Provider.HERMES:
                candidate_path = publisher.blob_path(blob_hash)
                payload_path = candidate_path if candidate_path.exists() else None
            sessions = _parse_one(
                provider,
                payload,
                source_path,
                payload_path=payload_path,
                archive_root=Path(blob_root_str).parent,
                fallback_id_override=fallback_id_override,
            )
            return raw_id, sessions, None
        if is_stream:
            with publisher.open(blob_hash) as stream_payload:
                sessions = _parse_stream(
                    provider, stream_payload, source_path, fallback_id_override=fallback_id_override
                )
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
                fallback_id_override=fallback_id_override,
            )
        return raw_id, sessions, None
    except Exception as exc:
        return raw_id, None, str(exc)


#: Providers whose parsed session identity is derived purely from payload
#: bytes, never from ``source_path`` -- safe to dedup census parse ACROSS
#: source paths sharing a ``blob_hash`` (polylogue-869u). Excluded
#: deliberately: ``Provider.BEADS`` derives workspace-scoped native ids from
#: ``source_path`` (``sources/parsers/beads.py:_repository_root``);
#: ``Provider.ANTIGRAVITY``'s brain-metadata mode derives its
#: ``profile_root``/artifact path from ``source_path``
#: (``sources/dispatch.py``'s ``antigravity.parse_brain_metadata`` call);
#: ``Provider.HERMES``'s ATOF/ATIF/verification-evidence modes likewise
#: derive ``profile_root`` from ``source_path``. Those three keep the
#: conservative same-path-only dedup below. ``Provider.UNKNOWN`` (browser
#: capture / unclassified) is also excluded out of caution -- its identity
#: derivation is not centrally audited here.
_PATH_INDEPENDENT_PARSE_PROVIDERS: Final[frozenset[Provider]] = frozenset(
    {
        Provider.CHATGPT,
        Provider.CLAUDE_AI,
        Provider.CLAUDE_DESIGN,
        Provider.CLAUDE_CODE,
        Provider.CODEX,
        Provider.GEMINI,
        Provider.GEMINI_CLI,
        Provider.GROK,
        Provider.DRIVE,
    }
)


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
    exception. Rows sharing the same ``blob_hash`` for a
    ``_PATH_INDEPENDENT_PARSE_PROVIDERS`` provider are parsed exactly once
    and the outcome fanned out, regardless of ``source_path``: identical
    bytes decode deterministically identically for those providers, so
    re-parsing them per row (even under a DIFFERENT acquired path -- a
    common re-acquisition/re-export shape) is pure waste. For every other
    provider the dedup key still includes ``source_path`` (some parsers
    derive identity from the path, e.g. beads workspace ids -- see
    ``_PATH_INDEPENDENT_PARSE_PROVIDERS``'s docstring for the excluded
    providers), so cross-path duplicates for those stay deliberately NOT
    deduplicated. Live evidence (polylogue-869u, 2026-07-19): 87,177
    newest-revision raws / 52.1 GiB but only 85,066 distinct blob hashes /
    43.4 GiB -- the same bytes (e.g. one 442 MB codex rollout) recur under
    up to 8 different ``logical_source_key``s / source paths and, before
    this cross-path widening, paid a full parse each time. Per-row
    ``revision_kind`` is re-attached from each row's own descriptor.

    ``prefetch_cache`` (polylogue-m6tp phase (a)) is consulted BEFORE any of
    the above: a raw_id already popped from the cache is used directly and
    excluded from dedup/dispatch entirely, so it costs neither a parse nor a
    process/thread-pool round trip here. Every raw_id NOT found in the cache
    (including all of them, when ``prefetch_cache`` is ``None`` -- the
    default for every existing caller) is parsed exactly as before.

    polylogue-oab7: after the per-page dedup grouping above, each group's
    ``(provider, blob_hash, dedup_path)`` key is also checked against
    ``prefetch_cache``'s CONTENT cache (``get_content``/``put_content``,
    distinct from the raw_id-keyed ``pop`` used above). A hit there means
    this exact content was already parsed on an EARLIER call to this
    function against the SAME ``prefetch_cache`` instance -- e.g. an earlier
    *page* of the same archive-wide rebuild, not just an earlier row of this
    same page -- and is reused without a second parse. A miss falls through
    to the unchanged dispatch path and, once parsed, is admitted into the
    content cache so a LATER page can reuse it. ``prefetch_cache=None``
    (every caller that does not opt in) skips this lookup/store entirely and
    is byte-identical to today's behavior.
    """
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]] = {}
    for raw_id in raw_ids:
        provider, blob_hash, source_path, kind, size = archive.raw_revision_descriptor(raw_id)
        # polylogue-6lyh1: resolve the same APPEND fallback-identity hint the
        # sequential path (``parse_retained_raw_sessions``) recovers, on this
        # calling thread, so every parallel dispatch path below can apply the
        # identical fallback -- see ``census_parse_worker``'s docstring.
        native_id = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
        descriptors[raw_id] = (provider, blob_hash, source_path, kind, size, native_id)
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
    grouped: dict[ContentCacheKey, list[str]] = {}
    for raw_id in remaining_raw_ids:
        provider, blob_hash, source_path, _kind, _size, native_id = descriptors[raw_id]
        dedup_path = "" if provider in _PATH_INDEPENDENT_PARSE_PROVIDERS else source_path
        grouped.setdefault((provider, blob_hash, dedup_path, native_id), []).append(raw_id)

    content_hits: dict[ContentCacheKey, tuple[list[ParsedSession], int, RawRevisionKind]] = {}
    representatives: list[str] = []
    for key, members in grouped.items():
        content_hit = prefetch_cache.get_content(key) if prefetch_cache is not None else None
        if content_hit is not None:
            content_hits[key] = content_hit
        else:
            representatives.append(members[0])

    unique = _parse_unique_retained_raws(
        archive, representatives, descriptors=descriptors, ingest_workers=ingest_workers
    )

    if prefetch_cache is not None:
        for key, members in grouped.items():
            if key in content_hits:
                continue
            fresh_outcome = unique[members[0]]
            if isinstance(fresh_outcome, Exception):
                continue
            sessions, rep_size, rep_kind = fresh_outcome
            prefetch_cache.put_content(key, sessions, payload_bytes=rep_size, revision_kind=rep_kind)

    for key, members in grouped.items():
        content_outcome = content_hits.get(key)
        outcome: tuple[list[ParsedSession], int, RawRevisionKind] | Exception = (
            content_outcome if content_outcome is not None else unique[members[0]]
        )
        for raw_id in members:
            if isinstance(outcome, Exception):
                results[raw_id] = outcome
            else:
                sessions, _rep_size, _rep_kind = outcome
                _provider, _blob_hash, _source_path, kind, size, _native_id = descriptors[raw_id]
                results[raw_id] = (sessions, size, kind)
    _enrich_retained_parse_results(archive, descriptors=descriptors, results=results)
    return results


def _enrich_retained_parse_results(
    archive: ArchiveStore,
    *,
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]],
    results: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception],
) -> None:
    """Apply replay-safe provider assembly to decoded retained raws.

    Initial file ingest routes every parsed session through the provider
    assembly layer before hashing and writing it. Raw replay historically
    skipped that layer, so even deterministic fallbacks (for example Codex's
    first-human-message title) changed the session hash and made byte-proven
    source cohorts permanently non-adoptable.

    Replay may consume only frozen source evidence. It therefore uses the
    earliest persisted ``history_sidecars`` snapshot for an acquisition path,
    plus durable Codex title hook events, and otherwise supplies an explicit
    empty sidecar set. It never rediscovers mutable files beside the original
    source path. Providers whose enrichment was never captured remain
    conservatively unenriched and will still fail an exact hash comparison
    rather than being reconstructed from ambient filesystem state.
    """
    # Unit-level parser/dedupe probes deliberately pass tiny protocol fakes;
    # enrichment is an ArchiveStore production concern and is covered through
    # real source-tier replay fixtures below. Do not turn those pure decode
    # probes into accidental SQLite integration tests.
    if not isinstance(archive, ArchiveStore):
        return
    from polylogue.sources.assembly_codex import read_codex_thread_title_hook_events

    source_conn = archive._ensure_source_conn()
    codex_hook_titles = read_codex_thread_title_hook_events(source_conn)
    sidecars_by_path: dict[tuple[Origin, str], object] = {}
    for raw_id, outcome in tuple(results.items()):
        if isinstance(outcome, Exception):
            continue
        provider, _blob_hash, source_path, _descriptor_kind, _size, _native_id = descriptors[raw_id]
        sessions, payload_bytes, kind = outcome
        if sessions:
            provider = Provider.from_string(sessions[0].source_name)
        results[raw_id] = (
            _replay_safe_enrich_sessions(
                source_conn,
                provider=provider,
                source_path=source_path,
                sessions=sessions,
                sidecars_by_path=sidecars_by_path,
                codex_hook_titles=codex_hook_titles,
            ),
            payload_bytes,
            kind,
        )


def _replay_safe_enrich_sessions(
    source_conn: sqlite3.Connection,
    *,
    provider: Provider,
    source_path: str,
    sessions: list[ParsedSession],
    sidecars_by_path: dict[tuple[Origin, str], object],
    codex_hook_titles: dict[str, str],
) -> list[ParsedSession]:
    """Enrich one retained parse from frozen source-tier sidecars only."""
    from polylogue.sources.assembly import SidecarData, get_assembly_spec
    from polylogue.storage.sqlite.archive_tiers.source_write import read_earliest_history_sidecar_for_path

    spec = get_assembly_spec(provider)
    if spec is None:
        return sessions
    origin = origin_from_provider(provider)
    sidecar_key = (origin, source_path)
    cached = sidecars_by_path.get(sidecar_key)
    if cached is None:
        persisted = read_earliest_history_sidecar_for_path(
            source_conn,
            origin=origin,
            source_path=source_path,
        )
        sidecar_data = cast("SidecarData", dict(persisted.payload) if persisted is not None else {})
        if provider is Provider.CODEX and codex_hook_titles:
            sidecar_data = cast("SidecarData", {**sidecar_data, "hook_event_titles": codex_hook_titles})
        sidecars_by_path[sidecar_key] = sidecar_data
    else:
        sidecar_data = cast("SidecarData", cached)
    return [spec.enrich_session(session, sidecar_data) for session in sessions]


def _parse_unique_retained_raws_via_threads(
    archive: ArchiveStore,
    raw_ids: list[str],
    *,
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]],
    ingest_workers: int,
) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
    """Thread-parallel parse over every raw, regardless of size.

    A plain ``ThreadPoolExecutor`` shares parsed ``ParsedSession`` object
    graphs by reference between threads, so neither cost that a process pool
    has to amortize applies here: no pickle-back of the return value (#3136
    measured 0.63x/net-loss above 256KiB) and no per-worker spawn+import tax
    (#3149's ~1.5-2s floor -- threads share the one already-imported
    interpreter, they never re-pay ``import polylogue``). Every raw in
    ``raw_ids`` therefore dispatches, with no payload-size ceiling and no
    aggregate floor; the heuristics that enforced those under the GIL are
    retired along with the process-pool path itself.

    Dispatches ``census_parse_worker`` deliberately -- NOT
    ``_parse_retained_raw(archive, raw_id)``
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
            provider, blob_hash, source_path, kind, _payload_size, native_id = descriptors[raw_id]
            future = pool.submit(
                census_parse_worker,
                raw_id,
                provider.value,
                blob_hash,
                source_path,
                is_stream_record_provider(source_path, str(provider)),
                blob_root_str,
                source_db_path_str,
                kind.value,
                native_id,
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
            _provider, _blob_hash, _source_path, kind, payload_size, _native_id = descriptors[raw_id]
            results[raw_id] = (sessions or [], payload_size, kind)
    return results


def _parse_unique_retained_raws(
    archive: ArchiveStore,
    raw_ids: list[str],
    *,
    descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]],
    ingest_workers: int,
) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
    """Parse already-deduplicated raws, optionally in parallel.

    Read-only blob->ParsedSession decode is authority-neutral and
    embarrassingly parallel; callers apply archive writes afterwards in a
    fixed deterministic order independent of completion order here, so
    parallel and sequential execution produce byte-identical archive state
    regardless of which raws take a parallel path versus the sequential
    path.

    Parallelism is a plain ``ThreadPoolExecutor`` over every raw, with no
    size partition and no amortization floor: parsed ``ParsedSession`` graphs
    are shared between threads by reference, so neither of the costs that
    used to require tuning applies.

    Parallelism requires a genuinely free-threaded interpreter. ``>=3.14`` in
    ``requires-python`` does NOT guarantee that -- a standard GIL 3.14 build
    satisfies it -- so this still probes, and parses sequentially when the GIL
    is enabled. That is a safety guard, not a tuning knob: polylogue-7mtf
    measured GIL-build parse threads giving no speedup (0.93x-0.96x) while
    inflating a concurrent SQLite writer thread's commit latency ~5000x
    (208ms against an ~0.04ms/5ms cadence), so threads must never engage under
    a GIL.

    The retired process-pool alternative existed solely to get parallelism
    under the GIL, and carried two measured heuristics to stay a net win
    there: a 256 KiB payload ceiling (pickling ``ParsedSession`` graphs back
    across the process boundary measured 0.63x, a net loss) and a 48 MiB
    aggregate floor (per-worker spawn+import ~1.5-2.0s). On the reference
    archive that ceiling routed 98.5% of all bytes to a single core, which is
    what made a full index rebuild a ~9-hour job on a 24-thread machine.
    """
    results: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception] = {}
    plan = resolve_revision_backfill_census_dispatch(
        ingest_workers=ingest_workers, record_count=len(raw_ids), free_threaded=parallel_threads_effective()
    )
    if plan.pool_kind is PoolKind.SEQUENTIAL:
        if ingest_workers > 1 and len(raw_ids) > 1:
            _LOGGER.warning(
                "parsing %d raws sequentially: this interpreter has the GIL enabled, and "
                "parse threads under a GIL starve the archive writer rather than speeding "
                "parse up. Run polylogue on a free-threaded build (3.14t) for parallel parse.",
                len(raw_ids),
            )
        for raw_id in raw_ids:
            try:
                results[raw_id] = _parse_retained_raw(archive, raw_id)
            except Exception as exc:
                results[raw_id] = exc
        return results

    return _parse_unique_retained_raws_via_threads(
        archive, raw_ids, descriptors=descriptors, ingest_workers=ingest_workers
    )


def parse_retained_raw_sessions(archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
    """Parse retained raw evidence without eagerly loading stream records.

    Raw-revision replay is shared by historical repair and the live full and
    append routes.  Keeping the provider-shape decision here prevents a
    seemingly harmless live replay helper from reintroducing ``read_all()``
    for Codex/Claude JSONL evidence.
    """
    provider, blob_hash, source_path, kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    # polylogue-u19l: an append-kind raw's own record stream may carry no
    # self-describing identity of its own (a Codex append delta has no
    # session_meta record) -- recover the identity hint recorded at write
    # time (``sources/live/batch.py``'s ``_append_payload_for_provider`` /
    # ``write_raw_payload``'s ``native_id``) and use it as the parser's
    # fallback_id instead of the bare filename stem. Historical rows
    # (written before this) have no recorded native_id and fall through to
    # the unchanged stem-based fallback -- their stored bytes still carry
    # the synthetic session_meta line that made this unnecessary for them.
    fallback_id_override = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
    if provider is Provider.UNKNOWN:
        # Source-only acquisition deliberately retains unknown ZIP members
        # without decoding them.  Recovery is the first lawful point to
        # inspect the durable bytes and resolve their parser, before deciding
        # whether their filename is a stream route.
        with archive.open_raw_revision_material(raw_id) as (_stream_provider, payload, _stream_path, _stream_kind):
            provider, _evidence = _detect_unknown_retained_provider(payload, source_path)
        if is_stream_record_provider(source_path, str(provider)):
            with archive.open_raw_revision_material(raw_id) as (_stream_provider, payload, stream_path, _stream_kind):
                return _parse_stream(provider, payload, stream_path, fallback_id_override=fallback_id_override)
        _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
        if provider is Provider.UNKNOWN:
            provider, _evidence = detect_provider_from_raw_bytes_evidence(
                eager_payload,
                Path(source_path).name,
                provider,
            )
        payload_path = archive.blob_path_for_hash(blob_hash) if provider is Provider.HERMES else None
        return _parse_one(
            provider,
            eager_payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive.archive_root,
            fallback_id_override=fallback_id_override,
        )
    if is_stream_record_provider(source_path, str(provider)):
        with archive.open_raw_revision_material(raw_id) as (stream_provider, payload, stream_path, _stream_kind):
            return _parse_stream(stream_provider, payload, stream_path, fallback_id_override=fallback_id_override)
    _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
    payload_path = archive.blob_path_for_hash(blob_hash) if provider is Provider.HERMES else None
    return _parse_one(
        provider,
        eager_payload,
        source_path,
        payload_path=payload_path,
        archive_root=archive.archive_root,
        fallback_id_override=fallback_id_override,
    )


def _replay_retained_codex_state_evidence(archive: ArchiveStore, raw_id: str) -> bool:
    """Apply a retained, in-scope Codex state snapshot without minting a session.

    Source-only acquisition snapshots named Codex databases before it can
    inspect their schema.  Once recovery owns the derived tier, only a
    recognized retained snapshot may become thread evidence.  The parser
    reads the immutable blob path, never the original mutable state DB.
    """
    provider, blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    if provider is not Provider.CODEX:
        return False
    state_path = archive.blob_path_for_hash(blob_hash)
    if state_path is None:
        return False
    state_kind = codex_state.classify_codex_sqlite_path(state_path, immutable=True)
    if state_kind not in codex_state.IN_SCOPE_KINDS:
        return False
    if state_kind == "thread_state":
        write_codex_thread_state_evidence(
            archive,
            codex_state.parse_codex_state_db(state_path, immutable=True),
            source_path=source_path,
            acquired_at_ms=archive.raw_revision_observed_at_ms(raw_id),
        )
    archive.replace_raw_membership_census(
        raw_id,
        [],
        parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
        censused_at_ms=0,
        detail="retained Codex state evidence applied",
        retire_full_revision_governance=True,
    )
    archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CODEX)
    return True


#: Lever-A prefetch-buffer budget clamp (estimated tree bytes) -- same
#: adaptive formula as ``_ParsedSessionSpill``'s hot decoded cache (physical
#: RAM / 16 within these bounds), but deliberately its own pair of constants:
#: tests shrink the spill's class constants to force the sqlite/reparse
#: for_raw fallbacks WITHOUT also starving the prefetcher whose job is to
#: hide exactly those fallbacks.
_PREFETCH_BUFFER_MIN_TREE_BYTES: Final[int] = 256 * 1024 * 1024
_PREFETCH_BUFFER_MAX_TREE_BYTES: Final[int] = 2 * 1024 * 1024 * 1024

#: Minimum replay cohort count for the AUTO ``pipeline_decode`` default to
#: engage. Below this, the worker's fixed setup cost (thread spawn, two read
#: connections, a raw_sessions/raw_session_memberships plan scan) cannot be
#: repaid -- most notably the live raw-materialization path
#: (``storage/repair.py``), which replays exactly one authority component
#: per ``backfill_historical_revision_evidence`` call. Explicit
#: ``pipeline_decode=True``/``False`` bypasses this floor entirely.
_PIPELINE_DECODE_MIN_COHORTS: Final[int] = 8


class _ReplaySpillPrefetcher:
    """Decode upcoming replay cohorts' parsed sessions off the writer thread.

    polylogue Lever A (parse ∥ apply): on the real full-rebuild receipt,
    ``spill_load`` -- the replay loop's inline ``_ParsedSessionSpill.for_raw``
    decode (pickle.loads from the sqlite spill, or a full reparse from raw
    blob bytes once the bounded spill has evicted/refused an entry) -- was
    2,830s of strictly SERIAL work interleaved with the single SQLite
    writer's own apply work, i.e. ``parse_s + apply_s == total`` exactly.
    This class moves that decode onto ONE background thread that walks the
    same logical-key order the writer consumes, so the writer's
    ``for_raw()`` becomes a buffer pop for every raw the prefetcher reached
    first.

    Correctness model -- a miss is always safe, mirroring
    :class:`RawParsePrefetchCache`:

    * The prefetcher performs **zero archive writes**. Every index/source
      write stays on the calling (writer) thread in unchanged order, so a
      pipelined and a sequential run produce byte-identical archive state.
    * Decode sources are exactly the two ``for_raw()`` fallbacks, executed
      off-thread with thread-private handles: a second read connection to
      the disposable spill sqlite (same file, same rows, same
      ``pickle.loads``) and :func:`census_parse_worker` (the same pure
      blob->ParsedSession function the parallel census dispatches; its
      sequential-vs-parallel equivalence is already pinned by
      ``test_thread_parse_matches_sequential_archive_state``).
    * An empty, late, failed, or budget-refused prefetch degrades to the
      exact unmodified inline decode path -- never a different outcome. A
      decode error is deliberately NOT buffered so the writer's own inline
      decode raises the identical exception at the identical point.

    Memory bound: buffered decoded trees are accounted in ESTIMATED TREE
    BYTES (``estimate_parsed_tree_bytes``, same currency as the spill's hot
    cache) against an adaptive budget of physical RAM / 16 clamped to
    [256 MiB, 2 GiB] (identical formula to
    ``_ParsedSessionSpill._decoded_budget``). The worker blocks before
    decoding the next item whenever the buffer is at budget, so peak
    residency is ``budget + one in-flight tree``; items whose raw payload
    alone exceeds ``budget // 4`` are never prefetched at all (they decode
    inline, exactly as today), which caps the in-flight tree. Entries the
    writer skipped (deferred cohorts, retirement) are dropped -- and their
    budget released -- when the writer enters the next logical key.

    Threading contract: only engaged on a genuinely free-threaded
    interpreter (``parallel_threads_effective``), matching every other
    CPU-bound parse thread in this module -- polylogue-7mtf measured GIL
    parse threads starving the writer instead of helping. The worker reads
    ``_ParsedSessionSpill._decoded``/``_whales`` dicts concurrently with
    writer mutation (per-op-atomic dict access; a stale read only causes a
    harmless duplicate decode) and opens its own ``mode=ro`` source.db
    connection (WAL -- snapshot reads never block the writer) plus its own
    spill-file connection (``busy_timeout`` bridges the journal-less spill's
    short exclusive write windows).
    """

    def __init__(
        self,
        spill: _ParsedSessionSpill,
        *,
        archive_root: Path,
        max_buffered_tree_bytes: int | None = None,
    ) -> None:
        self._spill = spill
        self._archive_root = archive_root
        self._source_db_path = archive_root / "source.db"
        self._blob_root = archive_root / "blob"
        if max_buffered_tree_bytes is not None:
            self._budget = max_buffered_tree_bytes
        else:
            physical = effective_physical_memory_bytes() or 0
            floor = _PREFETCH_BUFFER_MIN_TREE_BYTES
            ceiling = _PREFETCH_BUFFER_MAX_TREE_BYTES
            self._budget = max(floor, min(ceiling, physical // 16)) if physical else floor
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        #: raw_id -> (sessions, payload_bytes, tree_bytes, from_reparse, seq)
        self._buffer: dict[str, tuple[list[ParsedSession], int, int, bool, int]] = {}
        self._buffered_tree_bytes = 0
        self._key_start_seq: dict[str, int] = {}
        self._writer_floor_seq = 0
        self._generation = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        # Stats are worker/writer-shared simple counters; merged into the
        # caller's stage-timings dict by ``close()`` on the writer thread.
        # ``hits``/``reparse_hits`` count worker-side decodes (production,
        # not necessarily consumption -- an entry the writer overtook is
        # decoded but dropped); ``consumed`` counts writer-side pops that
        # actually served a replay ``for_raw``, which is the number that
        # proves the pipeline carried real work off the writer thread.
        self.hits = 0
        self.reparse_hits = 0
        self.consumed = 0
        self.decode_seconds = 0.0

    def start_phase(self, ordered_keys: Sequence[str], extra_members: dict[str, set[str]]) -> None:
        """Begin prefetching one replay phase's cohorts, in writer order.

        Called on the writer thread. ``ordered_keys`` is the exact iteration
        order of the writer's upcoming loop; ``extra_members`` supplements
        the durable key->raw_ids mapping with in-memory census knowledge
        (``provisional_full_raw_ids`` / ``membership_candidates``) so raws
        whose durable binding sits in an uncommitted batch window are still
        reachable. Any previous phase's worker is stopped and its unconsumed
        buffer dropped first.
        """
        previous = self._thread
        with self._wakeup:
            self._generation += 1
            generation = self._generation
            self._drop_buffer_locked()
            self._key_start_seq = {}
            self._writer_floor_seq = 0
            self._wakeup.notify_all()
        if previous is not None:
            previous.join()
        if self._closed:
            return
        keys = tuple(ordered_keys)
        members_snapshot = {key: frozenset(raw_ids) for key, raw_ids in extra_members.items()}
        worker = threading.Thread(
            target=self._run,
            args=(generation, keys, members_snapshot),
            name="replay-spill-prefetch",
            daemon=True,
        )
        self._thread = worker
        worker.start()

    def pop(self, raw_id: str) -> tuple[list[ParsedSession], int, bool] | None:
        """Consume one prefetched decode, releasing its budget share."""
        with self._wakeup:
            entry = self._buffer.pop(raw_id, None)
            if entry is None:
                return None
            sessions, payload_bytes, tree_bytes, from_reparse, _seq = entry
            self._buffered_tree_bytes -= tree_bytes
            self.consumed += 1
            self._wakeup.notify_all()
            return sessions, payload_bytes, from_reparse

    def enter_key(self, logical_key: str) -> None:
        """Writer progress signal: drop buffered entries from earlier keys."""
        with self._wakeup:
            floor = self._key_start_seq.get(logical_key)
            if floor is None:
                return
            self._writer_floor_seq = max(self._writer_floor_seq, floor)
            stale = [raw_id for raw_id, entry in self._buffer.items() if entry[4] < floor]
            for raw_id in stale:
                _sessions, _payload, tree_bytes, _from_reparse, _seq = self._buffer.pop(raw_id)
                self._buffered_tree_bytes -= tree_bytes
            if stale:
                self._wakeup.notify_all()

    def close(self) -> dict[str, float]:
        """Stop the worker, drop the buffer, and return merge-able stats."""
        with self._wakeup:
            self._closed = True
            self._generation += 1
            self._drop_buffer_locked()
            self._wakeup.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        stats: dict[str, float] = {}
        if self.hits or self.reparse_hits or self.decode_seconds:
            stats["spill_prefetch.hits"] = float(self.hits)
            stats["spill_prefetch.reparse_hits"] = float(self.reparse_hits)
            stats["spill_prefetch.consumed"] = float(self.consumed)
            stats["spill_prefetch.decode_concurrent"] = self.decode_seconds
        return stats

    def _drop_buffer_locked(self) -> None:
        self._buffer.clear()
        self._buffered_tree_bytes = 0

    def _run(self, generation: int, keys: tuple[str, ...], extra_members: dict[str, frozenset[str]]) -> None:
        try:
            self._run_inner(generation, keys, extra_members)
        except Exception:
            # A prefetch failure must never take down the replay: the writer
            # keeps decoding inline, identical to prefetching never having
            # been enabled.
            _LOGGER.warning("replay spill prefetch worker failed; falling back to inline decode", exc_info=True)

    def _run_inner(self, generation: int, keys: tuple[str, ...], extra_members: dict[str, frozenset[str]]) -> None:
        if not keys:
            return
        # NOTE: ``with sqlite3.connect(...)`` would only manage a
        # transaction, not the connection lifetime -- close explicitly.
        source_conn = sqlite3.connect(f"file:{self._source_db_path}?mode=ro", uri=True, timeout=30.0)
        spill_conn: sqlite3.Connection | None = None
        try:
            plan, descriptors = self._build_plan(source_conn, keys, extra_members)
            if not plan:
                return
            from polylogue.sources.assembly_codex import read_codex_thread_title_hook_events

            codex_hook_titles = read_codex_thread_title_hook_events(source_conn)
            sidecars_by_path: dict[tuple[Origin, str], object] = {}
            spill_conn = sqlite3.connect(self._spill.path, timeout=30.0)
            spill_conn.execute("PRAGMA busy_timeout = 30000")
            for seq, raw_id in plan:
                if self._wait_for_budget(generation, seq) is False:
                    return
                with self._lock:
                    if seq < self._writer_floor_seq or raw_id in self._buffer:
                        # Already passed by the writer (decoding it now would
                        # be pure waste) or already buffered.
                        continue
                if raw_id in self._spill._decoded or raw_id in self._spill._whales:
                    continue
                decoded = self._decode(
                    spill_conn,
                    source_conn,
                    raw_id,
                    descriptors,
                    sidecars_by_path=sidecars_by_path,
                    codex_hook_titles=codex_hook_titles,
                )
                if decoded is None:
                    continue
                sessions, payload_bytes, from_reparse = decoded
                tree_bytes = estimate_parsed_tree_bytes(sessions)
                with self._wakeup:
                    if self._generation != generation or self._closed:
                        return
                    if seq < self._writer_floor_seq:
                        # The writer already moved past this key while we
                        # were decoding; buffering it would only pin budget.
                        continue
                    self._buffer[raw_id] = (sessions, payload_bytes, tree_bytes, from_reparse, seq)
                    self._buffered_tree_bytes += tree_bytes
        finally:
            if spill_conn is not None:
                spill_conn.close()
            source_conn.close()

    def _wait_for_budget(self, generation: int, seq: int) -> bool:
        """Block until buffer headroom exists; False means phase over."""
        with self._wakeup:
            while True:
                if self._generation != generation or self._closed:
                    return False
                if not self._buffer or self._buffered_tree_bytes < self._budget:
                    return True
                if seq < self._writer_floor_seq:
                    # Writer overtook this position while we were parked;
                    # skip ahead rather than decode already-passed work.
                    return True
                self._wakeup.wait(timeout=1.0)

    def _build_plan(
        self,
        source_conn: sqlite3.Connection,
        keys: tuple[str, ...],
        extra_members: dict[str, frozenset[str]],
    ) -> tuple[list[tuple[int, str]], dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]]]:
        """Resolve (seq, raw_id) decode order plus per-raw parse descriptors.

        Durable membership comes from source.db on this thread's own
        read-only connection (both mappings are committed before replay
        begins; retirement-phase additions living in an open batch window
        are covered by ``extra_members`` instead). Descriptors mirror
        ``raw_revision_descriptor``'s SELECT exactly (plus ``native_id``,
        polylogue-6lyh1), minus the writer-side connection affinity.
        """
        wanted = set(keys)
        members: dict[str, list[str]] = {key: [] for key in keys}
        for table in ("raw_sessions", "raw_session_memberships"):
            for row in source_conn.execute(
                f"SELECT logical_source_key, raw_id FROM {table} WHERE logical_source_key IS NOT NULL"
            ):
                key = str(row[0])
                if key in wanted:
                    members[key].append(str(row[1]))
        for key, extra in extra_members.items():
            if key in wanted:
                members[key].extend(extra)
        plan: list[tuple[int, str]] = []
        key_start_seq: dict[str, int] = {}
        seen: set[str] = set()
        seq = 0
        for key in keys:
            key_start_seq[key] = seq
            for raw_id in sorted(set(members[key])):
                if raw_id in seen:
                    continue
                seen.add(raw_id)
                plan.append((seq, raw_id))
                seq += 1
            # A key with no members still advances nothing; its start seq
            # equals the next key's, which is exactly what enter_key needs.
        with self._lock:
            self._key_start_seq = key_start_seq
        descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]] = {}
        planned = [raw_id for _seq, raw_id in plan]
        for start in range(0, len(planned), 500):
            chunk = planned[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in source_conn.execute(
                "SELECT raw_id, origin, capture_mode, lower(hex(blob_hash)), source_path, revision_kind, "
                "blob_size, native_id "
                f"FROM raw_sessions WHERE raw_id IN ({placeholders})",
                chunk,
            ):
                provider = provider_from_origin(Origin.from_string(str(row[1])), family_hint=row[2])
                kind = RawRevisionKind(str(row[5]))
                # polylogue-6lyh1: mirror ``census_parse_worker``'s fallback-id
                # contract exactly -- only an APPEND raw's recovered native_id
                # is ever used as a fallback identity; see that function's
                # docstring for why this must match the sequential path.
                raw_native_id_value = row[7]
                native_id = (
                    str(raw_native_id_value)
                    if kind is RawRevisionKind.APPEND
                    and isinstance(raw_native_id_value, str)
                    and raw_native_id_value.strip()
                    else None
                )
                descriptors[str(row[0])] = (provider, str(row[3]), str(row[4]), kind, int(row[6]), native_id)
        return plan, descriptors

    def _decode(
        self,
        spill_conn: sqlite3.Connection,
        source_conn: sqlite3.Connection,
        raw_id: str,
        descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int, str | None]],
        *,
        sidecars_by_path: dict[tuple[Origin, str], object],
        codex_hook_titles: dict[str, str],
    ) -> tuple[list[ParsedSession], int, bool] | None:
        started = time.perf_counter()
        rows = spill_conn.execute(
            "SELECT parsed, payload_bytes FROM parsed_sessions WHERE raw_id = ? ORDER BY logical_key", (raw_id,)
        ).fetchall()
        if rows:
            sessions = [cast(ParsedSession, pickle.loads(bytes(row[0]))) for row in rows]
            self.hits += 1
            self.decode_seconds += time.perf_counter() - started
            return sessions, int(rows[0][1]), False
        descriptor = descriptors.get(raw_id)
        if descriptor is None:
            return None
        provider, blob_hash, source_path, kind, payload_bytes, native_id = descriptor
        if payload_bytes > self._budget // 4:
            # Oversized decode: leave it to the writer's inline path so one
            # in-flight tree can never balloon far past the buffer budget.
            return None
        _same_raw_id, sessions_or_none, error = census_parse_worker(
            raw_id,
            provider.value,
            blob_hash,
            source_path,
            is_stream_record_provider(source_path, str(provider)),
            str(self._blob_root),
            str(self._source_db_path),
            kind.value,
            native_id,
        )
        if error is not None or sessions_or_none is None:
            # Do not buffer failures: the writer's inline decode raises the
            # identical error at the identical point in the identical order.
            return None
        sessions_or_none = _replay_safe_enrich_sessions(
            source_conn,
            provider=provider,
            source_path=source_path,
            sessions=sessions_or_none,
            sidecars_by_path=sidecars_by_path,
            codex_hook_titles=codex_hook_titles,
        )
        self.reparse_hits += 1
        self.decode_seconds += time.perf_counter() - started
        return sessions_or_none, payload_bytes, True


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
    #: Decoded-session RAM layer budget, accounted in ESTIMATED TREE BYTES
    #: (polylogue-xb4i estimator) rather than raw payload bytes -- parsed
    #: trees inflate payload 2-14x, so a payload-denominated budget either
    #: wastes RAM headroom on text-dense sessions or overshoots on
    #: structure-dense ones. Adaptive: physical RAM / 16, clamped to
    #: [256 MiB, 2 GiB]; falls back to the floor when RAM is unknown.
    _DECODED_CACHE_MIN_TREE_BYTES: Final[int] = 256 * 1024 * 1024
    _DECODED_CACHE_MAX_TREE_BYTES: Final[int] = 2 * 1024 * 1024 * 1024

    #: Whale-residency ceiling (polylogue-odm1): a single parsed tree that
    #: exceeds ``_decoded_budget`` above is, by construction, never retained
    #: by the hot multi-entry cache -- it previously fell all the way through
    #: to the sqlite spill (pickle.dumps at census, pickle.loads on every
    #: later ``for_raw()``), or, once the spill's own payload-byte budget was
    #: also exhausted, was silently dropped and had to be REPARSED FROM RAW
    #: BYTES on every subsequent ``for_raw()`` call -- the dominant cost on
    #: whale-bearing rebuild pages (v42 walk receipts: spill_load = 41% of a
    #: 1440.2s page). A whale is a transient, effectively single-occupant
    #: resident (the hot cache already handles the "many small/medium trees"
    #: case), so it can safely claim a much larger fraction of budgeted RAM
    #: than the multi-entry hot pool without changing the pool's own sizing
    #: philosophy: same ``effective_physical_memory_bytes()`` input, a wider
    #: divisor (physical RAM / 4 instead of / 16) and a proportionally wider
    #: absolute ceiling (8 GiB instead of 2 GiB), floored at whatever
    #: ``_decoded_budget`` resolved to so the whale tier is never narrower
    #: than the hot tier it complements. Bypassing the sqlite round trip
    #: entirely (no pickle.dumps, no pickle.loads) is safe: the durable raw
    #: bytes remain the replay fallback (``_parse_retained_raw``) if this
    #: process dies before ``for_raw()`` is called, so nothing is lost, only
    #: possibly reparsed once more on restart -- identical to today's
    #: existing crash-recovery contract for the sqlite-backed spill.
    _WHALE_CACHE_MAX_TREE_BYTES: Final[int] = 8 * 1024 * 1024 * 1024

    def __init__(
        self,
        archive_root: Path,
        *,
        index_path: Path | None = None,
        max_cached_payload_bytes: int | None,
    ) -> None:
        # Place the spill beside the RESOLVED index tier, not the archive
        # root: on deployments where the .db files are symlinks (e.g. root
        # SSD config dir -> NVMe data disk), a spill in archive_root would
        # put census churn on the wear-limited disk the symlinks exist to
        # protect.
        resolved_index_path = index_path or ArchiveLocation.resolve(archive_root).active_index_path
        spill_dir = resolved_index_path.resolve().parent if resolved_index_path.exists() else archive_root
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
        # Lever A: a ``_ReplaySpillPrefetcher`` reads this same file on its
        # own connection while replay may still write here (for_raw fallback
        # adds, whale-eviction spills). With no journal, both sides rely on
        # plain file locking; a busy_timeout on each connection bridges the
        # other side's short exclusive windows instead of failing instantly.
        self.conn.execute("PRAGMA busy_timeout = 30000")
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
        self._decoded: dict[str, tuple[list[ParsedSession], int, int]] = {}
        self._decoded_tree_bytes = 0
        physical = effective_physical_memory_bytes() or 0
        self._decoded_budget = (
            max(self._DECODED_CACHE_MIN_TREE_BYTES, min(self._DECODED_CACHE_MAX_TREE_BYTES, physical // 16))
            if physical
            else self._DECODED_CACHE_MIN_TREE_BYTES
        )
        # Whale-residency tier (polylogue-odm1): a bounded, FIFO-evicted
        # sibling of ``_decoded`` for parsed trees too large for the hot
        # cache's own budget. See ``_WHALE_CACHE_MAX_TREE_BYTES`` for the
        # sizing rationale. Its accounting is fully independent of
        # ``_decoded_tree_bytes`` -- the two tiers never compete for the same
        # budgeted bytes, so peak RAM is bounded by the sum of both ceilings,
        # not by either alone.
        self._whales: dict[str, tuple[list[ParsedSession], int, int]] = {}
        self._whale_tree_bytes = 0
        self._whale_budget = (
            max(self._decoded_budget, min(self._WHALE_CACHE_MAX_TREE_BYTES, physical // 4))
            if physical
            else self._decoded_budget
        )
        #: Optional Lever-A decode prefetcher (attached by
        #: ``backfill_historical_revision_evidence`` when pipelined decode is
        #: engaged). ``for_raw`` consults it AFTER the free RAM tiers and
        #: BEFORE the sqlite/reparse fallbacks it exists to hide.
        self._prefetcher: _ReplaySpillPrefetcher | None = None
        self._replay_sidecars_by_path: dict[tuple[Origin, str], object] = {}
        self._codex_hook_titles: dict[str, str] | None = None

    def attach_prefetcher(self, prefetcher: _ReplaySpillPrefetcher) -> None:
        self._prefetcher = prefetcher

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
        tree_bytes = estimate_parsed_tree_bytes(sessions)
        if tree_bytes > self._decoded_budget and self._retain_whale(
            raw_id, sessions, payload_bytes=payload_bytes, tree_bytes=tree_bytes
        ):
            # Size-partitioned spill (polylogue-odm1): a tree too big for the
            # hot cache but within the whale ceiling bypasses the sqlite
            # spill entirely -- no pickle.dumps now, no pickle.loads later.
            # Correctness fallback: ``_retain_whale`` returns False (and
            # execution falls through to the unchanged sqlite path below)
            # whenever holding it resident would exceed the whale budget, so
            # a session too large for EITHER tier still gets spilled exactly
            # as before.
            return
        if self._spill_to_sqlite(raw_id, sessions, payload_bytes=payload_bytes):
            self._retain_decoded(raw_id, sessions, payload_bytes=payload_bytes, tree_bytes=tree_bytes)

    def _spill_to_sqlite(self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int) -> bool:
        """Pickle ``sessions`` into the sqlite spill, subject to the caller's
        overall payload-byte budget. Returns whether the write happened.
        """
        if self.max_cached_payload_bytes is None or payload_bytes > self.max_cached_payload_bytes:
            return False
        if self.cached_payload_bytes + payload_bytes > self.max_cached_payload_bytes:
            return False
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
        return True

    def _retain_decoded(
        self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int, tree_bytes: int
    ) -> None:
        if tree_bytes > self._decoded_budget:
            return
        while self._decoded and self._decoded_tree_bytes + tree_bytes > self._decoded_budget:
            oldest_raw = next(iter(self._decoded))
            _sessions, _payload, evicted_tree = self._decoded.pop(oldest_raw)
            self._decoded_tree_bytes -= evicted_tree
        self._decoded[raw_id] = (sessions, payload_bytes, tree_bytes)
        self._decoded_tree_bytes += tree_bytes

    def _retain_whale(self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int, tree_bytes: int) -> bool:
        """Hold an outsized parsed tree resident, bypassing the sqlite spill.

        Returns ``False`` (retaining nothing) when ``tree_bytes`` alone
        exceeds ``_whale_budget`` -- the caller must then fall back to the
        existing spill-or-reparse path rather than blow the memory budget.
        Otherwise evicts oldest whale entries (FIFO, mirroring
        ``_retain_decoded``) until the new entry fits and retains it.

        A rare multi-whale page can still exceed the whale budget across
        several distinct oversized raws even though each individually fits.
        An entry evicted here to make room was never written to the sqlite
        spill (that write was deliberately skipped to avoid the pickle
        cost) -- so, unlike the hot cache's plain eviction, degrade it into
        the sqlite spill as a courtesy on the way out (best-effort; if the
        sqlite spill's own payload budget is also exhausted the entry is
        dropped exactly as today's pre-lever code would drop it, falling
        back to reparse-from-source on next access -- never worse than the
        unpatched baseline).
        """
        if tree_bytes > self._whale_budget:
            return False
        while self._whales and self._whale_tree_bytes + tree_bytes > self._whale_budget:
            oldest_raw = next(iter(self._whales))
            evicted_sessions, evicted_payload, evicted_tree = self._whales.pop(oldest_raw)
            self._whale_tree_bytes -= evicted_tree
            self._spill_to_sqlite(oldest_raw, evicted_sessions, payload_bytes=evicted_payload)
        self._whales[raw_id] = (sessions, payload_bytes, tree_bytes)
        self._whale_tree_bytes += tree_bytes
        return True

    def for_raw(self, archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int]:
        decoded = self._decoded.get(raw_id)
        if decoded is not None:
            return decoded[0], decoded[1]
        whale = self._whales.get(raw_id)
        if whale is not None:
            return whale[0], whale[1]
        if self._prefetcher is not None:
            prefetched = self._prefetcher.pop(raw_id)
            if prefetched is not None:
                sessions, payload_bytes, from_reparse = prefetched
                if from_reparse:
                    # Mirror the inline reparse fallback below exactly: a
                    # freshly reparsed raw is (re)admitted to the cache tiers
                    # so a later ``for_raw`` for the same raw hits RAM/sqlite
                    # instead of reparsing a third time. sqlite-sourced
                    # prefetch entries skip this -- their spill row is still
                    # present, identical to the inline sqlite-hit path.
                    self.add(raw_id, sessions, payload_bytes=payload_bytes)
                return sessions, payload_bytes
        rows = self.conn.execute(
            "SELECT parsed, payload_bytes FROM parsed_sessions WHERE raw_id = ? ORDER BY logical_key", (raw_id,)
        ).fetchall()
        if rows:
            return [pickle.loads(bytes(row[0])) for row in rows], int(rows[0][1])
        sessions, payload_bytes, _kind = _parse_retained_raw(archive, raw_id)
        if self._codex_hook_titles is None:
            from polylogue.sources.assembly_codex import read_codex_thread_title_hook_events

            self._codex_hook_titles = read_codex_thread_title_hook_events(archive._ensure_source_conn())
        provider, _blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
        sessions = _replay_safe_enrich_sessions(
            archive._ensure_source_conn(),
            provider=provider,
            source_path=source_path,
            sessions=sessions,
            sidecars_by_path=self._replay_sidecars_by_path,
            codex_hook_titles=self._codex_hook_titles,
        )
        self.add(raw_id, sessions, payload_bytes=payload_bytes)
        return sessions, payload_bytes


def _declared_non_session_artifact_classification(
    provider: Provider,
    source_path: str,
    *,
    sample: Sequence[object] = (),
) -> ArtifactClassification | None:
    """Classify a declared non-session raw revision before session admission.

    polylogue-b508: retained raw revisions include OriginSpec-declared fact
    artifacts (``agent-*.meta.json`` sidecars, ``workflows/*.json`` run
    snapshots, ``subagents/workflows/*/journal.jsonl``, ``adopt.json``
    manifests) admitted as raw authority by ``sources/live/batch.py`` even
    though their ``parse_policy`` is ``"fact"``, never ``"session"`` --
    intentional, so the retained bytes stay durable raw evidence. The live
    daemon's ingest path (``ingest_worker.py``/``batch.py``) already consults
    this same OriginSpec rule before parsing and refuses to session-parse
    these; this replay engine (used by ``polylogue ops reset --index`` /
    ``devtools`` rebuild-index) is a SEPARATE parse chokepoint that did not,
    and would silently recreate exactly the ``<agent>.meta`` phantom sessions
    that fix is meant to eliminate on every future rebuild. A positive JSONL
    session proof is the one deliberate exception, matching the live route:
    a source-only outage may retain bytes before it can inspect a path that
    normally carries fact evidence, and recovery must not make that filename
    permanently override later decoded session authority.

    polylogue-9ykn: a path-declared rule is only half of the live path's
    gate. ``pipeline/services/ingest_worker.py`` also runs every sampled
    JSONL payload through ``archive.artifact_taxonomy.classify_artifact`` --
    the richer, CONTENT-based classifier that catches a non-conversational
    record sitting under a watched Claude Code directory with no matching
    path rule at all (e.g. a third-party analysis index such as
    ``conversation_relationships.jsonl`` that happens to satisfy the loose
    per-record shape check). Without the same content check here, replay
    (this module) and rebuild (``maintenance/rebuild_index.py`` -> this
    module) silently resurrect exactly the phantom sessions the live gate
    now refuses, on every future rebuild -- the two "single chokepoints"
    disagreeing is the location-as-identity defect recurring at a second
    layer. ``sample`` -- the first up to 64 decoded records, mirroring the
    live path's own sample bound (``ingest_worker.py``'s
    ``_sample_jsonl_payload_with_detail(..., max_samples=64)``) -- is
    classified only when no path rule already decided the question; an
    empty ``sample`` (the default) preserves the original path-only
    behavior exactly, so every existing caller is unaffected until it opts
    in.
    """
    from polylogue.archive.artifact_taxonomy import classify_artifact

    rule = artifact_rule_for_path(provider, source_path)
    if rule is not None and rule.parse_policy != "session" and not sample:
        classification = classify_artifact([], provider=provider, source_path=source_path)
        if not classification.parse_as_session:
            return classification
    if not sample:
        return None
    from polylogue.core.json import JSONValue

    # ``sample`` records come from ``_iter_json_stream`` (this module's own
    # decode path, typed ``JsonValue`` -- ``core/query_identity.py``'s
    # structurally-equivalent but nominally distinct alias) rather than
    # ``classify_artifact``'s own ``core.json.JSONValue``; both describe the
    # same decoded-JSON shape ijson/json.loads ever produce, so the cast is
    # a type-identity bridge, not a real behavior narrowing.
    classification = classify_artifact(cast(list[JSONValue], list(sample)), provider=provider, source_path=source_path)
    # ``UNKNOWN`` means the taxonomy cannot decide, not that the payload is
    # a proved sidecar.  Codex append deltas intentionally omit the
    # session_meta header and become materializable only after the live
    # revision layer supplies its recorded native-id hint.  Treating that
    # undecided partial stream as an artifact skips revision binding and
    # silently loses its append frontier.  Keep the raw evidence on the
    # normal parser path, which either uses the hint or records a typed parse
    # failure; only a positive non-session cohort may bypass session parsing.
    if classification.kind is ArtifactKind.UNKNOWN:
        return None
    return classification if not classification.parse_as_session else None


def _detected_provider_for_empty_replay(
    archive: ArchiveStore,
    raw_id: str,
    *,
    stored_provider: Provider,
    source_path: str,
) -> Provider:
    """Resolve a provider before terminalizing an empty retained replay."""
    if stored_provider is not Provider.UNKNOWN:
        return stored_provider
    with archive.open_raw_revision_material(raw_id) as (_provider, payload, _path, _kind):
        provider, _evidence = _detect_unknown_retained_provider(payload, source_path)
    if provider is not Provider.UNKNOWN:
        return provider
    _provider, full_payload, _path, _kind = archive.raw_revision_material(raw_id)
    provider, _evidence = detect_provider_from_raw_bytes_evidence(
        full_payload,
        Path(source_path).name,
        stored_provider,
    )
    return provider


def _persist_terminal_non_session_artifact(
    archive: ArchiveStore,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    manage_transaction: bool,
) -> bool:
    """Record replay-confirmed source-only artifact authority once.

    Replay reaches this function only after the real parser has consumed the
    complete stream and produced no conversational session. The terminal
    receipt therefore follows that one authoritative parse result instead of
    reclassifying the raw through a second, weaker JSONL shape scan.
    """
    if provider is Provider.UNKNOWN:
        return False
    classification = _declared_non_session_artifact_classification(provider, source_path)
    if classification is None:
        return False
    origin = origin_from_provider(provider)
    observed_at_ms = archive.raw_revision_observed_at_ms(raw_id)
    upsert_raw_artifact(
        archive._ensure_source_conn(),
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=origin.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            artifact_kind=classification.cohort,
            classification_reason=classification.reason,
            parse_as_session=False,
            schema_eligible=classification.schema_eligible,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        ),
        manage_transaction=manage_transaction,
    )
    apply_source_raw_state_update(
        archive._ensure_source_conn(),
        raw_id,
        state=_raw_parse_success_state(provider),
        manage_transaction=manage_transaction,
    )
    return True


def _is_declared_non_session_artifact(
    provider: Provider,
    source_path: str,
    *,
    sample: Sequence[object] = (),
) -> bool:
    """Return whether this raw revision must not be session-parsed on replay."""
    return _declared_non_session_artifact_classification(provider, source_path, sample=sample) is not None


def _parse_one(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    # polylogue-9ykn: replay must apply the same positive-conversational-
    # evidence gate the live ingest paths apply, on top of the path/shape
    # gate above (``_is_declared_non_session_artifact``, polylogue-6mpy) --
    # a source can pass that gate (its shape IS a recognized Claude Code
    # JSONL file with no path rule) yet still parse to zero real messages
    # (e.g. a file containing only file-history-snapshot records).
    return require_positive_conversational_evidence(
        _parse_one_raw(
            provider,
            payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive_root,
            fallback_id_override=fallback_id_override,
        ),
        provider=provider,
        source_path=source_path,
    )


def _parse_one_raw(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    if provider is Provider.ANTIGRAVITY and Path(source_path).suffix.lower() == ".pb":
        trajectory_path = Path(source_path)
        root = trajectory_path.parent.parent
        if trajectory_path.parent.name != "conversations" or not trajectory_path.is_file():
            raise RuntimeError(
                f"Antigravity raw replay requires its original conversations/<cascade_id>.pb trajectory: {source_path}"
            )
        cascade_id = trajectory_path.stem
        sessions = list(antigravity.iter_language_server_exports(root, only_cascade_ids=frozenset({cascade_id})))
        if len(sessions) != 1 or sessions[0].provider_session_id != cascade_id:
            raise RuntimeError(
                "Antigravity raw replay did not reproduce exactly one requested trajectory "
                f"{cascade_id!r} from {source_path}"
            )
        return sessions
    source_name = Path(source_path).name
    fallback_id = fallback_id_override or Path(source_path).stem
    if provider is Provider.HERMES and looks_like_sqlite_bytes(payload):
        with _sqlite_payload_path(payload, payload_path, archive_root) as sqlite_path:
            if hermes_state.looks_like_state_db_path(sqlite_path, immutable=True):
                return hermes_state.parse_state_db(
                    sqlite_path,
                    fallback_id=fallback_id,
                    profile_root=Path(source_path).parent,
                    immutable=True,
                )
            if hermes_verification.looks_like_verification_evidence_db_path(sqlite_path, immutable=True):
                return hermes_verification.parse_verification_evidence_db(
                    sqlite_path,
                    fallback_id=fallback_id,
                    profile_root=Path(source_path).parent,
                    immutable=True,
                )
    rule = artifact_rule_for_path(provider, source_path)
    declared_path_session_evidence = False
    if rule is not None and rule.parse_policy != "session" and is_jsonl_source_path(source_path):
        from polylogue.archive.raw_payload.decode import jsonl_session_artifact

        declared_path_session_evidence = jsonl_session_artifact(payload, provider=provider) is not None
    if is_stream_record_provider(source_path, str(provider)):
        records = list(_iter_json_stream(BytesIO(payload), source_name))
        if not declared_path_session_evidence and _is_declared_non_session_artifact(
            provider, source_path, sample=records[:64]
        ):
            return []
        return parse_stream_payload(
            provider,
            records,
            fallback_id,
            source_path=source_path,
        )
    records = list(_iter_json_stream(BytesIO(payload), source_name))
    if not declared_path_session_evidence and _is_declared_non_session_artifact(
        provider, source_path, sample=records[:64]
    ):
        return []
    return parse_payload(
        provider,
        records,
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


def _parse_stream(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    # polylogue-9ykn: see ``_parse_one``'s comment -- the same positive-
    # conversational-evidence gate applies to the streaming replay path.
    return require_positive_conversational_evidence(
        _parse_stream_raw(provider, payload, source_path, fallback_id_override=fallback_id_override),
        provider=provider,
        source_path=source_path,
    )


def _parse_stream_raw(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = fallback_id_override or Path(source_path).stem
    stream = _iter_json_stream(payload, source_name)
    return parse_stream_payload(
        provider,
        stream,
        fallback_id,
        source_path=source_path,
    )


__all__ = [
    "RAW_AUTHORITY_PARSER_FINGERPRINT",
    "RawParsePrefetchCache",
    "RawRevisionReplayResourceBlockedError",
    "RebuildDeadlineExceededError",
    "RevisionBackfillResult",
    "RevisionCensusResult",
    "backfill_historical_revision_evidence",
    "census_historical_revision_evidence",
    "census_parse_worker",
    "record_resource_blocked_revision_census",
    "require_current_parser_source_census",
    "uncensused_historical_revision_raw_ids",
    "parse_retained_raw_sessions",
]
