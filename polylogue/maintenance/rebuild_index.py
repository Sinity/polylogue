"""Daemon-safe source-to-index rebuild execution.

The operation owns the write-side rebuild protocol; CLI and HTTP are adapters.
Callers must hold the daemon writer coordinator for an online rebuild.  The
offline guard rejects every other live-daemon caller, preserving break-glass
operation after the daemon has stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
from polylogue.storage.fts.fts_lifecycle import rebuild_command_trigram_index_sync, rebuild_fts_index_sync
from polylogue.storage.fts.sql import FTS_REBUILD_SQL, TRIGRAM_REBUILD_DELETE_ALL_SQL
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.delegation_facts import rebuild_all_delegation_facts_sync
from polylogue.storage.table_existence import table_exists

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache

_PLANNER_STATS_ANALYSIS_LIMIT = 1000
# A fresh generation begins with representative bootstrap statistics, but a
# replay eventually needs measured selectivities as it grows.  Refreshing
# after every resume page is needlessly expensive for small pages: ANALYZE
# must revisit a large set of indexes even when the generation changed by a
# fraction of a percent.  Keep the measured statistics within one bounded
# source page of the materialized corpus instead.
_PLANNER_STATS_REFRESH_RAW_INTERVAL = 1000
# Bulk-build replay keeps the FTS/trigram stores empty until final readiness,
# so analyzing their virtual-table backing stores does not improve any replay
# plan and can dominate a large archive's checkpoint.  These row stores are
# the tables used by the writer-hot replacement/link/action-pair queries.
_PLANNER_STATS_ANALYZE_STATEMENTS = (
    "ANALYZE sessions",
    "ANALYZE messages",
    "ANALYZE blocks",
    "ANALYZE session_links",
    "ANALYZE action_pairs",
)

logger = get_logger(__name__)


def _should_refresh_generation_planner_statistics(
    *,
    processed_before: int | None,
    processed_after: int,
) -> bool:
    """Return whether this replay pass crossed a measured-statistics boundary.

    Unbounded/one-shot rebuilds have no transaction cursor and always refresh
    after replay.  Resumable rebuilds retain their representative bootstrap
    statistics until the first measured tranche is large enough, then refresh
    whenever another bounded tranche has landed.  This preserves writer-hot
    query plans without making a 25 GiB generation pay an archive-wide
    ANALYZE for every small recovery page.
    """
    if processed_before is None:
        return True
    return processed_before // _PLANNER_STATS_REFRESH_RAW_INTERVAL < (
        processed_after // _PLANNER_STATS_REFRESH_RAW_INTERVAL
    )


def _clear_bulk_build_derived_stores(index_path: Path) -> None:
    """Idempotently empty ``messages_fts``/``blocks_command_trigram``.

    polylogue-v6i3: a fresh generation already starts with both derived
    stores empty by construction (bootstrap creates empty schema), so the
    very first pass of a brand-new bulk-build transaction never has
    meaningful work to do here -- but calling this unconditionally on every
    resumed pass (guarded by the transaction's own ``derived_stores_cleared``
    marker so it fires at most once per operation) converts "derived stores
    are empty throughout bulk-build replay" from an assumption inherited from
    generation creation into an explicit, verified invariant. This mirrors
    the manual pre-promote recovery script's clearing action
    (``/realm/tmp/trigram-restore-pre-promote.py``, the live incident this
    bead productizes), now automatic. Delete-all on an already-empty table is
    near-instant (28.7s was measured against a *populated* table during the
    live incident this bead responds to; an empty one is orders of magnitude
    faster), so this is cheap even when it turns out to be a no-op.
    """
    with contextlib.closing(sqlite3.connect(index_path, timeout=60)) as conn:
        conn.execute("PRAGMA busy_timeout = 60000")
        if table_exists(conn, "messages_fts"):
            conn.execute(FTS_REBUILD_SQL)
        if table_exists(conn, "blocks_command_trigram"):
            conn.execute(TRIGRAM_REBUILD_DELETE_ALL_SQL)
        conn.commit()


def _repopulate_bulk_build_derived_state(index_path: Path) -> dict[str, float]:
    """One archive-wide repopulate of every surface bulk-build replay skipped.

    polylogue-v6i3: ``write_parsed_session_to_archive``'s ``bulk_build`` mode
    leaves ``messages_fts``, ``blocks_command_trigram``, ``action_pairs``, and
    ``delegation_facts`` empty (or stale from a prior page) throughout replay
    -- this runs exactly once, right before readiness, to bring all four back
    into exact sync from ``blocks``/``messages``/``session_links`` in one bulk
    delete+insert per surface instead of the per-session maintenance replay
    skipped. Order matters: ``action_pairs`` must be repopulated before
    ``delegation_facts`` (the latter's ``delegation_facts_source`` view joins
    through the ``actions`` view, which reads ``action_pairs``).
    """
    timings_s: dict[str, float] = {}
    with contextlib.closing(sqlite3.connect(index_path, timeout=600)) as conn:
        conn.execute("PRAGMA busy_timeout = 600000")
        started_at = time.perf_counter()
        rebuild_fts_index_sync(conn, resume_from_empty_message_index=True)
        timings_s["fts"] = time.perf_counter() - started_at
        started_at = time.perf_counter()
        rebuild_command_trigram_index_sync(conn)
        timings_s["command_trigram"] = time.perf_counter() - started_at
        started_at = time.perf_counter()
        rebuild_all_action_pairs_sync(conn)
        timings_s["action_pairs"] = time.perf_counter() - started_at
        started_at = time.perf_counter()
        rebuild_all_delegation_facts_sync(conn)
        timings_s["delegation_facts"] = time.perf_counter() - started_at
        started_at = time.perf_counter()
        conn.commit()
        timings_s["commit"] = time.perf_counter() - started_at
    return timings_s


def _refresh_generation_planner_statistics(index_path: Path) -> None:
    """Replace bootstrap-seeded planner stats after a bounded replay tranche.

    A generation is bulk-written from empty, so the relative selectivities the
    planner needs (session-scoped indexes are narrow, type-scoped ones are not)
    drift fast as tables grow.  Bounded periodic ANALYZE of only writer-hot row
    stores keeps per-session plans (e.g. ``action_pairs`` refresh) on
    session-scoped indexes; analyzing bulk-build's empty FTS virtual tables
    adds archive-scale I/O without improving replay.  Skipping measured row-
    store statistics altogether reproduced an O(N^2) replay at >20x slower.
    Failures are non-fatal: stale stats degrade speed, never correctness.
    """
    try:
        with contextlib.closing(sqlite3.connect(index_path, timeout=60)) as conn:
            conn.execute(f"PRAGMA analysis_limit = {_PLANNER_STATS_ANALYSIS_LIMIT}")
            for statement in _PLANNER_STATS_ANALYZE_STATEMENTS:
                conn.execute(statement)
            conn.commit()
    except sqlite3.Error:
        return


@dataclass(frozen=True, slots=True)
class RebuildIndexRequest:
    """One bounded source snapshot replay request."""

    archive_root: Path
    only_missing: bool = False
    raw_ids: tuple[str, ...] = ()
    max_blob_mb: float | None = None
    promote: bool = True
    operation_id: str | None = None
    raw_batch_size: int = 500
    pass_byte_budget_mb: float | None = None
    pass_deadline_seconds: float | None = None
    # polylogue-gd6v: daemon-internal callers only (never CLI/HTTP -- there is
    # no JSON wire shape for a live cache object). Lets the daemon's bulk
    # rebuild routing substitute parse output already computed off the
    # writer hold (``DaemonParseStage``) for this pass's census phase. Every
    # existing caller leaves this ``None`` and gets the exact unmodified
    # parse path.
    prefetch_cache: RawParsePrefetchCache | None = None


@dataclass(frozen=True, slots=True)
class RebuildPassCost:
    """What one rebuild pass cost, and what that implies for the whole run.

    Three full rebuilds completed with no cost breakdown persisted anywhere.
    The only forensics available afterwards was receipt file mtimes -- enough
    to show 88% of a 74-hour run was idle wall-clock, but not enough to say
    where the remaining 9.2 hours of compute went.

    ``replay_s`` / ``checkpoint_s``  where the pass went. NOTE: ``replay_source``
        does not expose a parse-vs-apply split, so ``replay_s`` covers both.
        Splitting it is the next instrumentation step and is what would say
        whether decode or the single writer is the bottleneck; recording a
        fabricated split would be worse than recording none.
    ``mib_per_s`` / ``raws_per_s``  is throughput holding, or degrading as the
        index grows?
    ``free_threaded`` / ``parse_workers``  did parallel parse actually engage?
        A GIL build silently parses ~98.5% of this corpus' bytes on ONE core,
        which is exactly how a 9-hour rebuild happened. That belongs in the
        durable artifact, not only a log line read afterwards.
    ``percent_bytes`` / ``eta_s``  how far in and how long left, from THIS
        run's observed byte rate. Progress is in BYTES because cost is
        bytes-bound -- passes end ``deferred`` on a byte budget, so a row-count
        percentage would call a rebuild half done with most of the payload left.
    """

    replay_s: float
    checkpoint_s: float
    pass_s: float
    raws: int
    bytes_in: int
    processed_raws: int
    processed_bytes: int
    total_raws: int
    total_bytes: int
    free_threaded: bool
    parse_workers: int

    @property
    def mib_per_s(self) -> float:
        return (self.bytes_in / (1024 * 1024) / self.pass_s) if self.pass_s > 0 else 0.0

    @property
    def raws_per_s(self) -> float:
        return (self.raws / self.pass_s) if self.pass_s > 0 else 0.0

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.total_bytes - self.processed_bytes)

    @property
    def eta_s(self) -> float | None:
        """Seconds remaining at this pass's observed byte rate, or None."""
        if self.pass_s <= 0 or self.bytes_in <= 0 or self.total_bytes <= 0:
            return None
        return self.remaining_bytes / (self.bytes_in / self.pass_s)

    def to_dict(self) -> dict[str, object]:
        eta = self.eta_s
        return {
            "replay_s": round(self.replay_s, 3),
            "checkpoint_s": round(self.checkpoint_s, 3),
            "pass_s": round(self.pass_s, 3),
            "raws": self.raws,
            "bytes_in": self.bytes_in,
            "mib_per_s": round(self.mib_per_s, 2),
            "raws_per_s": round(self.raws_per_s, 2),
            "processed_raws": self.processed_raws,
            "processed_bytes": self.processed_bytes,
            "total_raws": self.total_raws,
            "total_bytes": self.total_bytes,
            "percent_bytes": round(100.0 * self.processed_bytes / self.total_bytes, 2) if self.total_bytes else 0.0,
            "eta_s": round(eta, 1) if eta is not None else None,
            "free_threaded": self.free_threaded,
            "parse_workers": self.parse_workers,
        }


@dataclass(frozen=True, slots=True)
class RebuildIndexReceipt:
    """Typed evidence emitted after one source-to-index rebuild pass."""

    archive_root: str
    raw_session_count: int
    selected_raw_count: int
    skipped_by_blob_limit_count: int
    status: str
    materialized: bool
    materialization: dict[str, object]
    generation: dict[str, object]
    readiness: dict[str, object]
    replay: dict[str, object]
    transaction: dict[str, object] | None = None
    #: Wall-clock seconds per rebuild stage for THIS pass.
    #:
    #: Three full rebuilds ran without this, so the only cost breakdown
    #: available afterwards was receipt file mtimes -- enough to show 88% of a
    #: 74h run was idle, but not enough to say where the remaining 9.2h of
    #: compute went. The terminal stages were already logged as structured
    #: events; logs are not the durable artifact and per-pass parse/apply was
    #: not measured at all. Persisting it here makes the next optimisation
    #: evidence-based rather than a guess.
    timings_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": self.archive_root,
            "raw_session_count": self.raw_session_count,
            "selected_raw_count": self.selected_raw_count,
            "skipped_by_blob_limit_count": self.skipped_by_blob_limit_count,
            "status": self.status,
            "materialized": self.materialized,
            "materialization": self.materialization,
            "generation": self.generation,
            "readiness": self.readiness,
            "transaction": self.transaction,
            "timings_s": self.timings_s,
            **self.replay,
        }


def validate_rebuild_index_request(request: RebuildIndexRequest) -> None:
    """Reject selection and transaction combinations that cannot be promoted safely."""
    if request.raw_ids and request.only_missing:
        raise ValueError("--raw-id cannot be combined with --only-missing")
    if (request.raw_ids or request.only_missing) and request.promote:
        raise ValueError("partial rebuild selections require --no-promote and can never replace the active index")
    if request.max_blob_mb is not None and request.max_blob_mb <= 0:
        raise ValueError("max blob size must be positive")
    if request.max_blob_mb is not None and not request.raw_ids and not request.only_missing:
        raise ValueError("--max-blob-mb requires --only-missing or --raw-id")
    if request.raw_batch_size <= 0:
        raise ValueError("raw batch size must be positive")
    if request.pass_byte_budget_mb is not None and request.pass_byte_budget_mb <= 0:
        raise ValueError("pass byte budget must be positive")
    if request.pass_deadline_seconds is not None and request.pass_deadline_seconds <= 0:
        raise ValueError("pass deadline must be positive")
    if request.operation_id is not None and (
        request.raw_ids or request.only_missing or request.max_blob_mb is not None
    ):
        raise ValueError("--operation-id only resumes an unfiltered full-source rebuild")
    if request.operation_id is not None and (
        request.pass_byte_budget_mb is not None or request.pass_deadline_seconds is not None
    ):
        raise ValueError("resumed rebuild budgets are durable; omit pass budget options with --operation-id")


def count_source_raw_sessions(root: Path) -> int:
    source_db = root / "source.db"
    if not source_db.exists():
        return 0
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()
    return int(row[0]) if row is not None else 0


def total_source_blob_bytes(root: Path) -> int:
    """Total blob payload the rebuild has to replay, for progress and ETA.

    Rebuild cost is bytes-bound -- bounded passes end ``deferred`` on a byte
    budget, not a row count -- so percent-complete and ETA are only meaningful
    against total BYTES. Counting rows would have reported a rebuild as
    "half done" while the remaining half held most of the payload.
    """
    source_db = root / "source.db"
    if not source_db.exists():
        return 0
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        row = conn.execute("SELECT COALESCE(SUM(blob_size), 0) FROM raw_sessions").fetchone()
    return int(row[0]) if row is not None else 0


def missing_index_raw_ids(root: Path) -> list[str]:
    """Return source raw_ids that have not yet reached ``index.sessions``.

    polylogue-ogn1: a missing/lost ``index.db`` (fresh archive, or one just
    reset via ``ops reset --index``) means every source row is missing from
    the index by definition -- return the full source set instead of an
    empty list, so ``--only-missing`` actually rebuilds something on a
    fresh/lost index rather than silently doing nothing.
    """
    source_db = root / "source.db"
    if not source_db.exists():
        return []
    index_db = ArchiveLocation.resolve(root).active_index_path
    if not index_db.exists():
        return all_index_rebuild_raw_ids(root)
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        conn.execute("ATTACH DATABASE ? AS idx", (str(index_db),))
        rows = conn.execute(
            """
            SELECT r.raw_id FROM raw_sessions r
            WHERE NOT EXISTS (SELECT 1 FROM idx.sessions s WHERE s.raw_id = r.raw_id)
            ORDER BY r.acquired_at_ms, r.raw_id
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def all_index_rebuild_raw_ids(root: Path) -> list[str]:
    source_db = root / "source.db"
    if not source_db.exists():
        return []
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        rows = conn.execute("SELECT raw_id FROM raw_sessions ORDER BY acquired_at_ms, raw_id").fetchall()
    return [str(row[0]) for row in rows]


def filter_raw_ids_by_max_blob_size(root: Path, raw_ids: list[str], max_blob_mb: float | None) -> list[str]:
    if max_blob_mb is None or not raw_ids:
        return raw_ids
    source_db = root / "source.db"
    placeholders = ",".join("?" for _ in raw_ids)
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        rows = conn.execute(
            f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({placeholders}) AND blob_size <= ? "
            "ORDER BY acquired_at_ms, raw_id",
            (*raw_ids, int(max_blob_mb * 1024 * 1024)),
        ).fetchall()
    return [str(row[0]) for row in rows]


def select_rebuild_raw_ids(request: RebuildIndexRequest) -> tuple[int, list[str], int]:
    """Select source rows deterministically before the replay starts."""
    root = request.archive_root
    raw_count = count_source_raw_sessions(root)
    raw_ids = (
        list(dict.fromkeys(request.raw_ids))
        if request.raw_ids
        else missing_index_raw_ids(root)
        if request.only_missing
        else all_index_rebuild_raw_ids(root)
    )
    unfiltered_count = len(raw_ids)
    selected = filter_raw_ids_by_max_blob_size(root, raw_ids, request.max_blob_mb)
    return raw_count, selected, unfiltered_count - len(selected)


async def rebuild_index_from_source(request: RebuildIndexRequest) -> RebuildIndexReceipt:
    """Replay one source snapshot into an owned generation and optionally promote it.

    Acquires :class:`~polylogue.storage.archive_identity.OwnedArchiveLocation`
    over ``request.archive_root`` before any generation directory or SQLite
    tier is touched (polylogue-ovme.2 AC3): an offline rebuild is exactly the
    maintenance/campaign writer ``OwnedArchiveLocation`` exists for, and this
    is orthogonal to ``RebuildLease`` below (that lease serializes concurrent
    *rebuild* invocations specifically; this proves the caller still owns
    the *location* it resolved, catching e.g. a concurrent devtools campaign
    or a foreign/rotated root before this rebuild can act on stale identity).
    """
    validate_rebuild_index_request(request)
    root = request.archive_root
    location = ArchiveLocation.resolve(root)
    active_config = Config(
        archive_root=root,
        render_root=render_root(),
        sources=[],
        db_path=location.active_index_path,
    )
    if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
        raise RuntimeError(reason)

    owned = OwnedArchiveLocation.acquire(location)
    try:
        assert_owns_archive_location(owned, location)
        return await _rebuild_index_from_source_owned(request, root=root, owned=owned)
    finally:
        owned.release()


async def _rebuild_index_from_source_owned(
    request: RebuildIndexRequest, *, root: Path, owned: OwnedArchiveLocation
) -> RebuildIndexReceipt:
    """Ownership-proven body of :func:`rebuild_index_from_source`."""
    from polylogue.maintenance.archive_verification import verify_archive
    from polylogue.maintenance.replay import rebuild_index_from_source as replay_source
    from polylogue.storage.archive_readiness import archive_readiness_status
    from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, source_revision_snapshot
    from polylogue.storage.repair import repair_session_insights

    generation_store = IndexGenerationStore(owned.location)
    with RebuildLease(root):
        raw_count = count_source_raw_sessions(root)
        if raw_count == 0:
            return RebuildIndexReceipt(
                archive_root=str(root),
                raw_session_count=0,
                selected_raw_count=0,
                skipped_by_blob_limit_count=0,
                status="empty-source",
                materialized=False,
                materialization={},
                generation={},
                readiness={},
                replay={},
            )
        resumable_full_source = not request.raw_ids and not request.only_missing and request.max_blob_mb is None
        transaction = None
        page = None
        pass_started_at_ms = int(time.time() * 1000)
        if resumable_full_source:
            transaction = (
                generation_store.load_transaction(request.operation_id)
                if request.operation_id is not None
                else generation_store.create_transaction(
                    source_snapshot=source_revision_snapshot(root),
                    pass_byte_budget=(
                        int(request.pass_byte_budget_mb * 1024 * 1024)
                        if request.pass_byte_budget_mb is not None
                        else None
                    ),
                    pass_deadline_ms=(
                        int(request.pass_deadline_seconds * 1000) if request.pass_deadline_seconds is not None else None
                    ),
                )
            )
            if transaction.status in {"promoted", "stale"}:
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is {transaction.status}; start a new operation"
                )
            if source_revision_snapshot(root) != transaction.source_snapshot:
                generation_store.checkpoint_transaction(
                    transaction,
                    status="stale",
                    error="source evidence changed since this rebuild was planned",
                )
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                )
            generation = generation_store.load(transaction.generation_id)
            if generation.owner_id != transaction.generation_owner_id or generation.state != "inactive":
                raise RuntimeError(f"rebuild operation {transaction.operation_id} lost its inactive candidate")
            if not transaction.derived_stores_cleared:
                _clear_bulk_build_derived_stores(Path(generation.index_path))
                transaction = generation_store.checkpoint_transaction(
                    transaction,
                    status=transaction.status,
                    derived_stores_cleared=True,
                )
            page = generation_store.next_raw_page(transaction, limit=request.raw_batch_size)
            selected_raw_ids = [raw_id for raw_id, _acquired_at_ms, _blob_size in page.rows]
            selected_raw_count = len(selected_raw_ids)
            skipped_by_blob_limit_count = 0
        else:
            raw_count, selected_raw_ids, skipped_by_blob_limit_count = select_rebuild_raw_ids(request)
            selected_raw_count = len(selected_raw_ids)
            generation = generation_store.create(source_snapshot=source_revision_snapshot(root))
        source_drifted = False
        try:
            generation_root = Path(generation.index_path).parent
            config = Config(
                archive_root=generation_root,
                render_root=render_root(),
                sources=[],
                db_path=Path(generation.index_path),
            )
            pass_started_at_s = time.perf_counter()
            replay = await replay_source(
                config,
                raw_ids=selected_raw_ids,
                raw_batch_size=request.raw_batch_size,
                ingest_workers=None,
                materialize=True,
                progress_callback=None,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
                # polylogue-crd8: this is the offline rebuild path (an owned
                # inactive generation, never the live daemon ingest path), so
                # the guard-gated bulk FTS mode is safe to enable unconditionally
                # here -- it collapses whale prefix-sharing lineage cascades'
                # per-row messages_fts trigger storm into one bulk delete+insert
                # per affected session.
                bulk_fts=True,
                # polylogue-v6i3: the broader bulk-generation-build lifecycle --
                # every per-session messages_fts/blocks_command_trigram/
                # action_pairs/delegation_facts refresh is skipped during this
                # replay (safe for a full OR partial/diagnostic selection: a
                # repopulate from `blocks` always matches whatever sessions
                # actually got replayed into this generation); see
                # _repopulate_bulk_build_derived_state, called below right
                # before readiness.
                bulk_build=True,
                prefetch_cache=request.prefetch_cache,
            )
            pass_elapsed_s = time.perf_counter() - pass_started_at_s
            processed_before = transaction.processed_raw_count if transaction is not None else None
            if selected_raw_ids and _should_refresh_generation_planner_statistics(
                processed_before=processed_before,
                processed_after=(processed_before or 0) + len(selected_raw_ids),
            ):
                _refresh_generation_planner_statistics(Path(generation.index_path))
            if transaction is not None and selected_raw_ids:
                if source_revision_snapshot(root) != transaction.source_snapshot:
                    transaction = generation_store.checkpoint_transaction(
                        transaction,
                        status="stale",
                        error="source evidence changed during this bounded rebuild pass",
                    )
                    source_drifted = True
                    raise RuntimeError(
                        f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                    )
                assert page is not None
                last_raw_id, last_acquired_at_ms, _blob_size = page.rows[-1]
                elapsed_ms = int(time.time() * 1000) - pass_started_at_ms
                deadline_expired = (
                    transaction.pass_deadline_ms is not None and elapsed_ms >= transaction.pass_deadline_ms
                )
                status = "deferred" if page.deferred_reason == "byte-budget" or deadline_expired else "paused"
                transaction = generation_store.checkpoint_transaction(
                    transaction,
                    status=status,
                    last_acquired_at_ms=last_acquired_at_ms,
                    last_raw_id=last_raw_id,
                    processed_raw_count=transaction.processed_raw_count + len(selected_raw_ids),
                    processed_blob_bytes=transaction.processed_blob_bytes + sum(row[2] for row in page.rows),
                )
                if page.has_more or deadline_expired:
                    from polylogue.pipeline.services.process_pool import (
                        parallel_threads_effective,
                        resolve_parse_worker_count,
                    )

                    pass_cost = RebuildPassCost(
                        replay_s=pass_elapsed_s,
                        checkpoint_s=0.0,
                        pass_s=time.perf_counter() - pass_started_at_s,
                        raws=selected_raw_count,
                        bytes_in=sum(row[2] for row in page.rows),
                        processed_raws=transaction.processed_raw_count,
                        processed_bytes=transaction.processed_blob_bytes,
                        total_raws=raw_count,
                        total_bytes=total_source_blob_bytes(root),
                        free_threaded=parallel_threads_effective(),
                        parse_workers=resolve_parse_worker_count(),
                    )
                    logger.info(
                        "rebuild_pass_cost",
                        generation_id=generation.generation_id,
                        **pass_cost.to_dict(),
                    )
                    pass_receipt = RebuildIndexReceipt(
                        archive_root=str(root),
                        raw_session_count=raw_count,
                        selected_raw_count=selected_raw_count,
                        skipped_by_blob_limit_count=0,
                        status=status,
                        materialized=False,
                        materialization={},
                        generation=cast(dict[str, object], asdict(generation)),
                        readiness={},
                        replay=replay,
                        transaction=cast(dict[str, object], asdict(transaction)),
                        timings_s=cast(dict[str, float], pass_cost.to_dict()),
                    )
                    generation_store.save_pass_receipt(transaction.operation_id, pass_receipt.to_dict())
                    return pass_receipt
            terminal_started_at = time.perf_counter()
            insight_result = repair_session_insights(
                config,
                dry_run=False,
                archive_root_override=generation_root,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
            )
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="session_insights",
                elapsed_s=round(time.perf_counter() - terminal_started_at, 3),
            )
            if not insight_result.success:
                raise RuntimeError(f"session insight materialization failed: {insight_result.detail}")
            if source_revision_snapshot(root) != generation.source_snapshot:
                if transaction is not None:
                    transaction = generation_store.checkpoint_transaction(
                        transaction,
                        status="stale",
                        error="source evidence changed before terminal readiness",
                    )
                    source_drifted = True
                raise RuntimeError(f"source evidence changed while rebuilding {generation.generation_id}")
            # polylogue-v6i3: bulk-build replay (bulk_build=True above) left
            # messages_fts/blocks_command_trigram/action_pairs/delegation_facts
            # empty or stale for every session -- repopulate all four
            # archive-wide exactly once here, then prove exact parity before
            # readiness can observe (and silently accept) a mismatch.
            bulk_timings_s = _repopulate_bulk_build_derived_state(Path(generation.index_path))
            for stage, elapsed_s in bulk_timings_s.items():
                logger.info(
                    "rebuild_terminal_stage_complete",
                    generation_id=generation.generation_id,
                    stage=f"bulk_build.{stage}",
                    elapsed_s=round(elapsed_s, 3),
                )
            terminal_started_at = time.perf_counter()
            parity_report = verify_archive(generation_root, checks=["fts-parity"])
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="fts_parity",
                elapsed_s=round(time.perf_counter() - terminal_started_at, 3),
            )
            if parity_report.blocking:
                failing = "; ".join(check.summary for check in parity_report.checks if check.status.value == "error")
                raise RuntimeError(
                    f"bulk-build FTS/trigram parity failed for generation {generation.generation_id}: {failing}"
                )
            terminal_started_at = time.perf_counter()
            readiness = archive_readiness_status(generation_root)
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="readiness",
                elapsed_s=round(time.perf_counter() - terminal_started_at, 3),
            )
            if not readiness.get("checked") or int(readiness.get("blocked_surface_count", 1)) != 0:
                blocked = [
                    name
                    for name, info in cast(dict[str, dict[str, object]], readiness.get("surfaces", {})).items()
                    if info.get("ready") is not True
                ]
                detail = (
                    f"reason: {readiness.get('reason')}"
                    if not readiness.get("checked")
                    else "blocked surfaces: " + ", ".join(blocked)
                )
                raise RuntimeError(f"inactive generation {generation.generation_id} is not exact-ready; {detail}")
            if transaction is not None:
                transaction = generation_store.checkpoint_transaction(transaction, status="ready")
            if request.promote:
                # Re-prove ownership immediately before the activation swap:
                # a long-running rebuild pass can outlast a concurrent
                # promotion of a different generation, and this must be
                # caught before clobbering someone else's activation rather
                # than after (polylogue-ovme.2 AC3).
                assert_owns_archive_location(owned, ArchiveLocation.resolve(root))
                terminal_started_at = time.perf_counter()
                generation = generation_store.promote(generation)
                logger.info(
                    "rebuild_terminal_stage_complete",
                    generation_id=generation.generation_id,
                    stage="promote",
                    elapsed_s=round(time.perf_counter() - terminal_started_at, 3),
                )
                if transaction is not None:
                    transaction = generation_store.checkpoint_transaction(transaction, status="promoted")
        except Exception:
            if transaction is not None and not source_drifted:
                with contextlib.suppress(Exception):
                    generation_store.checkpoint_transaction(
                        transaction,
                        status="failed",
                        error="bounded rebuild pass failed; candidate retained for diagnosis or explicit recovery",
                    )
            else:
                with contextlib.suppress(Exception):
                    generation_store.discard_if_inactive(generation)
            raise
    final_receipt = RebuildIndexReceipt(
        archive_root=str(root),
        raw_session_count=raw_count,
        selected_raw_count=selected_raw_count,
        skipped_by_blob_limit_count=skipped_by_blob_limit_count,
        status="replayed",
        materialized=True,
        materialization=cast(dict[str, object], insight_result.to_dict()),
        generation=cast(dict[str, object], asdict(generation)),
        readiness=cast(dict[str, object], readiness),
        replay=replay,
        transaction=cast(dict[str, object], asdict(transaction)) if transaction is not None else None,
    )
    if transaction is not None:
        generation_store.save_pass_receipt(transaction.operation_id, final_receipt.to_dict())
    return final_receipt


def rebuild_index_from_source_sync(request: RebuildIndexRequest) -> RebuildIndexReceipt:
    """Synchronous adapter for offline CLI callers."""
    return asyncio.run(rebuild_index_from_source(request))


__all__ = [
    "RebuildIndexReceipt",
    "RebuildIndexRequest",
    "all_index_rebuild_raw_ids",
    "count_source_raw_sessions",
    "filter_raw_ids_by_max_blob_size",
    "missing_index_raw_ids",
    "rebuild_index_from_source",
    "rebuild_index_from_source_sync",
    "select_rebuild_raw_ids",
    "validate_rebuild_index_request",
]
