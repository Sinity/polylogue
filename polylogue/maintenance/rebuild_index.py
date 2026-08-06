"""Daemon-safe source-to-index rebuild execution.

The operation owns the write-side rebuild protocol; CLI and HTTP are adapters.
Callers must hold the daemon writer coordinator for an online rebuild.  The
offline guard rejects every other live-daemon caller, preserving break-glass
operation after the daemon has stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.config import Config
from polylogue.core.errors import PolylogueError
from polylogue.logging import get_logger
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
from polylogue.storage.fts.fts_lifecycle import rebuild_command_trigram_index_sync, rebuild_fts_index_sync
from polylogue.storage.fts.sql import FTS_REBUILD_SQL, TRIGRAM_REBUILD_DELETE_ALL_SQL
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.connection_profile import BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS
from polylogue.storage.sqlite.delegation_facts import rebuild_all_delegation_facts_sync

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache
    from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore, IndexRebuildTransaction

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


class RebuildProvenanceError(RuntimeError):
    """Raised when rebuild evidence is no longer valid for a mutation."""


class RebuildDerivedStateProvenanceError(RebuildProvenanceError):
    """A derived-state stage was blocked by a failed provenance recheck."""


class RebuildSchemaCurrencyError(PolylogueError):
    """The durable tiers do not match the package that would rebuild them."""

    http_status_code = HTTPStatus.CONFLICT

    def __init__(self, diagnostic: dict[str, object]) -> None:
        self.diagnostic = diagnostic
        blocked = diagnostic["blocking_tiers"]
        assert isinstance(blocked, list)
        detail = ", ".join(
            f"{item['tier']}.db:{item['actual_user_version']}!={item['expected_user_version']}"
            for item in blocked
            if isinstance(item, dict)
        )
        super().__init__(f"rebuild schema currency preflight failed: {detail}")


def rebuild_schema_currency_preflight(root: Path) -> dict[str, object]:
    """Report whether durable source evidence matches this runtime package.

    ``index.db`` is intentionally absent: rebuilding it is the operation's
    purpose, while a durable-tier mismatch means this package can interpret or
    write durable evidence using a schema it does not own.
    """
    from polylogue.storage.archive_readiness import probe_archive_tier
    from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS

    checks: list[dict[str, object]] = []
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        probe = probe_archive_tier(tier, root / f"{tier.value}.db")
        checks.append(
            {
                "tier": tier.value,
                "path": probe.path,
                "actual_user_version": probe.user_version,
                "expected_user_version": probe.expected_user_version,
                "status": probe.version_status,
            }
        )
    blocking = [check for check in checks if check["status"] != "ok"]
    return {
        "kind": "rebuild-schema-currency",
        "archive_root": str(root),
        "status": "ready" if not blocking else "blocked",
        "tiers": checks,
        "blocking_tiers": blocking,
    }


def require_rebuild_schema_currency(root: Path) -> dict[str, object]:
    """Reject a rebuild before it consumes evidence or creates a generation."""
    diagnostic = rebuild_schema_currency_preflight(root)
    if diagnostic["status"] != "ready":
        raise RebuildSchemaCurrencyError(diagnostic)
    return diagnostic


@dataclass(frozen=True, slots=True)
class RebuildProvenanceContext:
    """Validated evidence shared by every mutation in one rebuild pass.

    ``validate`` is the guard for operations that consume source evidence.
    Cleanup has a narrower contract: it may remove only a generation that was
    created under this already-validated context, and never reads or writes
    source evidence.  Keeping that distinction lets a failed receipt still
    clean up non-promotable scratch generations without treating the stale
    receipt as authorization to continue replaying.
    """

    root: Path
    receipt_path: Path | None
    source_snapshot: str
    consumed_evidence: dict[str, object]

    def validate(self) -> None:
        _validate_rebuild_provenance_receipt(self.root, self.receipt_path)
        from polylogue.maintenance.schema_inference_gate import rebuild_source_revision_snapshot

        if rebuild_source_revision_snapshot(self.root) != self.source_snapshot:
            raise RebuildProvenanceError(
                "rebuild schema-inference preflight gate failed: source evidence changed during replay"
            )

    def validate_cleanup(self) -> None:
        """Authorize failure cleanup from the immutable pre-mutation proof."""
        if not self.consumed_evidence or not self.source_snapshot:
            raise RuntimeError("rebuild cleanup has no validated provenance context")


def _validate_rebuild_provenance_receipt(root: Path, receipt_path: Path | None) -> dict[str, object]:
    """Validate rebuild provenance at the current ownership boundary."""
    from polylogue.maintenance.schema_inference_gate import (
        SchemaInferenceGateError,
        validate_schema_inference_receipt,
    )

    try:
        return validate_schema_inference_receipt(root, receipt_path)
    except SchemaInferenceGateError as exc:
        raise RebuildProvenanceError(f"rebuild schema-inference preflight gate failed: {exc}") from exc


def _validate_before_derived_state(provenance: RebuildProvenanceContext) -> None:
    """Validate immediately before a derived-state mutation begins."""
    try:
        provenance.validate()
    except Exception as exc:
        raise RebuildDerivedStateProvenanceError(str(exc)) from exc


def _discard_generation_after_provenance_failure(
    generation_store: IndexGenerationStore, generation: IndexGeneration, provenance: RebuildProvenanceContext
) -> list[BaseException]:
    """Discard a fresh candidate without rereading mutable source evidence."""
    errors: list[BaseException] = []
    try:
        provenance.validate_cleanup()
        discarded = generation_store.discard_if_inactive(generation)
        if not discarded:
            errors.append(RuntimeError(f"candidate {generation.generation_id} was not discarded"))
    except BaseException as exc:
        errors.append(exc)
    return errors


def _discard_transaction_after_provenance_failure(
    generation_store: IndexGenerationStore,
    transaction: IndexRebuildTransaction,
    provenance: RebuildProvenanceContext,
) -> list[BaseException]:
    """Discard a fresh candidate and transaction, attempting both independently."""
    errors: list[BaseException] = []
    try:
        generation = generation_store.load(transaction.generation_id)
    except FileNotFoundError:
        generation = None
    except BaseException as exc:
        errors.append(exc)
        generation = None
    if generation is not None:
        errors.extend(_discard_generation_after_provenance_failure(generation_store, generation, provenance))
    try:
        discarded = generation_store.discard_transaction(transaction.operation_id)
        if not discarded:
            errors.append(
                RuntimeError(f"transaction {transaction.operation_id} was not discarded because its record was missing")
            )
    except BaseException as exc:
        errors.append(exc)
    return errors


def _cleanup_transaction_after_provenance_failure(
    generation_store: IndexGenerationStore,
    transaction: IndexRebuildTransaction,
    root: Path,
    receipt_path: Path | None,
    consumed_evidence: dict[str, object],
    primary: BaseException,
) -> None:
    """Clean a fresh transaction and candidate while preserving its failure."""
    cleanup_context = RebuildProvenanceContext(
        root=root,
        receipt_path=receipt_path,
        source_snapshot=str(consumed_evidence.get("source_snapshot", "")),
        consumed_evidence=consumed_evidence,
    )
    try:
        cleanup_errors = _discard_transaction_after_provenance_failure(generation_store, transaction, cleanup_context)
    except BaseException as cleanup_error:
        cleanup_errors = [cleanup_error]
    _add_cleanup_failure_notes(primary, cleanup_errors, label="rebuild transaction")


def _add_cleanup_failure_notes(primary: BaseException, errors: list[BaseException], *, label: str) -> None:
    """Keep the primary exception while making cleanup failures visible."""
    if errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        primary.add_note(f"{label} cleanup also failed: {detail}")


def _cleanup_nonresumable_generation_failure(
    generation_store: IndexGenerationStore,
    generation: IndexGeneration,
    *,
    root: Path,
    receipt_path: Path | None,
    consumed_evidence: dict[str, object],
    primary: BaseException,
) -> None:
    """Retire a one-shot candidate and expose every discard outcome."""
    cleanup_context = RebuildProvenanceContext(
        root=root,
        receipt_path=receipt_path,
        source_snapshot=str(consumed_evidence.get("source_snapshot", "")),
        consumed_evidence=consumed_evidence,
    )
    try:
        cleanup_errors = _discard_generation_after_provenance_failure(generation_store, generation, cleanup_context)
    except BaseException as cleanup_error:
        cleanup_errors = [cleanup_error]
    _add_cleanup_failure_notes(primary, cleanup_errors, label="nonresumable rebuild")


def _create_rebuild_transaction_after_receipt_validation(
    generation_store: IndexGenerationStore,
    request: RebuildIndexRequest,
    root: Path,
) -> IndexRebuildTransaction:
    """Create the first candidate only after an ownership-bound validation."""
    from polylogue.storage.index_generation import rebuild_source_evidence_snapshot

    consumed_evidence = _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
    source_snapshot = rebuild_source_evidence_snapshot(root)
    transaction = generation_store.create_transaction(
        source_snapshot=source_snapshot,
        pass_byte_budget=(
            int(request.pass_byte_budget_mb * 1024 * 1024) if request.pass_byte_budget_mb is not None else None
        ),
        pass_deadline_ms=(
            int(request.pass_deadline_seconds * 1000) if request.pass_deadline_seconds is not None else None
        ),
    )
    try:
        _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
        if rebuild_source_evidence_snapshot(root) != source_snapshot:
            raise RebuildProvenanceError(
                "rebuild schema-inference preflight gate failed: source evidence changed during transaction creation"
            )
    except BaseException as exc:
        _cleanup_transaction_after_provenance_failure(
            generation_store,
            transaction,
            root,
            request.schema_inference_receipt_path,
            consumed_evidence,
            exc,
        )
        raise
    return transaction


def _checkpoint_rebuild_transaction_after_receipt_validation(
    generation_store: IndexGenerationStore,
    transaction: IndexRebuildTransaction,
    root: Path,
    receipt_path: Path | None,
    *,
    status: str,
    last_blob_hash_hex: str | None = None,
    last_raw_id: str | None = None,
    processed_raw_count: int | None = None,
    processed_blob_bytes: int | None = None,
    error: str | None = None,
    derived_stores_cleared: bool | None = None,
) -> IndexRebuildTransaction:
    """Validate immediately before every persisted rebuild state transition."""
    _validate_rebuild_provenance_receipt(root, receipt_path)
    return generation_store.checkpoint_transaction(
        transaction,
        status=status,
        last_blob_hash_hex=last_blob_hash_hex,
        last_raw_id=last_raw_id,
        processed_raw_count=processed_raw_count,
        processed_blob_bytes=processed_blob_bytes,
        error=error,
        derived_stores_cleared=derived_stores_cleared,
    )


def _save_rebuild_pass_receipt_after_receipt_validation(
    generation_store: IndexGenerationStore,
    operation_id: str,
    pass_receipt: RebuildIndexReceipt,
    root: Path,
    receipt_path: Path | None,
) -> None:
    """Validate before publishing a pass receipt tied to rebuild state."""
    _validate_rebuild_provenance_receipt(root, receipt_path)
    generation_store.save_pass_receipt(operation_id, pass_receipt.to_dict())


#: Passed through to ``CensusParseStage.warm_raw_ids``'s ``max_payload_bytes``
#: parameter for symmetry with the daemon's own call site
#: (``daemon/bulk_rebuild.py``); the parameter is currently accepted but not
#: consulted inside ``warm_raw_ids`` itself, so this value has no observable
#: effect today, but every caller supplies one so a future budget check does
#: not silently start unbounded for whichever caller forgot to pass it.
_OFFLINE_PREFETCH_WARM_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


def _warm_offline_prefetch_cache(config: Config, raw_ids: list[str]) -> RawParsePrefetchCache | None:
    """Pre-parse this pass's raw ids in a bounded thread pool before replay.

    polylogue-czq2: closes the gap where every rebuild caller EXCEPT the
    daemon's own bulk-rebuild loop (``daemon/bulk_rebuild.py``, #3168) left
    ``RebuildIndexRequest.prefetch_cache`` at its default ``None`` -- the
    offline ``polylogue ops maintenance rebuild-index`` CLI and the daemon's
    own ``/api/maintenance/rebuild-index`` HTTP route both construct a
    ``RebuildIndexRequest`` without ever threading one, so ``census``'s parse
    step (and the ``spill_load`` reload it feeds) always paid the full
    unwarmed cost on those routes even though the exact machinery to avoid it
    (``CensusParseStage``/``RawParsePrefetchCache``,
    ``polylogue.sources.census_parse_stage``) already existed and was fully
    wired through ``backfill_historical_revision_evidence``.

    Called from ``_rebuild_index_from_source_owned`` only when
    ``request.prefetch_cache is None`` (a caller that already warmed its own
    cache off a writer hold it does not yet hold -- the daemon's bulk-rebuild
    loop -- is never overridden here). Returns ``None`` for an empty
    ``raw_ids`` (nothing to warm); a construction/warm failure is never
    raised -- this is a pure optimization over the unmodified parse path, so
    any failure here must degrade to that path exactly like an ordinary
    prefetch miss, never abort the rebuild pass itself.
    """
    if not raw_ids:
        return None
    from polylogue.sources.census_parse_stage import CensusParseStage

    stage = CensusParseStage()
    try:
        warmed = stage.warm_raw_ids(config, raw_ids=raw_ids, max_payload_bytes=_OFFLINE_PREFETCH_WARM_MAX_PAYLOAD_BYTES)
        logger.info(
            "rebuild_index_offline_prefetch_warm",
            requested=len(raw_ids),
            warmed=warmed,
        )
    except Exception:
        logger.warning("rebuild_index_offline_prefetch_warm_failed", exc_info=True)
        stage.shutdown()
        return None
    stage.shutdown()
    return stage.cache


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


def _open_bulk_build_maintenance_connection(index_path: Path, *, timeout: int) -> sqlite3.Connection:
    """Open a terminal-stage maintenance connection with the bulk-build profile.

    The terminal repopulate/clear stages act on the same owned INACTIVE
    generation the replay just bulk-wrote: never read until promoted, and
    discarded wholesale if the pass raises. That is exactly the licence
    ``BULK_BUILD_WRITE_CONNECTION_PROFILE`` documents (``journal_mode=MEMORY``,
    ``synchronous=OFF``, large cache/mmap) -- previously these connections ran
    with SQLite's stock defaults (rollback journal on disk, ``synchronous=FULL``,
    ~2 MiB cache), which made the archive-wide FTS/trigram/action-pair
    repopulate pay a full journal round-trip and fsync per committed batch at
    full-table scale.
    """
    conn = sqlite3.connect(index_path, timeout=timeout)
    try:
        for statement in BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS:
            conn.execute(statement)
    except BaseException:
        conn.close()
        raise
    return conn


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
    with contextlib.closing(_open_bulk_build_maintenance_connection(index_path, timeout=60)) as conn:
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
    with contextlib.closing(_open_bulk_build_maintenance_connection(index_path, timeout=600)) as conn:
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
        with contextlib.closing(_open_bulk_build_maintenance_connection(index_path, timeout=60)) as conn:
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
    candidate_acceptance_checks: tuple[str, ...] | None = None
    operation_id: str | None = None
    schema_inference_receipt_path: Path | None = None
    raw_batch_size: int = 500
    pass_byte_budget_mb: float | None = None
    pass_deadline_seconds: float | None = None
    # polylogue-gd6v: in-process callers only (never CLI/HTTP wire params --
    # there is no JSON shape for a live cache object). Lets a caller that
    # already computed parse output off a writer hold (the daemon's own
    # ``CensusParseStage``, e.g. ``daemon/bulk_rebuild.py``) substitute it for
    # this pass's census phase. Leaving this ``None`` (every CLI/HTTP-facing
    # caller) does NOT skip prefetching any more: ``_rebuild_index_from_source_owned``
    # warms one itself for exactly this pass's selected raw ids before
    # replaying (see ``_warm_offline_prefetch_cache``) -- a caller only needs
    # to set this explicitly to reuse a cache warmed AHEAD of a writer hold it
    # does not yet hold, which is what the daemon's bulk-rebuild loop does.
    prefetch_cache: RawParsePrefetchCache | None = None
    # polylogue-pzxm: split this pass's selected raw ids into shard_count
    # owned-inactive generations built in parallel (free-threaded
    # interpreter: threads share parsed graphs), merged sequentially into
    # this pass's real target generation before the existing terminal
    # stages run. 1 (the default) is the unchanged single-writer path --
    # every existing caller (CLI, daemon HTTP route, bulk-rebuild loop) is
    # unaffected until it opts in. See
    # polylogue.maintenance.sharded_rebuild for the merge/graph-resolution
    # correctness argument.
    shard_count: int = 1


@dataclass(frozen=True, slots=True)
class RebuildPassCost:
    """What one rebuild pass cost, and what that implies for the whole run.

    Three full rebuilds completed with no cost breakdown persisted anywhere.
    The only forensics available afterwards was receipt file mtimes -- enough
    to show 88% of a 74-hour run was idle wall-clock, but not enough to say
    where the remaining 9.2 hours of compute went.

    ``replay_s`` / ``checkpoint_s``  where the pass went. ``replay_s`` is the
        wall-clock time around the whole ``replay_source`` call (parse +
        apply + the small async-dispatch overhead between them); ``parse_s``
        and ``apply_s`` (below) are its breakdown.
    ``parse_s`` / ``apply_s``  the parse-vs-apply split (polylogue-623q).
        ``parse_s`` is read-only decode (census parse + spill-cache reload of
        already-parsed content) -- embarrassingly parallel, scales with
        ``parse_workers``. ``apply_s`` is everything charged to the single
        SQLite writer (index/FTS/projection writes) -- serialized, does not
        scale with worker count. Sourced from
        ``revision_backfill.split_parse_and_apply_seconds`` over the
        ``stage_timings_s`` dict threaded back through ``replay_source``'s
        return value; before that threading existed, ``replay_s`` was the
        only number recorded and there was no way to tell decode and writer
        cost apart. Both are ``0.0`` if the pass replayed zero raws (no
        stage ever ran).
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
    parse_s: float = 0.0
    apply_s: float = 0.0
    #: polylogue-6mvg: wall-clock seconds this pass spent choosing WHICH raws
    #: to replay -- the resumable path's ``next_raw_page`` keyset query, or
    #: the one-shot path's ``select_rebuild_raw_ids`` (full-source/only-
    #: missing/max-blob-mb scan plus size filter) -- before any census/parse/
    #: apply work started. Previously invisible: a live full rebuild spent
    #: ~86s of one CPU core on selection alone before the inactive generation
    #: held a single session, with nothing durable recording where that time
    #: went.
    selection_s: float = 0.0
    #: Time spent classifying byte and membership authority cohorts. This is
    #: a diagnostic rollup over the replay stage ledger, retained separately
    #: from the parse/apply split for phase-level receipts.
    cohort_s: float = 0.0
    #: Terminal insight materialization time. Deferred/paused passes carry
    #: the explicit zero because they have not reached terminal stages.
    insight_s: float = 0.0
    #: Sum of terminal stages after replay, excluding raw selection.
    terminal_s: float = 0.0

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
            "selection_s": round(self.selection_s, 3),
            "cohort_s": round(self.cohort_s, 3),
            "replay_s": round(self.replay_s, 3),
            "parse_s": round(self.parse_s, 3),
            "apply_s": round(self.apply_s, 3),
            "insight_s": round(self.insight_s, 3),
            "terminal_s": round(self.terminal_s, 3),
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


_COHORT_TIMING_PREFIXES = ("replay.classify_cohort", "replay.adoptable_check", "membership.")


def _cohort_seconds(stage_timings_s: object) -> float:
    """Roll up the durable replay timing keys that decide authority cohorts."""
    if not isinstance(stage_timings_s, dict):
        return 0.0
    return sum(
        float(value)
        for key, value in stage_timings_s.items()
        if isinstance(key, str)
        and key.startswith(_COHORT_TIMING_PREFIXES)
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    )


def _receipt_timings(
    *,
    selection_s: float,
    replay: dict[str, object],
    terminal_timings_s: dict[str, float],
) -> dict[str, float]:
    """Build one stable phase vocabulary plus existing granular timings."""
    stage_timings_s = replay.get("stage_timings_s", {})
    parse_s = replay.get("parse_s", 0.0)
    apply_s = replay.get("apply_s", 0.0)
    resolved_parse_s = float(parse_s) if isinstance(parse_s, int | float) else 0.0
    resolved_apply_s = float(apply_s) if isinstance(apply_s, int | float) else 0.0
    insight_s = float(terminal_timings_s.get("terminal.session_insights", 0.0))
    terminal_s = sum(float(value) for key, value in terminal_timings_s.items() if key != "selection_s")
    rollups = {
        "selection_s": float(selection_s),
        "cohort_s": _cohort_seconds(stage_timings_s),
        "parse_s": resolved_parse_s,
        "apply_s": resolved_apply_s,
        "insight_s": insight_s,
        "terminal_s": terminal_s,
    }
    return {
        **{key: round(value, 3) for key, value in rollups.items()},
        **{key: round(float(value), 3) for key, value in terminal_timings_s.items()},
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
    # Explicit operation evidence, retained in every pass receipt as well as
    # returned to CLI/HTTP callers.  ``transaction`` remains the backwards
    # compatible full checkpoint payload; this compact view is the stable
    # operator contract for ownership, heartbeat, cursor, delta, and recovery.
    operation: dict[str, object] = field(default_factory=dict)
    # Rebuild-owned evidence for the exact raw selection replayed into this
    # candidate. The IDs themselves are deliberately not duplicated in every
    # receipt; the stable commitment is enough for a verifier holding the
    # requested IDs to prove the set, count, source snapshot, and candidate
    # identity all agree.
    selection_evidence: dict[str, object] = field(default_factory=dict)
    # The immutable source-evidence hash taken after replay. This is separate
    # from the generation's before-replay source_snapshot so report
    # consumption can reject source drift without treating parser or
    # governance state as part of the canary's identity.
    source_evidence_after: str | None = None
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
    consumed_evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_schema_version": 4,
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
            "operation": self.operation,
            "selection_evidence": self.selection_evidence,
            "source_evidence_after": self.source_evidence_after,
            "timings_s": self.timings_s,
            "consumed_evidence": self.consumed_evidence,
            **self.replay,
        }


def rebuild_selection_evidence(
    raw_ids: list[str] | tuple[str, ...],
    *,
    archive_root: Path,
    generation_id: str,
    generation_owner_id: str,
    candidate_index: Path,
    source_snapshot: str,
) -> dict[str, object]:
    """Commit the requested and production-expanded replay closure.

    The replay path can widen a raw-id hint after census discovers durable
    membership and logical-source-key relationships.  Persisting only the
    caller's hints would let a later source mutation change which raws and
    cohorts the candidate actually represents without invalidating its
    receipt.
    """

    replay_closure = _rebuild_replay_closure_evidence(archive_root, raw_ids)

    canonical = {
        "archive_root": str(archive_root.resolve()),
        "candidate_generation_id": generation_id,
        "candidate_index_path": str(candidate_index.resolve()),
        "candidate_owner_id": generation_owner_id,
        "raw_ids": sorted(raw_ids),
        "replay_closure": replay_closure,
        "source_snapshot": source_snapshot,
    }
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "algorithm": "sha256-canonical-json-v1",
        "raw_id_count": len(raw_ids),
        "raw_ids_sha256": sha256(encoded).hexdigest(),
        **{key: value for key, value in canonical.items() if key != "raw_ids"},
    }


def _rebuild_replay_closure_evidence(archive_root: Path, raw_ids: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Read the same durable closure primitive used by source replay.

    Missing source tiers are tolerated for standalone structural tests that
    exercise selection serialization without an archive. Real rebuilds always
    have ``source.db`` and therefore record the full expanded closure and every
    membership row participating in it.
    """

    source_db = archive_root / "source.db"
    if not source_db.exists():
        return {"raw_ids": sorted(raw_ids), "logical_source_keys": [], "raw_session_memberships": []}

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as connection:
        expanded, logical_keys = ArchiveStore.expand_raw_membership_selection_sync(connection, list(raw_ids))
        placeholders_raw = ",".join("?" for _ in expanded)
        placeholders_key = ",".join("?" for _ in logical_keys)
        clauses: list[str] = []
        parameters: list[str] = []
        if expanded:
            clauses.append(f"raw_id IN ({placeholders_raw})")
            parameters.extend(expanded)
        if logical_keys:
            clauses.append(f"logical_source_key IN ({placeholders_key})")
            parameters.extend(logical_keys)
        rows: list[dict[str, object]] = []
        if clauses:
            selected = connection.execute(
                "SELECT raw_id, logical_source_key, provider_session_id, source_revision, "
                "normalized_content_hash, message_count, predecessor_raw_id, acquisition_generation, "
                "revision_authority, decision, decided_at_ms "
                "FROM raw_session_memberships WHERE " + " OR ".join(clauses) + " ORDER BY logical_source_key, raw_id",
                parameters,
            )
            for row in selected:
                rows.append(
                    {
                        "raw_id": str(row[0]),
                        "logical_source_key": str(row[1]),
                        "provider_session_id": str(row[2]),
                        "source_revision": str(row[3]),
                        "normalized_content_hash": bytes(row[4]).hex(),
                        "message_count": int(row[5]),
                        "predecessor_raw_id": None if row[6] is None else str(row[6]),
                        "acquisition_generation": int(row[7]),
                        "revision_authority": str(row[8]),
                        "decision": None if row[9] is None else str(row[9]),
                        "decided_at_ms": None if row[10] is None else int(row[10]),
                    }
                )
    return {
        "raw_ids": list(expanded),
        "logical_source_keys": list(logical_keys),
        "raw_session_memberships": rows,
    }


def _persist_candidate_receipt(generation: IndexGeneration, receipt: dict[str, object]) -> None:
    """Atomically retain the completed rebuild receipt with its candidate."""

    path = Path(generation.index_path).parent / "rebuild-receipt.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _operation_evidence(
    root: Path,
    *,
    generation: IndexGeneration | None,
    transaction: IndexRebuildTransaction | None,
    recovery_state: str,
) -> dict[str, object]:
    """Return the compact, durable operation receipt shared by every pass.

    This is intentionally assembled from the existing archive-root lease and
    transaction checkpoint.  It adds no competing ownership mechanism and
    keeps the full checkpoint in ``transaction`` for callers that need it.
    """
    from polylogue.storage.index_generation import rebuild_lease_status, rebuild_source_evidence_snapshot

    transaction_payload = asdict(transaction) if transaction is not None else {}
    generation_payload = asdict(generation) if generation is not None else {}
    source_snapshot = transaction_payload.get("source_snapshot") or generation_payload.get("source_snapshot")
    current_snapshot = rebuild_source_evidence_snapshot(root) if (root / "source.db").exists() else None
    return {
        "owner": {
            "generation_owner_id": transaction_payload.get("generation_owner_id") or generation_payload.get("owner_id"),
            "pid": transaction_payload.get("owner_pid"),
            "host": transaction_payload.get("owner_host"),
            "lease": rebuild_lease_status(root).to_dict(),
        },
        "generation": {
            "generation_id": generation_payload.get("generation_id"),
            "state": generation_payload.get("state"),
        },
        "heartbeat": {"at_ms": transaction_payload.get("heartbeat_at_ms")},
        "cursor": transaction.cursor if transaction is not None else None,
        "delta": {
            "source_snapshot_matches": current_snapshot == source_snapshot if current_snapshot is not None else None,
            "current_source_snapshot": current_snapshot,
            "transaction_source_snapshot": source_snapshot,
        },
        "recovery_state": recovery_state,
    }


def validate_rebuild_index_request(request: RebuildIndexRequest) -> None:
    """Reject selection and transaction combinations that cannot be promoted safely."""
    if request.raw_ids and request.only_missing:
        raise ValueError("--raw-id cannot be combined with --only-missing")
    if (request.raw_ids or request.only_missing) and request.promote:
        raise ValueError("partial rebuild selections require --no-promote and can never replace the active index")
    if request.promote and request.candidate_acceptance_checks is not None:
        raise ValueError("caller-supplied candidate acceptance profiles require --no-promote")
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
    if request.shard_count < 1:
        raise ValueError("shard_count must be positive")
    if request.shard_count > 1 and request.pass_deadline_seconds is not None:
        # polylogue-pzxm/polylogue-uhgm interaction: the sharded path has no
        # deadline_check seam threaded to its K concurrent shard replays (see
        # the dispatch site's comment in _rebuild_index_from_source_owned) --
        # reject the combination up front rather than silently ignoring the
        # deadline.
        raise ValueError("--shard-count does not yet honor --pass-deadline-seconds; use one or the other")
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
    from polylogue.storage.index_generation import RebuildLease
    from polylogue.storage.sqlite.connection_profile import (
        check_mapped_bytes_budget_against_cgroup_limit,
        log_mapped_bytes_budget_check,
    )

    # polylogue-e98k: this is the other path (besides daemon startup) that can
    # hold a ``BULK_BUILD_WRITE_CONNECTION_PROFILE`` connection (4 GiB mmap) --
    # log the budget-vs-cgroup-limit comparison before replay starts, not only
    # discoverable by symptom after a throttled/stalled rebuild.
    log_mapped_bytes_budget_check(logger, check_mapped_bytes_budget_against_cgroup_limit())
    validate_rebuild_index_request(request)
    root = request.archive_root
    require_rebuild_schema_currency(root)
    consumed_evidence = _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
    location = ArchiveLocation.resolve(root)
    # The joined raw-frontier projection is rooted at the co-located active
    # index. A split-root canary intentionally points that index elsewhere and
    # validates the selected active generation through its own receipt-bound
    # route below, so a missing root/index.db must not masquerade as raw debt.
    if count_source_raw_sessions(root) and location.active_index_path.parent == root:
        from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

        if reason := raw_frontier_source_selection_block_reason(root):
            raise RuntimeError(f"reindex source preflight gate failed: raw frontier integrity: {reason}")
    active_config = Config(
        archive_root=root,
        render_root=render_root(),
        sources=[],
        db_path=location.active_index_path,
    )
    if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
        raise RuntimeError(reason)
    from polylogue.maintenance.archive_verification import REINDEX_SOURCE_PREFLIGHT_CHECKS, verify_archive

    source_liveness = verify_archive(root, checks=REINDEX_SOURCE_PREFLIGHT_CHECKS)
    if source_liveness.blocking:
        failing = "; ".join(
            f"{check.name}: {check.summary}"
            for check in source_liveness.checks
            if check.status.value == "error" and getattr(check, "waived_bead_id", None) is None
        )
        raise RuntimeError(f"reindex source preflight gate failed: {failing}")

    # Ownership acquisition itself only claims the lock. Revalidate after it
    # so receipt expiry/source/external-corpus drift between the cheap
    # preflight and ownership acquisition cannot reach the rebuild lease or a
    # candidate mutation.
    owned = OwnedArchiveLocation.acquire(location)
    try:
        assert_owns_archive_location(owned, location)
        require_rebuild_schema_currency(root)
        consumed_evidence = _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
        # The lease is itself lifecycle state guarded by the provenance gate.
        # Revalidate again under the lease immediately before the owned body
        # can create or mutate a candidate/transaction.
        with RebuildLease(root):
            consumed_evidence = _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
            return await _rebuild_index_from_source_owned(
                request, root=root, owned=owned, consumed_evidence=consumed_evidence
            )
    finally:
        owned.release()


async def _rebuild_index_from_source_owned(
    request: RebuildIndexRequest,
    *,
    root: Path,
    owned: OwnedArchiveLocation,
    consumed_evidence: dict[str, object],
) -> RebuildIndexReceipt:
    """Ownership-proven body of :func:`rebuild_index_from_source`."""
    from polylogue.maintenance.archive_verification import (
        REINDEX_ACCEPTANCE_CHECKS,
        REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
        passes_strict_acceptance,
        strict_acceptance_failures,
        verify_archive,
    )
    from polylogue.maintenance.replay import rebuild_index_from_source as replay_source
    from polylogue.sources.revision_backfill import RebuildDeadlineExceededError
    from polylogue.storage.archive_readiness import archive_readiness_status
    from polylogue.storage.index_generation import (
        IndexGenerationStore,
        rebuild_source_evidence_snapshot,
    )
    from polylogue.storage.repair import repair_session_insights

    generation_store = IndexGenerationStore(owned.location)
    # ``rebuild_index_from_source`` already acquired this root's
    # ``RebuildLease`` before any operation mutation.  Retain this scope only
    # to preserve the body's indentation and make the outer ownership boundary
    # explicit at the public entry point.
    with contextlib.nullcontext():
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
                operation=_operation_evidence(root, generation=None, transaction=None, recovery_state="empty-source"),
                consumed_evidence=consumed_evidence,
            )
        resumable_full_source = not request.raw_ids and not request.only_missing and request.max_blob_mb is None
        transaction = None
        transaction_created_here = False
        page = None
        pass_started_at_ms = int(time.time() * 1000)
        if resumable_full_source:
            if request.operation_id is not None:
                transaction = generation_store.load_transaction(request.operation_id)
            else:
                transaction = _create_rebuild_transaction_after_receipt_validation(
                    generation_store,
                    request,
                    root,
                )
                transaction_created_here = True
            if transaction.status in {"promoted", "stale"}:
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is {transaction.status}; start a new operation"
                )
            if rebuild_source_evidence_snapshot(root) != transaction.source_snapshot:
                if transaction_created_here:
                    mismatch = RebuildProvenanceError(
                        "rebuild schema-inference preflight gate failed: "
                        "source evidence changed since this rebuild was planned"
                    )
                    _cleanup_transaction_after_provenance_failure(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        consumed_evidence,
                        mismatch,
                    )
                    raise mismatch
                try:
                    _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="stale",
                        error="source evidence changed since this rebuild was planned",
                    )
                except RebuildProvenanceError as exc:
                    if transaction_created_here:
                        _cleanup_transaction_after_provenance_failure(
                            generation_store,
                            transaction,
                            root,
                            request.schema_inference_receipt_path,
                            consumed_evidence,
                            exc,
                        )
                    raise
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                )
            generation = generation_store.load(transaction.generation_id)
            if generation.owner_id != transaction.generation_owner_id or generation.state != "inactive":
                raise RuntimeError(f"rebuild operation {transaction.operation_id} lost its inactive candidate")
            if not transaction.derived_stores_cleared:
                try:
                    _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
                    _clear_bulk_build_derived_stores(Path(generation.index_path))
                    transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status=transaction.status,
                        derived_stores_cleared=True,
                    )
                except RebuildProvenanceError as exc:
                    if transaction_created_here:
                        _cleanup_transaction_after_provenance_failure(
                            generation_store,
                            transaction,
                            root,
                            request.schema_inference_receipt_path,
                            consumed_evidence,
                            exc,
                        )
                    raise
            # polylogue-6mvg: the pass's SELECTION phase -- which raws
            # replay THIS pass -- previously had no durable timing at all. A
            # live full rebuild spent ~86s of one CPU core on selection
            # alone before the inactive generation held a single session,
            # visible only as an unexplained gap between the pass starting
            # and its first "backfill stage timings" log line.
            selection_started_at = time.perf_counter()
            page = generation_store.next_raw_page(transaction, limit=request.raw_batch_size)
            selection_elapsed_s = time.perf_counter() - selection_started_at
            selected_raw_ids = [raw_id for raw_id, _blob_hash_hex, _blob_size in page.rows]
            selected_raw_count = len(selected_raw_ids)
            skipped_by_blob_limit_count = 0
        else:
            selection_started_at = time.perf_counter()
            raw_count, selected_raw_ids, skipped_by_blob_limit_count = select_rebuild_raw_ids(request)
            selection_elapsed_s = time.perf_counter() - selection_started_at
            selected_raw_count = len(selected_raw_ids)
            precreate_provenance = RebuildProvenanceContext(
                root=root,
                receipt_path=request.schema_inference_receipt_path,
                source_snapshot=str(consumed_evidence.get("source_snapshot", "")),
                consumed_evidence=consumed_evidence,
            )
            precreate_provenance.validate()
            generation = generation_store.create(source_snapshot=rebuild_source_evidence_snapshot(root))
        try:
            selection_evidence = rebuild_selection_evidence(
                selected_raw_ids,
                archive_root=root,
                generation_id=generation.generation_id,
                generation_owner_id=generation.owner_id,
                candidate_index=Path(generation.index_path),
                source_snapshot=generation.source_snapshot,
            )
        except BaseException as exc:
            if transaction is None:
                _cleanup_nonresumable_generation_failure(
                    generation_store,
                    generation,
                    root=root,
                    receipt_path=request.schema_inference_receipt_path,
                    consumed_evidence=consumed_evidence,
                    primary=exc,
                )
            raise
        provenance = RebuildProvenanceContext(
            root=root,
            receipt_path=request.schema_inference_receipt_path,
            source_snapshot=str(consumed_evidence.get("source_snapshot", "")),
            consumed_evidence=consumed_evidence,
        )
        sharded_replay = request.shard_count > 1 and len(selected_raw_ids) >= request.shard_count
        source_drifted = False
        try:
            provenance.validate()
            generation_root = Path(generation.index_path).parent
            config = Config(
                archive_root=generation_root,
                render_root=render_root(),
                sources=[],
                db_path=Path(generation.index_path),
            )
            # polylogue-czq2: warm THIS pass's own raw ids before replay when
            # the caller did not already hand us a prefetch cache (the
            # daemon's bulk-rebuild loop is the only caller that does --
            # `resolve_or_start_daemon_bulk_rebuild_transaction` warms off a
            # writer hold it does not yet hold, so its cache is left alone
            # here). Every other caller (offline CLI, the daemon's own HTTP
            # rebuild-index route) gets the same census-parse seam the daemon
            # loop always had, closing the gap that let ``spill_load`` pay a
            # full serial re-parse/reload cost on those routes. Reads from the
            # REAL archive root (`root`), not `generation_root`: source.db and
            # the blob store live beside the outer archive, never inside a
            # not-yet-promoted generation directory.
            effective_prefetch_cache = request.prefetch_cache
            if effective_prefetch_cache is None and selected_raw_ids:
                warm_config = Config(archive_root=root, render_root=render_root(), sources=[])
                effective_prefetch_cache = await asyncio.to_thread(
                    _warm_offline_prefetch_cache, warm_config, selected_raw_ids
                )
            pass_started_at_s = time.perf_counter()
            if sharded_replay:
                # polylogue-pzxm: build request.shard_count owned-inactive
                # generations in parallel and merge them into `generation`
                # instead of replaying `selected_raw_ids` through the single
                # writer directly. Every terminal stage below (planner
                # statistics, bulk-build repopulate, FTS parity, readiness,
                # promote) is unchanged: it observes `generation.index_path`
                # after the merge exactly as it would after a sequential
                # replay of the same raw ids.
                #
                # polylogue-uhgm interaction: the sharded path does NOT yet
                # honor mid-replay pass deadlines (`_check_pass_deadline`
                # below is wired only through the non-sharded
                # `replay_source` call's `deadline_check` param) -- a shard's
                # own `backfill_historical_revision_evidence` call has no
                # deadline threaded to it, and there is no natural per-cohort
                # checkpoint to interrupt across K concurrently-running
                # shards the way one single-writer replay has. This is
                # deliberately rejected up front instead of silently
                # ignored: `validate_rebuild_index_request` refuses
                # `shard_count > 1` together with `pass_deadline_seconds`,
                # and this defends the resumed-operation case (a deadline
                # set on transaction CREATION, before this pass's own
                # `request.shard_count` was chosen) the request-level
                # validator cannot see.
                if transaction is not None and transaction.pass_deadline_ms is not None:
                    raise ValueError(
                        f"rebuild operation {transaction.operation_id} carries a pass deadline "
                        f"({transaction.pass_deadline_ms}ms); shard_count > 1 does not yet honor mid-replay "
                        "deadlines -- resume with shard_count=1, or start a new undeadlined operation"
                    )
                from polylogue.maintenance.sharded_rebuild import replay_selected_raw_ids_sharded

                replay = await replay_selected_raw_ids_sharded(
                    root=root,
                    generation_store=generation_store,
                    generation=generation,
                    selected_raw_ids=selected_raw_ids,
                    raw_batch_size=request.raw_batch_size,
                    shard_count=request.shard_count,
                    prefetch_cache=effective_prefetch_cache,
                    provenance=provenance,
                )
            else:
                # polylogue-uhgm: the pass deadline used to be checked only AFTER
                # this whole page's replay_source() call returned, so a page that
                # expanded into a much larger authority cohort (or simply ran
                # slow) could overshoot ``pass_deadline_ms`` by an entire page --
                # live evidence: a 300s deadline, ~8-9 minute pages. This closure
                # is threaded down to backfill_historical_revision_evidence's
                # REPLAY-phase cohort loops (see RebuildDeadlineExceededError's
                # docstring), which call it between cohorts. It is a no-op for
                # every non-resumable selection (``transaction is None`` --
                # raw_ids/--only-missing/--max-blob-mb runs never carry a
                # pass_deadline_ms in the first place, see
                # validate_rebuild_index_request).
                def _check_pass_deadline() -> None:
                    if transaction is None or transaction.pass_deadline_ms is None:
                        return
                    elapsed_ms = int(time.time() * 1000) - pass_started_at_ms
                    if elapsed_ms >= transaction.pass_deadline_ms:
                        raise RebuildDeadlineExceededError(
                            f"rebuild pass deadline ({transaction.pass_deadline_ms}ms) exceeded mid-replay "
                            f"after {elapsed_ms}ms; stopping before the next cohort"
                        )

                try:
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
                        prefetch_cache=effective_prefetch_cache,
                        deadline_check=_check_pass_deadline if transaction is not None else None,
                    )
                except RebuildDeadlineExceededError as exc:
                    # Mid-page interrupt: at least one cohort was durably
                    # committed (or none, if the very first cohort tripped it),
                    # but never the whole requested page. Do NOT advance the
                    # transaction's cursor/processed counters -- leaving them at
                    # their pre-pass values means the next pass re-derives from
                    # exactly the same source-order position. Re-applying any
                    # cohort this pass DID commit is a safe idempotent no-op
                    # (content-hash upsert), so this can never duplicate or skip
                    # a raw/cohort; it can only redo bounded work.
                    assert transaction is not None  # deadline_check is only wired when transaction is not None
                    pass_elapsed_s = time.perf_counter() - pass_started_at_s
                    if rebuild_source_evidence_snapshot(root) != transaction.source_snapshot:
                        transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                            generation_store,
                            transaction,
                            root,
                            request.schema_inference_receipt_path,
                            status="stale",
                            error="source evidence changed during deadline-interrupted rebuild pass",
                        )
                        source_drifted = True
                        raise RuntimeError(
                            f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                        ) from exc
                    transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="deferred",
                        error=str(exc),
                    )
                    from polylogue.pipeline.services.process_pool import (
                        parallel_threads_effective,
                        resolve_parse_worker_count,
                    )

                    pass_cost = RebuildPassCost(
                        selection_s=selection_elapsed_s,
                        cohort_s=0.0,
                        replay_s=pass_elapsed_s,
                        checkpoint_s=0.0,
                        pass_s=time.perf_counter() - pass_started_at_s,
                        raws=selected_raw_count,
                        bytes_in=sum(row[2] for row in page.rows) if page is not None else 0,
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
                        deferred_reason="pass-deadline-mid-replay",
                        **pass_cost.to_dict(),
                    )
                    pass_receipt = RebuildIndexReceipt(
                        archive_root=str(root),
                        raw_session_count=raw_count,
                        selected_raw_count=selected_raw_count,
                        skipped_by_blob_limit_count=0,
                        status="deferred",
                        materialized=False,
                        materialization={},
                        generation=cast(dict[str, object], asdict(generation)),
                        readiness={},
                        # Zero-shaped like a normal replay dict (not merely a bare
                        # marker) so every existing consumer that indexes
                        # replay-dict keys unconditionally (the daemon-mode CLI
                        # text formatter, the daemon HTTP route's raw
                        # ``receipt.to_dict()`` response) keeps working: this pass
                        # made no *measured* progress by design (see the "do not
                        # advance the cursor" comment above), so reporting zeros
                        # here is accurate, not merely a placeholder shape.
                        replay={
                            "scanned_raw_count": 0,
                            "classified_full_count": 0,
                            "replayed_logical_source_count": 0,
                            "quarantined_raw_count": 0,
                            "adoption_deferred_raw_count": 0,
                            "authority_selection_expanded": True,
                            "scheduled_raw_count": len(selected_raw_ids),
                            "raw_batch_size": request.raw_batch_size,
                            "ingest_workers": None,
                            "parse_s": 0.0,
                            "apply_s": 0.0,
                            "stage_timings_s": {},
                            "deferred_reason": "pass-deadline-mid-replay",
                        },
                        transaction=cast(dict[str, object], asdict(transaction)),
                        operation=_operation_evidence(
                            root, generation=generation, transaction=transaction, recovery_state="deferred"
                        ),
                        selection_evidence=selection_evidence,
                        timings_s=cast(dict[str, float], pass_cost.to_dict()),
                        consumed_evidence=consumed_evidence,
                    )
                    _save_rebuild_pass_receipt_after_receipt_validation(
                        generation_store,
                        transaction.operation_id,
                        pass_receipt,
                        root,
                        request.schema_inference_receipt_path,
                    )
                    return pass_receipt
            pass_elapsed_s = time.perf_counter() - pass_started_at_s
            processed_before = transaction.processed_raw_count if transaction is not None else None
            _validate_before_derived_state(provenance)
            if selected_raw_ids and _should_refresh_generation_planner_statistics(
                processed_before=processed_before,
                processed_after=(processed_before or 0) + len(selected_raw_ids),
            ):
                _refresh_generation_planner_statistics(Path(generation.index_path))
            if transaction is not None and selected_raw_ids:
                if rebuild_source_evidence_snapshot(root) != transaction.source_snapshot:
                    if transaction_created_here:
                        raise RebuildProvenanceError(
                            "rebuild schema-inference preflight gate failed: "
                            "source evidence changed during this bounded rebuild pass"
                        )
                    _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
                    transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="stale",
                        error="source evidence changed during this bounded rebuild pass",
                    )
                    source_drifted = True
                    raise RuntimeError(
                        f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                    )
                assert page is not None
                last_raw_id, last_blob_hash_hex, _blob_size = page.rows[-1]
                elapsed_ms = int(time.time() * 1000) - pass_started_at_ms
                deadline_expired = (
                    transaction.pass_deadline_ms is not None and elapsed_ms >= transaction.pass_deadline_ms
                )
                status = "deferred" if page.deferred_reason == "byte-budget" or deadline_expired else "paused"
                transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                    generation_store,
                    transaction,
                    root,
                    request.schema_inference_receipt_path,
                    status=status,
                    last_blob_hash_hex=last_blob_hash_hex,
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
                        selection_s=selection_elapsed_s,
                        cohort_s=_cohort_seconds(replay.get("stage_timings_s", {})),
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
                        parse_s=cast(float, replay.get("parse_s", 0.0)),
                        apply_s=cast(float, replay.get("apply_s", 0.0)),
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
                        operation=_operation_evidence(
                            root, generation=generation, transaction=transaction, recovery_state=status
                        ),
                        selection_evidence=selection_evidence,
                        timings_s=cast(dict[str, float], pass_cost.to_dict()),
                        consumed_evidence=consumed_evidence,
                    )
                    _save_rebuild_pass_receipt_after_receipt_validation(
                        generation_store,
                        transaction.operation_id,
                        pass_receipt,
                        root,
                        request.schema_inference_receipt_path,
                    )
                    return pass_receipt
            # polylogue-o56w: terminal-stage costs used to survive only as log
            # lines; collect them here and persist them on the final receipt
            # so a full rebuild's cost breakdown is durable forensics.
            #
            # polylogue-6mvg: ``selection_s`` (this pass's raw-id/page
            # selection cost, measured above before replay even started) is
            # folded in here too -- extending the SAME receipt vocabulary
            # the deferred/paused ``RebuildPassCost`` timings already use,
            # not a parallel one, so every receipt shape (deferred, paused,
            # replayed) carries the identical key for this phase.
            terminal_timings_s: dict[str, float] = {"selection_s": selection_elapsed_s}
            # Census can create membership rows and logical-source keys while
            # replay is running. Recompute the receipt commitment from the
            # post-replay source tier so it names the closure the candidate
            # actually consumed, not only the caller's raw-id hints.
            selection_evidence = rebuild_selection_evidence(
                selected_raw_ids,
                archive_root=root,
                generation_id=generation.generation_id,
                generation_owner_id=generation.owner_id,
                candidate_index=Path(generation.index_path),
                source_snapshot=generation.source_snapshot,
            )
            if rebuild_source_evidence_snapshot(root) != generation.source_snapshot:
                if transaction is not None and transaction_created_here:
                    raise RebuildProvenanceError(
                        "rebuild schema-inference preflight gate failed: "
                        "source evidence changed before terminal readiness"
                    )
                if transaction is not None:
                    _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
                    transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="stale",
                        error="source evidence changed before terminal readiness",
                    )
                    source_drifted = True
                raise RuntimeError(f"source evidence changed while rebuilding {generation.generation_id}")
            source_evidence_after = rebuild_source_evidence_snapshot(root)
            # polylogue-v6i3: bulk-build replay (bulk_build=True above) left
            # messages_fts/blocks_command_trigram/action_pairs/delegation_facts
            # empty or stale for every session -- repopulate all four
            # archive-wide exactly once here, then prove exact parity before
            # readiness can observe (and silently accept) a mismatch.
            _validate_before_derived_state(provenance)
            bulk_timings_s = _repopulate_bulk_build_derived_state(Path(generation.index_path))
            for stage, elapsed_s in bulk_timings_s.items():
                terminal_timings_s[f"terminal.bulk_build.{stage}"] = elapsed_s
                logger.info(
                    "rebuild_terminal_stage_complete",
                    generation_id=generation.generation_id,
                    stage=f"bulk_build.{stage}",
                    elapsed_s=round(elapsed_s, 3),
                )
            terminal_started_at = time.perf_counter()
            # polylogue-t0m73: the index-only reindex acceptance gate -- every
            # ground-truth check whose universe is satisfiable from a
            # generation directory's index.db alone (source.db/user.db/
            # embeddings.db live once at the archive root, not per
            # generation, so cross-tier checks are excluded; see
            # REINDEX_ACCEPTANCE_CHECKS' docstring). This subsumed the older
            # fts-parity-only gate.
            acceptance_reports = (
                verify_archive(generation_root, checks=REINDEX_ACCEPTANCE_CHECKS),
                # polylogue-f1vg: an inactive generation has only index.db, so
                # corpus fidelity combines that candidate with the durable
                # source tier at the archive root before promotion.
                verify_archive(
                    root,
                    checks=(
                        request.candidate_acceptance_checks
                        if request.candidate_acceptance_checks is not None
                        else REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS
                    ),
                    index_path_override=Path(generation.index_path),
                ),
            )
            terminal_timings_s["terminal.reindex_acceptance"] = time.perf_counter() - terminal_started_at
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="reindex_acceptance",
                elapsed_s=round(terminal_timings_s["terminal.reindex_acceptance"], 3),
            )
            acceptance_requirements = (
                REINDEX_ACCEPTANCE_CHECKS,
                request.candidate_acceptance_checks
                if request.candidate_acceptance_checks is not None
                else REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
            )
            acceptance_failures = tuple(
                failure
                for report, required_checks in zip(acceptance_reports, acceptance_requirements, strict=True)
                if not passes_strict_acceptance(report, required_checks=required_checks)
                for failure in strict_acceptance_failures(report, required_checks=required_checks)
            )
            if acceptance_failures:
                failing = "; ".join(acceptance_failures)
                raise RuntimeError(
                    f"reindex acceptance gate failed for generation {generation.generation_id}: {failing}"
                )
            # Derived insight materialization assumes a coherent lineage graph.
            # Reject a structurally invalid inactive candidate before invoking
            # it, so the acceptance receipt names the actual bad invariant
            # instead of reporting an incidental derived-model failure.
            terminal_started_at = time.perf_counter()
            _validate_before_derived_state(provenance)
            insight_result = repair_session_insights(
                config,
                dry_run=False,
                archive_root_override=generation_root,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
            )
            terminal_timings_s["terminal.session_insights"] = time.perf_counter() - terminal_started_at
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="session_insights",
                elapsed_s=round(terminal_timings_s["terminal.session_insights"], 3),
            )
            if not insight_result.success:
                raise RuntimeError(f"session insight materialization failed: {insight_result.detail}")
            terminal_started_at = time.perf_counter()
            readiness = archive_readiness_status(generation_root)
            terminal_timings_s["terminal.readiness"] = time.perf_counter() - terminal_started_at
            logger.info(
                "rebuild_terminal_stage_complete",
                generation_id=generation.generation_id,
                stage="readiness",
                elapsed_s=round(terminal_timings_s["terminal.readiness"], 3),
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
                transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                    generation_store,
                    transaction,
                    root,
                    request.schema_inference_receipt_path,
                    status="ready",
                )
            if request.promote:
                # Re-prove ownership immediately before the activation swap:
                # a long-running rebuild pass can outlast a concurrent
                # promotion of a different generation, and this must be
                # caught before clobbering someone else's activation rather
                # than after (polylogue-ovme.2 AC3).
                assert_owns_archive_location(owned, ArchiveLocation.resolve(root))
                _validate_rebuild_provenance_receipt(root, request.schema_inference_receipt_path)
                terminal_started_at = time.perf_counter()
                generation = generation_store.promote(generation)
                terminal_timings_s["terminal.promote"] = time.perf_counter() - terminal_started_at
                logger.info(
                    "rebuild_terminal_stage_complete",
                    generation_id=generation.generation_id,
                    stage="promote",
                    elapsed_s=round(terminal_timings_s["terminal.promote"], 3),
                )
                if transaction is not None:
                    transaction = _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="promoted",
                    )
        except Exception as exc:
            fresh_provenance_failure = isinstance(exc, RebuildProvenanceError) and (
                transaction_created_here or transaction is None or sharded_replay
            )
            if fresh_provenance_failure:
                if transaction is not None:
                    cleanup_errors = _discard_transaction_after_provenance_failure(
                        generation_store, transaction, provenance
                    )
                else:
                    cleanup_errors = _discard_generation_after_provenance_failure(
                        generation_store, generation, provenance
                    )
                _add_cleanup_failure_notes(exc, cleanup_errors, label="rebuild provenance")
            elif transaction is not None and not source_drifted:
                try:
                    # A process can fail after ``checkpoint_transaction`` has
                    # atomically published the output batch but before this
                    # frame receives its returned replacement object.  Never
                    # let this stale local value rewind that committed cursor
                    # while recording recovery state.
                    transaction = generation_store.load_transaction(transaction.operation_id)
                    _checkpoint_rebuild_transaction_after_receipt_validation(
                        generation_store,
                        transaction,
                        root,
                        request.schema_inference_receipt_path,
                        status="failed",
                        error="bounded rebuild pass failed; candidate retained for diagnosis or explicit recovery",
                    )
                except BaseException as checkpoint_error:
                    exc.add_note(
                        "resumable rebuild failure-state checkpoint also failed: "
                        f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                    )
            elif transaction is None:
                _cleanup_nonresumable_generation_failure(
                    generation_store,
                    generation,
                    root=root,
                    receipt_path=request.schema_inference_receipt_path,
                    consumed_evidence=consumed_evidence,
                    primary=exc,
                )
            raise
    try:
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
            operation=_operation_evidence(
                root,
                generation=generation,
                transaction=transaction,
                recovery_state="promoted" if request.promote else "ready",
            ),
            selection_evidence=selection_evidence,
            source_evidence_after=source_evidence_after,
            timings_s=_receipt_timings(
                selection_s=selection_elapsed_s,
                replay=replay,
                terminal_timings_s=terminal_timings_s,
            ),
            consumed_evidence=consumed_evidence,
        )
        _persist_candidate_receipt(generation, final_receipt.to_dict())
        if transaction is not None:
            _save_rebuild_pass_receipt_after_receipt_validation(
                generation_store,
                transaction.operation_id,
                final_receipt,
                root,
                request.schema_inference_receipt_path,
            )
    except BaseException as exc:
        if transaction is None:
            _cleanup_nonresumable_generation_failure(
                generation_store,
                generation,
                root=root,
                receipt_path=request.schema_inference_receipt_path,
                consumed_evidence=consumed_evidence,
                primary=exc,
            )
        raise
    return final_receipt


def rebuild_index_from_source_sync(request: RebuildIndexRequest) -> RebuildIndexReceipt:
    """Synchronous adapter for offline CLI callers."""
    return asyncio.run(rebuild_index_from_source(request))


def rebuild_status(
    archive_root: Path,
    *,
    operation_id: str | None = None,
    include_daemon_bulk_rebuild: bool = True,
) -> dict[str, object]:
    """Consolidated raw-replay rebuild status for operator/agent surfaces.

    polylogue-b5l.1 AC5: one read gives lease ownership, the active
    generation, the resumable transaction's cursor/delta, and explicit
    stale-lock/failed-transaction recovery guidance -- instead of an operator
    hand-cross-referencing ``.index-rebuild.lock``, ``.index-active-pointer``,
    and a transaction JSON file under ``.index-rebuild-transactions/``.

    ``operation_id`` selects which persisted transaction to report. When
    omitted and ``include_daemon_bulk_rebuild`` is True (the default), this
    falls back to the daemon's own well-known bulk-rebuild operation id
    (``DAEMON_BULK_REBUILD_OPERATION_ID``) -- the common case for
    ``ops reset --index && polylogued run``, where the daemon never has an
    explicit operation id to hand the caller. Read-only throughout: never
    acquires ``RebuildLease``, never mutates any transaction or generation.
    """
    from polylogue.daemon.bulk_rebuild import DAEMON_BULK_REBUILD_OPERATION_ID
    from polylogue.storage.index_generation import (
        IndexGenerationStore,
        rebuild_lease_status,
        rebuild_source_evidence_snapshot,
    )

    location = ArchiveLocation.resolve(archive_root)
    lease = rebuild_lease_status(archive_root)
    store = IndexGenerationStore(location)

    active_generation: dict[str, object] | None = None
    try:
        active_target = store.active_pointer.resolve(strict=True)
    except OSError:
        active_target = None
    if active_target is not None:
        for metadata_path in store.generations_root.glob("gen-*/generation.json"):
            try:
                generation = store.load(metadata_path.parent.name)
                if generation.state == "active" and Path(generation.index_path).resolve(strict=True) == active_target:
                    active_generation = cast(dict[str, object], asdict(generation))
                    break
            except (OSError, ValueError, TypeError):
                continue

    schema_version: int | None = None
    try:
        with contextlib.closing(sqlite3.connect(f"file:{store.active_pointer}?mode=ro", uri=True, timeout=5.0)) as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
        schema_version = int(row[0]) if row is not None else None
    except sqlite3.Error:
        schema_version = None

    resolved_operation_id = operation_id
    if resolved_operation_id is None and include_daemon_bulk_rebuild:
        resolved_operation_id = DAEMON_BULK_REBUILD_OPERATION_ID

    transaction_payload: dict[str, object] | None = None
    delta: dict[str, object] | None = None
    transaction = None
    if resolved_operation_id is not None:
        try:
            transaction = store.load_transaction(resolved_operation_id)
        except FileNotFoundError:
            transaction = None
        except (OSError, ValueError, TypeError, KeyError):
            transaction = None
        if transaction is not None:
            transaction_payload = cast(dict[str, object], asdict(transaction))
            current_snapshot = (
                rebuild_source_evidence_snapshot(archive_root) if (archive_root / "source.db").exists() else None
            )
            delta = {
                "source_snapshot_matches": (
                    current_snapshot is not None and current_snapshot == transaction.source_snapshot
                ),
                "current_source_snapshot": current_snapshot,
                "transaction_source_snapshot": transaction.source_snapshot,
            }

    recovery: list[str] = []
    if lease.stale:
        recovery.append(
            f"lease lock file records dead pid={lease.holder_pid} host={lease.holder_host!r}; "
            "the next RebuildLease acquisition reclaims it automatically -- no manual action required "
            "unless a fresh attempt still refuses"
        )
    if transaction_payload is not None and transaction_payload.get("status") == "failed":
        recovery.append(
            f"transaction {resolved_operation_id!r} is failed: {transaction_payload.get('error')!r}; "
            "resume with the same --operation-id to retry the same candidate, or discard it to start fresh"
        )
    if delta is not None and delta.get("source_snapshot_matches") is False:
        recovery.append(
            f"transaction {resolved_operation_id!r} source snapshot no longer matches current source.db; "
            "the next pass against this operation id will refuse as stale -- start a new operation"
        )

    return {
        "archive_root": str(archive_root),
        "lease": lease.to_dict(),
        "generation": active_generation,
        "schema_version": schema_version,
        "operation_id": resolved_operation_id,
        "transaction": transaction_payload,
        "operation": _operation_evidence(
            archive_root,
            generation=None,
            transaction=transaction,
            recovery_state=(str(transaction_payload["status"]) if transaction_payload is not None else "idle"),
        ),
        "delta": delta,
        "recovery": recovery,
    }


__all__ = [
    "RebuildIndexReceipt",
    "RebuildIndexRequest",
    "all_index_rebuild_raw_ids",
    "count_source_raw_sessions",
    "filter_raw_ids_by_max_blob_size",
    "missing_index_raw_ids",
    "rebuild_index_from_source",
    "rebuild_index_from_source_sync",
    "rebuild_status",
    "select_rebuild_raw_ids",
    "validate_rebuild_index_request",
]
