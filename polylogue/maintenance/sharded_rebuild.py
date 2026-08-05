"""Parallel sharded generation build for from-empty index rebuilds.

polylogue-pzxm: the single-writer invariant
(``connection_profile.BULK_BUILD_WRITE_CONNECTION_PROFILE``'s
``locking_mode=EXCLUSIVE`` rationale) protects an *owned inactive*
generation with exactly one writer and zero readers until
:meth:`~polylogue.storage.index_generation.IndexGenerationStore.promote`
swaps the active symlink -- it is not load-bearing across *multiple*
owned-inactive generations built concurrently, each with its own single
writer. This module exploits that: split a rebuild pass's selected raw ids
into ``shard_count`` buckets, build each bucket into its own owned inactive
generation in parallel (:func:`_build_one_shard`), then fold every shard's
``index.db`` into the pass's real target generation with one sequential
ATTACH + ``INSERT OR REPLACE ... SELECT`` merge per table
(:func:`merge_shards_into_target`), and finally re-run cross-session graph
resolution over the merged generation (:func:`resolve_cross_shard_session_graph`)
so a parent/child pair split across two shards still composes exactly like a
single-writer replay would have produced.

Non-goals (see polylogue-pzxm bead + lane brief): this module never touches
the promote()/lease machinery itself, the online (non-from-empty) single
writer contract, or ``sources/revision_backfill.py``'s replay-loop
internals -- it calls the same public
:func:`polylogue.maintenance.replay.rebuild_index_from_source` entry point
every non-sharded rebuild pass already uses, once per shard, against a
distinct owned inactive generation.

Deferred surfaces are unaffected: ``bulk_build=True`` (passed to every shard
replay, matching the non-sharded rebuild caller) leaves ``messages_fts``,
``blocks_command_trigram``, ``action_pairs``, and ``delegation_facts`` empty
throughout -- exactly like a single-writer bulk-build pass -- so those
surfaces are never merged here; the existing archive-wide
``_repopulate_bulk_build_derived_state`` terminal stage in
``maintenance/rebuild_index.py`` repopulates them once, unchanged, from the
merged ``blocks``/``messages``/``session_links`` content after this module
returns.

Measured honestly (``tests/benchmarks/test_sharded_rebuild.py``'s K=1/4/8
sweep, synthetic corpus, see the polylogue-pzxm PR body for the exact
numbers): at 96-400 synthetic raws sharding was SLOWER than the single-writer
path, not faster -- K=4 measured 0.63-0.72x and K=8 measured 0.51-0.73x of
the K=1 baseline. Per-shard generation bootstrap (a full schema DDL replay
per shard), the merge's ``PRAGMA foreign_key_check``, and the post-merge
per-session graph-resolution pass are fixed overheads that do not shrink
with more shards, and this harness's synthetic payloads make the shardable
apply-phase work too small to amortize them. Whether the real archive's
apply-phase cost (graph_resolve alone measured ~156s real-run per the bead)
is large enough to cross over into a net win is NOT yet demonstrated by this
module -- it is the natural next measurement, not assumed here.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.paths import render_root
from polylogue.storage.index_generation import IndexGeneration
from polylogue.storage.sqlite.connection_profile import BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS

if TYPE_CHECKING:
    from polylogue.maintenance.rebuild_index import RebuildProvenanceContext
    from polylogue.sources.revision_backfill import RawParsePrefetchCache
    from polylogue.storage.index_generation import IndexGenerationStore

logger = get_logger(__name__)

#: Every table ``write_parsed_session_to_archive`` writes during a
#: ``bulk_build=True`` replay pass, in the order they must be merged so
#: FOREIGN KEY enforcement (deliberately disabled for the duration of the
#: merge, see :func:`merge_shards_into_target`) never needs to matter --
#: order is cosmetic here, correctness comes from the post-merge
#: ``PRAGMA foreign_key_check``. Every one of these tables keys off
#: content-derived identity (``session_id``/``message_id``/``block_id``
#: generated columns, or an explicit composite PRIMARY KEY) -- see
#: ``storage/sqlite/archive_tiers/index.py``'s ``CREATE TABLE`` statements --
#: never a bare ``INTEGER PRIMARY KEY`` surrogate referenced elsewhere, so an
#: ``INSERT OR REPLACE ... SELECT`` merge across independently-built shards
#: is identity-safe: two shards that (via cohort-completeness expansion,
#: see ``replay.rebuild_index_from_source``'s docstring) both replay the same
#: raw converge on the same content-derived row, and REPLACE is a no-op.
#:
#: ``raw_revision_heads``/``raw_revision_applications`` are included even
#: though they hold cohort-authority bookkeeping rather than session
#: content, so a FUTURE incremental rebuild off the promoted generation
#: still sees every raw this pass applied as applied -- omitting them would
#: silently break ``--only-missing`` resumability for exactly the raws this
#: pass sharded.
MERGE_TABLES: tuple[str, ...] = (
    "raw_revision_heads",
    "raw_revision_applications",
    "sessions",
    "messages",
    "blocks",
    "session_links",
    "session_events",
    "session_agent_policies",
    "attachments",
    "attachment_refs",
    "attachment_native_ids",
    "paste_spans",
    "file_edits",
    "session_working_dirs",
    "session_refs",
    "repos",
    "repo_checkouts",
    "session_repos",
    "session_commits",
    "session_model_usage",
    "session_provider_usage_events",
    "web_content_constructs",
    "threads",
    "thread_sessions",
    "session_tags",
)

#: The subset of :data:`MERGE_TABLES` the polylogue-pzxm correctness bar
#: names explicitly ("byte-identical schema SHA, per-table row counts, and
#: content SHA over ordered sessions/messages/blocks/session_links/
#: action_pairs vs a sequential build"). ``action_pairs`` is deliberately
#: NOT in ``MERGE_TABLES`` (bulk_build leaves it empty per-shard, populated
#: only by the shared terminal repopulate stage) but IS part of the
#: equivalence surface, since it is derived from the merged content and
#: must still match byte-for-byte once that terminal stage runs.
EQUIVALENCE_TABLES: tuple[str, ...] = ("sessions", "messages", "blocks", "session_links", "action_pairs")


def _cohort_keys_for_raw_ids(
    root: Path, raw_ids: list[str], *, prefetch_cache: RawParsePrefetchCache | None
) -> dict[str, str]:
    """Map each raw id to the revision-cohort key it must stay grouped by.

    Two distinct cohort-crossing hazards exist, and only one has a cheap
    pre-parse fix:

    - Byte-growth revision chains (older/head members of the same logical
      source) are grouped by ``raw_sessions.logical_source_key``/
      ``source_path`` (``storage/sqlite/archive_tiers/source.py``) --
      exactly what ``classify_untyped_full_revision_groups``
      (revision_backfill.py, read only, not modified by this module) groups
      candidates by before calling ``revision_replay.plan_revision_replay``.
      Splitting a chain across two shards makes each shard's own census see
      a lone candidate and ``plan_revision_replay`` raises (empirically
      confirmed: "revision replay requires at least one candidate").
      ``source_path``/``logical_source_key`` are available from ``source.db``
      BEFORE any parse, since they are set at acquire/prior-classification
      time.

    - Ambiguous-identity pairs (two raws whose PARSED content both claim the
      same ``f"{origin}:{provider_session_id}"``) are grouped by that parsed
      identity, computed fresh every pass inside ``_parse_retained_raws``
      (``membership_candidates``/``provisional_full_raw_ids`` in
      revision_backfill.py) -- NOT available from ``source.db`` before a
      parse on a from-empty rebuild's first pass, where
      ``logical_source_key`` is still NULL for every raw. Splitting an
      ambiguous pair across two shards makes each shard materialize its own
      raw as an unambiguous session instead of correctly arbitrating the
      conflict (empirically confirmed: sharded row counts exceeded
      sequential row counts by exactly the split-pair count).

    ``prefetch_cache`` (the SAME ``RawParsePrefetchCache``
    ``_rebuild_index_from_source_owned`` already warms for every selected
    raw id before this pass's replay -- see
    ``rebuild_index.py::_warm_offline_prefetch_cache``) closes the second
    gap for free: it was already going to parse every selected raw once
    regardless of sharding, so reading its
    :meth:`~RawParsePrefetchCache.peek_logical_keys` here costs nothing
    extra and yields the REAL parsed identity, matching this bead's "threads
    share parsed graphs" framing -- one shared parse feeds both cohort
    partitioning and (via the same cache, still populated afterwards) each
    shard's own replay. A raw id the cache does not cover (cache is ``None``,
    admission-budget-rejected, or a multi-session bundle raw) falls back to
    the ``source.db`` key, which is exact for chains but NOT for an
    ambiguous pair split this way -- a known residual gap, see this
    function's own docstring above and the polylogue-pzxm PR body.
    """
    if not raw_ids:
        return {}
    keys: dict[str, str] = dict(prefetch_cache.peek_logical_keys()) if prefetch_cache is not None else {}
    missing = [raw_id for raw_id in raw_ids if raw_id not in keys]
    if missing:
        source_db = root / "source.db"
        if source_db.exists():
            placeholders = ",".join("?" for _ in missing)
            with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
                rows = conn.execute(
                    f"SELECT raw_id, COALESCE(logical_source_key, source_path) FROM raw_sessions "
                    f"WHERE raw_id IN ({placeholders})",
                    missing,
                ).fetchall()
            keys.update({str(raw_id): str(cohort_key) for raw_id, cohort_key in rows})
    return keys


def shard_raw_ids(
    root: Path,
    raw_ids: list[str],
    shard_count: int,
    *,
    prefetch_cache: RawParsePrefetchCache | None = None,
) -> list[list[str]]:
    """Deterministically partition ``raw_ids`` into ``shard_count`` buckets.

    Partitions by revision-cohort key (see :func:`_cohort_keys_for_raw_ids`),
    not bare ``raw_id`` -- every raw belonging to the same cohort is
    guaranteed to land in the same bucket, so a bucket's shard-local replay
    always sees a complete cohort, matching what a sequential replay of the
    same raw ids would have seen. A raw id with no cohort evidence at all
    (missing from both the prefetch cache and ``source.db``) falls back to
    its own raw_id as a singleton cohort key. Hashing (not round-robin/
    contiguous slicing) so bucket membership does not depend on query row
    order, and empty buckets are dropped by the caller rather than spun up as
    a no-op shard build.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    cohort_key_by_raw_id = _cohort_keys_for_raw_ids(root, raw_ids, prefetch_cache=prefetch_cache)
    bucket_by_cohort_key: dict[str, int] = {}
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    for raw_id in raw_ids:
        cohort_key = cohort_key_by_raw_id.get(raw_id, raw_id)
        bucket = bucket_by_cohort_key.get(cohort_key)
        if bucket is None:
            digest = hashlib.sha256(cohort_key.encode("utf-8")).digest()
            bucket = digest[0] % shard_count
            bucket_by_cohort_key[cohort_key] = bucket
        buckets[bucket].append(raw_id)
    return buckets


@dataclass(frozen=True, slots=True)
class ShardBuildStats:
    """Per-shard evidence folded into the aggregated receipt."""

    generation_id: str
    raw_count: int
    replay: dict[str, object]
    build_s: float


async def _build_one_shard(
    *,
    generation_store: IndexGenerationStore,
    source_snapshot: str,
    raw_ids: list[str],
    raw_batch_size: int,
    prefetch_cache: RawParsePrefetchCache | None,
    provenance: RebuildProvenanceContext,
    created_generations: list[IndexGeneration],
) -> tuple[IndexGeneration, ShardBuildStats]:
    """Replay one shard's raw ids into its own owned inactive generation."""
    from polylogue.maintenance.replay import rebuild_index_from_source as replay_source

    provenance.validate()
    generation = generation_store.create(source_snapshot=source_snapshot)
    created_generations.append(generation)
    provenance.validate()
    generation_root = Path(generation.index_path).parent
    config = Config(
        archive_root=generation_root,
        render_root=render_root(),
        sources=[],
        db_path=Path(generation.index_path),
    )
    started_at = time.perf_counter()
    replay = await replay_source(
        config,
        raw_ids=raw_ids,
        raw_batch_size=raw_batch_size,
        ingest_workers=None,
        materialize=True,
        progress_callback=None,
        owned_inactive_generation=(generation.generation_id, generation.owner_id),
        bulk_fts=True,
        bulk_build=True,
        prefetch_cache=prefetch_cache,
        deadline_check=provenance.validate,
    )
    provenance.validate()
    build_s = time.perf_counter() - started_at
    return generation, ShardBuildStats(
        generation_id=generation.generation_id,
        raw_count=len(raw_ids),
        replay=replay,
        build_s=build_s,
    )


def _open_merge_connection(index_path: Path, *, timeout: int) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path, timeout=timeout)
    try:
        for statement in BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS:
            if statement.strip().upper() == "PRAGMA FOREIGN_KEYS = ON":
                # Merge deliberately runs with FK enforcement OFF: a shard's
                # own sessions table can reference a parent that only exists
                # in ANOTHER shard's sessions table, and the ATTACH+INSERT
                # merge order across shards/tables is not guaranteed to place
                # parents before children within one INSERT...SELECT (SQLite
                # checks IMMEDIATE foreign keys per row, not per statement).
                # `merge_shards_into_target` re-enables enforcement and runs
                # `PRAGMA foreign_key_check` before returning, so this is a
                # scoped, verified relaxation, not a weakened invariant.
                conn.execute("PRAGMA foreign_keys = OFF")
                continue
            conn.execute(statement)
    except BaseException:
        conn.close()
        raise
    return conn


def _non_generated_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names to move in a merge, excluding GENERATED (STORED/VIRTUAL) columns.

    ``INSERT INTO t SELECT * FROM other.t`` is NOT a reliable way to skip
    generated columns across an ATTACHed database (unlike a same-connection
    self-select, it does not consistently apply SQLite's implicit-column-list
    narrowing -- observed empirically: ``sessions`` has two generated columns
    (``session_id``, ``sort_key_ms``) and a cross-attach ``SELECT *`` raised
    ``table sessions has 38 columns but 40 values were supplied``). Building
    an explicit column list from ``PRAGMA table_xinfo`` (``hidden`` 2=VIRTUAL,
    3=STORED) is correct regardless of that ATTACH-vs-self-select
    inconsistency, and self-documents as schema changes add/remove generated
    columns.
    """
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({table})").fetchall() if int(row[6]) not in (2, 3)]


def merge_shards_into_target(
    target_index_path: Path,
    shard_index_paths: list[Path],
    *,
    provenance: RebuildProvenanceContext,
) -> dict[str, int]:
    """ATTACH every shard's ``index.db`` and fold its rows into the target.

    Returns per-table total row counts written (across all shards, before
    REPLACE de-duplication -- i.e. attempted rows, not necessarily the
    target's final row count, which callers can read directly if needed).
    Raises ``RuntimeError`` if the merged generation fails
    ``PRAGMA foreign_key_check`` -- this is the load-bearing correctness
    check for the FK-relaxed merge above.
    """
    from polylogue.storage.fts.sql import FTS_BULK_SESSION_WRITE_GUARD

    row_counts: dict[str, int] = dict.fromkeys(MERGE_TABLES, 0)
    aliases = [f"shard{i}" for i in range(len(shard_index_paths))]
    provenance.validate()
    with contextlib.closing(_open_merge_connection(target_index_path, timeout=600)) as conn:
        conn.execute("PRAGMA busy_timeout = 600000")
        attached: list[str] = []
        try:
            # polylogue-pzxm: messages_fts/blocks_command_trigram's INSERT/
            # UPDATE/DELETE triggers on `blocks` are unconditional at the SQL
            # level EXCEPT for this one guard row (`storage/fts/sql.py`'s
            # `FTS_BULK_SESSION_WRITE_GUARD`, `NOT EXISTS (...)` in every
            # trigger's WHEN clause) -- a normal per-session bulk_build write
            # holds it only around that session's own block inserts. This
            # merge inserts every shard's blocks in one connection with no
            # such per-row scoping, so without holding the SAME guard for the
            # whole merge, the target's triggers fire uncontrolled and leave
            # messages_fts/blocks_command_trigram non-empty and inconsistent
            # (empirically confirmed: merged messages_fts ended up with MORE
            # rows than text-bearing blocks). Holding it here reproduces the
            # exact invariant a sequential bulk_build replay already
            # maintains (both derived stores stay empty until the shared
            # terminal `_repopulate_bulk_build_derived_state` stage), so that
            # stage's `resume_from_empty_message_index=True` precondition
            # holds for the sharded path too.
            provenance.validate()
            conn.execute(
                "INSERT OR IGNORE INTO derived_refresh_guard (guard_name) VALUES (?)", (FTS_BULK_SESSION_WRITE_GUARD,)
            )
            for alias, shard_path in zip(aliases, shard_index_paths, strict=True):
                provenance.validate()
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_path),))
                attached.append(alias)
            for table in MERGE_TABLES:
                columns = _non_generated_columns(conn, table)
                column_list = ", ".join(columns)
                for alias in aliases:
                    provenance.validate()
                    cursor = conn.execute(
                        f"INSERT OR REPLACE INTO {table} ({column_list}) SELECT {column_list} FROM {alias}.{table}"
                    )
                    row_counts[table] += cursor.rowcount if cursor.rowcount > 0 else 0
                    provenance.validate()
            provenance.validate()
            conn.execute("DELETE FROM derived_refresh_guard WHERE guard_name = ?", (FTS_BULK_SESSION_WRITE_GUARD,))
            conn.commit()
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(
                    f"sharded rebuild merge produced {len(violations)} foreign-key violations "
                    f"(first 10: {violations[:10]!r}); merged generation discarded by caller"
                )
        finally:
            for alias in attached:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"DETACH DATABASE {alias}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    return row_counts


def resolve_cross_shard_session_graph(
    target_index_path: Path,
    *,
    provenance: RebuildProvenanceContext,
) -> float:
    """Re-run per-session graph resolution over every merged session.

    Each shard's own replay already resolved links whose OTHER endpoint
    existed within the same shard (via ``write.py``'s
    ``_resolve_session_graph``, called per session at write time). A
    parent/child pair split across two shards never had its counterpart
    present during either shard's own replay, so ``session_links``/
    ``parent_session_id``/``root_session_id`` for that pair are still
    unresolved after :func:`merge_shards_into_target`. Re-running resolution
    for every session in the merged generation closes that gap; it is safe
    (not merely convenient) to do so because ``_resolve_session_graph`` is
    idempotent and fast-paths sessions whose projection is already current
    (``_root_projection_current`` in write.py), so already-intra-shard-
    resolved sessions cost only the fast-path check, not a full re-resolve.

    Returns the wall-clock seconds spent, folded into the aggregated
    receipt's ``apply_s`` by the caller (this is single-writer, serialized
    work -- it does not shard further).
    """
    from polylogue.storage.sqlite.archive_tiers.write import _resolve_session_graph

    provenance.validate()
    started_at = time.perf_counter()
    with contextlib.closing(_open_merge_connection(target_index_path, timeout=600)) as conn:
        conn.execute("PRAGMA busy_timeout = 600000")
        conn.execute("PRAGMA foreign_keys = ON")
        rows = conn.execute("SELECT session_id, native_id, origin FROM sessions ORDER BY session_id").fetchall()
        signature_cache: dict[str, list[tuple[str, str]]] = {}
        for session_id, native_id, origin in rows:
            provenance.validate()
            _resolve_session_graph(
                conn,
                session_id,
                native_id,
                origin,
                cache=signature_cache,
                add_timing=None,
                bulk_fts=True,
                bulk_build=True,
            )
        provenance.validate()
        conn.commit()
    return time.perf_counter() - started_at


def _cleanup_shard_generations(
    generation_store: IndexGenerationStore,
    shard_generations: list[IndexGeneration],
    provenance: RebuildProvenanceContext,
) -> list[BaseException]:
    """Attempt every shard cleanup and return failures without short-circuiting."""
    errors: list[BaseException] = []
    for shard_generation in shard_generations:
        try:
            provenance.validate_cleanup()
            discarded = generation_store.discard_if_inactive(shard_generation)
            if not discarded:
                raise RuntimeError(f"{shard_generation.generation_id} was not discarded")
        except BaseException as exc:
            cleanup_error = RuntimeError(f"{shard_generation.generation_id}: {exc}")
            cleanup_error.__cause__ = exc
            errors.append(cleanup_error)
    return errors


def _surface_shard_cleanup_failures(primary: BaseException | None, cleanup_errors: list[BaseException]) -> None:
    """Preserve a primary failure while surfacing every cleanup failure."""
    if not cleanup_errors:
        return
    detail = "; ".join(str(error) for error in cleanup_errors)
    if primary is None:
        raise RuntimeError(f"shard cleanup failed: {detail}") from cleanup_errors[0]
    primary.add_note(f"shard cleanup also failed: {detail}")


async def replay_selected_raw_ids_sharded(
    *,
    root: Path,
    generation_store: IndexGenerationStore,
    generation: IndexGeneration,
    selected_raw_ids: list[str],
    raw_batch_size: int,
    shard_count: int,
    prefetch_cache: RawParsePrefetchCache | None,
    provenance: RebuildProvenanceContext,
) -> dict[str, object]:
    """Replay ``selected_raw_ids`` into ``generation`` via ``shard_count`` parallel shards.

    Drop-in replacement for a single ``replay.rebuild_index_from_source``
    call targeting the SAME already-created ``generation`` -- callers (see
    ``maintenance/rebuild_index.py``) keep every terminal stage
    (planner-statistics refresh, ``_repopulate_bulk_build_derived_state``,
    FTS parity, readiness, promote) unchanged; this function's only
    contract is that ``generation.index_path`` ends up holding exactly what
    a sequential replay of ``selected_raw_ids`` would have written to the
    tables in :data:`EQUIVALENCE_TABLES`, after its terminal stage runs.

    Shard generations are always discarded (``discard_if_inactive``) before
    returning, success or failure -- they are scratch, never promotable.
    """
    buckets = [
        bucket for bucket in shard_raw_ids(root, selected_raw_ids, shard_count, prefetch_cache=prefetch_cache) if bucket
    ]
    if not buckets:
        return {
            "scanned_raw_count": 0,
            "classified_full_count": 0,
            "replayed_logical_source_count": 0,
            "quarantined_raw_count": 0,
            "adoption_deferred_raw_count": 0,
            "authority_selection_expanded": True,
            "scheduled_raw_count": 0,
            "raw_batch_size": raw_batch_size,
            "ingest_workers": 0,
            "parse_s": 0.0,
            "apply_s": 0.0,
            "stage_timings_s": {},
            "shard_count": 0,
        }
    logger.info(
        "sharded_rebuild_build_start",
        generation_id=generation.generation_id,
        shard_count=len(buckets),
        selected_raw_count=len(selected_raw_ids),
    )
    created_generations: list[IndexGeneration] = []
    build_results = await asyncio.gather(
        *[
            _build_one_shard(
                generation_store=generation_store,
                source_snapshot=generation.source_snapshot,
                raw_ids=bucket,
                raw_batch_size=raw_batch_size,
                prefetch_cache=prefetch_cache,
                provenance=provenance,
                created_generations=created_generations,
            )
            for bucket in buckets
        ],
        return_exceptions=True,
    )
    failures = [result for result in build_results if isinstance(result, BaseException)]
    if failures:
        primary = failures[0]
        cleanup_errors = _cleanup_shard_generations(generation_store, created_generations, provenance)
        _surface_shard_cleanup_failures(primary, cleanup_errors)
        raise primary
    shard_generations = [cast(tuple[IndexGeneration, ShardBuildStats], result)[0] for result in build_results]
    shard_stats = [cast(tuple[IndexGeneration, ShardBuildStats], result)[1] for result in build_results]
    try:
        merge_started_at = time.perf_counter()
        row_counts = merge_shards_into_target(
            Path(generation.index_path), [Path(sg.index_path) for sg in shard_generations], provenance=provenance
        )
        merge_s = time.perf_counter() - merge_started_at
        graph_resolve_s = resolve_cross_shard_session_graph(Path(generation.index_path), provenance=provenance)
    except BaseException as primary:
        cleanup_errors = _cleanup_shard_generations(generation_store, shard_generations, provenance)
        _surface_shard_cleanup_failures(primary, cleanup_errors)
        raise
    else:
        cleanup_errors = _cleanup_shard_generations(generation_store, shard_generations, provenance)
        _surface_shard_cleanup_failures(None, cleanup_errors)
    logger.info(
        "sharded_rebuild_merge_complete",
        generation_id=generation.generation_id,
        shard_count=len(buckets),
        merge_s=round(merge_s, 3),
        graph_resolve_s=round(graph_resolve_s, 3),
        row_counts=row_counts,
    )
    parse_s = max((cast(float, s.replay.get("parse_s", 0.0)) for s in shard_stats), default=0.0)
    shard_apply_s = max((cast(float, s.replay.get("apply_s", 0.0)) for s in shard_stats), default=0.0)
    apply_s = shard_apply_s + merge_s + graph_resolve_s
    stage_timings_s: dict[str, float] = {"shard.merge_s": merge_s, "shard.graph_resolve_s": graph_resolve_s}
    for stats in shard_stats:
        stage_timings_s[f"shard.{stats.generation_id}.build_s"] = stats.build_s
        for key, value in cast(dict[str, float], stats.replay.get("stage_timings_s", {})).items():
            stage_timings_s[f"shard.{stats.generation_id}.{key}"] = value
    ingest_workers = int(cast(int, shard_stats[0].replay.get("ingest_workers", 0))) if shard_stats else 0
    return {
        "scanned_raw_count": sum(cast(int, s.replay.get("scanned_raw_count", 0)) for s in shard_stats),
        "classified_full_count": sum(cast(int, s.replay.get("classified_full_count", 0)) for s in shard_stats),
        "replayed_logical_source_count": sum(
            cast(int, s.replay.get("replayed_logical_source_count", 0)) for s in shard_stats
        ),
        "quarantined_raw_count": sum(cast(int, s.replay.get("quarantined_raw_count", 0)) for s in shard_stats),
        "adoption_deferred_raw_count": sum(
            cast(int, s.replay.get("adoption_deferred_raw_count", 0)) for s in shard_stats
        ),
        "authority_selection_expanded": True,
        "scheduled_raw_count": len(selected_raw_ids),
        "raw_batch_size": raw_batch_size,
        "ingest_workers": ingest_workers,
        "parse_s": round(parse_s, 6),
        "apply_s": round(apply_s, 6),
        "stage_timings_s": {key: round(value, 6) for key, value in stage_timings_s.items()},
        "shard_count": len(buckets),
        "shard_row_counts": row_counts,
    }
