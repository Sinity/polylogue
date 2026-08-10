"""Derived-tier rebuild-safety and rebuild-differential lab scenarios.

Two related, evidence-driven proof obligations that neither had an
executable lane before this module (polylogue-1xc.8, polylogue-hjwr):

1. **rebuild-safety** (1xc.8): resetting the derived tier (``index.db``) and
   rebuilding it from durable ``source.db`` evidence must be lossless and
   idempotent -- the same source replayed twice produces the same derived
   content, and the durable ``user.db`` tier (assertions, corrections) is
   never touched by a derived-tier reset.
2. **rebuild-differential** (hjwr): the two ways a session's derived rows
   come to exist -- a full from-scratch rebuild
   (``polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync``)
   versus live incremental ingest (per-session
   ``write_parsed_session_to_archive``) followed by the daemon's own
   convergence-stage catch-up (``daemon.convergence_stages.make_insights_stage``)
   -- must agree on the derived content they produce. ``polylogue-a7xr.2``
   (closed) was exactly a case where they didn't: the converger and
   ``storage/repair.py`` encoded different staleness predicates for
   ``session_profiles``, an inconsistency this scenario is the general,
   ongoing gate against a *class* of, not merely that one instance.

Both scenarios share one seed corpus (real, parseable Codex/Claude Code raws
via ``tests.infra.rebuild_cost_model.build_stratum_sample_corpus`` -- the
same synthesis already used by the stratified rebuild-cost benchmark, not a
new generator) and one census/diff engine: every derived (``index.db``)
table is either diffed or explicitly allowlisted with a documented reason
(``_ALLOWLISTED_DERIVED_TABLES``); an undeclared new table fails the
scenario instead of silently escaping comparison (the auto-census
requirement both beads' acceptance criteria name).
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.core.enums import Provider
from polylogue.daemon.convergence_stages import make_insights_stage
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.live_ingest import write_session_sync
from tests.infra.pipeline_roundtrip import parse_payload_roundtrip
from tests.infra.rebuild_cost_model import Stratum, build_stratum_sample_corpus
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

REBUILD_SAFETY_SCENARIO_NAME = "rebuild-safety"
REBUILD_DIFFERENTIAL_SCENARIO_NAME = "rebuild-differential"

#: Columns excluded from every table's diff: wall-clock materialization
#: stamps, not content derived from source evidence. Two independently
#: timed rebuild passes of identical source content legitimately disagree
#: on *when* they ran without disagreeing on *what* they produced.
_VOLATILE_COLUMNS = frozenset({"materialized_at", "materialized_at_ms", "decided_at_ms"})

#: Tables intentionally excluded from the row-level diff, each for a
#: documented reason -- the census in ``_assert_every_table_covered`` fails
#: the scenario if a new derived table appears in neither this set nor the
#: diffed-table set, so growth in ``archive_tiers`` DDL cannot silently
#: escape this lane's coverage.
_ALLOWLISTED_DERIVED_TABLES = frozenset(
    {
        # FTS5 virtual tables and their opaque shadow-table internals: binary
        # segment/index formats, not row content meaningfully comparable
        # across two independently built FTS indexes of the same logical text.
        "messages_fts",
        "messages_fts_config",
        "messages_fts_data",
        "messages_fts_docsize",
        "messages_fts_idx",
        "messages_fts_identity",
        "blocks_command_trigram",
        "blocks_command_trigram_config",
        "blocks_command_trigram_data",
        "blocks_command_trigram_docsize",
        "blocks_command_trigram_idx",
        "session_work_events_fts",
        "session_work_events_fts_config",
        "session_work_events_fts_data",
        "session_work_events_fts_docsize",
        "session_work_events_fts_idx",
        "threads_fts",
        "threads_fts_config",
        "threads_fts_content",
        "threads_fts_data",
        "threads_fts_docsize",
        "threads_fts_idx",
        # Seeded fixed reference data (pricing catalog), not derived from this
        # scenario's source corpus at all -- identical on every fresh init.
        "price_catalogs",
        # Pure convergence/refresh bookkeeping: run-scoped progress markers,
        # not content derived from source raws.
        "fts_freshness_state",
        "derived_refresh_guard",
        "query_unit_frame_state",
        # Replay provenance receipts are emitted by the full source replay
        # engine, while the incremental index writer intentionally consumes
        # already-frozen source authority without recreating those receipts.
        # They are source/replay bookkeeping, not derived content; the
        # source census and the content-bearing index tables remain compared.
        "raw_revision_applications",
        "raw_revision_heads",
        "insight_materialization",
    }
)


@dataclass(frozen=True, slots=True)
class TableDiff:
    table: str
    only_in_a: tuple[tuple[object, ...], ...]
    only_in_b: tuple[tuple[object, ...], ...]

    @property
    def is_empty(self) -> bool:
        return not self.only_in_a and not self.only_in_b


@dataclass(frozen=True, slots=True)
class RebuildComparisonResult:
    scenario_name: str
    diffs: tuple[TableDiff, ...]
    covered_tables: frozenset[str]
    census_tables: frozenset[str]
    extra_checks: dict[str, bool] = field(default_factory=dict)

    @property
    def diverging_tables(self) -> tuple[TableDiff, ...]:
        return tuple(diff for diff in self.diffs if not diff.is_empty)

    @property
    def all_passed(self) -> bool:
        return not self.diverging_tables and all(self.extra_checks.values())

    def format_report(self) -> str:
        lines = [f"scenario: {self.scenario_name}"]
        uncovered = self.census_tables - self.covered_tables - _ALLOWLISTED_DERIVED_TABLES
        lines.append(f"census: {len(self.census_tables)} tables, {len(uncovered)} uncovered")
        if uncovered:
            lines.append(f"  UNCOVERED (neither diffed nor allowlisted): {sorted(uncovered)}")
        for diff in self.diffs:
            status = "OK" if diff.is_empty else "DIVERGED"
            lines.append(f"  {diff.table}: {status}")
            if not diff.is_empty:
                lines.append(f"    only in A: {diff.only_in_a[:5]}{'...' if len(diff.only_in_a) > 5 else ''}")
                lines.append(f"    only in B: {diff.only_in_b[:5]}{'...' if len(diff.only_in_b) > 5 else ''}")
        for name, passed in self.extra_checks.items():
            lines.append(f"  {name}: {'OK' if passed else 'FAILED'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared seed corpus
# ---------------------------------------------------------------------------

_DEMO_STRATA: tuple[Stratum, ...] = (
    Stratum("rebuild-safety/codex-session", Provider.CODEX, count=4, total_bytes=4 * 4_000),
    Stratum("rebuild-safety/claude-code-session", Provider.CLAUDE_CODE, count=4, total_bytes=4 * 3_000),
)


def _seed_demo_corpus(archive_root: Path) -> list[str]:
    """Seed a small, real, multi-origin raw corpus into a fresh archive root."""
    raw_ids: list[str] = []
    for stratum in _DEMO_STRATA:
        raw_ids.extend(build_stratum_sample_corpus(archive_root, stratum, sample_n=stratum.count))
    return raw_ids


def _seed_user_assertion(archive_root: Path) -> str:
    """Write one durable user.db assertion the rebuild must never touch."""
    user_db = archive_root / "user.db"
    conn = sqlite3.connect(user_db)
    try:
        assertion_id = "rebuild-safety-canary"
        conn.execute(
            """
            INSERT INTO assertions (
                assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms
            ) VALUES (?, 'workspace:rebuild-safety', 'note', 'rebuild-safety canary assertion', 1, 1)
            """,
            (assertion_id,),
        )
        conn.commit()
        return assertion_id
    finally:
        conn.close()


def _read_user_assertion(archive_root: Path, assertion_id: str) -> tuple[object, ...] | None:
    conn = sqlite3.connect(archive_root / "user.db")
    try:
        row = conn.execute(
            "SELECT assertion_id, target_ref, kind, body_text FROM assertions WHERE assertion_id = ?",
            (assertion_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Census + dump + diff engine
# ---------------------------------------------------------------------------


def _index_table_names() -> frozenset[str]:
    """Every table a fresh ``index.db`` DDL init declares (the census)."""
    with TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "census-index.db"
        conn = sqlite3.connect(scratch)
        try:
            initialize_archive_tier(conn, ArchiveTier.INDEX)
            return frozenset(
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            )
        finally:
            conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _dump_table(conn: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    columns = [c for c in _table_columns(conn, table) if c not in _VOLATILE_COLUMNS]
    if not columns:
        return ()
    quoted = ", ".join(f'"{c}"' for c in columns)
    rows = conn.execute(f"SELECT {quoted} FROM {table} ORDER BY {quoted}").fetchall()
    return tuple(tuple(row) for row in rows)


def _diff_table(
    table: str, dump_a: tuple[tuple[object, ...], ...], dump_b: tuple[tuple[object, ...], ...]
) -> TableDiff:
    set_a, set_b = set(dump_a), set(dump_b)
    return TableDiff(
        table=table, only_in_a=tuple(sorted(set_a - set_b, key=str)), only_in_b=tuple(sorted(set_b - set_a, key=str))
    )


def _diff_index_databases(index_db_a: Path, index_db_b: Path, *, scenario_name: str) -> RebuildComparisonResult:
    census = _index_table_names()
    diffable_tables = sorted(census - _ALLOWLISTED_DERIVED_TABLES)
    conn_a = sqlite3.connect(index_db_a)
    conn_b = sqlite3.connect(index_db_b)
    try:
        diffs = tuple(
            _diff_table(table, _dump_table(conn_a, table), _dump_table(conn_b, table)) for table in diffable_tables
        )
    finally:
        conn_a.close()
        conn_b.close()
    return RebuildComparisonResult(
        scenario_name=scenario_name,
        diffs=diffs,
        covered_tables=frozenset(diffable_tables),
        census_tables=census,
    )


@contextmanager
def _archive_root_env(archive_root: Path) -> Iterator[None]:
    """Scope ``POLYLOGUE_ARCHIVE_ROOT`` to ``archive_root`` for one call.

    Some replay internals (revision-backfill's owned-inactive-generation
    store) resolve the archive root from process config rather than the
    request object passed to them. Restores the prior value (present or
    absent) on exit so this scenario never leaks its scratch root into a
    caller's later config resolution.
    """
    prior = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = prior


def _full_rebuild(archive_root: Path) -> None:
    """Reset index.db and replay the full source.db raw population (full rebuild).

    An unfiltered ``RebuildIndexRequest`` (no ``raw_ids``/``only_missing``)
    selects every raw row via ``all_index_rebuild_raw_ids`` and is the only
    selection ``validate_rebuild_index_request`` allows to combine with
    ``promote=True`` -- a partial selection can never replace the active
    index, by the same rule ``polylogue ops maintenance rebuild-index``'s
    CLI enforces.
    """
    index_db = archive_root / "index.db"
    if index_db.exists():
        index_db.unlink()
    conn = sqlite3.connect(index_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
    finally:
        conn.close()
    receipt_path = archive_root.parent / f"{archive_root.name}-schema-inference-gate-receipt.json"
    if not receipt_path.exists():
        write_valid_rebuild_receipt(archive_root, receipt_path)
    request = RebuildIndexRequest(
        archive_root=archive_root,
        promote=True,
        schema_inference_receipt_path=receipt_path,
    )
    # The revision-backfill step inside the replay engine resolves its own
    # inactive-generation store from the process-wide Config root, not from
    # ``request.archive_root`` directly -- the same env-var scoping
    # ``tests.infra.rebuild_cost_model._run_one_rebuild_pass`` uses for the
    # identical reason.
    with _archive_root_env(archive_root):
        rebuild_index_from_source_sync(request)


# ---------------------------------------------------------------------------
# rebuild-safety (polylogue-1xc.8)
# ---------------------------------------------------------------------------


def run_rebuild_safety() -> RebuildComparisonResult:
    """Prove derived-tier reset+rebuild is lossless, idempotent, and user.db-safe."""
    with TemporaryDirectory() as tmp:
        archive_root = Path(tmp) / "archive"
        initialize_active_archive_root(archive_root)
        _raw_ids = _seed_demo_corpus(archive_root)
        backfill_historical_revision_evidence(archive_root, ingest_workers=1)
        assertion_id = _seed_user_assertion(archive_root)
        before_assertion = _read_user_assertion(archive_root, assertion_id)

        first_pass = Path(tmp) / "index-first.db"
        _full_rebuild(archive_root)
        (archive_root / "index.db").rename(first_pass)

        second_pass = Path(tmp) / "index-second.db"
        _full_rebuild(archive_root)
        (archive_root / "index.db").rename(second_pass)

        after_assertion = _read_user_assertion(archive_root, assertion_id)

        result = _diff_index_databases(first_pass, second_pass, scenario_name=REBUILD_SAFETY_SCENARIO_NAME)
        extra_checks = {
            "user_db_untouched": before_assertion is not None and before_assertion == after_assertion,
        }
        return RebuildComparisonResult(
            scenario_name=result.scenario_name,
            diffs=result.diffs,
            covered_tables=result.covered_tables,
            census_tables=result.census_tables,
            extra_checks=extra_checks,
        )


# ---------------------------------------------------------------------------
# rebuild-differential (polylogue-hjwr)
# ---------------------------------------------------------------------------


def _incremental_ingest_and_converge(archive_root: Path, raw_ids: list[str]) -> None:
    """Path B: per-session live write + explicit convergence-stage catch-up.

    Mirrors real daemon operation: a new session lands via
    ``write_parsed_session_to_archive`` one at a time (never through the
    from-scratch rebuild engine), and derived insight tables
    (``session_profiles``/``session_latency_profiles``/``session_work_events``/
    ``session_phases``/threads) are populated afterward by the SAME
    convergence stage the live daemon runs
    (``daemon.convergence_stages.make_insights_stage``), not a
    differently-derived approximation of it.
    """
    index_db = archive_root / "index.db"
    if index_db.exists():
        index_db.unlink()
    conn = sqlite3.connect(index_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
    finally:
        conn.close()

    blob_store = BlobStore(archive_root / "blob")
    source_db = archive_root / "source.db"
    session_ids: list[str] = []
    conn = sqlite3.connect(source_db)
    try:
        conn.row_factory = sqlite3.Row
        for raw_id in raw_ids:
            row = conn.execute(
                "SELECT origin, blob_hash, capture_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)
            ).fetchone()
            if row is None:
                continue
            blob_hash_hex = bytes(row["blob_hash"]).hex()
            raw_bytes = blob_store.read_all(blob_hash_hex)
            roundtrip = parse_payload_roundtrip(str(row["capture_mode"] or row["origin"]), raw_bytes, raw_id)
            session_ids.append(write_session_sync(index_db, roundtrip.parsed, raw_id=raw_id))
    finally:
        conn.close()

    insights_stage = make_insights_stage(index_db)
    insights_stage.execute_sessions(session_ids)


def run_rebuild_differential() -> RebuildComparisonResult:
    """Prove full rebuild and incremental-ingest-plus-convergence agree."""
    with TemporaryDirectory() as tmp:
        archive_root = Path(tmp) / "archive"
        initialize_active_archive_root(archive_root)
        raw_ids = _seed_demo_corpus(archive_root)
        backfill_historical_revision_evidence(archive_root, ingest_workers=1)

        full_pass = Path(tmp) / "index-full.db"
        _full_rebuild(archive_root)
        (archive_root / "index.db").rename(full_pass)

        full_rerun_pass = Path(tmp) / "index-full-rerun.db"
        _full_rebuild(archive_root)
        (archive_root / "index.db").rename(full_rerun_pass)

        determinism = _diff_index_databases(full_pass, full_rerun_pass, scenario_name="rebuild-determinism")

        incremental_pass = Path(tmp) / "index-incremental.db"
        _incremental_ingest_and_converge(archive_root, raw_ids)
        (archive_root / "index.db").rename(incremental_pass)

        differential = _diff_index_databases(
            full_pass, incremental_pass, scenario_name=REBUILD_DIFFERENTIAL_SCENARIO_NAME
        )
        return RebuildComparisonResult(
            scenario_name=REBUILD_DIFFERENTIAL_SCENARIO_NAME,
            diffs=differential.diffs,
            covered_tables=differential.covered_tables,
            census_tables=differential.census_tables,
            extra_checks={"full_rebuild_is_deterministic": determinism.all_passed},
        )


__all__ = [
    "REBUILD_DIFFERENTIAL_SCENARIO_NAME",
    "REBUILD_SAFETY_SCENARIO_NAME",
    "RebuildComparisonResult",
    "TableDiff",
    "run_rebuild_differential",
    "run_rebuild_safety",
]
