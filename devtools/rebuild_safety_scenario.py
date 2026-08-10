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
(``_ALLOWLISTED_DERIVED_TABLES``). Logical FTS surfaces are queried through
their MATCH/LIKE paths, while only opaque FTS shadow tables are allowlisted.
An undeclared new table fails the scenario instead of silently escaping
comparison (the auto-census requirement both beads' acceptance criteria
name).
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
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.fts.sql import message_identity_mismatch_sql
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.archive_canonical_snapshot import capture_canonical_snapshot
from tests.infra.live_ingest import write_session_sync
from tests.infra.pipeline_roundtrip import decode_source_payload, parse_payload_roundtrip
from tests.infra.rebuild_cost_model import Stratum, build_stratum_sample_corpus
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

REBUILD_SAFETY_SCENARIO_NAME = "rebuild-safety"
REBUILD_DIFFERENTIAL_SCENARIO_NAME = "rebuild-differential"

#: Columns excluded from every table's diff: wall-clock materialization
#: stamps, not content derived from source evidence. Two independently
#: timed rebuild passes of identical source content legitimately disagree
#: on *when* they ran without disagreeing on *what* they produced.
_VOLATILE_COLUMNS = frozenset({"materialized_at", "materialized_at_ms", "decided_at_ms"})

# ``session_profiles.priced_at_ms`` records when the pricing projection ran,
# not source evidence or the selected price catalog. It differs across two
# otherwise identical replay passes just like ``materialized_at_ms``.
_SESSION_PROFILE_VOLATILE_COLUMNS = _VOLATILE_COLUMNS | frozenset({"priced_at_ms"})

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
        "messages_fts_config",
        "messages_fts_data",
        "messages_fts_docsize",
        "messages_fts_idx",
        "messages_fts",
        "blocks_command_trigram_config",
        "blocks_command_trigram_data",
        "blocks_command_trigram_docsize",
        "blocks_command_trigram_idx",
        "blocks_command_trigram",
        "session_work_events_fts_config",
        "session_work_events_fts_content",
        "session_work_events_fts_data",
        "session_work_events_fts_docsize",
        "session_work_events_fts_idx",
        "session_work_events_fts",
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
    }
)

_LOGICAL_FTS_SURFACES = frozenset(
    {
        "messages_fts",
        "blocks_command_trigram",
        "session_work_events_fts",
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

_CONTENT_BEARING_FIXTURES: tuple[tuple[Provider, str], ...] = (
    (Provider.CODEX, "tests/data/codex_event_stream/tool_call_stream.jsonl"),
    (Provider.CLAUDE_CODE, "tests/fixtures/claude-code/claude-normalization-main.jsonl"),
)

_LINEAGE_WITNESS_PAYLOAD = b"\n".join(
    (
        b'{"type":"session_meta","payload":{"id":"rebuild-safety-child","timestamp":"2025-01-15T14:00:00Z","cwd":"/repo/polylogue","forked_from_id":"rebuild-safety-parent"}}',
        b'{"type":"response_item","payload":{"type":"message","id":"rebuild-safety-user","role":"user","content":[{"type":"input_text","text":"continue rebuild safety"}]}}',
        b'{"type":"response_item","payload":{"type":"function_call","id":"rebuild-safety-command","call_id":"rebuild-safety-command","name":"Bash","arguments":"{\\"command\\":\\"printf rebuild-safety-trigram\\"}"}}',
    )
)


def _seed_demo_corpus(archive_root: Path) -> list[str]:
    """Seed parser witnesses that exercise content-bearing derived relations."""
    raw_ids: list[str] = []
    for stratum in _DEMO_STRATA:
        raw_ids.extend(build_stratum_sample_corpus(archive_root, stratum, sample_n=stratum.count))
    repo_root = Path(__file__).resolve().parents[1]
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for provider, relative_path in _CONTENT_BEARING_FIXTURES:
            fixture_path = repo_root / relative_path
            raw_ids.append(
                archive.write_raw_payload(
                    provider=provider,
                    payload=fixture_path.read_bytes(),
                    source_path=f"rebuild-safety/{relative_path}",
                    acquired_at_ms=len(raw_ids) + 1,
                )
            )
        raw_ids.append(
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_LINEAGE_WITNESS_PAYLOAD,
                source_path="rebuild-safety/codex-lineage-witness.jsonl",
                acquired_at_ms=len(raw_ids) + 1,
            )
        )
    return raw_ids


def _write_attachment_witness(archive_root: Path) -> tuple[str, ...]:
    """Write a browser-acquired attachment through the production writer.

    Rebuild replay deliberately consumes source raw bytes without re-running a
    browser's authenticated attachment acquisition. The witness therefore uses
    the same preacquired-blob handoff as the live browser-ingest path, after
    each compared replay has completed.
    """
    fixture_path = Path(__file__).resolve().parents[1] / "tests/fixtures/chatgpt/native-browser-capture-v1.json"
    roundtrip = parse_payload_roundtrip("browser-capture", fixture_path.read_bytes(), "rebuild-safety-attachment")
    parsed = roundtrip.parsed.model_copy(
        update={
            "attachments": tuple(attachment for attachment in roundtrip.parsed.attachments if attachment.inline_bytes)
        }
    )
    blob_store = BlobStore(archive_root / "blob")
    preacquired: dict[int, tuple[bytes | None, int, str]] = {}
    blob_hashes: list[str] = []
    for attachment in parsed.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, byte_count = blob_store.write_from_bytes(attachment.inline_bytes)
        blob_hashes.append(blob_hash)
        preacquired[id(attachment)] = (bytes.fromhex(blob_hash), byte_count, "acquired")
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        write_parsed_session_to_archive(
            conn,
            parsed,
            content_hash=session_content_hash(parsed),
            preacquired_attachment_blobs=preacquired,
        )
    finally:
        conn.close()
    return tuple(blob_hashes)


def _discard_attachment_witness_blobs(archive_root: Path, blob_hashes: tuple[str, ...]) -> None:
    """Remove temporary witness blobs before source replay validates the store."""
    blob_store = BlobStore(archive_root / "blob")
    for blob_hash in blob_hashes:
        if not blob_store.remove(blob_hash):
            raise RuntimeError(f"attachment witness blob disappeared before cleanup: {blob_hash}")


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


def _user_tier_snapshot(archive_root: Path) -> tuple[object, ...]:
    """Capture every canonical durable-user relation and column before reset."""
    return capture_canonical_snapshot(archive_root).user_state


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
    if table == "messages_fts_identity":
        # ``rowid`` is a local SQLite allocation artifact: the same block can
        # receive a different rowid when live incremental writes arrive in a
        # different order from raw replay. Compare every durable identity
        # field exactly, and validate the rowid-to-block binding separately.
        rows = conn.execute(
            """
            SELECT block_id, source_hash, recipe_id
            FROM messages_fts_identity
            ORDER BY block_id, source_hash, recipe_id
            """
        ).fetchall()
        return tuple(tuple(row) for row in rows)
    volatile_columns = _SESSION_PROFILE_VOLATILE_COLUMNS if table == "session_profiles" else _VOLATILE_COLUMNS
    columns = [c for c in _table_columns(conn, table) if c not in volatile_columns]
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


def _fts_probe_term(conn: sqlite3.Connection, table: str) -> str | None:
    vocab_name = f"rebuild_safety_{table}_vocab"
    conn.execute(f"CREATE VIRTUAL TABLE temp.{vocab_name} USING fts5vocab(main, {table}, row)")
    try:
        row = conn.execute(f"SELECT term FROM temp.{vocab_name} WHERE doc > 0 ORDER BY term LIMIT 1").fetchone()
        return str(row[0]) if row is not None else None
    finally:
        conn.execute(f"DROP TABLE temp.{vocab_name}")


def _logical_fts_rows(conn: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    if table == "messages_fts":
        term = _fts_probe_term(conn, table)
        if term is None:
            return ()
        rows = conn.execute(
            """
            SELECT blocks.block_id
            FROM messages_fts
            JOIN blocks ON blocks.rowid = messages_fts.rowid
            WHERE messages_fts MATCH ?
            ORDER BY blocks.block_id
            """,
            (term,),
        ).fetchall()
    elif table == "blocks_command_trigram":
        # FTS5's vocabulary extension does not expose a usable probe for this
        # partial external-content trigram index. Its indexed contract is the
        # tool-detail text, so derive a real LIKE probe from a tool-use block.
        row = conn.execute(
            """
            SELECT tool_detail_text
            FROM blocks
            WHERE block_type = 'tool_use' AND tool_detail_text != ''
            ORDER BY block_id
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return ()
        term = str(row[0])
        rows = conn.execute(
            """
            SELECT block_id
            FROM blocks
            WHERE rowid IN (
                SELECT rowid
                FROM blocks_command_trigram
                WHERE tool_detail_text LIKE ?
            )
            ORDER BY block_id
            """,
            (f"%{term}%",),
        ).fetchall()
    elif table == "session_work_events_fts":
        term = _fts_probe_term(conn, table)
        if term is None:
            return ()
        rows = conn.execute(
            """
            SELECT session_work_events.event_id
            FROM session_work_events_fts
            JOIN session_work_events ON session_work_events.event_id = session_work_events_fts.event_id
            WHERE session_work_events_fts MATCH ?
            ORDER BY session_work_events.event_id
            """,
            (term,),
        ).fetchall()
    else:
        raise ValueError(f"unsupported logical FTS surface: {table}")
    return tuple((term, str(row[0])) for row in rows)


def _messages_fts_identity_is_consistent(conn: sqlite3.Connection) -> bool:
    """Require every ledger row to retain its own database's block-row binding."""
    return int(conn.execute(message_identity_mismatch_sql()).fetchone()[0]) == 0


def _diff_index_databases(index_db_a: Path, index_db_b: Path, *, scenario_name: str) -> RebuildComparisonResult:
    census = _index_table_names()
    diffable_tables = sorted(census - _ALLOWLISTED_DERIVED_TABLES)
    conn_a = sqlite3.connect(index_db_a)
    conn_b = sqlite3.connect(index_db_b)
    try:
        table_diffs = tuple(
            _diff_table(table, _dump_table(conn_a, table), _dump_table(conn_b, table)) for table in diffable_tables
        )
        logical_fts_rows_a = {table: _logical_fts_rows(conn_a, table) for table in _LOGICAL_FTS_SURFACES}
        logical_fts_rows_b = {table: _logical_fts_rows(conn_b, table) for table in _LOGICAL_FTS_SURFACES}
        fts_diffs = tuple(
            _diff_table(f"{table} logical query", logical_fts_rows_a[table], logical_fts_rows_b[table])
            for table in sorted(_LOGICAL_FTS_SURFACES)
        )
        identity_consistent_a = _messages_fts_identity_is_consistent(conn_a)
        identity_consistent_b = _messages_fts_identity_is_consistent(conn_b)
    finally:
        conn_a.close()
        conn_b.close()
    return RebuildComparisonResult(
        scenario_name=scenario_name,
        diffs=(*table_diffs, *fts_diffs),
        covered_tables=frozenset(diffable_tables),
        census_tables=census,
        extra_checks={
            "messages_fts_identity_a_is_consistent": identity_consistent_a,
            "messages_fts_identity_b_is_consistent": identity_consistent_b,
            **{
                f"{table}_logical_query_populated": bool(logical_fts_rows_a[table]) and bool(logical_fts_rows_b[table])
                for table in sorted(_LOGICAL_FTS_SURFACES)
            },
        },
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
        _seed_user_assertion(archive_root)
        before_user_tier = _user_tier_snapshot(archive_root)

        first_pass = Path(tmp) / "index-first.db"
        _full_rebuild(archive_root)
        first_witness_blobs = _write_attachment_witness(archive_root)
        (archive_root / "index.db").rename(first_pass)
        _discard_attachment_witness_blobs(archive_root, first_witness_blobs)

        second_pass = Path(tmp) / "index-second.db"
        _full_rebuild(archive_root)
        second_witness_blobs = _write_attachment_witness(archive_root)
        after_user_tier = _user_tier_snapshot(archive_root)
        (archive_root / "index.db").rename(second_pass)
        _discard_attachment_witness_blobs(archive_root, second_witness_blobs)

        result = _diff_index_databases(first_pass, second_pass, scenario_name=REBUILD_SAFETY_SCENARIO_NAME)
        extra_checks = {
            **result.extra_checks,
            "user_db_untouched": before_user_tier == after_user_tier,
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
            payload = decode_source_payload(raw_bytes)
            detected = detect_provider(payload)
            if detected is None:
                raise RuntimeError(f"incremental parser could not detect raw payload {raw_id}")
            for parsed in parse_payload(detected, payload, raw_id):
                session_ids.append(write_session_sync(index_db, parsed, raw_id=raw_id))
    finally:
        conn.close()

    insights_stage = make_insights_stage(index_db)
    execute_sessions = insights_stage.execute_sessions
    if execute_sessions is None:
        raise RuntimeError("insights convergence stage does not expose session-scoped execution")
    if not execute_sessions(session_ids):
        raise RuntimeError("insights convergence remained pending")


def run_rebuild_differential() -> RebuildComparisonResult:
    """Prove full rebuild and incremental-ingest-plus-convergence agree."""
    with TemporaryDirectory() as tmp:
        archive_root = Path(tmp) / "archive"
        initialize_active_archive_root(archive_root)
        raw_ids = _seed_demo_corpus(archive_root)
        backfill_historical_revision_evidence(archive_root, ingest_workers=1)

        full_pass = Path(tmp) / "index-full.db"
        _full_rebuild(archive_root)
        full_witness_blobs = _write_attachment_witness(archive_root)
        (archive_root / "index.db").rename(full_pass)
        _discard_attachment_witness_blobs(archive_root, full_witness_blobs)

        full_rerun_pass = Path(tmp) / "index-full-rerun.db"
        _full_rebuild(archive_root)
        rerun_witness_blobs = _write_attachment_witness(archive_root)
        (archive_root / "index.db").rename(full_rerun_pass)
        _discard_attachment_witness_blobs(archive_root, rerun_witness_blobs)

        determinism = _diff_index_databases(full_pass, full_rerun_pass, scenario_name="rebuild-determinism")

        incremental_pass = Path(tmp) / "index-incremental.db"
        _incremental_ingest_and_converge(archive_root, raw_ids)
        incremental_witness_blobs = _write_attachment_witness(archive_root)
        (archive_root / "index.db").rename(incremental_pass)
        _discard_attachment_witness_blobs(archive_root, incremental_witness_blobs)

        differential = _diff_index_databases(
            full_pass, incremental_pass, scenario_name=REBUILD_DIFFERENTIAL_SCENARIO_NAME
        )
        return RebuildComparisonResult(
            scenario_name=REBUILD_DIFFERENTIAL_SCENARIO_NAME,
            diffs=differential.diffs,
            covered_tables=differential.covered_tables,
            census_tables=differential.census_tables,
            extra_checks={
                **differential.extra_checks,
                "full_rebuild_is_deterministic": determinism.all_passed,
            },
        )


__all__ = [
    "REBUILD_DIFFERENTIAL_SCENARIO_NAME",
    "REBUILD_SAFETY_SCENARIO_NAME",
    "RebuildComparisonResult",
    "TableDiff",
    "run_rebuild_differential",
    "run_rebuild_safety",
]
