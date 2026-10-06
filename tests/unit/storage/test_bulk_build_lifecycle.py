"""Bulk-generation-build lifecycle (polylogue-v6i3): empty FTS state
throughout replay, one archive-wide repopulate at readiness.

Background: a whale offline rebuild measured a 3h+ stall doing per-session
FTS maintenance one session at a time, where an ARCHIVE-WIDE
``messages_fts`` delete-all took 28.7s (bead polylogue-v6i3). ``write_parsed_session_to_
archive(..., bulk_build=True)`` -- the offline rebuild path's mode, layered
on top of the existing ``bulk_fts`` guard-gated bulk FTS mode (#3152) --
skips ALL per-session maintenance of these two derived surfaces (not just
the whale prefix-reextract cascade #3152 already handles) and defers
everything to one archive-wide repopulate the caller runs once at readiness
(``maintenance/rebuild_index.py``'s ``_repopulate_bulk_build_derived_state``).

These tests prove: (a) ``bulk_build=True`` writes leave ``messages_fts``
empty for the written session, where mode-off leaves it populated; (b) the readiness repopulate produces byte-identical
content to trickle-mode population of the same corpus, including a
prefix-sharing lineage cascade; (c) the guard row never leaks past an
exception; (d) skipping one readiness surface makes the parity comparison
fail. ``action_pairs`` remains a compact indexed relation without duplicated
payload text.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
from polylogue.storage.fts.sql import FTS_BULK_SESSION_WRITE_GUARD
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.action_pairs import rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.archive_tiers import write as _write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.revision_application import assert_session_fts_exact_sync
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard, prepared_session_rows_from_shard
from polylogue.storage.sqlite.delegation_facts import rebuild_all_delegation_facts_sync
from polylogue.storage.sqlite.runtime_indexes import (
    DEFERRED_SECONDARY_INDEX_NAMES,
    defer_secondary_indexes_sync,
    restore_deferred_secondary_indexes_sync,
)
from polylogue.storage.sqlite.session_shard import SessionShard
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import published_fixture_index_batch, write_fixture_index_session
from tests.infra.revision_backfill_benchmark import (
    FinishedBuildMeasurement,
    build_large_parent_shared_prefix_sessions,
)

T = TypeVar("T")


def _connect(path: Path) -> sqlite3.Connection:
    # Shard transport attaches a read-only ``file:`` URI, as production's
    # archive write connection does. Keeping this fixture URI-capable makes
    # the combined fresh-shard path exercise SQLite's actual attachment mode.
    # Each Index is its archive root's active index.db; compared builds
    # each get their own root. The root is bootstrapped before any build
    # defers its secondary indexes.
    path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_archive_root(path.parent)
    conn = connect_measured(path, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _tool_pair(idx: int, position: int, *, tool_id_prefix: str = "tool") -> ParsedMessage:
    tool_id = f"{tool_id_prefix}-{idx}"
    return ParsedMessage(
        provider_message_id=f"m{idx}",
        role=Role.ASSISTANT,
        text=None,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Bash",
                tool_id=tool_id,
                tool_input={"command": f"echo {idx}"},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text=f"output {idx}",
                is_error=False,
                exit_code=0,
            ),
        ],
    )


def _text_message(idx: int, position: int, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"text{idx}",
        role=Role.USER,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _session(session_id: str, *, n_pairs: int = 2) -> ParsedSession:
    messages = [_text_message(0, 0, f"session {session_id} opening remark")]
    for i in range(n_pairs):
        messages.append(_tool_pair(i, position=i + 1, tool_id_prefix=f"{session_id}-tool"))
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=f"session {session_id}",
        messages=messages,
    )


def _archive_session_id(session: ParsedSession) -> str:
    return archive_session_id(origin_from_provider(session.source_name).value, session.provider_session_id)


def _reader_index_names(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    )


def _logical_table_digests(path: Path, *, tables: tuple[str, ...] | None = None) -> dict[str, str]:
    """Digest canonical typed row encodings without treating DDL whitespace as data."""
    table_digests: dict[str, hashlib._Hash] = {}
    active_digest: hashlib._Hash | None = None
    for line in logical_export_bytes(path, tables=tables).splitlines(keepends=True):
        record = json.loads(line)
        if isinstance(record, dict) and "table" in record:
            active_digest = hashlib.sha256()
            table_digests[str(record["table"])] = active_digest
        elif active_digest is not None:
            active_digest.update(line)
    return {table: digest.hexdigest() for table, digest in table_digests.items()}


#: Write-history state, not archive product: each relation's query-frame
#: epoch counts the writes that touched it, so two routes to the same rows
#: (per-session writes versus one shard copy) finish with different epochs.
_WRITE_HISTORY_TABLES = frozenset({"query_unit_frame_state"})


def _stable_finished_table_digests(path: Path) -> dict[str, str]:
    """Digest archive rows, excluding FTS5 implementation tables and write-history epochs."""
    with sqlite3.connect(path) as conn:
        tables = tuple(
            str(name)
            for name, sql in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if not str(sql).lstrip().upper().startswith("CREATE VIRTUAL TABLE")
            and not str(name).startswith("messages_fts_")
            and str(name) not in _WRITE_HISTORY_TABLES
        )
    return _logical_table_digests(path, tables=tables)


def _finished_output_snapshot(path: Path) -> tuple[dict[str, str], list[tuple[object, ...]]]:
    """The completed archive product, not FTS5's implementation tables."""
    with sqlite3.connect(path) as conn:
        return _stable_finished_table_digests(path), _fts_rows(conn)


def _finish_bulk_build(conn: sqlite3.Connection, *, checkpoint: bool = True) -> None:
    """Run the same reader-shape boundary the cold replay owns."""
    rebuild_fts_index_sync(conn)
    rebuild_all_action_pairs_sync(conn)
    rebuild_all_delegation_facts_sync(conn)
    if checkpoint:
        conn.commit()


def _measure(operation: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    return operation(), perf_counter() - started


def _canonical_rows_digest(rows: list[tuple[object, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _finished_output_digests(path: Path) -> dict[str, str]:
    table_digests, fts_rows = _finished_output_snapshot(path)
    return {
        **table_digests,
        "messages_fts": _canonical_rows_digest(fts_rows),
    }


def _measure_batch(
    batch: AbstractContextManager[list[str]], finish: Callable[[], object]
) -> tuple[float, float, float]:
    """Time a published batch's import, in-scope finish and committing exit."""
    _, import_seconds = _measure(batch.__enter__)
    try:
        _, finish_seconds = _measure(finish)
    except BaseException as failure:
        if not batch.__exit__(type(failure), failure, failure.__traceback__):
            raise
        raise
    _, checkpoint_seconds = _measure(lambda: batch.__exit__(None, None, None))
    return import_seconds, finish_seconds, checkpoint_seconds


def _finished_build_counts(path: Path, *, offered_count: int, deferred_count: int) -> tuple[int, int, int, int, int]:
    with sqlite3.connect(path) as conn:
        ingested_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        output_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    return offered_count, ingested_count, max(0, offered_count - ingested_count), deferred_count, output_count


def _write_fresh_shard_arm(conn: sqlite3.Connection, directory: Path, sessions: list[ParsedSession]) -> None:
    dropped = defer_secondary_indexes_sync(conn)
    assert set(dropped) == set(DEFERRED_SECONDARY_INDEX_NAMES)
    shard = prepare_session_shard(directory, sessions)
    seen: set[str] = set()
    for session in sessions:
        write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepared_session_rows_from_shard(shard.path, _archive_session_id(session)),
            fresh_build=True,
            fresh_build_batch=seen,
            bulk_build=True,
        )


def _lineage_scenario(conn: sqlite3.Connection, *, bulk_fts: bool, bulk_build: bool) -> tuple[str, str]:
    """A prefix-sharing child+parent pair, mirroring
    ``test_bulk_fts_prefix_reextract.py``'s partial-tail scenario, extended
    with tool_use/tool_result pairs so action_pairs content exists
    to compare, not just messages_fts."""
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-child",
        title="lineage child",
        parent_session_provider_id="lineage-parent",
        messages=[
            _text_message(0, 0, "hello"),
            _tool_pair(0, position=1, tool_id_prefix="child"),
            _text_message(1, 2, "child diverges here"),
        ],
    )
    child_id = write_fixture_index_session(conn, child, bulk_fts=bulk_fts, bulk_build=bulk_build)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="lineage-parent",
        title="lineage parent",
        messages=[
            _text_message(0, 0, "hello"),
            _tool_pair(0, position=1, tool_id_prefix="parent"),
            _text_message(1, 2, "parent continues alone"),
        ],
    )
    parent_id = write_fixture_index_session(conn, parent, bulk_fts=bulk_fts, bulk_build=bulk_build)
    return child_id, parent_id


def _build_corpus(conn: sqlite3.Connection, *, bulk_build: bool) -> list[str]:
    """Two independent sessions plus one prefix-sharing lineage pair."""
    session_ids = []
    for label in ("alpha", "beta"):
        session_ids.append(
            write_fixture_index_session(conn, _session(label), bulk_fts=bulk_build, bulk_build=bulk_build)
        )
    child_id, parent_id = _lineage_scenario(conn, bulk_fts=bulk_build, bulk_build=bulk_build)
    session_ids.extend([child_id, parent_id])
    return session_ids


def _fts_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    # polylogue-wohv: `messages_fts` is contentless and declares only the
    # indexed `text` column. Identity is resolved through the `blocks` rowid
    # join, exactly as every production reader does -- selecting the former
    # UNINDEXED columns straight off the index returned four NULLs, so the
    # comparison below used to prove parity on `text` alone.
    rows = conn.execute(
        """
        SELECT b.block_id, b.message_id, b.session_id, b.block_type, f.text
        FROM messages_fts AS f
        JOIN blocks AS b ON b.rowid = f.rowid
        ORDER BY b.block_id
        """
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def _action_pair_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    rows = conn.execute(
        """
        SELECT tool_use_block_id, session_id, message_id, tool_id, use_rank, tool_name,
               semantic_type, tool_command, tool_path, tool_result_block_id, is_error, exit_code
        FROM action_pairs
        ORDER BY tool_use_block_id
        """
    ).fetchall()
    return sorted(tuple(row) for row in rows)


def test_bulk_build_write_leaves_derived_surfaces_empty(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session_id = write_fixture_index_session(conn, _session("solo"), bulk_fts=True, bulk_build=True)

    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM action_pairs WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
    # The guard row must never leak past the write it protected.
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    # Real rows exist to index -- an empty derived surface here is a
    # deliberate skip, not an artifact of an empty session.
    assert conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0] > 0
    conn.close()


def test_bulk_build_off_matches_todays_per_session_population(tmp_path: Path) -> None:
    """Without bulk_build, the same session write populates the FTS surface
    immediately. The action-pairs view is available in either mode."""
    conn = _connect(tmp_path / "index.db")
    session_id = write_fixture_index_session(conn, _session("solo"))

    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM action_pairs WHERE session_id = ?", (session_id,)).fetchone()[0] > 0
    conn.close()


def test_bulk_build_readiness_repopulate_matches_trickle_mode(tmp_path: Path) -> None:
    """THE key equivalence proof: a bulk-build corpus, repopulated once at
    readiness, must be byte-identical to the same corpus built entirely in
    today's per-session trickle mode."""
    conn_trickle = _connect(tmp_path / "trickle" / "index.db")
    _build_corpus(conn_trickle, bulk_build=False)
    fts_trickle = _fts_rows(conn_trickle)
    action_pairs_trickle = _action_pair_rows(conn_trickle)
    conn_trickle.close()

    conn_bulk = _connect(tmp_path / "bulk" / "index.db")
    _build_corpus(conn_bulk, bulk_build=True)
    # Confirm the empty-throughout invariant actually held for this corpus
    # before repopulating -- otherwise the parity check below could pass
    # vacuously if bulk_build silently did nothing.
    assert conn_bulk.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert conn_bulk.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0

    rebuild_fts_index_sync(conn_bulk)
    rebuild_all_action_pairs_sync(conn_bulk)
    conn_bulk.commit()

    fts_bulk = _fts_rows(conn_bulk)
    action_pairs_bulk = _action_pair_rows(conn_bulk)

    assert fts_bulk == fts_trickle
    assert action_pairs_bulk == action_pairs_trickle
    assert fts_bulk, "corpus produced no messages_fts rows -- comparison would be vacuous"
    assert action_pairs_bulk, "corpus produced no action_pairs rows -- comparison would be vacuous"
    conn_bulk.close()


def test_bulk_build_exact_sync_assertion_accepts_empty_state_but_still_checks_triggers(tmp_path: Path) -> None:
    """``assert_session_fts_exact_sync(..., bulk_build=True)`` must not raise
    for a session left deliberately unindexed, but must still fail if the
    canonical triggers are somehow missing (the trigger-presence half of the
    proof is unaffected by bulk-build mode)."""
    conn = _connect(tmp_path / "index.db")
    session_id = write_fixture_index_session(conn, _session("solo"), bulk_fts=True, bulk_build=True)

    # Deliberately out of sync (0 indexed vs >0 expected) -- must not raise.
    assert_session_fts_exact_sync(conn, session_id, bulk_build=True)

    # Without bulk_build, the same out-of-sync state must be caught.
    with pytest.raises(RuntimeError, match="FTS proof failed"):
        assert_session_fts_exact_sync(conn, session_id, bulk_build=False)
    conn.close()


def test_bulk_build_guard_row_cleared_even_on_exception(tmp_path: Path) -> None:
    """A failure mid-write must not leave the whole-transaction guard row set."""
    conn = _connect(tmp_path / "index.db")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected write failure")

    original = _write_module._write_blocks
    _write_module._write_blocks = _boom
    try:
        with pytest.raises(RuntimeError, match="injected write failure"):
            write_fixture_index_session(conn, _session("solo"), bulk_fts=True, bulk_build=True)
    finally:
        _write_module._write_blocks = original

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM derived_refresh_guard WHERE guard_name = ?",
            (FTS_BULK_SESSION_WRITE_GUARD,),
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_fresh_build_refuses_duplicate_session_instead_of_replacing(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = _session("fresh")
    write_fixture_index_session(conn, session, fresh_build=True)
    with pytest.raises(AssertionError, match="fresh_build requires an absent session_id"):
        write_fixture_index_session(conn, session, fresh_build=True)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    conn.close()


def test_fresh_build_refuses_nonempty_generation_even_for_new_session(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    write_fixture_index_session(conn, _session("existing"))
    with pytest.raises(AssertionError, match="fresh_build requires an empty archive generation"):
        write_fixture_index_session(conn, _session("other"), fresh_build=True)
    conn.close()


def test_fresh_build_batch_allows_distinct_sessions_after_empty_check(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    seen: set[str] = set()
    write_fixture_index_session(conn, _session("first"), fresh_build=True, fresh_build_batch=seen)
    write_fixture_index_session(conn, _session("second"), fresh_build=True, fresh_build_batch=seen)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    conn.close()


def test_fresh_build_skips_stale_replace_probe_but_keeps_its_absence_assertion(tmp_path: Path) -> None:
    """Fresh mode must avoid compare work without turning its safety check into a hint."""
    conn = _connect(tmp_path / "index.db")
    statements: list[str] = []
    session = _session("fresh-timestamp").model_copy(update={"updated_at": "2026-01-01T00:00:02Z"})
    conn.set_trace_callback(statements.append)
    write_fixture_index_session(conn, session, fresh_build=True)
    conn.set_trace_callback(None)

    normalized = [statement.upper() for statement in statements]
    assert not any("SELECT UPDATED_AT_MS FROM SESSIONS" in statement for statement in normalized)
    assert any("SELECT 1 FROM SESSIONS WHERE SESSION_ID" in statement for statement in normalized)
    conn.close()


def test_deferred_secondary_indexes_round_trip_without_losing_rows(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    before = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'")
    }
    assert set(DEFERRED_SECONDARY_INDEX_NAMES) <= before
    dropped = defer_secondary_indexes_sync(conn)
    assert set(dropped) == set(DEFERRED_SECONDARY_INDEX_NAMES)
    assert not any(
        conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)).fetchone()
        for name in DEFERRED_SECONDARY_INDEX_NAMES
    )
    write_fixture_index_session(conn, _session("deferred"), fresh_build=True)
    restore_deferred_secondary_indexes_sync(conn)
    conn.commit()
    after = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'")
    }
    assert set(DEFERRED_SECONDARY_INDEX_NAMES) <= after
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    conn.close()


def test_fresh_shard_build_finishes_equivalent_to_retained_indexes(tmp_path: Path) -> None:
    """A small completed-build comparison for the combined B+C writer path.

    The retained arm uses ordinary row binding and reader indexes throughout.
    The fresh arm uses a sealed stage-A shard, has the writer ATTACH/C-copy it
    while secondary indexes are deferred, then performs the normal reader
    finalization. Canonical typed row digests compare every non-virtual table
    after both arms are finished, while the reader-index comparison proves the
    declared index set was restored.
    """
    sessions = [_session("equivalent-alpha"), _session("equivalent-beta", n_pairs=1)]

    retained_path = tmp_path / "retained" / "index.db"
    retained = _connect(retained_path)
    for session in sessions:
        write_fixture_index_session(retained, session, content_hash=str(session_content_hash(session)))
    _finish_bulk_build(retained)
    retained_indexes = _reader_index_names(retained)
    retained.close()

    fresh_path = tmp_path / "fresh-shard" / "index.db"
    fresh = _connect(fresh_path)
    _write_fresh_shard_arm(fresh, tmp_path / "shards", sessions)

    assert fresh.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
    assert fresh.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
    restore_deferred_secondary_indexes_sync(fresh)
    _finish_bulk_build(fresh)
    fresh_indexes = _reader_index_names(fresh)
    fresh.close()

    assert fresh_indexes == retained_indexes
    fresh_output = _finished_output_snapshot(fresh_path)
    retained_output = _finished_output_snapshot(retained_path)
    if fresh_output != retained_output:
        fresh_digests, fresh_fts = fresh_output
        retained_digests, retained_fts = retained_output
        differing_tables = {
            table: (retained_digests.get(table), fresh_digests.get(table))
            for table in retained_digests.keys() | fresh_digests.keys()
            if retained_digests.get(table) != fresh_digests.get(table)
        }
        assert not differing_tables and fresh_fts == retained_fts, (
            f"fresh-shard output differs from retained output: tables={differing_tables!r}, "
            f"fts_equal={fresh_fts == retained_fts}"
        )
    with sqlite3.connect(fresh_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == len(sessions)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > len(sessions)


def test_fresh_shard_finished_output_comparison_rejects_missing_finalization_or_rows(tmp_path: Path) -> None:
    """The completed-output witness fails for an unfinalized build and for a lost logical row."""
    sessions = [_session("red-twin-alpha"), _session("red-twin-beta", n_pairs=1)]

    retained_path = tmp_path / "retained" / "index.db"
    retained = _connect(retained_path)
    for session in sessions:
        write_fixture_index_session(retained, session, content_hash=str(session_content_hash(session)))
    _finish_bulk_build(retained)
    retained.close()
    expected = _finished_output_snapshot(retained_path)
    assert expected[1], "retained control produced no logical FTS rows"

    fresh_path = tmp_path / "fresh-shard" / "index.db"
    fresh = _connect(fresh_path)
    _write_fresh_shard_arm(fresh, tmp_path / "shards", sessions)
    restore_deferred_secondary_indexes_sync(fresh)
    fresh.close()
    assert _finished_output_snapshot(fresh_path) != expected

    fresh = _connect(fresh_path)
    _finish_bulk_build(fresh)
    fresh.execute("DELETE FROM blocks WHERE block_id = (SELECT block_id FROM blocks ORDER BY block_id LIMIT 1)")
    fresh.commit()
    fresh.close()
    assert _finished_output_snapshot(fresh_path) != expected


@pytest.mark.uses_real_clock("completed writer routes emit measured phase durations")
def test_finished_build_measurement_protocol_compares_completed_routes(tmp_path: Path) -> None:
    """Retain completed-route measurements with caller-owned transactions.

    Timing is descriptive, not a speed threshold. Removing a writer's
    manage_transaction=False creates extra COMMIT boundaries and fails the
    trace assertion; discarding the receipt fails its readback assertion.
    """

    def record_boundaries(conn: sqlite3.Connection) -> list[str]:
        boundaries: list[str] = []

        def record(statement: str) -> None:
            # The Index write scope opens its transaction as BEGIN IMMEDIATE,
            # and its post-commit verification opens one it rolls back; only
            # committed transactions are boundaries of the build.
            keyword = statement.strip().upper().split(" ", 1)[0]
            if keyword in {"BEGIN", "COMMIT"}:
                boundaries.append(keyword)
            elif keyword == "ROLLBACK" and boundaries and boundaries[-1] == "BEGIN":
                boundaries.pop()

        conn.set_trace_callback(record)
        return boundaries

    def complete_retained() -> FinishedBuildMeasurement:
        (sessions, conn), construction_seconds = _measure(
            lambda: (build_large_parent_shared_prefix_sessions(), _connect(tmp_path / "retained" / "index.db"))
        )
        boundaries = record_boundaries(conn)
        try:
            batch = published_fixture_index_batch(conn, sessions, archive_root=tmp_path / "retained")
            import_seconds, derived_fts_finalization_seconds, checkpoint_seconds = _measure_batch(
                batch, lambda: _finish_bulk_build(conn, checkpoint=False)
            )
            assert boundaries == ["BEGIN", "COMMIT"]
        finally:
            conn.close()
        path = tmp_path / "retained" / "index.db"
        offered, ingested, refused, deferred, output = _finished_build_counts(
            path, offered_count=len(sessions), deferred_count=0
        )
        return FinishedBuildMeasurement(
            route="retained",
            construction_seconds=construction_seconds,
            import_seconds=import_seconds,
            index_restoration_seconds=0.0,
            derived_fts_finalization_seconds=derived_fts_finalization_seconds,
            checkpoint_seconds=checkpoint_seconds,
            canonical_output_digests=_finished_output_digests(path),
            offered_count=offered,
            ingested_count=ingested,
            refused_count=refused,
            deferred_count=deferred,
            output_count=output,
        )

    def complete_deferred() -> FinishedBuildMeasurement:
        (sessions, conn), construction_seconds = _measure(
            lambda: (build_large_parent_shared_prefix_sessions(), _connect(tmp_path / "deferred" / "index.db"))
        )
        seen: set[str] = set()
        boundaries = record_boundaries(conn)
        restoration: list[float] = []

        def restore_then_finish() -> None:
            _, restored = _measure(lambda: restore_deferred_secondary_indexes_sync(conn))
            restoration.append(restored)
            _finish_bulk_build(conn, checkpoint=False)

        try:
            batch = published_fixture_index_batch(
                conn,
                sessions,
                archive_root=tmp_path / "deferred",
                before_publish=lambda: defer_secondary_indexes_sync(conn),
                fresh_build=True,
                fresh_build_batch=seen,
                bulk_build=True,
            )
            import_seconds, finish_seconds, checkpoint_seconds = _measure_batch(batch, restore_then_finish)
            (index_restoration_seconds,) = restoration
            derived_fts_finalization_seconds = finish_seconds - index_restoration_seconds
            assert boundaries == ["BEGIN", "COMMIT"]
        finally:
            conn.close()
        path = tmp_path / "deferred" / "index.db"
        offered, ingested, refused, deferred, output = _finished_build_counts(
            path, offered_count=len(sessions), deferred_count=len(sessions)
        )
        return FinishedBuildMeasurement(
            route="deferred",
            construction_seconds=construction_seconds,
            import_seconds=import_seconds,
            index_restoration_seconds=index_restoration_seconds,
            derived_fts_finalization_seconds=derived_fts_finalization_seconds,
            checkpoint_seconds=checkpoint_seconds,
            canonical_output_digests=_finished_output_digests(path),
            offered_count=offered,
            ingested_count=ingested,
            refused_count=refused,
            deferred_count=deferred,
            output_count=output,
        )

    def complete_shard() -> FinishedBuildMeasurement:
        def construct_shard() -> tuple[list[ParsedSession], sqlite3.Connection, SessionShard]:
            sessions = build_large_parent_shared_prefix_sessions()
            conn = _connect(tmp_path / "shard" / "index.db")
            return sessions, conn, prepare_session_shard(tmp_path / "shards", sessions)

        (sessions, conn, shard), construction_seconds = _measure(construct_shard)
        seen: set[str] = set()
        boundaries = record_boundaries(conn)
        try:

            def defer() -> None:
                dropped = defer_secondary_indexes_sync(conn)
                assert set(dropped) == set(DEFERRED_SECONDARY_INDEX_NAMES)

            restoration: list[float] = []

            def restore_then_finish() -> None:
                _, restored = _measure(lambda: restore_deferred_secondary_indexes_sync(conn))
                restoration.append(restored)
                _finish_bulk_build(conn, checkpoint=False)

            batch = published_fixture_index_batch(
                conn,
                sessions,
                archive_root=tmp_path / "shard",
                before_publish=defer,
                prepared_rows=[
                    prepared_session_rows_from_shard(shard.path, _archive_session_id(session)) for session in sessions
                ],
                fresh_build=True,
                fresh_build_batch=seen,
                bulk_build=True,
            )
            import_seconds, finish_seconds, checkpoint_seconds = _measure_batch(batch, restore_then_finish)
            (index_restoration_seconds,) = restoration
            derived_fts_finalization_seconds = finish_seconds - index_restoration_seconds
            assert boundaries == ["BEGIN", "COMMIT"]
        finally:
            conn.close()
        path = tmp_path / "shard" / "index.db"
        offered, ingested, refused, deferred, output = _finished_build_counts(
            path, offered_count=len(sessions), deferred_count=len(sessions)
        )
        return FinishedBuildMeasurement(
            route="shard",
            construction_seconds=construction_seconds,
            import_seconds=import_seconds,
            index_restoration_seconds=index_restoration_seconds,
            derived_fts_finalization_seconds=derived_fts_finalization_seconds,
            checkpoint_seconds=checkpoint_seconds,
            canonical_output_digests=_finished_output_digests(path),
            offered_count=offered,
            ingested_count=ingested,
            refused_count=refused,
            deferred_count=deferred,
            output_count=output,
        )

    measurements = [complete_retained(), complete_deferred(), complete_shard()]
    assert [measurement.route for measurement in measurements] == ["retained", "deferred", "shard"]
    assert len({json.dumps(measurement.canonical_output_digests, sort_keys=True) for measurement in measurements}) == 1
    assert all(measurement.canonical_output_digests["messages_fts"] for measurement in measurements)
    assert all(
        measurement.offered_count == measurement.ingested_count == measurement.output_count
        for measurement in measurements
    )
    assert all(measurement.refused_count == 0 for measurement in measurements)
    assert [measurement.deferred_count for measurement in measurements] == [0, 9, 9]
    receipt = {"routes": [asdict(measurement) for measurement in measurements]}
    emitted = emit_receipt("finished-build-protocol-synthetic", receipt)
    assert json.loads(emitted.read_text(encoding="utf-8"))["measurement"] == receipt


def test_bulk_build_anti_vacuity_repopulate_is_load_bearing(tmp_path: Path) -> None:
    """Skip the messages_fts half of the readiness repopulate and show the
    equivalence proof then fails -- confirming the parity test above is
    actually exercising ``rebuild_fts_index_sync``, not passing by accident."""
    conn_trickle = _connect(tmp_path / "trickle" / "index.db")
    _build_corpus(conn_trickle, bulk_build=False)
    fts_trickle = _fts_rows(conn_trickle)
    conn_trickle.close()

    conn_bulk = _connect(tmp_path / "bulk" / "index.db")
    _build_corpus(conn_bulk, bulk_build=True)
    # Deliberately DO NOT call rebuild_fts_index_sync here.
    rebuild_all_action_pairs_sync(conn_bulk)
    conn_bulk.commit()

    fts_bulk = _fts_rows(conn_bulk)
    assert fts_bulk != fts_trickle
    assert fts_bulk == []
    assert fts_trickle, "trickle-mode reference produced no rows -- comparison would be vacuous"
    conn_bulk.close()


#: Session-owned projections a full replace clears before rewriting. Each
#: cascades from ``sessions`` or ``messages``, so none can hold rows for a
#: session saved for the first time.
_REPLACE_PRELUDE_TABLES = (
    "session_identity_claims",
    "action_pairs",
    "web_content_constructs",
    "session_links",
    "session_tags",
)


def _prelude_deletes(statements: list[str]) -> set[str]:
    return {
        table
        for statement in statements
        for table in _REPLACE_PRELUDE_TABLES
        if statement.lstrip().upper().startswith("DELETE") and f"DELETE FROM {table} " in statement
    }


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh-build", "ordinary"])
def test_a_first_save_issues_no_replace_prelude_deletes(tmp_path: Path, fresh: bool) -> None:
    """polylogue-ctha5: a from-empty build deletes nothing it cannot match.

    A later save of an existing session still clears what it replaces.
    Anti-vacuity: run the replace prelude unconditionally (the predecessor)
    and every first save deletes from each of these tables.
    """
    conn = _connect(tmp_path / "prelude" / "index.db")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    seen: set[str] = set()
    for session in (_session("prelude-alpha"), _session("prelude-beta", n_pairs=1)):
        write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            fresh_build=fresh,
            fresh_build_batch=seen if fresh else None,
        )
    conn.commit()
    assert _prelude_deletes(statements) == set(), _prelude_deletes(statements)

    statements.clear()
    revised = _session("prelude-alpha", n_pairs=1)
    write_fixture_index_session(conn, revised, content_hash=str(session_content_hash(revised)))
    conn.commit()
    conn.set_trace_callback(None)
    assert {"session_identity_claims", "action_pairs", "web_content_constructs", "session_tags"} <= _prelude_deletes(
        statements
    )
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    conn.close()
