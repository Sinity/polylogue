"""Counter-based complexity, fairness, and resumability laws for raw rebuilds."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, PassCursor, converge
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.sources import revision_backfill
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.fts.fts_lifecycle import reset_message_fts_index_sync
from polylogue.storage.fts.sql import (
    FTS_MESSAGES_IDENTITY_RECIPE_ID,
    insert_all_message_identity_rows_sql,
    insert_all_message_rows_sql,
    insert_session_identity_rows_sql,
    insert_session_rows_sql,
    repair_message_identity_rows_range_sql,
)
from polylogue.storage.sqlite.action_pairs import action_pairs_refresh_sql, rebuild_all_action_pairs_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import rebuild_archive_messages_fts
from polylogue.storage.sqlite.delegation_facts import rebuild_all_delegation_facts_sync
from tests.infra.growth_budgets import GrowthObservation
from tests.infra.prepared_replay import run_on_convergence_owner
from tests.infra.sqlite_work_counter import mutating_statements, sqlite_work_counter

_COMMENTED_REFRESHES = {
    "leading-block": '/* refresh */ UPDATE main."action_pairs" SET tool_name=tool_name',
    "leading-line": "-- refresh\nUPDATE action_pairs SET tool_name=tool_name",
    "update-target": 'UPDATE /* refresh */ main."action_pairs" SET tool_name=tool_name',
    "update-qualified": 'UPDATE main /* refresh */ . /* refresh */ "action_pairs" SET tool_name=tool_name',
    "update-conflict": "UPDATE /* refresh */ OR /* refresh */ IGNORE /* refresh */ action_pairs SET tool_name=tool_name",
    "delete-from": "DELETE /* refresh */ FROM action_pairs",
    "delete-target": "DELETE FROM /* refresh */ action_pairs",
    "insert-into": "INSERT /* refresh */ INTO messages_fts(messages_fts) VALUES('delete-all')",
    "insert-target": "INSERT INTO /* refresh */ messages_fts(messages_fts) VALUES('delete-all')",
    "insert-conflict": "INSERT /* refresh */ OR /* refresh */ REPLACE /* refresh */ INTO /* refresh */ messages_fts(messages_fts) VALUES('delete-all')",
    "replace-into": "REPLACE /* refresh */ INTO /* refresh */ messages_fts(rowid, text) SELECT rowid, search_text FROM blocks WHERE search_text != ''",
    "drop-table": "DROP /* refresh */ TABLE action_pairs",
    "drop-target": "DROP TABLE /* refresh */ main.action_pairs",
    "fts-control": "INSERT INTO messages_fts /* refresh */ (messages_fts) /* refresh */ VALUES('delete-all')",
    "cte": "WITH /* refresh */ all_rows AS (SELECT rowid, search_text FROM blocks WHERE search_text != '') /* refresh */ INSERT /* refresh */ OR REPLACE INTO messages_fts(rowid, text) SELECT rowid, search_text FROM all_rows",
    "quoted-literal": "UPDATE action_pairs SET tool_name='/* keep */ -- keep ''quoted'' UPDATE action_pairs'",
}


def _run(
    root: Path, *, limit: int, raw_ids: tuple[str, ...] = (), cursor: PassCursor | None = None
) -> DerivationReport:
    return run_on_convergence_owner(
        root,
        "test.rebuild.converge",
        lambda compute: converge(
            DerivationRegistry((RawObservationDerivation(root, compute_adapter=compute),)),
            raw_observation_frame(root, raw_ids=raw_ids),
            budget=Budget(page=limit, discovery=limit, inspection=2 * limit, compute=limit, publication=limit),
            # Each publication enters the owner's writer admission, as the
            # daemon's derivation publisher does; preparation stays outside it.
            publisher=admit_stage_write,
            cursor=cursor,
        ),
    )


def _tool_call_payload(native_id: str) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"run {native_id}"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": f"{native_id}-call",
                "call_id": f"{native_id}-call-id",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "printf hello"}),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": f"{native_id}-call-id",
                "output": "hello",
            },
        },
    ]
    return b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)


def _seed_raw_archive(root: Path, count: int, *, prefix: str = "session") -> list[str]:
    initialize_active_archive_root(root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            native_id = f"{prefix}-{index}"
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=_tool_call_payload(native_id),
                    source_path=f"{native_id}.jsonl",
                    canonical_source_path=f"{native_id}.jsonl",
                    acquired_at_ms=index + 1,
                )
            )
    return raw_ids


def _materialize_all(root: Path, raw_count: int) -> None:
    result = _run(root, limit=raw_count)
    assert result.failed == result.pending == 0
    assert result.done == raw_count


def _run_component_measurement(
    tmp_path: Path,
    archive_size: int,
    *,
    component_count: int,
) -> GrowthObservation:
    if component_count < 1 or component_count > archive_size:
        raise ValueError("component_count must be between one and archive_size")
    root = tmp_path / f"component-{archive_size}"
    _seed_raw_archive(root, archive_size, prefix="existing")
    _materialize_all(root, archive_size)

    selected: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(component_count):
            native_id = f"target-{index}"
            selected.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=_tool_call_payload(native_id),
                    source_path=f"{native_id}.jsonl",
                    canonical_source_path=f"{native_id}.jsonl",
                    acquired_at_ms=archive_size + index + 1,
                )
            )

    with sqlite_work_counter(step_interval=1) as counter:
        result = _run(root, limit=component_count, raw_ids=tuple(selected))

    assert result.done == component_count
    return GrowthObservation(
        tier=str(archive_size),
        size=archive_size,
        metrics={
            "component_derived_vm_steps": float(counter.metric("derived_vm_steps")),
            "archive_wide_derived_statements": float(counter.metric("archive_wide_derived_statements")),
            "component_rows_scanned": float(result.work.discovered),
            "component_rows_written": float(result.work.published),
            "component_bytes": float(sum(len(_tool_call_payload(f"target-{i}")) for i in range(component_count))),
            "component_passes": float(result.done),
            "selected_component_count": float(component_count),
        },
    )


def _assert_component_shape(observations: list[GrowthObservation]) -> None:
    assert observations
    assert len({observation.metric("selected_component_count") for observation in observations}) == 1
    measured = "\n".join(f"  {observation.tier}: {dict(observation.metrics)}" for observation in observations)
    assert all(observation.metric("archive_wide_derived_statements") == 0 for observation in observations), (
        f"incremental component route emitted archive-wide derived writes; measured counters:\n{measured}"
    )
    assert all(observation.metric("component_derived_vm_steps") > 0 for observation in observations), (
        f"production route reported no derived work; measured counters:\n{measured}"
    )
    assert all(
        observation.metric("component_derived_vm_steps") <= observations[0].metric("component_derived_vm_steps")
        for observation in observations[1:]
    ), f"fixed component work grew with unrelated archive rows; measured counters:\n{measured}"


@pytest.mark.timeout(0)
def test_incremental_component_has_no_archive_wide_derived_writes(tmp_path: Path) -> None:
    # Each retained component prepares in a cancellable process and publishes
    # its own progress. The suite's fixed 120-second cutoff can interrupt a
    # valid progressing 32-session seed; keep cancellation with the managed run.
    observations = [
        _run_component_measurement(
            tmp_path,
            archive_size,
            component_count=component_count,
        )
        for archive_size, component_count in ((2, 1), (8, 1), (32, 1))
    ]

    _assert_component_shape(observations)


@pytest.mark.parametrize(
    "mutation",
    [
        "delete",
        "update",
        "delete-tautology",
        "update-tautology",
        "action-pairs-rebuild",
        "action-pairs-tautology-scope",
        "fts-rebuild",
        "fts-reset",
        "fts-row-range-tautology",
        "fts-identity-rebuild",
        "fts-session-union",
        "fts-identity-session-union",
        "fts-literal-scope",
        "fts-tautology-scope",
        "fts-delete-all",
        "delegation-copy",
        "delegation-rebuild",
        *[
            f"quoted-{operation}-{quote}"
            for operation in ("update", "delete", "drop", "control")
            for quote in ("double", "bracket", "backtick", "single")
        ],
        *[
            f"conflict-{operation}-{mode}"
            for operation in ("update", "insert")
            for mode in ("rollback", "abort", "replace", "fail", "ignore")
        ],
        "qualified-main",
        "qualified-temp",
        "mixed-case",
        *[f"comment-{name}" for name in _COMMENTED_REFRESHES],
    ],
)
def test_incremental_law_rejects_once_per_pass_archive_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Removing the zero-write oracle lets this real SQL refresh pass."""
    original = _run
    refreshes = 0

    def refresh_after_pass(*args: Any, **kwargs: Any) -> DerivationReport:
        nonlocal refreshes
        result = original(*args, **kwargs)
        refreshes += 1
        root = args[0]
        # The mutant runs once after an ordinary pass, on a production-opened
        # index connection. It changes unrelated derived rows archive-wide.
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            count = archive._conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0]
            assert count == 9
            if mutation.startswith("comment-"):
                name = mutation.removeprefix("comment-")
                changes = archive._conn.total_changes
                assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
                archive._conn.execute(_COMMENTED_REFRESHES[name])
                if name.startswith("drop-"):
                    assert (
                        archive._conn.execute(
                            "SELECT COUNT(*) FROM sqlite_master WHERE name='action_pairs'"
                        ).fetchone()[0]
                        == 0
                    )
                else:
                    assert archive._conn.total_changes > changes
                    if name.startswith("delete-"):
                        assert archive._conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
                    elif name.startswith("insert-") or name == "fts-control":
                        assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
                    elif name == "quoted-literal":
                        assert [
                            row[0] for row in archive._conn.execute("SELECT DISTINCT tool_name FROM action_pairs")
                        ] == ["/* keep */ -- keep 'quoted' UPDATE action_pairs"]
            elif mutation.startswith("quoted-"):
                _, operation, quote = mutation.split("-")
                opening, closing = {
                    "double": ('"', '"'),
                    "bracket": ("[", "]"),
                    "backtick": ("`", "`"),
                    "single": ("'", "'"),
                }[quote]
                table = f"{opening}action_pairs{closing}"
                changes = archive._conn.total_changes
                if operation == "update":
                    assert archive._conn.execute(f"UPDATE {table} SET tool_name=tool_name").rowcount == count
                    assert archive._conn.total_changes > changes
                elif operation == "delete":
                    assert archive._conn.execute(f"DELETE FROM {table}").rowcount == count
                    assert archive._conn.total_changes > changes
                    assert archive._conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
                elif operation == "drop":
                    archive._conn.execute(f"DROP TABLE main.{table}")
                    assert (
                        archive._conn.execute(
                            "SELECT COUNT(*) FROM sqlite_master WHERE name='action_pairs'"
                        ).fetchone()[0]
                        == 0
                    )
                else:
                    assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
                    fts = f"{opening}messages_fts{closing}"
                    archive._conn.execute(f"INSERT INTO main.{fts}({fts}) VALUES('delete-all')")
                    assert archive._conn.total_changes > changes
                    assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
            elif mutation.startswith("conflict-"):
                _, operation, mode = mutation.split("-")
                changes = archive._conn.total_changes
                if operation == "update":
                    assert (
                        archive._conn.execute(f"UPDATE OR {mode.upper()} action_pairs SET tool_name=tool_name").rowcount
                        == count
                    )
                else:
                    populated = archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
                    assert populated > 0
                    offset = archive._conn.execute("SELECT MAX(rowid) FROM messages_fts").fetchone()[0] + 1
                    # Mutate the ordinary archive-wide FTS owner, with fresh
                    # rowids so every conflict mode performs real writes.
                    sql = (
                        insert_all_message_rows_sql()
                        .replace("INSERT INTO", f"INSERT OR {mode.upper()} INTO", 1)
                        .replace("SELECT rowid,", f"SELECT rowid + {offset},", 1)
                    )
                    assert archive._conn.execute(sql).rowcount > 0
                    assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > populated
                assert archive._conn.total_changes > changes
            elif mutation in {"qualified-main", "qualified-temp", "mixed-case"}:
                if mutation == "qualified-temp":
                    # CTAS is fixture preparation, not a counted DML refresh;
                    # the following UPDATE is the sole global-write witness.
                    archive._conn.execute("CREATE TEMP TABLE action_pairs AS SELECT * FROM main.action_pairs")
                    table = "temp.action_pairs"
                else:
                    table = "main.action_pairs" if mutation == "qualified-main" else '"MAIN" . "AcTiOn_PaIrS"'
                assert archive._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count
                changes = archive._conn.total_changes
                assert archive._conn.execute(f"uPdAtE {table} SET tool_name=tool_name").rowcount == count
                assert archive._conn.total_changes > changes
            elif mutation == "delete":
                assert archive._conn.execute("DELETE FROM action_pairs").rowcount == count
            elif mutation == "update":
                assert archive._conn.execute("UPDATE action_pairs SET tool_name = tool_name").rowcount == count
            elif mutation == "delete-tautology":
                assert archive._conn.execute("DELETE FROM action_pairs WHERE 1").rowcount == count
            elif mutation == "update-tautology":
                assert archive._conn.execute("UPDATE action_pairs SET tool_name = tool_name WHERE 1").rowcount == count
            elif mutation == "action-pairs-rebuild":
                rebuild_all_action_pairs_sync(archive._conn)
                assert archive._conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == count
            elif mutation == "action-pairs-tautology-scope":
                sessions = set(archive._conn.execute("SELECT DISTINCT session_id FROM action_pairs"))
                sql = action_pairs_refresh_sql("'absent' OR 1").replace(
                    "INSERT INTO action_pairs", "INSERT OR REPLACE INTO action_pairs", 1
                )
                changes = archive._conn.total_changes
                assert archive._conn.execute(sql).rowcount > 0
                assert archive._conn.total_changes > changes
                assert set(archive._conn.execute("SELECT DISTINCT session_id FROM action_pairs")) == sessions
            elif mutation == "fts-reset":
                assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
                reset_message_fts_index_sync(archive._conn)
                assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
            elif mutation == "fts-row-range-tautology":
                sql = (
                    repair_message_identity_rows_range_sql()
                    .replace("AND b.rowid <= ?", "AND b.rowid <= ? OR 1")
                    .replace(f"'{FTS_MESSAGES_IDENTITY_RECIPE_ID}'", "'mutant-recipe'")
                )
                changes = archive._conn.total_changes
                archive._conn.execute(sql, (0, 0))
                assert archive._conn.total_changes > changes
                assert (
                    archive._conn.execute(
                        "SELECT COUNT(*) FROM messages_fts_identity WHERE recipe_id = 'mutant-recipe'"
                    ).fetchone()[0]
                    > 0
                )
            elif mutation == "fts-rebuild":
                assert rebuild_archive_messages_fts(archive._conn) > 0
            elif mutation == "fts-delete-all":
                assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
                archive._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
                assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
            elif mutation == "fts-identity-rebuild":
                assert archive._conn.execute(insert_all_message_identity_rows_sql()).rowcount > 0
            elif mutation in {"fts-session-union", "fts-identity-session-union"}:
                table = "messages_fts" if mutation == "fts-session-union" else "messages_fts_identity"
                builder = (
                    insert_session_rows_sql if mutation == "fts-session-union" else insert_session_identity_rows_sql
                )
                populated = archive._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert populated > 0
                sql = builder(1).replace("VALUES (?)", "VALUES (?) UNION SELECT session_id FROM sessions")
                changes = archive._conn.total_changes
                archive._conn.execute(sql, ("absent",))
                assert archive._conn.total_changes > changes
                assert archive._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == populated
            elif mutation == "fts-literal-scope":
                assert (
                    archive._conn.execute(
                        "INSERT OR REPLACE INTO messages_fts(rowid, text) "
                        "SELECT b.rowid, 'b.session_id = 1' FROM blocks AS b WHERE b.search_text != ''"
                    ).rowcount
                    > 0
                )
            elif mutation == "fts-tautology-scope":
                assert (
                    archive._conn.execute(
                        "INSERT OR REPLACE INTO messages_fts(rowid, text) "
                        "SELECT b.rowid, b.search_text FROM blocks AS b "
                        "WHERE b.session_id = 'absent-session' OR 1"
                    ).rowcount
                    > 0
                )
            else:
                # Build canonical dispatch/link evidence; normal triggers
                # derive the populated facts the mutant rewrites.
                parent = archive._conn.execute(
                    "SELECT session_id FROM sessions WHERE native_id = 'existing-0'"
                ).fetchone()[0]
                child = archive._conn.execute(
                    "SELECT session_id FROM sessions WHERE native_id = 'existing-1'"
                ).fetchone()[0]
                archive._conn.execute(
                    "UPDATE blocks SET semantic_type = 'subagent' WHERE session_id = ? AND block_type = 'tool_use'",
                    (parent,),
                )
                block_id = archive._conn.execute(
                    "SELECT block_id FROM blocks WHERE session_id = ? AND block_type = 'tool_use'", (parent,)
                ).fetchone()[0]
                archive._conn.execute(
                    "INSERT INTO session_links(src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id, parent_tool_use_block_id, observed_at_ms) VALUES (?, 'codex-session', 'existing-0', 'subagent', ?, ?, 1)",
                    (child, parent, block_id),
                )
                facts = archive._conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0]
                assert facts > 0
                if mutation == "delegation-copy":
                    assert (
                        archive._conn.execute(
                            "INSERT OR REPLACE INTO delegation_facts SELECT * FROM delegation_facts WHERE 1"
                        ).rowcount
                        == facts
                    )
                else:
                    rebuild_all_delegation_facts_sync(archive._conn)
                    assert archive._conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0] == facts
            archive.commit()
        return result

    # Seed/materialize through the ordinary route before activating the mutant.
    calls = 0

    def mutate_selected_pass(*args: Any, **kwargs: Any) -> DerivationReport:
        nonlocal calls
        calls += 1
        if kwargs.get("raw_ids"):
            return refresh_after_pass(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(__import__(__name__, fromlist=["_run"]), "_run", mutate_selected_pass)
    observation = _run_component_measurement(tmp_path, 8, component_count=1)
    assert calls == 2
    assert refreshes == 1
    assert observation.metric("archive_wide_derived_statements") > 0
    with pytest.raises(AssertionError):
        _assert_component_shape([observation])


def test_commented_scoped_writes_preserve_scope_and_quoted_evidence(tmp_path: Path) -> None:
    """Removing quote-aware comment handling miscounts these real writes."""
    root = tmp_path / "commented-scopes"
    _seed_raw_archive(root, 2, prefix="comments")
    _materialize_all(root, 2)
    with sqlite_work_counter(step_interval=1) as counter:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            session_id = archive._conn.execute("SELECT session_id FROM action_pairs LIMIT 1").fetchone()[0]
            archive._conn.execute("CREATE TEMP TABLE \"action_pairs/* keep */ -- keep\" AS SELECT 'before' AS value")
            value = "/* keep */ -- keep 'quoted'"
            changes = archive._conn.total_changes
            assert (
                archive._conn.execute(
                    "/* scope */ UPDATE /* scope */ action_pairs SET tool_name=? WHERE /* scope */ session_id = ?",
                    (value, session_id),
                ).rowcount
                > 0
            )
            assert archive._conn.total_changes > changes
            assert (
                archive._conn.execute(
                    "SELECT tool_name FROM action_pairs WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                == value
            )
            changes = archive._conn.total_changes
            assert (
                archive._conn.execute(
                    "DELETE /* scope */ FROM action_pairs WHERE session_id = ?", (session_id,)
                ).rowcount
                == 1
            )
            assert archive._conn.total_changes > changes
            rowid = archive._conn.execute("SELECT MAX(rowid) FROM messages_fts").fetchone()[0] + 1
            changes = archive._conn.total_changes
            archive._conn.execute("INSERT /* scope */ INTO messages_fts(rowid, text) VALUES(?, ?)", (rowid, value))
            assert archive._conn.total_changes > changes
            assert (
                archive._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE rowid = ?", (rowid,)).fetchone()[0] == 1
            )
            native_sql = insert_session_rows_sql(1)
            assert "INSERT INTO messages_fts" in native_sql
            sql = native_sql.replace("WITH", "WITH /* scope */", 1).replace(
                "INSERT INTO", "INSERT /* scope */ INTO /* scope */", 1
            )
            assert "INSERT /* scope */ INTO /* scope */ messages_fts" in sql
            changes = archive._conn.total_changes
            archive._conn.execute(sql, (session_id,))
            assert archive._conn.total_changes > changes
            for opening, closing in (('"', '"'), ("[", "]"), ("`", "`"), ("'", "'")):
                table = f"{opening}action_pairs/* keep */ -- keep{closing}"
                assert archive._conn.execute(f"UPDATE {table} SET value='after'").rowcount == 1
            assert archive._conn.execute('SELECT value FROM "action_pairs/* keep */ -- keep"').fetchone()[0] == "after"
            assert counter.metric("archive_wide_derived_statements") == 0
            assert counter.metric("derived_vm_steps") > 0
            archive.commit()
    with mutating_statements() as recorded:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            changes = archive._conn.total_changes
            assert (
                archive._conn.execute(
                    "-- refresh\nUPDATE /* refresh */ action_pairs SET tool_name='/* keep */ -- keep'"
                ).rowcount
                == 1
            )
            assert archive._conn.total_changes > changes
            archive.commit()
    # Opening an archive also prepares temporary/bootstrap relations; SQLite
    # repeats the UPDATE trace for triggers. Neither changes this witness.
    index_mutations = [sql for tier, sql in recorded if tier == "index"]
    assert index_mutations
    assert all(sql == "update action_pairs set tool_name='/* keep */ -- keep'" for sql in index_mutations)


def test_fts_shadow_execution_retains_derived_vm_accounting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Comment-stripping the VM trace drops actual FTS5 shadow work."""
    import tests.infra.sqlite_work_counter as work_module

    root = tmp_path / "fts-shadow-work"
    _seed_raw_archive(root, 2, prefix="fts-shadow")
    _materialize_all(root, 2)
    original = work_module._mentions_derived_surface
    shadow_vm = 0
    measuring = False

    def observe_vm_frame(sql: str) -> bool:
        nonlocal shadow_vm
        derived = original(sql)
        # This observes each real VM progress callback without altering its
        # classification. SQLite's annotated FTS shadow frames are execution
        # evidence, whereas e.g. its data_version PRAGMA is not derived work.
        if measuring and sql.startswith("--") and "messages_fts" in sql and derived:
            shadow_vm += 1
        return derived

    monkeypatch.setattr(work_module, "_mentions_derived_surface", observe_vm_frame)
    with sqlite_work_counter(step_interval=1) as counter:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            populated = archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            assert populated > 0
            archive._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
            assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0
            before_vm = counter.metric("vm_steps")
            before_derived = counter.metric("derived_vm_steps")
            before_changes = archive._conn.total_changes
            measuring = True
            assert archive._conn.execute(insert_all_message_rows_sql()).rowcount > 0
            measuring = False
            measured_vm = counter.metric("vm_steps") - before_vm
            measured_derived = counter.metric("derived_vm_steps") - before_derived
            assert shadow_vm > 0
            assert measured_vm >= measured_derived >= shadow_vm
            assert archive._conn.total_changes > before_changes
            assert archive._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == populated
            archive.commit()


@pytest.mark.timeout(0)
def test_incremental_law_rejects_archive_wide_derived_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _run
    scans = 0

    def read_after_selected_pass(*args: Any, **kwargs: Any) -> DerivationReport:
        nonlocal scans
        result = original(*args, **kwargs)
        if kwargs.get("raw_ids"):
            scans += 1
            with ArchiveStore.open_existing(args[0], read_only=True) as archive:
                rows = archive._conn.execute("SELECT * FROM action_pairs").fetchall()
                assert len(rows) > 0
        return result

    monkeypatch.setattr(__import__(__name__, fromlist=["_run"]), "_run", read_after_selected_pass)
    observations = [_run_component_measurement(tmp_path, size, component_count=1) for size in (2, 8)]
    assert scans == 2
    assert all(observation.metric("archive_wide_derived_statements") == 0 for observation in observations)
    assert observations[1].metric("component_derived_vm_steps") > observations[0].metric("component_derived_vm_steps")
    with pytest.raises(AssertionError):
        _assert_component_shape(observations)


def test_bounded_replay_work_is_batch_bounded_independent_of_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_size = 2
    observed: list[tuple[int, int, int, int]] = []
    for archive_size in (4, 16, 64):
        root = tmp_path / f"batch-{archive_size}"
        _seed_raw_archive(root, archive_size, prefix="batch")
        selected_work = 0
        original_backfill = revision_backfill.apply_prepared_revision_replay

        def counted_backfill(*args: Any, _original: Any = original_backfill, **kwargs: Any) -> Any:
            nonlocal selected_work
            result = _original(*args, **kwargs)
            # The census commits in place during preparation, so the replay
            # scans no census rows; its selected work is the logical sources
            # it replays.
            selected_work += result.replayed_logical_sources
            return result

        with monkeypatch.context() as mutation:
            mutation.setattr(revision_backfill, "apply_prepared_revision_replay", counted_backfill)
            result = _run(root, limit=batch_size)

        assert result.done == batch_size
        assert result.work.discovered <= batch_size
        observed.append(
            (
                archive_size,
                selected_work,
                result.done,
                result.work.published,
            )
        )

    assert observed == [(4, 2, 2, 2), (16, 2, 2, 2), (64, 2, 2, 2)], (
        f"bounded replay exceeded batch bound={batch_size}; observed={observed}"
    )


def test_mixed_hot_cold_large_small_components_all_receive_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fair-mixed"
    raw_ids = _seed_raw_archive(root, 4, prefix="mixed")
    large_raw_id = raw_ids[2]
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (512 * 1024 * 1024, large_raw_id),
        )
        conn.commit()

    original_backfill = revision_backfill.apply_prepared_revision_replay
    attempted = Counter[str]()

    def fail_hot_component(*args: Any, **kwargs: Any) -> Any:
        selected_raw_ids = kwargs.get("selected_raw_ids")
        assert isinstance(selected_raw_ids, list)
        attempted[selected_raw_ids[0]] += 1
        if selected_raw_ids == [raw_ids[0]]:
            raise RuntimeError("injected hot component retry")
        return original_backfill(*args, **kwargs)

    with monkeypatch.context() as mutation:
        mutation.setattr(revision_backfill, "apply_prepared_revision_replay", fail_hot_component)
        cursor = None
        for _ in range(4):
            result = _run(root, limit=1, cursor=cursor)
            cursor = result.cursor
            assert result.work.discovered == 1

    assert set(attempted) == set(raw_ids)
    assert all(count == 1 for count in attempted.values())


def test_progress_counter_is_monotonic_and_resumable_across_bounded_passes(tmp_path: Path) -> None:
    root = tmp_path / "resumable"
    raw_count = 7
    _seed_raw_archive(root, raw_count, prefix="resume")

    remaining: list[int] = []
    repaired: list[int] = []
    selected: list[str] = []
    cursor = None
    for _ in range(4):
        result = _run(root, limit=2, cursor=cursor)
        cursor = result.cursor
        with sqlite3.connect(root / "index.db") as conn:
            remaining.append(raw_count - conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        repaired.append(result.done)
        selected.extend(outcome.key.key for outcome in result.outcomes if outcome.outcome.value == "done")
        if remaining[-1] == 0:
            break

    assert remaining == [5, 3, 1, 0]
    assert repaired == [2, 2, 2, 1]
    assert len(selected) == raw_count
    assert len(set(selected)) == raw_count
    assert _run(root, limit=2).work.published == 0
