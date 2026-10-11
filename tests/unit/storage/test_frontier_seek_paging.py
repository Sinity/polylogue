"""Neutral real scratch paging: exact output and SQLite VM work, not timing."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.storage import frontier_inspection as frontier
from polylogue.storage.raw_retention import _IndexRawRevisionHead
from polylogue.storage.sqlite.connection_profile import open_scratch_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


@pytest.fixture
def selection(tmp_path: Path) -> Iterator[frontier._FrontierSelection]:
    # These components consume only scratch SQL. A real native parent owns
    # physical cleanup; this fixture never supplies Source authority.
    parent = open_scratch_connection(tmp_path / "parent.db")
    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    item = frontier._FrontierSelection(
        cast(PreparedIndexMutation, parent), Path(scratch.name) / "selection.db", scratch_directory=scratch
    )
    try:
        yield item
    finally:
        item._owner.close()
        parent.close()


def _keys(count: int) -> list[str]:
    return sorted(["", "é", "Ω", "😀", *(f"key-{number:08d}" for number in range(count - 4))])


@pytest.mark.parametrize("kind", ["keys", "paths", "pending_raws"])
def test_real_keyset_pagers_preserve_output_and_seek_work_growth(
    selection: frontier._FrontierSelection, kind: str, record_property: Any
) -> None:
    conn = selection._owner.require_connection()
    table, column = {
        "keys": ("selected_keys", "logical_key"),
        "paths": ("selected_paths", "source_path"),
        "pending_raws": ("pending_heads", "raw_id"),
    }[kind]
    actual_work = []
    reference_work = []
    for count in (4096, 8192):
        conn.execute(f"DELETE FROM {table}")
        expected = _keys(count)
        if kind == "pending_raws":
            conn.executemany("INSERT INTO pending_heads VALUES(?,?,?,?)", ((key, key, "{}", "{}") for key in expected))
        else:
            conn.executemany(f"INSERT INTO {table} VALUES(?)", ((key,) for key in expected))
        progress = 0

        def step() -> int:
            nonlocal progress
            progress += 1
            return 0

        trace: list[str] = []
        conn.set_progress_handler(step, 100)
        conn.set_trace_callback(trace.append)
        try:
            if kind == "keys":
                actual = list(selection.keys())
            elif kind == "paths":
                actual = list(selection.paths())
            else:
                actual = list(frontier._pending_raw_ids(selection))
        finally:
            conn.set_progress_handler(None, 0)
            conn.set_trace_callback(None)
        actual_work.append(progress * 100)
        assert actual == expected
        pages = [sql for sql in trace if sql.startswith("SELECT")]
        assert len(pages) == count // 256 + 1
        assert ">" not in pages[0]  # None starts the first page; empty text is a real key.
        for sql in pages[1:]:
            plan = " ".join(str(row[-1]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql))
            assert "SEARCH" in plan and ">?" in plan
            assert any(row[1] == "SeekGT" for row in conn.execute("EXPLAIN " + sql))
        progress = 0
        conn.set_progress_handler(step, 100)
        try:
            after = None
            reference: list[str] = []
            while True:
                page = list(
                    conn.execute(
                        f"SELECT DISTINCT {column} FROM {table} WHERE (? IS NULL OR {column}>?) "
                        f"ORDER BY {column} LIMIT 256",
                        (after, after),
                    )
                )
                if not page:
                    break
                reference.extend(row[0] for row in page)
                after = page[-1][0]
        finally:
            conn.set_progress_handler(None, 0)
        reference_work.append(progress * 100)
        assert reference == actual
    record_property(f"{kind}_seek_vm_ops", actual_work)
    record_property(f"{kind}_nullable_vm_ops", reference_work)
    # Doubling neutral input doubles seek work; the old OR re-visits each prefix.
    assert actual_work[1] < actual_work[0] * 2.3
    assert reference_work[1] > reference_work[0] * 3
    assert actual_work[1] < reference_work[1] // 3


def test_byte_head_pager_keeps_complete_path_kind_and_unicode_scope(
    selection: frontier._FrontierSelection,
) -> None:
    expected = _keys(1024)
    for key in expected:
        selection.retain_head(_IndexRawRevisionHead(key, "raw", "revision", "byte", 5, 0, 5), "path")
        selection.retain_head(
            _IndexRawRevisionHead("semantic-" + key, "raw", "revision", "semantic", 1, 0, None), "path"
        )
        selection.retain_head(_IndexRawRevisionHead("other-" + key, "raw", "revision", "byte", 5, 0, 5), "other")
    conn = selection._owner.require_connection()
    trace: list[str] = []
    conn.set_trace_callback(trace.append)
    try:
        heads = list(selection.byte_heads_for_path("path"))
    finally:
        conn.set_trace_callback(None)
    assert [head.logical_source_key for head in heads] == expected
    assert list(selection.byte_heads_for_path("missing")) == []
    for sql in trace[1:]:
        plan = " ".join(str(row[-1]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql))
        assert "frontier_heads_by_path" in plan and "logical_key>?" in plan


def test_checkpoint_pagers_and_ack_comparison_visit_members_after_first_page(
    selection: frontier._FrontierSelection,
) -> None:
    import json
    import sqlite3

    conn = selection._owner.require_connection()
    before = {"decision": None, "revision_authority": "quarantined"}
    after = {
        "decision": "applied",
        "revision_authority": "byte_proven",
        "decided_at_ms": 1,
        "acquisition_generation": 0,
    }
    keys = _keys(1024)
    conn.execute("INSERT INTO pending_raws VALUES('raw','digest')")
    conn.execute("INSERT INTO current_raws VALUES('raw','digest')")
    conn.executemany("INSERT INTO membership_inputs VALUES('raw',?,?)", ((key, json.dumps(before)) for key in keys))
    conn.executemany("INSERT INTO current_memberships VALUES('raw',?,?)", ((key, json.dumps(after)) for key in keys))
    assert frontier._ack_raw_transition(selection, "raw")
    conn.execute(
        "UPDATE current_memberships SET disposition_json=? WHERE logical_source_key=?",
        ('{"decision":"rejected"}', keys[-1]),
    )
    assert not frontier._ack_raw_transition(selection, "raw")
    for key in keys:
        selection.add_raw(key)
    selection.add_raw("raw")
    with sqlite3.connect(":memory:") as ops:
        ops.executescript(
            "CREATE TABLE raw_frontier_pending_heads(logical_source_key TEXT PRIMARY KEY,raw_id,inputs_json,contribution_json);"
            "CREATE TABLE raw_frontier_pending_raws(raw_id TEXT PRIMARY KEY,structural_digest);"
            "CREATE TABLE raw_frontier_membership_inputs(raw_id,logical_source_key,disposition_json,PRIMARY KEY(raw_id,logical_source_key));"
        )
        frontier._prepare_checkpoint_delta(selection, ops)
    assert conn.execute("SELECT count(*) FROM checkpoint_member_writes").fetchone()[0] == len(keys)
