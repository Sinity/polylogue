"""Codex state exports must materialize completely with bounded working sets.

Two defects, one logical export (polylogue-xrba4):

1. State rows and long text need bounded pages with addressable continuation.
2. The materials that route produces carry ``referrer_ref =
   codex-session:<thread>`` and own their bytes through
   ``material_observations.blob_hash`` alone -- no ``sessions`` row, no
   ``raw_sessions`` row, no ``blob_refs`` row. ``excise --session`` resolved
   raw targets only from ``sessions.raw_id``, so it left that content
   readable and re-admissible.

Fixtures are synthetic: invented thread uuids, invented objectives, no
operator paths and no transcript bytes.
"""

from __future__ import annotations

import json
import sqlite3
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.sources.codex_state_evidence import (
    CodexStateMaterializationReceipt,
    materialize_codex_state_content,
)
from polylogue.storage.materials import MaterialObservation, list_materials, list_materials_page, read_material
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_THREAD_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
_THREAD_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
_SESSION_A = f"codex-session:{_THREAD_A}"
_SESSION_B = f"codex-session:{_THREAD_B}"


def _write_goals_db(path: Path, goals: list[tuple[str, str, str]]) -> None:
    """Write a synthetic ``goals_1.sqlite`` holding ``(thread, goal, objective)``."""
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE thread_goals (
                thread_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL,
                time_used_seconds INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO thread_goals VALUES (?, ?, ?, 'active', 100, 1, 2, 1000, 2000)",
            goals,
        )
        conn.commit()


def _materialize(root: Path, goals_path: Path, **limits: int) -> CodexStateMaterializationReceipt | None:
    with ArchiveStore(root) as archive:
        receipt = materialize_codex_state_content(
            archive,
            "raw-codex-state",
            state_path=goals_path,
            source_path="/synthetic/codex/goals_1.sqlite",
            state_kind="goals",
            acquired_at_ms=5_000,
            **limits,
        )
        archive.commit()
    return receipt


def test_state_materialization_continues_past_each_work_window(tmp_path: Path) -> None:
    """A small row and byte window must still publish every row and text part."""
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{index:02d}", f"goal-{index:02d}", "x" * 500) for index in range(6)],
    )

    receipt = _materialize(root, goals_path, row_limit=2, text_char_limit=16, aggregate_byte_limit=1024)
    assert receipt is not None
    assert receipt.rows_available == 6
    assert receipt.rows_materialized == 6
    assert receipt.rows_declined_row_cap == 0
    assert receipt.rows_declined_byte_cap == 0
    assert receipt.clipped_item_ids == ()
    assert receipt.bounded is False
    assert receipt.bytes_materialized > receipt.aggregate_byte_cap
    with sqlite3.connect(root / "source.db") as conn:
        cursor = None
        observed: list[MaterialObservation] = []
        while True:
            page = list_materials_page(conn, after=cursor, limit=3)
            observed.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert len(observed) == 6 * (1 + 31)
        chunks = [json.loads(read_material(conn, item.material_id)) for item in observed]
        assert any(part.get("item_id") == "goal-05" and part.get("offset_chars") == 496 for part in chunks)
        assert all(part.get("text_continuation") or part.get("record_type") == "goals" for part in chunks)


@pytest.mark.timeout(300)
def test_large_goal_export_last_row_is_reachable(tmp_path: Path) -> None:
    """A valid row after the old 10,000-row limit survives materialization."""
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{index:05d}", f"goal-{index:05d}", "objective") for index in range(10_001)],
    )
    receipt = _materialize(root, goals_path)
    assert receipt is not None and receipt.rows_materialized == 10_001
    with sqlite3.connect(root / "source.db") as conn:
        last = list_materials_page(conn, evidence_ref="codex-session:thread-10000", limit=2)
        assert len(last.items) == 1
        assert json.loads(read_material(conn, last.items[0].material_id))["goal_id"] == "goal-10000"


def test_long_memory_text_reassembles_from_material_pages(tmp_path: Path) -> None:
    """Every character after the old clip is addressable through material reads."""
    root = tmp_path / "archive"
    root.mkdir()
    memory_path = tmp_path / "memories_1.sqlite"
    memory = "Ω" * 65_001
    with sqlite3.connect(memory_path) as conn:
        conn.execute(
            "CREATE TABLE stage1_outputs (thread_id TEXT PRIMARY KEY, source_updated_at INTEGER, "
            "generated_at INTEGER, raw_memory TEXT, rollout_summary TEXT, usage_count INTEGER, "
            "rollout_slug TEXT, selected_for_phase2 INTEGER)"
        )
        conn.execute(
            "INSERT INTO stage1_outputs VALUES (?, 1, 2, ?, 'summary', 3, 'slug', 1)", ("thread-memory", memory)
        )
    with ArchiveStore(root) as archive:
        receipt = materialize_codex_state_content(
            archive,
            "raw-memory",
            state_path=memory_path,
            source_path="/synthetic/codex/memories_1.sqlite",
            state_kind="memories",
            acquired_at_ms=5_000,
        )
        archive.commit()
    assert receipt is not None and receipt.rows_materialized == 1 and not receipt.bounded
    with sqlite3.connect(root / "source.db") as conn:
        page = list_materials_page(conn, evidence_ref="codex-session:thread-memory", limit=2)
        assert len(page.items) == 2
        parts = [json.loads(read_material(conn, item.material_id)) for item in page.items]
    header = next(part for part in parts if "raw_memory" in part)
    suffix = next(part for part in parts if part.get("field") == "raw_memory")
    assert header["raw_memory"] + suffix["text"] == memory
    assert suffix["offset_chars"] == 64_000 and suffix["final"] is True


def test_parser_row_pages_do_not_retain_the_export(tmp_path: Path) -> None:
    """A larger export must not become one tuple or one retained payload list."""
    from polylogue.sources.parsers.codex_state import iter_codex_state_parts

    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{i:04d}", f"goal-{i:04d}", "x" * 10_000) for i in range(1000)],
    )
    tracemalloc.start()
    try:
        count = sum(1 for _ in iter_codex_state_parts(goals_path, state_kind="goals", page_size=8))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 1000
    assert peak < 8 * 1024 * 1024


def test_invalid_goal_row_is_named_as_partial(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(goals_path, [("valid-thread", "valid-goal", "objective"), ("", "invalid-goal", "objective")])
    receipt = _materialize(root, goals_path, row_limit=1)
    assert receipt is not None
    assert receipt.rows_available == 2
    assert receipt.rows_materialized == 1
    assert receipt.rows_skipped_invalid == 1
    assert receipt.bounded
    assert "partial goal materialization" in receipt.as_detail()


def test_interrupted_state_projection_resumes_without_duplicate_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed page leaves no terminal receipt and replay fills the suffix."""
    from polylogue.sources import codex_state_evidence

    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(goals_path, [(f"thread-{i}", f"goal-{i}", "objective") for i in range(8)])
    original = codex_state_evidence._upsert_codex_material
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic interruption")
        original(*args, **kwargs)

    monkeypatch.setattr(codex_state_evidence, "_upsert_codex_material", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _materialize(root, goals_path, row_limit=2)
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 2
    monkeypatch.setattr(codex_state_evidence, "_upsert_codex_material", original)
    receipt = _materialize(root, goals_path, row_limit=2)
    assert receipt is not None and receipt.rows_materialized == 8
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 8


def test_excising_a_thread_removes_its_codex_state_materials(tmp_path: Path) -> None:
    """``excise --session`` must reach state materials retained for that thread.

    Anti-vacuity: removing the ``_session_material_targets`` call from
    ``resolve_session_excision_target`` (or the material deletion from
    ``apply_session_excision``) leaves thread A's material row, its blob hash
    absent from ``excised_content``, and its objective text still readable
    through ``list_materials``. Thread B's material is asserted untouched, so
    a resolver that simply deleted every material fails too.
    """
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [
            (_THREAD_A, "goal-a", "synthetic objective for thread a"),
            (_THREAD_B, "goal-b", "synthetic objective for thread b"),
        ],
    )
    _materialize(root, goals_path)

    # Precondition: both threads' state evidence is retained and readable.
    with sqlite3.connect(root / "source.db") as conn:
        assert len(list_materials(conn, evidence_ref=_SESSION_A)) == 1
        assert len(list_materials(conn, evidence_ref=_SESSION_B)) == 1
        hash_a = bytes(
            conn.execute(
                "SELECT blob_hash FROM material_observations WHERE referrer_ref = ?",
                (_SESSION_A,),
            ).fetchone()[0]
        )

    # A session with no index row at all is still a real excision target when
    # state materials name it.
    target = resolve_session_excision_target(root, _SESSION_A)
    assert target.session_exists is False
    assert target.found is True
    assert len(target.material_ids) == 1

    plan = plan_session_excision(root, _SESSION_A)
    assert plan.source_materials == 1

    receipt = apply_session_excision(root, _SESSION_A, reason="operator request", actor="tests")
    assert receipt.found is True
    assert receipt.counts["source_materials"] == 1
    assert hash_a.hex() in receipt.removed_blob_hashes

    with sqlite3.connect(root / "source.db") as conn:
        assert list_materials(conn, evidence_ref=_SESSION_A) == []
        # The other thread's evidence, in the same export, is untouched.
        assert len(list_materials(conn, evidence_ref=_SESSION_B)) == 1
        # The bytes are durably refused on re-acquisition, not merely unlinked.
        excised = {bytes(row[0]) for row in conn.execute("SELECT removed_hash FROM excised_content")}
        assert hash_a in excised


def test_an_unresolvable_index_does_not_read_as_a_current_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (polylogue-hu24g): a failed index resolution returned
    ``True`` ("already current"), so the Codex thread-state projection was
    skipped as done instead of running and failing visibly."""
    import polylogue.storage.archive_identity as archive_identity
    from polylogue.sources.codex_state_evidence import _thread_state_projection_is_current

    def unresolvable(_root: Path) -> Path:
        raise OSError("active generation pointer unreadable")

    monkeypatch.setattr(archive_identity, "resolve_active_index_path", unresolvable)
    assert _thread_state_projection_is_current(tmp_path) is False
