"""Codex state exports must materialize under declared bounds, and the
thread-scoped evidence they produce must be reachable by session excision.

Two defects, one logical export (polylogue-xrba4):

1. ``materialize_codex_state_content`` read every ``thread_goals`` /
   ``stage1_outputs`` row with no cap and serialized each one into its own
   durable blob. A state database is an untrusted provider artifact, so this
   was both an OOM surface and a durable amplification surface -- and because
   materialization precedes terminal parse marking, it retriggered on every
   restart.
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

import sqlite3
from pathlib import Path

from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.sources.codex_state_evidence import (
    CodexStateMaterializationReceipt,
    materialize_codex_state_content,
)
from polylogue.storage.materials import list_materials
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


def test_state_materialization_is_bounded_and_receipts_what_it_declined(tmp_path: Path) -> None:
    """Declared caps must bound the materialization AND name the remainder.

    Anti-vacuity: dropping the ``LIMIT ?`` from ``parse_codex_goals_db``, the
    ``substr`` clip on ``objective``, or the aggregate-byte break in
    ``materialize_codex_state_content`` each makes one of these assertions
    red -- and deleting the receipt (returning ``None``, or reporting
    ``rows_materialized == rows_available``) makes the truncation
    indistinguishable from an export that held only two rows, which the
    ``rows_available``/``rows_declined`` assertions reject.
    """
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{index:02d}", f"goal-{index:02d}", "x" * 500) for index in range(6)],
    )

    receipt = _materialize(root, goals_path, row_limit=2, text_char_limit=16)
    assert receipt is not None
    assert receipt.rows_available == 6
    assert receipt.rows_materialized == 2
    assert receipt.rows_declined_row_cap == 4
    assert receipt.rows_declined_byte_cap == 0
    # The oversized objective is clipped, not dropped, and it is named.
    assert receipt.clipped_item_ids == ("goal-00", "goal-01")
    assert receipt.bounded is True
    assert "4 declined by row cap 2" in receipt.as_detail()

    # Only the rows the row cap admitted became durable blobs.
    with sqlite3.connect(root / "source.db") as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0]) == 2

    # The aggregate byte cap is a second, independent bound: the rows were
    # read but never admitted, and the receipt separates the two reasons.
    byte_capped_root = tmp_path / "archive-bytes"
    byte_capped_root.mkdir()
    byte_receipt = _materialize(
        byte_capped_root, goals_path, row_limit=100, text_char_limit=1000, aggregate_byte_limit=700
    )
    assert byte_receipt is not None
    assert byte_receipt.rows_declined_row_cap == 0
    assert byte_receipt.rows_declined_byte_cap > 0
    assert byte_receipt.rows_materialized + byte_receipt.rows_declined_byte_cap == 6
    with sqlite3.connect(byte_capped_root / "source.db") as conn:
        admitted = int(conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0])
    assert admitted == byte_receipt.rows_materialized


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
