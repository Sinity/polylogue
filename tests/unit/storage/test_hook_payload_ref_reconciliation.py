"""polylogue-tfzw0: classify orphaned pre-v22 hook-payload blob refs.

Before source schema v22, ``write_source_hook_event`` wrote every hook
event's blob ref as ``ref_type='raw_payload'`` keyed by a synthetic id
(``deterministic_raw_session_id(origin, source_path, source_index=0,
blob_hash, native_id)``) that never matches a real ``raw_sessions`` row.
These tests build that exact pre-v22 shape directly (bypassing the
now-fixed writer) and prove the classifier recomputes the same id from the
``raw_hook_events`` row's own fields and matches it back to its orphaned ref.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import polylogue.storage.hook_payload_ref_reconciliation as reconciliation
from polylogue.storage.hook_payload_ref_reconciliation import plan_hook_payload_ref_reconciliation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    deterministic_blob_hash,
    deterministic_raw_session_id,
)


def _seed_pre_v22_hook_ref(
    conn: sqlite3.Connection,
    *,
    hook_event_id: str,
    origin: str,
    source_path: str,
    native_id: str,
    payload: bytes,
) -> bytes:
    """Insert the exact pre-v22 orphan shape: a 'raw_payload' ref keyed by a
    synthetic id, plus a raw_hook_events row with blob_hash still NULL.
    """
    blob_hash = deterministic_blob_hash(payload)
    synthetic_ref_id = deterministic_raw_session_id(origin, source_path, 0, blob_hash, native_id)
    conn.execute(
        """
        INSERT INTO raw_hook_events (
            hook_event_id, origin, native_id, session_native_id, source_path, event_type,
            payload_json, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, 'PostToolUse', '{}', 1)
        """,
        (hook_event_id, origin, native_id, "session-1", source_path),
    )
    conn.execute(
        """
        INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, ?, 1)
        """,
        (blob_hash, synthetic_ref_id, source_path, len(payload)),
    )
    return blob_hash


def test_plan_matches_orphaned_ref_to_its_hook_event(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = _seed_pre_v22_hook_ref(
            conn,
            hook_event_id="hook-1",
            origin="codex-session",
            source_path="/hooks/a.jsonl",
            native_id="native-1",
            payload=b'{"event":"PostToolUse"}',
        )
        conn.commit()

        plan = plan_hook_payload_ref_reconciliation(conn)

    assert plan.scanned_count == 1
    assert plan.unmatched_count == 0
    assert len(plan.matched) == 1
    candidate = plan.matched[0]
    assert candidate.hook_event_id == "hook-1"
    assert candidate.blob_hash == blob_hash
    assert candidate.source_path == "/hooks/a.jsonl"
    assert candidate.size_bytes == len(b'{"event":"PostToolUse"}')
    assert plan.matched_bytes == len(b'{"event":"PostToolUse"}')


def test_plan_leaves_ambiguous_collisions_unmatched(tmp_path: Path) -> None:
    """Two hook events sharing a source_path whose recomputed ids BOTH match
    the same orphaned ref (a synthetic collision) must not be guessed at.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = deterministic_blob_hash(b"shared-payload")
        source_path = "/hooks/collide.jsonl"
        native_id = "native-collide"
        synthetic_ref_id = deterministic_raw_session_id("codex-session", source_path, 0, blob_hash, native_id)
        # Two distinct hook_event_id rows both recomputing to the SAME id
        # (same origin/source_path/native_id/blob_hash) -- a genuine
        # ambiguity the planner must not resolve by guessing.
        for suffix in ("a", "b"):
            conn.execute(
                """
                INSERT INTO raw_hook_events (
                    hook_event_id, origin, native_id, session_native_id, source_path, event_type,
                    payload_json, observed_at_ms
                ) VALUES (?, 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1)
                """,
                (f"hook-{suffix}", native_id, source_path),
            )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 15, 1)
            """,
            (blob_hash, synthetic_ref_id, source_path),
        )
        conn.commit()

        plan = plan_hook_payload_ref_reconciliation(conn)

    assert plan.scanned_count == 1
    assert plan.matched == ()
    assert plan.unmatched_count == 1


def test_plan_leaves_ref_with_no_source_path_match_unmatched(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = deterministic_blob_hash(b"lonely-payload")
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-orphan-no-candidates', 'raw_payload', '/hooks/nothing.jsonl', 5, 1)
            """,
            (blob_hash,),
        )
        conn.commit()

        plan = plan_hook_payload_ref_reconciliation(conn)

    assert plan.scanned_count == 1
    assert plan.matched == ()
    assert plan.unmatched_count == 1


def test_plan_ignores_raw_payload_refs_with_a_real_raw_sessions_row(tmp_path: Path) -> None:
    """A genuine session raw_payload ref (the common case) is not "orphaned"
    -- it must never enter the scanned population at all.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session(
            conn,
            origin="codex-session",
            source_path="/sessions/real.jsonl",
            source_index=0,
            payload=b'{"real":"session"}',
            acquired_at_ms=1,
            native_id="real-session",
        )
        plan = plan_hook_payload_ref_reconciliation(conn)

    assert plan.scanned_count == 0
    assert plan.matched == ()


@pytest.mark.parametrize(
    "checkpoint",
    (
        *(f"created:{table}" for table in reconciliation._STAGE_TABLES),
        "population_began",
    ),
)
def test_match_stage_failure_clears_every_temp_table_then_rebuilds(tmp_path: Path, checkpoint: str) -> None:
    """No build checkpoint may leave a reusable partial reconciliation stage."""

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as conn:
        _seed_pre_v22_hook_ref(
            conn,
            hook_event_id="hook-1",
            origin="codex-session",
            source_path="/hooks/a.jsonl",
            native_id="native-1",
            payload=b'{"event":"PostToolUse"}',
        )

        def fail_at_current_checkpoint(event: str) -> None:
            if event == checkpoint:
                raise RuntimeError(f"injected failure at {event}")

        with pytest.raises(RuntimeError, match="injected failure"):
            reconciliation._create_match_stage(conn, failure_injector=fail_at_current_checkpoint)

        remaining = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_temp_master WHERE type = 'table'")}
        assert not remaining.intersection(reconciliation._STAGE_TABLES)
        assert reconciliation._match_stage_readiness(conn) is None

        rebuilt = plan_hook_payload_ref_reconciliation(conn)

    assert rebuilt.scanned_count == 1
    assert rebuilt.unmatched_count == 0
    assert tuple(candidate.hook_event_id for candidate in rebuilt.matched) == ("hook-1",)
