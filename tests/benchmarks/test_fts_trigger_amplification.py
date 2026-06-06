"""Benchmark: measure blocks FTS trigger work per incremental block insert.

#1606 — the legacy ``content_blocks`` FTS triggers re-projected the entire
message text (4-leg UNION ALL over messages.text + all content_blocks fields)
on every single-block INSERT, giving O(N²) total trigger work for a message
with N blocks inserted incrementally.

The archive replaces ``content_blocks`` with the archive `blocks`
table whose ``blocks_fts_ai`` trigger projects only the inserted row's columns
(O(1) per insert). This test measures the wall-time growth curve for
incremental vs batch insertion against the archive `blocks` table and asserts
the per-insert work stays flat.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


def _resolve_bench_message(conn: sqlite3.Connection) -> tuple[str, str, int]:
    """Return (message_id, session_id, next_position) for the seeded message.

    The seeded message already owns one text block at position 0, so
    incremental inserts begin one past the highest existing block position.
    """
    row = conn.execute("SELECT message_id, session_id FROM messages LIMIT 1").fetchone()
    assert row is not None, "seeded session has no message"
    next_position = int(
        conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM blocks WHERE message_id = ?",
            (row[0],),
        ).fetchone()[0]
    )
    return row[0], row[1], next_position


def _timed_insert(conn: sqlite3.Connection, message_id: str, session_id: str, position: int) -> float:
    """Insert one archive `blocks` row, return elapsed wall time in seconds."""
    start = time.perf_counter()
    conn.execute(
        """INSERT INTO blocks(message_id, session_id, position, block_type, text)
        VALUES (?, ?, ?, 'text', ?)""",
        (
            message_id,
            session_id,
            position,
            f"block {position} text content for fts indexing benchmark",
        ),
    )
    conn.commit()
    return time.perf_counter() - start


@pytest.mark.slow
def test_content_blocks_fts_trigger_growth_is_subquadratic(tmp_path: Path) -> None:
    """Insert 200 blocks incrementally; per-block time must not grow linearly."""
    from tests.infra.storage_records import SessionBuilder

    block_counts = (20, 50, 100, 200)
    means: dict[int, float] = {}

    for n in block_counts:
        # SessionBuilder writes the archive `index.db` into the db_path's
        # parent directory, so isolate each tier under its own subdirectory.
        root = tmp_path / f"incr_{n}"
        root.mkdir()
        db_path = root / "index.db"
        (
            SessionBuilder(db_path, "conv-bench")
            .provider("test")
            .add_message("msg-bench", role="user", text="initial message text")
            .save()
        )
        conn = sqlite3.connect(str(db_path))
        message_id, session_id, base_position = _resolve_bench_message(conn)
        times: list[float] = []
        for i in range(n):
            elapsed = _timed_insert(conn, message_id, session_id, base_position + i)
            times.append(elapsed)
        conn.close()
        means[n] = sum(times) / len(times)

    # With the #1606 fix, INSERT triggers are O(1) per block.
    # Per-block time should stay nearly flat: mean@200 ≈ mean@20.
    ratio = means[200] / means[20]
    assert ratio < 2.0, (
        f"Per-block FTS trigger time still grows superlinearly: "
        f"mean@200={means[200]:.6f}s vs mean@20={means[20]:.6f}s "
        f"(ratio={ratio:.1f}, expected < 2.0 after #1606 fix)."
    )


@pytest.mark.slow
def test_content_blocks_batch_insert_is_linear(tmp_path: Path) -> None:
    """Bulk-inserting N blocks scales linearly — comparison baseline."""
    from tests.infra.storage_records import SessionBuilder

    block_counts = (20, 50, 100, 200)
    timings: dict[int, float] = {}

    for n in block_counts:
        root = tmp_path / f"batch_{n}"
        root.mkdir()
        db_path = root / "index.db"
        (
            SessionBuilder(db_path, "conv-bench")
            .provider("test")
            .add_message("msg-bench", role="user", text="initial message text")
            .save()
        )
        conn = sqlite3.connect(str(db_path))
        message_id, session_id, base_position = _resolve_bench_message(conn)
        start = time.perf_counter()
        conn.executemany(
            """INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, 'text', ?)""",
            [
                (
                    message_id,
                    session_id,
                    base_position + i,
                    f"block {i} text content",
                )
                for i in range(n)
            ],
        )
        conn.commit()
        timings[n] = time.perf_counter() - start
        conn.close()

    # Bulk insert should scale roughly linearly: per-block time ~constant.
    ratio = (timings[200] / 200) / (timings[20] / 20)
    assert ratio < 3.0, (
        f"Bulk insert per-block time not linear: "
        f"{timings[200]:.4f}s for 200 vs {timings[20]:.4f}s for 20 "
        f"(per-block ratio={ratio:.1f})"
    )
