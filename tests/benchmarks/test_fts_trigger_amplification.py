"""Benchmark: measure content_blocks FTS trigger work per incremental block insert.

#1606 — the content_blocks FTS triggers re-project the entire message text
(4-leg UNION ALL over messages.text + all content_blocks fields) on every
single-block INSERT. For a message with N blocks inserted incrementally,
total trigger work is 1+2+…+N = O(N²).

This test measures the wall-time growth curve for incremental vs batch
insertion and asserts the documented O(N²) behavior exists (so future
optimizations have a measured baseline).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


def _timed_insert(conn: sqlite3.Connection, block_index: int) -> float:
    """Insert one content_block row, return elapsed wall time in seconds."""
    start = time.perf_counter()
    conn.execute(
        """INSERT INTO content_blocks(block_id, message_id, conversation_id,
           block_index, type, text, tool_name, tool_id, tool_input, metadata,
           semantic_type)
        VALUES (?, ?, ?, ?, 'text', ?, '', '', '', '', '')""",
        (
            f"block-{block_index}",
            "msg-bench",
            "conv-bench",
            block_index,
            f"block {block_index} text content for fts indexing benchmark",
        ),
    )
    conn.commit()
    return time.perf_counter() - start


@pytest.mark.slow
def test_content_blocks_fts_trigger_growth_is_subquadratic(tmp_path: Path) -> None:
    """Insert 200 blocks incrementally; per-block time must not grow linearly."""
    from tests.infra.storage_records import ConversationBuilder

    block_counts = (20, 50, 100, 200)
    means: dict[int, float] = {}

    for n in block_counts:
        db_path = tmp_path / f"incr_{n}.db"
        (
            ConversationBuilder(db_path, "conv-bench")
            .provider("test")
            .add_message("msg-bench", role="user", text="initial message text")
            .save()
        )
        conn = sqlite3.connect(str(db_path))
        times: list[float] = []
        for i in range(n):
            elapsed = _timed_insert(conn, i)
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
    from tests.infra.storage_records import ConversationBuilder

    block_counts = (20, 50, 100, 200)
    timings: dict[int, float] = {}

    for n in block_counts:
        db_path = tmp_path / f"batch_{n}.db"
        (
            ConversationBuilder(db_path, "conv-bench")
            .provider("test")
            .add_message("msg-bench", role="user", text="initial message text")
            .save()
        )
        conn = sqlite3.connect(str(db_path))
        start = time.perf_counter()
        conn.executemany(
            """INSERT INTO content_blocks(block_id, message_id, conversation_id,
               block_index, type, text, tool_name, tool_id, tool_input, metadata,
               semantic_type)
            VALUES (?, ?, ?, ?, 'text', ?, '', '', '', '', '')""",
            [
                (
                    f"block-{i}",
                    "msg-bench",
                    "conv-bench",
                    i,
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
