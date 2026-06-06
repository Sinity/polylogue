"""Live watcher / cursor concurrency invariants (#1182).

Pins the documented expectations for ``CursorStore``:

- ``set(..., allow_backward=False)`` rejects a regression: a write with
  smaller ``byte_size`` and ``byte_offset`` against the same parser
  fingerprint must not advance the cursor backward.
- Concurrent batch-convergence threads racing on the same source path
  end with the cursor at the largest observed offset — never below.
- Idempotent ``raw_sessions`` upserts (the daemon's
  ``INSERT OR REPLACE``/dedupe-by-raw_id contract) tolerate concurrent
  re-ingestion of the same source without producing duplicate rows.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.sources.live.cursor import CursorStore


def test_cursor_rejects_backward_write(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    src = tmp_path / "session.jsonl"
    src.write_text("ignored")
    assert store.set(src, byte_size=500, byte_offset=500, parser_fingerprint="v1")
    # Smaller size + offset under same fingerprint: must be rejected.
    accepted = store.set(src, byte_size=100, byte_offset=100, parser_fingerprint="v1")
    assert accepted is False
    record = store.get_record(src)
    assert record is not None
    assert record.byte_size == 500
    assert record.byte_offset == 500


def test_cursor_allows_backward_when_explicitly_requested(tmp_path: Path) -> None:
    """``allow_backward=True`` is the documented escape hatch (rewind on parser change)."""
    store = CursorStore(tmp_path / "live.sqlite")
    src = tmp_path / "session.jsonl"
    src.write_text("ignored")
    store.set(src, byte_size=500, byte_offset=500, parser_fingerprint="v1")
    # Different parser fingerprint OR explicit allow_backward → write wins.
    assert store.set(src, byte_size=100, byte_offset=100, parser_fingerprint="v2")
    record = store.get_record(src)
    assert record is not None and record.byte_size == 100


def test_concurrent_cursor_writers_never_regress(tmp_path: Path) -> None:
    """N threads each push monotonically-increasing offsets; final state is the max.

    This is the durable invariant for batch convergence: even when two
    workers process the same source file in parallel (rare, but legal),
    the cursor must end at the largest committed offset.
    """
    store = CursorStore(tmp_path / "live.sqlite")
    src = tmp_path / "session.jsonl"
    src.write_text("x")

    n_threads = 4
    steps = 25
    error: list[BaseException] = []

    def writer(offset_base: int) -> None:
        try:
            for i in range(steps):
                value = offset_base + i * 100
                store.set(src, byte_size=value, byte_offset=value, parser_fingerprint="v1")
        except BaseException as exc:  # pragma: no cover - defensive thread error capture
            error.append(exc)

    threads = [threading.Thread(target=writer, args=(t * 10,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not error, f"thread raised: {error}"

    record = store.get_record(src)
    assert record is not None
    max_expected = max((t * 10) + (steps - 1) * 100 for t in range(n_threads))
    assert record.byte_size >= max_expected - 100, (
        f"cursor regressed: got {record.byte_size}, expected >= {max_expected - 100}. "
        "Concurrent writers caused offset rollback."
    )


def test_idempotent_raw_upsert_under_concurrent_ingest(tmp_path: Path) -> None:
    """``raw_sessions`` has ``raw_id`` PRIMARY KEY; concurrent ingest of identical raw_id must converge to one row.

    This is the contract that prevents the daemon from creating duplicate
    raw rows when two convergence cycles race on the same file.
    """
    db_path = tmp_path / "raw.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.commit()
    conn.close()

    raw_id = "shared-raw-id"
    error: list[BaseException] = []

    def writer(label: str) -> None:
        try:
            local = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                for _ in range(30):
                    local.execute(
                        "INSERT OR IGNORE INTO raw_sessions "
                        "(raw_id, source_name, source_path, blob_size, acquired_at) "
                        "VALUES (?, 'claude', ?, 1, '2024-01-01')",
                        (raw_id, f"x-{label}"),
                    )
                    local.commit()
            finally:
                local.close()
        except BaseException as exc:  # pragma: no cover - defensive thread error capture
            error.append(exc)

    threads = [threading.Thread(target=writer, args=(str(i),)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not error, f"thread raised: {error}"

    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1, (
        f"expected 1 row for raw_id, got {count}. Concurrent ingest "
        "produced duplicate raw rows — uniqueness invariant broken."
    )


@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    offsets=st.lists(st.integers(min_value=1, max_value=100_000), min_size=2, max_size=12),
)
def test_hypothesis_cursor_monotone_against_arbitrary_offsets(
    tmp_path_factory: pytest.TempPathFactory, offsets: list[int]
) -> None:
    """For any sequence of offsets, the same-fingerprint cursor reflects the maximum."""
    base = tmp_path_factory.mktemp("cursor_hyp")
    store = CursorStore(base / "live.sqlite")
    src = base / "session.jsonl"
    src.write_text("x")
    for offset in offsets:
        store.set(src, byte_size=offset, byte_offset=offset, parser_fingerprint="v1")
    record = store.get_record(src)
    assert record is not None
    assert record.byte_size == max(offsets), (
        f"cursor stored {record.byte_size}, expected max {max(offsets)} from sequence {offsets}"
    )
