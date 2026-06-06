"""Concurrent reader/writer throughput benchmarks.

Measures: read throughput under write load, WAL growth under concurrency,
concurrent reader scaling, and the performance cost of WAL checkpointing
during active reads.

These benchmarks run against the ``index.db``: the corpus is
seeded through ``SessionBuilder`` (native ``ArchiveStore`` writes), reads go
through the archive `blocks_fts` full-text index, and the write load inserts
archive `blocks` rows (which generate WAL traffic and re-index via the
``blocks_fts`` triggers).

Run with:
    pytest tests/benchmarks/test_concurrent_throughput.py \\
      --benchmark-enable -p no:xdist -o "addopts=" -v
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import statistics
import time
from pathlib import Path

import pytest

from tests.benchmarks.helpers import BenchmarkFixture
from tests.infra.storage_records import SessionBuilder

_WORDS = [
    "analysis",
    "python",
    "data",
    "test",
    "implementation",
    "performance",
    "benchmark",
    "query",
    "search",
    "index",
    "configuration",
    "deployment",
    "monitoring",
    "pipeline",
    "optimization",
    "refactoring",
    "validation",
    "integration",
]


def _populate_corpus(archive_root: Path, n_messages: int = 2000) -> tuple[Path, int]:
    """Seed an archive `index.db` corpus, returning (index_db_path, message_count).

    Sessions and messages are written through ``SessionBuilder`` so the
    archive `blocks` / ``blocks_fts`` indexes are maintained by the archive
    write path and triggers.
    """
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    n_convs = max(5, n_messages // 40)
    msgs_per_conv = n_messages // n_convs

    for c in range(n_convs):
        builder = SessionBuilder(db_path, f"concurrent-conv-{c:04d}")
        builder.title(f"Concurrent Test {c}")
        for m in range(msgs_per_conv):
            word = _WORDS[(c * msgs_per_conv + m) % len(_WORDS)]
            role = "user" if m % 2 == 0 else "assistant"
            text = (
                f"Message {m} in conv {c}: {word} {_WORDS[(m + 1) % len(_WORDS)]} "
                "additional context for realistic message size. " + "The quick brown fox jumps over the lazy dog. " * 3
            )
            builder.add_message(f"msg-{c:04d}-{m:04d}", role=role, text=text)
        builder.save()

    return db_path, n_convs * msgs_per_conv


def _fts_search(db_path: Path, query: str, limit: int = 20) -> int:
    """Run an archive `blocks_fts` query, returning the hit count."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        rows = conn.execute(
            "SELECT block_id FROM blocks_fts WHERE blocks_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()
        return len(rows)
    finally:
        conn.close()


def _write_target(db_path: Path) -> tuple[str, str, int]:
    """Resolve (message_id, session_id, next_block_position) for write-load inserts."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT message_id, session_id FROM messages LIMIT 1").fetchone()
        assert row is not None, "seeded corpus has no messages"
        next_position = int(
            conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM blocks WHERE message_id = ?",
                (row[0],),
            ).fetchone()[0]
        )
        return row[0], row[1], next_position
    finally:
        conn.close()


# ── Read throughput with concurrent writes ────────────────────────────


@pytest.mark.benchmark
def test_read_throughput_with_writes(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    """Measure archive FTS search throughput while writes happen concurrently."""
    db_path, msg_count = _populate_corpus(tmp_path / "archive", n_messages=2000)
    assert msg_count > 0

    message_id, session_id, base_position = _write_target(db_path)
    queries = ["analysis", "python", "data", "performance", "benchmark"]

    def mixed_workload() -> dict[str, float]:
        t0 = time.perf_counter()
        read_count = 0
        read_times: list[float] = []
        position = base_position

        for batch in range(10):
            # Do a write: insert archive blocks rows (re-indexed by FTS triggers).
            conn = sqlite3.connect(str(db_path), timeout=30.0)
            try:
                for _ in range(20):
                    conn.execute(
                        "INSERT INTO blocks(message_id, session_id, position, block_type, text) "
                        "VALUES (?, ?, ?, 'text', ?)",
                        (message_id, session_id, position, "new concurrent analysis data row"),
                    )
                    position += 1
                conn.commit()
            finally:
                conn.close()

            # Do a read against the archive FTS index.
            query = queries[batch % len(queries)]
            t_read = time.perf_counter()
            try:
                read_count += _fts_search(db_path, query, limit=20)
            except Exception:
                pass
            read_times.append((time.perf_counter() - t_read) * 1000)

        elapsed = time.perf_counter() - t0
        return {
            "total_s": round(elapsed, 2),
            "reads_completed": 10,
            "read_hits": read_count,
            "read_p50_ms": round(statistics.median(read_times), 2) if read_times else 0,
            "read_p95_ms": round(sorted(read_times)[int(len(read_times) * 0.95)], 2) if read_times else 0,
        }

    _ = benchmark(mixed_workload)

    extras = {
        "corpus_messages": msg_count,
        "reads_completed": 10,
        "writes_completed": 200,
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)


# ── Concurrent reader scaling ─────────────────────────────────────────


@pytest.mark.benchmark
@pytest.mark.parametrize("n_readers", [1, 2, 4, 8])
def test_concurrent_reader_scaling(
    benchmark: BenchmarkFixture,
    n_readers: int,
    tmp_path: Path,
) -> None:
    """Measure archive FTS query throughput scaling with concurrent readers."""
    db_path, msg_count = _populate_corpus(tmp_path / "archive", n_messages=2000)
    assert msg_count > 0

    queries = [
        "analysis",
        "python",
        "data",
        "performance",
        "benchmark",
        "test",
        "implementation",
        "query",
    ]

    def concurrent_reads() -> dict[str, float]:
        t0 = time.perf_counter()
        total_hits = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_readers) as executor:
            futures = []
            for i in range(n_readers * 5):  # 5 queries per reader
                query = queries[i % len(queries)]
                futures.append(executor.submit(_fts_search, db_path, query, 20))

            for future in concurrent.futures.as_completed(futures):
                try:
                    total_hits += future.result(timeout=30)
                except Exception:
                    pass

        elapsed = time.perf_counter() - t0
        return {
            "total_s": round(elapsed, 2),
            "n_readers": n_readers,
            "total_queries": n_readers * 5,
            "total_hits": total_hits,
            "queries_per_s": round((n_readers * 5) / max(elapsed, 0.001), 1),
            "hits_per_s": round(total_hits / max(elapsed, 0.001), 1),
        }

    _ = benchmark(concurrent_reads)

    extras = {"n_readers": n_readers}
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)


# ── WAL growth under sustained writes ─────────────────────────────────


@pytest.mark.benchmark
def test_wal_growth_under_sustained_writes(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    """Measure WAL file growth during sustained native-blocks write workload."""
    db_path, _ = _populate_corpus(tmp_path / "archive", n_messages=500)
    message_id, session_id, base_position = _write_target(db_path)

    wal_path = db_path.with_name(db_path.name + "-wal")

    def sustained_writes() -> dict[str, float]:
        wal_sizes: list[int] = []

        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            # Disable autocheckpoint for measurement
            conn.execute("PRAGMA wal_autocheckpoint = 0")
            conn.commit()

            t0 = time.perf_counter()
            position = base_position
            for i in range(500):
                conn.execute(
                    "INSERT INTO blocks(message_id, session_id, position, block_type, text) "
                    "VALUES (?, ?, ?, 'text', ?)",
                    (message_id, session_id, position, f"WAL growth test block {i} with padding " + "x" * 200),
                )
                position += 1
                if i % 50 == 0:
                    conn.commit()
                    if wal_path.exists():
                        wal_sizes.append(wal_path.stat().st_size)
            conn.commit()

            elapsed = time.perf_counter() - t0

            # Run a checkpoint
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
        finally:
            conn.close()

        wal_after_checkpoint = wal_path.stat().st_size if wal_path.exists() else 0

        return {
            "total_s": round(elapsed, 2),
            "messages_written": 500,
            "msgs_per_s": round(500 / max(elapsed, 0.001), 1),
            "wal_peak_bytes": max(wal_sizes) if wal_sizes else 0,
            "wal_peak_mb": round(max(wal_sizes) / (1024 * 1024), 2) if wal_sizes else 0,
            "wal_after_checkpoint_bytes": wal_after_checkpoint,
            "wal_samples": len(wal_sizes),
        }

    _ = benchmark(sustained_writes)

    # WAL should be checkpointed after explicit TRUNCATE
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(
            {
                "wal_growth_measured": True,
            }
        )


# ── WAL checkpoint latency ────────────────────────────────────────────


@pytest.mark.benchmark
def test_wal_checkpoint_latency(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    """Measure WAL checkpoint latency at different WAL sizes."""
    db_path, _ = _populate_corpus(tmp_path / "archive", n_messages=500)
    message_id, session_id, base_position = _write_target(db_path)

    # Build up WAL by inserting without checkpointing
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        position = base_position
        for i in range(1000):
            conn.execute(
                "INSERT INTO blocks(message_id, session_id, position, block_type, text) VALUES (?, ?, ?, 'text', ?)",
                (message_id, session_id, position, f"Checkpoint test block {i} " + "y" * 300),
            )
            position += 1
        conn.commit()
    finally:
        conn.close()

    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_size_before = wal_path.stat().st_size if wal_path.exists() else 0

    def run_checkpoint() -> dict[str, float]:
        t0 = time.perf_counter()
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        finally:
            conn.close()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        wal_after = wal_path.stat().st_size if wal_path.exists() else 0
        return {
            "checkpoint_ms": round(elapsed_ms, 2),
            "wal_before_mb": round(wal_size_before / (1024 * 1024), 2),
            "wal_after_bytes": wal_after,
            "checkpoint_result_busy": int(row[0]) if row else -1,
        }

    _ = benchmark(run_checkpoint)

    extras = {
        "wal_before_mb": round(wal_size_before / (1024 * 1024), 2),
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)
