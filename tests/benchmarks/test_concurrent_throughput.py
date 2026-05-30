"""Concurrent reader/writer throughput benchmarks.

Measures: read throughput under write load, WAL growth under concurrency,
concurrent reader scaling, and the performance cost of WAL checkpointing
during active reads.

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

from polylogue.storage.search import search_messages
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.benchmarks.helpers import BenchmarkFixture
from tests.infra.storage_records import (
    ConversationBuilder,
    DbFactory,
)


def _populate_corpus(db_path: Path, n_messages: int = 2000) -> int:
    """Populate a database with test conversations and return message count."""
    _ = DbFactory(db_path)  # initialize schema
    words = [
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

    n_convs = max(5, n_messages // 40)
    msgs_per_conv = n_messages // n_convs

    for c in range(n_convs):
        conv_id = f"concurrent-conv-{c:04d}"
        builder = ConversationBuilder(db_path, conv_id)
        builder.title(f"Concurrent Test {c}")

        for m in range(msgs_per_conv):
            word = words[(c * msgs_per_conv + m) % len(words)]
            role = "user" if m % 2 == 0 else "assistant"
            text = f"Message {m} in conv {c}: {word} {words[(m + 1) % len(words)]} "
            text += "additional context for realistic message size. "
            text += "The quick brown fox jumps over the lazy dog. " * 3
            builder.add_message(f"msg-{c:04d}-{m:04d}", role=role, text=text)

        builder.save()

    with open_connection(db_path) as conn:
        from polylogue.storage.index import rebuild_index

        rebuild_index(conn)

    return n_convs * msgs_per_conv


# ── Read throughput with concurrent writes ────────────────────────────


@pytest.mark.benchmark
def test_read_throughput_with_writes(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    """Measure search throughput while writes are happening concurrently."""
    db_path = tmp_path / "concurrent.db"
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    msg_count = _populate_corpus(db_path, n_messages=2000)
    assert msg_count > 0

    # Pre-create additional messages to insert during benchmark
    new_messages: list[tuple[str, str, str, str]] = []
    for i in range(200):
        new_messages.append(
            (
                f"new-msg-{i:04d}",
                "concurrent-conv-0000",
                "user" if i % 2 == 0 else "assistant",
                f"New concurrent message {i} with searchable terms analysis data",
            )
        )

    queries = ["analysis", "python", "data", "performance", "benchmark"]

    def mixed_workload() -> dict[str, float]:
        t0 = time.perf_counter()
        read_count = 0
        read_times: list[float] = []

        for batch in range(10):
            # Do a write
            with open_connection(db_path) as conn:
                for j in range(20):
                    idx = (batch * 20 + j) % len(new_messages)
                    msg_id, conv_id, role, text = new_messages[idx]
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO messages "
                            "(message_id, conversation_id, role, text, source_name, version) "
                            "VALUES (?, ?, ?, ?, 'test', 1)",
                            (msg_id, conv_id, role, text),
                        )
                    except sqlite3.OperationalError:
                        pass
                conn.commit()

            # Do a read
            query = queries[batch % len(queries)]
            t_read = time.perf_counter()
            try:
                result = search_messages(query, archive_root=archive_root, db_path=db_path, limit=20)
                read_count += len(result.hits)
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
    """Measure query throughput scaling with concurrent readers."""
    db_path = tmp_path / "readers.db"
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    msg_count = _populate_corpus(db_path, n_messages=2000)
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
                futures.append(
                    executor.submit(
                        search_messages,
                        query,
                        archive_root=archive_root,
                        db_path=db_path,
                        limit=20,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    total_hits += len(result.hits)
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
    """Measure WAL file growth during sustained write workload."""
    db_path = tmp_path / "wal_growth.db"
    _populate_corpus(db_path, n_messages=500)

    wal_path = db_path.with_suffix(".db-wal")

    def sustained_writes() -> dict[str, float]:
        wal_sizes: list[int] = []

        with open_connection(db_path) as conn:
            # Disable autocheckpoint for measurement
            conn.execute("PRAGMA wal_autocheckpoint = 0")
            conn.commit()

            t0 = time.perf_counter()
            for i in range(500):
                conn.execute(
                    "INSERT INTO messages (message_id, conversation_id, role, text, source_name, version) "
                    "VALUES (?, ?, ?, ?, 'test', 1)",
                    (
                        f"wal-msg-{i:04d}",
                        "concurrent-conv-0000",
                        "user" if i % 2 == 0 else "assistant",
                        f"WAL growth test message {i} with padding " + "x" * 200,
                    ),
                )
                if i % 50 == 0:
                    conn.commit()
                    if wal_path.exists():
                        wal_sizes.append(wal_path.stat().st_size)
            conn.commit()

            elapsed = time.perf_counter() - t0

            # Run a checkpoint
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()

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
    db_path = tmp_path / "checkpoint.db"
    _populate_corpus(db_path, n_messages=500)

    # Build up WAL by inserting without checkpointing
    with open_connection(db_path) as conn:
        conn.execute("PRAGMA wal_autocheckpoint = 0")
        for i in range(1000):
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, source_name, version) "
                "VALUES (?, ?, ?, ?, 'test', 1)",
                (f"cp-msg-{i:04d}", "concurrent-conv-0000", "user", f"Checkpoint test message {i} " + "y" * 300),
            )
        conn.commit()

    wal_path = db_path.with_suffix(".db-wal")
    wal_size_before = wal_path.stat().st_size if wal_path.exists() else 0

    def run_checkpoint() -> dict[str, float]:
        t0 = time.perf_counter()
        with open_connection(db_path, timeout=5.0) as conn:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
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
