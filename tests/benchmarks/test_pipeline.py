"""Pipeline benchmark tests.

Covers: semantic classification throughput, FTS5 rebuild at scale,
incremental FTS update.

Run with:
    pytest tests/benchmarks/test_pipeline.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from pathlib import Path
from sqlite3 import Connection

import pytest

from polylogue.archive.viewport.viewports import classify_tool
from polylogue.core.json import JSONDocument
from polylogue.lib.hashing import hash_payload, hash_text
from polylogue.pipeline.prepare import PrepareCache
from polylogue.pipeline.semantic_metadata import extract_tool_metadata
from polylogue.storage.action_events.rebuild_runtime import (
    rebuild_action_event_read_model_sync,
    valid_action_event_source_ids_sync,
)
from polylogue.storage.fts.fts_lifecycle import repair_fts_index_sync
from polylogue.storage.index import rebuild_index, update_index_for_conversations
from tests.benchmarks.helpers import (
    BenchmarkFixture,
    benchmark_connection_call,
    benchmark_store_call,
)


def _make_diverse_tool_inputs(n: int) -> list[tuple[str, JSONDocument]]:
    """Create n diverse (tool_name, tool_input) pairs for classification benchmarks."""
    patterns: list[tuple[str, JSONDocument]] = [
        ("Read", {"file_path": "/path/to/file.py"}),
        ("Write", {"file_path": "/path/to/output.py", "content": "print('hello')"}),
        ("Edit", {"file_path": "/path/to/file.py", "old_string": "old", "new_string": "new"}),
        ("Bash", {"command": "git commit -m 'fix: update'"}),
        ("Bash", {"command": "ls -la /tmp"}),
        ("Task", {"subagent_type": "general-purpose", "prompt": "analyze this code", "description": "analyze"}),
        ("Glob", {"pattern": "**/*.py"}),
        ("Grep", {"pattern": "class Foo", "path": "."}),
        ("Bash", {"command": "git push origin main"}),
        ("Bash", {"command": "pytest -q tests/"}),
    ]
    result = []
    for i in range(n):
        name, inp = patterns[i % len(patterns)]
        result.append((name, inp))
    return result


@pytest.mark.benchmark
@pytest.mark.parametrize("n_tools", [10, 100, 1000])
def test_bench_classify_tool_calls(benchmark: BenchmarkFixture, n_tools: int) -> None:
    """classify_tool() throughput across ToolCategory variants."""
    tool_inputs = _make_diverse_tool_inputs(n_tools)
    benchmark(lambda: [classify_tool(name, inp) for name, inp in tool_inputs])


@pytest.mark.benchmark
def test_bench_extract_tool_metadata(benchmark: BenchmarkFixture) -> None:
    """extract_tool_metadata() for git, file, subagent variants."""
    inputs: list[tuple[str, JSONDocument]] = [
        ("Bash", {"command": "git commit -m 'fix: update schema'"}),
        ("Read", {"file_path": "/workspace/polylogue/polylogue/lib/models.py"}),
        ("Write", {"file_path": "/tmp/output.py", "content": "result = 42"}),
        ("Edit", {"file_path": "/tmp/file.py", "old_string": "old_val", "new_string": "new_val"}),
        ("Task", {"subagent_type": "explore", "prompt": "explore the codebase", "description": "explore"}),
    ] * 20  # 100 total

    benchmark(lambda: [extract_tool_metadata(name, inp) for name, inp in inputs])


@pytest.mark.benchmark
def test_bench_fts_rebuild_1k(benchmark: BenchmarkFixture, bench_db_1k: Path) -> None:
    """FTS5 full rebuild on 1k messages."""
    benchmark_connection_call(benchmark, bench_db_1k, rebuild_index)


@pytest.mark.benchmark
def test_bench_fts_rebuild_5k(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """FTS5 full rebuild on 5k messages — shows O(N) scaling."""
    benchmark_connection_call(benchmark, bench_db_5k, rebuild_index)


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [1, 10, 50])
def test_bench_fts_incremental_update(benchmark: BenchmarkFixture, bench_db_5k: Path, n: int) -> None:
    """update_index_for_conversations() for 1, 10, 50 conversations."""
    ids = [f"bench-conv-{i:05d}" for i in range(n)]
    benchmark_connection_call(
        benchmark,
        bench_db_5k,
        lambda conn: update_index_for_conversations(ids, conn),
    )


@pytest.mark.benchmark
def test_bench_action_event_repair_rebuild(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """Action-event repair loop over a realistic seeded archive."""

    def repair_action_events(conn: Connection) -> int:
        targets = valid_action_event_source_ids_sync(conn)
        conn.execute("DELETE FROM action_events")
        conn.execute("DELETE FROM action_events_fts")
        conn.commit()
        rebuilt = rebuild_action_event_read_model_sync(conn, conversation_ids=targets or None)
        if targets:
            repair_fts_index_sync(conn, targets)
        conn.commit()
        return rebuilt

    benchmark_connection_call(benchmark, bench_db_5k, repair_action_events)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1_000, 10_000])
def test_bench_hash_text(benchmark: BenchmarkFixture, size: int) -> None:
    """hash_text() throughput — NFC normalization + SHA-256. Tests text sizes."""
    text = "α" * size  # NFC normalization-heavy Unicode
    benchmark(lambda: hash_text(text))


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", [1, 10, 50])
def test_bench_hash_payload(benchmark: BenchmarkFixture, depth: int) -> None:
    """hash_payload() — JSON serialization + SHA-256 for varying object complexity."""

    def _make_nested_payload(d: int) -> dict[str, object]:
        node: dict[str, object] = {"value": "leaf", "index": d}
        for level in range(d):
            node = {"level": level, "child": node, "data": list(range(10))}
        return node

    payload = _make_nested_payload(depth)
    benchmark(lambda: hash_payload(payload))


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [100, 500])
def test_bench_prepare_cache_load(benchmark: BenchmarkFixture, bench_db_5k: Path, n: int) -> None:
    """PrepareCache.load() — bulk-loads N existing conversations in 2 queries."""
    cids = {f"bench-conv-{i:05d}" for i in range(n)}
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: PrepareCache.load(store.backend, cids),
    )
