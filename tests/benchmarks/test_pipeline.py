"""Pipeline benchmark tests.

Covers: semantic classification throughput, FTS5 rebuild at scale,
incremental FTS update.

Run with:
    pytest tests/benchmarks/test_pipeline.py --benchmark-enable -p no:xdist -v
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from polylogue.lib.viewports import ToolCategory, classify_tool
from polylogue.pipeline.semantic import extract_tool_metadata
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index, update_index_for_conversations


def _make_diverse_tool_inputs(n: int) -> list[tuple[str, dict[str, Any]]]:
    """Create n diverse (tool_name, tool_input) pairs for classification benchmarks."""
    patterns: list[tuple[str, dict[str, Any]]] = [
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
def test_bench_classify_tool_calls(benchmark, n_tools: int) -> None:
    """classify_tool() throughput across ToolCategory variants."""
    tool_inputs = _make_diverse_tool_inputs(n_tools)
    benchmark(lambda: [classify_tool(name, inp) for name, inp in tool_inputs])


@pytest.mark.benchmark
def test_bench_extract_tool_metadata(benchmark) -> None:
    """extract_tool_metadata() for git, file, subagent variants."""
    inputs: list[tuple[str, dict[str, Any]]] = [
        ("Bash", {"command": "git commit -m 'fix: update schema'"}),
        ("Read", {"file_path": "/realm/project/polylogue/polylogue/lib/models.py"}),
        ("Write", {"file_path": "/tmp/output.py", "content": "result = 42"}),
        ("Edit", {"file_path": "/tmp/file.py", "old_string": "old_val", "new_string": "new_val"}),
        ("Task", {"subagent_type": "explore", "prompt": "explore the codebase", "description": "explore"}),
    ] * 20  # 100 total

    benchmark(lambda: [extract_tool_metadata(name, inp) for name, inp in inputs])


@pytest.mark.benchmark
def test_bench_fts_rebuild_1k(benchmark, bench_db_1k: Path) -> None:
    """FTS5 full rebuild on 1k messages."""
    with open_connection(bench_db_1k) as conn:
        benchmark(lambda: rebuild_index(conn))


@pytest.mark.benchmark
def test_bench_fts_rebuild_5k(benchmark, bench_db_5k: Path) -> None:
    """FTS5 full rebuild on 5k messages — shows O(N) scaling."""
    with open_connection(bench_db_5k) as conn:
        benchmark(lambda: rebuild_index(conn))


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [1, 10, 50])
def test_bench_fts_incremental_update(benchmark, bench_db_5k: Path, n: int) -> None:
    """update_index_for_conversations() for 1, 10, 50 conversations."""
    ids = [f"bench-conv-{i:05d}" for i in range(n)]
    with open_connection(bench_db_5k) as conn:
        benchmark(lambda: update_index_for_conversations(ids, conn))
