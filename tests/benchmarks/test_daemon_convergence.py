"""Daemon convergence performance probe.

Generates synthetic JSONL at controlled scale tiers, runs the live watcher
batch path with daemon post-ingest convergence stages, and measures timing
for regression tracking.

Run with:
    pytest tests/benchmarks/test_daemon_convergence.py \\
      --benchmark-enable -p no:xdist -o "addopts=" -v
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# ── Synthetic data generation ──────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_claude_code_session(uuid: str, n_messages: int, *, include_tools: bool = True) -> list[dict[str, object]]:
    """Generate a realistic Claude Code session JSONL."""
    records: list[dict[str, object]] = []
    tool_names = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"]
    for i in range(n_messages):
        is_user = i % 2 == 0
        record: dict[str, object] = {
            "parentUuid": None if i == 0 else f"msg-{i - 1:04d}",
            "sessionId": uuid,
            "type": "user" if is_user else "assistant",
            "message": {
                "role": "user" if is_user else "assistant",
                "content": f"Synthetic message {i} in session {uuid}. "
                f"{'The quick brown fox jumps over the lazy dog. ' * 20}",
            },
            "uuid": f"msg-{i:04d}",
            "timestamp": f"2026-05-05T00:{i // 60:02d}:{i % 60:02d}.000Z",
            "cwd": "/realm/project/polylogue",
            "version": "1.0.6",
            "isSidechain": False,
            "userType": "external",
        }
        if not is_user and include_tools and i % 4 == 1:
            tool = tool_names[i % len(tool_names)]
            record["message"]["content"] = [  # type: ignore[index]
                {
                    "type": "tool_use",
                    "name": tool,
                    "id": f"tool-{i:04d}",
                    "input": {"command": f"echo 'hello {i}'"},
                }
            ]
            record["type"] = "assistant"
        records.append(record)
    return records


# ── Scale tiers ─────────────────────────────────────────────────────


_SCALE_TIERS = {
    "xs-tiny-files": {"files": 10, "msgs_per_file": 10},
    "sm-small-corpus": {"files": 50, "msgs_per_file": 50},
    "md-medium-corpus": {"files": 100, "msgs_per_file": 100},
    "lg-few-large": {"files": 5, "msgs_per_file": 1000},
    "xl-single-giant": {"files": 1, "msgs_per_file": 10000},
}


def _generate_corpus(tmp_path: Path, tier: str) -> Path:
    """Generate a synthetic corpus at the given scale tier."""
    spec = _SCALE_TIERS[tier]
    root = tmp_path / "corpus" / "test-project"
    for i in range(spec["files"]):
        uuid = f"deadbeef-0000-0000-0000-{i:012x}"
        records = _make_claude_code_session(uuid, spec["msgs_per_file"])
        _write_jsonl(root / f"{uuid}.jsonl", records)
    return root.parent


# ── Probe: convergence model ────────────────────────────────────────


def _run_convergence_probe(
    corpus_root: Path,
    tmp_path: Path,
) -> dict[str, float]:
    """Run the canonical daemon live-ingest path against a synthetic corpus.

    Returns per-stage timing dict.
    """
    import asyncio

    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    # Use a fresh DB for clean measurement.
    db_path = tmp_path / "polylogue.db"
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(tmp_path)

    # Collect all JSONL files.
    files = list(corpus_root.rglob("*.jsonl"))

    converger = DaemonConverger(stages=make_default_convergence_stages(db_path), max_workers=4)
    polylogue = _BenchmarkPolylogue(tmp_path, db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="benchmark", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="benchmark-v1",
        converger=converger,
    )

    timings: dict[str, float] = {}

    # Measure canonical batched live ingestion with post-ingest convergence.
    t_total = time.perf_counter()
    asyncio.run(processor.ingest_files(files, emit_event=False))
    timings["total_s"] = time.perf_counter() - t_total
    timings["files"] = float(len(files))

    summary = converger.summary()
    timings["converged"] = float(summary["converged"])
    timings["failed"] = float(summary["failed"])
    timings["total_files"] = float(summary["total"])
    timings["parse_source_calls"] = float(polylogue.parse_source_calls)

    return timings


class _BenchmarkPolylogue:
    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)
        self.parse_source_calls = 0

    async def parse_sources(self, *, sources: list[Any], download_assets: bool) -> None:  # noqa: ARG002
        self.parse_source_calls += 1


# ── Benchmark tests ─────────────────────────────────────────────────


@pytest.mark.benchmark
@pytest.mark.parametrize("tier", list(_SCALE_TIERS))
def test_convergence_scale_tier(benchmark, tier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Measure convergence throughput at each scale tier."""
    corpus_root = _generate_corpus(tmp_path, tier)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    result = benchmark(_run_convergence_probe, corpus_root, tmp_path)

    spec = _SCALE_TIERS[tier]
    total_msgs = spec["files"] * spec["msgs_per_file"]
    if result["total_s"] > 0:
        msgs_per_s = total_msgs / result["total_s"]
        # Round-trip via benchmark extra_info for pytest-benchmark.
        extras = {
            "tier": tier,
            "files": spec["files"],
            "msgs_per_file": spec["msgs_per_file"],
            "total_msgs": total_msgs,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs_per_s, 1),
            "converged": int(result["converged"]),
            "parse_source_calls": int(result["parse_source_calls"]),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        # Assert basic correctness.
        assert result["parse_source_calls"] == 1
        assert result["converged"] >= result["total_files"] * 0.8, (
            f"Only {result['converged']}/{result['total_files']} converged"
        )
    else:
        pytest.fail("Zero elapsed time — measurement broken")


@pytest.mark.benchmark
def test_convergence_single_file_perf(benchmark, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Detailed timing on a single 1000-message file."""
    root = tmp_path / "corpus" / "test"
    records = _make_claude_code_session("single-test", 1000)
    _write_jsonl(root / "single.jsonl", records)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    result = benchmark(_run_convergence_probe, root.parent, tmp_path)
    msgs = 1000
    if result["total_s"] > 0:
        extras = {
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs / result["total_s"], 1),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
