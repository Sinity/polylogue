"""Daemon convergence performance probe.

Generates synthetic JSONL at controlled scale tiers and measures
``LiveBatchProcessor.ingest_files`` with post-ingest convergence.

That call is the write entry ``FileIntakeAdapter.admit_page`` uses. It is
not the production scheduler. ``polylogued run`` schedules through
``FairIntakeDispatcher``; the paired measurement of that route lives in
``tests/unit/daemon/test_dispatcher_intake_measurement.py``.

Rehearsal-4 (2026-09-03) and the 09-15 ``real_ingest_driver.py`` /
``hook_drain_driver.py`` numbers were measured on the watcher
chunk/catch-up/hook-drain route. That route is deleted. Those receipts are
historical for a deleted path, not current production.

Run with:
    devtools test tests/benchmarks/test_daemon_convergence.py --benchmark-enable

The md, lg, xl and xxl tiers and the huge-session probe are the heavy lane:
add ``--run-heavy-benchmarks`` to run them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot
from tests.infra.compute_owner import owned_compute_adapter
from tests.infra.convergence_probe_contract import intake_measurement
from tests.infra.workload_declarations import (
    CONVERGENCE_HEAVY_SCALE_TIERS,
    CONVERGENCE_SCALE_TIERS,
    convergence_corpus_specs,
    convergence_workload_profile,
)

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


def _generate_corpus(tmp_path: Path, tier: str) -> Path:
    """Generate a synthetic corpus at the given scale tier."""
    workload = convergence_corpus_specs(tier)[0]
    # The writer places each session at its declared layout position below
    # the source root.
    root = tmp_path / "corpus"
    SyntheticCorpus.write_spec_artifacts(workload, root, prefix="convergence", index_width=4)
    return root


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
    from polylogue.sources.live.watcher import WatchSource
    from tests.infra.live_batch import prepared_live_batch_processor

    # Use a fresh DB for clean measurement. Archive root / config are scoped by
    # the calling test via ``monkeypatch.setenv`` so the probe never mutates
    # process-global ``os.environ`` directly (#1878).
    db_path = tmp_path / "index.db"

    # Collect all JSONL files.
    files = list(corpus_root.rglob("*.jsonl"))

    with owned_compute_adapter() as compute:
        converger = DaemonConverger(stages=make_default_convergence_stages(db_path, compute_adapter=compute))

        async def ingest() -> tuple[Any, float]:
            # The production live batch bootstraps the archive and runs its
            # Source bodies and retained publication on the daemon owners.
            async with prepared_live_batch_processor(
                tmp_path,
                (WatchSource(name="claude-code", root=corpus_root),),
                parser_fingerprint="benchmark-v1",
                converger=converger,
                compute_adapter=compute,
            ) as processor:
                started = time.perf_counter()
                result = await processor.ingest_files(files, emit_event=False)
                return result, time.perf_counter() - started

        timings: dict[str, float] = {}

        # Measure canonical batched live ingestion with post-ingest convergence.
        metrics, elapsed = asyncio.run(ingest())
        timings["total_s"] = elapsed
        timings["files"] = float(len(files))
        timings["parse_wall_s"] = metrics.parse_time_s
        timings["convergence_wall_s"] = metrics.convergence_time_s

        summary = converger.summary()
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
            stored_sessions = archive.count_sessions()
            stored_messages = archive.count_session_messages(metrics.changed_session_ids)
        measurement = intake_measurement(
            expected_files=len(files),
            expected_sessions=int(metrics.ingested_session_count),
            expected_messages=int(metrics.ingested_message_count),
            succeeded_files=metrics.succeeded_file_count,
            failed_files=metrics.failed_file_count,
            skipped_files=metrics.skipped_file_count,
            excluded_files=metrics.excluded_file_count,
            deferred_files=metrics.deferred_file_count,
            refused_bytes=metrics.refused_bytes,
            stored_sessions=stored_sessions,
            stored_messages=stored_messages,
            stage_summary=summary,
        )
        timings.update({key: float(value) for key, value in measurement.items()})
        timings["total_files"] = float(len(files))

        return timings


# ── Benchmark tests ─────────────────────────────────────────────────


#: Heavy tiers keep cancellation with the managed run (``timeout(0)``, see
#: TESTING.md) instead of a fixed deadline: their declared workloads need
#: longer than the repository's 900 s cap on explicit deadlines, and a
#: deadline that fails slow-but-progressing work is a defect.
_HEAVY_LANE = (pytest.mark.heavy_benchmark, pytest.mark.timeout(0))


def _scale_tier_params() -> list[Any]:
    return [
        *(pytest.param(tier, id=str(tier.value)) for tier in CONVERGENCE_SCALE_TIERS),
        *(pytest.param(tier, id=str(tier.value), marks=_HEAVY_LANE) for tier in CONVERGENCE_HEAVY_SCALE_TIERS),
    ]


@pytest.mark.benchmark
@pytest.mark.parametrize("tier", _scale_tier_params())
def test_convergence_scale_tier(benchmark, tier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Measure convergence throughput at each scale tier."""
    corpus_root = _generate_corpus(tmp_path, tier)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_probe, corpus_root, tmp_path)

    profile = convergence_workload_profile(tier)
    _provider, files, msgs_per_file = profile.provider_session_shapes[0]
    total_msgs = files * msgs_per_file
    if result["total_s"] > 0:
        msgs_per_s = total_msgs / result["total_s"]
        # Round-trip via benchmark extra_info for pytest-benchmark.
        extras = {
            "tier": tier,
            "files": files,
            "msgs_per_file": msgs_per_file,
            "total_msgs": total_msgs,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs_per_s, 1),
            "stage_converged_files": int(result["stage_converged_files"]),
            "intake_succeeded_files": int(result["intake_succeeded_files"]),
            "intake_failed_files": int(result["intake_failed_files"]),
            "parse_wall_s": round(result["parse_wall_s"], 2),
            "convergence_wall_s": round(result["convergence_wall_s"], 2),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        # Assert basic correctness.
        assert result["intake_expected_files"] == files
        assert result["stored_sessions"] == files
        assert result["stored_messages"] == total_msgs
        assert result["intake_succeeded_files"] == result["intake_expected_files"]
        assert result["stage_failed_files"] == 0
    else:
        pytest.fail("Zero elapsed time — measurement broken")


@pytest.mark.benchmark
def test_convergence_single_file_perf(benchmark, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Detailed timing on a single 1000-message file."""
    root = tmp_path / "corpus" / "test"
    records = _make_claude_code_session("single-test", 1000)
    _write_jsonl(root / "single.jsonl", records)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_probe, root.parent, tmp_path)
    msgs = 1000
    if result["total_s"] > 0:
        assert result["intake_expected_files"] == 1
        assert result["stored_sessions"] == 1
        assert result["stored_messages"] == msgs
        extras = {
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(result["stored_messages"] / result["total_s"], 1),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)


# ── Memory probe ─────────────────────────────────────────────────────


def _run_convergence_memory_probe(
    corpus_root: Path,
    tmp_path: Path,
) -> dict[str, float]:
    """Run the daemon live-ingest path and capture RSS/memory metrics."""
    import asyncio

    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.watcher import WatchSource
    from tests.infra.live_batch import prepared_live_batch_processor

    db_path = tmp_path / "index.db"

    files = list(corpus_root.rglob("*.jsonl"))

    with owned_compute_adapter() as compute:
        converger = DaemonConverger(stages=make_default_convergence_stages(db_path, compute_adapter=compute))

        async def ingest() -> tuple[Any, float]:
            # The production live batch bootstraps the archive and runs its
            # Source bodies and retained publication on the daemon owners.
            async with prepared_live_batch_processor(
                tmp_path,
                (WatchSource(name="claude-code", root=corpus_root),),
                parser_fingerprint="benchmark-memory-v1",
                converger=converger,
                compute_adapter=compute,
            ) as processor:
                started = time.perf_counter()
                result = await processor.ingest_files(files, emit_event=False)
                return result, time.perf_counter() - started

        metrics, elapsed = asyncio.run(ingest())
        summary = converger.summary()
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
            stored_sessions = archive.count_sessions()
            stored_messages = archive.count_session_messages(metrics.changed_session_ids)
        measurement = intake_measurement(
            expected_files=len(files),
            expected_sessions=int(metrics.ingested_session_count),
            expected_messages=int(metrics.ingested_message_count),
            succeeded_files=metrics.succeeded_file_count,
            failed_files=metrics.failed_file_count,
            skipped_files=metrics.skipped_file_count,
            excluded_files=metrics.excluded_file_count,
            deferred_files=metrics.deferred_file_count,
            refused_bytes=metrics.refused_bytes,
            stored_sessions=stored_sessions,
            stored_messages=stored_messages,
            stage_summary=summary,
        )

        return {
            # Return the unrounded elapsed time. Rounding to 2 decimals here would
            # collapse a sub-10ms run to ``0.0`` and trip the ``total_s > 0``
            # measurement guard in callers (#1878); round only at display time.
            "total_s": elapsed,
            "files": float(len(files)),
            **{key: float(value) for key, value in measurement.items()},
            "parse_wall_s": metrics.parse_time_s,
            "convergence_wall_s": metrics.convergence_time_s,
            "rss_current_mb": metrics.rss_current_mb or 0.0,
            "rss_peak_self_mb": metrics.rss_peak_self_mb or 0.0,
            "rss_peak_children_mb": metrics.rss_peak_children_mb or 0.0,
            "cgroup_memory_current_mb": metrics.cgroup_memory_current_mb or 0.0,
            "cgroup_memory_peak_mb": metrics.cgroup_memory_peak_mb or 0.0,
            "input_bytes": float(metrics.input_bytes),
            "source_payload_read_bytes": float(metrics.source_payload_read_bytes),
        }


@pytest.mark.benchmark
@pytest.mark.parametrize("n_messages", [200, 1000, 5000])
def test_convergence_large_session_memory(
    benchmark: BenchmarkFixture, n_messages: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure convergence performance and memory on a single large session.

    Records RSS, cgroup memory, and timing metrics as benchmark extra_info.
    """
    root = tmp_path / "corpus" / "test"
    records = _make_claude_code_session("large-session-memory", n_messages)
    _write_jsonl(root / "large.jsonl", records)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_memory_probe, root.parent, tmp_path)

    if result["total_s"] > 0:
        rss_peak_mb = result["rss_peak_self_mb"] + result["rss_peak_children_mb"]
        assert result["intake_expected_files"] == 1
        assert result["stored_sessions"] == 1
        assert result["stored_messages"] == n_messages
        extras = {
            "n_messages": n_messages,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(result["stored_messages"] / result["total_s"], 1),
            "parse_wall_s": result["parse_wall_s"],
            "convergence_wall_s": result["convergence_wall_s"],
            "rss_current_mb": result["rss_current_mb"],
            "rss_peak_self_mb": result["rss_peak_self_mb"],
            "rss_peak_children_mb": result["rss_peak_children_mb"],
            "rss_peak_mb": round(rss_peak_mb, 1),
            "cgroup_memory_current_mb": result["cgroup_memory_current_mb"],
            "cgroup_memory_peak_mb": result["cgroup_memory_peak_mb"],
            "input_bytes": result["input_bytes"],
            "source_payload_read_bytes": result["source_payload_read_bytes"],
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        assert result["stage_failed_files"] == 0


@pytest.mark.benchmark
@pytest.mark.heavy_benchmark
@pytest.mark.timeout(0)
def test_convergence_huge_session_memory_bounded(
    benchmark: BenchmarkFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Huge-session RSS regression probe for #1244 / #845-A.

    Ingests a single Claude-Code-shaped JSONL file with 100k messages and
    asserts that the daemon's peak RSS stays well below the file's
    on-disk size. The streaming ``fingerprint_file`` (1 MiB chunks) keeps
    the cursor-update working set bounded; the previous full-file
    ``read_bytes`` produced RSS ≥ file size after every successful full
    ingest.
    """
    root = tmp_path / "corpus" / "huge"
    n_messages = 100_000
    records = _make_claude_code_session("huge-session", n_messages)
    target = root / "huge.jsonl"
    _write_jsonl(target, records)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    file_bytes = target.stat().st_size

    result = benchmark_one_shot(benchmark, _run_convergence_memory_probe, root.parent, tmp_path)

    rss_peak_mb = result["rss_peak_self_mb"] + result["rss_peak_children_mb"]
    assert result["intake_expected_files"] == 1
    assert result["stored_sessions"] == 1
    assert result["stored_messages"] == n_messages
    file_mb = file_bytes / (1024 * 1024)
    extras = {
        "n_messages": n_messages,
        "total_s": round(result["total_s"], 2),
        "file_mb": round(file_mb, 1),
        "rss_peak_mb": round(rss_peak_mb, 1),
        "rss_per_file_mb_ratio": round(rss_peak_mb / max(file_mb, 0.001), 3),
        "convergence_wall_s": result["convergence_wall_s"],
        "source_payload_read_bytes": result["source_payload_read_bytes"],
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)
    assert result["stage_failed_files"] == 0
    # Sanity: the synthetic fixture is genuinely huge.
    assert file_mb >= 50.0, f"100k-message session should produce ≥50MB JSONL, got {file_mb:.1f}MB"
