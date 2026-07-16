"""Multi-provider daemon convergence benchmarks.

Extends the Claude Code-only convergence benchmarks to cover all major
providers: Codex, ChatGPT, and Gemini. Measures throughput, memory, and
correctness per provider at controlled scale tiers.

Run with:
    pytest tests/benchmarks/test_daemon_convergence.py \\
      tests/benchmarks/test_daemon_convergence_multi_provider.py \\
      --benchmark-enable -p no:xdist -o "addopts=" -v
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot

_SCALE_TIERS = {
    "xs-tiny": {"files": 5, "msgs_per_file": 10},
    "sm-small": {"files": 20, "msgs_per_file": 25},
    "md-medium": {"files": 50, "msgs_per_file": 50},
}


def _generate_corpus(tmp_path: Path, tier: str, provider: str) -> Path:
    spec = _SCALE_TIERS[tier]
    root = tmp_path / "corpus" / f"{provider}-project"
    workload = CorpusSpec.for_provider(
        provider,
        count=spec["files"],
        messages_min=spec["msgs_per_file"],
        messages_max=spec["msgs_per_file"],
        seed=42,
        origin="generated.schema-convergence-benchmark",
        tags=("synthetic", "schema", "benchmark", "convergence"),
    )
    SyntheticCorpus.write_spec_artifacts(workload, root, prefix=provider, index_width=4)
    return root.parent


# ── Probe ──────────────────────────────────────────────────────────────


def _run_convergence_probe(
    corpus_root: Path,
    tmp_path: Path,
) -> dict[str, float]:
    import asyncio

    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    # Archive root / config are scoped by the calling test via
    # ``monkeypatch.setenv`` so the probe never mutates process-global
    # ``os.environ`` directly (#1878). The convergence path reads the archive
    # root from the ``_BenchmarkPolylogue`` object, not the env.
    db_path = tmp_path / "index.db"

    files = list(corpus_root.rglob("*.jsonl")) + list(corpus_root.rglob("*.json"))
    # Filter only session files (skip metadata)
    files = [f for f in files if not f.name.startswith(".")]

    converger = DaemonConverger(stages=make_default_convergence_stages(db_path), max_workers=4)
    polylogue = _BenchmarkPolylogue(tmp_path, db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="benchmark", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="benchmark-multi-v1",
        converger=converger,
    )

    t_total = time.perf_counter()
    metrics = asyncio.run(processor.ingest_files(files, emit_event=False))
    elapsed = time.perf_counter() - t_total

    return {
        # Unrounded elapsed: rounding to 2 decimals would collapse a sub-10ms
        # run to ``0.0`` and trip the ``total_s > 0`` guard (#1878). Round at
        # display time only.
        "total_s": elapsed,
        "files": float(len(files)),
        "total_files": float(len(files)),
        "succeeded_files": float(metrics.succeeded_file_count),
        "failed_files": float(metrics.failed_file_count),
        "parse_wall_s": metrics.parse_time_s,
        "convergence_wall_s": metrics.convergence_time_s,
    }


class _BenchmarkPolylogue:
    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)


# ── Parameterized benchmark tests ─────────────────────────────────────


def _provider_tier_params() -> list[Any]:
    params: list[Any] = []
    for provider in ["claude-code", "codex"]:  # ChatGPT needs special handling
        for tier in _SCALE_TIERS:
            params.append(pytest.param(provider, tier, id=f"{provider}-{tier}"))
    return params


@pytest.mark.benchmark
@pytest.mark.parametrize("provider,tier", _provider_tier_params())
def test_convergence_per_provider(
    benchmark: BenchmarkFixture,
    provider: str,
    tier: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure convergence throughput for each provider at each scale tier."""
    corpus_root = _generate_corpus(tmp_path, tier, provider)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_probe, corpus_root, tmp_path)

    spec = _SCALE_TIERS[tier]
    total_msgs = spec["files"] * spec["msgs_per_file"]
    if result["total_s"] > 0:
        msgs_per_s = total_msgs / result["total_s"]
        extras = {
            "provider": provider,
            "tier": tier,
            "files": spec["files"],
            "msgs_per_file": spec["msgs_per_file"],
            "total_msgs": total_msgs,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs_per_s, 1),
            "parse_wall_s": result["parse_wall_s"],
            "convergence_wall_s": result["convergence_wall_s"],
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        assert result["succeeded_files"] == result["total_files"], (
            f"{provider}/{tier}: {result['failed_files']} files failed"
        )
    else:
        pytest.fail("Zero elapsed time — measurement broken")


@pytest.mark.benchmark
@pytest.mark.parametrize("provider", ["claude-code", "codex"])
def test_convergence_single_file_per_provider(
    benchmark: BenchmarkFixture,
    provider: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-provider throughput on a single 500-message file."""
    root = tmp_path / "corpus" / "test"
    workload = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=500,
        messages_max=500,
        seed=43,
        origin="generated.schema-convergence-benchmark",
        tags=("synthetic", "schema", "benchmark", "convergence"),
    )
    SyntheticCorpus.write_spec_artifacts(workload, root, prefix=f"single-{provider}")

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_probe, root.parent, tmp_path)
    msgs = 500
    if result["total_s"] > 0:
        extras = {
            "provider": provider,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs / result["total_s"], 1),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        assert result["failed_files"] == 0


# ── Cross-provider correctness assertion ──────────────────────────────


@pytest.mark.benchmark
def test_cross_provider_convergence_correctness(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All providers should converge with zero failures on a mixed corpus."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True)

    # Generate files from all providers into subdirectories
    providers = ["claude-code", "codex"]
    for provider in providers:
        tier = "xs-tiny"
        _generate_corpus(tmp_path, tier, provider)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark_one_shot(benchmark, _run_convergence_probe, corpus_root, tmp_path)

    assert result["failed_files"] == 0, f"Cross-provider convergence had failures: {result}"
    assert result["succeeded_files"] >= len(providers), (
        f"Expected at least {len(providers)} converged, got {result['succeeded_files']}"
    )

    extras = {
        "providers": ",".join(providers),
        "total_files": result["files"],
        "total_s": round(result["total_s"], 2),
        "succeeded": int(result["succeeded_files"]),
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)
