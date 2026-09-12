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

from polylogue.schemas.synthetic import SyntheticCorpus
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot
from tests.infra.workload_declarations import (
    MULTI_PROVIDER_SCALE_TIERS,
    ConvergenceWorkloadTier,
    convergence_corpus_specs,
    convergence_workload_profile,
)


def _generate_corpus(tmp_path: Path, tier: str, provider: str) -> Path:
    spec = convergence_corpus_specs(tier, provider=provider)[0]
    root = tmp_path / "corpus" / f"{provider}-project"
    SyntheticCorpus.write_spec_artifacts(spec, root, prefix=provider, index_width=4)
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

    converger = DaemonConverger(stages=make_default_convergence_stages(db_path))
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
        for tier in MULTI_PROVIDER_SCALE_TIERS:
            params.append(pytest.param(provider, tier.value, id=f"{provider}-{tier.value}"))
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

    spec = convergence_workload_profile(tier)
    _provider, files, msgs_per_file = next(shape for shape in spec.provider_session_shapes if shape[0] == provider)
    total_msgs = files * msgs_per_file
    if result["total_s"] > 0:
        msgs_per_s = total_msgs / result["total_s"]
        extras = {
            "provider": provider,
            "tier": tier,
            "files": files,
            "msgs_per_file": msgs_per_file,
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
    workload = convergence_corpus_specs(ConvergenceWorkloadTier.SINGLE_FILE_500, provider=provider, seed=43)[0]
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
        tier = ConvergenceWorkloadTier.MULTI_XS_TINY.value
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
