"""Writable benchmark archives materialized from shared workload artifacts.

Benchmarks need a private mutable archive per tier and the populations that
tier contains. Both come from the shared artifact: construction measures the
published tree once and records it on the manifest, so a benchmark fixture
reads those numbers instead of reopening its clone to count rows again.
"""

from __future__ import annotations

from pathlib import Path

from tests.infra.workload_artifacts import (
    SeededArchiveArtifact,
    benchmark_workload_profile,
    benchmark_workload_tier,
    build_benchmark_archive,
    clone_seeded_archive,
)


def _measured_benchmark_artifact(target_messages: int, *, seed: int) -> SeededArchiveArtifact:
    """Resolve the shared artifact for a tier, refusing a mis-sized construction."""
    tier = benchmark_workload_tier(target_messages)
    artifact = build_benchmark_archive(tier, seed=seed)
    produced = artifact.manifest.resources.row_counts.get("messages")
    expected = benchmark_workload_profile(tier).target_messages
    if produced != expected:
        raise RuntimeError(f"benchmark workload {tier.value} produced {produced} messages, expected {expected}")
    return artifact


def seed_benchmark_archive(db_path: Path, target_messages: int, seed: int = 42) -> dict[str, int]:
    """Clone one private writable benchmark tier and report its measured shape."""
    artifact = _measured_benchmark_artifact(target_messages, seed=seed)
    clone_seeded_archive(artifact, db_path.parent)
    resources = artifact.manifest.resources
    return {
        "sessions": resources.row_counts["sessions"],
        "messages": resources.row_counts["messages"],
        "content_blocks": resources.row_counts["blocks"],
        "bytes": resources.total_bytes,
    }


__all__ = ["seed_benchmark_archive"]
