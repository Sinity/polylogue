"""Long-haul benchmark campaign runner.

Executes reproducible benchmark campaigns against synthetic archives
and produces durable JSON + Markdown reports under .local/benchmark-campaigns/.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.scenarios import CorpusSourceKind, dispatch_execution

from .authored_scenario_catalog import build_authored_scenario_catalog
from .benchmark_catalog import BenchmarkCampaignEntry
from .synthetic_benchmark_runtime import CampaignResult, resolve_synthetic_benchmark_runner

SYNTHETIC_CAMPAIGNS: dict[str, BenchmarkCampaignEntry] = (
    build_authored_scenario_catalog().synthetic_benchmark_campaign_index()
)

async def run_synthetic_benchmark_campaign(name: str, db_path: Path) -> CampaignResult:
    """Dispatch one synthetic benchmark campaign by authored scenario id."""

    campaign = SYNTHETIC_CAMPAIGNS[name]
    if campaign.execution is None:
        raise ValueError(f"Synthetic benchmark campaign {campaign.name!r} has no execution")
    result = await dispatch_execution(
        campaign.execution,
        runner_resolver=resolve_synthetic_benchmark_runner,
        runner_args=(db_path,),
    )
    if not isinstance(result, CampaignResult):
        raise TypeError(
            f"Synthetic benchmark campaign {campaign.name!r} returned unexpected result type {type(result).__name__}"
        )
    result.origin = campaign.origin
    result.path_targets = list(campaign.path_targets)
    result.artifact_targets = list(campaign.artifact_targets)
    result.operation_targets = list(campaign.operation_targets)
    result.tags = list(campaign.tags)
    return result


async def run_full_campaign(
    scale_level: str,
    output_dir: Path,
    *,
    corpus_source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
) -> list[CampaignResult]:
    """Run all benchmark campaigns at a given scale level.

    Generates a synthetic archive at the specified scale, then runs
    each campaign against the resulting database.

    Args:
        scale_level: One of "small", "medium", "large", "stretch".
        output_dir: Directory for archive and report output.

    Returns:
        List of CampaignResult for all campaigns.
    """
    from devtools.large_archive_generator import (
        ScaleLevel,
        generate_archive,
        get_default_spec,
    )

    level = ScaleLevel(scale_level)
    spec = get_default_spec(level)

    archive_dir = output_dir / f"archive-{scale_level}"
    source_kind = CorpusSourceKind(corpus_source)
    print(
        f"Generating {scale_level} archive from {source_kind.value} corpus source "
        f"({spec.conversations} conversations, ~{spec.message_count} messages)..."
    )
    archive_metrics = await generate_archive(spec, archive_dir, corpus_source=source_kind)
    print(
        f"Archive generated in {archive_metrics.wall_time_s:.1f}s "
        f"({archive_metrics.conversation_count} convs, "
        f"{archive_metrics.message_count} msgs, "
        f"{archive_metrics.db_size_bytes / 1024 / 1024:.1f} MB)"
    )

    db_path = archive_dir / "benchmark.db"
    results: list[CampaignResult] = []

    for campaign in SYNTHETIC_CAMPAIGNS.values():
        if campaign.scale_targets and scale_level not in campaign.scale_targets:
            continue
        print(f"Running {campaign.name} campaign...")
        result = await run_synthetic_benchmark_campaign(campaign.name, db_path)
        result.scale_level = scale_level
        results.append(result)
        metric_value = result.metrics.get(campaign.summary_metric, 0)
        print(f"  -> {metric_value:.4f}{campaign.summary_label}")

    return results

__all__ = [
    "CampaignResult",
    "SYNTHETIC_CAMPAIGNS",
    "run_full_campaign",
    "run_synthetic_benchmark_campaign",
]
