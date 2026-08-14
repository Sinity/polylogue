"""CLI entry point for running benchmark campaigns.

Usage:
    devtools bench synthetic --scale medium --output .local/benchmark-campaigns/
    devtools bench synthetic --scale large --campaign fts-rebuild
    devtools bench synthetic --list
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import shutil
import sys
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.scenarios import CorpusSourceKind

ROOT = _get_root()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible benchmark campaigns against synthetic archives.",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large", "stretch"],
        default="small",
        help="Scale level for synthetic archive (default: small)",
    )
    parser.add_argument(
        "--campaign",
        default="all",
        help="Specific campaign name or 'all' (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".local" / "benchmark-campaigns",
        help="Output directory for generated archives (default: .local/benchmark-campaigns/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_campaigns",
        help="List available campaigns and exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--corpus-source",
        choices=[kind.value for kind in CorpusSourceKind],
        default=CorpusSourceKind.DEFAULT.value,
        help="Synthetic corpus source to use for archive generation (default: default)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    from devtools.campaign_archive_location import CampaignArchiveLocation
    from devtools.large_archive_generator import (
        ScaleLevel,
        generate_archive,
        get_default_spec,
    )
    from devtools.synthetic_benchmark_runtime import SYNTHETIC_BENCHMARK_RUNNERS, CampaignResult

    async def run_synthetic_benchmark(name: str, db_path: Path) -> CampaignResult:
        try:
            runner = SYNTHETIC_BENCHMARK_RUNNERS[name]
        except KeyError as exc:
            raise ValueError(f"Unknown synthetic benchmark runner {name!r}") from exc
        result = runner(db_path)
        if inspect.isawaitable(result):
            return await result
        return result

    if args.list_campaigns:
        print("Available synthetic benchmark runners:")
        for name in SYNTHETIC_BENCHMARK_RUNNERS:
            print(f"  {name}")
        print("\nScale levels: small, medium, large, stretch")
        return 0

    if args.campaign == "all":
        level = ScaleLevel(args.scale)
        spec = get_default_spec(level)
        archive_dir = args.output / f"archive-{args.scale}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        with CampaignArchiveLocation.acquire(archive_dir) as location:
            print(f"Generating {args.scale} archive from {args.corpus_source} corpus source...")
            await generate_archive(
                spec,
                archive_dir,
                corpus_source=CorpusSourceKind(args.corpus_source),
                location=location,
            )
            results: list[CampaignResult] = []
            for name in SYNTHETIC_BENCHMARK_RUNNERS:
                print(f"Running {name}...")
                result = await run_synthetic_benchmark(name, location.active_index_path)
                result.scale_level = args.scale
                results.append(result)
    else:
        # Generate archive first
        if args.campaign not in SYNTHETIC_BENCHMARK_RUNNERS:
            print(f"Unknown campaign: {args.campaign}")
            print(f"Available: {', '.join(SYNTHETIC_BENCHMARK_RUNNERS)}")
            return 1

        archive_dir = args.output / f"archive-{args.scale}"

        # CampaignArchiveLocation is acquired once and held for archive
        # generation (if any) and the benchmark reopen below -- ownership,
        # generation, and target tier stay stable across both steps
        # (polylogue-ovme.3). A wrong/unowned archive_dir fails here, before
        # any work starts.
        if args.campaign == "daemon-live-convergence":
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            archive_dir.mkdir(parents=True, exist_ok=True)

            with CampaignArchiveLocation.acquire(archive_dir) as location:
                result = await run_synthetic_benchmark(args.campaign, location.active_index_path)
        else:
            level = ScaleLevel(args.scale)
            spec = get_default_spec(level)

            # Override seed if provided
            if args.seed != 42:
                from dataclasses import replace

                spec = replace(spec, seed=args.seed)

            print(f"Generating {args.scale} archive from {args.corpus_source} corpus source...")

            archive_dir.mkdir(parents=True, exist_ok=True)
            with CampaignArchiveLocation.acquire(archive_dir) as location:
                await generate_archive(
                    spec,
                    archive_dir,
                    corpus_source=CorpusSourceKind(args.corpus_source),
                    location=location,
                )
                result = await run_synthetic_benchmark(args.campaign, location.active_index_path)

        result.scale_level = args.scale
        results = [result]

    for result in results:
        print(f"{result.campaign_name}: {json.dumps(result.metrics, sort_keys=True)}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
