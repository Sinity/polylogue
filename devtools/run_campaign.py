"""CLI entry point for running benchmark campaigns.

Usage:
    devtools bench synthetic --scale medium --output .local/benchmark-campaigns/
    devtools bench synthetic --scale large --campaign fts-rebuild
    devtools bench synthetic --list
"""

from __future__ import annotations

import argparse
import asyncio
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
        help="Output directory for reports (default: .local/benchmark-campaigns/)",
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
    from devtools.benchmark_campaigns import (
        SYNTHETIC_CAMPAIGNS,
        run_full_campaign,
        run_synthetic_benchmark_campaign,
    )
    from devtools.campaign_report import save_campaign_reports
    from devtools.large_archive_generator import (
        ScaleLevel,
        generate_archive,
        get_default_spec,
    )

    if args.list_campaigns:
        print("Available campaigns:")
        for campaign in SYNTHETIC_CAMPAIGNS.values():
            print(f"  {campaign.name}: {campaign.description}")
        print("\nScale levels: small, medium, large, stretch")
        return 0

    if args.campaign == "all":
        results = await run_full_campaign(
            args.scale,
            args.output,
            corpus_source=CorpusSourceKind(args.corpus_source),
        )
    else:
        # Generate archive first
        if args.campaign not in SYNTHETIC_CAMPAIGNS:
            print(f"Unknown campaign: {args.campaign}")
            print(f"Available: {', '.join(SYNTHETIC_CAMPAIGNS)}")
            return 1

        archive_dir = args.output / f"archive-{args.scale}"
        db_path = archive_dir / "benchmark.db"
        if args.campaign == "daemon-live-convergence":
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            archive_dir.mkdir(parents=True, exist_ok=True)
        else:
            level = ScaleLevel(args.scale)
            spec = get_default_spec(level)

            # Override seed if provided
            if args.seed != 42:
                from dataclasses import replace

                spec = replace(spec, seed=args.seed)

            print(f"Generating {args.scale} archive from {args.corpus_source} corpus source...")
            await generate_archive(
                spec,
                archive_dir,
                corpus_source=CorpusSourceKind(args.corpus_source),
            )

        result = await run_synthetic_benchmark_campaign(args.campaign, db_path)
        result.scale_level = args.scale
        results = [result]

    # Save reports
    saved = save_campaign_reports(results, args.output)
    print("\nReports saved:")
    for path in saved:
        print(f"  {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
