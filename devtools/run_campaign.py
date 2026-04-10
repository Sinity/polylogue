"""CLI entry point for running benchmark campaigns.

Usage:
    python -m devtools run-benchmark-campaigns --scale medium --output .local/benchmark-campaigns/
    python -m devtools run-benchmark-campaigns --scale large --campaign fts-rebuild
    python -m devtools run-benchmark-campaigns --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    from devtools.benchmark_campaigns import (
        CAMPAIGN_REGISTRY,
        run_filter_scan_campaign,
        run_fts_rebuild_campaign,
        run_full_campaign,
        run_incremental_index_campaign,
        run_startup_health_campaign,
    )
    from devtools.campaign_report import save_campaign_reports
    from devtools.large_archive_generator import (
        ScaleLevel,
        generate_archive,
        get_default_spec,
    )

    if args.list_campaigns:
        print("Available campaigns:")
        for name, desc in CAMPAIGN_REGISTRY.items():
            print(f"  {name}: {desc}")
        print("\nScale levels: small, medium, large, stretch")
        return 0

    if args.campaign == "all":
        results = await run_full_campaign(args.scale, args.output)
    else:
        # Generate archive first
        if args.campaign not in CAMPAIGN_REGISTRY:
            print(f"Unknown campaign: {args.campaign}")
            print(f"Available: {', '.join(CAMPAIGN_REGISTRY)}")
            return 1

        level = ScaleLevel(args.scale)
        spec = get_default_spec(level)

        # Override seed if provided
        if args.seed != 42:
            from dataclasses import replace

            spec = replace(spec, seed=args.seed)

        archive_dir = args.output / f"archive-{args.scale}"
        print(f"Generating {args.scale} archive...")
        await generate_archive(spec, archive_dir)
        db_path = archive_dir / "benchmark.db"

        # Run the specific campaign
        match args.campaign:
            case "fts-rebuild":
                result = run_fts_rebuild_campaign(db_path)
            case "incremental-index":
                result = await run_incremental_index_campaign(db_path)
            case "filter-scan":
                result = await run_filter_scan_campaign(db_path)
            case "startup-health":
                result = await run_startup_health_campaign(db_path)
            case _:
                print(f"Unknown campaign: {args.campaign}")
                return 1

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
