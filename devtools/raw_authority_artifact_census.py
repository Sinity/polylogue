"""Run the quarantined raw-authority artifact census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.raw_authority_artifact_census import (
    RawAuthorityArtifactCensusError,
    render_report,
    run_raw_authority_artifact_census,
)
from polylogue.paths import archive_root as default_archive_root


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to inspect; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows scanned; dry-run defaults to the full quarantine and --apply defaults to 500.",
    )
    parser.add_argument(
        "--after-raw-id",
        default=None,
        help="Exclusive cursor from the durable checkpoint; requires --census-id for --apply.",
    )
    parser.add_argument(
        "--census-id",
        default=None,
        help="Durable checkpoint identity returned by the preceding apply page.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upsert only artifact observations. Requires --backup-manifest; never changes raw authority.",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        default=None,
        help="Verified source-tier backup manifest required with --apply.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write one immutable JSON receipt for dry-run only. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete report as JSON; required with --apply.",
    )
    args = parser.parse_args(argv)
    root = args.archive_root if args.archive_root is not None else default_archive_root()
    if args.apply and not args.json:
        print("refused: --apply requires --json for its stdout report", file=stdout)
        return 1
    try:
        report = run_raw_authority_artifact_census(
            root,
            apply=args.apply,
            backup_manifest=args.backup_manifest,
            limit=args.limit,
            after_raw_id=args.after_raw_id,
            receipt_path=args.receipt,
            census_id=args.census_id,
        )
    except (RawAuthorityArtifactCensusError, FileNotFoundError, ValueError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=stdout)
    else:
        render_report(report, stdout=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
