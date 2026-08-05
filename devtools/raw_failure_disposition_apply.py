"""Apply reviewed terminal dispositions to historical raw parse failures.

The supplied JSONL manifest is dry-run checked by default. ``--apply`` also
requires a verified source-tier backup manifest and writes immutable receipts;
it retains raw payload bytes and the original parser diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.raw_failure_disposition_apply import (
    RawFailureDispositionApplyError,
    apply_raw_failure_dispositions,
)
from polylogue.paths import archive_root as default_archive_root


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-manifest", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.archive_root if args.archive_root is not None else default_archive_root()
    try:
        report = apply_raw_failure_dispositions(
            root,
            manifest_path=args.manifest_path,
            backup_manifest=args.backup_manifest,
            dry_run=not args.apply,
        )
    except (RawFailureDispositionApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1
    payload = {
        "applied": report.applied,
        "manifest_sha256": report.manifest_sha256,
        "candidate_count": report.candidate_count,
        "disposed_raw_ids": list(report.disposed_raw_ids),
        "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
    else:
        mode = "APPLIED" if report.applied else "dry-run (no mutation performed)"
        print(f"mode: {mode}", file=stdout)
        print(f"terminal dispositions: {report.candidate_count}", file=stdout)
        print(f"manifest sha256: {report.manifest_sha256}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
