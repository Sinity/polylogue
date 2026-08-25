"""Preview or apply GC for the shared seeded-archive fixture cache.

The command has one reachability authority: the generated inventory in
``tests.infra.workload_artifacts``. It never treats an observed cache entry as
reachable and it delegates all lock, lease, corruption, grace, receipt, and
deletion behavior to ``gc_seeded_archive_artifacts``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

from tests.infra.workload_artifacts import (
    ArtifactGcReport,
    SeededArchiveReachabilityInventory,
    current_seeded_archive_reachability,
    default_cache_root,
    gc_seeded_archive_artifacts,
    validate_seeded_archive_reachability,
)


def _report_payload(report: ArtifactGcReport, *, inventory: SeededArchiveReachabilityInventory) -> dict[str, object]:
    payload = report.to_payload()
    payload["reachability"] = inventory.to_payload()
    payload["dispositions"] = dict(sorted(Counter(entry.disposition.value for entry in report.entries).items()))
    return payload


def _render_report(
    report: ArtifactGcReport,
    *,
    inventory: SeededArchiveReachabilityInventory,
    receipt: Path,
    stdout: TextIO,
) -> None:
    mode = "APPLIED" if not report.dry_run else "preview (no mutation performed -- pass --apply to delete)"
    print(f"mode: {mode}", file=stdout)
    print(f"cache root: {report.cache_root}", file=stdout)
    print(f"reachable recipes: {len(inventory.entries)}", file=stdout)
    print(f"reclaimable bytes: {report.reclaimable_bytes}", file=stdout)
    print(f"deleted bytes: {report.deleted_bytes}", file=stdout)
    dispositions = Counter(entry.disposition.value for entry in report.entries)
    print("dispositions:", file=stdout)
    for disposition, count in sorted(dispositions.items()):
        print(f"  {disposition}: {count}", file=stdout)
    print(f"receipt: {receipt}", file=stdout)


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Seeded-artifact cache root; defaults to the NVMe cache root.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Bounded JSON GC receipt path (defaults inside the cache root).",
    )
    parser.add_argument(
        "--grace-period-s",
        type=float,
        default=24 * 60 * 60,
        help="Minimum age before an unreachable artifact can be deleted.",
    )
    parser.add_argument(
        "--protected-worktree",
        type=Path,
        action="append",
        default=[],
        help="Additional artifact/worktree path to retain (repeatable).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete eligible aged artifacts. Without this flag, preview only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the complete report as JSON.")
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    root = (args.cache_root or default_cache_root()).expanduser()
    receipt = (args.receipt or root / ".seeded-archive-gc-receipt.json").expanduser()

    try:
        inventory = current_seeded_archive_reachability()
        validate_seeded_archive_reachability(inventory)
        report = gc_seeded_archive_artifacts(
            cache_root=root,
            reachable_keys=inventory.keys,
            grace_period_s=args.grace_period_s,
            dry_run=not args.apply,
            protected_worktrees=args.protected_worktree,
            receipt_path=receipt,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"refused: {exc}", file=output)
        return 1

    if args.json:
        print(json.dumps(_report_payload(report, inventory=inventory), indent=2, sort_keys=True), file=output)
    else:
        _render_report(report, inventory=inventory, receipt=receipt, stdout=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
