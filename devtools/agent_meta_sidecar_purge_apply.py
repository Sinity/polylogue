"""Actuator: purge ``agent-*.meta.json`` subagent-sidecar phantom sessions.

polylogue-ioz7: ``devtools/agent_meta_sidecar_sweep_report.py`` only
classifies and reports. This command acts on the report by deleting the
flagged ``sessions`` rows from ``index.db`` (via
``polylogue.maintenance.agent_meta_sidecar_purge_apply``, which reuses the
tested, incident-hardened ``ArchiveStore.delete_sessions`` bulk-delete
primitive) and writing one immutable receipt per purged row.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually delete rows, which additionally requires ``--backup-manifest``
pointing at a verified backup manifest that includes ``index.db`` (the
default backup profile omits the rebuildable index tier -- use
``polylogue backup --output-dir <dir> --profile full_evidence --verify`` or
an equivalent profile). ``raw_sessions`` rows and blobs in ``source.db`` are
never touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.agent_meta_sidecar_purge_apply import (
    AgentMetaSidecarPurgeApplyError,
    apply_agent_meta_sidecar_purge,
)
from polylogue.paths import archive_root as default_archive_root


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to operate on; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of matching sessions classified/purged (unbounded by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched sessions. Without this flag, nothing is mutated.",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        default=None,
        help="Verified backup manifest covering index.db. Required with --apply.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()

    try:
        report = apply_agent_meta_sidecar_purge(
            root,
            backup_manifest=args.backup_manifest,
            limit=args.limit,
            dry_run=not args.apply,
        )
    except (AgentMetaSidecarPurgeApplyError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        payload = {
            "applied": report.applied,
            "scanned_count": report.scanned_count,
            "purged_count": report.purged_count,
            "purged_bytes": report.purged_bytes,
            "purged_session_ids": list(report.purged_session_ids),
            "shape_mismatch_count": report.shape_mismatch_count,
            "backup_manifest": str(report.backup_manifest) if report.backup_manifest is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    def _mb(byte_count: int) -> str:
        return f"{byte_count / (1024 * 1024):.2f} MB"

    mode = "APPLIED" if report.applied else "dry-run (no mutation performed -- pass --apply to purge)"
    print(f"mode: {mode}", file=stdout)
    print(f"agent-meta-sidecar sessions scanned: {report.scanned_count}", file=stdout)
    print(
        f"{'purged' if report.applied else 'purgeable'}: {report.purged_count:>7}  ({_mb(report.purged_bytes)})",
        file=stdout,
    )
    if report.shape_mismatch_count:
        print(
            f"WARNING: {report.shape_mismatch_count} candidates fail the native_id shape cross-check "
            "-- --apply is refused until this is investigated",
            file=stdout,
        )
    if report.applied:
        print(f"backup manifest used: {report.backup_manifest}", file=stdout)
        print(
            "Each purged session has an immutable receipt in "
            "agent_meta_sidecar_purge_receipts. raw_sessions rows and blobs in "
            "source.db were not touched.",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
