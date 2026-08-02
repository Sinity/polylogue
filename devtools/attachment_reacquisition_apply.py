"""Actuator: backfill acquisition for historically-unfetched attachments.

polylogue-pfdf: ``devtools workspace attachment-reacquisition`` only
classifies and reports (see :mod:`polylogue.storage.attachment_reacquisition`).
This command acts on the classifier's verdict: it publishes freshly re-derived
bytes for every ``reacquirable`` attachment (a re-parse of its still-durable
``raw_sessions`` payload with today's parser code reproduces its content-
identity hash *with* inline bytes) and marks every ``unrecoverable`` attachment
(currently: ChatGPT Code Interpreter sandbox output, which the export never
carries) as ``acquisition_status='unavailable'`` instead of leaving it
ambiguously ``'unfetched'`` forever. Attachments this pass cannot prove either
way (``undetermined`` -- most commonly Drive/OAuth attachments needing a live
authenticated fetch this module never performs) are left untouched.

Default mode is dry-run (report only, zero mutation). Pass ``--apply`` to
actually mutate, which additionally requires ``--manifest-path`` (an immutable
JSONL receipt of every row acted on) and ``--backup-manifest`` pointing at a
verified backup manifest for the ``index`` tier (see
``polylogue backup --output-dir <dir> --verify``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.attachment_reacquisition import (
    AttachmentReacquisitionError,
    apply_attachment_reacquisition,
)


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to operate on; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--raw-row-limit",
        type=int,
        default=None,
        help="Cap how many raw_sessions rows the reparse scan inspects (unbounded when omitted).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Cap how many reacquirable/unrecoverable attachments one apply pass acts on.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually mutate. Without this flag, nothing is written.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path for the immutable JSONL receipt of every row acted on. Required with --apply.",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        default=None,
        help="Verified backup manifest for the index tier. Required with --apply.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()

    try:
        result = apply_attachment_reacquisition(
            root,
            manifest_path=args.manifest_path,
            backup_manifest=args.backup_manifest,
            raw_row_limit=args.raw_row_limit,
            max_count=args.max_count,
            dry_run=not args.apply,
        )
    except (AttachmentReacquisitionError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=stdout)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True), file=stdout)
        return 0

    mode = "APPLIED" if result.applied else "dry-run (no mutation performed -- pass --apply to act)"
    print(f"mode: {mode}", file=stdout)
    print(f"unfetched attachments seen: {result.unfetched_count}", file=stdout)
    print(
        f"{'reacquired' if result.applied else 'reacquirable'}:       {result.reacquired_count:>6} "
        f"(of {result.reacquirable_count} classified reacquirable, {result.reacquired_bytes} bytes)",
        file=stdout,
    )
    print(
        f"{'marked unavailable' if result.applied else 'unrecoverable'}: "
        f"{result.marked_unavailable_count:>6} (of {result.unrecoverable_count} classified unrecoverable)",
        file=stdout,
    )
    print(f"left unfetched, undetermined:      {result.undetermined_count:>6}", file=stdout)
    if result.errors:
        print(f"errors ({len(result.errors)}):", file=stdout)
        for error in result.errors:
            print(f"  {error}", file=stdout)
    if result.applied:
        print(f"manifest: {result.manifest_path}", file=stdout)
        print(f"backup manifest used: {result.backup_manifest}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
