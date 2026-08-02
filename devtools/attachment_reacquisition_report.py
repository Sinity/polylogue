"""Read-only report: classify historically-unfetched attachments for backfill.

polylogue-pfdf: 79% of ``attachments`` rows sit at
``acquisition_status='unfetched'`` -- bytes were never fetched. This command
runs ``polylogue.storage.attachment_reacquisition.plan_attachment_reacquisition``
against the live archive (``index.db``/``source.db`` both opened strictly
read-only) and reports, per unfetched attachment, whether re-parsing its
still-durable raw payload with today's parser proves it recoverable, a static
signature proves it structurally unrecoverable (e.g. ChatGPT Code Interpreter
sandbox output), or neither is provable yet (most commonly a Drive/OAuth
attachment that would need a live authenticated fetch this classifier never
performs).

This is a dry-run classifier only. It never mutates ``index.db`` or the blob
store -- acting on a "reacquirable"/"unrecoverable" classification is the
separate, explicitly operator-authorized
``devtools workspace attachment-reacquisition-apply``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.paths import archive_root as default_archive_root
from polylogue.storage.attachment_reacquisition import plan_attachment_reacquisition


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root to inspect; defaults to the configured active archive root.",
    )
    parser.add_argument(
        "--raw-row-limit",
        type=int,
        default=None,
        help="Cap how many raw_sessions rows the reparse scan inspects (unbounded by default).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="How many rows per bucket to print in the human-readable report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full classified plan as JSON.")
    args = parser.parse_args(argv)

    root = args.archive_root if args.archive_root is not None else default_archive_root()
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        print(f"no index.db at {index_db}", file=stdout)
        return 1
    if not source_db.exists():
        print(f"no source.db at {source_db}", file=stdout)
        return 1

    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        plan = plan_attachment_reacquisition(
            index_conn,
            source_conn,
            archive_root=root,
            blob_root=root / "blob",
            raw_row_limit=args.raw_row_limit,
        )
    finally:
        index_conn.close()
        source_conn.close()

    if args.json:
        payload = plan.to_dict()
        payload["reacquirable_sample"] = [
            {
                "attachment_id": item.attachment_id,
                "raw_id": item.raw_id,
                "session_id": item.session_id,
                "byte_count": item.byte_count,
                "mime_type": item.mime_type,
            }
            for item in plan.reacquirable[: args.sample_limit]
        ]
        payload["unrecoverable_sample"] = [
            {"attachment_id": item.attachment_id, "reason": item.reason}
            for item in plan.unrecoverable[: args.sample_limit]
        ]
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0

    print(f"unfetched attachments: {plan.unfetched_count}", file=stdout)
    print(
        f"reacquirable (raw reparse now yields bytes): {len(plan.reacquirable):>6} "
        f"({sum(item.byte_count for item in plan.reacquirable)} bytes)",
        file=stdout,
    )
    print(f"unrecoverable (structurally impossible):     {len(plan.unrecoverable):>6}", file=stdout)
    print(f"undetermined (left unfetched, not resolved): {plan.undetermined_count:>6}", file=stdout)
    print(f"raw_sessions rows scanned: {plan.raw_rows_scanned} of {plan.raw_rows_total}", file=stdout)
    print("", file=stdout)
    print(
        "This is a read-only classification. No attachment or blob was mutated -- "
        "run `devtools workspace attachment-reacquisition-apply` to act on it.",
        file=stdout,
    )
    if plan.reacquirable:
        print(f"\nreacquirable sample (up to {args.sample_limit}):", file=stdout)
        for item in plan.reacquirable[: args.sample_limit]:
            print(f"  {item.attachment_id}  raw={item.raw_id}  {item.byte_count}B  {item.mime_type}", file=stdout)
    if plan.unrecoverable:
        print(f"\nunrecoverable sample (up to {args.sample_limit}):", file=stdout)
        for unrecoverable_item in plan.unrecoverable[: args.sample_limit]:
            print(f"  {unrecoverable_item.attachment_id}  {unrecoverable_item.reason}", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
