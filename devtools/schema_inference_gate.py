"""Run the schema-inference prerequisite and persist its go/no-go receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from polylogue.maintenance.schema_inference_gate import run_schema_inference_gate


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True, help="Archive root to inspect read-only.")
    parser.add_argument(
        "--blob-hash-receipt",
        type=Path,
        required=True,
        help="Required receipt from a separate full blob-hash verification run.",
    )
    parser.add_argument("--receipt", type=Path, required=True, help="Destination for the durable gate receipt.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Maximum evidence rows per failure class.")
    parser.add_argument("--json", action="store_true", help="Emit the durable receipt as JSON.")
    args = parser.parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    result = run_schema_inference_gate(
        args.archive_root,
        blob_hash_receipt=args.blob_hash_receipt,
        receipt_path=args.receipt,
        sample_limit=args.sample_limit,
    )
    if args.json:
        json.dump(result.payload, output, indent=2, sort_keys=True)
        print(file=output)
    else:
        print(f"Schema-inference gate: {result.payload['archive_root']}", file=output)
        print(f"  receipt: {args.receipt}", file=output)
        print(f"  verdict: {result.payload['verdict']}", file=output)
        reasons = result.payload.get("pass_fail_reasons", [])
        if isinstance(reasons, list):
            for reason in reasons:
                print(f"  reason: {reason}", file=output)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
