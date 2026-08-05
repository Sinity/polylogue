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
        "--ground-truth-root",
        action="append",
        default=[],
        metavar="ORIGIN=PATH",
        help="External source root to scan and reconcile for an origin; repeat for each origin in source.db.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="External destination named schema-inference-gate-receipt.json.",
    )
    parser.add_argument("--sample-limit", type=int, default=10, help="Maximum evidence rows per failure class.")
    parser.add_argument("--json", action="store_true", help="Emit the durable receipt as JSON.")
    args = parser.parse_args(argv)
    ground_truth_roots: dict[str, list[Path]] = {}
    for raw in args.ground_truth_root:
        origin, separator, path = raw.partition("=")
        if not separator or not origin or not path:
            parser.error("--ground-truth-root must be ORIGIN=PATH")
        ground_truth_roots.setdefault(origin, []).append(Path(path))
    output = stdout if stdout is not None else sys.stdout
    result = run_schema_inference_gate(
        args.archive_root,
        receipt_path=args.receipt,
        ground_truth_roots=ground_truth_roots,
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
