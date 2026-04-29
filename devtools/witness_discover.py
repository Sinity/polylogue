"""Save an arbitrary input that triggered a failure as a local witness.

Usage:
  devtools witness-discover --input <path> --witness-id <id> [--origin <origin>]
  devtools witness-discover --stdin --witness-id <id> [--origin <origin>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from polylogue.lib.json import dumps as json_dumps
from polylogue.proof.witnesses import (
    LOCAL_WITNESS_INBOX,
    WITNESS_SCHEMA_VERSION,
    WitnessLifecycle,
    WitnessMetadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save a failure-triggering input as a local witness.")
    parser.add_argument("--input", type=Path, help="Path to the input file that triggered the failure.")
    parser.add_argument("--stdin", action="store_true", help="Read witness content from stdin.")
    parser.add_argument("--witness-id", required=True, type=str, help="Unique witness identifier (slug).")
    parser.add_argument(
        "--origin",
        type=str,
        default="regression",
        choices=["golden-surface-snapshot", "live-derived", "synthetic", "external", "regression"],
        help="Provenance of the witness (default: regression).",
    )
    parser.add_argument(
        "--semantic-facts",
        type=str,
        nargs="*",
        default=(),
        help="Semantic facts this witness preserves (space-separated).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        nargs="*",
        default=(),
        help="Optional notes about the discovery.",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.stdin and args.input is None:
        parser.error("one of --input or --stdin is required")
    if args.stdin and args.input is not None:
        parser.error("--input and --stdin are mutually exclusive")

    LOCAL_WITNESS_INBOX.mkdir(parents=True, exist_ok=True)

    if args.stdin:
        content = sys.stdin.read()
        witness_path = LOCAL_WITNESS_INBOX / f"{args.witness_id}.witness.txt"
        witness_path.write_text(content.strip(), encoding="utf-8")
    elif args.input is not None:
        witness_path = LOCAL_WITNESS_INBOX / f"{args.witness_id}.witness{args.input.suffix}"
        shutil.copy2(args.input, witness_path)
    else:
        return 1

    lifecycle = WitnessLifecycle.new()
    metadata = WitnessMetadata(
        witness_id=args.witness_id,
        path=str(witness_path),
        origin=args.origin,
        provenance={
            "discovery_command": " ".join(sys.argv),
            "discovery_cwd": str(Path.cwd()),
        },
        preserved_semantic_facts=tuple(args.semantic_facts),
        minimization_status="raw",
        lifecycle=lifecycle,
        committed=False,
        notes=tuple(args.notes),
        schema_version=WITNESS_SCHEMA_VERSION,
    )

    metadata_path = LOCAL_WITNESS_INBOX / f"{args.witness_id}.metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_payload(), indent=2), encoding="utf-8")

    if args.json:
        print(json_dumps({"witness_id": args.witness_id, "path": str(witness_path), "state": "discovered"}))
    else:
        print(f"discovered: {args.witness_id}")
        print(f"  witness: {witness_path}")
        print(f"  metadata: {metadata_path}")
        print("  state: discovered")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
