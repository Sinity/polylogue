"""Promote a local witness to the committed witness directory.

Usage:
  devtools witness-promote <witness-id>
  devtools witness-promote <witness-id> --known-failing --rejection-reason '...'
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from polylogue.proof.witnesses import (
    COMMITTED_WITNESS_DIR,
    LOCAL_WITNESS_INBOX,
    WITNESS_SCHEMA_VERSION,
    WitnessLifecycle,
    WitnessMetadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a local witness to the committed directory.")
    parser.add_argument("witness_id", type=str, help="Witness identifier to promote.")
    parser.add_argument("--known-failing", action="store_true", help="Mark the witness as known failing (xfail).")
    parser.add_argument(
        "--rejection-reason", type=str, help="Required rationale when promoting a known-failing witness."
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def _build_witness_index(witness_dir: Path) -> None:
    """Rebuild tests/witnesses/index.json."""
    index_entries: list[dict[str, object]] = []
    for meta_path in sorted(witness_dir.glob("*.metadata.json")):
        try:
            meta = WitnessMetadata.from_payload(json.loads(meta_path.read_text(encoding="utf-8")))
            index_entries.append(
                {
                    "witness_id": meta.witness_id,
                    "origin": meta.origin,
                    "minimization_status": meta.minimization_status,
                    "last_exercised_at": meta.lifecycle.last_exercised_at if meta.lifecycle else None,
                }
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    index_path = witness_dir / "index.json"
    index_path.write_text(json.dumps(index_entries, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    metadata_path = LOCAL_WITNESS_INBOX / f"{args.witness_id}.metadata.json"
    if not metadata_path.exists():
        print(f"error: witness '{args.witness_id}' not found in {LOCAL_WITNESS_INBOX}", file=sys.stderr)
        return 1

    metadata = WitnessMetadata.from_payload(json.loads(metadata_path.read_text(encoding="utf-8")))
    if args.known_failing and not (args.rejection_reason or metadata.rejection_reason):
        print("error: --known-failing requires --rejection-reason", file=sys.stderr)
        return 1

    if metadata.privacy_classification is None:
        print(
            "error: witness must have a privacy_classification before promotion. "
            "Run 'devtools witness-minimize --privacy-classification <cls>' first.",
            file=sys.stderr,
        )
        return 1

    COMMITTED_WITNESS_DIR.mkdir(parents=True, exist_ok=True)

    witness_path = Path(metadata.path)
    committed_witness = COMMITTED_WITNESS_DIR / witness_path.name
    committed_metadata = COMMITTED_WITNESS_DIR / f"{args.witness_id}.metadata.json"

    if witness_path.exists():
        shutil.copy2(witness_path, committed_witness)

    committed_metadata_obj = WitnessMetadata(
        witness_id=metadata.witness_id,
        path=str(committed_witness),
        origin=metadata.origin,
        provenance=metadata.provenance,
        preserved_semantic_facts=metadata.preserved_semantic_facts,
        minimization_status=metadata.minimization_status,
        privacy=metadata.privacy,
        privacy_classification=metadata.privacy_classification,
        lifecycle=(
            metadata.lifecycle.transition("committed")
            if metadata.lifecycle is not None
            else WitnessLifecycle.new().transition("committed")
        ),
        committed=True,
        known_failing=args.known_failing or metadata.known_failing,
        xfail_strict=args.known_failing,
        rejection_reason=args.rejection_reason or metadata.rejection_reason,
        notes=metadata.notes,
        schema_version=WITNESS_SCHEMA_VERSION,
    )

    committed_metadata.write_text(json.dumps(committed_metadata_obj.to_payload(), indent=2), encoding="utf-8")
    _build_witness_index(COMMITTED_WITNESS_DIR)

    # Remove from inbox
    metadata_path.unlink(missing_ok=True)
    if witness_path.exists():
        witness_path.unlink(missing_ok=True)

    if args.json:
        print(json.dumps({"witness_id": args.witness_id, "state": "committed"}))
    else:
        print(f"promoted: {args.witness_id}")
        print(f"  witness: {committed_witness}")
        print(f"  metadata: {committed_metadata}")
        if args.known_failing:
            print(f"  xfail: strict ({args.rejection_reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
