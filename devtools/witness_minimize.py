"""Apply minimization heuristics to a local witness.

Usage:
  devtools witness-minimize <witness-id>
  devtools witness-minimize <witness-id> --privacy-classification <classification>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polylogue.core.json import dumps as json_dumps
from polylogue.proof.witnesses import (
    LOCAL_WITNESS_INBOX,
    WITNESS_SCHEMA_VERSION,
    PrivacyClassification,
    WitnessLifecycle,
    WitnessMetadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimize a local witness — shrink, redact, and normalize.")
    parser.add_argument("witness_id", type=str, help="Witness identifier to minimize.")
    parser.add_argument(
        "--privacy-classification",
        type=str,
        choices=["synthetic", "redacted", "public"],
        help="Set privacy classification after minimization.",
    )
    parser.add_argument(
        "--semantic-facts",
        type=str,
        nargs="*",
        help="Replace preserved semantic facts after minimization.",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def _minimize_json(content: str) -> str:
    """Shrink a JSON payload: strip whitespace, remove null fields."""
    try:
        data = json.loads(content)
        minimized = _strip_nulls(data)
        return json_dumps(minimized)
    except (json.JSONDecodeError, TypeError):
        return content


def _strip_nulls(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    metadata_path = LOCAL_WITNESS_INBOX / f"{args.witness_id}.metadata.json"
    if not metadata_path.exists():
        print(f"error: witness '{args.witness_id}' not found in {LOCAL_WITNESS_INBOX}", file=sys.stderr)
        return 1

    metadata = WitnessMetadata.from_payload(json.loads(metadata_path.read_text(encoding="utf-8")))

    # Apply JSON minimization to the witness file if it's JSON
    witness_path = Path(metadata.path)
    if witness_path.exists() and witness_path.suffix in (".json",):
        original = witness_path.read_text(encoding="utf-8")
        minimized = _minimize_json(original)
        if len(minimized) < len(original):
            witness_path.write_text(minimized, encoding="utf-8")

    privacy_class: PrivacyClassification | None = None
    if args.privacy_classification:
        privacy_class = args.privacy_classification

    updated_metadata = WitnessMetadata(
        witness_id=metadata.witness_id,
        path=metadata.path,
        origin=metadata.origin,
        provenance={**metadata.provenance, "minimized_at": str(Path.cwd())},
        preserved_semantic_facts=(
            tuple(args.semantic_facts) if args.semantic_facts else metadata.preserved_semantic_facts
        ),
        minimization_status="minimized",
        privacy=metadata.privacy,
        privacy_classification=privacy_class,
        lifecycle=(
            metadata.lifecycle.transition("minimized")
            if metadata.lifecycle is not None
            else WitnessLifecycle.new().transition("minimized")
        ),
        committed=False,
        notes=metadata.notes,
        schema_version=WITNESS_SCHEMA_VERSION,
    )

    metadata_path.write_text(json.dumps(updated_metadata.to_payload(), indent=2), encoding="utf-8")

    if args.json:
        print(json_dumps({"witness_id": args.witness_id, "state": "minimized"}))
    else:
        print(f"minimized: {args.witness_id}")
        if privacy_class:
            print(f"  privacy_classification: {privacy_class}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
