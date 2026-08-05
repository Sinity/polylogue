"""Commit a real full-corpus schema generation into committed packages.

``devtools lab schema generate`` only ever produces a preview
``GenerationResult`` -- it never writes to ``polylogue/schemas/providers/``.
This command is the actual persisting entry point (polylogue-k45pq):
it calls ``generate_all_schemas`` for real via
``polylogue.schemas.operator.commit.commit_provider_schema`` and reports
which package versions changed, plus whether any previously-committed leaf
type was lost or narrowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polylogue.cli.shared.schema_command_support import build_schema_privacy_config
from polylogue.config import get_config
from polylogue.schemas.operator.commit import commit_provider_schema
from polylogue.schemas.operator.models import SchemaCommitRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "polylogue" / "schemas" / "providers"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit a real full-corpus provider schema generation into committed packages."
    )
    parser.add_argument("--provider", required=True, help="Provider to generate and commit schema for.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Committed schema package root to write into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for generation.")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        default=True,
        help="Bypass all sample caps for full-corpus schema generation (default: on).",
    )
    parser.add_argument(
        "--no-full-corpus",
        dest="full_corpus",
        action="store_false",
        help="Generate from a capped sample window instead of the full corpus.",
    )
    parser.add_argument(
        "--privacy",
        choices=("strict", "standard", "permissive"),
        default=None,
        help="Privacy preset level. Defaults to standard.",
    )
    parser.add_argument("--privacy-config", type=Path, default=None, help="Path to TOML privacy config overrides.")
    parser.add_argument(
        "--schema-inference-gate-receipt",
        type=Path,
        required=True,
        help="Accepted PASS receipt from devtools verify schema-inference-gate.",
    )
    parser.add_argument(
        "--dry-run",
        "--check",
        dest="dry_run",
        action="store_true",
        help="Preview what a commit would change without writing to --output-dir.",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        privacy_config = build_schema_privacy_config(
            privacy=args.privacy,
            privacy_config_path=args.privacy_config,
        )
    except ValueError as exc:
        print(f"schema-commit: {exc}", file=sys.stderr)
        return 1
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    result = commit_provider_schema(
        SchemaCommitRequest(
            provider=str(args.provider),
            output_dir=output_dir,
            db_path=get_config().db_path,
            max_samples=args.max_samples,
            privacy_config=privacy_config,
            full_corpus=bool(args.full_corpus),
            dry_run=bool(args.dry_run),
            schema_inference_gate_receipt_path=args.schema_inference_gate_receipt,
        )
    )

    if not result.success:
        error = result.generation.error or "Schema generation failed"
        if args.json:
            print(json.dumps({"provider": result.provider, "success": False, "error": error}, sort_keys=True))
        else:
            print(f"schema-commit: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    else:
        mode = "DRY RUN (no files written)" if result.dry_run else f"committed to {output_dir}"
        print(f"schema-commit: {result.provider} -- {mode}")
        print(f"  sample_count={result.generation.sample_count}")
        if result.handoff is not None:
            print(f"  handoff_digest={result.handoff.receipt_digest}")
            if result.handoff_path is not None:
                print(f"  handoff_path={result.handoff_path}")
        for version_report in result.versions:
            flags = []
            if version_report.narrowed_paths:
                flags.append(f"NARROWED({len(version_report.narrowed_paths)})")
            if version_report.added_paths:
                flags.append(f"added({len(version_report.added_paths)})")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"  {version_report.version}: {version_report.status} "
                f"sample_count={version_report.sample_count}{suffix}"
            )
        if result.narrowed:
            print(
                "  WARNING: a previously-committed leaf type was lost or narrowed -- "
                "review before trusting this commit.",
                file=sys.stderr,
            )

    return 1 if result.narrowed else 0


if __name__ == "__main__":
    raise SystemExit(main())
