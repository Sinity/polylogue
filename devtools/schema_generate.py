"""Generate provider schema packages from the devtools surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polylogue.cli.schema_command_support import build_schema_privacy_config
from polylogue.cli.schema_rendering import render_schema_generate_result
from polylogue.config import get_config
from polylogue.schemas.operator_models import SchemaInferRequest
from polylogue.schemas.operator_workflow import infer_schema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate provider schema packages and optional evidence clusters.")
    parser.add_argument("--provider", required=True, help="Provider to generate schema for.")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also cluster observed samples by structural fingerprint.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for generation.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument(
        "--privacy",
        choices=("strict", "standard", "permissive"),
        default=None,
        help="Privacy preset level. Defaults to standard.",
    )
    parser.add_argument("--privacy-config", type=Path, default=None, help="Path to TOML privacy config overrides.")
    parser.add_argument("--report", action="store_true", help="Write a redaction report alongside the schema.")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Bypass all sample caps for full-corpus schema generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        privacy_config = build_schema_privacy_config(
            privacy=args.privacy,
            privacy_config_path=args.privacy_config,
        )
        result = infer_schema(
            SchemaInferRequest(
                provider=str(args.provider),
                db_path=get_config().db_path,
                max_samples=args.max_samples,
                privacy_config=privacy_config,
                cluster=bool(args.cluster),
                full_corpus=bool(args.full_corpus),
            )
        )
    except ValueError as exc:
        print(f"schema-generate: {exc}", file=sys.stderr)
        return 1

    generation = result.generation
    if not generation.success:
        print(f"schema-generate: {generation.error or 'Schema generation failed'}", file=sys.stderr)
        return 1
    if args.cluster and result.manifest is None:
        print("schema-generate: No samples found for clustering", file=sys.stderr)
        return 1

    render_schema_generate_result(
        provider=str(args.provider),
        result=result,
        json_output=bool(args.json),
        report=bool(args.report),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
