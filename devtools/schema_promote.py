"""Promote schema evidence clusters from the devtools surface."""

from __future__ import annotations

import argparse
import sys

from polylogue.cli.shared.schema_rendering import render_schema_promote_result
from polylogue.config import get_config
from polylogue.schemas.operator.models import SchemaPromoteRequest
from polylogue.schemas.operator.workflow import promote_schema_cluster


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote an evidence cluster to a registered schema package.")
    parser.add_argument("--provider", required=True, help="Provider name.")
    parser.add_argument("--cluster", dest="cluster_id", required=True, help="Evidence cluster ID to promote.")
    parser.add_argument("--with-samples", action="store_true", help="Re-load samples for full schema generation.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples when using --with-samples.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = promote_schema_cluster(
            SchemaPromoteRequest(
                provider=str(args.provider),
                cluster_id=str(args.cluster_id),
                db_path=get_config().db_path,
                with_samples=bool(args.with_samples),
                max_samples=int(args.max_samples),
            )
        )
    except ValueError as exc:
        print(f"schema-promote: {exc}", file=sys.stderr)
        return 1

    render_schema_promote_result(result=result, json_output=bool(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
