"""Inspect committed schema packages from the devtools lab surface."""

from __future__ import annotations

import argparse
import sys

from polylogue.cli.shared.schema_rendering import (
    render_schema_compare_result,
    render_schema_explain_result,
    render_schema_list_result,
)
from polylogue.schemas.operator.models import SchemaCompareRequest, SchemaExplainRequest, SchemaListRequest
from polylogue.schemas.operator.workflow import compare_schema_versions, explain_schema, list_schemas


def _list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List available schema packages, versions, and evidence manifests.")
    parser.add_argument("--provider", "-p", default=None, help="Filter to specific provider.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    return parser


def _compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two schema package versions for a provider.")
    parser.add_argument("--provider", "-p", required=True, help="Provider name.")
    parser.add_argument("--from", dest="from_version", required=True, help="Source version, e.g. v1.")
    parser.add_argument("--to", dest="to_version", required=True, help="Target version, e.g. v2.")
    parser.add_argument("--element", dest="element_kind", default=None, help="Element kind inside the package.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown.")
    return parser


def _explain_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain a package element schema with evidence and annotations.")
    parser.add_argument("--provider", "-p", required=True, help="Provider name.")
    parser.add_argument("--version", default="latest", help="Schema version. Defaults to latest.")
    parser.add_argument("--element", dest="element_kind", default=None, help="Element kind inside the package.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show semantic roles and coverage.")
    parser.add_argument("--review-evidence", action="store_true", help="Show evidence for role assignment decisions.")
    return parser


def list_main(argv: list[str] | None = None) -> int:
    args = _list_parser().parse_args(argv)
    result = list_schemas(SchemaListRequest(provider=args.provider))
    render_schema_list_result(provider=args.provider, result=result, json_output=bool(args.json))
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    args = _compare_parser().parse_args(argv)
    try:
        result = compare_schema_versions(
            SchemaCompareRequest(
                provider=args.provider,
                from_version=args.from_version,
                to_version=args.to_version,
                element_kind=args.element_kind,
            )
        )
    except ValueError as exc:
        print(f"schema-compare: {exc}", file=sys.stderr)
        return 1
    render_schema_compare_result(result=result, json_output=bool(args.json), md_output=bool(args.markdown))
    return 0


def explain_main(argv: list[str] | None = None) -> int:
    args = _explain_parser().parse_args(argv)
    try:
        result = explain_schema(
            SchemaExplainRequest(
                provider=args.provider,
                version=args.version,
                element_kind=args.element_kind,
                review_evidence=bool(args.review_evidence),
            )
        )
    except ValueError as exc:
        print(f"schema-explain: {exc}", file=sys.stderr)
        return 1
    render_schema_explain_result(result=result, json_output=bool(args.json), verbose=bool(args.verbose))
    return 0


__all__ = ["compare_main", "explain_main", "list_main"]
