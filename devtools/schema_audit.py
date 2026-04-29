"""Audit committed schema packages from the devtools surface."""

from __future__ import annotations

import argparse

from polylogue.cli.shared.schema_rendering import render_schema_audit_result
from polylogue.schemas.operator.models import SchemaAuditRequest
from polylogue.schemas.operator.workflow import audit_schemas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quality checks on committed schema packages.")
    parser.add_argument("--provider", default=None, help="Audit a specific provider. Defaults to all providers.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = audit_schemas(SchemaAuditRequest(provider=args.provider))
    render_schema_audit_result(report=report, json_output=bool(args.json))
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
