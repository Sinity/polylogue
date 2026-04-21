"""Capture pipeline-probe summaries as durable local regression cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from devtools.regression_cases import (
    DEFAULT_REGRESSION_CASE_DIR,
    RegressionCase,
    RegressionCaseStore,
    json_input_document,
    regression_case_path_payload,
)


def _read_input(input_path: Path | None) -> str:
    if input_path is None or str(input_path) == "-":
        return sys.stdin.read()
    return input_path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a devtools pipeline-probe JSON summary as a durable regression case.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Pipeline-probe JSON summary path. Reads stdin when omitted or set to '-'.",
    )
    parser.add_argument("--name", required=True, help="Human-readable regression case name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REGRESSION_CASE_DIR,
        help=f"Output directory for captured cases (default: {DEFAULT_REGRESSION_CASE_DIR}).",
    )
    parser.add_argument("--tag", action="append", default=[], help="Tag to attach to the captured case. Repeatable.")
    parser.add_argument("--note", action="append", default=[], help="Note to attach to the captured case. Repeatable.")
    parser.add_argument("--json", action="store_true", help="Emit the captured case payload as JSON.")
    args = parser.parse_args(argv)

    summary = json_input_document(_read_input(args.input))
    case = RegressionCase.from_probe_summary(
        name=args.name,
        summary=summary,
        tags=tuple(args.tag),
        notes=tuple(args.note),
    )
    output_path = RegressionCaseStore(args.output_dir).write(case)

    if args.json:
        print(json.dumps(regression_case_path_payload(case, output_path), indent=2, sort_keys=True))
        return 0
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
