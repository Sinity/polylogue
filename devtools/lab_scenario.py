"""Verification-lab showcase scenario runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polylogue.showcase.qa_runner_reporting import format_qa_summary
from polylogue.showcase.qa_runner_request import QAStage, build_qa_session_request
from polylogue.showcase.qa_runner_workflow import run_qa_session
from polylogue.showcase.qa_session_payload import generate_qa_session

_SCENARIO_NAMES = ("archive-smoke",)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verification-lab showcase scenarios.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a named showcase scenario set.")
    run_parser.add_argument("scenario", choices=_SCENARIO_NAMES, help="Scenario set to run.")
    run_parser.add_argument(
        "--live", action="store_true", help="Run against the active archive instead of a seeded workspace."
    )
    run_parser.add_argument("--tier", type=int, default=None, help="Only run exercises at this tier.")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Directory for scenario artifacts.")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable QA session payload.")
    run_parser.add_argument("--verbose", action="store_true", help="Print exercise outputs.")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first exercise failure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action != "run":
        parser.error(f"unknown action: {args.action}")
    request = build_qa_session_request(
        synthetic=not bool(args.live),
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=QAStage.EXERCISES,
        skip_stages=(),
        workspace=None,
        report_dir=args.report_dir,
        verbose=bool(args.verbose),
        fail_fast=bool(args.fail_fast),
        tier_filter=args.tier,
    )
    result = run_qa_session(request)
    if args.json:
        print(json.dumps(generate_qa_session(result), indent=2))
    else:
        print(format_qa_summary(result))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
