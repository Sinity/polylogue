"""Diff proof obligations between two git refs.

Usage:
  devtools obligation-diff --base <ref> --head <ref>
  devtools obligation-diff --base origin/master --head HEAD --json
"""

from __future__ import annotations

import argparse

from polylogue.lib.json import dumps
from polylogue.proof.diffing import (
    build_affected_obligation_report,
    changed_paths_between_refs,
    render_affected_obligations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff proof obligations between git refs.")
    parser.add_argument("--base", required=True, type=str, help="Base git ref (e.g. origin/master).")
    parser.add_argument("--head", type=str, default="HEAD", help="Head git ref (default: HEAD).")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = changed_paths_between_refs(args.base, args.head)
    if not paths:
        print("No changed paths found.")
        return 0

    report = build_affected_obligation_report(paths)
    if args.json:
        print(dumps(report.to_payload()))
    else:
        rendered = render_affected_obligations(report)
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
