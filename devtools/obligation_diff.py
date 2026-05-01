"""Diff proof obligations between two git refs.

Usage:
  devtools obligation-diff --base-ref <ref> --head-ref <ref>
  devtools obligation-diff --base-ref origin/master --head-ref HEAD --json
"""

from __future__ import annotations

import argparse

from polylogue.core.json import dumps
from polylogue.proof.diffing import (
    build_affected_obligation_report,
    changed_paths_between_refs,
    obligation_ids_for_ref,
    render_affected_obligations,
    render_affected_obligations_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff proof obligations between git refs.")
    parser.add_argument("--base-ref", required=True, type=str, help="Base git ref (e.g. origin/master).")
    parser.add_argument("--head-ref", type=str, default="HEAD", help="Head git ref (default: HEAD).")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--markdown", action="store_true", help="Output PR-comment-ready Markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = changed_paths_between_refs(args.base_ref, args.head_ref)
    if not paths:
        print("No changed paths found.")
        return 0

    report = build_affected_obligation_report(
        paths,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        base_obligation_ids=obligation_ids_for_ref(args.base_ref),
        head_obligation_ids=obligation_ids_for_ref(args.head_ref),
    )
    if args.json:
        print(dumps(report.to_payload()))
    elif args.markdown:
        print(render_affected_obligations_markdown(report))
    else:
        rendered = render_affected_obligations(report)
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
