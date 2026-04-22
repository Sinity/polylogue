"""Render proof obligations affected by changed paths or refs."""

from __future__ import annotations

import argparse
import json
import sys

from polylogue.proof.diffing import (
    build_affected_obligation_report,
    changed_paths_between_refs,
    obligation_ids_for_ref,
    render_affected_obligations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="master", help="Base git ref for changed-path and obligation diffing.")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref for changed-path and obligation diffing.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Changed repo-relative path. Repeat to bypass git diff path discovery.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable affected-obligation report.")
    args = parser.parse_args(argv)

    paths = tuple(args.path) if args.path else changed_paths_between_refs(args.base_ref, args.head_ref)
    base_ids = obligation_ids_for_ref(args.base_ref)
    head_ids = obligation_ids_for_ref(args.head_ref)
    report = build_affected_obligation_report(
        paths,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        base_obligation_ids=base_ids,
        head_obligation_ids=head_ids,
    )
    if args.json:
        json.dump(report.to_payload(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    sys.stdout.write(render_affected_obligations(report))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
