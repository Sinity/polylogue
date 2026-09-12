"""Verify provider/importer package completeness declarations."""

from __future__ import annotations

import argparse
import sys

from polylogue.sources.provider_completeness import accepted_blockers, provider_package_completeness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report provider/importer package completeness.")
    parser.add_argument(
        "--origin",
        help="Limit the report to a public origin or provider-wire token.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when an accepted provider package has missing or partial required evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the declaration-backed provider completeness report."""
    args = _parser().parse_args(argv)
    report = provider_package_completeness(origin=args.origin)
    blockers = accepted_blockers(report)

    if args.json:
        print(report.to_json())
    else:
        totals = report.totals
        print(
            "provider completeness: "
            f"{totals.complete}/{totals.total} complete; "
            f"{totals.partial} partial; {totals.missing} missing; "
            f"{totals.reserved} reserved"
        )
        for row in report.rows:
            suffix = f" ({'; '.join(row.blockers)})" if row.blockers else ""
            print(f"  {row.package_ref}: {row.status}{suffix}")
        if blockers and args.check:
            print("accepted provider completeness blockers:", file=sys.stderr)
            for blocker in blockers:
                print(f"  - {blocker}", file=sys.stderr)

    return 1 if args.check and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
