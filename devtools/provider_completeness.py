"""Report provider/importer package completeness."""

from __future__ import annotations

import argparse
import json
import sys

from polylogue.sources.provider_completeness import accepted_blockers, provider_package_completeness
from polylogue.surfaces.payloads import ProviderPackageCompletenessPayload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", help="Limit rows to a public origin or provider-wire token.")
    parser.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when an accepted provider package has missing or partial required items.",
    )
    return parser


def _print_table(report: ProviderPackageCompletenessPayload) -> None:
    rows = report.rows
    print("Provider package completeness")
    print()
    print(f"{'origin':<24} {'capture mode':<30} {'maturity':<9} {'status':<8} blockers")
    print(f"{'-' * 24} {'-' * 30} {'-' * 9} {'-' * 8} {'-' * 8}")
    for row in rows:
        blocker_text = "; ".join(row.blockers[:2])
        if len(row.blockers) > 2:
            blocker_text += f"; +{len(row.blockers) - 2} more"
        print(f"{row.origin:<24} {row.capture_mode:<30} {row.maturity:<9} {row.status:<8} {blocker_text}")
    print()
    print(
        "totals: "
        f"{report.totals.total} rows, "
        f"{report.totals.complete} complete, "
        f"{report.totals.partial} partial, "
        f"{report.totals.proposed} proposed, "
        f"{report.totals.accepted_blocked} accepted blocked"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = provider_package_completeness(origin=args.origin)
    if args.json:
        json.dump(report.model_dump(mode="json"), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(report)
    blockers = accepted_blockers(report)
    if args.check and blockers:
        if not args.json:
            print("\naccepted provider-package blockers:", file=sys.stderr)
            for blocker in blockers:
                print(f"- {blocker}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
