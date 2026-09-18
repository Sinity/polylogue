"""Declare, record and check the schema-source frontier.

The frontier is the set of source roots a provider schema run is allowed to
read, plus the recorded membership of each one. Generation binds the recorded
baseline digest instead of a reading taken over live mutable roots, so a moved
or emptied root fails the check rather than silently narrowing the sample set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Declare, record and check the schema-source frontier.")
    parser.add_argument("--frontier", type=Path, default=None, help="Frontier document (default: archive state).")
    parser.add_argument("--subject", action="append", default=[], help="Limit to these schema subjects.")
    parser.add_argument("--record", action="store_true", help="Re-record the baseline, hashing every member.")
    parser.add_argument(
        "--verify-content",
        action="store_true",
        help="Re-hash every admitted member instead of comparing byte counts only.",
    )
    parser.add_argument("--list", action="store_true", help="Print the declaration without checking live roots.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args(argv)

    from polylogue.core.schema_subjects import INFERENCE_EXCLUDED_SUBJECTS
    from polylogue.schemas.source_frontier import (
        SchemaFrontierError,
        check_frontier,
        load_frontier,
        record_frontier,
        write_frontier,
    )

    subjects = tuple(args.subject) or None

    try:
        frontier = load_frontier(args.frontier)
    except SchemaFrontierError as exc:
        print(f"schema-frontier: {exc}", file=sys.stderr)
        return 1

    # Declared non-applicability is code-owned, not document-owned: a subject
    # here has a zero denominator because it was never admissible, so it can
    # carry neither a declared root nor a recorded member. Reporting it beside
    # the declaration is what makes "zero eligible material" checkable instead
    # of merely absent.
    non_applicable = {
        token: reason
        for token, reason in sorted(INFERENCE_EXCLUDED_SUBJECTS.items())
        if subjects is None or token in subjects
    }

    def _print_non_applicable() -> None:
        for token, reason in non_applicable.items():
            print(f"  {token}: declared non-applicable, zero eligible material by declaration")
            print(f"      reason: {reason}")

    if args.list:
        payload = frontier.to_payload()
        payload["declared_non_applicable"] = dict(non_applicable)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"schema-frontier: declaration_digest={frontier.declaration_digest}")
            print(f"  baseline_digest={frontier.baseline_digest or 'not recorded'}")
            for subject in frontier.subjects:
                if subjects is not None and subject.subject not in subjects:
                    continue
                print(f"  {subject.subject}:")
                for root in subject.roots:
                    baseline = frontier.baseline_for(subject.subject, str(root.path))
                    membership = f"{baseline.member_count} members" if baseline is not None else "not recorded"
                    print(f"    {root.path} -- {root.scope} [{membership}]")
                    for exclusion in root.exclusions:
                        owner = f" (owner: {exclusion.owner})" if exclusion.owner else ""
                        print(f"      excluded {exclusion.pattern}: {exclusion.reason}{owner}")
                    if root.zero_material_reason is not None:
                        print(f"      zero eligible material: {root.zero_material_reason}")
            _print_non_applicable()
        return 0

    if args.record:
        try:
            recorded = record_frontier(frontier, subjects=subjects)
            path = write_frontier(recorded, args.frontier)
        except SchemaFrontierError as exc:
            print(f"schema-frontier: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(recorded.to_payload(), indent=2, sort_keys=True))
        else:
            print(f"schema-frontier: recorded {path}")
            print(f"  baseline_digest={recorded.baseline_digest}")
            for baseline in recorded.baselines:
                print(
                    f"  {baseline.subject} {baseline.root}: "
                    f"{baseline.member_count} members, {baseline.byte_count} bytes"
                )
        return 0

    try:
        check = check_frontier(frontier, verify_content=args.verify_content, subjects=subjects)
    except SchemaFrontierError as exc:
        print(f"schema-frontier: {exc}", file=sys.stderr)
        return 1

    if args.json:
        check_payload = check.to_payload()
        check_payload["declared_non_applicable"] = dict(non_applicable)
        print(json.dumps(check_payload, indent=2, sort_keys=True))
    else:
        state = "OK" if check.ok else "RED"
        print(f"schema-frontier: {state}")
        print(f"  baseline_digest={check.baseline_digest or 'not recorded'}")
        print(f"  checked_roots={check.checked_roots} checked_members={check.checked_members}")
        print(f"  content_verified={str(check.content_verified).lower()}")
        print(f"  errors={len(check.errors)} notices={len(check.notices)}")
        _print_non_applicable()
        for finding in check.notices:
            member = f" [{finding.member}]" if finding.member else ""
            print(f"  notice {finding.kind}: {finding.subject} {finding.root}{member} -- {finding.detail}")
        for finding in check.errors:
            member = f" [{finding.member}]" if finding.member else ""
            print(f"  {finding.kind}: {finding.subject} {finding.root}{member} -- {finding.detail}", file=sys.stderr)
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
