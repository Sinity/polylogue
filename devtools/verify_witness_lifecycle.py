"""Verify committed witness lifecycle health — staleness, unexercised, stale xfails.

Usage:
  devtools verify-witness-lifecycle
  devtools verify-witness-lifecycle --json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from polylogue.core.json import dumps
from polylogue.proof.witnesses import COMMITTED_WITNESS_DIR, load_committed_witnesses

_STALE_DAYS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check witness lifecycle health.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def _is_stale(last_exercised: str | None, now: datetime) -> bool:
    if last_exercised is None:
        return True
    try:
        dt = datetime.fromisoformat(last_exercised)
        return (now - dt).days > _STALE_DAYS
    except (ValueError, TypeError):
        return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    now = datetime.now(tz=timezone.utc)
    witnesses = load_committed_witnesses(COMMITTED_WITNESS_DIR)

    issues: list[dict[str, object]] = []
    ok_count = 0

    for w in witnesses:
        validation_errors = w.validation_errors()
        if validation_errors:
            issues.append(
                {
                    "witness_id": w.witness_id,
                    "kind": "validation_error",
                    "details": list(validation_errors),
                }
            )
            continue

        stale = False
        unexercised = False
        if w.lifecycle is not None:
            if w.lifecycle.state == "committed" and w.lifecycle.last_exercised_at is None:
                unexercised = True
            elif w.lifecycle.last_exercised_at and _is_stale(w.lifecycle.last_exercised_at, now):
                stale = True

        if unexercised:
            issues.append(
                {
                    "witness_id": w.witness_id,
                    "kind": "unexercised",
                    "state": w.lifecycle.state if w.lifecycle else "unknown",
                }
            )
        elif stale:
            days = "unknown"
            if w.lifecycle and w.lifecycle.last_exercised_at:
                try:
                    dt = datetime.fromisoformat(w.lifecycle.last_exercised_at)
                    days = str((now - dt).days)
                except (ValueError, TypeError):
                    pass
            issues.append(
                {
                    "witness_id": w.witness_id,
                    "kind": "stale",
                    "last_exercised_at": w.lifecycle.last_exercised_at if w.lifecycle else None,
                    "days_since_exercise": days,
                }
            )
        else:
            ok_count += 1

    if args.json:
        print(dumps({"witnesses": len(witnesses), "ok": ok_count, "issues": issues}))
    else:
        for issue in issues:
            wid = issue["witness_id"]
            kind = issue["kind"]
            if kind == "validation_error":
                print(f"  ✗ {wid}  VALIDATION: {issue['details']}")
            elif kind == "unexercised":
                print(f"  ✗ {wid}  UNEXERCISED (never run)")
            elif kind == "stale":
                print(f"  ⚠ {wid}  STALE (last exercised {issue.get('days_since_exercise', '?')}d ago)")
        if ok_count:
            print(f"  ✓ {ok_count} witnesses OK")

    if issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
