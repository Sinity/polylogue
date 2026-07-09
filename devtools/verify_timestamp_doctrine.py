"""Verify durable-tier DDL never stores a timestamp as TEXT.

Background
----------

Polylogue's time doctrine is UTC epoch-ms canon: every temporal column is an
INTEGER count of milliseconds since the epoch, never a TEXT string. A TEXT
timestamp re-introduces the exact ambiguity the doctrine exists to kill
(timezone-unknown values, lexicographic sort diverging from temporal sort).

This lint scans the durable tiers (``source.db``, ``user.db`` — the tiers
that require an explicit additive migration to change, so a doctrine
violation there is expensive to fix later) for any column whose name looks
like a timestamp (segments ``at``/``ms``/``time``/``date``, or containing
``timestamp``) but whose declared SQL type is not ``INTEGER``.

Derived tiers (``index.db``, ``embeddings.db``) are out of scope: they are
rebuilt from source on a schema bump, so a doctrine violation there is cheap
to fix and not the expensive-migration risk this lint exists to catch.

Wired into ``devtools verify --lab`` alongside the schema-versioning policy
check, not the fast default path, since this is an architectural boundary
check rather than a per-edit gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

_COLUMN_LINE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s+(TEXT|INTEGER|BLOB|REAL)\b")
_TIMESTAMP_SEGMENTS = {"at", "ms", "time", "date"}


@dataclass(frozen=True, slots=True)
class TimestampViolation:
    tier: str
    column: str
    declared_type: str


def _is_timestamp_like_name(name: str) -> bool:
    lowered = name.lower()
    if "timestamp" in lowered:
        return True
    return any(segment in _TIMESTAMP_SEGMENTS for segment in lowered.split("_"))


def scan_ddl_for_text_timestamps(ddl: str, *, tier: str) -> list[TimestampViolation]:
    """Return every column in *ddl* that looks like a timestamp but is TEXT.

    Exposed standalone (not just via the CLI) so a test can feed a synthetic
    DDL fixture directly, per the bead's verify clause.
    """
    violations: list[TimestampViolation] = []
    for line in ddl.splitlines():
        match = _COLUMN_LINE.match(line)
        if match is None:
            continue
        column, declared_type = match.group(1), match.group(2)
        if declared_type == "TEXT" and _is_timestamp_like_name(column):
            violations.append(TimestampViolation(tier=tier, column=column, declared_type=declared_type))
    return violations


def _collect_durable_tier_violations() -> list[TimestampViolation]:
    from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
    from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

    violations: list[TimestampViolation] = []
    violations.extend(scan_ddl_for_text_timestamps(SOURCE_DDL, tier="source"))
    violations.extend(scan_ddl_for_text_timestamps(USER_DDL, tier="user"))
    return violations


def _format_report(violations: list[TimestampViolation]) -> str:
    if not violations:
        return "Time doctrine intact: no TEXT-typed timestamp columns in durable-tier DDL."
    lines = [f"TEXT-typed timestamp columns found in durable tiers: {len(violations)}", ""]
    for violation in violations:
        lines.append(f"  {violation.tier}: {violation.column} {violation.declared_type}")
    lines.append("")
    lines.append(
        "Policy violation: durable-tier timestamps must be INTEGER epoch-ms, "
        "never TEXT (docs/internals.md time doctrine)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    violations = _collect_durable_tier_violations()

    if args.json:
        payload = {
            "violations": [{"tier": v.tier, "column": v.column, "declared_type": v.declared_type} for v in violations],
            "ok": not violations,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(violations))

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
