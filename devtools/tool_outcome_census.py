"""Archive-wide census of tool-result outcomes and their unknown reasons.

Usage:
  devtools archive tool-outcome-census [--json] [--archive-root PATH]

Classifies every ``tool_result`` block by origin, construct, structural
outcome and unknown reason, and counts the four shapes the outcome contract
forbids. Reads only structured index-tier columns -- never result text, never
paths -- so the report is safe to persist and share.

``construct`` is the paired invocation's ``tool_name``: the construct family a
result belongs to is the tool that produced it. Results the ``actions``
projection could not pair carry the ``<unpaired>`` construct, which is a
census fact, not a defect (an interrupted invocation is a real recorded
state).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.config import get_config
from polylogue.core.enums import Origin, ToolOutcome
from polylogue.sources.origin_specs import tool_outcome_unknown_reasons_for_origin
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

__all__ = [
    "UNPAIRED_CONSTRUCT",
    "ToolOutcomeCensus",
    "compute_tool_outcome_census",
    "main",
    "render_census",
]

UNPAIRED_CONSTRUCT = "<unpaired>"

_KNOWN_OUTCOMES = (ToolOutcome.OK.value, ToolOutcome.ERROR.value, ToolOutcome.NO_RESULT.value)

#: ``actions.result_state`` a result block's own ``tool_outcome`` implies.
_EXPECTED_RESULT_STATE = {
    ToolOutcome.OK.value: "outcome_success",
    ToolOutcome.ERROR.value: "outcome_error",
    ToolOutcome.UNKNOWN.value: "outcome_unknown",
    ToolOutcome.NO_RESULT.value: "no_result",
}


@dataclass(frozen=True, slots=True)
class ToolOutcomeCensus:
    """Structural counts for one census pass."""

    total_tool_results: int
    #: ``(origin, construct, outcome, reason)`` -> count. ``reason`` is ``""``
    #: when the outcome is known.
    by_classification: dict[tuple[str, str, str, str], int] = field(default_factory=dict)
    unknown_without_reason: int = 0
    known_with_reason: int = 0
    unsupported_without_owner: int = 0
    public_projection_disagreement: int = 0

    @property
    def defect_counts(self) -> dict[str, int]:
        return {
            "unknown_without_reason": self.unknown_without_reason,
            "known_with_reason": self.known_with_reason,
            "unsupported_without_owner": self.unsupported_without_owner,
            "public_projection_disagreement": self.public_projection_disagreement,
        }

    @property
    def is_clean(self) -> bool:
        return not any(self.defect_counts.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "total_tool_results": self.total_tool_results,
            "by_classification": [
                {"origin": origin, "construct": construct, "outcome": outcome, "reason": reason or None, "count": n}
                for (origin, construct, outcome, reason), n in sorted(self.by_classification.items())
            ],
            **self.defect_counts,
            "clean": self.is_clean,
        }


def compute_tool_outcome_census(conn: sqlite3.Connection) -> ToolOutcomeCensus:
    """Compute the census from an ``index.db`` connection (read-only safe)."""
    rows = conn.execute(
        """
        SELECT s.origin AS origin,
               COALESCE(a.tool_name, ?) AS construct,
               b.tool_outcome AS outcome,
               COALESCE(b.tool_result_outcome_unknown_reason, '') AS reason,
               a.result_state AS result_state,
               COUNT(*) AS n
        FROM blocks b
        JOIN sessions s ON s.session_id = b.session_id
        LEFT JOIN actions a ON a.tool_result_block_id = b.block_id
        WHERE b.block_type = 'tool_result'
        GROUP BY origin, construct, outcome, reason, result_state
        """,
        (UNPAIRED_CONSTRUCT,),
    ).fetchall()

    by_classification: dict[tuple[str, str, str, str], int] = {}
    total = 0
    unknown_without_reason = 0
    known_with_reason = 0
    unsupported_without_owner = 0
    disagreement = 0
    declared_by_origin = _declared_reasons_by_origin()

    for origin, construct, outcome, reason, result_state, count in rows:
        total += count
        key = (str(origin), str(construct), str(outcome or ""), str(reason))
        by_classification[key] = by_classification.get(key, 0) + count
        # A NULL outcome is the same hole as an unreasoned unknown: the row
        # states nothing and attributes nothing. The writer cannot produce it,
        # so any row carrying one arrived past the production route.
        if (outcome == ToolOutcome.UNKNOWN.value or not outcome) and not reason:
            unknown_without_reason += count
        if outcome in _KNOWN_OUTCOMES and reason:
            known_with_reason += count
        if reason and reason not in declared_by_origin.get(str(origin), frozenset()):
            unsupported_without_owner += count
        if result_state is not None and result_state != _EXPECTED_RESULT_STATE.get(str(outcome or "")):
            disagreement += count

    return ToolOutcomeCensus(
        total_tool_results=total,
        by_classification=by_classification,
        unknown_without_reason=unknown_without_reason,
        known_with_reason=known_with_reason,
        unsupported_without_owner=unsupported_without_owner,
        public_projection_disagreement=disagreement,
    )


def _declared_reasons_by_origin() -> Mapping[str, frozenset[str]]:
    return {
        origin.value: frozenset(reason.value for reason in tool_outcome_unknown_reasons_for_origin(origin))
        for origin in Origin
    }


def render_census(census: ToolOutcomeCensus) -> Sequence[str]:
    """Render the census as plaintext lines."""
    lines = [f"tool results: {census.total_tool_results}"]
    for (origin, construct, outcome, reason), count in sorted(census.by_classification.items()):
        lines.append(f"  {origin} | {construct} | {outcome} | {reason or '-'} | {count}")
    for name, count in census.defect_counts.items():
        lines.append(f"{name}: {count}")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools archive tool-outcome-census",
        description="Classify every archived tool result by origin, construct, outcome and unknown reason.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--index-db", type=Path, default=None, help="Read a specific index.db.")
    parser.add_argument("--json", action="store_true", help="Emit the census as JSON on stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.index_db is not None:
        index_db = args.index_db.expanduser().resolve()
    elif args.archive_root is not None:
        index_db = args.archive_root.expanduser().resolve() / "index.db"
    else:
        index_db = get_config().db_path
    if not index_db.exists():
        print(f"tool-outcome-census: no index.db found at {index_db}", file=sys.stderr)
        return 1
    conn = open_readonly_connection(index_db)
    try:
        census = compute_tool_outcome_census(conn)
    finally:
        conn.close()
    if args.json:
        sys.stdout.write(json.dumps(census.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        for line in render_census(census):
            print(line)
    return 0 if census.is_clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
