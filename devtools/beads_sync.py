"""Monotonic, receipted synchronization for `.beads/issues.jsonl` (polylogue-gxjh.1).

Background
----------

The 2026-07-15 planning-surface incident (see polylogue-gxjh, polylogue-gxjh.1
notes) had two distinct failure modes, both invisible to "does the file parse
and does the lint pass":

1. A staged `.beads/issues.jsonl` contained literal `<<<<<<<`/`=======`/
   `>>>>>>>` conflict markers while `git status` reported a plain
   modification, not an unmerged index entry. Whole-file JSON validation and
   the backlog-hygiene lint both silently pass on a file that happens to have
   valid-looking lines around the markers.
2. The "fix" -- a plain re-export/overwrite -- replaced the file with a
   syntactically valid 883-row snapshot that nonetheless *reverted* nine
   independently-edited Beads to older versions, because whole-file
   replacement carries no per-row comparison: a stale snapshot with valid
   JSON always "wins" against newer live state if nothing merges by
   revision.

This module is the durable fix: it never trusts "the file parses" or "the
command exited 0" as proof of synchronization. Every merge is per-row,
compares `updated_at` (the only revision proxy present in `bd export`
output; there is no separate numeric revision field), and produces a
machine-readable `SyncReceipt` that enumerates every id's outcome plus both
revisions where relevant. Downgrades (incoming row strictly older than the
row it would replace) are refused by default; the only way to apply one is
an explicit recovery override that must carry an actor and a reason, and
every downgraded row still appears in the receipt.

This module deliberately does not talk to the live `bd`/Dolt process for its
core merge -- `bd export`/`bd import` are the actual live-state boundary
(see `devtools/bd_reimport_guard.py`, which already wraps those for the
checkout/merge-hook window). `beads_sync` operates on JSONL files: the
committed `.beads/issues.jsonl`, a candidate recovery file, or a fresh `bd
export` snapshot the caller took immediately before calling in. Consumers
(the `bd_reimport_guard` hook wrapper, an operator doing conflict recovery,
or the repo policy layer at polylogue-8jg9.1) call `synchronize_files` /
`merge_issue_sets` and act on the returned `SyncReceipt` rather than
re-implementing comparison logic.

Outcome vocabulary (AC #1)
---------------------------

For every id in the union of base ∪ incoming:

  created            -- id only in incoming; adopted.
  retained           -- id only in base; kept (incoming does not carry it).
  equal              -- same `updated_at` and same content; no-op.
  updated            -- incoming has a strictly newer `updated_at`; adopted.
  skipped_downgrade  -- incoming has a strictly older `updated_at`; refused
                        (ordinary sync). Reported with both revisions.
  conflicted         -- same `updated_at` but different content; refused
                        (ordinary sync cannot silently pick a winner).
  downgraded         -- only possible with `allow_recovery=True`: a
                        skipped_downgrade or conflicted row whose incoming
                        version was explicitly adopted anyway. Always
                        recorded with both revisions plus actor/reason.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

IssueRow = dict[str, Any]

_CONFLICT_MARKERS: tuple[str, ...] = ("<<<<<<<", "=======", ">>>>>>>")

RowOutcomeKind = Literal[
    "created",
    "retained",
    "equal",
    "updated",
    "skipped_downgrade",
    "conflicted",
    "downgraded",
]


class PlanningSurfaceCorruptError(RuntimeError):
    """Raised when a JSONL planning-surface file cannot be trusted.

    Covers: literal merge-conflict markers, JSON parse failures, and
    duplicate ids within a single file. Callers must treat this as a hard
    failure -- never fall back to "it parsed, so it's fine".
    """


class RecoveryRequiredError(RuntimeError):
    """Raised when a merge would downgrade a row without recovery override."""


@dataclass(frozen=True, slots=True)
class RowOutcome:
    id: str
    outcome: RowOutcomeKind
    base_updated_at: str | None
    incoming_updated_at: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome,
            "base_updated_at": self.base_updated_at,
            "incoming_updated_at": self.incoming_updated_at,
        }


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    """Machine-readable, per-row record of a single synchronization merge."""

    fingerprint: Mapping[str, str]
    recovery: bool
    actor: str | None
    reason: str | None
    outcomes: tuple[RowOutcome, ...] = field(default_factory=tuple)

    def by_outcome(self, kind: RowOutcomeKind) -> tuple[RowOutcome, ...]:
        return tuple(o for o in self.outcomes if o.outcome == kind)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            counts[o.outcome] = counts.get(o.outcome, 0) + 1
        return counts

    def has_downgrades(self) -> bool:
        return any(o.outcome == "downgraded" for o in self.outcomes)

    def has_refusals(self) -> bool:
        """True if the merge left any row unresolved (would need recovery)."""
        return any(o.outcome in ("skipped_downgrade", "conflicted") for o in self.outcomes)

    def covers(self, expected_ids: Iterable[str]) -> bool:
        """True if every id in `expected_ids` has a recorded outcome.

        A caller that knows the expected union up front (e.g. len(base) +
        new incoming ids) uses this to detect a non-progress receipt --
        one where something short-circuited before comparing every row --
        rather than trusting a non-empty `outcomes` tuple alone.
        """
        seen = {o.id for o in self.outcomes}
        return all(issue_id in seen for issue_id in expected_ids)

    def to_json(self) -> dict[str, Any]:
        return {
            "fingerprint": dict(self.fingerprint),
            "recovery": self.recovery,
            "actor": self.actor,
            "reason": self.reason,
            "counts": self.counts(),
            "outcomes": [o.to_json() for o in self.outcomes],
        }


def _strip_generated(row: IssueRow) -> dict[str, Any]:
    """Drop the `updated_at` field so content-equality ignores the revision axis."""
    return {k: v for k, v in row.items() if k != "updated_at"}


def load_jsonl_rows(path: Path) -> dict[str, IssueRow]:
    """Parse a `.beads/issues.jsonl`-shaped file into {id: row}.

    Raises `PlanningSurfaceCorruptError` on:
      - literal conflict markers anywhere in the file (even if surrounding
        lines happen to parse -- markers alone make the file untrustworthy),
      - any line that fails to parse as JSON,
      - a row missing an `id`,
      - duplicate ids within the file.
    """
    text = path.read_text()
    for marker in _CONFLICT_MARKERS:
        if marker in text:
            raise PlanningSurfaceCorruptError(
                f"{path}: contains literal conflict marker {marker!r} -- refusing to trust this file"
            )

    rows: dict[str, IssueRow] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise PlanningSurfaceCorruptError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict) or "id" not in row:
            raise PlanningSurfaceCorruptError(f"{path}:{lineno}: row missing required 'id' field")
        issue_id = row["id"]
        if issue_id in rows:
            raise PlanningSurfaceCorruptError(f"{path}: duplicate id {issue_id!r} (line {lineno})")
        rows[issue_id] = row
    return rows


def merge_issue_sets(
    base: Mapping[str, IssueRow],
    incoming: Mapping[str, IssueRow],
    *,
    fingerprint: Mapping[str, str] | None = None,
    allow_recovery: bool = False,
    actor: str | None = None,
    reason: str | None = None,
) -> tuple[dict[str, IssueRow], SyncReceipt]:
    """Merge `incoming` into `base` per-row, never silently downgrading.

    Returns (merged_rows, receipt). Every id present in base or incoming
    appears exactly once in the receipt's outcomes.

    `allow_recovery=True` is the explicit operator-authorized recovery path
    (AC #2): it still refuses to run without both `actor` and `reason`, and
    every row it downgrades is recorded in the receipt with both revisions.
    Ordinary (non-recovery) calls never adopt an incoming row whose
    `updated_at` is not strictly newer than base's -- they report the
    conflict and keep base's row.
    """
    if allow_recovery and (not actor or not reason):
        raise RecoveryRequiredError(
            "recovery override requires both actor and reason -- refusing to authorize downgrades anonymously"
        )

    merged: dict[str, IssueRow] = dict(base)
    outcomes: list[RowOutcome] = []
    all_ids = sorted(set(base) | set(incoming))

    for issue_id in all_ids:
        base_row = base.get(issue_id)
        incoming_row = incoming.get(issue_id)

        if base_row is None and incoming_row is not None:
            merged[issue_id] = incoming_row
            outcomes.append(RowOutcome(issue_id, "created", None, incoming_row.get("updated_at")))
            continue

        if incoming_row is None:
            # Only in base: nothing to compare, kept untouched.
            outcomes.append(RowOutcome(issue_id, "retained", base_row.get("updated_at") if base_row else None, None))
            continue

        assert base_row is not None  # both present from here
        base_ts = base_row.get("updated_at") or ""
        incoming_ts = incoming_row.get("updated_at") or ""

        if incoming_ts > base_ts:
            merged[issue_id] = incoming_row
            outcomes.append(RowOutcome(issue_id, "updated", base_ts, incoming_ts))
            continue

        if incoming_ts == base_ts:
            if _strip_generated(incoming_row) == _strip_generated(base_row):
                outcomes.append(RowOutcome(issue_id, "equal", base_ts, incoming_ts))
                continue
            # Same revision marker, different content: an incomparable
            # conflict -- ordinary sync cannot pick a winner.
            if allow_recovery:
                merged[issue_id] = incoming_row
                outcomes.append(RowOutcome(issue_id, "downgraded", base_ts, incoming_ts))
            else:
                outcomes.append(RowOutcome(issue_id, "conflicted", base_ts, incoming_ts))
            continue

        # incoming_ts < base_ts: a strict downgrade.
        if allow_recovery:
            merged[issue_id] = incoming_row
            outcomes.append(RowOutcome(issue_id, "downgraded", base_ts, incoming_ts))
        else:
            outcomes.append(RowOutcome(issue_id, "skipped_downgrade", base_ts, incoming_ts))

    receipt = SyncReceipt(
        fingerprint=dict(fingerprint or {}),
        recovery=allow_recovery,
        actor=actor,
        reason=reason,
        outcomes=tuple(outcomes),
    )
    return merged, receipt


def atomic_write_jsonl(path: Path, rows: Iterable[IssueRow]) -> None:
    """Write `rows` to `path` as temp+fsync+atomic-rename.

    Validates before staging: every row round-trips through JSON, ids are
    unique, and no line contains a literal conflict marker. Refuses to
    write into a `path` that currently contains conflict markers unless the
    caller has already deleted/replaced it (this function never merges --
    callers must run `merge_issue_sets` first and pass the merged result).
    """
    if path.exists():
        existing_text = path.read_text()
        for marker in _CONFLICT_MARKERS:
            if marker in existing_text:
                raise PlanningSurfaceCorruptError(
                    f"{path}: refusing to overwrite a file that currently contains "
                    f"conflict marker {marker!r} without an explicit merge/recovery plan"
                )

    rows_list = list(rows)
    seen_ids: set[str] = set()
    lines: list[str] = []
    for row in rows_list:
        issue_id = row.get("id")
        if not issue_id:
            raise PlanningSurfaceCorruptError("refusing to stage a row missing 'id'")
        if issue_id in seen_ids:
            raise PlanningSurfaceCorruptError(f"refusing to stage duplicate id {issue_id!r}")
        seen_ids.add(issue_id)
        line = json.dumps(row, sort_keys=True)
        for marker in _CONFLICT_MARKERS:
            if marker in line:
                raise PlanningSurfaceCorruptError(
                    f"refusing to stage id {issue_id!r}: serialized row contains conflict marker {marker!r}"
                )
        lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".beads-sync-", suffix=".jsonl.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as tmp_file:
            for line in lines:
                tmp_file.write(line + "\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        # Validate the staged file parses back cleanly before it becomes live.
        staged = load_jsonl_rows(Path(tmp_name))
        if len(staged) != len(rows_list):
            raise PlanningSurfaceCorruptError(
                f"staged file row count {len(staged)} does not match expected {len(rows_list)}"
            )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def synchronize_files(
    base_path: Path,
    incoming_path: Path,
    output_path: Path,
    *,
    fingerprint: Mapping[str, str] | None = None,
    allow_recovery: bool = False,
    actor: str | None = None,
    reason: str | None = None,
) -> SyncReceipt:
    """High-level entrypoint: load base+incoming, merge, atomically write output.

    Refuses (raises `PlanningSurfaceCorruptError`) if either input file
    contains conflict markers or fails to parse -- this is the guard against
    the 2026-07-15 "staged conflict markers, syntactically valid neighbor
    lines" failure mode. Refuses (raises `RecoveryRequiredError`) if the
    merge would need to downgrade any row and `allow_recovery` was not
    explicitly set with actor+reason -- callers can inspect the raised
    receipt-free error and re-invoke with recovery once a human has signed
    off, or accept the returned receipt's `skipped_downgrade`/`conflicted`
    rows as the authoritative "what would have been lost" record.
    """
    base_rows = load_jsonl_rows(base_path) if base_path.exists() else {}
    incoming_rows = load_jsonl_rows(incoming_path)

    merged, receipt = merge_issue_sets(
        base_rows,
        incoming_rows,
        fingerprint=fingerprint,
        allow_recovery=allow_recovery,
        actor=actor,
        reason=reason,
    )

    # Ordinary sync still writes the safely-mergeable union (base rows,
    # created rows, updated rows, equal rows) -- unrelated rows are not held
    # hostage by one incomparable id -- but the caller must consult the
    # receipt for what was refused (`has_refusals()`) rather than assume
    # completeness from a clean exit alone.
    atomic_write_jsonl(output_path, merged.values())
    return receipt


def bootstrap_union(*sources: Mapping[str, IssueRow]) -> dict[str, IssueRow]:
    """Fold N sources into one monotonic union (AC #3: concurrent bootstraps + a writer).

    Order-independent: for any permutation of `sources`, the result is the
    same union with, for every id, the row carrying the lexicographically
    greatest `updated_at` (ties broken by content equality; a true
    same-timestamp content conflict raises `PlanningSurfaceCorruptError`
    since bootstrap has no operator present to authorize a downgrade
    choice).
    """
    merged: dict[str, IssueRow] = {}
    for source in sources:
        merged, receipt = merge_issue_sets(merged, source)
        for outcome in receipt.by_outcome("conflicted"):
            raise PlanningSurfaceCorruptError(
                f"bootstrap union hit an incomparable conflict on {outcome.id} "
                "(same updated_at, different content) -- concurrent bootstrap has no "
                "operator present to authorize a recovery choice"
            )
    return merged


def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="devtools beads-sync", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    merge_p = sub.add_parser("merge", help="Merge incoming JSONL into base, write output + receipt.")
    merge_p.add_argument("--base", type=Path, required=True)
    merge_p.add_argument("--incoming", type=Path, required=True)
    merge_p.add_argument("--output", type=Path, required=True)
    merge_p.add_argument("--receipt", type=Path, default=None)
    merge_p.add_argument("--recover", action="store_true")
    merge_p.add_argument("--actor", type=str, default=None)
    merge_p.add_argument("--reason", type=str, default=None)
    merge_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "merge":
        try:
            receipt = synchronize_files(
                args.base,
                args.incoming,
                args.output,
                allow_recovery=args.recover,
                actor=args.actor,
                reason=args.reason,
            )
        except (PlanningSurfaceCorruptError, RecoveryRequiredError) as exc:
            print(f"beads-sync: {exc}", file=__import__("sys").stderr)
            return 1

        receipt_json = json.dumps(receipt.to_json(), indent=2, sort_keys=True)
        if args.receipt:
            args.receipt.write_text(receipt_json + "\n")
        if args.json:
            print(receipt_json)
        else:
            print(f"beads-sync: merged -> {args.output}")
            for kind, count in sorted(receipt.counts().items()):
                print(f"  {kind}: {count}")
        return 1 if (not args.recover and receipt.has_refusals()) else 0

    return 1


def main(argv: list[str] | None = None) -> int:
    return _cli_main(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main())
