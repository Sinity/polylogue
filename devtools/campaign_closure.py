"""Campaign work selection computed from the blocks-only dependency closure.

Campaign membership is defined by ``blocks`` edges, not by labels, not by
``metadata.campaign_id``, and not by whatever ``bd ready`` chooses to show.
Those are descriptive; a selection drawn from any of them omits most of the
prerequisite graph, so a "no actionable work remains" conclusion drawn from one
is unsound (polylogue-eug7l).

This module is campaign machinery, not a product gate: it is deliberately not
registered in :mod:`devtools.command_catalog` and not part of any gate set.

Input is ``bd export`` JSONL. Each exported issue carries its own edges under
``dependencies`` as ``{"issue_id": X, "depends_on_id": Y, "type": T}``, read as
*X depends on Y*. Only ``T == "blocks"`` participates here; ``relates-to`` and
``discovered-from`` are explicitly excluded, because mixing them is how a
traversal silently acquires members whose prerequisites it then cannot reason
about.

A campaign root is the record that cannot be finished until its campaign is:
``polylogue-reindex-2026`` carries 29 ``blocks`` prerequisites and
``polylogue-reindex-2026.1`` carries 83. The closure from a root therefore
descends through ``depends_on_id`` -- the members are the root's prerequisites,
transitively. A member is *unblocked* when every one of its own
blocks-prerequisites is closed.

Direction matters and is easy to invert: ascending to dependents instead of
descending to prerequisites returns 59 records against the same export where the
correct descent returns 552.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The disposition vocabulary. Exactly these six tokens; anything else on a
#: record is not a disposition and the record still counts as undispositioned.
DISPOSITIONS: frozenset[str] = frozenset(
    {
        "implementation-residual",
        "specification-required",
        "bounded-evidence",
        "final-build-evidence",
        "already-satisfied",
        "scope-adjudicated",
    }
)

#: The edge type that defines campaign membership. Nothing else does.
BLOCKS = "blocks"

#: Statuses that mean the record no longer holds work.
CLOSED_STATUSES: frozenset[str] = frozenset({"closed", "done", "completed"})


def load_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse ``bd export`` JSONL into issue records, ignoring blank lines."""

    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"export line is not an object: {line[:80]!r}")
        if record.get("_type", "issue") != "issue":
            continue
        records.append(record)
    return records


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """``metadata`` as a mapping; it is sometimes exported as a JSON string."""

    raw = record.get("metadata")
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def is_closed(record: Mapping[str, Any]) -> bool:
    status = record.get("status")
    return isinstance(status, str) and status.lower() in CLOSED_STATUSES


def disposition_of(record: Mapping[str, Any]) -> str | None:
    """The record's disposition token, or None when it carries no valid one.

    An out-of-vocabulary value is *not* a disposition. Accepting it would let a
    typo silently discharge criterion 1.
    """

    value = _metadata(record).get("disposition")
    if isinstance(value, str) and value in DISPOSITIONS:
        return value
    return None


def blocks_prerequisites(record: Mapping[str, Any]) -> list[str]:
    """Ids this record depends on via ``blocks`` edges only."""

    prerequisites: list[str] = []
    edges = record.get("dependencies")
    if not isinstance(edges, list):
        return prerequisites
    own_id = record.get("id")
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        if edge.get("type") != BLOCKS:
            continue
        # An export can carry an edge on either endpoint's record; only the
        # one whose issue_id is this record states *this* record's prerequisite.
        if edge.get("issue_id") != own_id:
            continue
        target = edge.get("depends_on_id")
        if isinstance(target, str) and target:
            prerequisites.append(target)
    return prerequisites


def _prerequisite_index(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """id -> the ids it depends on via ``blocks``."""

    index: dict[str, list[str]] = {}
    for record in records:
        own_id = record.get("id")
        if not isinstance(own_id, str):
            continue
        index[own_id] = blocks_prerequisites(record)
    return index


def closure_ids(records: Sequence[Mapping[str, Any]], roots: Sequence[str]) -> set[str]:
    """Ids any root transitively depends on through ``blocks`` edges.

    The roots themselves are included. Cycles terminate: an id already seen is
    not re-expanded.
    """

    index = _prerequisite_index(records)
    seen: set[str] = set()
    frontier = list(roots)
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(index.get(current, ()))
    return seen


@dataclass(frozen=True)
class Member:
    """One nonclosed closure member and why it is or is not selectable."""

    id: str
    title: str
    priority: int | None
    unblocked: bool
    disposition: str | None
    open_prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class ClosureReport:
    roots: tuple[str, ...]
    closure_size: int
    nonclosed: tuple[Member, ...] = field(default=())
    #: Ids named as prerequisites or roots that the export does not contain.
    missing: tuple[str, ...] = field(default=())

    @property
    def unblocked(self) -> tuple[Member, ...]:
        return tuple(member for member in self.nonclosed if member.unblocked)

    @property
    def violations(self) -> tuple[Member, ...]:
        """Unblocked members carrying no disposition -- criterion 1's count."""

        return tuple(member for member in self.unblocked if member.disposition is None)

    def to_json(self) -> dict[str, Any]:
        return {
            "roots": list(self.roots),
            "closure_size": self.closure_size,
            "nonclosed_count": len(self.nonclosed),
            "unblocked_count": len(self.unblocked),
            "violation_count": len(self.violations),
            "missing": list(self.missing),
            "members": [
                {
                    "id": member.id,
                    "title": member.title,
                    "priority": member.priority,
                    "unblocked": member.unblocked,
                    "disposition": member.disposition,
                    "open_prerequisites": list(member.open_prerequisites),
                }
                for member in self.nonclosed
            ],
        }


def build_report(records: Sequence[Mapping[str, Any]], roots: Sequence[str]) -> ClosureReport:
    """Compute the blocks-only closure report.

    No label, ``campaign_id``, ``execution_shape`` or readiness filter is
    applied anywhere in this function. A record missing all of that metadata is
    still a member and still reported -- that absence is what the filtered views
    were dropping.
    """

    by_id = {record["id"]: record for record in records if isinstance(record.get("id"), str)}
    member_ids = closure_ids(records, roots)

    named: set[str] = set(member_ids)
    members: list[Member] = []
    for member_id in sorted(member_ids):
        record = by_id.get(member_id)
        if record is None or is_closed(record):
            continue
        named.update(blocks_prerequisites(record))
        open_prerequisites = tuple(
            sorted(
                prerequisite
                for prerequisite in blocks_prerequisites(record)
                # An id absent from the export cannot be shown to be closed, so
                # it is treated as open rather than quietly ignored.
                if prerequisite not in by_id or not is_closed(by_id[prerequisite])
            )
        )
        priority = record.get("priority")
        members.append(
            Member(
                id=member_id,
                title=str(record.get("title", "")),
                priority=priority if isinstance(priority, int) else None,
                unblocked=not open_prerequisites,
                disposition=disposition_of(record),
                open_prerequisites=open_prerequisites,
            )
        )

    return ClosureReport(
        roots=tuple(roots),
        closure_size=len(member_ids),
        nonclosed=tuple(members),
        missing=tuple(sorted(name for name in named if name not in by_id)),
    )


def render_text(report: ClosureReport) -> str:
    lines = [
        f"roots                 {', '.join(report.roots)}",
        f"closure records       {report.closure_size}",
        f"nonclosed             {len(report.nonclosed)}",
        f"unblocked             {len(report.unblocked)}",
        f"undispositioned       {len(report.violations)}",
    ]
    if report.missing:
        lines.append(f"absent from export    {len(report.missing)}")
    if report.violations:
        lines.append("")
        lines.append("unblocked records carrying no disposition:")
        for member in report.violations:
            priority = "P?" if member.priority is None else f"P{member.priority}"
            lines.append(f"  {priority} {member.id}  {member.title[:72]}")
    return "\n".join(lines)


def _read_lines(source: str) -> Iterator[str]:
    if source == "-":
        yield from sys.stdin
        return
    with Path(source).open(encoding="utf-8") as handle:
        yield from handle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="campaign-closure",
        description=("Report the blocks-only campaign closure and the unblocked records that carry no disposition."),
    )
    parser.add_argument(
        "--export",
        default="-",
        help="bd export JSONL path, or '-' for stdin (default).",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        required=True,
        help="Closure root id; repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    report = build_report(load_records(_read_lines(args.export)), args.roots)
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(render_text(report))
    # Nonzero when the population still holds undispositioned actionable work,
    # so the check is an assertion rather than a printout.
    return 1 if report.violations else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
