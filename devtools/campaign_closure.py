"""Blocks-only campaign closure over an exported Beads graph.

Usage:
  python -m devtools.campaign_closure [--export PATH|-] [--root ID ...]
                                      [--ready-ids PATH] [--backfill-plan]
                                      [--json] [--summary] [--check]

This is campaign machinery, not a product gate: it is deliberately not
registered in ``devtools/command_catalog.py`` and not in any gate set
(polylogue-eug7l design: "do NOT add it to devtools/command_catalog.py or the
quick gate set unless the operator asks"). It writes nothing --
``--backfill-plan`` prints the ``bd update`` lines a backfill would run.

Campaign membership is defined by ``blocks`` edges, not by a label, not by
``metadata.campaign_id`` and not by whatever ``bd ready`` happens to surface.
Those are descriptive views: each of them omits part of the real prerequisite
graph, so a "no actionable work remains" claim resting on one of them is
unsound. This module computes the population the claim actually needs -- the
transitive ``blocks`` closure of a declared root -- and prints enough
arithmetic that a second reader can tell whether the answer is *complete*
rather than merely plausible.

Three properties are load-bearing, and each has a named test that goes red
under its own mutation (``tests/unit/devtools/test_campaign_closure.py``):

*Only ``blocks`` edges are followed.* A ``bd export`` record carries every
relation type in one ``dependencies`` list -- ``blocks`` alongside
``relates-to``, ``discovered-from``, ``parent-child``, ``supersedes`` and
more. Treating any of the others as a prerequisite manufactures phantom
blockers (a real incident: four "deadlock cycles" across 29 beads that did not
exist, because ``relates-to`` edges were counted as blocking). The report
names every edge type it saw and marks each FOLLOWED or IGNORED, so the filter
is visible in the output instead of being an invisible assumption.

*Direction.* In the export an edge is ``{"issue_id": X, "depends_on_id": Y,
"type": "blocks"}`` and means **Y blocks X** -- Y is a prerequisite of X, and
``bd dep list X`` lists Y. The closure therefore walks *from* a root *to* its
prerequisites. Reversing it silently produces the dependents instead, which
looks equally well-formed and is entirely wrong.

*Closed exclusion, with unknown kept separate.* Only ``closed`` is a terminal
status in this tracker; ``open``, ``in_progress``, ``blocked``, ``deferred``,
``pinned`` and ``hooked`` are all live. A member is *unblocked* only when
every one of its ``blocks`` prerequisites is known-closed. A prerequisite id
that is absent from the export is neither closed nor open -- it gets its own
term (``blocked_unknown``) instead of being folded into the nearest answer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CLOSED_STATUS",
    "DEFAULT_ROOTS",
    "DESCRIPTIVE_CAMPAIGN_LABEL",
    "DISPOSITION_VOCABULARY",
    "SELECTION_EDGE_TYPE",
    "BackfillProposal",
    "CampaignGraph",
    "ClosureReport",
    "ExportRefusedError",
    "MemberClassification",
    "ReadyComparison",
    "Record",
    "compute_closure",
    "load_graph",
    "main",
    "propose_dispositions",
    "render_backfill_plan",
    "render_report",
]

#: The one relation type that defines campaign membership and blocking.
SELECTION_EDGE_TYPE = "blocks"

#: The only terminal status in this tracker (``bd statuses``: category "done").
CLOSED_STATUS = "closed"

#: Roots of the reindex campaign. ``.1`` is itself inside the parent's
#: closure; it is listed so a caller who passes only the sub-root still gets a
#: well-defined population.
DEFAULT_ROOTS: tuple[str, ...] = ("polylogue-reindex-2026",)

#: Exhaustive disposition vocabulary (polylogue-eug7l acceptance criterion 1).
DISPOSITION_VOCABULARY: tuple[str, ...] = (
    "implementation-residual",
    "specification-required",
    "bounded-evidence",
    "final-build-evidence",
    "already-satisfied",
    "scope-adjudicated",
)

#: Descriptive only. Reported as a counter-example, never used as a filter.
DESCRIPTIVE_CAMPAIGN_LABEL = "campaign:reindex-2026"


class ExportRefusedError(Exception):
    """The export could not support a closure claim; refuse rather than guess."""


@dataclass(frozen=True, slots=True)
class Record:
    """One exported issue, reduced to the fields selection depends on."""

    id: str
    status: str
    priority: int | None
    title: str
    labels: frozenset[str]
    metadata: Mapping[str, str]
    has_acceptance_text: bool

    @property
    def is_closed(self) -> bool:
        return self.status == CLOSED_STATUS

    @property
    def disposition(self) -> str | None:
        """The top-level ``metadata.disposition`` selection token.

        Deliberately *not* ``acceptance_contract_v1.closure.disposition``: that
        nested field exists on many records and is a different concept. A
        2026-09-20 note read the nested field's occurrences as evidence that
        this one was populated; it was not.
        """

        value = self.metadata.get("disposition")
        return value if isinstance(value, str) and value else None

    @property
    def disposition_is_valid(self) -> bool:
        return self.disposition in DISPOSITION_VOCABULARY


@dataclass(frozen=True, slots=True)
class CampaignGraph:
    """Records plus the ``blocks``-only prerequisite index built from them."""

    records: Mapping[str, Record]
    #: ``id`` -> ids that block it (its prerequisites). Blocks edges only.
    prerequisites: Mapping[str, frozenset[str]]
    #: Every relation type seen in the export, with its edge count.
    edge_type_counts: Mapping[str, int]

    @property
    def ignored_edge_types(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self.edge_type_counts if name != SELECTION_EDGE_TYPE))

    def prerequisites_of(self, identifier: str) -> frozenset[str]:
        return self.prerequisites.get(identifier, frozenset())


@dataclass(frozen=True, slots=True)
class MemberClassification:
    """Why one nonclosed closure member is or is not selectable."""

    id: str
    status: str
    priority: int | None
    title: str
    state: str
    disposition: str | None
    disposition_state: str
    open_prerequisites: tuple[str, ...]
    unknown_prerequisites: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "priority": self.priority,
            "title": self.title,
            "state": self.state,
            "disposition": self.disposition,
            "disposition_state": self.disposition_state,
            "open_prerequisites": list(self.open_prerequisites),
            "unknown_prerequisites": list(self.unknown_prerequisites),
        }


@dataclass(frozen=True, slots=True)
class ReadyComparison:
    """How a ``bd ready`` view relates to the closure it is used to stand for."""

    ready_ids: tuple[str, ...]
    ready_inside_closure: tuple[str, ...]
    ready_outside_closure: tuple[str, ...]
    unblocked_absent_from_ready: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready_rows": len(self.ready_ids),
            "ready_inside_closure": list(self.ready_inside_closure),
            "ready_outside_closure": list(self.ready_outside_closure),
            "unblocked_absent_from_ready": list(self.unblocked_absent_from_ready),
        }


@dataclass(frozen=True, slots=True)
class ClosureReport:
    """One observation of the blocks-only closure."""

    roots: tuple[str, ...]
    export_records: int
    edge_type_counts: Mapping[str, int]
    ignored_edge_types: tuple[str, ...]
    members: tuple[str, ...]
    dangling_references: tuple[str, ...]
    closed_members: tuple[str, ...]
    classifications: tuple[MemberClassification, ...]
    label_carriers: tuple[str, ...]
    ready: ReadyComparison | None = None

    @property
    def nonclosed(self) -> tuple[MemberClassification, ...]:
        return self.classifications

    def _by_state(self, state: str) -> tuple[MemberClassification, ...]:
        return tuple(entry for entry in self.classifications if entry.state == state)

    @property
    def unblocked(self) -> tuple[MemberClassification, ...]:
        return self._by_state("unblocked")

    @property
    def blocked_open(self) -> tuple[MemberClassification, ...]:
        return self._by_state("blocked_open")

    @property
    def blocked_unknown(self) -> tuple[MemberClassification, ...]:
        return self._by_state("blocked_unknown")

    @property
    def violations(self) -> tuple[MemberClassification, ...]:
        """Unblocked members whose disposition is missing or off-vocabulary."""

        return tuple(entry for entry in self.unblocked if entry.disposition_state != "valid")

    @property
    def disposition_counts(self) -> dict[str, int]:
        counts = Counter(entry.disposition or "<absent>" for entry in self.unblocked)
        return dict(sorted(counts.items()))

    @property
    def resolved_members(self) -> int:
        """Closure members that resolve to an exported record."""

        return len(self.members) - len(self.dangling_references)

    @property
    def partition_holds(self) -> bool:
        """The three nonclosed states partition the nonclosed population."""

        return len(self.unblocked) + len(self.blocked_open) + len(self.blocked_unknown) == len(self.classifications)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "followed_edge_type": SELECTION_EDGE_TYPE,
            "ignored_edge_types": list(self.ignored_edge_types),
            "edge_type_counts": dict(sorted(self.edge_type_counts.items())),
            "roots": list(self.roots),
            "export_records": self.export_records,
            "closure_members": len(self.members),
            "closed_members": len(self.closed_members),
            "nonclosed_members": len(self.classifications),
            "unblocked": len(self.unblocked),
            "blocked_open": len(self.blocked_open),
            "blocked_unknown": len(self.blocked_unknown),
            "partition_holds": self.partition_holds,
            "dangling_references": list(self.dangling_references),
            "disposition_counts": self.disposition_counts,
            "violations": [entry.to_dict() for entry in self.violations],
            "members": [entry.to_dict() for entry in self.classifications],
            "label_filter_counterexample": {
                "label": DESCRIPTIVE_CAMPAIGN_LABEL,
                "resolved_members": self.resolved_members,
                "members_carrying_label": len(self.label_carriers),
                "members_lacking_label": self.resolved_members - len(self.label_carriers),
            },
        }
        if self.ready is not None:
            payload["ready_comparison"] = self.ready.to_dict()
        return payload


def _as_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str)}


def _as_labels(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))


def _record_from_payload(payload: Mapping[str, object]) -> Record:
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ExportRefusedError(f"export record without a usable id: {payload!r:.120}")
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ExportRefusedError(f"{identifier}: export record without a status; status decides closedness")
    priority = payload.get("priority")
    acceptance = payload.get("acceptance_criteria")
    title = payload.get("title")
    return Record(
        id=identifier,
        status=status,
        priority=priority if isinstance(priority, int) else None,
        title=title if isinstance(title, str) else "",
        labels=_as_labels(payload.get("labels")),
        metadata=_as_mapping(payload.get("metadata")),
        has_acceptance_text=isinstance(acceptance, str) and bool(acceptance.strip()),
    )


def _iter_payloads(lines: Iterable[str]) -> Iterator[Mapping[str, object]]:
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ExportRefusedError(f"export line {number} is not JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ExportRefusedError(f"export line {number} is not a JSON object")
        if payload.get("_type", "issue") != "issue":
            continue
        yield payload


def load_graph(lines: Iterable[str]) -> CampaignGraph:
    """Build the blocks-only prerequisite graph from ``bd export`` JSONL."""

    records: dict[str, Record] = {}
    prerequisites: dict[str, set[str]] = {}
    edge_type_counts: Counter[str] = Counter()
    for payload in _iter_payloads(lines):
        record = _record_from_payload(payload)
        records[record.id] = record
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, list):
            continue
        for edge in dependencies:
            if not isinstance(edge, dict):
                continue
            edge_type = edge.get("type")
            dependent = edge.get("issue_id")
            prerequisite = edge.get("depends_on_id")
            if not isinstance(edge_type, str) or not isinstance(dependent, str) or not isinstance(prerequisite, str):
                continue
            edge_type_counts[edge_type] += 1
            if edge_type != SELECTION_EDGE_TYPE:
                continue
            # Direction: `depends_on_id` BLOCKS `issue_id`. The prerequisite of
            # the dependent is the thing it depends on -- never the reverse.
            prerequisites.setdefault(dependent, set()).add(prerequisite)
    if not records:
        raise ExportRefusedError("export contained no issue records")
    return CampaignGraph(
        records=records,
        prerequisites={key: frozenset(value) for key, value in prerequisites.items()},
        edge_type_counts=dict(edge_type_counts),
    )


def compute_closure(
    graph: CampaignGraph,
    roots: Sequence[str] = DEFAULT_ROOTS,
    *,
    ready_ids: Sequence[str] | None = None,
) -> ClosureReport:
    """Walk ``blocks`` edges from ``roots`` to their transitive prerequisites."""

    if not roots:
        raise ExportRefusedError("no root given; the closure is undefined without one")
    missing_roots = [root for root in roots if root not in graph.records]
    if missing_roots:
        raise ExportRefusedError(f"root(s) absent from the export: {', '.join(sorted(missing_roots))}")

    members: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in members:
            continue
        members.add(current)
        stack.extend(graph.prerequisites_of(current))

    dangling = sorted(identifier for identifier in members if identifier not in graph.records)
    known = sorted(identifier for identifier in members if identifier in graph.records)

    closed: list[str] = []
    classifications: list[MemberClassification] = []
    for identifier in known:
        record = graph.records[identifier]
        if record.is_closed:
            closed.append(identifier)
            continue
        open_prerequisites: list[str] = []
        unknown_prerequisites: list[str] = []
        for prerequisite in sorted(graph.prerequisites_of(identifier)):
            blocker = graph.records.get(prerequisite)
            if blocker is None:
                unknown_prerequisites.append(prerequisite)
            elif not blocker.is_closed:
                open_prerequisites.append(prerequisite)
        if unknown_prerequisites:
            state = "blocked_unknown"
        elif open_prerequisites:
            state = "blocked_open"
        else:
            state = "unblocked"
        if record.disposition is None:
            disposition_state = "absent"
        elif record.disposition_is_valid:
            disposition_state = "valid"
        else:
            disposition_state = "off-vocabulary"
        classifications.append(
            MemberClassification(
                id=identifier,
                status=record.status,
                priority=record.priority,
                title=record.title,
                state=state,
                disposition=record.disposition,
                disposition_state=disposition_state,
                open_prerequisites=tuple(open_prerequisites),
                unknown_prerequisites=tuple(unknown_prerequisites),
            )
        )

    label_carriers = tuple(
        identifier for identifier in known if DESCRIPTIVE_CAMPAIGN_LABEL in graph.records[identifier].labels
    )

    ready: ReadyComparison | None = None
    if ready_ids is not None:
        ready_set = dict.fromkeys(ready_ids)
        unblocked_ids = {entry.id for entry in classifications if entry.state == "unblocked"}
        ready = ReadyComparison(
            ready_ids=tuple(ready_set),
            ready_inside_closure=tuple(identifier for identifier in ready_set if identifier in members),
            ready_outside_closure=tuple(identifier for identifier in ready_set if identifier not in members),
            unblocked_absent_from_ready=tuple(sorted(unblocked_ids - set(ready_set))),
        )

    return ClosureReport(
        roots=tuple(roots),
        export_records=len(graph.records),
        edge_type_counts=dict(graph.edge_type_counts),
        ignored_edge_types=graph.ignored_edge_types,
        members=tuple(sorted(members)),
        dangling_references=tuple(dangling),
        closed_members=tuple(closed),
        classifications=tuple(classifications),
        label_carriers=label_carriers,
        ready=ready,
    )


@dataclass(frozen=True, slots=True)
class BackfillProposal:
    """A proposed ``metadata.disposition`` for one violating member.

    Only one token is *derivable*: acceptance criterion 3 states that a broad
    or underspecified record is ``specification-required`` and stays visible.
    The remaining five tokens encode a judgement about the record's content
    (is the residual implementation, bounded evidence, final-build evidence,
    already satisfied, or an adjudicated scope change?) that no field in the
    export decides. Those proposals are defaults carrying
    ``basis="unadjudicated-default"`` and must be confirmed by a human before
    anything is written.
    """

    id: str
    priority: int | None
    title: str
    proposed: str
    basis: str

    @property
    def is_derived(self) -> bool:
        return self.basis != "unadjudicated-default"

    @property
    def command(self) -> str:
        """The merging write.

        ``bd update --metadata`` takes a whole JSON object and *replaces* the
        record's metadata map, which on these records would destroy
        ``write_scope``, ``execution_shape``, ``dispatch_group`` and
        ``conflict_keys``. ``--set-metadata key=value`` merges one key, so it
        is the only form this plan will ever emit.
        """

        return f"bd update {self.id} --set-metadata disposition={self.proposed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "priority": self.priority,
            "title": self.title,
            "proposed": self.proposed,
            "basis": self.basis,
            "derived": self.is_derived,
            "command": self.command,
        }


def propose_dispositions(graph: CampaignGraph, report: ClosureReport) -> tuple[BackfillProposal, ...]:
    """Derive a disposition proposal for every violating unblocked member."""

    proposals: list[BackfillProposal] = []
    for entry in report.violations:
        record = graph.records[entry.id]
        shape = record.metadata.get("execution_shape")
        if not record.has_acceptance_text:
            proposed, basis = "specification-required", "no-acceptance-text"
        elif not shape:
            proposed, basis = "specification-required", "no-execution-shape"
        elif shape == "decision":
            proposed, basis = "specification-required", "execution_shape=decision"
        else:
            proposed, basis = "implementation-residual", "unadjudicated-default"
        proposals.append(
            BackfillProposal(
                id=entry.id,
                priority=entry.priority,
                title=entry.title,
                proposed=proposed,
                basis=basis,
            )
        )
    return tuple(sorted(proposals, key=lambda item: (item.priority is None, item.priority, item.id)))


def render_backfill_plan(proposals: Sequence[BackfillProposal]) -> str:
    """A dry run: the exact writes, never executed by this module."""

    derived = [item for item in proposals if item.is_derived]
    defaults = [item for item in proposals if not item.is_derived]
    lines: list[str] = []
    lines.append("")
    lines.append("disposition backfill plan (DRY RUN -- this module never writes to the tracker)")
    lines.append(f"  proposals                       {len(proposals):>6}")
    lines.append(f"    derived from criterion 3      {len(derived):>6}  (underspecified -> specification-required)")
    lines.append(f"    unadjudicated defaults        {len(defaults):>6}  (needs a human decision before any write)")
    lines.append("")
    counts = Counter(item.proposed for item in proposals)
    for token, count in sorted(counts.items()):
        lines.append(f"  {token:<28} {count:>6}")
    lines.append("")
    for item in proposals:
        lines.append(f"  # {_priority_token(item.priority)} basis={item.basis}")
        lines.append(f"  {item.command}")
    return "\n".join(lines) + "\n"


def _priority_token(priority: int | None) -> str:
    return f"P{priority}" if priority is not None else "P?"


def render_report(report: ClosureReport, *, list_members: bool = True) -> str:
    """Human-readable report whose arithmetic a second reader can check."""

    lines: list[str] = []
    lines.append("campaign closure (blocks-only)")
    lines.append(f"  roots                  {', '.join(report.roots)}")
    lines.append(f"  edge type followed     {SELECTION_EDGE_TYPE}")
    ignored = ", ".join(report.ignored_edge_types) or "<none present>"
    lines.append(f"  edge types IGNORED     {ignored}")
    lines.append("  (an ignored type is not a prerequisite; counting one manufactures phantom blockers)")
    lines.append("  direction              depends_on_id BLOCKS issue_id; the walk goes root -> prerequisites")
    lines.append("")
    lines.append("edge census (whole export)")
    for name, count in sorted(report.edge_type_counts.items()):
        marker = "FOLLOWED" if name == SELECTION_EDGE_TYPE else "ignored "
        lines.append(f"  {marker}  {name:<16} {count:>6}")
    lines.append("")
    lines.append("population")
    lines.append(f"  export records                 {report.export_records:>6}")
    lines.append(f"  closure members                {len(report.members):>6}")
    lines.append(f"    resolved to a record         {len(report.members) - len(report.dangling_references):>6}")
    lines.append(f"    dangling (absent from export){len(report.dangling_references):>6}")
    lines.append(f"  closed members                 {len(report.closed_members):>6}")
    lines.append(f"  nonclosed members              {len(report.classifications):>6}")
    lines.append(f"    unblocked                    {len(report.unblocked):>6}")
    lines.append(f"    blocked_open                 {len(report.blocked_open):>6}")
    lines.append(f"    blocked_unknown              {len(report.blocked_unknown):>6}")
    partition = "holds" if report.partition_holds else "BROKEN"
    lines.append(f"  partition unblocked+blocked_open+blocked_unknown == nonclosed: {partition}")
    if report.dangling_references:
        lines.append("  dangling prerequisite ids (status unmeasured, never counted as closed):")
        for identifier in report.dangling_references:
            lines.append(f"    {identifier}")
    lines.append("")
    lines.append("label filter counter-example (the label is descriptive, not membership)")
    lines.append(f"  resolved closure members:                            {report.resolved_members}")
    lines.append(f"  closure members carrying {DESCRIPTIVE_CAMPAIGN_LABEL}: {len(report.label_carriers)}")
    lines.append(
        f"  closure members lacking it:                          {report.resolved_members - len(report.label_carriers)}"
    )
    if report.ready is not None:
        lines.append("")
        lines.append("bd ready comparison (a ready view is not the population)")
        lines.append(f"  ready rows                       {len(report.ready.ready_ids):>6}")
        lines.append(f"    inside the closure             {len(report.ready.ready_inside_closure):>6}")
        lines.append(f"    outside the closure            {len(report.ready.ready_outside_closure):>6}")
        lines.append(f"  unblocked closure members absent from ready: {len(report.ready.unblocked_absent_from_ready)}")
        for identifier in report.ready.unblocked_absent_from_ready:
            lines.append(f"    {identifier}")
    lines.append("")
    lines.append("disposition (metadata.disposition; NOT acceptance_contract_v1.closure.disposition)")
    lines.append(f"  vocabulary: {', '.join(DISPOSITION_VOCABULARY)}")
    for token, count in report.disposition_counts.items():
        lines.append(f"  {token:<28} {count:>6}")
    lines.append(f"  unblocked without a valid disposition: {len(report.violations)}")
    if list_members and report.violations:
        lines.append("")
        lines.append("violations (unblocked, disposition missing or off-vocabulary)")
        for entry in sorted(report.violations, key=lambda item: (item.priority is None, item.priority, item.id)):
            lines.append(
                f"  {_priority_token(entry.priority)} {entry.id:<28} [{entry.disposition_state}] {entry.title[:70]}"
            )
    if list_members and report.blocked_open:
        lines.append("")
        lines.append("blocked_open members (first blocker shown)")
        for entry in sorted(report.blocked_open, key=lambda item: (item.priority is None, item.priority, item.id)):
            blocker = entry.open_prerequisites[0] if entry.open_prerequisites else "?"
            lines.append(f"  {_priority_token(entry.priority)} {entry.id:<28} blocked by {blocker}")
    return "\n".join(lines) + "\n"


def _read_export(source: str | None) -> list[str]:
    if source == "-":
        return sys.stdin.read().splitlines()
    if source is not None:
        return Path(source).read_text(encoding="utf-8").splitlines()
    try:
        completed = subprocess.run(  # fixed argv, no shell
            ["bd", "export"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:  # pragma: no cover - environment-dependent
        raise ExportRefusedError("bd is not on PATH; pass --export PATH with a `bd export` JSONL file") from error
    if completed.returncode != 0:  # pragma: no cover - environment-dependent
        raise ExportRefusedError(f"`bd export` failed with exit {completed.returncode}: {completed.stderr.strip()}")
    return completed.stdout.splitlines()


def _read_ready_ids(source: str) -> list[str]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ExportRefusedError("--ready-ids JSON must be a list")
        identifiers: list[str] = []
        for item in payload:
            if isinstance(item, str):
                identifiers.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                identifiers.append(str(item["id"]))
        return identifiers
    return [line.strip() for line in stripped.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devtools.campaign_closure",
        description="Blocks-only campaign closure over a bd export. Labels and ready views are not membership.",
    )
    parser.add_argument(
        "--export", default=None, help="bd export JSONL path, or - for stdin (default: run `bd export`)"
    )
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        default=None,
        help=f"closure root (repeatable; default: {', '.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument(
        "--ready-ids", default=None, help="`bd ready --json` output or newline-separated ids to compare"
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when an unblocked member lacks a valid disposition"
    )
    parser.add_argument("--summary", action="store_true", help="omit the per-member listings")
    parser.add_argument(
        "--backfill-plan",
        action="store_true",
        help="print the proposed metadata.disposition writes as a dry run; nothing is written",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = load_graph(_read_export(args.export))
        ready_ids = _read_ready_ids(args.ready_ids) if args.ready_ids else None
        report = compute_closure(graph, tuple(args.roots) if args.roots else DEFAULT_ROOTS, ready_ids=ready_ids)
    except ExportRefusedError as error:
        print(f"campaign-closure: refused: {error}", file=sys.stderr)
        return 2
    proposals = propose_dispositions(graph, report) if args.backfill_plan else ()
    if args.json:
        payload = report.to_dict()
        if args.backfill_plan:
            payload["backfill_plan"] = [item.to_dict() for item in proposals]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_report(report, list_members=not args.summary), end="")
        if args.backfill_plan:
            print(render_backfill_plan(proposals), end="")
    if not report.partition_holds:  # pragma: no cover - defensive arithmetic guard
        print("campaign-closure: refused: state partition does not sum to the nonclosed population", file=sys.stderr)
        return 2
    if args.check and report.violations:
        print(
            f"campaign-closure: {len(report.violations)} unblocked closure member(s) carry no valid "
            f"metadata.disposition",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
