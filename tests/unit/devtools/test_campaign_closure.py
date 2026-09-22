"""The blocks-only campaign closure must survive its own inversions.

Anti-vacuity (each mutation is executed, not asserted):

* reverse the edge direction in ``load_graph``
  (``prerequisites.setdefault(prerequisite, ...).add(dependent)``) ->
  ``test_closure_walks_to_prerequisites_not_dependents`` and
  ``test_unblocked_requires_every_blocker_closed`` go red;
* drop the ``if edge_type != SELECTION_EDGE_TYPE: continue`` guard so
  ``relates-to``/``discovered-from`` become prerequisites ->
  ``test_only_blocks_edges_enter_the_closure`` and
  ``test_report_names_followed_and_ignored_edges`` go red;
* delete the ``if record.is_closed: ... continue`` branch in
  ``compute_closure`` -> ``test_closed_members_leave_the_selection`` and
  ``test_state_partition_sums_to_nonclosed`` go red.

The fixture is built so each mutation moves a *different* observable, which is
why it carries a dependent (``bead-dependent``), a ``relates-to`` sibling
(``bead-sibling``), two closed records and a dangling prerequisite id.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from devtools.campaign_closure import (
    DESCRIPTIVE_CAMPAIGN_LABEL,
    SELECTION_EDGE_TYPE,
    ClosureReport,
    ExportRefusedError,
    compute_closure,
    load_graph,
    main,
    propose_dispositions,
    render_backfill_plan,
    render_report,
)

ROOT = "bead-root"

#: (id, status, extra fields) for the synthetic campaign.
_RECORDS: tuple[dict[str, object], ...] = (
    {
        "id": ROOT,
        "status": "open",
        "priority": 0,
        "title": "campaign root",
        "labels": [DESCRIPTIVE_CAMPAIGN_LABEL],
        "dependencies": [
            {"issue_id": ROOT, "depends_on_id": "bead-impl", "type": "blocks"},
            {"issue_id": ROOT, "depends_on_id": "bead-shipped", "type": "blocks"},
            {"issue_id": ROOT, "depends_on_id": "bead-pending", "type": "blocks"},
            {"issue_id": ROOT, "depends_on_id": "bead-sibling", "type": "relates-to"},
        ],
    },
    {
        "id": "bead-impl",
        "status": "open",
        "priority": 1,
        "title": "implementation residual",
        "labels": [DESCRIPTIVE_CAMPAIGN_LABEL],
        "acceptance_criteria": "1. it works",
        "metadata": {"disposition": "implementation-residual", "execution_shape": "leaf"},
        "dependencies": [{"issue_id": "bead-impl", "depends_on_id": "bead-done", "type": "blocks"}],
    },
    {"id": "bead-done", "status": "closed", "priority": 1, "title": "already closed prerequisite"},
    {
        "id": "bead-shipped",
        "status": "closed",
        "priority": 2,
        "title": "closed, but still carries a live prerequisite",
        "dependencies": [{"issue_id": "bead-shipped", "depends_on_id": "bead-orphan", "type": "blocks"}],
    },
    # Carries no label, no execution_shape, no acceptance text and no
    # disposition: exactly the record a selector-metadata filter would drop.
    {"id": "bead-orphan", "status": "open", "priority": 2, "title": "underspecified but real work"},
    {
        "id": "bead-pending",
        "status": "in_progress",
        "priority": 1,
        "title": "waits on an id the export does not contain",
        "metadata": {"disposition": "bounded-evidence"},
        "dependencies": [{"issue_id": "bead-pending", "depends_on_id": "bead-absent", "type": "blocks"}],
    },
    {
        "id": "bead-dependent",
        "status": "open",
        "priority": 3,
        "title": "blocked BY the root; must never enter the closure",
        "dependencies": [{"issue_id": "bead-dependent", "depends_on_id": ROOT, "type": "blocks"}],
    },
    {
        "id": "bead-sibling",
        "status": "open",
        "priority": 3,
        "title": "merely relates-to the root",
        "dependencies": [{"issue_id": "bead-sibling", "depends_on_id": ROOT, "type": "discovered-from"}],
    },
)

#: Closure members under the correct direction and the blocks-only filter.
EXPECTED_MEMBERS = frozenset(
    {ROOT, "bead-impl", "bead-done", "bead-shipped", "bead-orphan", "bead-pending", "bead-absent"}
)


def _export_lines(records: tuple[dict[str, object], ...] = _RECORDS) -> list[str]:
    return [json.dumps({"_type": "issue", **record}) for record in records]


def _report(*, ready_ids: Sequence[str] | None = None) -> ClosureReport:
    return compute_closure(load_graph(_export_lines()), [ROOT], ready_ids=ready_ids)


def test_closure_walks_to_prerequisites_not_dependents() -> None:
    report = _report()

    assert set(report.members) == EXPECTED_MEMBERS
    # bead-dependent depends on the root. Reversing the edge direction pulls it
    # in and drops every genuine prerequisite.
    assert "bead-dependent" not in report.members
    assert "bead-impl" in report.members


def test_unblocked_requires_every_blocker_closed() -> None:
    report = _report()
    states = {entry.id: entry.state for entry in report.classifications}

    # bead-impl's only blocks-prerequisite (bead-done) is closed.
    assert states["bead-impl"] == "unblocked"
    # The root still waits on bead-impl and bead-pending.
    assert states[ROOT] == "blocked_open"
    blocked = {entry.id: entry.open_prerequisites for entry in report.blocked_open}
    assert set(blocked[ROOT]) == {"bead-impl", "bead-pending"}


def test_only_blocks_edges_enter_the_closure() -> None:
    report = _report()

    # bead-sibling is reachable from the root by relates-to and reaches the
    # root by discovered-from. Neither is a prerequisite.
    assert "bead-sibling" not in report.members
    assert len(report.members) == len(EXPECTED_MEMBERS)
    assert set(report.ignored_edge_types) == {"relates-to", "discovered-from"}


def test_report_names_followed_and_ignored_edges() -> None:
    text = render_report(_report())

    assert f"edge type followed     {SELECTION_EDGE_TYPE}" in text
    assert "edge types IGNORED     discovered-from, relates-to" in text
    assert f"FOLLOWED  {SELECTION_EDGE_TYPE}" in text
    assert "ignored   relates-to" in text
    assert "depends_on_id BLOCKS issue_id" in text


def test_closed_members_leave_the_selection() -> None:
    report = _report()

    assert set(report.closed_members) == {"bead-done", "bead-shipped"}
    nonclosed = {entry.id for entry in report.classifications}
    assert nonclosed == {ROOT, "bead-impl", "bead-orphan", "bead-pending"}
    assert "bead-done" not in nonclosed
    assert "bead-shipped" not in nonclosed


def test_state_partition_sums_to_nonclosed() -> None:
    report = _report()

    assert len(report.unblocked) == 2
    assert len(report.blocked_open) == 1
    assert len(report.blocked_unknown) == 1
    assert len(report.classifications) == 4
    assert report.partition_holds


def test_unknown_prerequisite_is_not_a_closed_one() -> None:
    report = _report()

    assert report.dangling_references == ("bead-absent",)
    pending = next(entry for entry in report.blocked_unknown)
    assert pending.id == "bead-pending"
    assert pending.unknown_prerequisites == ("bead-absent",)
    # It carries a valid disposition, so only the unknown blocker keeps it out.
    assert pending.disposition == "bounded-evidence"
    assert {entry.id for entry in report.unblocked} == {"bead-impl", "bead-orphan"}


def test_underspecified_record_stays_in_population() -> None:
    report = _report()
    orphan = next(entry for entry in report.unblocked if entry.id == "bead-orphan")

    assert orphan.state == "unblocked"
    assert orphan.disposition_state == "absent"
    assert orphan.id in {entry.id for entry in report.violations}
    assert "bead-orphan" in render_report(report)


def test_label_filter_would_drop_closure_members() -> None:
    report = _report()

    assert set(report.label_carriers) == {ROOT, "bead-impl"}
    assert len(report.label_carriers) < len(report.members)
    text = render_report(report)
    assert f"closure members carrying {DESCRIPTIVE_CAMPAIGN_LABEL}: 2" in text


def test_ready_view_omits_unblocked_members() -> None:
    report = _report(ready_ids=["bead-impl", "bead-dependent"])

    assert report.ready is not None
    assert report.ready.ready_inside_closure == ("bead-impl",)
    assert report.ready.ready_outside_closure == ("bead-dependent",)
    assert report.ready.unblocked_absent_from_ready == ("bead-orphan",)


def test_off_vocabulary_disposition_is_a_violation() -> None:
    records = tuple(
        {**record, "metadata": {"disposition": "needs-work"}} if record["id"] == "bead-orphan" else record
        for record in _RECORDS
    )
    report = compute_closure(load_graph(_export_lines(records)), [ROOT])
    orphan = next(entry for entry in report.unblocked if entry.id == "bead-orphan")

    assert orphan.disposition == "needs-work"
    assert orphan.disposition_state == "off-vocabulary"
    assert orphan.id in {entry.id for entry in report.violations}


def test_nested_closure_disposition_is_not_token() -> None:
    """``acceptance_contract_v1.closure.disposition`` is a different field."""

    nested = json.dumps({"closure": {"disposition": "already-satisfied"}})
    records = tuple(
        {**record, "metadata": {"acceptance_contract_v1": nested}} if record["id"] == "bead-orphan" else record
        for record in _RECORDS
    )
    report = compute_closure(load_graph(_export_lines(records)), [ROOT])
    orphan = next(entry for entry in report.unblocked if entry.id == "bead-orphan")

    assert orphan.disposition is None
    assert orphan.disposition_state == "absent"
    assert orphan.id in {entry.id for entry in report.violations}


def test_missing_root_is_refused() -> None:
    graph = load_graph(_export_lines())

    with pytest.raises(ExportRefusedError, match="root"):
        compute_closure(graph, ["bead-nonexistent"])
    with pytest.raises(ExportRefusedError, match="no root"):
        compute_closure(graph, [])


def test_check_exit_codes_follow_violations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = tmp_path / "export.jsonl"
    export.write_text("\n".join(_export_lines()), encoding="utf-8")

    assert main(["--export", str(export), "--root", ROOT]) == 0
    assert main(["--export", str(export), "--root", ROOT, "--check"]) == 1
    capsys.readouterr()

    dispositioned = tuple(
        {**record, "metadata": {"disposition": "specification-required"}} if record["id"] == "bead-orphan" else record
        for record in _RECORDS
    )
    export.write_text("\n".join(_export_lines(dispositioned)), encoding="utf-8")
    assert main(["--export", str(export), "--root", ROOT, "--check"]) == 0
    capsys.readouterr()

    assert main(["--export", str(export), "--root", ROOT, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["followed_edge_type"] == SELECTION_EDGE_TYPE
    assert payload["unblocked"] == 2
    assert payload["violations"] == []


def test_backfill_plan_derives_only_criterion_three() -> None:
    """Only `specification-required` is derivable; the rest are defaults."""

    graph = load_graph(_export_lines())
    report = compute_closure(graph, [ROOT])
    proposals = {item.id: item for item in propose_dispositions(graph, report)}

    # bead-orphan has no acceptance text -> criterion 3 decides it mechanically.
    assert proposals["bead-orphan"].proposed == "specification-required"
    assert proposals["bead-orphan"].basis == "no-acceptance-text"
    assert proposals["bead-orphan"].is_derived
    assert proposals["bead-orphan"].command == "bd update bead-orphan --metadata disposition=specification-required"

    # bead-impl already carries a valid disposition, so it is not a violation.
    assert "bead-impl" not in proposals


def test_backfill_default_is_flagged_unadjudicated() -> None:
    records = tuple(
        {
            **record,
            "acceptance_criteria": "1. a stated criterion",
            "metadata": {"execution_shape": "leaf"},
        }
        if record["id"] == "bead-orphan"
        else record
        for record in _RECORDS
    )
    graph = load_graph(_export_lines(records))
    report = compute_closure(graph, [ROOT])
    (proposal,) = propose_dispositions(graph, report)

    assert proposal.id == "bead-orphan"
    assert proposal.proposed == "implementation-residual"
    assert proposal.basis == "unadjudicated-default"
    assert not proposal.is_derived
    plan = render_backfill_plan((proposal,))
    assert "DRY RUN" in plan
    assert "unadjudicated defaults" in plan
    assert proposal.command in plan


def test_decision_shape_is_specification_required() -> None:
    records = tuple(
        {
            **record,
            "acceptance_criteria": "1. a stated criterion",
            "metadata": {"execution_shape": "decision"},
        }
        if record["id"] == "bead-orphan"
        else record
        for record in _RECORDS
    )
    graph = load_graph(_export_lines(records))
    (proposal,) = propose_dispositions(graph, compute_closure(graph, [ROOT]))

    assert proposal.proposed == "specification-required"
    assert proposal.basis == "execution_shape=decision"
