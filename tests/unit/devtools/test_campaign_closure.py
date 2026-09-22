"""Blocks-only campaign closure selection.

Fixtures are neutral synthetic graphs. Nothing here reads the real tracker: the
defect under test is a selection rule, and a rule is only demonstrable on a
graph whose membership is known by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.campaign_closure import (
    DISPOSITIONS,
    build_report,
    closure_ids,
    disposition_of,
    load_records,
    main,
)

ROOT = "syn-root"


def issue(
    ident: str,
    *,
    status: str = "open",
    priority: int = 2,
    blocks_on: tuple[str, ...] = (),
    other_edges: tuple[tuple[str, str], ...] = (),
    disposition: str | None = None,
    labels: tuple[str, ...] = (),
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """One synthetic export record.

    ``blocks_on`` are ``blocks`` prerequisites; ``other_edges`` are
    ``(target, type)`` pairs for edge types that must NOT confer membership.
    """

    edges = [{"issue_id": ident, "depends_on_id": target, "type": "blocks"} for target in blocks_on]
    edges += [{"issue_id": ident, "depends_on_id": target, "type": kind} for target, kind in other_edges]
    meta = dict(metadata or {})
    if disposition is not None:
        meta["disposition"] = disposition
    return {
        "_type": "issue",
        "id": ident,
        "title": f"synthetic {ident}",
        "status": status,
        "priority": priority,
        "labels": list(labels),
        "metadata": meta,
        "dependencies": edges,
        "dependency_count": len(edges),
    }


def as_jsonl(records: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


def test_closure_follows_only_blocks_edges() -> None:
    """A relates-to or discovered-from neighbour is not a campaign member.

    Anti-vacuity: accepting every edge type puts ``syn-rel`` and ``syn-disc``
    in the closure, and this assertion goes red.
    """

    records = [
        issue(
            ROOT,
            blocks_on=("syn-a",),
            other_edges=(("syn-rel", "relates-to"), ("syn-disc", "discovered-from")),
        ),
        issue("syn-a", blocks_on=("syn-b",)),
        issue("syn-b"),
        issue("syn-rel"),
        issue("syn-disc"),
    ]
    assert closure_ids(records, [ROOT]) == {ROOT, "syn-a", "syn-b"}


def test_population_ignores_labels_and_campaign_metadata() -> None:
    """Selection may not shrink because a record lacks selector metadata.

    This is the bead's own anti-vacuity condition: reintroducing a
    ``campaign:*`` label filter or a readiness filter drops the bare record and
    this assertion goes red.
    """

    labelled = issue("syn-labelled", labels=("campaign:x",))
    bare = issue("syn-bare")
    assert bare["labels"] == []
    assert bare["metadata"] == {}

    root = issue(ROOT, blocks_on=("syn-labelled", "syn-bare"))
    report = build_report([root, labelled, bare], [ROOT])
    assert {member.id for member in report.unblocked} == {
        "syn-labelled",
        "syn-bare",
    }


def test_unblocked_needs_every_prerequisite_closed() -> None:
    """One open prerequisite is enough to hold a record back.

    Anti-vacuity: treating "any prerequisite closed" as unblocked makes
    ``syn-waiting`` unblocked and this assertion goes red.
    """

    records = [
        issue(ROOT, blocks_on=("syn-waiting",)),
        issue("syn-waiting", blocks_on=("syn-done", "syn-open")),
        issue("syn-done", status="closed"),
        issue("syn-open"),
    ]
    report = build_report(records, [ROOT])
    waiting = next(m for m in report.nonclosed if m.id == "syn-waiting")
    assert waiting.unblocked is False
    assert waiting.open_prerequisites == ("syn-open",)


def test_absent_prerequisite_is_not_assumed_closed() -> None:
    """An id the export does not contain cannot be shown closed.

    Anti-vacuity: skipping unknown prerequisites makes ``syn-x`` unblocked.
    """

    records = [issue(ROOT, blocks_on=("syn-x",)), issue("syn-x", blocks_on=("syn-?",))]
    report = build_report(records, [ROOT])
    member = next(m for m in report.nonclosed if m.id == "syn-x")
    assert member.unblocked is False
    assert "syn-?" in report.missing


def test_violations_are_unblocked_without_disposition() -> None:
    """Criterion 1's count, and its exit status.

    Anti-vacuity: deleting the disposition from ``syn-ok`` raises the count to
    two; accepting an arbitrary token drops it to zero.
    """

    records = [
        issue(ROOT, blocks_on=("syn-ok", "syn-none", "syn-bogus")),
        issue("syn-ok", disposition="implementation-residual"),
        issue("syn-none"),
        issue("syn-bogus", disposition="totally-made-up"),
    ]
    report = build_report(records, [ROOT])
    assert {member.id for member in report.violations} == {"syn-none", "syn-bogus"}


@pytest.mark.parametrize("token", sorted(DISPOSITIONS))
def test_every_vocabulary_token_discharges(token: str) -> None:
    """Each of the six tokens counts; nothing outside the six does."""

    record = issue("syn-t", disposition=token)
    assert disposition_of(record) == token
    report = build_report([issue(ROOT, blocks_on=("syn-t",)), record], [ROOT])
    assert report.violations == ()


def test_underspecified_record_stays_visible_as_a_member() -> None:
    """Criterion 3: a bare record is reported, not dropped.

    Anti-vacuity: filtering members on ``execution_shape`` or acceptance text
    removes ``syn-broad`` from ``nonclosed`` and this goes red.
    """

    bare = issue("syn-broad", disposition="specification-required")
    report = build_report([issue(ROOT, blocks_on=("syn-broad",)), bare], [ROOT])
    member = next(m for m in report.nonclosed if m.id == "syn-broad")
    assert member.disposition == "specification-required"
    assert member.unblocked is True


def test_metadata_exported_as_a_json_string_is_read() -> None:
    """``metadata`` is sometimes a JSON string in an export."""

    record = issue("syn-s")
    record["metadata"] = json.dumps({"disposition": "bounded-evidence"})
    assert disposition_of(record) == "bounded-evidence"


def test_cycle_terminates() -> None:
    """A blocks cycle must not hang the traversal."""

    records = [
        issue(ROOT, blocks_on=("syn-p",)),
        issue("syn-p", blocks_on=("syn-q",)),
        issue("syn-q", blocks_on=("syn-p",)),
    ]
    assert closure_ids(records, [ROOT]) == {ROOT, "syn-p", "syn-q"}


def test_edge_on_other_endpoint_is_not_a_prereq() -> None:
    """Only an edge whose ``issue_id`` is this record states its prerequisite.

    Anti-vacuity: ignoring ``issue_id`` makes the mirrored edge on ``syn-m``
    read as ``syn-m`` depending on ``syn-n``, and this goes red.
    """

    mirrored = issue("syn-m")
    edges = mirrored["dependencies"]
    assert isinstance(edges, list)
    edges.append({"issue_id": "syn-n", "depends_on_id": "syn-z", "type": "blocks"})
    report = build_report([issue(ROOT, blocks_on=("syn-m",)), mirrored], [ROOT])
    member = next(m for m in report.nonclosed if m.id == "syn-m")
    assert member.open_prerequisites == ()
    assert "syn-z" not in report.missing


def test_main_exits_nonzero_on_violations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The check asserts; it does not merely print.

    Anti-vacuity: returning 0 unconditionally makes this go red.
    """

    path = tmp_path / "export.jsonl"
    path.write_text(
        as_jsonl([issue(ROOT, blocks_on=("syn-v",)), issue("syn-v")]),
        encoding="utf-8",
    )
    assert main(["--export", str(path), "--root", ROOT]) == 1
    assert "syn-v" in capsys.readouterr().out

    path.write_text(
        as_jsonl(
            [
                issue(ROOT, blocks_on=("syn-v",)),
                issue("syn-v", disposition="already-satisfied"),
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--export", str(path), "--root", ROOT, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["violation_count"] == 0
    assert payload["unblocked_count"] == 1


def test_load_records_skips_blanks_and_non_issue_rows() -> None:
    text = as_jsonl([issue(ROOT)]) + "\n" + json.dumps({"_type": "memory"}) + "\n"
    records = load_records(text.splitlines())
    assert [record["id"] for record in records] == [ROOT]
