from __future__ import annotations

import json
import subprocess

import pytest

from devtools import verify_bead_graph


def _issue(
    id: str,
    *,
    status: str = "open",
    labels: list[str] | None = None,
    acceptance_criteria: str = "some AC",
    dependencies: list[dict[str, str]] | None = None,
    priority: int = 2,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": id,
        "title": f"Title for {id}",
        "status": status,
        "labels": labels if labels is not None else [],
        "acceptance_criteria": acceptance_criteria,
        "dependencies": dependencies if dependencies is not None else [],
        "priority": priority,
        "metadata": metadata if metadata is not None else {},
    }


def test_clean_graph_has_no_findings() -> None:
    issues = [_issue("polylogue-a", labels=["wave:1"]), _issue("polylogue-b", labels=["wave:1"])]
    assert verify_bead_graph.collect_findings(issues) == []


def test_duplicate_wave_labels_flagged() -> None:
    issues = [_issue("polylogue-a", labels=["wave:1", "wave:2"])]
    findings = verify_bead_graph.collect_findings(issues)
    assert [f.kind for f in findings] == ["duplicate-wave"]
    assert findings[0].bead_id == "polylogue-a"


def test_missing_acceptance_criteria_flagged() -> None:
    issues = [_issue("polylogue-a", acceptance_criteria="")]
    findings = verify_bead_graph.collect_findings(issues)
    assert [f.kind for f in findings] == ["missing-ac"]


def test_closed_beads_are_exempt() -> None:
    issues = [_issue("polylogue-a", status="closed", acceptance_criteria="", labels=["wave:1", "wave:2"])]
    assert verify_bead_graph.collect_findings(issues) == []


def test_wave_inversion_flagged_when_blocker_has_later_wave() -> None:
    issues = [
        _issue("polylogue-a", labels=["wave:1"], dependencies=[{"type": "blocks", "depends_on_id": "polylogue-b"}]),
        _issue("polylogue-b", labels=["wave:2"]),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    kinds = {f.kind for f in findings}
    assert "wave-inversion" in kinds
    inversion = next(f for f in findings if f.kind == "wave-inversion")
    assert inversion.bead_id == "polylogue-a"
    assert "polylogue-b" in inversion.detail


def test_wave_inversion_not_flagged_against_closed_blocker() -> None:
    issues = [
        _issue("polylogue-a", labels=["wave:1"], dependencies=[{"type": "blocks", "depends_on_id": "polylogue-b"}]),
        _issue("polylogue-b", labels=["wave:2"], status="closed"),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    assert not any(f.kind == "wave-inversion" for f in findings)


def test_non_blocks_dependency_type_ignored_for_wave_inversion() -> None:
    issues = [
        _issue("polylogue-a", labels=["wave:1"], dependencies=[{"type": "related", "depends_on_id": "polylogue-b"}]),
        _issue("polylogue-b", labels=["wave:2"]),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    assert not any(f.kind == "wave-inversion" for f in findings)


def test_malformed_wave_label_is_flagged_not_silently_skipped() -> None:
    """A non-numeric `wave:` value (e.g. `wave:later`) must fail loudly.

    The old bash script's Python subprocess would raise ValueError -- a
    malformed label is a lint-worthy data-entry mistake, not a "no wave"
    no-op. Regression coverage for the swallowed-ValueError bug caught in
    PR review on polylogue-kapb.
    """
    issues = [_issue("polylogue-a", labels=["wave:later"])]
    findings = verify_bead_graph.collect_findings(issues)
    assert [f.kind for f in findings] == ["malformed-wave"]
    assert findings[0].bead_id == "polylogue-a"
    assert "wave:later" in findings[0].detail


def test_malformed_wave_label_reported_once_per_bead() -> None:
    """A malformed wave must not be re-reported once per referencing dependent."""
    issues = [
        _issue(
            "polylogue-a", labels=["wave:1"], dependencies=[{"type": "blocks", "depends_on_id": "polylogue-broken"}]
        ),
        _issue(
            "polylogue-b", labels=["wave:1"], dependencies=[{"type": "blocks", "depends_on_id": "polylogue-broken"}]
        ),
        _issue("polylogue-broken", labels=["wave:soon"]),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    malformed = [f for f in findings if f.kind == "malformed-wave"]
    assert len(malformed) == 1
    assert malformed[0].bead_id == "polylogue-broken"


def test_malformed_wave_excluded_from_inversion_check() -> None:
    """An unparseable wave can't participate in ordinal comparison either way."""
    issues = [
        _issue("polylogue-a", labels=["wave:1"], dependencies=[{"type": "blocks", "depends_on_id": "polylogue-b"}]),
        _issue("polylogue-b", labels=["wave:soon"]),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    assert not any(f.kind == "wave-inversion" for f in findings)
    assert any(f.kind == "malformed-wave" and f.bead_id == "polylogue-b" for f in findings)


def test_closed_bead_with_malformed_wave_is_exempt() -> None:
    issues = [_issue("polylogue-a", status="closed", labels=["wave:later"])]
    assert verify_bead_graph.collect_findings(issues) == []


def test_main_exits_nonzero_and_reports_malformed_wave(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (True, "✓ No dependency cycles detected"))
    monkeypatch.setattr(
        verify_bead_graph,
        "_run_bd_list_all",
        lambda: [_issue("polylogue-a", labels=["wave:later"])],
    )

    rc = verify_bead_graph.main([])

    out = capsys.readouterr().out
    assert rc == 1
    assert "malformed-wave: polylogue-a" in out
    assert "malformed_wave=1" in out


def test_main_exits_zero_when_all_waves_are_well_formed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (True, ""))
    monkeypatch.setattr(verify_bead_graph, "_run_bd_list_all", lambda: [_issue("polylogue-a", labels=["wave:1"])])

    rc = verify_bead_graph.main([])

    out = capsys.readouterr().out
    assert rc == 0
    assert "violations: dup_labels=0 inversions=0 missing_ac=0 malformed_wave=0 parent_integrity=0" in out


def test_parent_child_validation_allows_zero_or_one_canonical_parent() -> None:
    issues = [
        _issue("polylogue-parent"),
        _issue("polylogue-child", dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-parent"}]),
        _issue("polylogue-root"),
    ]

    assert verify_bead_graph.canonical_parent_map(issues) == {
        "polylogue-parent": None,
        "polylogue-child": "polylogue-parent",
        "polylogue-root": None,
    }
    assert not [finding for finding in verify_bead_graph.collect_findings(issues) if "parent" in finding.kind]


def test_parent_child_validation_rejects_multiple_missing_and_cyclic_parents() -> None:
    issues = [
        _issue("polylogue-a", dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-b"}]),
        _issue(
            "polylogue-b",
            dependencies=[
                {"type": "parent-child", "depends_on_id": "polylogue-a"},
                {"type": "parent-child", "depends_on_id": "polylogue-c"},
            ],
        ),
        _issue("polylogue-d", dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-absent"}]),
        _issue("polylogue-cycle-a", dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-cycle-b"}]),
        _issue("polylogue-cycle-b", dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-cycle-a"}]),
    ]

    findings = verify_bead_graph.collect_findings(issues)
    assert {(finding.kind, finding.bead_id) for finding in findings} >= {
        ("multiple-parents", "polylogue-b"),
        ("missing-parent", "polylogue-d"),
        ("parent-cycle", "polylogue-cycle-a"),
    }


def test_parent_child_validation_rejects_duplicate_edge_records() -> None:
    issues = [
        _issue("polylogue-parent"),
        _issue(
            "polylogue-child",
            dependencies=[
                {"type": "parent-child", "depends_on_id": "polylogue-parent"},
                {"type": "parent-child", "depends_on_id": "polylogue-parent"},
            ],
        ),
    ]

    findings = verify_bead_graph.collect_findings(issues)

    assert verify_bead_graph.canonical_parent_map(issues)["polylogue-child"] is None
    assert ("multiple-parents", "polylogue-child") in {(finding.kind, finding.bead_id) for finding in findings}


@pytest.mark.parametrize("payload", [["not-an-issue"], [{"id": ""}], [{"id": 42}]])
def test_bd_list_rejects_each_malformed_issue_record(monkeypatch: pytest.MonkeyPatch, payload: list[object]) -> None:
    class Completed:
        stdout = json.dumps(payload)

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())

    with pytest.raises(RuntimeError, match="expected object with non-empty string id|has no non-empty string id"):
        verify_bead_graph._run_bd_list_all()


def test_main_reports_bd_cycle_launch_failure_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_cycles() -> tuple[bool, str]:
        raise OSError("bd unavailable")

    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", fail_cycles)

    assert verify_bead_graph.main(["--json"]) == 1
    assert json.loads(capsys.readouterr().out) == {"error": "bd unavailable", "report_version": 1}


def test_parent_relationship_survives_json_import_export_and_merge_shape() -> None:
    """The production census relies only on structured dependency records."""
    exported = [
        _issue("polylogue-program"),
        _issue(
            "polylogue-child",
            dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-program"}],
            acceptance_criteria="",
        ),
    ]
    imported = json.loads(json.dumps(exported))
    merged = [*imported, _issue("polylogue-unrelated")]

    report = verify_bead_graph.build_report(merged, cycles_ok=True, cycles_output="")

    assert report["counts"].get("multiple-parents", 0) == 0
    item = report["missing_ac_census"]["items"][0]
    assert item["id"] == "polylogue-child"
    assert item["program_or_parent"] == "polylogue-program"


def test_json_report_lists_every_missing_ac_with_deterministic_partitions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    issues = [
        _issue("polylogue-parent", status="open"),
        _issue(
            "polylogue-a",
            status="open",
            priority=1,
            acceptance_criteria="",
            dependencies=[{"type": "parent-child", "depends_on_id": "polylogue-parent"}],
            labels=["campaign:reindex"],
        ),
        _issue("polylogue-b", status="in_progress", priority=2, acceptance_criteria=""),
        _issue("polylogue-c", status="closed", acceptance_criteria=""),
    ]
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (True, "cycle check clean"))
    monkeypatch.setattr(verify_bead_graph, "_run_bd_list_all", lambda: issues)

    assert verify_bead_graph.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)

    census = payload["missing_ac_census"]
    assert census["total"] == 2
    assert [item["id"] for item in census["items"]] == ["polylogue-b", "polylogue-a"]
    assert census["by_status"]["open"] == {"count": 1, "ids": ["polylogue-a"]}
    assert census["by_priority"]["1"] == {"count": 1, "ids": ["polylogue-a"]}
    assert census["by_program_or_parent"]["polylogue-parent"]["ids"] == ["polylogue-a"]
    assert census["by_campaign_relevance"]["declared"]["ids"] == ["polylogue-a"]
