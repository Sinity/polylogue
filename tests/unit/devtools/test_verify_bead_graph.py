from __future__ import annotations

from devtools import verify_bead_graph


def _issue(
    id: str,
    *,
    status: str = "open",
    labels: list[str] | None = None,
    acceptance_criteria: str = "some AC",
    dependencies: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "id": id,
        "title": f"Title for {id}",
        "status": status,
        "labels": labels if labels is not None else [],
        "acceptance_criteria": acceptance_criteria,
        "dependencies": dependencies if dependencies is not None else [],
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
