from __future__ import annotations

import pytest

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
    assert "violations: dup_labels=0 inversions=0 missing_ac=0 malformed_wave=0" in out
