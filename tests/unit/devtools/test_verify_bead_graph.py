from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools import verify_bead_graph


def _issue(bead_id: str, *, dependencies: object = None) -> dict[str, object]:
    return {"id": bead_id, "dependencies": [] if dependencies is None else dependencies}


def test_clean_structured_graph_has_no_findings() -> None:
    issues = [
        _issue("parent"),
        _issue("child", dependencies=[{"type": "parent-child", "depends_on_id": "parent"}]),
        _issue("dependent", dependencies=[{"type": "blocks", "depends_on_id": "child"}]),
    ]
    assert verify_bead_graph.collect_findings(issues) == []
    assert verify_bead_graph.canonical_parent_map(issues)["child"] == "parent"


def test_dependency_integrity_rejects_malformed_duplicate_missing_and_self_edges() -> None:
    issues = [
        _issue("a", dependencies="invalid"),
        _issue("b", dependencies=[None, {"type": "blocks", "depends_on_id": ""}]),
        _issue(
            "c",
            dependencies=[
                {"type": "relates-to", "depends_on_id": "missing"},
                {"type": "blocks", "depends_on_id": "c"},
                {"type": "blocks", "depends_on_id": "c"},
            ],
        ),
    ]
    kinds = {finding.kind for finding in verify_bead_graph.collect_findings(issues)}
    assert kinds >= {
        "malformed-dependencies",
        "malformed-dependency",
        "missing-dependency-target",
        "self-dependency",
        "duplicate-dependency",
    }


def test_parent_cardinality_and_parent_cycle_are_rejected() -> None:
    issues = [
        _issue("a", dependencies=[{"type": "parent-child", "depends_on_id": "b"}]),
        _issue("b", dependencies=[{"type": "parent-child", "depends_on_id": "a"}]),
        _issue(
            "c",
            dependencies=[
                {"type": "parent-child", "depends_on_id": "a"},
                {"type": "parent-child", "depends_on_id": "b"},
            ],
        ),
    ]
    findings = verify_bead_graph.collect_findings(issues)
    assert {finding.kind for finding in findings} >= {"multiple-parents", "parent-cycle"}
    assert verify_bead_graph.canonical_parent_map(issues)["c"] is None


def test_blocks_cycle_is_rejected_from_export_without_invoking_bd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "issues.jsonl"
    export.write_text(
        "\n".join(
            json.dumps(issue)
            for issue in [
                _issue("a", dependencies=[{"type": "blocks", "depends_on_id": "b"}]),
                _issue("b", dependencies=[{"type": "blocks", "depends_on_id": "a"}]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("bd must not run"))
    assert verify_bead_graph.main(["--export", str(export), "--json"]) == 1


@pytest.mark.parametrize("payload", [["not-an-issue"], [{"id": ""}], [{"id": 42}], [{"id": "a"}, {"id": "a"}]])
def test_issue_population_validation_is_fail_closed(payload: list[object]) -> None:
    with pytest.raises(RuntimeError):
        verify_bead_graph._validated_issues(payload, source="test")


def test_export_json_error_names_line(tmp_path: Path) -> None:
    export = tmp_path / "issues.jsonl"
    export.write_text('{"id":"a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"issues\.jsonl:2"):
        verify_bead_graph._load_export(export)


def test_main_loads_live_population_and_reports_clean_graph(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (True, "no cycles"))
    monkeypatch.setattr(verify_bead_graph, "_run_bd_list_all", lambda: [_issue("a")])
    assert verify_bead_graph.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["issues_scanned"] == 1
    assert report["findings"] == []


def test_main_fails_closed_when_live_cycle_probe_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_bead_graph, "_run_bd_dep_cycles", lambda: (False, "a -> b -> a"))
    assert verify_bead_graph.main(["--json"]) == 1
    assert "dependency cycle check failed" in json.loads(capsys.readouterr().out)["error"]
