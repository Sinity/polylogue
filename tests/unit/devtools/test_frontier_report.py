from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import frontier_report


def _issue(
    bead_id: str,
    *,
    status: str = "open",
    priority: int = 2,
    dependencies: list[dict[str, str]] | None = None,
    design: str = "",
    labels: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": bead_id,
        "title": f"title for {bead_id}",
        "status": status,
        "priority": priority,
        "issue_type": "task",
        "dependencies": dependencies or [],
        "design": design,
        "labels": labels or [],
        "metadata": metadata or {},
    }


def test_execution_focus_derives_ready_priority_leverage_claim_and_resource_constraints() -> None:
    issues = [
        _issue("claim-schema", status="in_progress", design="migration 008_add_column.sql"),
        _issue("claim-overlap", status="in_progress", design="polylogue/mcp/server.py"),
        _issue("schema-candidate", priority=0, design="migration 009_add_other.sql"),
        _issue("overlap-candidate", priority=1, design="polylogue/mcp/server.py"),
        _issue("leverage", priority=1),
        _issue("ordinary", priority=1),
        _issue("blocked", dependencies=[{"type": "blocks", "depends_on_id": "leverage"}]),
    ]
    report = frontier_report.build_report(
        issues,
        [issue for issue in issues if issue["id"] in {"schema-candidate", "overlap-candidate", "leverage", "ordinary"}],
        repo=Path("/repo"),
    )

    focus = report["execution_focus"]
    assert focus["occupied_claims"] == ["claim-overlap", "claim-schema"]
    assert [item["id"] for item in focus["focus"]] == ["leverage", "ordinary"]
    assert focus["focus"][0]["critical_path_leverage"] == 1
    deferred = {item["id"]: item for item in focus["deferred"]}
    assert "schema-lane occupied by claim(s): claim-schema" in deferred["schema-candidate"]["reason"]
    assert deferred["overlap-candidate"]["conflicts_with_claims"] == ["claim-overlap"]
    assert report["counts"] == {
        "ambition": 7,
        "active_set": 0,
        "claims": 2,
        "dependency_ready": 4,
        "execution_focus": 2,
        "deferred": 2,
    }


def test_execution_focus_reports_active_set_without_mutating_admission() -> None:
    issue = _issue("active", metadata={"frontier": "active", "frontier_program_ref": "program"})
    report = frontier_report.build_report([issue], [issue], repo=Path("/repo"))

    assert report["active_set"] == ["active"]
    assert issue["metadata"] == {"frontier": "active", "frontier_program_ref": "program"}
    assert report["execution_focus"]["focus"][0]["id"] == "active"


def test_main_uses_unbounded_live_surfaces_and_emits_complete_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    issues = [_issue("a"), _issue("b")]

    def fake_run(_repo: Path, args: list[str]) -> list[dict[str, object]]:
        calls.append(args)
        return issues

    monkeypatch.setattr(frontier_report, "_run_bd", fake_run)

    assert frontier_report.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [["list", "--all", "--limit", "0"], ["ready", "--limit", "0"]]
    assert [item["id"] for item in payload["execution_focus"]["candidates"]] == ["a", "b"]
    assert payload["execution_focus"]["resource_policy"] == frontier_report.RESOURCE_POLICY
