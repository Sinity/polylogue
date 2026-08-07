from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    issue_type: str = "task",
) -> dict[str, object]:
    return {
        "id": bead_id,
        "title": f"title for {bead_id}",
        "status": status,
        "priority": priority,
        "issue_type": issue_type,
        "dependencies": dependencies or [],
        "design": design,
        "labels": labels or [],
        "metadata": metadata or {},
    }


def test_execution_focus_derives_ready_priority_leverage_claim_and_resource_constraints() -> None:
    issues = [
        _issue("claim-schema", status="in_progress", design="migration 008_add_column.sql"),
        _issue("claim-overlap", status="in_progress", design="polylogue/mcp/server.py"),
        _issue("schema-candidate", priority=0, design="migration 009_add_other.sql", metadata={"frontier": "active"}),
        _issue("overlap-candidate", priority=1, design="polylogue/mcp/server.py", metadata={"frontier": "active"}),
        _issue("leverage", priority=1, design="polylogue/core/leverage.py", metadata={"frontier": "active"}),
        _issue("ordinary", priority=1, design="devtools/ordinary.py", metadata={"frontier": "active"}),
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
        "active_set": 4,
        "claims": 2,
        "dependency_ready": 4,
        "execution_focus": 2,
        "deferred": 2,
    }


def test_execution_focus_reports_active_set_without_mutating_admission() -> None:
    issue = _issue(
        "active",
        design="devtools/frontier_report.py",
        metadata={"frontier": "active", "frontier_program_ref": "program"},
    )
    report = frontier_report.build_report([issue], [issue], repo=Path("/repo"))

    assert report["active_set"] == ["active"]
    assert issue["metadata"] == {"frontier": "active", "frontier_program_ref": "program"}
    assert report["execution_focus"]["focus"][0]["id"] == "active"


def test_build_report_rejects_ready_ids_absent_from_the_issue_snapshot() -> None:
    issue = _issue("listed", design="devtools/frontier_report.py")
    ready = _issue("ready-only", design="devtools/frontier_report.py")

    with pytest.raises(RuntimeError, match="ready snapshot contains IDs absent"):
        frontier_report.build_report([issue], [ready], repo=Path("/repo"))


def test_schema_tier_ddl_occupies_the_schema_lane() -> None:
    issue = _issue(
        "ddl",
        design="Add the new table to polylogue/storage/sqlite/archive_tiers/index.py.",
        metadata={"frontier": "active"},
    )

    report = frontier_report.build_report([issue], [issue], repo=Path("/repo"))

    assert report["execution_focus"]["focus"][0]["resource_classes"] == ["schema-lane"]


def test_execution_focus_excludes_unadmitted_ready_work_and_emits_full_ambition() -> None:
    issues = [
        _issue("active", design="devtools/frontier_report.py", metadata={"frontier": "active"}),
        _issue("unadmitted", design="polylogue/core/unadmitted.py"),
        _issue("claimed", status="in_progress", design="polylogue/core/claimed.py"),
        _issue("program", issue_type="epic"),
    ]

    report = frontier_report.build_report(issues, [issues[0], issues[1], issues[3]], repo=Path("/repo"))

    assert [item["id"] for item in report["execution_focus"]["candidates"]] == ["active"]
    assert [item["id"] for item in report["execution_focus"]["focus"]] == ["active"]
    assert report["counts"]["ambition"] == 4
    assert report["ambition"] == [
        {
            "id": "active",
            "title": "title for active",
            "status": "open",
            "priority": 2,
            "issue_type": "task",
            "frontier": "active",
            "frontier_program_ref": None,
            "horizons": [],
        },
        {
            "id": "claimed",
            "title": "title for claimed",
            "status": "in_progress",
            "priority": 2,
            "issue_type": "task",
            "frontier": None,
            "frontier_program_ref": None,
            "horizons": [],
        },
        {
            "id": "program",
            "title": "title for program",
            "status": "open",
            "priority": 2,
            "issue_type": "epic",
            "frontier": None,
            "frontier_program_ref": None,
            "horizons": [],
        },
        {
            "id": "unadmitted",
            "title": "title for unadmitted",
            "status": "open",
            "priority": 2,
            "issue_type": "task",
            "frontier": None,
            "frontier_program_ref": None,
            "horizons": [],
        },
    ]


def test_execution_focus_only_counts_executable_claimed_leaves() -> None:
    issues = [
        _issue("claimed-epic", status="in_progress", issue_type="epic", design="migration 008_program.sql"),
        _issue("claimed-program", status="in_progress", issue_type="program", design="migration 009_program.sql"),
        _issue("claimed-leaf", status="in_progress", design="migration 010_leaf.sql"),
        _issue("candidate", design="migration 011_candidate.sql", metadata={"frontier": "active"}),
    ]

    report = frontier_report.build_report(issues, [issues[-1]], repo=Path("/repo"))

    assert report["counts"]["claims"] == 1
    assert report["execution_focus"]["occupied_claims"] == ["claimed-leaf"]
    assert [item["id"] for item in report["execution_focus"]["deferred"]] == ["candidate"]
    assert report["execution_focus"]["deferred"][0]["reason"] == "schema-lane occupied by claim(s): claimed-leaf"
    assert [item["id"] for item in report["ambition"]] == [
        "candidate",
        "claimed-epic",
        "claimed-leaf",
        "claimed-program",
    ]


def test_execution_focus_defers_selected_footprints_and_respects_integer_resource_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(frontier_report.RESOURCE_POLICY["schema-lane"], "max_parallel", 2)
    issues = [
        _issue("devtools-first", priority=0, design="devtools/frontier_report.py", metadata={"frontier": "active"}),
        _issue("devtools-second", priority=1, design="devtools/frontier_report.py", metadata={"frontier": "active"}),
        _issue("schema-first", priority=2, design="migration 010_first.sql", metadata={"frontier": "active"}),
        _issue("schema-second", priority=3, design="migration 011_second.sql", metadata={"frontier": "active"}),
        _issue("schema-third", priority=4, design="migration 012_third.sql", metadata={"frontier": "active"}),
        _issue(
            "schema-unnumbered",
            priority=5,
            design="edit canonical index DDL and bump INDEX_SCHEMA_VERSION",
            metadata={"frontier": "active"},
        ),
    ]

    report = frontier_report.build_report(issues, issues, repo=Path("/repo"))

    focus = report["execution_focus"]
    assert [item["id"] for item in focus["focus"]] == ["devtools-first", "schema-first", "schema-second"]
    deferred = {item["id"]: item for item in focus["deferred"]}
    assert deferred["devtools-second"]["reason"] == "footprint conflict with selected focus: devtools-first"
    assert deferred["schema-third"]["reason"] == "schema-lane occupied by selected focus: schema-first, schema-second"
    assert deferred["schema-unnumbered"]["reason"] == "footprint is ambiguous; confirm ownership before parallel focus"


def test_execution_focus_defers_ambiguous_footprints_and_mixed_resource_work() -> None:
    issues = [
        _issue("live-claim", status="in_progress", design="polylogue/daemon/live.py", labels=["resource:live-state"]),
        _issue(
            "mixed-candidate",
            design="migration 010_mixed.sql",
            labels=["resource:live-state"],
            metadata={"frontier": "active"},
        ),
        _issue("ambiguous-candidate", metadata={"frontier": "active"}),
    ]

    report = frontier_report.build_report(
        issues,
        [issues[1], issues[2]],
        repo=Path("/repo"),
    )

    deferred = {item["id"]: item for item in report["execution_focus"]["deferred"]}
    assert deferred["mixed-candidate"]["resource_classes"] == ["schema-lane", "live-state"]
    assert deferred["mixed-candidate"]["reason"] == "live-state occupied by claim(s): live-claim"
    assert (
        deferred["ambiguous-candidate"]["reason"] == "footprint is ambiguous; confirm ownership before parallel focus"
    )


def test_execution_focus_defers_candidates_when_an_active_claim_has_ambiguous_footprint() -> None:
    issues = [
        _issue("ambiguous-claim", status="in_progress"),
        _issue("candidate", design="devtools/frontier_report.py", metadata={"frontier": "active"}),
    ]

    report = frontier_report.build_report(issues, [issues[1]], repo=Path("/repo"))

    assert report["execution_focus"]["focus"] == []
    assert report["execution_focus"]["deferred"][0]["reason"] == (
        "active claim footprint is ambiguous; confirm ownership before parallel focus: ambiguous-claim"
    )


def test_active_set_rows_expose_blockers_readiness_and_program() -> None:
    issues = [
        _issue("blocker", status="open", design="devtools/blocker.py"),
        _issue(
            "active",
            design="devtools/frontier_report.py",
            dependencies=[{"type": "blocks", "depends_on_id": "blocker"}],
            metadata={"frontier": "active", "frontier_program_ref": "program"},
        ),
    ]

    report = frontier_report.build_report(issues, [], repo=Path("/repo"))

    assert report["active_set_rows"] == [
        {
            "id": "active",
            "title": "title for active",
            "status": "open",
            "priority": 2,
            "dependency_ready": False,
            "blocked_by": ["blocker"],
            "frontier_program_ref": "program",
        }
    ]


def test_active_set_rows_preserve_dangling_blockers() -> None:
    issue = _issue(
        "active",
        design="devtools/frontier_report.py",
        dependencies=[{"type": "blocks", "depends_on_id": "missing"}],
        metadata={"frontier": "active"},
    )

    report = frontier_report.build_report([issue], [], repo=Path("/repo"))

    assert report["active_set_rows"][0]["blocked_by"] == ["missing (missing)"]


def test_execution_focus_counts_only_dependents_the_candidate_unblocks() -> None:
    issues = [
        _issue("blocker-a", design="devtools/frontier_report.py", metadata={"frontier": "active"}),
        _issue("blocker-b", design="devtools/verify_bead_graph.py", metadata={"frontier": "active"}),
        _issue(
            "single-dependent",
            dependencies=[{"type": "blocks", "depends_on_id": "blocker-a"}],
        ),
        _issue(
            "multiply-blocked",
            dependencies=[
                {"type": "blocks", "depends_on_id": "blocker-a"},
                {"type": "blocks", "depends_on_id": "blocker-b"},
            ],
        ),
        _issue(
            "transitive-dependent",
            dependencies=[{"type": "blocks", "depends_on_id": "single-dependent"}],
        ),
    ]

    report = frontier_report.build_report(issues, issues[:2], repo=Path("/repo"))
    candidates = {item["id"]: item for item in report["execution_focus"]["candidates"]}

    assert candidates["blocker-a"]["critical_path_leverage"] == 2
    assert candidates["blocker-b"]["critical_path_leverage"] == 0


def test_build_report_rejects_malformed_live_records() -> None:
    with pytest.raises(RuntimeError, match="bd list record 0 has no non-empty string id"):
        frontier_report.build_report([{"id": ""}], [], repo=Path("/repo"))


def test_build_report_rejects_duplicate_live_ids() -> None:
    with pytest.raises(RuntimeError, match="bd list record 1 duplicates id 'same'"):
        frontier_report.build_report([{"id": "same"}, {"id": "same"}], [], repo=Path("/repo"))


def test_render_markdown_includes_counts_focus_and_deferrals() -> None:
    rendered = frontier_report._render_markdown(
        {
            "repo": "/repo",
            "counts": {
                "ambition": 3,
                "active_set": 1,
                "claims": 1,
                "dependency_ready": 2,
                "execution_focus": 1,
                "deferred": 1,
            },
            "execution_focus": {
                "focus": [{"id": "focus", "priority": 1, "critical_path_leverage": 2, "title": "Focus title"}],
                "deferred": [{"id": "wait", "priority": 2, "reason": "schema-lane occupied by claim(s): claim"}],
            },
            "active_set_rows": [
                {
                    "id": "active",
                    "title": "Active title",
                    "priority": 1,
                    "dependency_ready": True,
                    "blocked_by": [],
                    "frontier_program_ref": "program",
                }
            ],
            "ambition": [
                {
                    "id": "ambition",
                    "priority": 3,
                    "status": "open",
                    "issue_type": "task",
                    "frontier_program_ref": "program",
                    "horizons": ["horizon:frontier"],
                    "title": "Ambition title",
                }
            ],
        }
    )

    assert rendered == (
        "# Execution Focus\n\n"
        "repo: `/repo`\n"
        "counts: ambition=3 active_set=1 claims=1 dependency_ready=2 execution_focus=1 deferred=1\n\n"
        "## Active Set\n\n"
        "- `active` P1 ready=True blocked_by=none program=program Active title\n\n"
        "## Focus\n\n"
        "- `focus` P1 leverage=2 Focus title\n\n"
        "## Deferred\n\n"
        "- `wait` P2 schema-lane occupied by claim(s): claim\n\n"
        "## Full Ambition\n\n"
        "- `ambition` P3 open task program=program horizons=horizon:frontier Ambition title"
    )


def test_main_uses_unbounded_live_surfaces_and_emits_complete_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    issues: list[Any] = [
        _issue("a", design="polylogue/core/a.py", metadata={"frontier": "active"}),
        _issue("b", design="polylogue/core/b.py"),
    ]
    ready: list[Any] = [_issue("a", design="polylogue/core/a.py")]

    def fake_run(_repo: Path, args: list[str]) -> list[object]:
        calls.append(args)
        if args[0] == "list":
            return issues
        if args[0] == "ready":
            return ready
        raise AssertionError(f"unexpected bd arguments: {args}")

    monkeypatch.setattr(frontier_report, "_run_bd", fake_run)

    assert frontier_report.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [["list", "--all"], ["ready"]]
    assert [item["id"] for item in payload["execution_focus"]["candidates"]] == ["a"]
    assert [item["id"] for item in payload["ambition"]] == ["a", "b"]
    assert payload["counts"]["dependency_ready"] == 1
    assert payload["execution_focus"]["resource_policy"] == frontier_report.RESOURCE_POLICY
