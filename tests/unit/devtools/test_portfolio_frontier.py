from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from devtools.portfolio_frontier import ActivePolicy, PortfolioPolicyError, build_views, enumerate_complete


def row(bead_id: str, *, status: str = "open", issue_type: str = "task", **meta: Any) -> dict[str, Any]:
    return {
        "id": bead_id,
        "status": status,
        "issue_type": issue_type,
        "priority": 1,
        "design": "design",
        "acceptance_criteria": "AC",
        "labels": ["area:devtools", "horizon:frontier"],
        "metadata": {"frontier": "active", "frontier_program_ref": "program", **meta},
        "dependencies": [],
    }


def program() -> dict[str, Any]:
    return {
        "id": "program",
        "status": "open",
        "issue_type": "epic",
        "labels": ["horizon:frontier"],
        "metadata": {"frontier_program": "active"},
    }


def test_active_is_broad_and_focus_is_derived_without_count_cap() -> None:
    leaves = [row(f"leaf-{index}") for index in range(55)]
    report = build_views([program(), *leaves], policy=ActivePolicy(focus_limit=3))
    assert report["active"]["count"] == 55  # anti-vacuity: a 16/50 cap must not hide rows
    assert len(report["active"]["leaves"]) == 55
    assert len(report["execution_focus"]) == 3
    assert report["diagnostics"]


def test_focus_respects_conflicts_and_prioritizes_unlock_leverage() -> None:
    blocker = row("blocker", conflict_keys="shared")
    blocked = row("blocked", conflict_keys="other")
    blocked["dependencies"] = [{"depends_on_id": "blocker", "type": "blocks"}]
    competing = row("competing", conflict_keys="shared", priority=2)
    report = build_views([program(), blocker, blocked, competing], policy=ActivePolicy(focus_limit=3))
    assert [item["id"] for item in report["execution_focus"]] == ["blocker"]


def test_invalid_program_epic_and_missing_contract_fail_with_actionable_errors() -> None:
    bad_epic = row("epic-leaf", issue_type="epic")
    bad = row("bad", frontier_program_ref="missing", design="", labels=["horizon:frontier"])
    with pytest.raises(
        PortfolioPolicyError, match="active epics.*invalid active program ref|invalid active program ref.*active epics"
    ):
        build_views([program(), bad_epic, bad])


def test_blockers_and_parent_integrity_are_reported() -> None:
    parent = row("parent", frontier="active")
    child = row("child")
    child["dependencies"] = [{"depends_on_id": "unknown", "type": "blocks"}]
    child["parent"] = "missing-parent"
    child["metadata"].pop("parent", None)
    with pytest.raises(PortfolioPolicyError, match="missing canonical parent.*missing dependency"):
        build_views([program(), parent, child])


def test_complete_pages_reject_repeating_or_truncated_streams() -> None:
    def repeating(cursor: str | None, size: int) -> tuple[list[dict[str, str]], str]:
        return ([{"id": "a"}], "same")

    with pytest.raises(PortfolioPolicyError, match="repeating|non-progressing"):
        enumerate_complete(repeating)

    def oversized(cursor: str | None, size: int) -> tuple[list[dict[str, str]], None]:
        return ([{"id": str(index)} for index in range(size + 1)], None)

    with pytest.raises(PortfolioPolicyError, match="exceeded requested bound"):
        enumerate_complete(oversized, page_size=2)


def test_incomplete_sync_receipt_fails_before_policy() -> None:
    with pytest.raises(PortfolioPolicyError, match="planning-surface-corrupt"):
        build_views([program(), row("leaf")], receipt={"complete": False})


def test_full_ambition_remains_queryable_by_horizon() -> None:
    mid = row("mid", frontier="", frontier_program_ref=None)
    mid["labels"] = ["area:devtools", "horizon:mid"]
    mid["metadata"] = {}
    report = build_views([program(), row("frontier"), mid])
    assert report["ambition"]["frontier"] == ["frontier", "program"]
    assert report["ambition"]["mid"] == ["mid"]


def test_views_keep_priority_readiness_and_claims_as_separate_dimensions() -> None:
    ready = row("ready", status="in_progress", claimed_at="2026-08-27T10:00:00Z", owner="alice")
    ready["metadata"]["critical_path"] = True
    blocked = row("blocked")
    blocked["dependencies"] = [{"depends_on_id": "ready", "type": "blocks"}]
    report = build_views([program(), ready, blocked], now=datetime(2026, 8, 27, tzinfo=UTC))
    assert report["priority"]["1"] == ["blocked", "ready"]
    assert report["active"]["readiness"] == {
        "ready": [],
        "blocked-near-next": ["blocked"],
        "in_progress": ["ready"],
    }
    assert report["active"]["claims"]["claimed"] == ["ready"]
    assert report["execution_focus"][0]["id"] == "ready"


def test_receipt_row_count_and_stale_claims_fail_closed() -> None:
    with pytest.raises(PortfolioPolicyError, match="receipt says"):
        build_views(
            [program(), row("leaf")],
            receipt={"schema": 1, "complete": True, "source_fingerprint": "x", "rows": 99},
        )
    with pytest.raises(PortfolioPolicyError, match="stale claim"):
        build_views(
            [program(), row("leaf", claimed_at="2026-08-01T00:00:00Z", owner="alice")],
            now=datetime(2026, 8, 27, tzinfo=UTC),
        )


def test_production_scale_input_is_not_truncated() -> None:
    report = build_views([program(), *(row(f"leaf-{i}") for i in range(1000))])
    assert report["active"]["count"] == 1000
    assert len(report["ambition"]["frontier"]) == 1001
