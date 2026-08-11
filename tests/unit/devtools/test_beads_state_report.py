"""Behavior tests for devtools.beads_state_report.

The report's contract: interpretation is computed, never fossilized.  These
tests pin the load-bearing computations -- temporal snapshot reconstruction,
conditional insight emission, parallel-frontier independence, and the health
queues that must survive presentation changes.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from devtools.beads_state_report import (
    Facts,
    compute_insights,
    json_payload,
    render,
)

NOW = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)


def _bead(
    bid: str,
    *,
    status: str = "open",
    priority: int = 2,
    itype: str = "task",
    created: str = "2026-07-01T00:00:00Z",
    closed: str | None = None,
    updated: str | None = None,
    title: str | None = None,
    notes: str = "",
    deps: list[dict[str, str]] | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_type": "issue",
        "id": bid,
        "title": title or f"title of {bid}",
        "status": status,
        "priority": priority,
        "issue_type": itype,
        "created_at": created,
        "closed_at": closed,
        "updated_at": updated or created,
        "notes": notes,
        "labels": labels or [],
        "dependencies": deps or [],
    }


def _dep(src: str, dst: str, kind: str, created: str = "2026-07-02T00:00:00Z") -> dict[str, str]:
    return {"issue_id": src, "depends_on_id": dst, "type": kind, "created_at": created}


def _facts(beads: list[dict[str, Any]], now: dt.datetime = NOW) -> Facts:
    issues = {b["id"]: b for b in beads}
    edges = [
        (d["issue_id"], d["depends_on_id"], d["type"], d.get("created_at", ""))
        for b in beads
        for d in b["dependencies"]
    ]
    return Facts(issues, edges, now)


# ---------------------------------------------------------------------------
# temporal reconstruction
# ---------------------------------------------------------------------------
class TestSnapshot:
    def test_open_and_closed_reconstructed_at_past_instant(self) -> None:
        facts = _facts(
            [
                _bead("a", created="2026-07-01T00:00:00Z"),
                _bead("b", created="2026-07-20T00:00:00Z"),
                _bead(
                    "c",
                    status="closed",
                    created="2026-07-01T00:00:00Z",
                    closed="2026-07-15T00:00:00Z",
                ),
            ]
        )
        # On 2026-07-10: a exists (open), b not created yet, c not yet closed.
        snap = facts.snapshot(dt.datetime(2026, 7, 10, tzinfo=dt.UTC))
        assert snap == {"total": 2, "open": 2, "closed": 0, "p0_open": 0, "blocked": 0, "ready": 2}
        # On 2026-07-25: all three exist, c closed.
        snap = facts.snapshot(dt.datetime(2026, 7, 25, tzinfo=dt.UTC))
        assert (snap["total"], snap["open"], snap["closed"]) == (3, 2, 1)

    def test_blocked_respects_edge_creation_time(self) -> None:
        facts = _facts(
            [
                _bead("blocker", created="2026-07-01T00:00:00Z"),
                _bead(
                    "blocked",
                    created="2026-07-01T00:00:00Z",
                    deps=[_dep("blocked", "blocker", "blocks", created="2026-07-10T00:00:00Z")],
                ),
            ]
        )
        before_edge = facts.snapshot(dt.datetime(2026, 7, 5, tzinfo=dt.UTC))
        after_edge = facts.snapshot(dt.datetime(2026, 7, 15, tzinfo=dt.UTC))
        assert before_edge["blocked"] == 0 and before_edge["ready"] == 2
        assert after_edge["blocked"] == 1 and after_edge["ready"] == 1

    def test_daily_series_spans_first_creation_to_now(self) -> None:
        facts = _facts([_bead("a", created="2026-07-30T00:00:00Z")])
        series = facts.daily_series()
        assert series[0][0] == "2026-07-30"
        assert series[-1][0] == "2026-08-01"
        assert all(open_n == 1 for _, open_n, _ in series)


# ---------------------------------------------------------------------------
# conditional insights
# ---------------------------------------------------------------------------
class TestInsights:
    def _insight_titles(self, facts: Facts) -> list[str]:
        snaps = {
            "now": facts.snapshot(NOW),
            "7d": facts.snapshot(NOW - dt.timedelta(days=7)),
            "14d": facts.snapshot(NOW - dt.timedelta(days=14)),
        }
        return [i.title for i in compute_insights(facts, {}, snaps)]

    def test_cycle_detection_flips_the_insight(self) -> None:
        acyclic = _facts(
            [
                _bead("a", deps=[_dep("a", "b", "blocks")]),
                _bead("b"),
            ]
        )
        assert any("No cycles" in t for t in self._insight_titles(acyclic))
        cyclic = _facts(
            [
                _bead("a", deps=[_dep("a", "b", "blocks")]),
                _bead("b", deps=[_dep("b", "a", "blocks")]),
            ]
        )
        titles = self._insight_titles(cyclic)
        assert any("cycle(s)" in t for t in titles)
        assert not any("No cycles" in t for t in titles)

    def test_priority_ladder_inversion_detected(self) -> None:
        # P0 lead 10d, P1 lead 1d -> lead inversion at P0->P1.
        inverted = _facts(
            [
                _bead("p0", priority=0, status="closed", created="2026-07-01T00:00:00Z", closed="2026-07-11T00:00:00Z"),
                _bead("p1", priority=1, status="closed", created="2026-07-01T00:00:00Z", closed="2026-07-02T00:00:00Z"),
                _bead("p1b", priority=1),
            ]
        )
        assert any("inverts" in t for t in self._insight_titles(inverted))

    def test_stale_claim_and_aged_urgent_emitted(self) -> None:
        facts = _facts(
            [
                _bead(
                    "zombie",
                    status="in_progress",
                    created="2026-07-01T00:00:00Z",
                    updated="2026-07-10T00:00:00Z",
                ),
                _bead("aged-p0", priority=0, created="2026-07-01T00:00:00Z"),
            ]
        )
        titles = self._insight_titles(facts)
        assert any("in-progress claim(s) untouched" in t for t in titles)
        assert any("P0/P1" in t for t in titles)
        assert facts.stale_claims and facts.stale_claims[0][0] == "zombie"
        assert facts.aged_urgent == ["aged-p0"]

    def test_dangling_reference_insight_names_target(self) -> None:
        facts = _facts([_bead("a", deps=[_dep("a", "nonexistent", "blocks")])])
        snaps = {"now": facts.snapshot(NOW), "7d": facts.snapshot(NOW), "14d": facts.snapshot(NOW)}
        insights = compute_insights(facts, {}, snaps)
        dangling = [i for i in insights if "do not exist" in i.title]
        assert dangling and "nonexistent" in dangling[0].body


# ---------------------------------------------------------------------------
# parallel frontier
# ---------------------------------------------------------------------------
def _epic_with_kids(eid: str, n_kids: int, cross_edge_to: str | None = None) -> list[dict[str, Any]]:
    beads = [_bead(eid, itype="epic")]
    for i in range(n_kids):
        kid = f"{eid}.{i}"
        deps = [_dep(kid, eid, "parent-child")]
        if cross_edge_to and i == 0:
            deps.append(_dep(kid, cross_edge_to, "blocks"))
        beads.append(_bead(kid, deps=deps))
    return beads


class TestParallelFrontier:
    def test_disjoint_epics_are_parallel(self) -> None:
        facts = _facts(_epic_with_kids("e1", 4) + _epic_with_kids("e2", 4))
        frontier = facts.parallel_frontier()
        assert {c["id"] for c in frontier} == {"e1", "e2"}

    def test_blocks_edge_between_subtrees_collapses_frontier(self) -> None:
        # e2's first child blocks on e1's first child -> shared component.
        facts = _facts(_epic_with_kids("e1", 4) + _epic_with_kids("e2", 4, cross_edge_to="e1.0"))
        frontier = facts.parallel_frontier()
        assert len(frontier) == 1


# ---------------------------------------------------------------------------
# render invariants
# ---------------------------------------------------------------------------
class TestRender:
    def test_health_queues_survive(self, tmp_path: Any) -> None:
        beads = [
            _bead("parent", deps=[]),
            _bead(
                "kid",
                status="closed",
                closed="2026-07-20T00:00:00Z",
                deps=[_dep("kid", "parent", "parent-child")],
            ),
            _bead("dangler", deps=[_dep("dangler", "20d.14", "blocks")]),
        ]
        facts = _facts(beads)
        assert facts.open_parent_all_closed == ["parent"]
        html = render(facts, tmp_path / "issues.jsonl", NOW, {})
        # graph-health queues preserved
        assert "open parent, all children closed" in html
        assert "dangling dependency references" in html
        assert "20d.14" in html
        # deferred status renders in the legend
        assert "deferred" in html

    def test_no_fossilized_epic_ids_in_generator_output_for_foreign_data(self, tmp_path: Any) -> None:
        # A population that never contained the historically hard-coded epics
        # must not mention them: interpretation is computed, not transcribed.
        facts = _facts(_epic_with_kids("e1", 9) + _epic_with_kids("e2", 9))
        html = render(facts, tmp_path / "issues.jsonl", NOW, {})
        assert "polylogue-9e5" not in html
        assert "polylogue-9l5" not in html

    def test_json_payload_contract(self) -> None:
        facts = _facts([_bead("a")])
        payload = json_payload(facts, {}, NOW)
        for key in ("snapshots", "velocity", "parallel_frontier", "insights", "generated", "total"):
            assert key in payload
        assert payload["snapshots"]["now"]["open"] == 1
        assert all({"sev", "title", "body", "ev"} <= set(i) for i in payload["insights"])
