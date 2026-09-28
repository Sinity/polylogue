"""The stacked-review-threads gate over synthetic GraphQL responses.

Each test drives the production traversal through a fake transport that
answers the gate's GraphQL requests with response-shaped fixtures. A gate that
stopped descending past one level, walked a reused child branch name only
once, counted a PR merged into a child after the child merged, excused a PR
merged into the head before the root opened, or dropped a paginated thread
would fail here.
"""

from __future__ import annotations

from typing import Any

import pytest

from devtools.stacked_review_threads import StackedThreadGate, main


def _threads(*resolved: bool, number: int, has_next: bool = False) -> dict[str, Any]:
    return {
        "pageInfo": {"hasNextPage": has_next, "endCursor": "t1" if has_next else None},
        "nodes": [
            {
                "isResolved": is_resolved,
                "comments": {"nodes": [{"url": f"https://example.test/pull/{number}#discussion_r{index}"}]},
            }
            for index, is_resolved in enumerate(resolved)
        ],
    }


def _merged(number: int, head: str, merged: str, *threads: bool, more_threads: bool = False) -> dict[str, Any]:
    return {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "headRefName": head,
        "mergedAt": merged,
        "_threads": _threads(*threads, number=number, has_next=more_threads),
    }


def _root(number: int, head: str) -> dict[str, Any]:
    return {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "state": "OPEN",
        "baseRefName": "master",
        "headRefName": head,
        "isCrossRepository": False,
    }


class FakeGitHub:
    """Answers the gate's queries from a branch -> merged PRs table."""

    def __init__(
        self,
        roots: list[dict[str, Any]],
        merged_into: dict[str, list[dict[str, Any]]],
        extra_threads: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.roots = roots
        self.merged_into = merged_into
        self.extra_threads = extra_threads or {}
        self.queried_branches: list[str] = []
        self.thread_queries: list[int] = []

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        page = {"hasNextPage": False, "endCursor": None}
        if "states: OPEN" in query:
            repo: dict[str, Any] = {"pullRequests": {"pageInfo": page, "nodes": self.roots}}
        elif "pullRequest(number: $number) { ...Root }" in query:
            repo = {"pullRequest": next(r for r in self.roots if r["number"] == variables["number"])}
        elif "states: MERGED" in query:
            self.queried_branches.append(variables["branch"])
            nodes = [
                {k: v for k, v in pr.items() if k != "_threads"} for pr in self.merged_into.get(variables["branch"], [])
            ]
            repo = {"pullRequests": {"pageInfo": page, "nodes": nodes}}
        elif "reviewThreads(first: 100, after: $cursor)" in query:
            self.thread_queries.append(variables["number"])
            if variables["cursor"] is None:
                prs = [pr for prs in self.merged_into.values() for pr in prs]
                threads = next(pr for pr in prs if pr["number"] == variables["number"])["_threads"]
            else:
                threads = self.extra_threads[variables["number"]]
            repo = {"pullRequest": {"reviewThreads": threads}}
        else:
            raise AssertionError(f"unexpected query: {query}")
        return {"data": {"repository": repo}}


def _offenders(fake: FakeGitHub, number: int) -> dict[int, tuple[str, ...]]:
    gate = StackedThreadGate(fake, "owner", "repo")
    root = gate.root(number)
    assert root is not None
    return {pr.number: pr.unresolved_thread_urls for pr in gate.evaluate(root).offenders}


def test_unresolved_threads_two_levels_down_fail_the_root() -> None:
    fake = FakeGitHub(
        roots=[_root(10, "feat")],
        merged_into={
            "feat": [_merged(11, "feat2", "2026-01-05T00:00:00Z", True)],
            "feat2": [_merged(12, "feat3", "2026-01-04T00:00:00Z", False, True, False)],
        },
    )
    assert _offenders(fake, 10) == {
        12: ("https://example.test/pull/12#discussion_r0", "https://example.test/pull/12#discussion_r2"),
    }


def test_all_threads_resolved_passes() -> None:
    fake = FakeGitHub(
        roots=[_root(10, "feat")],
        merged_into={"feat": [_merged(11, "feat2", "2026-01-03T00:00:00Z", True, True)]},
    )
    assert _offenders(fake, 10) == {}


def test_pr_merged_into_the_head_before_the_root_opened_still_counts() -> None:
    # Rebases and squash merges erase ancestry, so an early merge is not excused.
    fake = FakeGitHub(
        roots=[_root(20, "feat")],
        merged_into={"feat": [_merged(5, "old", "2025-01-02T00:00:00Z", False)]},
    )
    assert set(_offenders(fake, 20)) == {5}


def test_pr_merged_into_a_child_before_the_child_opened_counts() -> None:
    fake = FakeGitHub(
        roots=[_root(100, "feature-a")],
        merged_into={
            "feature-a": [_merged(101, "feature-b", "2026-01-06T00:00:00Z")],
            "feature-b": [_merged(102, "feature-c", "2026-01-03T00:00:00Z", False)],
        },
    )
    assert set(_offenders(fake, 100)) == {102}


def test_pr_merged_into_a_child_after_the_child_merged_does_not_reach_the_root() -> None:
    fake = FakeGitHub(
        roots=[_root(30, "feat")],
        merged_into={
            "feat": [_merged(31, "feat2", "2026-01-04T00:00:00Z")],
            "feat2": [_merged(32, "feat3", "2026-01-06T00:00:00Z", False)],
        },
    )
    assert _offenders(fake, 30) == {}
    assert 32 not in fake.thread_queries


def test_each_lifetime_of_a_reused_child_branch_is_walked_with_its_own_bound() -> None:
    fake = FakeGitHub(
        roots=[_root(80, "feat")],
        merged_into={
            "feat": [
                _merged(81, "child", "2026-01-04T00:00:00Z"),
                _merged(82, "child", "2026-01-07T00:00:00Z"),
            ],
            "child": [
                _merged(83, "g1", "2026-01-03T00:00:00Z", False),
                _merged(84, "g2", "2026-01-06T00:00:00Z", False),
            ],
        },
    )
    assert set(_offenders(fake, 80)) == {83, 84}
    assert fake.queried_branches.count("child") == 2


def test_threads_past_the_first_page_are_counted() -> None:
    fake = FakeGitHub(
        roots=[_root(40, "feat")],
        merged_into={"feat": [_merged(41, "feat2", "2026-01-03T00:00:00Z", True, more_threads=True)]},
        extra_threads={41: _threads(False, number=141)},
    )
    assert _offenders(fake, 40) == {41: ("https://example.test/pull/141#discussion_r0",)}


def test_branch_cycle_terminates() -> None:
    fake = FakeGitHub(
        roots=[_root(50, "a")],
        merged_into={
            "a": [_merged(51, "b", "2026-01-03T00:00:00Z")],
            "b": [_merged(52, "a", "2026-01-02T12:00:00Z", False)],
        },
    )
    assert set(_offenders(fake, 50)) == {52}
    assert fake.queried_branches == ["a", "b", "a"]


def test_cli_exit_status_reflects_offenders(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeGitHub(
        roots=[_root(60, "feat"), _root(70, "clean")],
        merged_into={"feat": [_merged(61, "feat2", "2026-01-03T00:00:00Z", False)]},
    )
    assert main(["--repo", "owner/repo"], transport=fake) == 1
    out = capsys.readouterr().out
    assert "https://example.test/pull/61#discussion_r0" in out
    assert "#70 https://example.test/pull/70: success" in out
    assert main(["--repo", "owner/repo", "--pr", "70"], transport=fake) == 0
