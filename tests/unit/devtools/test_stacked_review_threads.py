"""The stacked-review-threads gate over synthetic GraphQL responses.

Each test drives the production traversal through a fake transport that
answers the gate's GraphQL requests with response-shaped fixtures. A gate that
stopped descending past one level, trusted a reused branch name, counted a PR
merged into a child after the child merged, or dropped a paginated thread
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


def _merged(
    number: int,
    head: str,
    *,
    created: str,
    merged: str,
    oid: str,
    threads: tuple[bool, ...] = (),
    more_threads: bool = False,
) -> dict[str, Any]:
    return {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "headRefName": head,
        "createdAt": created,
        "mergedAt": merged,
        "mergeCommit": {"oid": oid},
        "_threads": _threads(*threads, number=number, has_next=more_threads),
    }


def _root(number: int, head: str, *, created: str) -> dict[str, Any]:
    return {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "state": "OPEN",
        "baseRefName": "master",
        "headRefName": head,
        "createdAt": created,
        "isCrossRepository": False,
    }


class FakeGitHub:
    """Answers the gate's queries from branch -> merged PRs and PR -> commits tables."""

    def __init__(
        self,
        roots: list[dict[str, Any]],
        merged_into: dict[str, list[dict[str, Any]]],
        commits: dict[int, list[str]],
        extra_threads: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.roots = roots
        self.merged_into = merged_into
        self.commits = commits
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
                first = next(
                    pr for prs in self.merged_into.values() for pr in prs if pr["number"] == variables["number"]
                )
                threads = first["_threads"]
            else:
                threads = self.extra_threads[variables["number"]]
            repo = {"pullRequest": {"reviewThreads": threads}}
        elif "commits(first: 100" in query:
            oids = self.commits.get(variables["number"], [])
            repo = {"pullRequest": {"commits": {"pageInfo": page, "nodes": [{"commit": {"oid": o}} for o in oids]}}}
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
        roots=[_root(10, "feat", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    11,
                    "feat2",
                    created="2026-01-02T00:00:00Z",
                    merged="2026-01-05T00:00:00Z",
                    oid="m11",
                    threads=(True,),
                )
            ],
            "feat2": [
                _merged(
                    12,
                    "feat3",
                    created="2026-01-03T00:00:00Z",
                    merged="2026-01-04T00:00:00Z",
                    oid="m12",
                    threads=(False, True, False),
                )
            ],
        },
        commits={10: ["c1", "m12", "m11"]},
    )
    assert _offenders(fake, 10) == {
        12: ("https://example.test/pull/12#discussion_r0", "https://example.test/pull/12#discussion_r2"),
    }


def test_all_threads_resolved_passes() -> None:
    fake = FakeGitHub(
        roots=[_root(10, "feat", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    11,
                    "feat2",
                    created="2026-01-02T00:00:00Z",
                    merged="2026-01-03T00:00:00Z",
                    oid="m11",
                    threads=(True, True),
                )
            ]
        },
        commits={10: ["m11"]},
    )
    assert _offenders(fake, 10) == {}


def test_reused_branch_name_merged_before_the_root_opened_is_ignored() -> None:
    fake = FakeGitHub(
        roots=[_root(20, "feat", created="2026-03-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    5, "old", created="2025-01-01T00:00:00Z", merged="2025-01-02T00:00:00Z", oid="m5", threads=(False,)
                )
            ]
        },
        commits={20: ["c1"]},
    )
    assert _offenders(fake, 20) == {}


def test_merged_after_root_opened_counts_even_when_the_root_was_rebased() -> None:
    fake = FakeGitHub(
        roots=[_root(20, "feat", created="2026-03-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    21,
                    "feat2",
                    created="2026-03-02T00:00:00Z",
                    merged="2026-03-03T00:00:00Z",
                    oid="gone",
                    threads=(False,),
                )
            ]
        },
        commits={20: ["rebased"]},
    )
    assert set(_offenders(fake, 20)) == {21}


def test_pr_merged_into_a_child_after_the_child_merged_does_not_reach_the_root() -> None:
    fake = FakeGitHub(
        roots=[_root(30, "feat", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [_merged(31, "feat2", created="2026-01-02T00:00:00Z", merged="2026-01-04T00:00:00Z", oid="m31")],
            "feat2": [
                _merged(
                    32,
                    "feat3",
                    created="2026-01-03T00:00:00Z",
                    merged="2026-01-06T00:00:00Z",
                    oid="m32",
                    threads=(False,),
                )
            ],
        },
        commits={30: ["m31"]},
    )
    assert _offenders(fake, 30) == {}


def test_threads_past_the_first_page_are_counted() -> None:
    fake = FakeGitHub(
        roots=[_root(40, "feat", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    41,
                    "feat2",
                    created="2026-01-02T00:00:00Z",
                    merged="2026-01-03T00:00:00Z",
                    oid="m41",
                    threads=(True,),
                    more_threads=True,
                )
            ]
        },
        commits={40: ["m41"]},
        extra_threads={41: _threads(False, number=141)},
    )
    assert _offenders(fake, 40) == {41: ("https://example.test/pull/141#discussion_r0",)}


def test_branch_cycle_terminates() -> None:
    fake = FakeGitHub(
        roots=[_root(50, "a", created="2026-01-01T00:00:00Z")],
        merged_into={
            "a": [_merged(51, "b", created="2026-01-02T00:00:00Z", merged="2026-01-03T00:00:00Z", oid="m51")],
            "b": [
                _merged(
                    52, "a", created="2026-01-02T00:00:00Z", merged="2026-01-02T12:00:00Z", oid="m52", threads=(False,)
                )
            ],
        },
        commits={50: ["m51", "m52"]},
    )
    assert set(_offenders(fake, 50)) == {52}
    assert fake.queried_branches == ["a", "b", "a"]


def test_each_lifetime_of_a_reused_child_branch_is_walked_with_its_own_window() -> None:
    fake = FakeGitHub(
        roots=[_root(80, "feat", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(81, "child", created="2026-01-02T00:00:00Z", merged="2026-01-04T00:00:00Z", oid="x81"),
                _merged(82, "child", created="2026-01-05T00:00:00Z", merged="2026-01-07T00:00:00Z", oid="x82"),
            ],
            "child": [
                _merged(
                    83, "g1", created="2026-01-02T00:00:00Z", merged="2026-01-03T00:00:00Z", oid="x83", threads=(False,)
                ),
                _merged(
                    84, "g2", created="2026-01-05T00:00:00Z", merged="2026-01-06T00:00:00Z", oid="x84", threads=(False,)
                ),
            ],
        },
        commits={80: ["rebased"]},
    )
    assert set(_offenders(fake, 80)) == {83, 84}


def test_child_merged_before_a_squashed_parent_opened_is_found_through_the_parent_commits() -> None:
    fake = FakeGitHub(
        roots=[_root(100, "feature-a", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feature-a": [
                _merged(
                    101, "feature-b", created="2026-01-05T00:00:00Z", merged="2026-01-06T00:00:00Z", oid="squash101"
                )
            ],
            "feature-b": [
                _merged(
                    102,
                    "feature-c",
                    created="2026-01-02T00:00:00Z",
                    merged="2026-01-03T00:00:00Z",
                    oid="m102",
                    threads=(False,),
                )
            ],
        },
        commits={100: ["squash101"], 101: ["b1", "m102"]},
    )
    assert set(_offenders(fake, 100)) == {102}


def test_threads_are_read_only_for_prs_in_the_stack() -> None:
    fake = FakeGitHub(
        roots=[_root(90, "feat", created="2026-03-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    91,
                    "old",
                    created="2025-01-01T00:00:00Z",
                    merged="2025-01-02T00:00:00Z",
                    oid="m91",
                    threads=(False,),
                ),
                _merged(92, "new", created="2026-03-02T00:00:00Z", merged="2026-03-03T00:00:00Z", oid="m92"),
            ]
        },
        commits={90: ["m92"]},
    )
    assert _offenders(fake, 90) == {}
    assert fake.thread_queries == [92]


def test_cli_exit_status_reflects_offenders(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeGitHub(
        roots=[_root(60, "feat", created="2026-01-01T00:00:00Z"), _root(70, "clean", created="2026-01-01T00:00:00Z")],
        merged_into={
            "feat": [
                _merged(
                    61,
                    "feat2",
                    created="2026-01-02T00:00:00Z",
                    merged="2026-01-03T00:00:00Z",
                    oid="m61",
                    threads=(False,),
                )
            ]
        },
        commits={60: ["m61"]},
    )
    assert main(["--repo", "owner/repo"], transport=fake) == 1
    out = capsys.readouterr().out
    assert "https://example.test/pull/61#discussion_r0" in out
    assert "#70 https://example.test/pull/70: success" in out
    assert main(["--repo", "owner/repo", "--pr", "70"], transport=fake) == 0
