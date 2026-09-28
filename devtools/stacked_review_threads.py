"""Fail a master-bound PR while a PR stacked into its head has open review threads.

Branch protection requires conversation resolution on ``master`` only. A PR
merged into another PR's head branch is never gated, yet its code reaches
``master`` through the parent. This check walks every merged PR whose base is
the evaluated PR's head branch, recursively (a PR merged into a branch that was
itself merged into the head), and reports each unresolved review thread.

Every PR ever merged into the root's head branch counts, and so does every PR
merged into a stacked PR's head branch before that stacked PR itself merged. A
PR merged into a child after the child merged never reached the root. No other
exclusion is attempted: merge-commit ancestry is erased by rebases and squash
merges, so a PR merged into an earlier branch of the same name fails the gate
too, named like any other, and resolving its threads clears it.

Stdlib only, so the CircleCI ``stacked-review-threads`` job runs it without
syncing the project. The GitHub token comes from ``GH_TOKEN`` or
``GITHUB_TOKEN``, else from ``gh auth token``.

Usage::

    python -m devtools.stacked_review_threads --repo OWNER/NAME [--pr N]

The exit status is 1 when any evaluated PR has stacked unresolved threads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

GRAPHQL_URL = "https://api.github.com/graphql"

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

_OPEN_ROOTS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: "master", first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { ...Root }
    }
  }
}
"""

_ONE_ROOT_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { ...Root }
  }
}
"""

_ROOT_FRAGMENT = """
fragment Root on PullRequest {
  number url state baseRefName headRefName isCrossRepository
}
"""

_CHILDREN_QUERY = """
query($owner: String!, $name: String!, $branch: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, baseRefName: $branch, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number url headRefName mergedAt
      }
    }
  }
}
"""

_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { isResolved comments(first: 1) { nodes { url } } }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class RootPullRequest:
    number: int
    url: str
    head_ref: str
    cross_repository: bool


@dataclass(frozen=True)
class StackedPullRequest:
    number: int
    url: str
    unresolved_thread_urls: tuple[str, ...]
    parent_number: int


@dataclass
class Verdict:
    root: RootPullRequest
    offenders: list[StackedPullRequest] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "failure" if self.offenders else "success"

    def report_lines(self) -> list[str]:
        lines = [f"#{self.root.number} {self.root.url}: {self.state}"]
        for pr in self.offenders:
            lines.append(f"  stacked #{pr.number} {pr.url} (merged into #{pr.parent_number})")
            lines.extend(f"    unresolved: {url}" for url in pr.unresolved_thread_urls)
        return lines


def _token() -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        completed = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("no GitHub token: set GH_TOKEN or GITHUB_TOKEN, or log in with gh") from exc
    return completed.stdout.strip()


def github_transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Run one GraphQL request against the GitHub API."""
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {_token()}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as reply:
        response: dict[str, Any] = json.load(reply)
    if response.get("errors"):
        raise RuntimeError(f"GraphQL errors: {response['errors']}")
    return response


class StackedThreadGate:
    def __init__(self, transport: Transport, owner: str, name: str) -> None:
        self._transport = transport
        self._repo = {"owner": owner, "name": name}

    def _query(self, query: str, **variables: Any) -> dict[str, Any]:
        response = self._transport(query, {**self._repo, **variables})
        repository: dict[str, Any] = response["data"]["repository"]
        return repository

    def open_roots(self) -> Iterator[RootPullRequest]:
        cursor: str | None = None
        while True:
            page = self._query(_OPEN_ROOTS_QUERY + _ROOT_FRAGMENT, cursor=cursor)["pullRequests"]
            for node in page["nodes"]:
                yield _parse_root(node)
            if not page["pageInfo"]["hasNextPage"]:
                return
            cursor = page["pageInfo"]["endCursor"]

    def root(self, number: int) -> RootPullRequest | None:
        node = self._query(_ONE_ROOT_QUERY + _ROOT_FRAGMENT, number=number)["pullRequest"]
        if node is None or node["state"] != "OPEN" or node["baseRefName"] != "master":
            return None
        return _parse_root(node)

    def evaluate(self, root: RootPullRequest) -> Verdict:
        verdict = Verdict(root)
        # (branch, parent number, parent merged_at or None for the root). Expansion
        # is keyed by parent PR, not branch name: a reused branch name is a
        # different lifetime with its own merge bound.
        pending: list[tuple[str, int, str | None]] = [(root.head_ref, root.number, None)]
        expanded: set[int] = set()
        while pending:
            branch, parent_number, parent_merged = pending.pop()
            if parent_number in expanded or branch == "master":
                continue
            expanded.add(parent_number)
            for child in self._merged_into(branch):
                if child["number"] in expanded or child["number"] == root.number:
                    continue
                if parent_merged is not None and child["mergedAt"] > parent_merged:
                    continue
                unresolved = self._unresolved_threads(child["number"])
                if unresolved and all(pr.number != child["number"] for pr in verdict.offenders):
                    verdict.offenders.append(
                        StackedPullRequest(
                            number=child["number"],
                            url=child["url"],
                            unresolved_thread_urls=tuple(unresolved),
                            parent_number=parent_number,
                        )
                    )
                pending.append((child["headRefName"], child["number"], child["mergedAt"]))
        verdict.offenders.sort(key=lambda pr: pr.number)
        return verdict

    def _merged_into(self, branch: str) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            page = self._query(_CHILDREN_QUERY, branch=branch, cursor=cursor)["pullRequests"]
            yield from page["nodes"]
            if not page["pageInfo"]["hasNextPage"]:
                return
            cursor = page["pageInfo"]["endCursor"]

    def _unresolved_threads(self, number: int) -> list[str]:
        urls: list[str] = []
        cursor: str | None = None
        while True:
            threads = self._query(_THREADS_QUERY, number=number, cursor=cursor)["pullRequest"]["reviewThreads"]
            urls.extend(_unresolved_urls(threads["nodes"]))
            if not threads["pageInfo"]["hasNextPage"]:
                return urls
            cursor = threads["pageInfo"]["endCursor"]


def _parse_root(node: dict[str, Any]) -> RootPullRequest:
    return RootPullRequest(
        number=node["number"],
        url=node["url"],
        head_ref=node["headRefName"],
        cross_repository=node["isCrossRepository"],
    )


def _unresolved_urls(threads: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for thread in threads:
        if thread["isResolved"]:
            continue
        comments = thread["comments"]["nodes"]
        urls.append(comments[0]["url"] if comments else "(thread without comments)")
    return urls


def main(argv: list[str] | None = None, transport: Transport = github_transport) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--pr", type=int, help="evaluate one PR instead of every open PR targeting master")
    args = parser.parse_args(argv)
    owner, name = args.repo.split("/", 1)
    gate = StackedThreadGate(transport, owner, name)
    if args.pr is not None:
        single = gate.root(args.pr)
        roots = [single] if single is not None else []
    else:
        roots = list(gate.open_roots())
    verdicts = [gate.evaluate(root) for root in roots if not root.cross_repository]

    lines: list[str] = []
    for verdict in verdicts:
        lines.extend(verdict.report_lines())
    print("\n".join(lines) if lines else "No open PR targeting master to evaluate")
    return 1 if any(verdict.offenders for verdict in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
