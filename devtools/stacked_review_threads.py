"""Fail a master-bound PR while a PR stacked into its head has open review threads.

Branch protection requires conversation resolution on ``master`` only. A PR
merged into another PR's head branch is never gated, yet its code reaches
``master`` through the parent. This check walks every merged PR whose base is
the evaluated PR's head branch, recursively (a PR merged into a branch that was
itself merged into the head), and reports each unresolved review thread.

A merged PR belongs to the stack when its merge commit is in the root PR's
commit list. When it is not (the parent branch was rebased, or the branch name
was reused), the PR still counts if it merged while its parent's branch was the
live one: after the parent PR was opened and, below the root, before the parent
itself merged. A PR merged into a child after that child merged never reached
the root, so it is excluded.

Stdlib only: the workflow runs it without installing the project.

Usage::

    python -m devtools.stacked_review_threads --repo OWNER/NAME [--pr N] [--post-status]

Without ``--post-status`` the exit status is 1 when any evaluated PR has
stacked unresolved threads. With it, each verdict is posted as the
``stacked-review-threads`` commit status on the PR head and the exit status
reports only transport errors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

STATUS_CONTEXT = "stacked-review-threads"
STATUS_DESCRIPTION_LIMIT = 140

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
  number url state baseRefName headRefName headRefOid createdAt isCrossRepository
  commits(last: 1) {
    nodes { commit { status { contexts { context state description } } } }
  }
}
"""

_CHILDREN_QUERY = """
query($owner: String!, $name: String!, $branch: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, baseRefName: $branch, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number url headRefName createdAt mergedAt
        mergeCommit { oid }
        reviewThreads(first: 100) {
          pageInfo { hasNextPage endCursor }
          nodes { isResolved comments(first: 1) { nodes { url } } }
        }
      }
    }
  }
}
"""

_MORE_THREADS_QUERY = """
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

_COMMITS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { commit { oid } }
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
    head_oid: str
    created_at: str
    cross_repository: bool
    current_status: tuple[str, str] | None


@dataclass(frozen=True)
class StackedPullRequest:
    number: int
    url: str
    head_ref: str
    created_at: str
    merged_at: str
    merge_oid: str | None
    unresolved_thread_urls: tuple[str, ...]
    parent_number: int


@dataclass
class Verdict:
    root: RootPullRequest
    offenders: list[StackedPullRequest] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "failure" if self.offenders else "success"

    @property
    def description(self) -> str:
        if not self.offenders:
            return "No unresolved review threads on stacked PRs"
        parts = ", ".join(f"#{pr.number} ({len(pr.unresolved_thread_urls)})" for pr in self.offenders)
        text = f"Unresolved threads on stacked PRs: {parts}"
        if len(text) > STATUS_DESCRIPTION_LIMIT:
            text = text[: STATUS_DESCRIPTION_LIMIT - 3] + "..."
        return text

    def report_lines(self) -> list[str]:
        lines = [f"#{self.root.number} {self.root.url}: {self.state}"]
        for pr in self.offenders:
            lines.append(f"  stacked #{pr.number} {pr.url} (merged into #{pr.parent_number})")
            lines.extend(f"    unresolved: {url}" for url in pr.unresolved_thread_urls)
        return lines


def gh_transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Run one GraphQL request through the authenticated ``gh`` CLI."""
    body = json.dumps({"query": query, "variables": variables})
    completed = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=body,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"gh api graphql failed: {completed.stderr.strip()}")
    response: dict[str, Any] = json.loads(completed.stdout)
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
        root_commits: set[str] | None = None
        # (branch, parent number, parent created_at, parent merged_at or None for the root)
        pending: list[tuple[str, int, str, str | None]] = [(root.head_ref, root.number, root.created_at, None)]
        visited: set[str] = set()
        while pending:
            branch, parent_number, parent_created, parent_merged = pending.pop()
            if branch in visited or branch == "master":
                continue
            visited.add(branch)
            for child in self._merged_into(branch, parent_number):
                if child.merge_oid is not None:
                    if root_commits is None:
                        root_commits = self._commits(root.number)
                    in_root_history = child.merge_oid in root_commits
                else:
                    in_root_history = False
                merged_while_live = child.merged_at >= parent_created and (
                    parent_merged is None or child.merged_at <= parent_merged
                )
                if not (in_root_history or merged_while_live):
                    continue
                if child.unresolved_thread_urls:
                    verdict.offenders.append(child)
                pending.append((child.head_ref, child.number, child.created_at, child.merged_at))
        verdict.offenders.sort(key=lambda pr: pr.number)
        return verdict

    def _merged_into(self, branch: str, parent_number: int) -> Iterator[StackedPullRequest]:
        cursor: str | None = None
        while True:
            page = self._query(_CHILDREN_QUERY, branch=branch, cursor=cursor)["pullRequests"]
            for node in page["nodes"]:
                threads = node["reviewThreads"]
                unresolved = _unresolved_urls(threads["nodes"])
                if threads["pageInfo"]["hasNextPage"]:
                    unresolved.extend(self._more_unresolved(node["number"], threads["pageInfo"]["endCursor"]))
                yield StackedPullRequest(
                    number=node["number"],
                    url=node["url"],
                    head_ref=node["headRefName"],
                    created_at=node["createdAt"],
                    merged_at=node["mergedAt"],
                    merge_oid=(node.get("mergeCommit") or {}).get("oid"),
                    unresolved_thread_urls=tuple(unresolved),
                    parent_number=parent_number,
                )
            if not page["pageInfo"]["hasNextPage"]:
                return
            cursor = page["pageInfo"]["endCursor"]

    def _more_unresolved(self, number: int, cursor: str) -> list[str]:
        urls: list[str] = []
        next_cursor: str | None = cursor
        while next_cursor is not None:
            threads = self._query(_MORE_THREADS_QUERY, number=number, cursor=next_cursor)["pullRequest"][
                "reviewThreads"
            ]
            urls.extend(_unresolved_urls(threads["nodes"]))
            next_cursor = threads["pageInfo"]["endCursor"] if threads["pageInfo"]["hasNextPage"] else None
        return urls

    def _commits(self, number: int) -> set[str]:
        oids: set[str] = set()
        cursor: str | None = None
        while True:
            commits = self._query(_COMMITS_QUERY, number=number, cursor=cursor)["pullRequest"]["commits"]
            oids.update(node["commit"]["oid"] for node in commits["nodes"])
            if not commits["pageInfo"]["hasNextPage"]:
                return oids
            cursor = commits["pageInfo"]["endCursor"]


def _parse_root(node: dict[str, Any]) -> RootPullRequest:
    current: tuple[str, str] | None = None
    for commit_node in node["commits"]["nodes"]:
        status = commit_node["commit"].get("status") or {}
        for context in status.get("contexts") or []:
            if context["context"] == STATUS_CONTEXT:
                current = (context["state"].lower(), context["description"] or "")
    return RootPullRequest(
        number=node["number"],
        url=node["url"],
        head_ref=node["headRefName"],
        head_oid=node["headRefOid"],
        created_at=node["createdAt"],
        cross_repository=node["isCrossRepository"],
        current_status=current,
    )


def _unresolved_urls(threads: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for thread in threads:
        if thread["isResolved"]:
            continue
        comments = thread["comments"]["nodes"]
        urls.append(comments[0]["url"] if comments else "(thread without comments)")
    return urls


def post_status(repo: str, verdict: Verdict, target_url: str | None) -> None:
    args = [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{repo}/statuses/{verdict.root.head_oid}",
        "-f",
        f"state={verdict.state}",
        "-f",
        f"context={STATUS_CONTEXT}",
        "-f",
        f"description={verdict.description}",
    ]
    if target_url:
        args += ["-f", f"target_url={target_url}"]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"posting status for #{verdict.root.number} failed: {completed.stderr.strip()}")


def main(argv: list[str] | None = None, transport: Transport = gh_transport) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--pr", type=int, help="evaluate one PR instead of every open PR targeting master")
    parser.add_argument("--post-status", action="store_true", help=f"post the {STATUS_CONTEXT} commit status")
    parser.add_argument("--target-url", help="details link for posted statuses")
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
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("## Stacked review threads\n\n```\n" + "\n".join(lines) + "\n```\n")

    if args.post_status:
        for verdict in verdicts:
            if verdict.root.current_status == (verdict.state, verdict.description):
                continue
            post_status(args.repo, verdict, args.target_url)
        return 0
    return 1 if any(verdict.offenders for verdict in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
