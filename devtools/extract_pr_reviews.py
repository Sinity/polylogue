"""Extract PR-level reviews from GitHub — complements inline comments from pulls/comments.

GitHub's bulk `gh api pulls/comments` returns only inline review comments.
PR-level reviews (the top-level review body + state) require per-PR:
    gh pr view <N> --json reviews

This tool extracts those missing PR-level reviews for comprehensive bot-feedback auditing.

Output: JSON lines — one object per review, with PR metadata attached.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass
class Review:
    pr_number: int
    pr_title: str
    pr_url: str
    pr_state: str  # MERGED / CLOSED
    pr_merged_at: str | None
    review_id: str
    author: str
    state: str  # APPROVED / COMMENTED / CHANGES_REQUESTED / DISMISSED
    body: str
    submitted_at: str | None
    last_modified_at: str | None
    html_url: str


def run_gh(args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess[str]:
    """Run gh CLI and return completed process. Raises on failure."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        input=stdin,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"[WARN] gh {' '.join(args)} → rc={result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
    return result


def list_prs(owner: str, repo: str, state: str = "merged") -> list[dict[str, Any]]:
    """List all PRs in a given state (merged/closed)."""
    prs: list[dict[str, Any]] = []
    cmd = [
        "pr",
        "list",
        "--repo",
        f"{owner}/{repo}",
        "--state",
        state,
        "--limit",
        "1000",
        "--json",
        "number,title,url,state,mergedAt,closedAt",
    ]
    result = run_gh(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Failed to list {state} PRs", file=sys.stderr)
        return prs
    return cast(list[dict[str, Any]], json.loads(result.stdout))


def get_pr_reviews(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Get PR-level reviews for a single PR."""
    cmd = [
        "pr",
        "view",
        str(pr_number),
        "--repo",
        f"{owner}/{repo}",
        "--json",
        "reviews",
    ]
    result = run_gh(cmd)
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[WARN] PR #{pr_number}: invalid JSON from gh pr view", file=sys.stderr)
        return []
    return cast(list[dict[str, Any]], data.get("reviews", []))


def extract_reviews(
    owner: str,
    repo: str,
    states: tuple[str, ...] = ("merged", "closed"),
    reviewer_filter: set[str] | None = None,
) -> list[Review]:
    """Extract all PR-level reviews for a repository.

    Args:
        owner: GitHub org/user
        repo: Repository name
        states: PR states to include
        reviewer_filter: If set, only include reviews from these logins (case-insensitive)
    """
    all_prs: list[dict[str, Any]] = []
    for state in states:
        prs = list_prs(owner, repo, state)
        all_prs.extend(prs)
        print(f"  Listed {len(prs)} {state} PRs", file=sys.stderr)

    # Deduplicate by PR number (a PR may appear in both merged and closed)
    seen: set[int] = set()
    unique_prs: list[dict[str, Any]] = []
    for pr in all_prs:
        n = pr["number"]
        if n not in seen:
            seen.add(n)
            unique_prs.append(pr)

    total = len(unique_prs)
    reviews: list[Review] = []
    for i, pr in enumerate(unique_prs):
        n = pr["number"]
        if i % 50 == 0:
            print(f"  [{i}/{total}] Extracting PR-level reviews...", file=sys.stderr)

        pr_reviews = get_pr_reviews(owner, repo, n)
        if not pr_reviews:
            continue

        for r in pr_reviews:
            author = (r.get("author", {}) or {}).get("login", "unknown")
            if reviewer_filter and author.lower() not in reviewer_filter:
                continue

            body = r.get("body", "") or ""
            reviews.append(
                Review(
                    pr_number=n,
                    pr_title=pr.get("title", ""),
                    pr_url=pr.get("url", ""),
                    pr_state=pr.get("state", "UNKNOWN"),
                    pr_merged_at=pr.get("mergedAt"),
                    review_id=r.get("id", ""),
                    author=author,
                    state=r.get("state", "UNKNOWN"),
                    body=body,
                    submitted_at=r.get("submittedAt"),
                    last_modified_at=r.get("lastModifiedAt"),
                    html_url=r.get("url", ""),
                )
            )

    return reviews


def reviews_to_jsonl(reviews: list[Review], path: Path) -> None:
    """Write reviews as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in reviews:
            f.write(
                json.dumps(
                    {
                        "pr_number": r.pr_number,
                        "pr_title": r.pr_title,
                        "pr_url": r.pr_url,
                        "pr_state": r.pr_state,
                        "pr_merged_at": r.pr_merged_at,
                        "review_id": r.review_id,
                        "author": r.author,
                        "state": r.state,
                        "body": r.body,
                        "submitted_at": r.submitted_at,
                        "last_modified_at": r.last_modified_at,
                        "html_url": r.html_url,
                    }
                )
                + "\n"
            )


def print_summary(reviews: list[Review]) -> None:
    """Print summary statistics."""
    by_author: dict[str, list[Review]] = {}
    by_state: dict[str, int] = {}
    for r in reviews:
        by_author.setdefault(r.author, []).append(r)
        by_state[r.state] = by_state.get(r.state, 0) + 1

    print("\n=== PR-Level Review Summary ===")
    print(f"Total reviews: {len(reviews)}")
    print(f"Unique PRs:    {len({r.pr_number for r in reviews})}")

    print("\nBy author:")
    for author, items in sorted(by_author.items(), key=lambda x: -len(x[1])):
        prs = len({r.pr_number for r in items})
        print(f"  {author}: {len(items)} reviews across {prs} PRs")

    print("\nBy state:")
    for state, count in sorted(by_state.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    # Reviews with substantive bodies (>50 chars)
    substantive = [r for r in reviews if len(r.body.strip()) > 50]
    print(f"\nSubstantive reviews (body > 50 chars): {len(substantive)}/{len(reviews)}")

    # Bodies that look like boilerplate
    boilerplate_markers = [
        "rate limit",
        "rate-limit",
        "config",
        "unrecognized",
        "onboarding",
        "welcome",
        "summary",
        "walk-through",
    ]
    boilerplate = [r for r in reviews if any(m in r.body.lower() for m in boilerplate_markers)]
    print(f"Likely boilerplate: {len(boilerplate)}/{len(reviews)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract PR-level reviews from GitHub (not inline comments)")
    parser.add_argument("--owner", default="Sinity", help="GitHub org/user")
    parser.add_argument("--repo", default="sinex", help="Repository name")
    parser.add_argument(
        "--reviewer", action="append", default=None, help="Filter by reviewer login (can repeat, case-insensitive)"
    )
    parser.add_argument("--output", default=None, help="Output JSONL file (default: pr_level_reviews_<repo>.jsonl)")
    parser.add_argument(
        "--states", default="merged,closed", help="PR states to include (comma-separated, default: merged,closed)"
    )
    args = parser.parse_args()

    states = tuple(s.strip() for s in args.states.split(","))
    reviewer_filter = {r.lower() for r in args.reviewer} if args.reviewer else None
    output = Path(args.output) if args.output else Path(f"pr_level_reviews_{args.repo}.jsonl")

    print(f"Extracting PR-level reviews for {args.owner}/{args.repo}", file=sys.stderr)
    print(f"  States: {states}", file=sys.stderr)
    if reviewer_filter:
        print(f"  Reviewer filter: {reviewer_filter}", file=sys.stderr)

    reviews = extract_reviews(args.owner, args.repo, states, reviewer_filter)

    reviews_to_jsonl(reviews, output)
    print(f"\nWrote {len(reviews)} reviews to {output}", file=sys.stderr)

    print_summary(reviews)


if __name__ == "__main__":
    main()
