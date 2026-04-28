"""Parse structured findings from PR-level review bodies (Copilot, CodeRabbit, Codex).

Extracts:
- CodeRabbit: "Outside diff range" comments (missed by inline-only extraction)
- Copilot: All substantive feedback sections
- Codex: Summary content beyond inline-comment wrappers

Output: JSONL — one object per finding, with file path, line range, severity, description.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    pr_number: int
    pr_title: str
    bot: str  # copilot / coderabbit / codex
    source: str  # "outside_diff_range" | "pr_level_body" | "inline_comment_listing"
    severity: str  # critical / p1 / p2 / info / unknown
    file_path: str | None
    line_range: str | None  # e.g. "311-330" or "405"
    category: str | None  # bug / security / performance / style / nit / unknown
    title: str  # one-line summary
    description: str  # full finding text
    suggested_fix: str | None


# ── CodeRabbit parser ──────────────────────────────────────────────

CR_OUTSIDE_DIFF_RE = re.compile(
    r"<summary>(?P<file>[^<]+) \((?P<count>\d+)\)</summary>\s*<blockquote>\s*"
    r"(?:.*?)"  # skip to the actual content
    r"`(?P<lines>[\d\-, ]+)`:\s*(?:_\s*)?(?P<severity_label>[^|]+?)(?:\s*_\s*\|\s*_(?P<severity>[^_]+)_)?\s*"
    r"\*\*(?P<title>[^*]+)\*\*",
    re.DOTALL,
)

# Simpler approach: split on file-level patterns
CR_FILE_HEADER = re.compile(
    r"<summary>(?P<file>[^<]+?)\s*\((?P<count>\d+)\)</summary>",
)

CR_FINDING_BLOCK = re.compile(
    r"`(?P<lines>[\d\-, ]+)`:\s*(?:_\s*)?(?P<severity_label>[^|]+?)(?:\s*\|\s*_(?P<severity>[^_]+)_)?\s*"
    r"\*\*(?P<title>[^*]+?)\*\*\s*"
    r"(?P<body>.*?)"
    r"(?=<details>|</blockquote>|`\d+[\d\-, ]*`:\s*_|$)",
    re.DOTALL,
)

CR_SEVERITY = re.compile(r"(?:🔴|🟡|🟢|⚠️|🛑)\s*(?:Critical|P1|P2|Nit|Minor|Info)", re.IGNORECASE)


def _cr_severity(text: str) -> str:
    """Extract severity from CodeRabbit finding text."""
    t = text.lower()
    if "critical" in t or "🔴" in t or "🛑" in t:
        return "critical"
    if "p1" in t or "high" in t:
        return "p1"
    if "p2" in t or "medium" in t:
        return "p2"
    if "nit" in t or "minor" in t or "style" in t:
        return "nit"
    if "info" in t or "note" in t:
        return "info"
    return "unknown"


def parse_coderabbit_review(review: dict[str, Any]) -> list[Finding]:
    """Extract findings from a CodeRabbit PR-level review body.

    Focuses on 'Outside diff range comments' which were missed by inline extraction.
    Also captures the inline comment listings for completeness.
    """
    body = review.get("body", "")
    findings: list[Finding] = []
    pr = review["pr_number"]
    title = review.get("pr_title", "")

    # Split into sections
    outside_diff_section = ""

    # Look for outside-diff section
    od_match = re.search(
        r"outside.*?diff.*?range.*?comments.*?\((\d+)\)</summary>\s*<blockquote>(.*?)</blockquote>",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if od_match:
        outside_diff_section = od_match.group(2)

    # Parse outside-diff findings by file blocks
    for file_match in CR_FILE_HEADER.finditer(outside_diff_section):
        file_path = file_match.group("file").strip()
        # Find the content for this file block
        start = file_match.end()
        # Find next file block or end
        rest = outside_diff_section[start:]
        end_match = CR_FILE_HEADER.search(rest)
        block = rest[: end_match.start()] if end_match else rest

        # Extract individual findings
        for f_match in CR_FINDING_BLOCK.finditer(block):
            findings.append(
                Finding(
                    pr_number=pr,
                    pr_title=title,
                    bot="coderabbit",
                    source="outside_diff_range",
                    severity=_cr_severity(f_match.group("severity_label") or ""),
                    file_path=file_path,
                    line_range=f_match.group("lines"),
                    category=_cr_severity(f_match.group("severity_label") or ""),
                    title=f_match.group("title").strip(),
                    description=f_match.group("body").strip()[:2000],
                    suggested_fix=_extract_code_block(f_match.group("body")),
                )
            )

    # Also capture inline comment listings from the prompt section
    inline_prompt = re.search(r"Inline comments:\s*\n(.*?)(?=</details>|$)", body, re.DOTALL)
    if inline_prompt:
        inline_text = inline_prompt.group(1)
        # Parse file-level inline listings
        for inline_file in re.finditer(r"In `@([^`]+)`:\s*\n(.*?)(?=In `@|$)", inline_text, re.DOTALL):
            file_path = inline_file.group(1)
            remaining = inline_file.group(2)
            # Extract line-level references
            for line_ref in re.finditer(r"(?:Around|-) line[s]?\s*(\d[\d\-, ]*)", remaining, re.IGNORECASE):
                line_clause = line_ref.group(0)
                desc_match = re.search(
                    r"(?:The |\-\s*)(.*?)(?=\.\s*(?:\n|In `@|</details>|```|$))", remaining[line_ref.end() :], re.DOTALL
                )
                desc = desc_match.group(1)[:500] if desc_match else ""

                findings.append(
                    Finding(
                        pr_number=pr,
                        pr_title=title,
                        bot="coderabbit",
                        source="inline_comment_listing",
                        severity="unknown",
                        file_path=file_path,
                        line_range=line_ref.group(1),
                        category=None,
                        title=line_clause.strip(),
                        description=desc.strip(),
                        suggested_fix=None,
                    )
                )

    return findings


# ── Copilot parser ──────────────────────────────────────────────────


def parse_copilot_review(review: dict[str, Any]) -> list[Finding]:
    """Extract findings from a Copilot PR-level review body.

    Copilot reviews are structured as:
    - PR overview
    - File-by-file change summaries (table)
    - Categorized comments section (if comments were generated)
    """
    body = review.get("body", "")
    findings: list[Finding] = []
    pr = review["pr_number"]
    title = review.get("pr_title", "")

    # Extract the file summary table entries — each describes what changed
    # Format: | path | description |
    for row in re.finditer(r"\|\s*`?((?:crate|xtask|nixos|tests|docs)[^`|]+)`?\s*\|\s*([^|]+)\s*\|", body):
        file_path = row.group(1).strip()
        desc = row.group(2).strip()
        if len(desc) > 10:  # Non-trivial description
            findings.append(
                Finding(
                    pr_number=pr,
                    pr_title=title,
                    bot="copilot",
                    source="file_summary",
                    severity="info",
                    file_path=file_path,
                    line_range=None,
                    category=None,
                    title=f"File changed: {file_path}",
                    description=desc,
                    suggested_fix=None,
                )
            )

    # Look for explicit "Comments" or finding sections
    # Copilot sometimes has categorized issue sections
    comment_count = re.search(r"generated (\d+) comments?", body)
    if comment_count and int(comment_count.group(1)) > 0:
        # Look for category sections (common in Copilot reviews)
        for cat_match in re.finditer(
            r"(?:###|####)\s*(?:🔴|🟡|🟢|⚠️)?\s*(Category\s*\d+|Issue|Finding|Problem|Risk|Suggestion)[:—\s]*(.*?)(?=(?:###|####)|\Z)",
            body,
            re.DOTALL,
        ):
            cat_body = cat_match.group(0)
            cat_name = cat_match.group(2).strip() if cat_match.lastindex and cat_match.lastindex >= 2 else ""
            severity = _copilot_category_severity(cat_name)
            findings.append(
                Finding(
                    pr_number=pr,
                    pr_title=title,
                    bot="copilot",
                    source="categorized_comment",
                    severity=severity,
                    file_path=None,
                    line_range=None,
                    category=cat_name[:100] if cat_name else None,
                    title=cat_match.group(1),
                    description=cat_body[:2000],
                    suggested_fix=None,
                )
            )

    return findings


def _copilot_category_severity(category: str) -> str:
    c = category.lower()
    if any(w in c for w in ["critical", "bug", "security", "p1", "must fix"]):
        return "critical"
    if any(w in c for w in ["p2", "should fix", "important"]):
        return "p2"
    if any(w in c for w in ["nit", "style", "minor", "typo"]):
        return "nit"
    return "info"


# ── Codex parser ────────────────────────────────────────────────────


def parse_codex_review(review: dict[str, Any]) -> list[Finding]:
    """Extract findings from a Codex PR-level review body.

    Codex PR-level reviews are typically short wrappers linking to the Codex platform.
    The actual findings came through inline comments (144 total), already captured.
    We extract any additional context from the PR-level body.
    """
    body = review.get("body", "")
    findings: list[Finding] = []
    pr = review["pr_number"]
    title = review.get("pr_title", "")

    # Extract the P1/P2 summary if present
    p1_match = re.search(r"(?:P1|Priority 1|🔴)[:\s]*(\d+)", body, re.IGNORECASE)
    p2_match = re.search(r"(?:P2|Priority 2|🟡)[:\s]*(\d+)", body, re.IGNORECASE)

    if p1_match or p2_match:
        findings.append(
            Finding(
                pr_number=pr,
                pr_title=title,
                bot="codex",
                source="pr_level_summary",
                severity="info",
                file_path=None,
                line_range=None,
                category="review_summary",
                title=f"Codex review summary for PR #{pr}",
                description=f"P1 count: {p1_match.group(1) if p1_match else '0'}, "
                f"P2 count: {p2_match.group(1) if p2_match else '0'}",
                suggested_fix=None,
            )
        )

    # Check if there's substantive content beyond the wrapper
    # Remove known boilerplate
    clean = re.sub(r"<details>.*?</details>", "", body, flags=re.DOTALL)
    clean = re.sub(r"!\[.*?\]\(.*?\)", "", clean)
    clean = re.sub(r"\*\*Reviewed commit:\*\*.*", "", clean)
    clean = clean.strip()

    if len(clean) > 200:
        findings.append(
            Finding(
                pr_number=pr,
                pr_title=title,
                bot="codex",
                source="pr_level_body",
                severity="info",
                file_path=None,
                line_range=None,
                category=None,
                title=f"Codex PR-level body for PR #{pr}",
                description=clean[:2000],
                suggested_fix=None,
            )
        )

    return findings


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_code_block(text: str) -> str | None:
    """Extract first code block from markdown text."""
    m = re.search(r"```(?:diff|rust|python|nix|bash|shell)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip()[:3000] if m else None


# ── Main ────────────────────────────────────────────────────────────


def parse_all(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Parse all reviews and write findings JSONL. Returns stats."""
    with open(input_path) as f:
        reviews = [json.loads(line) for line in f]

    all_findings: list[Finding] = []
    stats = {"coderabbit": 0, "copilot": 0, "codex": 0}

    for r in reviews:
        author = r["author"]
        if author == "coderabbitai":
            findings = parse_coderabbit_review(r)
            stats["coderabbit"] += len(findings)
        elif author == "copilot-pull-request-reviewer":
            findings = parse_copilot_review(r)
            stats["copilot"] += len(findings)
        elif author == "chatgpt-codex-connector":
            findings = parse_codex_review(r)
            stats["codex"] += len(findings)
        else:
            continue
        all_findings.extend(findings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for finding in all_findings:
            f.write(
                json.dumps(
                    {
                        "pr_number": finding.pr_number,
                        "pr_title": finding.pr_title,
                        "bot": finding.bot,
                        "source": finding.source,
                        "severity": finding.severity,
                        "file_path": finding.file_path,
                        "line_range": finding.line_range,
                        "category": finding.category,
                        "title": finding.title,
                        "description": finding.description,
                        "suggested_fix": finding.suggested_fix,
                    }
                )
                + "\n"
            )

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse findings from PR-level review JSONL")
    parser.add_argument("input", type=Path, help="Input JSONL from extract_pr_reviews.py")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output findings JSONL")
    args = parser.parse_args()

    stats = parse_all(args.input, args.output)
    print("Extracted findings:", file=sys.stderr)
    for bot, count in stats.items():
        print(f"  {bot}: {count}", file=sys.stderr)
    print(f"  total: {sum(stats.values())}", file=sys.stderr)
    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
