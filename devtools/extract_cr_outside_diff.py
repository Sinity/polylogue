"""Extract CodeRabbit 'outside diff range' findings from PR-level review bodies.

These findings were on lines outside the PR diff and couldn't be posted as inline
comments — so the bulk `gh api pulls/comments` extraction missed them entirely.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def extract_outside_diff_findings(body: str, pr_number: int) -> list[dict[str, Any]]:
    """Extract structured findings from CodeRabbit outside-diff-range sections."""
    findings: list[dict[str, Any]] = []

    # Find the outside-diff section
    od_section = re.search(
        r"(?:Outside|outside).*?diff.*?range.*?comments?\s*\((\d+)\)</summary>\s*<blockquote>(.*?)(?=</blockquote>\s*</details>)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if not od_section:
        return findings

    content = od_section.group(2)

    # Parse per-file blocks
    # Each file is in: <details><summary>path (N)</summary><blockquote>...findings...</blockquote></details>
    file_blocks = re.finditer(
        r"<summary>(?P<file>[^<]+?)\s*\((?P<count>\d+)\)</summary>\s*<blockquote>\s*"
        r"(?P<body>.*?)"
        r"</blockquote>\s*</details>",
        content,
        re.DOTALL,
    )

    for fb in file_blocks:
        file_path = fb.group("file").strip()
        block_body = fb.group("body")

        # Each finding starts with a line range in backticks
        finding_blocks = re.finditer(
            r"`(?P<lines>[\d\-,\s]+)`:\s*"
            r"(?P<annotation>_[^_]+_(?:\s*\|\s*_[^_]+_)?\s*)"
            r"\*\*(?P<title>[^*]+?)\*\*\s*"
            r"(?P<body>.*?)"
            r"(?=<details>|</blockquote>|`\d|$)",
            block_body,
            re.DOTALL,
        )

        for f_match in finding_blocks:
            annotation = f_match.group("annotation")
            title = f_match.group("title").strip()
            f_body = f_match.group("body").strip()

            # Extract severity
            severity = "unknown"
            if "critical" in annotation.lower() or "🔴" in annotation:
                severity = "critical"
            elif "p1" in annotation.lower() or "high" in annotation.lower():
                severity = "p1"
            elif "p2" in annotation.lower() or "medium" in annotation.lower():
                severity = "p2"
            elif "nit" in annotation.lower() or "minor" in annotation.lower():
                severity = "nit"

            # Extract suggested fix
            fix_match = re.search(r"```(?:diff|rust|python|nix|bash|shell)?\s*\n(.*?)```", f_body, re.DOTALL)
            suggested_fix = fix_match.group(1).strip()[:3000] if fix_match else None

            # Extract AI verification prompt
            prompt_match = re.search(r"Verify each finding against the current code.*?```", f_body, re.DOTALL)
            verification_prompt = prompt_match.group(0) if prompt_match else None

            findings.append(
                {
                    "pr_number": pr_number,
                    "bot": "coderabbit",
                    "source": "outside_diff_range",
                    "file_path": file_path,
                    "line_range": f_match.group("lines"),
                    "severity": severity,
                    "title": title,
                    "description": f_body[:3000],
                    "suggested_fix": suggested_fix,
                    "verification_prompt": verification_prompt,
                }
            )

    return findings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract CodeRabbit outside-diff-range findings from PR reviews")
    parser.add_argument("input", type=Path, help="Input JSONL from extract_pr_reviews.py")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output findings JSONL")
    args = parser.parse_args()

    with open(args.input) as f:
        reviews = [json.loads(line) for line in f]

    cr_reviews = [r for r in reviews if r["author"] == "coderabbitai"]
    print(f"Processing {len(cr_reviews)} CodeRabbit PR-level reviews...", file=sys.stderr)

    all_findings: list[dict[str, Any]] = []
    prs_with_findings: set[int] = set()
    for r in cr_reviews:
        findings = extract_outside_diff_findings(r["body"], r["pr_number"])
        if findings:
            prs_with_findings.add(r["pr_number"])
            all_findings.extend(findings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for finding in all_findings:
            f.write(json.dumps(finding) + "\n")

    print(
        f"Extracted {len(all_findings)} outside-diff-range findings from {len(prs_with_findings)} PRs", file=sys.stderr
    )
    print(f"PRs: {sorted(prs_with_findings)}", file=sys.stderr)

    # Stats by severity
    by_severity: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for finding in all_findings:
        by_severity[finding["severity"]] = by_severity.get(finding["severity"], 0) + 1
        fp = finding.get("file_path", "unknown")
        by_file[fp] = by_file.get(fp, 0) + 1

    print("\nBy severity:", file=sys.stderr)
    for sev, count in sorted(by_severity.items()):
        print(f"  {sev}: {count}", file=sys.stderr)

    print("\nTop files:", file=sys.stderr)
    for fp, count in sorted(by_file.items(), key=lambda x: -x[1])[:15]:
        print(f"  {fp}: {count}", file=sys.stderr)

    print(f"\nWritten to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
