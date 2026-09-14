"""Check the maintained codebase atlas for broken evidence anchors.

Atlas sections carry citations in ``(path/to/file.py:123)`` form.  This check
verifies that every cited file exists and that every cited line range lies
inside it.  Both findings are provable refusals -- the citation points at
something that is not there -- so they need no stamp, no history walk, and no
judgement about whether the surrounding prose is still accurate.  Findings are
an explicit queue for re-verification or deletion; the tool never edits
documentation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from devtools import repo_root

# Citations are commonly grouped as ``(`path.py:10`; `other.py:20`)``.
# Match the path token independently of the surrounding Markdown punctuation.
_CITATION_RE = re.compile(r"`?([A-Za-z0-9_./-]+):(\d+)(?:-(\d+))?`?")


@dataclass(frozen=True, slots=True)
class Finding:
    page: str
    section: str
    kind: str
    detail: str


def _sections(text: str) -> list[tuple[str, str, int, int]]:
    lines = text.splitlines()
    starts = [(index, line) for index, line in enumerate(lines) if line.startswith("## ")]
    return [
        (line[3:].strip(), "\n".join(lines[start:end]), start + 1, end)
        for (start, line), (end, _next) in zip(starts, [*starts[1:], (len(lines), "")], strict=True)
    ]


def inspect(root: Path) -> list[Finding]:
    atlas = root / "docs" / "atlas"
    findings: list[Finding] = []
    for page in sorted(atlas.glob("*.md")):
        relative_page = page.relative_to(root).as_posix()
        text = page.read_text(encoding="utf-8")
        for section, body, _start, _end in _sections(text):
            for match in _CITATION_RE.finditer(body):
                cited_path, first_line, last_line = match.groups()
                cited = root / cited_path
                if not cited.is_file():
                    findings.append(Finding(relative_page, section, "missing-file", cited_path))
                    continue
                line_count = len(cited.read_text(encoding="utf-8").splitlines())
                end_line = int(last_line or first_line)
                if int(first_line) < 1 or end_line > line_count:
                    findings.append(
                        Finding(
                            relative_page,
                            section,
                            "missing-anchor",
                            f"{cited_path}:{first_line}-{end_line} (file has {line_count} lines)",
                        )
                    )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (defaults to the current Git worktree)"
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable re-verification queue")
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    findings = inspect(root)
    payload: dict[str, Any] = {
        "atlas_root": str((root / "docs/atlas").relative_to(root)),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "status": "needs-attention" if findings else "current",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    elif findings:
        print("Atlas citations point at files or lines that do not exist:")
        for finding in findings:
            print(f"[BLOCK] {finding.page} :: {finding.section} :: {finding.kind} :: {finding.detail}")
    else:
        print("Every atlas citation resolves to an existing file and line range")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
