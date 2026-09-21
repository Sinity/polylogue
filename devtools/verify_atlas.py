"""Check the maintained codebase atlas for broken evidence anchors.

Atlas sections carry citations in ``(path/to/file.py:123)`` form.  This check
verifies that every cited file exists and that every cited line range lies
inside it.  Both findings are provable refusals -- the citation points at
something that is not there -- so they need no stamp, no history walk, and no
judgement about whether the surrounding prose is still accurate.  Findings are
an explicit queue for re-verification or deletion; the tool never edits
documentation.

A section may additionally declare the gate that enforces its invariant, on a
line of the form ``**Owning gate**: `<name>```.  That name is resolved against
``devtools.gate.GATES_BY_NAME``, so a doctrine cannot name a gate that no
longer exists.  The denominator is the live registry rather than a copy of it
here: a deleted gate disappears from ``GATES`` and the declaration goes red on
the next run without anyone editing this module.

The declaration is a claim of ownership, not a runnable example, and it is
deliberately written as a bare gate name.  ``devtools gate doc-commands``
already rejects the *runnable* spelling ``devtools gate <name>`` wherever it
appears inside a Markdown code segment, but it never sees an owner named this
way -- that residual is what this check covers.  Both spellings are accepted
so a sheet that prefers the runnable form is still resolved here.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from devtools import repo_root
from devtools.gate import GATES_BY_NAME

# Citations are commonly grouped as ``(`path.py:10`; `other.py:20`)``.
# Match the path token independently of the surrounding Markdown punctuation.
_CITATION_RE = re.compile(r"`?([A-Za-z0-9_./-]+):(\d+)(?:-(\d+))?`?")

#: ``**Owning gate**: `layering`, `atlas``` or ``**Owning gate**: none -- why``.
_OWNING_GATE_RE = re.compile(r"^\*\*Owning gate\*\*:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_GATE_TOKEN_RE = re.compile(r"`([^`]+)`")
#: Punctuation allowed between declared gate names, so a stray word cannot hide
#: beside a resolvable one.
_DECLARATION_SEPARATORS = " \t,;."


def declared_gate_names(value: str) -> tuple[str, ...] | None:
    """Parse one ``**Owning gate**`` value, or ``None`` when it is unparsable.

    An empty tuple means the section declared that no gate owns the invariant.
    A value that is neither ``none`` nor a list of backticked gate tokens is
    refused rather than ignored: silently skipping an unrecognised declaration
    is how a doctrine would keep an owner nobody checks.
    """

    if re.match(r"^none\b", value):
        return ()
    tokens = _GATE_TOKEN_RE.findall(value)
    if not tokens:
        return None
    if _GATE_TOKEN_RE.sub("", value).strip(_DECLARATION_SEPARATORS):
        return None
    names: list[str] = []
    for token in tokens:
        parts = token.split()
        if parts[:2] == ["devtools", "gate"] and len(parts) == 3:
            names.append(parts[2])
        elif len(parts) == 1:
            names.append(parts[0])
        else:
            return None
    return tuple(names)


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
            for declaration in _OWNING_GATE_RE.finditer(body):
                value = declaration.group(1)
                names = declared_gate_names(value)
                if names is None:
                    findings.append(Finding(relative_page, section, "unparsable-owning-gate", value))
                    continue
                findings.extend(
                    Finding(relative_page, section, "unknown-gate", name) for name in names if name not in GATES_BY_NAME
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
        print("Atlas anchors do not resolve (cited file, cited line range, or declared owning gate):")
        for finding in findings:
            print(f"[BLOCK] {finding.page} :: {finding.section} :: {finding.kind} :: {finding.detail}")
    else:
        print("Every atlas citation and declared owning gate resolves")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
