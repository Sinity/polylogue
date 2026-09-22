"""Check the maintained codebase atlas for broken evidence anchors.

Atlas sections carry citations in ``(path/to/file.py:123)`` form.  This check
verifies that every cited file exists, that every cited line range lies inside
it, and that a citation whose prose names a symbol actually points at code
where that symbol appears.  All three findings are provable refusals -- the
citation points at something that is not there -- so they need no stamp and no
history walk.  Findings are an explicit queue for re-verification or deletion;
the tool never edits documentation.

The symbol check exists because the first two are satisfied by any in-range
line numbers at all: an anchor whose range drifted onto unrelated code was
indistinguishable from a correct one, and one did drift 486 lines while the
gate stayed green (polylogue-72cbt).  The claim is taken from the prose that
already introduces the citation -- a backticked identifier in the same
sentence -- so nothing new is hand-maintained.  A pinned content digest was
rejected deliberately: it would have to be regenerated after every unrelated
edit inside the range, so it would be regenerated reflexively without anyone
re-reading the prose, which looks maintained and checks nothing.

The check is deliberately a floor, not a completeness requirement, and it
reports its own coverage rather than implying it examined every citation:

* A consecutive run of citations is one evidence bundle for one sentence, so
  the symbol must appear in at least one of the bundle's ranges.  Requiring it
  in every range would refuse the ordinary case where a sentence's clauses are
  each anchored separately.
* A range inside the ``def``/``class`` the prose names counts, because a
  method body does not repeat its own owner's name.
* Prose that names no backticked identifier, and prose whose identifier does
  not occur anywhere in the cited file, cannot be checked this way.  Both are
  counted and printed as unchecked rather than silently passed -- reporting
  success over citations nobody examined is the defect this check is about.

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
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from devtools import repo_root
from devtools.gate import GATES_BY_NAME

# Citations are commonly grouped as ``(`path.py:10`; `other.py:20`)``.
# Match the path token independently of the surrounding Markdown punctuation.
_CITATION_RE = re.compile(r"`?([A-Za-z0-9_./-]+):(\d+)(?:-(\d+))?`?")

#: Any backticked span, masked out before sentence boundaries are located so a
#: ``.py`` inside a cited path cannot read as the end of a sentence.
_BACKTICK_SPAN_RE = re.compile(r"`[^`\n]*`")
_BACKTICK_TOKEN_RE = re.compile(r"`([^`\n]+)`")
#: A dotted or bare Python name, the only backtick shape read as a symbol claim.
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Sentence end, table-cell wall, list-item start, blank line, or heading.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?]\s|\||\n\s*(?:[-*+]|\d+\.)\s|\n\s*\n|\n#)")
#: Punctuation that joins citations into one bundle: ``(`a.py:1`; `b.py:2`)``.
_CITATION_GLUE_RE = re.compile(r"^[\s`;,()\[\]]*(?:and[\s`;,()\[\]]*)?$")
#: Lines of tolerance around a cited range, for a decorator or a wrapped
#: signature immediately outside it.
_ANCHOR_SLACK_LINES = 3
#: Shorter backticked words are too ambiguous to read as a symbol claim.
_MIN_CLAIM_LENGTH = 3

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


@dataclass(slots=True)
class ClaimCoverage:
    """How much of the atlas the symbol check actually examined.

    ``checked`` plus the three unchecked counts is the number of citation
    bundles whose target is a readable Python file; the gate prints all of
    them so a green run cannot be read as a claim about every citation.
    """

    citations: int = 0
    bundles: int = 0
    checked: int = 0
    misaimed: int = 0
    no_symbol_claim: int = 0
    claim_absent_from_target: int = 0
    non_python_target: int = 0

    @property
    def unchecked(self) -> int:
        return self.no_symbol_claim + self.claim_absent_from_target + self.non_python_target

    def summary(self) -> str:
        return (
            f"{self.citations} atlas citations in {self.bundles} evidence bundles: "
            f"{self.checked} carry a checkable symbol claim ({self.misaimed} misaimed), "
            f"{self.unchecked} unchecked "
            f"({self.no_symbol_claim} name no backticked symbol, "
            f"{self.claim_absent_from_target} name a symbol absent from the cited file, "
            f"{self.non_python_target} target a non-Python file)"
        )


@dataclass(frozen=True, slots=True)
class AtlasAudit:
    findings: tuple[Finding, ...]
    coverage: ClaimCoverage = field(default_factory=ClaimCoverage)


class _SourceIndex:
    """Per-run reader for cited Python files: lines, identifiers, definitions."""

    def __init__(self) -> None:
        self._lines: dict[Path, tuple[str, ...]] = {}
        self._identifiers: dict[Path, frozenset[str]] = {}
        self._definitions: dict[Path, tuple[tuple[str, int, int], ...]] = {}

    def lines(self, path: Path) -> tuple[str, ...]:
        if path not in self._lines:
            self._lines[path] = tuple(path.read_text(encoding="utf-8", errors="replace").splitlines())
        return self._lines[path]

    def identifiers(self, path: Path) -> frozenset[str]:
        if path not in self._identifiers:
            self._identifiers[path] = frozenset(_IDENTIFIER_RE.findall("\n".join(self.lines(path))))
        return self._identifiers[path]

    def definitions(self, path: Path) -> tuple[tuple[str, int, int], ...]:
        """``(name, first line, last line)`` for every ``def``/``class``."""

        if path not in self._definitions:
            try:
                tree = ast.parse("\n".join(self.lines(path)))
            except SyntaxError:
                self._definitions[path] = ()
            else:
                spans: list[tuple[str, int, int]] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        first = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
                        spans.append((node.name, first, node.end_lineno or first))
                self._definitions[path] = tuple(spans)
        return self._definitions[path]


def _masked(body: str) -> str:
    return _BACKTICK_SPAN_RE.sub(lambda match: " " * len(match.group(0)), body)


def claimed_symbols(text: str) -> tuple[str, ...]:
    """Symbols the prose offers as what a citation is about.

    A backticked token counts only when it is a dotted or bare Python name;
    the trailing component is the symbol (``blocks.tool_outcome`` claims
    ``tool_outcome``).  An all-lowercase word with no underscore is dropped:
    ``origin``, ``actions`` and ``record`` are ordinary English as often as
    they are code, and reading them as claims produces refusals that name
    nothing an author can repoint.
    """

    symbols: list[str] = []
    for token in _BACKTICK_TOKEN_RE.findall(text):
        candidate = token.strip()
        if not _DOTTED_NAME_RE.match(candidate):
            continue
        name = candidate.rsplit(".", 1)[-1]
        if len(name) < _MIN_CLAIM_LENGTH or (name.islower() and "_" not in name):
            continue
        if name not in symbols:
            symbols.append(name)
    return tuple(symbols)


def _sentence(body: str, start: int, end: int) -> str:
    """The sentence, list item or table cell holding ``body[start:end]``."""

    masked = _masked(body)
    opening = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(masked, 0, start):
        opening = boundary.end()
    closing = _SENTENCE_BOUNDARY_RE.search(masked, end)
    return body[opening:start] + " " + body[end : closing.start() if closing else len(body)]


def _citation_bundles(body: str) -> list[list[re.Match[str]]]:
    """Group consecutive citations joined only by punctuation into one bundle."""

    bundles: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in _CITATION_RE.finditer(body):
        if current and _CITATION_GLUE_RE.match(body[current[-1].end() : match.start()]):
            current.append(match)
        else:
            if current:
                bundles.append(current)
            current = [match]
    if current:
        bundles.append(current)
    return bundles


def _symbol_is_anchored(index: _SourceIndex, path: Path, first: int, last: int, symbol: str) -> bool:
    lines = index.lines(path)
    window = "\n".join(lines[max(1, first - _ANCHOR_SLACK_LINES) - 1 : min(len(lines), last + _ANCHOR_SLACK_LINES)])
    # ``_?`` so prose may name a private helper without its underscore, and
    # ``s?`` so ``executable `OriginSpec`s`` still matches ``OriginSpecs``.
    if re.search(rf"\b_?{re.escape(symbol)}s?\b", window):
        return True
    return any(name == symbol and start <= first and last <= end for name, start, end in index.definitions(path))


def _sections(text: str) -> list[tuple[str, str, int, int]]:
    lines = text.splitlines()
    starts = [(index, line) for index, line in enumerate(lines) if line.startswith("## ")]
    return [
        (line[3:].strip(), "\n".join(lines[start:end]), start + 1, end)
        for (start, line), (end, _next) in zip(starts, [*starts[1:], (len(lines), "")], strict=True)
    ]


def audit(root: Path) -> AtlasAudit:
    atlas = root / "docs" / "atlas"
    findings: list[Finding] = []
    coverage = ClaimCoverage()
    index = _SourceIndex()
    for page in sorted(atlas.glob("*.md")):
        relative_page = page.relative_to(root).as_posix()
        text = page.read_text(encoding="utf-8")
        for section, body, _start, _end in _sections(text):
            for bundle in _citation_bundles(body):
                coverage.citations += len(bundle)
                coverage.bundles += 1
                targets: list[tuple[str, int, int, Path]] = []
                foreign = False
                for match in bundle:
                    cited_path, first_line, last_line = match.groups()
                    cited = root / cited_path
                    if not cited.is_file():
                        findings.append(Finding(relative_page, section, "missing-file", cited_path))
                        continue
                    line_count = len(index.lines(cited))
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
                        continue
                    if cited.suffix != ".py":
                        foreign = True
                        continue
                    targets.append((cited_path, int(first_line), end_line, cited))
                if not targets:
                    coverage.non_python_target += int(foreign)
                    continue
                symbols = claimed_symbols(_sentence(body, bundle[0].start(), bundle[-1].end()))
                named = {
                    symbol
                    for symbol in symbols
                    for _path, _first, _last, cited in targets
                    if index.identifiers(cited) & {symbol, f"_{symbol}", f"{symbol}s"}
                }
                if not symbols:
                    coverage.no_symbol_claim += 1
                    continue
                if not named:
                    coverage.claim_absent_from_target += 1
                    continue
                coverage.checked += 1
                if any(
                    _symbol_is_anchored(index, cited, first, last, symbol)
                    for symbol in named
                    for _path, first, last, cited in targets
                ):
                    continue
                coverage.misaimed += 1
                where = "; ".join(f"{path}:{first}-{last}" for path, first, last, _cited in targets)
                findings.append(
                    Finding(
                        relative_page,
                        section,
                        "misaimed-anchor",
                        f"{where} contains none of {sorted(named)} named by the citing sentence",
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
    return AtlasAudit(tuple(findings), coverage)


def inspect(root: Path) -> list[Finding]:
    """Every refusal in the atlas, without the coverage report."""

    return list(audit(root).findings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (defaults to the current Git worktree)"
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable re-verification queue")
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    result = audit(root)
    findings = result.findings
    payload: dict[str, Any] = {
        "atlas_root": str((root / "docs/atlas").relative_to(root)),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "claim_coverage": asdict(result.coverage),
        "status": "needs-attention" if findings else "current",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 1 if findings else 0
    if findings:
        print("Atlas anchors do not resolve (cited file, cited line range, named symbol, or declared owning gate):")
        for finding in findings:
            print(f"[BLOCK] {finding.page} :: {finding.section} :: {finding.kind} :: {finding.detail}")
    else:
        print("Every atlas citation and declared owning gate resolves")
    # Printed on the passing path too: a green run is a claim about what was
    # checked, and the unchecked citations are part of that claim.
    print(result.coverage.summary())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
