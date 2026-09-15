"""Verify CLI/MCP/Python operation parity and the library documentation.

Two checks, both derived from live sources:

1. **Matrix parity** -- :func:`polylogue.api.parity.validate_parity` proves the
   declared operation matrix is total over the public facade and that every
   live binding resolves.
2. **Library documentation** -- ``docs/library-api.md`` is read section by
   section; every documented facade call must name a real callable, use the
   correct ``await`` form for its asyncness, pass keyword arguments the live
   signature accepts, and cover every semantic operation bound to the facade.

Anti-vacuity: renaming or deleting a facade method, flipping one from ``async
def`` to ``def``, adding a public method without classifying it, documenting a
keyword the signature does not accept, or dropping a documented operation from
``docs/library-api.md`` each make this exit non-zero. Emptying the doc's code
examples fails as ``no_documented_operations`` rather than passing vacuously.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.api.parity import (
    FACADE_SYMBOL,
    ParityFinding,
    is_async_operation,
    operation_python_names,
    public_facade_callables,
    semantic_operations,
    validate_parity,
)

ROOT = _get_root()
LIBRARY_DOC = "docs/library-api.md"
REPAIR = "devtools verify api-parity --check"

#: Receivers that denote a live ``Polylogue`` facade instance in the docs.
_FACADE_RECEIVERS = ("archive", "poly", "polylogue")

_FENCE = re.compile(r"^```(\w*)\s*$")
_CALL = re.compile(r"(?P<await>await\s+)?\b(?P<receiver>" + "|".join(_FACADE_RECEIVERS) + r")\.(?P<name>\w+)\(")
_KEYWORD = re.compile(r"^\s*(?P<keyword>[A-Za-z_]\w*)\s*=(?!=)")


def _call_arguments(line: str, open_paren: int) -> tuple[str, int]:
    """Return the balanced argument text of a call and the index after it."""

    depth = 0
    for index in range(open_paren, len(line)):
        character = line[index]
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return line[open_paren + 1 : index], index + 1
    return line[open_paren + 1 :], len(line)


def _top_level_keywords(arguments: str) -> tuple[str, ...]:
    """Return keyword names passed at the outermost call depth only."""

    keywords: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(arguments):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            match = _KEYWORD.match(arguments[start:index])
            if match is not None:
                keywords.append(match.group("keyword"))
            start = index + 1
    match = _KEYWORD.match(arguments[start:])
    if match is not None:
        keywords.append(match.group("keyword"))
    return tuple(keywords)


_HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True, slots=True)
class DocumentedCall:
    """One facade call observed inside a documentation code surface."""

    name: str
    section: str
    line: int
    awaited: bool
    chained: bool
    keywords: tuple[str, ...]


def _facade_class() -> type:
    from polylogue.api import Polylogue

    return Polylogue


def documented_calls(text: str) -> tuple[DocumentedCall, ...]:
    """Return every facade call inside a fenced code block, with its section."""

    calls: list[DocumentedCall] = []
    section = "(preamble)"
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        fence = _FENCE.match(line)
        if fence is not None:
            in_fence = not in_fence
            continue
        heading = _HEADING.match(line)
        if heading is not None and not in_fence:
            section = heading.group("title")
            continue
        if not in_fence:
            continue
        for match in _CALL.finditer(line):
            arguments, after = _call_arguments(line, match.end() - 1)
            calls.append(
                DocumentedCall(
                    name=match.group("name"),
                    section=section,
                    line=number,
                    awaited=bool(match.group("await")),
                    chained=line[after : after + 1] == ".",
                    keywords=_top_level_keywords(arguments),
                )
            )
    return tuple(calls)


def _accepts_keyword(name: str, keyword: str) -> bool:
    try:
        signature = inspect.signature(getattr(_facade_class(), name))
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if keyword in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def validate_library_doc(path: Path) -> tuple[ParityFinding, ...]:
    """Return every library-documentation defect, deterministically ordered."""

    findings: list[ParityFinding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (ParityFinding("missing_library_doc", str(path), f"{path} does not exist", REPAIR),)

    live = set(public_facade_callables())
    calls = documented_calls(text)
    covered: set[str] = set()
    for call in calls:
        if call.name not in live:
            findings.append(
                ParityFinding(
                    "unknown_documented_callable",
                    f"{path}:{call.line}:{call.name}",
                    f"{LIBRARY_DOC} documents `{call.name}` in section {call.section!r}, "
                    f"which is not a public callable on {FACADE_SYMBOL}",
                    REPAIR,
                )
            )
            continue
        covered.add(call.name)
        # ``await archive.X(...)`` asserts X is awaitable. A chained call
        # (``await archive.filter().list()``) awaits its terminal, not X, and a
        # bare ``archive.X()`` may be awaited elsewhere (``asyncio.gather``),
        # so only the direct-await form carries a checkable claim.
        if call.awaited and not call.chained and not is_async_operation(call.name):
            findings.append(
                ParityFinding(
                    "async_mismatch",
                    f"{path}:{call.line}:{call.name}",
                    f"{LIBRARY_DOC} awaits `{call.name}` directly, but the live facade callable is not a "
                    f"coroutine function",
                    REPAIR,
                )
            )
        if call.chained:
            continue
        for keyword in call.keywords:
            if not _accepts_keyword(call.name, keyword):
                findings.append(
                    ParityFinding(
                        "unknown_documented_keyword",
                        f"{path}:{call.line}:{call.name}.{keyword}",
                        f"{LIBRARY_DOC} passes `{keyword}=` to `{call.name}`, which its live signature does not accept",
                        REPAIR,
                    )
                )

    if not covered:
        findings.append(
            ParityFinding(
                "no_documented_operations",
                str(path),
                f"{LIBRARY_DOC} documents no live facade call; the verifier would otherwise pass vacuously",
                REPAIR,
            )
        )
    for name in sorted(set(operation_python_names()) - covered):
        findings.append(
            ParityFinding(
                "undocumented_operation",
                name,
                f"semantic operation callable `{name}` is not documented in {LIBRARY_DOC}",
                REPAIR,
            )
        )
    return tuple(sorted(findings, key=lambda item: (item.code, item.subject)))


def run(*, doc_path: Path) -> tuple[ParityFinding, ...]:
    return tuple(sorted((*validate_parity(), *validate_library_doc(doc_path)), key=lambda i: (i.code, i.subject)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify CLI/MCP/Python operation parity and library docs.")
    parser.add_argument("--json", action="store_true", help="Emit the parity report as JSON.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any parity or documentation finding is reported (default behavior).",
    )
    parser.add_argument(
        "--doc", default=str(ROOT / LIBRARY_DOC), help=f"Library doc to verify (default: {LIBRARY_DOC})"
    )
    args = parser.parse_args(argv)

    findings = run(doc_path=Path(args.doc))
    operations = semantic_operations()
    if args.json:
        print(
            json.dumps(
                {
                    "operations": len(operations),
                    "facade_callables": len(public_facade_callables()),
                    "findings": [
                        {
                            "code": finding.code,
                            "subject": finding.subject,
                            "message": finding.message,
                            "repair_command": finding.repair_command,
                        }
                        for finding in findings
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"api-parity: {len(operations)} semantic operations, {len(public_facade_callables())} facade callables")
        for finding in findings:
            print(f"api-parity: {finding.code}: {finding.subject}: {finding.message}", file=sys.stderr)
        if findings:
            print(f"api-parity: repair with `{REPAIR}`", file=sys.stderr)
        else:
            print("api-parity: OK")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
