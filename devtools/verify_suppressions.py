"""Inspect and enforce suppression registry expiry dates.

Every suppression in ``docs/plans/suppressions.yaml`` must have an expiry
date. The lint fails when any suppression is past its expiry date, forcing
review and either renewal or removal.

The command also discovers source-level exception mechanisms so an empty
registry cannot be mistaken for an absence of suppressions. Beyond raw
discovery, the scanner enforces discipline rules native tools cannot
express:

- ``pytest.skip`` / ``pytest.mark.skip`` / ``pytest.mark.skipif`` must
  carry a substantive ``reason=`` (not empty, ``TODO``, ``skip``, or
  ``todo``);
- ``pytest.mark.xfail`` / ``pytest.xfail`` must carry a ``reason=`` that
  references an issue (``#NNN``) and, for marker form, declare
  ``strict=True``;
- broad ``# pragma: no cover`` directives must include an inline
  justification comment (``# pragma: no cover - <reason>``).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.proof.suppressions import Suppression, load_suppressions, validate_suppressions

ROOT = _get_root()
REGISTRY = ROOT / "docs" / "plans" / "suppressions.yaml"
SCAN_DIRS = ("polylogue", "tests", "devtools")
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("noqa", re.compile(r"#\s*noqa(?::|\b)")),
    ("type_ignore", re.compile(r"#\s*type:\s*ignore(?:\[[^]]+\])?")),
    ("no_cover", re.compile(r"pragma:\s*no cover|coverage:\s*ignore")),
)

# Reasons that are too generic to count as substantive on a skip/xfail.
_NON_SUBSTANTIVE_REASONS = frozenset({"", "todo", "skip", "fixme", "xxx", "tbd"})

# Issue link pattern: ``#NNN`` anywhere inside the reason text.
_ISSUE_REFERENCE_PATTERN = re.compile(r"#\d+")


@dataclass(frozen=True, slots=True)
class DiscoveredSuppression:
    kind: str
    path: str
    line: int
    text: str
    registered: bool
    discipline_errors: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=REGISTRY)
    p.add_argument("--json", action="store_true")
    p.add_argument("--scan-root", type=Path, default=ROOT)
    p.add_argument(
        "--enforce-discovered",
        action="store_true",
        help="Block on discovered source suppressions not covered by registry paths.",
    )
    p.add_argument(
        "--enforce-kinds",
        default="",
        help="Comma-separated suppression kinds to enforce registration for (e.g. type_ignore,noqa).",
    )
    args = p.parse_args(argv)

    enforce_kinds: frozenset[str] = frozenset(k.strip() for k in args.enforce_kinds.split(",") if k.strip())

    suppressions = load_suppressions(registry=args.yaml)
    errors = validate_suppressions(suppressions)
    discovered = discover_source_suppressions(args.scan_root, suppressions=suppressions)
    unregistered = [item for item in discovered if not item.registered]
    unregistered_enforced = [item for item in unregistered if item.kind in enforce_kinds] if enforce_kinds else []
    discipline_violations = [item for item in discovered if item.discipline_errors]
    blocking = (
        bool(errors)
        or (args.enforce_discovered and bool(unregistered))
        or bool(unregistered_enforced)
        or bool(discipline_violations)
    )

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "expired": errors,
                "total": len(suppressions),
                "discovered_total": len(discovered),
                "discovered_by_kind": dict(Counter(item.kind for item in discovered)),
                "unregistered_total": len(unregistered),
                "unregistered_enforced_total": len(unregistered_enforced),
                "discipline_violations_total": len(discipline_violations),
                "enforce_kinds": sorted(enforce_kinds),
                "unregistered": [item.to_payload() for item in unregistered],
                "discipline_violations": [item.to_payload() for item in discipline_violations],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if errors:
            for error in errors:
                print(f"[BLOCK] {error}")
        else:
            print(f"verify-suppressions: all {len(suppressions)} suppressions current")
        if discovered:
            print(f"discovered source suppressions: {len(discovered)}")
            for kind, count in sorted(Counter(item.kind for item in discovered).items()):
                print(f"    {kind}: {count}")
        if discipline_violations:
            print(f"[BLOCK] suppression discipline violations: {len(discipline_violations)}")
            for item in discipline_violations[:20]:
                joined = "; ".join(item.discipline_errors)
                print(f"    {item.path}:{item.line}: {item.kind}: {joined}")
            if len(discipline_violations) > 20:
                print(f"    ... {len(discipline_violations) - 20} more")
        if unregistered_enforced:
            print(f"[BLOCK] unregistered {sorted(enforce_kinds)} suppressions: {len(unregistered_enforced)}")
            for item in unregistered_enforced[:20]:
                print(f"    {item.path}:{item.line}: {item.kind}: {item.text}")
            if len(unregistered_enforced) > 20:
                print(f"    ... {len(unregistered_enforced) - 20} more")
        elif unregistered:
            label = "[BLOCK]" if args.enforce_discovered else "[warn]"
            print(f"{label} unregistered source suppressions: {len(unregistered)}")
            for item in unregistered[:20]:
                print(f"    {item.path}:{item.line}: {item.kind}: {item.text}")
            if len(unregistered) > 20:
                print(f"    ... {len(unregistered) - 20} more")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


def discover_source_suppressions(
    root: Path = ROOT,
    *,
    suppressions: list[Suppression] | None = None,
) -> list[DiscoveredSuppression]:
    """Discover source-level suppression and exception forms."""
    registry_paths = _registry_paths(suppressions or [])
    results: list[DiscoveredSuppression] = []
    for path in _iter_scan_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.splitlines()
        for kind, line, errors in _pytest_suppression_lines(text):
            results.append(
                DiscoveredSuppression(
                    kind=kind,
                    path=rel,
                    line=line,
                    text=_line_text(lines, line),
                    registered=_path_registered(rel, registry_paths),
                    discipline_errors=tuple(errors),
                )
            )
        for kind, line, comment, errors in _comment_suppression_lines(text):
            results.append(
                DiscoveredSuppression(
                    kind=kind,
                    path=rel,
                    line=line,
                    text=comment.strip(),
                    registered=_path_registered(rel, registry_paths),
                    discipline_errors=tuple(errors),
                )
            )
    return results


def _pytest_suppression_lines(text: str) -> list[tuple[str, int, list[str]]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, RecursionError, ValueError, MemoryError):
        return []
    results: list[tuple[str, int, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kind = _pytest_call_kind(node.func)
            if kind is not None:
                errors = _pytest_call_discipline_errors(kind, node)
                results.append((kind, node.lineno, errors))
    return results


def _pytest_call_discipline_errors(kind: str, node: ast.Call) -> list[str]:
    """Return discipline errors for a pytest skip/xfail call."""
    errors: list[str] = []
    reason = _extract_reason(node)
    if kind in ("pytest_skip", "pytest_skip_marker"):
        if not _is_substantive_reason(reason):
            errors.append("pytest skip requires substantive reason=")
    elif kind == "pytest_xfail":
        if not _is_substantive_reason(reason):
            errors.append("pytest xfail requires substantive reason=")
        elif not _ISSUE_REFERENCE_PATTERN.search(reason or ""):
            errors.append("pytest xfail reason must reference an issue (#NNN)")
        if _is_marker_xfail(node) and not _has_strict_true(node):
            errors.append("pytest.mark.xfail must declare strict=True")
    return errors


def _extract_reason(node: ast.Call) -> str | None:
    """Return the textual value of a ``reason=`` keyword, if any.

    The first positional argument also acts as the reason for ``pytest.skip``
    and ``pytest.xfail`` (per pytest's API), so it is treated as a reason
    when present. Both literal strings and f-strings (``JoinedStr``) are
    recognised, since both express runtime reason content.
    """
    for kw in node.keywords:
        if kw.arg == "reason":
            value = _reason_value(kw.value)
            if value is not None:
                return value
    if node.args:
        return _reason_value(node.args[0])
    return None


def _reason_value(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        # f-string: concatenate the literal parts and a placeholder for each
        # substitution so generic patterns like f"{x}" still register as a
        # non-substantive reason while informative literals dominate.
        parts: list[str] = []
        for value in expr.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{...}")
        joined = "".join(parts).strip()
        return joined or None
    return None


def _is_substantive_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return reason.strip().lower() not in _NON_SUBSTANTIVE_REASONS


def _is_marker_xfail(node: ast.Call) -> bool:
    return _attribute_parts(node.func) == ["pytest", "mark", "xfail"]


def _has_strict_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _pytest_call_kind(func: ast.expr) -> str | None:
    parts = _attribute_parts(func)
    match parts:
        case ["pytest", "skip"]:
            return "pytest_skip"
        case ["pytest", "xfail"]:
            return "pytest_xfail"
        case ["pytest", "mark", "xfail"]:
            return "pytest_xfail"
        case ["pytest", "mark", "skip" | "skipif"]:
            return "pytest_skip_marker"
        case _:
            return None


def _attribute_parts(expr: ast.expr) -> list[str]:
    if isinstance(expr, ast.Name):
        return [expr.id]
    if isinstance(expr, ast.Attribute):
        return [*_attribute_parts(expr.value), expr.attr]
    return []


def _comment_suppression_lines(text: str) -> list[tuple[str, int, str, list[str]]]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        comments = [(token.start[0], token.string) for token in tokens if token.type == tokenize.COMMENT]
    except tokenize.TokenError:
        return []
    results: list[tuple[str, int, str, list[str]]] = []
    for line, comment in comments:
        for kind, pattern in _PATTERNS:
            if pattern.search(comment):
                errors = _comment_discipline_errors(kind, comment)
                results.append((kind, line, comment, errors))
    return results


_BARE_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?!\[)")
_NO_COVER_WITH_JUSTIFICATION = re.compile(r"pragma:\s*no cover\s*[-\u2013\u2014#:]\s*\S")


def _comment_discipline_errors(kind: str, comment: str) -> list[str]:
    """Return discipline errors for a suppression comment."""
    errors: list[str] = []
    if kind == "type_ignore" and _BARE_TYPE_IGNORE.search(comment):
        errors.append("bare '# type: ignore' must use bracketed '[error-code]'")
    if kind == "no_cover" and not _NO_COVER_WITH_JUSTIFICATION.search(comment):
        errors.append("'# pragma: no cover' must include inline justification (e.g. '- defensive')")
    return errors


def _line_text(lines: list[str], line: int) -> str:
    index = line - 1
    if 0 <= index < len(lines):
        return lines[index].strip()
    return ""


def _iter_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirname in SCAN_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


def _registry_paths(suppressions: list[Suppression]) -> tuple[str, ...]:
    paths: list[str] = []
    for suppression in suppressions:
        paths.extend(suppression.paths)
    return tuple(paths)


def _path_registered(path: str, registry_paths: tuple[str, ...]) -> bool:
    return any(path == registered or fnmatch(path, registered) for registered in registry_paths)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
