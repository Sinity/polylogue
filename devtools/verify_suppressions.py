"""Inspect and enforce suppression registry expiry dates.

Every suppression in ``docs/plans/suppressions.yaml`` must have an expiry
date. The lint fails when any suppression is past its expiry date, forcing
review and either renewal or removal.

The command also discovers source-level exception mechanisms so an empty
registry cannot be mistaken for an absence of suppressions.
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
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class DiscoveredSuppression:
    kind: str
    path: str
    line: int
    text: str
    registered: bool

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
    args = p.parse_args(argv)

    suppressions = load_suppressions(registry=args.yaml)
    errors = validate_suppressions(suppressions)
    discovered = discover_source_suppressions(args.scan_root, suppressions=suppressions)
    unregistered = [item for item in discovered if not item.registered]
    blocking = bool(errors) or (args.enforce_discovered and bool(unregistered))

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "expired": errors,
                "total": len(suppressions),
                "discovered_total": len(discovered),
                "discovered_by_kind": dict(Counter(item.kind for item in discovered)),
                "unregistered_total": len(unregistered),
                "unregistered": [item.to_payload() for item in unregistered],
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
        if unregistered:
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
        results.extend(
            DiscoveredSuppression(
                kind=kind,
                path=rel,
                line=line,
                text=_line_text(lines, line),
                registered=_path_registered(rel, registry_paths),
            )
            for kind, line in _pytest_suppression_lines(text)
        )
        results.extend(
            DiscoveredSuppression(
                kind=kind,
                path=rel,
                line=line,
                text=comment.strip(),
                registered=_path_registered(rel, registry_paths),
            )
            for kind, line, comment in _comment_suppression_lines(text)
        )
    return results


def _pytest_suppression_lines(text: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kind = _pytest_call_kind(node.func)
            if kind is not None:
                results.append((kind, node.lineno))
    return results


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


def _comment_suppression_lines(text: str) -> list[tuple[str, int, str]]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        comments = [(token.start[0], token.string) for token in tokens if token.type == tokenize.COMMENT]
    except tokenize.TokenError:
        return []
    results: list[tuple[str, int, str]] = []
    for line, comment in comments:
        for kind, pattern in _PATTERNS:
            if pattern.search(comment):
                results.append((kind, line, comment))
    return results


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
