"""Verify every production module >150 AST statement lines has a matching test or exemption.

Walks ``polylogue/*.py`` (and subdirectories), counts non-blank
non-comment lines, and for each module with >50 substantive lines checks
that either:

  1. A matching test file exists under ``tests/unit/``
     (e.g. ``polylogue/core/hashing.py`` → ``tests/unit/core/test_hashing.py``),
     or
  2. The module path is listed with a rationale in
     ``docs/plans/test-coverage-exemptions.yaml``.

The line count uses AST-parsed statements + docstrings, not raw file lines,
so module docstrings and imports don't inflate the count.

Returns 0 when all contracts are satisfied, 1 when violations exist.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()
EXEMPTIONS_PATH = ROOT / "docs" / "plans" / "test-coverage-exemptions.yaml"
LINE_THRESHOLD = 150


# ── line counting ────────────────────────────────────────────────────


def _ast_stmt_lines(path: Path) -> int:
    """Count AST-level statement lines (non-blank, non-comment, non-import-only).

    We count docstrings, class/function bodies, and standalone expressions.
    Pure-import modules (re-export shims) land at 0-2 lines.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return 9999  # unparseable → flag it

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Docstring — count it.
            count += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Each function/class definition counts as one structured line.
            count += 1
            # Count body statements.
            for stmt in node.body:
                if (
                    not isinstance(stmt, ast.Expr)
                    or not isinstance(stmt.value, ast.Constant)
                    or not isinstance(stmt.value.value, str)
                ):
                    count += 1
    # Floor: module-level statements count but we want a lower bound.
    # If the module has a docstring + __future__ + a few imports, it
    # may still have substantive code in functions that we counted above.
    return count


def _production_modules() -> Iterator[Path]:
    """Yield all ``.py`` files under polylogue/."""
    for py_file in sorted(ROOT.glob("polylogue/**/*.py")):
        if py_file.name == "__init__.py":
            init_dir = py_file.parent
            # Skip directories that contain only __init__.py
            siblings = list(init_dir.glob("*.py"))
            if len(siblings) == 1:
                continue
        yield py_file


def _test_for_module(module_path: Path) -> Path | None:
    """Return the expected test path for a production module.

    ``polylogue/core/hashing.py`` → ``tests/unit/core/test_hashing.py``.
    """
    rel = module_path.relative_to(ROOT / "polylogue")
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
        if not parts:
            return None
    else:
        parts[-1] = parts[-1].replace(".py", "")
        parts[-1] = f"test_{parts[-1]}.py"
    candidate = ROOT / "tests" / "unit" / Path(*parts)
    if candidate.exists():
        return candidate

    flat_archive_tiers = _flat_archive_tiers_test_for_module(rel)
    if flat_archive_tiers is not None and flat_archive_tiers.exists():
        return flat_archive_tiers
    flat_parser = _flat_parser_test_for_module(rel)
    if flat_parser is not None and flat_parser.exists():
        return flat_parser
    return None


def _flat_archive_tiers_test_for_module(rel: Path) -> Path | None:
    """Return the flat archive-tier storage test path for consolidated helper tests."""
    parts = rel.parts
    if len(parts) != 4 or parts[:3] != ("storage", "sqlite", "archive_tiers"):
        return None
    stem = parts[-1].removesuffix(".py")
    return ROOT / "tests" / "unit" / "storage" / f"test_archive_tiers_{stem}.py"


def _flat_parser_test_for_module(rel: Path) -> Path | None:
    """Return the flat parser test path for parser modules with consolidated tests."""
    parts = rel.parts
    if len(parts) != 3 or parts[:2] != ("sources", "parsers"):
        return None
    stem = parts[-1].removesuffix(".py")
    return ROOT / "tests" / "unit" / "sources" / f"test_parsers_{stem}.py"


# ── exemptions ───────────────────────────────────────────────────────


def _parse_exemptions(path: Path) -> dict[str, str]:
    """Parse a simple ``- path: reason`` YAML list.

    The file MUST exist (created by the operator). Returns ``{path: reason}``.
    """
    if not path.exists():
        # If the file genuinely does not exist yet, create it with a
        # bootstrap message. But we report every module as uncovered
        # so the operator seeds the exemptions deliberately.
        return {}

    entries: dict[str, str] = {}
    current_path: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            rest = stripped[2:]
            if ": " in rest:
                key, _, val = rest.partition(": ")
                if key == "path":
                    # New exemption entry — save previous before starting new.
                    current_path = val.strip()
                elif key == "reason" and current_path is not None:
                    entries[current_path] = val.strip()
        elif current_path is not None and ": " in stripped:
            key, _, val = stripped.partition(": ")
            if key == "reason":
                entries[current_path] = val.strip()
    return entries


# ── main ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify test coverage contracts")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--threshold", type=int, default=LINE_THRESHOLD, help=f"Line threshold (default: {LINE_THRESHOLD})"
    )
    args = parser.parse_args(argv)

    exemptions = _parse_exemptions(EXEMPTIONS_PATH)
    uncovered: list[dict[str, str]] = []

    for module_path in _production_modules():
        stmt_lines = _ast_stmt_lines(module_path)
        if stmt_lines <= args.threshold:
            continue

        rel = str(module_path.relative_to(ROOT))
        if rel in exemptions:
            continue

        test = _test_for_module(module_path)
        if test is None:
            uncovered.append(
                {
                    "path": rel,
                    "lines": str(stmt_lines),
                    "issue": "no matching test file",
                }
            )

    # Report
    blocking = len(uncovered) > 0

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "threshold": args.threshold,
                "uncovered": uncovered,
                "exemption_count": len(exemptions),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        total_checked = sum(1 for m in _production_modules() if _ast_stmt_lines(m) > args.threshold)
        print(f"threshold: {args.threshold} AST statement lines")
        print(f"checked: {total_checked} modules")
        print(f"exemptions: {len(exemptions)}")
        if uncovered:
            print(f"\n[BLOCK] uncovered: {len(uncovered)}")
            for u in uncovered:
                print(f"  {u['path']} ({u['lines']} lines) — {u['issue']}")
        else:
            print("\nAll covered.")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
