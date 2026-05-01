"""Verify each production module has at least one test that imports it.

Walks ``polylogue/**/*.py`` and ``tests/unit/**/test_*.py``. For each
production module, looks for tests that import it (directly via
``polylogue.<dotted>`` or ``from polylogue.<dotted> import …``). Reports:

  * uncovered: production module not imported by any unit test.
  * orphan_tests: test files that don't import any production module
    (typically infrastructure or fixtures, listed in `shared:`).

Modules that legitimately do not require unit tests (entry points,
re-export shims, generated code) are listed in ``untested:`` of the
manifest with a justification.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "plans" / "test-ownership.yaml"


def parse_yaml(text: str) -> dict[str, list[dict[str, str]]]:
    """Tiny YAML reader for the test-ownership schema."""
    sections: dict[str, list[dict[str, str]]] = {"untested": [], "shared": []}
    current_section: str | None = None
    current_item: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            if key in sections:
                if current_item is not None and current_section is not None:
                    sections[current_section].append(current_item)
                current_section = key
                current_item = None
            continue
        if current_section is None:
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current_item is not None:
                sections[current_section].append(current_item)
            current_item = {}
            rest = stripped[2:]
            if ": " in rest:
                k, _, v = rest.partition(": ")
                current_item[k] = _coerce(v.strip())
        elif current_item is not None and ": " in stripped:
            k, _, v = stripped.partition(": ")
            current_item[k] = _coerce(v.strip())
    if current_item is not None and current_section is not None:
        sections[current_section].append(current_item)
    return sections


def _coerce(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def production_modules() -> list[str]:
    out: list[str] = []
    for root_dir in ("polylogue", "devtools"):
        for p in (ROOT / root_dir).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel.endswith("/__init__.py"):
                continue
            if rel.endswith("/__main__.py"):
                continue
            out.append(rel)
    return sorted(out)


def test_files() -> list[Path]:
    out: list[Path] = []
    for p in (ROOT / "tests" / "unit").rglob("test_*.py"):
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return sorted(out)


PRODUCTION_PREFIXES = ("polylogue", "devtools")


def imports_in(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(prefix) for prefix in PRODUCTION_PREFIXES):
                    found.add(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and any(node.module.startswith(prefix) for prefix in PRODUCTION_PREFIXES)
        ):
            found.add(node.module)
    return found


def production_module_names() -> dict[str, str]:
    """Map dotted module name → relative path."""
    out: dict[str, str] = {}
    for rel in production_modules():
        dotted = rel[: -len(".py")].replace("/", ".")
        out[dotted] = rel
    # Also map package __init__ paths to their package dotted name.
    for root_dir in ("polylogue", "devtools"):
        for p in (ROOT / root_dir).rglob("__init__.py"):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(ROOT).as_posix()
            dotted = rel[: -len("/__init__.py")].replace("/", ".")
            out[dotted] = rel
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=MANIFEST)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    manifest = parse_yaml(args.yaml.read_text())
    untested_paths = {entry["path"] for entry in manifest["untested"]}
    shared_paths = {entry["path"] for entry in manifest["shared"]}

    name_to_path = production_module_names()
    coverage: dict[str, set[str]] = defaultdict(set)
    test_imports_count: dict[str, int] = {}

    for tf in test_files():
        rel = tf.relative_to(ROOT).as_posix()
        if rel in shared_paths:
            continue
        imports = imports_in(tf)
        production_imports = {name_to_path[n] for n in imports if n in name_to_path}
        # Also resolve sub-module imports: e.g. import polylogue.lib triggers any descendant.
        for imp in imports:
            for name, prod_rel in name_to_path.items():
                if imp == name or imp.startswith(name + "."):
                    production_imports.add(prod_rel)
                if name.startswith(imp + "."):
                    production_imports.add(prod_rel)
        test_imports_count[rel] = len(production_imports)
        for prod in production_imports:
            coverage[prod].add(rel)

    uncovered = sorted(set(production_modules()) - set(coverage.keys()) - untested_paths)
    stale_untested = sorted(untested_paths - set(production_modules()))
    orphan_tests = sorted(rel for rel, count in test_imports_count.items() if count == 0)

    blocking = bool(uncovered) or bool(stale_untested)

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "counts": {
                    "production_modules": len(production_modules()),
                    "covered": len(coverage),
                    "uncovered": len(uncovered),
                    "untested_declared": len(untested_paths),
                    "stale_untested": len(stale_untested),
                    "orphan_tests": len(orphan_tests),
                },
                "uncovered": uncovered,
                "stale_untested": stale_untested,
                "orphan_tests": orphan_tests,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        prod_total = len(production_modules())
        print(f"production modules: {prod_total}")
        print(f"covered: {len(coverage)}")
        print(f"untested (declared): {len(untested_paths)}")
        if uncovered:
            print(f"[BLOCK] uncovered: {len(uncovered)}")
            for u in uncovered[:25]:
                print(f"    {u}")
            if len(uncovered) > 25:
                print(f"    ... and {len(uncovered) - 25} more")
        if stale_untested:
            print(f"[BLOCK] stale untested entries (file no longer present): {len(stale_untested)}")
            for s in stale_untested:
                print(f"    {s}")
        if orphan_tests:
            print(f"[warn] orphan tests (no production import): {len(orphan_tests)}")
            for o in orphan_tests[:10]:
                print(f"    {o}")
            if len(orphan_tests) > 10:
                print(f"    ... and {len(orphan_tests) - 10} more")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
