"""Enforce per-file LOC budgets declared in docs/plans/file-size-budgets.yaml.

Reports overages with the relevant file and its budget. Exceptions are explicit
per-file ceilings for known large files; they still block further growth, and
stale exceptions are reported when the file disappears.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ROOT / "docs" / "plans" / "file-size-budgets.yaml"


def parse_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML reader for the file-size-budgets schema.

    Schema is fixed: top-level keys ``defaults``, ``per_package``, ``exceptions``.
    Avoids a PyYAML dependency.
    """
    out: dict[str, Any] = {"defaults": {}, "per_package": {}, "exceptions": []}
    section: str | None = None
    pkg_key: str | None = None
    current_exc: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            if key in {"defaults", "per_package", "exceptions"}:
                section = key
                pkg_key = None
                current_exc = None
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if section == "defaults" and indent == 2 and ": " in stripped:
            k, _, v = stripped.partition(": ")
            out["defaults"][k] = int(v)
        elif section == "per_package":
            if indent == 2 and stripped.endswith(":"):
                pkg_key = stripped.rstrip(":")
                out["per_package"][pkg_key] = {}
            elif indent == 4 and pkg_key and ": " in stripped:
                k, _, v = stripped.partition(": ")
                out["per_package"][pkg_key][k] = int(v)
        elif section == "exceptions":
            if stripped.startswith("- "):
                if current_exc is not None:
                    out["exceptions"].append(current_exc)
                current_exc = {}
                rest = stripped[2:]
                if ": " in rest:
                    k, _, v = rest.partition(": ")
                    current_exc[k] = _coerce(v.strip())
            elif current_exc is not None and ": " in stripped:
                k, _, v = stripped.partition(": ")
                current_exc[k] = _coerce(v.strip())
    if current_exc is not None:
        out["exceptions"].append(current_exc)
    return out


def _coerce(value: str) -> int | str:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.read_text().splitlines())
    except OSError:
        return 0


def is_test(rel: str) -> bool:
    return rel.startswith("tests/")


def budget_for(rel: str, budgets: dict[str, Any]) -> tuple[int, str]:
    """Resolve the ceiling for a path. Returns (ceiling, source_label)."""
    # Exceptions win.
    for exc in budgets["exceptions"]:
        if exc.get("path") == rel:
            return int(exc["ceiling"]), "exception"
    # Per-package ceilings (longest prefix wins).
    pkg_ceiling: tuple[int, str] | None = None
    for pkg_prefix, settings in sorted(budgets["per_package"].items(), key=lambda x: -len(x[0])):
        if rel.startswith(pkg_prefix):
            key = "test_loc_ceiling" if is_test(rel) else "source_loc_ceiling"
            if key in settings:
                pkg_ceiling = (int(settings[key]), f"per_package[{pkg_prefix}]")
                break
    if pkg_ceiling is not None:
        return pkg_ceiling
    # Defaults.
    key = "test_loc_ceiling" if is_test(rel) else "source_loc_ceiling"
    return int(budgets["defaults"][key]), f"defaults.{key}"


def walk_files() -> Iterable[Path]:
    for d in ("polylogue", "devtools", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=BUDGETS)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    budgets = parse_yaml(args.yaml.read_text())
    overages: list[dict[str, Any]] = []
    declared_exceptions = {exc["path"] for exc in budgets["exceptions"]}
    used_exceptions: set[str] = set()

    for path in walk_files():
        rel = path.relative_to(ROOT).as_posix()
        ceiling, source = budget_for(rel, budgets)
        n = loc(path)
        if n > ceiling:
            overages.append({"path": rel, "loc": n, "ceiling": ceiling, "source": source})
        if rel in declared_exceptions:
            used_exceptions.add(rel)

    stale = sorted(declared_exceptions - used_exceptions)

    blocking = bool(overages)

    if args.json:
        json.dump({"blocking": blocking, "overages": overages, "stale_exceptions": stale}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if overages:
            print(f"[BLOCK] file-size overages: {len(overages)}")
            for o in overages:
                print(f"    {o['path']}: {o['loc']} > {o['ceiling']} ({o['source']})")
        if stale:
            print(f"[warn] stale exceptions (file no longer present): {len(stale)}")
            for s in stale:
                print(f"    {s}")
        if not overages and not stale:
            print("file-size budgets: clean")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
