#!/usr/bin/env python3
"""Inventory imports of tests.infra modules across the test tree.

This finds unused or narrowly consumed shared helpers before agents add another
parallel fixture/algebra.  Import count is only a routing signal: a one-consumer
helper can be excellent, and pytest plugin modules may be loaded by string.
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = ROOT / "tests"
INFRA = TESTS / "infra"
OUTPUT = Path(__file__).resolve().parent / "test-infra-consumers.tsv"


def _module_for(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _infra_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.infra"):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("tests.infra"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("tests.infra"):
            # pytest_plugins loads modules by dotted string rather than an
            # import statement. Treat those as consumers too.
            imports.add(node.value)
    return imports


def _nonblank_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    module_paths = {
        _module_for(path): path for path in sorted(INFRA.rglob("*.py")) if not path.name.startswith("test_")
    }
    consumers: dict[str, set[str]] = defaultdict(set)
    for path in sorted(TESTS.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for imported in _infra_imports(path):
            # Attribute imports name the module in ImportFrom. Plain imports may
            # name a child that is not itself a source module; walk upward to the
            # nearest known module.
            candidate = imported
            while candidate not in module_paths and candidate.startswith("tests.infra."):
                candidate = candidate.rsplit(".", 1)[0]
            if candidate in module_paths and module_paths[candidate] != path:
                consumers[candidate].add(relative)

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["module", "physical_loc", "nonblank_loc", "consumer_count", "consumers"])
        rows = []
        for module, path in module_paths.items():
            module_consumers = sorted(consumers[module])
            rows.append(
                (
                    module,
                    len(path.read_text(encoding="utf-8").splitlines()),
                    _nonblank_lines(path),
                    len(module_consumers),
                    ",".join(module_consumers),
                )
            )
        rows.sort(key=lambda row: (int(row[3]), -int(row[2]), str(row[0])))
        writer.writerows(rows)

    zero = [row for row in rows if row[3] == 0]
    one = [row for row in rows if row[3] == 1]
    print(f"modules={len(rows)} zero_consumers={len(zero)} one_consumer={len(one)}")


if __name__ == "__main__":
    main()
