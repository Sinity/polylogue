"""Verify every directory holding collectible test modules is a package.

pytest's default ``prepend`` import mode derives a test module's import name
from the shortest path that is not a package: a module in a directory without
``__init__.py`` is imported under its bare basename.  Two such modules that
share a basename then collide, and pytest aborts the *whole* collection with
"import file mismatch".

The failure is invisible to any selection that touches only one side of a
collision, so it hides until the full corpus is collected as one unit --
exactly the run CLAUDE.md requires and the one nobody completes.  Keeping
every test directory a package makes each module name package-qualified by
path, so a duplicate basename is structurally harmless and this class of
break cannot be reintroduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# Mirrors [tool.pytest.ini_options] python_files, plus conftest.py, which is
# imported by the same machinery.
_COLLECTIBLE = ("test_*.py", "*_test.py", "fuzz_*.py", "conftest.py")


def required_package_dirs(tests_root: Path = TESTS_ROOT) -> list[Path]:
    """Every directory that must carry ``__init__.py``, nearest-first.

    A directory qualifies when it holds a collectible module; each of its
    ancestors up to and including ``tests/`` qualifies too, because a package
    chain broken anywhere above still leaves the module name unqualified.
    """
    required: set[Path] = set()
    for pattern in _COLLECTIBLE:
        for path in tests_root.rglob(pattern):
            if "__pycache__" in path.parts:
                continue
            directory = path.parent
            while True:
                required.add(directory)
                if directory == tests_root:
                    break
                directory = directory.parent
    return sorted(required)


def missing_packages(tests_root: Path = TESTS_ROOT) -> list[Path]:
    return [d for d in required_package_dirs(tests_root) if not (d / "__init__.py").exists()]


def _format_report(missing: list[Path]) -> str:
    if not missing:
        return "test-packages: every directory holding collectible test modules is a package"
    lines = [
        f"test-packages: {len(missing)} test director{'y is' if len(missing) == 1 else 'ies are'} not a package.",
        "",
        "Modules there import under their bare basename, so a duplicate basename",
        "anywhere in the corpus aborts the whole collection with 'import file",
        "mismatch'. Add an empty __init__.py to each:",
        "",
    ]
    lines += [f"  {path.relative_to(REPO_ROOT)}/__init__.py" for path in missing]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    missing = missing_packages()

    if args.json:
        payload = {
            "missing": [str(path.relative_to(REPO_ROOT)) for path in missing],
            "ok": not missing,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(missing))

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
