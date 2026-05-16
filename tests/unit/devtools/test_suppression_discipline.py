"""Backstop discipline tests for native suppression enforcement (#1062).

These tests pin the load-bearing invariants the verify-suppressions scanner
enforces, but read the codebase directly so a regression cannot hide behind
a configuration change.
"""

from __future__ import annotations

import re
from pathlib import Path

from devtools import repo_root

ROOT = repo_root()
SCAN_DIRS = ("polylogue", "tests", "devtools")
_BARE_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?!\[)")


def _iter_python_sources() -> list[Path]:
    files: list[Path] = []
    for dirname in SCAN_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


# Files that legitimately mention the pattern inside string literals (the
# scanner implementation itself and the tests that exercise it).
_PATTERN_AUTHORITIES = frozenset(
    {
        "devtools/verify_suppressions.py",
        "tests/unit/devtools/test_verify_suppressions.py",
        "tests/unit/devtools/test_suppression_discipline.py",
        "tests/unit/devtools/test_no_cover_pragmas.py",
    }
)


def test_no_bare_type_ignore_directives() -> None:
    """Every ``# type: ignore`` must carry a bracketed error code.

    This is the load-bearing claim for ``enable_error_code =
    ["ignore-without-code"]`` in ``[tool.mypy]``; the mypy gate fails
    locally, but this test is a grep-level backstop that catches the
    pattern even if mypy is temporarily disabled in a CI matrix entry.
    """
    offenders: list[str] = []
    for path in _iter_python_sources():
        rel = path.relative_to(ROOT).as_posix()
        if rel in _PATTERN_AUTHORITIES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BARE_TYPE_IGNORE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "bare '# type: ignore' directives found; require bracketed [error-code]:\n  " + "\n  ".join(
        offenders
    )


def test_pyproject_enables_ignore_without_code() -> None:
    """``[tool.mypy].enable_error_code`` must include ``ignore-without-code``."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Locate the [tool.mypy] section.
    mypy_section = pyproject.split("[tool.mypy]", 1)
    assert len(mypy_section) == 2, "expected [tool.mypy] section in pyproject.toml"
    # Cut to next table header.
    body = mypy_section[1].split("\n[tool.", 1)[0]
    assert "enable_error_code" in body, "[tool.mypy] must declare enable_error_code"
    assert "ignore-without-code" in body, (
        "[tool.mypy].enable_error_code must include 'ignore-without-code' so "
        "bare '# type: ignore' is rejected by the mypy gate"
    )
