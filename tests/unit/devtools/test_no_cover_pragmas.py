"""Backstop discipline test for ``# pragma: no cover`` (#1062).

Mirrors the verify-suppressions scanner's classifier so the codebase keeps
inline justification on every coverage pragma, independent of whether the
scanner is wired into a particular CI matrix entry.
"""

from __future__ import annotations

import re

from devtools import repo_root

ROOT = repo_root()
SCAN_DIRS = ("polylogue", "tests", "devtools")

_PRAGMA_PATTERN = re.compile(r"pragma:\s*no cover")
# Match the exact justification grammar the scanner enforces. Allow ASCII
# hyphen, en-dash, em-dash, hash, or colon as the separator.
_JUSTIFIED_PRAGMA = re.compile(r"pragma:\s*no cover\s*[-\u2013\u2014#:]\s*\S")


def test_no_cover_pragmas_carry_justification() -> None:
    """Every ``# pragma: no cover`` must be followed by inline rationale.

    Acceptable forms include ``# pragma: no cover - defensive``,
    ``# pragma: no cover  # script entry``, or
    ``# pragma: no cover \u2014 returns RootModeRequest``.
    """
    offenders: list[str] = []
    for path in sorted(
        p
        for dirname in SCAN_DIRS
        if (ROOT / dirname).exists()
        for p in (ROOT / dirname).rglob("*.py")
        if "__pycache__" not in p.parts
    ):
        # Skip this test itself (the pattern appears inside string literals).
        if path.name in {"test_no_cover_pragmas.py", "test_verify_suppressions.py", "verify_suppressions.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PRAGMA_PATTERN.search(line) and not _JUSTIFIED_PRAGMA.search(line):
                rel = path.relative_to(ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, "'# pragma: no cover' directives without inline justification:\n  " + "\n  ".join(offenders)
