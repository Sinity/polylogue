"""Architecture guard for the single controlled archive-read boundary."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
CONTROLLED_READER = "polylogue/archive/query/execution_control.py"


def _direct_archive_open_lines(path: Path) -> list[tuple[int, bool | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[tuple[int, bool | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "open_existing" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "ArchiveStore":
            continue
        read_only: bool | None = None
        for keyword in node.keywords:
            if keyword.arg == "read_only" and isinstance(keyword.value, ast.Constant):
                read_only = keyword.value.value if isinstance(keyword.value.value, bool) else None
        lines.append((node.lineno, read_only))
    return lines


def test_all_direct_archive_opens_use_the_controlled_reader_or_explicit_write_mode() -> None:
    """Every production module participates; new adapters cannot evade the boundary."""

    violations: list[str] = []
    controlled_reader_opens = 0
    for path in sorted((REPO_ROOT / "polylogue").rglob("*.py")):
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for line, read_only in _direct_archive_open_lines(path):
            if relative_path == CONTROLLED_READER:
                controlled_reader_opens += 1
                continue
            if read_only is not False:
                violations.append(f"{relative_path}:{line} read_only={read_only!r}")

    assert controlled_reader_opens > 0, "the controlled reader must own the read-only ArchiveStore open"
    assert not violations, "direct archive opens must be controlled reads or explicit writer paths: " + ", ".join(
        violations
    )
