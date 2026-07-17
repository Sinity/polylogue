"""Architecture guard for the single controlled archive-read boundary."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
READ_SURFACES = (
    "polylogue/api/archive.py",
    "polylogue/api/insights.py",
    "polylogue/annotations/join.py",
    "polylogue/archive/query/archive_execution.py",
    "polylogue/archive/query/search_hits.py",
    "polylogue/cli/archive_query.py",
    "polylogue/cli/commands/diagnostics.py",
    "polylogue/cli/commands/maintenance/_archive_read.py",
    "polylogue/cli/read_views/standard.py",
    "polylogue/cli/shell_completion_values.py",
    "polylogue/daemon/http.py",
    "polylogue/demo/verify.py",
    "polylogue/mcp/archive_support.py",
    "polylogue/mcp/server_prompts.py",
)


def _direct_archive_open_lines(path: Path) -> list[tuple[int, bool | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[tuple[int, bool | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "open_existing" or not isinstance(node.func.value, ast.Attribute):
            continue
        if node.func.value.attr != "ArchiveStore":
            continue
        read_only: bool | None = None
        for keyword in node.keywords:
            if keyword.arg == "read_only" and isinstance(keyword.value, ast.Constant):
                read_only = keyword.value.value if isinstance(keyword.value.value, bool) else None
        lines.append((node.lineno, read_only))
    return lines


def test_read_surface_direct_opens_are_absent_or_explicit_writer_paths() -> None:
    """A read adapter must not silently bypass admission/deadline/cancellation."""

    violations: list[str] = []
    for relative_path in READ_SURFACES:
        path = REPO_ROOT / relative_path
        for line, read_only in _direct_archive_open_lines(path):
            if read_only is not False:
                violations.append(f"{relative_path}:{line} read_only={read_only!r}")

    assert not violations, "direct archive opens must be controlled reads or explicit writer paths: " + ", ".join(
        violations
    )
