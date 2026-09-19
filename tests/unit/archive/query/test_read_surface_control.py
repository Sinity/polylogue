"""Architecture guard for the single controlled archive-read boundary."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
CONTROLLED_READER = "polylogue/archive/query/execution_control.py"
CONTROLLED_OPERATION_READ = ("polylogue/operations/operation_context.py", "open_operation_read")


def _direct_archive_open_lines(path: Path) -> list[tuple[int, bool | None, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[tuple[int, bool | None, str | None]] = []

    class ArchiveOpenVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_names: list[str] = []
            super().__init__()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_names.append(node.name)
            self.generic_visit(node)
            self.function_names.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_names.append(node.name)
            self.generic_visit(node)
            self.function_names.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "open_existing"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ArchiveStore"
            ):
                read_only: bool | None = None
                for keyword in node.keywords:
                    if keyword.arg == "read_only" and isinstance(keyword.value, ast.Constant):
                        read_only = keyword.value.value if isinstance(keyword.value.value, bool) else None
                lines.append((node.lineno, read_only, self.function_names[-1] if self.function_names else None))
            self.generic_visit(node)

    ArchiveOpenVisitor().visit(tree)
    return lines


def _is_declared_controlled_reader(relative_path: str, function_name: str | None) -> bool:
    return relative_path == CONTROLLED_READER or (relative_path, function_name) == CONTROLLED_OPERATION_READ


def test_operation_context_exemption_names_only_the_snapshot_boundary() -> None:
    """Mutation: exempt the whole module and an accidental sibling open passes the census."""

    path, function_name = CONTROLLED_OPERATION_READ
    assert _is_declared_controlled_reader(path, function_name)
    assert not _is_declared_controlled_reader(path, "unbounded_operation_read")


def test_all_direct_archive_opens_use_the_controlled_reader_or_explicit_write_mode() -> None:
    """Every production module participates; new adapters cannot evade the boundary."""

    violations: list[str] = []
    controlled_reader_opens = 0
    for path in sorted((REPO_ROOT / "polylogue").rglob("*.py")):
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for line, read_only, function_name in _direct_archive_open_lines(path):
            if _is_declared_controlled_reader(relative_path, function_name):
                controlled_reader_opens += 1
                continue
            if read_only is not False:
                violations.append(f"{relative_path}:{line} ({function_name}) read_only={read_only!r}")

    assert controlled_reader_opens > 0, "the controlled reader must own the read-only ArchiveStore open"
    assert not violations, "direct archive opens must be controlled reads or explicit writer paths: " + ", ".join(
        violations
    )
