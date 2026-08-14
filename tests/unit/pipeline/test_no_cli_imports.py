"""Pipeline must not import from CLI surfaces (#430).

Topology rule: surfaces present existing meaning; substrate owns it. A
pipeline module importing from ``polylogue.cli.*`` inverts that rule and
creates a substrate↔surface cycle that ``devtools verify-cluster-cohesion``
flags.
"""

from __future__ import annotations

import ast
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[3] / "polylogue" / "pipeline"


def _find_cli_imports(source: str, *, filename: str = "<source>") -> list[str]:
    imports: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name.startswith("polylogue.cli"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("polylogue.cli"):
            imports.append(node.module)
    return imports


def test_pipeline_does_not_import_cli() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(PIPELINE_ROOT.rglob("*.py")):
        rel = path.relative_to(PIPELINE_ROOT.parents[1])
        text = path.read_text(encoding="utf-8")
        violations = _find_cli_imports(text, filename=str(path))
        if violations:
            offenders[str(rel)] = violations
    assert not offenders, "polylogue/pipeline/* must not import from polylogue/cli/*; offenders: " + repr(offenders)


def test_pipeline_cli_import_analyzer_handles_aliases_and_multiline_imports() -> None:
    planted = """
import polylogue.cli.archive_query as query_surface
from polylogue.cli.commands import (
    diagnostics as diagnostics_surface,
)
"""

    assert _find_cli_imports(planted) == ["polylogue.cli.archive_query", "polylogue.cli.commands"]
