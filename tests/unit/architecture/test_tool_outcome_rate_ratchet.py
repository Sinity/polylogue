"""Tool-outcome rate ratchet: one gated route for a success rate (cuxz.4 AC4).

``blocks.tool_outcome`` records ``unknown`` as a deliberate retained refusal,
and on the live archive most tool_result rows carry exactly that. A success
or error rate computed over that population describes the classified
minority while reading as a statement about the whole, so the only sanctioned
way to produce one is
:mod:`polylogue.analysis.measurement.outcome_coverage`, which refuses the
bare scalar below :data:`TOOL_OUTCOME_COVERAGE_FLOOR` and reports
``degraded`` with the coverage gap named.

A helper nobody is required to call is advice, not a floor. This ratchet is
what makes the floor binding: any module on a read path that both speaks the
tool-outcome vocabulary and names a rate is a finding unless it is the gated
module itself. The forbidden regrowth is concrete -- a second, ungated place
that divides successes by calls.

Anti-vacuity: add a ``success_rate`` (or ``error_rate``/``failure_rate``/
``pass_rate``) over ``tool_outcome`` to any module under ``polylogue/analysis``,
``polylogue/surfaces``, ``polylogue/cli``, ``polylogue/mcp``, ``polylogue/api``
or ``polylogue/operations`` and this test goes red naming that file. Deleting
the sanctioned module, or stripping its floor, turns
``test_sanctioned_route_still_gates_the_scalar`` red -- so the ratchet cannot
pass by having nothing left to protect.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from polylogue.analysis.measurement.outcome_coverage import (
    TOOL_OUTCOME_COVERAGE_FLOOR,
    build_tool_outcome_aggregate,
)
from polylogue.core.enums import ToolOutcome

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Read paths a misleading scalar could reach a reader from.
READ_PATH_ROOTS = (
    REPO_ROOT / "polylogue" / "analysis",
    REPO_ROOT / "polylogue" / "surfaces",
    REPO_ROOT / "polylogue" / "cli",
    REPO_ROOT / "polylogue" / "mcp",
    REPO_ROOT / "polylogue" / "api",
    REPO_ROOT / "polylogue" / "operations",
)

#: The one module allowed to produce a tool-outcome rate.
SANCTIONED = REPO_ROOT / "polylogue" / "analysis" / "measurement" / "outcome_coverage.py"

#: Names that mean "a proportion of tool calls that succeeded or failed".
RATE_NAME = re.compile(
    r"(?<![A-Za-z])(?:success|succeeded|failure|failed|error|ok|pass|passing)"
    r"_(?:rate|pct|percent|percentage|ratio|share)",
    re.IGNORECASE,
)

#: A module only counts as speaking tool-outcome vocabulary if it names one of
#: these. A rate over HTTP routes or schema drift is not this concern.
TOOL_OUTCOME_TOKENS = ("tool_outcome", "ToolOutcome", "tool_result_is_error")


def _iter_read_path_modules() -> list[Path]:
    files: list[Path] = []
    for root in READ_PATH_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _rate_names(source: str) -> list[str]:
    """Every identifier or literal key in ``source`` that names a rate.

    Covers the shapes a rate actually lands in: a property or function, an
    assigned or annotated attribute, a model field, and a payload dict key
    (string constants, because a JSON projection publishes the number under
    a literal name rather than an identifier).
    """

    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            candidates.append(node.name)
        elif isinstance(node, ast.Name):
            candidates.append(node.id)
        elif isinstance(node, ast.Attribute):
            candidates.append(node.attr)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            candidates.append(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidates.append(node.value)
        hits.extend(name for name in candidates if RATE_NAME.search(name))
    return hits


@pytest.mark.parametrize("path", _iter_read_path_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_ungated_tool_outcome_rate_on_a_read_path(path: Path) -> None:
    """A tool-outcome rate outside the gated module is the forbidden regrowth."""

    source = path.read_text(encoding="utf-8")
    if not any(token in source for token in TOOL_OUTCOME_TOKENS):
        return
    hits = sorted(set(_rate_names(source)))
    if not hits:
        return
    if path == SANCTIONED:
        return
    relative = path.relative_to(REPO_ROOT)
    pytest.fail(
        f"{relative} computes a tool-outcome rate ({hits}) outside the gated route. "
        "A rate over `blocks.tool_outcome` must go through "
        "polylogue.analysis.measurement.outcome_coverage.build_tool_outcome_aggregate, "
        f"which refuses the bare scalar below TOOL_OUTCOME_COVERAGE_FLOOR "
        f"({TOOL_OUTCOME_COVERAGE_FLOOR}) and reports `degraded` with the coverage gap "
        "named. `unknown` is a retained refusal, never a success."
    )


def test_sanctioned_route_still_gates_the_scalar() -> None:
    """The ratchet must not pass by having nothing left to protect."""

    assert SANCTIONED.exists()
    below = build_tool_outcome_aggregate({ToolOutcome.OK: 1, ToolOutcome.UNKNOWN: 9})
    assert below.success_rate is None
    assert below.outcome.state == "degraded"
    assert _rate_names(SANCTIONED.read_text(encoding="utf-8"))


def test_ratchet_detects_the_shapes_a_rate_actually_lands_in() -> None:
    """The detector is not keyed to one syntactic form."""

    assert _rate_names("def success_rate(self): ...")
    assert _rate_names("error_rate = ok / total")
    assert _rate_names("class Row:\n    failure_rate: float")
    assert _rate_names('payload = {"pass_rate": value}')
    assert _rate_names("Model(success_pct=value)")
    # Unrelated vocabulary must not trip it.
    assert not _rate_names("latency_ms = 3\nsample_rate = 4\nbit_rate = 5")
