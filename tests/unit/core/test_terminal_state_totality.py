"""TerminalState is declared once and every consumer table is total (polylogue-hjvow).

Three tables keyed on the terminal-state vocabulary each carried their own
untyped member list with a silent default, so ``refused`` and ``truncated``
-- both real structural outcomes emitted by ``_terminal_state`` -- fell
through to "ambiguous"/0.25/severity-0 everywhere, while two dead keys
(``clean_finish``, ``agent_hanging``) no producer emits stayed in.

Anti-vacuity: deleting any key from one of the three tables, or adding a
member to ``TerminalState`` without extending them, raises at import time and
makes this module red on collection; loosening a table back to
``dict[str, ...]`` with a ``.get`` default makes the explicit-mapping
assertions below red.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import pytest

from polylogue.analysis.archive_rollups import ABANDONMENT_SEVERITY_RANK
from polylogue.analysis.objective_posture import _STRUCTURAL_POSTURE_BY_TERMINAL_STATE
from polylogue.analysis.resume import _TERMINAL_WEIGHT, _terminal_weight
from polylogue.core.enums import TERMINAL_STATE_VALUES, TerminalState

_RUNTIME_SOURCE = Path(__file__).resolve().parents[3] / "polylogue" / "archive" / "session" / "runtime.py"


def test_declared_vocabulary_matches_its_frozenset() -> None:
    assert frozenset(get_args(TerminalState)) == TERMINAL_STATE_VALUES
    assert "clean_finish" not in TERMINAL_STATE_VALUES
    assert "agent_hanging" not in TERMINAL_STATE_VALUES


@pytest.mark.parametrize(
    "table",
    [_STRUCTURAL_POSTURE_BY_TERMINAL_STATE, _TERMINAL_WEIGHT, ABANDONMENT_SEVERITY_RANK],
    ids=["structural_posture", "resume_weight", "abandonment_severity"],
)
def test_consumer_tables_are_total(table: dict[TerminalState, object]) -> None:
    assert set(table) == TERMINAL_STATE_VALUES


def test_refused_and_truncated_are_not_defaulted() -> None:
    assert _STRUCTURAL_POSTURE_BY_TERMINAL_STATE["refused"] == "awaiting_operator"
    assert _STRUCTURAL_POSTURE_BY_TERMINAL_STATE["truncated"] == "awaiting_effect"
    # 'truncated' is strong resume material, not noise at the unknown weight.
    assert _terminal_weight("truncated") > _terminal_weight("question_left")
    assert _terminal_weight("truncated") > _terminal_weight("unknown")
    assert ABANDONMENT_SEVERITY_RANK["truncated"] > ABANDONMENT_SEVERITY_RANK["tool_left"]
    assert ABANDONMENT_SEVERITY_RANK["refused"] > ABANDONMENT_SEVERITY_RANK["unknown"]


def test_every_state_the_producer_emits_is_declared() -> None:
    """``_terminal_state`` may not return a label outside the declared vocabulary."""
    tree = ast.parse(_RUNTIME_SOURCE.read_text(encoding="utf-8"))
    producer = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_terminal_state"
    )
    emitted = {
        element.value
        for node in ast.walk(producer)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
        for element in node.value.elts[:1]
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    assert emitted, "expected _terminal_state to return literal labels"
    assert emitted <= TERMINAL_STATE_VALUES, f"undeclared terminal states: {sorted(emitted - TERMINAL_STATE_VALUES)}"
