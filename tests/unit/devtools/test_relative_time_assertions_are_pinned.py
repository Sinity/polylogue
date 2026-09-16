"""A rendered relative-time string may not be asserted against an unpinned clock.

``relative_time`` is the distance from *now* to a fixture timestamp, so a test
that asserts its rendered label is correct the day it is written and goes red
weeks later. Two such tests were repaired in 2026-08, each costing a fresh
diagnosis, and the calendar schedules the next one. This check closes the class
structurally: a test asserting one of those labels must pin the instant it
renders against.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[2]

#: The shapes ``polylogue.surfaces.query_rows.relative_time`` renders.
_RELATIVE_LABEL = re.compile(r"^(just now|\d+[mhdw] ago)$")

#: Names that pin the instant a row renders against.
_PINNING_NAMES = frozenset({"frozen_clock", "frozen_clock_modules", "now"})


def _pins_the_clock(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    arguments = {argument.arg for argument in function.args.args} | {
        argument.arg for argument in function.args.kwonlyargs
    }
    if arguments & _PINNING_NAMES:
        return True
    for decorator in function.decorator_list:
        if any(isinstance(node, ast.Attribute) and node.attr in _PINNING_NAMES for node in ast.walk(decorator)):
            return True
    return any(isinstance(node, ast.keyword) and node.arg in _PINNING_NAMES for node in ast.walk(function)) or any(
        isinstance(node, ast.Name) and node.id in _PINNING_NAMES for node in ast.walk(function)
    )


def _unpinned_relative_time_assertions() -> list[str]:
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            literals = [
                node.value
                for node in ast.walk(function)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and _RELATIVE_LABEL.match(node.value)
            ]
            if literals and not _pins_the_clock(function):
                rel = path.relative_to(_TESTS_ROOT.parent).as_posix()
                offenders.append(f"{rel}::{function.name} asserts {literals[0]!r} without pinning the clock")
    return offenders


def test_no_test_asserts_a_relative_time_label_against_an_unpinned_clock() -> None:
    """Anti-vacuity: delete the frozen_clock fixture from test_select.py's row
    contract test and this check names it, where today it stays green until the
    fixture date drifts one more week."""
    assert _unpinned_relative_time_assertions() == []


def test_the_check_recognises_an_unpinned_assertion() -> None:
    """The scan is not vacuous: an unpinned label in a test function is caught."""
    source = "def test_row_label() -> None:\n    assert row.relative_time == '16w ago'\n"
    tree = ast.parse(source)
    (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert not _pins_the_clock(function)
    assert any(
        isinstance(node, ast.Constant) and isinstance(node.value, str) and _RELATIVE_LABEL.match(node.value)
        for node in ast.walk(function)
    )
