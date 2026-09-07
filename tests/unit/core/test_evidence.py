"""The evidence union keeps "no answer" from being spelled as an answer.

Anti-vacuity: delete ``Empty``/``Unavailable``/``Degraded`` from the union, or
give any of them a ``value`` attribute, or hand ``resolve`` a default for one
of its handlers, and one of these assertions fails. Static proof that a
renderer ignoring the state does not type-check is
``test_renderer_that_ignores_state_fails_type_check``.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.core.evidence import (
    Degraded,
    Empty,
    Evidence,
    Measured,
    Unavailable,
    evidence_state,
    measured_or_none,
    resolve,
)


def test_only_the_value_carrying_cases_expose_a_value() -> None:
    assert Measured(3).value == 3
    assert Degraded(3, reason="partial").value == 3
    assert not hasattr(Empty(), "value")
    assert not hasattr(Unavailable(reason="tier_missing"), "value")


def test_resolve_requires_a_handler_for_every_case() -> None:
    parameters = inspect.signature(resolve).parameters
    handlers = {name: p for name, p in parameters.items() if name != "evidence"}
    assert set(handlers) == {"measured", "empty", "unavailable", "degraded"}
    for name, parameter in handlers.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is inspect.Parameter.empty, name


@pytest.mark.parametrize(
    ("evidence", "state"),
    [
        (Measured(1), "measured"),
        (Empty(), "empty"),
        (Unavailable(reason="tier_missing"), "unavailable"),
        (Degraded(1, reason="partial"), "degraded"),
    ],
)
def test_state_names_the_case_without_yielding_the_value(evidence: Evidence[int], state: str) -> None:
    assert evidence_state(evidence) == state


def test_empty_and_unavailable_are_not_interchangeable() -> None:
    empty: Evidence[int] = Empty()
    unavailable: Evidence[int] = Unavailable(reason="tier_missing")
    assert empty != unavailable
    assert evidence_state(empty) != evidence_state(unavailable)


def test_measured_or_none_reports_absence_as_none_not_zero() -> None:
    assert measured_or_none(Measured(0)) == 0
    assert measured_or_none(Empty()) is None
    assert measured_or_none(Unavailable(reason="scalar_read_failed")) is None
    assert measured_or_none(Degraded(7, reason="partial")) == 7


def test_renderer_that_ignores_state_fails_type_check(tmp_path: Path) -> None:
    """A payload built from ``.value`` alone, or a partial ``resolve``, is a type error."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from polylogue.core.evidence import Evidence, resolve\n"
        "\n"
        "def render(e: Evidence[int]) -> str:\n"
        "    return str(e.value)\n"
        "\n"
        "def partial(e: Evidence[int]) -> str:\n"
        "    return resolve(e, measured=str, empty=lambda: '-')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", "--cache-dir", str(tmp_path / "cache"), str(probe)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode != 0, result.stdout
    assert 'has no attribute "value"' in result.stdout
    assert 'Missing named argument "unavailable"' in result.stdout
    assert 'Missing named argument "degraded"' in result.stdout
