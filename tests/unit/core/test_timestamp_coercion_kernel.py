"""One owner for timestamp coercion (polylogue-z3sv).

Seven private ``_timestamp_ms`` / ``_iso_from_epoch_ms`` copies disagreed by a
factor of 1000 on the same all-digit string, because each baked in its own
answer to "what unit is a bare number". The kernel makes that unit a declared
argument, and the census test below keeps the copies from growing back.

Anti-vacuity: restoring any private copy (or dropping ``numeric_unit`` so one
unit becomes an implicit default) makes ``test_no_private_timestamp_coercion_copies``
or the unit-boundary assertions red.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

import polylogue
from polylogue.core.timestamps import iso_from_epoch_ms, to_epoch_ms

_EPOCH_SECONDS = 1735689600  # 2025-01-01T00:00:00Z
_EPOCH_MS = _EPOCH_SECONDS * 1000


@pytest.mark.parametrize("value", [_EPOCH_SECONDS, float(_EPOCH_SECONDS), str(_EPOCH_SECONDS)])
def test_seconds_unit_scales_numbers_and_digit_strings(value: object) -> None:
    assert to_epoch_ms(value, numeric_unit="seconds") == _EPOCH_MS


@pytest.mark.parametrize("value", [_EPOCH_MS, float(_EPOCH_MS), str(_EPOCH_MS)])
def test_milliseconds_unit_passes_numbers_and_digit_strings_through(value: object) -> None:
    assert to_epoch_ms(value, numeric_unit="milliseconds") == _EPOCH_MS


def test_the_two_units_differ_by_exactly_one_thousand() -> None:
    seconds = to_epoch_ms(str(_EPOCH_SECONDS), numeric_unit="seconds")
    milliseconds = to_epoch_ms(str(_EPOCH_SECONDS), numeric_unit="milliseconds")
    assert seconds == milliseconds * 1000 != 0


def test_iso_input_is_unit_independent() -> None:
    iso = "2025-01-01T00:00:00Z"
    assert to_epoch_ms(iso, numeric_unit="seconds") == _EPOCH_MS
    assert to_epoch_ms(iso, numeric_unit="milliseconds") == _EPOCH_MS


def test_naive_iso_is_read_as_utc() -> None:
    assert to_epoch_ms("2025-01-01T00:00:00", numeric_unit="seconds") == _EPOCH_MS


def test_aware_datetime_is_unit_independent() -> None:
    moment = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert to_epoch_ms(moment, numeric_unit="seconds") == _EPOCH_MS
    assert to_epoch_ms(moment, numeric_unit="milliseconds") == _EPOCH_MS


@pytest.mark.parametrize("value", [None, True, False, "", "   ", "not-a-time", "{'a': 1}", object()])
def test_unparseable_values_are_none_not_zero(value: object) -> None:
    assert to_epoch_ms(value, numeric_unit="seconds") is None
    assert to_epoch_ms(value, numeric_unit="milliseconds") is None


def test_small_digit_string_is_not_read_as_a_seconds_instant() -> None:
    """'2025' is a bare year, not four seconds past the epoch."""
    assert to_epoch_ms("2025", numeric_unit="seconds") is None
    assert to_epoch_ms("2025", numeric_unit="milliseconds") == 2025


def test_iso_from_epoch_ms_round_trips_the_kernel() -> None:
    assert iso_from_epoch_ms(_EPOCH_MS) == "2025-01-01T00:00:00+00:00"
    assert iso_from_epoch_ms(str(_EPOCH_MS)) == "2025-01-01T00:00:00+00:00"
    assert iso_from_epoch_ms(None) is None
    assert iso_from_epoch_ms(True) is None
    assert iso_from_epoch_ms("nonsense") is None


_FORKED_NAMES = {"_timestamp_ms", "_iso_from_epoch_ms", "_epoch_ms"}


def test_no_private_timestamp_coercion_copies() -> None:
    package_root = Path(polylogue.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _FORKED_NAMES:
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno} {node.name}")
    assert offenders == [], "timestamp coercion re-forked; import polylogue.core.timestamps instead: " + ", ".join(
        offenders
    )
