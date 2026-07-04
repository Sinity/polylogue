"""Round-trip contract for :class:`MaintenanceScopeFilter` (#1303).

Pins the typed-filter round-trip independent of any surface:

* ``to_dict()`` always yields a JSON-shaped dict that lists every known
  dimension exactly once (so absent dimensions stay explicitly
  ``None`` instead of being silently dropped);
* ``from_dict(to_dict(filter))`` is the identity for every dimension
  combination — verified both by example and by a Hypothesis property
  that draws across all eight dimensions;
* ``None`` and ``{}`` both rehydrate to the canonical empty filter so
  surfaces that omit the field do not desynchronize from surfaces that
  send an empty body.

Companion to :mod:`tests.unit.maintenance.test_scope_filter` (which
pins the model's shape and frozen-ness) and
:mod:`tests.unit.maintenance.test_scope_filter_envelope_contract`
(which pins cross-surface parity).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.maintenance.scope import MaintenanceScopeFilter

_EXPECTED_DIMENSIONS = frozenset(
    {
        "session_ids",
        "provider",
        "source_family",
        "source_root",
        "time_range",
        "failure_kind",
        "parser_version",
    }
)


class TestScopeFilterDictShape:
    """``to_dict`` is byte-stable and exhaustive."""

    def test_to_dict_lists_every_dimension_for_empty_filter(self) -> None:
        payload = MaintenanceScopeFilter().to_dict()
        assert set(payload.keys()) == _EXPECTED_DIMENSIONS
        # Every absent dimension is explicit ``None`` rather than missing.
        for dim in _EXPECTED_DIMENSIONS:
            assert payload[dim] is None, f"{dim} should serialize as None when unset"

    def test_to_dict_lists_every_dimension_for_populated_filter(self) -> None:
        full = MaintenanceScopeFilter(
            session_ids=("c1", "c2"),
            provider="claude",
            source_family="claude-code-session",
            source_root=Path("/data/claude"),
            time_range=(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
            failure_kind="ValidationError",
            parser_version="v3",
        )
        payload = full.to_dict()
        assert set(payload.keys()) == _EXPECTED_DIMENSIONS

    def test_to_dict_is_json_serializable(self) -> None:
        """``mode='json'`` must coerce Path/datetime/tuple to JSON primitives.

        Surfaces serialize the payload straight through :func:`json.dumps`
        without a custom encoder — leaking a ``Path`` or ``datetime`` here
        would crash the daemon HTTP response or the MCP envelope.
        """
        full = MaintenanceScopeFilter(
            session_ids=("c1",),
            source_root=Path("/data"),
            time_range=(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
        )
        text = json.dumps(full.to_dict())
        # Round-trip via JSON must not lose information.
        rehydrated = MaintenanceScopeFilter.from_dict(json.loads(text))
        assert rehydrated == full


class TestScopeFilterRoundTrip:
    """from_dict(to_dict(f)) == f for every dimension combination."""

    @pytest.mark.parametrize(
        "filter_kwargs",
        [
            {},
            {"session_ids": ("c1",)},
            {"session_ids": ("c1", "c2", "c3")},
            {"provider": "claude"},
            {"source_family": "claude-code-session"},
            {"source_root": Path("/data/claude")},
            {
                "time_range": (
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 2, 1, tzinfo=timezone.utc),
                )
            },
            {"failure_kind": "ValidationError"},
            {"parser_version": "v3"},
            {
                "session_ids": ("c1", "c2"),
                "provider": "claude",
                "source_family": "claude-code-session",
                "source_root": Path("/data"),
                "time_range": (
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 2, 1, tzinfo=timezone.utc),
                ),
                "failure_kind": "decode-error",
                "parser_version": "v3",
            },
        ],
    )
    def test_round_trip_preserves_filter(self, filter_kwargs: dict[str, Any]) -> None:
        original = MaintenanceScopeFilter(**filter_kwargs)
        recovered = MaintenanceScopeFilter.from_dict(original.to_dict())
        assert recovered == original

    def test_double_round_trip_is_idempotent(self) -> None:
        original = MaintenanceScopeFilter(
            session_ids=("c1",),
            provider="claude",
            source_root=Path("/data"),
            time_range=(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
        )
        once = MaintenanceScopeFilter.from_dict(original.to_dict())
        twice = MaintenanceScopeFilter.from_dict(once.to_dict())
        assert original == once == twice
        # The serialized payload itself is byte-stable, too.
        assert original.to_dict() == once.to_dict() == twice.to_dict()

    def test_from_dict_none_is_empty_filter(self) -> None:
        assert MaintenanceScopeFilter.from_dict(None) == MaintenanceScopeFilter()

    def test_from_dict_empty_dict_is_empty_filter(self) -> None:
        assert MaintenanceScopeFilter.from_dict({}) == MaintenanceScopeFilter()

    def test_from_dict_with_explicit_none_dimensions_is_empty(self) -> None:
        """A payload that names every dimension as ``None`` is the empty filter.

        This is the shape ``to_dict()`` emits, so the empty-filter
        round-trip must not depend on the caller pruning ``None`` keys.
        """
        all_none = dict.fromkeys(_EXPECTED_DIMENSIONS)
        assert MaintenanceScopeFilter.from_dict(all_none) == MaintenanceScopeFilter()


# ---------------------------------------------------------------------------
# Hypothesis property: round-trip identity across the whole dimension space
# ---------------------------------------------------------------------------


_ASCII_TEXT = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=24,
)


@st.composite
def _scope_filter_kwargs(draw: Any) -> dict[str, Any]:
    """Draw kwargs for an arbitrary :class:`MaintenanceScopeFilter`.

    Each dimension is independently present-or-absent, so the property
    explores the full 2^8 presence lattice rather than only fully-populated
    or fully-empty filters.
    """
    kwargs: dict[str, Any] = {}
    if draw(st.booleans()):
        kwargs["session_ids"] = tuple(draw(st.lists(_ASCII_TEXT, min_size=1, max_size=4, unique=True)))
    if draw(st.booleans()):
        kwargs["provider"] = draw(_ASCII_TEXT)
    if draw(st.booleans()):
        kwargs["source_family"] = draw(_ASCII_TEXT)
    if draw(st.booleans()):
        kwargs["source_root"] = Path(draw(_ASCII_TEXT))
    if draw(st.booleans()):
        # Two distinct tz-aware datetimes so the (since, until) ordering
        # is meaningful but we never have to worry about naive timestamps.
        since = draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 12, 31),
                timezones=st.just(timezone.utc),
            )
        )
        until = draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 12, 31),
                timezones=st.just(timezone.utc),
            )
        )
        kwargs["time_range"] = (since, until)
    if draw(st.booleans()):
        kwargs["failure_kind"] = draw(_ASCII_TEXT)
    if draw(st.booleans()):
        kwargs["parser_version"] = draw(_ASCII_TEXT)
    return kwargs


@given(_scope_filter_kwargs())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_round_trip_property(kwargs: dict[str, Any]) -> None:
    """Property: any valid filter round-trips through ``to_dict``/``from_dict``."""
    original = MaintenanceScopeFilter(**kwargs)
    recovered = MaintenanceScopeFilter.from_dict(original.to_dict())
    assert recovered == original
    # And the serialized payload is itself byte-stable across one round trip.
    assert original.to_dict() == recovered.to_dict()
