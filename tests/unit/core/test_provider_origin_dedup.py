"""Regression tests for the provider<->origin reverse-lookup dedup (polylogue-9e5.8 Step 1).

Ground truth found while executing polylogue-9e5.8's plan: the bead's design
field described ``archive/query/archive_execution.py``'s ``_ORIGIN_TO_PROVIDER``
dict as uniquely, silently collapsing ``aistudio-drive`` onto
``Provider.GEMINI``. Reading the live source showed this collapse is actually
a *deliberate, documented* choice already centralized in
``core/sources.py::provider_from_origin`` (``Origin.AISTUDIO_DRIVE`` is
non-injective -- both ``Provider.GEMINI`` and ``Provider.DRIVE`` produce it --
and ``GEMINI`` is the documented canonical choice so a Gemini session
round-trips). The real, verified bug was that three independent modules
(``archive/query/archive_execution.py``, ``storage/sqlite/archive_tiers/
archive.py``, ``storage/sqlite/queries/tool_usage.py``) each hand-rolled an
*independent copy* of that exact table instead of delegating to the
canonical function -- and had already silently drifted: all three copies were
missing a ``grok-export`` entry, so a Grok-origin session would resolve to
``Provider.UNKNOWN`` instead of ``Provider.GROK`` through these three call
sites specifically, while the canonical ``core/sources.py`` table already
handled it correctly.

The fix replaces all three hand-rolled dicts with delegation to
``provider_from_origin``. These tests prove: (1) all three now agree with the
canonical function for every ``Origin`` value, including the previously-drifted
``grok-export`` case, and (2) the well-known, deliberate GEMINI/DRIVE collapse
is preserved (not "fixed" -- that needs a Source-family disambiguator, Step 5
of the plan, out of scope here).

Un-collapsing DRIVE from GEMINI is explicitly *not* what this fix does or
claims to do -- see polylogue-9e5.8's design field, "The non-injective
GEMINI/DRIVE blocker" section.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import polylogue.mcp.server_insight_tools as server_insight_tools_module
from polylogue.archive.query.archive_execution import _provider_for_origin as _archive_execution_provider_for_origin
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.mcp.insight_tool_contracts import _origin_to_provider_token as _contracts_origin_to_provider_token
from polylogue.storage.sqlite.archive_tiers.archive import _provider_for_origin as _archive_tiers_provider_for_origin
from polylogue.storage.sqlite.queries.tool_usage import _provider_for_origin as _tool_usage_provider_for_origin

_DUPLICATED_PROVIDER_FOR_ORIGIN: tuple[Callable[[str], Provider], ...] = (
    _archive_execution_provider_for_origin,
    _archive_tiers_provider_for_origin,
    _tool_usage_provider_for_origin,
)


@pytest.mark.parametrize("fn", _DUPLICATED_PROVIDER_FOR_ORIGIN)
def test_provider_for_origin_agrees_with_canonical_for_every_origin(fn: Callable[[str], Provider]) -> None:
    """Every de-duplicated call site must agree with the single source of
    truth for every known Origin token -- no more silent drift between
    hand-copies."""
    for origin in Origin:
        assert fn(origin.value) == provider_from_origin(origin), (
            f"{fn.__module__}.{fn.__qualname__}({origin.value!r}) diverges from "
            "the canonical polylogue.core.sources.provider_from_origin"
        )


@pytest.mark.parametrize("fn", _DUPLICATED_PROVIDER_FOR_ORIGIN)
def test_grok_export_no_longer_silently_falls_back_to_unknown(fn: Callable[[str], Provider]) -> None:
    """The regression this dedup fixes: all three hand-rolled dicts were
    missing a grok-export entry and fell back to Provider.UNKNOWN. Delegating
    to the canonical function fixes this at all three call sites at once."""
    assert fn("grok-export") == Provider.GROK


@pytest.mark.parametrize("fn", _DUPLICATED_PROVIDER_FOR_ORIGIN)
def test_aistudio_drive_still_canonically_resolves_to_gemini(fn: Callable[[str], Provider]) -> None:
    """Documents, rather than "fixes", the deliberate non-injective collapse:
    Origin.AISTUDIO_DRIVE is produced by both Provider.GEMINI and
    Provider.DRIVE, and GEMINI is the documented canonical reverse choice
    (core/sources.py). Un-collapsing this needs a Source-family
    disambiguator (polylogue-9e5.8 Step 5) -- out of scope for this fix."""
    assert fn("aistudio-drive") == Provider.GEMINI


@pytest.mark.parametrize("fn", _DUPLICATED_PROVIDER_FOR_ORIGIN)
def test_unknown_origin_token_falls_back_to_unknown_provider(fn: Callable[[str], Provider]) -> None:
    assert fn("not-a-real-origin") == Provider.UNKNOWN


def test_origin_to_provider_token_is_a_single_shared_definition() -> None:
    """polylogue-9e5.8's design field flagged _origin_to_provider_token as
    defined twice, independently, in mcp/insight_tool_contracts.py and
    mcp/server_insight_tools.py -- itself evidence of an ad hoc, uncentralized
    shim. server_insight_tools now imports the contracts module's definition
    instead of re-declaring it; this asserts there is exactly one function
    object, not two behaviorally-identical copies."""
    assert getattr(server_insight_tools_module, "_origin_to_provider_token") is _contracts_origin_to_provider_token  # noqa: B009


def test_origin_to_provider_token_round_trips_known_origins() -> None:
    for origin in Origin:
        assert _contracts_origin_to_provider_token(origin.value) == provider_from_origin(origin).value


def test_origin_to_provider_token_none_and_empty_are_none() -> None:
    assert _contracts_origin_to_provider_token(None) is None
    assert _contracts_origin_to_provider_token("") is None
