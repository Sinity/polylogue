"""Regression tests for source-wire mappings and canonical origin filters."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from polylogue.archive.query.archive_execution import _provider_for_origin as _archive_execution_provider_for_origin
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.storage.sqlite.archive_tiers.archive import (
    _origin_for_tool_usage_filter as _archive_tiers_origin_for_tool_usage_filter,
)
from polylogue.storage.sqlite.archive_tiers.archive import _provider_for_origin as _archive_tiers_provider_for_origin
from polylogue.storage.sqlite.queries.tool_usage import (
    _origin_for_tool_usage_filter as _query_origin_for_tool_usage_filter,
)
from polylogue.storage.sqlite.queries.tool_usage import _provider_for_origin as _tool_usage_provider_for_origin

_DUPLICATED_PROVIDER_FOR_ORIGIN: tuple[Callable[[str], Provider], ...] = (
    _archive_execution_provider_for_origin,
    _archive_tiers_provider_for_origin,
    _tool_usage_provider_for_origin,
)

_TOOL_USAGE_ORIGIN_FILTERS: tuple[Callable[[str | None], str | None], ...] = (
    _archive_tiers_origin_for_tool_usage_filter,
    _query_origin_for_tool_usage_filter,
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


@pytest.mark.parametrize("fn", _TOOL_USAGE_ORIGIN_FILTERS)
def test_tool_usage_origin_filters_preserve_every_canonical_origin(
    fn: Callable[[str | None], str | None],
) -> None:
    for origin in Origin:
        assert fn(origin.value) == origin.value
