"""Behavior tests for canonical origin filters."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.archive import (
    _origin_for_tool_usage_filter as _archive_tiers_origin_for_tool_usage_filter,
)
from polylogue.storage.sqlite.queries.tool_usage import (
    _origin_for_tool_usage_filter as _query_origin_for_tool_usage_filter,
)

_TOOL_USAGE_ORIGIN_FILTERS: tuple[Callable[[str | None], str | None], ...] = (
    _archive_tiers_origin_for_tool_usage_filter,
    _query_origin_for_tool_usage_filter,
)


@pytest.mark.parametrize("fn", _TOOL_USAGE_ORIGIN_FILTERS)
def test_tool_usage_origin_filters_preserve_every_canonical_origin(
    fn: Callable[[str | None], str | None],
) -> None:
    for origin in Origin:
        assert fn(origin.value) == origin.value
