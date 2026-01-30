"""Hypothesis strategies for filter composition testing.

These strategies generate filter combinations for testing the filter
algebra properties (monotonicity, composition, etc).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from hypothesis import strategies as st


# =============================================================================
# Filter Type Strategies
# =============================================================================


@st.composite
def filter_type_strategy(draw: st.DrawFn) -> str:
    """Generate a filter type name."""
    return draw(st.sampled_from([
        "provider",
        "contains",
        "since",
        "until",
        "limit",
        "offset",
        "sort",
        "role",
        "has_attachments",
        "min_words",
        "max_words",
    ]))


@st.composite
def provider_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for provider filter."""
    return {
        "type": "provider",
        "value": draw(st.sampled_from(["chatgpt", "claude", "claude-code", "codex"])),
    }


@st.composite
def contains_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for contains (text search) filter."""
    return {
        "type": "contains",
        "value": draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=("L", "N", "P"),
            blacklist_characters='"\'\\',
        ))),
    }


@st.composite
def date_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for date filters (since/until)."""
    filter_type = draw(st.sampled_from(["since", "until"]))
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    offset_days = draw(st.integers(min_value=-365, max_value=365))
    date = base_date + timedelta(days=offset_days)

    return {
        "type": filter_type,
        "value": date.isoformat(),
    }


@st.composite
def limit_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for limit filter."""
    return {
        "type": "limit",
        "value": draw(st.integers(min_value=1, max_value=1000)),
    }


@st.composite
def offset_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for offset filter."""
    return {
        "type": "offset",
        "value": draw(st.integers(min_value=0, max_value=100)),
    }


@st.composite
def sort_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for sort filter."""
    return {
        "type": "sort",
        "field": draw(st.sampled_from(["created_at", "updated_at", "title"])),
        "direction": draw(st.sampled_from(["asc", "desc"])),
    }


@st.composite
def role_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for role filter."""
    return {
        "type": "role",
        "value": draw(st.sampled_from(["user", "assistant", "system"])),
    }


@st.composite
def word_count_filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate arguments for word count filters."""
    filter_type = draw(st.sampled_from(["min_words", "max_words"]))
    return {
        "type": filter_type,
        "value": draw(st.integers(min_value=1, max_value=1000)),
    }


# =============================================================================
# Composite Filter Strategies
# =============================================================================


@st.composite
def filter_arg_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate any valid filter argument."""
    return draw(st.one_of(
        provider_filter_arg_strategy(),
        contains_filter_arg_strategy(),
        date_filter_arg_strategy(),
        limit_filter_arg_strategy(),
        offset_filter_arg_strategy(),
        sort_filter_arg_strategy(),
        role_filter_arg_strategy(),
        word_count_filter_arg_strategy(),
    ))


@st.composite
def filter_chain_strategy(
    draw: st.DrawFn,
    min_filters: int = 1,
    max_filters: int = 5,
) -> list[dict[str, Any]]:
    """Generate a chain of filters for composition testing.

    Properties to test with this strategy:
    - Monotonicity: adding filters never increases result count
    - Commutativity: filter order doesn't affect final count (for idempotent filters)
    - Idempotence: applying same filter twice has no additional effect
    """
    return draw(st.lists(
        filter_arg_strategy(),
        min_size=min_filters,
        max_size=max_filters,
    ))


@st.composite
def pagination_filter_chain_strategy(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Generate pagination filter combinations.

    Tests that offset + limit work correctly together.
    """
    offset = draw(st.integers(min_value=0, max_value=50))
    limit = draw(st.integers(min_value=1, max_value=100))

    return [
        {"type": "offset", "value": offset},
        {"type": "limit", "value": limit},
    ]


@st.composite
def date_range_filter_chain_strategy(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Generate date range filter combinations.

    Tests that since + until work correctly together.
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start_offset = draw(st.integers(min_value=-365, max_value=0))
    end_offset = draw(st.integers(min_value=0, max_value=365))

    since = base + timedelta(days=start_offset)
    until = base + timedelta(days=end_offset)

    # Ensure since <= until
    if since > until:
        since, until = until, since

    return [
        {"type": "since", "value": since.isoformat()},
        {"type": "until", "value": until.isoformat()},
    ]
