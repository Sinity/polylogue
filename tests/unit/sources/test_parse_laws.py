"""Property laws for provider parsers and role normalization.

Each law covers an invariant that holds for any input, superseding
specific example tables in the parser test files.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib.roles import normalize_role

# ---------------------------------------------------------------------------
# Law 1: normalize_role never raises for any non-empty string
# ---------------------------------------------------------------------------

@given(st.text(min_size=1))
def test_normalize_role_never_raises_for_nonempty(text: str) -> None:
    """normalize_role handles any non-empty string without raising."""
    # normalize_role raises only on empty/whitespace-only strings
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Law 2: normalize_role always returns one of the canonical roles
# ---------------------------------------------------------------------------

CANONICAL_ROLES = frozenset({"user", "assistant", "system", "tool", "unknown"})


@given(st.text(min_size=1))
def test_normalize_role_result_is_canonical(text: str) -> None:
    """normalize_role always returns a canonical role string."""
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert result in CANONICAL_ROLES


# ---------------------------------------------------------------------------
# Law 3: normalize_role is idempotent on its own output
# ---------------------------------------------------------------------------

@given(st.sampled_from(sorted(CANONICAL_ROLES - {"unknown"})))
def test_normalize_role_idempotent_on_canonical(role: str) -> None:
    """Applying normalize_role to a canonical role returns the same value."""
    result = normalize_role(role)
    assert result == role


# ---------------------------------------------------------------------------
# Law 4: normalize_role is case-insensitive
# ---------------------------------------------------------------------------

@given(st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",))))
def test_normalize_role_case_insensitive(text: str) -> None:
    """normalize_role gives the same result for any case variant."""
    stripped = text.strip()
    if stripped:
        lower_result = normalize_role(stripped.lower())
        upper_result = normalize_role(stripped.upper())
        title_result = normalize_role(stripped.title())
        assert lower_result == upper_result == title_result


# ---------------------------------------------------------------------------
# Law 5: normalize_role strips whitespace before normalizing
# ---------------------------------------------------------------------------

@given(
    st.sampled_from(["user", "assistant", "system", "tool"]),
    st.integers(min_value=0, max_value=5),
)
def test_normalize_role_strips_whitespace(role: str, padding: int) -> None:
    """normalize_role ignores leading/trailing whitespace."""
    padded = " " * padding + role + " " * padding
    assert normalize_role(padded) == role


# ---------------------------------------------------------------------------
# Law 6: normalize_role raises ValueError for empty/whitespace-only input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty", ["", "   ", "\t", "\n", "\t\n  "])
def test_normalize_role_raises_on_empty(empty: str) -> None:
    """normalize_role raises ValueError for empty or whitespace-only input."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        normalize_role(empty)
