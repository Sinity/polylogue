"""Hypothesis strategies for search query and FTS5 edge-case testing.

These strategies generate search queries that exercise FTS5 escaping,
unicode handling, operator injection, and ranking edge cases.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import strategies as st

from tests.infra.adversarial_cases import FTS5_OPERATORS, SQL_INJECTION_PAYLOADS

# See also: adversarial.fts5_operator_strategy() which draws from the same FTS5_OPERATORS

_LETTER_NUMBER_CATEGORIES: tuple[Literal["L"], Literal["N"]] = ("L", "N")
_LETTER_NUMBER_PUNCT_SPACE_CATEGORIES: tuple[Literal["L"], Literal["N"], Literal["P"], Literal["Z"]] = (
    "L",
    "N",
    "P",
    "Z",
)
_LETTER_ONLY_CATEGORIES: tuple[Literal["L"]] = ("L",)

# =============================================================================
# Search Query Strategies
# =============================================================================


@st.composite
def search_query_strategy(draw: st.DrawFn) -> str:
    """Generate a search query that exercises FTS5 edge cases.

    Covers:
    - Simple alphanumeric terms
    - Unicode text (CJK, Cyrillic, accented, emoji)
    - FTS5 operator injection (AND, OR, NOT, NEAR)
    - Special characters (quotes, asterisks, parens)
    - SQL injection payloads
    - Empty/whitespace
    """
    value = draw(
        st.one_of(
            # Simple terms
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=_LETTER_NUMBER_CATEGORIES),
            ),
            # FTS5 operators as literals
            st.sampled_from(FTS5_OPERATORS),
            # SQL injection payloads
            st.sampled_from(SQL_INJECTION_PAYLOADS),
            # Unicode text
            st.text(min_size=1, max_size=30),
            # Quoted strings with internal quotes
            st.builds(
                lambda t: f'"{t}"',
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=_LETTER_NUMBER_PUNCT_SPACE_CATEGORIES),
                ),
            ),
            # Operator-prefix patterns (e.g. "NOT foo", "NEAR bar")
            st.builds(
                lambda op, term: f"{op} {term}",
                st.sampled_from(["NOT", "NEAR", "AND", "OR"]),
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=_LETTER_NUMBER_CATEGORIES),
                ),
            ),
            # Empty and whitespace
            st.sampled_from(["", " ", "  \t  ", "\n"]),
        )
    )
    assert isinstance(value, str)
    return value


@st.composite
def fts5_match_text_strategy(draw: st.DrawFn) -> str:
    """Generate text suitable for FTS5 indexing and matching.

    Returns text that contains searchable terms — useful for creating
    test data that can be found via FTS5 queries.
    """
    words = draw(
        st.lists(
            st.text(
                min_size=2,
                max_size=15,
                alphabet=st.characters(whitelist_categories=_LETTER_ONLY_CATEGORIES),
            ),
            min_size=1,
            max_size=20,
        )
    )
    return " ".join(words)


@st.composite
def search_with_since_strategy(
    draw: st.DrawFn,
) -> tuple[str, str | None]:
    """Generate a (query, since_date) pair for testing --since filtering.

    Returns a tuple of (search_term, optional_iso_date).
    """
    query = draw(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=_LETTER_NUMBER_CATEGORIES),
        )
    )
    since = draw(
        st.one_of(
            st.none(),
            st.dates().map(lambda d: d.isoformat()),
        )
    )
    return query, since
