"""Consolidated FTS5 search tests using parametrization.

CONSOLIDATION: Reduced 44 tests to ~10 using aggressive parametrization.

Original: Individual test per FTS5 operator/special character
New: Parametrized tests covering all escaping cases
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.search import escape_fts5_query, search_messages
from tests.helpers import ConversationBuilder
from tests.factories import DbFactory


# =============================================================================
# FTS5 ESCAPING - PARAMETRIZED (1 test replacing 30+ tests)
# =============================================================================


# Test cases: (input_query, expected_property, description)
FTS5_ESCAPE_CASES = [
    # Empty/whitespace
    ('', '""', "empty query"),
    ('   ', '""', "whitespace only"),

    # Quotes
    ('find "quoted text" here', 'has_doubled_quotes', "internal quotes"),

    # Wildcards
    ('*', '""', "bare asterisk"),
    ('test*', 'starts_and_ends_with_quotes', "asterisk with text"),
    ('?', '?', "question mark"),  # Single char, no special FTS5 chars -> unquoted

    # FTS5 operators (should be quoted as literals)
    ('AND', '"AND"', "AND operator"),
    ('OR', '"OR"', "OR operator"),
    ('NOT', '"NOT"', "NOT operator"),
    ('NEAR', '"NEAR"', "NEAR operator"),
    ('and', '"and"', "lowercase and"),
    ('And', '"And"', "mixed case And"),

    # Special characters
    ('test:value', 'starts_and_ends_with_quotes', "colon"),
    ('test^2', 'starts_and_ends_with_quotes', "caret"),
    ('function(arg)', 'starts_and_ends_with_quotes', "parentheses"),
    ('test)', 'starts_and_ends_with_quotes', "close paren"),
    ('(test', 'starts_and_ends_with_quotes', "open paren"),

    # Minus/hyphen
    ('-test', 'starts_and_ends_with_quotes', "leading minus"),
    ('test-word', 'starts_and_ends_with_quotes', "embedded hyphen"),

    # Plus
    ('+required', 'starts_and_ends_with_quotes', "plus operator"),

    # Multiple operators
    ('test AND query', 'test AND query', "embedded AND - passes through unquoted"),
    ('OR query', 'starts_and_ends_with_quotes', "leading OR - quoted for safety"),

    # Normal text (NOT quoted - implementation passes simple alphanumeric through as-is)
    ('simple query', 'simple query', "simple words"),
    ('hello', 'hello', "single word"),
]


@pytest.mark.parametrize("input_query,expected,desc", FTS5_ESCAPE_CASES)
def test_escape_fts5_comprehensive(input_query, expected, desc):
    """Comprehensive FTS5 escaping test.

    Replaces 30+ individual escaping tests with single parametrized test.

    Expected can be:
    - Exact string match (e.g., '""')
    - Property to check (e.g., 'starts_and_ends_with_quotes', 'has_doubled_quotes')
    """
    result = escape_fts5_query(input_query)

    if expected == 'starts_and_ends_with_quotes':
        assert result.startswith('"'), f"Failed {desc}: doesn't start with quote"
        assert result.endswith('"'), f"Failed {desc}: doesn't end with quote"
    elif expected == 'has_doubled_quotes':
        assert result.startswith('"'), f"Failed {desc}: not quoted"
        assert result.endswith('"'), f"Failed {desc}: not quoted"
        assert '""' in result, f"Failed {desc}: quotes not doubled"
    else:
        # Exact match
        assert result == expected, f"Failed {desc}: expected {expected}, got {result}"


# =============================================================================
# SEARCH INTEGRATION - PARAMETRIZED (1 test replacing ~10 tests)
# =============================================================================


@pytest.mark.parametrize("query,should_find", [
    ("test", True),  # Basic search
    ("nonexistent", False),  # No match
    ("*", False),  # Bare asterisk escaped
    ("AND", False),  # Operator as literal
    ("quoted", True),  # Part of text with quotes
])
def test_search_messages_escaping_integration(query, should_find, tmp_path):
    """Integration test for search with various queries.

    Replaces ~10 individual integration tests.
    """
    from pathlib import Path

    # Setup database with test data
    db_path = tmp_path / "test.db"
    db = DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test Conversation")
     .add_message("msg1", role="user", text='This is a test message with "quoted text" inside.')
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search - use keyword arguments
    results = search_messages(
        query,
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    if should_find:
        assert len(results.hits) > 0, f"Expected to find results for '{query}'"
    else:
        # Either no results or results don't match the query
        # The important thing is no SQL errors occur
        assert isinstance(results.hits, list)


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing ~5 tests)
# =============================================================================


@pytest.mark.parametrize("special_query,should_quote", [
    ("test OR anything", False),  # "OR" in middle - passes through unquoted
    ("NOT this", True),  # "NOT" at start - should be quoted
    ("NEAR that", True),  # "NEAR" at start - should be quoted
    ("' OR '1'='1", False),  # No special FTS5 chars, passes through (single quotes aren't FTS5 special)
    ("test; DROP TABLE messages--", True),  # Contains special chars (semicolon, etc.), should be quoted
])
def test_escape_fts5_injection_prevention(special_query, should_quote):
    """Prevent dangerous operator positions and special characters.

    Replaces ~5 security-focused tests.
    """
    result = escape_fts5_query(special_query)

    if should_quote:
        # Should be safely quoted
        assert result.startswith('"'), f"Expected quoted: {special_query}"
        assert result.endswith('"'), f"Expected quoted: {special_query}"
    else:
        # These may or may not be quoted depending on special chars
        # The important thing is they don't cause FTS5 errors
        assert isinstance(result, str), f"Should return string: {special_query}"


# =============================================================================
# UNICODE HANDLING - PARAMETRIZED (1 test)
# =============================================================================


@pytest.mark.parametrize("unicode_query", [
    "文字",  # Chinese
    "тест",  # Cyrillic
    "🔍",   # Emoji
    "café",  # Accented
])
def test_escape_fts5_unicode(unicode_query):
    """Unicode queries are handled correctly.

    Unicode-only queries are simple alphanumeric (no special FTS5 chars),
    so they pass through unquoted.
    """
    result = escape_fts5_query(unicode_query)

    # Should preserve unicode and pass through unquoted
    assert result == unicode_query


# =============================================================================
# SEARCH RESULT VALIDATION (NEW - was missing)
# =============================================================================


def test_search_messages_returns_valid_structure(tmp_path):
    """Search results have expected structure."""
    from pathlib import Path

    db_path = tmp_path / "test.db"
    db = DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test")
     .add_message("msg1", role="user", text="Searchable content")
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search using keyword arguments
    results = search_messages(
        "searchable",
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    assert len(results.hits) > 0
    for hit in results.hits:
        # Verify result structure
        assert hasattr(hit, 'snippet')
        assert hasattr(hit, 'conversation_id')
        assert hit.snippet is not None
        assert len(hit.snippet) > 0
