"""Tests for polylogue.search module FTS5 escaping."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.storage.db import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.store import ConversationRecord, MessageRecord, store_records
from tests.factories import DbFactory


class TestFTS5QueryEscaping:
    """Tests for FTS5 query string escaping via escape_fts5_query()."""

    def test_escape_fts5_empty_query(self):
        """Empty query should return safe default."""
        result = escape_fts5_query('')
        assert result == '""'

    def test_escape_fts5_whitespace_only(self):
        """Whitespace-only query should return safe default."""
        result = escape_fts5_query('   ')
        assert result == '""'

    def test_escape_fts5_quotes(self):
        """Quotes in search query should be escaped."""
        result = escape_fts5_query('find "quoted text" here')
        # Should be wrapped and internal quotes doubled
        assert result.startswith('"')
        assert result.endswith('"')
        # Internal quotes should be doubled
        assert '""' in result

    def test_escape_fts5_bare_asterisk(self):
        """Bare asterisk (dangerous prefix search) should be escaped."""
        result = escape_fts5_query('*')
        assert result == '""'  # Should not allow bare asterisk

    def test_escape_fts5_asterisk_with_text(self):
        """Asterisk attached to text (valid prefix search) should be preserved in quotes."""
        result = escape_fts5_query('test*')
        # Asterisk makes it special, so should be quoted
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_operators_and(self):
        """AND operator as literal should be quoted."""
        result = escape_fts5_query('AND')
        assert result == '"AND"'

    def test_escape_fts5_operators_or(self):
        """OR operator as literal should be quoted."""
        result = escape_fts5_query('OR')
        assert result == '"OR"'

    def test_escape_fts5_operators_not(self):
        """NOT operator as literal should be quoted."""
        result = escape_fts5_query('NOT')
        assert result == '"NOT"'

    def test_escape_fts5_operators_near(self):
        """NEAR operator as literal should be quoted."""
        result = escape_fts5_query('NEAR')
        assert result == '"NEAR"'

    def test_escape_fts5_operators_case_insensitive(self):
        """Operators should be recognized case-insensitively."""
        result_lower = escape_fts5_query('and')
        result_mixed = escape_fts5_query('And')
        # Both should be quoted
        assert result_lower == '"and"'
        assert result_mixed == '"And"'

    def test_escape_fts5_colon(self):
        """Colon (column search operator) should be escaped."""
        result = escape_fts5_query('test:value')
        assert result.startswith('"')
        assert result.endswith('"')
        assert ':' in result  # Colon is inside quotes now

    def test_escape_fts5_caret(self):
        """Caret (boost operator) should be escaped."""
        result = escape_fts5_query('test^2')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_parentheses_open(self):
        """Open parenthesis (grouping) should be escaped."""
        result = escape_fts5_query('function(arg)')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_parentheses_close(self):
        """Close parenthesis (grouping) should be escaped."""
        result = escape_fts5_query('test)')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_hyphen(self):
        """Hyphen (NOT operator) should be escaped."""
        result = escape_fts5_query('test-word')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_plus(self):
        """Plus (inclusion operator) should be escaped."""
        result = escape_fts5_query('+test')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_pipe(self):
        """Pipe (should not be FTS5 operator but escape anyway) should be escaped."""
        result = escape_fts5_query('a|b')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_ampersand(self):
        """Ampersand should be escaped."""
        result = escape_fts5_query('a&b')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_exclamation(self):
        """Exclamation mark should be escaped."""
        result = escape_fts5_query('test!')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_curly_braces(self):
        """Curly braces should be escaped."""
        result = escape_fts5_query('{test}')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_square_brackets(self):
        """Square brackets should be escaped."""
        result = escape_fts5_query('[test]')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_unclosed_quote(self):
        """Unclosed quote should be escaped."""
        result = escape_fts5_query('"unclosed')
        assert result.startswith('"')
        assert result.endswith('"')
        # Internal quote should be doubled
        assert '""' in result

    def test_escape_fts5_multiple_quotes(self):
        """Multiple quotes should be escaped."""
        result = escape_fts5_query('"first" and "second"')
        assert result.startswith('"')
        assert result.endswith('"')
        assert result.count('""') >= 4  # Each quote doubled

    def test_escape_fts5_simple_word(self):
        """Simple alphanumeric word should pass through."""
        result = escape_fts5_query('hello')
        assert result == 'hello'

    def test_escape_fts5_simple_words(self):
        """Simple words separated by spaces should pass through."""
        result = escape_fts5_query('hello world')
        assert result == 'hello world'

    def test_escape_fts5_numbers(self):
        """Numbers should pass through."""
        result = escape_fts5_query('12345')
        assert result == '12345'

    def test_escape_fts5_mixed_safe_chars(self):
        """Safe mix of letters, numbers, spaces should pass through."""
        result = escape_fts5_query('test message 123 ok')
        assert result == 'test message 123 ok'

    def test_escape_fts5_unicode(self):
        """Unicode characters without special chars should pass through."""
        result = escape_fts5_query('café')
        assert result == 'café'

    def test_escape_fts5_unicode_with_special(self):
        """Unicode with special chars should be quoted."""
        result = escape_fts5_query('café:special')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_escape_fts5_all_asterisks(self):
        """Multiple asterisks should be escaped."""
        result = escape_fts5_query('***')
        assert result == '""'

    def test_escape_fts5_leading_trailing_spaces(self):
        """Leading/trailing spaces should be stripped."""
        result = escape_fts5_query('  hello world  ')
        assert result == 'hello world'


class TestSearchWithSpecialCharacters:
    """Integration tests: search with special characters should not raise SQL errors."""

    @pytest.fixture
    def indexed_db(self, tmp_path):
        """Create a database with FTS index."""
        db_path = tmp_path / "search.db"
        with open_connection(db_path) as conn:
            # Create conversation with messages
            conv = ConversationRecord(
                conversation_id="c1",
                provider_name="test",
                provider_conversation_id="ext-c1",
                title="Test Conversation",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                content_hash=uuid4().hex,
                version=1,
            )

            msgs = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="c1",
                    role="user",
                    text="hello world",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                    version=1,
                ),
                MessageRecord(
                    message_id="m2",
                    conversation_id="c1",
                    role="assistant",
                    text="test:value with special characters",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                    version=1,
                ),
            ]

            store_records(conversation=conv, messages=msgs, attachments=[], conn=conn)
            rebuild_index(conn)

        return db_path

    @pytest.fixture
    def archive_root(self, tmp_path):
        """Create archive root with render directory."""
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "render").mkdir()
        return archive

    def test_search_unclosed_quote_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with unclosed quote should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('"unclosed quote', archive_root=archive_root)
            # Should return SearchResult, even if empty
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Unclosed quote raised {exc}")

    def test_search_bare_or_operator_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with bare OR should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('test OR', archive_root=archive_root)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Bare OR raised {exc}")

    def test_search_unbalanced_paren_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with unbalanced parenthesis should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('(unbalanced', archive_root=archive_root)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Unbalanced paren raised {exc}")

    def test_search_colon_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with colon should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('col:umn', archive_root=archive_root)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Colon raised {exc}")

    def test_search_multiple_asterisks_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with multiple asterisks should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('***', archive_root=archive_root)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Multiple asterisks raised {exc}")

    def test_search_consecutive_operators_no_error(self, indexed_db, archive_root, monkeypatch):
        """Search with consecutive AND/OR should not raise SQL error."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        try:
            result = search_messages('a AND AND b', archive_root=archive_root)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"Consecutive operators raised {exc}")

    def test_search_all_special_chars_batch(self, indexed_db, archive_root, monkeypatch):
        """Batch test: all special characters should not crash."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

        dangerous_queries = [
            '"unclosed quote',
            'test OR',
            '(unbalanced',
            'col:umn',
            '***',
            'a AND AND b',
            ')',
            '^boost',
            '{test}',
            '[test]',
            'a|b',
            'a&b',
            'test!',
            'a+b',
            'a-b',
        ]

        for query in dangerous_queries:
            try:
                result = search_messages(query, archive_root=archive_root)
                assert result is not None, f"Query {query!r} returned None"
            except Exception as exc:
                pytest.fail(f"Query {query!r} raised {exc}")


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @pytest.fixture
    def search_indexed_db(self, tmp_path):
        """Create indexed database with various content for search tests."""
        db_path = tmp_path / "search_integration.db"

        with open_connection(db_path) as conn:
            factory = DbFactory(db_path)

            # Create multiple conversations with searchable content
            factory.create_conversation(
                id="c1",
                title="Python Tips",
                messages=[
                    {
                        "id": "m1",
                        "role": "user",
                        "text": "How do I use Python decorators?",
                    },
                    {
                        "id": "m2",
                        "role": "assistant",
                        "text": "Decorators are functions that modify other functions.",
                    },
                ],
            )

            factory.create_conversation(
                id="c2",
                title="API Design",
                messages=[
                    {
                        "id": "m3",
                        "role": "user",
                        "text": "What's best practice for REST APIs?",
                    },
                    {
                        "id": "m4",
                        "role": "assistant",
                        "text": "Use consistent naming and proper HTTP methods.",
                    },
                ],
            )

            # Rebuild FTS index
            rebuild_index(conn)

        return db_path

    @pytest.fixture
    def search_archive_root(self, tmp_path):
        """Create archive root for search tests."""
        archive = tmp_path / "search_archive"
        archive.mkdir()
        (archive / "render").mkdir()
        return archive

    def test_search_simple_word(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search should find simple words."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("decorators", archive_root=search_archive_root)
        assert len(result.hits) > 0
        assert any("decorators" in h.snippet.lower() for h in result.hits)

    def test_search_multiple_words(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search should find multiple word queries."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("HTTP methods", archive_root=search_archive_root)
        assert len(result.hits) > 0

    def test_search_returns_snippet(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search results should include snippets."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("decorators", archive_root=search_archive_root)
        assert len(result.hits) > 0
        hit = result.hits[0]
        assert hit.snippet is not None
        assert len(hit.snippet) > 0

    def test_search_returns_conversation_metadata(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search results should include conversation metadata."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("decorators", archive_root=search_archive_root)
        assert len(result.hits) > 0
        hit = result.hits[0]
        assert hit.conversation_id is not None
        assert hit.provider_name is not None
        assert hit.message_id is not None

    def test_search_respects_limit(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search should respect limit parameter."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("a", archive_root=search_archive_root, limit=1)
        assert len(result.hits) <= 1

    def test_search_empty_result(self, search_indexed_db, search_archive_root, monkeypatch):
        """Search with no matches should return empty result."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(search_archive_root))
        result = search_messages("xyzabc", archive_root=search_archive_root, db_path=search_indexed_db)
        assert result is not None
        assert len(result.hits) == 0
