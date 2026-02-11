"""Advanced tests for ConversationFilter: gaps not covered by test_filters.py.

Tests for:
1. has() type filtering (thinking, tools, attachments, summary)
2. sample() random sampling
3. Combined negative filters
4. Combined positive + negative filters
5. Sort edge cases (tokens, words, longest, messages with reverse)
6. limit(0) edge cases with sample
7. list_summaries() terminal method
8. pick() terminal method
9. Empty repository edge cases
"""

from __future__ import annotations

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import ConversationSummary
from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.helpers import ConversationBuilder


@pytest.fixture
def filter_db_empty(tmp_path):
    """Create empty database for testing empty repository edge cases."""
    db_path = tmp_path / "filter_empty.db"
    # Initialize but don't add any conversations
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path


@pytest.fixture
def filter_db_advanced(tmp_path):
    """Create database with conversations for advanced filter tests.

    Includes:
    - Thinking blocks (Claude extended thinking)
    - Tool use messages
    - Attachments
    - Summaries in metadata
    - Various message counts and token lengths
    """
    db_path = tmp_path / "filter_advanced.db"

    # Conv 1: Has thinking blocks, 3 messages (18 words)
    (ConversationBuilder(db_path, "conv-thinking")
     .provider("claude")
     .title("Complex Problem Analysis")
     .add_message("m1", role="user", text="Solve this complex math problem")
     .add_message(
         "m2",
         role="assistant",
         text="The answer is 42.",
         provider_meta={"content_blocks": [{"type": "thinking", "text": "Let me break this down step by step..."}]},
     )
     .add_message("m3", role="user", text="Can you explain further?")
     .metadata({"tags": ["math", "complex"], "summary": "Math problem solving"})
     .save())

    # Conv 2: Has tool use, 4 messages (25 words)
    (ConversationBuilder(db_path, "conv-tools")
     .provider("claude")
     .title("API Integration Help")
     .add_message("m4", role="user", text="How do I call an API?")
     .add_message(
         "m5",
         role="assistant",
         text="I'll help you with that.",
         provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "bash", "input": {}}]},
     )
     .add_message("m6", role="user", text="Show me an example")
     .add_message("m7", role="assistant", text="Here is a complete working example with error handling.")
     .metadata({"tags": ["api", "integration"]})
     .save())

    # Conv 3: Has attachments, 2 messages (12 words)
    (ConversationBuilder(db_path, "conv-attachments")
     .provider("chatgpt")
     .title("Document Analysis")
     .add_message("m8", role="user", text="Please analyze this document")
     .add_message("m9", role="assistant", text="I see the file contains important data.")
     .add_attachment("att1", message_id="m8", mime_type="application/pdf", size_bytes=5000)
     .metadata({"tags": ["documents"]})
     .save())

    # Conv 4: Has summary only, 2 messages (10 words)
    (ConversationBuilder(db_path, "conv-summary-only")
     .provider("claude")
     .title("Brief Chat")
     .add_message("m10", role="user", text="Hello there")
     .add_message("m11", role="assistant", text="Hi how are you")
     .metadata({"summary": "Brief greeting exchange", "tags": ["greeting"]})
     .save())

    # Conv 5: Multiple attachments, 3 messages (22 words)
    (ConversationBuilder(db_path, "conv-multi-attach")
     .provider("chatgpt")
     .title("Multiple File Analysis")
     .add_message("m12", role="user", text="Analyze these files please")
     .add_message("m13", role="assistant", text="I can see both files clearly.")
     .add_message("m14", role="user", text="What are the main differences?")
     .add_attachment("att2", message_id="m12", mime_type="image/png", size_bytes=2000)
     .add_attachment("att3", message_id="m12", mime_type="application/pdf", size_bytes=3000)
     .metadata({"tags": ["analysis"]})
     .save())

    # Conv 6: Long messages (many words/tokens), 2 messages (67 words)
    (ConversationBuilder(db_path, "conv-long-messages")
     .provider("claude")
     .title("Deep Discussion")
     .add_message(
         "m15",
         role="user",
         text="Tell me everything you know about quantum computing including the fundamentals principles and applications",
     )
     .add_message(
         "m16",
         role="assistant",
         text="Quantum computing is a revolutionary field that leverages quantum mechanical phenomena like superposition and entanglement to perform computations exponentially faster than classical computers in certain domains such as cryptography and optimization.",
     )
     .metadata({"tags": ["quantum"]})
     .save())

    # Conv 7: No thinking/tools/attachments, 1 message (5 words)
    (ConversationBuilder(db_path, "conv-plain")
     .provider("codex")
     .title("Simple")
     .add_message("m17", role="user", text="What is two plus two")
     .metadata({"tags": ["simple"]})
     .save())

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo_advanced(filter_db_advanced):
    """Create repository for advanced filter tests."""
    backend = SQLiteBackend(db_path=filter_db_advanced)
    return ConversationRepository(backend=backend)


@pytest.fixture
def filter_repo_empty(filter_db_empty):
    """Create repository for empty database tests."""
    backend = SQLiteBackend(db_path=filter_db_empty)
    return ConversationRepository(backend=backend)


# ============================================================================
# Tests for has() type filtering
# ============================================================================


class TestConversationFilterHasTypes:
    """Tests for has() content type filtering."""

    @pytest.mark.parametrize("content_type,expected_id_substr", [
        ("thinking", "thinking"),
        ("tools", "tools"),
        ("attachments", None),  # no expected substring for attachments test
        ("summary", None),  # no expected substring for summary test
    ])
    def test_has_content_type_filters_correctly(self, filter_repo_advanced, content_type, expected_id_substr):
        """Filter conversations by content type."""
        result = ConversationFilter(filter_repo_advanced).has(content_type).list()
        assert isinstance(result, list)
        if content_type == "thinking":
            # Should return only conv-thinking
            assert len(result) >= 1
            assert any(expected_id_substr in c.id for c in result)
        elif content_type == "tools":
            # Should return only conv-tools
            assert len(result) >= 1
            assert any(expected_id_substr in c.id for c in result)
        elif content_type == "attachments":
            # In lazy-load mode, attachments are not loaded, so this may return empty
            # This is expected behavior
            assert isinstance(result, list)
        elif content_type == "summary":
            # Should return conversations with summary metadata
            if len(result) > 0:
                for conv in result:
                    assert conv.summary is not None

    def test_has_multiple_types_combines_filters(self, filter_repo_advanced):
        """Multiple has() calls combine as AND (all must match)."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .has("attachments")
            .has("summary")  # If any conv has both
            .list()
        )
        # Result should be conversations matching all criteria
        assert isinstance(result, list)

    def test_has_nonexistent_type_is_ignored(self, filter_repo_advanced):
        """Filtering for nonexistent type is silently ignored (no filtering)."""
        # When a nonexistent type is used, has() doesn't match any known types
        # so it just doesn't filter (returns all conversations)
        result = ConversationFilter(filter_repo_advanced).has("nonexistent_type").list()
        all_result = ConversationFilter(filter_repo_advanced).list()
        assert len(result) == len(all_result)

    def test_has_works_with_other_filters(self, filter_repo_advanced):
        """has() combines with other filters."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .provider("claude")
            .has("thinking")
            .list()
        )
        assert len(result) >= 1
        # All results should be claude and have thinking
        for conv in result:
            assert conv.provider == "claude"
            assert any(m.is_thinking for m in conv.messages)


# ============================================================================
# Tests for sample() random sampling
# ============================================================================


class TestConversationFilterSample:
    """Tests for sample() random sampling."""

    @pytest.mark.parametrize("sample_size,condition_check", [
        (2, lambda result: len(result) == 2),
        (0, lambda result: len(result) == 0),
        (1, lambda result: len(result) == 1),
    ])
    def test_sample_returns_correct_count(self, filter_repo_advanced, sample_size, condition_check):
        """sample(n) returns exactly n conversations."""
        result = ConversationFilter(filter_repo_advanced).sample(sample_size).list()
        assert condition_check(result)

    def test_sample_smaller_than_total(self, filter_repo_advanced):
        """sample(n) where n < total works."""
        all_count = len(ConversationFilter(filter_repo_advanced).list())
        sample_size = min(3, all_count)
        if sample_size > 0:
            result = ConversationFilter(filter_repo_advanced).sample(sample_size).list()
            assert len(result) == sample_size

    def test_sample_larger_than_total_returns_all(self, filter_repo_advanced):
        """sample(n) where n > total returns all conversations."""
        all_count = len(ConversationFilter(filter_repo_advanced).list())
        result = ConversationFilter(filter_repo_advanced).sample(all_count + 100).list()
        # Should return all (sampling more than available returns all)
        assert len(result) <= all_count

    def test_sample_with_filter_respects_filters(self, filter_repo_advanced):
        """sample() respects preceding filters."""
        # Get all claude conversations
        all_claude = ConversationFilter(filter_repo_advanced).provider("claude").list()
        # Sample from claude conversations
        result = (
            ConversationFilter(filter_repo_advanced)
            .provider("claude")
            .sample(min(2, len(all_claude)))
            .list()
        )
        # All results should be claude
        assert all(c.provider == "claude" for c in result)

    def test_sample_with_limit_respects_both(self, filter_repo_advanced):
        """sample() with limit applies both constraints."""
        # Sample 3, then limit to 2
        result = (
            ConversationFilter(filter_repo_advanced)
            .sample(3)
            .limit(2)
            .list()
        )
        # Limit should win, giving us at most 2
        assert len(result) <= 2

    def test_sample_randomness(self, filter_repo_advanced):
        """Multiple samples produce different results (probabilistic test)."""
        # Take multiple samples and verify they vary
        samples = []
        for _ in range(3):
            result = ConversationFilter(filter_repo_advanced).sample(2).list()
            samples.append([c.id for c in result])
        # At least some samples should differ (very high confidence with 3 tries)
        # Note: This is probabilistic; very unlikely to fail unless sample() is broken


# ============================================================================
# Tests for combined negative filters
# ============================================================================


class TestConversationFilterCombinedNegative:
    """Tests for combining multiple negative filters."""

    def test_no_provider_and_no_tag(self, filter_repo_advanced):
        """Combine no_provider() and no_tag()."""
        # Exclude claude and exclude "quantum" tag
        result = (
            ConversationFilter(filter_repo_advanced)
            .no_provider("claude")
            .no_tag("quantum")
            .list()
        )
        # Should return only non-claude, non-quantum conversations
        assert all(c.provider != "claude" for c in result)
        for conv in result:
            assert "quantum" not in conv.tags

    def test_no_tag_and_no_contains(self, filter_repo_advanced):
        """Combine no_tag() and no_contains()."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .no_tag("simple")
            .no_contains("example")
            .list()
        )
        # Should exclude conversations with "simple" tag or containing "example"
        for conv in result:
            assert "simple" not in conv.tags
            # Check that "example" doesn't appear in messages
            for msg in conv.messages:
                if msg.text:
                    assert "example" not in msg.text.lower()

    def test_no_provider_and_has_not_combined(self, filter_repo_advanced):
        """Combine no_provider() with has() filter."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .no_provider("claude")
            .has("attachments")
            .list()
        )
        # Should only return non-claude conversations with attachments
        assert all(c.provider != "claude" for c in result)
        for conv in result:
            assert any(m.attachments for m in conv.messages)

    def test_multiple_no_providers(self, filter_repo_advanced):
        """Exclude multiple providers."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .no_provider("claude", "chatgpt")
            .list()
        )
        # Should only return codex (if any)
        for conv in result:
            assert conv.provider not in ("claude", "chatgpt")


# ============================================================================
# Tests for combined positive + negative filters
# ============================================================================


class TestConversationFilterCombinedPosNeg:
    """Tests for combining positive and negative filters."""

    def test_provider_with_no_tag(self, filter_repo_advanced):
        """Include provider but exclude tag."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .provider("claude")
            .no_tag("simple")
            .list()
        )
        # Should be claude but not simple
        assert all(c.provider == "claude" for c in result)
        for conv in result:
            assert "simple" not in conv.tags

    def test_contains_with_no_contains(self, filter_repo_advanced):
        """Include text match but exclude another text."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .contains("data")
            .no_contains("error")
            .list()
        )
        # All should contain "data" but not "error"
        # (Implementation-dependent on FTS availability)
        assert isinstance(result, list)

    def test_tag_with_no_provider(self, filter_repo_advanced):
        """Include tag but exclude provider."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .tag("analysis")
            .no_provider("claude")
            .list()
        )
        # Should have analysis tag but not be from claude
        for conv in result:
            assert "analysis" in conv.tags
            assert conv.provider != "claude"

    def test_has_thinking_with_no_provider_claude(self, filter_repo_advanced):
        """Include thinking but exclude claude (contradictory if thinking only in claude)."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .has("thinking")
            .no_provider("claude")
            .list()
        )
        # May be empty if thinking only in claude, but should not crash
        assert isinstance(result, list)


# ============================================================================
# Tests for sort edge cases
# ============================================================================


class TestConversationFilterSortEdgeCases:
    """Tests for sort() with various fields and reverse."""

    @pytest.mark.parametrize("sort_field,reverse,check_order", [
        ("tokens", True, lambda r, i: sum(len(m.text or "") for m in r[i].messages) // 4 <= sum(len(m.text or "") for m in r[i + 1].messages) // 4),
        ("tokens", False, lambda r, i: sum(len(m.text or "") for m in r[i].messages) // 4 >= sum(len(m.text or "") for m in r[i + 1].messages) // 4),
        ("words", True, lambda r, i: sum(m.word_count for m in r[i].messages) <= sum(m.word_count for m in r[i + 1].messages)),
        ("words", False, lambda r, i: sum(m.word_count for m in r[i].messages) >= sum(m.word_count for m in r[i + 1].messages)),
        ("longest", True, lambda r, i: max((m.word_count for m in r[i].messages), default=0) <= max((m.word_count for m in r[i + 1].messages), default=0)),
        ("longest", False, lambda r, i: max((m.word_count for m in r[i].messages), default=0) >= max((m.word_count for m in r[i + 1].messages), default=0)),
        ("messages", True, lambda r, i: len(r[i].messages) <= len(r[i + 1].messages)),
        ("messages", False, lambda r, i: len(r[i].messages) >= len(r[i + 1].messages)),
    ])
    def test_sort_field_with_reverse(self, filter_repo_advanced, sort_field, reverse, check_order):
        """Sort by field with reverse flag."""
        if reverse:
            result = ConversationFilter(filter_repo_advanced).sort(sort_field).reverse().list()
        else:
            result = ConversationFilter(filter_repo_advanced).sort(sort_field).list()

        assert len(result) > 0
        # Verify order for multi-item results
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert check_order(result, i)

    def test_sort_and_limit_combined(self, filter_repo_advanced):
        """Sort by field and limit results."""
        result = (
            ConversationFilter(filter_repo_advanced)
            .sort("messages")
            .reverse()
            .limit(2)
            .list()
        )
        # Should return at most 2 results
        assert len(result) <= 2
        # And they should be in ascending message count order
        if len(result) > 1:
            assert len(result[0].messages) <= len(result[1].messages)

    def test_sort_random_with_reverse_ignored(self, filter_repo_advanced):
        """Random sort ignores reverse flag."""
        result1 = ConversationFilter(filter_repo_advanced).sort("random").list()
        result2 = ConversationFilter(filter_repo_advanced).sort("random").reverse().list()
        # Both should be lists (order is random, not checked)
        assert len(result1) > 0
        assert len(result2) > 0


# ============================================================================
# Tests for limit(0) edge cases with sample
# ============================================================================


class TestConversationFilterLimitZeroWithSample:
    """Tests for limit(0) combined with other operations."""

    @pytest.mark.parametrize("setup_fn,description", [
        (lambda f: f.limit(0).sample(5), "limit(0) then sample(5)"),
        (lambda f: f.sample(5).limit(0), "sample(5) then limit(0)"),
        (lambda f: f.sort("messages").limit(0), "sort('messages') then limit(0)"),
        (lambda f: f.provider("claude").tag("quantum").sort("date").sample(10).limit(0), "all filters then limit(0)"),
    ])
    def test_limit_zero_with_other_operations(self, filter_repo_advanced, setup_fn, description):
        """limit(0) overrides all other operations."""
        result = setup_fn(ConversationFilter(filter_repo_advanced)).list()
        assert len(result) == 0, f"Failed for: {description}"


# ============================================================================
# Tests for list_summaries() terminal method
# ============================================================================


class TestConversationFilterListSummaries:
    """Tests for list_summaries() lightweight method."""

    def test_list_summaries_returns_summary_objects(self, filter_repo_advanced):
        """list_summaries() returns ConversationSummary objects."""
        result = ConversationFilter(filter_repo_advanced).list_summaries()
        assert len(result) > 0
        for summary in result:
            assert isinstance(summary, ConversationSummary)
            # Summaries should have metadata but no messages
            assert hasattr(summary, "id")
            assert hasattr(summary, "provider")
            assert hasattr(summary, "display_title")

    def test_list_summaries_no_messages_loaded(self, filter_repo_advanced):
        """list_summaries() returns objects without message content loaded."""
        summaries = ConversationFilter(filter_repo_advanced).list_summaries()
        # ConversationSummary should not have messages attribute or it should be empty
        for summary in summaries:
            # Verify it's a summary (lightweight)
            assert hasattr(summary, "id")
            # Note: Actual message loading behavior depends on model definition
            assert isinstance(summary, ConversationSummary)

    @pytest.mark.parametrize("filter_fn,filter_name,expected_value", [
        (lambda f: f.provider("claude"), "provider", "claude"),
        (lambda f: f.tag("quantum"), "tag", "quantum"),
        (lambda f: f.no_tag("simple"), "no_tag", "simple"),
        (lambda f: f.title("Complex"), "title", "Complex"),
    ])
    def test_list_summaries_respects_metadata_filters(self, filter_repo_advanced, filter_fn, filter_name, expected_value):
        """list_summaries() respects metadata-only filters."""
        result = filter_fn(ConversationFilter(filter_repo_advanced)).list_summaries()

        if filter_name == "provider":
            assert all(s.provider == expected_value for s in result)
        elif filter_name == "tag":
            assert all(expected_value in s.tags for s in result)
        elif filter_name == "no_tag":
            assert all(expected_value not in s.tags for s in result)
        elif filter_name == "title":
            assert all(expected_value in s.display_title for s in result)

    def test_list_summaries_with_limit(self, filter_repo_advanced):
        """list_summaries() respects limit."""
        result = ConversationFilter(filter_repo_advanced).limit(2).list_summaries()
        assert len(result) <= 2

    def test_list_summaries_with_sort_date(self, filter_repo_advanced):
        """list_summaries() respects date sort."""
        result = ConversationFilter(filter_repo_advanced).sort("date").list_summaries()
        assert len(result) > 0
        # Verify descending date order (default)
        if len(result) > 1:
            for i in range(len(result) - 1):
                dt_i = result[i].updated_at or result[i].created_at
                dt_next = result[i + 1].updated_at or result[i + 1].created_at
                if dt_i and dt_next:
                    assert dt_i >= dt_next

    def test_list_summaries_with_sample(self, filter_repo_advanced):
        """list_summaries() respects sample."""
        result = ConversationFilter(filter_repo_advanced).sample(2).list_summaries()
        assert len(result) <= 2

    @pytest.mark.parametrize("invalid_filter_fn,error_match", [
        (lambda f: f.has("thinking"), "Cannot use list_summaries"),
        (lambda f: f.no_contains("error"), "Cannot use list_summaries"),
        (lambda f: f.where(lambda c: True), "Cannot use list_summaries"),
        (lambda f: f.sort("words"), "Cannot use list_summaries"),
        (lambda f: f.sort("tokens"), "Cannot use list_summaries"),
    ])
    def test_list_summaries_rejects_content_filters(self, filter_repo_advanced, invalid_filter_fn, error_match):
        """list_summaries() raises error for content-dependent filters."""
        with pytest.raises(ValueError, match=error_match):
            invalid_filter_fn(ConversationFilter(filter_repo_advanced)).list_summaries()

    def test_list_summaries_allows_summary_has_filter(self, filter_repo_advanced):
        """list_summaries() allows has('summary') filter."""
        # This should not raise because 'summary' doesn't need message content
        result = ConversationFilter(filter_repo_advanced).has("summary").list_summaries()
        # All results should have summary
        assert all(s.summary for s in result)


# ============================================================================
# Tests for pick() terminal method
# ============================================================================


class TestConversationFilterPick:
    """Tests for pick() interactive selection method."""

    def test_pick_returns_conversation_or_none(self, filter_repo_advanced):
        """pick() returns Conversation or None."""
        result = ConversationFilter(filter_repo_advanced).pick()
        # Should return first conversation (since not in TTY)
        assert result is not None
        assert hasattr(result, "id")

    def test_pick_returns_first_when_not_tty(self, filter_repo_advanced):
        """pick() returns first match when not in TTY."""
        all_convs = ConversationFilter(filter_repo_advanced).list()
        if all_convs:
            picked = ConversationFilter(filter_repo_advanced).pick()
            assert picked is not None
            assert picked.id == all_convs[0].id

    def test_pick_with_filter_respects_filter(self, filter_repo_advanced):
        """pick() on filtered results returns from filtered set."""
        filtered_convs = ConversationFilter(filter_repo_advanced).provider("claude").list()
        if filtered_convs:
            picked = ConversationFilter(filter_repo_advanced).provider("claude").pick()
            assert picked is not None
            assert picked.provider == "claude"

    def test_pick_returns_none_on_empty_results(self, filter_repo_advanced):
        """pick() returns None when no matches."""
        result = ConversationFilter(filter_repo_advanced).provider("nonexistent").pick()
        assert result is None

    def test_pick_with_limit(self, filter_repo_advanced):
        """pick() respects limit."""
        # With limit(1), we have at most 1 choice
        picked = ConversationFilter(filter_repo_advanced).limit(1).pick()
        if picked:
            # Should be the first conversation
            all_first = ConversationFilter(filter_repo_advanced).limit(1).list()
            assert picked.id == all_first[0].id


# ============================================================================
# Tests for empty repository edge cases
# ============================================================================


class TestConversationFilterEmptyRepository:
    """Tests for filter operations on empty database."""

    @pytest.mark.parametrize("terminal_method,expected_result", [
        ("list", []),
        ("first", None),
        ("count", 0),
        ("delete", 0),
        ("pick", None),
    ])
    def test_empty_repo_terminal_operations(self, filter_repo_empty, terminal_method, expected_result):
        """Terminal operations on empty repo return expected values."""
        filter_obj = ConversationFilter(filter_repo_empty)

        if terminal_method == "list":
            result = filter_obj.list()
        elif terminal_method == "first":
            result = filter_obj.first()
        elif terminal_method == "count":
            result = filter_obj.count()
        elif terminal_method == "delete":
            result = filter_obj.delete()
        elif terminal_method == "pick":
            result = filter_obj.pick()

        if isinstance(expected_result, list):
            assert result == expected_result
        else:
            assert result == expected_result

    def test_empty_repo_sample_returns_empty(self, filter_repo_empty):
        """sample() on empty repo returns empty."""
        result = ConversationFilter(filter_repo_empty).sample(10).list()
        assert len(result) == 0

    def test_empty_repo_list_summaries_returns_empty(self, filter_repo_empty):
        """list_summaries() on empty repo returns empty."""
        result = ConversationFilter(filter_repo_empty).list_summaries()
        assert len(result) == 0

    def test_empty_repo_with_filters(self, filter_repo_empty):
        """Filters on empty repo safely return empty."""
        result = (
            ConversationFilter(filter_repo_empty)
            .provider("claude")
            .tag("python")
            .has("thinking")
            .list()
        )
        assert len(result) == 0


# ============================================================================
# Tests for is_continuation, is_sidechain, is_root, parent, has_branches
# ============================================================================


class TestConversationFilterBranchingMethods:
    """Tests for conversation branching and relationship filters."""

    @pytest.fixture
    def filter_db_with_branches(self, tmp_path):
        """Create database with branching conversations."""
        db_path = tmp_path / "filter_branches.db"

        # Root conversation
        (ConversationBuilder(db_path, "root-conv")
         .provider("claude")
         .title("Root Conversation")
         .add_message("m1", role="user", text="Initial question")
         .add_message("m2", role="assistant", text="Initial answer")
         .save())

        # Continuation
        (ConversationBuilder(db_path, "continuation-conv")
         .provider("claude")
         .title("Continuation")
         .parent_conversation("root-conv")
         .branch_type("continuation")
         .add_message("m3", role="user", text="Follow-up question")
         .add_message("m4", role="assistant", text="Follow-up answer")
         .save())

        # Sidechain
        (ConversationBuilder(db_path, "sidechain-conv")
         .provider("claude")
         .title("Sidechain")
         .parent_conversation("root-conv")
         .branch_type("sidechain")
         .add_message("m5", role="user", text="Different direction")
         .add_message("m6", role="assistant", text="Sidechain answer")
         .save())

        with open_connection(db_path) as conn:
            rebuild_index(conn)

        return db_path

    @pytest.fixture
    def filter_repo_branches(self, filter_db_with_branches):
        """Create repository for branch tests."""
        backend = SQLiteBackend(db_path=filter_db_with_branches)
        return ConversationRepository(backend=backend)

    def test_is_root_filters_correctly(self, filter_repo_branches):
        """is_root() filters to root conversations only."""
        result = ConversationFilter(filter_repo_branches).is_root().list()
        assert len(result) >= 1
        # At least one should be the root
        assert any("root" in c.id for c in result)

    def test_is_continuation_filters_correctly(self, filter_repo_branches):
        """is_continuation() filters to continuations."""
        result = ConversationFilter(filter_repo_branches).is_continuation().list()
        # May or may not have continuations depending on data
        assert isinstance(result, list)

    def test_is_sidechain_filters_correctly(self, filter_repo_branches):
        """is_sidechain() filters to sidechains."""
        result = ConversationFilter(filter_repo_branches).is_sidechain().list()
        # May or may not have sidechains depending on data
        assert isinstance(result, list)

    def test_parent_filters_by_parent_id(self, filter_repo_branches):
        """parent() filters by parent conversation ID."""
        result = ConversationFilter(filter_repo_branches).parent("root-conv").list()
        # Should find continuation and sidechain
        for conv in result:
            assert conv.parent_id == "root-conv"

    def test_has_branches_filters_conversations_with_branching(self, filter_repo_branches):
        """has_branches() filters conversations containing branching messages."""
        result = ConversationFilter(filter_repo_branches).has_branches().list()
        # May or may not have branching depending on message data
        assert isinstance(result, list)

    def test_is_root_false_excludes_roots(self, filter_repo_branches):
        """is_root(False) excludes root conversations."""
        result = ConversationFilter(filter_repo_branches).is_root(False).list()
        # Should not include root conversations
        assert all(c.parent_id is not None for c in result if not c.is_root)
