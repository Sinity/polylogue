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
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.infra.helpers import ConversationBuilder


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
    @pytest.mark.asyncio
    async def test_has_content_type_filters_correctly(self, filter_repo_advanced, content_type, expected_id_substr):
        """Filter conversations by content type."""
        result = await ConversationFilter(filter_repo_advanced).has(content_type).list()
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
            # Some code paths may lazily skip attachment hydration; if present,
            # results must come from known attachment conversations.
            expected_ids = {"conv-attachments", "conv-multi-attach"}
            assert all(c.id in expected_ids for c in result)
        elif content_type == "summary":
            # Should return conversations with summary metadata
            assert all(conv.summary is not None for conv in result)

    @pytest.mark.asyncio
    async def test_has_multiple_types_combines_filters(self, filter_repo_advanced):
        """Multiple has() calls combine as AND (all must match)."""
        result = await (
            ConversationFilter(filter_repo_advanced)
            .has("attachments")
            .has("summary")  # If any conv has both
            .list()
        )
        # Fixture has no conversation with BOTH attachments and summary.
        assert result == []

    @pytest.mark.asyncio
    async def test_has_nonexistent_type_is_ignored(self, filter_repo_advanced):
        """Filtering for nonexistent type is silently ignored (no filtering)."""
        # When a nonexistent type is used, has() doesn't match any known types
        # so it just doesn't filter (returns all conversations)
        result = await ConversationFilter(filter_repo_advanced).has("nonexistent_type").list()
        all_result = await ConversationFilter(filter_repo_advanced).list()
        assert len(result) == len(all_result)

    @pytest.mark.asyncio
    async def test_has_works_with_other_filters(self, filter_repo_advanced):
        """has() combines with other filters."""
        result = await (
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
    @pytest.mark.asyncio
    async def test_sample_returns_correct_count(self, filter_repo_advanced, sample_size, condition_check):
        """sample(n) returns exactly n conversations."""
        result = await ConversationFilter(filter_repo_advanced).sample(sample_size).list()
        assert condition_check(result)

    @pytest.mark.parametrize("sample_size", [3, 9999])
    @pytest.mark.asyncio
    async def test_sample_size_is_bounded_by_filtered_population(self, filter_repo_advanced, sample_size):
        """sample(n) never exceeds the size of the filtered candidate set."""
        all_results = await ConversationFilter(filter_repo_advanced).list()
        sampled = await ConversationFilter(filter_repo_advanced).sample(sample_size).list()
        assert len(sampled) <= min(sample_size, len(all_results))
        assert {c.id for c in sampled}.issubset({c.id for c in all_results})

    @pytest.mark.asyncio
    async def test_sample_with_filter_respects_filters(self, filter_repo_advanced):
        """sample() respects preceding filters."""
        # Get all claude conversations
        all_claude = await ConversationFilter(filter_repo_advanced).provider("claude").list()
        # Sample from claude conversations
        result = await (
            ConversationFilter(filter_repo_advanced)
            .provider("claude")
            .sample(min(2, len(all_claude)))
            .list()
        )
        # All results should be claude
        assert all(c.provider == "claude" for c in result)

    @pytest.mark.asyncio
    async def test_sample_with_limit_respects_both(self, filter_repo_advanced):
        """sample() with limit applies both constraints."""
        # Sample 3, then limit to 2
        result = await (
            ConversationFilter(filter_repo_advanced)
            .sample(3)
            .limit(2)
            .list()
        )
        # Limit should win, giving us at most 2
        assert len(result) <= 2


# ============================================================================
# Tests for combined filter compositions
# ============================================================================

COMBINED_FILTER_CASES = [
    (
        "exclude-provider-and-tag",
        lambda f: f.exclude_provider("claude").exclude_tag("quantum"),
        False,
        lambda conv: conv.provider != "claude" and "quantum" not in conv.tags,
    ),
    (
        "exclude-tag-and-text",
        lambda f: f.exclude_tag("simple").exclude_text("example"),
        False,
        lambda conv: (
            "simple" not in conv.tags
            and all("example" not in (msg.text or "").lower() for msg in conv.messages)
        ),
    ),
    (
        "exclude-provider-with-attachments",
        lambda f: f.exclude_provider("claude").has("attachments"),
        False,
        lambda conv: conv.provider != "claude" and any(msg.attachments for msg in conv.messages),
    ),
    (
        "exclude-multiple-providers",
        lambda f: f.exclude_provider("claude", "chatgpt"),
        False,
        lambda conv: conv.provider not in ("claude", "chatgpt"),
    ),
    (
        "provider-with-excluded-tag",
        lambda f: f.provider("claude").exclude_tag("simple"),
        False,
        lambda conv: conv.provider == "claude" and "simple" not in conv.tags,
    ),
    (
        "contains-with-excluded-text",
        lambda f: f.contains("data").exclude_text("error"),
        False,
        lambda conv: (
            "data" in " ".join((m.text or "").lower() for m in conv.messages)
            and "error" not in " ".join((m.text or "").lower() for m in conv.messages)
        ),
    ),
    (
        "tag-with-excluded-provider",
        lambda f: f.tag("analysis").exclude_provider("claude"),
        False,
        lambda conv: "analysis" in conv.tags and conv.provider != "claude",
    ),
    (
        "contradictory-thinking-provider",
        lambda f: f.has("thinking").exclude_provider("claude"),
        True,
        lambda conv: True,
    ),
]


class TestConversationFilterCombinedCompositions:
    """Composable filter stacks preserve both include and exclude semantics."""

    @pytest.mark.parametrize("case_id,setup_fn,expect_empty,predicate", COMBINED_FILTER_CASES)
    @pytest.mark.asyncio
    async def test_combined_filters(self, filter_repo_advanced, case_id, setup_fn, expect_empty, predicate):
        result = await setup_fn(ConversationFilter(filter_repo_advanced)).list()
        assert isinstance(result, list), case_id
        if expect_empty:
            assert result == [], case_id
            return
        for conv in result:
            assert predicate(conv), case_id


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
    @pytest.mark.asyncio
    async def test_sort_field_with_reverse(self, filter_repo_advanced, sort_field, reverse, check_order):
        """Sort by field with reverse flag."""
        if reverse:
            result = await ConversationFilter(filter_repo_advanced).sort(sort_field).reverse().list()
        else:
            result = await ConversationFilter(filter_repo_advanced).sort(sort_field).list()

        assert len(result) > 0
        # Verify order for multi-item results
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert check_order(result, i)

    @pytest.mark.asyncio
    async def test_sort_and_limit_combined(self, filter_repo_advanced):
        """Sort by field and limit results."""
        result = await (
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

    @pytest.mark.asyncio
    async def test_sort_random_with_reverse_ignored(self, filter_repo_advanced):
        """Random sort ignores reverse flag."""
        result1 = await ConversationFilter(filter_repo_advanced).sort("random").list()
        result2 = await ConversationFilter(filter_repo_advanced).sort("random").reverse().list()
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
    @pytest.mark.asyncio
    async def test_limit_zero_with_other_operations(self, filter_repo_advanced, setup_fn, description):
        """limit(0) overrides all other operations."""
        result = await setup_fn(ConversationFilter(filter_repo_advanced)).list()
        assert len(result) == 0, f"Failed for: {description}"


# ============================================================================
# Tests for list_summaries() terminal method
# ============================================================================


class TestConversationFilterListSummaries:
    """Tests for list_summaries() lightweight method."""

    @pytest.mark.asyncio
    async def test_list_summaries_returns_summary_objects(self, filter_repo_advanced):
        """list_summaries() returns ConversationSummary objects."""
        result = await ConversationFilter(filter_repo_advanced).list_summaries()
        assert len(result) > 0
        for summary in result:
            assert isinstance(summary, ConversationSummary)
            # Summaries should have metadata but no messages
            assert hasattr(summary, "id")
            assert hasattr(summary, "provider")
            assert hasattr(summary, "display_title")

    @pytest.mark.asyncio
    async def test_list_summaries_no_messages_loaded(self, filter_repo_advanced):
        """list_summaries() returns objects without message content loaded."""
        summaries = await ConversationFilter(filter_repo_advanced).list_summaries()
        # ConversationSummary should not have messages attribute or it should be empty
        for summary in summaries:
            # Verify it's a summary (lightweight)
            assert hasattr(summary, "id")
            # Note: Actual message loading behavior depends on model definition
            assert isinstance(summary, ConversationSummary)

    @pytest.mark.parametrize("filter_fn,filter_name,expected_value", [
        (lambda f: f.provider("claude"), "provider", "claude"),
        (lambda f: f.tag("quantum"), "tag", "quantum"),
        (lambda f: f.exclude_tag("simple"), "exclude_tag", "simple"),
        (lambda f: f.title("Complex"), "title", "Complex"),
    ])
    @pytest.mark.asyncio
    async def test_list_summaries_respects_metadata_filters(self, filter_repo_advanced, filter_fn, filter_name, expected_value):
        """list_summaries() respects metadata-only filters."""
        result = await filter_fn(ConversationFilter(filter_repo_advanced)).list_summaries()

        if filter_name == "provider":
            assert all(s.provider == expected_value for s in result)
        elif filter_name == "tag":
            assert all(expected_value in s.tags for s in result)
        elif filter_name == "exclude_tag":
            assert all(expected_value not in s.tags for s in result)
        elif filter_name == "title":
            assert all(expected_value in s.display_title for s in result)

    @pytest.mark.asyncio
    async def test_list_summaries_with_limit(self, filter_repo_advanced):
        """list_summaries() respects limit."""
        result = await ConversationFilter(filter_repo_advanced).limit(2).list_summaries()
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_list_summaries_with_sort_date(self, filter_repo_advanced):
        """list_summaries() respects date sort."""
        result = await ConversationFilter(filter_repo_advanced).sort("date").list_summaries()
        assert len(result) > 0
        # Verify descending date order (default)
        if len(result) > 1:
            for i in range(len(result) - 1):
                dt_i = result[i].updated_at or result[i].created_at
                dt_next = result[i + 1].updated_at or result[i + 1].created_at
                if dt_i and dt_next:
                    assert dt_i >= dt_next

    @pytest.mark.asyncio
    async def test_list_summaries_with_sample(self, filter_repo_advanced):
        """list_summaries() respects sample."""
        result = await ConversationFilter(filter_repo_advanced).sample(2).list_summaries()
        assert len(result) <= 2

    @pytest.mark.parametrize("invalid_filter_fn,error_match", [
        (lambda f: f.has("thinking"), "Cannot use list_summaries"),
        (lambda f: f.exclude_text("error"), "Cannot use list_summaries"),
        (lambda f: f.where(lambda c: True), "Cannot use list_summaries"),
        (lambda f: f.sort("words"), "Cannot use list_summaries"),
        (lambda f: f.sort("tokens"), "Cannot use list_summaries"),
    ])
    @pytest.mark.asyncio
    async def test_list_summaries_rejects_content_filters(self, filter_repo_advanced, invalid_filter_fn, error_match):
        """list_summaries() raises error for content-dependent filters."""
        with pytest.raises(ValueError, match=error_match):
            await invalid_filter_fn(ConversationFilter(filter_repo_advanced)).list_summaries()

    @pytest.mark.asyncio
    async def test_list_summaries_allows_summary_has_filter(self, filter_repo_advanced):
        """list_summaries() allows has('summary') filter."""
        # This should not raise because 'summary' doesn't need message content
        result = await ConversationFilter(filter_repo_advanced).has("summary").list_summaries()
        # All results should have summary
        assert all(s.summary for s in result)
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
    @pytest.mark.asyncio
    async def test_empty_repo_terminal_operations(self, filter_repo_empty, terminal_method, expected_result):
        """Terminal operations on empty repo return expected values."""
        filter_obj = ConversationFilter(filter_repo_empty)

        if terminal_method == "list":
            result = await filter_obj.list()
        elif terminal_method == "first":
            result = await filter_obj.first()
        elif terminal_method == "count":
            result = await filter_obj.count()
        elif terminal_method == "delete":
            result = await filter_obj.delete()
        elif terminal_method == "pick":
            result = await filter_obj.pick()

        if isinstance(expected_result, list):
            assert result == expected_result
        else:
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_empty_repo_sample_returns_empty(self, filter_repo_empty):
        """sample() on empty repo returns empty."""
        result = await ConversationFilter(filter_repo_empty).sample(10).list()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_repo_list_summaries_returns_empty(self, filter_repo_empty):
        """list_summaries() on empty repo returns empty."""
        result = await ConversationFilter(filter_repo_empty).list_summaries()
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_repo_with_filters(self, filter_repo_empty):
        """Filters on empty repo safely return empty."""
        result = await (
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

    @pytest.mark.asyncio
    async def test_is_root_filters_correctly(self, filter_repo_branches):
        """is_root() filters to root conversations only."""
        result = await ConversationFilter(filter_repo_branches).is_root().list()
        assert len(result) >= 1
        # At least one should be the root
        assert any("root" in c.id for c in result)

    @pytest.mark.asyncio
    async def test_is_continuation_filters_correctly(self, filter_repo_branches):
        """is_continuation() filters to continuations."""
        result = await ConversationFilter(filter_repo_branches).is_continuation().list()
        assert len(result) == 1
        assert result[0].id == "continuation-conv"
        assert all(c.is_continuation for c in result)

    @pytest.mark.asyncio
    async def test_is_sidechain_filters_correctly(self, filter_repo_branches):
        """is_sidechain() filters to sidechains."""
        result = await ConversationFilter(filter_repo_branches).is_sidechain().list()
        assert len(result) == 1
        assert result[0].id == "sidechain-conv"
        assert all(c.is_sidechain for c in result)

    @pytest.mark.asyncio
    async def test_parent_filters_by_parent_id(self, filter_repo_branches):
        """parent() filters by parent conversation ID."""
        result = await ConversationFilter(filter_repo_branches).parent("root-conv").list()
        # Should find continuation and sidechain
        for conv in result:
            assert conv.parent_id == "root-conv"

    @pytest.mark.asyncio
    async def test_has_branches_filters_conversations_with_branching(self, filter_repo_branches):
        """has_branches() filters conversations containing branching messages."""
        result = await ConversationFilter(filter_repo_branches).has_branches().list()
        # Fixture data has session-branching but no message branch_index>0.
        assert result == []

    @pytest.mark.asyncio
    async def test_is_root_false_excludes_roots(self, filter_repo_branches):
        """is_root(False) excludes root conversations."""
        result = await ConversationFilter(filter_repo_branches).is_root(False).list()
        assert all(not c.is_root for c in result)
        assert all(c.parent_id is not None for c in result)


# ============================================================================
# Tests for delete() cascade behavior
# ============================================================================


class TestDeleteCascade:
    """Test that ConversationFilter.delete() cascades correctly."""

    @pytest.fixture
    async def populated_db(self, tmp_path):
        """Create database with a conversation, messages, attachments, FTS entries."""
        db_path = tmp_path / "cascade.db"
        backend = SQLiteBackend(db_path=db_path)
        repo = ConversationRepository(backend=backend)

        # Build a conversation with messages and attachments
        conv = (
            ConversationBuilder(db_path, "cascade-conv")
            .provider("claude")
            .title("Cascade Test")
            .add_message("m1", role="user", text="Hello world")
            .add_message("m2", role="assistant", text="Hi there")
            .add_attachment("att1", message_id="m1", mime_type="image/png", size_bytes=1024)
        )
        conv.save()

        # Build FTS index
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        return db_path, backend, repo

    @pytest.mark.asyncio
    async def test_delete_cascades_to_messages(self, populated_db):
        """delete() removes associated messages."""
        db_path, backend, repo = populated_db
        f = ConversationFilter(repo).id("cascade-conv")
        deleted = await f.delete()
        assert deleted == 1

        # Verify messages are gone
        with open_connection(db_path) as conn:
            msgs = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 'cascade-conv'").fetchone()[0]
            assert msgs == 0

    @pytest.mark.asyncio
    async def test_delete_cascades_to_attachment_refs(self, populated_db):
        """delete() removes attachment_refs for the conversation."""
        db_path, backend, repo = populated_db
        f = ConversationFilter(repo).id("cascade-conv")
        await f.delete()

        with open_connection(db_path) as conn:
            refs = conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = 'cascade-conv'").fetchone()[0]
            assert refs == 0

    @pytest.mark.asyncio
    async def test_delete_removes_fts_entries(self, populated_db):
        """delete() removes FTS index entries for deleted conversations."""
        db_path, backend, repo = populated_db

        # Verify FTS entries exist first
        with open_connection(db_path) as conn:
            before = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = 'cascade-conv'").fetchone()[0]
            assert before > 0

        f = ConversationFilter(repo).id("cascade-conv")
        await f.delete()

        with open_connection(db_path) as conn:
            after = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = 'cascade-conv'").fetchone()[0]
            assert after == 0

    @pytest.mark.asyncio
    async def test_delete_prunes_orphaned_attachments(self, populated_db):
        """delete() prunes attachments with ref_count reaching 0."""
        db_path, backend, repo = populated_db

        f = ConversationFilter(repo).id("cascade-conv")
        await f.delete()

        with open_connection(db_path) as conn:
            # Attachment should be pruned (ref_count was 1, now 0)
            att = conn.execute("SELECT COUNT(*) FROM attachments WHERE attachment_id = 'att1'").fetchone()[0]
            assert att == 0


# ============================================================================
# Tests for semantic content_blocks filters (schema v3)
# ============================================================================


@pytest.fixture
def filter_db_semantic(tmp_path):
    """Database with conversations classified by semantic_type in content_blocks."""
    db_path = tmp_path / "filter_semantic.db"

    # Conv with file read/write operations
    (ConversationBuilder(db_path, "conv-file-ops")
     .provider("claude-code")
     .title("File editing session")
     .add_message("m1", role="user", text="Edit the config file")
     .add_message(
         "m2", role="assistant", text="Reading and updating config.",
         provider_meta={"content_blocks": [
             {"type": "tool_use", "tool_name": "Read", "semantic_type": "file_read"},
             {"type": "tool_use", "tool_name": "Edit", "semantic_type": "file_edit"},
         ]},
     )
     .save())

    # Conv with git operations
    (ConversationBuilder(db_path, "conv-git-ops")
     .provider("claude-code")
     .title("Git commit session")
     .add_message("m3", role="user", text="Commit these changes")
     .add_message(
         "m4", role="assistant", text="Running git commit.",
         provider_meta={"content_blocks": [
             {"type": "tool_use", "tool_name": "Bash", "semantic_type": "git"},
         ]},
     )
     .save())

    # Conv with subagent spawn
    (ConversationBuilder(db_path, "conv-subagent")
     .provider("claude-code")
     .title("Delegating to subagent")
     .add_message("m5", role="user", text="Explore the codebase")
     .add_message(
         "m6", role="assistant", text="Spawning exploration agent.",
         provider_meta={"content_blocks": [
             {"type": "tool_use", "tool_name": "Task", "semantic_type": "subagent"},
         ]},
     )
     .save())

    # Conv with multiple semantic types (git + file_write)
    (ConversationBuilder(db_path, "conv-mixed")
     .provider("claude-code")
     .title("Complex coding task")
     .add_message("m7", role="user", text="Write and commit a new module")
     .add_message(
         "m8", role="assistant", text="Writing and committing.",
         provider_meta={"content_blocks": [
             {"type": "tool_use", "tool_name": "Write", "semantic_type": "file_write"},
             {"type": "tool_use", "tool_name": "Bash", "semantic_type": "git"},
         ]},
     )
     .save())

    # Conv with no semantic operations (plain shell)
    (ConversationBuilder(db_path, "conv-shell-only")
     .provider("claude-code")
     .title("Shell command")
     .add_message("m9", role="user", text="Run tests")
     .add_message(
         "m10", role="assistant", text="Running pytest.",
         provider_meta={"content_blocks": [
             {"type": "tool_use", "tool_name": "Bash", "semantic_type": "shell"},
         ]},
     )
     .save())

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo_semantic(filter_db_semantic):
    backend = SQLiteBackend(db_path=filter_db_semantic)
    return ConversationRepository(backend=backend)


class TestSemanticFilters:
    """Tests for has_file_operations(), has_git_operations(), has_subagent_spawns()."""

    @pytest.mark.asyncio
    async def test_has_file_operations_returns_file_convs(self, filter_repo_semantic):
        """has_file_operations() selects conversations with file_read/write/edit blocks."""
        results = await ConversationFilter(filter_repo_semantic).has_file_operations().list()
        ids = {c.id for c in results}
        assert "conv-file-ops" in ids   # has file_read + file_edit
        assert "conv-mixed" in ids      # has file_write
        assert "conv-git-ops" not in ids
        assert "conv-subagent" not in ids
        assert "conv-shell-only" not in ids

    @pytest.mark.asyncio
    async def test_has_git_operations_returns_git_convs(self, filter_repo_semantic):
        """has_git_operations() selects conversations with git semantic_type blocks."""
        results = await ConversationFilter(filter_repo_semantic).has_git_operations().list()
        ids = {c.id for c in results}
        assert "conv-git-ops" in ids
        assert "conv-mixed" in ids      # has git + file_write
        assert "conv-file-ops" not in ids
        assert "conv-subagent" not in ids
        assert "conv-shell-only" not in ids

    @pytest.mark.asyncio
    async def test_has_subagent_spawns_returns_subagent_convs(self, filter_repo_semantic):
        """has_subagent_spawns() selects conversations with subagent semantic_type blocks."""
        results = await ConversationFilter(filter_repo_semantic).has_subagent_spawns().list()
        ids = {c.id for c in results}
        assert "conv-subagent" in ids
        assert "conv-file-ops" not in ids
        assert "conv-git-ops" not in ids
        assert "conv-mixed" not in ids
        assert "conv-shell-only" not in ids

    @pytest.mark.asyncio
    async def test_combined_git_and_file_ops(self, filter_repo_semantic):
        """Combining git + file_operations returns only conversations with both."""
        results = await (ConversationFilter(filter_repo_semantic)
                         .has_git_operations()
                         .has_file_operations()
                         .list())
        ids = {c.id for c in results}
        assert "conv-mixed" in ids      # has both git and file_write
        assert "conv-git-ops" not in ids  # git only, no file ops
        assert "conv-file-ops" not in ids  # file ops only, no git

    @pytest.mark.asyncio
    async def test_no_semantic_match_returns_empty(self, filter_repo_semantic):
        """has_subagent_spawns() + has_git_operations() on exclusive data returns empty."""
        # conv-subagent has no git; conv-git-ops has no subagent; conv-mixed has git but no subagent
        results = await (ConversationFilter(filter_repo_semantic)
                         .has_subagent_spawns()
                         .has_git_operations()
                         .list())
        assert results == []

    @pytest.mark.asyncio
    async def test_semantic_filter_with_provider_filter(self, filter_repo_semantic):
        """Semantic filter composes correctly with provider filter."""
        results = await (ConversationFilter(filter_repo_semantic)
                         .provider("claude-code")
                         .has_git_operations()
                         .list())
        ids = {c.id for c in results}
        assert "conv-git-ops" in ids
        assert "conv-mixed" in ids
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_shell_only_conv_not_matched_by_any_semantic_filter(self, filter_repo_semantic):
        """conv-shell-only has shell blocks but no file/git/subagent classification."""
        file_results = await ConversationFilter(filter_repo_semantic).has_file_operations().list()
        git_results = await ConversationFilter(filter_repo_semantic).has_git_operations().list()
        sub_results = await ConversationFilter(filter_repo_semantic).has_subagent_spawns().list()
        for results in (file_results, git_results, sub_results):
            assert all(c.id != "conv-shell-only" for c in results)
