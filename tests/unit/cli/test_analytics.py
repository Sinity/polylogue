"""Tests for analytics metrics computation."""

from __future__ import annotations

from polylogue.cli.analytics import ProviderMetrics, compute_provider_comparison
from tests.infra.helpers import make_conversation, make_message


class TestProviderMetrics:
    """Test ProviderMetrics dataclass."""

    def test_tool_use_percentage_with_data(self):
        """Tool use percentage is calculated correctly."""
        metrics = ProviderMetrics(
            provider_name="test",
            conversation_count=100,
            message_count=500,
            user_message_count=200,
            assistant_message_count=300,
            avg_messages_per_conversation=5.0,
            avg_user_words=50.0,
            avg_assistant_words=150.0,
            tool_use_count=25,
            thinking_count=10,
            total_conversations_with_tools=20,
            total_conversations_with_thinking=8,
        )
        assert metrics.tool_use_percentage == 20.0

    def test_tool_use_percentage_zero_conversations(self):
        """Tool use percentage returns 0 when no conversations."""
        metrics = ProviderMetrics(
            provider_name="empty",
            conversation_count=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            avg_messages_per_conversation=0.0,
            avg_user_words=0.0,
            avg_assistant_words=0.0,
            tool_use_count=0,
            thinking_count=0,
            total_conversations_with_tools=0,
            total_conversations_with_thinking=0,
        )
        assert metrics.tool_use_percentage == 0.0

    def test_thinking_percentage_with_data(self):
        """Thinking percentage is calculated correctly."""
        metrics = ProviderMetrics(
            provider_name="test",
            conversation_count=50,
            message_count=200,
            user_message_count=100,
            assistant_message_count=100,
            avg_messages_per_conversation=4.0,
            avg_user_words=40.0,
            avg_assistant_words=120.0,
            tool_use_count=5,
            thinking_count=15,
            total_conversations_with_tools=3,
            total_conversations_with_thinking=10,
        )
        assert metrics.thinking_percentage == 20.0

    def test_thinking_percentage_zero_conversations(self):
        """Thinking percentage returns 0 when no conversations."""
        metrics = ProviderMetrics(
            provider_name="empty",
            conversation_count=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            avg_messages_per_conversation=0.0,
            avg_user_words=0.0,
            avg_assistant_words=0.0,
            tool_use_count=0,
            thinking_count=0,
            total_conversations_with_tools=0,
            total_conversations_with_thinking=0,
        )
        assert metrics.thinking_percentage == 0.0


class TestComputeProviderComparison:
    """Test compute_provider_comparison function."""

    async def test_empty_database(self, workspace_env):
        """Empty database returns empty list."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        result = await compute_provider_comparison(db_path=db_path)
        assert result == []

    async def test_single_provider(self, workspace_env, storage_repository):
        """Single provider aggregation."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("conv-1", provider_name="claude", provider_meta={"source": "inbox"})
        msgs = [
            make_message("msg-1", "conv-1", text="Hello world test"),
            make_message("msg-2", "conv-1", role="assistant", text="Response with more words for testing average calculation"),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].provider_name == "claude"
        assert result[0].conversation_count == 1
        assert result[0].message_count == 2
        assert result[0].user_message_count == 1
        assert result[0].assistant_message_count == 1

    async def test_multiple_providers_sorted(self, workspace_env, storage_repository):
        """Multiple providers sorted by conversation count descending."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Create 2 claude conversations
        for i in range(2):
            conv = make_conversation(f"claude-{i}", provider_name="claude", title=f"Claude {i}", provider_meta={"source": "inbox"})
            msgs = [make_message(f"cmsg-{i}", f"claude-{i}", text="Hello")]
            await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Create 3 chatgpt conversations
        for i in range(3):
            conv = make_conversation(f"chatgpt-{i}", provider_name="chatgpt", title=f"ChatGPT {i}", provider_meta={"source": "inbox"})
            msgs = [make_message(f"gmsg-{i}", f"chatgpt-{i}", text="Hi")]
            await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 2
        # ChatGPT has more conversations, should be first
        assert result[0].provider_name == "chatgpt"
        assert result[0].conversation_count == 3
        assert result[1].provider_name == "claude"
        assert result[1].conversation_count == 2

    async def test_user_assistant_segregation(self, workspace_env, storage_repository):
        """User and assistant messages are counted separately."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("conv-roles", title="Roles Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("rmsg-1", "conv-roles", text="User one"),
            make_message("rmsg-2", "conv-roles", role="assistant", text="Assistant one"),
            make_message("rmsg-3", "conv-roles", text="User two"),
            make_message("rmsg-4", "conv-roles", role="assistant", text="Assistant two three four"),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].user_message_count == 2
        assert result[0].assistant_message_count == 2
        assert result[0].message_count == 4

    async def test_avg_messages_per_conversation(self, workspace_env, storage_repository):
        """Average messages per conversation is computed correctly."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Conv 1: 2 messages
        conv1 = make_conversation("avg-1", title="Avg 1", provider_meta={"source": "inbox"})
        msgs1 = [
            make_message("avg-msg-1a", "avg-1", text="Hi"),
            make_message("avg-msg-1b", "avg-1", role="assistant", text="Hello"),
        ]
        await storage_repository.save_conversation(conversation=conv1, messages=msgs1, attachments=[])

        # Conv 2: 4 messages
        conv2 = make_conversation("avg-2", title="Avg 2", provider_meta={"source": "inbox"})
        msgs2 = [
            make_message("avg-msg-2a", "avg-2", text="Q1"),
            make_message("avg-msg-2b", "avg-2", role="assistant", text="A1"),
            make_message("avg-msg-2c", "avg-2", text="Q2"),
            make_message("avg-msg-2d", "avg-2", role="assistant", text="A2"),
        ]
        await storage_repository.save_conversation(conversation=conv2, messages=msgs2, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Total 6 messages across 2 conversations = 3.0 average
        assert result[0].avg_messages_per_conversation == 3.0

    async def test_tool_use_detection(self, workspace_env, storage_repository):
        """Tool use is detected from content_blocks."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("tool-conv", provider_name="claude", title="Tool Use Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("tool-msg-1", "tool-conv", role="assistant", text="Let me search for that",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "search", "id": "toolu_123"}]}),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 1
        assert result[0].total_conversations_with_tools == 1
        assert result[0].tool_use_percentage == 100.0

    async def test_thinking_detection(self, workspace_env, storage_repository):
        """Thinking is detected from content_blocks."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("think-conv", provider_name="claude", title="Thinking Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("think-msg-1", "think-conv", role="assistant", text="Let me think about this",
                        provider_meta={"content_blocks": [{"type": "thinking", "thinking": "Reasoning..."}]}),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].thinking_count == 1
        assert result[0].total_conversations_with_thinking == 1
        assert result[0].thinking_percentage == 100.0

    async def test_conversations_with_tools_set_dedup(self, workspace_env, storage_repository):
        """Multiple tool uses in same conversation counted once."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("multi-tool", provider_name="claude", title="Multi Tool", provider_meta={"source": "inbox"})
        msgs = [
            make_message("mt-msg-1", "multi-tool", role="assistant", text="Tool 1",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "a"}]}),
            make_message("mt-msg-2", "multi-tool", role="assistant", text="Tool 2",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "b"}]}),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 2  # Two tool use messages
        assert result[0].total_conversations_with_tools == 1  # But one conversation
        assert result[0].tool_use_percentage == 100.0

    async def test_division_by_zero_protection(self, workspace_env, storage_repository):
        """Metrics handle zero counts gracefully."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Conversation with system-only messages (no user/assistant)
        conv = make_conversation("zero-div", title="Zero Division", provider_meta={"source": "inbox"})
        msgs = [make_message("zero-msg-1", "zero-div", role="system", text="System message")]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = await compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Should not raise division by zero
        assert result[0].avg_user_words == 0.0
        assert result[0].avg_assistant_words == 0.0
        assert result[0].tool_use_percentage == 0.0
        assert result[0].thinking_percentage == 0.0


# ============================================================================
# _seed_db helper: create test DB with custom row data
# ============================================================================


async def _seed_db(tmp_path, rows):
    """Seed database with raw rows: (provider, role, text, provider_meta_or_None).

    Returns: db_path (Path)

    Args:
        tmp_path: pytest tmp_path fixture
        rows: list of tuples (provider, role, text, provider_meta_dict_or_None)

    Creates conversations and messages from rows, returns db_path.
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)

    # Group rows by provider and conversation
    convos_by_provider = {}
    for provider, role, text, provider_meta in rows:
        if provider not in convos_by_provider:
            convos_by_provider[provider] = []
        convos_by_provider[provider].append((role, text, provider_meta))

    # Create conversations and save
    msg_counter = 0
    for provider, messages in convos_by_provider.items():
        conv = make_conversation(
            f"conv-{provider}",
            provider_name=provider,
            title=f"{provider} Test Conversation",
            provider_meta={"source": "test"}
        )
        msgs = []
        for role, text, provider_meta in messages:
            msg_counter += 1
            msg = make_message(
                f"msg-{msg_counter}",
                conv.conversation_id,
                role=role,
                text=text,
                provider_meta=provider_meta,
            )
            msgs.append(msg)

        await repo.save_conversation(conversation=conv, messages=msgs, attachments=[])

    await backend.close()
    return db_path


# ============================================================================
# Word count SQL edge cases
# ============================================================================


class TestWordCountEdgeCases:
    """Verify word count SQL handles edge cases correctly."""

    async def test_spaces_only_text_counts_zero_words(self, tmp_path):
        """Space-only messages should count as 0 words."""
        db = await _seed_db(tmp_path, [
            ("test", "user", "   ", None),
            ("test", "user", "     ", None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert len(results) == 1
        assert results[0].avg_user_words == 0.0

    async def test_tabs_newlines_are_not_stripped(self, tmp_path):
        """SQLite TRIM only strips spaces — tabs/newlines count as 1 word.

        This documents a known approximation: SQLite's TRIM() only removes
        ASCII space (0x20), not tabs (0x09) or newlines (0x0A). In practice
        this doesn't matter because real messages never contain only whitespace.
        """
        db = await _seed_db(tmp_path, [
            ("test", "user", "\t\t", None),
        ])
        results = await compute_provider_comparison(db_path=db)
        # Tab-only text passes TRIM check (TRIM doesn't strip tabs)
        # and the formula counts it as 1 "word"
        assert results[0].avg_user_words == 1.0

    async def test_single_word_counts_one(self, tmp_path):
        """A single word with no spaces counts as 1."""
        db = await _seed_db(tmp_path, [
            ("test", "user", "Hello", None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].avg_user_words == 1.0

    async def test_multiple_spaces_between_words(self, tmp_path):
        """Multiple spaces between words still count correctly.

        The SQL formula counts spaces, so 'a  b' has 2 spaces = 3 words.
        This is a known approximation - documenting the actual behavior.
        """
        db = await _seed_db(tmp_path, [
            ("test", "user", "hello  world", None),  # 2 spaces
        ])
        results = await compute_provider_comparison(db_path=db)
        # With double space: LENGTH("hello  world")=12, REPLACE removes spaces: "helloworld"=10
        # 12 - 10 + 1 = 3 (overcounts by 1 due to double space)
        # This documents the known approximation behavior
        assert results[0].avg_user_words == 3.0

    async def test_empty_text_counts_zero(self, tmp_path):
        """Empty string text counts as 0 words."""
        db = await _seed_db(tmp_path, [
            ("test", "user", "", None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].avg_user_words == 0.0

    async def test_none_text_counts_zero(self, tmp_path):
        """NULL text counts as 0 words."""
        db = await _seed_db(tmp_path, [
            ("test", "user", None, None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].avg_user_words == 0.0


# ============================================================================
# LIKE pattern resistance tests
# ============================================================================


class TestLikePatternResistance:
    """Verify LIKE-based tool_use/thinking detection doesn't false-positive."""

    async def test_tool_use_in_message_text_not_detected(self, tmp_path):
        """Text containing 'tool_use' string should NOT count as tool use."""
        db = await _seed_db(tmp_path, [
            ("test", "assistant", 'The tool_use feature is great', None),
            ("test", "assistant", 'I used "type":"tool_use" in my message', None),
        ])
        results = await compute_provider_comparison(db_path=db)
        # tool_use_count should be 0 — the LIKE is on provider_meta, not text
        assert results[0].tool_use_count == 0

    async def test_thinking_in_message_text_not_detected(self, tmp_path):
        """Text containing 'thinking' should NOT count as thinking."""
        db = await _seed_db(tmp_path, [
            ("test", "assistant", 'I was thinking about "type":"thinking" blocks', None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].thinking_count == 0

    async def test_tool_role_fallback_detected(self, tmp_path):
        """Messages with role='tool' should be counted as tool use."""
        db = await _seed_db(tmp_path, [
            ("test", "tool", "Tool result here", None),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].tool_use_count == 1

    async def test_tool_use_in_provider_meta_detected(self, tmp_path):
        """Tool use in provider_meta content_blocks is detected."""
        meta = {"content_blocks": [{"type": "tool_use", "name": "search", "id": "t1"}]}
        db = await _seed_db(tmp_path, [
            ("test", "assistant", "Using a tool", meta),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].tool_use_count == 1
        assert results[0].total_conversations_with_tools == 1

    async def test_thinking_in_provider_meta_detected(self, tmp_path):
        """Thinking blocks in provider_meta are detected."""
        meta = {"content_blocks": [{"type": "thinking", "thinking": "Let me consider..."}]}
        db = await _seed_db(tmp_path, [
            ("test", "assistant", "Here's my answer", meta),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].thinking_count == 1
        assert results[0].total_conversations_with_thinking == 1

    async def test_mixed_content_blocks_counted_correctly(self, tmp_path):
        """Message with both tool_use and thinking blocks counts both."""
        meta = {"content_blocks": [
            {"type": "thinking", "thinking": "Planning..."},
            {"type": "tool_use", "name": "search", "id": "t1"},
            {"type": "text", "text": "Result"},
        ]}
        db = await _seed_db(tmp_path, [
            ("test", "assistant", "Result from tool", meta),
        ])
        results = await compute_provider_comparison(db_path=db)
        assert results[0].tool_use_count == 1
        assert results[0].thinking_count == 1


# ============================================================================
# Cross-provider integration test
# ============================================================================


class TestCrossProviderConsistency:
    """Verify SQL detection works across different provider data structures."""

    async def test_multiple_providers_with_tool_use(self, tmp_path):
        """Tool use is detected correctly across ChatGPT and Claude providers."""
        chatgpt_meta = {"content_blocks": [{"type": "tool_use", "name": "browser"}]}
        claude_meta = {"content_blocks": [{"type": "tool_use", "name": "computer", "id": "toolu_1"}]}

        db = await _seed_db(tmp_path, [
            ("chatgpt", "assistant", "ChatGPT used a tool", chatgpt_meta),
            ("chatgpt", "user", "Thanks", None),
            ("claude-ai", "assistant", "Claude used a tool", claude_meta),
            ("claude-ai", "user", "Thanks", None),
        ])
        results = await compute_provider_comparison(db_path=db)

        by_provider = {r.provider_name: r for r in results}
        assert by_provider["chatgpt"].tool_use_count == 1
        assert by_provider["claude-ai"].tool_use_count == 1
