"""Tests for analytics metrics computation."""

from __future__ import annotations

from polylogue.analytics.metrics import ProviderMetrics, compute_provider_comparison
from tests.helpers import make_conversation, make_message


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

    def test_empty_database(self, workspace_env):
        """Empty database returns empty list."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        result = compute_provider_comparison(db_path=db_path)
        assert result == []

    def test_single_provider(self, workspace_env, storage_repository):
        """Single provider aggregation."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("conv-1", provider_name="claude", provider_meta={"source": "inbox"})
        msgs = [
            make_message("msg-1", "conv-1", text="Hello world test"),
            make_message("msg-2", "conv-1", role="assistant", text="Response with more words for testing average calculation"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].provider_name == "claude"
        assert result[0].conversation_count == 1
        assert result[0].message_count == 2
        assert result[0].user_message_count == 1
        assert result[0].assistant_message_count == 1

    def test_multiple_providers_sorted(self, workspace_env, storage_repository):
        """Multiple providers sorted by conversation count descending."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Create 2 claude conversations
        for i in range(2):
            conv = make_conversation(f"claude-{i}", provider_name="claude", title=f"Claude {i}", provider_meta={"source": "inbox"})
            msgs = [make_message(f"cmsg-{i}", f"claude-{i}", text="Hello")]
            storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Create 3 chatgpt conversations
        for i in range(3):
            conv = make_conversation(f"chatgpt-{i}", provider_name="chatgpt", title=f"ChatGPT {i}", provider_meta={"source": "inbox"})
            msgs = [make_message(f"gmsg-{i}", f"chatgpt-{i}", text="Hi")]
            storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 2
        # ChatGPT has more conversations, should be first
        assert result[0].provider_name == "chatgpt"
        assert result[0].conversation_count == 3
        assert result[1].provider_name == "claude"
        assert result[1].conversation_count == 2

    def test_user_assistant_segregation(self, workspace_env, storage_repository):
        """User and assistant messages are counted separately."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("conv-roles", title="Roles Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("rmsg-1", "conv-roles", text="User one"),
            make_message("rmsg-2", "conv-roles", role="assistant", text="Assistant one"),
            make_message("rmsg-3", "conv-roles", text="User two"),
            make_message("rmsg-4", "conv-roles", role="assistant", text="Assistant two three four"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].user_message_count == 2
        assert result[0].assistant_message_count == 2
        assert result[0].message_count == 4

    def test_avg_messages_per_conversation(self, workspace_env, storage_repository):
        """Average messages per conversation is computed correctly."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Conv 1: 2 messages
        conv1 = make_conversation("avg-1", title="Avg 1", provider_meta={"source": "inbox"})
        msgs1 = [
            make_message("avg-msg-1a", "avg-1", text="Hi"),
            make_message("avg-msg-1b", "avg-1", role="assistant", text="Hello"),
        ]
        storage_repository.save_conversation(conversation=conv1, messages=msgs1, attachments=[])

        # Conv 2: 4 messages
        conv2 = make_conversation("avg-2", title="Avg 2", provider_meta={"source": "inbox"})
        msgs2 = [
            make_message("avg-msg-2a", "avg-2", text="Q1"),
            make_message("avg-msg-2b", "avg-2", role="assistant", text="A1"),
            make_message("avg-msg-2c", "avg-2", text="Q2"),
            make_message("avg-msg-2d", "avg-2", role="assistant", text="A2"),
        ]
        storage_repository.save_conversation(conversation=conv2, messages=msgs2, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Total 6 messages across 2 conversations = 3.0 average
        assert result[0].avg_messages_per_conversation == 3.0

    def test_tool_use_detection(self, workspace_env, storage_repository):
        """Tool use is detected from content_blocks."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("tool-conv", provider_name="claude", title="Tool Use Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("tool-msg-1", "tool-conv", role="assistant", text="Let me search for that",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "search", "id": "toolu_123"}]}),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 1
        assert result[0].total_conversations_with_tools == 1
        assert result[0].tool_use_percentage == 100.0

    def test_thinking_detection(self, workspace_env, storage_repository):
        """Thinking is detected from content_blocks."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("think-conv", provider_name="claude", title="Thinking Test", provider_meta={"source": "inbox"})
        msgs = [
            make_message("think-msg-1", "think-conv", role="assistant", text="Let me think about this",
                        provider_meta={"content_blocks": [{"type": "thinking", "thinking": "Reasoning..."}]}),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].thinking_count == 1
        assert result[0].total_conversations_with_thinking == 1
        assert result[0].thinking_percentage == 100.0

    def test_conversations_with_tools_set_dedup(self, workspace_env, storage_repository):
        """Multiple tool uses in same conversation counted once."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        conv = make_conversation("multi-tool", provider_name="claude", title="Multi Tool", provider_meta={"source": "inbox"})
        msgs = [
            make_message("mt-msg-1", "multi-tool", role="assistant", text="Tool 1",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "a"}]}),
            make_message("mt-msg-2", "multi-tool", role="assistant", text="Tool 2",
                        provider_meta={"content_blocks": [{"type": "tool_use", "name": "b"}]}),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 2  # Two tool use messages
        assert result[0].total_conversations_with_tools == 1  # But one conversation
        assert result[0].tool_use_percentage == 100.0

    def test_division_by_zero_protection(self, workspace_env, storage_repository):
        """Metrics handle zero counts gracefully."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Conversation with system-only messages (no user/assistant)
        conv = make_conversation("zero-div", title="Zero Division", provider_meta={"source": "inbox"})
        msgs = [make_message("zero-msg-1", "zero-div", role="system", text="System message")]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Should not raise division by zero
        assert result[0].avg_user_words == 0.0
        assert result[0].avg_assistant_words == 0.0
        assert result[0].tool_use_percentage == 0.0
        assert result[0].thinking_percentage == 0.0
