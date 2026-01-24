"""Tests for analytics metrics computation."""

from __future__ import annotations

import pytest

from polylogue.analytics.metrics import ProviderMetrics, compute_provider_comparison
from polylogue.storage.store import ConversationRecord, MessageRecord


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
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"
        result = compute_provider_comparison(db_path=db_path)
        assert result == []

    def test_single_provider(self, workspace_env, storage_repository):
        """Single provider aggregation."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        # Create a conversation
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="1000",
            updated_at="1000",
            content_hash="hash1",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                provider_message_id="pmsg-1",
                role="user",
                text="Hello world test",
                timestamp="1000",
                content_hash="mhash1",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                provider_message_id="pmsg-2",
                role="assistant",
                text="Response with more words for testing average calculation",
                timestamp="1001",
                content_hash="mhash2",
                provider_meta=None,
            ),
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
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        # Create 2 claude conversations
        for i in range(2):
            conv = ConversationRecord(
                conversation_id=f"claude-{i}",
                provider_name="claude",
                provider_conversation_id=f"pc-{i}",
                title=f"Claude {i}",
                created_at="1000",
                updated_at="1000",
                content_hash=f"chash-{i}",
                provider_meta={"source": "inbox"},
            )
            msgs = [
                MessageRecord(
                    message_id=f"cmsg-{i}",
                    conversation_id=f"claude-{i}",
                    provider_message_id=f"cpmsg-{i}",
                    role="user",
                    text="Hello",
                    timestamp="1000",
                    content_hash=f"cmhash-{i}",
                    provider_meta=None,
                ),
            ]
            storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Create 3 chatgpt conversations
        for i in range(3):
            conv = ConversationRecord(
                conversation_id=f"chatgpt-{i}",
                provider_name="chatgpt",
                provider_conversation_id=f"gp-{i}",
                title=f"ChatGPT {i}",
                created_at="1000",
                updated_at="1000",
                content_hash=f"ghash-{i}",
                provider_meta={"source": "inbox"},
            )
            msgs = [
                MessageRecord(
                    message_id=f"gmsg-{i}",
                    conversation_id=f"chatgpt-{i}",
                    provider_message_id=f"gpmsg-{i}",
                    role="user",
                    text="Hi",
                    timestamp="1000",
                    content_hash=f"gmhash-{i}",
                    provider_meta=None,
                ),
            ]
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
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="conv-roles",
            provider_name="test",
            provider_conversation_id="prov-roles",
            title="Roles Test",
            created_at="1000",
            updated_at="1000",
            content_hash="rolehash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="rmsg-1",
                conversation_id="conv-roles",
                provider_message_id="rpmsg-1",
                role="user",
                text="User one",
                timestamp="1000",
                content_hash="rmhash1",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="rmsg-2",
                conversation_id="conv-roles",
                provider_message_id="rpmsg-2",
                role="assistant",
                text="Assistant one",
                timestamp="1001",
                content_hash="rmhash2",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="rmsg-3",
                conversation_id="conv-roles",
                provider_message_id="rpmsg-3",
                role="user",
                text="User two",
                timestamp="1002",
                content_hash="rmhash3",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="rmsg-4",
                conversation_id="conv-roles",
                provider_message_id="rpmsg-4",
                role="assistant",
                text="Assistant two three four",
                timestamp="1003",
                content_hash="rmhash4",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].user_message_count == 2
        assert result[0].assistant_message_count == 2
        assert result[0].message_count == 4

    def test_avg_messages_per_conversation(self, workspace_env, storage_repository):
        """Average messages per conversation is computed correctly."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        # Conv 1: 2 messages
        conv1 = ConversationRecord(
            conversation_id="avg-1",
            provider_name="test",
            provider_conversation_id="pavg-1",
            title="Avg 1",
            created_at="1000",
            updated_at="1000",
            content_hash="avghash1",
            provider_meta={"source": "inbox"},
        )
        msgs1 = [
            MessageRecord(
                message_id="avg-msg-1a",
                conversation_id="avg-1",
                provider_message_id="avgpmsg-1a",
                role="user",
                text="Hi",
                timestamp="1000",
                content_hash="avgmhash1a",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="avg-msg-1b",
                conversation_id="avg-1",
                provider_message_id="avgpmsg-1b",
                role="assistant",
                text="Hello",
                timestamp="1001",
                content_hash="avgmhash1b",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv1, messages=msgs1, attachments=[])

        # Conv 2: 4 messages
        conv2 = ConversationRecord(
            conversation_id="avg-2",
            provider_name="test",
            provider_conversation_id="pavg-2",
            title="Avg 2",
            created_at="1000",
            updated_at="1000",
            content_hash="avghash2",
            provider_meta={"source": "inbox"},
        )
        msgs2 = [
            MessageRecord(
                message_id="avg-msg-2a",
                conversation_id="avg-2",
                provider_message_id="avgpmsg-2a",
                role="user",
                text="Q1",
                timestamp="1000",
                content_hash="avgmhash2a",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="avg-msg-2b",
                conversation_id="avg-2",
                provider_message_id="avgpmsg-2b",
                role="assistant",
                text="A1",
                timestamp="1001",
                content_hash="avgmhash2b",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="avg-msg-2c",
                conversation_id="avg-2",
                provider_message_id="avgpmsg-2c",
                role="user",
                text="Q2",
                timestamp="1002",
                content_hash="avgmhash2c",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="avg-msg-2d",
                conversation_id="avg-2",
                provider_message_id="avgpmsg-2d",
                role="assistant",
                text="A2",
                timestamp="1003",
                content_hash="avgmhash2d",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv2, messages=msgs2, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Total 6 messages across 2 conversations = 3.0 average
        assert result[0].avg_messages_per_conversation == 3.0

    def test_tool_use_detection(self, workspace_env, storage_repository):
        """Tool use is detected from content_blocks."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="tool-conv",
            provider_name="claude",
            provider_conversation_id="ptool",
            title="Tool Use Test",
            created_at="1000",
            updated_at="1000",
            content_hash="toolhash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="tool-msg-1",
                conversation_id="tool-conv",
                provider_message_id="tpmsg-1",
                role="assistant",
                text="Let me search for that",
                timestamp="1000",
                content_hash="toolmhash1",
                provider_meta={
                    "content_blocks": [
                        {"type": "tool_use", "name": "search", "id": "toolu_123"}
                    ]
                },
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 1
        assert result[0].total_conversations_with_tools == 1
        assert result[0].tool_use_percentage == 100.0

    def test_thinking_detection(self, workspace_env, storage_repository):
        """Thinking is detected from content_blocks."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="think-conv",
            provider_name="claude",
            provider_conversation_id="pthink",
            title="Thinking Test",
            created_at="1000",
            updated_at="1000",
            content_hash="thinkhash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="think-msg-1",
                conversation_id="think-conv",
                provider_message_id="thpmsg-1",
                role="assistant",
                text="Let me think about this",
                timestamp="1000",
                content_hash="thinkmhash1",
                provider_meta={
                    "content_blocks": [
                        {"type": "thinking", "thinking": "Reasoning..."}
                    ]
                },
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].thinking_count == 1
        assert result[0].total_conversations_with_thinking == 1
        assert result[0].thinking_percentage == 100.0

    def test_conversations_with_tools_set_dedup(self, workspace_env, storage_repository):
        """Multiple tool uses in same conversation counted once."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="multi-tool",
            provider_name="claude",
            provider_conversation_id="pmtool",
            title="Multi Tool",
            created_at="1000",
            updated_at="1000",
            content_hash="mtoolhash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="mt-msg-1",
                conversation_id="multi-tool",
                provider_message_id="mtpmsg-1",
                role="assistant",
                text="Tool 1",
                timestamp="1000",
                content_hash="mtmhash1",
                provider_meta={"content_blocks": [{"type": "tool_use", "name": "a"}]},
            ),
            MessageRecord(
                message_id="mt-msg-2",
                conversation_id="multi-tool",
                provider_message_id="mtpmsg-2",
                role="assistant",
                text="Tool 2",
                timestamp="1001",
                content_hash="mtmhash2",
                provider_meta={"content_blocks": [{"type": "tool_use", "name": "b"}]},
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        assert result[0].tool_use_count == 2  # Two tool use messages
        assert result[0].total_conversations_with_tools == 1  # But one conversation
        assert result[0].tool_use_percentage == 100.0

    def test_division_by_zero_protection(self, workspace_env, storage_repository):
        """Metrics handle zero counts gracefully."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        # Conversation with system-only messages (no user/assistant)
        conv = ConversationRecord(
            conversation_id="zero-div",
            provider_name="test",
            provider_conversation_id="pzero",
            title="Zero Division",
            created_at="1000",
            updated_at="1000",
            content_hash="zerohash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="zero-msg-1",
                conversation_id="zero-div",
                provider_message_id="zpmsg-1",
                role="system",
                text="System message",
                timestamp="1000",
                content_hash="zeromhash1",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        result = compute_provider_comparison(db_path=db_path)

        assert len(result) == 1
        # Should not raise division by zero
        assert result[0].avg_user_words == 0.0
        assert result[0].avg_assistant_words == 0.0
        assert result[0].tool_use_percentage == 0.0
        assert result[0].thinking_percentage == 0.0
