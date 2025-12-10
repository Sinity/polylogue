"""Unit tests for DatabaseRenderer."""
import sqlite3
from pathlib import Path

import pytest

from polylogue.renderers.db_renderer import DatabaseRenderer, ConversationData, MessageData


def test_database_renderer_init():
    """Test DatabaseRenderer initialization."""
    renderer = DatabaseRenderer()
    assert renderer.db_path is None

    renderer = DatabaseRenderer(db_path=Path("/tmp/test.db"))
    assert renderer.db_path == Path("/tmp/test.db")


def test_conversation_data_dataclass():
    """Test ConversationData dataclass."""
    data = ConversationData(
        provider="chatgpt",
        conversation_id="123",
        slug="test-slug",
        title="Test Title",
        current_branch="main",
        root_message_id="msg1",
        last_updated="2025-01-01",
        content_hash="abc123",
        metadata={"key": "value"},
    )

    assert data.provider == "chatgpt"
    assert data.conversation_id == "123"
    assert data.slug == "test-slug"
    assert data.title == "Test Title"
    assert data.metadata == {"key": "value"}


def test_message_data_dataclass():
    """Test MessageData dataclass."""
    data = MessageData(
        message_id="msg1",
        parent_id=None,
        position=0,
        timestamp="2025-01-01T00:00:00Z",
        role="user",
        content_hash="abc",
        rendered_text="Hello world",
        raw_json='{"text": "Hello world"}',
        token_count=2,
        word_count=2,
        attachment_count=0,
        attachment_names=None,
        metadata=None,
    )

    assert data.message_id == "msg1"
    assert data.role == "user"
    assert data.rendered_text == "Hello world"
    assert data.token_count == 2


def test_query_conversations_empty(tmp_path):
    """Test query_conversations with empty database."""
    db_path = tmp_path / "test.db"

    # Create empty database with schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            root_message_id TEXT,
            last_updated TEXT,
            content_hash TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        )
    """)
    conn.commit()
    conn.close()

    renderer = DatabaseRenderer(db_path=db_path)
    conversations = renderer.query_conversations()

    assert conversations == []


def test_query_conversations_with_data(tmp_path):
    """Test query_conversations with data."""
    db_path = tmp_path / "test.db"

    # Create database with test data
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            root_message_id TEXT,
            last_updated TEXT,
            content_hash TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        )
    """)
    conn.execute("""
        INSERT INTO conversations VALUES (
            'chatgpt', 'conv1', 'test-conv', 'Test Conversation',
            'main', 'msg1', '2025-01-01T00:00:00Z', 'hash123', NULL
        )
    """)
    conn.commit()
    conn.close()

    renderer = DatabaseRenderer(db_path=db_path)
    conversations = renderer.query_conversations()

    assert len(conversations) == 1
    assert conversations[0].provider == "chatgpt"
    assert conversations[0].conversation_id == "conv1"
    assert conversations[0].title == "Test Conversation"


def test_query_conversations_with_provider_filter(tmp_path):
    """Test query_conversations with provider filter."""
    db_path = tmp_path / "test.db"

    # Create database with multiple providers
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            root_message_id TEXT,
            last_updated TEXT,
            content_hash TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        )
    """)
    conn.execute("""
        INSERT INTO conversations VALUES
        ('chatgpt', 'conv1', 'test-conv1', 'Chat 1', 'main', 'msg1', '2025-01-01', 'hash1', NULL),
        ('claude', 'conv2', 'test-conv2', 'Claude 1', 'main', 'msg2', '2025-01-02', 'hash2', NULL)
    """)
    conn.commit()
    conn.close()

    renderer = DatabaseRenderer(db_path=db_path)

    # Query all
    all_convs = renderer.query_conversations()
    assert len(all_convs) == 2

    # Query chatgpt only
    chatgpt_convs = renderer.query_conversations(provider="chatgpt")
    assert len(chatgpt_convs) == 1
    assert chatgpt_convs[0].provider == "chatgpt"

    # Query claude only
    claude_convs = renderer.query_conversations(provider="claude")
    assert len(claude_convs) == 1
    assert claude_convs[0].provider == "claude"
