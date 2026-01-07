from datetime import datetime
import sqlite3
import pytest
from polylogue.lib.models import Conversation, Message, Attachment
from polylogue.lib.repository import ConversationRepository
from polylogue.store import ConversationRecord, MessageRecord, AttachmentRecord
from polylogue.db import open_connection


@pytest.fixture
def mock_db(tmp_path):
    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass
    return db_path


def test_semantic_models():
    # Test rich methods
    msg_user = Message(id="1", role="user", text="hello")
    msg_bot = Message(id="2", role="assistant", text="hi")
    conv = Conversation(id="c1", provider="test", messages=[msg_user, msg_bot])

    # Test filtering
    user_only = conv.user_only()
    assert len(user_only.messages) == 1
    assert user_only.messages[0].id == "1"

    # Test text_only
    txt = conv.text_only()
    assert "user: hello" in txt
    assert "assistant: hi" in txt


def test_repository(mock_db):
    repo = ConversationRepository(mock_db)

    # Seed data manually via SQL to avoid circular dependency on Store/Ingest for this unit test
    # (Or helper if available)
    with sqlite3.connect(mock_db) as conn:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, content_hash, version) 
            VALUES ('c1', 'chatgpt', 'ext-1', 'hash', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) 
            VALUES ('m1', 'c1', 'user', 'hello world', 'hash1', 1)
            """
        )
        conn.commit()

    # Test get
    conv = repo.get("c1")
    assert conv is not None
    assert conv.id == "c1"
    assert len(conv.messages) == 1
    assert conv.messages[0].text == "hello world"

    # Test list
    lst = repo.list()
    assert len(lst) == 1
    assert lst[0].id == "c1"
