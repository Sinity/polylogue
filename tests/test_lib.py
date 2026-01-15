import pytest

from polylogue.db import open_connection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.repository import ConversationRepository


@pytest.fixture
def mock_db(tmp_path):
    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass
    return db_path


def test_semantic_models():
    # Test rich methods
    msg_user = Message(id="1", role="user", text="hello, how are you today?")
    msg_bot = Message(id="2", role="assistant", text="I'm doing well, thanks for asking!")
    conv = Conversation(id="c1", provider="test", messages=[msg_user, msg_bot])

    # Test filtering
    user_only = conv.user_only()
    assert len(user_only.messages) == 1
    assert user_only.messages[0].id == "1"

    # Test to_text
    txt = conv.to_text()
    assert "user:" in txt
    assert "assistant:" in txt
    assert "hello" in txt

    # Test new classification properties
    assert msg_user.is_user
    assert msg_bot.is_assistant
    assert msg_user.is_substantive
    assert not msg_user.is_tool_use
    assert not msg_user.is_noise

    # Test projections
    clean = conv.without_noise()
    assert len(clean.messages) == 2  # Both are substantive

    # Test statistics
    assert conv.message_count == 2
    assert conv.user_message_count == 1


def test_repository(mock_db):
    repo = ConversationRepository(mock_db)

    # Seed data
    from tests.factories import DbFactory

    factory = DbFactory(mock_db)
    factory.create_conversation(
        id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
    )

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
