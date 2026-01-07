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
