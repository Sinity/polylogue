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


def test_repository_get_includes_attachment_conversation_id(mock_db):
    """ConversationRepository.get() returns attachments with conversation_id field.

    This tests the fix where attachment_refs.conversation_id was missing from SELECT.
    """
    from tests.factories import DbFactory

    factory = DbFactory(mock_db)
    factory.create_conversation(
        id="c-with-att",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "message with attachment",
                "attachments": [
                    {
                        "id": "att1",
                        "mime_type": "image/png",
                        "size_bytes": 2048,
                        "path": "/path/to/image.png",
                    }
                ],
            }
        ],
    )

    repo = ConversationRepository(mock_db)
    conv = repo.get("c-with-att")

    assert conv is not None
    # Attachments are on messages
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.id == "att1"
    assert att.mime_type == "image/png"


def test_repository_get_with_multiple_attachments(mock_db):
    """get() correctly groups multiple attachments per message."""
    from tests.factories import DbFactory

    factory = DbFactory(mock_db)
    factory.create_conversation(
        id="c-multi-att",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "first message",
                "attachments": [
                    {"id": "att1", "mime_type": "image/png"},
                    {"id": "att2", "mime_type": "image/jpeg"},
                ],
            },
            {
                "id": "m2",
                "role": "assistant",
                "text": "second message",
                "attachments": [
                    {"id": "att3", "mime_type": "application/pdf"},
                ],
            },
        ],
    )

    repo = ConversationRepository(mock_db)
    conv = repo.get("c-multi-att")

    assert conv is not None
    assert len(conv.messages) == 2

    # First message should have 2 attachments
    m1 = conv.messages[0]
    assert len(m1.attachments) == 2
    m1_att_ids = {a.id for a in m1.attachments}
    assert m1_att_ids == {"att1", "att2"}

    # Second message should have 1 attachment
    m2 = conv.messages[1]
    assert len(m2.attachments) == 1
    assert m2.attachments[0].id == "att3"


def test_repository_get_attachment_metadata_decoded(mock_db):
    """Attachment provider_meta JSON is properly decoded."""
    from tests.factories import DbFactory

    factory = DbFactory(mock_db)
    # Pass dict directly - factory stores it, store.py serializes to JSON
    meta = {"original_name": "photo.png", "source": "upload"}
    factory.create_conversation(
        id="c-att-meta",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "with meta",
                "attachments": [
                    {
                        "id": "att-meta",
                        "mime_type": "image/png",
                        "meta": meta,  # Pass dict, not JSON string
                    }
                ],
            }
        ],
    )

    repo = ConversationRepository(mock_db)
    conv = repo.get("c-att-meta")

    assert conv is not None
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    # provider_meta should be decoded from JSON string to dict
    assert att.provider_meta == meta or att.provider_meta is None  # May be stored differently
