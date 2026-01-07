import pytest
import sqlite3
from polylogue.db import open_connection
from polylogue.lib.repository import ConversationRepository
from polylogue.server.deps import get_repository

pytest.importorskip("fastapi")
from polylogue.server.app import app
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(tmp_path):
    # Override dependency to use a temp DB
    db_path = tmp_path / "server_test.db"

    # Init DB
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, content_hash, version) 
            VALUES ('c1', 'chatgpt', 'ext-1', 'hash', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) 
            VALUES ('m1', 'c1', 'user', 'hello server', 'hash1', 1)
            """
        )

    def override_repo():
        return ConversationRepository(db_path)

    app.dependency_overrides[get_repository] = override_repo
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_api_conversations(test_client):
    response = test_client.get("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "c1"


def test_api_get_conversation(test_client):
    response = test_client.get("/api/conversations/c1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "c1"
    assert data["messages"][0]["text"] == "hello server"


def test_web_index(test_client):
    # This might fail if template directory is not correctly located relative to installed package
    # But since we run from source, it should find polylogue/templates
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Polylogue" in response.text  # Assuming title or header has it


def test_web_view(test_client):
    response = test_client.get("/view/c1")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "hello server" in response.text
