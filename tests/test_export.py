"""Tests for polylogue.export functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.export import export_jsonl
from tests.helpers import ConversationBuilder


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Provide a database populated with test data."""
    (ConversationBuilder(db_path, "conv-1")
     .provider("chatgpt")
     .title("First Conversation")
     .add_message("msg-1-1", role="user", text="Hello")
     .add_message("msg-1-2", role="assistant", text="Hi there")
     .save())

    (ConversationBuilder(db_path, "conv-2")
     .provider("claude")
     .title("With Attachments")
     .add_message("msg-2-1", role="user", text="Here is an image")
     .add_attachment("att-1", message_id="msg-2-1", mime_type="image/png", size_bytes=1024)
     .save())

    return db_path


def test_export_jsonl_creates_file(populated_db, tmp_path):
    """export_jsonl creates the output file."""
    # export_jsonl uses open_connection(None) which uses default_db_path.
    # We must patch open_connection or ensure it uses our fixture db.
    # export_jsonl allows `open_connection(None)`?
    # Actually, export_jsonl implementation:
    # with open_connection(None) as conn:
    # It hardcodes connecting to DEFAULT DB if filtered?
    # No, open_connection(None) connects to standard path.
    # We need to test against our `db_path`.
    # But `export_jsonl` does NOT accept a db_path or connection!
    # It imports `open_connection` directly.
    # We must patch `polylogue.export.open_connection`.

    from unittest.mock import patch

    db_path = populated_db

    # We need to mock open_connection to return a connection to our test db
    with patch("polylogue.export.open_connection") as mock_conn:
        # Mock context manager behavior
        from polylogue.storage.backends.sqlite import connection_context

        mock_conn.side_effect = lambda _: connection_context(db_path)

        output_dir = tmp_path / "archive"
        output_dir.mkdir()

        output_file = export_jsonl(archive_root=output_dir)

        assert output_file.exists()
        assert output_file.name == "conversations.jsonl"
        assert output_file.parent == output_dir / "exports"


def test_export_content_correctness(populated_db, tmp_path):
    """Exported JSONL contains correct data."""
    from unittest.mock import patch

    from polylogue.storage.backends.sqlite import connection_context

    db_path = populated_db

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "out.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        # Parse and verify
        convs = [json.loads(line) for line in lines]
        convs_by_id = {c["conversation"]["conversation_id"]: c for c in convs}

        assert "conv-1" in convs_by_id
        assert "conv-2" in convs_by_id

        c1 = convs_by_id["conv-1"]
        assert c1["conversation"]["title"] == "First Conversation"
        assert len(c1["messages"]) == 2
        assert c1["messages"][0]["text"] == "Hello"
        assert len(c1["attachments"]) == 0

        c2 = convs_by_id["conv-2"]
        assert len(c2["attachments"]) == 1
        att = c2["attachments"][0]
        assert att["attachment_id"] == "att-1"
        assert att["mime_type"] == "image/png"


def test_export_handles_empty_db(db_path, tmp_path):
    """Exporting empty DB produces empty file."""
    from unittest.mock import patch

    # Ensure tables exist by opening a connection (which auto-creates schema)
    from polylogue.storage.backends.sqlite import connection_context, open_connection

    with open_connection(db_path):
        pass

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "empty.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8").strip()
        assert content == ""
