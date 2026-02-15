"""Tests for export functionality.

Extracted from test_core.py: TestExportJsonl.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from polylogue.export import export_jsonl
from polylogue.storage.backends.sqlite import connection_context, open_connection


class TestExportJsonl:
    """Tests for export_jsonl() function."""

    def test_export_jsonl_creates_output_file(self, workspace_env):
        """Creates output file in correct location."""
        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        assert result_path.suffix == ".jsonl"

    def test_export_jsonl_custom_output_path(self, workspace_env):
        """Uses custom output path when provided."""
        archive_root = workspace_env["archive_root"]
        custom_output = workspace_env["data_root"] / "custom_export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=custom_output)

        assert result_path == custom_output
        assert custom_output.exists()

    def test_export_jsonl_empty_database(self, workspace_env):
        """Handles empty database gracefully."""
        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        content = result_path.read_text()
        assert content == ""

    def test_export_jsonl_with_conversation(self, workspace_env, db_path):
        """Exports conversation with messages and attachments."""
        from tests.helpers import ConversationBuilder

        (
            ConversationBuilder(db_path, "test-conv")
            .title("Export Test")
            .provider("chatgpt")
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi there!")
            .save()
        )

        archive_root = workspace_env["archive_root"]
        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        content = result_path.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 1

        export_data = json.loads(lines[0])
        assert "conversation" in export_data
        assert "messages" in export_data
        assert export_data["conversation"]["title"] == "Export Test"
        assert len(export_data["messages"]) == 2

    def test_export_jsonl_creates_parent_dirs(self, workspace_env):
        """Creates parent directories for output path."""
        archive_root = workspace_env["archive_root"]
        nested_output = workspace_env["data_root"] / "deeply" / "nested" / "export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=nested_output)

        assert result_path.exists()
        assert result_path.parent.exists()


# ---------------------------------------------------------------------------
# Standalone export integration tests (use populated_db fixture)
# ---------------------------------------------------------------------------


def test_export_jsonl_creates_file(populated_db, tmp_path):
    """export_jsonl creates the output file."""
    db_path = populated_db

    with patch("polylogue.export.open_connection") as mock_conn:
        mock_conn.side_effect = lambda _: connection_context(db_path)

        output_dir = tmp_path / "archive"
        output_dir.mkdir()

        output_file = export_jsonl(archive_root=output_dir)

        assert output_file.exists()
        assert output_file.name == "conversations.jsonl"
        assert output_file.parent == output_dir / "exports"


def test_export_content_correctness(populated_db, tmp_path):
    """Exported JSONL contains correct data."""
    db_path = populated_db

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "out.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

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
    with open_connection(db_path):
        pass

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "empty.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8").strip()
        assert content == ""
