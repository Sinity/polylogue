from pathlib import Path
import json

import pytest

from polylogue.importers.chatgpt import list_chatgpt_conversations, import_chatgpt_export
from polylogue.importers.claude_ai import list_claude_conversations, import_claude_export
from polylogue.services.conversation_registrar import create_default_registrar


def _write_chatgpt_export(tmp_path: Path, count: int = 2000) -> Path:
    data = [
        {
            "id": f"chat-{i}",
            "conversation_id": f"chat-{i}",
            "title": f"Chat {i}",
            "update_time": "2024-01-01T00:00:00Z",
            "create_time": "2023-12-31T00:00:00Z",
            "mapping": {},
        }
        for i in range(count)
    ]
    export_dir = tmp_path / "chatgpt"
    export_dir.mkdir()
    (export_dir / "conversations.json").write_text(json.dumps(data), encoding="utf-8")
    return export_dir


def _write_claude_export(tmp_path: Path, count: int = 2000) -> Path:
    data = {
        "conversations": [
            {
                "uuid": f"claude-{i}",
                "id": f"claude-{i}",
                "name": f"Claude {i}",
                "updated_at": "2024-01-01T00:00:00Z",
                "created_at": "2023-12-31T00:00:00Z",
                "chat_messages": [],
            }
            for i in range(count)
        ]
    }
    export_dir = tmp_path / "claude"
    export_dir.mkdir()
    (export_dir / "conversations.json").write_text(json.dumps(data), encoding="utf-8")
    return export_dir


def test_list_chatgpt_streams_without_loading_all(tmp_path: Path):
    export_dir = _write_chatgpt_export(tmp_path, count=500)
    conversations = list_chatgpt_conversations(export_dir)
    assert len(conversations) == 500


def test_list_claude_streams_without_loading_all(tmp_path: Path):
    export_dir = _write_claude_export(tmp_path, count=500)
    conversations = list_claude_conversations(export_dir)
    assert len(conversations) == 500


def test_import_chatgpt_streaming_handles_many(tmp_path: Path):
    export_dir = _write_chatgpt_export(tmp_path, count=50)
    out_dir = tmp_path / "out"
    registrar = create_default_registrar()
    results = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=25,
        html=False,
        html_theme="light",
        force=False,
        allow_dirty=False,
        registrar=registrar,
    )
    assert len(results) == 50


def test_import_claude_streaming_handles_many(tmp_path: Path):
    export_dir = _write_claude_export(tmp_path, count=50)
    out_dir = tmp_path / "out"
    registrar = create_default_registrar()
    results = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=25,
        html=False,
        html_theme="light",
        force=False,
        allow_dirty=False,
        registrar=registrar,
    )
    assert len(results) == 50
