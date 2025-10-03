import json
from pathlib import Path

from chatmd.importers.chatgpt import import_chatgpt_export
from chatmd.importers.claude_ai import import_claude_export
from chatmd.importers.claude_code import import_claude_code_session


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_import_chatgpt_export_basic(tmp_path):
    export_dir = tmp_path / "chatgpt"
    export_dir.mkdir()
    conversations = [
        {
            "id": "conv-test",
            "title": "Sample Chat",
            "create_time": 1,
            "update_time": 2,
            "mapping": {
                "node-user": {
                    "id": "node-user",
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                    },
                },
                "node-assistant": {
                    "id": "node-assistant",
                    "message": {
                        "id": "msg-assistant",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi there"]},
                    },
                },
            },
        }
    ]
    _write(export_dir / "conversations.json", conversations)

    out_dir = tmp_path / "out"
    results = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    assert len(results) == 1
    result = results[0]
    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "Sample Chat" in text
    assert "Hello" in text
    assert "Hi there" in text


def test_import_claude_export_basic(tmp_path):
    export_dir = tmp_path / "claude"
    export_dir.mkdir()
    conversations = {
        "conversations": [
            {
                "uuid": "claude-1",
                "name": "Claude Chat",
                "chat_messages": [
                    {"sender": "user", "content": [{"type": "text", "text": "Hi"}]},
                    {"sender": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                ],
            }
        ]
    }
    _write(export_dir / "conversations.json", conversations)

    out_dir = tmp_path / "out"
    results = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    assert len(results) == 1
    result = results[0]
    assert result.markdown_path.exists()
    body = result.markdown_path.read_text(encoding="utf-8")
    assert "Claude Chat" in body
    assert "Hi" in body
    assert "Hello" in body


def test_import_claude_code_session_basic(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / "session-abc.jsonl"
    lines = [
        json.dumps({"type": "summary", "summary": "Initial summary"}),
        json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Run command"}]},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Command output"}]},
            }
        ),
    ]
    session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    result = import_claude_code_session(
        session_path.name,
        base_dir=sessions_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "Run command" in text
    assert "Command output" in text
    assert "Initial summary" in text
