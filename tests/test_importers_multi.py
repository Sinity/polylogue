import json
import zipfile
from pathlib import Path

import pytest

from polylogue.importers.chatgpt import import_chatgpt_export
from polylogue.importers.claude_ai import import_claude_export
from polylogue.importers.claude_code import import_claude_code_session


@pytest.fixture(autouse=True)
def _polylogue_state_home(tmp_path, monkeypatch):
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    return state_home


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


def test_import_chatgpt_tool_call_pairing(tmp_path):
    export_dir = tmp_path / "chatgpt_tools"
    export_dir.mkdir()
    conversations = [
        {
            "id": "conv-tool",
            "title": "Tool Chat",
            "mapping": {
                "node-user": {
                    "id": "node-user",
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["run tool"]},
                    },
                },
                "node-tool": {
                    "id": "node-tool",
                    "message": {
                        "id": "msg-tool",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "tool", "parts": []},
                        "metadata": {},
                    },
                },
            },
        }
    ]
    conversations[0]["mapping"]["node-tool"]["message"]["content"] = {
        "content_type": "text",
        "parts": [
            {
                "type": "tool_calls",
                "id": "tool-1",
                "name": "bash",
                "input": "{\"cmd\": \"echo hi\"}"
            }
        ],
    }
    conversations[0]["mapping"]["node-toolout"] = {
        "id": "node-toolout",
        "message": {
            "id": "msg-toolout",
            "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": ["Tool result output"]},
        },
    }
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")
    out_dir = tmp_path / "tool_out"
    results = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    body = results[0].markdown_path.read_text(encoding="utf-8")
    assert "Tool call" in body
    assert "Tool result" in body


def test_import_chatgpt_export_zip_with_attachment(tmp_path):
    export_dir = tmp_path / "chatgpt_zip"
    files_dir = export_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "log.txt").write_text("attachment payload", encoding="utf-8")
    conversations = [
        {
            "id": "conv-zip",
            "title": "Zip Chat",
            "mapping": {
                "node-user": {
                    "id": "node-user",
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["User line"]},
                    },
                },
                "node-assistant": {
                    "id": "node-assistant",
                    "message": {
                        "id": "msg-assistant",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Assistant line"]},
                        "metadata": {
                            "attachments": [
                                {
                                    "id": "log.txt",
                                    "name": "log.txt",
                                }
                            ]
                        },
                    },
                },
            },
        }
    ]
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")

    zip_path = tmp_path / "chatgpt_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in export_dir.rglob("*"):
            arcname = file.relative_to(export_dir)
            if file.is_dir():
                continue
            zf.write(file, arcname.as_posix())

    out_dir = tmp_path / "out_zip"
    results = import_chatgpt_export(
        zip_path,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )

    assert results and results[0].attachments_dir is not None
    attachments = list(results[0].attachments_dir.iterdir())
    assert attachments and attachments[0].name.startswith("log")
    assert attachments[0].read_text(encoding="utf-8") == "attachment payload"


def test_import_claude_export_zip_with_attachment(tmp_path):
    export_dir = tmp_path / "claude_zip"
    attachments_dir = export_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    (attachments_dir / "result.txt").write_text("claude attachment", encoding="utf-8")
    conversations = {
        "conversations": [
            {
                "uuid": "claude-zip",
                "name": "Zip Claude",
                "chat_messages": [
                    {"sender": "user", "content": [{"type": "text", "text": "Hi"}]},
                    {
                        "sender": "assistant",
                        "content": [
                            {
                                "type": "file",
                                "file_id": "result.txt",
                                "file_name": "result.txt",
                            }
                        ],
                    },
                ],
            }
        ]
    }
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")

    zip_path = tmp_path / "claude_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in export_dir.rglob("*"):
            if file.is_dir():
                continue
            zf.write(file, file.relative_to(export_dir).as_posix())

    out_dir = tmp_path / "out_claude_zip"
    results = import_claude_export(
        zip_path,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )

    assert results and results[0].attachments_dir is not None
    attachments = list(results[0].attachments_dir.iterdir())
    assert attachments and attachments[0].name.startswith("result")
    assert attachments[0].read_text(encoding="utf-8") == "claude attachment"


def test_import_chatgpt_export_repeat_skips(tmp_path):
    export_dir = tmp_path / "chatgpt_repeat"
    export_dir.mkdir()
    conversations = [
        {
            "id": "repeat-id",
            "title": "Repeat Chat",
            "create_time": 1,
            "update_time": 2,
            "mapping": {
                "user": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                    }
                }
            },
        }
    ]
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")

    out_dir = tmp_path / "out_repeat"
    first = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    assert len(first) == 1
    assert not first[0].skipped

    second = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
    )
    assert len(second) == 1
    assert second[0].skipped
    assert second[0].document is None
    assert second[0].markdown_path == first[0].markdown_path
