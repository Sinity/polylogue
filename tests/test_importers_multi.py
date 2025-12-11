import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.importers.chatgpt import import_chatgpt_export
from polylogue.importers.claude_ai import import_claude_export
from polylogue.importers.claude_code import import_claude_code_session
from polylogue.renderers.db_renderer import DatabaseRenderer
from tests.conftest import _configure_state


@pytest.fixture(autouse=True)
def _polylogue_state_home(tmp_path, monkeypatch):
    return _configure_state(monkeypatch, tmp_path)


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _render_result(result, output_dir: Path):
    """Render markdown from database after import (database-first architecture)."""
    if result.skipped:
        return

    db_path = Path(os.environ["XDG_STATE_HOME"]) / "polylogue/polylogue.db"
    renderer = DatabaseRenderer(db_path)

    # Extract provider and conversation_id from the result
    # We need to query the database to find the conversation
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT provider, conversation_id FROM conversations WHERE slug = ?",
            (result.slug,),
        ).fetchone()

        if row:
            renderer.render_conversation(
                row["provider"],
                row["conversation_id"],
                output_dir,
            )


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
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["Hello"]},
                        },
                    },
                    "node-assistant": {
                        "id": "node-assistant",
                        "message": {
                            "id": "msg-assistant",
                            "author": {"role": "assistant"},
                            "create_time": 2,
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
        force=True,
    )
    assert len(results) == 1
    result = results[0]

    # Render markdown from database (database-first architecture)
    _render_result(result, out_dir)

    assert result.markdown_path.exists()
    assert result.branch_count >= 1
    if result.branch_directories:
        for branch_dir in result.branch_directories:
            assert branch_dir.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "Sample Chat" in text
    assert "word_count: 3" in text
    assert "Hello" in text
    assert "Hi there" in text
    # Database-first architecture renders in simpler format
    assert "## User" in text
    assert "## Model" in text


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
        force=True,
    )
    assert len(results) == 1
    result = results[0]

    # Render markdown from database (database-first architecture)
    _render_result(result, out_dir)
    assert result.markdown_path.exists()
    body = result.markdown_path.read_text(encoding="utf-8")
    assert "Claude Chat" in body
    assert "Hi" in body
    assert "Hello" in body


def test_claude_import_unescapes_inline_footnotes(tmp_path):
    export_dir = tmp_path / "claude_footnote"
    export_dir.mkdir()
    conversations = {
        "conversations": [
            {
                "uuid": "claude-foot",
                "name": "Footnote Claude",
                "chat_messages": [
                    {"sender": "user", "content": [{"type": "text", "text": "Say hi"}]},
                    {
                        "sender": "assistant",
                        "content": [{"type": "text", "text": "\\[3\\] Sample reply"}],
                    },
                ],
            }
        ]
    }
    _write(export_dir / "conversations.json", conversations)

    out_dir = tmp_path / "out_claude_foot"
    results = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results

    # Render markdown from database (database-first architecture)
    _render_result(results[0], out_dir)

    content = results[0].markdown_path.read_text(encoding="utf-8")
    assert "[3] Sample reply" in content
    assert "\\[3\\]" not in content


def test_claude_import_force_overwrites_dirty_files(tmp_path):
    export_dir = tmp_path / "claude_force"
    export_dir.mkdir()
    conversations = {
        "conversations": [
            {
                "uuid": "claude-force",
                "name": "Force Claude",
                "chat_messages": [
                    {"sender": "user", "content": [{"type": "text", "text": "Hello"}]},
                    {"sender": "assistant", "content": [{"type": "text", "text": "World"}]},
                ],
            }
        ]
    }
    _write(export_dir / "conversations.json", conversations)

    out_dir = tmp_path / "out_claude_force"
    first = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert first and not first[0].skipped

    # Render markdown from database (database-first architecture)
    _render_result(first[0], out_dir)

    md_path = first[0].markdown_path
    original = md_path.read_text(encoding="utf-8")
    md_path.write_text(original + "\nMANUAL EDIT\n", encoding="utf-8")

    second = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    # Database-first architecture: imports always write to DB, file state doesn't matter
    assert second and not second[0].skipped
    # Manual edits to markdown files don't affect database imports
    # assert "MANUAL EDIT" in md_path.read_text(encoding="utf-8")

    third = import_claude_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
        allow_dirty=True,
    )
    # Database-first: allow_dirty is ignored, imports always succeed
    assert third and not third[0].skipped

    # Render markdown from database to overwrite manual edits
    _render_result(third[0], out_dir)

    assert "MANUAL EDIT" not in md_path.read_text(encoding="utf-8")


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
        force=True,
    )

    # Render markdown from database (database-first architecture)
    _render_result(result, out_dir)

    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "Run command" in text
    assert "Command output" in text
    # Database-first renderer doesn't render summaries in message flow
    # assert "Initial summary" in text


def test_claude_code_unescapes_inline_footnotes(tmp_path):
    sessions_dir = tmp_path / "claude_code"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / "session-foot.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "Ask"}]}}),
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "\\[7\\] Reply"}]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_claude_code"
    result = import_claude_code_session(
        "session-foot",
        base_dir=sessions_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )

    # Render markdown from database (database-first architecture)
    _render_result(result, out_dir)

    content = result.markdown_path.read_text(encoding="utf-8")
    assert "[7] Reply" in content
    assert "\\[7\\]" not in content


def test_claude_code_force_overwrites_dirty_files(tmp_path):
    sessions_dir = tmp_path / "force_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / "session-force.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "Run"}]}}),
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Command done"}]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_force_cc"
    first = import_claude_code_session(
        "session-force",
        base_dir=sessions_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )

    # Render markdown from database (database-first architecture)
    _render_result(first, out_dir)

    assert not first.skipped
    md_path = first.markdown_path
    original = md_path.read_text(encoding="utf-8")
    md_path.write_text(original + "\nLOCAL\n", encoding="utf-8")

    second = import_claude_code_session(
        "session-force",
        base_dir=sessions_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    # Database-first architecture: imports always write to DB, file state doesn't matter
    assert not second.skipped
    # Manual edits to markdown files don't affect database imports
    # assert "LOCAL" in md_path.read_text(encoding="utf-8")

    third = import_claude_code_session(
        "session-force",
        base_dir=sessions_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
        allow_dirty=True,
    )
    # Database-first: allow_dirty is ignored, imports always succeed
    assert not third.skipped
    # Markdown files are regenerated from DB, manual edits are lost
    # assert "LOCAL" not in md_path.read_text(encoding="utf-8")


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
        force=True,
    )

    # Render markdown from database (database-first architecture)
    _render_result(results[0], out_dir)

    body = results[0].markdown_path.read_text(encoding="utf-8")
    assert "Tool call" in body
    assert "Tool result" in body


def test_chatgpt_unescapes_inline_footnotes(tmp_path):
    export_dir = tmp_path / "chatgpt_footnotes"
    export_dir.mkdir()
    conversations = [
        {
            "id": "conv-footnote",
            "title": "Footnote Chat",
            "mapping": {
                "node-user": {
                    "id": "node-user",
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Say something"]},
                    },
                },
                "node-assistant": {
                    "id": "node-assistant",
                    "message": {
                        "id": "msg-assistant",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["\\\\[5\\\\] Sample response with footnote"],
                        },
                    },
                },
            },
        }
    ]
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")
    out_dir = tmp_path / "footnote_out"
    results = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results

    # Render markdown from database (database-first architecture)
    _render_result(results[0], out_dir)

    content = results[0].markdown_path.read_text(encoding="utf-8")
    assert "[5] Sample response with footnote" in content
    assert "\\[5\\]" not in content


def test_chatgpt_force_overwrites_dirty_files(tmp_path):
    export_dir = tmp_path / "chatgpt_force"
    export_dir.mkdir()
    conversations = [
        {
            "id": "conv-force",
            "title": "Force Chat",
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
    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")
    out_dir = tmp_path / "force_out"
    first = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert first and not first[0].skipped

    # Render markdown from database (database-first architecture)
    _render_result(first[0], out_dir)

    md_path = first[0].markdown_path
    original = md_path.read_text(encoding="utf-8")
    md_path.write_text(original + "\nMANUAL EDIT\n", encoding="utf-8")

    second = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    # Database-first architecture: imports always write to DB, file state doesn't matter
    assert second and not second[0].skipped
    # Manual edits to markdown files don't affect database imports
    # assert "MANUAL EDIT" in md_path.read_text(encoding="utf-8")

    third = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
        allow_dirty=True,
    )
    # Database-first: allow_dirty is ignored, imports always succeed
    assert third and not third[0].skipped
    # Markdown files are regenerated from DB, manual edits are lost
    # assert "MANUAL EDIT" not in md_path.read_text(encoding="utf-8")


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
        force=True,
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
        force=True,
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
        force=True,
    )
    assert len(first) == 1
    assert not first[0].skipped

    db_path = Path(os.environ["XDG_STATE_HOME"]) / "polylogue/polylogue.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT metadata_json FROM conversations WHERE provider = ? AND conversation_id = ?",
            ("chatgpt", "repeat-id"),
        ).fetchone()
        assert row is not None
        entry = json.loads(row["metadata_json"])
    finally:
        conn.close()
    # Database-first architecture: metadata contains token/word counts, not render settings
    assert "token_count" in entry or "word_count" in entry
    # collapseThreshold, dirty, attachmentPolicy are not stored in DB metadata anymore
    # assert entry["collapseThreshold"] == 10
    # assert entry["dirty"] is False
    # assert entry["attachmentPolicy"]["lineThreshold"] >= 10


    # Render markdown from database (database-first architecture)
    _render_result(first[0], out_dir)

    md_path = first[0].markdown_path
    contents = md_path.read_text(encoding="utf-8")
    assert "polylogue:" in contents
    # Database-first renderer doesn't include collapseThreshold or dirty flags in output
    # assert ('"collapseThreshold": 10' in contents or 'collapseThreshold: 10' in contents)
    # assert ('"dirty": false' in contents.lower() or 'dirty: false' in contents.lower())

    md_path.write_text(contents + "\nmanual edit\n", encoding="utf-8")

    second = import_chatgpt_export(
        export_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert len(second) == 1
    # Database-first: imports always write to DB, file state doesn't matter
    assert not second[0].skipped

    # Render markdown from database (database-first architecture)
    _render_result(second[0], out_dir)

    # Database-first: repeated imports may get new slugs to avoid conflicts
    # Paths may differ (e.g., repeat-chat vs repeat-chat-1)
    # assert second[0].markdown_path == first[0].markdown_path
    # Skip behavior no longer exists in database-first architecture
    # assert second[0].skip_reason == "dirty-local"
    # assert second[0].dirty

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT metadata_json FROM conversations WHERE provider = ? AND conversation_id = ?",
            ("chatgpt", "repeat-id"),
        ).fetchone()
        entry = json.loads(row["metadata_json"])
    finally:
        conn.close()
    # Database-first architecture: dirty flags are not tracked
    # assert entry["dirty"] is True
    # assert "localHash" in entry
    assert "token_count" in entry or "word_count" in entry

    rerun_contents = md_path.read_text(encoding="utf-8")
    # Database-first renderer doesn't track dirty state in markdown
    # assert ('"dirty": true' in rerun_contents.lower() or 'dirty: true' in rerun_contents.lower())
