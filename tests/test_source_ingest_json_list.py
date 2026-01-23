from __future__ import annotations

import json
from pathlib import Path

from polylogue.config import Source
from polylogue.ingestion import iter_source_conversations


def test_iter_source_conversations_handles_codex_json_list(tmp_path: Path):
    """Test that Codex/Claude-Code/Gemini single-conversation JSON lists are not unpacked."""
    # A Codex/Claude-Code export is often a list of messages representing one conversation
    # If unpacked, this would look like N conversations with 0 messages each.
    # If not unpacked, it looks like 1 conversation with N messages.
    payload = [
        {"prompt": "Hello", "completion": "Hi"},
        {"prompt": "How are you?", "completion": "Good"},
    ]
    
    # Write as a single JSON file
    source_file = tmp_path / "codex_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    # Name hint helps detection
    source = Source(name="codex", path=source_file)
    
    conversations = list(iter_source_conversations(source))
    
    # Should result in ONE conversation with messages, NOT multiple empty ones
    assert len(conversations) == 1
    convo = conversations[0]
    assert convo.provider_name == "codex"
    # 2 items * 2 messages (prompt+completion) = 4 messages
    assert len(convo.messages) == 4
    assert convo.messages[0].text == "Hello"


def test_iter_source_conversations_handles_claude_code_json_list(tmp_path: Path):
    """Test that Claude Code single-conversation JSON lists are not unpacked."""
    payload = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "message": {"content": "Hello"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "message": {"content": "Hi"},
        },
    ]
    
    source_file = tmp_path / "claude-code_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="claude-code", path=source_file)
    
    conversations = list(iter_source_conversations(source))
    
    assert len(conversations) == 1
    convo = conversations[0]
    assert convo.provider_name == "claude-code"
    assert len(convo.messages) == 2
    assert convo.messages[0].text == "Hello"


def test_iter_source_conversations_still_unpacks_chatgpt_json_list(tmp_path: Path):
    """Test that ChatGPT list of conversations IS unpacked (default behavior)."""
    # ChatGPT export is a list of conversation objects
    payload = [
        {
            "title": "Conv 1",
            "mapping": {
                "n1": {"message": {"content": {"parts": ["Msg 1"]}, "author": {"role": "user"}}}
            }
        },
        {
            "title": "Conv 2",
            "mapping": {
                "n2": {"message": {"content": {"parts": ["Msg 2"]}, "author": {"role": "user"}}}
            }
        }
    ]
    
    source_file = tmp_path / "chatgpt_export.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="chatgpt", path=source_file)
    
    conversations = list(iter_source_conversations(source))
    
    # Should unpack into 2 conversations
    assert len(conversations) == 2
    assert conversations[0].messages[0].text == "Msg 1"
    assert conversations[1].messages[0].text == "Msg 2"
