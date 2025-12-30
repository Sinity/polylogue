from __future__ import annotations

import json

from polylogue.config import Source
from polylogue.source_ingest import iter_source_conversations


def test_auto_detect_chatgpt_and_claude(tmp_path):
    chatgpt_payload = {
        "id": "conv-chatgpt",
        "mapping": {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello"]},
                    "create_time": 1,
                },
            }
        },
    }
    claude_payload = {
        "conversations": [
            {
                "id": "conv-claude",
                "name": "Claude Chat",
                "chat_messages": [
                    {
                        "id": "msg-1",
                        "sender": "user",
                        "content": [{"type": "text", "text": "Hi"}],
                    }
                ],
            }
        ]
    }
    (tmp_path / "chatgpt.json").write_text(json.dumps(chatgpt_payload), encoding="utf-8")
    (tmp_path / "claude.json").write_text(json.dumps(claude_payload), encoding="utf-8")

    source = Source(name="inbox", type="auto", path=tmp_path)
    conversations = list(iter_source_conversations(source))
    providers = {convo.provider_name for convo in conversations}
    assert "chatgpt" in providers
    assert "claude" in providers


def test_claude_chat_messages_attachments(tmp_path):
    payload = {
        "chat_messages": [
            {
                "id": "msg-1",
                "sender": "assistant",
                "content": [{"type": "text", "text": "Files"}],
                "attachments": [
                    {"id": "file-1", "name": "notes.txt", "size": 12, "mimeType": "text/plain"}
                ],
            }
        ]
    }
    source_file = tmp_path / "claude.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    source = Source(name="inbox", type="auto", path=source_file)
    conversations = list(iter_source_conversations(source))
    assert conversations
    convo = conversations[0]
    assert convo.attachments
    attachment = convo.attachments[0]
    assert attachment.provider_attachment_id == "file-1"
    assert attachment.name == "notes.txt"
