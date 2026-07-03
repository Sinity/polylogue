"""Tests for raw-materialization artifact classification."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.archive.raw_materialization import parsed_non_session_artifact_reason


def _write_blob(root: Path, blob_hash: str, text: str) -> bytes:
    blob_path = root / "blob" / blob_hash[:2] / blob_hash[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_text(text, encoding="utf-8")
    return bytes.fromhex(blob_hash)


def test_claude_code_bridge_prompt_sidecars_are_non_session_artifacts(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "55" * 32,
        "\n".join(
            (
                '{"type":"last-prompt","leafUuid":"leaf","sessionId":"session"}',
                '{"type":"bridge-session","sessionId":"session","bridgeSessionId":"bridge"}',
            )
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-code-session",
            source_path=str(tmp_path / "session.jsonl"),
            blob_hash=blob_hash,
        )
        == "Claude Code bridge/last-prompt sidecar"
    )


def test_claude_ai_empty_conversations_are_non_session_artifacts(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "1b" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "name": "",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "",
                        "content": [],
                        "sender": "human",
                        "attachments": [],
                        "files": [],
                    },
                    {
                        "uuid": "m2",
                        "text": "",
                        "content": [],
                        "sender": "assistant",
                        "attachments": [],
                        "files": [],
                    },
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        == "Claude.ai empty conversation artifact"
    )


def test_claude_ai_attachment_shells_without_content_are_empty(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "3d" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "",
                        "content": [],
                        "sender": "human",
                        "attachments": [
                            {
                                "file_name": "",
                                "file_size": 0,
                                "file_type": "pdf",
                                "extracted_content": "",
                            }
                        ],
                        "files": [{"file_uuid": "file-1", "file_name": ""}],
                    }
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        == "Claude.ai empty conversation artifact"
    )


def test_claude_ai_empty_content_blocks_are_empty(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "5f" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "",
                        "content": [
                            {
                                "type": "text",
                                "text": "",
                                "start_timestamp": "2025-01-25T16:34:30Z",
                                "stop_timestamp": "2025-01-25T16:34:30Z",
                                "citations": [],
                            }
                        ],
                        "sender": "human",
                        "attachments": [],
                        "files": [],
                    }
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        == "Claude.ai empty conversation artifact"
    )


def test_claude_ai_conversation_without_messages_is_non_session_metadata(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "7b" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "name": "Janus worldview and philosophy compilation",
                "summary": "",
                "chat_messages": [],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        == "Claude.ai empty conversation artifact"
    )


def test_claude_ai_conversation_with_text_remains_session_shaped(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "2c" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "real prompt",
                        "content": [],
                        "sender": "human",
                        "attachments": [],
                        "files": [],
                    }
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        is None
    )


def test_claude_ai_conversation_with_content_text_remains_session_shaped(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "6a" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "",
                        "content": [{"type": "text", "text": "real prompt", "citations": []}],
                        "sender": "human",
                        "attachments": [],
                        "files": [],
                    }
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        is None
    )


def test_claude_ai_conversation_with_attachment_content_remains_session_shaped(tmp_path: Path) -> None:
    blob_hash = _write_blob(
        tmp_path,
        "4e" * 32,
        json.dumps(
            {
                "uuid": "conversation",
                "chat_messages": [
                    {
                        "uuid": "m1",
                        "text": "",
                        "content": [],
                        "sender": "human",
                        "attachments": [
                            {
                                "file_name": "paper.pdf",
                                "file_size": 12,
                                "file_type": "pdf",
                                "extracted_content": "",
                            }
                        ],
                        "files": [],
                    }
                ],
            }
        ),
    )

    assert (
        parsed_non_session_artifact_reason(
            archive_root=tmp_path,
            origin="claude-ai-export",
            source_path=str(tmp_path / "claude-ai.zip:conversations.json"),
            blob_hash=blob_hash,
        )
        is None
    )
