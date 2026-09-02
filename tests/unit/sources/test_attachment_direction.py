"""Attachment direction must come from the owning turn's role evidence."""

from __future__ import annotations

from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedAttachment
from polylogue.sources.parsers.browser_capture import parse as parse_browser_capture
from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping
from polylogue.sources.parsers.claude.common import normalize_chat_messages
from polylogue.sources.parsers.drive import parse_chunked_prompt


def test_parsed_attachment_has_no_direction_without_evidence() -> None:
    """The model must not silently classify an attachment as user input."""
    assert ParsedAttachment(provider_attachment_id="unclassified").direction is None


def test_chatgpt_attachment_direction_uses_message_role() -> None:
    mapping = {
        "user-node": {
            "message": {
                "id": "user-message",
                "author": {"role": "user"},
                "content": {"parts": ["Read this"]},
                "metadata": {"attachments": [{"id": "user-file", "name": "input.txt"}]},
            }
        },
        "assistant-node": {
            "message": {
                "id": "assistant-message",
                "author": {"role": "assistant"},
                "content": {"parts": ["Here is the result"]},
                "metadata": {"attachments": [{"id": "output-file", "name": "output.txt"}]},
            }
        },
    }

    _messages, attachments = extract_messages_from_mapping(mapping)
    by_id = {attachment.provider_attachment_id: attachment for attachment in attachments}

    assert by_id["user-file"].direction == "user_input"
    assert by_id["user-file"].producer_ref is None
    assert by_id["output-file"].direction == "model_output"
    assert by_id["output-file"].producer_ref == "message:assistant-message"


def test_gemini_attachment_direction_uses_chunk_role_for_all_attachment_shapes() -> None:
    session = parse_chunked_prompt(
        Provider.GEMINI,
        {
            "chunkedPrompt": {
                "chunks": [
                    {
                        "id": "user-turn",
                        "role": "user",
                        "text": "Use these files",
                        "driveDocument": {"id": "user-drive-file"},
                    },
                    {
                        "id": "model-turn",
                        "role": "model",
                        "parts": [
                            {"inlineData": {"mimeType": "text/plain", "data": "b3V0"}},
                            {"fileData": {"fileUri": "drive://output-file"}},
                        ],
                    },
                ]
            }
        },
        "gemini-direction",
    )
    by_kind = {attachment.attachment_kind: attachment for attachment in session.attachments}

    assert by_kind[None].direction == "user_input"
    assert by_kind[None].producer_ref is None
    assert by_kind["inline_data"].direction == "model_output"
    assert by_kind["inline_data"].producer_ref == "message:model-turn"
    assert by_kind["file_data"].direction == "model_output"
    assert by_kind["file_data"].producer_ref == "message:model-turn"


def test_claude_attachment_direction_uses_message_role() -> None:
    normalized = normalize_chat_messages(
        [
            {"id": "user-message", "role": "user", "text": "Read this", "attachments": [{"id": "input"}]},
            {
                "id": "assistant-message",
                "role": "assistant",
                "text": "Generated this",
                "attachments": [{"id": "output"}],
            },
        ]
    )
    by_id = {attachment.provider_attachment_id: attachment for attachment in normalized.attachments}

    assert by_id["input"].direction == "user_input"
    assert by_id["output"].direction == "model_output"
    assert by_id["output"].producer_ref == "message:assistant-message"


def test_browser_capture_attachment_direction_uses_turn_role() -> None:
    parsed = parse_browser_capture(
        {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "gemini:direction",
            "provenance": {
                "source_url": "https://example.test/chat",
                "captured_at": "2026-08-31T00:00:00Z",
                "adapter_name": "test",
            },
            "session": {
                "provider": "gemini",
                "provider_session_id": "direction",
                "turns": [
                    {
                        "provider_turn_id": "user-turn",
                        "role": "user",
                        "text": "Read this",
                        "attachments": [{"provider_attachment_id": "input"}],
                    },
                    {
                        "provider_turn_id": "assistant-turn",
                        "role": "assistant",
                        "text": "Generated this",
                        "attachments": [{"provider_attachment_id": "output"}],
                    },
                ],
            },
        },
        "direction",
    )
    by_id = {attachment.provider_attachment_id: attachment for attachment in parsed.attachments}

    assert by_id["input"].direction == "user_input"
    assert by_id["output"].direction == "model_output"
    assert by_id["output"].producer_ref == "message:assistant-turn"
