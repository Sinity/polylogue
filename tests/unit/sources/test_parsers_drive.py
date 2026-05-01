"""Focused contracts for the Drive/Gemini chunked-prompt parser.

Broader generated-provider coverage already lives in:
- `test_parsers_props.py`
- `test_source_laws.py`
- `test_unified_semantic_laws.py`

This file keeps only the concrete parser behaviors that are still clearest as
direct contracts.
"""

from __future__ import annotations

import json

import pytest

from polylogue.core.json import JSONDocument
from polylogue.scenarios import CorpusSpec
from polylogue.sources.parsers.drive import (
    _attachment_from_doc,
    _collect_drive_docs,
    parse_chunked_prompt,
)
from polylogue.sources.parsers.drive_support import extract_text_from_chunk


def _provider_content_blocks(
    message_index: int, result_provider_meta: dict[str, object] | None
) -> list[dict[str, object]]:
    assert isinstance(result_provider_meta, dict), f"message {message_index} is missing provider_meta"
    blocks = result_provider_meta.get("content_blocks")
    assert isinstance(blocks, list), f"message {message_index} has no structured content_blocks"
    typed_blocks = [block for block in blocks if isinstance(block, dict)]
    assert len(typed_blocks) == len(blocks), f"message {message_index} content_blocks should all be dict payloads"
    return typed_blocks


@pytest.fixture
def synthetic_gemini_payload() -> JSONDocument:
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "gemini",
            count=1,
            messages_min=4,
            messages_max=7,
            seed=42,
            origin="generated.test-drive-parser",
            tags=("synthetic", "test", "drive-parser"),
        )
    )[0]
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ({"text": "hello"}, "hello"),
        ({"content": "alt"}, "alt"),
        ({"message": "msg"}, "msg"),
        ({"parts": [{"text": "alpha"}, "beta", {"text": "gamma"}]}, "alpha\nbeta\ngamma"),
        ({"text": None, "parts": [{"text": "fallback"}]}, "fallback"),
        ({"data": {"text": "nested"}}, None),
        ("not a dict", None),
        (None, None),
    ],
    ids=[
        "text",
        "content",
        "message",
        "parts",
        "parts-fallback",
        "nested-dict-not-recursed",
        "string-input",
        "none-input",
    ],
)
def test_extract_text_from_chunk_contract(chunk: object, expected: str | None) -> None:
    assert extract_text_from_chunk(chunk) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not a dict", []),
        ({"driveDocument": "doc-1"}, ["doc-1"]),
        ({"driveDocuments": [{"id": "doc-2"}, "doc-3"]}, [{"id": "doc-2"}, "doc-3"]),
        (
            {"metadata": {"driveDocument": "nested-doc"}},
            ["nested-doc"],
        ),
    ],
    ids=["non-dict", "single-doc", "list-docs", "nested-doc"],
)
def test_collect_drive_docs_contract(payload: object, expected: list[object]) -> None:
    assert _collect_drive_docs(payload) == expected


@pytest.mark.parametrize(
    ("doc", "expected_id", "expected_size"),
    [
        ("doc-string-id", "doc-string-id", None),
        ({"id": "doc-1", "sizeBytes": "5000"}, "doc-1", 5000),
        ({"fileId": "doc-2", "size": 12}, "doc-2", 12),
        (123, None, None),
        ({"name": "missing-id"}, None, None),
    ],
    ids=["string-doc", "size-bytes-string", "file-id-int-size", "invalid-type", "missing-id"],
)
def test_attachment_from_doc_contract(doc: object, expected_id: str | None, expected_size: int | None) -> None:
    attachment = _attachment_from_doc(doc if isinstance(doc, dict | str) else {}, "msg-1")
    if expected_id is None:
        assert attachment is None
    else:
        assert attachment is not None
        assert attachment.provider_attachment_id == expected_id
        assert attachment.message_provider_id == "msg-1"
        assert attachment.size_bytes == expected_size
        if isinstance(doc, dict):
            assert attachment.provider_meta is not None
            for key, value in doc.items():
                assert attachment.provider_meta[key] == value


def test_parse_chunked_prompt_preserves_core_conversation_metadata() -> None:
    payload: JSONDocument = {
        "id": "gemini-conv",
        "displayName": "Gemini Session",
        "createTime": "2024-01-15T10:30:00Z",
        "updateTime": "2024-01-15T11:45:00Z",
        "chunkedPrompt": {
            "chunks": [
                {"id": "msg-user", "role": "user", "text": "Question"},
                {"id": "msg-model", "role": "model", "text": "Answer"},
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.provider_name == "gemini"
    assert result.provider_conversation_id == "gemini-conv"
    assert result.title == "Gemini Session"
    assert result.provider_meta == {"title_source": "imported:displayName"}
    assert result.created_at == "2024-01-15T10:30:00Z"
    assert result.updated_at == "2024-01-15T11:45:00Z"
    assert [message.provider_message_id for message in result.messages] == ["msg-user", "msg-model"]
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert [message.text for message in result.messages] == ["Question", "Answer"]


def test_parse_chunked_prompt_records_fallback_title_source() -> None:
    payload: JSONDocument = {
        "id": "gemini-fallback-title",
        "chunkedPrompt": {"chunks": [{"id": "msg-user", "role": "user", "text": "Question"}]},
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.title == "fallback-id"
    assert result.provider_meta == {"title_source": "fallback:id"}


def test_parse_chunked_prompt_preserves_reasoning_code_tool_results_and_attachments() -> None:
    payload: JSONDocument = {
        "id": "gemini-rich",
        "displayName": "Gemini Rich",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-user",
                    "role": "user",
                    "text": "question",
                    "driveDocument": {
                        "id": "doc-1",
                        "name": "spec.pdf",
                        "mimeType": "application/pdf",
                        "sizeBytes": "12",
                    },
                },
                {
                    "id": "msg-thought",
                    "role": "model",
                    "text": "reasoning",
                    "isThought": True,
                    "thinkingBudget": 32,
                },
                {
                    "id": "msg-code",
                    "role": "model",
                    "parts": [{"text": "inline"}],
                    "executableCode": {"language": "python", "code": "print('ok')"},
                    "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "ok"},
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.provider_message_id for message in result.messages] == [
        "msg-user",
        "msg-thought",
        "msg-code",
    ]
    assert [block.type for block in result.messages[0].content_blocks] == ["text", "document"]
    assert [block.type for block in result.messages[1].content_blocks] == ["thinking"]
    assert [block.type for block in result.messages[2].content_blocks] == ["text", "code", "tool_result"]
    user_blocks = _provider_content_blocks(0, result.messages[0].provider_meta)
    assert user_blocks[0]["type"] == "text"
    assert user_blocks[0]["text"] == "question"
    assert user_blocks[1]["type"] == "document"
    metadata = user_blocks[1].get("metadata")
    assert isinstance(metadata, dict)
    drive_document = metadata.get("driveDocument")
    assert isinstance(drive_document, dict)
    assert drive_document["id"] == "doc-1"

    thought_blocks = _provider_content_blocks(1, result.messages[1].provider_meta)
    assert thought_blocks[0]["type"] == "thinking"
    assert thought_blocks[0]["text"] == "reasoning"
    thought_meta = result.messages[1].provider_meta
    assert isinstance(thought_meta, dict)
    reasoning_traces = thought_meta.get("reasoning_traces")
    assert reasoning_traces == [{"text": "reasoning", "token_count": 32, "provider": "gemini"}]
    code_blocks = _provider_content_blocks(2, result.messages[2].provider_meta)
    assert [block["type"] for block in code_blocks] == ["text", "code", "tool_result"]
    assert code_blocks[0]["text"] == "inline"
    assert code_blocks[1]["text"] == "print('ok')"
    assert code_blocks[2]["text"] == "ok"
    assert len(result.attachments) == 1
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[0].mime_type == "application/pdf"
    assert result.attachments[0].size_bytes == 12


def test_parse_chunked_prompt_preserves_attachment_only_chunks_and_chunk_timestamps() -> None:
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-doc",
                    "role": "user",
                    "createTime": "2024-01-15T10:30:00Z",
                    "driveDocument": {"id": "doc-1", "name": "notes.txt"},
                },
                {
                    "id": "msg-inline",
                    "role": "user",
                    "createTime": "2024-01-15T10:31:00Z",
                    "inlineFile": {"mimeType": "text/plain", "data": "aGVsbG8="},
                },
                {
                    "id": "msg-video",
                    "role": "model",
                    "createTime": "2024-01-15T10:32:00Z",
                    "youtubeVideo": {"id": "vid-1"},
                },
            ]
        }
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.provider_message_id for message in result.messages] == ["msg-doc", "msg-inline", "msg-video"]
    assert [message.timestamp for message in result.messages] == [
        "2024-01-15T10:30:00Z",
        "2024-01-15T10:31:00Z",
        "2024-01-15T10:32:00Z",
    ]
    assert [message.text for message in result.messages] == [None, None, None]
    assert result.created_at == "2024-01-15T10:30:00Z"
    assert result.updated_at == "2024-01-15T10:32:00Z"
    assert [block.type for block in result.messages[0].content_blocks] == ["document"]
    assert len(result.attachments) == 3
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[1].provider_attachment_id.startswith("inline-file-")
    assert result.attachments[1].mime_type == "text/plain"
    assert result.attachments[1].size_bytes == 5
    assert result.attachments[2].provider_attachment_id == "youtube-video-vid-1"
    assert result.attachments[2].mime_type == "video/youtube"


def test_parse_chunked_prompt_skips_chunks_without_text_or_role() -> None:
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                "string chunk without role",
                {"text": "missing role"},
                {"role": "user"},
                {"role": "user", "text": "kept"},
                {"role": "model", "parts": [{"text": "also kept"}]},
            ]
        }
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert [message.text for message in result.messages] == ["kept", "also kept"]


def test_parse_chunked_prompt_accepts_synthetic_exports(synthetic_gemini_payload: JSONDocument) -> None:
    result = parse_chunked_prompt("gemini", synthetic_gemini_payload, "synthetic-fallback")

    assert result.provider_name == "gemini"
    assert result.messages
    assert all(message.text for message in result.messages)
    assert all(message.provider_meta is not None for message in result.messages)
