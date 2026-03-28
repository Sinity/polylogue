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
from typing import Any

import pytest

from polylogue.sources.parsers.drive import extract_text_from_chunk, parse_chunked_prompt


@pytest.fixture
def synthetic_gemini_payload() -> dict[str, Any]:
    from polylogue.schemas.synthetic import SyntheticCorpus

    corpus = SyntheticCorpus.for_provider("gemini")
    raw = corpus.generate(count=1, messages_per_conversation=range(4, 8), seed=42)[0]
    return json.loads(raw)


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


def test_parse_chunked_prompt_preserves_core_conversation_metadata() -> None:
    payload = {
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
    assert result.created_at == "2024-01-15T10:30:00Z"
    assert result.updated_at == "2024-01-15T11:45:00Z"
    assert [message.provider_message_id for message in result.messages] == ["msg-user", "msg-model"]
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert [message.text for message in result.messages] == ["Question", "Answer"]


def test_parse_chunked_prompt_preserves_reasoning_code_tool_results_and_attachments() -> None:
    payload = {
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
    assert result.messages[0].provider_meta["content_blocks"] == [{"type": "text", "text": "question"}]
    assert result.messages[1].provider_meta["content_blocks"] == [{"type": "thinking", "text": "reasoning"}]
    assert result.messages[1].provider_meta["reasoning_traces"] == [
        {"text": "reasoning", "token_count": 32, "provider": "gemini"}
    ]
    assert result.messages[2].provider_meta["content_blocks"] == [
        {"type": "text", "text": "inline"},
        {"type": "code", "text": "print('ok')"},
        {"type": "tool_result", "text": "ok"},
    ]
    assert len(result.attachments) == 1
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[0].mime_type == "application/pdf"
    assert result.attachments[0].size_bytes == 12


def test_parse_chunked_prompt_skips_chunks_without_text_or_role() -> None:
    payload = {
        "chunkedPrompt": {
            "chunks": [
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


def test_parse_chunked_prompt_accepts_synthetic_exports(synthetic_gemini_payload: dict[str, Any]) -> None:
    result = parse_chunked_prompt("gemini", synthetic_gemini_payload, "synthetic-fallback")

    assert result.provider_name == "gemini"
    assert result.messages
    assert all(message.text for message in result.messages)
    assert all(message.provider_meta is not None for message in result.messages)
