"""Pinned provider content-extraction regressions.

Broad semantic equivalence/content-block laws live in
``tests/unit/sources/test_unified_semantic_laws.py``.
"""

from __future__ import annotations

import pytest

from polylogue.lib.provider_semantics import extract_claude_code_text
from polylogue.sources.providers.chatgpt import ChatGPTAuthor, ChatGPTContent, ChatGPTMessage
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.gemini import GeminiMessage


def test_extract_claude_code_text_excludes_thinking_blocks() -> None:
    blocks = [
        {"type": "thinking", "thinking": "Let me reason about this..."},
        {"type": "text", "text": "Here is my answer."},
    ]
    text = extract_claude_code_text(blocks)
    assert "Let me reason" not in text
    assert "Here is my answer." in text


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (None, ""),
        (["text", None], "text"),
        ([None, None, None], ""),
        ([{"text": "Unicode: 你好 мир 🌍"}, "English"], "Unicode: 你好 мир 🌍\nEnglish"),
    ],
    ids=["none", "mixed-none", "all-none", "unicode-dict-parts"],
)
def test_chatgpt_parts_none_guard_contract(parts, expected: str) -> None:
    message = ChatGPTMessage(
        id="m1",
        author=ChatGPTAuthor(role="assistant"),
        content=ChatGPTContent(content_type="text", parts=parts),
    )
    assert message.text_content == expected


@pytest.mark.parametrize(
    ("kwargs", "expected_text", "expected_block_types"),
    [
        (
            {"text": "", "role": "model", "parts": [{"fileData": {"mimeType": "application/pdf", "fileUri": "uri..."}}]},
            "",
            [],
        ),
        (
            {
                "text": "Here's the code",
                "role": "model",
                "executableCode": {"language": "python", "code": "print('hello')"},
                "codeExecutionResult": {"outcome": "OK", "output": "hello"},
            },
            "Here's the code",
            ["text", "code", "tool_result"],
        ),
        (
            {
                "text": "",
                "role": "model",
                "parts": [{"text": None}, {"text": None}, {"text": "Finally"}, {"text": None}],
            },
            "Finally",
            [],
        ),
    ],
    ids=["file-data", "code-execution", "none-text-parts"],
)
def test_gemini_special_part_contract(kwargs: dict[str, object], expected_text: str, expected_block_types: list[str]) -> None:
    message = GeminiMessage(**kwargs)
    assert message.text_content == expected_text
    if expected_block_types:
        assert [block.type.value for block in message.extract_content_blocks()] == expected_block_types


def test_claude_code_only_thinking_yields_empty_text_and_reasoning_traces() -> None:
    record = ClaudeCodeRecord(
        type="assistant",
        message={
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking 1"},
                {"type": "thinking", "thinking": "Thinking 2"},
            ],
        },
    )

    assert record.text_content == ""
    assert [trace.text for trace in record.extract_reasoning_traces()] == ["Thinking 1", "Thinking 2"]
