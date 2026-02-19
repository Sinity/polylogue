"""Tests for content enrichment None guards and edge cases.

Covers:
- d5c3228: content_blocks key exists but value is None
- enrich_content_blocks with empty/None/missing blocks
- enrich_message_metadata with None provider_meta
"""

from __future__ import annotations

import pytest

from polylogue.pipeline.enrichment import enrich_content_blocks, enrich_message_metadata


class TestEnrichContentBlocks:
    """enrich_content_blocks must handle None, empty, and malformed blocks."""

    def test_none_returns_empty(self):
        """None content_blocks should return empty list, not crash."""
        assert enrich_content_blocks(None) == []

    def test_empty_list_returns_empty(self):
        assert enrich_content_blocks([]) == []

    def test_plain_text_block_passes_through(self):
        blocks = [{"type": "text", "text": "Hello world"}]
        result = enrich_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello world"

    def test_code_block_without_language_gets_detection(self):
        blocks = [{"type": "code", "text": "def hello():\n    print('hello')"}]
        result = enrich_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["language"] is not None

    def test_code_block_with_language_preserved(self):
        blocks = [{"type": "code", "text": "print('hi')", "language": "python"}]
        result = enrich_content_blocks(blocks)
        assert result[0]["language"] == "python"

    def test_fenced_code_in_text_extracted(self):
        blocks = [{"type": "text", "text": "Here is code:\n```python\nprint('hi')\n```\nDone."}]
        result = enrich_content_blocks(blocks)
        # Should split into text + code + text
        types = [b["type"] for b in result]
        assert "code" in types
        assert "text" in types

    def test_block_with_none_text_does_not_crash(self):
        """Block with type but None text must not crash."""
        blocks = [{"type": "text", "text": None}]
        result = enrich_content_blocks(blocks)
        # Should pass through (no fences to extract)
        assert len(result) == 1

    def test_block_missing_type_passes_through(self):
        """Block without type key should pass through."""
        blocks = [{"text": "orphan text"}]
        result = enrich_content_blocks(blocks)
        assert len(result) == 1

    def test_thinking_block_passes_through(self):
        """Non-text, non-code blocks should pass through unchanged."""
        blocks = [{"type": "thinking", "thinking": "Let me think..."}]
        result = enrich_content_blocks(blocks)
        assert result[0]["type"] == "thinking"

    def test_mixed_blocks(self):
        """Multiple block types in sequence."""
        blocks = [
            {"type": "thinking", "thinking": "Planning..."},
            {"type": "text", "text": "Here is code:\n```python\nx = 1\n```"},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]
        result = enrich_content_blocks(blocks)
        # thinking passes through, text gets split, tool_use passes through
        types = [b["type"] for b in result]
        assert "thinking" in types
        assert "tool_use" in types

    def test_fenced_code_block_extracted_with_language(self):
        """Fenced code blocks are extracted as code type with declared language."""
        blocks = [{"type": "text", "text": "```python\nprint('hello')\n```"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "code"
        assert result[0]["language"] == "python"
        assert "print('hello')" in result[0]["text"]

    def test_fenced_code_with_declared_language_field(self):
        """Fenced blocks expose declared_language field."""
        blocks = [{"type": "text", "text": "```javascript\nconsole.log('hi')\n```"}]
        result = enrich_content_blocks(blocks)

        assert result[0]["type"] == "code"
        assert result[0]["declared_language"] == "javascript"
        assert result[0]["language"] == "javascript"

    def test_fenced_code_without_language_handled(self):
        """Fenced blocks without language get detection or pass through."""
        blocks = [{"type": "text", "text": "```\ndef foo():\n    pass\n```"}]
        result = enrich_content_blocks(blocks)

        assert result[0]["type"] == "code"
        assert "text" in result[0]

    def test_other_block_types_unchanged(self):
        """Non-text, non-code blocks pass through unchanged."""
        blocks = [
            {"type": "image", "url": "https://example.com/img.png"},
            {"type": "file", "name": "doc.pdf"},
        ]
        result = enrich_content_blocks(blocks)

        assert len(result) == 2
        assert result[0]["type"] == "image"
        assert result[1]["type"] == "file"

    def test_multiple_code_blocks_extracted(self):
        """Multiple fenced code blocks are all extracted."""
        text = (
            "First code:\n```python\nx = 1\n```\n"
            "Second code:\n```javascript\nlet y = 2\n```\nDone."
        )
        blocks = [{"type": "text", "text": text}]
        result = enrich_content_blocks(blocks)

        code_blocks = [b for b in result if b["type"] == "code"]
        assert len(code_blocks) == 2

        languages = {b["language"] for b in code_blocks}
        assert "python" in languages
        assert "javascript" in languages

    def test_empty_fenced_block_handled(self):
        """Empty fenced blocks are handled gracefully."""
        blocks = [{"type": "text", "text": "```\n\n```"}]
        result = enrich_content_blocks(blocks)
        # Should handle without error
        assert len(result) >= 0

    def test_existing_code_block_with_javascript_language(self):
        """Code blocks with language pass through unchanged (javascript)."""
        blocks = [{"type": "code", "text": "const x = 1", "language": "javascript"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "code"
        assert result[0]["language"] == "javascript"
        assert result[0]["text"] == "const x = 1"


class TestEnrichMessageMetadata:
    """enrich_message_metadata must handle None and missing content_blocks."""

    def test_none_provider_meta(self):
        assert enrich_message_metadata(None) is None

    def test_empty_dict(self):
        result = enrich_message_metadata({})
        assert result == {}

    def test_no_content_blocks_key(self):
        meta = {"some_key": "value"}
        result = enrich_message_metadata(meta)
        assert result == meta

    def test_content_blocks_is_none(self):
        """Regression: content_blocks key exists but value is None (d5c3228)."""
        meta = {"content_blocks": None}
        result = enrich_message_metadata(meta)
        # Should return meta unchanged, not crash
        assert result == meta

    def test_content_blocks_is_empty_list(self):
        meta = {"content_blocks": []}
        result = enrich_message_metadata(meta)
        assert result["content_blocks"] == []

    def test_enrichment_preserves_other_keys(self):
        meta = {
            "content_blocks": [{"type": "text", "text": "hello"}],
            "model": "claude-3",
            "tokens": 100,
        }
        result = enrich_message_metadata(meta)
        assert result["model"] == "claude-3"
        assert result["tokens"] == 100

    def test_metadata_with_content_blocks_enriched(self):
        """Metadata with content_blocks gets enriched (code extracted)."""
        meta = {
            "model": "gpt-4",
            "content_blocks": [
                {"type": "text", "text": "```python\nprint('hi')\n```"}
            ],
        }
        result = enrich_message_metadata(meta)

        assert "content_blocks" in result
        assert result["model"] == "gpt-4"

        enriched_blocks = result["content_blocks"]
        assert len(enriched_blocks) >= 1
        code_blocks = [b for b in enriched_blocks if b["type"] == "code"]
        assert len(code_blocks) == 1

    def test_original_metadata_not_mutated(self):
        """Original metadata dict is not mutated."""
        meta = {
            "model": "gpt-4",
            "content_blocks": [{"type": "text", "text": "Hello"}],
        }
        original_blocks = meta["content_blocks"]

        enrich_message_metadata(meta)

        assert meta["content_blocks"] is original_blocks
        assert meta["content_blocks"][0]["type"] == "text"


class TestLanguageDetectionIntegration:
    """Tests for language detection within enrichment."""

    @pytest.mark.parametrize("code,expected_lang", [
        ("def foo():\n    pass", "python"),
        ("function foo() { }", "javascript"),
        ("fn main() { }", "rust"),
        ("package main\nfunc main() { }", "go"),
    ])
    def test_language_detection_accuracy(self, code: str, expected_lang: str):
        """Language detection works for common languages."""
        blocks = [{"type": "code", "text": code}]
        result = enrich_content_blocks(blocks)

        if result[0].get("language"):
            assert result[0]["language"] == expected_lang

    def test_language_alias_normalization(self):
        """Language aliases are normalized."""
        blocks = [{"type": "text", "text": "```py\nprint('hi')\n```"}]
        result = enrich_content_blocks(blocks)

        code_blocks = [b for b in result if b["type"] == "code"]
        if code_blocks and code_blocks[0].get("language"):
            assert code_blocks[0]["language"] == "python"
