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
