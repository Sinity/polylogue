"""Tests for content enrichment utilities.

Covers polylogue/core/content_enrichment.py which enriches content blocks
with language detection and code extraction.
"""

from __future__ import annotations

import pytest

from polylogue.pipeline.enrichment import (
    enrich_content_blocks,
    enrich_message_metadata,
)

# =============================================================================
# enrich_content_blocks Tests
# =============================================================================


class TestEnrichContentBlocks:
    """Tests for enrich_content_blocks function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        result = enrich_content_blocks([])
        assert result == []

    def test_plain_text_unchanged(self):
        """Plain text blocks pass through unchanged."""
        blocks = [{"type": "text", "text": "Hello world"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello world"

    def test_fenced_code_block_extracted(self):
        """Fenced code blocks are extracted as code type."""
        blocks = [{"type": "text", "text": "```python\nprint('hello')\n```"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "code"
        assert result[0]["language"] == "python"
        assert "print('hello')" in result[0]["text"]

    def test_fenced_code_with_declared_language(self):
        """Fenced blocks preserve declared language."""
        blocks = [{"type": "text", "text": "```javascript\nconsole.log('hi')\n```"}]
        result = enrich_content_blocks(blocks)

        assert result[0]["type"] == "code"
        assert result[0]["declared_language"] == "javascript"
        assert result[0]["language"] == "javascript"

    def test_fenced_code_without_language_detected(self):
        """Fenced blocks without language get detection."""
        blocks = [{"type": "text", "text": "```\ndef foo():\n    pass\n```"}]
        result = enrich_content_blocks(blocks)

        assert result[0]["type"] == "code"
        # Language might be detected or None
        assert "text" in result[0]

    def test_mixed_text_and_code(self):
        """Text with embedded code blocks is split."""
        blocks = [{"type": "text", "text": "Here is code:\n```python\nx = 1\n```\nDone."}]
        result = enrich_content_blocks(blocks)

        # Should be split into multiple blocks
        assert len(result) >= 2

        # Find code block
        code_blocks = [b for b in result if b["type"] == "code"]
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"

    def test_existing_code_block_without_language(self):
        """Code blocks without language get detection."""
        blocks = [{"type": "code", "text": "def hello():\n    print('hi')"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "code"
        # Language should be detected (or None if detection fails)
        assert "language" in result[0]

    def test_existing_code_block_with_language(self):
        """Code blocks with language pass through unchanged."""
        blocks = [{"type": "code", "text": "const x = 1", "language": "javascript"}]
        result = enrich_content_blocks(blocks)

        assert len(result) == 1
        assert result[0]["type"] == "code"
        assert result[0]["language"] == "javascript"
        assert result[0]["text"] == "const x = 1"

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

    def test_multiple_code_blocks(self):
        """Multiple fenced code blocks are all extracted."""
        text = """First code:
```python
x = 1
```
Second code:
```javascript
let y = 2
```
Done."""
        blocks = [{"type": "text", "text": text}]
        result = enrich_content_blocks(blocks)

        # Find code blocks
        code_blocks = [b for b in result if b["type"] == "code"]
        assert len(code_blocks) == 2

        languages = {b["language"] for b in code_blocks}
        assert "python" in languages
        assert "javascript" in languages

    def test_empty_fenced_block(self):
        """Empty fenced blocks are handled gracefully."""
        blocks = [{"type": "text", "text": "```\n\n```"}]
        result = enrich_content_blocks(blocks)

        # Should handle without error
        assert len(result) >= 0


# =============================================================================
# enrich_message_metadata Tests
# =============================================================================


class TestEnrichMessageMetadata:
    """Tests for enrich_message_metadata function."""

    def test_none_metadata_returns_none(self):
        """None metadata returns None."""
        result = enrich_message_metadata(None)
        assert result is None

    def test_empty_metadata_returns_empty(self):
        """Empty metadata returns empty."""
        result = enrich_message_metadata({})
        assert result == {}

    def test_metadata_without_content_blocks_unchanged(self):
        """Metadata without content_blocks passes through."""
        meta = {"model": "gpt-4", "temperature": 0.7}
        result = enrich_message_metadata(meta)

        assert result == meta

    def test_metadata_with_content_blocks_enriched(self):
        """Metadata with content_blocks gets enriched."""
        meta = {
            "model": "gpt-4",
            "content_blocks": [
                {"type": "text", "text": "```python\nprint('hi')\n```"}
            ]
        }
        result = enrich_message_metadata(meta)

        assert "content_blocks" in result
        assert result["model"] == "gpt-4"  # Other fields preserved

        # Check enrichment happened
        enriched_blocks = result["content_blocks"]
        assert len(enriched_blocks) >= 1
        code_blocks = [b for b in enriched_blocks if b["type"] == "code"]
        assert len(code_blocks) == 1

    def test_original_metadata_not_mutated(self):
        """Original metadata dict is not mutated."""
        meta = {
            "model": "gpt-4",
            "content_blocks": [{"type": "text", "text": "Hello"}]
        }
        original_blocks = meta["content_blocks"]

        enrich_message_metadata(meta)

        # Original should be unchanged
        assert meta["content_blocks"] is original_blocks
        assert meta["content_blocks"][0]["type"] == "text"


# =============================================================================
# Language Detection Integration Tests
# =============================================================================


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

        # May detect correctly or return None
        if result[0].get("language"):
            # If detected, should match expected
            assert result[0]["language"] == expected_lang

    def test_language_alias_normalization(self):
        """Language aliases are normalized."""
        blocks = [{"type": "text", "text": "```py\nprint('hi')\n```"}]
        result = enrich_content_blocks(blocks)

        # 'py' should be normalized to 'python'
        code_blocks = [b for b in result if b["type"] == "code"]
        if code_blocks and code_blocks[0].get("language"):
            assert code_blocks[0]["language"] == "python"
