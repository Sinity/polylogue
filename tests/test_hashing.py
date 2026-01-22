"""Comprehensive tests for polylogue.core.hashing module."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from polylogue.core.hashing import (
    hash_file,
    hash_payload,
    hash_text,
    hash_text_short,
)


class TestHashText:
    """Tests for hash_text function."""

    def test_hash_text_returns_64_chars(self):
        """SHA-256 hex digest is always 64 characters."""
        result = hash_text("hello world")
        assert len(result) == 64

    def test_hash_text_deterministic(self):
        """Same input produces same output."""
        text = "deterministic test"
        hash1 = hash_text(text)
        hash2 = hash_text(text)
        assert hash1 == hash2

    def test_hash_text_different_inputs(self):
        """Different inputs produce different hashes."""
        hash1 = hash_text("input one")
        hash2 = hash_text("input two")
        assert hash1 != hash2

    def test_hash_text_empty_string(self):
        """Handles empty string correctly."""
        result = hash_text("")
        assert len(result) == 64
        # Empty string has known SHA-256 hash
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_hash_text_unicode(self):
        """Handles emoji and CJK characters."""
        unicode_text = "Hello 世界 🌍 emoji"
        result = hash_text(unicode_text)
        assert len(result) == 64
        # Verify deterministic
        assert result == hash_text(unicode_text)

    def test_hash_text_newlines(self):
        """Handles multiline text."""
        multiline = "line one\nline two\nline three"
        result = hash_text(multiline)
        assert len(result) == 64
        # Different from single line version
        assert result != hash_text("line oneline twoline three")


class TestHashTextShort:
    """Tests for hash_text_short function."""

    def test_hash_text_short_default_length(self):
        """Returns 16 characters by default."""
        result = hash_text_short("test")
        assert len(result) == 16

    def test_hash_text_short_custom_length(self):
        """Respects custom length parameter."""
        text = "custom length test"
        assert len(hash_text_short(text, length=8)) == 8
        assert len(hash_text_short(text, length=32)) == 32
        assert len(hash_text_short(text, length=64)) == 64

    def test_hash_text_short_deterministic(self):
        """Same input produces same truncated output."""
        text = "deterministic short"
        hash1 = hash_text_short(text)
        hash2 = hash_text_short(text)
        assert hash1 == hash2

    def test_hash_text_short_is_prefix(self):
        """Short hash is prefix of full hash."""
        text = "prefix test"
        full = hash_text(text)
        short = hash_text_short(text, length=10)
        assert full.startswith(short)

    def test_hash_text_short_different_inputs(self):
        """Different inputs produce different short hashes."""
        hash1 = hash_text_short("input A")
        hash2 = hash_text_short("input B")
        assert hash1 != hash2


class TestHashPayload:
    """Tests for hash_payload function."""

    def test_hash_payload_dict(self):
        """Hashes dict correctly."""
        payload = {"name": "test", "value": 42}
        result = hash_payload(payload)
        assert len(result) == 64

    def test_hash_payload_list(self):
        """Hashes list correctly."""
        payload = [1, 2, 3, "four", "five"]
        result = hash_payload(payload)
        assert len(result) == 64

    def test_hash_payload_nested(self):
        """Handles nested structures."""
        payload = {
            "outer": {
                "inner": [1, 2, {"deep": "value"}],
                "another": "field",
            },
            "list": [{"a": 1}, {"b": 2}],
        }
        result = hash_payload(payload)
        assert len(result) == 64

    def test_hash_payload_key_order_independent(self):
        """Dict key order doesn't affect hash (sort_keys=True)."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "a": 1, "b": 2}
        payload3 = {"b": 2, "c": 3, "a": 1}
        hash1 = hash_payload(payload1)
        hash2 = hash_payload(payload2)
        hash3 = hash_payload(payload3)
        assert hash1 == hash2 == hash3

    def test_hash_payload_deterministic(self):
        """Same object produces same hash."""
        payload = {"complex": [1, 2, {"nested": True}], "value": 123}
        hash1 = hash_payload(payload)
        hash2 = hash_payload(payload)
        assert hash1 == hash2

    def test_hash_payload_different_values(self):
        """Different values produce different hashes."""
        hash1 = hash_payload({"key": "value1"})
        hash2 = hash_payload({"key": "value2"})
        assert hash1 != hash2

    def test_hash_payload_primitives(self):
        """Handles primitive types."""
        assert len(hash_payload(42)) == 64
        assert len(hash_payload("string")) == 64
        assert len(hash_payload(True)) == 64
        assert len(hash_payload(None)) == 64
        assert len(hash_payload(3.14159)) == 64


class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_file_basic(self, tmp_path: Path):
        """Hashes file contents correctly."""
        file = tmp_path / "test.txt"
        file.write_text("Hello, world!")
        result = hash_file(file)
        assert len(result) == 64

    def test_hash_file_deterministic(self, tmp_path: Path):
        """Same file produces same hash."""
        file = tmp_path / "test.txt"
        file.write_text("deterministic content")
        hash1 = hash_file(file)
        hash2 = hash_file(file)
        assert hash1 == hash2

    def test_hash_file_empty(self, tmp_path: Path):
        """Handles empty file."""
        file = tmp_path / "empty.txt"
        file.write_text("")
        result = hash_file(file)
        assert len(result) == 64
        # Empty file has same hash as empty string
        assert result == hash_text("")

    def test_hash_file_binary_content(self, tmp_path: Path):
        """Handles binary data."""
        file = tmp_path / "binary.dat"
        file.write_bytes(bytes(range(256)))
        result = hash_file(file)
        assert len(result) == 64

    def test_hash_file_large_file(self, tmp_path: Path):
        """Handles files larger than chunk size (1MB)."""
        file = tmp_path / "large.dat"
        # Create 2.5 MB file (more than 2 chunks)
        chunk_size = 1024 * 1024  # 1MB
        data = b"x" * (chunk_size * 2 + chunk_size // 2)
        file.write_bytes(data)
        result = hash_file(file)
        assert len(result) == 64

    def test_hash_file_different_contents(self, tmp_path: Path):
        """Different file contents produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content one")
        file2.write_text("content two")
        hash1 = hash_file(file1)
        hash2 = hash_file(file2)
        assert hash1 != hash2

    def test_hash_file_unicode_content(self, tmp_path: Path):
        """Handles Unicode text in files."""
        file = tmp_path / "unicode.txt"
        file.write_text("Hello 世界 🌍 emoji", encoding="utf-8")
        result = hash_file(file)
        assert len(result) == 64

    def test_hash_file_newlines(self, tmp_path: Path):
        """Preserves newline differences."""
        file_lf = tmp_path / "lf.txt"
        file_crlf = tmp_path / "crlf.txt"
        file_lf.write_bytes(b"line1\nline2\n")
        file_crlf.write_bytes(b"line1\r\nline2\r\n")
        hash_lf = hash_file(file_lf)
        hash_crlf = hash_file(file_crlf)
        assert hash_lf != hash_crlf


class TestCrossFunctionConsistency:
    """Tests for consistency between different hash functions."""

    def test_hash_text_vs_hash_payload_string(self):
        """hash_payload on string differs from hash_text (JSON encoding)."""
        text = "test string"
        # hash_payload wraps in JSON quotes
        assert hash_payload(text) == hash_text('"test string"')
        assert hash_payload(text) != hash_text(text)

    def test_hash_text_vs_hash_file(self, tmp_path: Path):
        """hash_file matches hash_text for text content."""
        text = "file content test"
        file = tmp_path / "test.txt"
        file.write_text(text, encoding="utf-8")
        # Should match since file uses UTF-8 encoding
        assert hash_file(file) == hash_text(text)

    def test_hash_text_short_consistency(self):
        """hash_text_short is consistent with hash_text prefix."""
        text = "consistency test"
        for length in [1, 8, 16, 32, 48, 64]:
            short = hash_text_short(text, length=length)
            full = hash_text(text)
            assert short == full[:length]


class TestUnicodeNormalization:
    """Tests for Unicode normalization in hashing.

    Issue: core/hashing.py:14-16 doesn't normalize Unicode, so
    visually identical strings (NFC vs NFD) produce different hashes.
    """

    def test_nfc_nfd_produce_same_hash(self):
        """Visually identical Unicode strings MUST hash identically.

        This test SHOULD FAIL until Unicode normalization is added to hash_text().
        Fix: Apply unicodedata.normalize("NFC", text) before hashing.
        """
        # "café" in NFC (precomposed) vs NFD (decomposed)
        nfc = unicodedata.normalize("NFC", "café")  # é as single codepoint
        nfd = unicodedata.normalize("NFD", "café")  # e + combining acute

        # Verify they're visually identical but byte-different
        # Note: Python does NOT normalize on comparison, so nfc != nfd as strings
        assert nfc.encode("utf-8") != nfd.encode("utf-8")  # Bytes differ

        hash_nfc = hash_text(nfc)
        hash_nfd = hash_text(nfd)

        # MUST be equal for content-addressable hashing to work correctly
        assert hash_nfc == hash_nfd, f"Unicode normalization bug: NFC hash {hash_nfc[:16]}... != NFD hash {hash_nfd[:16]}..."

    def test_combining_characters_hash_same(self):
        """Precomposed and decomposed characters MUST hash identically.

        This test SHOULD FAIL until normalization is implemented.
        """
        # ñ as single char vs n + combining tilde
        precomposed = "\u00f1"  # ñ
        decomposed = "n\u0303"  # n + combining tilde

        hash1 = hash_text(precomposed)
        hash2 = hash_text(decomposed)

        assert hash1 == hash2, "Combining characters must hash same as precomposed"

    def test_emoji_with_modifiers(self):
        """Test emoji with skin tone modifiers hash consistently."""
        # Base emoji
        wave = "👋"
        # Same emoji with skin tone modifier
        wave_light = "👋🏻"

        hash_base = hash_text(wave)
        hash_modified = hash_text(wave_light)

        # These SHOULD be different (different characters)
        assert hash_base != hash_modified

    def test_zero_width_characters(self):
        """Test that zero-width characters affect hash."""
        normal = "hello"
        with_zwj = "hel\u200dlo"  # Zero-width joiner in middle

        hash_normal = hash_text(normal)
        hash_zwj = hash_text(with_zwj)

        # Zero-width chars DO affect hash (may or may not be desired)
        assert hash_normal != hash_zwj


@given(st.text())
def test_hash_text_unicode_normalization_invariant(text: str):
    """Hash MUST be invariant under Unicode normalization.

    This test SHOULD FAIL until normalization is added.
    """
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)

    hash_nfc = hash_text(nfc)
    hash_nfd = hash_text(nfd)

    # MUST be equal regardless of normalization form
    assert hash_nfc == hash_nfd, f"Normalization variant for {repr(text[:20])}..."
