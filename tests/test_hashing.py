"""Consolidated hashing tests using parametrization.

CONSOLIDATION: 34 tests → 8 parametrized test functions with 34+ test cases.

Original: Separate test classes per function (hash_text, hash_text_short, hash_payload, hash_file)
New: Parametrized tests covering all hash functions and edge cases
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.core.hashing import (
    hash_file,
    hash_payload,
    hash_text,
    hash_text_short,
)


# =============================================================================
# HASH_TEXT - PARAMETRIZED (1 test replacing 6)
# =============================================================================


HASH_TEXT_CASES = [
    # Basic properties
    ("hello world", 64, "length is 64 chars"),
    ("deterministic test", "deterministic", "same input → same output"),
    ("input one", "differs from 'input two'", "different inputs"),

    # Empty string has known SHA-256 hash
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty string known hash"),

    # Unicode handling
    ("Hello 世界 🌍 emoji", 64, "unicode emoji CJK"),

    # Newlines preserved
    ("line one\nline two\nline three", "multiline", "newlines preserved"),
]


@pytest.mark.parametrize("text,expected,desc", HASH_TEXT_CASES)
def test_hash_text_comprehensive(text, expected, desc):
    """Comprehensive hash_text test.

    Replaces 6 individual tests from TestHashText.
    """
    result = hash_text(text)

    if isinstance(expected, int):
        # Length assertion
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        # Determinism check
        assert result == hash_text(text), f"Failed {desc}"
    elif expected.startswith("differs"):
        # Different inputs
        other_text = "input two"
        assert result != hash_text(other_text), f"Failed {desc}"
    elif expected == "multiline":
        # Multiline differs from concatenated
        concatenated = "line oneline twoline three"
        assert result != hash_text(concatenated), f"Failed {desc}"
    elif expected == "unicode":
        # Unicode determinism
        assert result == hash_text(text), f"Failed {desc}"
    else:
        # Exact hash match (empty string case)
        assert result == expected, f"Failed {desc}"


# =============================================================================
# HASH_TEXT_SHORT - PARAMETRIZED (1 test replacing 5)
# =============================================================================


HASH_TEXT_SHORT_CASES = [
    # Default length
    ("test", None, 16, "default length"),

    # Custom lengths
    ("custom length test", 8, 8, "custom length 8"),
    ("custom length test", 32, 32, "custom length 32"),
    ("custom length test", 64, 64, "custom length 64"),

    # Determinism
    ("deterministic short", None, "deterministic", "same input → same output"),

    # Prefix property
    ("prefix test", 10, "prefix", "short is prefix of full"),

    # Different inputs
    ("input A", None, "differs from 'input B'", "different inputs"),
]


@pytest.mark.parametrize("text,length,expected,desc", HASH_TEXT_SHORT_CASES)
def test_hash_text_short_comprehensive(text, length, expected, desc):
    """Comprehensive hash_text_short test.

    Replaces 5 individual tests from TestHashTextShort.
    """
    if length is not None:
        result = hash_text_short(text, length=length)
    else:
        result = hash_text_short(text)

    if isinstance(expected, int):
        # Length assertion
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        # Determinism check
        if length is not None:
            assert result == hash_text_short(text, length=length), f"Failed {desc}"
        else:
            assert result == hash_text_short(text), f"Failed {desc}"
    elif expected == "prefix":
        # Prefix property
        full = hash_text(text)
        assert full.startswith(result), f"Failed {desc}"
    elif expected.startswith("differs"):
        # Different inputs
        other_text = "input B"
        if length is not None:
            assert result != hash_text_short(other_text, length=length), f"Failed {desc}"
        else:
            assert result != hash_text_short(other_text), f"Failed {desc}"


# =============================================================================
# HASH_PAYLOAD - PARAMETRIZED (1 test replacing 8)
# =============================================================================


HASH_PAYLOAD_CASES = [
    # Data structures
    ({"name": "test", "value": 42}, 64, "dict"),
    ([1, 2, 3, "four", "five"], 64, "list"),
    ({"outer": {"inner": [1, 2, {"deep": "value"}], "another": "field"}, "list": [{"a": 1}, {"b": 2}]}, 64, "nested"),

    # Key order independence
    ({"a": 1, "b": 2, "c": 3}, "key_order_independent", "key order independence"),

    # Determinism
    ({"complex": [1, 2, {"nested": True}], "value": 123}, "deterministic", "determinism"),

    # Different values
    ({"key": "value1"}, "differs from value2", "different values"),

    # Primitives
    (42, 64, "int primitive"),
    ("string", 64, "string primitive"),
    (True, 64, "bool primitive"),
    (None, 64, "None primitive"),
    (3.14159, 64, "float primitive"),
]


@pytest.mark.parametrize("payload,expected,desc", HASH_PAYLOAD_CASES)
def test_hash_payload_comprehensive(payload, expected, desc):
    """Comprehensive hash_payload test.

    Replaces 8 individual tests from TestHashPayload.
    """
    result = hash_payload(payload)

    if isinstance(expected, int):
        # Length assertion
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "key_order_independent":
        # Key order shouldn't matter
        variant1 = {"c": 3, "a": 1, "b": 2}
        variant2 = {"b": 2, "c": 3, "a": 1}
        hash1 = hash_payload(payload)
        hash2 = hash_payload(variant1)
        hash3 = hash_payload(variant2)
        assert hash1 == hash2 == hash3, f"Failed {desc}"
    elif expected == "deterministic":
        # Same object → same hash
        assert result == hash_payload(payload), f"Failed {desc}"
    elif expected.startswith("differs"):
        # Different values → different hashes
        other = {"key": "value2"}
        assert result != hash_payload(other), f"Failed {desc}"


# =============================================================================
# HASH_FILE - PARAMETRIZED (1 test replacing 8)
# =============================================================================


HASH_FILE_CASES = [
    ("Hello, world!", 64, None, "basic"),
    ("deterministic content", "deterministic", None, "deterministic"),
    ("", "empty", None, "empty file"),
    (bytes(range(256)), 64, "binary", "binary content"),
    (b"x" * (1024 * 1024 * 2 + 1024 * 512), 64, "binary", "large file"),
    ("content one", "differs from 'content two'", None, "different contents"),
    ("Hello 世界 🌍 emoji", 64, "utf-8", "unicode content"),
    (None, "newlines", "binary", "newline differences"),
]


@pytest.mark.parametrize("content,expected,encoding,desc", HASH_FILE_CASES)
def test_hash_file_comprehensive(tmp_path: Path, content, expected, encoding, desc):
    """Comprehensive hash_file test.

    Replaces 8 individual tests from TestHashFile.
    """
    file = tmp_path / "test.dat"

    # Special case: newline differences test
    if expected == "newlines":
        file_lf = tmp_path / "lf.txt"
        file_crlf = tmp_path / "crlf.txt"
        file_lf.write_bytes(b"line1\nline2\n")
        file_crlf.write_bytes(b"line1\r\nline2\r\n")
        hash_lf = hash_file(file_lf)
        hash_crlf = hash_file(file_crlf)
        assert hash_lf != hash_crlf, f"Failed {desc}"
        return

    # Write content
    if isinstance(content, bytes):
        file.write_bytes(content)
    else:
        file.write_text(content, encoding=encoding or "utf-8")

    result = hash_file(file)

    if isinstance(expected, int):
        # Length assertion
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        # Same file → same hash
        assert result == hash_file(file), f"Failed {desc}"
    elif expected == "empty":
        # Empty file matches empty string hash
        assert result == hash_text(""), f"Failed {desc}"
    elif expected.startswith("differs"):
        # Different contents → different hashes
        file2 = tmp_path / "test2.dat"
        file2.write_text("content two")
        hash2 = hash_file(file2)
        assert result != hash2, f"Failed {desc}"


# =============================================================================
# CROSS-FUNCTION CONSISTENCY - PARAMETRIZED (1 test replacing 3)
# =============================================================================


CONSISTENCY_CASES = [
    # hash_payload wraps strings in JSON quotes
    ("hash_payload on string", "test string", "payload_vs_text", "payload wraps in quotes"),

    # hash_file matches hash_text for text content
    ("hash_file matches hash_text", "file content test", "file_vs_text", "file vs text consistency"),

    # hash_text_short is prefix of hash_text
    ("consistency test", None, "short_prefix", "short is prefix of full"),
]


@pytest.mark.parametrize("label,text,test_type,desc", CONSISTENCY_CASES)
def test_cross_function_consistency(tmp_path: Path, label, text, test_type, desc):
    """Cross-function consistency tests.

    Replaces 3 individual tests from TestCrossFunctionConsistency.
    """
    if test_type == "payload_vs_text":
        # hash_payload wraps in JSON quotes
        assert hash_payload(text) == hash_text(f'"{text}"'), f"Failed {desc}"
        assert hash_payload(text) != hash_text(text), f"Failed {desc}"

    elif test_type == "file_vs_text":
        # hash_file matches hash_text for UTF-8 text
        file = tmp_path / "test.txt"
        file.write_text(text, encoding="utf-8")
        assert hash_file(file) == hash_text(text), f"Failed {desc}"

    elif test_type == "short_prefix":
        # hash_text_short is consistent prefix
        # Use label as text when text is None
        test_text = text if text is not None else label
        for length in [1, 8, 16, 32, 48, 64]:
            short = hash_text_short(test_text, length=length)
            full = hash_text(test_text)
            assert short == full[:length], f"Failed {desc} at length {length}"


# =============================================================================
# UNICODE NORMALIZATION - PARAMETRIZED (1 test + 1 property test replacing 4)
# =============================================================================


UNICODE_NORMALIZATION_CASES = [
    # NFC vs NFD (SHOULD FAIL until normalization added)
    ("café", "nfc_nfd", "NFC vs NFD same hash"),

    # Precomposed vs decomposed (SHOULD FAIL)
    ("\u00f1", "combining", "precomposed vs decomposed"),

    # Emoji modifiers (different characters → different hashes)
    ("👋", "emoji_modifiers", "emoji with modifiers"),

    # Zero-width characters (DO affect hash)
    ("hello", "zero_width", "zero-width characters"),
]


@pytest.mark.parametrize("text,test_type,desc", UNICODE_NORMALIZATION_CASES)
def test_unicode_normalization_comprehensive(text, test_type, desc):
    """Unicode normalization tests.

    Replaces 4 individual tests from TestUnicodeNormalization.
    Note: Some tests SHOULD FAIL until normalization is implemented.
    """
    if test_type == "nfc_nfd":
        # NFC vs NFD MUST hash same (currently fails)
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)

        # Verify bytes differ
        assert nfc.encode("utf-8") != nfd.encode("utf-8")

        hash_nfc = hash_text(nfc)
        hash_nfd = hash_text(nfd)

        # MUST be equal (this test SHOULD FAIL until normalization added)
        assert hash_nfc == hash_nfd, f"Unicode normalization bug: {desc}"

    elif test_type == "combining":
        # Precomposed vs decomposed MUST hash same
        precomposed = text  # ñ
        decomposed = "n\u0303"  # n + combining tilde

        hash1 = hash_text(precomposed)
        hash2 = hash_text(decomposed)

        # SHOULD FAIL until normalization implemented
        assert hash1 == hash2, f"Combining characters must hash same: {desc}"

    elif test_type == "emoji_modifiers":
        # Different emoji (base vs with modifier) → different hashes
        wave = text
        wave_light = "👋🏻"

        hash_base = hash_text(wave)
        hash_modified = hash_text(wave_light)

        # These SHOULD be different
        assert hash_base != hash_modified, f"Failed {desc}"

    elif test_type == "zero_width":
        # Zero-width characters DO affect hash
        normal = text
        with_zwj = "hel\u200dlo"  # Zero-width joiner

        hash_normal = hash_text(normal)
        hash_zwj = hash_text(with_zwj)

        assert hash_normal != hash_zwj, f"Failed {desc}"


# =============================================================================
# PROPERTY-BASED UNICODE NORMALIZATION (1 test - kept from original)
# =============================================================================


@given(st.text())
def test_hash_text_unicode_normalization_invariant(text: str):
    """Hash MUST be invariant under Unicode normalization.

    This test SHOULD FAIL until normalization is added to hash_text().
    Uses Hypothesis for property-based testing.
    """
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)

    hash_nfc = hash_text(nfc)
    hash_nfd = hash_text(nfd)

    # MUST be equal regardless of normalization form
    assert hash_nfc == hash_nfd, f"Normalization variant for {repr(text[:20])}..."
