"""Tests for hashing functions.

Consolidated from test_hashing.py.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib.hashing import hash_file, hash_payload, hash_text, hash_text_short


HASH_TEXT_CASES = [
    ("hello world", 64, "length is 64 chars"),
    ("deterministic test", "deterministic", "same input → same output"),
    ("input one", "differs from 'input two'", "different inputs"),
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty string known hash"),
    ("Hello 世界 🌍 emoji", 64, "unicode emoji CJK"),
    ("line one\nline two\nline three", "multiline", "newlines preserved"),
]


@pytest.mark.parametrize("text,expected,desc", HASH_TEXT_CASES)
def test_hash_text_comprehensive(text, expected, desc):
    """Comprehensive hash_text test."""
    result = hash_text(text)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_text(text), f"Failed {desc}"
    elif expected.startswith("differs"):
        other_text = "input two"
        assert result != hash_text(other_text), f"Failed {desc}"
    elif expected == "multiline":
        concatenated = "line oneline twoline three"
        assert result != hash_text(concatenated), f"Failed {desc}"
    elif expected == "unicode":
        assert result == hash_text(text), f"Failed {desc}"
    else:
        assert result == expected, f"Failed {desc}"


HASH_TEXT_SHORT_CASES = [
    ("test", None, 16, "default length"),
    ("custom length test", 8, 8, "custom length 8"),
    ("custom length test", 32, 32, "custom length 32"),
    ("custom length test", 64, 64, "custom length 64"),
    ("deterministic short", None, "deterministic", "same input → same output"),
    ("prefix test", 10, "prefix", "short is prefix of full"),
    ("input A", None, "differs from 'input B'", "different inputs"),
]


@pytest.mark.parametrize("text,length,expected,desc", HASH_TEXT_SHORT_CASES)
def test_hash_text_short_comprehensive(text, length, expected, desc):
    """Comprehensive hash_text_short test."""
    result = hash_text_short(text, length=length) if length is not None else hash_text_short(text)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        if length is not None:
            assert result == hash_text_short(text, length=length), f"Failed {desc}"
        else:
            assert result == hash_text_short(text), f"Failed {desc}"
    elif expected == "prefix":
        full = hash_text(text)
        assert full.startswith(result), f"Failed {desc}"
    elif expected.startswith("differs"):
        other_text = "input B"
        if length is not None:
            assert result != hash_text_short(other_text, length=length), f"Failed {desc}"
        else:
            assert result != hash_text_short(other_text), f"Failed {desc}"


HASH_PAYLOAD_CASES = [
    ({"name": "test", "value": 42}, 64, "dict"),
    ([1, 2, 3, "four", "five"], 64, "list"),
    ({"outer": {"inner": [1, 2, {"deep": "value"}], "another": "field"}, "list": [{"a": 1}, {"b": 2}]}, 64, "nested"),
    ({"a": 1, "b": 2, "c": 3}, "key_order_independent", "key order independence"),
    ({"complex": [1, 2, {"nested": True}], "value": 123}, "deterministic", "determinism"),
    ({"key": "value1"}, "differs from value2", "different values"),
    (42, 64, "int primitive"),
    ("string", 64, "string primitive"),
    (True, 64, "bool primitive"),
    (None, 64, "None primitive"),
    (3.14159, 64, "float primitive"),
]


@pytest.mark.parametrize("payload,expected,desc", HASH_PAYLOAD_CASES)
def test_hash_payload_comprehensive(payload, expected, desc):
    """Comprehensive hash_payload test."""
    result = hash_payload(payload)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "key_order_independent":
        variant1 = {"c": 3, "a": 1, "b": 2}
        variant2 = {"b": 2, "c": 3, "a": 1}
        hash1 = hash_payload(payload)
        hash2 = hash_payload(variant1)
        hash3 = hash_payload(variant2)
        assert hash1 == hash2 == hash3, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_payload(payload), f"Failed {desc}"
    elif expected.startswith("differs"):
        other = {"key": "value2"}
        assert result != hash_payload(other), f"Failed {desc}"


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
    """Comprehensive hash_file test."""
    file = tmp_path / "test.dat"

    if expected == "newlines":
        file_lf = tmp_path / "lf.txt"
        file_crlf = tmp_path / "crlf.txt"
        file_lf.write_bytes(b"line1\nline2\n")
        file_crlf.write_bytes(b"line1\r\nline2\r\n")
        hash_lf = hash_file(file_lf)
        hash_crlf = hash_file(file_crlf)
        assert hash_lf != hash_crlf, f"Failed {desc}"
        return

    if isinstance(content, bytes):
        file.write_bytes(content)
    else:
        file.write_text(content, encoding=encoding or "utf-8")

    result = hash_file(file)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_file(file), f"Failed {desc}"
    elif expected == "empty":
        assert result == hash_text(""), f"Failed {desc}"
    elif expected.startswith("differs"):
        file2 = tmp_path / "test2.dat"
        file2.write_text("content two")
        hash2 = hash_file(file2)
        assert result != hash2, f"Failed {desc}"


CONSISTENCY_CASES = [
    ("hash_payload on string", "test string", "payload_vs_text", "payload wraps in quotes"),
    ("hash_file matches hash_text", "file content test", "file_vs_text", "file vs text consistency"),
    ("consistency test", None, "short_prefix", "short is prefix of full"),
]


@pytest.mark.parametrize("label,text,test_type,desc", CONSISTENCY_CASES)
def test_cross_function_consistency(tmp_path: Path, label, text, test_type, desc):
    """Cross-function consistency tests."""
    if test_type == "payload_vs_text":
        assert hash_payload(text) == hash_text(f'"{text}"'), f"Failed {desc}"
        assert hash_payload(text) != hash_text(text), f"Failed {desc}"

    elif test_type == "file_vs_text":
        file = tmp_path / "test.txt"
        file.write_text(text, encoding="utf-8")
        assert hash_file(file) == hash_text(text), f"Failed {desc}"

    elif test_type == "short_prefix":
        test_text = text if text is not None else label
        for length in [1, 8, 16, 32, 48, 64]:
            short = hash_text_short(test_text, length=length)
            full = hash_text(test_text)
            assert short == full[:length], f"Failed {desc} at length {length}"


UNICODE_NORMALIZATION_CASES = [
    ("café", "nfc_nfd", "NFC vs NFD same hash"),
    ("\u00f1", "combining", "precomposed vs decomposed"),
    ("👋", "emoji_modifiers", "emoji with modifiers"),
    ("hello", "zero_width", "zero-width characters"),
]


@pytest.mark.parametrize("text,test_type,desc", UNICODE_NORMALIZATION_CASES)
def test_unicode_normalization_comprehensive(text, test_type, desc):
    """Unicode normalization tests."""
    if test_type == "nfc_nfd":
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)

        assert nfc.encode("utf-8") != nfd.encode("utf-8")

        hash_nfc = hash_text(nfc)
        hash_nfd = hash_text(nfd)

        assert hash_nfc == hash_nfd, f"Unicode normalization bug: {desc}"

    elif test_type == "combining":
        precomposed = text
        decomposed = "n\u0303"

        hash1 = hash_text(precomposed)
        hash2 = hash_text(decomposed)

        assert hash1 == hash2, f"Combining characters must hash same: {desc}"

    elif test_type == "emoji_modifiers":
        wave = text
        wave_light = "👋🏻"

        hash_base = hash_text(wave)
        hash_modified = hash_text(wave_light)

        assert hash_base != hash_modified, f"Failed {desc}"

    elif test_type == "zero_width":
        normal = text
        with_zwj = "hel\u200dlo"

        hash_normal = hash_text(normal)
        hash_zwj = hash_text(with_zwj)

        assert hash_normal != hash_zwj, f"Failed {desc}"


@given(st.text())
def test_hash_text_unicode_normalization_invariant(text: str):
    """Hash MUST be invariant under Unicode normalization."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)

    hash_nfc = hash_text(nfc)
    hash_nfd = hash_text(nfd)

    assert hash_nfc == hash_nfd, f"Normalization variant for {repr(text[:20])}..."
