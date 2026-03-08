"""Robustness tests for raw-file acquisition edge cases.

Covers: BOM-prefixed files, Windows CRLF line endings, and
latin-1 encoded exports that fall through to the utf-8 error-ignore path.

All of these exist in real-world provider exports. The _decode_json_bytes
function in source.py probes a tuple of encodings (_ENCODING_GUESSES) before
falling back to utf-8 with errors="ignore", so these tests verify the chain
works end-to-end without crashing.
"""

from __future__ import annotations

import json

import pytest

from polylogue.sources.source import _decode_json_bytes


# ---------------------------------------------------------------------------
# BOM handling
# ---------------------------------------------------------------------------

def test_utf8_bom_prefix_decoded() -> None:
    """UTF-8 BOM (EF BB BF) prefix is stripped cleanly by the utf-8-sig probe."""
    payload = {"role": "user", "content": "hello"}
    bom_bytes = b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8")
    result = _decode_json_bytes(bom_bytes)
    assert result is not None, "BOM-prefixed bytes returned None"
    parsed = json.loads(result)
    assert parsed == payload


def test_utf16_bom_prefix_decoded() -> None:
    """UTF-16 BOM (FF FE or FE FF) prefix is handled by the utf-16 probe."""
    payload = {"role": "assistant", "content": "hi"}
    utf16_bytes = json.dumps(payload).encode("utf-16")  # includes BOM automatically
    result = _decode_json_bytes(utf16_bytes)
    assert result is not None, "UTF-16 BOM bytes returned None"
    # The decoded string must be valid JSON
    parsed = json.loads(result)
    assert parsed["role"] == "assistant"


# ---------------------------------------------------------------------------
# CRLF / Windows line endings
# ---------------------------------------------------------------------------

def test_json_with_crlf_parsed() -> None:
    """CRLF line endings inside JSON string values do not break decoding."""
    # JSON strings can contain \r\n — the JSON spec is fine with it
    payload = {"note": "line1\r\nline2"}
    raw = json.dumps(payload).encode("utf-8")
    result = _decode_json_bytes(raw)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["note"] == "line1\r\nline2"


def test_jsonl_crlf_line_separator() -> None:
    """CRLF between JSONL lines (\r\n) decodes without stripping content."""
    line1 = json.dumps({"idx": 1})
    line2 = json.dumps({"idx": 2})
    raw = (line1 + "\r\n" + line2).encode("utf-8")
    result = _decode_json_bytes(raw)
    assert result is not None
    # Both records should be parseable from the result
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["idx"] == 1
    assert json.loads(lines[1])["idx"] == 2


# ---------------------------------------------------------------------------
# Encoding fallback
# ---------------------------------------------------------------------------

def test_latin1_file_does_not_crash() -> None:
    """A latin-1 byte sequence falls through to utf-8/ignore and returns a string."""
    # 0xe9 = 'é' in latin-1, but invalid as a standalone byte in UTF-8
    raw = b'{"note": "caf\xe9"}'
    result = _decode_json_bytes(raw)
    # Must not crash — invalid bytes are silently dropped by the ignore path
    assert result is not None
    # The returned string must be non-empty
    assert len(result.strip()) > 0


def test_all_ascii_roundtrips_exactly() -> None:
    """Plain ASCII JSON (no BOM, no special encoding) roundtrips without loss."""
    payload = {"provider": "chatgpt", "count": 42, "active": True}
    raw = json.dumps(payload).encode("ascii")
    result = _decode_json_bytes(raw)
    assert result is not None
    assert json.loads(result) == payload


def test_null_bytes_stripped() -> None:
    """Embedded NUL bytes (\\x00) are stripped by the cleaner step."""
    payload = '{"key": "value"}'
    raw_with_nulls = (payload[:5] + "\x00\x00" + payload[5:]).encode("utf-8")
    result = _decode_json_bytes(raw_with_nulls)
    assert result is not None
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_empty_bytes_returns_none() -> None:
    """Empty byte input returns None rather than crashing."""
    result = _decode_json_bytes(b"")
    assert result is None
