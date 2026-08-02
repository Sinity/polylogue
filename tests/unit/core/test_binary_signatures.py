"""Tests for the shared magic-byte binary-format detector.

polylogue-hbtj2: this is the single, shared detector every acquisition/
detection call site consults to refuse non-conversational binary payloads
(SQLite being the concrete miscapture the audit found) before any session
parser sees the bytes. See ``core.binary_signatures`` module docstring.
"""

from __future__ import annotations

from polylogue.core.binary_signatures import (
    SQLITE_MAGIC_HEADER,
    detect_binary_signature,
    looks_like_sqlite_bytes,
    looks_like_zip_bytes,
)


def test_detect_binary_signature_recognizes_sqlite() -> None:
    payload = SQLITE_MAGIC_HEADER + b"\x00" * 100
    signature = detect_binary_signature(payload)
    assert signature is not None
    assert signature.name == "sqlite"


def test_detect_binary_signature_recognizes_png() -> None:
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    signature = detect_binary_signature(payload)
    assert signature is not None
    assert signature.name == "png"


def test_detect_binary_signature_returns_none_for_json() -> None:
    payload = b'{"sessionId": "abc"}'
    assert detect_binary_signature(payload) is None


def test_detect_binary_signature_returns_none_for_empty() -> None:
    assert detect_binary_signature(b"") is None


def test_looks_like_sqlite_bytes() -> None:
    assert looks_like_sqlite_bytes(SQLITE_MAGIC_HEADER + b"rest")
    assert not looks_like_sqlite_bytes(b"not sqlite")


def test_looks_like_zip_bytes_is_reporting_only_not_in_binary_signatures() -> None:
    """ZIP is a legitimate session-source shape (GDPR export bundles), so it
    must never be a member of the auto-refusal registry, only reachable via
    the dedicated reporting helper."""
    zip_bytes = b"PK\x03\x04" + b"\x00" * 20
    assert looks_like_zip_bytes(zip_bytes)
    assert detect_binary_signature(zip_bytes) is None
