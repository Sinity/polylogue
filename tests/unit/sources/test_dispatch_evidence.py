"""Tests for the evidence-tagged detection variants used by acquisition logging.

``detect_provider_evidence``/``detect_provider_from_raw_bytes_evidence`` are the
single source of truth for detection (``detect_provider``/
``_detect_provider_from_raw_bytes`` are thin wrappers that discard the evidence
label) -- these tests both exercise the evidence label content and guard
against the wrapper functions drifting from the evidence-returning
implementation.
"""

from __future__ import annotations

import json

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.dispatch import (
    detect_provider,
    detect_provider_evidence,
    detect_provider_from_raw_bytes_evidence,
)

_CLAUDE_CODE_RECORD = {"sessionId": "s1", "uuid": "u1", "type": "user", "cwd": "/tmp"}
_CHATGPT_LIKE_RECORD = {
    "mapping": {
        "node-1": {
            "id": "node-1",
            "message": {"id": "m1", "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["hi"]}},
            "parent": None,
            "children": [],
        }
    },
    "current_node": "node-1",
    "create_time": 1700000000.0,
    "conversation_id": "conv-1",
}
_CLAUDE_AI_RECORD = {"chat_messages": [{"sender": "human", "text": "hi"}]}


@pytest.mark.parametrize(
    ("payload", "expected_provider"),
    [
        ([_CLAUDE_CODE_RECORD], Provider.CLAUDE_CODE),
        (_CHATGPT_LIKE_RECORD, Provider.CHATGPT),
        (_CLAUDE_AI_RECORD, Provider.CLAUDE_AI),
        ({"unrelated": "shape"}, None),
        ([], None),
    ],
)
def test_detect_provider_evidence_matches_detect_provider(payload: object, expected_provider: Provider | None) -> None:
    """No drift: the wrapper's result must equal the evidence function's own result."""
    provider, evidence = detect_provider_evidence(payload)
    assert provider is expected_provider
    assert detect_provider(payload) is provider
    assert evidence


def test_detect_provider_evidence_names_the_deciding_rule_for_claude_code() -> None:
    _, evidence = detect_provider_evidence([_CLAUDE_CODE_RECORD])
    assert "claude.looks_like_code" in evidence


def test_detect_provider_evidence_reports_no_match_reason() -> None:
    provider, evidence = detect_provider_evidence({"unrelated": "shape"})
    assert provider is None
    assert "no detector matched" in evidence


def test_detect_provider_from_raw_bytes_evidence_matches_document_detection() -> None:
    raw_bytes = json.dumps(_CHATGPT_LIKE_RECORD).encode()

    provider, evidence = detect_provider_from_raw_bytes_evidence(raw_bytes, "export.json", Provider.UNKNOWN)

    assert provider is Provider.CHATGPT
    assert "chatgpt" in evidence.lower()


def test_detect_provider_from_raw_bytes_evidence_falls_back_with_reason_on_garbage() -> None:
    raw_bytes = b"not json at all \x00\x01\x02"

    provider, evidence = detect_provider_from_raw_bytes_evidence(
        raw_bytes,
        "mystery.dat",
        Provider.UNKNOWN,
        truncated_tail_ok=True,
    )

    assert provider is Provider.UNKNOWN
    assert "fallback_provider" in evidence


def test_detect_provider_from_raw_bytes_evidence_refuses_sqlite_before_json_decode() -> None:
    """polylogue-hbtj2: a SQLite-shaped raw byte stream must be positively refused
    here (the single shared raw-bytes detection chokepoint used by production
    per-file acquisition and the unclaimed-file sweep), by magic bytes -- not
    by falling through to an incidental JSON-decode failure."""
    raw_bytes = b"SQLite format 3\x00" + b"\x00" * 100

    provider, evidence = detect_provider_from_raw_bytes_evidence(
        raw_bytes,
        "state_5.sqlite",
        Provider.CODEX,
    )

    assert provider is Provider.CODEX
    assert "sqlite" in evidence.lower()
    assert "refused" in evidence.lower()
