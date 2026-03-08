"""Tests for shared provider identity normalization."""

from __future__ import annotations

from polylogue.lib.provider_identity import (
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.types import Provider


def test_canonical_runtime_provider_aliases() -> None:
    assert canonical_runtime_provider("gpt") == "chatgpt"
    assert canonical_runtime_provider("openai") == "chatgpt"
    assert canonical_runtime_provider("claude-ai") == "claude"
    assert canonical_runtime_provider("anthropic") == "claude"
    assert canonical_runtime_provider("CLAUDE_CODE") == "claude-code"


def test_canonical_runtime_provider_preserves_unknown_when_requested() -> None:
    assert canonical_runtime_provider("my-inbox", preserve_unknown=True) == "my-inbox"
    assert canonical_runtime_provider("my-inbox", preserve_unknown=False) == "unknown"


def test_canonical_schema_provider_mapping() -> None:
    assert canonical_schema_provider("claude") == "claude-ai"
    assert canonical_schema_provider("claude-ai") == "claude-ai"
    assert canonical_schema_provider("openai") == "chatgpt"


def test_provider_enum_from_string_uses_shared_runtime_identity() -> None:
    assert Provider.from_string("claude-ai") is Provider.CLAUDE
    assert Provider.from_string("openai") is Provider.CHATGPT
    assert Provider.from_string("nonexistent-provider") is Provider.UNKNOWN


def test_build_raw_payload_envelope_normalizes_fallback_identity() -> None:
    raw = b'{"id":"x"}'  # not enough for detect_provider
    assert build_raw_payload_envelope(raw, source_path=None, fallback_provider="claude-ai").provider == "claude"
    assert build_raw_payload_envelope(raw, source_path=None, fallback_provider="my-inbox").provider == "my-inbox"
