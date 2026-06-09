"""Edge-case synthetic corpus extension tests.

Extends the synthetic generator to produce edge-case payloads and validates
each against its committed provider schema.

Ref #1736.
"""

from __future__ import annotations

import json

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.validator import validate_provider_export

# Providers with committed schemas that the synthetic generator supports.
_SYNTHETIC_PROVIDERS = ("chatgpt", "claude-code", "claude-ai", "codex", "gemini")

# Providers with JSONL wire format — records are individual JSON objects.
_JSONL_PROVIDERS = {"claude-code", "codex"}


# ── helpers ─────────────────────────────────────────────────────────────


def _generate_payload(provider: str, n_messages: int = 5, seed: int = 42) -> object:
    """Generate a raw synthetic payload for the given provider.

    For JSONL providers, returns a list of record dicts.
    For JSON providers, returns a dict.
    """
    spec = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=n_messages,
        messages_max=n_messages,
        seed=seed,
        origin="generated.test-edge-cases",
        tags=("synthetic", "test", "edge-cases"),
    )
    batch = SyntheticCorpus.generate_batch_for_spec(spec)
    raw_bytes = batch.raw_items[0]
    if provider in _JSONL_PROVIDERS:
        return [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    return json.loads(raw_bytes)


def _validate_payload(provider: str, payload: object) -> None:
    """Assert the payload validates against the committed provider schema.

    For JSONL providers (whose schema expects a single record object, not an
    array), validate each record individually.
    """
    if provider in _JSONL_PROVIDERS and isinstance(payload, list):
        for idx, record in enumerate(payload):
            result = validate_provider_export(record, provider, strict=False)
            assert result.is_valid, f"Schema validation failed for {provider} record {idx}: {result.errors}"
    else:
        result = validate_provider_export(payload, provider, strict=False)
        assert result.is_valid, f"Schema validation failed for {provider}: {result.errors}"


def _has_text_content(payload: object, provider: str) -> bool:
    """Check whether a generated payload contains any text content.

    This works across provider-specific payload shapes by parsing the
    generated session and checking the parsed message text.
    """
    from polylogue.sources.dispatch import parse_payload

    sessions = parse_payload(provider, payload, fallback_id="test")
    for conv in sessions:
        for msg in conv.messages:
            if msg.text and msg.text.strip():
                return True
            # Also check content_blocks for text.
            for blk in msg.content_blocks:
                if blk.text and blk.text.strip():
                    return True
    return False


# ── schema validation of normal payloads ────────────────────────────────


@pytest.mark.parametrize("provider", _SYNTHETIC_PROVIDERS)
def test_synthetic_payload_validates_against_provider_schema(provider: str) -> None:
    """A normally-generated payload must validate against its provider schema."""
    payload = _generate_payload(provider)
    _validate_payload(provider, payload)


@pytest.mark.parametrize("provider", _SYNTHETIC_PROVIDERS)
def test_multiple_message_counts_all_valid(provider: str) -> None:
    """Payloads with different message counts must all validate."""
    for n in (3, 7, 12):
        payload = _generate_payload(provider, n_messages=n, seed=n * 17)
        _validate_payload(provider, payload)


# ── edge case: message count extremes ───────────────────────────────────


@pytest.mark.parametrize("provider", _SYNTHETIC_PROVIDERS)
def test_single_message_session(provider: str) -> None:
    """A session with exactly one message must validate."""
    payload = _generate_payload(provider, n_messages=1, seed=1)
    _validate_payload(provider, payload)


@pytest.mark.parametrize("provider", _SYNTHETIC_PROVIDERS)
def test_many_message_session(provider: str) -> None:
    """A session with 50 messages must validate."""
    payload = _generate_payload(provider, n_messages=50, seed=2)
    _validate_payload(provider, payload)


# ── edge case: block-type checks via parse ──────────────────────────────


@pytest.mark.parametrize("provider", _SYNTHETIC_PROVIDERS)
def test_payload_produces_messages_with_text(provider: str) -> None:
    """A payload must produce messages with text content when parsed."""
    payload = _generate_payload(provider, n_messages=8, seed=42)
    assert _has_text_content(payload, provider), f"No text content found in {provider} payload after parsing"


def test_generated_claude_code_contains_varied_block_types() -> None:
    """With enough messages, claude-code synthetic payloads include varied block types."""
    from polylogue.sources.dispatch import parse_payload

    payload = _generate_payload("claude-code", n_messages=30, seed=42)
    assert isinstance(payload, list), f"Expected list payload for claude-code, got {type(payload)}"

    sessions = parse_payload("claude-code", payload, fallback_id="test")
    block_types_found: set[str] = set()
    for conv in sessions:
        for msg in conv.messages:
            for blk in msg.content_blocks:
                block_types_found.add(str(blk.type))
            if msg.text:
                block_types_found.add("text")

    assert len(block_types_found) >= 2, f"Expected at least 2 block types, got {block_types_found}"


# ── edge case: empty / missing text ─────────────────────────────────────


def test_claude_code_message_with_empty_text_parses() -> None:
    """A Claude Code record with empty string content must parse without error."""
    from polylogue.sources.dispatch import parse_payload

    records = [
        {
            "type": "user",
            "uuid": "test-uuid-1",
            "sessionId": "test-session-empty",
            "timestamp": 1700000000.0,
            "message": {"content": ""},
        },
        {
            "type": "assistant",
            "uuid": "test-uuid-2",
            "sessionId": "test-session-empty",
            "parentUuid": "test-uuid-1",
            "timestamp": 1700000060.0,
            "message": {"content": "ok"},
        },
    ]

    parsed = parse_payload("claude-code", records, fallback_id="test-empty")
    assert len(parsed) >= 1, f"Expected at least 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    assert result.messages, "Should produce at least one message"
    user_msgs = [m for m in result.messages if m.role and str(m.role) == "user"]
    assert len(user_msgs) >= 1, "Empty-text user message should be parsed"


def test_chatgpt_message_with_null_content_parses() -> None:
    """A ChatGPT message with null content parts must not crash the parser."""
    from polylogue.sources.parsers.chatgpt import parse

    payload = {
        "title": "Null Content Test",
        "mapping": {
            "msg-1": {
                "id": "msg-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [None]},
                    "create_time": 1700000000.0,
                },
            },
            "msg-2": {
                "id": "msg-2",
                "parent": "msg-1",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 1700000060.0,
                },
            },
        },
    }

    result = parse(payload, fallback_id="test-null")
    assert result.messages, "Should produce at least one message"


# ── edge case: role patterns ────────────────────────────────────────────


def test_claude_code_user_only_roles() -> None:
    """A claude-code session with only user-type records must parse."""
    from polylogue.sources.dispatch import parse_payload

    records: list[dict[str, object]] = [
        {
            "type": "user",
            "uuid": f"user-{i:04d}",
            "sessionId": "test-user-only",
            "timestamp": 1700000000.0 + i * 600.0,
            "message": {"content": f"Question {i + 1}"},
        }
        for i in range(3)
    ]

    parsed = parse_payload("claude-code", records, fallback_id="test-user-only")
    assert len(parsed) == 1, f"Expected 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    assert len(result.messages) == 3, f"Expected 3 messages, got {len(result.messages)}"


def test_claude_code_assistant_only_roles() -> None:
    """A claude-code session with only assistant-type records must parse."""
    from polylogue.sources.dispatch import parse_payload

    records: list[dict[str, object]] = [
        {
            "type": "assistant",
            "uuid": f"assistant-{i:04d}",
            "sessionId": "test-assistant-only",
            "timestamp": 1700000000.0 + i * 600.0,
            "message": {"content": f"Answer {i + 1}"},
        }
        for i in range(3)
    ]

    parsed = parse_payload("claude-code", records, fallback_id="test-assistant-only")
    assert len(parsed) == 1, f"Expected 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    assert len(result.messages) == 3, f"Expected 3 messages, got {len(result.messages)}"


def test_claude_code_system_records_not_messages() -> None:
    """System-type records (queue-operation, file-history-snapshot) must NOT
    produce message rows.  This is the contract that the parser skips
    internal bookkeeping records."""
    from polylogue.sources.dispatch import parse_payload

    records: list[dict[str, object]] = [
        {
            "type": "user",
            "uuid": "user-1",
            "sessionId": "test-sys-skip",
            "timestamp": 1700000000.0,
            "message": {"content": "hello"},
        },
        {
            "type": "queue-operation",
            "uuid": "queue-1",
            "sessionId": "test-sys-skip",
            "timestamp": 1700000030.0,
            "operation": "enqueue",
        },
        {
            "type": "file-history-snapshot",
            "uuid": "fhs-1",
            "sessionId": "test-sys-skip",
            "timestamp": 1700000060.0,
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "sessionId": "test-sys-skip",
            "timestamp": 1700000090.0,
            "message": {"content": "ack"},
        },
    ]

    parsed = parse_payload("claude-code", records, fallback_id="test-sys-skip")
    assert len(parsed) == 1, f"Expected 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    role_texts = {str(m.role) for m in result.messages if m.role}
    assert "system" not in role_texts, f"System records should not produce message rows, got roles: {role_texts}"
    assert len(result.messages) == 2, f"Expected 2 messages (user + assistant), got {len(result.messages)}"
