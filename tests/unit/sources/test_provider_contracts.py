"""Provider-specific contract tests.

For each committed provider schema, generate a schema-conformant payload
and assert the parser accepts it without error.  Includes regression tests
for known issues (#1617, unknown-field resilience).

Ref #1736.
"""

from __future__ import annotations

import json

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.validator import validate_provider_export
from polylogue.schemas.validator_resolution import available_providers
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedSession

# Providers that have both committed schemas AND synthetic generator support.
_VALIDATOR_PROVIDERS = set(available_providers())
_SYNTHETIC_PROVIDERS = set(SyntheticCorpus.available_providers())
_PROVIDERS_WITH_SCHEMAS = tuple(sorted(_VALIDATOR_PROVIDERS & _SYNTHETIC_PROVIDERS))

# JSONL wire-format providers.
_JSONL_PROVIDERS = {"claude-code", "codex"}

# ── helpers ─────────────────────────────────────────────────────────────


def _load_payload(raw_bytes: bytes, provider: str) -> object:
    """Load a synthetic payload, handling both JSON and JSONL formats."""
    if provider in _JSONL_PROVIDERS:
        return [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    return json.loads(raw_bytes)


def _generate_and_parse(provider: str) -> tuple[object, list[ParsedSession]]:
    """Generate a synthetic payload for a provider and parse it.

    Returns (raw_payload, parsed_sessions).
    """
    spec = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=5,
        messages_max=8,
        seed=42,
        origin="generated.test-provider-contracts",
        tags=("synthetic", "test", "provider-contracts"),
    )
    batch = SyntheticCorpus.generate_batch_for_spec(spec)
    raw_bytes = batch.raw_items[0]
    payload = _load_payload(raw_bytes, provider)
    result = parse_payload(provider, payload, fallback_id="test")
    return payload, result


def _validate_payload_for_provider(provider: str, payload: object) -> bool:
    """Validate a payload against the committed provider schema.

    For JSONL providers, each record is validated individually since the
    schema's root type is 'object', not 'array'.
    """
    if provider in _JSONL_PROVIDERS and isinstance(payload, list):
        for record in payload:
            result = validate_provider_export(record, provider, strict=False)
            if not result.is_valid:
                return False
        return True
    result = validate_provider_export(payload, provider, strict=False)
    return result.is_valid


# ── schema-conformant parse acceptance ──────────────────────────────────


@pytest.mark.parametrize("provider", _PROVIDERS_WITH_SCHEMAS)
def test_schema_conformant_payload_parses_without_error(provider: str) -> None:
    """Every committed provider schema must produce a payload the parser accepts."""
    _payload, sessions = _generate_and_parse(provider)
    assert len(sessions) >= 1, f"Parser returned zero sessions for {provider}"
    conv = sessions[0]
    assert hasattr(conv, "messages"), f"Parsed session for {provider} has no messages attribute"
    assert len(conv.messages) >= 1, f"Parsed session for {provider} has zero messages"


@pytest.mark.parametrize("provider", _PROVIDERS_WITH_SCHEMAS)
def test_schema_conformant_payload_validates(provider: str) -> None:
    """Every committed provider schema payload must pass schema validation."""
    payload, _ = _generate_and_parse(provider)
    assert _validate_payload_for_provider(provider, payload), f"Schema validation failed for {provider}"


# ── Claude Code: progress records regression (#1617) ────────────────────


def test_claude_code_progress_records_produce_zero_messages() -> None:
    """type=progress records must produce zero message rows.

    Regression test for #1617: progress records are hook lifecycle events
    carried alongside tool invocations.  They are NOT message content and
    must be dropped by the parser.
    """
    from polylogue.sources.dispatch import parse_payload

    records: list[dict[str, object]] = [
        {
            "type": "user",
            "uuid": "user-1",
            "sessionId": "test-progress-skip",
            "timestamp": 1700000000.0,
            "message": {"content": "run the tests"},
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "sessionId": "test-progress-skip",
            "timestamp": 1700000030.0,
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "toolu_01",
                        "input": {"command": "pytest"},
                    }
                ]
            },
        },
        {
            "type": "progress",
            "uuid": "progress-1",
            "sessionId": "test-progress-skip",
            "timestamp": 1700000040.0,
            "hookEvent": "PostToolUse",
            "hookName": "polylogue-hook",
            "command": "bash",
            "toolUseId": "toolu_01",
        },
        {
            "type": "user",
            "uuid": "user-2",
            "parentUuid": "assistant-1",
            "sessionId": "test-progress-skip",
            "timestamp": 1700000060.0,
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "3 passed"},
                ]
            },
        },
    ]

    # Use parse_payload (the public API) rather than the private
    # _parse_code_records.  parse_payload returns a list; for
    # claude-code the list has one element since all records belong
    # to the same session.
    parsed = parse_payload("claude-code", records, fallback_id="test-progress")
    assert len(parsed) == 1, f"Expected 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    assert result.messages, "Should have at least some messages"

    # No message should have text containing progress/hook data.
    progress_texts = [m.text for m in result.messages if m.text and ("hookEvent" in m.text or "progress" in m.text)]
    assert not progress_texts, f"Progress record data leaked into messages: {progress_texts}"

    # The message count should be 3 (user, assistant with tool_use, user with tool_result),
    # not 4 (no separate progress message).
    assert len(result.messages) == 3, (
        f"Expected 3 messages (user + assistant + tool_result user), got {len(result.messages)}"
    )


def test_claude_code_init_records_produce_zero_messages() -> None:
    """type=init and type=result records must produce zero message rows.

    These are internal lifecycle records, not message content.
    """
    from polylogue.sources.dispatch import parse_payload

    records: list[dict[str, object]] = [
        {
            "type": "init",
            "uuid": "init-1",
            "sessionId": "test-init-skip",
            "timestamp": 1700000000.0,
            "version": "1.0.0",
        },
        {
            "type": "user",
            "uuid": "user-1",
            "sessionId": "test-init-skip",
            "timestamp": 1700000060.0,
            "message": {"content": "hello"},
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "sessionId": "test-init-skip",
            "timestamp": 1700000120.0,
            "message": {"content": "hi"},
        },
    ]

    parsed = parse_payload("claude-code", records, fallback_id="test-init")
    assert len(parsed) == 1, f"Expected 1 parsed session, got {len(parsed)}"
    result = parsed[0]
    assert len(result.messages) == 2, (
        f"Expected 2 messages (user + assistant), got {len(result.messages)}. Init record should not produce a message."
    )


# ── unknown field resilience ────────────────────────────────────────────


@pytest.mark.parametrize("provider", ("chatgpt", "claude-code", "codex", "gemini"))
def test_unknown_fields_in_payload_do_not_crash_parser(provider: str) -> None:
    """Schema-conformant payloads with extra unknown fields must not crash parsers."""
    spec = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=4,
        messages_max=6,
        seed=42,
        origin="generated.test-unknown-fields",
        tags=("synthetic", "test", "unknown-fields"),
    )
    batch = SyntheticCorpus.generate_batch_for_spec(spec)
    raw_bytes = batch.raw_items[0]
    payload = _load_payload(raw_bytes, provider)

    # Inject unknown fields into the payload.
    _inject_unknown_fields(payload, depth=0)

    # Parse must succeed without exception.
    result = parse_payload(provider, payload, fallback_id="test-unknown")
    assert len(result) >= 1, f"Parser returned zero sessions for {provider} with unknown fields"
    assert len(result[0].messages) >= 1, f"Parser returned zero messages for {provider} with unknown fields"


@pytest.mark.parametrize("provider", ("chatgpt", "claude-code", "codex", "gemini"))
def test_unknown_fields_at_record_level_do_not_crash(provider: str) -> None:
    """Unknown fields at the top-level record must not cause parse failures."""
    spec = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=3,
        messages_max=5,
        seed=99,
        origin="generated.test-unknown-top-level",
        tags=("synthetic", "test", "unknown-fields"),
    )
    batch = SyntheticCorpus.generate_batch_for_spec(spec)
    raw_bytes = batch.raw_items[0]
    payload = _load_payload(raw_bytes, provider)

    # Add unknown fields at root level.
    if isinstance(payload, dict):
        payload["__unknown_synthetic_field__"] = {"nested": [1, 2, 3], "flag": True}
        payload["__another_unknown__"] = "should-be-harmless"
    elif isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                record["__unknown_synthetic_field__"] = 42
                record["__another_unknown__"] = None

    result = parse_payload(provider, payload, fallback_id="test-unknown-top")
    assert len(result) >= 1, f"Parser returned zero sessions for {provider} with unknown top-level fields"


def _inject_unknown_fields(payload: object, *, depth: int = 0) -> None:
    """Inject benign unknown fields into a payload structure.

    Guard against infinite recursion with a max depth.
    """
    if depth > 4:
        return
    if isinstance(payload, dict):
        payload["__polylogue_test_unknown__"] = {
            "nested_field": True,
            "array_field": [1, "two", None],
            "deep": {"a": 1, "b": 2},
        }
        for key, value in list(payload.items()):
            if key.startswith("__polylogue"):
                continue
            if isinstance(value, dict):
                _inject_unknown_fields(value, depth=depth + 1)
            elif isinstance(value, list):
                for item in value:
                    _inject_unknown_fields(item, depth=depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            _inject_unknown_fields(item, depth=depth + 1)


# ── provider dispatch: detection accuracy ───────────────────────────────


@pytest.mark.parametrize("provider", _PROVIDERS_WITH_SCHEMAS)
def test_generated_payload_is_detected_as_correct_provider(provider: str) -> None:
    """A generated payload must be detected as its own provider."""
    from polylogue.sources.dispatch import detect_provider

    payload, _ = _generate_and_parse(provider)
    detected = detect_provider(payload)
    assert detected is not None, f"detect_provider returned None for {provider} payload"
    assert str(detected) == provider, f"detect_provider returned {detected} instead of {provider}"


# ── parse_payload for all committed providers ───────────────────────────


@pytest.mark.parametrize("provider", _PROVIDERS_WITH_SCHEMAS)
def test_parse_payload_returns_typed_session(provider: str) -> None:
    """parse_payload must return a ParsedSession with the provider set."""
    _payload, sessions = _generate_and_parse(provider)
    assert len(sessions) >= 1
    conv = sessions[0]
    assert conv.source_name is not None, f"source_name is None for {provider}"
    assert len(conv.messages) >= 1, f"No messages for {provider}"
    first_msg = conv.messages[0]
    assert first_msg.role is not None, f"First message role is None for {provider}"
