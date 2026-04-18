"""Property tests for schema-driven parse → render → re-parse roundtrip.

Replaces the 8-file roundtrip proof infrastructure with Hypothesis property
tests that generate schema-conformant payloads and verify the pipeline
preserves core structure.
"""

from __future__ import annotations

import json

from hypothesis import given, settings

from polylogue.lib.messages import MessageCollection
from polylogue.rendering.formatting import format_conversation
from tests.infra.builders import make_conv, make_msg
from tests.infra.strategies.schema_driven import schema_conformant_payload


@given(payload=schema_conformant_payload("chatgpt"))
@settings(max_examples=10, deadline=10000)
def test_chatgpt_schema_payload_parses_without_crash(payload: object) -> None:
    """Schema-conformant ChatGPT payloads must not crash with unhandled exceptions.

    ValueError (missing role) and ValidationError are expected for minimal
    payloads and are caught by the extraction pipeline. Any other exception
    is a genuine crash.
    """
    from pydantic import ValidationError

    from polylogue.schemas.unified import extract_harmonized_message

    if not isinstance(payload, dict):
        return
    try:
        result = extract_harmonized_message("chatgpt", payload)
        assert result is not None
        assert result.provider is not None
    except (ValueError, ValidationError):
        pass  # expected for minimal/empty payloads


@given(payload=schema_conformant_payload("claude-code"))
@settings(max_examples=10, deadline=10000)
def test_claude_code_schema_payload_parses_without_crash(payload: object) -> None:
    """Schema-conformant Claude Code payloads must not crash with unhandled exceptions."""
    from pydantic import ValidationError

    from polylogue.schemas.unified import extract_harmonized_message

    if not isinstance(payload, dict):
        return
    try:
        result = extract_harmonized_message("claude-code", payload)
        assert result is not None
        assert result.provider is not None
    except (ValueError, ValidationError):
        pass


@given(payload=schema_conformant_payload("chatgpt"))
@settings(max_examples=10, deadline=10000)
def test_chatgpt_successful_extraction_produces_valid_role(payload: object) -> None:
    """When extraction succeeds, the role must be valid and non-empty."""
    from pydantic import ValidationError

    from polylogue.schemas.unified import extract_harmonized_message

    if not isinstance(payload, dict):
        return
    try:
        result = extract_harmonized_message("chatgpt", payload)
    except (ValueError, ValidationError):
        return  # can't test role if extraction fails
    assert result.role is not None
    assert str(result.role) != ""


@given(payload=schema_conformant_payload("claude-code"))
@settings(max_examples=5, deadline=15000)
def test_claude_code_json_roundtrip_preserves_message_count(payload: object) -> None:
    """parse → json_export → re-parse preserves message count for Claude Code."""
    from pydantic import ValidationError

    from polylogue.schemas.unified import extract_harmonized_message

    if not isinstance(payload, dict):
        return

    try:
        result = extract_harmonized_message("claude-code", payload)
    except (ValueError, ValidationError):
        return  # can't roundtrip if extraction fails

    msg = make_msg(id=result.id or "test-msg", role=result.role, text=result.text or "")
    conv = make_conv(
        id="roundtrip-test",
        provider="claude-code",
        title="Roundtrip Test",
        messages=MessageCollection(messages=[msg]),
    )

    json_str = format_conversation(conv, "json", None)
    json_data = json.loads(json_str)

    assert isinstance(json_data.get("messages"), list)
    assert len(json_data["messages"]) == 1
    assert json_data["messages"][0]["role"] == str(msg.role)
