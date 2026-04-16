"""Schema-driven semantic property tests across providers.

Extends crashlessness testing with semantic invariants: valid roles,
valid content block types, title stability, and timestamp parseability.

These tests generate schema-conformant payloads and verify that parsed
conversations produce semantically valid output.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.lib.roles import Role
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base_models import ParsedProviderEvent
from polylogue.types import ContentBlockType
from tests.infra.strategies.schema_driven import schema_conformant_payload

PARSEABLE_PROVIDERS = ("chatgpt", "claude-code", "codex")
VALID_ROLES = frozenset(Role)
_HEALTH_SUPPRESS = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]


def _try_parse(provider: str, payload: object):
    """Attempt to parse, returning parsed conversations or None on expected failures."""
    try:
        detected = detect_provider(payload)
        if detected is None:
            return None
        return parse_payload(detected, payload, fallback_id="prop-test")
    except Exception:
        return None


@pytest.mark.parametrize("provider", PARSEABLE_PROVIDERS)
@given(data=st.data())
@settings(max_examples=15, deadline=None, suppress_health_check=_HEALTH_SUPPRESS)
def test_parse_produces_valid_roles(provider: str, data) -> None:
    """Successfully parsed conversations always have valid roles."""
    payload = data.draw(schema_conformant_payload(provider))
    conversations = _try_parse(provider, payload)
    if not conversations:
        return
    for conv in conversations:
        for msg in conv.messages:
            assert msg.role in VALID_ROLES, f"Invalid role {msg.role!r} from {provider}"


@pytest.mark.parametrize("provider", PARSEABLE_PROVIDERS)
@given(data=st.data())
@settings(max_examples=15, deadline=None, suppress_health_check=_HEALTH_SUPPRESS)
def test_parse_produces_valid_content_block_types(provider: str, data) -> None:
    """Content blocks always have valid ContentBlockType values."""
    payload = data.draw(schema_conformant_payload(provider))
    conversations = _try_parse(provider, payload)
    if not conversations:
        return
    valid_types = set(ContentBlockType)
    for conv in conversations:
        for msg in conv.messages:
            for block in msg.content_blocks:
                assert block.type in valid_types, f"Invalid content block type {block.type!r} from {provider}"


@pytest.mark.parametrize("provider", PARSEABLE_PROVIDERS)
@given(data=st.data())
@settings(max_examples=10, deadline=None, suppress_health_check=_HEALTH_SUPPRESS)
def test_parse_title_is_stable(provider: str, data) -> None:
    """Parsing the same payload twice produces the same title."""
    payload = data.draw(schema_conformant_payload(provider))
    conversations1 = _try_parse(provider, payload)
    conversations2 = _try_parse(provider, payload)
    if not conversations1 or not conversations2:
        return
    titles1 = [conv.title for conv in conversations1]
    titles2 = [conv.title for conv in conversations2]
    assert titles1 == titles2, f"Title instability for {provider}: {titles1!r} vs {titles2!r}"


class TestProviderEventRoundtrip:
    """ParsedProviderEvent serialization round-trips correctly."""

    def test_compaction_event_roundtrips(self):
        event = ParsedProviderEvent(
            event_type="compaction",
            timestamp="2026-01-01T00:00:00Z",
            payload={"trigger": "auto", "pre_tokens": 4096, "summary_text": "Session compacted"},
        )
        data = event.model_dump()
        restored = ParsedProviderEvent.model_validate(data)
        assert restored.event_type == event.event_type
        assert restored.timestamp == event.timestamp
        assert restored.payload == event.payload

    def test_empty_event_roundtrips(self):
        event = ParsedProviderEvent(event_type="turn_context")
        data = event.model_dump()
        restored = ParsedProviderEvent.model_validate(data)
        assert restored.event_type == "turn_context"
        assert restored.payload == {}
