"""Pinned parser regressions that still add value beyond the law suites."""

from __future__ import annotations

from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.codex import parse as codex_parse
from polylogue.sources.source import parse_payload

# =============================================================================
# Claude Code cost aggregation regressions
# =============================================================================


def test_claude_code_cost_usd_non_numeric_string():
    """Test that Claude Code parser handles non-numeric costUSD strings.

    The key is that it doesn't crash during aggregation of costUSD values
    that are non-numeric strings.
    """
    payload = [
        {
            "type": "user",
            "uuid": "msg1",
            "message": {"role": "user", "content": "hello"},
            "timestamp": 1700000000,
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            "timestamp": 1700000001,
            "costUSD": "error",  # Non-numeric string - should be skipped in aggregation
            "durationMs": "pending",  # Also non-numeric - should be skipped in aggregation
        },
    ]

    # Should not crash and should produce a valid ParsedConversation
    # The parser uses _safe_float() which returns 0.0 for non-numeric strings
    result = parse_code(payload, "test-session")
    assert result is not None
    assert result.provider_name == "claude-code"
    # The parser only includes messages that validate properly via ClaudeCodeRecord
    # At least one message should be parsed
    assert len(result.messages) >= 1
    # The _safe_float() converter is used for costUSD aggregation
    # Non-numeric strings should result in 0.0, and 0.0 values are skipped in aggregation
    # So total_cost_usd should not be set or should be 0
    if result.provider_meta:
        total_cost = result.provider_meta.get("total_cost_usd")
        assert total_cost is None or total_cost == 0


def test_claude_code_cost_usd_valid_numeric_string():
    """Test that Claude Code parser handles numeric string costUSD correctly."""
    payload = [
        {
            "type": "user",
            "uuid": "msg1",
            "message": {"role": "user", "content": "hello"},
            "timestamp": 1700000000,
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            "timestamp": 1700000001,
            "costUSD": "0.05",  # Numeric string
            "durationMs": "1000",  # Numeric string
        },
    ]

    result = parse_code(payload, "test-session")
    assert result is not None
    assert len(result.messages) == 2
    # Should aggregate valid numeric strings
    assert result.provider_meta is not None
    assert result.provider_meta.get("total_cost_usd") == 0.05
    assert result.provider_meta.get("total_duration_ms") == 1000


def test_claude_code_cost_usd_mixed_valid_invalid():
    """Test that Claude Code aggregates valid costs and skips invalid ones."""
    payload = [
        {
            "type": "assistant",
            "uuid": "msg1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response1"}]},
            "timestamp": 1700000000,
            "costUSD": "0.02",  # Valid
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response2"}]},
            "timestamp": 1700000001,
            "costUSD": "invalid",  # Invalid, should be skipped
        },
        {
            "type": "assistant",
            "uuid": "msg3",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response3"}]},
            "timestamp": 1700000002,
            "costUSD": "0.03",  # Valid
        },
    ]

    result = parse_code(payload, "test-session")
    assert result is not None
    assert result.provider_meta is not None
    # Should aggregate only valid costs: 0.02 + 0.03 = 0.05
    assert result.provider_meta.get("total_cost_usd") == 0.05


# =============================================================================
# Codex role normalization regressions
# =============================================================================


def test_codex_role_normalization_human_to_user():
    """Test that Codex parser normalizes 'human' role to 'user'."""
    payload = [
        {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"},
        {
            "type": "message",
            "role": "human",  # Should be normalized to "user"
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        },
    ]

    result = codex_parse(payload, "test-codex")
    assert result is not None
    assert len(result.messages) == 2
    # First message should have role "user" after normalization
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "hello"
    # Second message should be "assistant"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "hi"


def test_codex_role_normalization_model_to_assistant():
    """Test that Codex parser normalizes 'model' role to 'assistant'."""
    payload = [
        {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"},
        {
            "type": "message",
            "role": "model",  # Should be normalized to "assistant"
            "content": [{"type": "output_text", "text": "response"}],
        },
    ]

    result = codex_parse(payload, "test-codex")
    assert result is not None
    assert len(result.messages) == 1
    assert result.messages[0].role == "assistant"


# =============================================================================
# parse_payload recursion limit
# =============================================================================


def test_parse_payload_recursion_depth_limit():
    """Test that deeply nested payloads don't cause stack overflow."""
    # Build a deeply nested payload with conversations key at depth > 10
    # Start with depth 12 (exceeds MAX_PARSE_DEPTH=10)
    # Construct the deeply nested structure step by step
    payload = {
        "conversations": [
            {
                "conversations": [
                    {
                        "conversations": [
                            {
                                "conversations": [
                                    {
                                        "conversations": [
                                            {
                                                "conversations": [
                                                    {
                                                        "conversations": [
                                                            {
                                                                "conversations": [
                                                                    {
                                                                        "conversations": [
                                                                            {
                                                                                "conversations": [
                                                                                    {
                                                                                        "conversations": [
                                                                                            {"id": "nested", "mapping": {}}
                                                                                        ]
                                                                                    }
                                                                                ]
                                                                            }
                                                                        ]
                                                                    }
                                                                ]
                                                            }
                                                        ]
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Should return a list (not crash on deep recursion)
    # The recursion limit prevents infinite loops but still returns an empty conversation
    result = parse_payload("chatgpt", payload, "test-deep")
    assert isinstance(result, list)
    # Deep nesting with empty mapping produces no conversations or empty conversations
    assert all(len(c.messages) == 0 for c in result)


def test_parse_payload_shallow_nesting_succeeds():
    """Test that moderately nested payloads within depth limit are parsed."""
    # Build a nested payload at depth 5 (within MAX_PARSE_DEPTH=10)
    payload = {
        "conversations": [
            {
                "conversations": [
                    {
                        "mapping": {
                            "node1": {
                                "id": "node1",
                                "message": {
                                    "id": "msg1",
                                    "author": {"role": "user"},
                                    "content": {"content_type": "text", "parts": ["hello"]},
                                    "create_time": 1700000000,
                                },
                                "children": [],
                            }
                        }
                    }
                ]
            }
        ]
    }

    result = parse_payload("chatgpt", payload, "test-shallow")
    assert isinstance(result, list)
    assert len(result) > 0


# =============================================================================
# INTEGRATION: Verify parsers don't crash on edge cases
# =============================================================================


def test_chatgpt_full_parse_with_string_author():
    """Test that full ChatGPT parse function handles string author gracefully."""
    payload = {
        "id": "conv1",
        "title": "Test",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg1",
                    "author": "system",  # String author
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 1700000000,
                },
                "children": [],
            },
            "node2": {
                "id": "node2",
                "message": {
                    "id": "msg2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hi"]},
                    "create_time": 1700000001,
                },
                "children": [],
            },
        },
    }

    result = chatgpt_parse(payload, "test-conv")
    assert result is not None
    assert result.provider_name == "chatgpt"
    # Should have only 1 message (msg2), msg1 skipped due to string author
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
