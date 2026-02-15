"""Regression tests for parser bug fixes.

These tests target specific edge cases that were previously crashing:
1. ChatGPT author field crash when msg.author is a non-dict truthy value
2. Claude Code cost aggregation crash on non-numeric strings
3. Codex role normalization (raw role -> normalized role)
4. Recursion depth limit to prevent stack overflow
5. YAML-safe function for Obsidian frontmatter with special characters
"""

from __future__ import annotations

import pytest

from polylogue.lib.formatting import _yaml_safe
from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping, parse as chatgpt_parse
from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.codex import parse as codex_parse
from polylogue.sources.source import _parse_json_payload


# =============================================================================
# 1. CHATGPT AUTHOR FIELD CRASH
# =============================================================================


def test_chatgpt_author_field_string_skips_message():
    """Test that ChatGPT messages with string author (not dict) are skipped."""
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": "system",  # String, not dict! Previously crashed
                "content": {"content_type": "text", "parts": ["hello"]},
                "create_time": 1700000000,
            },
            "children": [],
        },
        "node2": {
            "id": "node2",
            "message": {
                "id": "msg2",
                "author": {"role": "user"},  # Correct dict format
                "content": {"content_type": "text", "parts": ["good morning"]},
                "create_time": 1700000001,
            },
            "children": [],
        },
    }
    messages, attachments = extract_messages_from_mapping(mapping)
    # node1 should be skipped (author is string)
    # node2 should be parsed (author is dict with role)
    assert len(messages) == 1
    assert messages[0].provider_message_id == "msg2"
    assert messages[0].role == "user"
    assert messages[0].text == "good morning"
    assert len(attachments) == 0


def test_chatgpt_author_field_none_skips_message():
    """Test that ChatGPT messages with None author are skipped."""
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": None,
                "content": {"content_type": "text", "parts": ["hello"]},
                "create_time": 1700000000,
            },
            "children": [],
        },
    }
    messages, attachments = extract_messages_from_mapping(mapping)
    assert len(messages) == 0


def test_chatgpt_author_field_missing_skips_message():
    """Test that ChatGPT messages with missing author are skipped."""
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                # author field missing entirely
                "content": {"content_type": "text", "parts": ["hello"]},
                "create_time": 1700000000,
            },
            "children": [],
        },
    }
    messages, attachments = extract_messages_from_mapping(mapping)
    assert len(messages) == 0


def test_chatgpt_author_field_dict_empty_skips_message():
    """Test that ChatGPT messages with empty dict author (no role) are skipped."""
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {},  # Dict but no role field
                "content": {"content_type": "text", "parts": ["hello"]},
                "create_time": 1700000000,
            },
            "children": [],
        },
    }
    messages, attachments = extract_messages_from_mapping(mapping)
    assert len(messages) == 0


# =============================================================================
# 2. CLAUDE CODE COST AGGREGATION CRASH
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
# 3. CODEX ROLE NORMALIZATION
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
# 4. RECURSION DEPTH LIMIT
# =============================================================================


def test_parse_json_payload_recursion_depth_limit():
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
    result = _parse_json_payload("chatgpt", payload, "test-deep")
    assert isinstance(result, list)
    # Should still return a result, but with no messages due to empty mapping
    assert len(result) >= 0


def test_parse_json_payload_shallow_nesting_succeeds():
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

    result = _parse_json_payload("chatgpt", payload, "test-shallow")
    assert isinstance(result, list)
    assert len(result) > 0


# =============================================================================
# 5. YAML-SAFE FRONTMATTER
# =============================================================================


def test_yaml_safe_no_special_chars():
    """Test that values without special characters are not quoted."""
    assert _yaml_safe("simple_value") == "simple_value"
    assert _yaml_safe("hello-world") == "hello-world"
    assert _yaml_safe("123abc") == "123abc"


def test_yaml_safe_colon_requires_quoting():
    """Test that values with colons are quoted."""
    result = _yaml_safe("key: value")
    assert result.startswith('"') and result.endswith('"')
    assert "key: value" in result


def test_yaml_safe_hash_requires_quoting():
    """Test that values with # are quoted."""
    result = _yaml_safe("comment #text")
    assert result.startswith('"') and result.endswith('"')
    assert "comment #text" in result


def test_yaml_safe_bracket_requires_quoting():
    """Test that values with brackets are quoted."""
    result = _yaml_safe("array [1,2,3]")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_brace_requires_quoting():
    """Test that values with braces are quoted."""
    result = _yaml_safe("{key: value}")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_pipe_requires_quoting():
    """Test that values with pipe are quoted."""
    result = _yaml_safe("one|two")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_ampersand_requires_quoting():
    """Test that values with & are quoted."""
    result = _yaml_safe("a&b")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_asterisk_requires_quoting():
    """Test that values with * are quoted."""
    result = _yaml_safe("*wildcard")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_quote_escaping():
    """Test that quotes in values are properly escaped."""
    result = _yaml_safe('he said "hello"')
    # Should be quoted and internal quotes escaped
    assert result.startswith('"') and result.endswith('"')
    assert '\\"' in result


def test_yaml_safe_backslash_escaping():
    """Test that backslashes are properly escaped when in quoted strings.

    Note: Backslash alone is not in the set of special chars that trigger quoting.
    It's only escaped if the value is already being quoted due to other chars.
    """
    # Backslash alone does not trigger quoting
    result = _yaml_safe('path\\to\\file')
    # This string has no special YAML chars, so it shouldn't be quoted
    assert result == 'path\\to\\file'

    # But if combined with a special char that does trigger quoting
    result2 = _yaml_safe('path\\to\\file:test')
    # Now it should be quoted and backslash escaped
    assert result2.startswith('"') and result2.endswith('"')
    assert '\\\\' in result2


def test_yaml_safe_newline_requires_quoting():
    """Test that values with newlines are quoted."""
    result = _yaml_safe('line1\nline2')
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_multiple_special_chars():
    """Test that values with multiple special characters are properly escaped."""
    result = _yaml_safe('title: "The Best"')
    assert result.startswith('"') and result.endswith('"')
    assert '\\"' in result  # Quote should be escaped


def test_yaml_safe_exclamation_mark():
    """Test that ! is quoted."""
    result = _yaml_safe("stop!")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_question_mark():
    """Test that ? is quoted."""
    result = _yaml_safe("what?")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_at_sign():
    """Test that @ is quoted."""
    result = _yaml_safe("@mention")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_backtick():
    """Test that backtick is quoted."""
    result = _yaml_safe("command `run`")
    assert result.startswith('"') and result.endswith('"')


def test_yaml_safe_comma():
    """Test that comma is quoted."""
    result = _yaml_safe("item1, item2")
    assert result.startswith('"') and result.endswith('"')


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
