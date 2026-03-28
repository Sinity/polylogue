"""Consolidated message classification tests.

CONSOLIDATION: Reduced 31 tests to 7 using parametrization.

Original: Individual tests for each role variant, detection method
New: Parametrized tests covering all variants
"""

import pytest

from polylogue.lib.models import Message

# =============================================================================
# ROLE CLASSIFICATION - PARAMETRIZED (1 test replacing 8 tests)
# =============================================================================


# Test cases: (role_string, is_user, is_assistant, is_system, is_dialogue, description)
ROLE_TEST_CASES = [
    ("user", True, False, False, True, "user role"),
    ("human", True, False, False, True, "human alias"),
    ("USER", True, False, False, True, "uppercase user"),
    ("assistant", False, True, False, True, "assistant role"),
    ("model", False, True, False, True, "model alias (Gemini)"),
    ("ASSISTANT", False, True, False, True, "uppercase assistant"),
    ("system", False, False, True, False, "system role"),
    ("tool", False, False, False, False, "tool role"),
]


@pytest.mark.parametrize("role,exp_user,exp_asst,exp_sys,exp_dial,desc", ROLE_TEST_CASES)
def test_role_classification_comprehensive(role, exp_user, exp_asst, exp_sys, exp_dial, desc):
    """Comprehensive role classification test.

    Replaces 8 individual role tests with single parametrized test.
    """
    msg = Message(id="1", role=role, text="Test")

    assert msg.is_user == exp_user, f"Wrong is_user for {desc}"
    assert msg.is_assistant == exp_asst, f"Wrong is_assistant for {desc}"
    assert msg.is_system == exp_sys, f"Wrong is_system for {desc}"
    assert msg.is_dialogue == exp_dial, f"Wrong is_dialogue for {desc}"


# =============================================================================
# THINKING DETECTION - PARAMETRIZED (1 test replacing 5 tests)
# =============================================================================


THINKING_TEST_CASES = [
    # (provider_meta, expected_is_thinking, description)
    ({"content_blocks": [{"type": "thinking", "text": "Analysis..."}]}, True, "content_blocks"),
    ({"isThought": True}, True, "Gemini isThought"),
    ({"raw": {"isThought": True}}, True, "Gemini nested isThought"),
    ({"raw": {"content": {"content_type": "thoughts"}}}, True, "ChatGPT thoughts"),
    ({"raw": {"content": {"content_type": "reasoning_recap"}}}, True, "ChatGPT reasoning"),
    ({}, False, "no markers"),
    (None, False, "None provider_meta"),
]


@pytest.mark.parametrize("provider_meta,expected,desc", THINKING_TEST_CASES)
def test_is_thinking_detection(provider_meta, expected, desc):
    """Comprehensive thinking detection test.

    Replaces 5 individual thinking detection tests.
    """
    msg = Message(
        id="1",
        role="assistant",
        text="Thinking content...",
        provider_meta=provider_meta
    )
    assert msg.is_thinking == expected, f"Wrong is_thinking for {desc}"


# =============================================================================
# TOOL USE DETECTION - PARAMETRIZED (1 test replacing 6 tests)
# =============================================================================


TOOL_USE_TEST_CASES = [
    # (role, provider_meta, expected_is_tool_use, description)
    ("tool", {}, True, "role=tool"),
    ("assistant", {"content_blocks": [{"type": "tool_use"}]}, True, "content_blocks tool_use"),
    ("assistant", {"raw": {"isSidechain": True}}, True, "Claude sidechain"),
    ("assistant", {"raw": {"isMeta": True}}, True, "Claude meta marker"),
    ("assistant", {}, False, "normal assistant"),
    ("user", {}, False, "user message"),
]


@pytest.mark.parametrize("role,provider_meta,expected,desc", TOOL_USE_TEST_CASES)
def test_is_tool_use_detection(role, provider_meta, expected, desc):
    """Comprehensive tool use detection test.

    Replaces 6 individual tool use tests.
    """
    msg = Message(
        id="1",
        role=role,
        text="Tool content",
        provider_meta=provider_meta
    )
    assert msg.is_tool_use == expected, f"Wrong is_tool_use for {desc}"


# =============================================================================
# CONTEXT DUMP DETECTION - PARAMETRIZED (1 test replacing 4 tests)
# =============================================================================


CONTEXT_DUMP_TEST_CASES = [
    # (text, expected_is_context_dump, description)
    ("```\n```\n```\n```\n```\n```\nCode", True, "6+ backticks (3+ code blocks)"),
    ("<system>Long context</system>\n" * 5, True, "multiple system tags"),
    ("Normal message with one ```code block```", False, "single code block"),
    ("Regular text without markers", False, "plain text"),
]


@pytest.mark.parametrize("text,expected,desc", CONTEXT_DUMP_TEST_CASES)
def test_is_context_dump_detection(text, expected, desc):
    """Comprehensive context dump detection test.

    Replaces 4 individual context dump tests.
    """
    msg = Message(id="1", role="user", text=text)
    assert msg.is_context_dump == expected, f"Wrong is_context_dump for {desc}"


# =============================================================================
# NOISE AND SUBSTANTIVE - PARAMETRIZED (1 test replacing 5 tests)
# =============================================================================


NOISE_TEST_CASES = [
    # (role, text, provider_meta, expected_is_noise, expected_is_substantive, description)
    ("system", "System prompt", {}, True, False, "system message"),
    ("assistant", "A slightly longer message", {}, False, True, "text >10 chars"),
    ("assistant", "Thinking...", {"isThought": True}, False, False, "thinking block (noise via is_thinking in substantive check)"),
    ("tool", "Tool result", {}, True, False, "tool message"),
    ("assistant", "This is a substantial answer with details.", {}, False, True, "substantive"),
    ("user", "Regular question here?", {}, False, True, "user question"),
]


@pytest.mark.parametrize("role,text,meta,exp_noise,exp_subst,desc", NOISE_TEST_CASES)
def test_noise_and_substantive_classification(role, text, meta, exp_noise, exp_subst, desc):
    """Comprehensive noise/substantive classification.

    Replaces 5 individual classification tests.
    """
    msg = Message(id="1", role=role, text=text, provider_meta=meta)

    assert msg.is_noise == exp_noise, f"Wrong is_noise for {desc}"
    assert msg.is_substantive == exp_subst, f"Wrong is_substantive for {desc}"


# =============================================================================
# METADATA EXTRACTION - PARAMETRIZED (1 test replacing 3 tests)
# =============================================================================


METADATA_TEST_CASES = [
    # (provider_meta, expected_cost, expected_duration, expected_word_count, description)
    ({"costUSD": 0.005}, 0.005, None, None, "cost only"),
    ({"durationMs": 2500}, None, 2500, None, "duration only"),
    ({"costUSD": 0.01, "durationMs": 5000}, 0.01, 5000, None, "both"),
    ({}, None, None, None, "no metadata"),
]


@pytest.mark.parametrize("meta,exp_cost,exp_dur,exp_words,desc", METADATA_TEST_CASES)
def test_metadata_extraction(meta, exp_cost, exp_dur, exp_words, desc):
    """Comprehensive metadata extraction test.

    Replaces 3 individual metadata tests.
    """
    msg = Message(id="1", role="assistant", text="Response text", provider_meta=meta)

    assert msg.cost_usd == exp_cost, f"Wrong cost_usd for {desc}"
    assert msg.duration_ms == exp_dur, f"Wrong duration_ms for {desc}"
    # word_count is always calculated from text
    assert msg.word_count > 0
