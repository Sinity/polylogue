"""Property-based contracts for the unified harmonization layer.

Each law covers an invariant that must hold across all providers and
arbitrary inputs, superseding example-driven role/content tests in
test_models.py (sources).
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.types import Provider


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_PROVIDERS = [p.value for p in Provider]
_ROLES = ["user", "assistant", "system", "tool", "unknown",
          "human", "model", "ASSISTANT", "USER", "  system  ", ""]
_CANONICAL_ROLES = frozenset({"user", "assistant", "system", "tool", "unknown"})


@st.composite
def _claude_code_raw(draw: st.DrawFn) -> dict:
    role = draw(st.sampled_from(["user", "assistant", "system"]))
    text = draw(st.text(max_size=200))
    return {
        "uuid": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
        "type": role,
        "message": {"role": role, "content": text},
        "timestamp": "2025-01-01T00:00:00Z",
    }


@st.composite
def _chatgpt_raw(draw: st.DrawFn) -> dict:
    role = draw(st.sampled_from(["user", "assistant", "system"]))
    text = draw(st.text(max_size=200))
    return {
        "id": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
        "author": {"role": role},
        "content": {"content_type": "text", "parts": [text]},
        "create_time": 1700000000.0,
    }


@st.composite
def _gemini_raw(draw: st.DrawFn) -> dict:
    role = draw(st.sampled_from(["user", "model"]))
    text = draw(st.text(max_size=200))
    return {
        "role": role,
        "text": text,
    }


@st.composite
def _codex_raw(draw: st.DrawFn) -> dict:
    role = draw(st.sampled_from(["user", "assistant"]))
    text = draw(st.text(max_size=200))
    return {
        "id": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


@st.composite
def _claude_ai_raw(draw: st.DrawFn) -> dict:
    sender = draw(st.sampled_from(["human", "assistant", "system"]))
    text = draw(st.text(max_size=200))
    return {
        "uuid": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))),
        "sender": sender,
        "text": text,
    }


_PROVIDER_STRATEGIES = {
    Provider.CLAUDE_CODE: _claude_code_raw(),
    Provider.CHATGPT: _chatgpt_raw(),
    Provider.GEMINI: _gemini_raw(),
    Provider.CODEX: _codex_raw(),
    Provider.CLAUDE: _claude_ai_raw(),
}


# ---------------------------------------------------------------------------
# Law 1: HarmonizedMessage.role is always a canonical role value
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(_PROVIDER_STRATEGIES.keys())).flatmap(
    lambda p: _PROVIDER_STRATEGIES[p].map(lambda raw: (p, raw))
))
def test_harmonized_role_is_canonical(args: tuple) -> None:
    """extract_harmonized_message always produces a canonical role."""
    provider, raw = args
    msg = extract_harmonized_message(provider, raw)
    assert msg.role.value in _CANONICAL_ROLES


# ---------------------------------------------------------------------------
# Law 2: HarmonizedMessage.text is always a string (never None)
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(_PROVIDER_STRATEGIES.keys())).flatmap(
    lambda p: _PROVIDER_STRATEGIES[p].map(lambda raw: (p, raw))
))
def test_harmonized_text_is_string(args: tuple) -> None:
    """extract_harmonized_message always produces a str text field."""
    provider, raw = args
    msg = extract_harmonized_message(provider, raw)
    assert isinstance(msg.text, str)


# ---------------------------------------------------------------------------
# Law 3: HarmonizedMessage.provider matches the extraction provider
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(_PROVIDER_STRATEGIES.keys())).flatmap(
    lambda p: _PROVIDER_STRATEGIES[p].map(lambda raw: (p, raw))
))
def test_harmonized_provider_matches(args: tuple) -> None:
    """The provider field in HarmonizedMessage matches the input provider."""
    provider, raw = args
    msg = extract_harmonized_message(provider, raw)
    assert msg.provider == provider


# ---------------------------------------------------------------------------
# Law 4: HarmonizedMessage coerces any string role to canonical Role
# ---------------------------------------------------------------------------

@given(st.sampled_from(["user", "assistant", "system", "tool", "unknown",
                         "human", "model", "ASSISTANT", "SYSTEM", "USER"]))
def test_harmonized_message_role_coercion(role_str: str) -> None:
    """HarmonizedMessage coerces any recognized role string to canonical Role."""
    msg = HarmonizedMessage(role=role_str, text="test", provider=Provider.CLAUDE_CODE)
    assert msg.role.value in _CANONICAL_ROLES


# ---------------------------------------------------------------------------
# Law 5: extract_reasoning_traces never crashes on arbitrary content
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.one_of(
            st.just({"type": "thinking", "thinking": "thought"}),
            st.just({"type": "text", "text": "text"}),
            st.just("not a dict"),
            st.just(42),
        ),
        max_size=10,
    ),
    st.sampled_from(list(_PROVIDER_STRATEGIES.keys())),
)
def test_extract_reasoning_traces_never_crashes(content: list, provider: Provider) -> None:
    """extract_reasoning_traces never crashes on arbitrary content."""
    traces = extract_reasoning_traces(content, provider)
    assert isinstance(traces, list)


# ---------------------------------------------------------------------------
# Law 6: extract_tool_calls never crashes on arbitrary content
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.one_of(
            st.just({"type": "tool_use", "name": "Read", "input": {}}),
            st.just({"type": "text", "text": "text"}),
            st.just("not a dict"),
            st.just(None),
        ),
        max_size=10,
    ),
    st.sampled_from(list(_PROVIDER_STRATEGIES.keys())),
)
def test_extract_tool_calls_never_crashes(content: list, provider: Provider) -> None:
    """extract_tool_calls never crashes on arbitrary content."""
    calls = extract_tool_calls(content, provider)
    assert isinstance(calls, list)


# ---------------------------------------------------------------------------
# Law 7: HarmonizedMessage.provider coerces string providers
# ---------------------------------------------------------------------------

@given(st.sampled_from(["chatgpt", "claude", "claude-code", "gemini", "codex"]))
def test_harmonized_provider_coercion(provider_str: str) -> None:
    """HarmonizedMessage coerces provider strings to Provider enum."""
    msg = HarmonizedMessage(role="user", text="test", provider=provider_str)
    assert isinstance(msg.provider, Provider)


# ---------------------------------------------------------------------------
# Law 8: extract_reasoning_traces returns traces only for thinking blocks
# ---------------------------------------------------------------------------

def test_extract_reasoning_traces_only_for_thinking() -> None:
    """extract_reasoning_traces only extracts thinking blocks, not text."""
    content = [
        {"type": "thinking", "thinking": "deep thought"},
        {"type": "text", "text": "response"},
        {"type": "tool_use", "name": "Read", "input": {}},
    ]
    traces = extract_reasoning_traces(content, Provider.CLAUDE_CODE)
    assert len(traces) == 1
    assert "deep thought" in traces[0].text


# ---------------------------------------------------------------------------
# Law 9: extract_tool_calls returns calls only for tool_use blocks
# ---------------------------------------------------------------------------

def test_extract_tool_calls_only_for_tool_use() -> None:
    """extract_tool_calls only extracts tool_use blocks, not other types."""
    content = [
        {"type": "thinking", "thinking": "thought"},
        {"type": "text", "text": "response"},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
    ]
    calls = extract_tool_calls(content, Provider.CLAUDE_CODE)
    assert len(calls) == 1
    assert calls[0].name == "Bash"


# ---------------------------------------------------------------------------
# Law 10: HarmonizedMessage.has_reasoning is consistent with reasoning_traces
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(_PROVIDER_STRATEGIES.keys())).flatmap(
    lambda p: _PROVIDER_STRATEGIES[p].map(lambda raw: (p, raw))
))
def test_has_reasoning_consistent_with_traces(args: tuple) -> None:
    """has_reasoning iff reasoning_traces is non-empty."""
    provider, raw = args
    msg = extract_harmonized_message(provider, raw)
    assert msg.has_reasoning == (len(msg.reasoning_traces) > 0)


# ---------------------------------------------------------------------------
# Law 11: HarmonizedMessage.has_tool_use is consistent with tool_calls
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(_PROVIDER_STRATEGIES.keys())).flatmap(
    lambda p: _PROVIDER_STRATEGIES[p].map(lambda raw: (p, raw))
))
def test_has_tool_use_consistent_with_calls(args: tuple) -> None:
    """has_tool_use iff tool_calls is non-empty."""
    provider, raw = args
    msg = extract_harmonized_message(provider, raw)
    assert msg.has_tool_use == (len(msg.tool_calls) > 0)
