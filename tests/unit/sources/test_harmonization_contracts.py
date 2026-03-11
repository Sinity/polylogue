"""Property-based contracts for the unified harmonization layer.

Each law covers an invariant that must hold across all providers and
arbitrary inputs, superseding example-driven role/content tests in
test_models.py (sources).
"""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_codex_text,
    extract_content_blocks,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.types import Provider
from tests.infra.strategies import content_block_strategy

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


def _expected_reasoning_texts(content: list[object]) -> list[str]:
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            text = block.get("thinking") or block.get("text")
            if text:
                texts.append(text)
        elif block.get("isThought") and block.get("text"):
            texts.append(block["text"])
    return texts


def _expected_tool_blocks(content: list[object]) -> list[dict[str, object]]:
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _expected_content_types(content: list[object]) -> list[str]:
    expected: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "text")
        if block_type == "text":
            expected.append("text")
        elif block_type == "thinking":
            expected.append("thinking")
        elif block_type == "tool_use":
            expected.append("tool_use")
        elif block_type == "tool_result":
            expected.append("tool_result")
        elif block_type == "code":
            expected.append("code")
    return expected


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
# Law 5: extract_reasoning_traces returns every reasoning block in order
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.fixed_dictionaries({"isThought": st.just(True), "text": st.text(min_size=1, max_size=100)}),
            st.just("not a dict"),
            st.just(42),
        ),
        max_size=10,
    ),
    st.sampled_from(list(_PROVIDER_STRATEGIES.keys())),
)
def test_extract_reasoning_traces_preserve_reasoning_blocks(content: list, provider: Provider) -> None:
    """extract_reasoning_traces preserves every reasoning block in order."""
    traces = extract_reasoning_traces(content, provider)
    expected_texts = _expected_reasoning_texts(content)
    assert [trace.text for trace in traces] == expected_texts
    assert all(trace.provider == provider for trace in traces)


# ---------------------------------------------------------------------------
# Law 6: extract_tool_calls returns every tool_use block in order
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.just("not a dict"),
            st.just(None),
        ),
        max_size=10,
    ),
    st.sampled_from(list(_PROVIDER_STRATEGIES.keys())),
)
def test_extract_tool_calls_preserve_tool_use_blocks(content: list, provider: Provider) -> None:
    """extract_tool_calls preserves every tool_use block in order."""
    calls = extract_tool_calls(content, provider)
    expected = _expected_tool_blocks(content)
    assert [call.name for call in calls] == [str(block.get("name", "")) for block in expected]
    assert [call.id for call in calls] == [block.get("id") for block in expected]
    assert [call.input for call in calls] == [
        block.get("input", {}) if isinstance(block.get("input"), dict) else {}
        for block in expected
    ]
    assert all(call.provider == provider for call in calls)


# ---------------------------------------------------------------------------
# Law 7: HarmonizedMessage.provider coerces string providers
# ---------------------------------------------------------------------------

@given(st.sampled_from(["chatgpt", "claude", "claude-code", "gemini", "codex"]))
def test_harmonized_provider_coercion(provider_str: str) -> None:
    """HarmonizedMessage coerces provider strings to Provider enum."""
    msg = HarmonizedMessage(role="user", text="test", provider=provider_str)
    assert isinstance(msg.provider, Provider)


# ---------------------------------------------------------------------------
# Law 8: extract_content_blocks classifies every recognized block in order
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.fixed_dictionaries({"type": st.just("unknown"), "text": st.text(max_size=50)}),
            st.just("not a dict"),
        ),
        max_size=10,
    )
)
def test_extract_content_blocks_preserve_recognized_block_order(content: list[object]) -> None:
    """extract_content_blocks preserves recognized block order and classification."""
    blocks = extract_content_blocks(content)
    assert [block.type.value for block in blocks] == _expected_content_types(content)


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


@given(
    st.lists(
        st.one_of(
            st.fixed_dictionaries(
                {
                    "text": st.text(max_size=40),
                    "input_text": st.text(max_size=40),
                    "output_text": st.text(max_size=40),
                }
            ),
            st.just({"type": "image", "url": "https://example.com"}),
            st.just("not a dict"),
        ),
        max_size=8,
    )
)
def test_extract_codex_text_prefers_first_available_text_field(content: list[object]) -> None:
    expected: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if isinstance(text, str) and text:
            expected.append(text)

    assert extract_codex_text(content) == "\n".join(expected)
