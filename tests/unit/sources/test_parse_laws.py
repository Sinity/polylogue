"""Property laws for provider parsers and role normalization.

Each law covers an invariant that holds for any input, superseding
specific example tables in the parser test files.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib.roles import normalize_role
from polylogue.sources.providers.chatgpt import (
    ChatGPTAuthor,
    ChatGPTContent,
    ChatGPTConversation,
    ChatGPTMessage,
    ChatGPTNode,
)
from polylogue.sources.providers.claude_code import (
    ClaudeCodeThinkingBlock,
    ClaudeCodeToolUse,
    ClaudeCodeUsage,
)

# ---------------------------------------------------------------------------
# Law 1: normalize_role never raises for any non-empty string
# ---------------------------------------------------------------------------

@given(st.text(min_size=1))
def test_normalize_role_never_raises_for_nonempty(text: str) -> None:
    """normalize_role handles any non-empty string without raising."""
    # normalize_role raises only on empty/whitespace-only strings
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Law 2: normalize_role always returns one of the canonical roles
# ---------------------------------------------------------------------------

CANONICAL_ROLES = frozenset({"user", "assistant", "system", "tool", "unknown"})


@given(st.text(min_size=1))
def test_normalize_role_result_is_canonical(text: str) -> None:
    """normalize_role always returns a canonical role string."""
    stripped = text.strip()
    if stripped:
        result = normalize_role(text)
        assert result in CANONICAL_ROLES


# ---------------------------------------------------------------------------
# Law 3: normalize_role is idempotent on its own output
# ---------------------------------------------------------------------------

@given(st.sampled_from(sorted(CANONICAL_ROLES - {"unknown"})))
def test_normalize_role_idempotent_on_canonical(role: str) -> None:
    """Applying normalize_role to a canonical role returns the same value."""
    result = normalize_role(role)
    assert result == role


# ---------------------------------------------------------------------------
# Law 4: normalize_role is case-insensitive
# ---------------------------------------------------------------------------

@given(st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",))))
def test_normalize_role_case_insensitive(text: str) -> None:
    """normalize_role gives the same result for any case variant."""
    stripped = text.strip()
    if stripped:
        lower_result = normalize_role(stripped.lower())
        upper_result = normalize_role(stripped.upper())
        title_result = normalize_role(stripped.title())
        assert lower_result == upper_result == title_result


# ---------------------------------------------------------------------------
# Law 5: normalize_role strips whitespace before normalizing
# ---------------------------------------------------------------------------

@given(
    st.sampled_from(["user", "assistant", "system", "tool"]),
    st.integers(min_value=0, max_value=5),
)
def test_normalize_role_strips_whitespace(role: str, padding: int) -> None:
    """normalize_role ignores leading/trailing whitespace."""
    padded = " " * padding + role + " " * padding
    assert normalize_role(padded) == role


# ---------------------------------------------------------------------------
# Law 6: normalize_role raises ValueError for empty/whitespace-only input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty", ["", "   ", "\t", "\n", "\t\n  "])
def test_normalize_role_raises_on_empty(empty: str) -> None:
    """normalize_role raises ValueError for empty or whitespace-only input."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        normalize_role(empty)


@pytest.mark.parametrize(
    ("roles", "expected_pairs"),
    [
        (["user", "assistant"], [("m0", "m1")]),
        (["system", "user", "assistant", "assistant"], [("m1", "m2")]),
        (["user", "tool", "assistant", "user", "assistant"], [("m3", "m4")]),
        (["assistant", "user"], []),
    ],
)
def test_chatgpt_iter_user_assistant_pairs_contract(
    roles: list[str],
    expected_pairs: list[tuple[str, str]],
) -> None:
    mapping: dict[str, ChatGPTNode] = {}
    children = [f"node-{idx}" for idx in range(len(roles))]
    mapping["root"] = ChatGPTNode(id="root", parent=None, children=children[:1])
    for idx, role in enumerate(roles):
        node_id = f"node-{idx}"
        next_child = [f"node-{idx + 1}"] if idx + 1 < len(roles) else []
        mapping[node_id] = ChatGPTNode(
            id=node_id,
            parent="root" if idx == 0 else f"node-{idx - 1}",
            children=next_child,
            message=ChatGPTMessage(
                id=f"m{idx}",
                author=ChatGPTAuthor(role=role),
                content=ChatGPTContent(content_type="text", parts=[f"{role}-{idx}"]),
            ),
        )

    conversation = ChatGPTConversation(
        id="conv-pairs",
        conversation_id="conv-pairs",
        title="pairs",
        create_time=1700000000.0,
        update_time=1700000100.0,
        mapping=mapping,
        current_node=f"node-{len(roles) - 1}" if roles else "root",
    )

    assert [
        (user.id, assistant.id)
        for user, assistant in conversation.iter_user_assistant_pairs()
    ] == expected_pairs


def test_claude_code_helper_conversion_contracts() -> None:
    tool = ClaudeCodeToolUse(id="tool-1", name="bash", input={"command": "git status"})
    trace = ClaudeCodeThinkingBlock(thinking="chain of thought")
    usage = ClaudeCodeUsage(
        input_tokens=12,
        output_tokens=34,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=6,
    )

    tool_call = tool.to_tool_call()
    reasoning = trace.to_reasoning_trace()
    token_usage = usage.to_token_usage()

    assert tool_call.name == "bash"
    assert tool_call.id == "tool-1"
    assert tool_call.input == {"command": "git status"}
    assert tool_call.provider == "claude-code"
    assert tool_call.raw == tool.model_dump()

    assert reasoning.text == "chain of thought"
    assert reasoning.provider == "claude-code"
    assert reasoning.raw == trace.model_dump()

    assert token_usage.input_tokens == 12
    assert token_usage.output_tokens == 34
    assert token_usage.cache_write_tokens == 5
    assert token_usage.cache_read_tokens == 6
