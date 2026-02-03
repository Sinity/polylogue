"""Property-based tests for polylogue.types module.

SYSTEMATIZATION: 45 spot checks → 5 property tests + 3 parametrized tests.

NewType wrappers (ConversationId, MessageId, AttachmentId, ContentHash) are
compile-time markers that become str at runtime. Property tests verify:
1. String semantics preserved
2. Equality reflects underlying string
3. Hash/dict/set behavior matches str

Provider enum tests verify the from_string() normalization logic.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.types import (
    AttachmentId,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
)


# =============================================================================
# NewType Property Tests (replacing 36 spot checks with 4 property tests)
# =============================================================================

# All NewType string wrappers share identical runtime behavior
NEWTYPE_WRAPPERS = [ConversationId, MessageId, AttachmentId, ContentHash]
NEWTYPE_IDS = ["ConversationId", "MessageId", "AttachmentId", "ContentHash"]


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_preserves_string_semantics(wrapper, text: str) -> None:
    """NewType wrappers are str at runtime - all string operations work.

    Subsumes: test_creation_from_string, test_is_string_type, test_string_methods_available
    for all 4 NewType wrappers (12 spot checks).
    """
    wrapped = wrapper(text)

    # Runtime type is str
    assert isinstance(wrapped, str)

    # Value equality
    assert wrapped == text

    # String methods work
    assert wrapped.upper() == text.upper()
    assert wrapped.lower() == text.lower()
    assert len(wrapped) == len(text)

    # Containment check
    if text:
        assert text[0] in wrapped


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text(), st.text())
def test_newtype_equality_reflects_string(wrapper, a: str, b: str) -> None:
    """Equality between wrapped values reflects underlying string equality.

    Subsumes: test_equality_same_value, test_inequality_different_value
    for all 4 wrappers (8 spot checks).
    """
    wrapped_a = wrapper(a)
    wrapped_b = wrapper(b)

    # Equality matches string equality
    assert (wrapped_a == wrapped_b) == (a == b)

    # Inequality is the opposite
    assert (wrapped_a != wrapped_b) == (a != b)


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_hashable_as_string(wrapper, text: str) -> None:
    """NewType values hash identically to their underlying string.

    Subsumes: test_use_as_dict_key, test_use_in_set for all 4 wrappers (8 spot checks).
    """
    wrapped = wrapper(text)

    # Hash matches string hash
    assert hash(wrapped) == hash(text)

    # Works as dict key
    d = {wrapped: "value"}
    assert d[text] == "value"  # Can look up via plain string
    assert d[wrapped] == "value"

    # Works in set
    s = {wrapped}
    assert text in s
    assert wrapped in s


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.lists(st.text(), min_size=0, max_size=20))
def test_newtype_set_deduplication(wrapper, texts: list[str]) -> None:
    """Set deduplication works correctly with NewType values.

    Subsumes: test_use_in_set with deduplication logic (4 spot checks).
    """
    wrapped_values = [wrapper(t) for t in texts]
    wrapped_set = set(wrapped_values)
    string_set = set(texts)

    # Same number of unique values
    assert len(wrapped_set) == len(string_set)


# =============================================================================
# Provider Enum Property Tests (replacing 9 spot checks with property tests)
# =============================================================================


@given(st.sampled_from(list(Provider)))
def test_provider_is_string_enum(provider: Provider) -> None:
    """Provider enum inherits from str - string operations work.

    Subsumes: test_is_string_enum, test_string_methods_available.
    """
    # Is a string
    assert isinstance(provider, str)

    # String value matches
    assert str(provider) == provider.value

    # String methods available
    assert provider.upper() == provider.value.upper()
    assert provider.lower() == provider.value.lower()


def test_provider_enum_values() -> None:
    """Provider enum has all expected values.

    Kept as explicit test - enumerates all known providers.
    """
    assert Provider.CHATGPT.value == "chatgpt"
    assert Provider.CLAUDE.value == "claude"
    assert Provider.CLAUDE_CODE.value == "claude-code"
    assert Provider.CODEX.value == "codex"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.DRIVE.value == "drive"
    assert Provider.UNKNOWN.value == "unknown"


# =============================================================================
# Provider.from_string() Property Tests (replacing 19 spot checks)
# =============================================================================


# Known valid provider values that should parse to their enum
PROVIDER_EXACT_MATCHES = [
    ("chatgpt", Provider.CHATGPT),
    ("claude", Provider.CLAUDE),
    ("claude-code", Provider.CLAUDE_CODE),
    ("codex", Provider.CODEX),
    ("gemini", Provider.GEMINI),
    ("drive", Provider.DRIVE),
]


@pytest.mark.parametrize("value,expected", PROVIDER_EXACT_MATCHES)
def test_provider_from_string_exact_match(value: str, expected: Provider) -> None:
    """Exact match returns correct enum."""
    assert Provider.from_string(value) == expected


# Alias mappings
PROVIDER_ALIASES = [
    # ChatGPT aliases
    ("gpt", Provider.CHATGPT),
    ("openai", Provider.CHATGPT),
    ("GPT", Provider.CHATGPT),
    ("OPENAI", Provider.CHATGPT),
    # Claude aliases
    ("claude-ai", Provider.CLAUDE),
    ("anthropic", Provider.CLAUDE),
    ("CLAUDE-AI", Provider.CLAUDE),
    ("ANTHROPIC", Provider.CLAUDE),
]


@pytest.mark.parametrize("alias,expected", PROVIDER_ALIASES)
def test_provider_from_string_alias(alias: str, expected: Provider) -> None:
    """Aliases map to correct provider."""
    assert Provider.from_string(alias) == expected


@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_case_insensitive(value: str) -> None:
    """Provider matching is case-insensitive.

    Subsumes: test_case_insensitive_match.
    """
    # Various case transformations should all work
    assert Provider.from_string(value.upper()).value == value.lower()
    assert Provider.from_string(value.lower()).value == value.lower()
    assert Provider.from_string(value.title()).value == value.lower()


@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_whitespace_stripped(value: str) -> None:
    """Whitespace is stripped before matching.

    Subsumes: test_whitespace_stripped.
    """
    # With various whitespace
    assert Provider.from_string(f"  {value}  ").value == value
    assert Provider.from_string(f"\t{value}\n").value == value
    assert Provider.from_string(f" {value}").value == value


@given(st.text().filter(lambda s: s.strip().lower() not in [
    "chatgpt", "claude", "claude-code", "codex", "gemini", "drive", "unknown",
    "gpt", "openai", "claude-ai", "anthropic"
]))
def test_provider_from_string_unknown_fallback(value: str) -> None:
    """Unknown provider strings return UNKNOWN.

    Subsumes: test_unknown_for_invalid.
    """
    result = Provider.from_string(value)
    assert result == Provider.UNKNOWN


@pytest.mark.parametrize("empty_value", [None, "", "   ", "\t\n"])
def test_provider_from_string_empty_returns_unknown(empty_value) -> None:
    """Empty/None values return UNKNOWN.

    Subsumes: test_none_returns_unknown, test_empty_string_returns_unknown,
    test_whitespace_only_returns_unknown.
    """
    assert Provider.from_string(empty_value) == Provider.UNKNOWN


# =============================================================================
# Interoperability Tests (kept - tests realistic usage patterns)
# =============================================================================


@given(st.data())
def test_all_id_types_as_dict_keys(data) -> None:
    """All ID types work together as dict keys.

    Subsumes: test_all_ids_as_dict_keys.
    """
    conv_str = data.draw(st.text(min_size=1, max_size=50), label="conv_str")
    msg_str = data.draw(st.text(min_size=1, max_size=50), label="msg_str")
    att_str = data.draw(st.text(min_size=1, max_size=50), label="att_str")
    # Ensure hash_str differs from conv_str to avoid dict key collision
    hash_str = data.draw(
        st.text(min_size=1, max_size=50).filter(lambda s: s != conv_str),
        label="hash_str"
    )

    cid = ConversationId(conv_str)
    mid = MessageId(msg_str)
    aid = AttachmentId(att_str)
    ch = ContentHash(hash_str)

    # Build a realistic data structure
    d = {
        cid: {"messages": [mid], "attachments": [aid]},
        ch: "dedup-info",
    }

    # Access works
    assert d[cid]["messages"][0] == mid
    assert d[ch] == "dedup-info"


@given(st.lists(st.text(), min_size=1, max_size=10))
def test_ids_in_list_and_set(texts: list[str]) -> None:
    """ID types work correctly in collections.

    Subsumes: test_ids_in_list_and_set.
    """
    cids = [ConversationId(t) for t in texts]

    # List preserves all
    assert len(cids) == len(texts)

    # Set deduplicates correctly
    unique_cids = set(cids)
    unique_texts = set(texts)
    assert len(unique_cids) == len(unique_texts)


@given(st.sampled_from(list(Provider)), st.text(min_size=1, max_size=50))
def test_provider_enum_interop_with_ids(provider: Provider, conv_str: str) -> None:
    """Provider enum works with ID types in data structures.

    Subsumes: test_provider_enum_interop_with_ids.
    """
    cid = ConversationId(conv_str)

    metadata = {"conversation_id": cid, "provider": provider}

    assert metadata["conversation_id"] == cid
    assert metadata["provider"] == provider
    assert str(metadata["provider"]) == provider.value
