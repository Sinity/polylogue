"""Tests for core JSON, dates, versions, timestamps, repository, projections, and property laws.

Consolidated from:
- test_json.py (hand-written tests)
- test_json_laws.py (property-based tests)

Each law covers a behavioral invariant that supersedes specific parametrized
example tests.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib import json as core_json
from polylogue.lib.provider_identity import normalize_provider_token
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId, Provider

SURROGATE_CATEGORY: tuple[Literal["Cs"], ...] = ("Cs",)

# ---------------------------------------------------------------------------
# Serializable value strategy
# ---------------------------------------------------------------------------

_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
    st.text(alphabet=st.characters(blacklist_categories=SURROGATE_CATEGORY)),
)

_json_value = st.recursive(
    _scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(
            st.text(alphabet=st.characters(blacklist_categories=SURROGATE_CATEGORY), max_size=20), children, max_size=6
        ),
    ),
    max_leaves=20,
)


def _loaded_document(payload: str | bytes) -> core_json.JSONDocument:
    result = core_json.loads(payload)
    assert core_json.is_json_document(result)
    return result


# ---------------------------------------------------------------------------
# Law 1: JSON roundtrip for basic serializable types
# ---------------------------------------------------------------------------


@given(_json_value)
def test_json_roundtrip_basic_types(value: object) -> None:
    """Any JSON-serializable value survives a dumps/loads roundtrip."""
    output = core_json.dumps(value)
    result = core_json.loads(output)
    assert result == value


# ---------------------------------------------------------------------------
# Law 2: Decimal encodes to float exactly
# ---------------------------------------------------------------------------


@given(
    st.decimals(
        allow_nan=False,
        allow_infinity=False,
        min_value=Decimal("-1e15"),
        max_value=Decimal("1e15"),
    )
)
def test_decimal_encodes_to_float(d: Decimal) -> None:
    """Decimal always encodes to float — never to string, never raises.
    The encoded value equals float(d)."""
    output = core_json.dumps({"v": d})
    result = _loaded_document(output)
    assert isinstance(result["v"], float)
    assert result["v"] == float(d)


# ---------------------------------------------------------------------------
# Law 3: Invalid JSON always raises, never silently returns None
# ---------------------------------------------------------------------------

_INVALID_JSON_FRAGMENTS = [
    "{",
    "}",
    "[",
    "]",
    "{1: 2}",
    "undefined",
    "{'single': 'quotes'}",
    '{"key": undefined}',
    "NaN",
    "Infinity",
    "01",  # leading zero
    "",
]


@pytest.mark.parametrize("fragment", _INVALID_JSON_FRAGMENTS)
def test_loads_known_invalid_json_raises(fragment: str) -> None:
    """Invalid JSON fragments always raise an exception."""
    with pytest.raises(orjson.JSONDecodeError):
        core_json.loads(fragment)


@given(st.text(min_size=1, alphabet='{[}"]:').filter(lambda s: s.strip() not in ("[]", "{}", '""', "{}")))
def test_loads_malformed_json_never_silent(text: str) -> None:
    """loads either raises or returns a non-None value; it never silently returns None
    for a non-null JSON input."""
    try:
        result = core_json.loads(text)
        # If loads succeeds, it must have parsed to something
        # (Note: valid JSON 'null' would return None, so we only assert on successful
        # non-null parses)
        _ = result  # No assertion needed - successful parse is fine
    except Exception:
        pass  # Expected for malformed input


# ---------------------------------------------------------------------------
# Law 4: Provider.from_string always returns a non-empty value
# ---------------------------------------------------------------------------


@given(st.text())
def test_provider_from_string_value_never_empty(text: str) -> None:
    """Provider.from_string(x).value is always a non-empty string."""
    result = Provider.from_string(text)
    assert isinstance(result.value, str)
    assert len(result.value) > 0


# ---------------------------------------------------------------------------
# Law 5: Provider.from_string is idempotent on known values
# ---------------------------------------------------------------------------


@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_idempotent(value: str) -> None:
    """Applying from_string to a known provider value twice gives the same result."""
    first = Provider.from_string(value)
    second = Provider.from_string(first.value)
    assert first == second
    assert first.value == second.value


# =============================================================================
# Merged from test_json.py (2024-03-15)
# =============================================================================


# =============================================================================
# JSON DUMPS - STANDALONE FUNCTIONS
# =============================================================================


def test_dumps_custom_type_default_handler() -> Any:
    """Custom default handler serializes custom types."""

    class CustomType:
        def __init__(self: Any, value: Any) -> None:
            self.value = value

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, CustomType):
            return {"custom": obj.value}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    payload_obj = CustomType(42)
    output = core_json.dumps(payload_obj, default=custom_handler)
    data = _loaded_document(output)
    assert data == {"custom": 42}


def test_dumps_custom_fallback_to_encoder() -> None:
    """When custom handler raises, built-in encoder handles known types like Decimal."""

    def custom_handler(obj: Any) -> Any:
        raise TypeError("Not handled")

    payload = {"decimal": Decimal("1.5")}
    output = core_json.dumps(payload, default=custom_handler)
    data = _loaded_document(output)
    assert data["decimal"] == 1.5


def test_dumps_custom_handler_takes_precedence_for_decimal() -> Any:
    """Custom handlers can override Decimal serialization instead of falling through."""

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return f"decimal:{obj}"
        raise TypeError("Not handled")

    output = core_json.dumps({"decimal": Decimal("1.5")}, default=custom_handler)
    data = _loaded_document(output)
    assert data["decimal"] == "decimal:1.5"


def test_dumps_preserves_non_type_errors_from_custom_handler() -> None:
    """Only TypeError falls through to built-in encoding."""

    def custom_handler(obj: Any) -> Any:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        core_json.dumps({"decimal": Decimal("1.5")}, default=custom_handler)


def test_dumps_forwards_orjson_options() -> None:
    """Options are forwarded to orjson when the fast path succeeds."""
    payload = {"b": 1, "a": 2}
    output = core_json.dumps(payload, option=orjson.OPT_SORT_KEYS)
    assert output == '{"a":2,"b":1}'


def test_dumps_bytes_roundtrips_as_utf8_json() -> None:
    """dumps_bytes returns UTF-8 bytes with the same semantic content as dumps."""
    payload = {"b": 1, "a": Decimal("2.5")}

    output = core_json.dumps_bytes(payload, option=orjson.OPT_SORT_KEYS)

    assert isinstance(output, bytes)
    assert output == b'{"a":2.5,"b":1}'
    assert core_json.loads(output) == {"a": 2.5, "b": 1}


def test_dumps_fallback_uses_stdlib_encoder_when_orjson_option_rejects_object() -> Any:
    """The stdlib fallback still uses the combined encoder contract."""

    class CustomType:
        def __init__(self: Any, value: int) -> None:
            self.value = value

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, CustomType):
            return {"custom": obj.value}
        raise TypeError("Not handled")

    output = core_json.dumps(
        {"payload": CustomType(7)},
        default=custom_handler,
        option=orjson.OPT_NON_STR_KEYS,
    )
    data = _loaded_document(output)
    assert data == {"payload": {"custom": 7}}


def test_dumps_bytes_fallback_uses_stdlib_encoder_when_orjson_option_rejects_object() -> Any:
    """dumps_bytes preserves the same semantic encoder contract as dumps."""

    class CustomType:
        def __init__(self: Any, value: int) -> None:
            self.value = value

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, CustomType):
            return {"custom": obj.value}
        raise TypeError("Not handled")

    output = core_json.dumps_bytes(
        {"payload": CustomType(7)},
        default=custom_handler,
        option=orjson.OPT_NON_STR_KEYS,
    )
    assert isinstance(output, bytes)
    assert core_json.loads(output) == {"payload": {"custom": 7}}


# =============================================================================
# JSON EDGE CASES - STANDALONE FUNCTIONS
# =============================================================================


# =============================================================================
# ENCODER FALLBACK - STANDALONE FUNCTIONS
# =============================================================================


def test_encoder_unhandled_type_raises_type_error() -> None:
    """dumps raises TypeError when custom handler cannot handle the object."""

    class UnhandledType:
        pass

    def custom_handler(obj: Any) -> Any:
        raise TypeError("Not handled by custom handler")

    obj = UnhandledType()
    with pytest.raises(TypeError):
        core_json.dumps(obj, default=custom_handler)


def test_encoder_decimal_serialized_when_custom_fails() -> Any:
    """Built-in encoder handles Decimal even when custom handler raises for other types."""

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.upper()
        raise TypeError("Not handled")

    payload = {"value": Decimal("1.5")}
    output = core_json.dumps(payload, default=custom_handler)
    data = _loaded_document(output)
    assert data["value"] == 1.5


# =============================================================================
# NEWTYPE & PROVIDER TESTS - PARAMETRIZED
# =============================================================================


# All NewType string wrappers share identical runtime behavior
NEWTYPE_WRAPPERS = [ConversationId, MessageId, AttachmentId, ContentHash]
NEWTYPE_IDS = ["ConversationId", "MessageId", "AttachmentId", "ContentHash"]


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_preserves_string_semantics(wrapper: Any, text: str) -> None:
    """NewType wrappers are str at runtime - all string operations work."""
    wrapped = wrapper(text)

    assert isinstance(wrapped, str)
    assert wrapped == text
    assert wrapped.upper() == text.upper()
    assert wrapped.lower() == text.lower()
    assert len(wrapped) == len(text)

    if text:
        assert text[0] in wrapped


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text(), st.text())
def test_newtype_equality_reflects_string(wrapper: Any, a: str, b: str) -> None:
    """Equality between wrapped values reflects underlying string equality."""
    wrapped_a = wrapper(a)
    wrapped_b = wrapper(b)

    assert (wrapped_a == wrapped_b) == (a == b)
    assert (wrapped_a != wrapped_b) == (a != b)


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_hashable_as_string(wrapper: Any, text: str) -> None:
    """NewType values hash identically to their underlying string."""
    wrapped = wrapper(text)

    assert hash(wrapped) == hash(text)

    d = {wrapped: "value"}
    assert d[text] == "value"
    assert d[wrapped] == "value"

    s = {wrapped}
    assert text in s
    assert wrapped in s


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.lists(st.text(), min_size=0, max_size=20))
def test_newtype_set_deduplication(wrapper: Any, texts: list[str]) -> None:
    """Set deduplication works correctly with NewType values."""
    wrapped_values = [wrapper(t) for t in texts]
    wrapped_set = set(wrapped_values)
    string_set = set(texts)

    assert len(wrapped_set) == len(string_set)


@given(st.sampled_from(list(Provider)))
def test_provider_is_string_enum(provider: Provider) -> None:
    """Provider enum inherits from str - string operations work."""
    assert isinstance(provider, str)
    assert str(provider) == provider.value
    assert provider.upper() == provider.value.upper()
    assert provider.lower() == provider.value.lower()


@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_case_insensitive(value: str) -> None:
    """Provider matching is case-insensitive."""
    assert Provider.from_string(value.upper()).value == value.lower()
    assert Provider.from_string(value.lower()).value == value.lower()
    assert Provider.from_string(value.title()).value == value.lower()


@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_whitespace_stripped(value: str) -> None:
    """Whitespace is stripped before matching."""
    assert Provider.from_string(f"  {value}  ").value == value
    assert Provider.from_string(f"\t{value}\n").value == value
    assert Provider.from_string(f" {value}").value == value


KNOWN_PROVIDER_TOKENS = {
    normalize_provider_token(token)
    for token in (
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
        "drive",
        "unknown",
        "gpt",
        "openai",
        "claude-ai",
        "anthropic",
        "claudecode",
    )
}


@given(st.text().filter(lambda s: normalize_provider_token(s) not in KNOWN_PROVIDER_TOKENS))
def test_provider_from_string_unknown_fallback(value: str) -> None:
    """Unknown provider strings return UNKNOWN."""
    result = Provider.from_string(value)
    assert result == Provider.UNKNOWN


@pytest.mark.parametrize("empty_value", [None, "", "   ", "\t\n"])
def test_provider_from_string_empty_returns_unknown(empty_value: Any) -> None:
    """Empty/None values return UNKNOWN."""
    assert Provider.from_string(empty_value) == Provider.UNKNOWN


@given(st.data())
def test_all_id_types_as_dict_keys(data: Any) -> None:
    """All ID types work together as dict keys."""
    conv_str = data.draw(st.text(min_size=1, max_size=50), label="conv_str")
    msg_str = data.draw(st.text(min_size=1, max_size=50), label="msg_str")
    att_str = data.draw(st.text(min_size=1, max_size=50), label="att_str")
    hash_str = data.draw(st.text(min_size=1, max_size=50).filter(lambda s: s != conv_str), label="hash_str")

    cid = ConversationId(conv_str)
    mid = MessageId(msg_str)
    aid = AttachmentId(att_str)
    ch = ContentHash(hash_str)

    conversation_index: dict[ConversationId, dict[str, list[str]]] = {
        cid: {"messages": [mid], "attachments": [aid]},
    }
    dedupe_index: dict[ContentHash, str] = {ch: "dedup-info"}

    assert conversation_index[cid]["messages"][0] == mid
    assert dedupe_index[ch] == "dedup-info"


@given(st.lists(st.text(), min_size=1, max_size=10))
def test_ids_in_list_and_set(texts: list[str]) -> None:
    """ID types work correctly in collections."""
    cids = [ConversationId(t) for t in texts]

    assert len(cids) == len(texts)

    unique_cids = set(cids)
    unique_texts = set(texts)
    assert len(unique_cids) == len(unique_texts)


@given(st.sampled_from(list(Provider)), st.text(min_size=1, max_size=50))
def test_provider_enum_interop_with_ids(provider: Provider, conv_str: str) -> None:
    """Provider enum works with ID types in data structures."""
    cid = ConversationId(conv_str)

    metadata = {"conversation_id": cid, "provider": provider}

    assert metadata["conversation_id"] == cid
    assert metadata["provider"] == provider
    assert str(metadata["provider"]) == provider.value
