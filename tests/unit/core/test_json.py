"""Tests for core JSON, environment, dates, versions, timestamps, repository, and projections.

Consolidated from:
- test_core_json.py
- test_core_utilities.py
- test_lib.py
- test_services.py
- test_timestamps.py
- test_projections.py
- test_types.py
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib import json as core_json
from polylogue.lib.provider_identity import normalize_provider_token
from polylogue.services import build_runtime_services
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId, Provider

# =============================================================================
# JSON DUMPS - STANDALONE FUNCTIONS
# =============================================================================


def test_dumps_custom_type_default_handler():
    """Custom default handler serializes custom types."""

    class CustomType:
        def __init__(self, value):
            self.value = value

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, CustomType):
            return {"custom": obj.value}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    payload_obj = CustomType(42)
    output = core_json.dumps(payload_obj, default=custom_handler)
    data = core_json.loads(output)
    assert data == {"custom": 42}


def test_dumps_custom_fallback_to_encoder():
    """When custom handler raises, built-in encoder handles known types like Decimal."""

    def custom_handler(obj: Any) -> Any:
        raise TypeError("Not handled")

    payload = {"decimal": Decimal("1.5")}
    output = core_json.dumps(payload, default=custom_handler)
    data = core_json.loads(output)
    assert data["decimal"] == 1.5


def test_dumps_custom_handler_takes_precedence_for_decimal():
    """Custom handlers can override Decimal serialization instead of falling through."""

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return f"decimal:{obj}"
        raise TypeError("Not handled")

    output = core_json.dumps({"decimal": Decimal("1.5")}, default=custom_handler)
    data = core_json.loads(output)
    assert data["decimal"] == "decimal:1.5"


def test_dumps_preserves_non_type_errors_from_custom_handler():
    """Only TypeError falls through to built-in encoding."""

    def custom_handler(obj: Any) -> Any:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        core_json.dumps({"decimal": Decimal("1.5")}, default=custom_handler)


def test_dumps_forwards_orjson_options():
    """Options are forwarded to orjson when the fast path succeeds."""
    payload = {"b": 1, "a": 2}
    output = core_json.dumps(payload, option=orjson.OPT_SORT_KEYS)
    assert output == '{"a":2,"b":1}'


def test_dumps_fallback_uses_stdlib_encoder_when_orjson_option_rejects_object():
    """The stdlib fallback still uses the combined encoder contract."""

    class CustomType:
        def __init__(self, value: int):
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
    data = core_json.loads(output)
    assert data == {"payload": {"custom": 7}}


# =============================================================================
# JSON EDGE CASES - STANDALONE FUNCTIONS
# =============================================================================



# =============================================================================
# ENCODER FALLBACK - STANDALONE FUNCTIONS
# =============================================================================


def test_encoder_unhandled_type_raises_type_error():
    """dumps raises TypeError when custom handler cannot handle the object."""

    class UnhandledType:
        pass

    def custom_handler(obj: Any) -> Any:
        raise TypeError("Not handled by custom handler")

    obj = UnhandledType()
    with pytest.raises(TypeError):
        core_json.dumps(obj, default=custom_handler)


def test_encoder_decimal_serialized_when_custom_fails():
    """Built-in encoder handles Decimal even when custom handler raises for other types."""

    def custom_handler(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.upper()
        raise TypeError("Not handled")

    payload = {"value": Decimal("1.5")}
    output = core_json.dumps(payload, default=custom_handler)
    data = core_json.loads(output)
    assert data["value"] == 1.5


# =============================================================================
# ENVIRONMENT VARIABLE TESTS - PARAMETRIZED
# =============================================================================


GET_ENV_CASES = [
    ("prefixed_precedence", "VOYAGE_API_KEY", {"POLYLOGUE_VOYAGE_API_KEY": "local-key", "VOYAGE_API_KEY": "global-key"}, "local-key", None, "POLYLOGUE_* variable takes precedence over unprefixed"),
    ("unprefixed_fallback", "VOYAGE_API_KEY", {"VOYAGE_API_KEY": "global-key"}, "global-key", None, "Falls back to unprefixed when prefixed not set"),
    ("default_value", "MISSING_VAR", {}, "fallback", "fallback", "Returns default when neither variable is set"),
    ("none_without_default", "TOTALLY_MISSING", {}, None, None, "Returns None when neither variable is set and no default given"),
    ("empty_string_fallback", "EMPTY", {"POLYLOGUE_EMPTY": "", "EMPTY": "real_value"}, "real_value", None, "Empty string values are treated as falsy (falls through)"),
]


@pytest.mark.parametrize("scenario,var_name,env_vars,expected,default_val,desc", GET_ENV_CASES)
def test_get_env_comprehensive(scenario, var_name, env_vars, expected, default_val, desc, monkeypatch):
    """Comprehensive get_env() tests."""
    from polylogue.lib.env import get_env

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    result = get_env(var_name, default_val) if default_val is not None else get_env(var_name)

    assert result == expected, f"Failed {desc}"


GET_ENV_MULTI_CASES = [
    ("prefixed_precedence", ("GOOGLE_API_KEY", "GEMINI_API_KEY"), {"POLYLOGUE_GOOGLE_API_KEY": "polylogue_key", "GOOGLE_API_KEY": "google_key", "GEMINI_API_KEY": "gemini_key"}, "polylogue_key", None, "Prefixed variable takes precedence over all"),
    ("unprefixed_primary", ("GOOGLE_API_KEY", "GEMINI_API_KEY"), {"GOOGLE_API_KEY": "google_key", "GEMINI_API_KEY": "gemini_key"}, "google_key", None, "Unprefixed primary takes precedence over alternatives"),
    ("alternative_fallback", ("GOOGLE_API_KEY", "GEMINI_API_KEY"), {"GEMINI_API_KEY": "gemini_key"}, "gemini_key", None, "Falls back to alternative when primary not set"),
    ("default_value", ("MISSING", "ALSO_MISSING"), {}, "default_val", "default_val", "Returns default when no variables are set"),
    ("single_var", ("SINGLE_KEY",), {"SINGLE_KEY": "single_value"}, "single_value", None, "Works with no alternative names provided"),
]


@pytest.mark.parametrize("scenario,var_names,env_vars,expected,default_val,desc", GET_ENV_MULTI_CASES)
def test_get_env_multi_comprehensive(scenario, var_names, env_vars, expected, default_val, desc, monkeypatch):
    """Comprehensive get_env_multi() tests."""
    from polylogue.lib.env import get_env_multi

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    result = get_env_multi(*var_names, default=default_val) if default_val is not None else get_env_multi(*var_names)

    assert result == expected, f"Failed {desc}"





async def test_repository(test_db):
    """Test ConversationRepository basic operations."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    from tests.infra.storage_records import DbFactory

    factory = DbFactory(test_db)
    factory.create_conversation(
        id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
    )

    # Test get
    conv = await repo.get("c1")
    assert conv is not None
    assert conv.id == "c1"
    assert len(conv.messages) == 1
    assert conv.messages[0].text == "hello world"

    # Test list
    lst = await repo.list()
    assert len(lst) == 1
    assert lst[0].id == "c1"


async def test_repository_get_includes_attachment_conversation_id(test_db):
    """ConversationRepository.get_eager() returns attachments with conversation_id field."""
    from tests.infra.storage_records import DbFactory

    factory = DbFactory(test_db)
    factory.create_conversation(
        id="c-with-att",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "message with attachment",
                "attachments": [
                    {
                        "id": "att1",
                        "mime_type": "image/png",
                        "size_bytes": 2048,
                        "path": "/path/to/image.png",
                    }
                ],
            }
        ],
    )

    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    conv = await repo.get_eager("c-with-att")

    assert conv is not None
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.id == "att1"
    assert att.mime_type == "image/png"


async def test_repository_get_with_multiple_attachments(test_db):
    """get_eager() correctly groups multiple attachments per message."""
    from tests.infra.storage_records import DbFactory

    factory = DbFactory(test_db)
    factory.create_conversation(
        id="c-multi-att",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "first message",
                "attachments": [
                    {"id": "att1", "mime_type": "image/png"},
                    {"id": "att2", "mime_type": "image/jpeg"},
                ],
            },
            {
                "id": "m2",
                "role": "assistant",
                "text": "second message",
                "attachments": [
                    {"id": "att3", "mime_type": "application/pdf"},
                ],
            },
        ],
    )

    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    conv = await repo.get_eager("c-multi-att")

    assert conv is not None
    assert len(conv.messages) == 2

    m1 = conv.messages[0]
    assert len(m1.attachments) == 2
    m1_att_ids = {a.id for a in m1.attachments}
    assert m1_att_ids == {"att1", "att2"}

    m2 = conv.messages[1]
    assert len(m2.attachments) == 1
    assert m2.attachments[0].id == "att3"


async def test_repository_get_attachment_metadata_decoded(test_db):
    """Attachment provider_meta JSON is properly decoded."""
    from tests.infra.storage_records import DbFactory

    factory = DbFactory(test_db)
    meta = {"original_name": "photo.png", "source": "upload"}
    factory.create_conversation(
        id="c-att-meta",
        provider="test",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "text": "with meta",
                "attachments": [
                    {
                        "id": "att-meta",
                        "mime_type": "image/png",
                        "meta": meta,
                    }
                ],
            }
        ],
    )

    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    conv = await repo.get_eager("c-att-meta")

    assert conv is not None
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.provider_meta == meta or att.provider_meta is None




# =============================================================================
# SERVICES TESTS
# =============================================================================


class TestRuntimeServices:
    def test_repository_is_cached_per_runtime_scope(self, workspace_env):
        services = build_runtime_services()
        repo1 = services.get_repository()
        repo2 = services.get_repository()
        assert repo1 is repo2

    def test_backend_is_cached_per_runtime_scope(self, workspace_env):
        services = build_runtime_services()
        backend1 = services.get_backend()
        backend2 = services.get_backend()
        assert backend1 is backend2

    def test_repository_uses_runtime_backend(self, workspace_env):
        services = build_runtime_services()
        repo = services.get_repository()
        assert repo.backend is services.get_backend()

    def test_distinct_runtime_scopes_do_not_share_instances(self, workspace_env):
        services1 = build_runtime_services()
        services2 = build_runtime_services()
        assert services1.get_repository() is not services2.get_repository()
        assert services1.get_backend() is not services2.get_backend()



# =============================================================================
# NEWTYPE & PROVIDER TESTS - PARAMETRIZED
# =============================================================================


# All NewType string wrappers share identical runtime behavior
NEWTYPE_WRAPPERS = [ConversationId, MessageId, AttachmentId, ContentHash]
NEWTYPE_IDS = ["ConversationId", "MessageId", "AttachmentId", "ContentHash"]


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_preserves_string_semantics(wrapper, text: str) -> None:
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
def test_newtype_equality_reflects_string(wrapper, a: str, b: str) -> None:
    """Equality between wrapped values reflects underlying string equality."""
    wrapped_a = wrapper(a)
    wrapped_b = wrapper(b)

    assert (wrapped_a == wrapped_b) == (a == b)
    assert (wrapped_a != wrapped_b) == (a != b)


@pytest.mark.parametrize("wrapper", NEWTYPE_WRAPPERS, ids=NEWTYPE_IDS)
@given(st.text())
def test_newtype_hashable_as_string(wrapper, text: str) -> None:
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
def test_newtype_set_deduplication(wrapper, texts: list[str]) -> None:
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
        "claude",
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
def test_provider_from_string_empty_returns_unknown(empty_value) -> None:
    """Empty/None values return UNKNOWN."""
    assert Provider.from_string(empty_value) == Provider.UNKNOWN


@given(st.data())
def test_all_id_types_as_dict_keys(data) -> None:
    """All ID types work together as dict keys."""
    conv_str = data.draw(st.text(min_size=1, max_size=50), label="conv_str")
    msg_str = data.draw(st.text(min_size=1, max_size=50), label="msg_str")
    att_str = data.draw(st.text(min_size=1, max_size=50), label="att_str")
    hash_str = data.draw(
        st.text(min_size=1, max_size=50).filter(lambda s: s != conv_str),
        label="hash_str"
    )

    cid = ConversationId(conv_str)
    mid = MessageId(msg_str)
    aid = AttachmentId(att_str)
    ch = ContentHash(hash_str)

    d = {
        cid: {"messages": [mid], "attachments": [aid]},
        ch: "dedup-info",
    }

    assert d[cid]["messages"][0] == mid
    assert d[ch] == "dedup-info"


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
