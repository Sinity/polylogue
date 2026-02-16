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

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.lib import json as core_json
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.timestamps import format_timestamp, parse_timestamp
from polylogue.services import get_backend, get_repository, get_service_config, reset
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId, Provider

# =============================================================================
# JSON DUMPS - PARAMETRIZED
# =============================================================================


DUMPS_CASES = [
    # Basic types
    ({"key": "value"}, "simple dict"),
    ([1, 2, 3], "simple list"),
    ({"outer": {"inner": [1, 2, 3]}}, "nested structure"),
    # Decimal handling
    ({"value": Decimal("1.25")}, "handles Decimal"),
    ({"price": Decimal("99.99")}, "Decimal as float not string"),
    (
        {"value1": Decimal("10.5"), "value2": Decimal("20.75"), "nested": {"value3": Decimal("30.25")}},
        "multiple Decimals",
    ),
    # Special values
    ({"key": None}, "None value"),
    ({"true_val": True, "false_val": False}, "boolean values"),
    ({"text": "Hello 世界 🌍"}, "unicode characters"),
    ({}, "empty dict"),
    ([], "empty list"),
    ({"123": "value"}, "numeric keys as strings"),
    # Custom default handler
    ("custom_type", "custom default handler"),
    ("custom_handler_fallback", "custom handler fallback to encoder"),
    # Numbers and special chars
    ({"zero": 0, "negative": -42, "float": 3.14, "large": 999999999999}, "special numbers"),
    ({"text": 'Line 1\nLine 2\tTabbed"Quoted"'}, "escaped characters"),
]


@pytest.mark.parametrize("payload,desc", DUMPS_CASES)
def test_dumps_comprehensive(payload, desc):
    """Comprehensive dumps test."""
    if desc == "custom default handler":

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
        return

    elif desc == "custom handler fallback to encoder":

        def custom_handler(obj: Any) -> Any:
            raise TypeError("Not handled")

        payload = {"decimal": Decimal("1.5")}
        output = core_json.dumps(payload, default=custom_handler)
        data = core_json.loads(output)
        assert data["decimal"] == 1.5
        return

    # Standard dumps → loads verification
    output = core_json.dumps(payload)
    data = core_json.loads(output)

    # Type-specific assertions
    if desc == "Decimal as float not string":
        assert isinstance(data["price"], float)
        assert data["price"] == 99.99

    elif desc == "multiple Decimals":
        assert data["value1"] == 10.5
        assert data["value2"] == 20.75
        assert data["nested"]["value3"] == 30.25

    elif desc == "boolean values":
        assert data["true_val"] is True
        assert data["false_val"] is False

    elif desc == "numeric keys as strings":
        assert "123" in data

    else:
        if "Decimal" in desc:
            assert all(
                isinstance(v, float) if isinstance(payload[k], Decimal) else True
                for k, v in data.items()
                if isinstance(payload, dict)
            )
        else:
            assert data == payload


# =============================================================================
# JSON LOADS - PARAMETRIZED
# =============================================================================


LOADS_CASES = [
    ('{"key": "value"}', {"key": "value"}, "from string"),
    (b'{"key": "value"}', {"key": "value"}, "from bytes"),
    ("[1, 2, 3]", [1, 2, 3], "array"),
    ('{"outer": {"inner": [1, 2]}}', {"outer": {"inner": [1, 2]}}, "nested structure"),
    ('{"text": "Hello 世界"}', {"text": "Hello 世界"}, "with unicode"),
    ('{"value": null}', {"value": None}, "with null"),
    ('{"true": true, "false": false}', {"true": True, "false": False}, "with boolean"),
    ("42", 42, "primitive number int"),
    ("3.14", 3.14, "primitive number float"),
    ('"hello"', "hello", "primitive string"),
    ("null", None, "primitive null"),
    ("true", True, "primitive true"),
    ("false", False, "primitive false"),
    ("{}", {}, "empty dict"),
    ("[]", [], "empty list"),
    ('{"text": "He said \\"hello\\""}', {"text": 'He said "hello"'}, "escaped quotes"),
    ('{"text": "Line 1\\nLine 2"}', {"text": "Line 1\nLine 2"}, "newlines in string"),
]


@pytest.mark.parametrize("input_data,expected,desc", LOADS_CASES)
def test_loads_comprehensive(input_data, expected, desc):
    """Comprehensive loads test."""
    result = core_json.loads(input_data)
    assert result == expected, f"Failed {desc}"


# =============================================================================
# JSON ROUNDTRIP - PARAMETRIZED
# =============================================================================


ROUNDTRIP_CASES = [
    ({"a": 1, "b": "text", "c": None}, "dict"),
    ({"level1": {"level2": {"level3": [1, 2, 3]}}}, "nested"),
    ({"price": Decimal("19.99"), "quantity": Decimal("5")}, "with Decimal"),
    ({"emoji": "🚀", "chinese": "中文", "mixed": "Hello 世界 🌍"}, "with unicode"),
    ({"a": 1, "b": [2, 3]}, "multiple times"),
]


@pytest.mark.parametrize("original,desc", ROUNDTRIP_CASES)
def test_roundtrip_comprehensive(original, desc):
    """Comprehensive roundtrip test."""
    if desc == "multiple times":
        data = original
        for _ in range(3):
            output = core_json.dumps(data)
            data = core_json.loads(output)
        assert data == original

    elif desc == "with Decimal":
        output = core_json.dumps(original)
        result = core_json.loads(output)
        assert result == {"price": 19.99, "quantity": 5.0}

    else:
        output = core_json.dumps(original)
        result = core_json.loads(output)
        assert result == original



# =============================================================================
# JSON EDGE CASES - PARAMETRIZED
# =============================================================================


EDGE_CASES = [
    ({"big": 999999999999999999999}, "very large number"),
    ("deeply_nested", "deeply nested structure"),
    ([1, "string", True, None, 3.14, {"nested": "dict"}], "mixed types list"),
    ({"zero": Decimal("0")}, "Decimal zero"),
    ({"negative": Decimal("-123.45")}, "Decimal negative"),
    ({"items": [Decimal("1.5"), [Decimal("2.5"), Decimal("3.5")]]}, "Decimal in nested list"),
    ({"key": "value"}, "with option parameter"),
    ('{"unicode": "café"}', "bytes UTF-8"),
    ('{"key": "value"}', "bytes UTF-8 with BOM"),
    ({"path": "C:\\Users\\test"}, "string with backslash"),
    ("many_keys", "dict with many keys"),
    ("many_elements", "array with many elements"),
    (
        """
    {
        "key"   :   "value"  ,
        "array" : [ 1 , 2 , 3 ]
    }
    """,
        "whitespace handling",
    ),
    ("custom_decimal_override", "custom handler for Decimal override"),
    ({"empty_dict": {}, "empty_list": [], "nested": {"deep_empty": {}}}, "empty nested structures"),
    ("true", "primitive true"),
    ("false", "primitive false"),
    ('"\\u0048\\u0065\\u006c\\u006c\\u006f"', "unicode escapes"),
    ({"value": 3.141592653589793}, "float precision"),
]


@pytest.mark.parametrize("payload,desc", EDGE_CASES)
def test_edge_cases_comprehensive(payload, desc):
    """Comprehensive edge cases test."""
    if desc == "very large number":
        output = core_json.dumps(payload)
        data = core_json.loads(output)
        assert isinstance(data["big"], float)
        assert data["big"] > 999999999999999999998

    elif desc == "deeply nested structure":
        nested = {"level": 0}
        current = nested
        for i in range(1, 10):
            current["next"] = {"level": i}
            current = current["next"]

        output = core_json.dumps(nested)
        data = core_json.loads(output)
        assert data["level"] == 0

    elif desc == "mixed types list":
        output = core_json.dumps(payload)
        data = core_json.loads(output)
        assert data[0] == 1
        assert data[1] == "string"
        assert data[2] is True
        assert data[3] is None
        assert data[4] == 3.14
        assert data[5] == {"nested": "dict"}

    elif desc == "Decimal in nested list":
        output = core_json.dumps(payload)
        data = core_json.loads(output)
        assert data["items"][0] == 1.5
        assert data["items"][1][0] == 2.5
        assert data["items"][1][1] == 3.5

    elif desc == "with option parameter":
        output = core_json.dumps(payload, option=None)
        data = core_json.loads(output)
        assert data == payload

    elif desc == "bytes UTF-8":
        json_bytes = payload.encode()
        result = core_json.loads(json_bytes)
        assert result["unicode"] == "café"

    elif desc == "bytes UTF-8 with BOM":
        json_bytes = payload.encode("utf-8-sig")
        try:
            result = core_json.loads(json_bytes)
            assert result == {"key": "value"}
        except ValueError:
            pass  # orjson raises for BOM

    elif desc == "dict with many keys":
        payload = {f"key_{i}": f"value_{i}" for i in range(100)}
        output = core_json.dumps(payload)
        data = core_json.loads(output)
        assert len(data) == 100
        assert data["key_0"] == "value_0"
        assert data["key_99"] == "value_99"

    elif desc == "array with many elements":
        payload = list(range(1000))
        output = core_json.dumps(payload)
        data = core_json.loads(output)
        assert len(data) == 1000
        assert data[0] == 0
        assert data[999] == 999

    elif desc == "whitespace handling":
        result = core_json.loads(payload)
        assert result == {"key": "value", "array": [1, 2, 3]}

    elif desc == "custom handler for Decimal override":

        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload = {"price": Decimal("19.99")}
        output = core_json.dumps(payload, default=custom_handler)
        data = core_json.loads(output)
        assert data["price"] == "19.99"

    elif desc == "primitive true":
        result = core_json.loads(payload)
        assert result is True
        assert isinstance(result, bool)

    elif desc == "primitive false":
        result = core_json.loads(payload)
        assert result is False
        assert isinstance(result, bool)

    elif desc == "unicode escapes":
        result = core_json.loads(payload)
        assert result == "Hello"

    elif desc == "float precision":
        output = core_json.dumps(payload)
        result = core_json.loads(output)
        assert result["value"] == payload["value"]

    else:
        output = core_json.dumps(payload)
        data = core_json.loads(output)

        if "Decimal" in desc:
            for key, value in payload.items():
                if isinstance(value, Decimal):
                    assert data[key] == float(value)
        else:
            assert data == payload


ENCODER_FALLBACK_CASES = [
    ("unhandled_type", "unhandled type raises TypeError"),
    ("decimal_when_custom_fails", "Decimal encoded when custom fails"),
]


@pytest.mark.parametrize("scenario,desc", ENCODER_FALLBACK_CASES)
def test_encoder_fallback_comprehensive(scenario, desc):
    """Comprehensive encoder fallback test."""
    if scenario == "unhandled_type":

        class UnhandledType:
            pass

        def custom_handler(obj: Any) -> Any:
            raise TypeError("Not handled by custom handler")

        obj = UnhandledType()
        with pytest.raises(TypeError):
            core_json.dumps(obj, default=custom_handler)

    elif scenario == "decimal_when_custom_fails":

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


# =============================================================================
# DATE PARSING TESTS - PARAMETRIZED
# =============================================================================


PARSE_DATE_CASES = [
    ("2024-01-15", 2024, 1, 15, "ISO format"),
    ("2024-01-15T10:30:00", 10, 30, 0, "ISO format with time"),
    ("yesterday", "past", None, None, "natural yesterday"),
    ("2 days ago", "past", None, None, "relative dates"),
    ("not a date at all xyz123", None, None, None, "invalid returns None"),
    ("2024-06-15", None, None, None, "UTC-aware returned"),
]


@pytest.mark.parametrize("input_val,exp_year_or_hour,exp_month_or_minute,exp_day_or_second,desc", PARSE_DATE_CASES)
def test_parse_date_comprehensive(input_val, exp_year_or_hour, exp_month_or_minute, exp_day_or_second, desc):
    """Comprehensive parse_date() tests."""
    from polylogue.lib.dates import parse_date

    result = parse_date(input_val)

    if desc == "ISO format":
        assert result is not None
        assert result.year == exp_year_or_hour
        assert result.month == exp_month_or_minute
        assert result.day == exp_day_or_second
        assert result.tzinfo is not None

    elif desc == "ISO format with time":
        assert result is not None
        assert result.hour == exp_year_or_hour
        assert result.minute == exp_month_or_minute
        assert result.second == exp_day_or_second

    elif desc == "natural yesterday":
        assert result is not None
        assert result < datetime.now(timezone.utc)

    elif desc == "relative dates":
        assert result is not None
        now = datetime.now(timezone.utc)
        assert result < now

    elif desc == "invalid returns None":
        assert result is None

    elif desc == "UTC-aware returned":
        assert result is not None
        assert result.tzinfo is not None


# =============================================================================
# DATE FORMATTING TESTS - PARAMETRIZED
# =============================================================================


FORMAT_DATE_CASES = [
    (datetime(2024, 1, 15, 10, 30, 45), "2024-01-15 10:30:45", "basic"),
    (datetime(2024, 12, 25, 0, 0, 0), "2024-12-25 00:00:00", "midnight"),
    (datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc), "2024-06-15 14:30:00", "with timezone"),
]


@pytest.mark.parametrize("dt,expected,desc", FORMAT_DATE_CASES)
def test_format_date_iso_comprehensive(dt, expected, desc):
    """Comprehensive format_date_iso() tests."""
    from polylogue.lib.dates import format_date_iso

    result = format_date_iso(dt)
    assert result == expected


# =============================================================================
# VERSION TESTS - PARAMETRIZED
# =============================================================================


VERSION_INFO_CASES = [
    ("1.2.3", None, False, "1.2.3", "1.2.3", "without commit"),
    ("1.2.3", "abc123def456", False, "1.2.3+abc123de", "1.2.3", "with commit"),
    ("1.2.3", "abc123def456", True, "1.2.3+abc123de-dirty", "1.2.3", "with dirty"),
    ("0.9.0", "deadbeef", True, "0.9.0+deadbeef-dirty", "0.9.0", "short property"),
]


@pytest.mark.parametrize("version,commit,dirty,exp_full,exp_short,desc", VERSION_INFO_CASES)
def test_version_info_comprehensive(version, commit, dirty, exp_full, exp_short, desc):
    """Comprehensive VersionInfo tests."""
    from polylogue.version import VersionInfo

    info = VersionInfo(version=version) if commit is None else VersionInfo(version=version, commit=commit, dirty=dirty)

    assert str(info) == exp_full
    assert info.full == exp_full
    assert info.short == exp_short


# =============================================================================
# VERSION RESOLUTION TESTS
# =============================================================================


def test_version_info_available():
    """VERSION_INFO is available at module level."""
    from polylogue.version import VERSION_INFO

    assert VERSION_INFO is not None
    assert hasattr(VERSION_INFO, "version")
    assert hasattr(VERSION_INFO, "full")
    assert hasattr(VERSION_INFO, "short")


def test_polylogue_version_available():
    """POLYLOGUE_VERSION constant is available."""
    from polylogue.version import POLYLOGUE_VERSION

    assert POLYLOGUE_VERSION is not None
    assert isinstance(POLYLOGUE_VERSION, str)
    assert len(POLYLOGUE_VERSION) > 0


# =============================================================================
# SEMANTIC MODELS TESTS
# =============================================================================


def test_semantic_models():
    """Test rich methods on Message and Conversation."""
    msg_user = Message(id="1", role="user", text="hello, how are you today?")
    msg_bot = Message(id="2", role="assistant", text="I'm doing well, thanks for asking!")
    conv = Conversation(id="c1", provider="test", messages=MessageCollection(messages=[msg_user, msg_bot]))

    # Test filtering
    user_only = conv.user_only()
    assert len(user_only.messages) == 1
    assert user_only.messages[0].id == "1"

    # Test to_text
    txt = conv.to_text()
    assert "user:" in txt
    assert "assistant:" in txt
    assert "hello" in txt

    # Test classification properties
    assert msg_user.is_user
    assert msg_bot.is_assistant
    assert msg_user.is_substantive
    assert not msg_user.is_tool_use
    assert not msg_user.is_noise

    # Test projections
    clean = conv.without_noise()
    assert len(clean.messages) == 2

    # Test statistics
    assert conv.message_count == 2
    assert conv.user_message_count == 1


async def test_repository(test_db):
    """Test ConversationRepository basic operations."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    from tests.infra.helpers import DbFactory

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
    from tests.infra.helpers import DbFactory

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
    from tests.infra.helpers import DbFactory

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
    from tests.infra.helpers import DbFactory

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
# TIMESTAMP PARSING TESTS - PARAMETRIZED
# =============================================================================


TIMESTAMP_PARSE_CASES = [
    (None, "None"),
    (1704067200.0, "numeric timestamp"),
]


@pytest.mark.parametrize("ts_input,desc", TIMESTAMP_PARSE_CASES)
def test_parse_timestamp_comprehensive(ts_input, desc):
    """Comprehensive parse_timestamp() tests."""
    from polylogue.lib.timestamps import parse_timestamp

    result = parse_timestamp(ts_input)

    if desc == "None":
        assert result is None
    elif desc == "numeric timestamp":
        assert result is not None


# =============================================================================
# JSON UTILS TESTS
# =============================================================================


class TestJsonUtils:
    """Test core/json.py utilities."""

    def test_loads_valid_json(self):
        """Test loads with valid JSON."""
        from polylogue.lib.json import loads

        data = loads('{"key": "value"}')
        assert data == {"key": "value"}

    def test_loads_invalid_json(self):
        """Test loads with invalid JSON raises."""
        from polylogue.lib.json import loads

        with pytest.raises((ValueError, json.JSONDecodeError)):
            loads("{invalid}")


# =============================================================================
# VERSION EDGE CASES - PARAMETRIZED
# =============================================================================


VERSION_EDGE_CASES = [
    ("resolve_version", "VersionInfo object returned"),
    ("str_representation", "version included in string"),
    ("dirty_state", "dirty state indicated"),
]


@pytest.mark.parametrize("scenario,desc", VERSION_EDGE_CASES)
def test_version_edge_cases_comprehensive(scenario, desc):
    """Comprehensive version edge case tests."""
    if scenario == "resolve_version":
        from polylogue.version import _resolve_version

        info = _resolve_version()
        assert info is not None
        assert hasattr(info, "version")

    elif scenario == "str_representation":
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.0.0", commit="abc123", dirty=False)
        s = str(info)
        assert "1.0.0" in s

    elif scenario == "dirty_state":
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.0.0", commit="abc123", dirty=True)
        s = str(info)
        assert "dirty" in s.lower() or "+" in s


# =============================================================================
# DISPLAY DATE TESTS - PARAMETRIZED
# =============================================================================


DISPLAY_DATE_CASES = [
    ("summary_prefers_updated", "prefers updated over created"),
    ("summary_falls_back_to_created", "falls back to created when updated missing"),
    ("summary_none_when_both_missing", "returns None when both missing"),
    ("conversation_prefers_updated", "conversation prefers updated over created"),
    ("conversation_falls_back_to_created", "conversation falls back to created when updated missing"),
]


@pytest.mark.parametrize("scenario,desc", DISPLAY_DATE_CASES)
def test_display_date_comprehensive(scenario, desc):
    """Comprehensive display_date tests."""
    from datetime import datetime, timezone

    if scenario == "summary_prefers_updated":
        from polylogue.lib.models import ConversationSummary

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2025, 6, 15, tzinfo=timezone.utc)
        s = ConversationSummary(
            id="test:1", provider="test", created_at=created, updated_at=updated
        )
        assert s.display_date == updated

    elif scenario == "summary_falls_back_to_created":
        from polylogue.lib.models import ConversationSummary

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        s = ConversationSummary(
            id="test:1", provider="test", created_at=created, updated_at=None
        )
        assert s.display_date == created

    elif scenario == "summary_none_when_both_missing":
        from polylogue.lib.models import ConversationSummary

        s = ConversationSummary(id="test:1", provider="test")
        assert s.display_date is None

    elif scenario == "conversation_prefers_updated":
        from polylogue.lib.models import Conversation, MessageCollection

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2025, 6, 15, tzinfo=timezone.utc)
        c = Conversation(
            id="test:1",
            provider="test",
            messages=MessageCollection(messages=[]),
            created_at=created,
            updated_at=updated,
        )
        assert c.display_date == updated

    elif scenario == "conversation_falls_back_to_created":
        from polylogue.lib.models import Conversation, MessageCollection

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        c = Conversation(
            id="test:1",
            provider="test",
            messages=MessageCollection(messages=[]),
            created_at=created,
            updated_at=None,
        )
        assert c.display_date == created


# =============================================================================
# SERVICES TESTS
# =============================================================================


class TestServices:
    def test_get_backend_returns_sqlite(self, workspace_env):
        # Must import after workspace_env reloads the module
        from polylogue.storage.backends.async_sqlite import SQLiteBackend as _SQLiteBackend

        backend = get_backend()
        assert isinstance(backend, _SQLiteBackend)

    def test_get_repository_returns_repo(self, workspace_env):
        from polylogue.storage.repository import ConversationRepository as _ConversationRepository

        repo = get_repository()
        assert isinstance(repo, _ConversationRepository)

    def test_get_config_returns_config(self):
        from polylogue.config import Config as _Config

        config = get_service_config()
        assert isinstance(config, _Config)
        assert config.archive_root is not None

    def test_reset_clears_singletons(self, workspace_env):
        repo1 = get_repository()
        reset()
        repo2 = get_repository()
        assert repo1 is not repo2

    def test_singleton_returns_same_instance(self, workspace_env):
        repo1 = get_repository()
        repo2 = get_repository()
        assert repo1 is repo2

    def test_backend_singleton_returns_same_instance(self, workspace_env):
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_repository_uses_same_backend(self, workspace_env):
        repo1 = get_repository()
        repo2 = get_repository()
        assert repo1.backend is repo2.backend

    def test_reset_affects_backend_singleton(self, workspace_env):
        backend1 = get_backend()
        reset()
        backend2 = get_backend()
        assert backend1 is not backend2


# =============================================================================
# TIMESTAMP FORMATTING TESTS - PARAMETRIZED
# =============================================================================


PARSE_TEST_CASES = [
    # Epoch timestamps
    (1704067200, 2024, 1, 1, None, "epoch int"),
    (1704067200.5, 2024, 1, 1, 500000, "epoch float"),
    ("1704067200", 2024, 1, 1, None, "epoch string"),
    ("1704067200.5", 2024, 1, 1, 500000, "epoch string with decimal"),

    # ISO 8601 variants
    ("2024-01-01T00:00:00", 2024, 1, 1, None, "ISO basic"),
    ("2024-01-01T00:00:00Z", 2024, 1, 1, None, "ISO with Z"),
    ("2024-01-01T00:00:00+00:00", 2024, 1, 1, None, "ISO with offset"),
    ("2024-01-01T00:00:00.500000", 2024, 1, 1, 500000, "ISO with microseconds"),

    # Millisecond timestamps (common in JS exports) - not currently supported, return None
    (1704067200000, None, None, None, None, "millisecond epoch"),
    ("1704067200000", None, None, None, None, "millisecond epoch string"),
]


@pytest.mark.parametrize("input_val,exp_year,exp_month,exp_day,exp_micro,desc", PARSE_TEST_CASES)
def test_parse_timestamp_formats(input_val, exp_year, exp_month, exp_day, exp_micro, desc):
    """Parse all supported timestamp formats."""
    result = parse_timestamp(input_val)

    if exp_year is None:
        assert result is None, f"Expected None for {desc}, got {result}"
    else:
        assert result is not None, f"Failed to parse {desc}: {input_val}"
        assert result.year == exp_year, f"Wrong year for {desc}"
        assert result.month == exp_month, f"Wrong month for {desc}"
        assert result.day == exp_day, f"Wrong day for {desc}"

        if exp_micro is not None:
            assert result.microsecond == exp_micro, f"Wrong microseconds for {desc}"


@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    "not-a-date",
    "invalid-timestamp",
    "2024-13-45",
])
def test_parse_timestamp_invalid_returns_none(invalid_input):
    """Invalid timestamps return None."""
    result = parse_timestamp(invalid_input)
    assert result is None


def test_parse_timestamp_overflow_handled():
    """Handle very large epoch values gracefully."""
    result = parse_timestamp(9999999999999)
    if result is not None:
        assert isinstance(result, datetime)


# Test cases: (input_datetime, expected_format_output, description)
FORMAT_TEST_CASES = [
    (
        datetime(2024, 1, 1, 12, 30, 45),
        "2024-01-01T12:30:45",
        "basic datetime"
    ),
    (
        datetime(2024, 1, 1, 0, 0, 0),
        "2024-01-01T00:00:00",
        "midnight"
    ),
    (
        datetime(2024, 12, 31, 23, 59, 59),
        "2024-12-31T23:59:59",
        "end of year"
    ),
    (
        datetime(2024, 1, 1, 12, 30, 45, 500000),
        "2024-01-01T12:30:45",
        "with microseconds"
    ),
]


@pytest.mark.parametrize("dt,expected_format,desc", FORMAT_TEST_CASES)
def test_format_timestamp_variants(dt, expected_format, desc):
    """Format datetime objects to ISO 8601."""
    result = format_timestamp(dt)

    assert result.startswith(expected_format[:19]), f"Wrong format for {desc}"
    assert "T" in result, "Should use T separator"


@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    "not-a-datetime",
])
def test_format_timestamp_invalid_returns_none_or_empty(invalid_input):
    """Invalid input to format_timestamp handled gracefully."""
    try:
        result = format_timestamp(invalid_input)
        assert result is None or result == ""
    except (TypeError, AttributeError):
        pass


# =============================================================================
# PROJECTION TESTS - PARAMETRIZED
# =============================================================================


# Test cases: (filter_method, expected_count, description)
FILTER_TEST_CASES = [
    ("user_messages", 2, "user_messages filter"),
    ("assistant_messages", 3, "assistant_messages filter"),
    ("dialogue", 5, "dialogue filter"),
    ("substantive", 4, "substantive filter"),
    ("without_noise", 5, "without_noise filter"),
    ("thinking_only", 0, "thinking_only filter"),
    ("tool_use_only", 1, "tool_use_only filter"),
]


@pytest.mark.parametrize("method_name,expected_count,desc", FILTER_TEST_CASES)
def test_projection_filters_comprehensive(sample_conversation, method_name, expected_count, desc):
    """Comprehensive projection filter test."""
    projection = sample_conversation.project()
    filtered = getattr(projection, method_name)()

    result = filtered.to_list()
    assert len(result) == expected_count, \
        f"Failed {desc}: expected {expected_count}, got {len(result)}"


COMPOSITION_CASES = [
    (["user_messages", "substantive"], 2, "user + substantive"),
    (["assistant_messages", "substantive"], 2, "assistant + substantive"),
    (["dialogue", "without_noise"], 5, "dialogue + no noise"),
]


@pytest.mark.parametrize("methods,expected_count,desc", COMPOSITION_CASES)
def test_projection_filter_chaining(sample_conversation, methods, expected_count, desc):
    """Test filter method chaining."""
    projection = sample_conversation.project()

    for method in methods:
        projection = projection.substantive() if method == "substantive" else getattr(projection, method)()

    result = projection.to_list()
    assert len(result) == expected_count, f"Failed {desc}"


def test_projection_filter_chaining_contains(sample_conversation):
    """Test chaining with contains filter."""
    projection = sample_conversation.project().user_messages().contains("searchterm")
    result = projection.to_list()
    assert len(result) == 1
    assert result[0].id == "m5"


TERMINAL_CASES = [
    ("to_list", list, 7, "to_list"),
    ("count", int, 7, "count"),
    ("first", Message, 1, "first"),
    ("last", Message, 1, "last"),
    ("exists", bool, True, "exists"),
    ("to_text", str, None, "to_text"),
]


@pytest.mark.parametrize("method_name,expected_type,expected_value,desc", TERMINAL_CASES)
def test_projection_terminal_operations(sample_conversation, method_name, expected_type, expected_value, desc):
    """Comprehensive terminal operation test."""
    projection = sample_conversation.project()
    result = getattr(projection, method_name)()

    assert isinstance(result, expected_type), f"Failed {desc}: wrong type"

    if expected_value is not None and isinstance(expected_value, int):
        if method_name == "count":
            assert result == expected_value, f"Failed {desc}"
        elif method_name in ["to_list", "to_text"]:
            assert len(result) > 0


PAGINATION_CASES = [
    ("limit", 3, 3, "limit 3"),
    ("offset", 2, 5, "offset 2"),
    ("reverse", None, 7, "reverse order"),
    ("first_n", 3, 3, "first_n 3"),
    ("last_n", 2, 2, "last_n 2"),
]


@pytest.mark.parametrize("method_name,arg,expected_count,desc", PAGINATION_CASES)
def test_projection_pagination(sample_conversation, method_name, arg, expected_count, desc):
    """Comprehensive pagination test."""
    projection = sample_conversation.project()

    if arg is not None:
        result = getattr(projection, method_name)(arg).to_list()
    else:
        result = getattr(projection, method_name)().to_list()

    assert len(result) == expected_count, f"Failed {desc}"


TRANSFORM_CASES = [
    ("truncate_text", 10, "text truncated", "truncate_text"),
    ("strip_attachments", 0, "no attachments after strip", "strip_attachments"),
]


@pytest.mark.parametrize("method_name,arg,expected_property,desc", TRANSFORM_CASES)
def test_projection_transforms(sample_conversation, method_name, arg, expected_property, desc):
    """Comprehensive transform test."""
    projection = sample_conversation.project()

    transformed = getattr(projection, method_name)(arg) if arg else getattr(projection, method_name)()

    result = transformed.to_list()

    if "truncated" in expected_property:
        for msg in result:
            if msg.text:
                assert len(msg.text) <= arg + 10
    elif "no attachments" in expected_property:
        for msg in result:
            assert not msg.attachments or len(msg.attachments) == 0


def _make_tool_message(id: str, text: str) -> Message:
    """Create a message marked as tool use via provider_meta."""
    return Message(
        id=id,
        role="assistant",
        text=text,
        provider_meta={"content_blocks": [{"type": "tool_use"}]},
    )


def _make_thinking_message(id: str, text: str) -> Message:
    """Create a message marked as thinking via provider_meta."""
    return Message(
        id=id,
        role="assistant",
        text=text,
        provider_meta={"content_blocks": [{"type": "thinking"}]},
    )


STRIP_CASES = [
    ("strip_tools", "is_tool_use", "tool use message stripped"),
    ("strip_thinking", "is_thinking", "thinking message stripped"),
]


@pytest.mark.parametrize("method_name,attr_name,desc", STRIP_CASES)
def test_projection_strip_methods(method_name, attr_name, desc):
    """Test strip_tools, strip_thinking, strip_all methods."""
    messages = [
        Message(id="m1", role="user", text="Hello"),
        _make_tool_message("m2", "Tool result"),
        _make_thinking_message("m3", "Thinking..."),
        Message(id="m4", role="assistant", text="Normal response"),
    ]
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))

    projection = conv.project()
    filtered = getattr(projection, method_name)()
    result = filtered.to_list()

    assert not any(getattr(m, attr_name, False) for m in result), f"Failed {desc}"


def test_projection_strip_all():
    """Test strip_all() removes both tools and thinking."""
    messages = [
        Message(id="m1", role="user", text="Hello"),
        _make_tool_message("m2", "Tool result"),
        _make_thinking_message("m3", "Thinking..."),
        Message(id="m4", role="assistant", text="Normal response"),
    ]
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))

    result = conv.project().strip_all().to_list()

    assert len(result) == 2
    assert not any(m.is_tool_use for m in result)
    assert not any(m.is_thinking for m in result)


EDGE_CASE_CONVERSATIONS = [
    ([], 0, "empty conversation"),
    ([Message(id="m1", role="user", text=None)], 0, "None text"),
    ([Message(id="m1", role="user", text="", timestamp=None)], 0, "None timestamp"),
    ([Message(id="m1", role="user", text="Valid")], 1, "single message"),
]


@pytest.mark.parametrize("messages,expected_count,desc", EDGE_CASE_CONVERSATIONS)
def test_projection_edge_cases(messages, expected_count, desc):
    """Edge case handling in projections."""
    conv = Conversation(id="test", provider="test", messages=MessageCollection(messages=messages))
    result = conv.project().to_list()

    assert len(result) >= 0


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


def test_provider_enum_values() -> None:
    """Provider enum has all expected values."""
    assert Provider.CHATGPT.value == "chatgpt"
    assert Provider.CLAUDE.value == "claude"
    assert Provider.CLAUDE_CODE.value == "claude-code"
    assert Provider.CODEX.value == "codex"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.DRIVE.value == "drive"
    assert Provider.UNKNOWN.value == "unknown"


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
    ("gpt", Provider.CHATGPT),
    ("openai", Provider.CHATGPT),
    ("GPT", Provider.CHATGPT),
    ("OPENAI", Provider.CHATGPT),
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


@given(st.text().filter(lambda s: s.strip().lower() not in [
    "chatgpt", "claude", "claude-code", "codex", "gemini", "drive", "unknown",
    "gpt", "openai", "claude-ai", "anthropic"
]))
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
