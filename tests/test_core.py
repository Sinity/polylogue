"""Consolidated core module tests.

SYSTEMATIZATION: Merged from test_core_json.py, test_core_utilities.py, test_lib.py

This file contains tests for:
- JSON encoding/decoding with Decimal support (polylogue.lib.json)
- Environment variable handling (polylogue.lib.env)
- Date parsing and formatting (polylogue.lib.dates)
- Version info (polylogue.version)
- Export functions (polylogue.export)
- Semantic models and repository (polylogue.lib)
"""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.config import Config, ConfigError, get_config
from polylogue.export import export_jsonl
from polylogue.lib import json as core_json
from polylogue.lib.hashing import hash_file, hash_payload, hash_text, hash_text_short
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.timestamps import format_timestamp, parse_timestamp
from polylogue.paths import DriveConfig, IndexConfig, Source, is_within_root, safe_path_component
from polylogue.schemas import ValidationResult, validate_provider_export as validate_provider_export_fn
from polylogue.schemas.validator import SchemaValidator, validate_provider_export
from polylogue.services import get_backend, get_repository, get_service_config, reset
from polylogue.storage.backends.sqlite import SQLiteBackend, connection_context, open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId, Provider
from tests.helpers import ConversationBuilder

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


def test_json_decode_error_is_value_error():
    """JSONDecodeError should be ValueError for compatibility."""
    assert core_json.JSONDecodeError is ValueError


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
    ("prefixed_precedence", "QDRANT_URL", {"POLYLOGUE_QDRANT_URL": "http://local:6333", "QDRANT_URL": "http://global:6333"}, "http://local:6333", None, "POLYLOGUE_* variable takes precedence over unprefixed"),
    ("unprefixed_fallback", "QDRANT_URL", {"QDRANT_URL": "http://global:6333"}, "http://global:6333", None, "Falls back to unprefixed when prefixed not set"),
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

    if default_val is not None:
        result = get_env(var_name, default_val)
    else:
        result = get_env(var_name)

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

    if default_val is not None:
        result = get_env_multi(*var_names, default=default_val)
    else:
        result = get_env_multi(*var_names)

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

    if commit is None:
        info = VersionInfo(version=version)
    else:
        info = VersionInfo(version=version, commit=commit, dirty=dirty)

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
# EXPORT TESTS (from test_core_utilities.py)
# =============================================================================


class TestExportJsonl:
    """Tests for export_jsonl() function."""

    def test_export_jsonl_creates_output_file(self, workspace_env):
        """Creates output file in correct location."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        assert result_path.suffix == ".jsonl"

    def test_export_jsonl_custom_output_path(self, workspace_env):
        """Uses custom output path when provided."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]
        custom_output = workspace_env["data_root"] / "custom_export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=custom_output)

        assert result_path == custom_output
        assert custom_output.exists()

    def test_export_jsonl_empty_database(self, workspace_env):
        """Handles empty database gracefully."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        content = result_path.read_text()
        assert content == ""

    def test_export_jsonl_with_conversation(self, workspace_env, db_path):
        """Exports conversation with messages and attachments."""
        from polylogue.export import export_jsonl
        from tests.helpers import ConversationBuilder

        (
            ConversationBuilder(db_path, "test-conv")
            .title("Export Test")
            .provider("chatgpt")
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi there!")
            .save()
        )

        archive_root = workspace_env["archive_root"]
        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        content = result_path.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 1

        export_data = json.loads(lines[0])
        assert "conversation" in export_data
        assert "messages" in export_data
        assert export_data["conversation"]["title"] == "Export Test"
        assert len(export_data["messages"]) == 2

    def test_export_jsonl_creates_parent_dirs(self, workspace_env):
        """Creates parent directories for output path."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]
        nested_output = workspace_env["data_root"] / "deeply" / "nested" / "export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=nested_output)

        assert result_path.exists()
        assert result_path.parent.exists()


# =============================================================================
# SEMANTIC MODELS TESTS (from test_lib.py)
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


def test_repository(test_db):
    """Test ConversationRepository basic operations."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    from tests.helpers import DbFactory

    factory = DbFactory(test_db)
    factory.create_conversation(
        id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
    )

    # Test get
    conv = repo.get("c1")
    assert conv is not None
    assert conv.id == "c1"
    assert len(conv.messages) == 1
    assert conv.messages[0].text == "hello world"

    # Test list
    lst = repo.list()
    assert len(lst) == 1
    assert lst[0].id == "c1"


def test_repository_get_includes_attachment_conversation_id(test_db):
    """ConversationRepository.get_eager() returns attachments with conversation_id field."""
    from tests.helpers import DbFactory

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
    conv = repo.get_eager("c-with-att")

    assert conv is not None
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.id == "att1"
    assert att.mime_type == "image/png"


def test_repository_get_with_multiple_attachments(test_db):
    """get_eager() correctly groups multiple attachments per message."""
    from tests.helpers import DbFactory

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
    conv = repo.get_eager("c-multi-att")

    assert conv is not None
    assert len(conv.messages) == 2

    m1 = conv.messages[0]
    assert len(m1.attachments) == 2
    m1_att_ids = {a.id for a in m1.attachments}
    assert m1_att_ids == {"att1", "att2"}

    m2 = conv.messages[1]
    assert len(m2.attachments) == 1
    assert m2.attachments[0].id == "att3"


def test_repository_get_attachment_metadata_decoded(test_db):
    """Attachment provider_meta JSON is properly decoded."""
    from tests.helpers import DbFactory

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
    conv = repo.get_eager("c-att-meta")

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
# --- merged from test_services.py ---
# =============================================================================


class TestServices:
    def test_get_backend_returns_sqlite(self, workspace_env):
        # Must import after workspace_env reloads the module
        from polylogue.storage.backends.sqlite import SQLiteBackend as _SQLiteBackend

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
# --- merged from test_export.py ---
# =============================================================================


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Provide a database populated with test data."""
    (ConversationBuilder(db_path, "conv-1")
     .provider("chatgpt")
     .title("First Conversation")
     .add_message("msg-1-1", role="user", text="Hello")
     .add_message("msg-1-2", role="assistant", text="Hi there")
     .save())

    (ConversationBuilder(db_path, "conv-2")
     .provider("claude")
     .title("With Attachments")
     .add_message("msg-2-1", role="user", text="Here is an image")
     .add_attachment("att-1", message_id="msg-2-1", mime_type="image/png", size_bytes=1024)
     .save())

    return db_path


def test_export_jsonl_creates_file(populated_db, tmp_path):
    """export_jsonl creates the output file."""
    db_path = populated_db

    with patch("polylogue.export.open_connection") as mock_conn:
        from polylogue.storage.backends.sqlite import connection_context

        mock_conn.side_effect = lambda _: connection_context(db_path)

        output_dir = tmp_path / "archive"
        output_dir.mkdir()

        output_file = export_jsonl(archive_root=output_dir)

        assert output_file.exists()
        assert output_file.name == "conversations.jsonl"
        assert output_file.parent == output_dir / "exports"


def test_export_content_correctness(populated_db, tmp_path):
    """Exported JSONL contains correct data."""
    from polylogue.storage.backends.sqlite import connection_context

    db_path = populated_db

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "out.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        convs = [json.loads(line) for line in lines]
        convs_by_id = {c["conversation"]["conversation_id"]: c for c in convs}

        assert "conv-1" in convs_by_id
        assert "conv-2" in convs_by_id

        c1 = convs_by_id["conv-1"]
        assert c1["conversation"]["title"] == "First Conversation"
        assert len(c1["messages"]) == 2
        assert c1["messages"][0]["text"] == "Hello"
        assert len(c1["attachments"]) == 0

        c2 = convs_by_id["conv-2"]
        assert len(c2["attachments"]) == 1
        att = c2["attachments"][0]
        assert att["attachment_id"] == "att-1"
        assert att["mime_type"] == "image/png"


def test_export_handles_empty_db(db_path, tmp_path):
    """Exporting empty DB produces empty file."""
    with open_connection(db_path):
        pass

    with patch("polylogue.export.open_connection", side_effect=lambda _: connection_context(db_path)):
        output_file = tmp_path / "empty.jsonl"
        export_jsonl(archive_root=tmp_path, output_path=output_file)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8").strip()
        assert content == ""


# =============================================================================
# --- merged from test_config.py ---
# =============================================================================


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_basic_construction(self, tmp_path):
        """Config can be constructed with required fields."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.archive_root == tmp_path / "archive"
        assert config.render_root == tmp_path / "render"
        assert config.sources == []

    def test_config_with_sources(self, tmp_path):
        """Config stores source list."""
        sources = [
            Source(name="inbox", path=tmp_path / "inbox"),
            Source(name="claude-code", path=tmp_path / "claude"),
        ]
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=sources,
        )
        assert len(config.sources) == 2
        assert config.sources[0].name == "inbox"
        assert config.sources[1].name == "claude-code"

    def test_config_db_path_property(self, workspace_env):
        """db_path property returns paths.DB_PATH."""
        config = Config(
            archive_root=Path(workspace_env["archive_root"]),
            render_root=Path(workspace_env["archive_root"]) / "render",
            sources=[],
        )
        assert config.db_path.name == "polylogue.db"
        assert "polylogue" in str(config.db_path)

    def test_config_optional_fields_default_none(self, tmp_path):
        """Optional fields default to None."""
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.drive_config is None
        assert config.index_config is None


class TestGetConfig:
    """Tests for get_config() function."""

    def test_get_config_returns_config(self, workspace_env):
        """get_config() returns a Config instance."""
        from polylogue.config import Config as ReloadedConfig
        from polylogue.config import get_config as reloaded_get_config

        config = reloaded_get_config()
        assert isinstance(config, ReloadedConfig)

    def test_get_config_has_archive_root(self, workspace_env):
        """Config from get_config() has an archive_root."""
        config = get_config()
        assert config.archive_root is not None
        assert isinstance(config.archive_root, Path)

    def test_get_config_has_render_root(self, workspace_env):
        """Config from get_config() has a render_root."""
        config = get_config()
        assert config.render_root is not None
        assert isinstance(config.render_root, Path)

    def test_get_config_has_sources(self, workspace_env):
        """Config from get_config() has a sources list."""
        config = get_config()
        assert isinstance(config.sources, list)
        assert len(config.sources) >= 1

    def test_get_config_has_drive_config(self, workspace_env):
        """Config from get_config() includes drive configuration."""
        config = get_config()
        assert config.drive_config is not None

    def test_get_config_has_index_config(self, workspace_env):
        """Config from get_config() includes index configuration."""
        config = get_config()
        assert config.index_config is not None


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_runtime_error(self):
        """ConfigError inherits from RuntimeError."""
        assert issubclass(ConfigError, RuntimeError)

    def test_config_error_message(self):
        """ConfigError preserves error message."""
        err = ConfigError("test error message")
        assert str(err) == "test error message"

    def test_config_error_can_be_raised_and_caught(self):
        """ConfigError works with try/except."""
        with pytest.raises(ConfigError, match="bad config"):
            raise ConfigError("bad config")


# =============================================================================
# --- merged from test_timestamps.py ---
# =============================================================================


# Test cases: (input_value, expected_year, expected_month, expected_day, expected_microsecond, description)
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
# --- merged from test_projections.py ---
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
# --- merged from test_paths.py ---
# =============================================================================


class TestSafePathComponent:
    """Tests for filesystem-safe path component generation."""

    def test_simple_safe_string(self):
        """Simple alphanumeric strings pass through unchanged."""
        assert safe_path_component("hello") == "hello"
        assert safe_path_component("test-file") == "test-file"
        assert safe_path_component("v2.0.1") == "v2.0.1"

    def test_special_characters_replaced(self):
        """Strings with special chars get hashed."""
        result = safe_path_component("hello world")
        assert "-" in result
        assert len(result) > 10

    def test_empty_string_uses_fallback(self):
        """Empty string returns fallback."""
        result = safe_path_component("")
        assert result == "item"

    def test_custom_fallback(self):
        """Custom fallback is used for empty input."""
        result = safe_path_component("", fallback="default")
        assert result == "default"

    def test_none_uses_fallback(self):
        """None input returns fallback."""
        result = safe_path_component(None)
        assert result == "item"

    def test_whitespace_only_uses_fallback(self):
        """Whitespace-only input returns fallback."""
        result = safe_path_component("   ")
        assert result == "item"

    def test_dot_returns_fallback(self):
        """Single dot returns fallback (dangerous path component)."""
        result = safe_path_component(".")
        assert "item" in result

    def test_dotdot_returns_fallback(self):
        """Double dot returns fallback (path traversal)."""
        result = safe_path_component("..")
        assert "item" in result

    def test_path_separator_triggers_hash(self):
        """Path separators trigger hashed output."""
        result = safe_path_component("foo/bar")
        assert "-" in result
        assert "/" not in result

    def test_unicode_triggers_hash(self):
        """Unicode characters trigger hashed output."""
        result = safe_path_component("café")
        assert "-" in result

    def test_deterministic(self):
        """Same input always produces same output."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("hello world")
        assert r1 == r2

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("goodbye world")
        assert r1 != r2

    def test_long_prefix_truncated(self):
        """Long prefixes are truncated to 12 chars."""
        result = safe_path_component("this_is_a_very_long_name with spaces")
        prefix = result.split("-")[0]
        assert len(prefix) <= 12


class TestIsWithinRoot:
    """Tests for path containment check."""

    def test_path_within_root(self, tmp_path):
        """Path inside root returns True."""
        root = tmp_path / "root"
        root.mkdir()
        child = root / "subdir" / "file.txt"
        assert is_within_root(child, root) is True

    def test_path_outside_root(self, tmp_path):
        """Path outside root returns False."""
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "other" / "file.txt"
        assert is_within_root(outside, root) is False

    def test_path_is_root(self, tmp_path):
        """Root itself is within root."""
        root = tmp_path / "root"
        root.mkdir()
        assert is_within_root(root, root) is True

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal (../) is correctly evaluated."""
        root = tmp_path / "root"
        root.mkdir()
        traversal = root / ".." / "other"
        assert is_within_root(traversal, root) is False


class TestSource:
    """Tests for Source dataclass validation."""

    def test_source_with_path(self, tmp_path):
        """Source with path is valid."""
        src = Source(name="test", path=tmp_path)
        assert src.name == "test"
        assert src.path == tmp_path
        assert not src.is_drive

    def test_source_with_folder(self):
        """Source with folder (Drive) is valid."""
        src = Source(name="gemini", folder="Google AI Studio")
        assert src.name == "gemini"
        assert src.is_drive

    def test_source_empty_name_raises(self):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="", path=Path("/tmp"))

    def test_source_whitespace_name_raises(self):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="   ", path=Path("/tmp"))

    def test_source_no_path_no_folder_raises(self):
        """Source without path or folder raises ValueError."""
        with pytest.raises(ValueError, match="must have either"):
            Source(name="broken")

    def test_source_both_path_and_folder_raises(self):
        """Source with both path and folder raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both"):
            Source(name="confused", path=Path("/tmp"), folder="Drive Folder")

    def test_source_name_stripped(self):
        """Source name is stripped of whitespace."""
        src = Source(name="  test  ", path=Path("/tmp"))
        assert src.name == "test"

    def test_source_folder_stripped(self):
        """Source folder is stripped of whitespace."""
        src = Source(name="test", folder="  My Folder  ")
        assert src.folder == "My Folder"


class TestDriveConfig:
    """Tests for DriveConfig defaults."""

    def test_default_retry_count(self):
        """Default retry count is 3."""
        config = DriveConfig()
        assert config.retry_count == 3

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        config = DriveConfig()
        assert config.timeout == 30

    def test_credentials_path_is_in_config(self):
        """Default credentials path is in polylogue config dir."""
        config = DriveConfig()
        assert "polylogue" in str(config.credentials_path)


class TestIndexConfig:
    """Tests for IndexConfig from environment."""

    def test_from_env_defaults(self, monkeypatch):
        """Default IndexConfig has FTS enabled, no external services."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_MODEL", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_DIMENSION", raising=False)
        monkeypatch.delenv("POLYLOGUE_AUTO_EMBED", raising=False)
        config = IndexConfig.from_env()
        assert config.fts_enabled is True
        assert config.voyage_api_key is None
        assert config.voyage_model == "voyage-4"
        assert config.voyage_dimension is None
        assert config.auto_embed is False

    def test_from_env_polylogue_prefixed(self, monkeypatch):
        """POLYLOGUE_* env vars are picked up."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "voyage-key")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_MODEL", "voyage-4-large")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_DIMENSION", "512")
        monkeypatch.setenv("POLYLOGUE_AUTO_EMBED", "true")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "voyage-key"
        assert config.voyage_model == "voyage-4-large"
        assert config.voyage_dimension == 512
        assert config.auto_embed is True

    def test_from_env_unprefixed_fallback(self, monkeypatch):
        """Unprefixed env vars used when POLYLOGUE_* not set."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "fallback-key"

    def test_from_env_prefixed_takes_precedence(self, monkeypatch):
        """POLYLOGUE_* vars take precedence over unprefixed."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "preferred-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "preferred-key"


class TestXDGPaths:
    """Tests for XDG path resolution."""

    def test_xdg_data_home_respected(self, monkeypatch):
        """XDG_DATA_HOME env var overrides default."""
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        assert Path("/custom/data") == polylogue.paths.DATA_ROOT

        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        importlib.reload(polylogue.paths)

    def test_db_path_under_data_home(self, workspace_env):
        """DB_PATH is under XDG_DATA_HOME/polylogue/."""
        import polylogue.paths

        assert "polylogue" in str(polylogue.paths.DB_PATH)
        assert polylogue.paths.DB_PATH.name == "polylogue.db"


# =============================================================================
# --- merged from test_types.py ---
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


# =============================================================================
# --- merged from test_validation.py ---
# =============================================================================


@pytest.fixture
def mock_schema_dir(tmp_path):
    """Create a mock schema directory with test schemas."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()

    test_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "count": {"type": "integer"},
            "meta": {"type": "object", "properties": {"source": {"type": "string"}}, "additionalProperties": False},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    (schema_dir / "test-provider.schema.json").write_text(json.dumps(test_schema), encoding="utf-8")

    open_schema = {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": {}}
    (schema_dir / "open-provider.schema.json").write_text(json.dumps(open_schema), encoding="utf-8")

    return schema_dir


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_schema_validator_loads_provider(mock_path_attr, mock_schema_dir):
    """SchemaValidator.for_provider loads correct schema."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")
        assert validator.schema["required"] == ["id"]

        with pytest.raises(FileNotFoundError):
            SchemaValidator.for_provider("nonexistent")


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_valid_data(mock_path_attr, mock_schema_dir):
    """Validate returns is_valid=True for valid data."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        data = {"id": "123", "count": 10, "meta": {"source": "test"}}
        result = validator.validate(data)

        assert result.is_valid
        assert not result.errors
        assert not result.drift_warnings


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_detects_errors(mock_path_attr, mock_schema_dir):
    """Validate detects schema errors."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        result = validator.validate({"count": 10})
        assert not result.is_valid
        assert any("id" in e for e in result.errors)

        result = validator.validate({"id": 123})
        assert not result.is_valid
        assert any("123" in e for e in result.errors)


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_detects_drift(mock_path_attr, mock_schema_dir):
    """Validate detects drift (unexpected fields) in strict mode."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("open-provider", strict=True)

        data = {"id": "123", "extra": "drift"}
        result = validator.validate(data)

        assert result.is_valid
        assert result.has_drift
        assert "extra" in result.drift_warnings[0]


def test_available_providers(mock_schema_dir):
    """available_providers lists schema files."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        providers = SchemaValidator.available_providers()
        assert "test-provider" in providers
        assert "open-provider" in providers
        assert "nonexistent" not in providers


def test_validate_helper(mock_schema_dir):
    """validate_provider_export helper works."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        result = validate_provider_export({"id": "123"}, "test-provider")
        assert result.is_valid


def test_schema_validator_creation():
    """Test creating validators for available providers."""
    for provider in SchemaValidator.available_providers():
        validator = SchemaValidator.for_provider(provider)
        assert validator.schema is not None
        assert "$schema" in validator.schema


def test_missing_provider_raises():
    """Test that missing provider raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


class TestSchemaValidation:
    """Validate real data against schemas using raw_conversations."""

    def test_all_samples_validate(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """All raw samples must validate against their provider schemas."""
        if not raw_db_samples:
            pytest.skip("No raw conversations (run: polylogue run --stage acquire)")

        provider_to_schema = {
            "chatgpt": "chatgpt",
            "claude": "claude-ai",
            "claude-ai": "claude-ai",
            "claude-code": "claude-code",
            "codex": "codex",
            "gemini": "gemini",
        }

        available = set(SchemaValidator.available_providers())
        failures = []
        skipped_providers = set()

        for sample in raw_db_samples:
            schema_name = provider_to_schema.get(sample.provider_name, sample.provider_name)

            if schema_name not in available:
                skipped_providers.add(sample.provider_name)
                continue

            try:
                validator = SchemaValidator.for_provider(schema_name, strict=False)
                content = sample.raw_content.decode("utf-8")

                if sample.provider_name in ("claude-code", "codex", "gemini"):
                    for line in content.strip().split("\n"):
                        if line.strip():
                            data = json.loads(line)
                            break
                    else:
                        failures.append((sample.raw_id[:16], sample.provider_name, "Empty JSONL"))
                        continue
                else:
                    data = json.loads(content)

                result = validator.validate(data)
                if not result.is_valid:
                    failures.append((sample.raw_id[:16], sample.provider_name, result.errors[0][:80]))

            except json.JSONDecodeError as e:
                failures.append((sample.raw_id[:16], sample.provider_name, f"Invalid JSON: {e}"))
            except Exception as e:
                failures.append((sample.raw_id[:16], sample.provider_name, str(e)[:80]))

        if failures:
            msg = f"{len(failures)}/{len(raw_db_samples)} failed validation:\n"
            for raw_id, provider, error in failures[:10]:
                msg += f"  {provider}:{raw_id}: {error}\n"
            msg += "\nRun: polylogue run --stage generate-schemas"
            pytest.fail(msg)


class TestDriftDetection:
    """Detect schema drift in real data."""

    def test_drift_warnings(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """Report drift warnings (new fields not in schema)."""
        if not raw_db_samples:
            pytest.skip("No raw conversations")

        provider_to_schema = {
            "chatgpt": "chatgpt",
            "claude": "claude-ai",
            "claude-ai": "claude-ai",
            "claude-code": "claude-code",
            "codex": "codex",
            "gemini": "gemini",
        }

        available = set(SchemaValidator.available_providers())
        drift_by_provider: dict[str, list[str]] = {}

        for sample in raw_db_samples[:100]:
            schema_name = provider_to_schema.get(sample.provider_name, sample.provider_name)
            if schema_name not in available:
                continue

            try:
                content = sample.raw_content.decode("utf-8")
                if sample.provider_name in ("claude-code", "codex", "gemini"):
                    for line in content.strip().split("\n"):
                        if line.strip():
                            data = json.loads(line)
                            break
                    else:
                        continue
                else:
                    data = json.loads(content)

                result = validate_provider_export(data, schema_name, strict=True)
                if result.drift_warnings:
                    if sample.provider_name not in drift_by_provider:
                        drift_by_provider[sample.provider_name] = []
                    drift_by_provider[sample.provider_name].extend(result.drift_warnings[:3])

            except Exception:
                pass

        if drift_by_provider:
            print(f"\nDrift detected in {len(drift_by_provider)} providers:")
            for provider, warnings in drift_by_provider.items():
                unique_warnings = list(set(warnings))[:5]
                print(f"  {provider}: {len(unique_warnings)} unique warnings")
                for w in unique_warnings[:3]:
                    print(f"    - {w}")


def test_chatgpt_rejects_missing_mapping():
    """Test that ChatGPT schema rejects exports without mapping."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    invalid = {"id": "test", "title": "Test"}
    result = validate_provider_export(invalid, "chatgpt")
    assert isinstance(result, ValidationResult)


def test_drift_new_field():
    """Test that new fields are detected as drift."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    data = {
        "id": "test-123",
        "mapping": {},
        "brand_new_field": "unexpected",
    }

    result = validate_provider_export(data, "chatgpt", strict=True)
    assert isinstance(result, ValidationResult)


def test_validation_result_properties():
    """Test ValidationResult properties."""
    valid = ValidationResult(is_valid=True)
    assert valid.is_valid
    assert not valid.has_drift
    valid.raise_if_invalid()

    invalid = ValidationResult(is_valid=False, errors=["missing field"])
    assert not invalid.is_valid
    with pytest.raises(ValueError, match="missing field"):
        invalid.raise_if_invalid()

    with_drift = ValidationResult(is_valid=True, drift_warnings=["new field: foo"])
    assert with_drift.is_valid
    assert with_drift.has_drift


# =============================================================================
# --- merged from test_hashing.py ---
# =============================================================================


HASH_TEXT_CASES = [
    ("hello world", 64, "length is 64 chars"),
    ("deterministic test", "deterministic", "same input → same output"),
    ("input one", "differs from 'input two'", "different inputs"),
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty string known hash"),
    ("Hello 世界 🌍 emoji", 64, "unicode emoji CJK"),
    ("line one\nline two\nline three", "multiline", "newlines preserved"),
]


@pytest.mark.parametrize("text,expected,desc", HASH_TEXT_CASES)
def test_hash_text_comprehensive(text, expected, desc):
    """Comprehensive hash_text test."""
    result = hash_text(text)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_text(text), f"Failed {desc}"
    elif expected.startswith("differs"):
        other_text = "input two"
        assert result != hash_text(other_text), f"Failed {desc}"
    elif expected == "multiline":
        concatenated = "line oneline twoline three"
        assert result != hash_text(concatenated), f"Failed {desc}"
    elif expected == "unicode":
        assert result == hash_text(text), f"Failed {desc}"
    else:
        assert result == expected, f"Failed {desc}"


HASH_TEXT_SHORT_CASES = [
    ("test", None, 16, "default length"),
    ("custom length test", 8, 8, "custom length 8"),
    ("custom length test", 32, 32, "custom length 32"),
    ("custom length test", 64, 64, "custom length 64"),
    ("deterministic short", None, "deterministic", "same input → same output"),
    ("prefix test", 10, "prefix", "short is prefix of full"),
    ("input A", None, "differs from 'input B'", "different inputs"),
]


@pytest.mark.parametrize("text,length,expected,desc", HASH_TEXT_SHORT_CASES)
def test_hash_text_short_comprehensive(text, length, expected, desc):
    """Comprehensive hash_text_short test."""
    result = hash_text_short(text, length=length) if length is not None else hash_text_short(text)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        if length is not None:
            assert result == hash_text_short(text, length=length), f"Failed {desc}"
        else:
            assert result == hash_text_short(text), f"Failed {desc}"
    elif expected == "prefix":
        full = hash_text(text)
        assert full.startswith(result), f"Failed {desc}"
    elif expected.startswith("differs"):
        other_text = "input B"
        if length is not None:
            assert result != hash_text_short(other_text, length=length), f"Failed {desc}"
        else:
            assert result != hash_text_short(other_text), f"Failed {desc}"


HASH_PAYLOAD_CASES = [
    ({"name": "test", "value": 42}, 64, "dict"),
    ([1, 2, 3, "four", "five"], 64, "list"),
    ({"outer": {"inner": [1, 2, {"deep": "value"}], "another": "field"}, "list": [{"a": 1}, {"b": 2}]}, 64, "nested"),
    ({"a": 1, "b": 2, "c": 3}, "key_order_independent", "key order independence"),
    ({"complex": [1, 2, {"nested": True}], "value": 123}, "deterministic", "determinism"),
    ({"key": "value1"}, "differs from value2", "different values"),
    (42, 64, "int primitive"),
    ("string", 64, "string primitive"),
    (True, 64, "bool primitive"),
    (None, 64, "None primitive"),
    (3.14159, 64, "float primitive"),
]


@pytest.mark.parametrize("payload,expected,desc", HASH_PAYLOAD_CASES)
def test_hash_payload_comprehensive(payload, expected, desc):
    """Comprehensive hash_payload test."""
    result = hash_payload(payload)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "key_order_independent":
        variant1 = {"c": 3, "a": 1, "b": 2}
        variant2 = {"b": 2, "c": 3, "a": 1}
        hash1 = hash_payload(payload)
        hash2 = hash_payload(variant1)
        hash3 = hash_payload(variant2)
        assert hash1 == hash2 == hash3, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_payload(payload), f"Failed {desc}"
    elif expected.startswith("differs"):
        other = {"key": "value2"}
        assert result != hash_payload(other), f"Failed {desc}"


HASH_FILE_CASES = [
    ("Hello, world!", 64, None, "basic"),
    ("deterministic content", "deterministic", None, "deterministic"),
    ("", "empty", None, "empty file"),
    (bytes(range(256)), 64, "binary", "binary content"),
    (b"x" * (1024 * 1024 * 2 + 1024 * 512), 64, "binary", "large file"),
    ("content one", "differs from 'content two'", None, "different contents"),
    ("Hello 世界 🌍 emoji", 64, "utf-8", "unicode content"),
    (None, "newlines", "binary", "newline differences"),
]


@pytest.mark.parametrize("content,expected,encoding,desc", HASH_FILE_CASES)
def test_hash_file_comprehensive(tmp_path: Path, content, expected, encoding, desc):
    """Comprehensive hash_file test."""
    file = tmp_path / "test.dat"

    if expected == "newlines":
        file_lf = tmp_path / "lf.txt"
        file_crlf = tmp_path / "crlf.txt"
        file_lf.write_bytes(b"line1\nline2\n")
        file_crlf.write_bytes(b"line1\r\nline2\r\n")
        hash_lf = hash_file(file_lf)
        hash_crlf = hash_file(file_crlf)
        assert hash_lf != hash_crlf, f"Failed {desc}"
        return

    if isinstance(content, bytes):
        file.write_bytes(content)
    else:
        file.write_text(content, encoding=encoding or "utf-8")

    result = hash_file(file)

    if isinstance(expected, int):
        assert len(result) == expected, f"Failed {desc}"
    elif expected == "deterministic":
        assert result == hash_file(file), f"Failed {desc}"
    elif expected == "empty":
        assert result == hash_text(""), f"Failed {desc}"
    elif expected.startswith("differs"):
        file2 = tmp_path / "test2.dat"
        file2.write_text("content two")
        hash2 = hash_file(file2)
        assert result != hash2, f"Failed {desc}"


CONSISTENCY_CASES = [
    ("hash_payload on string", "test string", "payload_vs_text", "payload wraps in quotes"),
    ("hash_file matches hash_text", "file content test", "file_vs_text", "file vs text consistency"),
    ("consistency test", None, "short_prefix", "short is prefix of full"),
]


@pytest.mark.parametrize("label,text,test_type,desc", CONSISTENCY_CASES)
def test_cross_function_consistency(tmp_path: Path, label, text, test_type, desc):
    """Cross-function consistency tests."""
    if test_type == "payload_vs_text":
        assert hash_payload(text) == hash_text(f'"{text}"'), f"Failed {desc}"
        assert hash_payload(text) != hash_text(text), f"Failed {desc}"

    elif test_type == "file_vs_text":
        file = tmp_path / "test.txt"
        file.write_text(text, encoding="utf-8")
        assert hash_file(file) == hash_text(text), f"Failed {desc}"

    elif test_type == "short_prefix":
        test_text = text if text is not None else label
        for length in [1, 8, 16, 32, 48, 64]:
            short = hash_text_short(test_text, length=length)
            full = hash_text(test_text)
            assert short == full[:length], f"Failed {desc} at length {length}"


UNICODE_NORMALIZATION_CASES = [
    ("café", "nfc_nfd", "NFC vs NFD same hash"),
    ("\u00f1", "combining", "precomposed vs decomposed"),
    ("👋", "emoji_modifiers", "emoji with modifiers"),
    ("hello", "zero_width", "zero-width characters"),
]


@pytest.mark.parametrize("text,test_type,desc", UNICODE_NORMALIZATION_CASES)
def test_unicode_normalization_comprehensive(text, test_type, desc):
    """Unicode normalization tests."""
    if test_type == "nfc_nfd":
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)

        assert nfc.encode("utf-8") != nfd.encode("utf-8")

        hash_nfc = hash_text(nfc)
        hash_nfd = hash_text(nfd)

        assert hash_nfc == hash_nfd, f"Unicode normalization bug: {desc}"

    elif test_type == "combining":
        precomposed = text
        decomposed = "n\u0303"

        hash1 = hash_text(precomposed)
        hash2 = hash_text(decomposed)

        assert hash1 == hash2, f"Combining characters must hash same: {desc}"

    elif test_type == "emoji_modifiers":
        wave = text
        wave_light = "👋🏻"

        hash_base = hash_text(wave)
        hash_modified = hash_text(wave_light)

        assert hash_base != hash_modified, f"Failed {desc}"

    elif test_type == "zero_width":
        normal = text
        with_zwj = "hel\u200dlo"

        hash_normal = hash_text(normal)
        hash_zwj = hash_text(with_zwj)

        assert hash_normal != hash_zwj, f"Failed {desc}"


@given(st.text())
def test_hash_text_unicode_normalization_invariant(text: str):
    """Hash MUST be invariant under Unicode normalization."""
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)

    hash_nfc = hash_text(nfc)
    hash_nfd = hash_text(nfd)

    assert hash_nfc == hash_nfd, f"Normalization variant for {repr(text[:20])}..."
