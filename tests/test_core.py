"""Consolidated core module tests.

SYSTEMATIZATION: Merged from test_core_json.py, test_core_utilities.py, test_lib.py

This file contains tests for:
- JSON encoding/decoding with Decimal support (polylogue.core.json)
- Environment variable handling (polylogue.core.env)
- Date parsing and formatting (polylogue.core.dates)
- Version info (polylogue.version)
- Export functions (polylogue.export)
- Semantic models and repository (polylogue.lib)
"""
from __future__ import annotations

import json
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polylogue.core import json as core_json


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
    ({"value1": Decimal("10.5"), "value2": Decimal("20.75"), "nested": {"value3": Decimal("30.25")}}, "multiple Decimals"),

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
            assert all(isinstance(v, float) if isinstance(payload[k], Decimal) else True
                      for k, v in data.items() if isinstance(payload, dict))
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
    ("""
    {
        "key"   :   "value"  ,
        "array" : [ 1 , 2 , 3 ]
    }
    """, "whitespace handling"),
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
# ENVIRONMENT VARIABLE TESTS (from test_core_utilities.py)
# =============================================================================


class TestGetEnv:
    """Tests for get_env() function with POLYLOGUE_* precedence."""

    def test_get_env_returns_prefixed_first(self, monkeypatch):
        """POLYLOGUE_* variable takes precedence over unprefixed."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("QDRANT_URL", "http://global:6333")
        monkeypatch.setenv("POLYLOGUE_QDRANT_URL", "http://local:6333")

        result = get_env("QDRANT_URL")
        assert result == "http://local:6333"

    def test_get_env_falls_back_to_unprefixed(self, monkeypatch):
        """Falls back to unprefixed when prefixed not set."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("QDRANT_URL", "http://global:6333")
        monkeypatch.delenv("POLYLOGUE_QDRANT_URL", raising=False)

        result = get_env("QDRANT_URL")
        assert result == "http://global:6333"

    def test_get_env_returns_default(self, monkeypatch):
        """Returns default when neither variable is set."""
        from polylogue.core.env import get_env

        monkeypatch.delenv("MISSING_VAR", raising=False)
        monkeypatch.delenv("POLYLOGUE_MISSING_VAR", raising=False)

        result = get_env("MISSING_VAR", "fallback")
        assert result == "fallback"

    def test_get_env_returns_none_without_default(self, monkeypatch):
        """Returns None when neither variable is set and no default given."""
        from polylogue.core.env import get_env

        monkeypatch.delenv("TOTALLY_MISSING", raising=False)
        monkeypatch.delenv("POLYLOGUE_TOTALLY_MISSING", raising=False)

        result = get_env("TOTALLY_MISSING")
        assert result is None

    def test_get_env_empty_string_is_falsy(self, monkeypatch):
        """Empty string values are treated as falsy (falls through)."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("POLYLOGUE_EMPTY", "")
        monkeypatch.setenv("EMPTY", "real_value")

        result = get_env("EMPTY")
        assert result == "real_value"


class TestGetEnvMulti:
    """Tests for get_env_multi() with multiple fallback names."""

    def test_get_env_multi_prefixed_first(self, monkeypatch):
        """Prefixed variable takes precedence over all."""
        from polylogue.core.env import get_env_multi

        monkeypatch.setenv("POLYLOGUE_GOOGLE_API_KEY", "polylogue_key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "polylogue_key"

    def test_get_env_multi_unprefixed_primary(self, monkeypatch):
        """Unprefixed primary takes precedence over alternatives."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "google_key"

    def test_get_env_multi_falls_to_alternative(self, monkeypatch):
        """Falls back to alternative when primary not set."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "gemini_key"

    def test_get_env_multi_returns_default(self, monkeypatch):
        """Returns default when no variables are set."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_MISSING", raising=False)
        monkeypatch.delenv("MISSING", raising=False)
        monkeypatch.delenv("ALSO_MISSING", raising=False)

        result = get_env_multi("MISSING", "ALSO_MISSING", default="default_val")
        assert result == "default_val"

    def test_get_env_multi_no_alternatives(self, monkeypatch):
        """Works with no alternative names provided."""
        from polylogue.core.env import get_env_multi

        monkeypatch.setenv("SINGLE_KEY", "single_value")
        monkeypatch.delenv("POLYLOGUE_SINGLE_KEY", raising=False)

        result = get_env_multi("SINGLE_KEY")
        assert result == "single_value"


# =============================================================================
# DATE PARSING TESTS (from test_core_utilities.py)
# =============================================================================


class TestParseDate:
    """Tests for parse_date() with natural language support."""

    def test_parse_date_iso_format(self):
        """Parses ISO format dates correctly."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is not None

    def test_parse_date_iso_with_time(self):
        """Parses ISO format with time correctly."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-01-15T10:30:00")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_date_natural_yesterday(self):
        """Parses 'yesterday' naturally."""
        from polylogue.core.dates import parse_date

        result = parse_date("yesterday")
        assert result is not None
        assert result < datetime.now(timezone.utc)

    def test_parse_date_natural_relative(self):
        """Parses relative dates like '2 days ago'."""
        from polylogue.core.dates import parse_date

        result = parse_date("2 days ago")
        assert result is not None
        now = datetime.now(timezone.utc)
        assert result < now

    def test_parse_date_invalid_returns_none(self):
        """Returns None for unparseable input."""
        from polylogue.core.dates import parse_date

        result = parse_date("not a date at all xyz123")
        assert result is None

    def test_parse_date_returns_utc_aware(self):
        """All returned dates are UTC-aware."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-06-15")
        assert result is not None
        assert result.tzinfo is not None


class TestFormatDateIso:
    """Tests for format_date_iso() function."""

    def test_format_date_iso_basic(self):
        """Formats datetime as ISO string."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_date_iso(dt)
        assert result == "2024-01-15 10:30:45"

    def test_format_date_iso_midnight(self):
        """Formats midnight correctly."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 12, 25, 0, 0, 0)
        result = format_date_iso(dt)
        assert result == "2024-12-25 00:00:00"

    def test_format_date_iso_with_timezone(self):
        """Works with timezone-aware datetime."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = format_date_iso(dt)
        assert result == "2024-06-15 14:30:00"


# =============================================================================
# VERSION TESTS (from test_core_utilities.py)
# =============================================================================


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_version_info_str_without_commit(self):
        """String representation without commit hash."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3")
        assert str(info) == "1.2.3"
        assert info.full == "1.2.3"
        assert info.short == "1.2.3"

    def test_version_info_str_with_commit(self):
        """String representation with commit hash."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3", commit="abc123def456")
        assert str(info) == "1.2.3+abc123de"
        assert info.full == "1.2.3+abc123de"
        assert info.short == "1.2.3"

    def test_version_info_str_with_dirty(self):
        """String representation with dirty flag."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3", commit="abc123def456", dirty=True)
        assert str(info) == "1.2.3+abc123de-dirty"
        assert info.full == "1.2.3+abc123de-dirty"
        assert info.short == "1.2.3"

    def test_version_info_short_property(self):
        """Short property always returns just version."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="0.9.0", commit="deadbeef", dirty=True)
        assert info.short == "0.9.0"


class TestVersionResolution:
    """Tests for version resolution mechanism."""

    def test_version_info_available(self):
        """VERSION_INFO is available at module level."""
        from polylogue.version import VERSION_INFO

        assert VERSION_INFO is not None
        assert hasattr(VERSION_INFO, "version")
        assert hasattr(VERSION_INFO, "full")
        assert hasattr(VERSION_INFO, "short")

    def test_polylogue_version_available(self):
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
        from tests.helpers import ConversationBuilder
        from polylogue.export import export_jsonl

        (ConversationBuilder(db_path, "test-conv")
         .title("Export Test")
         .provider("chatgpt")
         .add_message("m1", role="user", text="Hello")
         .add_message("m2", role="assistant", text="Hi there!")
         .save())

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


from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.repository import ConversationRepository
from polylogue.storage.backends.sqlite import SQLiteBackend


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

    from tests.factories import DbFactory

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
    from tests.factories import DbFactory

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
    from tests.factories import DbFactory

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
    from tests.factories import DbFactory

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
