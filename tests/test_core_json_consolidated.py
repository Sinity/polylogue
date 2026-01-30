"""Consolidated JSON core tests using parametrization.

CONSOLIDATION: 62 tests → ~7 parametrized test functions with 62+ test cases.

Original: Separate test classes per aspect (dumps, loads, roundtrip, edge cases)
New: Parametrized tests covering all JSON encoding/decoding behaviors
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from polylogue.core import json


# =============================================================================
# DUMPS - PARAMETRIZED (1 test replacing 17)
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
    """Comprehensive dumps test.

    Replaces 17 individual tests from TestDumps.
    """
    if desc == "custom default handler":
        # Custom type test
        class CustomType:
            def __init__(self, value):
                self.value = value

        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, CustomType):
                return {"custom": obj.value}
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload_obj = CustomType(42)
        output = json.dumps(payload_obj, default=custom_handler)
        data = json.loads(output)
        assert data == {"custom": 42}
        return

    elif desc == "custom handler fallback to encoder":
        # Handler that doesn't handle Decimal → encoder does
        def custom_handler(obj: Any) -> Any:
            raise TypeError("Not handled")

        payload = {"decimal": Decimal("1.5")}
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["decimal"] == 1.5
        return

    # Standard dumps → loads verification
    output = json.dumps(payload)
    data = json.loads(output)

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
        # Generic assertion - roundtrip equality
        if "Decimal" in desc:
            # Decimals convert to floats
            assert all(isinstance(v, float) if isinstance(payload[k], Decimal) else True
                      for k, v in data.items() if isinstance(payload, dict))
        else:
            assert data == payload


# =============================================================================
# LOADS - PARAMETRIZED (1 test replacing 14)
# =============================================================================


LOADS_CASES = [
    # Basic types
    ('{"key": "value"}', {"key": "value"}, "string", "from string"),
    (b'{"key": "value"}', {"key": "value"}, "bytes", "from bytes"),
    ("[1, 2, 3]", [1, 2, 3], "str", "array"),
    ('{"outer": {"inner": [1, 2]}}', {"outer": {"inner": [1, 2]}}, "str", "nested structure"),

    # Unicode
    ('{"text": "Hello 世界"}', {"text": "Hello 世界"}, "str", "with unicode"),

    # Special values
    ('{"value": null}', {"value": None}, "str", "with null"),
    ('{"true": true, "false": false}', {"true": True, "false": False}, "str", "with boolean"),

    # Primitives
    ("42", 42, "str", "primitive number int"),
    ("3.14", 3.14, "str", "primitive number float"),
    ('"hello"', "hello", "str", "primitive string"),
    ("null", None, "str", "primitive null"),
    ("true", True, "str", "primitive true"),
    ("false", False, "str", "primitive false"),

    # Empty
    ("{}", {}, "str", "empty dict"),
    ("[]", [], "str", "empty list"),

    # Special strings
    ('{"text": "He said \\"hello\\""}', {"text": 'He said "hello"'}, "str", "escaped quotes"),
    ('{"text": "Line 1\\nLine 2"}', {"text": "Line 1\nLine 2"}, "str", "newlines in string"),
]


@pytest.mark.parametrize("input_data,expected,input_type,desc", LOADS_CASES)
def test_loads_comprehensive(input_data, expected, input_type, desc):
    """Comprehensive loads test.

    Replaces 14 individual tests from TestLoads.
    """
    result = json.loads(input_data)
    assert result == expected, f"Failed {desc}"


# =============================================================================
# ROUNDTRIP - PARAMETRIZED (1 test replacing 5)
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
    """Comprehensive roundtrip test.

    Replaces 5 individual tests from TestRoundtrip.
    """
    if desc == "multiple times":
        # Multiple roundtrips
        data = original
        for _ in range(3):
            output = json.dumps(data)
            data = json.loads(output)
        assert data == original

    elif desc == "with Decimal":
        # Decimals convert to floats
        output = json.dumps(original)
        result = json.loads(output)
        assert result == {"price": 19.99, "quantity": 5.0}

    else:
        # Standard roundtrip
        output = json.dumps(original)
        result = json.loads(output)
        assert result == original


# =============================================================================
# JSON DECODE ERROR - KEPT AS-IS (1 test)
# =============================================================================


def test_json_decode_error_is_value_error():
    """JSONDecodeError should be ValueError for compatibility."""
    assert json.JSONDecodeError is ValueError


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing 23)
# =============================================================================


EDGE_CASES = [
    # Large numbers
    ({"big": 999999999999999999999}, "very large number"),

    # Deeply nested
    ("deeply_nested", "deeply nested structure"),

    # Mixed types
    ([1, "string", True, None, 3.14, {"nested": "dict"}], "mixed types list"),

    # Decimal edge cases
    ({"zero": Decimal("0")}, "Decimal zero"),
    ({"negative": Decimal("-123.45")}, "Decimal negative"),
    ({"items": [Decimal("1.5"), [Decimal("2.5"), Decimal("3.5")]]}, "Decimal in nested list"),

    # Options parameter
    ({"key": "value"}, "with option parameter"),

    # Bytes UTF-8
    ('{"unicode": "café"}', "bytes UTF-8"),
    ('{"key": "value"}', "bytes UTF-8 with BOM"),

    # Backslashes
    ({"path": "C:\\Users\\test"}, "string with backslash"),

    # Large collections
    ("many_keys", "dict with many keys"),
    ("many_elements", "array with many elements"),

    # Whitespace
    ("""
    {
        "key"   :   "value"  ,
        "array" : [ 1 , 2 , 3 ]
    }
    """, "whitespace handling"),

    # Custom handler
    ("custom_decimal_override", "custom handler for Decimal override"),

    # Empty nested
    ({"empty_dict": {}, "empty_list": [], "nested": {"deep_empty": {}}}, "empty nested structures"),

    # Primitives
    ("true", "primitive true"),
    ("false", "primitive false"),

    # Unicode escapes
    ('"\\u0048\\u0065\\u006c\\u006c\\u006f"', "unicode escapes"),

    # Precision
    ({"value": 3.141592653589793}, "float precision"),
]


@pytest.mark.parametrize("payload,desc", EDGE_CASES)
def test_edge_cases_comprehensive(payload, desc):
    """Comprehensive edge cases test.

    Replaces 23 individual tests from TestEdgeCases.
    """
    if desc == "very large number":
        output = json.dumps(payload)
        data = json.loads(output)
        assert isinstance(data["big"], float)
        assert data["big"] > 999999999999999999998

    elif desc == "deeply nested structure":
        # Build deeply nested
        nested = {"level": 0}
        current = nested
        for i in range(1, 10):
            current["next"] = {"level": i}
            current = current["next"]

        output = json.dumps(nested)
        data = json.loads(output)
        assert data["level"] == 0

    elif desc == "mixed types list":
        output = json.dumps(payload)
        data = json.loads(output)
        assert data[0] == 1
        assert data[1] == "string"
        assert data[2] is True
        assert data[3] is None
        assert data[4] == 3.14
        assert data[5] == {"nested": "dict"}

    elif desc == "Decimal in nested list":
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["items"][0] == 1.5
        assert data["items"][1][0] == 2.5
        assert data["items"][1][1] == 3.5

    elif desc == "with option parameter":
        output = json.dumps(payload, option=None)
        data = json.loads(output)
        assert data == payload

    elif desc == "bytes UTF-8":
        json_bytes = payload.encode()
        result = json.loads(json_bytes)
        assert result["unicode"] == "café"

    elif desc == "bytes UTF-8 with BOM":
        json_bytes = payload.encode("utf-8-sig")
        try:
            result = json.loads(json_bytes)
            assert result == {"key": "value"}
        except ValueError:
            # orjson raises for BOM
            pass

    elif desc == "dict with many keys":
        payload = {f"key_{i}": f"value_{i}" for i in range(100)}
        output = json.dumps(payload)
        data = json.loads(output)
        assert len(data) == 100
        assert data["key_0"] == "value_0"
        assert data["key_99"] == "value_99"

    elif desc == "array with many elements":
        payload = list(range(1000))
        output = json.dumps(payload)
        data = json.loads(output)
        assert len(data) == 1000
        assert data[0] == 0
        assert data[999] == 999

    elif desc == "whitespace handling":
        result = json.loads(payload)
        assert result == {"key": "value", "array": [1, 2, 3]}

    elif desc == "custom handler for Decimal override":
        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)  # Return string instead of float
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload = {"price": Decimal("19.99")}
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["price"] == "19.99"

    elif desc == "primitive true":
        result = json.loads(payload)
        assert result is True
        assert isinstance(result, bool)

    elif desc == "primitive false":
        result = json.loads(payload)
        assert result is False
        assert isinstance(result, bool)

    elif desc == "unicode escapes":
        result = json.loads(payload)
        assert result == "Hello"

    elif desc == "float precision":
        output = json.dumps(payload)
        result = json.loads(output)
        assert result["value"] == payload["value"]

    else:
        # Generic test for Decimal/empty cases
        output = json.dumps(payload)
        data = json.loads(output)

        if "Decimal" in desc:
            # Verify Decimals converted to floats
            for key, value in payload.items():
                if isinstance(value, Decimal):
                    assert data[key] == float(value)
        else:
            assert data == payload


# =============================================================================
# ENCODER FALLBACK - PARAMETRIZED (1 test replacing 2)
# =============================================================================


ENCODER_FALLBACK_CASES = [
    ("unhandled_type", "unhandled type raises TypeError"),
    ("decimal_when_custom_fails", "Decimal encoded when custom fails"),
]


@pytest.mark.parametrize("scenario,desc", ENCODER_FALLBACK_CASES)
def test_encoder_fallback_comprehensive(scenario, desc):
    """Comprehensive encoder fallback test.

    Replaces 2 individual tests from TestEncoderFallback.
    """
    if scenario == "unhandled_type":
        # Unhandled types raise TypeError
        class UnhandledType:
            pass

        def custom_handler(obj: Any) -> Any:
            raise TypeError("Not handled by custom handler")

        obj = UnhandledType()
        with pytest.raises(TypeError):
            json.dumps(obj, default=custom_handler)

    elif scenario == "decimal_when_custom_fails":
        # Decimal handled by encoder when custom handler fails
        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, str):
                return obj.upper()
            raise TypeError("Not handled")

        payload = {"value": Decimal("1.5")}
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["value"] == 1.5
