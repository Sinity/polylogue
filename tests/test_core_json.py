from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from polylogue.core import json


class TestDumps:
    """Test json.dumps() functionality."""

    def test_dumps_simple_dict(self):
        payload = {"key": "value"}
        output = json.dumps(payload)
        assert json.loads(output) == payload

    def test_dumps_simple_list(self):
        payload = [1, 2, 3]
        output = json.dumps(payload)
        assert json.loads(output) == payload

    def test_dumps_nested_structure(self):
        payload = {"outer": {"inner": [1, 2, 3]}}
        output = json.dumps(payload)
        assert json.loads(output) == payload

    def test_dumps_handles_decimal(self):
        payload = {"value": Decimal("1.25")}

        output = json.dumps(payload)

        data = json.loads(output)
        assert data["value"] == 1.25

    def test_dumps_handles_decimal_as_string_representation(self):
        """Decimal should convert to float, not string."""
        payload = {"price": Decimal("99.99")}
        output = json.dumps(payload)
        data = json.loads(output)
        assert isinstance(data["price"], float)
        assert data["price"] == 99.99

    def test_dumps_handles_multiple_decimals(self):
        payload = {
            "value1": Decimal("10.5"),
            "value2": Decimal("20.75"),
            "nested": {"value3": Decimal("30.25")},
        }
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["value1"] == 10.5
        assert data["value2"] == 20.75
        assert data["nested"]["value3"] == 30.25

    def test_dumps_with_none_values(self):
        payload = {"key": None}
        output = json.dumps(payload)
        assert json.loads(output) == payload

    def test_dumps_with_boolean_values(self):
        payload = {"true_val": True, "false_val": False}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["true_val"] is True
        assert data["false_val"] is False

    def test_dumps_with_unicode_characters(self):
        payload = {"text": "Hello 世界 🌍"}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["text"] == "Hello 世界 🌍"

    def test_dumps_with_empty_dict(self):
        payload = {}
        output = json.dumps(payload)
        assert json.loads(output) == {}

    def test_dumps_with_empty_list(self):
        payload = []
        output = json.dumps(payload)
        assert json.loads(output) == []

    def test_dumps_with_numeric_keys_as_strings(self):
        """JSON keys are always strings."""
        payload = {"123": "value"}
        output = json.dumps(payload)
        data = json.loads(output)
        assert "123" in data

    def test_dumps_accepts_none_option(self):
        payload = {"value": 123}

        default_output = json.dumps(payload)
        explicit_output = json.dumps(payload, option=None)

        assert explicit_output == default_output

    def test_dumps_with_custom_default_handler(self):
        """Test custom default handler for non-serializable types."""

        class CustomType:
            def __init__(self, value):
                self.value = value

        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, CustomType):
                return {"custom": obj.value}
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload = CustomType(42)
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data == {"custom": 42}

    def test_dumps_custom_handler_fallback_to_encoder(self):
        """Custom handler that doesn't handle the type should fallback to encoder."""

        def custom_handler(obj: Any) -> Any:
            raise TypeError("Not handled")

        payload = {"decimal": Decimal("1.5")}
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["decimal"] == 1.5

    def test_dumps_with_special_numbers(self):
        payload = {
            "zero": 0,
            "negative": -42,
            "float": 3.14,
            "large": 999999999999,
        }
        output = json.dumps(payload)
        data = json.loads(output)
        assert data == payload

    def test_dumps_string_with_escaped_characters(self):
        payload = {"text": 'Line 1\nLine 2\tTabbed"Quoted"'}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["text"] == payload["text"]


class TestLoads:
    """Test json.loads() functionality."""

    def test_loads_from_string(self):
        json_str = '{"key": "value"}'
        result = json.loads(json_str)
        assert result == {"key": "value"}

    def test_loads_from_bytes(self):
        json_bytes = b'{"key": "value"}'
        result = json.loads(json_bytes)
        assert result == {"key": "value"}

    def test_loads_array(self):
        json_str = "[1, 2, 3]"
        result = json.loads(json_str)
        assert result == [1, 2, 3]

    def test_loads_nested_structure(self):
        json_str = '{"outer": {"inner": [1, 2]}}'
        result = json.loads(json_str)
        assert result == {"outer": {"inner": [1, 2]}}

    def test_loads_with_unicode(self):
        json_str = '{"text": "Hello 世界"}'
        result = json.loads(json_str)
        assert result["text"] == "Hello 世界"

    def test_loads_with_null(self):
        json_str = '{"value": null}'
        result = json.loads(json_str)
        assert result["value"] is None

    def test_loads_with_boolean(self):
        json_str = '{"true": true, "false": false}'
        result = json.loads(json_str)
        assert result["true"] is True
        assert result["false"] is False

    def test_loads_primitive_number(self):
        assert json.loads("42") == 42
        assert json.loads("3.14") == 3.14

    def test_loads_primitive_string(self):
        result = json.loads('"hello"')
        assert result == "hello"

    def test_loads_primitive_null(self):
        assert json.loads("null") is None

    def test_loads_primitive_boolean(self):
        assert json.loads("true") is True
        assert json.loads("false") is False

    def test_loads_empty_structures(self):
        assert json.loads("{}") == {}
        assert json.loads("[]") == []

    def test_loads_with_escaped_quotes(self):
        json_str = '{"text": "He said \\"hello\\""}'
        result = json.loads(json_str)
        assert result["text"] == 'He said "hello"'

    def test_loads_with_newlines_in_string(self):
        json_str = '{"text": "Line 1\\nLine 2"}'
        result = json.loads(json_str)
        assert result["text"] == "Line 1\nLine 2"


class TestRoundtrip:
    """Test that dumps -> loads preserves data (idempotency)."""

    def test_roundtrip_dict(self):
        original = {"a": 1, "b": "text", "c": None}
        output = json.dumps(original)
        result = json.loads(output)
        assert result == original

    def test_roundtrip_nested(self):
        original = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                },
            },
        }
        output = json.dumps(original)
        result = json.loads(output)
        assert result == original

    def test_roundtrip_with_decimal(self):
        original = {"price": Decimal("19.99"), "quantity": Decimal("5")}
        output = json.dumps(original)
        result = json.loads(output)
        # Decimals convert to floats
        assert result == {"price": 19.99, "quantity": 5.0}

    def test_roundtrip_with_unicode(self):
        original = {"emoji": "🚀", "chinese": "中文", "mixed": "Hello 世界 🌍"}
        output = json.dumps(original)
        result = json.loads(output)
        assert result == original

    def test_roundtrip_multiple_times(self):
        """Ensure multiple roundtrips are stable."""
        original = {"a": 1, "b": [2, 3]}
        data = original
        for _ in range(3):
            output = json.dumps(data)
            data = json.loads(output)
        assert data == original


class TestJSONDecodeError:
    """Test JSONDecodeError compatibility export."""

    def test_json_decode_error_is_value_error(self):
        """JSONDecodeError should be ValueError for compatibility."""
        assert json.JSONDecodeError is ValueError


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dumps_with_very_large_number(self):
        """Test handling of very large numbers (converted to float in JSON)."""
        payload = {"big": 999999999999999999999}
        output = json.dumps(payload)
        data = json.loads(output)
        # Large integers become floats in JSON, so we check approximate value
        assert isinstance(data["big"], float)
        assert data["big"] > 999999999999999999998

    def test_dumps_deeply_nested_structure(self):
        """Test deeply nested structures."""
        nested = {"level": 0}
        current = nested
        for i in range(1, 10):
            current["next"] = {"level": i}
            current = current["next"]

        output = json.dumps(nested)
        data = json.loads(output)
        assert data["level"] == 0
        # Navigate to deepest level
        current = data
        for i in range(1, 10):
            assert current.get("next", {}).get("level") == i
            current = current.get("next", {})

    def test_dumps_list_with_mixed_types(self):
        """Test list containing mixed types."""
        payload = [1, "string", True, None, 3.14, {"nested": "dict"}]
        output = json.dumps(payload)
        data = json.loads(output)
        assert data[0] == 1
        assert data[1] == "string"
        assert data[2] is True
        assert data[3] is None
        assert data[4] == 3.14
        assert data[5] == {"nested": "dict"}

    def test_dumps_decimal_zero(self):
        """Test Decimal zero."""
        payload = {"zero": Decimal("0")}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["zero"] == 0.0

    def test_dumps_decimal_negative(self):
        """Test negative Decimal."""
        payload = {"negative": Decimal("-123.45")}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["negative"] == -123.45

    def test_dumps_with_option_parameter(self):
        """Test dumps with explicit option parameter (e.g., orjson option flags)."""
        # Using option=None should work (default behavior)
        payload = {"key": "value"}
        output = json.dumps(payload, option=None)
        data = json.loads(output)
        assert data == payload

    def test_loads_bytes_utf8(self):
        """Test loads with UTF-8 encoded bytes."""
        json_bytes = '{"unicode": "café"}'.encode()
        result = json.loads(json_bytes)
        assert result["unicode"] == "café"

    def test_loads_bytes_utf8_with_bom(self):
        """Test loads with UTF-8 BOM bytes (orjson doesn't support BOM, raises error)."""
        json_bytes = '{"key": "value"}'.encode("utf-8-sig")
        # orjson explicitly doesn't support BOM, should raise an error
        try:
            result = json.loads(json_bytes)
            # If it somehow succeeds, that's fine too (future versions might support it)
            assert result == {"key": "value"}
        except ValueError:
            # orjson raises JSONDecodeError (ValueError) for BOM
            pass

    def test_dumps_string_with_backslash(self):
        """Test strings containing backslashes."""
        payload = {"path": "C:\\Users\\test"}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["path"] == "C:\\Users\\test"

    def test_dumps_dict_with_many_keys(self):
        """Test dict with many keys."""
        payload = {f"key_{i}": f"value_{i}" for i in range(100)}
        output = json.dumps(payload)
        data = json.loads(output)
        assert len(data) == 100
        assert data["key_0"] == "value_0"
        assert data["key_99"] == "value_99"

    def test_dumps_array_with_many_elements(self):
        """Test array with many elements."""
        payload = list(range(1000))
        output = json.dumps(payload)
        data = json.loads(output)
        assert len(data) == 1000
        assert data[0] == 0
        assert data[999] == 999

    def test_dumps_decimal_in_nested_list(self):
        """Test Decimal in nested list."""
        payload = {"items": [Decimal("1.5"), [Decimal("2.5"), Decimal("3.5")]]}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["items"][0] == 1.5
        assert data["items"][1][0] == 2.5
        assert data["items"][1][1] == 3.5

    def test_loads_whitespace_handling(self):
        """Test loads with various whitespace."""
        json_str = """
        {
            "key"   :   "value"  ,
            "array" : [ 1 , 2 , 3 ]
        }
        """
        result = json.loads(json_str)
        assert result == {"key": "value", "array": [1, 2, 3]}

    def test_dumps_with_custom_handler_for_decimal_override(self):
        """Test custom handler that overrides default Decimal handling."""

        def custom_handler(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)  # Return string instead of float
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload = {"price": Decimal("19.99")}
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["price"] == "19.99"

    def test_dumps_empty_nested_structures(self):
        """Test empty dicts and lists in nested structures."""
        payload = {"empty_dict": {}, "empty_list": [], "nested": {"deep_empty": {}}}
        output = json.dumps(payload)
        data = json.loads(output)
        assert data["empty_dict"] == {}
        assert data["empty_list"] == []
        assert data["nested"]["deep_empty"] == {}

    def test_loads_primitive_true(self):
        """Test loading primitive true value."""
        result = json.loads("true")
        assert result is True
        assert isinstance(result, bool)

    def test_loads_primitive_false(self):
        """Test loading primitive false value."""
        result = json.loads("false")
        assert result is False
        assert isinstance(result, bool)

    def test_loads_string_with_unicode_escapes(self):
        """Test loading strings with unicode escape sequences."""
        json_str = '"\\u0048\\u0065\\u006c\\u006c\\u006f"'
        result = json.loads(json_str)
        assert result == "Hello"

    def test_roundtrip_preserves_number_precision(self):
        """Test that float precision is preserved in roundtrip."""
        original = {"value": 3.141592653589793}
        output = json.dumps(original)
        result = json.loads(output)
        assert result["value"] == original["value"]


class TestEncoderFallback:
    """Test encoder fallback behavior."""

    def test_default_encoder_with_unhandled_type(self):
        """Test that unhandled types raise TypeError after custom handler fails."""

        class UnhandledType:
            pass

        def custom_handler(obj: Any) -> Any:
            # This handler doesn't handle UnhandledType, so encoder will raise
            raise TypeError("Not handled by custom handler")

        obj = UnhandledType()
        # Both orjson and stdlib json should raise TypeError for unhandled types
        with pytest.raises(TypeError):
            json.dumps(obj, default=custom_handler)

    def test_default_encoder_handles_decimal_when_custom_fails(self):
        """Test that when custom handler fails, Decimal is still encoded by default encoder."""

        def custom_handler(obj: Any) -> Any:
            # Handler that only handles specific types
            if isinstance(obj, str):
                return obj.upper()
            raise TypeError("Not handled")

        payload = {"value": Decimal("1.5")}
        # Should not raise - encoder will handle Decimal after custom fails
        output = json.dumps(payload, default=custom_handler)
        data = json.loads(output)
        assert data["value"] == 1.5
