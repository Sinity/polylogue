"""Schema annotation quality checks against packaged provider schemas."""

from __future__ import annotations

import json
from typing import Any

import pytest


def _load_schema(provider: str) -> dict | None:
    """Load a packaged provider schema, returning None if absent."""
    try:
        import gzip
        from pathlib import Path

        schema_dir = Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"
        path = schema_dir / f"{provider}.schema.json.gz"
        if not path.exists():
            return None
        return json.loads(gzip.decompress(path.read_bytes()))
    except Exception:
        return None


def _find_annotations(schema: dict, prefix: str = "x-polylogue-") -> dict[str, list[tuple[str, Any]]]:
    """Walk the schema tree and collect all x-polylogue-* annotations by key."""
    result: dict[str, list[tuple[str, Any]]] = {}

    def _walk(obj: Any, path: str) -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if key.startswith(prefix):
                result.setdefault(key, []).append((path, value))
            if isinstance(value, dict):
                _walk(value, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        _walk(item, f"{path}.{key}[{index}]")

    _walk(schema, "$")
    return result


def _get_nested(schema: dict, dotpath: str) -> dict | None:
    """Navigate a schema by dot-separated property path."""
    current = schema
    for part in dotpath.split("."):
        if part == "additionalProperties":
            current = current.get("additionalProperties", {})
        else:
            current = current.get("properties", {}).get(part, {})
        if not current:
            return None
        if "anyOf" in current:
            for variant in current["anyOf"]:
                if variant.get("type") == "object" and "properties" in variant:
                    current = variant
                    break
    return current or None


class TestSchemaAnnotations:
    """Packaged schemas should expose coherent annotation metadata."""

    def test_chatgpt_role_enum(self):
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        role_schema = _get_nested(schema, "mapping.additionalProperties.message.author.role")
        assert role_schema is not None
        values = role_schema.get("x-polylogue-values", [])
        assert "user" in values
        assert "assistant" in values

    def test_chatgpt_uuid_format(self):
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        assert schema["properties"]["current_node"].get("x-polylogue-format") == "uuid4"
        node_id = _get_nested(schema, "mapping.additionalProperties.id")
        assert node_id is not None
        assert node_id.get("x-polylogue-format") == "uuid4"

    def test_chatgpt_timestamp_format(self):
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        create_time = schema["properties"].get("create_time", {})
        fmt = create_time.get("x-polylogue-format")
        rng = create_time.get("x-polylogue-range")
        assert fmt == "unix-epoch" or rng is not None

    def test_chatgpt_reference_detection(self):
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        assert schema["properties"]["current_node"].get("x-polylogue-ref") == "$.mapping"

    def test_claude_code_has_annotations(self):
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        annotations = _find_annotations(schema)
        total = sum(len(values) for values in annotations.values())
        assert total > 100
        assert "x-polylogue-format" in annotations
        assert "x-polylogue-values" in annotations
        assert "x-polylogue-frequency" in annotations

    def test_claude_code_type_enum(self):
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        type_values = schema.get("properties", {}).get("type", {}).get("x-polylogue-values", [])
        assert any(value in type_values for value in ("user", "assistant", "human"))

    def test_claude_ai_sender_enum(self):
        schema = _load_schema("claude-ai")
        if schema is None:
            pytest.skip("Claude AI schema not available")

        messages = schema.get("properties", {}).get("chat_messages", {})
        item = messages.get("items", {})
        if "anyOf" in item:
            for variant in item["anyOf"]:
                sender = variant.get("properties", {}).get("sender", {})
                if "x-polylogue-values" in sender:
                    values = sender["x-polylogue-values"]
                    assert "human" in values
                    assert "assistant" in values
                    return
        sender = item.get("properties", {}).get("sender", {})
        assert "human" in sender.get("x-polylogue-values", [])

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_frequency_values_in_range(self, provider):
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, frequency in _find_annotations(schema).get("x-polylogue-frequency", []):
            assert 0.0 < frequency < 1.0, f"{provider} {path}: frequency {frequency} not in (0, 1)"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_numeric_ranges_plausible(self, provider):
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, value_range in _find_annotations(schema).get("x-polylogue-range", []):
            assert isinstance(value_range, list) and len(value_range) == 2
            low, high = value_range
            assert low <= high, f"{provider} {path}: range inverted: {low} > {high}"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_format_values_are_known(self, provider):
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        known_formats = {
            "uuid4",
            "uuid",
            "hex-id",
            "iso8601",
            "unix-epoch",
            "unix-epoch-str",
            "base64",
            "url",
            "email",
            "mime-type",
        }
        for path, fmt in _find_annotations(schema).get("x-polylogue-format", []):
            assert fmt in known_formats, f"{provider} {path}: unknown format {fmt!r}"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_values_are_nonempty_lists(self, provider):
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        for path, values in _find_annotations(schema).get("x-polylogue-values", []):
            assert isinstance(values, list), f"{provider} {path}: values not a list"
            assert values, f"{provider} {path}: empty values list"
            assert all(isinstance(value, str) for value in values)
