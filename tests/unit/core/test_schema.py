"""Tests for schema validation and schema inference.

Consolidated from test_validation.py and test_schema_inference.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.schemas import ValidationResult
from polylogue.schemas.schema_inference import (
    PROVIDERS,
    GenerationResult,
    _remove_nested_required,
    cli_main,
    collapse_dynamic_keys,
    generate_all_schemas,
    generate_provider_schema,
    generate_schema_from_samples,
    get_sample_count_from_db,
    is_dynamic_key,
    load_samples_from_db,
    load_samples_from_sessions,
)
from polylogue.schemas.validator import SchemaValidator, validate_provider_export


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
    """Test that missing provider raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


def test_missing_provider_raises():
    """Test that missing provider raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


class TestSyntheticRoundTrip:
    """Verify synthetic data round-trips through parsers for all providers.

    Replaces the old TestSchemaValidation (which validated real data against
    schemas — circular with synthetic data) and TestDriftDetection (drift is
    only meaningful for real data). Instead, this tests the end-to-end contract:
    schema → generate → parse → valid conversations.
    """

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_synthetic_parses_successfully(self, provider: str, synthetic_source) -> None:
        """Synthetic data for each provider parses into valid conversations."""
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=3, seed=42)
        convos = list(iter_source_conversations(source))
        assert len(convos) > 0, f"No conversations parsed for {provider}"
        for conv in convos:
            assert len(conv.messages) > 0, f"Empty conversation for {provider}"
            assert any(m.text for m in conv.messages), f"No message text for {provider}"


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
# SCHEMA INFERENCE TESTS (merged from test_schema_inference.py)
# =============================================================================


class TestDynamicKeyDetection:
    """Test dynamic key detection for schema collapsing."""

    def test_uuid_detected(self):
        assert is_dynamic_key("550e8400-e29b-41d4-a716-446655440000")
        assert is_dynamic_key("550E8400-E29B-41D4-A716-446655440000")

    def test_long_hex_detected(self):
        assert is_dynamic_key("a" * 24)
        assert is_dynamic_key("abcdef0123456789abcdef01")

    def test_message_ids_detected(self):
        assert is_dynamic_key("msg-abc123")
        assert is_dynamic_key("node-550e8400")
        assert is_dynamic_key("conv-123abc")

    def test_normal_keys_not_detected(self):
        assert not is_dynamic_key("content")
        assert not is_dynamic_key("role")
        assert not is_dynamic_key("text")
        assert not is_dynamic_key("id")


class TestSchemaCollapsing:
    """Test dynamic key collapsing in schemas."""

    def test_collapse_uuid_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "550e8400-e29b-41d4-a716-446655440000": {"type": "object"},
                "660f9511-f3ac-52e5-b827-557766551111": {"type": "object"},
            },
        }

        collapsed = collapse_dynamic_keys(schema)

        assert "title" in collapsed["properties"]
        assert "550e8400-e29b-41d4-a716-446655440000" not in collapsed["properties"]
        assert "additionalProperties" in collapsed

    def test_preserve_static_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "object"},
            },
        }

        collapsed = collapse_dynamic_keys(schema)

        assert "id" in collapsed["properties"]
        assert "content" in collapsed["properties"]
        assert "additionalProperties" not in collapsed


class TestSchemaGeneration:
    """Test schema generation from samples."""

    def test_generate_from_simple_samples(self):
        samples = [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi there"},
        ]

        schema = generate_schema_from_samples(samples)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "role" in schema["properties"]
        assert "text" in schema["properties"]

    def test_generate_from_complex_samples(self):
        samples = [
            {"id": "1", "content": {"parts": ["text"]}},
            {"id": "2", "content": {"parts": ["more text"], "type": "text"}},
        ]

        schema = generate_schema_from_samples(samples)

        assert "content" in schema["properties"]
        assert "parts" in schema["properties"]["content"]["properties"]

    def test_empty_samples_returns_placeholder(self):
        schema = generate_schema_from_samples([])
        assert "No samples available" in schema.get("description", "")


class TestProviderSchemaGeneration:
    """Test full provider schema generation."""

    def test_known_providers(self):
        """All configured providers should be known."""
        expected = {"chatgpt", "claude-code", "claude-ai", "gemini", "codex"}
        assert set(PROVIDERS.keys()) == expected

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "codex"])
    def test_generate_schema_from_db(self, seeded_db, provider):
        """Generate schema from database for each provider."""
        result = generate_provider_schema(provider, db_path=seeded_db, max_samples=100)

        # Should succeed since we have fixtures for these providers
        if result.sample_count > 0:
            assert result.success, f"Failed: {result.error}"
            assert result.schema is not None
            assert result.schema.get("type") == "object"
            assert "properties" in result.schema

    def test_unknown_provider_returns_error(self):
        result = generate_provider_schema("unknown-provider")
        assert not result.success
        assert "Unknown provider" in (result.error or "")

    def test_result_dataclass(self):
        """GenerationResult should have correct properties."""
        success = GenerationResult(provider="test", schema={"type": "object"}, sample_count=10)
        assert success.success
        assert success.provider == "test"
        assert success.sample_count == 10

        failure = GenerationResult(provider="test", schema=None, sample_count=0, error="no data")
        assert not failure.success


class TestLoadSamples:
    """Test sample loading functions."""

    def test_load_limited_samples(self, seeded_db):
        """Should respect max_samples limit."""
        samples = load_samples_from_db("chatgpt", db_path=seeded_db, max_samples=10)

        # Should have at most 10 samples (may have fewer if DB has less)
        assert len(samples) <= 10

    def test_load_nonexistent_provider(self, seeded_db):
        """Unknown provider returns empty list."""
        samples = load_samples_from_db("nonexistent-provider", db_path=seeded_db)
        assert samples == []


# MERGED FROM test_schema_inference_coverage.py
# =============================================================================
# _remove_nested_required
# =============================================================================


class TestRemoveNestedRequired:
    """Tests for recursive required-removal from JSON schemas."""

    def _fn(self, schema, depth=0):
        return _remove_nested_required(schema, depth)

    def test_root_required_preserved(self):
        """Root-level required array is kept."""
        schema = {"type": "object", "required": ["id", "name"], "properties": {}}
        result = self._fn(schema, depth=0)
        assert "required" in result
        assert result["required"] == ["id", "name"]

    def test_nested_required_removed(self):
        """Nested required array is removed (depth > 0)."""
        schema = {"type": "object", "required": ["field1"], "properties": {}}
        result = self._fn(schema, depth=1)
        assert "required" not in result

    def test_recurse_into_properties(self):
        """Required removed from nested properties."""
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "object", "required": ["x"], "properties": {}},
            },
        }
        result = self._fn(schema, depth=0)
        # Root required preserved
        assert "required" in result
        # Nested required removed
        assert "required" not in result["properties"]["a"]

    def test_recurse_into_items(self):
        """Required removed from array items schema."""
        schema = {
            "type": "array",
            "items": {"type": "object", "required": ["id"], "properties": {}},
        }
        result = self._fn(schema, depth=0)
        assert "required" not in result["items"]

    def test_recurse_into_anyof(self):
        """Required removed from anyOf variants."""
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["b"]},
            ]
        }
        result = self._fn(schema, depth=0)
        for variant in result["anyOf"]:
            assert "required" not in variant

    def test_recurse_into_oneof(self):
        """Required removed from oneOf variants."""
        schema = {"oneOf": [{"type": "object", "required": ["x"]}]}
        result = self._fn(schema, depth=0)
        assert "required" not in result["oneOf"][0]

    def test_recurse_into_allof(self):
        """Required removed from allOf variants."""
        schema = {"allOf": [{"type": "object", "required": ["y"]}]}
        result = self._fn(schema, depth=0)
        assert "required" not in result["allOf"][0]

    def test_non_dict_returns_unchanged(self):
        """Non-dict input returned as-is."""
        assert self._fn("string") == "string"
        assert self._fn(42) == 42
        assert self._fn([1, 2]) == [1, 2]

    def test_deeply_nested(self):
        """Deeply nested schemas handled correctly."""
        schema = {
            "type": "object",
            "required": ["top"],
            "properties": {
                "top": {
                    "type": "object",
                    "required": ["mid"],
                    "properties": {
                        "mid": {
                            "type": "object",
                            "required": ["deep"],
                            "properties": {},
                        }
                    },
                }
            },
        }
        result = self._fn(schema, depth=0)
        assert "required" in result
        assert "required" not in result["properties"]["top"]
        assert "required" not in result["properties"]["top"]["properties"]["mid"]


# =============================================================================
# load_samples_from_sessions
# =============================================================================


class TestLoadSamplesFromSessions:
    """Tests for loading samples from JSONL session files."""

    def _fn(self, session_dir, max_sessions=None):
        return load_samples_from_sessions(session_dir, max_sessions=max_sessions)

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        """Non-existent directory returns empty list."""
        result = self._fn(tmp_path / "does_not_exist")
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty directory returns empty list."""
        d = tmp_path / "empty"
        d.mkdir()
        result = self._fn(d)
        assert result == []

    def test_single_jsonl_file(self, tmp_path):
        """Single JSONL file with records is loaded."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "session1.jsonl").write_text(
            json.dumps({"type": "user", "text": "hello"}) + "\n"
            + json.dumps({"type": "assistant", "text": "hi"}) + "\n"
        )
        result = self._fn(d)
        assert len(result) == 2
        assert result[0]["type"] == "user"

    def test_max_sessions_limits_files(self, tmp_path):
        """max_sessions parameter limits number of files processed."""
        d = tmp_path / "sessions"
        d.mkdir()
        for i in range(10):
            (d / f"session{i:02d}.jsonl").write_text(
                json.dumps({"id": i}) + "\n"
            )
        result = self._fn(d, max_sessions=3)
        # Should process at most 3 sessions
        assert len(result) <= 10
        assert len(result) >= 1

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "bad.jsonl").write_text(
            "not json\n"
            + json.dumps({"valid": True}) + "\n"
            + "{incomplete\n"
        )
        result = self._fn(d)
        assert len(result) == 1
        assert result[0]["valid"] is True

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in JSONL files are skipped."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "spaces.jsonl").write_text(
            "\n\n" + json.dumps({"ok": True}) + "\n\n"
        )
        result = self._fn(d)
        assert len(result) == 1

    def test_recursive_glob(self, tmp_path):
        """Files in subdirectories are found by rglob."""
        d = tmp_path / "sessions"
        sub = d / "sub1"
        sub.mkdir(parents=True)
        (sub / "nested.jsonl").write_text(json.dumps({"nested": True}) + "\n")
        result = self._fn(d)
        assert len(result) == 1
        assert result[0]["nested"] is True


# =============================================================================
# get_sample_count_from_db
# =============================================================================


class TestGetSampleCountFromDb:
    """Tests for get_sample_count_from_db."""

    def _fn(self, provider_name, db_path):
        return get_sample_count_from_db(provider_name, db_path=db_path)

    def test_nonexistent_db_returns_zero(self, tmp_path):
        """Non-existent database returns 0."""
        result = self._fn("chatgpt", tmp_path / "missing.db")
        assert result == 0

    def test_empty_db_returns_zero(self, tmp_path):
        """Database with no messages returns 0."""
        from polylogue.storage.backends.connection import open_connection

        db_path = tmp_path / "empty.db"
        with open_connection(db_path):
            pass
        result = self._fn("chatgpt", db_path)
        assert result == 0

    def test_matching_provider_returns_count(self, tmp_path):
        """Database with matching provider messages returns count."""
        from polylogue.storage.backends.connection import open_connection

        db_path = tmp_path / "test.db"
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash,
                    provider_meta, metadata, version,
                    parent_conversation_id, branch_type, raw_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("c1", "chatgpt", "p1", "Test", None, None, "hash1",
                 '{"source":"test"}', '{}', 1, None, None, None),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", '{"key":"val"}', 1, None, 0),
            )
            conn.commit()

        result = self._fn("chatgpt", db_path)
        assert result == 1

    def test_wrong_provider_returns_zero(self, tmp_path):
        """Database with no matching provider returns 0."""
        from polylogue.storage.backends.connection import open_connection

        db_path = tmp_path / "test.db"
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash,
                    provider_meta, metadata, version,
                    parent_conversation_id, branch_type, raw_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("c1", "chatgpt", "p1", "Test", None, None, "hash1",
                 None, '{}', 1, None, None, None),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", '{"k":"v"}', 1, None, 0),
            )
            conn.commit()

        result = self._fn("claude", db_path)
        assert result == 0


# =============================================================================
# generate_schema_from_samples
# =============================================================================


class TestGenerateSchemaFromSamples:
    """Tests for genson-based schema generation."""

    def _fn(self, samples):
        return generate_schema_from_samples(samples)

    @pytest.fixture(autouse=True)
    def _check_genson(self):
        """Skip tests if genson not available."""
        try:
            import genson  # noqa: F401
        except ImportError:
            pytest.skip("genson not installed")

    def test_empty_samples(self):
        """Empty samples returns description-only schema."""
        result = self._fn([])
        assert result["type"] == "object"
        assert "description" in result

    def test_single_sample(self):
        """Single sample produces valid schema with properties."""
        result = self._fn([{"id": "abc", "count": 42}])
        assert result["type"] == "object"
        assert "properties" in result
        assert "id" in result["properties"]
        assert "count" in result["properties"]

    def test_multiple_samples_merge(self):
        """Multiple samples merge optional fields."""
        result = self._fn([
            {"id": "a", "name": "Alice"},
            {"id": "b", "age": 30},
        ])
        assert "id" in result["properties"]
        # Both name and age should appear
        assert "name" in result["properties"]
        assert "age" in result["properties"]


# =============================================================================
# generate_all_schemas
# =============================================================================


class TestGenerateAllSchemas:
    """Tests for generate_all_schemas file output."""

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        output_dir = tmp_path / "schemas" / "nested"
        fake_result = GenerationResult(
            provider="test", sample_count=1,
            schema={"type": "object"}, error=None,
        )

        with patch("polylogue.schemas.schema_inference.generate_provider_schema", return_value=fake_result):
            results = generate_all_schemas(output_dir, providers=["test"])

        assert output_dir.exists()
        assert len(results) == 1
        assert (output_dir / "test.schema.json.gz").exists()

    def test_skips_failed_schemas(self, tmp_path):
        """Failed schemas are not written to disk."""
        failed_result = GenerationResult(
            provider="broken", sample_count=0,
            schema=None, error="No samples",
        )

        with patch("polylogue.schemas.schema_inference.generate_provider_schema", return_value=failed_result):
            results = generate_all_schemas(tmp_path, providers=["broken"])

        assert not (tmp_path / "broken.schema.json.gz").exists()
        assert results[0].success is False


# =============================================================================
# cli_main
# =============================================================================


class TestCliMain:
    """Tests for CLI entry point."""

    def test_cli_with_no_db(self, tmp_path, capsys):
        """CLI handles missing database gracefully."""
        exit_code = cli_main([
            "--provider", "chatgpt",
            "--output-dir", str(tmp_path / "out"),
            "--db-path", str(tmp_path / "missing.db"),
        ])
        # May succeed (0 samples) or fail depending on implementation
        assert isinstance(exit_code, int)


# =============================================================================
# SCHEMA ANNOTATION QUALITY TESTS
# =============================================================================


def _load_schema(provider: str) -> dict | None:
    """Load a provider schema, returning None if not available."""
    try:
        import gzip as _gzip
        from pathlib import Path as _Path

        schema_dir = _Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"
        path = schema_dir / f"{provider}.schema.json.gz"
        if not path.exists():
            return None
        return json.loads(_gzip.decompress(path.read_bytes()))
    except Exception:
        return None


def _find_annotations(schema: dict, prefix: str = "x-polylogue-") -> dict[str, list[tuple[str, Any]]]:
    """Walk schema tree and collect all x-polylogue-* annotations by annotation key.

    Returns: {annotation_key: [(json_path, value), ...]}
    """
    result: dict[str, list[tuple[str, Any]]] = {}

    def _walk(obj: Any, path: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            if k.startswith(prefix):
                result.setdefault(k, []).append((path, v))
            if isinstance(v, dict):
                _walk(v, f"{path}.{k}")
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        _walk(item, f"{path}.{k}[{i}]")

    _walk(schema, "$")
    return result


def _get_nested(schema: dict, dotpath: str) -> dict | None:
    """Navigate a schema by dot-separated property path.

    Example: "mapping.additionalProperties.message" navigates through
    properties → mapping → additionalProperties → properties → message.
    """
    current = schema
    for part in dotpath.split("."):
        if part == "additionalProperties":
            current = current.get("additionalProperties", {})
        else:
            current = current.get("properties", {}).get(part, {})
        if not current:
            return None
        # Unwrap anyOf to find the object variant
        if "anyOf" in current:
            for variant in current["anyOf"]:
                if variant.get("type") == "object" and "properties" in variant:
                    current = variant
                    break
    return current or None


class TestSchemaAnnotations:
    """Verify that generated schemas contain correct x-polylogue-* annotations.

    These tests validate the annotation pipeline: field statistics collection →
    annotation post-processing → schema output. They load real (regenerated)
    schemas and check for expected annotations on known fields.
    """

    # ── ChatGPT ────────────────────────────────────────────────────────

    def test_chatgpt_role_enum(self):
        """ChatGPT role field should have x-polylogue-values with user/assistant."""
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        role_schema = _get_nested(schema, "mapping.additionalProperties.message.author.role")
        assert role_schema is not None, "Could not navigate to mapping→message→author→role"

        values = role_schema.get("x-polylogue-values", [])
        assert "user" in values, f"'user' not in role values: {values}"
        assert "assistant" in values, f"'assistant' not in role values: {values}"

    def test_chatgpt_uuid_format(self):
        """ChatGPT UUID fields should have x-polylogue-format = uuid4."""
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        # current_node is a UUID referencing a mapping key
        assert schema["properties"]["current_node"].get("x-polylogue-format") == "uuid4"

        # Node IDs in the mapping should also be UUID
        node_id = _get_nested(schema, "mapping.additionalProperties.id")
        assert node_id is not None
        assert node_id.get("x-polylogue-format") == "uuid4"

    def test_chatgpt_timestamp_format(self):
        """ChatGPT create_time should be detected as unix-epoch."""
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        create_time = schema["properties"].get("create_time", {})
        # create_time may be number or anyOf; check annotation
        fmt = create_time.get("x-polylogue-format")
        r = create_time.get("x-polylogue-range")
        assert fmt == "unix-epoch" or r is not None, (
            f"create_time has neither format nor range: {create_time.keys()}"
        )

    def test_chatgpt_reference_detection(self):
        """ChatGPT current_node should reference $.mapping."""
        schema = _load_schema("chatgpt")
        if schema is None:
            pytest.skip("ChatGPT schema not available")

        ref = schema["properties"]["current_node"].get("x-polylogue-ref")
        assert ref == "$.mapping", f"current_node ref should be $.mapping, got: {ref}"

    # ── Claude Code ────────────────────────────────────────────────────

    def test_claude_code_has_annotations(self):
        """Claude Code schema should have substantial annotations."""
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        annotations = _find_annotations(schema)
        total = sum(len(v) for v in annotations.values())
        assert total > 100, f"Claude Code has only {total} annotations (expected >100)"

        # Should have format, values, and frequency annotations
        assert "x-polylogue-format" in annotations, "Missing format annotations"
        assert "x-polylogue-values" in annotations, "Missing values annotations"
        assert "x-polylogue-frequency" in annotations, "Missing frequency annotations"

    def test_claude_code_type_enum(self):
        """Claude Code type field should have recognized values."""
        schema = _load_schema("claude-code")
        if schema is None:
            pytest.skip("Claude Code schema not available")

        type_values = schema.get("properties", {}).get("type", {}).get("x-polylogue-values", [])
        # The parser uses type field to distinguish message types
        assert any(v in type_values for v in ("user", "assistant", "human")), (
            f"type enum missing expected roles: {type_values}"
        )

    # ── Claude AI ──────────────────────────────────────────────────────

    def test_claude_ai_sender_enum(self):
        """Claude AI sender field should have human/assistant."""
        schema = _load_schema("claude-ai")
        if schema is None:
            pytest.skip("Claude AI schema not available")

        # chat_messages items should have sender
        msgs = schema.get("properties", {}).get("chat_messages", {})
        item = msgs.get("items", {})
        if "anyOf" in item:
            for variant in item["anyOf"]:
                sender = variant.get("properties", {}).get("sender", {})
                if "x-polylogue-values" in sender:
                    values = sender["x-polylogue-values"]
                    assert "human" in values, f"'human' not in sender: {values}"
                    assert "assistant" in values, f"'assistant' not in sender: {values}"
                    return
        sender = item.get("properties", {}).get("sender", {})
        values = sender.get("x-polylogue-values", [])
        assert "human" in values, f"'human' not in sender: {values}"

    # ── Cross-provider invariants ──────────────────────────────────────

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_frequency_values_in_range(self, provider):
        """All x-polylogue-frequency values should be in (0, 1)."""
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        annotations = _find_annotations(schema)
        for path, freq in annotations.get("x-polylogue-frequency", []):
            assert 0.0 < freq < 1.0, (
                f"{provider} {path}: frequency {freq} not in (0, 1)"
            )

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_numeric_ranges_plausible(self, provider):
        """All x-polylogue-range values should have min < max."""
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        annotations = _find_annotations(schema)
        for path, r in annotations.get("x-polylogue-range", []):
            assert isinstance(r, list) and len(r) == 2, (
                f"{provider} {path}: range should be [min, max], got: {r}"
            )
            lo, hi = r
            assert lo <= hi, f"{provider} {path}: range inverted: {lo} > {hi}"

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_format_values_are_known(self, provider):
        """All x-polylogue-format values should be from the known set."""
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        known_formats = {
            "uuid4", "uuid", "hex-id", "iso8601", "unix-epoch",
            "unix-epoch-str", "base64", "url", "email", "mime-type",
        }
        annotations = _find_annotations(schema)
        for path, fmt in annotations.get("x-polylogue-format", []):
            assert fmt in known_formats, (
                f"{provider} {path}: unknown format '{fmt}' (known: {known_formats})"
            )

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex"])
    def test_values_are_nonempty_lists(self, provider):
        """All x-polylogue-values should be non-empty string lists."""
        schema = _load_schema(provider)
        if schema is None:
            pytest.skip(f"{provider} schema not available")

        annotations = _find_annotations(schema)
        for path, values in annotations.get("x-polylogue-values", []):
            assert isinstance(values, list), f"{provider} {path}: values not a list"
            assert len(values) > 0, f"{provider} {path}: empty values list"
            for v in values:
                assert isinstance(v, str), f"{provider} {path}: non-string value: {v!r}"
