"""Focused schema-inference and schema-generation tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from polylogue.schemas.schema_inference import (
    PROVIDERS,
    GenerationResult,
    _remove_nested_required,
    cli_main,
    generate_all_schemas,
    generate_provider_schema,
    generate_schema_from_samples,
    get_sample_count_from_db,
    load_samples_from_db,
    load_samples_from_sessions,
)


class TestProviderSchemaGeneration:
    """Provider-level schema generation entrypoints."""

    def test_known_providers(self):
        expected_core = {"chatgpt", "claude-code", "claude-ai", "gemini", "codex"}
        assert expected_core.issubset(PROVIDERS.keys())
        assert all(name.strip() for name in PROVIDERS)

    @pytest.mark.slow
    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "codex"])
    def test_generate_schema_from_db(self, seeded_db, provider):
        result = generate_provider_schema(provider, db_path=seeded_db, max_samples=100)
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
        success = GenerationResult(provider="test", schema={"type": "object"}, sample_count=10)
        assert success.success
        assert success.provider == "test"
        assert success.sample_count == 10

        failure = GenerationResult(provider="test", schema=None, sample_count=0, error="no data")
        assert not failure.success


class TestLoadSamples:
    """Database-backed sample loading behavior."""

    def test_load_limited_samples(self, seeded_db):
        samples = load_samples_from_db("chatgpt", db_path=seeded_db, max_samples=10)
        assert len(samples) <= 10

    def test_load_nonexistent_provider(self, seeded_db):
        assert load_samples_from_db("nonexistent-provider", db_path=seeded_db) == []

    def test_load_limited_document_samples_stops_without_full_materialization(self, monkeypatch, tmp_path):
        db_path = tmp_path / "samples.db"
        db_path.write_text("")

        def _iter(*args, **kwargs):
            yield {"id": "one"}
            yield {"id": "two"}
            raise AssertionError("iterator should not be exhausted past the limit")

        monkeypatch.setattr("polylogue.schemas.sampling._iter_samples_from_db", _iter)
        samples = load_samples_from_db("chatgpt", db_path=db_path, max_samples=2)
        assert samples == [{"id": "one"}, {"id": "two"}]


def test_remove_nested_required_non_dict_returns_unchanged():
    """Non-dict values should pass through recursive cleanup unchanged."""
    assert _remove_nested_required("string", 0) == "string"
    assert _remove_nested_required(42, 0) == 42
    assert _remove_nested_required([1, 2], 0) == [1, 2]


class TestLoadSamplesFromSessions:
    """Session-dir sample loading edge cases."""

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert load_samples_from_sessions(tmp_path / "does_not_exist") == []

    def test_empty_dir_returns_empty(self, tmp_path):
        session_dir = tmp_path / "empty"
        session_dir.mkdir()
        assert load_samples_from_sessions(session_dir) == []

    def test_max_sessions_limits_files(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        for i in range(10):
            (session_dir / f"session{i:02d}.jsonl").write_text(json.dumps({"id": i}) + "\n")

        result = load_samples_from_sessions(session_dir, max_sessions=3)
        assert 1 <= len(result) <= 10


class TestGetSampleCountFromDb:
    """Sample-count queries against persisted conversations."""

    def test_nonexistent_db_returns_zero(self, tmp_path):
        assert get_sample_count_from_db("chatgpt", db_path=tmp_path / "missing.db") == 0

    def test_empty_db_returns_zero(self, tmp_path):
        from polylogue.storage.backends.connection import open_connection

        db_path = tmp_path / "empty.db"
        with open_connection(db_path):
            pass
        assert get_sample_count_from_db("chatgpt", db_path=db_path) == 0

    def test_matching_provider_returns_count(self, tmp_path):
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
                (
                    "c1",
                    "chatgpt",
                    "p1",
                    "Test",
                    None,
                    None,
                    "hash1",
                    '{"source":"test"}',
                    '{}',
                    1,
                    None,
                    None,
                    None,
                ),
            )
            conn.execute(
                """INSERT INTO messages
                   (message_id, conversation_id, provider_message_id,
                    role, text, sort_key, content_hash,
                    version, parent_message_id, branch_index)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", 1, None, 0),
            )
            conn.commit()

        assert get_sample_count_from_db("chatgpt", db_path=db_path) == 1

    def test_wrong_provider_returns_zero(self, tmp_path):
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
                ("c1", "chatgpt", "p1", "Test", None, None, "hash1", None, '{}', 1, None, None, None),
            )
            conn.execute(
                """INSERT INTO messages
                   (message_id, conversation_id, provider_message_id,
                    role, text, sort_key, content_hash,
                    version, parent_message_id, branch_index)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", 1, None, 0),
            )
            conn.commit()

        assert get_sample_count_from_db("claude", db_path=db_path) == 0


class TestGenerateSchemaFromSamples:
    """Focused schema-generation edge cases beyond the general laws."""

    @pytest.fixture(autouse=True)
    def _check_genson(self):
        pytest.importorskip("genson")

    def test_empty_samples(self):
        result = generate_schema_from_samples([])
        assert result["type"] == "object"
        assert "description" in result

    def test_single_sample(self):
        result = generate_schema_from_samples([{"id": "abc", "count": 42}])
        assert result["type"] == "object"
        assert "properties" in result
        assert "id" in result["properties"]
        assert "count" in result["properties"]

    def test_identifier_fields_do_not_emit_value_enums(self):
        result = generate_schema_from_samples(
            [
                {"resourceId": "1csAnmQr_ThZWh285_IH8hg50f-mLpS1r", "category": "HARM_CATEGORY_HATE_SPEECH"},
                {"resourceId": "12q0eVTU-RR-IMCjN0peXXKVg3LPwbmVW", "category": "HARM_CATEGORY_HARASSMENT"},
            ]
        )
        assert "x-polylogue-values" not in result["properties"]["resourceId"]
        assert "x-polylogue-values" in result["properties"]["category"]

    def test_high_entropy_tail_segments_are_filtered(self):
        result = generate_schema_from_samples(
            [
                {"promptId": "prompts/15BHmKFY05bDKHrhyk4We29IUInZjKq0p", "model": "models/gemini-2.5-pro"},
                {"promptId": "prompts/26CInJFZ16cELIshzl5Xf30JVJoaLr1q", "model": "models/gemini-2.5-pro"},
            ]
        )
        assert "x-polylogue-values" not in result["properties"]["promptId"]
        assert result["properties"]["model"].get("x-polylogue-values") == ["models/gemini-2.5-pro"]

    def test_high_entropy_values_filtered_even_without_identifier_field_name(self):
        result = generate_schema_from_samples(
            [
                {"channel": "1csAnmQr_ThZWh285_IH8hg50f-mLpS1r", "role": "assistant"},
                {"channel": "12q0eVTU-RR-IMCjN0peXXKVg3LPwbmVW", "role": "assistant"},
            ]
        )
        assert "x-polylogue-values" not in result["properties"]["channel"]
        assert result["properties"]["role"].get("x-polylogue-values") == ["assistant"]


class TestGenerateAllSchemas:
    """Filesystem effects of schema bundle generation."""

    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "schemas" / "nested"
        fake_result = GenerationResult(provider="test", sample_count=1, schema={"type": "object"}, error=None)

        with patch("polylogue.schemas.schema_generation.generate_provider_schema", return_value=fake_result):
            results = generate_all_schemas(output_dir, providers=["test"])

        assert output_dir.exists()
        assert len(results) == 1
        assert (output_dir / "test.schema.json.gz").exists()

    def test_skips_failed_schemas(self, tmp_path):
        failed_result = GenerationResult(provider="broken", sample_count=0, schema=None, error="No samples")

        with patch("polylogue.schemas.schema_generation.generate_provider_schema", return_value=failed_result):
            results = generate_all_schemas(tmp_path, providers=["broken"])

        assert not (tmp_path / "broken.schema.json.gz").exists()
        assert results[0].success is False


class TestCliMain:
    """CLI entry point behavior."""

    def test_cli_with_no_db(self, tmp_path):
        exit_code = cli_main(
            [
                "--provider",
                "chatgpt",
                "--output-dir",
                str(tmp_path / "out"),
                "--db-path",
                str(tmp_path / "missing.db"),
            ]
        )
        assert isinstance(exit_code, int)
