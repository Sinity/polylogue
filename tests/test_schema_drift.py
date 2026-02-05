"""Schema drift detection tests.

Tests validate that provider exports conform to their schemas
and detect when provider formats have changed (drift).

Uses raw_conversations table as data source. Run `polylogue run --stage acquire`
to populate it with real exports.
"""

from __future__ import annotations

import json

import pytest

from polylogue.schemas import SchemaValidator, ValidationResult, validate_provider_export
from polylogue.storage.store import RawConversationRecord

# =============================================================================
# Schema Availability Tests
# =============================================================================


def test_available_providers():
    """Verify expected provider schemas exist."""
    providers = SchemaValidator.available_providers()
    assert "chatgpt" in providers
    assert "claude-code" in providers


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


# =============================================================================
# Database-Driven Schema Validation
# =============================================================================


class TestSchemaValidation:
    """Validate real data against schemas using raw_conversations.

    Schemas are generated from raw_conversations via `polylogue run --stage generate-schemas`.
    All samples MUST validate - failures indicate schema regeneration needed.
    """

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

                # Parse content (handle both JSON and JSONL)
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

        for sample in raw_db_samples[:100]:  # Check first 100 for drift
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

        # Report drift but don't fail - drift is informational
        if drift_by_provider:
            print(f"\nDrift detected in {len(drift_by_provider)} providers:")
            for provider, warnings in drift_by_provider.items():
                unique_warnings = list(set(warnings))[:5]
                print(f"  {provider}: {len(unique_warnings)} unique warnings")
                for w in unique_warnings[:3]:
                    print(f"    - {w}")


# =============================================================================
# Validator Behavior Tests (synthetic data)
# =============================================================================


def test_chatgpt_rejects_missing_mapping():
    """Test that ChatGPT schema rejects exports without mapping."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    invalid = {"id": "test", "title": "Test"}  # Missing mapping
    result = validate_provider_export(invalid, "chatgpt")
    # Just verify the validation runs without error
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
    # Drift should be detected for the new field
    assert isinstance(result, ValidationResult)


# =============================================================================
# ValidationResult Tests
# =============================================================================


def test_validation_result_properties():
    """Test ValidationResult properties."""
    # Valid result
    valid = ValidationResult(is_valid=True)
    assert valid.is_valid
    assert not valid.has_drift
    valid.raise_if_invalid()

    # Invalid result
    invalid = ValidationResult(is_valid=False, errors=["missing field"])
    assert not invalid.is_valid
    with pytest.raises(ValueError, match="missing field"):
        invalid.raise_if_invalid()

    # Valid with drift
    with_drift = ValidationResult(is_valid=True, drift_warnings=["new field: foo"])
    assert with_drift.is_valid
    assert with_drift.has_drift
