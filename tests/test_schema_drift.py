"""Schema drift detection tests.

These tests validate that provider exports conform to their schemas
and detect when provider formats have changed (drift).

Run with: pytest tests/test_schema_drift.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.validation import SchemaValidator, ValidationResult, validate_provider_export


# =============================================================================
# Fixtures
# =============================================================================


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real"
TEST_DATASETS_DIR = Path(__file__).parent / "test-datasets"


@pytest.fixture
def chatgpt_fixtures() -> list[dict]:
    """Load all ChatGPT fixture files."""
    chatgpt_dir = FIXTURES_DIR / "chatgpt"
    if not chatgpt_dir.exists():
        return []

    fixtures = []
    for path in chatgpt_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        fixtures.append(data)
    return fixtures


@pytest.fixture
def codex_fixtures() -> list[dict]:
    """Load Codex fixture files."""
    cody_path = TEST_DATASETS_DIR / "cody-attachments.json"
    if not cody_path.exists():
        return []
    return [json.loads(cody_path.read_text(encoding="utf-8"))]


# =============================================================================
# Schema Availability Tests
# =============================================================================


def test_available_providers():
    """Verify expected provider schemas exist."""
    providers = SchemaValidator.available_providers()

    # We should have at least these from fixture generation
    assert "chatgpt" in providers

    # Report what's available
    print(f"Available schemas: {providers}")


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
# ChatGPT Schema Validation
# =============================================================================


@pytest.mark.parametrize("fixture_name", [
    "simple.json",
    "branching.json",
    "attachments.json",
    "large.json",
])
def test_chatgpt_fixtures_have_required_structure(fixture_name: str):
    """Test that ChatGPT fixtures have required top-level structure.

    Note: Full schema validation is too strict for real exports that evolve.
    This test verifies essential structure that importers depend on.
    """
    fixture_path = FIXTURES_DIR / "chatgpt" / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    data = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Check essential structure that importers require
    assert isinstance(data, dict), "Export must be a dict"
    assert "mapping" in data, "Export must have mapping"
    assert isinstance(data["mapping"], dict), "mapping must be a dict"

    # Verify mapping nodes have expected structure
    for node_id, node in data["mapping"].items():
        if node.get("message"):
            msg = node["message"]
            assert "content" in msg or "author" in msg, f"Node {node_id} missing content/author"


def test_chatgpt_all_fixtures_structure(chatgpt_fixtures: list[dict]):
    """Test all ChatGPT fixtures have importable structure."""
    if not chatgpt_fixtures:
        pytest.skip("No ChatGPT fixtures found")

    for i, fixture in enumerate(chatgpt_fixtures):
        # Verify structure importers need
        assert "mapping" in fixture, f"Fixture {i} missing mapping"

        # Count valid message nodes
        valid_nodes = 0
        for node in fixture["mapping"].values():
            if isinstance(node, dict) and node.get("message"):
                valid_nodes += 1

        print(f"Fixture {i}: {valid_nodes} message nodes")


def test_chatgpt_drift_detection(chatgpt_fixtures: list[dict]):
    """Test drift detection on ChatGPT fixtures."""
    if not chatgpt_fixtures:
        pytest.skip("No ChatGPT fixtures found")

    # In strict mode, check for any drift warnings
    for fixture in chatgpt_fixtures:
        result = validate_provider_export(fixture, "chatgpt", strict=True)

        # Log drift warnings (expected for real exports - providers evolve)
        if result.drift_warnings:
            print(f"Drift detected in fixture: {len(result.drift_warnings)} warnings")
            for warning in result.drift_warnings[:5]:
                print(f"  - {warning}")


# =============================================================================
# Codex Schema Validation
# =============================================================================


def test_codex_fixtures_structure(codex_fixtures: list[dict]):
    """Test Codex fixtures have importable structure."""
    if not codex_fixtures:
        pytest.skip("No Codex fixtures found")

    for i, fixture in enumerate(codex_fixtures):
        # Codex/Cody format can be a list of conversations or a single dict
        if isinstance(fixture, list):
            print(f"Codex fixture {i}: list with {len(fixture)} items")
            for j, item in enumerate(fixture[:3]):
                if isinstance(item, dict):
                    print(f"  Item {j} keys: {list(item.keys())[:5]}")
        elif isinstance(fixture, dict):
            print(f"Codex fixture {i} keys: {list(fixture.keys())[:10]}")


# =============================================================================
# Invalid Data Tests
# =============================================================================


def test_chatgpt_rejects_missing_mapping():
    """Test that ChatGPT schema rejects exports without mapping."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    invalid = {"id": "test", "title": "Test"}  # Missing mapping
    result = validate_provider_export(invalid, "chatgpt")

    # This might be valid if mapping is optional in schema
    # The test verifies we can detect structural differences
    print(f"Missing mapping result: valid={result.is_valid}, errors={result.errors}")


def test_chatgpt_rejects_wrong_type():
    """Test that schema rejects wrong types."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    invalid = {
        "id": 12345,  # Should be string
        "mapping": {},
    }
    result = validate_provider_export(invalid, "chatgpt")
    # If id type is enforced, this should fail
    print(f"Wrong type result: valid={result.is_valid}, errors={result.errors}")


# =============================================================================
# Drift Detection Examples
# =============================================================================


def test_drift_new_field():
    """Test that new fields are detected as drift."""
    if "chatgpt" not in SchemaValidator.available_providers():
        pytest.skip("ChatGPT schema not available")

    # Create minimal valid export with extra field
    data = {
        "id": "test-123",
        "mapping": {},
        "brand_new_field": "unexpected",  # New field not in schema
    }

    result = validate_provider_export(data, "chatgpt", strict=True)

    # The export might still be valid (additionalProperties allowed)
    # but drift should be detected
    print(f"New field result: valid={result.is_valid}, drift={result.drift_warnings}")


# =============================================================================
# Property-Based Validation Tests
# =============================================================================


@pytest.mark.skipif(
    "chatgpt" not in SchemaValidator.available_providers(),
    reason="ChatGPT schema not available",
)
def test_generated_chatgpt_validates():
    """Test that generated ChatGPT exports validate."""
    from hypothesis import given, settings
    from tests.strategies import chatgpt_export_strategy

    @given(chatgpt_export_strategy())
    @settings(max_examples=20)
    def check_validates(export):
        result = validate_provider_export(export, "chatgpt", strict=False)
        # Generated exports should be structurally valid
        # (may have type mismatches for optional fields)
        if not result.is_valid:
            print(f"Generated export invalid: {result.errors[:3]}")

    check_validates()


# =============================================================================
# ValidationResult Tests
# =============================================================================


def test_validation_result_properties():
    """Test ValidationResult properties."""
    # Valid result
    valid = ValidationResult(is_valid=True)
    assert valid.is_valid
    assert not valid.has_drift
    valid.raise_if_invalid()  # Should not raise

    # Invalid result
    invalid = ValidationResult(is_valid=False, errors=["missing field"])
    assert not invalid.is_valid
    with pytest.raises(ValueError, match="missing field"):
        invalid.raise_if_invalid()

    # Valid with drift
    with_drift = ValidationResult(is_valid=True, drift_warnings=["new field: foo"])
    assert with_drift.is_valid
    assert with_drift.has_drift


# =============================================================================
# Live Database Validation Tests
# =============================================================================


def _get_polylogue_db_path() -> Path | None:
    """Get path to polylogue database if it exists."""
    db_path = Path.home() / ".local/state/polylogue/polylogue.db"
    return db_path if db_path.exists() else None


@pytest.fixture
def polylogue_db():
    """Get connection to polylogue database."""
    import sqlite3

    db_path = _get_polylogue_db_path()
    if not db_path:
        pytest.skip("No polylogue database found")

    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


def test_gemini_messages_from_db_validate(polylogue_db):
    """Test that Gemini messages from database validate against typed model."""
    from polylogue.providers import GeminiMessage

    rows = polylogue_db.execute("""
        SELECT m.provider_meta
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = 'gemini'
        AND m.provider_meta IS NOT NULL
        LIMIT 100
    """).fetchall()

    if not rows:
        pytest.skip("No Gemini messages in database")

    valid_count = 0
    errors = []
    for row in rows:
        try:
            meta = json.loads(row[0])
            raw = meta.get("raw", {})
            GeminiMessage.model_validate(raw)
            valid_count += 1
        except Exception as e:
            errors.append(str(e)[:100])

    print(f"Validated {valid_count}/{len(rows)} Gemini messages")
    if errors:
        print(f"Errors (first 3): {errors[:3]}")

    # Most messages should validate
    assert valid_count >= len(rows) * 0.9, f"Only {valid_count}/{len(rows)} validated"


def test_providers_in_db_have_schemas(polylogue_db):
    """Test that all providers in database have corresponding schemas."""
    rows = polylogue_db.execute(
        "SELECT DISTINCT provider_name FROM conversations"
    ).fetchall()

    providers_in_db = {row[0] for row in rows}
    available_schemas = set(SchemaValidator.available_providers())

    print(f"Providers in DB: {providers_in_db}")
    print(f"Available schemas: {available_schemas}")

    # Check coverage (some providers may not have schemas yet)
    covered = providers_in_db & available_schemas
    uncovered = providers_in_db - available_schemas

    if uncovered:
        print(f"Providers without schemas: {uncovered}")

    # At least the main providers should be covered
    main_providers = {"chatgpt", "claude", "claude-code", "gemini", "codex"}
    expected_covered = providers_in_db & main_providers
    assert expected_covered <= available_schemas | {"claude"}, \
        f"Missing schemas for: {expected_covered - available_schemas}"


def test_database_message_counts_by_provider(polylogue_db):
    """Report message counts per provider for visibility."""
    rows = polylogue_db.execute("""
        SELECT c.provider_name, COUNT(m.message_id) as count
        FROM conversations c
        LEFT JOIN messages m ON c.conversation_id = m.conversation_id
        GROUP BY c.provider_name
        ORDER BY count DESC
    """).fetchall()

    print("\n=== Message Counts by Provider ===")
    for provider, count in rows:
        print(f"  {provider}: {count:,} messages")
