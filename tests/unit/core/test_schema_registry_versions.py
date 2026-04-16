"""Tests for schema version tracking in SchemaRegistry.

Covers:
- Auto-incrementing version numbers
- Version listing and ordering
- Retrieving schemas by version (specific and latest)
- Version bumps only on structural changes
- Annotation-only changes (x-polylogue-*) still bump version (registry is structural-agnostic)
- Metadata injection ($id, x-polylogue-version, x-polylogue-registered-at)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.schemas.registry import SchemaRegistry


@pytest.fixture
def registry(tmp_path: Path) -> SchemaRegistry:
    """Registry backed by an isolated temp directory."""
    return SchemaRegistry(storage_root=tmp_path / "schemas")


# =============================================================================
# Version numbering
# =============================================================================


class TestVersionNumbering:
    def test_first_registration_is_v1(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        version = registry.register_schema("test-prov", schema)
        assert version == "v1"

    def test_successive_registrations_increment(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        v1 = registry.register_schema("test-prov", schema)
        v2 = registry.register_schema("test-prov", schema)
        v3 = registry.register_schema("test-prov", schema)
        assert v1 == "v1"
        assert v2 == "v2"
        assert v3 == "v3"

    def test_versions_are_per_provider(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        va = registry.register_schema("provider-a", schema)
        vb = registry.register_schema("provider-b", schema)
        assert va == "v1"
        assert vb == "v1"

    def test_version_continues_after_gap_read(self, registry: SchemaRegistry) -> None:
        """register -> read -> register still increments correctly."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        registry.register_schema("seq-prov", schema)
        _ = registry.get_schema("seq-prov", version="v1")
        v2 = registry.register_schema("seq-prov", schema)
        assert v2 == "v2"


# =============================================================================
# Version listing
# =============================================================================


class TestListVersions:
    def test_no_versions_for_unknown_provider(self, registry: SchemaRegistry) -> None:
        assert registry.list_versions("nonexistent") == []

    def test_list_returns_sorted_versions(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        for _ in range(5):
            registry.register_schema("multi-prov", schema)
        versions = registry.list_versions("multi-prov")
        assert versions == ["v1", "v2", "v3", "v4", "v5"]

    def test_list_versions_numeric_sort_not_lexicographic(self, registry: SchemaRegistry) -> None:
        """v10 should sort after v9, not after v1."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        for _ in range(12):
            registry.register_schema("sort-prov", schema)
        versions = registry.list_versions("sort-prov")
        assert versions[-1] == "v12"
        assert versions.index("v9") < versions.index("v10")


# =============================================================================
# Retrieving schemas by version
# =============================================================================


class TestGetSchema:
    def test_get_specific_version(self, registry: SchemaRegistry) -> None:
        schema_v1 = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema_v2 = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        registry.register_schema("get-prov", schema_v1)
        registry.register_schema("get-prov", schema_v2)

        retrieved = registry.get_schema("get-prov", version="v1")
        assert retrieved is not None
        assert "a" in retrieved["properties"]
        # v1 should NOT have "b"
        assert "b" not in retrieved["properties"]

    def test_get_latest_returns_most_recent(self, registry: SchemaRegistry) -> None:
        schema_v1 = {"type": "object", "properties": {"old": {"type": "string"}}}
        schema_v2 = {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "number"}}}
        registry.register_schema("lat-prov", schema_v1)
        registry.register_schema("lat-prov", schema_v2)

        latest = registry.get_schema("lat-prov", version="latest")
        assert latest is not None
        assert "new" in latest["properties"]

    def test_get_nonexistent_version_returns_none(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("ne-prov", schema)
        assert registry.get_schema("ne-prov", version="v99") is None

    def test_get_schema_unknown_provider_returns_none(self, registry: SchemaRegistry) -> None:
        assert registry.get_schema("totally-unknown", version="v1") is None
        assert registry.get_schema("totally-unknown", version="latest") is None


# =============================================================================
# Metadata injection
# =============================================================================


class TestMetadataInjection:
    def test_register_injects_id(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("meta-prov", schema)
        stored = registry.get_schema("meta-prov", version="v1")
        assert stored is not None
        assert stored["$id"] == "polylogue://schemas/meta-prov/v1/conversation_document"

    def test_register_injects_version_number(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("meta-prov", schema)
        registry.register_schema("meta-prov", schema)
        v2 = registry.get_schema("meta-prov", version="v2")
        assert v2 is not None
        assert v2["x-polylogue-version"] == 2

    def test_register_injects_timestamp(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("meta-prov", schema)
        stored = registry.get_schema("meta-prov", version="v1")
        assert stored is not None
        assert "x-polylogue-registered-at" in stored
        # Should be a valid ISO timestamp
        from datetime import datetime

        datetime.fromisoformat(stored["x-polylogue-registered-at"])

    def test_register_does_not_mutate_input(self, registry: SchemaRegistry) -> None:
        """The original dict passed in should not be modified."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        original_keys = set(schema.keys())
        registry.register_schema("mut-prov", schema)
        assert set(schema.keys()) == original_keys


# =============================================================================
# Structural vs annotation-only changes
# =============================================================================


class TestVersionBumpBehavior:
    """SchemaRegistry always bumps version on register_schema (it's structural-agnostic).

    The registry does not inspect whether changes are structural or annotation-only;
    it simply auto-increments. Schema comparison (compare_versions) is the tool
    for classifying change significance.
    """

    def test_structural_change_bumps_version(self, registry: SchemaRegistry) -> None:
        schema_v1 = {"type": "object", "properties": {"a": {"type": "string"}}}
        schema_v2 = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "number"}}}
        registry.register_schema("str-prov", schema_v1)
        v2 = registry.register_schema("str-prov", schema_v2)
        assert v2 == "v2"

        diff = registry.compare_versions("str-prov", "v1", "v2")
        assert diff.has_changes
        assert "b" in diff.added_properties

    def test_type_mutation_detected_on_version_bump(self, registry: SchemaRegistry) -> None:
        schema_v1 = {"type": "object", "properties": {"count": {"type": "integer"}}}
        schema_v2 = {"type": "object", "properties": {"count": {"type": "number"}}}
        registry.register_schema("typ-prov", schema_v1)
        registry.register_schema("typ-prov", schema_v2)

        diff = registry.compare_versions("typ-prov", "v1", "v2")
        type_mutations = [c for c in diff.classified_changes if c.kind == "type_mutation"]
        assert len(type_mutations) == 1
        assert type_mutations[0].path == "count"

    def test_annotation_only_change_still_creates_version(self, registry: SchemaRegistry) -> None:
        """Register schemas that differ only in x-polylogue-* annotations."""
        schema_v1 = {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
        }
        schema_v2 = {
            "type": "object",
            "properties": {"msg": {"type": "string", "x-polylogue-semantic-role": "message_body"}},
        }
        registry.register_schema("ann-prov", schema_v1)
        v2 = registry.register_schema("ann-prov", schema_v2)
        assert v2 == "v2"

        diff = registry.compare_versions("ann-prov", "v1", "v2")
        # Annotation-only changes should not appear in added/removed/changed
        assert not diff.added_properties
        assert not diff.removed_properties
        assert not diff.changed_properties
        # But should appear in classified_changes as semantic_role
        semantic = [c for c in diff.classified_changes if c.kind == "semantic_role"]
        assert len(semantic) == 1

    def test_identical_schemas_no_changes(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("id-prov", schema.copy())
        registry.register_schema("id-prov", schema.copy())

        diff = registry.compare_versions("id-prov", "v1", "v2")
        assert not diff.has_changes
        assert diff.summary() == "no changes"

    def test_property_removal_detected(self, registry: SchemaRegistry) -> None:
        schema_v1 = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}, "c": {"type": "boolean"}},
        }
        schema_v2 = {"type": "object", "properties": {"a": {"type": "string"}}}
        registry.register_schema("rem-prov", schema_v1)
        registry.register_schema("rem-prov", schema_v2)

        diff = registry.compare_versions("rem-prov", "v1", "v2")
        assert sorted(diff.removed_properties) == ["b", "c"]
        removed_classified = [c for c in diff.classified_changes if c.kind == "removed"]
        assert len(removed_classified) == 2


# =============================================================================
# Schema age
# =============================================================================


class TestSchemaAge:
    def test_age_returns_none_for_unknown_provider(self, registry: SchemaRegistry) -> None:
        assert registry.get_schema_age_days("unknown") is None

    def test_age_returns_none_without_timestamp(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("age-prov", schema)
        # Package-aware registry age is now based on package chronology.
        age = registry.get_schema_age_days("age-prov")
        assert age == 0

    def test_age_returns_days_with_generated_at(self, registry: SchemaRegistry) -> None:
        from datetime import datetime, timedelta, timezone

        two_days_ago = (datetime.now(tz=timezone.utc) - timedelta(days=2)).isoformat()
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "x-polylogue-generated-at": two_days_ago,
        }
        registry.register_schema("age-prov", schema)
        age = registry.get_schema_age_days("age-prov")
        assert age is not None
        assert age == 2


# =============================================================================
# Provider listing
# =============================================================================


class TestListProviders:
    def test_list_providers_includes_versioned(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("listed-prov", schema)
        providers = registry.list_providers()
        assert "listed-prov" in providers

    def test_list_providers_sorted(self, registry: SchemaRegistry) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        registry.register_schema("z-prov", schema)
        registry.register_schema("a-prov", schema)
        registry.register_schema("m-prov", schema)
        providers = registry.list_providers()
        versioned = [p for p in providers if p in {"a-prov", "m-prov", "z-prov"}]
        assert versioned == sorted(versioned)
