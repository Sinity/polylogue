"""Tests for schema clustering, versioning, comparison, promotion, and packaged provider schemas."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from polylogue.schemas.registry import (
    ClusterManifest,
    PropertyChange,
    SchemaCluster,
    SchemaDiff,
    SchemaRegistry,
    _fingerprint_hash,
)


@pytest.fixture
def tmp_registry(tmp_path: Path) -> SchemaRegistry:
    """Registry with an isolated storage root."""
    return SchemaRegistry(storage_root=tmp_path / "schemas")


@pytest.fixture
def sample_schema_a() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["id", "title"],
    }


@pytest.fixture
def sample_schema_b() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "count": {"type": "number"},  # type mutation
            "status": {"type": "string"},  # added
        },
        "required": ["id"],  # title no longer required
    }


# =============================================================================
# SchemaDiff output methods
# =============================================================================


class TestSchemaDiffOutputs:
    def test_to_dict_no_changes(self) -> None:
        diff = SchemaDiff(provider="test", version_a="v1", version_b="v2")
        d = diff.to_dict()
        assert d["has_changes"] is False
        assert d["summary"] == "no changes"

    def test_to_dict_with_changes(self) -> None:
        diff = SchemaDiff(
            provider="test",
            version_a="v1",
            version_b="v2",
            added_properties=["status"],
            removed_properties=["old_field"],
            changed_properties=["count"],
            classified_changes=[
                PropertyChange(path="status", kind="added", detail="new property"),
                PropertyChange(path="old_field", kind="removed", detail="removed property"),
                PropertyChange(path="count", kind="type_mutation", detail="integer -> number"),
            ],
        )
        d = diff.to_dict()
        assert d["has_changes"] is True
        assert len(d["classified_changes"]) == 3
        assert d["classified_changes"][0]["kind"] == "added"

    def test_to_text_renders(self) -> None:
        diff = SchemaDiff(
            provider="chatgpt",
            version_a="v1",
            version_b="v2",
            added_properties=["new_field"],
            classified_changes=[
                PropertyChange(path="new_field", kind="added", detail="new property"),
            ],
        )
        text = diff.to_text()
        assert "chatgpt" in text
        assert "v1 -> v2" in text
        assert "new_field" in text

    def test_to_text_no_changes(self) -> None:
        diff = SchemaDiff(provider="test", version_a="v1", version_b="v2")
        text = diff.to_text()
        assert "No changes detected" in text

    def test_to_markdown_renders_table(self) -> None:
        diff = SchemaDiff(
            provider="claude-ai",
            version_a="v1",
            version_b="v3",
            changed_properties=["count"],
            classified_changes=[
                PropertyChange(path="count", kind="type_mutation", detail="integer -> number"),
                PropertyChange(path="title", kind="requiredness", detail="optional (was required)"),
            ],
        )
        md = diff.to_markdown()
        assert "# Schema Diff: claude-ai" in md
        assert "| Path | Detail |" in md
        assert "`count`" in md

    def test_to_markdown_no_changes(self) -> None:
        diff = SchemaDiff(provider="test", version_a="v1", version_b="v2")
        md = diff.to_markdown()
        assert "No changes detected" in md

    def test_to_text_falls_back_to_legacy_lists(self) -> None:
        """When classified_changes is empty, falls back to added/removed/changed lists."""
        diff = SchemaDiff(
            provider="test",
            version_a="v1",
            version_b="v2",
            added_properties=["x"],
            removed_properties=["y"],
            changed_properties=["z"],
        )
        text = diff.to_text()
        assert "+ x" in text
        assert "- y" in text
        assert "~ z" in text


# =============================================================================
# PropertyChange
# =============================================================================


class TestPropertyChange:
    def test_to_dict(self) -> None:
        c = PropertyChange(path="$.id", kind="added", detail="new property")
        d = c.to_dict()
        assert d == {"path": "$.id", "kind": "added", "detail": "new property"}


# =============================================================================
# SchemaCluster and ClusterManifest
# =============================================================================


class TestSchemaCluster:
    def test_to_dict(self) -> None:
        cluster = SchemaCluster(
            cluster_id="abc123",
            provider="chatgpt",
            sample_count=100,
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-03-01T00:00:00Z",
            dominant_keys=["id", "title", "mapping"],
            confidence=0.95,
        )
        d = cluster.to_dict()
        assert d["cluster_id"] == "abc123"
        assert d["sample_count"] == 100
        assert d["confidence"] == 0.95
        assert d["promoted_package_version"] is None


class TestClusterManifest:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        cluster = SchemaCluster(
            cluster_id="xyz",
            provider="test",
            sample_count=50,
            first_seen="2026-01-01",
            last_seen="2026-03-01",
            dominant_keys=["a", "b"],
            confidence=0.8,
            artifact_kind="conversation_document",
            promoted_package_version="v2",
        )
        manifest = ClusterManifest(
            provider="test",
            clusters=[cluster],
            generated_at="2026-03-15T00:00:00Z",
            artifact_counts={"conversation_document": 50},
            default_version="v2",
        )
        d = manifest.to_dict()
        assert d["provider"] == "test"
        assert d["cluster_count"] == 1

        restored = ClusterManifest.from_dict(d)
        assert restored.provider == "test"
        assert len(restored.clusters) == 1
        assert restored.clusters[0].cluster_id == "xyz"
        assert restored.clusters[0].promoted_package_version == "v2"
        assert restored.default_version == "v2"

    def test_generated_at_auto_set(self) -> None:
        manifest = ClusterManifest(provider="auto")
        assert manifest.generated_at != ""


# =============================================================================
# SchemaRegistry clustering and promotion
# =============================================================================


class TestRegistryClustering:
    def test_cluster_samples_single_shape(self, tmp_registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "hello", "count": 5},
            {"id": "2", "title": "world", "count": 10},
            {"id": "3", "title": "foo", "count": 15},
        ]
        manifest = tmp_registry.cluster_samples("test-provider", samples)
        assert manifest.provider == "test-provider"
        assert len(manifest.clusters) == 1
        assert manifest.clusters[0].sample_count == 3

    def test_cluster_samples_multiple_shapes(self, tmp_registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "hello"},
            {"id": "2", "title": "world"},
            {"name": "different", "value": 42},
        ]
        manifest = tmp_registry.cluster_samples("test-provider", samples)
        assert len(manifest.clusters) == 2
        # Largest cluster first
        assert manifest.clusters[0].sample_count >= manifest.clusters[1].sample_count

    def test_cluster_samples_with_source_paths(self, tmp_registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1"},
            {"id": "2"},
        ]
        paths = ["/data/file1.json", "/data/file2.json"]
        manifest = tmp_registry.cluster_samples("test", samples, source_paths=paths)
        assert manifest.clusters[0].representative_paths == ["/data/file1.json", "/data/file2.json"]

    def test_save_and_load_cluster_manifest(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"id": "1", "title": "hi"}]
        manifest = tmp_registry.cluster_samples("test-provider", samples)

        saved_path = tmp_registry.save_cluster_manifest(manifest)
        assert saved_path.exists()
        assert saved_path.name == "manifest.json"

        loaded = tmp_registry.load_cluster_manifest("test-provider")
        assert loaded is not None
        assert loaded.provider == manifest.provider
        assert len(loaded.clusters) == len(manifest.clusters)
        assert loaded.clusters[0].cluster_id == manifest.clusters[0].cluster_id

    def test_load_cluster_manifest_missing(self, tmp_registry: SchemaRegistry) -> None:
        assert tmp_registry.load_cluster_manifest("nonexistent") is None


class TestRegistryPromotion:
    def test_promote_cluster_creates_new_version(self, tmp_registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "a", "count": 1},
            {"id": "2", "title": "b", "count": 2},
        ]
        manifest = tmp_registry.cluster_samples("test-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = tmp_registry.promote_cluster("test-prov", cluster_id, samples=samples)

        assert version == "v1"
        schema = tmp_registry.get_schema("test-prov", version=version)
        assert schema is not None
        assert schema["x-polylogue-anchor-profile-family-id"] == cluster_id
        assert schema["x-polylogue-observed-artifact-count"] == 2

    def test_promote_cluster_without_samples(self, tmp_registry: SchemaRegistry) -> None:
        """Promotion without samples creates a minimal stub schema."""
        samples = [{"id": "1", "key": "val"}]
        manifest = tmp_registry.cluster_samples("stub-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = tmp_registry.promote_cluster("stub-prov", cluster_id)

        assert version == "v1"
        schema = tmp_registry.get_schema("stub-prov", version=version)
        assert schema is not None
        assert "id" in schema["properties"]
        assert "key" in schema["properties"]

    def test_promote_cluster_updates_manifest(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"a": 1}]
        manifest = tmp_registry.cluster_samples("up-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = tmp_registry.promote_cluster("up-prov", cluster_id)

        reloaded = tmp_registry.load_cluster_manifest("up-prov")
        assert reloaded is not None
        assert reloaded.clusters[0].promoted_package_version == version

    def test_promote_cluster_already_promoted_raises(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"x": 1}]
        manifest = tmp_registry.cluster_samples("dup-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        tmp_registry.promote_cluster("dup-prov", cluster_id)

        with pytest.raises(ValueError, match="already promoted"):
            tmp_registry.promote_cluster("dup-prov", cluster_id)

    def test_promote_cluster_not_found_raises(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"x": 1}]
        manifest = tmp_registry.cluster_samples("nf-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        with pytest.raises(ValueError, match="not found"):
            tmp_registry.promote_cluster("nf-prov", "nonexistent-id")

    def test_promote_cluster_no_manifest_raises(self, tmp_registry: SchemaRegistry) -> None:
        with pytest.raises(ValueError, match="No cluster manifest"):
            tmp_registry.promote_cluster("missing-prov", "some-id")


# =============================================================================
# Enhanced compare_versions (D4: classified changes)
# =============================================================================


class TestEnhancedCompare:
    def test_classified_changes_additive(
        self,
        tmp_registry: SchemaRegistry,
        sample_schema_a: dict[str, Any],
        sample_schema_b: dict[str, Any],
    ) -> None:
        tmp_registry.register_schema("cmp-prov", sample_schema_a)
        tmp_registry.register_schema("cmp-prov", sample_schema_b)

        diff = tmp_registry.compare_versions("cmp-prov", "v1", "v2")
        assert diff.has_changes

        added = [c for c in diff.classified_changes if c.kind == "added"]
        assert any(c.path == "status" for c in added)

    def test_classified_changes_type_mutation(
        self,
        tmp_registry: SchemaRegistry,
        sample_schema_a: dict[str, Any],
        sample_schema_b: dict[str, Any],
    ) -> None:
        tmp_registry.register_schema("cmp-prov", sample_schema_a)
        tmp_registry.register_schema("cmp-prov", sample_schema_b)

        diff = tmp_registry.compare_versions("cmp-prov", "v1", "v2")

        type_changes = [c for c in diff.classified_changes if c.kind == "type_mutation"]
        assert any(c.path == "count" for c in type_changes)

    def test_classified_changes_requiredness(
        self,
        tmp_registry: SchemaRegistry,
        sample_schema_a: dict[str, Any],
        sample_schema_b: dict[str, Any],
    ) -> None:
        tmp_registry.register_schema("cmp-prov", sample_schema_a)
        tmp_registry.register_schema("cmp-prov", sample_schema_b)

        diff = tmp_registry.compare_versions("cmp-prov", "v1", "v2")

        req_changes = [c for c in diff.classified_changes if c.kind == "requiredness"]
        assert any(c.path == "title" for c in req_changes)

    def test_classified_changes_semantic_role(self, tmp_registry: SchemaRegistry) -> None:
        schema_a = {
            "type": "object",
            "properties": {
                "msg": {"type": "string", "x-polylogue-semantic-role": "message_body"},
            },
        }
        schema_b = {
            "type": "object",
            "properties": {
                "msg": {"type": "string", "x-polylogue-semantic-role": "title"},
            },
        }
        tmp_registry.register_schema("sem-prov", schema_a)
        tmp_registry.register_schema("sem-prov", schema_b)

        diff = tmp_registry.compare_versions("sem-prov", "v1", "v2")
        sem_changes = [c for c in diff.classified_changes if c.kind == "semantic_role"]
        assert len(sem_changes) == 1
        assert "message_body" in sem_changes[0].detail

    def test_classified_changes_relational(self, tmp_registry: SchemaRegistry) -> None:
        schema_a = {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "x-polylogue-ref": "$.other.id"},
            },
        }
        schema_b = {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "x-polylogue-ref": "$.new.id"},
            },
        }
        tmp_registry.register_schema("rel-prov", schema_a)
        tmp_registry.register_schema("rel-prov", schema_b)

        diff = tmp_registry.compare_versions("rel-prov", "v1", "v2")
        rel_changes = [c for c in diff.classified_changes if c.kind == "relational"]
        assert len(rel_changes) >= 1

    def test_schema_level_relational_annotations(self, tmp_registry: SchemaRegistry) -> None:
        schema_a = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        schema_b = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "x-polylogue-foreign-keys": [{"source": "$.a", "target": "$.b", "match_ratio": 0.9}],
        }
        tmp_registry.register_schema("rl-prov", schema_a)
        tmp_registry.register_schema("rl-prov", schema_b)

        diff = tmp_registry.compare_versions("rl-prov", "v1", "v2")
        rel_changes = [c for c in diff.classified_changes if c.kind == "relational"]
        assert any("x-polylogue-foreign-keys added" in c.detail for c in rel_changes)


# =============================================================================
# D5: Promotion visible to get_schema and list_versions
# =============================================================================


class TestPromotionVisibility:
    def test_schema_version_visible_via_get_schema(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"field_a": "val", "field_b": 42}]
        manifest = tmp_registry.cluster_samples("vis-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = tmp_registry.promote_cluster("vis-prov", cluster_id, samples=samples)

        # Should be retrievable by version
        schema = tmp_registry.get_schema("vis-prov", version=version)
        assert schema is not None

        # Should be the latest
        latest = tmp_registry.get_schema("vis-prov", version="latest")
        assert latest is not None
        assert latest.get("x-polylogue-anchor-profile-family-id") == cluster_id

    def test_schema_version_in_list_versions(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"k": "v"}]
        manifest = tmp_registry.cluster_samples("lv-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = tmp_registry.promote_cluster("lv-prov", cluster_id)

        versions = tmp_registry.list_versions("lv-prov")
        assert version in versions

    def test_promoted_schema_has_cluster_provenance(self, tmp_registry: SchemaRegistry) -> None:
        samples = [{"data": "test"}]
        manifest = tmp_registry.cluster_samples("prov-prov", samples)
        tmp_registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        tmp_registry.promote_cluster("prov-prov", cluster_id, samples=samples)


class TestPackageEvidenceRoundtrip:
    def test_write_schema_version_preserves_package_and_element_evidence(
        self,
        tmp_registry: SchemaRegistry,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "x-polylogue-generated-at": "2026-01-01T00:00:00+00:00",
            "x-polylogue-anchor-profile-family-id": "family-a",
            "x-polylogue-element-first-seen": "2026-01-01T00:00:00+00:00",
            "x-polylogue-element-last-seen": "2026-01-02T00:00:00+00:00",
            "x-polylogue-element-bundle-scope-count": 3,
            "x-polylogue-observed-artifact-count": 7,
            "x-polylogue-profile-family-ids": ["family-a", "family-b"],
            "x-polylogue-package-profile-family-ids": ["family-a", "family-b", "family-c"],
            "x-polylogue-exact-structure-ids": ["exact-a"],
        }

        tmp_registry.write_schema_version("pkg-prov", "v1", schema)
        package = tmp_registry.get_package("pkg-prov", version="v1")

        assert package is not None
        assert package.anchor_profile_family_id == "family-a"
        assert package.profile_family_ids == ["family-a", "family-b", "family-c"]
        assert package.first_seen == "2026-01-01T00:00:00+00:00"
        assert package.last_seen == "2026-01-01T00:00:00+00:00"

        element = package.element("conversation_document")
        assert element is not None
        assert element.first_seen == "2026-01-01T00:00:00+00:00"
        assert element.last_seen == "2026-01-02T00:00:00+00:00"
        assert element.bundle_scope_count == 3
        assert element.artifact_count == 7
        assert element.observed_artifact_count == 7
        assert element.profile_family_ids == ["family-a", "family-b"]


class TestManifestVersionSelection:
    def test_latest_uses_manifest_default_and_payload_matching_uses_profile_tokens(
        self,
        tmp_registry: SchemaRegistry,
    ) -> None:
        manifest = ClusterManifest(
            provider="chatgpt",
            artifact_counts={
                "conversation_document": 10,
                "subagent_conversation_stream": 5,
            },
            default_version="v1",
            clusters=[
                SchemaCluster(
                    cluster_id="cluster-main",
                    provider="chatgpt",
                    sample_count=10,
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                    artifact_kind="conversation_document",
                    profile_tokens=["field:mapping", "shape:mapping:object", "anchor:mapping"],
                    promoted_package_version="v1",
                ),
                SchemaCluster(
                    cluster_id="cluster-sidecar",
                    provider="chatgpt",
                    sample_count=5,
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                    artifact_kind="subagent_conversation_stream",
                    profile_tokens=["bucket:type:user", "field:type:user:message"],
                    promoted_package_version="v2",
                ),
            ],
        )
        tmp_registry.replace_provider_schemas(
            "chatgpt",
            [
                ("v1", {"type": "object", "properties": {"mapping": {"type": "object"}}}),
                ("v2", {"type": "object", "properties": {"type": {"type": "string"}}}),
            ],
            manifest=manifest,
        )

        latest = tmp_registry.get_schema("chatgpt", version="latest")
        default = tmp_registry.get_schema("chatgpt", version="default")
        assert latest is not None
        assert default is not None
        assert latest["x-polylogue-version"] == 2
        assert default["x-polylogue-version"] == 1
        assert tmp_registry.match_payload_version("chatgpt", {"mapping": {}}) == "v1"


# =============================================================================
# _fingerprint_hash
# =============================================================================


class TestFingerprintHash:
    def test_deterministic(self) -> None:
        fp = ("object", (("id", ("string",)), ("count", ("number",))))
        h1 = _fingerprint_hash(fp)
        h2 = _fingerprint_hash(fp)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_fingerprints_different_hashes(self) -> None:
        fp1 = ("object", (("id", ("string",)),))
        fp2 = ("object", (("name", ("string",)),))
        assert _fingerprint_hash(fp1) != _fingerprint_hash(fp2)


# =============================================================================
# Merged from test_provider_schema_meta.py (2024-03-15)
# =============================================================================


def _provider_schema_paths() -> list[Path]:
    schema_dir = Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"
    return sorted(schema_dir.glob("*/versions/*/elements/*.schema.json.gz"))


@pytest.mark.parametrize("schema_path", _provider_schema_paths(), ids=lambda p: p.name)
def test_packaged_provider_schema_is_valid_draft202012(schema_path: Path) -> None:
    schema = json.loads(gzip.decompress(schema_path.read_bytes()).decode("utf-8"))
    Draft202012Validator.check_schema(schema)
