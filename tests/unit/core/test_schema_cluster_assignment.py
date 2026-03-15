"""Tests for schema cluster assignment and structural fingerprinting.

Covers:
- Structural fingerprinting groups structurally identical samples
- Different providers with same structure get same fingerprint
- Minor variations (optional/extra fields) create separate clusters
- Type mutations cause separate clusters
- ClusterManifest membership tracking
- Cluster ordering (largest cluster first)
- Confidence calculation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.schemas.registry import (
    ClusterManifest,
    SchemaCluster,
    SchemaRegistry,
    _fingerprint_hash,
)
from polylogue.schemas.schema_generation import _structure_fingerprint


@pytest.fixture
def registry(tmp_path: Path) -> SchemaRegistry:
    """Registry backed by an isolated temp directory."""
    return SchemaRegistry(storage_root=tmp_path / "schemas")


# =============================================================================
# Structural fingerprinting fundamentals
# =============================================================================


class TestStructuralFingerprinting:
    def test_same_structure_same_fingerprint(self) -> None:
        """Samples with identical key sets and types produce the same fingerprint."""
        a = {"id": "abc", "title": "hello", "count": 5}
        b = {"id": "xyz", "title": "world", "count": 99}
        assert _structure_fingerprint(a) == _structure_fingerprint(b)

    def test_different_values_same_fingerprint(self) -> None:
        """Fingerprint ignores concrete values, only looks at types."""
        a = {"name": "Alice", "age": 25}
        b = {"name": "Bob", "age": 99}
        assert _structure_fingerprint(a) == _structure_fingerprint(b)

    def test_different_keys_different_fingerprint(self) -> None:
        a = {"id": "1", "title": "x"}
        b = {"name": "1", "value": "x"}
        assert _structure_fingerprint(a) != _structure_fingerprint(b)

    def test_extra_field_different_fingerprint(self) -> None:
        """Adding a field changes the fingerprint (structural shape differs)."""
        base = {"id": "1", "title": "x"}
        extended = {"id": "1", "title": "x", "extra": "y"}
        assert _structure_fingerprint(base) != _structure_fingerprint(extended)

    def test_type_mutation_different_fingerprint(self) -> None:
        """Changing a field's type changes the fingerprint."""
        a = {"count": 5}  # int -> number
        b = {"count": "five"}  # str -> string
        assert _structure_fingerprint(a) != _structure_fingerprint(b)

    def test_nested_structure_fingerprint(self) -> None:
        """Nested objects contribute to the fingerprint."""
        a = {"meta": {"source": "web"}, "id": "1"}
        b = {"meta": {"source": "api"}, "id": "2"}
        assert _structure_fingerprint(a) == _structure_fingerprint(b)

    def test_nested_type_difference(self) -> None:
        a = {"meta": {"count": 1}}
        b = {"meta": {"count": "one"}}
        assert _structure_fingerprint(a) != _structure_fingerprint(b)

    def test_array_structure(self) -> None:
        """Array item shapes are captured."""
        a = {"tags": ["a", "b", "c"]}
        b = {"tags": ["x", "y"]}
        assert _structure_fingerprint(a) == _structure_fingerprint(b)

    def test_null_vs_string(self) -> None:
        a = {"val": None}
        b = {"val": "hello"}
        assert _structure_fingerprint(a) != _structure_fingerprint(b)

    def test_bool_vs_number(self) -> None:
        """Bools are fingerprinted distinctly from numbers."""
        a = {"flag": True}
        b = {"flag": 1}
        assert _structure_fingerprint(a) != _structure_fingerprint(b)


# =============================================================================
# Fingerprint hashing
# =============================================================================


class TestFingerprintHashing:
    def test_hash_deterministic(self) -> None:
        fp = ("object", (("id", ("string",)), ("count", ("number",))))
        assert _fingerprint_hash(fp) == _fingerprint_hash(fp)

    def test_hash_is_16_chars(self) -> None:
        fp = ("object", (("x", ("string",)),))
        assert len(_fingerprint_hash(fp)) == 16

    def test_different_structures_different_hashes(self) -> None:
        fp1 = ("object", (("id", ("string",)),))
        fp2 = ("object", (("name", ("string",)),))
        assert _fingerprint_hash(fp1) != _fingerprint_hash(fp2)

    def test_hash_from_sample_is_stable(self) -> None:
        sample = {"id": "1", "title": "hello"}
        fp = _structure_fingerprint(sample)
        h1 = _fingerprint_hash(fp)
        h2 = _fingerprint_hash(_structure_fingerprint({"id": "2", "title": "world"}))
        assert h1 == h2  # same structure -> same hash


# =============================================================================
# Cluster assignment via registry
# =============================================================================


class TestClusterAssignment:
    def test_identical_structure_single_cluster(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "a", "count": 1},
            {"id": "2", "title": "b", "count": 2},
            {"id": "3", "title": "c", "count": 3},
        ]
        manifest = registry.cluster_samples("test-prov", samples)
        assert len(manifest.clusters) == 1
        assert manifest.clusters[0].sample_count == 3

    def test_different_structures_multiple_clusters(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "a"},
            {"id": "2", "title": "b"},
            {"name": "x", "value": 42},
            {"name": "y", "value": 99},
        ]
        manifest = registry.cluster_samples("test-prov", samples)
        assert len(manifest.clusters) == 2

    def test_clusters_sorted_by_size_descending(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "a"},
            {"id": "2", "title": "b"},
            {"id": "3", "title": "c"},
            {"name": "x", "value": 42},
        ]
        manifest = registry.cluster_samples("test-prov", samples)
        assert len(manifest.clusters) == 2
        assert manifest.clusters[0].sample_count >= manifest.clusters[1].sample_count

    def test_same_structure_different_providers_same_fingerprint(self) -> None:
        """Two providers with structurally identical exports produce the same fingerprint."""
        sample_a = {"id": "1", "text": "hello"}
        sample_b = {"id": "2", "text": "world"}
        fp_a = _fingerprint_hash(_structure_fingerprint(sample_a))
        fp_b = _fingerprint_hash(_structure_fingerprint(sample_b))
        assert fp_a == fp_b

    def test_type_mutation_causes_separate_cluster(self, registry: SchemaRegistry) -> None:
        """Samples where a field changes type get assigned to different clusters."""
        samples = [
            {"id": "1", "count": 5},       # count is number
            {"id": "2", "count": 10},      # count is number
            {"id": "3", "count": "many"},  # count is string
        ]
        manifest = registry.cluster_samples("type-prov", samples)
        assert len(manifest.clusters) == 2

    def test_optional_field_presence_causes_separate_cluster(self, registry: SchemaRegistry) -> None:
        """A sample missing a field has a different structure than one with it."""
        samples = [
            {"id": "1", "title": "a", "status": "active"},
            {"id": "2", "title": "b", "status": "inactive"},
            {"id": "3", "title": "c"},  # missing "status"
        ]
        manifest = registry.cluster_samples("opt-prov", samples)
        assert len(manifest.clusters) == 2

    def test_single_sample_single_cluster(self, registry: SchemaRegistry) -> None:
        samples = [{"solo": True}]
        manifest = registry.cluster_samples("solo-prov", samples)
        assert len(manifest.clusters) == 1
        assert manifest.clusters[0].sample_count == 1

    def test_empty_samples_no_clusters(self, registry: SchemaRegistry) -> None:
        manifest = registry.cluster_samples("empty-prov", [])
        assert len(manifest.clusters) == 0


# =============================================================================
# Cluster metadata (dominant_keys, confidence, representative_paths)
# =============================================================================


class TestClusterMetadata:
    def test_dominant_keys_from_representative_sample(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "title": "a", "count": 1},
            {"id": "2", "title": "b", "count": 2},
        ]
        manifest = registry.cluster_samples("key-prov", samples)
        cluster = manifest.clusters[0]
        assert sorted(cluster.dominant_keys) == ["count", "id", "title"]

    def test_confidence_scales_with_cluster_proportion(self, registry: SchemaRegistry) -> None:
        """Larger clusters relative to total samples get higher confidence."""
        samples = [{"id": str(i)} for i in range(20)] + [{"name": "outlier"}]
        manifest = registry.cluster_samples("conf-prov", samples)
        big_cluster = manifest.clusters[0]
        small_cluster = manifest.clusters[1]
        assert big_cluster.confidence > small_cluster.confidence

    def test_representative_paths_collected(self, registry: SchemaRegistry) -> None:
        samples = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        paths = ["/data/a.json", "/data/b.json", "/data/c.json"]
        manifest = registry.cluster_samples("path-prov", samples, source_paths=paths)
        cluster = manifest.clusters[0]
        assert len(cluster.representative_paths) == 3
        assert "/data/a.json" in cluster.representative_paths

    def test_representative_paths_capped_at_5(self, registry: SchemaRegistry) -> None:
        samples = [{"id": str(i)} for i in range(10)]
        paths = [f"/data/{i}.json" for i in range(10)]
        manifest = registry.cluster_samples("cap-prov", samples, source_paths=paths)
        cluster = manifest.clusters[0]
        assert len(cluster.representative_paths) <= 5

    def test_representative_paths_deduplicated(self, registry: SchemaRegistry) -> None:
        samples = [{"id": "1"}, {"id": "2"}]
        paths = ["/data/same.json", "/data/same.json"]
        manifest = registry.cluster_samples("dup-prov", samples, source_paths=paths)
        cluster = manifest.clusters[0]
        assert cluster.representative_paths == ["/data/same.json"]

    def test_dominant_keys_capped_at_20(self, registry: SchemaRegistry) -> None:
        many_keys = {f"key_{i:03d}": "val" for i in range(30)}
        manifest = registry.cluster_samples("manykey-prov", [many_keys])
        cluster = manifest.clusters[0]
        assert len(cluster.dominant_keys) <= 20


# =============================================================================
# ClusterManifest membership tracking
# =============================================================================


class TestManifestMembership:
    def test_manifest_provider_set(self, registry: SchemaRegistry) -> None:
        samples = [{"x": 1}]
        manifest = registry.cluster_samples("mem-prov", samples)
        assert manifest.provider == "mem-prov"

    def test_manifest_generated_at_populated(self, registry: SchemaRegistry) -> None:
        samples = [{"x": 1}]
        manifest = registry.cluster_samples("mem-prov", samples)
        assert manifest.generated_at != ""

    def test_manifest_to_dict_cluster_count(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "a": 1},
            {"name": "2", "b": 2},
        ]
        manifest = registry.cluster_samples("cnt-prov", samples)
        d = manifest.to_dict()
        assert d["cluster_count"] == 2
        assert len(d["clusters"]) == 2

    def test_manifest_roundtrip(self, registry: SchemaRegistry) -> None:
        samples = [
            {"id": "1", "val": "a"},
            {"id": "2", "val": "b"},
            {"name": "3"},
        ]
        manifest = registry.cluster_samples("rt-prov", samples)
        d = manifest.to_dict()
        restored = ClusterManifest.from_dict(d)
        assert restored.provider == manifest.provider
        assert len(restored.clusters) == len(manifest.clusters)
        for orig, rest in zip(manifest.clusters, restored.clusters):
            assert orig.cluster_id == rest.cluster_id
            assert orig.sample_count == rest.sample_count
            assert orig.confidence == rest.confidence

    def test_manifest_persistence(self, registry: SchemaRegistry) -> None:
        samples = [{"x": 1}, {"x": 2}]
        manifest = registry.cluster_samples("per-prov", samples)
        path = registry.save_cluster_manifest(manifest)
        assert path.exists()

        loaded = registry.load_cluster_manifest("per-prov")
        assert loaded is not None
        assert loaded.clusters[0].cluster_id == manifest.clusters[0].cluster_id

    def test_manifest_records_promotion(self, registry: SchemaRegistry) -> None:
        samples = [{"field": "val"}]
        manifest = registry.cluster_samples("promo-prov", samples)
        registry.save_cluster_manifest(manifest)

        cluster_id = manifest.clusters[0].cluster_id
        version = registry.promote_cluster("promo-prov", cluster_id)

        reloaded = registry.load_cluster_manifest("promo-prov")
        assert reloaded is not None
        assert reloaded.clusters[0].promoted_version == version


# =============================================================================
# Parametrized: various sample shapes
# =============================================================================


@pytest.mark.parametrize(
    "samples, expected_cluster_count",
    [
        pytest.param(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            1,
            id="uniform-ints",
        ),
        pytest.param(
            [{"a": "x"}, {"a": "y"}, {"b": 1}],
            2,
            id="mixed-keys",
        ),
        pytest.param(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5}],
            2,
            id="optional-field",
        ),
        pytest.param(
            [{"a": [1, 2]}, {"a": [3]}],
            1,
            id="arrays-same-type",
        ),
        pytest.param(
            [{"a": [1]}, {"a": ["x"]}],
            2,
            id="arrays-different-types",
        ),
        pytest.param(
            [{"nested": {"x": 1}}, {"nested": {"x": 2}}],
            1,
            id="nested-same-structure",
        ),
        pytest.param(
            [{"nested": {"x": 1}}, {"nested": {"y": 2}}],
            2,
            id="nested-different-keys",
        ),
    ],
)
def test_cluster_count_parametrized(
    tmp_path: Path,
    samples: list[dict],
    expected_cluster_count: int,
) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    manifest = registry.cluster_samples("param-prov", samples)
    assert len(manifest.clusters) == expected_cluster_count
