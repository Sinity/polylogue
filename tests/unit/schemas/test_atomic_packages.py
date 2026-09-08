"""Publication faults must expose one coherent provider generation."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.models import GenerationResult, _ProviderBundle
from polylogue.schemas.generation.provider_bundle_packages import allocate_package_versions
from polylogue.schemas.generation.workflow import persist_generated_provider_bundle
from polylogue.schemas.generation.workload_profiles import workload_profile_identity
from polylogue.schemas.numeric_privacy import redact_observed_numeric_schema
from polylogue.schemas.package_publication import publish_provider_tree
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import ClusterManifest, SchemaRegistry


def package(version: str = "v1", family: str = "family-a", kind: str = "session_document") -> SchemaVersionPackage:
    return SchemaVersionPackage(
        provider="synthetic-publication",
        version=version,
        anchor_kind=kind,
        default_element_kind=kind,
        first_seen="2026-01-01T00:00:00+00:00",
        last_seen="2026-01-01T00:00:00+00:00",
        sample_count=1,
        bundle_scope_count=1,
        anchor_profile_family_id=family,
        elements=[SchemaElementManifest(kind, f"{kind}.schema.json.gz", 1, 1)],
        workload_profile_file="workload-profile.json.gz",
    )


def publish(registry: SchemaRegistry, *, version: str = "v1", generation: int = 1, family: str = "family-a") -> None:
    item = package(version, family)
    registry.replace_provider_packages(
        item.provider,
        SchemaPackageCatalog(
            provider=item.provider,
            packages=[item],
            generated_at="2026-01-01T00:00:00+00:00",
            latest_version=version,
            default_version=version,
            recommended_version=version,
        ),
        {
            version: {
                item.anchor_kind: {
                    "type": "object",
                    "properties": {f"field_{generation}": {"type": "string"}},
                    "x-polylogue-test-generation": generation,
                }
            }
        },
        package_workload_profiles={version: {"generation": generation}},
        cluster_manifest={"generation": generation},
    )


def schema(registry: SchemaRegistry, version: str = "default") -> JSONDocument:
    result = registry.get_schema("synthetic-publication", version)
    assert result is not None
    return result


@pytest.mark.parametrize("failed_write", ["write_package", "save_package_catalog"])
def test_staging_failure_preserves_catalog_schemas_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_write: str
) -> None:
    """Writing into the active tree would replace part of v1 before failing."""
    writer = SchemaRegistry(storage_root=tmp_path)
    publish(writer)
    original = getattr(SchemaRegistry, failed_write)

    def fail_after_write(self: SchemaRegistry, *args: object, **kwargs: object) -> object:
        original(self, *args, **kwargs)
        raise OSError("injected staged write failure")

    monkeypatch.setattr(SchemaRegistry, failed_write, fail_after_write)
    with pytest.raises(OSError, match="injected"):
        publish(writer, generation=2)
    fresh = SchemaRegistry(storage_root=tmp_path)
    assert schema(fresh)["x-polylogue-test-generation"] == 1
    assert fresh.get_workload_profile("synthetic-publication") == {"generation": 1}
    assert fresh.load_committed_package("synthetic-publication", "v1") is not None


def test_cached_catalog_binds_lazy_files_to_same_generation(tmp_path: Path) -> None:
    """A reader reopening reused v1 filenames after publication sees mixed evidence."""
    writer = SchemaRegistry(storage_root=tmp_path)
    publish(writer)
    reader = SchemaRegistry(storage_root=tmp_path)
    assert reader.load_package_catalog("synthetic-publication") is not None
    publish(writer, generation=2)
    assert schema(reader)["x-polylogue-test-generation"] == 1
    assert reader.get_workload_profile("synthetic-publication") == {"generation": 1}
    old_schema = reader.load_committed_schema_file("synthetic-publication", "v1", "session_document.schema.json.gz")
    assert old_schema is not None and old_schema["x-polylogue-test-generation"] == 1
    reader.clear_cache()
    assert schema(reader)["x-polylogue-test-generation"] == 2
    assert reader.get_workload_profile("synthetic-publication") == {"generation": 2}


def test_omitted_family_survives_without_blending_into_new_family(tmp_path: Path) -> None:
    """Deleting omitted versions loses history; merging all versions erases families."""
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry)
    publish(registry, version="v2", generation=2, family="family-b")
    assert registry.list_versions("synthetic-publication") == ["v1", "v2"]
    first_properties = schema(registry, "v1")["properties"]
    second_properties = schema(registry, "v2")["properties"]
    assert isinstance(first_properties, dict) and "field_1" in first_properties
    assert isinstance(second_properties, dict) and "field_1" not in second_properties
    assert registry.get_workload_profile("synthetic-publication", "v1") == {"generation": 1}


def test_missing_historical_schema_refuses_publication_atomically(tmp_path: Path) -> None:
    """A retained package without a schema cannot publish a catalog that names it."""
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry)
    schema_path = (
        tmp_path / "synthetic-publication" / "versions" / "v1" / "elements" / "session_document.schema.json.gz"
    )
    schema_path.unlink()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Existing package synthetic-publication/v1 is incomplete"):
        publish(registry, version="v2", generation=2, family="family-b")

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_new_earlier_family_does_not_renumber_existing_versions() -> None:
    """Ordinal labels shift v1 when an earlier structural family arrives."""
    prior = SchemaPackageCatalog(
        provider="synthetic-publication", packages=[package("v1", "z-family")], generated_at="fixed"
    )
    assert allocate_package_versions(prior, [("session_document", "a-family"), ("session_document", "z-family")]) == [
        "v2",
        "v1",
    ]
    assert allocate_package_versions(prior, [("session_document", "z-family"), ("session_document", "a-family")]) == [
        "v1",
        "v2",
    ]


def test_retained_provider_files_survive_replacement(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry)
    retained = tmp_path / "synthetic-publication" / "pins.yaml"
    retained.write_text("pins: []\n")
    publish(registry, generation=2)
    assert retained.read_text() == "pins: []\n"
    assert not list(tmp_path.glob(".*.staging-*"))


def test_concurrent_retained_pin_change_retries_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimistic snapshot must cover retained provider files."""
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry)
    retained = tmp_path / "synthetic-publication" / "pins.yaml"
    retained.write_text("pins: initial\n")

    def publish_after_pin_change(staged: Path, destination: Path, *, expected_snapshot: dict[str, bytes]) -> None:
        retained.write_text("pins: concurrent\n")
        publish_provider_tree(staged, destination, expected_snapshot=expected_snapshot)

    monkeypatch.setattr("polylogue.schemas.runtime_registry.publish_provider_tree", publish_after_pin_change)

    with pytest.raises(RuntimeError, match="changed during preparation"):
        publish(registry, generation=2)

    assert retained.read_text() == "pins: concurrent\n"


def test_fresh_full_frequency_does_not_inherit_old_partial_frequency() -> None:
    """Frequency one is omitted by annotation; an old explicit fraction must not survive."""
    merged = SchemaRegistry._merge_element_schema_with_existing(
        {"type": "string", "x-polylogue-frequency": 0.3, "x-polylogue-range": [1, 2]},
        {"type": "string", "x-polylogue-observed-distribution": {"documents": 10, "encountered_documents": 10}},
    )
    assert "x-polylogue-frequency" not in merged
    assert "x-polylogue-range" not in merged
    assert "x-polylogue-statistics-status" not in merged


def test_family_label_cannot_be_reassigned(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry)
    with pytest.raises(ValueError, match="another structural family"):
        publish(registry, generation=2, family="family-b")
    assert schema(SchemaRegistry(storage_root=tmp_path))["x-polylogue-test-generation"] == 1


def test_reappearing_field_replaces_historical_status() -> None:
    original: JSONDocument = {"type": "object", "properties": {"value": {"type": "string"}}}
    absent = SchemaRegistry._merge_element_schema_with_existing(original, {"type": "object"})
    reappeared = SchemaRegistry._merge_element_schema_with_existing(
        absent,
        {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "x-polylogue-observed-distribution": {"documents": 1, "encountered_documents": 1},
                }
            },
        },
    )
    properties = reappeared["properties"]
    assert isinstance(properties, dict)
    value = properties["value"]
    assert isinstance(value, dict)
    assert value.get("x-polylogue-observation-status") != "historical"
    assert value.get("x-polylogue-frequency", 1) == 1


def test_single_version_update_uses_fresh_catalog_for_default(tmp_path: Path) -> None:
    writer = SchemaRegistry(storage_root=tmp_path)
    publish(writer)
    assert writer.get_package("synthetic-publication") is not None
    other = SchemaRegistry(storage_root=tmp_path)
    publish(other, version="v2", generation=2, family="family-b")
    writer.write_schema_version("synthetic-publication", "v1", {"type": "object"})
    catalog = SchemaRegistry(storage_root=tmp_path).load_package_catalog("synthetic-publication")
    assert catalog is not None
    assert catalog.latest_version == catalog.default_version == "v2"


def test_generated_publication_redacts_numeric_history_after_merge(tmp_path: Path) -> None:
    """Old fields and omitted elements must retain structure without restoring private numbers."""
    registry = SchemaRegistry(storage_root=tmp_path)
    publish(registry, version="v2", generation=2, family="other-family")
    historical = schema(registry, "v2")
    numeric: JSONDocument = {
        "type": "number",
        "minimum": 0,
        "x-polylogue-range": [12.345, 67.89],
        "x-polylogue-evidence": {"range": [12.345, 67.89], "frequency": 1.0},
        "x-polylogue-observed-distribution": {
            "numeric": {"count": 2, "non_finite_count": 0, "min": 12.345, "max": 67.89, "mean": 40.1175},
        },
    }
    old_schema: JSONDocument = {"type": "object", "properties": {"retired": numeric}}
    old_schema["x-polylogue-time-deltas"] = [{"min_delta": 12.345, "max_delta": 67.89}]
    old_package = package()
    old_package.elements.append(SchemaElementManifest("retired_element", "retired.schema.json.gz", 1, 1))
    profile: JSONDocument = {
        "elements": {
            "session_document": {"field_profiles": {"$.retired": numeric["x-polylogue-observed-distribution"]}}
        },
    }
    profile["profile_id"] = workload_profile_identity(profile)
    registry.replace_provider_packages(
        old_package.provider,
        SchemaPackageCatalog(provider=old_package.provider, packages=[old_package], default_version="v1"),
        {"v1": {"session_document": old_schema, "retired_element": old_schema}},
        package_workload_profiles={"v1": profile},
    )
    ordinary = schema(SchemaRegistry(storage_root=tmp_path), "v1")
    ordinary_properties = ordinary["properties"]
    assert isinstance(ordinary_properties, dict)
    ordinary_retired = ordinary_properties["retired"]
    assert isinstance(ordinary_retired, dict)
    assert "x-polylogue-range" in ordinary_retired
    current: JSONDocument = {"type": "object", "properties": {"current": {"type": "string"}}}
    incoming = replace(old_package, elements=old_package.elements[:1])
    bundle = _ProviderBundle(
        result=GenerationResult(provider=old_package.provider, schema=current, sample_count=1),
        catalog=SchemaPackageCatalog(provider=old_package.provider, packages=[incoming], default_version="v1"),
        package_schemas={"v1": {"session_document": current}},
        package_workload_profiles={"v1": profile},
        manifest=ClusterManifest(provider=old_package.provider, default_version="v1"),
    )
    original_profile = copy.deepcopy(profile)
    persist_generated_provider_bundle(tmp_path, old_package.provider, bundle)
    fresh = SchemaRegistry(storage_root=tmp_path)
    assert schema(fresh, "v2") == historical
    for kind in ("session_document", "retired_element"):
        persisted = fresh.get_element_schema(old_package.provider, version="v1", element_kind=kind)
        assert persisted is not None
        properties = persisted["properties"]
        assert isinstance(properties, dict)
        retired = properties["retired"]
        assert isinstance(retired, dict)
        assert retired["type"] == "number" and retired["minimum"] == 0
        assert "x-polylogue-range" not in retired
        role_evidence = retired["x-polylogue-evidence"]
        assert isinstance(role_evidence, dict) and "range" not in role_evidence
        numeric_distribution = retired["x-polylogue-observed-distribution"]
        assert isinstance(numeric_distribution, dict)
        assert numeric_distribution["numeric"] == {"count": 2, "non_finite_count": 0}
        assert "x-polylogue-time-deltas" not in persisted
    stored_profile = fresh.get_workload_profile(old_package.provider, "v1")
    assert stored_profile is not None
    assert stored_profile["profile_id"] == workload_profile_identity(stored_profile)
    assert stored_profile != original_profile
    assert profile == original_profile
    assert "12.345" not in str(stored_profile)


def test_numeric_publication_projection_is_idempotent_and_preserves_constraints() -> None:
    """Projection must visit schema variants without treating schema property names as annotations."""
    source: JSONDocument = {
        "type": "object",
        "properties": {
            "x-polylogue-range": {"type": "number", "minimum": 5, "maximum": 9},
            "value": {"anyOf": [{"type": "number", "x-polylogue-range": [6, 8]}, {"type": "null"}]},
        },
    }
    original = copy.deepcopy(source)
    projected = redact_observed_numeric_schema(source)
    assert source == original
    assert redact_observed_numeric_schema(projected) == projected
    properties = projected["properties"]
    assert isinstance(properties, dict)
    assert properties["x-polylogue-range"] == {"type": "number", "minimum": 5, "maximum": 9}
    assert properties["value"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
