"""Publication faults must expose one coherent provider generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.provider_bundle_packages import allocate_package_versions
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import SchemaRegistry


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
    assert "field_1" in schema(registry, "v1")["properties"]
    assert "field_1" not in schema(registry, "v2")["properties"]
    assert registry.get_workload_profile("synthetic-publication", "v1") == {"generation": 1}


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
    original = {"type": "object", "properties": {"value": {"type": "string"}}}
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
    assert properties["value"].get("x-polylogue-observation-status") != "historical"
    assert properties["value"].get("x-polylogue-frequency", 1) == 1


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
