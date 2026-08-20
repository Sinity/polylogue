"""The publication gate must run the enum-value privacy guard.

`check_privacy_guards` catches UUIDs, hex ids and high-entropy tokens recorded
verbatim in `x-polylogue-values` annotations. The required bundle check below
uses that predicate through the production `SchemaRegistry` over every
committed provider package, including non-default versions.

That was not theoretical: `promote_cluster`'s samples path calls
`generate_schema_from_samples()`, which unlike the full `generate` pipeline has
no `privacy_config` plumbed through it. A low-cardinality "safe enum" heuristic
with no UUID exemption recorded seven literal conversation UUIDs into
claude-ai v2, while the committed promotion audit reported zero blockers
throughout -- because it did not duplicate the check.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.promotion_audit import audit_schema_artifacts
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry


def _write_element(root: Path, provider: str, document: dict[str, object]) -> None:
    element_dir = root / "providers" / provider / "versions" / "v1" / "elements"
    element_dir.mkdir(parents=True, exist_ok=True)
    target = element_dir / "session_document.schema.json.gz"
    target.write_bytes(gzip.compress(json.dumps(document).encode("utf-8")))


def test_committed_schema_bundle_privacy_guard_is_green() -> None:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    expected_scopes: set[str] = set()
    for provider in registry.list_providers():
        for version in registry.list_versions(provider):
            package = registry.get_package(provider, version=version)
            assert package is not None
            expected_scopes.update(f"{provider}/{version}/{element.element_kind}" for element in package.elements)
            expected_scopes.update(
                f"{provider}/{version}/{schema_file}"
                for schema_file in registry.list_committed_schema_files(provider, version)
                if not any(element.schema_file == schema_file for element in package.elements)
            )

    report = audit_schema_bundle_privacy(registry=registry)

    assert report.checks, "the required schema privacy registry must inspect committed elements"
    assert report.all_passed, [check.format_line() for check in report.checks if check.status.value == "error"]
    assert len(report.checks) == len(expected_scopes)
    assert {getattr(check, "provider", None) for check in report.checks} == expected_scopes


def test_committed_schema_bundle_privacy_guard_red_twin(tmp_path: Path) -> None:
    """A leak planted in a real copied package makes the required gate red."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package = registry.get_package(provider, version=version)
    assert package is not None
    element = next(element for element in package.elements if element.schema_file is not None)
    schema_file = element.schema_file
    assert schema_file is not None
    schema_path = bundle_root / provider / "versions" / version / "elements" / schema_file

    with gzip.open(schema_path, "rt", encoding="utf-8") as stream:
        schema = json.load(stream)
    assert isinstance(schema, dict)
    leaked_value = "0f1e2d3c-4b5a-4968-8776-655443332211"
    schema["$defs"] = {
        "red_twin": {
            "type": "object",
            "properties": {"id": {"type": "string", "x-polylogue-values": [leaked_value]}},
        }
    }
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump(schema, stream)

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert privacy_failures, "the registered privacy predicate must reject the planted committed-schema leak"
    assert any(check.name == "privacy_guards" for check in privacy_failures)
    assert any("UUID leak" in detail for check in privacy_failures for detail in check.details)
    rendered = report.format_text()
    assert leaked_value not in rendered
    assert "sha256:" in rendered


def test_privacy_guard_traverses_nested_schema_containers(tmp_path: Path) -> None:
    """Definitions and other schema containers cannot hide unsafe values."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package = registry.get_package(provider, version=version)
    assert package is not None
    element = next(element for element in package.elements if element.schema_file is not None)
    schema_file = element.schema_file
    assert schema_file is not None
    schema_path = bundle_root / provider / "versions" / version / "elements" / schema_file
    leaked_values = {
        "$defs": "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
        "patternProperties": "2b3c4d5e-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
        "if": "3c4d5e6f-7a8b-4c9d-0e1f-2a3b4c5d6e7f",
        "then": "4d5e6f7a-8b9c-4d0e-1f2a-3b4c5d6e7f8a",
        "else": "5e6f7a8b-9c0d-4e1f-2a3b-4c5d6e7f8a9b",
        "contains": "6f7a8b9c-0d1e-4f2a-3b4c-5d6e7f8a9b0c",
        "prefixItems": "7a8b9c0d-1e2f-4a3b-4c5d-6e7f8a9b0c1d",
        "nested_arrays": "8b9c0d1e-2f3a-4b4c-5d6e-7f8a9b0c1d2e",
    }

    def nested_schema(value: str) -> dict[str, object]:
        return {"type": "string", "x-polylogue-values": [value]}

    with gzip.open(schema_path, "rt", encoding="utf-8") as stream:
        schema = json.load(stream)
    assert isinstance(schema, dict)
    schema.update(
        {
            "$defs": {"definition": nested_schema(leaked_values["$defs"])},
            "patternProperties": {"^nested": nested_schema(leaked_values["patternProperties"])},
            "if": nested_schema(leaked_values["if"]),
            "then": nested_schema(leaked_values["then"]),
            "else": nested_schema(leaked_values["else"]),
            "contains": nested_schema(leaked_values["contains"]),
            "prefixItems": [nested_schema(leaked_values["prefixItems"])],
            "nested_arrays": [[nested_schema(leaked_values["nested_arrays"])]],
        }
    )
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump(schema, stream)

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    failures = [check for check in report.checks if check.status.value == "error"]
    assert failures
    details = [detail for check in failures for detail in check.details]
    expected_paths = {
        "$.$defs.definition": "$defs",
        "$.patternProperties.^nested": "patternProperties",
        "$.if": "if",
        "$.then": "then",
        "$.else": "else",
        "$.contains": "contains",
        "$.prefixItems[0]": "prefixItems",
        "$.nested_arrays[0][0]": "nested_arrays",
    }
    for field, leaked_value in leaked_values.items():
        digest = hashlib.sha256(leaked_value.encode("utf-8")).hexdigest()[:16]
        expected_redaction = f"sha256:{digest};length={len(leaked_value)}"
        expected_path = next(path for path, path_field in expected_paths.items() if path_field == field)
        assert any(
            expected_path in detail and "UUID leak" in detail and expected_redaction in detail for detail in details
        ), field
    assert len([detail for detail in details if "UUID leak" in detail]) >= len(expected_paths)
    assert all(leaked_value not in report.format_text() for leaked_value in leaked_values.values())


def test_empty_discovered_provider_version_is_a_privacy_guard_failure(tmp_path: Path) -> None:
    """A clean package cannot hide a discovered provider version with no elements."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    assert registry.list_providers()
    assert registry.list_versions(registry.list_providers()[0])

    empty_provider = "empty-provider"
    registry.save_package_catalog(
        SchemaPackageCatalog(
            provider=empty_provider,
            latest_version="v1",
            default_version="v1",
            recommended_version="v1",
            packages=[
                SchemaVersionPackage(
                    provider=empty_provider,
                    version="v1",
                    anchor_kind="session_document",
                    default_element_kind="session_document",
                    first_seen="",
                    last_seen="",
                    bundle_scope_count=0,
                    sample_count=0,
                )
            ],
        )
    )

    report = audit_schema_bundle_privacy(registry=registry)

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert privacy_failures, "a registered empty provider/version must make the bundle audit fail"
    assert any(
        getattr(check, "provider", None) == "empty-provider/v1" and "no auditable elements" in check.summary
        for check in privacy_failures
    )
    assert any(check.status.value == "ok" for check in report.checks), "the copied clean package must be audited too"


def test_declared_element_without_schema_file_is_a_privacy_guard_failure(tmp_path: Path) -> None:
    """A supported element without a schema artifact must not be skipped."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    catalog = registry.load_package_catalog(provider)
    assert catalog is not None
    package = catalog.package(version)
    assert package is not None
    element = package.elements[0]
    element_kind = element.element_kind
    element.schema_file = None
    registry.save_package_catalog(catalog)
    package_path = bundle_root / provider / "versions" / version / "package.json"
    package_manifest = json.loads(package_path.read_text(encoding="utf-8"))
    assert isinstance(package_manifest, dict)
    package_elements = package_manifest["elements"]
    assert isinstance(package_elements, list)
    package_element = next(item for item in package_elements if item["element_kind"] == element_kind)
    assert isinstance(package_element, dict)
    package_element["schema_file"] = None
    package_path.write_text(json.dumps(package_manifest), encoding="utf-8")

    report = audit_schema_bundle_privacy(registry=registry)

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert any(
        getattr(check, "provider", None) == f"{provider}/{version}/{element_kind}"
        and "schema file is missing" in check.summary
        for check in privacy_failures
    )


def test_committed_bundle_without_catalog_is_still_privacy_audited(tmp_path: Path) -> None:
    """A package and schema artifact remain visible to the gate without catalog.json."""
    provider = "orphan-provider"
    version = "v1"
    version_dir = tmp_path / provider / "versions" / version
    schema_file = "session_document.schema.json.gz"
    schema_path = version_dir / "elements" / schema_file
    schema_path.parent.mkdir(parents=True)
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "type": "object",
                "x-polylogue-values": ["0f1e2d3c-4b5a-4968-8776-655443332211"],
            },
            stream,
        )
    (version_dir / "package.json").write_text(
        json.dumps(
            {
                "provider": provider,
                "version": version,
                "anchor_kind": "session_document",
                "default_element_kind": "session_document",
                "first_seen": "",
                "last_seen": "",
                "bundle_scope_count": 0,
                "sample_count": 1,
                "elements": [
                    {
                        "element_kind": "session_document",
                        "schema_file": schema_file,
                        "sample_count": 1,
                        "artifact_count": 1,
                        "supported": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=tmp_path))

    failures = [check for check in report.checks if check.status.value == "error"]
    assert any(getattr(check, "provider", None) == provider for check in failures)
    assert any(getattr(check, "provider", None) == f"{provider}/{version}/session_document" for check in failures)
    assert any("UUID leak" in detail for check in failures for detail in check.details)


def test_orphan_committed_element_schema_is_privacy_audited(tmp_path: Path) -> None:
    """An element gzip absent from both manifests still participates in the gate."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package = registry.get_package(provider, version=version)
    assert package is not None
    orphan_file = "orphan-element.schema.json.gz"
    orphan_path = bundle_root / provider / "versions" / version / "elements" / orphan_file
    leaked_value = "6f5e4d3c-2b1a-4987-8765-554433221100"
    with gzip.open(orphan_path, "wt", encoding="utf-8") as stream:
        json.dump({"type": "string", "x-polylogue-values": [leaked_value]}, stream)

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    failures = [check for check in report.checks if check.status.value == "error"]
    orphan_scope = f"{provider}/{version}/{orphan_file}"
    assert any(getattr(check, "provider", None) == orphan_scope for check in failures)
    assert any("UUID leak" in detail for check in failures for detail in check.details)
    assert leaked_value not in report.format_text()


def test_unmanifested_committed_version_element_is_privacy_audited(tmp_path: Path) -> None:
    """A version gzip remains visible to the gate without package.json or catalog.json."""
    provider = "unmanifested-provider"
    version = "v9"
    schema_file = "unmanifested-element.schema.json.gz"
    schema_path = tmp_path / provider / "versions" / version / "elements" / schema_file
    schema_path.parent.mkdir(parents=True)
    leaked_value = "7a8b9c0d-1e2f-4a3b-4c5d-6e7f8a9b0c1d"
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump({"type": "string", "x-polylogue-values": [leaked_value]}, stream)

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=tmp_path))

    failures = [check for check in report.checks if check.status.value == "error"]
    artifact_scope = f"{provider}/{version}/{schema_file}"
    assert any(getattr(check, "provider", None) == artifact_scope for check in failures)
    assert any("UUID leak" in detail for check in failures for detail in check.details)
    assert leaked_value not in report.format_text()


def test_committed_alias_provider_path_is_audited_literally(tmp_path: Path) -> None:
    """Inventory discovery must not redirect an alias directory to its canonical sibling."""
    provider = "openai"
    version = "v1"
    schema_file = "session_document.schema.json.gz"
    version_dir = tmp_path / provider / "versions" / version
    schema_path = version_dir / "elements" / schema_file
    schema_path.parent.mkdir(parents=True)
    leaked_value = "abcdef12-3456-4789-abcd-ef1234567890"
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump({"type": "string", "x-polylogue-values": [leaked_value]}, stream)
    package = {
        "provider": provider,
        "version": version,
        "anchor_kind": "session_document",
        "default_element_kind": "session_document",
        "first_seen": "",
        "last_seen": "",
        "bundle_scope_count": 0,
        "sample_count": 1,
        "elements": [
            {
                "element_kind": "session_document",
                "schema_file": schema_file,
                "sample_count": 1,
                "artifact_count": 1,
                "supported": True,
            }
        ],
    }
    (version_dir / "package.json").write_text(json.dumps(package), encoding="utf-8")
    (tmp_path / provider / "catalog.json").write_text(
        json.dumps({"provider": provider, "packages": [package]}), encoding="utf-8"
    )

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=tmp_path))

    failures = [check for check in report.checks if check.status.value == "error"]
    assert any(
        getattr(check, "provider", None) == f"{provider}/{version}/session_document"
        and any("UUID leak" in detail for detail in check.details)
        for check in failures
    )
    assert leaked_value not in report.format_text()


def test_catalog_and_package_schema_file_disagreement_audits_both_artifacts(tmp_path: Path) -> None:
    """A package-only schema file cannot hide behind a safe catalog reference."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package_path = bundle_root / provider / "versions" / version / "package.json"
    package_manifest = json.loads(package_path.read_text(encoding="utf-8"))
    assert isinstance(package_manifest, dict)
    package_elements = package_manifest["elements"]
    assert isinstance(package_elements, list)
    package_element = next(item for item in package_elements if item["element_kind"] == "session_document")
    assert isinstance(package_element, dict)
    leaked_schema_file = "package-only-leak.schema.json.gz"
    package_element["schema_file"] = leaked_schema_file
    package_path.write_text(json.dumps(package_manifest), encoding="utf-8")
    leaked_schema_path = package_path.parent / "elements" / leaked_schema_file
    with gzip.open(leaked_schema_path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "type": "object",
                "x-polylogue-values": ["0f1e2d3c-4b5a-4968-8776-655443332211"],
            },
            stream,
        )

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    failures = [check for check in report.checks if check.status.value == "error"]
    element_scope = f"{provider}/{version}/session_document"
    assert any(
        getattr(check, "provider", None) == element_scope
        and check.summary == "Catalog/package schema_file disagreement"
        for check in failures
    )
    assert any(
        getattr(check, "provider", None) == element_scope and any("UUID leak" in detail for detail in check.details)
        for check in failures
    )


def test_intentionally_unsupported_element_without_schema_is_not_a_gate_failure(tmp_path: Path) -> None:
    """Unsupported manifest entries may omit an artifact without failing the gate."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package_path = bundle_root / provider / "versions" / version / "package.json"
    package_manifest = json.loads(package_path.read_text(encoding="utf-8"))
    catalog_path = bundle_root / provider / "catalog.json"
    catalog_manifest = json.loads(catalog_path.read_text(encoding="utf-8"))
    unsupported = {
        "element_kind": "future_element",
        "schema_file": None,
        "sample_count": 0,
        "artifact_count": 0,
        "supported": False,
    }
    assert isinstance(package_manifest, dict)
    package_elements = package_manifest["elements"]
    assert isinstance(package_elements, list)
    package_elements.append(unsupported)
    assert isinstance(catalog_manifest, dict)
    catalog_packages = catalog_manifest["packages"]
    assert isinstance(catalog_packages, list)
    catalog_package = next(item for item in catalog_packages if item["version"] == version)
    assert isinstance(catalog_package, dict)
    catalog_elements = catalog_package["elements"]
    assert isinstance(catalog_elements, list)
    catalog_elements.append(dict(unsupported))
    package_path.write_text(json.dumps(package_manifest), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog_manifest), encoding="utf-8")

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    assert report.all_passed, [check.format_line() for check in report.checks if check.status.value == "error"]
    assert not any(
        getattr(check, "provider", None) == f"{provider}/{version}/future_element" for check in report.checks
    )


def test_declared_schema_artifact_missing_is_a_privacy_guard_failure(tmp_path: Path) -> None:
    """A declared schema whose copied artifact is absent must fail the registry workflow."""
    bundle_root = tmp_path / "providers"
    shutil.copytree(SCHEMA_DIR, bundle_root)
    registry = SchemaRegistry(storage_root=bundle_root)
    provider = registry.list_providers()[0]
    version = registry.list_versions(provider)[0]
    package = registry.get_package(provider, version=version)
    assert package is not None
    element = next(element for element in package.elements if element.schema_file is not None)
    schema_file = element.schema_file
    assert schema_file is not None
    schema_path = bundle_root / provider / "versions" / version / "elements" / schema_file
    assert schema_path.is_file()
    schema_path.unlink()
    assert element.schema_file == schema_file

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert any(
        getattr(check, "provider", None) == f"{provider}/{version}/{element.element_kind}"
        and check.summary == "Committed element schema is missing"
        for check in privacy_failures
    )


def test_uuid_recorded_as_an_observed_value_blocks_promotion(tmp_path: Path) -> None:
    """The exact claude-ai v2 shape: real UUIDs captured as a "safe enum"."""
    _write_element(
        tmp_path,
        "example",
        {
            "type": "object",
            "properties": {
                "current_leaf_message_uuid": {
                    "type": "string",
                    "x-polylogue-values": [
                        "0f1e2d3c-4b5a-4968-8776-655443332211",
                        "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
                    ],
                }
            },
        },
    )

    report = audit_schema_artifacts(tmp_path)

    assert report.blockers, "a UUID in x-polylogue-values must block publication"
    assert any(item.category == "unsafe_enum_value" for item in report.blockers)
    assert report.to_payload()["verdict"] == "blocked"


def test_genuine_enum_values_do_not_block(tmp_path: Path) -> None:
    """The guard must not fire on the low-cardinality vocabularies it exists to allow."""
    _write_element(
        tmp_path,
        "example",
        {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "x-polylogue-values": ["ok", "success", "failed", "in_progress"],
                }
            },
        },
    )

    report = audit_schema_artifacts(tmp_path)

    assert not [item for item in report.blockers if item.category == "unsafe_enum_value"]
