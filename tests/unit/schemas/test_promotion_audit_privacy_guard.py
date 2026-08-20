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
    schema["x-polylogue-values"] = ["0f1e2d3c-4b5a-4968-8776-655443332211"]
    with gzip.open(schema_path, "wt", encoding="utf-8") as stream:
        json.dump(schema, stream)

    report = audit_schema_bundle_privacy(registry=SchemaRegistry(storage_root=bundle_root))

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert privacy_failures, "the registered privacy predicate must reject the planted committed-schema leak"
    assert any(check.name == "privacy_guards" for check in privacy_failures)
    assert any("UUID leak" in detail for check in privacy_failures for detail in check.details)


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
    """A declared element without a schema artifact must not be skipped."""
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

    report = audit_schema_bundle_privacy(registry=registry)

    privacy_failures = [check for check in report.checks if check.status.value == "error"]
    assert any(
        getattr(check, "provider", None) == f"{provider}/{version}/{element_kind}"
        and "schema file is missing" in check.summary
        for check in privacy_failures
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
