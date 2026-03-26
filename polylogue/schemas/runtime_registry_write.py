"""Write/update helpers for runtime schema packages."""

from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry_catalog import SchemaRegistryCatalogMixin
from polylogue.schemas.runtime_registry_support import canonical_schema_provider


class SchemaRegistryWriteMixin(SchemaRegistryCatalogMixin):
    """Package write and registration helpers."""

    def write_package(
        self,
        package: SchemaVersionPackage,
        *,
        element_schemas: dict[str, dict[str, Any]],
    ):
        provider_token = str(canonical_schema_provider(package.provider))
        package_dir = self._package_dir(provider_token, package.version)
        elements_dir = package_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        for element in package.elements:
            if element.schema_file is None:
                continue
            schema = copy.deepcopy(element_schemas[element.element_kind])
            schema["$id"] = f"polylogue://schemas/{provider_token}/{package.version}/{element.element_kind}"
            schema["x-polylogue-version"] = int(package.version[1:]) if package.version.startswith("v") else package.version
            schema["x-polylogue-package-version"] = package.version
            schema["x-polylogue-element-kind"] = element.element_kind
            schema["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()
            schema_path = elements_dir / element.schema_file
            schema_path.write_bytes(gzip.compress(json.dumps(schema, indent=2).encode("utf-8")))

        manifest_path = self._package_manifest_path(provider_token, package.version)
        manifest_path.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    def replace_provider_packages(
        self,
        provider: str,
        catalog: SchemaPackageCatalog,
        package_schemas: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        provider_token = str(canonical_schema_provider(provider))
        provider_dir = self._provider_dir(provider_token)
        provider_dir.mkdir(parents=True, exist_ok=True)
        versions_dir = provider_dir / "versions"
        if versions_dir.exists():
            for path in versions_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(versions_dir.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        versions_dir.mkdir(parents=True, exist_ok=True)

        for package in catalog.packages:
            self.write_package(package, element_schemas=package_schemas[package.version])
        self.save_package_catalog(catalog)

    def _single_element_package(
        self,
        provider: str,
        *,
        version: str,
        schema: dict[str, Any],
        element_kind: str = "conversation_document",
        first_seen: str | None = None,
        last_seen: str | None = None,
    ) -> tuple[SchemaVersionPackage, dict[str, dict[str, Any]]]:
        provider_token = str(canonical_schema_provider(provider))
        now = datetime.now(tz=timezone.utc).isoformat()
        observed_at = schema.get("x-polylogue-generated-at")
        profile_family_ids = [
            str(item)
            for item in schema.get(
                "x-polylogue-package-profile-family-ids",
                schema.get("x-polylogue-profile-family-ids", []),
            )
            if isinstance(item, str)
        ]
        if not profile_family_ids:
            profile_family_ids = [
                str(item)
                for item in schema.get("x-polylogue-profile-family-ids", [])
                if isinstance(item, str)
            ]
        element_profile_family_ids = [
            str(item)
            for item in schema.get("x-polylogue-profile-family-ids", profile_family_ids)
            if isinstance(item, str)
        ]
        anchor_profile_family_id = str(schema.get("x-polylogue-anchor-profile-family-id", "")).strip() or next(
            (item for item in profile_family_ids if item),
            "",
        )
        observed_artifact_count = int(schema.get("x-polylogue-observed-artifact-count", 0) or 0)
        package = SchemaVersionPackage(
            provider=provider_token,
            version=version,
            anchor_kind=element_kind,
            default_element_kind=element_kind,
            first_seen=first_seen or observed_at or now,
            last_seen=last_seen or first_seen or observed_at or now,
            bundle_scope_count=0,
            sample_count=int(schema.get("x-polylogue-sample-count", 0) or 0),
            anchor_profile_family_id=anchor_profile_family_id,
            profile_family_ids=profile_family_ids,
            elements=[
                SchemaElementManifest(
                    element_kind=element_kind,
                    schema_file=f"{element_kind}.schema.json.gz",
                    sample_count=int(schema.get("x-polylogue-sample-count", 0) or 0),
                    artifact_count=observed_artifact_count,
                    first_seen=str(schema.get("x-polylogue-element-first-seen", observed_at or "")),
                    last_seen=str(schema.get("x-polylogue-element-last-seen", observed_at or "")),
                    bundle_scope_count=int(schema.get("x-polylogue-element-bundle-scope-count", 0) or 0),
                    exact_structure_ids=[str(item) for item in schema.get("x-polylogue-exact-structure-ids", [])],
                    profile_family_ids=element_profile_family_ids,
                    profile_tokens=[
                        str(item)
                        for item in schema.get("x-polylogue-profile-tokens", [])
                        if isinstance(item, str)
                    ],
                    representative_paths=[
                        str(item)
                        for item in schema.get("x-polylogue-representative-paths", [])
                        if isinstance(item, str)
                    ],
                    observed_artifact_count=observed_artifact_count,
                )
            ],
        )
        return package, {element_kind: schema}

    def register_schema(self, provider: str, schema: dict[str, Any]) -> str:
        provider_token = str(canonical_schema_provider(provider))
        versions = self.list_versions(provider_token)
        new_version = f"v{int(versions[-1][1:]) + 1}" if versions else "v1"
        self.write_schema_version(provider_token, new_version, schema)
        return new_version

    def write_schema_version(self, provider: str, version: str, schema: dict[str, Any]):
        provider_token = str(canonical_schema_provider(provider))
        package, schemas = self._single_element_package(provider_token, version=version, schema=copy.deepcopy(schema))
        catalog = self.load_package_catalog(provider_token) or SchemaPackageCatalog(provider=provider_token)
        existing_packages = [item for item in catalog.packages if item.version != version]
        existing_packages.append(package)
        existing_packages.sort(key=lambda item: int(item.version[1:]))
        catalog.packages = existing_packages
        catalog.latest_version = existing_packages[-1].version if existing_packages else version
        catalog.default_version = catalog.latest_version
        catalog.recommended_version = catalog.latest_version
        self.write_package(package, element_schemas=schemas)
        self.save_package_catalog(catalog)
        return self._package_dir(provider_token, version) / "elements" / f"{package.default_element_kind}.schema.json.gz"


__all__ = ["SchemaRegistryWriteMixin"]
