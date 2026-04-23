from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import (
    SchemaRegistry,
    _ObservedPayload,
    _read_gzip_json_dict,
    _read_json_dict,
    _resolved_package_version,
    canonical_schema_provider,
)
from polylogue.types import Provider


def _package(
    version: str,
    *,
    default_element_kind: str = "conversation_document",
    exact_structure_ids: list[str] | None = None,
    bundle_scopes: list[str] | None = None,
    profile_tokens: list[str] | None = None,
    schema_file: str | None = "conversation_document.schema.json.gz",
) -> SchemaVersionPackage:
    return SchemaVersionPackage(
        provider="chatgpt",
        version=version,
        anchor_kind=default_element_kind,
        default_element_kind=default_element_kind,
        first_seen="2026-04-01T00:00:00Z",
        last_seen="2026-04-02T00:00:00Z",
        bundle_scope_count=1,
        sample_count=1,
        bundle_scopes=bundle_scopes or [],
        elements=[
            SchemaElementManifest(
                element_kind=default_element_kind,
                schema_file=schema_file,
                sample_count=1,
                artifact_count=1,
                bundle_scopes=bundle_scopes or [],
                exact_structure_ids=exact_structure_ids or [],
                profile_tokens=profile_tokens or [],
            ),
            SchemaElementManifest(
                element_kind="adjunct",
                schema_file=None,
                sample_count=0,
                artifact_count=0,
            ),
        ],
    )


def _catalog(*packages: SchemaVersionPackage) -> SchemaPackageCatalog:
    return SchemaPackageCatalog(
        provider="chatgpt",
        packages=list(packages),
        default_version=packages[0].version if packages else None,
        latest_version=packages[-1].version if packages else None,
        recommended_version=packages[0].version if packages else None,
    )


def test_runtime_registry_helper_contracts_cover_json_and_provider_normalization(tmp_path: Path) -> None:
    json_path = tmp_path / "schema.json"
    json_path.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    gzip_path = tmp_path / "schema.json.gz"
    gzip_path.write_bytes(gzip.compress(json.dumps({"type": "object"}).encode("utf-8")))

    assert _read_json_dict(json_path) == {"type": "object"}
    assert _read_gzip_json_dict(gzip_path) == {"type": "object"}
    assert canonical_schema_provider("chatgpt") is Provider.CHATGPT
    assert canonical_schema_provider(" custom-provider ") == "custom-provider"
    assert canonical_schema_provider("") is Provider.UNKNOWN

    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text(json.dumps(["bad"]), encoding="utf-8")
    bad_gzip_path = tmp_path / "bad.json.gz"
    bad_gzip_path.write_bytes(gzip.compress(json.dumps(["bad"]).encode("utf-8")))

    with pytest.raises(ValueError, match="Expected JSON object"):
        _read_json_dict(bad_json_path)

    with pytest.raises(ValueError, match="Expected gzipped JSON object"):
        _read_gzip_json_dict(bad_gzip_path)


def test_resolved_package_version_prefers_default_latest_and_recommended() -> None:
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        packages=[],
        default_version="v2",
        latest_version="v3",
        recommended_version="v1",
    )

    assert _resolved_package_version(catalog, "default") == "v2"
    assert _resolved_package_version(catalog, "latest") == "v3"
    assert _resolved_package_version(catalog, "recommended") == "v1"
    assert _resolved_package_version(catalog, "v9") == "v9"


def test_write_and_replace_provider_packages_remove_stale_versions(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    old_catalog = _catalog(_package("v1"))
    new_catalog = _catalog(_package("v2"))

    registry.replace_provider_packages(
        "chatgpt",
        old_catalog,
        {"v1": {"conversation_document": {"type": "object"}}},
    )
    old_manifest = registry._package_manifest_path("chatgpt", "v1")
    assert old_manifest.exists()

    registry.replace_provider_packages(
        "chatgpt",
        new_catalog,
        {"v2": {"conversation_document": {"type": "object"}}},
    )

    assert not old_manifest.exists()
    assert registry._package_manifest_path("chatgpt", "v2").exists()
    assert "chatgpt" in registry.list_providers()


def test_get_element_schema_handles_missing_elements_and_files(tmp_path: Path) -> None:
    registry = SchemaRegistry(storage_root=tmp_path / "schemas")
    catalog = _catalog(_package("v1", schema_file="conversation_document.schema.json.gz"))
    registry.replace_provider_packages(
        "chatgpt",
        catalog,
        {"v1": {"conversation_document": {"type": "object"}}},
    )

    assert registry.get_element_schema("chatgpt", version="v1", element_kind="adjunct") is None

    schema_path = registry._package_dir("chatgpt", "v1") / "elements" / "conversation_document.schema.json.gz"
    schema_path.unlink()
    registry.clear_cache()

    assert registry.get_element_schema("chatgpt", version="v1") is None


def test_resolve_payload_prefers_exact_structure_then_profile_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SchemaRegistry()
    catalog = _catalog(
        _package("v1", exact_structure_ids=["exact-1"], bundle_scopes=["document"], profile_tokens=["field:mapping"]),
        _package("v2", bundle_scopes=["session"], profile_tokens=["field:messages"]),
    )
    monkeypatch.setattr(registry, "load_package_catalog", lambda _provider: catalog)

    exact_then_profile = (
        _ObservedPayload(
            artifact_kind="conversation_document",
            bundle_scope="session",
            exact_structure_id=None,
            profile_tokens=("field:messages",),
        ),
        _ObservedPayload(
            artifact_kind="conversation_document",
            bundle_scope="document",
            exact_structure_id="exact-1",
            profile_tokens=("field:mapping",),
        ),
    )
    monkeypatch.setattr(
        SchemaRegistry,
        "_observed_payloads",
        lambda self, provider, payload, source_path=None: exact_then_profile,
    )

    exact = registry.resolve_payload("chatgpt", {"mapping": {}}, source_path="/tmp/chat.json")
    assert exact is not None
    assert exact.package_version == "v1"
    assert exact.reason == "exact_structure"

    profile_only = (
        _ObservedPayload(
            artifact_kind="conversation_document",
            bundle_scope=None,
            exact_structure_id=None,
            profile_tokens=("field:messages",),
        ),
    )
    monkeypatch.setattr(
        SchemaRegistry,
        "_observed_payloads",
        lambda self, provider, payload, source_path=None: profile_only,
    )

    profile = registry.resolve_payload("chatgpt", {"messages": []}, source_path="/tmp/chat.json")
    assert profile is not None
    assert profile.package_version == "v2"
    assert profile.reason == "profile_family"
    assert profile.profile_score is not None

    default_only = (
        _ObservedPayload(
            artifact_kind="conversation_document",
            bundle_scope="unmatched",
            exact_structure_id=None,
            profile_tokens=(),
        ),
    )
    monkeypatch.setattr(
        SchemaRegistry,
        "_observed_payloads",
        lambda self, provider, payload, source_path=None: default_only,
    )

    default = registry.resolve_payload("chatgpt", {"messages": []}, source_path="/tmp/chat.json")
    assert default is not None
    assert default.package_version == "v1"
    assert default.reason == "package_default"


def test_resolve_payload_returns_none_when_catalog_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SchemaRegistry()
    monkeypatch.setattr(registry, "load_package_catalog", lambda _provider: None)

    assert registry.resolve_payload("chatgpt", {"mapping": {}}) is None
