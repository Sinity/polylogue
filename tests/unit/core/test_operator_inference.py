from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from polylogue.lib.json import JSONDocument
from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.operator_inference import (
    audit_schemas,
    compare_schema_versions,
    infer_schema,
    list_inferred_corpus_scenarios,
    list_inferred_corpus_specs,
    list_schemas,
    promote_schema_cluster,
)
from polylogue.schemas.operator_models import (
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaInferRequest,
    SchemaListRequest,
    SchemaPromoteRequest,
)
from polylogue.schemas.operator_registry import SchemaRegistryLike
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaResolution,
    SchemaVersionPackage,
)
from polylogue.schemas.privacy_config import PrivacyConfig
from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster, SchemaDiff


@dataclass(frozen=True)
class _ProviderConfig:
    db_provider_name: str | None


class _FakeSchemaRegistry(SchemaRegistryLike):
    def __init__(
        self,
        *,
        catalogs: dict[str, SchemaPackageCatalog] | None = None,
        manifests: dict[str, ClusterManifest | None] | None = None,
        clustered_manifest: ClusterManifest | None = None,
        manifest_path: Path | None = None,
        diff: SchemaDiff | None = None,
        promoted_version: str = "v99",
        promoted_package: SchemaVersionPackage | None = None,
        promoted_schema: JSONDocument | None = None,
        schema_age_days: dict[str, int | None] | None = None,
    ) -> None:
        self._catalogs = catalogs or {}
        self._manifests = manifests or {}
        self._clustered_manifest = clustered_manifest
        self._manifest_path = manifest_path or Path("manifest.json")
        self._diff = diff or SchemaDiff(provider="test", version_a="v1", version_b="v2")
        self._promoted_version = promoted_version
        self._promoted_package = promoted_package
        self._promoted_schema = promoted_schema
        self._schema_age_days = schema_age_days or {}

    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        return self._catalogs.get(provider)

    def get_package(self, provider: str, version: str = "default") -> SchemaVersionPackage | None:
        if self._promoted_package is not None and version == self._promoted_version:
            return self._promoted_package
        catalog = self.load_package_catalog(provider)
        return catalog.package(version) if catalog is not None else None

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> JSONDocument | None:
        if version == self._promoted_version:
            return self._promoted_schema
        return None

    def list_versions(self, provider: str) -> list[str]:
        if self._promoted_package is not None:
            return [self._promoted_version]
        catalog = self.load_package_catalog(provider)
        return [package.version for package in catalog.packages] if catalog is not None else []

    def list_providers(self) -> list[str]:
        provider_names = set(self._catalogs) | set(self._manifests)
        return sorted(provider_names)

    def get_schema_age_days(self, provider: str) -> int | None:
        return self._schema_age_days.get(provider)

    def resolve_payload(
        self,
        provider: str,
        payload: Mapping[str, object],
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None:
        return None

    def compare_versions(
        self,
        provider: str,
        v1: str,
        v2: str,
        *,
        element_kind: str | None = None,
    ) -> SchemaDiff:
        return self._diff

    def cluster_samples(self, provider: str, samples: Sequence[Mapping[str, object]]) -> ClusterManifest:
        if self._clustered_manifest is None:
            raise AssertionError("cluster_samples should not be called without a manifest")
        return self._clustered_manifest

    def save_cluster_manifest(self, manifest: ClusterManifest) -> Path:
        return self._manifest_path

    def load_cluster_manifest(self, provider: str) -> ClusterManifest | None:
        return self._manifests.get(provider)

    def promote_cluster(
        self,
        provider: str,
        cluster_id: str,
        *,
        samples: Sequence[Mapping[str, object]] | None = None,
    ) -> str:
        return self._promoted_version


def test_infer_schema_emits_default_corpus_specs_without_clustering(tmp_path: Path) -> None:
    generation = GenerationResult(
        provider="chatgpt",
        schema={"type": "object"},
        sample_count=9,
        default_version="v3",
    )

    with patch("polylogue.schemas.generation_workflow.generate_provider_schema", return_value=generation):
        result = infer_schema(
            SchemaInferRequest(
                provider="chatgpt",
                db_path=tmp_path / "archive.db",
                cluster=False,
            )
        )

    assert result.manifest is None
    assert len(result.corpus_specs) == 1
    assert len(result.corpus_scenarios) == 1
    assert result.corpus_specs[0].provider == "chatgpt"
    assert result.corpus_specs[0].package_version == "v3"
    assert result.corpus_specs[0].profile.observed_sample_count == 9
    assert result.corpus_scenarios[0].provider == "chatgpt"
    assert result.corpus_scenarios[0].package_version == "v3"


def test_infer_schema_emits_cluster_backed_corpus_specs_when_manifest_exists(tmp_path: Path) -> None:
    generation = GenerationResult(
        provider="chatgpt",
        schema={"type": "object"},
        sample_count=15,
        default_version="v4",
        cluster_count=1,
    )
    manifest = ClusterManifest(
        provider="chatgpt",
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=15,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source.json"],
                dominant_keys=["id", "messages"],
                confidence=0.92,
                artifact_kind="conversation_document",
            )
        ],
    )
    fake_registry = _FakeSchemaRegistry(
        clustered_manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
    )
    fake_provider = _ProviderConfig(db_provider_name="chatgpt")

    with (
        patch("polylogue.schemas.generation_workflow.generate_provider_schema", return_value=generation),
        patch.dict("polylogue.schemas.observation.PROVIDERS", {"chatgpt": fake_provider}, clear=False),
        patch("polylogue.schemas.sampling.load_samples_from_db", return_value=[{"id": "1"}]),
        patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry),
    ):
        result = infer_schema(
            SchemaInferRequest(
                provider="chatgpt",
                db_path=tmp_path / "archive.db",
                cluster=True,
            )
        )

    assert result.manifest is manifest
    assert result.manifest_path == tmp_path / "manifest.json"
    assert len(result.corpus_specs) == 1
    assert len(result.corpus_scenarios) == 1
    assert result.corpus_specs[0].profile.family_ids == ("cluster-a",)
    assert result.corpus_specs[0].element_kind == "conversation_document"
    assert result.corpus_scenarios[0].corpus_specs[0].profile.family_ids == ("cluster-a",)


def test_list_inferred_corpus_specs_reads_registry_catalog_and_manifest() -> None:
    manifest = ClusterManifest(
        provider="chatgpt",
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=12,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source.json"],
                dominant_keys=["id", "messages"],
                confidence=0.92,
                artifact_kind="conversation_document",
                promoted_package_version="v7",
            )
        ],
        default_version="v7",
    )
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        default_version="v7",
        latest_version="v7",
        recommended_version="v7",
        packages=[
            SchemaVersionPackage(
                provider="chatgpt",
                version="v7",
                anchor_kind="conversation_document",
                default_element_kind="conversation_document",
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                bundle_scope_count=1,
                sample_count=12,
                elements=[
                    SchemaElementManifest(
                        element_kind="conversation_document",
                        schema_file="conversation_document.schema.json",
                        sample_count=12,
                        artifact_count=12,
                    )
                ],
            )
        ],
    )
    fake_registry = _FakeSchemaRegistry(
        catalogs={"chatgpt": catalog},
        manifests={"chatgpt": manifest},
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        specs = list_inferred_corpus_specs()

    assert len(specs) == 1
    assert specs[0].provider == "chatgpt"
    assert specs[0].package_version == "v7"
    assert specs[0].profile.family_ids == ("cluster-a",)
    assert specs[0].profile.observed_sample_count == 12


def test_list_inferred_corpus_scenarios_groups_specs_by_provider_and_version() -> None:
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        default_version="v7",
        packages=[
            SchemaVersionPackage(
                provider="chatgpt",
                version="v7",
                anchor_kind="conversation_document",
                default_element_kind="conversation_document",
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                bundle_scope_count=1,
                sample_count=12,
            )
        ],
    )
    manifest = ClusterManifest(
        provider="chatgpt",
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=7,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source-a.json"],
                dominant_keys=["id"],
                confidence=0.9,
                artifact_kind="conversation_document",
            ),
            SchemaCluster(
                cluster_id="cluster-b",
                provider="chatgpt",
                sample_count=5,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source-b.json"],
                dominant_keys=["id"],
                confidence=0.8,
                artifact_kind="conversation_document",
            ),
        ],
        default_version="v7",
    )
    fake_registry = _FakeSchemaRegistry(
        catalogs={"chatgpt": catalog},
        manifests={"chatgpt": manifest},
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        scenarios = list_inferred_corpus_scenarios()

    assert len(scenarios) == 1
    assert scenarios[0].provider == "chatgpt"
    assert scenarios[0].package_version == "v7"
    assert tuple(spec.profile.primary_family_id for spec in scenarios[0].corpus_specs) == ("cluster-a", "cluster-b")


def test_list_inferred_corpus_specs_can_scope_to_one_provider() -> None:
    chatgpt_catalog = SchemaPackageCatalog(
        provider="chatgpt",
        default_version="v3",
        packages=[
            SchemaVersionPackage(
                provider="chatgpt",
                version="v3",
                anchor_kind="conversation_document",
                default_element_kind="conversation_document",
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                bundle_scope_count=1,
                sample_count=9,
            )
        ],
    )
    codex_catalog = SchemaPackageCatalog(
        provider="codex",
        default_version="v1",
        packages=[
            SchemaVersionPackage(
                provider="codex",
                version="v1",
                anchor_kind="conversation_document",
                default_element_kind="conversation_document",
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                bundle_scope_count=1,
                sample_count=4,
            )
        ],
    )
    fake_registry = _FakeSchemaRegistry(
        catalogs={"chatgpt": chatgpt_catalog, "codex": codex_catalog},
        manifests={"chatgpt": None, "codex": None},
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        specs = list_inferred_corpus_specs(provider="chatgpt")

    assert tuple(spec.provider for spec in specs) == ("chatgpt",)
    assert specs[0].package_version == "v3"


def test_infer_schema_coerces_privacy_config_and_falls_back_without_db_provider(tmp_path: Path) -> None:
    captured_privacy: list[PrivacyConfig | None] = []
    generation = GenerationResult(
        provider="chatgpt",
        schema={"type": "object"},
        sample_count=5,
        default_version="v2",
    )
    fake_provider = _ProviderConfig(db_provider_name=None)

    def _generate_provider_schema(*args: object, **kwargs: object) -> GenerationResult:
        captured_privacy.append(cast(PrivacyConfig | None, kwargs["privacy_config"]))
        return generation

    with (
        patch("polylogue.schemas.generation_workflow.generate_provider_schema", side_effect=_generate_provider_schema),
        patch.dict("polylogue.schemas.observation.PROVIDERS", {"chatgpt": fake_provider}, clear=False),
    ):
        result = infer_schema(
            SchemaInferRequest(
                provider="chatgpt",
                db_path=tmp_path / "archive.db",
                cluster=True,
                privacy_config={
                    "level": "strict",
                    "field_overrides": {"$.id": "drop", "bad": 1},
                    "allow_value_patterns": ["ok", 2],
                    "deny_value_patterns": ["secret"],
                    "safe_enum_max_length": "bad",
                    "high_entropy_min_length": 14,
                    "cross_conv_min_count": 5,
                    "cross_conv_proportional": True,
                },
            )
        )

    privacy = cast(PrivacyConfig, captured_privacy[0])
    assert privacy is not None
    assert privacy.level == "strict"
    assert privacy.safe_enum_max_length == 30
    assert privacy.high_entropy_min_length == 14
    assert privacy.cross_conv_min_count == 5
    assert privacy.cross_conv_proportional is True
    assert privacy.field_overrides == {"$.id": "drop"}
    assert privacy.allow_value_patterns == ["ok"]
    assert privacy.deny_value_patterns == ["secret"]
    assert result.manifest is None
    assert len(result.corpus_scenarios) == 1


def test_list_schemas_returns_selected_and_all_provider_snapshots() -> None:
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        default_version="v7",
        latest_version="v7",
        packages=[
            SchemaVersionPackage(
                provider="chatgpt",
                version="v7",
                anchor_kind="conversation_document",
                default_element_kind="conversation_document",
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                bundle_scope_count=1,
                sample_count=12,
            )
        ],
    )
    manifest = ClusterManifest(
        provider="chatgpt",
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=12,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/source.json"],
                dominant_keys=["id"],
                confidence=0.9,
                artifact_kind="conversation_document",
            )
        ],
    )
    fake_registry = _FakeSchemaRegistry(
        catalogs={"chatgpt": catalog},
        manifests={"chatgpt": manifest, "codex": None},
        schema_age_days={"chatgpt": 3, "codex": None},
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        selected = list_schemas(SchemaListRequest(provider="chatgpt"))
        listing = list_schemas(SchemaListRequest())

    assert selected.selected is not None
    assert selected.selected.provider == "chatgpt"
    assert selected.selected.latest_age_days == 3
    assert selected.selected.to_dict()["provider"] == "chatgpt"
    assert [snapshot.provider for snapshot in listing.providers] == ["chatgpt", "codex"]


def test_compare_schema_versions_wraps_registry_diff() -> None:
    diff = SchemaDiff(
        provider="chatgpt",
        version_a="v1",
        version_b="v2",
        changed_properties=["messages"],
    )
    fake_registry = _FakeSchemaRegistry(diff=diff)

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        result = compare_schema_versions(SchemaCompareRequest(provider="chatgpt", from_version="v1", to_version="v2"))

    assert result.diff is diff
    assert result.to_dict()["provider"] == "chatgpt"


def test_promote_schema_cluster_with_samples_filters_matching_cluster(tmp_path: Path) -> None:
    promoted_package = SchemaVersionPackage(
        provider="chatgpt",
        version="v8",
        anchor_kind="conversation_document",
        default_element_kind="conversation_document",
        first_seen="2026-04-01T00:00:00Z",
        last_seen="2026-04-02T00:00:00Z",
        bundle_scope_count=1,
        sample_count=2,
    )
    fake_registry = _FakeSchemaRegistry(
        promoted_version="v8",
        promoted_package=promoted_package,
        promoted_schema={"type": "object"},
    )
    fake_provider = _ProviderConfig(db_provider_name="chatgpt")

    with (
        patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry),
        patch.dict("polylogue.schemas.observation.PROVIDERS", {"chatgpt": fake_provider}, clear=False),
        patch("polylogue.schemas.sampling.load_samples_from_db", return_value=[{"id": "one"}, {"id": "two"}]),
        patch("polylogue.schemas.shape_fingerprint._structure_fingerprint", lambda sample: sample["id"]),
        patch(
            "polylogue.schemas.observation.fingerprint_hash",
            lambda fingerprint: "cluster-a" if fingerprint == "one" else "other",
        ),
    ):
        result = promote_schema_cluster(
            SchemaPromoteRequest(
                provider="chatgpt",
                cluster_id="cluster-a",
                db_path=tmp_path / "archive.db",
                with_samples=True,
            )
        )

    assert result.package_version == "v8"
    assert result.package is promoted_package
    assert result.schema == {"type": "object"}
    assert result.versions == ["v8"]


def test_promote_schema_cluster_rejects_unknown_provider_and_missing_cluster_samples(tmp_path: Path) -> None:
    fake_registry = _FakeSchemaRegistry()

    with (
        patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry),
        patch.dict("polylogue.schemas.observation.PROVIDERS", {}, clear=True),
    ):
        with pytest.raises(ValueError, match="Unknown provider"):
            promote_schema_cluster(
                SchemaPromoteRequest(
                    provider="chatgpt",
                    cluster_id="cluster-a",
                    db_path=tmp_path / "archive.db",
                    with_samples=True,
                )
            )

    fake_provider = _ProviderConfig(db_provider_name="chatgpt")
    with (
        patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry),
        patch.dict("polylogue.schemas.observation.PROVIDERS", {"chatgpt": fake_provider}, clear=False),
        patch("polylogue.schemas.sampling.load_samples_from_db", return_value=[{"id": "two"}]),
        patch("polylogue.schemas.shape_fingerprint._structure_fingerprint", lambda sample: sample["id"]),
        patch("polylogue.schemas.observation.fingerprint_hash", lambda _fingerprint: "other"),
    ):
        with pytest.raises(ValueError, match="No samples match cluster"):
            promote_schema_cluster(
                SchemaPromoteRequest(
                    provider="chatgpt",
                    cluster_id="cluster-a",
                    db_path=tmp_path / "archive.db",
                    with_samples=True,
                )
            )


def test_audit_schemas_selects_provider_or_global_workflow() -> None:
    provider_report = object()
    all_report = object()

    with (
        patch("polylogue.schemas.audit_workflow.audit_provider", return_value=provider_report),
        patch("polylogue.schemas.audit_workflow.audit_all_providers", return_value=all_report),
    ):
        assert audit_schemas(SchemaAuditRequest(provider="chatgpt")) is provider_report
        assert audit_schemas(SchemaAuditRequest()) is all_report
