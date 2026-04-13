from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.operator_inference import infer_schema, list_inferred_corpus_specs
from polylogue.schemas.operator_models import SchemaInferRequest
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster


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
    assert result.corpus_specs[0].provider == "chatgpt"
    assert result.corpus_specs[0].package_version == "v3"
    assert result.corpus_specs[0].observed_sample_count == 9


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
    fake_registry = SimpleNamespace(
        cluster_samples=lambda provider, samples: manifest,
        save_cluster_manifest=lambda manifest: tmp_path / "manifest.json",
    )
    fake_provider = SimpleNamespace(db_provider_name="chatgpt")

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
    assert result.corpus_specs[0].profile_family_ids == ("cluster-a",)
    assert result.corpus_specs[0].element_kind == "conversation_document"


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
    fake_registry = SimpleNamespace(
        list_providers=lambda: ["chatgpt"],
        load_package_catalog=lambda provider: catalog,
        load_cluster_manifest=lambda provider: manifest,
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        specs = list_inferred_corpus_specs()

    assert len(specs) == 1
    assert specs[0].provider == "chatgpt"
    assert specs[0].package_version == "v7"
    assert specs[0].profile_family_ids == ("cluster-a",)
    assert specs[0].observed_sample_count == 12


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
    fake_registry = SimpleNamespace(
        list_providers=lambda: ["chatgpt", "codex"],
        load_package_catalog=lambda provider: {"chatgpt": chatgpt_catalog, "codex": codex_catalog}[provider],
        load_cluster_manifest=lambda provider: None,
    )

    with patch("polylogue.schemas.operator_inference.schema_registry", return_value=fake_registry):
        specs = list_inferred_corpus_specs(provider="chatgpt")

    assert tuple(spec.provider for spec in specs) == ("chatgpt",)
    assert specs[0].package_version == "v3"
