from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.operator_inference import infer_schema
from polylogue.schemas.operator_models import SchemaInferRequest
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
