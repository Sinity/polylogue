from __future__ import annotations

import pytest

from polylogue.scenarios import CorpusSpec, build_default_corpus_specs, build_inferred_corpus_specs
from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster


def test_corpus_spec_payload_round_trip_preserves_inference_fields() -> None:
    spec = CorpusSpec(
        provider="chatgpt",
        package_version="v3",
        element_kind="conversation_document",
        count=4,
        messages_min=5,
        messages_max=9,
        seed=7,
        style="showcase",
        profile_family_ids=("cluster-a",),
        artifact_kind="conversation_document",
        observed_sample_count=23,
        observed_confidence=0.75,
        representative_paths=("/tmp/example.json",),
        origin="inferred.schema",
        tags=("synthetic", "schema"),
    )

    payload = spec.to_payload()

    assert CorpusSpec.from_payload(payload) == spec
    assert spec.messages_per_conversation == range(5, 10)


def test_corpus_spec_rejects_invalid_message_bounds() -> None:
    with pytest.raises(ValueError, match="messages_max"):
        CorpusSpec(provider="chatgpt", messages_min=6, messages_max=5)


def test_build_default_corpus_specs_preserves_provider_order() -> None:
    specs = build_default_corpus_specs(
        providers=("codex", "chatgpt"),
        count=2,
        messages_min=4,
        messages_max=6,
        seed=11,
    )

    assert tuple(spec.provider for spec in specs) == ("codex", "chatgpt")
    assert all(spec.origin == "generated.synthetic-defaults" for spec in specs)


def test_build_inferred_corpus_specs_uses_cluster_families_when_present() -> None:
    manifest = ClusterManifest(
        provider="chatgpt",
        clusters=[
            SchemaCluster(
                cluster_id="cluster-a",
                provider="chatgpt",
                sample_count=12,
                first_seen="2026-04-01T00:00:00Z",
                last_seen="2026-04-02T00:00:00Z",
                representative_paths=["/tmp/one.json"],
                dominant_keys=["id"],
                confidence=0.8,
                artifact_kind="conversation_document",
            )
        ],
    )

    specs = build_inferred_corpus_specs(
        provider="chatgpt",
        package_version="v5",
        manifest=manifest,
        sample_count=12,
    )

    assert len(specs) == 1
    assert specs[0].package_version == "v5"
    assert specs[0].element_kind == "conversation_document"
    assert specs[0].profile_family_ids == ("cluster-a",)
    assert specs[0].observed_sample_count == 12
    assert specs[0].origin == "inferred.schema"
