from __future__ import annotations

import pytest

from polylogue.scenarios import (
    CorpusScenario,
    CorpusSpec,
    ScenarioProjectionSourceKind,
    build_corpus_scenarios,
    build_default_corpus_specs,
    build_inferred_corpus_specs,
    resolve_corpus_specs,
)
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


def test_build_default_corpus_specs_accepts_metadata_overrides() -> None:
    specs = build_default_corpus_specs(
        providers=("chatgpt",),
        count=1,
        origin="generated.test-suite",
        tags=("synthetic", "test", "fixtures"),
    )

    assert specs[0].origin == "generated.test-suite"
    assert specs[0].tags == ("synthetic", "test", "fixtures")


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


def test_resolve_corpus_specs_applies_generation_overrides_to_inferred_specs() -> None:
    inferred = (
        CorpusSpec(
            provider="chatgpt",
            package_version="v7",
            count=12,
            messages_min=4,
            messages_max=16,
            style="default",
            profile_family_ids=("cluster-a",),
            origin="inferred.schema",
            tags=("inferred", "schema", "synthetic"),
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "polylogue.schemas.operator_inference.list_inferred_corpus_specs",
            lambda registry=None: inferred,
        )
        specs = resolve_corpus_specs(
            source="inferred",
            count=2,
            messages_min=6,
            messages_max=8,
            seed=42,
            style="showcase",
        )

    assert len(specs) == 1
    assert specs[0].provider == "chatgpt"
    assert specs[0].package_version == "v7"
    assert specs[0].profile_family_ids == ("cluster-a",)
    assert specs[0].count == 2
    assert specs[0].messages_min == 6
    assert specs[0].messages_max == 8
    assert specs[0].seed == 42
    assert specs[0].style == "showcase"
    assert specs[0].origin == "generated.synthetic-inferred"
    assert specs[0].tags == ("synthetic", "generated", "inferred")


def test_corpus_spec_scope_label_includes_version_and_profile_family() -> None:
    spec = CorpusSpec(
        provider="chatgpt",
        package_version="v7",
        profile_family_ids=("cluster/a",),
    )

    assert spec.scope_label == "v7-cluster-a"


def test_build_corpus_scenarios_groups_specs_by_provider_and_version() -> None:
    scenarios = build_corpus_scenarios(
        (
            CorpusSpec(
                provider="chatgpt",
                package_version="v7",
                profile_family_ids=("cluster-a",),
                origin="inferred.schema",
                tags=("inferred", "schema", "synthetic"),
            ),
            CorpusSpec(
                provider="chatgpt",
                package_version="v7",
                profile_family_ids=("cluster-b",),
                origin="inferred.schema",
                tags=("inferred", "schema", "synthetic"),
            ),
        ),
        origin="compiled.inferred-corpus-scenario",
        tags=("inferred", "schema", "synthetic", "scenario"),
    )

    assert scenarios == (
        CorpusScenario(
            provider="chatgpt",
            package_version="v7",
            corpus_specs=(
                CorpusSpec(
                    provider="chatgpt",
                    package_version="v7",
                    profile_family_ids=("cluster-a",),
                    origin="inferred.schema",
                    tags=("inferred", "schema", "synthetic"),
                ),
                CorpusSpec(
                    provider="chatgpt",
                    package_version="v7",
                    profile_family_ids=("cluster-b",),
                    origin="inferred.schema",
                    tags=("inferred", "schema", "synthetic"),
                ),
            ),
            origin="compiled.inferred-corpus-scenario",
            tags=("inferred", "schema", "synthetic", "scenario"),
        ),
    )


def test_corpus_scenario_compiles_its_own_projection_entry() -> None:
    scenario = CorpusScenario(
        provider="chatgpt",
        package_version="v7",
        corpus_specs=(
            CorpusSpec(
                provider="chatgpt",
                package_version="v7",
                element_kind="conversation_document",
                profile_family_ids=("cluster/a",),
                observed_sample_count=12,
                origin="inferred.schema",
                tags=("inferred", "schema", "synthetic"),
            ),
        ),
        origin="compiled.inferred-corpus-scenario",
        tags=("inferred", "schema", "synthetic", "scenario"),
    )

    projection = scenario.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.INFERRED_CORPUS_SCENARIO
    assert projection.name == "chatgpt:v7"
    assert projection.description == "Compiled inferred corpus scenario for chatgpt v7 across 1 corpus variant(s)."
    assert projection.origin == "compiled.inferred-corpus-scenario"
    assert projection.tags == ("inferred", "schema", "synthetic", "scenario")
    assert projection.source_payload["provider"] == "chatgpt"
    assert projection.source_payload["package_version"] == "v7"
    assert projection.source_payload["variant_count"] == 1


def test_inferred_corpus_spec_compiles_its_own_projection_entry() -> None:
    spec = CorpusSpec(
        provider="chatgpt",
        package_version="v7",
        element_kind="conversation_document",
        profile_family_ids=("cluster/a",),
        observed_sample_count=12,
        origin="inferred.schema",
        tags=("inferred", "schema", "synthetic"),
    )

    projection = spec.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.INFERRED_CORPUS
    assert projection.name == "chatgpt:v7:cluster/a"
    assert projection.description == "Inferred synthetic corpus spec for chatgpt conversation_document from 12 observed sample(s)."
    assert projection.origin == "inferred.schema"
    assert projection.tags == ("inferred", "schema", "synthetic")
    assert projection.source_payload["provider"] == "chatgpt"
    assert projection.source_payload["package_version"] == "v7"
