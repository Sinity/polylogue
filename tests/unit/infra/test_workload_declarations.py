"""Tests for operational workload declarations and their semantic boundary."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus
from tests.infra.workload_artifacts import _manifest_file_entries, build_seeded_archive, seeded_archive_key
from tests.infra.workload_declarations import (
    BENCHMARK_WORKLOAD_PROFILES,
    NAMED_WORKLOAD_PROFILES,
    BenchmarkWorkloadTier,
    WorkloadProfile,
    WorkloadSessionShape,
    benchmark_corpus_specs,
    benchmark_workload_profile,
    named_corpus_specs,
    named_workload_profile,
    raw_sample_corpus_specs,
)


def test_benchmark_profiles_are_semantic_mixed_origin_exact_message_projections() -> None:
    """Benchmark targets are named workload contracts, not direct index seeds."""
    assert tuple(profile.tier for profile in BENCHMARK_WORKLOAD_PROFILES) == (
        BenchmarkWorkloadTier.SMOKE,
        BenchmarkWorkloadTier.REPRESENTATIVE,
        BenchmarkWorkloadTier.ARCHIVE_SCALE,
        BenchmarkWorkloadTier.STRESS,
    )
    assert tuple(profile.target_messages for profile in BENCHMARK_WORKLOAD_PROFILES) == (1_000, 5_000, 10_000, 50_000)
    for profile in BENCHMARK_WORKLOAD_PROFILES:
        specs = benchmark_corpus_specs(profile.tier)
        assert sum(spec.count * spec.messages_min for spec in specs) == profile.target_messages
        assert {spec.provider for spec in specs} == {"chatgpt", "claude-ai", "claude-code", "codex", "gemini"}
        assert {spec.messages_min for spec in specs} == {2, 8, profile.messages_per_session, 100}
        assert {spec.messages_max for spec in specs} == {2, 8, profile.messages_per_session, 100}
        assert {spec.profile.primary_family_id for spec in specs} == {"benchmark-archive"}
        assert {profile.tier.value for spec in specs if profile.tier.value in spec.tags} == {profile.tier.value}


def test_benchmark_profile_selection_is_deterministic_and_named() -> None:
    first = benchmark_corpus_specs(BenchmarkWorkloadTier.REPRESENTATIVE, seed=91)
    second = benchmark_corpus_specs("representative", seed=91)

    assert first == second
    assert benchmark_workload_profile("representative").target_messages == 5_000
    with pytest.raises(ValueError, match="not-a-tier"):
        BenchmarkWorkloadTier("not-a-tier")


def test_named_workload_profiles_are_semantic_and_build_deterministic_provider_specs() -> None:
    assert {profile.name for profile in NAMED_WORKLOAD_PROFILES} == {
        "schema-small",
        "schema-medium",
        "cli-chatgpt",
        "cli-mixed",
        "completion",
    }

    profile = named_workload_profile("completion")
    first = named_corpus_specs(profile.name)
    second = profile.corpus_specs()

    assert first == second
    assert profile.purpose == "completion"
    assert {spec.provider for spec in first} == {"chatgpt", "claude-ai"}
    assert {spec.seed for spec in first} == {1271}
    assert {spec.origin for spec in first} == {"generated.test-workload-completion"}
    assert {spec.profile.primary_family_id for spec in first} == {"test-workload"}
    assert {"completion", "provider-native"}.issubset(set(first[0].profile.profile_tokens))
    with pytest.raises(ValueError, match="unknown named seeded archive workload"):
        named_workload_profile("unknown")


def test_raw_sample_specs_are_deterministic_and_preserve_hash_fixture_shape() -> None:
    first = raw_sample_corpus_specs()
    second = raw_sample_corpus_specs()

    assert first == second
    assert {spec.provider for spec in first} == set(SyntheticCorpus.available_providers())
    assert {spec.count for spec in first} == {5}
    assert {(spec.messages_min, spec.messages_max) for spec in first} == {(3, 15)}
    assert {spec.seed for spec in first} == {42}
    assert {spec.origin for spec in first} == {"generated.test-raw-samples"}
    assert {spec.tags for spec in first} == {("synthetic", "test", "raw-samples")}


def test_profile_name_and_purpose_are_part_of_artifact_identity() -> None:
    profile = named_workload_profile("cli-chatgpt")
    shapes = tuple(
        WorkloadSessionShape(provider, count, profile.messages_min, profile.messages_max)
        for provider, count in profile.provider_session_counts
    )
    baseline = seeded_archive_key(profile.workload.corpus_specs(shapes))

    renamed = dataclasses.replace(profile.workload, name="cli-chatgpt-renamed")
    repurposed = dataclasses.replace(profile.workload, purpose="cli-write")

    assert seeded_archive_key(renamed.corpus_specs(shapes)).value != baseline.value
    assert seeded_archive_key(repurposed.corpus_specs(shapes)).value != baseline.value


def test_workload_identity_rejects_semantic_oracle_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="semantic metadata"):
        WorkloadProfile(
            name="invalid",
            purpose="fixture-shape",
            seed=1,
            family_ids=("test",),
            profile_tokens=("expected_sessions",),
            origin="generated.test-invalid",
            tags=("synthetic",),
        )

    with pytest.raises(ValueError, match="semantic metadata"):
        WorkloadProfile(
            name="invalid-family",
            purpose="fixture-shape",
            seed=1,
            family_ids=("expected_sessions",),
            profile_tokens=("fixture-shape",),
            origin="generated.test-invalid",
            tags=("synthetic",),
        )

    with pytest.raises(ValueError, match="semantic metadata"):
        WorkloadProfile(
            name="invalid-family",
            purpose="fixture-shape",
            seed=1,
            family_ids=("expected_sessions",),
            profile_tokens=("provider-native",),
            origin="generated.test-invalid",
            tags=("synthetic",),
        )

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="semantic metadata"):
        dataclasses.replace(
            artifact.manifest,
            receipt={**artifact.manifest.receipt, "expected_sessions": 64},
        )

    file_entry = dict(artifact.manifest.files[0])
    file_entry["expected_sessions"] = 64
    with pytest.raises(ValueError, match="semantic metadata"):
        _manifest_file_entries((file_entry,))

    with pytest.raises(ValueError, match="semantic metadata"):
        dataclasses.replace(
            artifact.manifest,
            files=({"path": "index.db", "size": 0, "sha256": "0" * 64, "expected_sessions": 64},),
        )

    for field, value in {
        "name": "expected_sessions",
        "purpose": "expected_sessions",
        "origin": "expected_sessions",
        "tags": ("expected_sessions",),
    }.items():
        with pytest.raises(ValueError, match="semantic metadata"):
            dataclasses.replace(
                cast(
                    Any,
                    WorkloadProfile(
                        name="valid-name",
                        purpose="fixture-shape",
                        seed=1,
                        family_ids=("test",),
                        profile_tokens=("fixture-shape",),
                        origin="generated.test-invalid",
                        tags=("synthetic",),
                    ),
                ),
                **{field: value},
            )


def test_workload_profile_wrappers_reject_semantic_fields_and_unknown_providers() -> None:
    shape = WorkloadSessionShape("chatgpt", 1, 2, 2)
    with pytest.raises(ValueError, match="unknown corpus provider"):
        dataclasses.replace(shape, provider="not-a-provider")
    with pytest.raises(ValueError, match="semantic metadata"):
        dataclasses.replace(shape, style="expected_sessions")

    named = named_workload_profile("cli-chatgpt")
    with pytest.raises(ValueError, match="named workload profile cannot repeat"):
        dataclasses.replace(named, provider_session_counts=(("chatgpt", 1), ("chatgpt", 2)))
    with pytest.raises(ValueError, match="semantic metadata"):
        dataclasses.replace(named, workload=dataclasses.replace(named.workload, purpose="expected_semantics"))

    benchmark = benchmark_workload_profile(BenchmarkWorkloadTier.SMOKE)
    with pytest.raises(ValueError, match="benchmark workload cannot repeat"):
        dataclasses.replace(benchmark, provider_session_counts=(("chatgpt", 1), ("chatgpt", 1)))
    with pytest.raises(ValueError, match="unknown corpus provider"):
        dataclasses.replace(benchmark, provider_session_counts=(("not-a-provider", 1),))


def test_named_and_benchmark_catalogs_share_one_semantic_spec_contract() -> None:
    catalog_profiles = (*NAMED_WORKLOAD_PROFILES, *BENCHMARK_WORKLOAD_PROFILES)

    assert all(isinstance(profile.workload, WorkloadProfile) for profile in catalog_profiles)
    assert all(profile.workload.name and profile.workload.purpose for profile in catalog_profiles)
    assert all("provider-native" in profile.workload.profile_tokens for profile in catalog_profiles)

    named = named_workload_profile("cli-mixed")
    benchmark = benchmark_workload_profile(BenchmarkWorkloadTier.SMOKE)
    assert {spec.origin for spec in named.corpus_specs()} == {named.workload.origin}
    assert {spec.origin for spec in benchmark_corpus_specs(benchmark.tier)} == {benchmark.workload.origin}
