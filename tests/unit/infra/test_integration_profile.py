"""Contracts for the disposable integration workload selection."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.infra.integration_profile import (
    IntegrationProfile,
    IntegrationSelection,
    IntegrationWitness,
    build_integration_archive,
    default_integration_selection,
)
from tests.infra.workload_artifacts import CorpusArtifactManifest


def test_selection_contains_constraints_and_no_semantic_oracle() -> None:
    selection = default_integration_selection()
    assert selection.profile.required_origins == (
        "generated.integration-chatgpt",
        "generated.integration-codex",
    )
    assert all(not hasattr(witness, "expected") for witness in selection.witnesses)
    assert all(
        any(token.startswith("integration-profile:") for token in spec.profile.profile_tokens)
        for spec in selection.corpus_specs()
    )


def test_selection_digest_binds_profile_and_witness_recipe() -> None:
    selection = default_integration_selection()
    changed_profile = replace(selection.profile, scale="archive-shaped")
    changed_witness = replace(selection.witnesses[0], recipe_digest="law-chatgpt-dialogue-v2")
    assert IntegrationSelection(changed_profile, selection.witnesses).digest != selection.digest
    assert (
        IntegrationSelection(selection.profile, (changed_witness, *selection.witnesses[1:])).digest != selection.digest
    )


def test_selection_rejects_missing_required_origin_and_semantic_metadata() -> None:
    profile = IntegrationProfile(name="bounded", required_origins=("generated.missing",))
    with pytest.raises(ValueError, match="lacks required origins"):
        IntegrationSelection(profile, (IntegrationWitness("law-v1", "chatgpt", "generated.present"),))
    with pytest.raises(ValueError, match="semantic case metadata"):
        IntegrationProfile(name="expected_profile", required_origins=("generated.present",))


def test_archive_is_published_as_authenticated_manifest(tmp_path: Path) -> None:
    artifact = build_integration_archive(cache_root=tmp_path / "cache")
    assert isinstance(artifact.manifest, CorpusArtifactManifest)
    assert artifact.manifest.key.startswith("seeded-archive:sha256:")
    assert artifact.manifest.files
