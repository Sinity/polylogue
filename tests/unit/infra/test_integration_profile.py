"""Contracts for the disposable integration workload selection."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.scenarios import CorpusProfile, CorpusSpec
from tests.infra.integration_profile import (
    IntegrationInteraction,
    IntegrationProfile,
    IntegrationSelection,
    IntegrationWitness,
    build_integration_archive,
    default_integration_selection,
)
from tests.infra.workload_artifacts import CorpusArtifactManifest, seeded_archive_key


def test_selection_derives_constraints_from_witness_recipes() -> None:
    selection = default_integration_selection()

    assert selection.profile.required_origins == ("chatgpt-export", "codex-session")
    assert {witness.origin for witness in selection.witnesses} == set(selection.profile.required_origins)
    assert all(not hasattr(witness, "expected") for witness in selection.witnesses)
    assert all(
        spec.session_native_ids == witness.session_native_ids
        for spec, witness in zip(selection.corpus_specs(), selection.witnesses, strict=True)
    )


def test_selection_digest_and_generated_shape_follow_recipe() -> None:
    selection = default_integration_selection()
    changed_profile = replace(selection.profile, scale="archive-shaped")
    changed_recipe = replace(selection.witnesses[0].recipe, seed=71)
    changed_witness = replace(selection.witnesses[0], recipe=changed_recipe)

    assert IntegrationSelection(changed_profile, selection.witnesses).digest != selection.digest
    changed = IntegrationSelection(selection.profile, (changed_witness, *selection.witnesses[1:]))
    assert changed.digest != selection.digest
    assert changed.corpus_specs()[0].seed == 71
    assert changed.corpus_specs()[0].session_native_ids != selection.corpus_specs()[0].session_native_ids


def test_selection_rejects_unrealizable_constraints_and_semantic_metadata() -> None:
    recipe = CorpusSpec.for_provider("chatgpt", count=1, messages_min=2, messages_max=2, seed=42)
    coexistence = IntegrationProfile(name="bounded", required_origins=("chatgpt-export",))
    with pytest.raises(ValueError, match="multiple witnesses"):
        IntegrationSelection(coexistence, (IntegrationWitness(recipe),))
    with pytest.raises(ValueError, match="multiple witnesses"):
        IntegrationSelection(
            IntegrationProfile(
                name="lifecycle",
                required_origins=("chatgpt-export",),
                interactions=(IntegrationInteraction.LIFECYCLE_SCHEDULE,),
            ),
            (IntegrationWitness(recipe, interactions=(IntegrationInteraction.LIFECYCLE_SCHEDULE,)),),
        )
    with pytest.raises(ValueError, match="semantic case metadata"):
        IntegrationProfile(
            name="bounded", required_origins=("chatgpt-export",), required_source_classes=("expected_success",)
        )
    with pytest.raises(ValueError, match="semantic case metadata"):
        IntegrationWitness(replace(recipe, tags=("oracle_result",)))
    with pytest.raises(ValueError, match="deterministic seed"):
        IntegrationWitness(CorpusSpec.for_provider("chatgpt"))
    with pytest.raises(ValueError, match="cannot repeat a witness recipe"):
        IntegrationSelection(
            IntegrationProfile(
                name="duplicate",
                required_origins=("chatgpt-export",),
                interactions=(IntegrationInteraction.IDENTITY_COLLISION,),
            ),
            (
                IntegrationWitness(recipe, interactions=(IntegrationInteraction.IDENTITY_COLLISION,)),
                IntegrationWitness(recipe, interactions=(IntegrationInteraction.IDENTITY_COLLISION,)),
            ),
        )
    with pytest.raises(ValueError, match="multiple providers"):
        IntegrationSelection(
            IntegrationProfile(
                name="registry",
                required_origins=("chatgpt-export",),
                interactions=(IntegrationInteraction.REGISTRY_MIX,),
            ),
            (IntegrationWitness(recipe, interactions=(IntegrationInteraction.REGISTRY_MIX,)),),
        )


def test_same_provider_witnesses_receive_distinct_session_identities(tmp_path: Path) -> None:
    recipe = CorpusSpec.for_provider("codex", count=1, messages_min=2, messages_max=2, seed=42)
    profile = IntegrationProfile(
        name="identity",
        required_origins=("codex-session",),
        interactions=(IntegrationInteraction.IDENTITY_COLLISION,),
    )
    selection = IntegrationSelection(
        profile,
        (
            IntegrationWitness(recipe, interactions=(IntegrationInteraction.IDENTITY_COLLISION,)),
            IntegrationWitness(
                replace(recipe, profile=CorpusProfile(family_ids=("law",), profile_tokens=("second",))),
                interactions=(IntegrationInteraction.IDENTITY_COLLISION,),
            ),
        ),
    )

    session_ids = tuple(native_id for spec in selection.corpus_specs() for native_id in spec.session_native_ids)
    assert len(session_ids) == len(set(session_ids))
    artifact = build_integration_archive(selection, cache_root=tmp_path / "cache")
    assert {fact.expected_session_id for fact in artifact.facts} == {
        f"codex-session:{native_id}" for native_id in session_ids
    }


def test_archive_publishes_selected_heterogeneous_contents(tmp_path: Path) -> None:
    selection = default_integration_selection()
    artifact = build_integration_archive(selection, cache_root=tmp_path / "cache")

    assert isinstance(artifact.manifest, CorpusArtifactManifest)
    assert artifact.manifest.key == seeded_archive_key(selection.corpus_specs()).value
    assert {fact.expected_session_id for fact in artifact.facts} == {
        f"{witness.origin}:{native_id}" for witness in selection.witnesses for native_id in witness.session_native_ids
    }
    with sqlite3.connect(artifact.root / "index.db") as conn:
        origins = {row[0] for row in conn.execute("SELECT DISTINCT origin FROM sessions")}
    assert origins == set(selection.profile.required_origins)
    assert {witness.source_class for witness in selection.witnesses} == set(selection.profile.required_source_classes)
