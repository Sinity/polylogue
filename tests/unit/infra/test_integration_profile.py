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
    """The default selection's origins and identities come from its witness recipes.

    Anti-vacuity: have ``IntegrationSelection.corpus_specs`` regenerate native
    ids instead of carrying ``witness.session_native_ids``, or drop a witness
    whose origin the profile requires, and the zip or origin set goes red.
    """
    selection = default_integration_selection()

    assert selection.profile.required_origins == ("chatgpt-export", "codex-session")
    assert {witness.origin for witness in selection.witnesses} == set(selection.profile.required_origins)
    assert all(not hasattr(witness, "expected") for witness in selection.witnesses)
    assert all(
        spec.session_native_ids == witness.session_native_ids
        for spec, witness in zip(selection.corpus_specs(), selection.witnesses, strict=True)
    )


def test_selection_digest_and_generated_shape_follow_recipe() -> None:
    """Profile and recipe changes move the selection digest and generated specs.

    Anti-vacuity: omit the profile digest or a witness ``recipe_digest`` from
    ``IntegrationSelection.digest``, or let ``corpus_spec`` ignore the recipe
    seed, and a changed selection keeps the old digest or corpus shape.
    """
    selection = default_integration_selection()
    changed_profile = replace(selection.profile, scale="smoke")
    changed_recipe = replace(selection.witnesses[0].recipe, seed=71)
    changed_witness = replace(selection.witnesses[0], recipe=changed_recipe)

    assert IntegrationSelection(changed_profile, selection.witnesses).digest != selection.digest
    changed = IntegrationSelection(selection.profile, (changed_witness, *selection.witnesses[1:]))
    assert changed.digest != selection.digest
    assert changed.corpus_specs()[0].seed == 71
    assert changed.corpus_specs()[0].session_native_ids != selection.corpus_specs()[0].session_native_ids


def test_selection_rejects_unrealizable_constraints_and_semantic_metadata() -> None:
    """Every constraint no witness set can realize is refused at construction.

    Anti-vacuity: delete any one check in ``IntegrationSelection.__post_init__``
    or ``IntegrationWitness.__post_init__`` (multiple witnesses, repeated
    recipe, registry mix, seed, provider token, semantic metadata) and its
    ``pytest.raises`` block stops raising.
    """
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
    with pytest.raises(ValueError, match="supported provider-wire token"):
        IntegrationWitness(CorpusSpec.for_provider("chatgpt-export", seed=42))


def test_selection_enforces_scale_against_guaranteed_message_population() -> None:
    """Scale is checked against the guaranteed minimum, not the generated maximum.

    Anti-vacuity: compute the population from ``messages_max`` or skip the
    ``_SCALE_MINIMUM_MESSAGES`` check, and the two-message-per-witness default
    is accepted as ``archive-shaped``.
    """
    selection = default_integration_selection()

    with pytest.raises(ValueError, match="archive-shaped.*16 messages"):
        IntegrationSelection(replace(selection.profile, scale="archive-shaped"), selection.witnesses)

    archive_shaped = IntegrationSelection(
        replace(selection.profile, scale="archive-shaped"),
        tuple(IntegrationWitness(replace(witness.recipe, count=4)) for witness in selection.witnesses),
    )

    assert sum(spec.count * spec.messages_min for spec in archive_shaped.corpus_specs()) == 16


def test_selection_preserves_law_owned_session_native_ids() -> None:
    """A recipe that pins native ids keeps them through ``corpus_spec``.

    Anti-vacuity: have ``IntegrationWitness.session_native_ids`` always derive
    ``integration-<digest>`` ids and the pinned id is replaced.
    """
    recipe = CorpusSpec.for_provider(
        "chatgpt",
        count=1,
        messages_min=2,
        messages_max=2,
        seed=42,
        session_native_ids=("law-owned-native-id",),
    )
    profile = IntegrationProfile(name="pinned", required_origins=("chatgpt-export",), scale="smoke")
    witness = IntegrationWitness(recipe)

    assert witness.corpus_spec(profile).session_native_ids == ("law-owned-native-id",)


def test_same_provider_witnesses_receive_distinct_session_identities(tmp_path: Path) -> None:
    """Two same-provider witnesses materialize as two archived sessions.

    Anti-vacuity: derive generated native ids from the index alone rather
    than the recipe digest, and both witnesses collapse onto one identity:
    the selection refuses to construct, and without that guard the archive's
    facts lose a session.
    """
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
    """The published archive holds exactly the selected witnesses' sessions.

    Anti-vacuity: bypass ``selection.corpus_specs()`` in
    ``build_integration_archive`` (for example, fall back to the default C03
    corpus) and the manifest key, facts and stored origins all diverge.
    """
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


def test_selection_rejects_two_witnesses_materializing_one_session_identity() -> None:
    """Anti-vacuity: drop the identity loop from IntegrationSelection.__post_init__
    and this construction succeeds, seeding an archive where one witness
    silently overwrites the other.

    The recipe-digest check does not catch this: the two recipes below differ
    (different seeds, so different digests) while pinning the same
    ``session_native_ids``, and the archive is keyed on identity, not recipe.
    """

    profile = default_integration_selection().profile
    pinned = ("integration-collision-000",)
    first = IntegrationWitness(CorpusSpec.for_provider("codex", count=1, seed=1, session_native_ids=pinned))
    second = IntegrationWitness(CorpusSpec.for_provider("codex", count=1, seed=2, session_native_ids=pinned))

    assert first.recipe_digest != second.recipe_digest
    assert first.origin == second.origin

    with pytest.raises(ValueError, match="materializes"):
        IntegrationSelection(profile, (first, second))
