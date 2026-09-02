"""Disposable integration workload selection over law-owned witnesses.

This module owns selection shape only. Witnesses remain small, law-owned
recipes and the artifact publisher remains the sole archive construction and
cache authority. No expected result is accepted here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.scenarios import CorpusProfile, CorpusSpec
from tests.infra.workload_artifacts import SeededArchiveArtifact, build_seeded_archive


class IntegrationInteraction(StrEnum):
    """Cross-law behavior that requires a composed archive."""

    COEXISTENCE = "coexistence"
    IDENTITY_COLLISION = "identity-collision"
    REGISTRY_MIX = "registry-mix"
    CROSS_CASE_INTERFERENCE = "cross-case-interference"
    CANDIDATE_DIFFER = "candidate-differ"
    LIFECYCLE_SCHEDULE = "lifecycle-schedule"


@dataclass(frozen=True, slots=True)
class IntegrationProfile:
    """Operational constraints for one disposable heterogeneous workload."""

    name: str
    required_origins: tuple[str, ...]
    required_source_classes: tuple[str, ...] = ()
    interactions: tuple[IntegrationInteraction, ...] = (IntegrationInteraction.COEXISTENCE,)
    temporal_operations: tuple[str, ...] = ()
    scale: str = "representative"

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith(("expected_", "oracle_", "case_", "pathology_")):
            raise ValueError("integration profile cannot carry semantic case metadata")
        if not self.required_origins:
            raise ValueError("integration profile requires a name and at least one origin")
        if self.scale not in {"smoke", "representative", "archive-shaped", "stress"}:
            raise ValueError("integration profile scale must be an operational profile")
        for value in (*self.required_origins, *self.required_source_classes, *self.temporal_operations):
            if not value or value.startswith(("expected_", "oracle_", "case_", "pathology_")):
                raise ValueError("integration profile cannot carry semantic case metadata")

    @property
    def digest(self) -> str:
        payload = {
            "name": self.name,
            "required_origins": self.required_origins,
            "required_source_classes": self.required_source_classes,
            "interactions": tuple(item.value for item in self.interactions),
            "temporal_operations": self.temporal_operations,
            "scale": self.scale,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class IntegrationWitness:
    """A law-owned recipe reference and its provider-shaped workload shape."""

    recipe_digest: str
    provider: str
    origin: str
    source_class: str = "provider-shaped"
    count: int = 1
    messages_min: int = 2
    messages_max: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.recipe_digest or not self.origin or self.count < 1:
            raise ValueError("integration witness requires identity and positive count")
        if self.messages_min < 1 or self.messages_max < self.messages_min:
            raise ValueError("integration witness has invalid message bounds")
        if self.recipe_digest.startswith(("expected_", "oracle_", "case_", "pathology_")):
            raise ValueError("integration witness identity cannot encode semantic metadata")

    def corpus_spec(self, profile: IntegrationProfile) -> CorpusSpec:
        return CorpusSpec.for_provider(
            self.provider,
            count=self.count,
            messages_min=self.messages_min,
            messages_max=self.messages_max,
            seed=self.seed,
            origin=self.origin,
            tags=("synthetic", "integration", profile.name, profile.scale, self.source_class),
            profile=CorpusProfile(
                family_ids=("integration-workload",),
                profile_tokens=(
                    "integration-profile:" + profile.digest,
                    "witness-recipe:" + self.recipe_digest,
                    "source-class:" + self.source_class,
                ),
                artifact_kind="archive",
            ),
        )


@dataclass(frozen=True, slots=True)
class IntegrationSelection:
    profile: IntegrationProfile
    witnesses: tuple[IntegrationWitness, ...]

    def __post_init__(self) -> None:
        origins = {witness.origin for witness in self.witnesses}
        missing = set(self.profile.required_origins) - origins
        if missing:
            raise ValueError(f"integration selection lacks required origins: {sorted(missing)}")
        source_classes = {witness.source_class for witness in self.witnesses}
        missing_classes = set(self.profile.required_source_classes) - source_classes
        if missing_classes:
            raise ValueError(f"integration selection lacks required source classes: {sorted(missing_classes)}")

    @property
    def digest(self) -> str:
        payload = {
            "profile": self.profile.digest,
            "witnesses": tuple(
                {
                    "recipe_digest": witness.recipe_digest,
                    "provider": witness.provider,
                    "origin": witness.origin,
                    "source_class": witness.source_class,
                    "count": witness.count,
                    "messages_min": witness.messages_min,
                    "messages_max": witness.messages_max,
                    "seed": witness.seed,
                }
                for witness in self.witnesses
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def corpus_specs(self) -> tuple[CorpusSpec, ...]:
        return tuple(witness.corpus_spec(self.profile) for witness in self.witnesses)


DEFAULT_INTEGRATION_PROFILE = IntegrationProfile(
    name="heterogeneous-representative",
    required_origins=("generated.integration-chatgpt", "generated.integration-codex"),
    required_source_classes=("provider-shaped",),
    interactions=(IntegrationInteraction.COEXISTENCE, IntegrationInteraction.REGISTRY_MIX),
)


def default_integration_selection() -> IntegrationSelection:
    """Return the smallest heterogeneous selection used by integration tests."""
    return IntegrationSelection(
        DEFAULT_INTEGRATION_PROFILE,
        (
            IntegrationWitness("law-chatgpt-dialogue-v1", "chatgpt", "generated.integration-chatgpt"),
            IntegrationWitness("law-codex-dialogue-v1", "codex", "generated.integration-codex"),
        ),
    )


def build_integration_archive(
    selection: IntegrationSelection | None = None,
    *,
    cache_root: Path | None = None,
) -> SeededArchiveArtifact:
    """Materialize a selected workload through the canonical artifact route."""
    resolved = selection or default_integration_selection()
    return build_seeded_archive(resolved.corpus_specs(), cache_root=cache_root)


__all__ = [
    "DEFAULT_INTEGRATION_PROFILE",
    "IntegrationInteraction",
    "IntegrationProfile",
    "IntegrationSelection",
    "IntegrationWitness",
    "build_integration_archive",
    "default_integration_selection",
]
