"""Disposable integration workload selection over composable witness recipes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from polylogue.core.sources import origin_from_provider
from polylogue.scenarios import CorpusProfile, CorpusSpec
from tests.infra.workload_artifacts import SeededArchiveArtifact, build_seeded_archive

_SEMANTIC_METADATA_PREFIXES = ("expected_", "oracle_", "case_", "pathology_")


def _reject_semantic_metadata(value: object) -> None:
    if isinstance(value, str):
        if not value or value.startswith(_SEMANTIC_METADATA_PREFIXES):
            raise ValueError("integration witness cannot carry semantic case metadata")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_semantic_metadata(key)
            _reject_semantic_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_semantic_metadata(item)


class IntegrationInteraction(StrEnum):
    """Cross-witness behavior that requires a composed archive."""

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
        _reject_semantic_metadata(
            (self.name, self.required_origins, self.required_source_classes, self.temporal_operations)
        )
        if not self.required_origins:
            raise ValueError("integration profile requires a name and at least one origin")
        if self.scale not in {"smoke", "representative", "archive-shaped", "stress"}:
            raise ValueError("integration profile scale must be an operational profile")
        if any(not isinstance(interaction, IntegrationInteraction) for interaction in self.interactions):
            raise ValueError("integration profile interactions must use declared interaction values")

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
    """One law-owned corpus recipe and the interactions it can support."""

    recipe: CorpusSpec
    source_class: str = "provider-shaped"
    interactions: tuple[IntegrationInteraction, ...] = (
        IntegrationInteraction.COEXISTENCE,
        IntegrationInteraction.REGISTRY_MIX,
    )
    temporal_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _reject_semantic_metadata((self.recipe.to_payload(), self.source_class, self.temporal_operations))
        if any(not isinstance(interaction, IntegrationInteraction) for interaction in self.interactions):
            raise ValueError("integration witness interactions must use declared interaction values")

    @property
    def recipe_digest(self) -> str:
        payload = json.dumps(self.recipe.to_payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def origin(self) -> str:
        return origin_from_provider(self.recipe.provider).value

    @property
    def session_native_ids(self) -> tuple[str, ...]:
        return tuple(f"integration-{self.recipe_digest[:16]}-{index:03d}" for index in range(self.recipe.count))

    def corpus_spec(self, profile: IntegrationProfile) -> CorpusSpec:
        return replace(
            self.recipe,
            session_native_ids=self.session_native_ids,
            tags=(*self.recipe.tags, "synthetic", "integration", profile.name, profile.scale, self.source_class),
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
        if not self.witnesses:
            raise ValueError("integration selection requires at least one witness")
        origins = {witness.origin for witness in self.witnesses}
        missing = set(self.profile.required_origins) - origins
        if missing:
            raise ValueError(f"integration selection lacks required origins: {sorted(missing)}")
        source_classes = {witness.source_class for witness in self.witnesses}
        missing_classes = set(self.profile.required_source_classes) - source_classes
        if missing_classes:
            raise ValueError(f"integration selection lacks required source classes: {sorted(missing_classes)}")
        available = {interaction for witness in self.witnesses for interaction in witness.interactions}
        missing_interactions = set(self.profile.interactions) - available
        if missing_interactions:
            raise ValueError(f"integration selection lacks interaction support: {sorted(missing_interactions)}")
        cross_witness_interactions = {
            IntegrationInteraction.COEXISTENCE,
            IntegrationInteraction.CROSS_CASE_INTERFERENCE,
            IntegrationInteraction.CANDIDATE_DIFFER,
            IntegrationInteraction.LIFECYCLE_SCHEDULE,
        }
        if cross_witness_interactions & set(self.profile.interactions) and len(self.witnesses) < 2:
            raise ValueError("declared interaction requires multiple witnesses")
        if (
            IntegrationInteraction.REGISTRY_MIX in self.profile.interactions
            and len({w.recipe.provider for w in self.witnesses}) < 2
        ):
            raise ValueError("registry mix requires multiple providers")
        if IntegrationInteraction.IDENTITY_COLLISION in self.profile.interactions:
            providers = {w.recipe.provider for w in self.witnesses}
            if not any(sum(w.recipe.provider == provider for w in self.witnesses) > 1 for provider in providers):
                raise ValueError("identity collision requires multiple witnesses for one provider")
        temporal_operations = {operation for witness in self.witnesses for operation in witness.temporal_operations}
        missing_operations = set(self.profile.temporal_operations) - temporal_operations
        if missing_operations:
            raise ValueError(f"integration selection lacks temporal operation support: {sorted(missing_operations)}")
        if self.profile.temporal_operations and IntegrationInteraction.LIFECYCLE_SCHEDULE not in available:
            raise ValueError("temporal operations require lifecycle schedule support")

    @property
    def digest(self) -> str:
        payload = {
            "profile": self.profile.digest,
            "witnesses": tuple(
                {
                    "recipe_digest": witness.recipe_digest,
                    "source_class": witness.source_class,
                    "interactions": tuple(item.value for item in witness.interactions),
                    "temporal_operations": witness.temporal_operations,
                }
                for witness in self.witnesses
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def corpus_specs(self) -> tuple[CorpusSpec, ...]:
        return tuple(witness.corpus_spec(self.profile) for witness in self.witnesses)


DEFAULT_INTEGRATION_PROFILE = IntegrationProfile(
    name="heterogeneous-representative",
    required_origins=("chatgpt-export", "codex-session"),
    required_source_classes=("provider-shaped",),
    interactions=(IntegrationInteraction.COEXISTENCE, IntegrationInteraction.REGISTRY_MIX),
)

_CHATGPT_DIALOGUE_RECIPE = CorpusSpec.for_provider(
    "chatgpt", count=1, messages_min=2, messages_max=4, seed=42, origin="generated.integration"
)
_CODEX_DIALOGUE_RECIPE = CorpusSpec.for_provider(
    "codex", count=1, messages_min=2, messages_max=4, seed=42, origin="generated.integration"
)


def default_integration_selection() -> IntegrationSelection:
    """Return the smallest heterogeneous selection used by integration tests."""
    return IntegrationSelection(
        DEFAULT_INTEGRATION_PROFILE,
        (IntegrationWitness(_CHATGPT_DIALOGUE_RECIPE), IntegrationWitness(_CODEX_DIALOGUE_RECIPE)),
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
