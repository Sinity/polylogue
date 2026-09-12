"""Operational workload declarations used to construct seeded artifacts.

This module owns provider-shaped workload recipes only. Artifact publication,
cache reuse, leases, clones, and collection live in ``workload_artifacts``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from enum import Enum

from polylogue.scenarios import CorpusProfile, CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus

SEMANTIC_METADATA_PREFIXES = ("expected_", "oracle_", "pathology_", "case_")
_KNOWN_PROVIDERS = frozenset(SyntheticCorpus.available_providers())
_PROVIDER_COMPONENT = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def reject_semantic_metadata(value: object, *, location: str) -> None:
    """Refuse oracle-shaped fields in operational declarations and receipts."""
    if isinstance(value, str) and value.startswith(SEMANTIC_METADATA_PREFIXES):
        raise ValueError(f"{location} cannot carry semantic metadata: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.startswith(SEMANTIC_METADATA_PREFIXES):
                raise ValueError(f"{location} cannot carry semantic metadata: {key}")
            reject_semantic_metadata(child, location=location)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_semantic_metadata(child, location=location)


def validate_workload_provider(provider: object) -> str:
    """Require a registered provider safe for provider-shaped fixtures."""
    if not isinstance(provider, str) or not _PROVIDER_COMPONENT.fullmatch(provider):
        raise ValueError("corpus provider must be one safe path component")
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(f"unknown corpus provider: {provider!r}")
    return provider


def c03_semantic_corpus_spec() -> CorpusSpec:
    """Smallest named semantic canary with a pinned selective Codex session."""
    count = 64
    native_ids = ("c03-target", *(f"c03-irrelevant-{index:03d}" for index in range(count - 1)))
    return CorpusSpec.for_provider(
        "codex",
        count=count,
        messages_min=4,
        messages_max=4,
        seed=71,
        style="tool-heavy",
        session_native_ids=native_ids,
        origin="generated.test-workload-c03",
        tags=("synthetic", "test", "workload-c03"),
    )


def schema_coverage_corpus_specs() -> tuple[CorpusSpec, ...]:
    """Named all-provider schema workload; no caller chooses ad-hoc shape."""
    return tuple(
        CorpusSpec.for_provider(
            provider,
            count=2,
            messages_min=4,
            messages_max=4,
            seed=42,
            origin="generated.test-schema-coverage",
            tags=("synthetic", "test", "schema-coverage"),
        )
        for provider in SyntheticCorpus.available_providers()
    )


@dataclass(frozen=True)
class WorkloadSessionShape:
    """One provider-native population within an operational workload."""

    provider: str
    count: int
    messages_min: int
    messages_max: int
    seed_offset: int = 0
    style: str = "tool-heavy"

    def __post_init__(self) -> None:
        validate_workload_provider(self.provider)
        reject_semantic_metadata(asdict(self), location="workload session shape")
        if self.count < 1:
            raise ValueError("workload session shape requires a positive session count")
        if self.messages_min < 1 or self.messages_max < self.messages_min:
            raise ValueError("workload session shape has invalid message bounds")
        if self.seed_offset < 0:
            raise ValueError("workload session shape seed offset must be non-negative")


@dataclass(frozen=True)
class WorkloadProfile:
    """Operational identity and provider-native spec constructor for workloads."""

    name: str
    purpose: str
    seed: int
    family_ids: tuple[str, ...]
    profile_tokens: tuple[str, ...]
    origin: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.purpose:
            raise ValueError("workload profile requires a name and purpose")
        if not self.family_ids or not self.profile_tokens:
            raise ValueError("workload profile requires corpus identity")
        reject_semantic_metadata(asdict(self), location="workload profile")

    @property
    def identity_tokens(self) -> tuple[str, ...]:
        """Return the complete operational identity carried by generated specs."""
        return tuple(
            dict.fromkeys(
                (
                    *self.profile_tokens,
                    f"workload-name:{self.name}",
                    f"workload-purpose:{self.purpose}",
                )
            )
        )

    def corpus_specs(self, shapes: tuple[WorkloadSessionShape, ...]) -> tuple[CorpusSpec, ...]:
        if not shapes:
            raise ValueError("workload profile requires provider-native session shapes")
        corpus_profile = CorpusProfile(
            family_ids=self.family_ids,
            profile_tokens=self.identity_tokens,
            artifact_kind="archive",
        )
        return tuple(
            CorpusSpec.for_provider(
                shape.provider,
                count=shape.count,
                messages_min=shape.messages_min,
                messages_max=shape.messages_max,
                seed=self.seed + shape.seed_offset,
                style=shape.style,
                profile=corpus_profile,
                origin=self.origin,
                tags=self.tags,
            )
            for shape in shapes
        )


@dataclass(frozen=True)
class NamedWorkloadProfile:
    """A deterministic workload used by shared test fixtures."""

    workload: WorkloadProfile
    provider_session_counts: tuple[tuple[str, int], ...]
    messages_min: int = 4
    messages_max: int = 11

    @property
    def name(self) -> str:
        return self.workload.name

    @property
    def purpose(self) -> str:
        return self.workload.purpose

    @property
    def seed(self) -> int:
        return self.workload.seed

    def __post_init__(self) -> None:
        if not self.provider_session_counts or any(count < 1 for _provider, count in self.provider_session_counts):
            raise ValueError("named workload profile requires positive provider session counts")
        providers = tuple(validate_workload_provider(provider) for provider, _count in self.provider_session_counts)
        if len(set(providers)) != len(providers):
            raise ValueError("named workload profile cannot repeat a provider")
        if self.messages_min < 1 or self.messages_max < self.messages_min:
            raise ValueError("named workload profile has invalid message bounds")
        reject_semantic_metadata(asdict(self), location="named workload profile")

    def corpus_specs(self) -> tuple[CorpusSpec, ...]:
        return self.workload.corpus_specs(
            tuple(
                WorkloadSessionShape(provider, count, self.messages_min, self.messages_max)
                for provider, count in self.provider_session_counts
            )
        )


def _named_workload(name: str, purpose: str, *, seed: int = 42) -> WorkloadProfile:
    return WorkloadProfile(
        name=name,
        purpose=purpose,
        seed=seed,
        family_ids=("test-workload",),
        profile_tokens=(name, purpose, "provider-native"),
        origin=f"generated.test-workload-{name}",
        tags=("synthetic", "test", name, purpose),
    )


NAMED_WORKLOAD_PROFILES = (
    NamedWorkloadProfile(_named_workload("schema-small", "schema-scaling"), (("chatgpt", 10),)),
    NamedWorkloadProfile(_named_workload("schema-medium", "schema-scaling"), (("chatgpt", 50),)),
    NamedWorkloadProfile(_named_workload("cli-chatgpt", "cli-read"), (("chatgpt", 2),)),
    NamedWorkloadProfile(_named_workload("cli-mixed", "cli-read"), (("chatgpt", 2), ("claude-code", 2))),
    NamedWorkloadProfile(_named_workload("completion", "completion", seed=1271), (("chatgpt", 3), ("claude-ai", 3))),
)


def named_workload_profile(name: str) -> NamedWorkloadProfile:
    """Resolve one finite named test workload."""
    try:
        return next(profile for profile in NAMED_WORKLOAD_PROFILES if profile.name == name)
    except StopIteration as exc:
        raise ValueError(f"unknown named seeded archive workload {name!r}") from exc


def named_corpus_specs(name: str) -> tuple[CorpusSpec, ...]:
    """Resolve one shared workload declaration for test consumers."""
    return named_workload_profile(name).corpus_specs()


class BenchmarkWorkloadTier(str, Enum):
    """Named benchmark projections backed by shared archive artifacts."""

    SMOKE = "smoke"
    REPRESENTATIVE = "representative"
    ARCHIVE_SCALE = "archive-scale"
    STRESS = "stress"


@dataclass(frozen=True)
class BenchmarkWorkloadProfile:
    """A deterministic mixed-origin benchmark projection."""

    tier: BenchmarkWorkloadTier
    workload: WorkloadProfile
    target_messages: int
    provider_session_counts: tuple[tuple[str, int], ...]
    messages_per_session: int = 10

    def __post_init__(self) -> None:
        if self.target_messages < 1 or self.messages_per_session < 1:
            raise ValueError("benchmark workload dimensions must be positive")
        if not self.provider_session_counts or any(count < 1 for _provider, count in self.provider_session_counts):
            raise ValueError("benchmark workload requires every configured provider to have sessions")
        providers = tuple(validate_workload_provider(provider) for provider, _count in self.provider_session_counts)
        if len(set(providers)) != len(providers):
            raise ValueError("benchmark workload cannot repeat a provider")
        if (
            sum(count for _provider, count in self.provider_session_counts) * self.messages_per_session
            != self.target_messages
        ):
            raise ValueError("benchmark workload session composition must exactly produce target_messages")
        reject_semantic_metadata(asdict(self), location="benchmark workload profile")

    @property
    def purpose(self) -> str:
        return self.workload.purpose


_BENCHMARK_PROVIDER_MIX = (
    ("claude-code", 80),
    ("codex", 15),
    ("chatgpt", 2),
    ("claude-ai", 1),
    ("gemini", 2),
)


def _benchmark_workload(tier: BenchmarkWorkloadTier, purpose: str) -> WorkloadProfile:
    return WorkloadProfile(
        name=tier.value,
        purpose=purpose,
        seed=42,
        family_ids=("benchmark-archive",),
        profile_tokens=(tier.value, "mixed-origin", "provider-native"),
        origin=f"generated.benchmark-{tier.value}",
        tags=("synthetic", "benchmark", tier.value),
    )


BENCHMARK_WORKLOAD_PROFILES = (
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.SMOKE,
        _benchmark_workload(BenchmarkWorkloadTier.SMOKE, "fast-benchmark"),
        1_000,
        _BENCHMARK_PROVIDER_MIX,
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.REPRESENTATIVE,
        _benchmark_workload(BenchmarkWorkloadTier.REPRESENTATIVE, "broad-benchmark"),
        5_000,
        tuple((provider, count * 5) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.ARCHIVE_SCALE,
        _benchmark_workload(BenchmarkWorkloadTier.ARCHIVE_SCALE, "archive-scale-benchmark"),
        10_000,
        tuple((provider, count * 10) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.STRESS,
        _benchmark_workload(BenchmarkWorkloadTier.STRESS, "stress-benchmark"),
        50_000,
        tuple((provider, count * 50) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
)


def benchmark_workload_profile(tier: BenchmarkWorkloadTier | str) -> BenchmarkWorkloadProfile:
    """Resolve one named benchmark workload without round-count labels."""
    resolved = BenchmarkWorkloadTier(tier)
    return next(profile for profile in BENCHMARK_WORKLOAD_PROFILES if profile.tier is resolved)


def benchmark_corpus_specs(tier: BenchmarkWorkloadTier | str, *, seed: int = 42) -> tuple[CorpusSpec, ...]:
    """Build provider-native corpus specs for a named benchmark tier."""
    profile = benchmark_workload_profile(tier)
    session_shapes: list[WorkloadSessionShape] = []
    for provider, count in profile.provider_session_counts:
        provider_shapes: tuple[tuple[int, int], ...]
        if provider == "claude-code":
            multiplier, remainder = divmod(count, 80)
            if remainder:
                raise ValueError("benchmark Claude Code composition must retain the 80-session provider mix")
            provider_shapes = ((50 * multiplier, 2), (25 * multiplier, 8), (5 * multiplier, 100))
        else:
            provider_shapes = ((count, profile.messages_per_session),)
        for shape_count, messages_per_session in provider_shapes:
            session_shapes.append(
                WorkloadSessionShape(
                    provider, shape_count, messages_per_session, messages_per_session, len(session_shapes)
                )
            )
    return replace(profile.workload, seed=seed).corpus_specs(tuple(session_shapes))


__all__ = [
    "BENCHMARK_WORKLOAD_PROFILES",
    "NAMED_WORKLOAD_PROFILES",
    "BenchmarkWorkloadProfile",
    "BenchmarkWorkloadTier",
    "NamedWorkloadProfile",
    "SEMANTIC_METADATA_PREFIXES",
    "WorkloadProfile",
    "WorkloadSessionShape",
    "benchmark_corpus_specs",
    "benchmark_workload_profile",
    "c03_semantic_corpus_spec",
    "named_corpus_specs",
    "named_workload_profile",
    "reject_semantic_metadata",
    "schema_coverage_corpus_specs",
    "validate_workload_provider",
]
