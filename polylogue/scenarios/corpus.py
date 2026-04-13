"""Typed synthetic corpus specifications shared across verification surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from .metadata import ScenarioMetadata
from .projections import ScenarioProjectionSource, ScenarioProjectionSourceKind
from .specs import ScenarioSpec

if TYPE_CHECKING:
    from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster


def _coerce_optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def _slugify_corpus_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "default"


class CorpusSourceKind(str, Enum):
    DEFAULT = "default"
    INFERRED = "inferred"


@dataclass(frozen=True, kw_only=True)
class CorpusSpec(ScenarioProjectionSource, ScenarioMetadata):
    """Authored or inferred synthetic corpus configuration."""

    provider: str
    package_version: str = "default"
    element_kind: str | None = None
    count: int = 5
    messages_min: int = 3
    messages_max: int = 15
    seed: int | None = None
    style: str = "default"
    profile_family_ids: tuple[str, ...] = ()
    artifact_kind: str | None = None
    observed_sample_count: int | None = None
    observed_confidence: float | None = None
    representative_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("CorpusSpec.provider must be non-empty")
        if self.count < 1:
            raise ValueError("CorpusSpec.count must be >= 1")
        if self.messages_min < 1:
            raise ValueError("CorpusSpec.messages_min must be >= 1")
        if self.messages_max < self.messages_min:
            raise ValueError("CorpusSpec.messages_max must be >= messages_min")

    @property
    def messages_per_conversation(self) -> range:
        return range(self.messages_min, self.messages_max + 1)

    @property
    def scope_label(self) -> str:
        scope = self.profile_family_ids[0] if self.profile_family_ids else (self.element_kind or self.artifact_kind or "default")
        parts: list[str] = []
        if self.package_version != "default":
            parts.append(self.package_version)
        parts.append(scope)
        return "-".join(_slugify_corpus_token(part) for part in parts)

    @property
    def projection_scope(self) -> str:
        return self.profile_family_ids[0] if self.profile_family_ids else (self.element_kind or self.artifact_kind or "default")

    @property
    def projection_name(self) -> str:
        return f"{self.provider}:{self.package_version}:{self.projection_scope}"

    @property
    def projection_description(self) -> str:
        target = self.element_kind or self.artifact_kind or "default"
        observed = (
            f" from {self.observed_sample_count} observed sample(s)"
            if self.observed_sample_count is not None
            else ""
        )
        return f"Inferred synthetic corpus spec for {self.provider} {target}{observed}."

    def with_generation_overrides(
        self,
        *,
        count: int | None = None,
        messages_min: int | None = None,
        messages_max: int | None = None,
        seed: int | None = None,
        style: str | None = None,
        origin: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> CorpusSpec:
        return replace(
            self,
            count=self.count if count is None else count,
            messages_min=self.messages_min if messages_min is None else messages_min,
            messages_max=self.messages_max if messages_max is None else messages_max,
            seed=self.seed if seed is None else seed,
            style=self.style if style is None else style,
            origin=self.origin if origin is None else origin,
            tags=self.tags if tags is None else tags,
        )

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        package_version: str = "default",
        element_kind: str | None = None,
        count: int = 5,
        messages_min: int = 3,
        messages_max: int = 15,
        seed: int | None = None,
        style: str = "default",
        origin: str = "authored",
        tags: tuple[str, ...] = (),
    ) -> CorpusSpec:
        return cls(
            provider=provider,
            package_version=package_version,
            element_kind=element_kind,
            count=count,
            messages_min=messages_min,
            messages_max=messages_max,
            seed=seed,
            style=style,
            origin=origin,
            tags=tags,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CorpusSpec:
        metadata = ScenarioMetadata.from_payload(payload)
        provider = _coerce_optional_string(payload.get("provider"))
        if provider is None:
            raise ValueError("CorpusSpec payload must include provider")
        return cls(
            provider=provider,
            package_version=_coerce_optional_string(payload.get("package_version")) or "default",
            element_kind=_coerce_optional_string(payload.get("element_kind")),
            count=_coerce_optional_int(payload.get("count")) or 5,
            messages_min=_coerce_optional_int(payload.get("messages_min")) or 3,
            messages_max=_coerce_optional_int(payload.get("messages_max")) or 15,
            seed=_coerce_optional_int(payload.get("seed")),
            style=_coerce_optional_string(payload.get("style")) or "default",
            profile_family_ids=_coerce_string_tuple(payload.get("profile_family_ids")),
            artifact_kind=_coerce_optional_string(payload.get("artifact_kind")),
            observed_sample_count=_coerce_optional_int(payload.get("observed_sample_count")),
            observed_confidence=_coerce_optional_float(payload.get("observed_confidence")),
            representative_paths=_coerce_string_tuple(payload.get("representative_paths")),
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload.update(
            {
                "provider": self.provider,
                "package_version": self.package_version,
                "count": self.count,
                "messages_min": self.messages_min,
                "messages_max": self.messages_max,
                "style": self.style,
            }
        )
        if self.element_kind is not None:
            payload["element_kind"] = self.element_kind
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.profile_family_ids:
            payload["profile_family_ids"] = list(self.profile_family_ids)
        if self.artifact_kind is not None:
            payload["artifact_kind"] = self.artifact_kind
        if self.observed_sample_count is not None:
            payload["observed_sample_count"] = self.observed_sample_count
        if self.observed_confidence is not None:
            payload["observed_confidence"] = self.observed_confidence
        if self.representative_paths:
            payload["representative_paths"] = list(self.representative_paths)
        return payload

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.INFERRED_CORPUS

    def projection_source_payload(self) -> Mapping[str, object]:
        return self.to_payload()


def _merge_unique_string_tuples(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return tuple(merged)


def _inferred_schema_metadata() -> ScenarioMetadata:
    return ScenarioMetadata(
        origin="inferred.schema",
        path_targets=("inferred-corpus-compilation-loop",),
        artifact_targets=(
            "schema_packages",
            "schema_cluster_manifests",
            "inferred_corpus_specs",
            "inferred_corpus_scenarios",
        ),
        operation_targets=("compile-inferred-corpus-specs", "compile-inferred-corpus-scenarios"),
        tags=("inferred", "schema", "synthetic"),
    )


@dataclass(frozen=True, kw_only=True)
class CorpusScenario(ScenarioSpec):
    """Named scenario compiled from one or more corpus specs."""

    provider: str
    package_version: str = "default"
    corpus_specs: tuple[CorpusSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("CorpusScenario.provider must be non-empty")
        if not self.corpus_specs:
            raise ValueError("CorpusScenario.corpus_specs must be non-empty")
        providers = {spec.provider for spec in self.corpus_specs}
        if providers != {self.provider}:
            raise ValueError(f"CorpusScenario.provider mismatch: expected only {self.provider!r}, got {sorted(providers)!r}")
        versions = {spec.package_version for spec in self.corpus_specs}
        if versions != {self.package_version}:
            raise ValueError(
                f"CorpusScenario.package_version mismatch: expected only {self.package_version!r}, got {sorted(versions)!r}"
            )

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.INFERRED_CORPUS_SCENARIO

    @property
    def projection_name(self) -> str:
        return f"{self.provider}:{self.package_version}"

    @property
    def projection_description(self) -> str:
        return (
            f"Compiled inferred corpus scenario for {self.provider} {self.package_version} "
            f"across {len(self.corpus_specs)} corpus variant(s)."
        )

    @property
    def target_labels(self) -> tuple[str, ...]:
        return tuple(spec.projection_scope for spec in self.corpus_specs)

    def projection_source_payload(self) -> Mapping[str, object]:
        payload = self.scenario_payload()
        payload.update(
            {
                "provider": self.provider,
                "package_version": self.package_version,
                "variant_count": len(self.corpus_specs),
                "target_labels": list(self.target_labels),
            }
        )
        return payload


def build_corpus_scenarios(
    corpus_specs: Iterable[CorpusSpec],
    *,
    origin: str = "compiled.corpus-scenario",
    tags: tuple[str, ...] = ("synthetic", "scenario"),
) -> tuple[CorpusScenario, ...]:
    grouped: dict[tuple[str, str], list[CorpusSpec]] = {}
    for spec in corpus_specs:
        grouped.setdefault((spec.provider, spec.package_version), []).append(spec)
    scenarios: list[CorpusScenario] = []
    for (provider, package_version), specs in sorted(grouped.items()):
        ordered_specs = tuple(
            sorted(
                specs,
                key=lambda spec: (
                    spec.profile_family_ids[0] if spec.profile_family_ids else "",
                    spec.element_kind or "",
                    spec.artifact_kind or "",
                ),
            )
        )
        scenarios.append(
            CorpusScenario(
                provider=provider,
                package_version=package_version,
                corpus_specs=ordered_specs,
                origin=origin,
                path_targets=_merge_unique_string_tuples(*(spec.path_targets for spec in ordered_specs)),
                artifact_targets=_merge_unique_string_tuples(*(spec.artifact_targets for spec in ordered_specs)),
                operation_targets=_merge_unique_string_tuples(*(spec.operation_targets for spec in ordered_specs)),
                tags=_merge_unique_string_tuples(tags, *(spec.tags for spec in ordered_specs)),
            )
        )
    return tuple(scenarios)


def flatten_corpus_specs(
    corpus_scenarios: Iterable[CorpusScenario],
) -> tuple[CorpusSpec, ...]:
    return tuple(spec for scenario in corpus_scenarios for spec in scenario.corpus_specs)


def build_default_corpus_specs(
    *,
    providers: Iterable[str],
    count: int = 5,
    messages_min: int = 3,
    messages_max: int = 15,
    seed: int | None = None,
    style: str = "default",
    package_version: str = "default",
    origin: str = "generated.synthetic-defaults",
    tags: tuple[str, ...] = ("synthetic", "generated"),
) -> tuple[CorpusSpec, ...]:
    return tuple(
        CorpusSpec.for_provider(
            provider,
            package_version=package_version,
            count=count,
            messages_min=messages_min,
            messages_max=messages_max,
            seed=seed,
            style=style,
            origin=origin,
            tags=tags,
        )
        for provider in providers
    )


def resolve_corpus_specs(
    *,
    providers: Iterable[str] | None = None,
    source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    count: int = 5,
    messages_min: int = 3,
    messages_max: int = 15,
    seed: int | None = None,
    style: str = "default",
    package_version: str = "default",
    origin: str | None = None,
    tags: tuple[str, ...] | None = None,
    registry: Any | None = None,
) -> tuple[CorpusSpec, ...]:
    source_kind = CorpusSourceKind(source)
    provider_names = tuple(providers) if providers is not None else None
    if source_kind is CorpusSourceKind.DEFAULT:
        return build_default_corpus_specs(
            providers=provider_names or (),
            count=count,
            messages_min=messages_min,
            messages_max=messages_max,
            seed=seed,
            style=style,
            package_version=package_version,
            origin=origin or "generated.synthetic-defaults",
            tags=tags or ("synthetic", "generated"),
        )

    from polylogue.schemas.operator_inference import list_inferred_corpus_specs

    inferred_specs = list_inferred_corpus_specs(registry=registry)
    if provider_names is not None:
        allowed = set(provider_names)
        inferred_specs = tuple(spec for spec in inferred_specs if spec.provider in allowed)
    return tuple(
        spec.with_generation_overrides(
            count=count,
            messages_min=messages_min,
            messages_max=messages_max,
            seed=seed,
            style=style,
            origin=origin or "generated.synthetic-inferred",
            tags=tags or ("synthetic", "generated", "inferred"),
        )
        for spec in inferred_specs
    )


def resolve_corpus_scenarios(
    *,
    providers: Iterable[str] | None = None,
    source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    count: int = 5,
    messages_min: int = 3,
    messages_max: int = 15,
    seed: int | None = None,
    style: str = "default",
    package_version: str = "default",
    origin: str = "compiled.synthetic-corpus-scenario",
    tags: tuple[str, ...] = ("synthetic", "scenario"),
    registry: Any | None = None,
) -> tuple[CorpusScenario, ...]:
    specs = resolve_corpus_specs(
        providers=providers,
        source=source,
        count=count,
        messages_min=messages_min,
        messages_max=messages_max,
        seed=seed,
        style=style,
        package_version=package_version,
        registry=registry,
    )
    return build_corpus_scenarios(specs, origin=origin, tags=tags)


def _cluster_to_corpus_spec(
    cluster: SchemaCluster,
    *,
    provider: str,
    package_version: str,
    default_count: int,
) -> CorpusSpec:
    observed_artifact_kind = cluster.artifact_kind if cluster.artifact_kind != "unspecified" else None
    metadata = _inferred_schema_metadata()
    return CorpusSpec(
        provider=provider,
        package_version=package_version,
        element_kind=observed_artifact_kind,
        count=max(1, min(cluster.sample_count, default_count)),
        messages_min=4,
        messages_max=16,
        style="default",
        profile_family_ids=(cluster.cluster_id,),
        artifact_kind=observed_artifact_kind,
        observed_sample_count=cluster.sample_count,
        observed_confidence=cluster.confidence,
        representative_paths=tuple(cluster.representative_paths),
        origin=metadata.origin,
        path_targets=metadata.path_targets,
        artifact_targets=metadata.artifact_targets,
        operation_targets=metadata.operation_targets,
        tags=metadata.tags,
    )


def build_inferred_corpus_specs(
    *,
    provider: str,
    package_version: str = "default",
    manifest: ClusterManifest | None = None,
    sample_count: int = 0,
    default_count: int = 5,
) -> tuple[CorpusSpec, ...]:
    metadata = _inferred_schema_metadata()
    if manifest is not None and manifest.clusters:
        return tuple(
            _cluster_to_corpus_spec(
                cluster,
                provider=provider,
                package_version=cluster.promoted_package_version or package_version,
                default_count=default_count,
            )
            for cluster in manifest.clusters
        )
    return (
        CorpusSpec(
            provider=provider,
            package_version=package_version,
            count=max(1, min(sample_count or 1, default_count)),
            messages_min=4,
            messages_max=16,
            style="default",
            origin=metadata.origin,
            observed_sample_count=sample_count or None,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
        ),
    )


__all__ = [
    "build_corpus_scenarios",
    "flatten_corpus_specs",
    "CorpusSourceKind",
    "CorpusScenario",
    "CorpusSpec",
    "build_default_corpus_specs",
    "build_inferred_corpus_specs",
    "resolve_corpus_scenarios",
    "resolve_corpus_specs",
]
