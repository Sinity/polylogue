"""Typed synthetic corpus specifications shared across verification surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from .metadata import ScenarioMetadata
from .payloads import (
    PayloadDict,
    PayloadMap,
    merge_unique_string_tuples,
    payload_int,
    payload_mapping,
    payload_optional_string,
    payload_string_tuple,
)
from .projections import ScenarioProjectionSource, ScenarioProjectionSourceKind
from .specs import ScenarioSpec

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
    from polylogue.schemas.tooling_models import ClusterManifest, SchemaCluster


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _slugify_corpus_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "default"


class CorpusSourceKind(str, Enum):
    DEFAULT = "default"
    INFERRED = "inferred"


@dataclass(frozen=True, kw_only=True)
class CorpusProfile:
    """Observed or inferred corpus-profile metadata separate from generation controls."""

    family_ids: tuple[str, ...] = ()
    profile_tokens: tuple[str, ...] = ()
    artifact_kind: str | None = None
    anchor_kind: str | None = None
    observed_sample_count: int | None = None
    observed_artifact_count: int | None = None
    observed_confidence: float | None = None
    bundle_scope_count: int | None = None
    representative_paths: tuple[str, ...] = ()
    first_seen: str | None = None
    last_seen: str | None = None

    @property
    def primary_family_id(self) -> str | None:
        return self.family_ids[0] if self.family_ids else None

    @property
    def is_empty(self) -> bool:
        return (
            not self.family_ids
            and not self.profile_tokens
            and self.artifact_kind is None
            and self.anchor_kind is None
            and self.observed_sample_count is None
            and self.observed_artifact_count is None
            and self.observed_confidence is None
            and self.bundle_scope_count is None
            and not self.representative_paths
            and self.first_seen is None
            and self.last_seen is None
        )

    def scope_token(self, *, element_kind: str | None = None) -> str:
        return self.primary_family_id or element_kind or self.artifact_kind or "default"

    def target_kind(self, *, element_kind: str | None = None) -> str:
        return element_kind or self.artifact_kind or "default"

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CorpusProfile:
        return cls(
            family_ids=payload_string_tuple(payload.get("family_ids")),
            profile_tokens=payload_string_tuple(payload.get("profile_tokens")),
            artifact_kind=payload_optional_string(payload.get("artifact_kind")),
            anchor_kind=payload_optional_string(payload.get("anchor_kind")),
            observed_sample_count=payload_int(payload.get("observed_sample_count"), "observed_sample_count"),
            observed_artifact_count=payload_int(payload.get("observed_artifact_count"), "observed_artifact_count"),
            observed_confidence=_coerce_optional_float(payload.get("observed_confidence")),
            bundle_scope_count=payload_int(payload.get("bundle_scope_count"), "bundle_scope_count"),
            representative_paths=payload_string_tuple(payload.get("representative_paths")),
            first_seen=payload_optional_string(payload.get("first_seen")),
            last_seen=payload_optional_string(payload.get("last_seen")),
        )

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {}
        if self.family_ids:
            payload["family_ids"] = list(self.family_ids)
        if self.profile_tokens:
            payload["profile_tokens"] = list(self.profile_tokens)
        if self.artifact_kind is not None:
            payload["artifact_kind"] = self.artifact_kind
        if self.anchor_kind is not None:
            payload["anchor_kind"] = self.anchor_kind
        if self.observed_sample_count is not None:
            payload["observed_sample_count"] = self.observed_sample_count
        if self.observed_artifact_count is not None:
            payload["observed_artifact_count"] = self.observed_artifact_count
        if self.observed_confidence is not None:
            payload["observed_confidence"] = self.observed_confidence
        if self.bundle_scope_count is not None:
            payload["bundle_scope_count"] = self.bundle_scope_count
        if self.representative_paths:
            payload["representative_paths"] = list(self.representative_paths)
        if self.first_seen is not None:
            payload["first_seen"] = self.first_seen
        if self.last_seen is not None:
            payload["last_seen"] = self.last_seen
        return payload


@dataclass(frozen=True, kw_only=True)
class CorpusRequest:
    """Typed synthetic corpus selection and generation request."""

    providers: tuple[str, ...] | None = None
    source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT
    count: int = 5
    messages_min: int = 3
    messages_max: int = 15
    seed: int | None = None
    style: str = "default"
    package_version: str = "default"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("CorpusRequest.count must be >= 1")
        if self.messages_min < 1:
            raise ValueError("CorpusRequest.messages_min must be >= 1")
        if self.messages_max < self.messages_min:
            raise ValueError("CorpusRequest.messages_max must be >= messages_min")

    @property
    def source_kind(self) -> CorpusSourceKind:
        return CorpusSourceKind(self.source)

    def available_providers(self) -> tuple[str, ...] | None:
        if self.providers is not None:
            return self.providers
        if self.source_kind is not CorpusSourceKind.DEFAULT:
            return None
        from polylogue.schemas.synthetic import SyntheticCorpus

        return tuple(SyntheticCorpus.available_providers())

    def resolve_specs(
        self,
        *,
        origin: str | None = None,
        tags: tuple[str, ...] | None = None,
        registry: Any | None = None,
    ) -> tuple[CorpusSpec, ...]:
        return resolve_corpus_specs(
            providers=self.available_providers(),
            source=self.source_kind,
            count=self.count,
            messages_min=self.messages_min,
            messages_max=self.messages_max,
            seed=self.seed,
            style=self.style,
            package_version=self.package_version,
            origin=origin,
            tags=tags,
            registry=registry,
        )

    def resolve_scenarios(
        self,
        *,
        origin: str = "compiled.synthetic-corpus-scenario",
        tags: tuple[str, ...] = ("synthetic", "scenario"),
        registry: Any | None = None,
    ) -> tuple[CorpusScenario, ...]:
        return resolve_corpus_scenarios(
            providers=self.available_providers(),
            source=self.source_kind,
            count=self.count,
            messages_min=self.messages_min,
            messages_max=self.messages_max,
            seed=self.seed,
            style=self.style,
            package_version=self.package_version,
            origin=origin,
            tags=tags,
            registry=registry,
        )


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
    profile: CorpusProfile = field(default_factory=CorpusProfile)

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
        scope = self.profile.scope_token(element_kind=self.element_kind)
        parts: list[str] = []
        if self.package_version != "default":
            parts.append(self.package_version)
        parts.append(scope)
        return "-".join(_slugify_corpus_token(part) for part in parts)

    @property
    def projection_scope(self) -> str:
        return self.profile.scope_token(element_kind=self.element_kind)

    @property
    def projection_name(self) -> str:
        return f"{self.provider}:{self.package_version}:{self.projection_scope}"

    @property
    def projection_description(self) -> str:
        target = self.profile.target_kind(element_kind=self.element_kind)
        observed = (
            f" from {self.profile.observed_sample_count} observed sample(s)"
            if self.profile.observed_sample_count is not None
            else ""
        )
        window = ""
        if self.profile.first_seen is not None or self.profile.last_seen is not None:
            window = f" window={self.profile.first_seen or '?'} -> {self.profile.last_seen or '?'}"
        return f"Inferred synthetic corpus spec for {self.provider} {target}{observed}{window}."

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
        profile: CorpusProfile | None = None,
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
            profile=profile or CorpusProfile(),
            count=count,
            messages_min=messages_min,
            messages_max=messages_max,
            seed=seed,
            style=style,
            origin=origin,
            tags=tags,
        )

    @classmethod
    def from_payload(cls, payload: PayloadMap) -> CorpusSpec:
        metadata = ScenarioMetadata.from_payload(payload)
        provider = payload_optional_string(payload.get("provider"))
        if provider is None:
            raise ValueError("CorpusSpec payload must include provider")
        profile_payload = payload_mapping(payload.get("profile"))
        profile = CorpusProfile.from_payload(profile_payload) if profile_payload is not None else CorpusProfile()
        return cls(
            provider=provider,
            package_version=payload_optional_string(payload.get("package_version")) or "default",
            element_kind=payload_optional_string(payload.get("element_kind")),
            profile=profile,
            count=payload_int(payload.get("count"), "count") or 5,
            messages_min=payload_int(payload.get("messages_min"), "messages_min") or 3,
            messages_max=payload_int(payload.get("messages_max"), "messages_max") or 15,
            seed=payload_int(payload.get("seed"), "seed"),
            style=payload_optional_string(payload.get("style")) or "default",
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
        )

    def to_payload(self) -> PayloadDict:
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
        if not self.profile.is_empty:
            payload["profile"] = self.profile.to_payload()
        return payload

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.INFERRED_CORPUS

    def projection_source_payload(self) -> PayloadMap:
        return self.to_payload()


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
            raise ValueError(
                f"CorpusScenario.provider mismatch: expected only {self.provider!r}, got {sorted(providers)!r}"
            )
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
                    spec.profile.primary_family_id or "",
                    spec.element_kind or "",
                    spec.profile.artifact_kind or "",
                ),
            )
        )
        scenarios.append(
            CorpusScenario(
                provider=provider,
                package_version=package_version,
                corpus_specs=ordered_specs,
                origin=origin,
                path_targets=merge_unique_string_tuples(*(spec.path_targets for spec in ordered_specs)),
                artifact_targets=merge_unique_string_tuples(*(spec.artifact_targets for spec in ordered_specs)),
                operation_targets=merge_unique_string_tuples(*(spec.operation_targets for spec in ordered_specs)),
                tags=merge_unique_string_tuples(tags, *(spec.tags for spec in ordered_specs)),
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


def _merge_string_tuples(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return merge_unique_string_tuples(*groups, skip_empty=True)


def _catalog_package_for_version(
    catalog: SchemaPackageCatalog | None,
    package_version: str,
) -> SchemaVersionPackage | None:
    if catalog is None:
        return None
    return catalog.package(package_version)


def _package_element(
    package: SchemaVersionPackage | None,
    element_kind: str | None,
) -> SchemaElementManifest | None:
    if package is None:
        return None
    return package.element(element_kind)


def _profile_from_package(
    package: SchemaVersionPackage | None,
    *,
    element_kind: str | None,
) -> CorpusProfile:
    element = _package_element(package, element_kind)
    inferred_kind = (
        element_kind
        or (element.element_kind if element is not None else None)
        or (package.default_element_kind if package is not None else None)
    )
    return CorpusProfile(
        family_ids=tuple(element.profile_family_ids)
        if element is not None
        else tuple(package.profile_family_ids)
        if package
        else (),
        profile_tokens=tuple(element.profile_tokens) if element is not None else (),
        artifact_kind=inferred_kind,
        anchor_kind=package.anchor_kind if package is not None else None,
        observed_sample_count=element.sample_count
        if element is not None
        else package.sample_count
        if package is not None
        else None,
        observed_artifact_count=(
            (element.observed_artifact_count or element.artifact_count) if element is not None else None
        ),
        bundle_scope_count=element.bundle_scope_count
        if element is not None
        else package.bundle_scope_count
        if package
        else None,
        representative_paths=(
            tuple(element.representative_paths)
            if element is not None
            else tuple(package.representative_paths)
            if package is not None
            else ()
        ),
        first_seen=element.first_seen if element is not None else package.first_seen if package is not None else None,
        last_seen=element.last_seen if element is not None else package.last_seen if package is not None else None,
    )


def _merge_corpus_profiles(*profiles: CorpusProfile) -> CorpusProfile:
    merged = CorpusProfile()
    for profile in profiles:
        if profile.is_empty:
            continue
        merged = CorpusProfile(
            family_ids=_merge_string_tuples(merged.family_ids, profile.family_ids),
            profile_tokens=_merge_string_tuples(merged.profile_tokens, profile.profile_tokens),
            artifact_kind=merged.artifact_kind or profile.artifact_kind,
            anchor_kind=merged.anchor_kind or profile.anchor_kind,
            observed_sample_count=merged.observed_sample_count or profile.observed_sample_count,
            observed_artifact_count=merged.observed_artifact_count or profile.observed_artifact_count,
            observed_confidence=merged.observed_confidence or profile.observed_confidence,
            bundle_scope_count=merged.bundle_scope_count or profile.bundle_scope_count,
            representative_paths=_merge_string_tuples(merged.representative_paths, profile.representative_paths),
            first_seen=merged.first_seen or profile.first_seen,
            last_seen=merged.last_seen or profile.last_seen,
        )
    return merged


def _cluster_to_corpus_spec(
    cluster: SchemaCluster,
    *,
    provider: str,
    package_version: str,
    default_count: int,
    catalog: SchemaPackageCatalog | None = None,
) -> CorpusSpec:
    observed_artifact_kind = cluster.artifact_kind if cluster.artifact_kind != "unspecified" else None
    metadata = _inferred_schema_metadata()
    resolved_package_version = cluster.promoted_package_version or package_version
    package_profile = _profile_from_package(
        _catalog_package_for_version(catalog, resolved_package_version),
        element_kind=observed_artifact_kind,
    )
    return CorpusSpec(
        provider=provider,
        package_version=resolved_package_version,
        element_kind=observed_artifact_kind,
        profile=_merge_corpus_profiles(
            package_profile,
            CorpusProfile(
                family_ids=(cluster.cluster_id,),
                profile_tokens=tuple(cluster.profile_tokens),
                artifact_kind=observed_artifact_kind,
                observed_sample_count=cluster.sample_count,
                observed_confidence=cluster.confidence,
                bundle_scope_count=cluster.bundle_scope_count or None,
                representative_paths=tuple(cluster.representative_paths),
                first_seen=cluster.first_seen or None,
                last_seen=cluster.last_seen or None,
            ),
        ),
        count=max(1, min(cluster.sample_count, default_count)),
        messages_min=4,
        messages_max=16,
        style="default",
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
    catalog: SchemaPackageCatalog | None = None,
) -> tuple[CorpusSpec, ...]:
    metadata = _inferred_schema_metadata()
    if manifest is not None and manifest.clusters:
        return tuple(
            _cluster_to_corpus_spec(
                cluster,
                provider=provider,
                package_version=package_version,
                default_count=default_count,
                catalog=catalog,
            )
            for cluster in manifest.clusters
        )
    package_profile = _profile_from_package(
        _catalog_package_for_version(catalog, package_version),
        element_kind=None,
    )
    return (
        CorpusSpec(
            provider=provider,
            package_version=package_version,
            element_kind=package_profile.artifact_kind,
            profile=_merge_corpus_profiles(
                package_profile,
                CorpusProfile(observed_sample_count=sample_count or None),
            ),
            count=max(1, min(sample_count or 1, default_count)),
            messages_min=4,
            messages_max=16,
            style="default",
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
        ),
    )


__all__ = [
    "build_corpus_scenarios",
    "flatten_corpus_specs",
    "CorpusRequest",
    "CorpusProfile",
    "CorpusSourceKind",
    "CorpusScenario",
    "CorpusSpec",
    "build_default_corpus_specs",
    "build_inferred_corpus_specs",
    "resolve_corpus_scenarios",
    "resolve_corpus_specs",
]
