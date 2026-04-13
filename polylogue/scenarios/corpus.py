"""Typed synthetic corpus specifications shared across verification surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .metadata import ScenarioMetadata

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


@dataclass(frozen=True, kw_only=True)
class CorpusSpec(ScenarioMetadata):
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


def _cluster_to_corpus_spec(
    cluster: SchemaCluster,
    *,
    provider: str,
    package_version: str,
    default_count: int,
) -> CorpusSpec:
    observed_artifact_kind = cluster.artifact_kind if cluster.artifact_kind != "unspecified" else None
    return CorpusSpec(
        provider=provider,
        package_version=package_version,
        element_kind=observed_artifact_kind,
        count=max(1, min(cluster.sample_count, default_count)),
        messages_min=4,
        messages_max=16,
        style="default",
        origin="inferred.schema",
        profile_family_ids=(cluster.cluster_id,),
        artifact_kind=observed_artifact_kind,
        observed_sample_count=cluster.sample_count,
        observed_confidence=cluster.confidence,
        representative_paths=tuple(cluster.representative_paths),
        tags=("inferred", "schema", "synthetic"),
    )


def build_inferred_corpus_specs(
    *,
    provider: str,
    package_version: str = "default",
    manifest: ClusterManifest | None = None,
    sample_count: int = 0,
    default_count: int = 5,
) -> tuple[CorpusSpec, ...]:
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
            origin="inferred.schema",
            observed_sample_count=sample_count or None,
            tags=("inferred", "schema", "synthetic"),
        ),
    )


__all__ = [
    "CorpusSpec",
    "build_default_corpus_specs",
    "build_inferred_corpus_specs",
]
