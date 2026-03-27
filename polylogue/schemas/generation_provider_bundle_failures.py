"""Failure/result helpers for provider-bundle assembly."""

from __future__ import annotations

from polylogue.schemas.generation_models import GenerationResult, _ProviderBundle


def build_provider_error_bundle(
    provider: str,
    *,
    error: str,
    sample_count: int = 0,
    cluster_count: int = 0,
    artifact_counts: dict[str, int] | None = None,
    manifest=None,
) -> _ProviderBundle:
    """Build a provider bundle carrying only an error result."""
    return _ProviderBundle(
        result=GenerationResult(
            provider=provider,
            schema=None,
            sample_count=sample_count,
            error=error,
            cluster_count=cluster_count,
            artifact_counts=artifact_counts or {},
        ),
        manifest=manifest,
    )


__all__ = ["build_provider_error_bundle"]
