"""Stable schema-observation identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.schemas.observation_models import PROVIDERS, ProviderConfig
from polylogue.types import Provider


def resolve_provider_config(provider_name: str | Provider) -> ProviderConfig:
    canonical_provider = Provider.from_string(provider_name)
    if canonical_provider in PROVIDERS:
        return PROVIDERS[canonical_provider]

    config = next((c for c in PROVIDERS.values() if c.db_provider_name == canonical_provider), None)
    if config is not None:
        return config

    return ProviderConfig(
        name=canonical_provider,
        description=f"{canonical_provider} export format",
        db_provider_name=canonical_provider,
        sample_granularity="document",
    )


def fingerprint_hash(fingerprint: object) -> str:
    """Compute a stable short hash for a structural fingerprint."""
    raw = repr(fingerprint).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def schema_cluster_id(cluster_payload: object, artifact_kind: str) -> str:
    """Compute a stable cluster identifier for a schema unit."""
    from polylogue.schemas.shape_fingerprint import _structure_fingerprint

    return fingerprint_hash((artifact_kind, _structure_fingerprint(cluster_payload)))


def profile_cluster_id(artifact_kind: str, profile_tokens: tuple[str, ...]) -> str:
    """Compute a stable identifier for a profile-token cohort."""
    return fingerprint_hash((artifact_kind, tuple(sorted(profile_tokens))))


def profile_similarity(left: set[str], right: set[str]) -> float:
    """Similarity score tolerant to optional fields but resistant to drift."""
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return ((overlap / len(left)) + (overlap / len(right))) / 2.0


def derive_bundle_scope(
    provider_name: str | Provider,
    source_path: str | Path | None,
) -> str | None:
    """Return the provider-specific bundle scope for a raw artifact path."""
    if source_path is None:
        return None

    provider_token = Provider.from_string(provider_name)
    path = Path(str(source_path))
    normalized = str(path)

    if provider_token is Provider.CLAUDE_CODE:
        if "/subagents/" in normalized:
            return path.parent.parent.name or None
        if path.name in {"bridge-pointer.json", "sessions-index.json"}:
            return path.parent.name or None
        if path.name.startswith("agent-") and path.name.endswith(".meta.json"):
            return path.parent.parent.name or None
        if path.suffix == ".jsonl":
            return path.stem or None

    if provider_token is Provider.CODEX and path.suffix == ".jsonl":
        return path.stem or None

    return path.stem or path.name or None
