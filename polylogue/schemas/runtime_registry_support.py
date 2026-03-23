"""Shared runtime schema registry support types and helpers."""

from __future__ import annotations

from pathlib import Path

from polylogue.lib.provider_identity import canonical_schema_provider as _canonical_schema_provider
from polylogue.lib.provider_identity import normalize_provider_token
from polylogue.types import Provider

SCHEMA_DIR = Path(__file__).parent / "providers"
SchemaProvider = Provider | str


def canonical_schema_provider(provider: str | Provider) -> SchemaProvider:
    normalized = normalize_provider_token(str(provider))
    if not normalized:
        return Provider.UNKNOWN

    canonical = _canonical_schema_provider(normalized, default="")
    if canonical:
        provider_token = Provider.from_string(canonical)
        if provider_token is not Provider.UNKNOWN:
            return provider_token
    return normalized


__all__ = ["SCHEMA_DIR", "SchemaProvider", "canonical_schema_provider"]
