"""Emission helpers for persisted generated schema bundles."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.generation_models import _ProviderBundle
from polylogue.schemas.registry import SchemaRegistry


def persist_generated_provider_bundle(output_dir: Path, provider: str, bundle: _ProviderBundle) -> None:
    """Persist a generated provider bundle into the registry storage."""
    result = bundle.result
    if not result.success or bundle.manifest is None or bundle.catalog is None:
        return

    registry = SchemaRegistry(storage_root=output_dir)
    registry.replace_provider_packages(provider, bundle.catalog, bundle.package_schemas)
    registry.save_cluster_manifest(bundle.manifest)

    for legacy_name in (f"{provider}.schema.json.gz", f"{provider}.schema.json"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


__all__ = ["persist_generated_provider_bundle"]
