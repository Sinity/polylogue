"""Schema generation from provider data.

This module infers JSON schemas from real data samples, which can be used for:
1. Validation of new imports (detect malformed data)
2. Drift detection (warn when provider format changes)
3. Property-based test generation via hypothesis-jsonschema

Can be used as:
- Module: `from polylogue.schemas.schema_inference import generate_provider_schema`
- CLI: `polylogue schema generate --provider chatgpt`

Implementation is split across:
- `schemas/sampling.py` — ProviderConfig, PROVIDERS, sample loading
- `schemas/schema_generation.py` — schema manipulation, annotation, generation
- `schemas/field_stats.py` — FieldStats, _collect_field_stats
- `schemas/privacy.py` — _is_safe_enum_value, _is_content_field
"""

from __future__ import annotations

from pathlib import Path

# Re-export the full public API so callers don't need to know the split.
from polylogue.schemas.field_stats import (
    FieldStats,
    UUID_PATTERN,
    _collect_field_stats,
    is_dynamic_key,
)
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
)
from polylogue.schemas.sampling import (
    PROVIDERS,
    ProviderConfig,
    _iter_samples_from_db,
    _iter_samples_from_sessions,
    _resolve_provider_config,
    _sample_provider_where_clause,
    get_sample_count_from_db,
    load_samples_from_db,
    load_samples_from_sessions,
)
from polylogue.schemas.schema_generation import (
    GenerationResult,
    _annotate_schema,
    _merge_schemas,
    _remove_nested_required,
    _structure_fingerprint,
    collapse_dynamic_keys,
    generate_all_schemas,
    generate_provider_schema,
    generate_schema_from_samples,
)


# =============================================================================
# CLI Entry Point
# =============================================================================


def cli_main(args: list[str] | None = None) -> int:
    """CLI entry point for schema generation.

    Returns exit code (0 = success).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate JSON schemas from polylogue data"
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()) + ["all"],
        default="all",
        help="Provider to generate schema for",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("polylogue/schemas/providers"),
        help="Output directory for schemas",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to polylogue database (default: XDG data home)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for fast/debug generation (default: full dataset)",
    )

    parsed = parser.parse_args(args)

    providers = None if parsed.provider == "all" else [parsed.provider]
    results = generate_all_schemas(
        output_dir=parsed.output_dir,
        db_path=parsed.db_path,
        providers=providers,
        max_samples=parsed.max_samples,
    )

    success = []
    failed = []
    for r in results:
        if r.success:
            print(f"✓ {r.provider}: {r.sample_count:,} samples")
            success.append(r.provider)
        else:
            print(f"✗ {r.provider}: {r.error}")
            failed.append(r.provider)

    print(f"\nGenerated {len(success)} schemas")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())


__all__ = [
    # From field_stats
    "FieldStats",
    "UUID_PATTERN",
    "_collect_field_stats",
    "is_dynamic_key",
    # From privacy
    "_is_content_field",
    "_is_safe_enum_value",
    # From sampling
    "PROVIDERS",
    "ProviderConfig",
    "_iter_samples_from_db",
    "_iter_samples_from_sessions",
    "_resolve_provider_config",
    "_sample_provider_where_clause",
    "get_sample_count_from_db",
    "load_samples_from_db",
    "load_samples_from_sessions",
    # From schema_generation
    "GenerationResult",
    "_annotate_schema",
    "_merge_schemas",
    "_remove_nested_required",
    "_structure_fingerprint",
    "collapse_dynamic_keys",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
    # CLI
    "cli_main",
]
