"""Schema generation from provider data.

This module infers JSON schemas from real data samples, which can be used for:
1. Validation of new imports (detect malformed data)
2. Drift detection (warn when provider format changes)
3. Property-based test generation via hypothesis-jsonschema

Can be used as:
- Module: `from polylogue.schemas.generator import generate_provider_schema`
- CLI: `polylogue schema generate --provider chatgpt`
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from genson import SchemaBuilder
    GENSON_AVAILABLE = True
except ImportError:
    GENSON_AVAILABLE = False


# Default database path - use same location as main storage backend
import contextlib

from polylogue.storage.backends.sqlite import default_db_path

DEFAULT_DB_PATH = default_db_path()

# UUID pattern for detecting dynamic keys
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass
class ProviderConfig:
    """Configuration for a provider's schema generation."""

    name: str
    description: str
    db_provider_name: str | None = None  # Provider name in polylogue DB
    session_dir: Path | None = None  # For JSONL session-based providers
    max_sessions: int | None = None


# Provider configurations
PROVIDERS: dict[str, ProviderConfig] = {
    "chatgpt": ProviderConfig(
        name="chatgpt",
        description="ChatGPT message format",
        db_provider_name="chatgpt",
    ),
    "claude-code": ProviderConfig(
        name="claude-code",
        description="Claude Code message format",
        db_provider_name="claude-code",
    ),
    "claude-ai": ProviderConfig(
        name="claude-ai",
        description="Claude AI web message format",
        db_provider_name="claude",  # DB uses "claude"
    ),
    "gemini": ProviderConfig(
        name="gemini",
        description="Gemini AI Studio message format",
        db_provider_name="gemini",
    ),
    "codex": ProviderConfig(
        name="codex",
        description="OpenAI Codex CLI session format",
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
    ),
}


@dataclass
class GenerationResult:
    """Result of schema generation."""

    provider: str
    schema: dict[str, Any] | None
    sample_count: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.schema is not None and self.error is None


# =============================================================================
# Schema Utilities
# =============================================================================


def is_dynamic_key(key: str) -> bool:
    """Check if a key looks like a dynamic identifier (UUID, hash, etc)."""
    if UUID_PATTERN.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))


def collapse_dynamic_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse dynamic key properties into additionalProperties.

    Handles objects like ChatGPT's `mapping` where keys are UUIDs.
    """
    if not isinstance(schema, dict):
        return schema

    if "properties" in schema:
        props = schema["properties"]
        static_props = {}
        dynamic_schemas = []

        for key, value in props.items():
            collapsed_value = collapse_dynamic_keys(value)
            if is_dynamic_key(key):
                dynamic_schemas.append(collapsed_value)
            else:
                static_props[key] = collapsed_value

        if dynamic_schemas:
            schema["properties"] = static_props
            merged = _merge_schemas(dynamic_schemas)
            schema["additionalProperties"] = merged
            if "required" in schema:
                schema["required"] = [r for r in schema["required"] if r in static_props]
        else:
            schema["properties"] = static_props

    if "items" in schema:
        schema["items"] = collapse_dynamic_keys(schema["items"])

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [collapse_dynamic_keys(s) for s in schema[keyword]]

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = collapse_dynamic_keys(schema["additionalProperties"])

    return schema


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple schemas into one using genson."""
    if not GENSON_AVAILABLE:
        return schemas[0] if schemas else {}
    builder = SchemaBuilder()
    for schema in schemas:
        builder.add_schema(schema)
    return dict(builder.to_schema())


# =============================================================================
# Sample Loaders
# =============================================================================


def load_samples_from_db(
    provider_name: str,
    db_path: Path = DEFAULT_DB_PATH,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from polylogue database.

    Args:
        provider_name: Provider name in database
        db_path: Path to polylogue.db
        max_samples: Optional limit (None = all)

    Returns:
        List of raw message dicts
    """
    if not db_path.exists():
        return []

    samples = []
    conn = sqlite3.connect(db_path)

    try:
        limit_clause = f"LIMIT {max_samples}" if max_samples else ""

        # Load from raw_conversations table (new approach)
        rows = conn.execute(f"""
            SELECT raw_content, provider_name
            FROM raw_conversations
            WHERE provider_name = ?
            {limit_clause}
        """, (provider_name,)).fetchall()

        for row in rows:
            try:
                content = row[0]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")

                provider = row[1]

                # JSONL providers: parse first line
                if provider in ("claude-code", "codex", "gemini"):
                    for line in content.strip().split("\n"):
                        if line.strip():
                            samples.append(json.loads(line))
                            break
                else:
                    # JSON providers: parse whole content
                    samples.append(json.loads(content))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    finally:
        conn.close()

    return samples


def load_samples_from_sessions(
    session_dir: Path,
    max_sessions: int | None = None,
) -> list[dict[str, Any]]:
    """Load samples from JSONL session files.

    Args:
        session_dir: Directory containing .jsonl session files
        max_sessions: Optional limit on sessions to process

    Returns:
        List of record dicts from all sessions
    """
    if not session_dir.exists():
        return []

    samples = []
    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if max_sessions and len(jsonl_files) > max_sessions:
        step = len(jsonl_files) // max_sessions
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            samples.append(json.loads(line))
        except OSError:
            pass

    return samples


def get_sample_count_from_db(
    provider_name: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Get total message count for a provider in database."""
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ? AND m.provider_meta IS NOT NULL
        """, (provider_name,)).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


# =============================================================================
# Schema Generation
# =============================================================================


def _remove_nested_required(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Remove 'required' arrays from nested objects.

    Genson marks fields as required if they appear in all samples, but this
    is too strict for real data where fields can be optional. We keep top-level
    required (depth=0) but remove from nested objects.

    Args:
        schema: JSON schema dict
        depth: Current nesting depth (0 = root)

    Returns:
        Modified schema with nested required arrays removed
    """
    if not isinstance(schema, dict):
        return schema

    # Remove 'required' from nested objects (depth > 0)
    if depth > 0 and "required" in schema:
        del schema["required"]

    # Recurse into properties
    if "properties" in schema:
        for key, prop in schema["properties"].items():
            schema["properties"][key] = _remove_nested_required(prop, depth + 1)

    # Recurse into items (arrays)
    if "items" in schema:
        schema["items"] = _remove_nested_required(schema["items"], depth + 1)

    # Handle anyOf/oneOf/allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_remove_nested_required(s, depth + 1) for s in schema[key]]

    return schema


def generate_schema_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate JSON schema from samples using genson.

    Args:
        samples: List of data dicts to infer schema from

    Returns:
        JSON Schema dict
    """
    if not GENSON_AVAILABLE:
        raise ImportError("genson is required for schema generation. Install with: pip install genson")

    if not samples:
        return {"type": "object", "description": "No samples available"}

    builder = SchemaBuilder()
    for sample in samples:
        builder.add_object(sample)

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)

    # Remove required arrays from nested objects - genson is too strict
    schema = _remove_nested_required(schema)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    return schema


def generate_provider_schema(
    provider: str,
    db_path: Path = DEFAULT_DB_PATH,
    max_samples: int | None = None,
) -> GenerationResult:
    """Generate schema for a provider.

    Args:
        provider: Provider name (chatgpt, claude-code, etc.)
        db_path: Path to polylogue database
        max_samples: Optional sample limit

    Returns:
        GenerationResult with schema or error
    """
    if not GENSON_AVAILABLE:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error="genson not installed",
        )

    config = PROVIDERS.get(provider)
    if not config:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error=f"Unknown provider: {provider}. Known: {list(PROVIDERS.keys())}",
        )

    # Load samples
    samples: list[dict[str, Any]] = []

    if config.db_provider_name:
        samples = load_samples_from_db(
            config.db_provider_name,
            db_path=db_path,
            max_samples=max_samples,
        )
    elif config.session_dir:
        samples = load_samples_from_sessions(
            config.session_dir,
            max_sessions=config.max_sessions,
        )

    if not samples:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error="No samples found",
        )

    try:
        schema = generate_schema_from_samples(samples)
        schema["title"] = f"{provider} export format"
        schema["description"] = config.description

        return GenerationResult(
            provider=provider,
            schema=schema,
            sample_count=len(samples),
        )
    except Exception as e:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=len(samples),
            error=str(e),
        )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path = DEFAULT_DB_PATH,
    providers: list[str] | None = None,
) -> list[GenerationResult]:
    """Generate schemas for all (or specified) providers.

    Args:
        output_dir: Directory to write schema files
        db_path: Path to polylogue database
        providers: Optional list of providers (default: all)

    Returns:
        List of GenerationResults
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []

    for provider in provider_list:
        result = generate_provider_schema(provider, db_path=db_path)
        results.append(result)

        if result.success and result.schema:
            output_path = output_dir / f"{provider}.schema.json.gz"
            compressed = gzip.compress(
                json.dumps(result.schema, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            )
            output_path.write_bytes(compressed)
            # Remove legacy uncompressed file if present
            legacy_path = output_dir / f"{provider}.schema.json"
            if legacy_path.exists():
                legacy_path.unlink()

    return results


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
        default=DEFAULT_DB_PATH,
        help="Path to polylogue database",
    )

    parsed = parser.parse_args(args)

    providers = None if parsed.provider == "all" else [parsed.provider]
    results = generate_all_schemas(
        output_dir=parsed.output_dir,
        db_path=parsed.db_path,
        providers=providers,
    )

    # Report results
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
