"""Schema audit loading and traversal helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from polylogue.schemas.runtime_registry import SchemaRegistry

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)


def _load_committed_schema(provider: str) -> dict[str, Any] | None:
    """Load a committed provider schema."""
    schema_root = Path(__file__).resolve().parent / "providers"
    return SchemaRegistry(storage_root=schema_root).get_schema(provider, version="default")


def _walk_values(schema: dict[str, Any], path: str = "$") -> list[tuple[str, list[str]]]:
    """Walk schema tree and collect all x-polylogue-values entries."""
    results: list[tuple[str, list[str]]] = []
    if not isinstance(schema, dict):
        return results

    values = schema.get("x-polylogue-values")
    if isinstance(values, list):
        results.append((path, [str(v) for v in values]))

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            results.extend(_walk_values(prop, f"{path}.{name}"))
    if isinstance(schema.get("additionalProperties"), dict):
        results.extend(_walk_values(schema["additionalProperties"], f"{path}.*"))
    if isinstance(schema.get("items"), dict):
        results.extend(_walk_values(schema["items"], f"{path}[*]"))
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema:
            for sub in schema[kw]:
                if isinstance(sub, dict):
                    results.extend(_walk_values(sub, path))

    return results


def _walk_semantic_roles(schema: dict[str, Any], path: str = "$") -> list[tuple[str, str, float]]:
    """Walk schema and collect (path, role, confidence) tuples."""
    results: list[tuple[str, str, float]] = []
    if not isinstance(schema, dict):
        return results

    role = schema.get("x-polylogue-semantic-role")
    confidence = schema.get("x-polylogue-confidence", 0.0)
    if role:
        results.append((path, role, confidence))

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            results.extend(_walk_semantic_roles(prop, f"{path}.{name}"))
    if isinstance(schema.get("additionalProperties"), dict):
        results.extend(_walk_semantic_roles(schema["additionalProperties"], f"{path}.*"))
    if isinstance(schema.get("items"), dict):
        results.extend(_walk_semantic_roles(schema["items"], f"{path}[*]"))
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema:
            for sub in schema[kw]:
                if isinstance(sub, dict):
                    results.extend(_walk_semantic_roles(sub, path))

    return results


__all__ = [
    "_HEX_RE",
    "_UUID_RE",
    "_load_committed_schema",
    "_walk_semantic_roles",
    "_walk_values",
]
