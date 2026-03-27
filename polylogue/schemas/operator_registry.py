"""Registry access helpers for schema operator workflows."""

from __future__ import annotations

from typing import Any


def schema_registry() -> Any:
    from polylogue.schemas.registry import SchemaRegistry

    return SchemaRegistry()


def runtime_schema_registry() -> Any:
    from polylogue.schemas.runtime_registry import SchemaRegistry

    return SchemaRegistry()


__all__ = ["runtime_schema_registry", "schema_registry"]
