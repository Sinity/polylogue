"""Shared DDL helpers for the archive target."""

from __future__ import annotations

from polylogue.core.enums import PolylogueStrEnum, nullable_sql_check_in, sql_check_in


def check(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a non-null enum CHECK expression."""
    return sql_check_in(column, enum_type)


def nullable_check(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a nullable enum CHECK expression."""
    return nullable_sql_check_in(column, enum_type)


CONTENT_HASH_CHECK = "CHECK(length(content_hash) = 32)"
JSON_TEXT_DEFAULT = "TEXT NOT NULL DEFAULT '{}'"


__all__ = ["CONTENT_HASH_CHECK", "JSON_TEXT_DEFAULT", "check", "nullable_check"]
