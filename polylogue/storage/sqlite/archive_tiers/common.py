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


def json_check(column: str, *, json_type: str | None = None, nullable: bool = False) -> str:
    """Return a SQLite JSON1 CHECK expression for a JSON text column.

    ``json_type()`` raises on malformed input, so the expression uses a
    ``CASE`` guard instead of relying on SQL short-circuiting.  Nullable
    payload columns opt in explicitly; non-null columns reject NULL because
    ``json_valid(NULL)`` is false.
    """
    valid_expression = f"json_valid({column})"
    if json_type is not None:
        valid_expression = f"CASE WHEN json_valid({column}) THEN json_type({column}) = '{json_type}' ELSE 0 END"
    if nullable:
        return f"({column} IS NULL OR {valid_expression})"
    return valid_expression


def json_object_check(column: str, *, nullable: bool = False) -> str:
    """Return a CHECK expression requiring a JSON object payload."""
    return json_check(column, json_type="object", nullable=nullable)


def json_array_check(column: str, *, nullable: bool = False) -> str:
    """Return a CHECK expression requiring a JSON array payload."""
    return json_check(column, json_type="array", nullable=nullable)


__all__ = [
    "CONTENT_HASH_CHECK",
    "JSON_TEXT_DEFAULT",
    "check",
    "json_array_check",
    "json_check",
    "json_object_check",
    "nullable_check",
]
