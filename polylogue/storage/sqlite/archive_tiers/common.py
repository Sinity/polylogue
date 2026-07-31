"""Shared DDL helpers for the archive target."""

from __future__ import annotations

from polylogue.core.enums import (
    PolylogueStrEnum,
    nullable_sql_check_in,
    sql_check_in,
    sql_string_literal,
)


def check(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a non-null enum CHECK expression."""
    return sql_check_in(column, enum_type)


def literal_check(column: str, *values: str) -> str:
    """Return a non-null ``column IN (...)`` expression for explicit literals.

    Mirrors :func:`check`, but for closed vocabularies expressed as
    ``typing.Literal`` aliases rather than ``PolylogueStrEnum`` types. Callers
    expand the alias with ``typing.get_args`` at the call site so this helper
    stays free of an ``insights`` import inside the storage substrate.

    Restored: PR #3458 deleted this as an "uncalled generator", but
    ``archive_tiers/index.py``'s ``delegation_facts`` DDL calls it twice
    (``mapping_state``, ``result_status`` CHECK clauses) -- the deletion
    broke the import chain for the whole ``archive_tiers`` package (and
    therefore ``ArchiveStore``, the CLI, and every test) on ``master`` as of
    5798b3dd1. The "zero call sites" audit was simply wrong; both call
    sites are plain ``literal_check(...)`` calls, not aliased or generated.
    Re-verify with a real grep before deleting again.
    """
    if not values:
        raise ValueError("literal_check requires at least one value")
    rendered = ", ".join(sql_string_literal(value) for value in values)
    return f"{column} IN ({rendered})"


def nullable_check(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a nullable enum CHECK expression."""
    return nullable_sql_check_in(column, enum_type)


def order_check(later: str, earlier: str, *, nullable: bool = True) -> str:
    """Return a same-row ordering CHECK expression: ``later >= earlier``.

    Mirrors :func:`check` / :func:`json_check` — a relationship
    between two columns of the *same row* generated from a single call site
    instead of hand-written per table, so every ordering constraint reads
    identically (see ``raw_authority_blockers.resolved_at_ms`` in
    ``archive_tiers/source.py``, the one existing hand-written instance this
    generalizes).

    ``nullable=True`` (the default) treats either side being ``NULL`` as
    satisfying the constraint — appropriate for optional timestamp pairs
    where "unknown" must stay representable and is not itself a violation.
    Pass ``nullable=False`` only when both columns are declared ``NOT NULL``;
    SQLite's three-valued CHECK logic would otherwise silently accept rows
    where a NOT NULL column is unexpectedly NULL.

    Not currently applied to any table. See polylogue-cuxz.10: the archive's
    inverted-range defect (session_phases/session_work_events, 13,743 rows,
    ~99% chatgpt-export) lives entirely in tables a sibling bead deletes
    outright, so no CHECK is warranted there. The other same-row order
    candidates found while measuring that bead (session_profiles,
    session_latency_profiles, threads: first_message_at/last_message_at)
    still have a small number of producer-side sentinel-date rows that would
    make this a rejecting CHECK reject genuinely-produced rows today — the
    materializer defect must be fixed first. This helper is prepared for
    those tables once that lands, to ride the next index-tier rebuild rather
    than force its own.
    """
    mutual_null = f"{later} IS NULL OR {earlier} IS NULL OR " if nullable else ""
    return f"({mutual_null}{later} >= {earlier})"


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
    "literal_check",
    "nullable_check",
    "order_check",
]
