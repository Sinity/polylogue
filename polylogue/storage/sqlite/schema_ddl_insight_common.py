"""Shared SQL column fragments for session-insight DDL composition.

The session-insight tables (`session_profiles`, `session_work_events`,
`session_phases`, `session_tag_rollups`, `work_threads`)
all carry the same materializer-lifecycle bookkeeping columns. Defining them
once here keeps the renders byte-identical across DDL files and prevents drift
when the lifecycle contract evolves.
"""

from __future__ import annotations

_INDENT = "            "

# Columns shared by every session-insight table, in canonical order.
_LIFECYCLE_COLUMNS_BASE: tuple[str, ...] = (
    "materializer_version INTEGER NOT NULL DEFAULT 5",
    "materialized_at TEXT NOT NULL",
    "source_updated_at TEXT",
)

# `source_sort_key` is present on every session-insight table except
# `work_threads`, which does not carry a per-row sort key.
_LIFECYCLE_COLUMN_SORT_KEY = "source_sort_key REAL"

_LIFECYCLE_COLUMNS_TAIL: tuple[str, ...] = (
    "input_high_water_mark TEXT",
    "input_high_water_mark_source TEXT",
    "input_row_count INTEGER NOT NULL DEFAULT 0",
)


def materialization_columns_sql(*, with_source_sort_key: bool = True) -> str:
    """Render the lifecycle column block as a SQL fragment.

    The returned string is one line per column, each indented to match the
    other columns in the insight `CREATE TABLE` statements, and each terminated
    with a comma (so the fragment can be spliced between other column
    definitions). It starts with a newline so it composes cleanly after a
    preceding column line.
    """

    columns: list[str] = list(_LIFECYCLE_COLUMNS_BASE)
    if with_source_sort_key:
        columns.append(_LIFECYCLE_COLUMN_SORT_KEY)
    columns.extend(_LIFECYCLE_COLUMNS_TAIL)
    return "\n" + "\n".join(f"{_INDENT}{col}," for col in columns)


# Convenience pre-rendered fragments for the two variants in active use.
MATERIALIZATION_COLUMNS_SQL = materialization_columns_sql()
MATERIALIZATION_COLUMNS_SQL_NO_SORT_KEY = materialization_columns_sql(with_source_sort_key=False)
