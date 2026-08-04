"""Table-driven column specifications for archive_tiers hot core.

Consolidates hand-aligned column lists, INSERT placeholders, and tuple ordering
into a single source of truth per table. Eliminates 388 manual row[col]
accessors and triplicates of each column list (DDL, INSERT, tuple order).

Design: Column specs are dataclass-derived (via dataclasses.fields()) with
expression support for NULL literals, sqlite_text coercions, JSON decoders, and
GENERATED column markers. Specs drive:
  - INSERT statement generation (column list + placeholders)
  - Tuple order (what order to yield values in)
  - Row extraction (type-safe mapping from sqlite3.Row to typed fields)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Specification for a single column in a table.

    name: SQL column name
    sql_type: SQL type (for reference only)
    is_generated: True if column is GENERATED ALWAYS (should not be in INSERT)
    extract: Function to extract value from source object, or None if N/A
    extract_placeholder: SQL expression for INSERT VALUES (?, json_extract(...), NULL, etc)
    """

    name: str
    sql_type: str = "TEXT"
    is_generated: bool = False
    extract: Callable[[Any], Any] | None = None
    extract_placeholder: str = "?"
    ddl_sql: str | None = None
    record_name: str | None = None
    select_expression: str | None = None
    record_transform: Callable[[Any], Any] | None = None
    domain_name: str | None = None
    domain_transform: Callable[[Any], Any] | None = None

    @property
    def ddl_definition(self) -> str:
        """Return the canonical SQL definition for this storage column."""
        if self.ddl_sql is not None:
            return self.ddl_sql
        generated = " GENERATED ALWAYS AS (...) STORED" if self.is_generated else ""
        return f"{self.name} {self.sql_type}{generated}"

    def select_sql(self, table_alias: str) -> str:
        """Return this column's row-projection expression.

        ``select_expression`` may use ``{alias}`` to refer to the table alias.
        The output is always labelled with ``record_name`` so mappers consume
        the same names regardless of whether a query selected a raw column or
        a derived expression.
        """
        if self.record_name is None:
            raise ValueError(f"Column {self.name} has no record projection")
        expression = self.select_expression or f"{table_alias}.{self.name}"
        return f"{expression.format(alias=table_alias)} AS {self.record_name}"


@dataclass(frozen=True, slots=True)
class TableColumnSpec:
    """Complete column specification for a table.

    table_name: SQL table name
    all_columns: All columns in table (including GENERATED)
    writable_columns: Columns that go in INSERT (excludes GENERATED)
    """

    table_name: str
    all_columns: tuple[ColumnSpec, ...]
    writable_columns: tuple[ColumnSpec, ...]
    record_only_columns: tuple[ColumnSpec, ...] = ()

    @property
    def record_columns(self) -> tuple[ColumnSpec, ...]:
        """Columns and derived fields used to construct the runtime record."""
        return tuple(col for col in (*self.all_columns, *self.record_only_columns) if col.record_name is not None)

    @property
    def ddl_column_definitions(self) -> str:
        """Render the table's column definitions from the storage declaration."""
        return ",\n    ".join(col.ddl_definition for col in self.all_columns)

    @property
    def insert_column_names(self) -> str:
        """Generate INSERT column list."""
        return ", ".join(col.name for col in self.writable_columns)

    @property
    def insert_placeholder_string(self) -> str:
        """Generate VALUES placeholder string (?, ?, NULL, etc)."""
        return ", ".join(col.extract_placeholder for col in self.writable_columns)

    @property
    def select_column_names(self) -> str:
        """Generate SELECT column list (all columns including GENERATED)."""
        return ", ".join(col.name for col in self.all_columns)

    def record_select_column_names(self, table_alias: str) -> str:
        """Generate the SELECT projection consumed by the record mapper."""
        return ",\n    ".join(col.select_sql(table_alias) for col in self.record_columns)

    def extract_tuple(self, source_obj: Any) -> tuple[Any, ...]:
        """Extract a tuple of values from a source object in writable column order.

        Skips columns with extract_placeholder != "?" (e.g., NULL literals).
        Those are included in the SQL VALUES clause but not in the tuple.
        """
        result = []
        for col in self.writable_columns:
            # Skip columns with non-standard placeholders (NULL, expressions, etc)
            if col.extract_placeholder != "?":
                continue
            if col.extract is not None:
                result.append(col.extract(source_obj))
            else:
                raise ValueError(f"No extractor defined for column {col.name}")
        return tuple(result)

    def row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a typed dict using column specs."""
        return {col.name: row[col.name] for col in self.all_columns}

    def row_to_record_kwargs(self, row: sqlite3.Row) -> dict[str, Any]:
        """Extract only selected record fields, preserving omitted-column defaults."""
        names = set(row.keys())
        result: dict[str, Any] = {}
        for col in self.record_columns:
            assert col.record_name is not None
            if col.record_name not in names:
                continue
            value = row[col.record_name]
            if col.record_transform is not None:
                value = col.record_transform(value)
            result[col.record_name] = value
        return result

    def domain_kwargs(self, record: Any) -> dict[str, Any]:
        """Project a runtime record into domain-model constructor kwargs."""
        result: dict[str, Any] = {}
        for col in self.record_columns:
            if col.domain_name is None or col.record_name is None:
                continue
            value = getattr(record, col.record_name)
            if col.domain_transform is not None:
                value = col.domain_transform(value)
            result[col.domain_name] = value
        return result

    def domain_value(self, record: Any, domain_name: str) -> Any:
        """Return one domain field using the declaration's record mapping."""
        for col in self.record_columns:
            if col.domain_name == domain_name and col.record_name is not None:
                value = getattr(record, col.record_name)
                return col.domain_transform(value) if col.domain_transform is not None else value
        raise KeyError(f"{self.table_name} has no domain field {domain_name!r}")

    def row_to_typed_dict(
        self, row: sqlite3.Row, type_mapper: dict[str, Callable[[Any], Any]] | None = None
    ) -> dict[str, Any]:
        """Convert a sqlite3.Row to a typed dict with optional per-column transformers."""
        result = {}
        for col in self.all_columns:
            value = row[col.name]
            if type_mapper and col.name in type_mapper:
                value = type_mapper[col.name](value)
            result[col.name] = value
        return result
