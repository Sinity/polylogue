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
