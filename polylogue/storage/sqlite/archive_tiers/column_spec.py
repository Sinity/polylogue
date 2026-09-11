"""Table-driven column specifications for archive_tiers hot core.

One declaration per column owns the mechanical row shape: DDL definition,
INSERT column list and placeholders, bound-value order, upsert conflict
policy, the SELECT projection a record mapper consumes, and the record-to-
domain keyword plumbing. Semantic validation, enum meaning and authored
domain fields stay with the typed models.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Specification for a single column in a table.

    name: SQL column name
    is_generated: True if column is GENERATED ALWAYS (should not be in INSERT)
    extract: Function to extract value from source object, or None if N/A
    extract_placeholder: SQL expression for INSERT VALUES (?, json_extract(...), NULL, etc)
    conflict_update: right-hand side this column contributes to an upsert's
        DO UPDATE SET, or None when a conflict must keep the stored value
    conflict_params: names of the bound values ``conflict_update``'s
        placeholders consume, in placeholder order
    """

    name: str
    is_generated: bool = False
    extract: Callable[[Any], Any] | None = None
    extract_placeholder: str = "?"
    ddl_sql: str | None = None
    record_name: str | None = None
    select_expression: str | None = None
    record_transform: Callable[[Any], Any] | None = None
    domain_name: str | None = None
    domain_transform: Callable[[Any], Any] | None = None
    conflict_update: str | None = None
    conflict_params: tuple[str, ...] = ()

    @property
    def ddl_definition(self) -> str:
        """Return the canonical SQL definition for this storage column."""
        if self.ddl_sql is None:
            raise ValueError(f"Column {self.name} has no DDL definition")
        return self.ddl_sql

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
    table_constraints: tuple[str, ...] = ()

    @property
    def record_columns(self) -> tuple[ColumnSpec, ...]:
        """Columns and derived fields used to construct the runtime record."""
        return tuple(col for col in (*self.all_columns, *self.record_only_columns) if col.record_name is not None)

    @property
    def ddl_body(self) -> str:
        """Render columns and table-level constraints for a CREATE TABLE body."""
        definitions = tuple(col.ddl_definition for col in self.all_columns)
        return ",\n    ".join((*definitions, *self.table_constraints))

    @property
    def insert_columns(self) -> tuple[ColumnSpec, ...]:
        """Writable columns whose value this table's canonical INSERT supplies.

        A writable column that declares neither an extractor nor a literal
        placeholder has no value at insert time; a later owner (lineage
        resolution, a rollup pass) writes it, so it stays out of the statement.
        """
        return tuple(col for col in self.writable_columns if col.extract is not None or col.extract_placeholder != "?")

    @property
    def insert_column_names(self) -> str:
        """Generate INSERT column list."""
        return ", ".join(col.name for col in self.insert_columns)

    @property
    def insert_placeholder_string(self) -> str:
        """Generate VALUES placeholder string (?, ?, NULL, etc)."""
        return ", ".join(col.extract_placeholder for col in self.insert_columns)

    def record_select_column_names(self, table_alias: str) -> str:
        """Generate the SELECT projection consumed by the record mapper."""
        return ",\n    ".join(col.select_sql(table_alias) for col in self.record_columns)

    @property
    def conflict_update_columns(self) -> tuple[ColumnSpec, ...]:
        """Columns whose value an upsert replaces on conflict, in declared order."""
        return tuple(col for col in self.all_columns if col.conflict_update is not None)

    def conflict_update_sql(self, indent: str = "") -> str:
        """Render the ``DO UPDATE SET`` assignments this table declares.

        A column with no ``conflict_update`` keeps its stored value, which is
        what the counter and identity columns require.
        """
        separator = ",\n" + indent
        return separator.join(f"{col.name} = {col.conflict_update}" for col in self.conflict_update_columns)

    @property
    def conflict_update_param_names(self) -> tuple[str, ...]:
        """Bound-value names the rendered ``DO UPDATE SET`` consumes, in order.

        Each column declares the values its own conflict expression reads, so
        the caller supplies them by name; the statement's parameter order is
        the declaration's, never a separately maintained tuple.
        """
        names: list[str] = []
        for col in self.conflict_update_columns:
            assert col.conflict_update is not None
            placeholders = col.conflict_update.count("?")
            if len(col.conflict_params) != placeholders:
                raise ValueError(
                    f"{self.table_name}.{col.name} conflict policy takes {placeholders} bound value(s) "
                    f"but declares {len(col.conflict_params)}"
                )
            names.extend(col.conflict_params)
        return tuple(names)

    def conflict_update_tuple(self, values: Mapping[str, Any]) -> tuple[Any, ...]:
        """Bind the upsert's conflict parameters from a name→value mapping."""
        return tuple(values[name] for name in self.conflict_update_param_names)

    def extract_tuple(self, source_obj: Any) -> tuple[Any, ...]:
        """Extract a tuple of values from a source object in insert-column order.

        Columns with a literal placeholder (NULL, an expression) appear in the
        VALUES clause but carry no bound value, so they are skipped here.
        """
        result = []
        for col in self.insert_columns:
            if col.extract_placeholder != "?":
                continue
            if col.extract is None:
                raise ValueError(f"No extractor defined for column {col.name}")
            result.append(col.extract(source_obj))
        return tuple(result)

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

    def domain_kwargs(self, record: Any, *, accepted: Container[str] | None = None) -> dict[str, Any]:
        """Project a runtime record into domain-model constructor kwargs.

        ``accepted`` restricts the projection to the domain names a target
        model declares (pass its ``model_fields``); one storage declaration
        then feeds several domain models that carry different subsets of the
        same row without any of them restating the mapping. Omitting it
        projects every declared domain field.
        """
        result: dict[str, Any] = {}
        for col in self.record_columns:
            if col.domain_name is None or col.record_name is None:
                continue
            if accepted is not None and col.domain_name not in accepted:
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
