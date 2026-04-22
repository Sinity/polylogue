"""Typed output documents for query CLI surfaces."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryOutputFormat
from polylogue.lib.json import JSONDocument

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


def _selected_field_names(fields: str | None) -> frozenset[str] | None:
    if not fields:
        return None
    return frozenset(field.strip() for field in fields.split(",") if field.strip())


def _selected_row(row: JSONDocument, selected: frozenset[str] | None) -> JSONDocument:
    if selected is None:
        return dict(row)
    return {key: value for key, value in row.items() if key in selected}


@dataclass(frozen=True, slots=True)
class StructuredRowsDocument:
    """Rows plus alternate renderers for deterministic query-list output."""

    rows: tuple[JSONDocument, ...]
    csv_headers: tuple[str, ...]
    csv_rows: tuple[tuple[object, ...], ...]
    text_lines: tuple[str, ...]

    def with_selected_fields(self, fields: str | None) -> StructuredRowsDocument:
        selected = _selected_field_names(fields)
        if selected is None:
            return self
        return StructuredRowsDocument(
            rows=tuple(_selected_row(row, selected) for row in self.rows),
            csv_headers=self.csv_headers,
            csv_rows=self.csv_rows,
            text_lines=self.text_lines,
        )

    def render(self, output_format: QueryOutputFormat) -> str:
        if output_format == "json":
            return json.dumps(list(self.rows), indent=2)
        if output_format == "yaml":
            import yaml

            return yaml.dump(list(self.rows), default_flow_style=False, allow_unicode=True)
        if output_format == "csv":
            return self._render_csv()
        return "\n".join(self.text_lines)

    def _render_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.csv_headers)
        writer.writerows(self.csv_rows)
        return buffer.getvalue().rstrip("\r\n")


@dataclass(frozen=True, slots=True)
class QueryOutputDocument:
    """Rendered content plus delivery contract for query output surfaces."""

    content: str
    output_format: QueryOutputFormat
    destinations: tuple[QueryDeliveryTarget, ...]
    conversation: Conversation | None = None


__all__ = [
    "QueryOutputDocument",
    "StructuredRowsDocument",
]
