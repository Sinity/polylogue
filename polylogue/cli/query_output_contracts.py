"""Typed output documents for query CLI surfaces."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryOutputFormat
from polylogue.core.json import JSONDocument

if TYPE_CHECKING:
    from polylogue.archive.models import Session


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
            # #1618: paginated envelope matches the MCP list_sessions
            # tool. The CLI doesn't paginate today so ``limit`` mirrors
            # ``total`` and ``offset`` is 0; future CLI pagination will
            # populate the same fields MCP already does.
            envelope = {
                "items": list(self.rows),
                "total": len(self.rows),
                "limit": len(self.rows),
                "offset": 0,
            }
            return json.dumps(envelope, indent=2)
        if output_format == "ndjson":
            # JSON Lines streaming form: one JSON document per line, no
            # outer array, no indentation. Stable for shell pipelines and
            # LLM tool-use harnesses that want incremental parsing (#1272).
            # Pagination context does not apply to a streaming form.
            return "\n".join(json.dumps(row, separators=(",", ":")) for row in self.rows)
        if output_format == "yaml":
            import yaml

            envelope = {
                "items": list(self.rows),
                "total": len(self.rows),
                "limit": len(self.rows),
                "offset": 0,
            }
            return yaml.dump(envelope, default_flow_style=False, allow_unicode=True, sort_keys=False)
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
    session: Session | None = None


__all__ = [
    "QueryOutputDocument",
    "StructuredRowsDocument",
]
