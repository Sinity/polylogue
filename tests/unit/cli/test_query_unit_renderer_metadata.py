"""CLI renderer wiring for query-unit descriptors."""

from __future__ import annotations

import click
import pytest

from polylogue.archive.query.metadata import QueryUnitDescriptor, query_unit_descriptors
from polylogue.archive.query.unit_results import _ROW_PAYLOAD_MODELS, _RUNTIME_TRANSFORM_QUERIES, _SQL_QUERY_METHODS
from polylogue.cli.archive_query import _query_unit_text_line


@pytest.mark.parametrize(
    "unit",
    [descriptor.unit for descriptor in query_unit_descriptors(terminal_supported=True)],
)
def test_terminal_query_unit_descriptors_resolve_plain_renderers(unit: str) -> None:
    """Terminal query units render through descriptor-owned CLI metadata."""
    assert callable(_query_unit_text_line(unit))


def test_unknown_query_unit_renderer_fails_typed() -> None:
    with pytest.raises(click.UsageError, match="Unsupported query unit: imaginary"):
        _query_unit_text_line("imaginary")


@pytest.mark.parametrize(
    "descriptor",
    list(query_unit_descriptors(terminal_supported=True)),
    ids=lambda descriptor: descriptor.unit,
)
def test_terminal_query_unit_descriptors_resolve_executors(descriptor: object) -> None:
    """Terminal descriptors must resolve to the executor wiring they advertise."""
    assert isinstance(descriptor, QueryUnitDescriptor)
    unit = descriptor.unit
    if descriptor.lowerer_kind == "runtime_transform":
        assert unit in _RUNTIME_TRANSFORM_QUERIES
        return
    assert descriptor.lowerer_kind == "sql"
    assert unit in _SQL_QUERY_METHODS
    assert descriptor.payload_model in _ROW_PAYLOAD_MODELS
