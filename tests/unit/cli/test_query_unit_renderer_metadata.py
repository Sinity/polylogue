"""CLI renderer wiring for query-unit descriptors."""

from __future__ import annotations

import click
import pytest

from polylogue.archive.query.metadata import QueryUnitDescriptor, query_unit_descriptors
from polylogue.archive.query.unit_results import _row_payload_model
from polylogue.cli.archive_query import _query_unit_text_line
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


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
    """Terminal descriptors must resolve to the SQL executor wiring they advertise."""
    assert isinstance(descriptor, QueryUnitDescriptor)
    # Every terminal query unit is now SQL-backed (#2006 removed the
    # runtime-transform lowerer path).
    assert descriptor.lowerer_kind == "sql"
    assert descriptor.sql_query_method is not None
    assert descriptor.runtime_query_method is None
    assert hasattr(ArchiveStore, descriptor.sql_query_method)
    # Aggregate support is optional: run/observed-event/context-snapshot are
    # SQL-backed terminal units without an aggregate lowerer yet (#2006).
    assert _row_payload_model(descriptor) is not None
