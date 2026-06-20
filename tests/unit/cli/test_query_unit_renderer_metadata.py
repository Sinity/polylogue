"""CLI renderer wiring for query-unit descriptors."""

from __future__ import annotations

import click
import pytest

from polylogue.archive.query.metadata import query_unit_descriptors
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
