"""Archive data product inspection commands."""

from __future__ import annotations

import click

from polylogue.cli.commands.products_aggregate import (
    register_aggregate_product_commands,
)
from polylogue.cli.commands.products_session import (
    register_session_product_commands,
)


@click.group("products")
def products_command() -> None:
    """Inspect durable archive data products."""


register_session_product_commands(products_command)
register_aggregate_product_commands(products_command)


__all__ = ["products_command"]
