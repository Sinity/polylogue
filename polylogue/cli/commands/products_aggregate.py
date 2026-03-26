"""Aggregate archive product commands."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering import (
    render_day_session_summaries,
    render_provider_analytics,
    render_session_tag_rollups,
    render_week_session_summaries,
)
from polylogue.cli.products_workflow import (
    list_day_session_summary_products,
    list_provider_analytics_products,
    list_session_tag_rollup_products,
    list_week_session_summary_products,
)
from polylogue.cli.types import AppEnv


def register_aggregate_product_commands(products_command: click.Group) -> None:
    products_command.add_command(products_tags_command)
    products_command.add_command(products_day_summaries_command)
    products_command.add_command(products_week_summaries_command)
    products_command.add_command(products_analytics_command)


@click.command("tags")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--query", default=None, help="Substring match against the tag name")
@click.option("--limit", type=int, default=100, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_tags_command(
    env: AppEnv,
    provider: str | None,
    since: str | None,
    until: str | None,
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable session-tag rollup products."""
    items = list_session_tag_rollup_products(
        env,
        provider=provider,
        since=since,
        until=until,
        query=query,
        limit=limit,
        offset=offset,
    )
    render_session_tag_rollups(items, json_mode=json_mode)


@click.command("day-summaries")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--limit", type=int, default=90, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_day_summaries_command(
    env: AppEnv,
    provider: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable day-level session summary products."""
    items = list_day_session_summary_products(
        env,
        provider=provider,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    render_day_session_summaries(items, json_mode=json_mode)


@click.command("week-summaries")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--limit", type=int, default=52, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_week_summaries_command(
    env: AppEnv,
    provider: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable week-level session summary products."""
    items = list_week_session_summary_products(
        env,
        provider=provider,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    render_week_session_summaries(items, json_mode=json_mode)


@click.command("analytics")
@click.option("--provider", default=None, help="Limit to one provider")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_analytics_command(
    env: AppEnv,
    provider: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List canonical provider-level analytics products."""
    items = list_provider_analytics_products(
        env,
        provider=provider,
        limit=limit,
        offset=offset,
    )
    render_provider_analytics(items, json_mode=json_mode)


__all__ = ["register_aggregate_product_commands"]
