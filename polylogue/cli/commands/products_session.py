"""Session-oriented archive product commands."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering import (
    render_session_enrichments,
    render_session_phases,
    render_session_profiles,
    render_session_work_events,
    render_work_threads,
)
from polylogue.cli.products_workflow import (
    list_session_enrichment_products,
    list_session_phase_products,
    list_session_profile_products,
    list_session_work_event_products,
    list_work_thread_products,
)
from polylogue.cli.types import AppEnv


def register_session_product_commands(products_command: click.Group) -> None:
    products_command.add_command(products_profiles_command)
    products_command.add_command(products_enrichments_command)
    products_command.add_command(products_work_events_command)
    products_command.add_command(products_phases_command)
    products_command.add_command(products_threads_command)


@click.command("profiles")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--first-message-since", default=None, help="Only sessions whose first message is on/after this timestamp")
@click.option("--first-message-until", default=None, help="Only sessions whose first message is on/before this timestamp")
@click.option("--session-date-since", default=None, help="Only sessions whose canonical session date is on/after this date")
@click.option("--session-date-until", default=None, help="Only sessions whose canonical session date is on/before this date")
@click.option(
    "--tier",
    type=click.Choice(["merged", "evidence", "inference"]),
    default="merged",
    show_default=True,
    help="Return merged, evidence-only, or inference-only profile products",
)
@click.option("--query", default=None, help="FTS query against product search text")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_profiles_command(
    env: AppEnv,
    provider: str | None,
    since: str | None,
    until: str | None,
    first_message_since: str | None,
    first_message_until: str | None,
    session_date_since: str | None,
    session_date_until: str | None,
    tier: str,
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable session-profile products."""
    items = list_session_profile_products(
        env,
        provider=provider,
        since=since,
        until=until,
        first_message_since=first_message_since,
        first_message_until=first_message_until,
        session_date_since=session_date_since,
        session_date_until=session_date_until,
        tier=tier,
        query=query,
        limit=limit,
        offset=offset,
    )
    render_session_profiles(items, json_mode=json_mode)


@click.command("enrichments")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--first-message-since", default=None, help="Only sessions whose first message is on/after this timestamp")
@click.option("--first-message-until", default=None, help="Only sessions whose first message is on/before this timestamp")
@click.option("--session-date-since", default=None, help="Only sessions whose canonical session date is on/after this date")
@click.option("--session-date-until", default=None, help="Only sessions whose canonical session date is on/before this date")
@click.option("--refined-work-kind", default=None, help="Only this refined enrichment work kind")
@click.option("--query", default=None, help="FTS query against enrichment search text")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_enrichments_command(
    env: AppEnv,
    provider: str | None,
    since: str | None,
    until: str | None,
    first_message_since: str | None,
    first_message_until: str | None,
    session_date_since: str | None,
    session_date_until: str | None,
    refined_work_kind: str | None,
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable probabilistic session-enrichment products."""
    items = list_session_enrichment_products(
        env,
        provider=provider,
        since=since,
        until=until,
        first_message_since=first_message_since,
        first_message_until=first_message_until,
        session_date_since=session_date_since,
        session_date_until=session_date_until,
        refined_work_kind=refined_work_kind,
        query=query,
        limit=limit,
        offset=offset,
    )
    render_session_enrichments(items, json_mode=json_mode)


@click.command("work-events")
@click.option("--conversation-id", default=None, help="Only events from one conversation")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--kind", default=None, help="Only this work-event kind")
@click.option("--query", default=None, help="FTS query against event search text")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_work_events_command(
    env: AppEnv,
    conversation_id: str | None,
    provider: str | None,
    since: str | None,
    until: str | None,
    kind: str | None,
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable work-event products."""
    items = list_session_work_event_products(
        env,
        conversation_id=conversation_id,
        provider=provider,
        since=since,
        until=until,
        kind=kind,
        query=query,
        limit=limit,
        offset=offset,
    )
    render_session_work_events(items, json_mode=json_mode)


@click.command("phases")
@click.option("--conversation-id", default=None, help="Only phases from one conversation")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--kind", default=None, help="Only this session phase kind")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_phases_command(
    env: AppEnv,
    conversation_id: str | None,
    provider: str | None,
    since: str | None,
    until: str | None,
    kind: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable session-phase products."""
    items = list_session_phase_products(
        env,
        conversation_id=conversation_id,
        provider=provider,
        since=since,
        until=until,
        kind=kind,
        limit=limit,
        offset=offset,
    )
    render_session_phases(items, json_mode=json_mode)


@click.command("threads")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--query", default=None, help="FTS query against thread search text")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_threads_command(
    env: AppEnv,
    since: str | None,
    until: str | None,
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable work-thread products."""
    items = list_work_thread_products(
        env,
        since=since,
        until=until,
        query=query,
        limit=limit,
        offset=offset,
    )
    render_work_threads(items, json_mode=json_mode)


__all__ = ["register_session_product_commands"]
