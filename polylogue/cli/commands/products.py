"""Archive data product inspection commands."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering import (
    render_archive_debt,
    render_day_session_summaries,
    render_maintenance_runs,
    render_products_status,
    render_provider_analytics,
    render_session_phases,
    render_session_profiles,
    render_session_tag_rollups,
    render_session_work_events,
    render_week_session_summaries,
    render_work_threads,
    summarize_archive_debt,
)
from polylogue.cli.products_workflow import (
    get_products_status,
    list_archive_debt_products,
    list_day_session_summary_products,
    list_maintenance_run_products,
    list_provider_analytics_products,
    list_session_phase_products,
    list_session_profile_products,
    list_session_tag_rollup_products,
    list_session_work_event_products,
    list_week_session_summary_products,
    list_work_thread_products,
)
from polylogue.cli.types import AppEnv


@click.group("products")
def products_command() -> None:
    """Inspect durable archive data products."""


@products_command.command("status")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_status_command(env: AppEnv, json_mode: bool) -> None:
    """Show durable product readiness status."""
    status, debt_items = get_products_status(env)
    render_products_status(
        status=status,
        debt_summary=summarize_archive_debt(debt_items),
        json_mode=json_mode,
    )


@products_command.command("profiles")
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


@products_command.command("work-events")
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


@products_command.command("phases")
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


@products_command.command("threads")
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


@products_command.command("tags")
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


@products_command.command("day-summaries")
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


@products_command.command("week-summaries")
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


@products_command.command("maintenance")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum rows")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_maintenance_command(
    env: AppEnv,
    limit: int,
    json_mode: bool,
) -> None:
    """List durable maintenance preview/apply lineage."""
    items = list_maintenance_run_products(env, limit=limit)
    render_maintenance_runs(items, json_mode=json_mode)


@products_command.command("analytics")
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


@products_command.command("debt")
@click.option("--category", default=None, help="Limit to debt category")
@click.option("--actionable-only", is_flag=True, help="Show only debt items with outstanding issues")
@click.option("--limit", type=int, default=50, show_default=True, help="Maximum rows")
@click.option("--offset", type=int, default=0, show_default=True, help="Start offset")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_debt_command(
    env: AppEnv,
    category: str | None,
    actionable_only: bool,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List governed live archive debt items."""
    items = list_archive_debt_products(
        env,
        category=category,
        actionable_only=actionable_only,
        limit=limit,
        offset=offset,
    )
    render_archive_debt(items, json_mode=json_mode)


__all__ = ["products_command"]
