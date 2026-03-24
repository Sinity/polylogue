"""Archive data product inspection commands."""

from __future__ import annotations

import click

from polylogue.archive_products import (
    ArchiveDebtProductQuery,
    DaySessionSummaryProductQuery,
    MaintenanceRunProductQuery,
    ProviderAnalyticsProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProductQuery,
    WorkThreadProductQuery,
)
from polylogue.cli.machine_errors import emit_success
from polylogue.cli.types import AppEnv
from polylogue.sync_bridge import run_coroutine_sync


def _emit_product_list(*, json_mode: bool, key: str, items: list[object]) -> None:
    if json_mode:
        emit_success(
            {
                "count": len(items),
                key: [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else item
                    for item in items
                ],
            }
        )


def _summarize_archive_debt(items: list[object]) -> dict[str, int]:
    actionable = [item for item in items if getattr(item, "healthy", True) is False]
    return {
        "tracked_items": len(items),
        "actionable_items": len(actionable),
        "issue_rows": sum(int(getattr(item, "issue_count", 0) or 0) for item in items),
    }


@click.group("products")
def products_command() -> None:
    """Inspect durable archive data products."""


@products_command.command("status")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
@click.pass_obj
def products_status_command(env: AppEnv, json_mode: bool) -> None:
    """Show durable product readiness status."""
    status = run_coroutine_sync(env.operations.get_session_product_status())
    debt_items = run_coroutine_sync(
        env.operations.list_archive_debt_products(ArchiveDebtProductQuery())
    )
    debt_summary = _summarize_archive_debt(debt_items)
    if json_mode:
        emit_success({"session_products": status, "archive_debt": debt_summary})
        return
    click.echo("Session Product Status:\n")
    for key, value in sorted(status.items()):
        click.echo(f"  {key}: {value}")
    click.echo("\nArchive Debt:\n")
    click.echo(f"  tracked_items: {debt_summary['tracked_items']}")
    click.echo(f"  actionable_items: {debt_summary['actionable_items']}")
    click.echo(f"  issue_rows: {debt_summary['issue_rows']}")


@products_command.command("profiles")
@click.option("--provider", default=None, help="Limit to provider")
@click.option("--since", default=None, help="Only rows on/after this timestamp")
@click.option("--until", default=None, help="Only rows on/before this timestamp")
@click.option("--first-message-since", default=None, help="Only sessions whose first message is on/after this timestamp")
@click.option("--first-message-until", default=None, help="Only sessions whose first message is on/before this timestamp")
@click.option("--session-date-since", default=None, help="Only sessions whose canonical session date is on/after this date")
@click.option("--session-date-until", default=None, help="Only sessions whose canonical session date is on/before this date")
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
    query: str | None,
    limit: int,
    offset: int,
    json_mode: bool,
) -> None:
    """List durable session-profile products."""
    items = run_coroutine_sync(
        env.operations.list_session_profile_products(
            SessionProfileProductQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="session_profiles", items=items)
        return
    if not items:
        click.echo("No session profiles matched.")
        return
    click.echo(f"Session profiles: {len(items)}\n")
    for item in items:
        profile = item.profile
        projects = ", ".join(profile.get("canonical_projects", [])[:3]) or "-"
        click.echo(
            f"  {item.conversation_id} [{item.provider_name}] "
            f"{item.primary_work_kind or '-'} "
            f"{item.title or '(untitled)'}"
        )
        click.echo(
            f"    first_message_at={item.first_message_at or '-'} "
            f"session_date={item.canonical_session_date or '-'} "
            f"engaged_minutes={item.engaged_minutes:g}"
        )
        click.echo(
            f"    messages={profile.get('message_count', 0)} "
            f"work_events={len(profile.get('work_events', []))} "
            f"projects={projects}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_session_work_event_products(
            SessionWorkEventProductQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="session_work_events", items=items)
        return
    if not items:
        click.echo("No work events matched.")
        return
    click.echo(f"Work events: {len(items)}\n")
    for item in items:
        event = item.event
        click.echo(
            f"  {item.event_id} [{item.provider_name}] {item.kind} "
            f"conversation={item.conversation_id}"
        )
        click.echo(
            f"    start={item.start_time or '-'} end={item.end_time or '-'} "
            f"session_date={item.canonical_session_date or '-'} duration_ms={item.duration_ms}"
        )
        click.echo(
            f"    summary={event.get('summary', '-') or '-'} "
            f"files={len(event.get('file_paths', []))} "
            f"tools={len(event.get('tools_used', []))}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_session_phase_products(
            SessionPhaseProductQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="session_phases", items=items)
        return
    if not items:
        click.echo("No session phases matched.")
        return
    click.echo(f"Session phases: {len(items)}\n")
    for item in items:
        phase = item.phase
        click.echo(
            f"  {item.phase_id} [{item.provider_name}] {item.kind} "
            f"conversation={item.conversation_id}"
        )
        click.echo(
            f"    start={item.start_time or '-'} end={item.end_time or '-'} "
            f"session_date={item.canonical_session_date or '-'} duration_ms={item.duration_ms}"
        )
        click.echo(
            f"    message_range={phase.get('message_range', [])} "
            f"tools={phase.get('tool_counts', {})} "
            f"words={phase.get('word_count', 0)}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_work_thread_products(
            WorkThreadProductQuery(
                since=since,
                until=until,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="work_threads", items=items)
        return
    if not items:
        click.echo("No work threads matched.")
        return
    click.echo(f"Work threads: {len(items)}\n")
    for item in items:
        thread = item.thread
        click.echo(
            f"  {item.thread_id} project={item.dominant_project or '-'} "
            f"sessions={thread.get('session_count', 0)} "
            f"messages={thread.get('total_messages', 0)}"
        )
        click.echo(
            f"    depth={thread.get('depth', 0)} "
            f"branches={thread.get('branch_count', 0)} "
            f"providers={thread.get('provider_breakdown', {})}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_session_tag_rollup_products(
            SessionTagRollupQuery(
                provider=provider,
                since=since,
                until=until,
                query=query,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="session_tag_rollups", items=items)
        return
    if not items:
        click.echo("No session tag rollups matched.")
        return
    click.echo(f"Session tag rollups: {len(items)}\n")
    for item in items:
        click.echo(
            f"  {item.tag} conversations={item.conversation_count} "
            f"explicit={item.explicit_count} auto={item.auto_count}"
        )
        click.echo(
            f"    providers={item.provider_breakdown} "
            f"projects={item.project_breakdown}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_day_session_summary_products(
            DaySessionSummaryProductQuery(
                provider=provider,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="day_session_summaries", items=items)
        return
    if not items:
        click.echo("No day summaries matched.")
        return
    click.echo(f"Day session summaries: {len(items)}\n")
    for item in items:
        summary = item.summary
        click.echo(
            f"  {item.date} sessions={summary.get('session_count', 0)} "
            f"messages={summary.get('total_messages', 0)} "
            f"projects={', '.join(summary.get('projects_active', [])[:3]) or '-'}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_week_session_summary_products(
            WeekSessionSummaryProductQuery(
                provider=provider,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="week_session_summaries", items=items)
        return
    if not items:
        click.echo("No week summaries matched.")
        return
    click.echo(f"Week session summaries: {len(items)}\n")
    for item in items:
        summary = item.summary
        click.echo(
            f"  {item.iso_week} sessions={summary.get('session_count', 0)} "
            f"messages={summary.get('total_messages', 0)} "
            f"days={len(summary.get('day_summaries', []))}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_maintenance_run_products(
            MaintenanceRunProductQuery(limit=limit)
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="maintenance_runs", items=items)
        return
    if not items:
        click.echo("No maintenance runs recorded.")
        return
    click.echo(f"Maintenance runs: {len(items)}\n")
    for item in items:
        targets = ", ".join(item.target_names) if item.target_names else "all selected"
        click.echo(
            f"  {item.executed_at} {item.mode} success={item.success} "
            f"preview={item.preview} targets={targets}"
        )


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
    items = run_coroutine_sync(
        env.operations.list_provider_analytics_products(
            ProviderAnalyticsProductQuery(
                provider=provider,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="provider_analytics", items=items)
        return
    if not items:
        click.echo("No provider analytics matched.")
        return
    click.echo(f"Provider analytics: {len(items)}\n")
    for item in items:
        click.echo(
            f"  {item.provider_name} conversations={item.conversation_count} "
            f"messages={item.message_count} avg_messages={item.avg_messages_per_conversation:.1f}"
        )
        click.echo(
            f"    tools={item.tool_use_count} ({item.tool_use_percentage:.1f}% convs) "
            f"thinking={item.thinking_count} ({item.thinking_percentage:.1f}% convs)"
        )


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
    items = run_coroutine_sync(
        env.operations.list_archive_debt_products(
            ArchiveDebtProductQuery(
                category=category,
                only_actionable=actionable_only,
                limit=limit,
                offset=offset,
            )
        )
    )
    if json_mode:
        _emit_product_list(json_mode=True, key="archive_debt", items=items)
        return
    if not items:
        click.echo("No archive debt matched.")
        return
    click.echo(f"Archive debt: {len(items)}\n")
    for item in items:
        click.echo(
            f"  {item.debt_name} category={item.category} healthy={item.healthy} "
            f"issue_count={item.issue_count} destructive={item.destructive}"
        )
        click.echo(f"    target={item.maintenance_target} detail={item.detail}")


__all__ = ["products_command"]
