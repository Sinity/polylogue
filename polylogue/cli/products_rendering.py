"""Rendering helpers for archive-product CLI commands."""

from __future__ import annotations

import click

from polylogue.cli.machine_errors import emit_success


def _model_payload(item: object) -> object:
    return item.model_dump(mode="json") if hasattr(item, "model_dump") else item


def emit_product_list(*, key: str, items: list[object]) -> None:
    emit_success({"count": len(items), key: [_model_payload(item) for item in items]})


def summarize_archive_debt(items: list[object]) -> dict[str, int]:
    actionable = [item for item in items if getattr(item, "healthy", True) is False]
    return {
        "tracked_items": len(items),
        "actionable_items": len(actionable),
        "issue_rows": sum(int(getattr(item, "issue_count", 0) or 0) for item in items),
    }


def render_products_status(*, status: dict[str, int | bool], debt_summary: dict[str, int], json_mode: bool) -> None:
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


def render_session_profiles(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="session_profiles", items=items)
        return
    if not items:
        click.echo("No session profiles matched.")
        return
    click.echo(f"Session profiles: {len(items)}\n")
    for item in items:
        evidence = item.evidence.model_dump(mode="json") if item.evidence is not None else {}
        inference = item.inference.model_dump(mode="json") if item.inference is not None else {}
        projects = ", ".join(inference.get("canonical_projects", [])[:3]) or "-"
        click.echo(
            f"  {item.conversation_id} [{item.provider_name}] "
            f"tier={item.semantic_tier} "
            f"{inference.get('primary_work_kind', '-') or '-'} "
            f"{item.title or '(untitled)'}"
        )
        click.echo(
            f"    first_message_at={evidence.get('first_message_at', '-') or '-'} "
            f"session_date={evidence.get('canonical_session_date', '-') or '-'} "
            f"engaged_minutes={inference.get('engaged_minutes', 0) or 0:g}"
        )
        click.echo(
            f"    messages={evidence.get('message_count', 0)} "
            f"work_events={inference.get('work_event_count', 0)} "
            f"projects={projects}"
        )


def render_session_work_events(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="session_work_events", items=items)
        return
    if not items:
        click.echo("No work events matched.")
        return
    click.echo(f"Work events: {len(items)}\n")
    for item in items:
        evidence = item.evidence.model_dump(mode="json")
        inference = item.inference.model_dump(mode="json")
        click.echo(
            f"  {item.event_id} [{item.provider_name}] {inference.get('kind', '-')} "
            f"conversation={item.conversation_id}"
        )
        click.echo(
            f"    start={evidence.get('start_time', '-') or '-'} end={evidence.get('end_time', '-') or '-'} "
            f"session_date={evidence.get('canonical_session_date', '-') or '-'} duration_ms={evidence.get('duration_ms', 0)}"
        )
        click.echo(
            f"    summary={inference.get('summary', '-') or '-'} "
            f"confidence={inference.get('confidence', 0) or 0:.2f} "
            f"files={len(evidence.get('file_paths', []))} "
            f"tools={len(evidence.get('tools_used', []))}"
        )


def render_session_phases(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="session_phases", items=items)
        return
    if not items:
        click.echo("No session phases matched.")
        return
    click.echo(f"Session phases: {len(items)}\n")
    for item in items:
        evidence = item.evidence.model_dump(mode="json")
        inference = item.inference.model_dump(mode="json")
        click.echo(
            f"  {item.phase_id} [{item.provider_name}] {inference.get('kind', '-')} "
            f"conversation={item.conversation_id}"
        )
        click.echo(
            f"    start={evidence.get('start_time', '-') or '-'} end={evidence.get('end_time', '-') or '-'} "
            f"session_date={evidence.get('canonical_session_date', '-') or '-'} duration_ms={evidence.get('duration_ms', 0)}"
        )
        click.echo(
            f"    confidence={inference.get('confidence', 0) or 0:.2f} "
            f"message_range={evidence.get('message_range', [])} "
            f"tools={evidence.get('tool_counts', {})} "
            f"words={evidence.get('word_count', 0)}"
        )


def render_work_threads(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="work_threads", items=items)
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


def render_session_tag_rollups(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="session_tag_rollups", items=items)
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


def render_day_session_summaries(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="day_session_summaries", items=items)
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


def render_week_session_summaries(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="week_session_summaries", items=items)
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


def render_maintenance_runs(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="maintenance_runs", items=items)
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


def render_provider_analytics(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="provider_analytics", items=items)
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


def render_archive_debt(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="archive_debt", items=items)
        return
    if not items:
        click.echo("No archive debt matched.")
        return
    click.echo(f"Archive debt: {len(items)}\n")
    for item in items:
        click.echo(
            f"  {item.debt_name} category={item.category} healthy={item.healthy} "
            f"issue_count={item.issue_count} destructive={item.destructive} "
            f"stage={item.governance_stage}"
        )
        click.echo(
            f"    target={item.maintenance_target} detail={item.detail}"
        )
        if item.lineage is not None:
            click.echo(
                f"    lineage preview={item.lineage.latest_preview_at or '-'} "
                f"apply={item.lineage.latest_apply_at or '-'} "
                f"ok_apply={item.lineage.latest_successful_apply_at or '-'}"
            )


__all__ = [
    "emit_product_list",
    "render_archive_debt",
    "render_day_session_summaries",
    "render_maintenance_runs",
    "render_products_status",
    "render_provider_analytics",
    "render_session_phases",
    "render_session_profiles",
    "render_session_tag_rollups",
    "render_session_work_events",
    "render_week_session_summaries",
    "render_work_threads",
    "summarize_archive_debt",
]
