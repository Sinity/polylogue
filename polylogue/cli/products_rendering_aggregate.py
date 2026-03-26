"""Aggregate archive-product CLI rendering."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering_support import emit_product_list


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


__all__ = [
    "render_day_session_summaries",
    "render_provider_analytics",
    "render_session_tag_rollups",
    "render_week_session_summaries",
]
