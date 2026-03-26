"""Session-scoped archive-product CLI rendering."""

from __future__ import annotations

import click

from polylogue.cli.products_rendering_support import emit_product_list


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


def render_session_enrichments(items: list[object], *, json_mode: bool) -> None:
    if json_mode:
        emit_product_list(key="session_enrichments", items=items)
        return
    if not items:
        click.echo("No session enrichments matched.")
        return
    click.echo(f"Session enrichments: {len(items)}\n")
    for item in items:
        enrichment = item.enrichment.model_dump(mode="json")
        click.echo(
            f"  {item.conversation_id} [{item.provider_name}] "
            f"{enrichment.get('refined_work_kind', '-') or '-'} "
            f"{item.title or '(untitled)'}"
        )
        click.echo(
            f"    confidence={enrichment.get('confidence', 0) or 0:.2f} "
            f"support={enrichment.get('support_level', '-') or '-'} "
            f"family={item.enrichment_provenance.enrichment_family}"
        )
        click.echo(
            f"    intent={enrichment.get('intent_summary', '-') or '-'} "
            f"outcome={enrichment.get('outcome_summary', '-') or '-'}"
        )
        blockers = enrichment.get("blockers", [])
        click.echo(
            f"    blockers={len(blockers)} "
            f"signals={', '.join(enrichment.get('support_signals', [])[:4]) or '-'}"
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


__all__ = [
    "render_session_enrichments",
    "render_session_phases",
    "render_session_profiles",
    "render_session_work_events",
    "render_work_threads",
]
