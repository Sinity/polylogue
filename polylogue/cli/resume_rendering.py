"""Plain terminal rendering for resume briefs."""

from __future__ import annotations

import click

from polylogue.archive_resume import ResumeBrief


def _line(label: str, value: object | None) -> None:
    if value is None or value == "" or value == ():
        return
    if isinstance(value, (list, tuple)):
        rendered = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        rendered = ", ".join(f"{key}={value[key]}" for key in sorted(value))
    else:
        rendered = str(value)
    if rendered:
        click.echo(f"  {label}: {rendered}")


def render_resume_brief(brief: ResumeBrief) -> None:
    """Render a concise human handoff from a typed resume brief."""
    facts = brief.facts
    inferences = brief.inferences

    click.echo("Resume Brief")
    click.echo(f"Session: {facts.conversation_id}")
    _line("Title", facts.title)
    _line("Provider", facts.provider_name)
    _line("Updated", facts.updated_at)
    _line("Messages", facts.message_count)
    _line("Parent", facts.parent_id)
    _line("Branch", facts.branch_type)
    if facts.last_message is not None:
        _line("Last message", f"{facts.last_message.role}: {facts.last_message.preview}")

    click.echo("\nFacts")
    _line("Tags", facts.tags)
    _line("Repos", facts.repo_paths)
    _line("CWDs", facts.cwd_paths)
    _line("Branches", facts.branch_names)
    _line("Files touched", facts.file_paths_touched[:6])
    _line("Tool categories", facts.tool_categories)

    click.echo("\nInferred State")
    _line("Intent", inferences.intent_summary)
    _line("Outcome", inferences.outcome_summary)
    _line("Blockers", inferences.blockers)
    _line("Support", inferences.support_level)
    _line("Repo names", inferences.repo_names)
    _line("Auto tags", inferences.auto_tags)
    for event in inferences.work_events:
        click.echo(f"  Work event: {event.kind} - {event.summary}")
    for phase in inferences.phases:
        click.echo(
            f"  Phase {phase.phase_index}: messages {phase.message_range[0]}-{phase.message_range[1]} "
            f"support={phase.support_level}"
        )
    if inferences.work_thread is not None:
        thread = inferences.work_thread
        click.echo(
            f"  Work thread: {thread.thread_id} sessions={thread.session_count} repo={thread.dominant_repo or '-'}"
        )

    if brief.related_sessions:
        click.echo("\nRelated Sessions")
        for related in brief.related_sessions:
            title = f" - {related.title}" if related.title else ""
            click.echo(f"  {related.relation}: {related.conversation_id}{title}")

    if brief.uncertainties:
        click.echo("\nUncertainties")
        for uncertainty in brief.uncertainties:
            click.echo(f"  {uncertainty.source}: {uncertainty.detail}")

    click.echo("\nNext Steps")
    for step in brief.next_steps:
        click.echo(f"  - {step}")


__all__ = ["render_resume_brief"]
