"""Action helpers for CLI query execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_contracts import QueryMutationSpec, result_date, result_id, result_provider, result_title

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import ConversationFilter
    from polylogue.archive.query.spec import ConversationQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import ConversationQueryRuntimeStore, TagStore


async def resolve_stream_target(
    repo: ConversationQueryRuntimeStore,
    filter_chain: ConversationFilter,
    selection: ConversationQuerySpec,
) -> str:
    """Resolve the conversation ID for a streaming query."""
    query_terms = selection.query_terms
    if selection.conversation_id:
        resolved = await repo.resolve_id(selection.conversation_id)
        if not resolved:
            click.echo(f"No conversation found matching: {selection.conversation_id}", err=True)
            raise SystemExit(2)
        return str(resolved)

    if selection.latest:
        summaries = await filter_chain.list_summaries()
        if not summaries:
            click.echo("No conversations matched.", err=True)
            raise SystemExit(2)
        return str(summaries[0].id)

    if selection.has_filters():
        summaries = await filter_chain.sort("date").limit(1).list_summaries()
        if not summaries:
            click.echo("No conversations matched filters.", err=True)
            raise SystemExit(2)
        return str(summaries[0].id)

    if query_terms:
        resolved = await repo.resolve_id(query_terms[0])
        if not resolved:
            click.echo(f"No conversation found matching: {query_terms[0]}", err=True)
            click.echo("Hint: use `list` to browse conversations, or --latest for most recent", err=True)
            raise SystemExit(2)
        return str(resolved)

    click.echo("--stream requires a specific conversation. Use --latest or specify an ID.", err=True)
    raise SystemExit(1)


async def apply_modifiers(
    env: AppEnv,
    results: Sequence[Conversation | ConversationSummary],
    mutation: QueryMutationSpec,
    repo: TagStore | None = None,
) -> None:
    """Apply metadata modifiers to matched conversations."""
    repo = repo or env.repository
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    dry_run = mutation.dry_run
    force = mutation.force
    count = len(results)

    operations: list[str] = []
    if mutation.set_meta:
        keys = [kv[0] for kv in mutation.set_meta]
        operations.append(f"set metadata: {', '.join(keys)}")
    if mutation.add_tags:
        operations.append(f"add tags: {', '.join(mutation.add_tags)}")

    op_desc = "; ".join(operations)

    if dry_run:
        click.echo(f"DRY-RUN: Would modify {count} conversation(s)")
        click.echo(f"Operations: {op_desc}")
        env.ui.console.print("\nSample of affected conversations:")
        for conv in results[:5]:
            title = result_title(conv)[:40]
            env.ui.console.print(f"  - {result_id(conv)[:24]} [{result_provider(conv)}] {title}")
        return

    if count > 10 and not force:
        click.echo(f"About to modify {count} conversations")
        click.echo(f"Operations: {op_desc}")
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    tags_added = 0
    meta_set = 0

    for conv in results:
        if mutation.set_meta:
            for kv in mutation.set_meta:
                key, value = kv[0], kv[1]
                await repo.update_metadata(result_id(conv), key, value)
                meta_set += 1

        if mutation.add_tags:
            for tag in mutation.add_tags:
                await repo.add_tag(result_id(conv), tag)
                tags_added += 1

    reports: list[str] = []
    if tags_added:
        reports.append(f"Added tags to {count} conversations")
    if meta_set:
        reports.append(f"Set {meta_set} metadata field(s)")

    for report in reports:
        click.echo(report)


async def delete_conversations(
    env: AppEnv,
    results: Sequence[Conversation | ConversationSummary],
    mutation: QueryMutationSpec,
    repo: ConversationQueryRuntimeStore | None = None,
) -> None:
    """Delete matched conversations."""
    from collections import Counter

    repo = repo or env.repository
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    dry_run = mutation.dry_run
    force = mutation.force
    count = len(results)

    provider_counts = Counter(result_provider(conv) for conv in results)
    dates = [dt for conv in results if (dt := result_date(conv)) is not None]
    date_min = min(dates) if dates else None
    date_max = max(dates) if dates else None

    def _print_breakdown() -> None:
        click.echo("  Providers:")
        for provider, pcount in provider_counts.most_common():
            click.echo(f"    {provider}: {pcount}")
        if date_min and date_max:
            fmt = "%Y-%m-%d"
            if date_min.date() == date_max.date():
                click.echo(f"  Date: {date_min.strftime(fmt)}")
            else:
                click.echo(f"  Date range: {date_min.strftime(fmt)} → {date_max.strftime(fmt)}")
        click.echo("  Sample:")
        for conv in results[:5]:
            title = result_title(conv)[:40]
            click.echo(f"    {result_id(conv)[:24]} [{result_provider(conv)}] {title}")
        if count > 5:
            click.echo(f"    ... and {count - 5} more")

    if dry_run:
        click.echo(f"DRY-RUN: Would delete {count} conversation(s)")
        _print_breakdown()
        return

    if count > 10 and not force:
        click.echo(f"About to DELETE {count} conversations:", err=True)
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return
    elif not force:
        click.echo(f"About to delete {count} conversation(s):")
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    deleted_count = 0
    for conv in results:
        if await repo.delete_conversation(result_id(conv)):
            deleted_count += 1

    click.echo(f"Deleted {deleted_count} conversation(s)")


def apply_transform(results: list[Conversation], transform: str) -> list[Conversation]:
    """Apply a transform to filter messages from conversations."""
    transformed = []
    for conv in results:
        proj = conv.project()
        if transform == "strip-tools":
            proj = proj.strip_tools()
        elif transform == "strip-thinking":
            proj = proj.strip_thinking()
        elif transform == "strip-all":
            proj = proj.strip_all()
        transformed.append(proj.execute())
    return transformed
