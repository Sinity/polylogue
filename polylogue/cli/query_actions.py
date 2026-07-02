"""Action helpers for CLI query execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import click

from polylogue.cli.query_contracts import QueryMutationSpec, result_date, result_id, result_origin, result_title

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.models import Session, SessionSummary
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.protocols import SessionQueryRuntimeStore, TagStore


async def resolve_stream_target(
    repo: SessionQueryRuntimeStore,
    filter_chain: SessionFilter,
    selection: SessionQuerySpec,
) -> str:
    """Resolve the session ID for a streaming query."""
    query_terms = selection.query_terms
    if selection.session_id:
        resolved = await repo.resolve_id(selection.session_id)
        if not resolved:
            click.echo(f"No session found matching: {selection.session_id}", err=True)
            raise SystemExit(2)
        return str(resolved)

    if selection.latest:
        summaries = await filter_chain.list_summaries()
        if not summaries:
            click.echo("No sessions matched.", err=True)
            raise SystemExit(2)
        return str(summaries[0].id)

    if selection.has_filters():
        summaries = await filter_chain.sort("date").limit(1).list_summaries()
        if not summaries:
            click.echo("No sessions matched filters.", err=True)
            raise SystemExit(2)
        return str(summaries[0].id)

    if query_terms:
        resolved = await repo.resolve_id(query_terms[0])
        if not resolved:
            click.echo(f"No session found matching: {query_terms[0]}", err=True)
            click.echo("Hint: use `list` to browse sessions, or --latest for most recent", err=True)
            raise SystemExit(2)
        return str(resolved)

    click.echo("--stream requires a specific session. Use --latest or specify an ID.", err=True)
    raise SystemExit(1)


async def apply_modifiers(
    env: AppEnv,
    results: Sequence[Session | SessionSummary],
    mutation: QueryMutationSpec,
    repo: TagStore | None = None,
) -> None:
    """Apply metadata modifiers to matched sessions.

    When the caller supplies a custom ``repo`` (test harnesses with
    fictional summaries, batch tools that have already validated
    existence), tag mutations are routed straight at the repo. With no
    custom repo, mutations go through ``env.polylogue.add_tag`` so the
    shared facade enforces idempotency and session-existence
    checks. Mixing the two paths against summaries that are not in
    the polylogue archive raises ``SessionNotFoundError`` (#1012).
    """
    if not results:
        env.ui.console.print("No sessions matched.")
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
        click.echo(f"DRY-RUN: Would modify {count} session(s)")
        click.echo(f"Operations: {op_desc}")
        env.ui.console.print("\nSample of affected sessions:")
        for conv in results[:5]:
            title = result_title(conv)[:40]
            env.ui.console.print(f"  - {result_id(conv)[:24]} [{result_origin(conv)}] {title}")
        return

    if count > 10 and not force:
        click.echo(f"About to modify {count} sessions")
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
                if repo is not None:
                    await repo.update_metadata(result_id(conv), key, value)
                else:
                    # Route through the facade so the metadata key is
                    # validated and the session existence check fires
                    # (#862). The facade returns a typed MetadataMutationResult.
                    await env.polylogue.set_metadata(result_id(conv), key, value)
                meta_set += 1

        if mutation.add_tags:
            for tag in mutation.add_tags:
                if repo is not None:
                    await repo.add_tag(result_id(conv), tag)
                    tags_added += 1
                else:
                    result = await env.polylogue.add_tag(result_id(conv), tag)
                    if result:
                        tags_added += 1

    reports: list[str] = []
    if tags_added:
        reports.append(f"Added tags to {count} sessions")
    if meta_set:
        reports.append(f"Set {meta_set} metadata field(s)")

    for report in reports:
        click.echo(report)


async def delete_sessions(
    env: AppEnv,
    results: Sequence[Session | SessionSummary],
    mutation: QueryMutationSpec,
    repo: SessionQueryRuntimeStore | None = None,
) -> None:
    """Delete matched sessions.

    Routes deletes through ``env.polylogue.delete_session_safe`` by
    default so resolution and idempotency stay centralized in
    :class:`ArchiveMutationsMixin` (#862). Tests/batch tools that supply a
    custom repository keep direct access for the same reason
    ``apply_modifiers`` accepts a custom tag repo.
    """
    from collections import Counter

    if not results:
        env.ui.console.print("No sessions matched.")
        return

    dry_run = mutation.dry_run
    force = mutation.force
    count = len(results)

    origin_counts = Counter(result_origin(conv) for conv in results)
    dates = [dt for conv in results if (dt := result_date(conv)) is not None]
    date_min = min(dates) if dates else None
    date_max = max(dates) if dates else None

    def _print_breakdown() -> None:
        click.echo("  Origins:")
        for origin, pcount in origin_counts.most_common():
            click.echo(f"    {origin}: {pcount}")
        if date_min and date_max:
            fmt = "%Y-%m-%d"
            if date_min.date() == date_max.date():
                click.echo(f"  Date: {date_min.strftime(fmt)}")
            else:
                click.echo(f"  Date range: {date_min.strftime(fmt)} → {date_max.strftime(fmt)}")
        click.echo("  Sample:")
        for conv in results[:5]:
            title = result_title(conv)[:40]
            click.echo(f"    {result_id(conv)[:24]} [{result_origin(conv)}] {title}")
        if count > 5:
            click.echo(f"    ... and {count - 5} more")

    if dry_run:
        click.echo(f"DRY-RUN: Would delete {count} session(s)")
        _print_breakdown()
        return

    if count > 10 and not force:
        click.echo(f"About to DELETE {count} sessions:", err=True)
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return
    elif not force:
        click.echo(f"About to delete {count} session(s):")
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    deleted_count = 0
    for conv in results:
        if repo is not None:
            if await repo.delete_session(result_id(conv)):
                deleted_count += 1
        else:
            result = await env.polylogue.delete_session_safe(result_id(conv))
            if result.outcome == "deleted":
                deleted_count += 1

    click.echo(f"Deleted {deleted_count} session(s)")
