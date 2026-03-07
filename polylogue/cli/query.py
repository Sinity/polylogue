"""Query execution for CLI query mode.

This module handles the execution of query-mode operations including:
- Filtering conversations via the filter chain API
- Formatting and outputting results
- Aggregation operations (--by-month, --by-provider, --by-tag)
- Modifier operations (--set, --add-tag, --delete)
- Streaming output for memory-efficient large conversation display
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.lib.formatting import format_conversation
from polylogue.lib.log import get_logger
from polylogue.lib.query_spec import ConversationQuerySpec, QuerySpecError

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.storage.repository import ConversationRepository


def _coerce_query_spec(params: dict[str, Any] | ConversationQuerySpec) -> ConversationQuerySpec:
    if isinstance(params, ConversationQuerySpec):
        return params
    return ConversationQuerySpec.from_params(params)


def _describe_filters(params: dict[str, Any] | ConversationQuerySpec) -> list[str]:
    """Build a human-readable list of active filters from params or spec."""
    return _coerce_query_spec(params).describe()


def _no_results(env: AppEnv, params: dict[str, Any] | ConversationQuerySpec, *, exit_code: int = 2) -> None:
    """Print a helpful no-results message and exit."""
    filters = _describe_filters(params)
    if filters:
        click.echo("No conversations matched filters:", err=True)
        for f in filters:
            click.echo(f"  {f}", err=True)
        click.echo("Hint: try broadening your filters or use --list to browse", err=True)
    else:
        click.echo("No conversations matched.", err=True)
    raise SystemExit(exit_code)


def execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Execute a query-mode command.

    Args:
        env: Application environment with UI and config
        params: Parsed CLI parameters
    """
    import asyncio

    asyncio.run(_async_execute_query(env, params))


async def _async_execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Async core of execute_query.

    Args:
        env: Application environment with UI and config
        params: Parsed CLI parameters
    """
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError
    from polylogue.storage.search_providers import create_vector_provider

    # Load config
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    repo = env.repository

    # Get vector provider (may be None if not configured)
    vector_provider = None
    try:
        vector_provider = create_vector_provider(config)
    except (ValueError, ImportError):
        pass  # Vector search not available
    except Exception as exc:
        logger.warning("Vector search setup failed: %s", exc)

    query_spec = ConversationQuerySpec.from_params(params)
    query_terms = query_spec.query_terms
    try:
        filter_chain = query_spec.build_filter(
            repo,
            vector_provider=vector_provider,
        )
    except QuerySpecError as exc:
        click.echo(f"Error: Cannot parse date: '{exc.value}'", err=True)
        click.echo(
            "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
            err=True,
        )
        raise SystemExit(1) from exc

    # Handle --count (uses SQL COUNT(*) fast path when possible)
    if params.get("count_only"):
        click.echo(await filter_chain.count())
        return

    # Execute query
    # Determine if we can use lightweight summaries
    list_mode = params.get("list_mode", False)
    use_summaries = (
        list_mode
        and not params.get("transform")
        and not params.get("dialogue_only")
        and not params.get("stream")
        and not params.get("set_meta")
        and not params.get("add_tag")
        and not params.get("delete_matched")
        and filter_chain.can_use_summaries()
    )

    if use_summaries:
        summary_results = await filter_chain.list_summaries()
        if not summary_results:
            _no_results(env, params)
        await _output_summary_list(env, summary_results, params, repo)
        return

    # Streaming path
    if params.get("stream"):
        # Use the filter chain to resolve the conversation ID, so that filters
        # (--provider, --since, --tag, etc.) are respected even in streaming mode.
        if query_spec.conversation_id:
            resolved = await repo.resolve_id(query_spec.conversation_id)
            if not resolved:
                click.echo(f"No conversation found matching: {query_spec.conversation_id}", err=True)
                raise SystemExit(2)
            full_id = str(resolved)
        elif query_spec.latest:
            # filter_chain already has .sort("date").limit(1) from line 172-173
            summaries = await filter_chain.list_summaries()
            if not summaries:
                _no_results(env, query_spec)
            full_id = str(summaries[0].id)
        elif query_spec.has_filters():
            # Filters active but no --latest: pick most recent match
            summaries = await filter_chain.sort("date").limit(1).list_summaries()
            if not summaries:
                _no_results(env, query_spec)
            full_id = str(summaries[0].id)
        else:
            # Try to resolve first query term as ID
            if query_terms:
                resolved = await repo.resolve_id(query_terms[0])
                if not resolved:
                    click.echo(f"No conversation found matching: {query_terms[0]}", err=True)
                    click.echo("Hint: use --list to browse conversations, or --latest for most recent", err=True)
                    raise SystemExit(2)
                full_id = str(resolved)
            else:
                click.echo("--stream requires a specific conversation. Use --latest or specify an ID.", err=True)
                raise SystemExit(1)

        # Warn about flags that are incompatible with streaming
        if params.get("transform"):
            click.echo("Warning: --transform is ignored in --stream mode (messages are streamed individually).", err=True)
        output_dest = params.get("output")
        if output_dest and output_dest != "stdout":
            click.echo(f"Warning: --output {output_dest} is ignored in --stream mode (output goes to stdout).", err=True)

        output_format = params.get("output_format") or "plaintext"
        stream_format = "json-lines" if output_format == "json" else output_format
        if stream_format not in ("plaintext", "markdown", "json-lines"):
            stream_format = "plaintext"

        await stream_conversation(
            env,
            repo,
            full_id,
            output_format=stream_format,
            dialogue_only=params.get("dialogue_only", False),
            message_limit=params.get("limit"),
        )
        return

    # Handle --stats early via SQL aggregation (avoids loading all messages)
    if params.get("stats_only"):
        await _output_stats_sql(env, filter_chain, repo)
        return

    results = await filter_chain.list()

    # Handle modifiers (write operations)
    if params.get("set_meta") or params.get("add_tag"):
        await _apply_modifiers(env, results, params, repo)
        return

    if params.get("delete_matched"):
        if not query_spec.has_filters():
            click.echo("Error: --delete requires at least one filter to prevent accidental deletion of the entire archive.", err=True)
            raise SystemExit(1)
        await _delete_conversations(env, results, params, repo)
        return

    # Apply transforms
    transform = params.get("transform")
    if transform:
        results = _apply_transform(results, transform)

    # Apply dialogue-only filter
    if params.get("dialogue_only"):
        results = [conv.dialogue_only() for conv in results]

    # Handle stats-by (still needs loaded conversations for grouping)
    if params.get("stats_by"):
        _output_stats_by(env, results, params["stats_by"])
        return

    # Handle --open (open in browser/editor)
    if params.get("open_result"):
        _open_result(env, results, params)
        return

    # Note: --stream is handled earlier via the streaming fast path
    # to avoid loading full conversations into memory

    # Regular output
    _output_results(env, results, params)


async def _apply_modifiers(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
    repo: ConversationRepository | None = None,
) -> None:
    """Apply metadata modifiers to matched conversations."""
    repo = repo or env.repository
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    dry_run = params.get("dry_run", False)
    force = params.get("force", False)
    count = len(results)

    # Build description of what will be modified
    operations: list[str] = []
    if params.get("set_meta"):
        keys = [kv[0] for kv in params["set_meta"]]
        operations.append(f"set metadata: {', '.join(keys)}")
    if params.get("add_tag"):
        operations.append(f"add tags: {', '.join(params['add_tag'])}")

    op_desc = "; ".join(operations)

    # Dry-run mode: show preview and exit
    if dry_run:
        click.echo(f"DRY-RUN: Would modify {count} conversation(s)")
        click.echo(f"Operations: {op_desc}")
        env.ui.console.print("\nSample of affected conversations:")
        for conv in results[:5]:
            title = conv.display_title[:40] if conv.display_title else conv.id[:20]
            env.ui.console.print(f"  - {conv.id[:24]} [{conv.provider}] {title}")
        return

    # Confirmation for bulk operations (>10 items)
    if count > 10 and not force:
        click.echo(f"About to modify {count} conversations")
        click.echo(f"Operations: {op_desc}")
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    # Track counts for reporting
    tags_added = 0
    meta_set = 0

    # Apply modifiers
    for conv in results:
        if params.get("set_meta"):
            for kv in params["set_meta"]:
                key, value = kv[0], kv[1]
                await repo.update_metadata(str(conv.id), key, value)
                meta_set += 1

        if params.get("add_tag"):
            for tag in params["add_tag"]:
                await repo.add_tag(str(conv.id), tag)
                tags_added += 1

    # Report results
    reports: list[str] = []
    if tags_added:
        reports.append(f"Added tags to {count} conversations")
    if meta_set:
        reports.append(f"Set {meta_set} metadata field(s)")

    for report in reports:
        click.echo(report)


async def _delete_conversations(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
    repo: ConversationRepository | None = None,
) -> None:
    """Delete matched conversations."""
    from collections import Counter

    repo = repo or env.repository
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    dry_run = params.get("dry_run", False)
    force = params.get("force", False)
    count = len(results)

    # Build breakdown summary for preview/confirmation
    provider_counts = Counter(conv.provider for conv in results)
    dates = [conv.created_at for conv in results if conv.created_at is not None]
    date_min = min(dates) if dates else None
    date_max = max(dates) if dates else None

    def _print_breakdown() -> None:
        """Print provider and date breakdown."""
        # Provider breakdown
        click.echo("  Providers:")
        for provider, pcount in provider_counts.most_common():
            click.echo(f"    {provider}: {pcount}")
        # Date range
        if date_min and date_max:
            fmt = "%Y-%m-%d"
            if date_min.date() == date_max.date():
                click.echo(f"  Date: {date_min.strftime(fmt)}")
            else:
                click.echo(f"  Date range: {date_min.strftime(fmt)} → {date_max.strftime(fmt)}")
        # Sample
        click.echo("  Sample:")
        for conv in results[:5]:
            title = conv.display_title[:40] if conv.display_title else conv.id[:20]
            click.echo(f"    {conv.id[:24]} [{conv.provider}] {title}")
        if count > 5:
            click.echo(f"    ... and {count - 5} more")

    # Dry-run mode: show preview and exit
    if dry_run:
        click.echo(f"DRY-RUN: Would delete {count} conversation(s)")
        _print_breakdown()
        return

    # Confirmation for bulk operations (>10 items)
    if count > 10 and not force:
        click.echo(f"About to DELETE {count} conversations:", err=True)
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    # Individual confirmation if not bulk but not forced
    elif not force:
        click.echo(f"About to delete {count} conversation(s):")
        _print_breakdown()
        if not env.ui.confirm("Proceed?", default=False):
            env.ui.console.print("Aborted.")
            return

    deleted_count = 0
    for conv in results:
        if await repo.delete_conversation(str(conv.id)):
            deleted_count += 1

    click.echo(f"Deleted {deleted_count} conversation(s)")


def _apply_transform(results: list[Conversation], transform: str) -> list[Conversation]:
    """Apply a transform to filter messages from conversations.

    Args:
        results: List of conversations to transform
        transform: Transform to apply: 'strip-tools', 'strip-thinking', or 'strip-all'

    Returns:
        List of transformed conversations with filtered messages
    """
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


async def _output_stats_sql(
    env: AppEnv,
    filter_chain: ConversationFilter,
    repo: ConversationRepository,
) -> None:
    """Output statistics using SQL aggregation (no full message loading).

    Two paths depending on whether filters are active:
    - Filtered: uses temp table join for per-role/word breakdown (fast for subsets)
    - Unfiltered: uses lightweight COUNT(*) and conversations table (instant for any DB size)
    """
    from datetime import datetime, timezone

    # Determine if we have active filters (not just "everything")
    has_filters = bool(filter_chain._describe_active_filters())

    if has_filters:
        # Filtered path: resolve matching IDs, then aggregate their messages
        summaries = await filter_chain.list_summaries() if filter_chain.can_use_summaries() else None
        if summaries is not None:
            if not summaries:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = [str(s.id) for s in summaries]
            conv_count = len(conv_ids)
        else:
            conv_count = await filter_chain.count()
            if conv_count == 0:
                env.ui.console.print("No conversations matched.")
                return
            conv_ids = None
    else:
        # Unfiltered: pass None to trigger fast path
        conv_ids = None
        conv_count = await filter_chain.count()
        if conv_count == 0:
            env.ui.console.print("No conversations in archive.")
            return

    stats = await repo.aggregate_message_stats(conv_ids)

    date_range = ""
    if stats["min_sort_key"] and stats["max_sort_key"]:
        min_date = datetime.fromtimestamp(stats["min_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_date = datetime.fromtimestamp(stats["max_sort_key"], tz=timezone.utc).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    out = env.ui.console.print
    out(f"\nConversations: {conv_count:,}\n")

    # Message stats — role breakdown available only for filtered queries
    if stats["user"] or stats["assistant"]:
        out(
            f"Messages: {stats['total']:,} total "
            f"({stats['user']:,} user, {stats['assistant']:,} assistant)"
        )
    else:
        out(f"Messages: {stats['total']:,}")

    if stats["words_approx"]:
        out(f"Words: ~{stats['words_approx']:,}")

    if stats.get("providers"):
        provider_parts = [f"{name} ({cnt:,})" for name, cnt in stats["providers"].items()]
        out(f"Providers: {', '.join(provider_parts)}")

    out(f"Attachments: {stats['attachments']:,}")
    if date_range:
        out(f"Date range: {date_range}")


def _output_stats_by(env: AppEnv, results: list[Conversation], dimension: str) -> None:
    """Output statistics grouped by a dimension."""
    from collections import defaultdict

    from rich.table import Table

    from polylogue.lib.theme import provider_color

    if not results:
        env.ui.console.print("No conversations matched.")
        return

    groups: dict[str, list[Conversation]] = defaultdict(list)
    for conv in results:
        if dimension == "provider":
            key = conv.provider or "unknown"
        elif dimension == "month":
            dt = conv.display_date
            key = dt.strftime("%Y-%m") if dt else "unknown"
        elif dimension == "year":
            dt = conv.display_date
            key = dt.strftime("%Y") if dt else "unknown"
        elif dimension == "day":
            dt = conv.display_date
            key = dt.strftime("%Y-%m-%d") if dt else "unknown"
        else:
            key = "all"
        groups[key].append(conv)

    # Sort groups: by date desc for temporal, alphabetically for provider
    if dimension in {"month", "year", "day"}:
        sorted_keys = sorted(groups.keys(), reverse=True)
    else:
        sorted_keys = sorted(groups.keys())

    env.ui.console.print(f"\nMatched: {len(results)} conversations (by {dimension})\n")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Messages", justify="right")
    table.add_column("Words", justify="right")

    total_convs = 0
    total_msgs = 0
    total_words = 0

    for key in sorted_keys:
        convs = groups[key]
        n_convs = len(convs)
        n_msgs = sum(len(c.messages) for c in convs)
        n_words = sum(sum(m.word_count for m in c.messages) for c in convs)
        # Color provider names with their brand color
        label = f"[{provider_color(key).hex}]{key}[/]" if dimension == "provider" else key
        table.add_row(label, f"{n_convs:,}", f"{n_msgs:,}", f"{n_words:,}")
        total_convs += n_convs
        total_msgs += n_msgs
        total_words += n_words

    table.add_section()
    table.add_row("[bold]TOTAL[/]", f"[bold]{total_convs:,}[/]", f"[bold]{total_msgs:,}[/]", f"[bold]{total_words:,}[/]")

    env.ui.console.print(table)


def _output_results(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Output query results."""
    if not results:
        _no_results(env, params)

    output_format = params.get("output_format", "markdown")
    output_dest = params.get("output", "stdout")
    list_mode = params.get("list_mode", False)
    fields = params.get("fields")

    # Parse output destinations
    destinations = [d.strip() for d in output_dest.split(",")] if output_dest else ["stdout"]

    # Single result and not list mode: show content
    if len(results) == 1 and not list_mode:
        conv = results[0]
        # Use Rich rendering for interactive terminal markdown display
        if output_format == "markdown" and destinations == ["stdout"] and not env.ui.plain:
            _render_conversation_rich(env, conv)
            return
        content = format_conversation(conv, output_format, fields)
        _send_output(env, content, destinations, output_format, conv)
        return

    # Multiple results or list mode: show list
    content = _format_list(results, output_format, fields)
    _send_output(env, content, destinations, output_format, None)


def _format_list(
    results: list[Conversation],
    output_format: str,
    fields: str | None,
) -> str:
    """Format a list of conversations for output."""
    from polylogue.lib.formatting import _conv_to_dict

    if output_format == "json":
        return json.dumps([_conv_to_dict(c, fields) for c in results], indent=2)
    elif output_format == "yaml":
        import yaml  # type: ignore[import-untyped]

        return str(yaml.dump([_conv_to_dict(c, fields) for c in results], default_flow_style=False, allow_unicode=True))
    elif output_format == "csv":
        return _conv_to_csv(results)
    else:
        # Plain text table (for piping — no Rich markup)
        lines = []
        for conv in results:
            date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
            raw_title = conv.display_title or conv.id[:20]
            title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
            msg_count = len(conv.messages)
            lines.append(f"{conv.id[:24]:24s}  {date:10s}  {conv.provider:12s}  {title} ({msg_count} msgs)")
        return "\n".join(lines)


async def _output_summary_list(
    env: AppEnv,
    summaries: list[ConversationSummary],
    params: dict[str, Any],
    repo: ConversationRepository | None = None,
) -> None:
    """Output a list of conversation summaries (memory-efficient).

    This is the fast path for --list mode that doesn't load messages.
    When a repo is provided, batch-fetches message counts for display.
    """
    output_format = params.get("output_format", "text")

    # Batch-fetch message counts if repo available (single SQL query)
    msg_counts: dict[str, int] = {}
    if repo:
        ids = [str(s.id) for s in summaries]
        msg_counts = await repo.get_message_counts_batch(ids)

    fields = params.get("fields")
    selected: set[str] | None = None
    if fields:
        selected = {f.strip() for f in fields.split(",")}

    if output_format == "json":
        data = [
            {
                "id": str(s.id),
                "provider": s.provider,
                "title": s.display_title,
                "date": s.display_date.isoformat() if s.display_date else None,
                "tags": s.tags,
                "summary": s.summary,
                "messages": msg_counts.get(str(s.id), 0),
            }
            for s in summaries
        ]
        if selected:
            data = [{k: v for k, v in d.items() if k in selected} for d in data]
        click.echo(json.dumps(data, indent=2))
    elif output_format == "yaml":
        import yaml

        data = [
            {
                "id": str(s.id),
                "provider": s.provider,
                "title": s.display_title,
                "date": s.display_date.isoformat() if s.display_date else None,
                "tags": s.tags,
                "messages": msg_counts.get(str(s.id), 0),
            }
            for s in summaries
        ]
        if selected:
            data = [{k: v for k, v in d.items() if k in selected} for d in data]
        click.echo(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    elif output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "date", "provider", "title", "messages", "tags", "summary"])
        for s in summaries:
            date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
            tags_str = ",".join(s.tags) if s.tags else ""
            writer.writerow([
                str(s.id),
                date,
                s.provider,
                s.display_title or "",
                msg_counts.get(str(s.id), 0),
                tags_str,
                s.summary or "",
            ])
        click.echo(buf.getvalue().rstrip())
    else:
        # Default table format
        if env.ui.plain:
            # Plain text table (no Rich markup — safe for piping and non-TTY)
            for s in summaries:
                date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
                raw_title = s.display_title or str(s.id)[:20]
                title = (raw_title[:47] + "...") if len(raw_title) > 50 else raw_title
                count = msg_counts.get(str(s.id), 0)
                click.echo(f"{str(s.id)[:24]:24s}  {date:10s}  {s.provider:12s}  {title} ({count} msgs)")
            return

        # Rich table format (interactive terminals)
        from rich.table import Table
        from rich.text import Text

        from polylogue.lib.theme import provider_color

        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False, show_edge=False)
        table.add_column("ID", style="dim", max_width=24, no_wrap=True)
        table.add_column("Date", style="dim")
        table.add_column("Provider")
        table.add_column("Title", ratio=1)
        table.add_column("Msgs", justify="right")

        for s in summaries:
            date = s.display_date.strftime("%Y-%m-%d") if s.display_date else ""
            raw_title = s.display_title or str(s.id)[:20]
            title = (raw_title[:60] + "...") if len(raw_title) > 63 else raw_title
            count = msg_counts.get(str(s.id), 0)
            pc = provider_color(s.provider)
            prov_text = Text(s.provider, style=pc.hex)
            table.add_row(str(s.id)[:24], date, prov_text, title, str(count))

        env.ui.console.print(table)


def _conv_to_csv(results: list[Conversation]) -> str:
    """Convert conversations to CSV string."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "provider", "title", "messages", "words", "tags", "summary"])

    for conv in results:
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else ""
        tags_str = ",".join(conv.tags) if conv.tags else ""
        writer.writerow(
            [
                str(conv.id),
                date,
                conv.provider,
                conv.display_title or "",
                len(conv.messages),
                sum(m.word_count for m in conv.messages),
                tags_str,
                conv.summary or "",
            ]
        )

    return output.getvalue()


def _render_conversation_rich(env: AppEnv, conv: Conversation) -> None:
    """Render a conversation with Rich role colors and thinking block styling.

    Only used for interactive terminal display — never for piped/file output.
    """
    from rich import box
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    from polylogue.lib.theme import THINKING_STYLE, provider_color, role_color

    console = env.ui.console

    # Header
    title = conv.display_title or conv.id
    pc = provider_color(conv.provider)
    header = Text()
    header.append(title, style="bold")
    if conv.display_date:
        header.append(f"  {conv.display_date.strftime('%Y-%m-%d %H:%M')}", style="dim")
    header.append(f"  [{pc.hex}]{conv.provider}[/{pc.hex}]")
    console.print(header)
    console.print()

    for msg in conv.messages:
        if not msg.text:
            continue

        role = msg.role or "unknown"
        rc = role_color(role)

        # Check for thinking blocks
        is_thinking = msg.is_thinking

        if is_thinking:
            # Thinking blocks: dimmed, italic, with 💭 indicator
            content = Text(msg.text[:500], style=THINKING_STYLE["rich_style"])
            if len(msg.text) > 500:
                content.append(f"\n... ({len(msg.text):,} chars)", style="dim")
            panel = Panel(
                content,
                title=f"{THINKING_STYLE['icon']} Thinking",
                title_align="left",
                border_style=THINKING_STYLE["border_color"],
                box=box.SIMPLE,
                padding=(0, 1),
            )
            console.print(panel)
        else:
            # Regular messages: role-colored border
            role_label = role.capitalize()
            try:
                md = Markdown(msg.text)
                panel = Panel(
                    md,
                    title=f"[{rc.label}]{role_label}[/{rc.label}]",
                    title_align="left",
                    border_style=rc.hex,
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                console.print(panel)
            except Exception:
                # Fallback for markdown parsing failures
                console.print(f"[{rc.label}]{role_label}:[/{rc.label}] {msg.text[:200]}")

        console.print()


# =============================================================================
# Streaming Output (Memory-Efficient)
# =============================================================================


async def stream_conversation(
    env: AppEnv,
    repo: ConversationRepository,
    conversation_id: str,
    *,
    output_format: str = "plaintext",
    dialogue_only: bool = False,
    message_limit: int | None = None,
) -> int:
    """Stream conversation messages to stdout without buffering.

    This is the memory-efficient alternative to _output_results() for
    displaying large conversations. Messages are written to stdout one
    at a time, keeping memory usage constant regardless of conversation size.

    Args:
        env: Application environment with UI
        repo: ConversationRepository for data access
        conversation_id: ID of conversation to stream
        output_format: Output format - "plaintext", "markdown", or "json-lines"
        dialogue_only: If True, only stream user/assistant messages
        message_limit: Maximum messages to output. None = no limit.

    Returns:
        Number of messages streamed
    """
    # Get conversation metadata for header
    conv_record = await repo.get_conversation(conversation_id)
    if not conv_record:
        click.echo(f"Conversation not found: {conversation_id}", err=True)
        raise SystemExit(1)

    # Get stats for progress indication
    stats = await repo.get_conversation_stats(conversation_id)

    # Print header based on format
    if output_format == "markdown":
        title = conv_record.title or conversation_id[:24]
        sys.stdout.write(f"# {title}\n\n")
        if dialogue_only and stats:
            sys.stdout.write(f"_Showing {stats['dialogue_messages']} dialogue messages")
            if message_limit:
                sys.stdout.write(f" (limit: {message_limit})")
            sys.stdout.write(f" of {stats['total_messages']} total_\n\n")
        sys.stdout.flush()
    elif output_format == "json-lines":
        # JSONL header with metadata
        header = {
            "type": "header",
            "conversation_id": conversation_id,
            "title": conv_record.title,
            "dialogue_only": dialogue_only,
            "message_limit": message_limit,
            "stats": stats,
        }
        sys.stdout.write(json.dumps(header) + "\n")
        sys.stdout.flush()

    # Stream messages
    count = 0
    async for msg in repo.iter_messages(
        conversation_id,
        dialogue_only=dialogue_only,
        limit=message_limit,
    ):
        _write_message_streaming(msg, output_format)
        count += 1

    # Print footer
    if output_format == "markdown":
        sys.stdout.write(f"\n---\n_Streamed {count} messages_\n")
        sys.stdout.flush()
    elif output_format == "json-lines":
        footer = {"type": "footer", "message_count": count}
        sys.stdout.write(json.dumps(footer) + "\n")
        sys.stdout.flush()

    return count


def _write_message_streaming(msg: Message, output_format: str) -> None:
    """Write a single message to stdout in streaming mode.

    Args:
        msg: Message to write
        output_format: Format - "plaintext", "markdown", or "json-lines"
    """
    if output_format == "plaintext":
        # Simple format: [ROLE] text
        role_label = (msg.role or "unknown").upper().replace("[", "").replace("]", "")
        if msg.text:
            sys.stdout.write(f"[{role_label}]\n{msg.text}\n\n")
        sys.stdout.flush()

    elif output_format == "markdown":
        # Markdown format with headers — skip empty-text messages (tool progress, etc.)
        if msg.text:
            role_label = (msg.role or "unknown").capitalize()
            sys.stdout.write(f"## {role_label}\n\n{msg.text}\n\n")
            sys.stdout.flush()

    elif output_format == "json-lines":
        # JSONL format - one JSON object per line
        record = {
            "type": "message",
            "id": msg.id,
            "role": msg.role,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "text": msg.text,
            "word_count": msg.word_count,
        }
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _send_output(
    env: AppEnv,
    content: str,
    destinations: list[str],
    output_format: str,
    conv: Conversation | None,
) -> None:
    """Send output to specified destinations."""
    for dest in destinations:
        if dest == "stdout":
            click.echo(content)
        elif dest == "browser":
            _open_in_browser(env, content, output_format, conv)
        elif dest == "clipboard":
            _copy_to_clipboard(env, content)
        else:
            # Treat as file path
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            env.ui.console.print(f"Wrote to {path}")


def _open_in_browser(
    env: AppEnv,
    content: str,
    output_format: str,
    conv: Conversation | None,
) -> None:
    """Open content in browser."""
    import tempfile
    import webbrowser

    # Ensure HTML format for browser
    if output_format != "html":
        if conv:  # noqa: SIM108
            from polylogue.lib.formatting import _conv_to_html

            content = _conv_to_html(conv)
        else:
            from html import escape as _html_escape

            content = f"<html><body><pre>{_html_escape(content)}</pre></body></html>"

    # Write to temp file and open
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(content)
        temp_path = f.name

    webbrowser.open(f"file://{temp_path}")
    env.ui.console.print(f"Opened in browser: {temp_path}")


def _copy_to_clipboard(env: AppEnv, content: str) -> None:
    """Copy content to clipboard."""
    import subprocess

    # Try multiple clipboard commands
    clipboard_cmds = [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],  # macOS
        ["clip"],  # Windows
    ]

    for cmd in clipboard_cmds:
        try:
            subprocess.run(
                cmd,
                input=content.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            env.ui.console.print("Copied to clipboard.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    click.echo("Could not copy to clipboard (no clipboard tool found).", err=True)


def _open_result(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Open result in browser or editor."""
    if not results:
        env.ui.console.print("No conversations matched.")
        raise SystemExit(2)

    # Get the first/only result
    conv = results[0]

    # Try to find rendered file for this conversation
    from polylogue.cli.helpers import load_effective_config

    try:
        config = load_effective_config(env)
    except Exception as exc:
        logger.warning(
            "Config load failed, falling back to defaults: %s", exc
        )
        config = None

    render_root = None
    if config and hasattr(config, "render_root") and config.render_root:
        render_root = Path(config.render_root)
    else:
        # Try environment variable
        import os

        render_root_env = os.environ.get("POLYLOGUE_RENDER_ROOT")
        if render_root_env:
            render_root = Path(render_root_env)

    if not render_root or not render_root.exists():
        click.echo("No rendered outputs found.", err=True)
        click.echo("Run 'polylogue run' first to render conversations.", err=True)
        raise SystemExit(1)

    # Search for rendered file matching this conversation ID
    conv_id_short = str(conv.id)[:8] if conv.id else ""
    html_files = list(render_root.rglob(f"*{conv_id_short}*/conversation.html"))
    md_files = list(render_root.rglob(f"*{conv_id_short}*/conversation.md"))

    # Prefer HTML, fallback to MD
    render_file = None
    if html_files:
        render_file = html_files[0]
    elif md_files:
        render_file = md_files[0]
    else:
        # Fallback: find most recent render
        from polylogue.cli.helpers import latest_render_path

        render_file = latest_render_path(render_root)

    if not render_file:
        click.echo("No rendered output found for this conversation.", err=True)
        click.echo("Run 'polylogue run' to render conversations.", err=True)
        raise SystemExit(1)

    # Open in browser
    import webbrowser

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")
