"""Query execution for CLI query mode.

This module handles the execution of query-mode operations including:
- Filtering conversations via the filter chain API
- Formatting and outputting results
- Aggregation operations (--by-month, --by-provider, --by-tag)
- Modifier operations (--set, --add-tag, --delete)
- Streaming output for memory-efficient large conversation display
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.storage.repository import ConversationRepository


def execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Execute a query-mode command.

    Args:
        env: Application environment with UI and config
        params: Parsed CLI parameters
    """
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError
    from polylogue.services import get_repository
    from polylogue.storage.search_providers import create_vector_provider

    # Load config
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    # Build repository and vector provider
    conv_repo = get_repository()

    # Get vector provider (may be None if not configured)
    vector_provider = None
    with contextlib.suppress(ValueError, ImportError):
        vector_provider = create_vector_provider(config)

    # Create filter chain with vector provider
    from polylogue.lib.filters import ConversationFilter

    filter_chain = ConversationFilter(conv_repo, vector_provider=vector_provider)

    # Apply query terms (positional args)
    query_terms = params.get("query", ())
    for term in query_terms:
        filter_chain = filter_chain.contains(term)

    # Apply --contains
    for term in params.get("contains", ()):
        filter_chain = filter_chain.contains(term)

    # Apply --exclude-text
    for term in params.get("exclude_text", ()):
        filter_chain = filter_chain.no_contains(term)

    # Apply --provider (comma-separated)
    if params.get("provider"):
        providers = [p.strip() for p in params["provider"].split(",")]
        filter_chain = filter_chain.provider(*providers)

    # Apply --exclude-provider (comma-separated)
    if params.get("exclude_provider"):
        excluded = [p.strip() for p in params["exclude_provider"].split(",")]
        filter_chain = filter_chain.no_provider(*excluded)

    # Apply --tag (comma-separated)
    if params.get("tag"):
        tags = [t.strip() for t in params["tag"].split(",")]
        filter_chain = filter_chain.tag(*tags)

    # Apply --exclude-tag (comma-separated)
    if params.get("exclude_tag"):
        excluded = [t.strip() for t in params["exclude_tag"].split(",")]
        filter_chain = filter_chain.no_tag(*excluded)

    # Apply --title
    if params.get("title"):
        filter_chain = filter_chain.title(params["title"])

    # Apply --has
    for content_type in params.get("has_type", ()):
        filter_chain = filter_chain.has(content_type)

    # Apply --since
    if params.get("since"):
        filter_chain = filter_chain.since(params["since"])

    # Apply --until
    if params.get("until"):
        filter_chain = filter_chain.until(params["until"])

    # Apply --latest (= --sort date --limit 1)
    if params.get("latest"):
        filter_chain = filter_chain.sort("date").limit(1)

    # Apply --sort
    if params.get("sort"):
        filter_chain = filter_chain.sort(params["sort"])

    # Apply --reverse
    if params.get("reverse"):
        filter_chain = filter_chain.reverse()

    # Apply --limit
    if params.get("limit"):
        filter_chain = filter_chain.limit(params["limit"])

    # Apply --sample
    if params.get("sample"):
        filter_chain = filter_chain.sample(params["sample"])

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
        summary_results = filter_chain.list_summaries()
        if not summary_results:
            env.ui.console.print("No conversations matched.")
            raise SystemExit(2)
        _output_summary_list(env, summary_results, params)
        return

    # Streaming path
    if params.get("stream"):
        if params.get("latest"):
            summaries = conv_repo.list_summaries(limit=1)
            if not summaries:
                env.ui.console.print("No conversations in archive.")
                raise SystemExit(2)
            full_id = str(summaries[0].id)
        else:
            # Try to resolve first query term as ID
            if query_terms:
                full_id = conv_repo.resolve_id(query_terms[0])
                if not full_id:
                    env.ui.console.print(f"[red]No conversation found matching: {query_terms[0]}[/red]")
                    raise SystemExit(2)
            else:
                env.ui.console.print(
                    "[yellow]--stream requires a specific conversation. Use --latest or specify an ID.[/yellow]"
                )
                raise SystemExit(1)

        output_format = params.get("output_format") or "plaintext"
        stream_format = "json-lines" if output_format == "json" else output_format
        if stream_format not in ("plaintext", "markdown", "json-lines"):
            stream_format = "plaintext"

        from polylogue.cli.query import stream_conversation

        stream_conversation(
            env,
            conv_repo,
            full_id,
            output_format=stream_format,
            dialogue_only=params.get("dialogue_only", False),
        )
        return

    results = filter_chain.list()

    # Handle modifiers (write operations)
    if params.get("set_meta") or params.get("add_tag"):
        _apply_modifiers(env, results, params)
        return

    if params.get("delete_matched"):
        _delete_conversations(env, results, params)
        return

    # Apply transforms
    transform = params.get("transform")
    if transform:
        results = _apply_transform(results, transform)

    # Apply dialogue-only filter
    if params.get("dialogue_only"):
        results = [conv.dialogue_only() for conv in results]

    # Handle stats-only output
    if params.get("stats_only"):
        _output_stats(env, results)
        return

    # Handle --open (open in browser/editor)
    if params.get("open_result"):
        _open_result(env, results, params)
        return

    # Note: --stream is handled earlier via the streaming fast path
    # to avoid loading full conversations into memory

    # Regular output
    _output_results(env, results, params)


def _apply_modifiers(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Apply metadata modifiers to matched conversations."""
    from polylogue.services import get_repository

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
        env.ui.console.print(f"[yellow]DRY-RUN: Would modify {count} conversation(s)[/yellow]")
        env.ui.console.print(f"Operations: {op_desc}")
        env.ui.console.print("\nSample of affected conversations:")
        for conv in results[:5]:
            title = conv.display_title[:40] if conv.display_title else conv.id[:20]
            env.ui.console.print(f"  - {conv.id[:24]} [{conv.provider}] {title}")
        return

    # Confirmation for bulk operations (>10 items)
    if count > 10 and not force:
        env.ui.console.print(f"[yellow]About to modify {count} conversations[/yellow]")
        env.ui.console.print(f"Operations: {op_desc}")
        env.ui.console.print("\nUse --force to skip this prompt, or --dry-run to preview.")
        raise SystemExit(1)

    # Load repository
    repo = get_repository()

    # Track counts for reporting
    tags_added = 0
    meta_set = 0

    # Apply modifiers
    for conv in results:
        if params.get("set_meta"):
            for kv in params["set_meta"]:
                key, value = kv[0], kv[1]
                repo.update_metadata(str(conv.id), key, value)
                meta_set += 1

        if params.get("add_tag"):
            for tag in params["add_tag"]:
                repo.add_tag(str(conv.id), tag)
                tags_added += 1

    # Report results
    reports: list[str] = []
    if tags_added:
        reports.append(f"Added tags to {count} conversations")
    if meta_set:
        reports.append(f"Set {meta_set} metadata field(s)")

    for report in reports:
        env.ui.console.print(f"[green]{report}[/green]")


def _delete_conversations(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Delete matched conversations."""
    from polylogue.services import get_repository

    if not results:
        env.ui.console.print("No conversations matched.")
        return

    dry_run = params.get("dry_run", False)
    force = params.get("force", False)
    count = len(results)

    # Dry-run mode: show preview and exit
    if dry_run:
        env.ui.console.print(f"[yellow]DRY-RUN: Would delete {count} conversation(s)[/yellow]")
        env.ui.console.print("\nSample of conversations to be deleted:")
        for conv in results[:5]:
            title = conv.display_title[:40] if conv.display_title else conv.id[:20]
            env.ui.console.print(f"  - {conv.id[:24]} [{conv.provider}] {title}")
        return

    # Confirmation for bulk operations (>10 items)
    if count > 10 and not force:
        env.ui.console.print(f"[red]About to DELETE {count} conversations[/red]")
        env.ui.console.print("\nUse --force to skip this prompt, or --dry-run to preview.")
        raise SystemExit(1)

    # Individual confirmation if not bulk but not forced
    if not force and not env.ui.confirm("Are you sure you want to delete these conversations?", default=False):
        env.ui.console.print("Aborted.")
        return

    # Load repository
    repo = get_repository()

    deleted_count = 0
    for conv in results:
        if repo.delete_conversation(str(conv.id)):
            deleted_count += 1

    env.ui.console.print(f"[green]Deleted {deleted_count} conversation(s)[/green]")


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


def _output_stats(env: AppEnv, results: list[Conversation]) -> None:
    """Output statistics for matched conversations."""
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    total_messages = sum(len(c.messages) for c in results)
    total_words = sum(sum(m.word_count for m in c.messages) for c in results)
    user_messages = sum(1 for c in results for m in c.messages if m.role == "user")
    assistant_messages = sum(1 for c in results for m in c.messages if m.role == "assistant")
    thinking_traces = sum(1 for c in results for m in c.messages if m.is_thinking)
    tool_calls = sum(1 for c in results for m in c.messages if m.is_tool_use)
    attachments = sum(len(m.attachments) for c in results for m in c.messages)

    dates = [c.updated_at for c in results if c.updated_at]
    date_range = ""
    if dates:
        min_date = min(dates).strftime("%Y-%m-%d")
        max_date = max(dates).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"

    env.ui.console.print(f"\nMatched: {len(results)} conversations\n")
    env.ui.console.print(f"Messages: {total_messages} total ({user_messages} user, {assistant_messages} assistant)")
    env.ui.console.print(f"Words: {total_words:,}")
    env.ui.console.print(f"Thinking: {thinking_traces} traces")
    env.ui.console.print(f"Tool use: {tool_calls} calls")
    env.ui.console.print(f"Attachments: {attachments}")
    if date_range:
        env.ui.console.print(f"Date range: {date_range}")


def _output_results(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Output query results."""
    if not results:
        env.ui.console.print("No conversations matched.")
        raise SystemExit(2)  # Exit code 2 for no results

    output_format = params.get("output_format", "markdown")
    output_dest = params.get("output", "stdout")
    list_mode = params.get("list_mode", False)
    fields = params.get("fields")

    # Parse output destinations
    destinations = [d.strip() for d in output_dest.split(",")] if output_dest else ["stdout"]

    # Single result and not list mode: show content
    if len(results) == 1 and not list_mode:
        conv = results[0]
        content = _format_conversation(conv, output_format, fields)
        _send_output(env, content, destinations, output_format, conv)
        return

    # Multiple results or list mode: show list
    content = _format_list(results, output_format, fields)
    _send_output(env, content, destinations, output_format, None)


def _format_conversation(
    conv: Conversation,
    output_format: str,
    fields: str | None,
) -> str:
    """Format a single conversation for output."""
    if output_format == "json":
        return json.dumps(_conv_to_dict(conv, fields), indent=2)
    elif output_format == "yaml":
        return _conv_to_yaml(conv, fields)
    elif output_format == "html":
        return _conv_to_html(conv)
    elif output_format == "obsidian":
        return _conv_to_obsidian(conv)
    elif output_format == "org":
        return _conv_to_org(conv)
    elif output_format == "plaintext":
        return _conv_to_plaintext(conv)
    else:  # markdown
        return _conv_to_markdown(conv)


def _format_list(
    results: list[Conversation],
    output_format: str,
    fields: str | None,
) -> str:
    """Format a list of conversations for output."""
    if output_format == "json":
        return json.dumps([_conv_to_dict(c, fields) for c in results], indent=2)
    elif output_format == "yaml":
        import yaml

        return yaml.dump([_conv_to_dict(c, fields) for c in results], default_flow_style=False, allow_unicode=True)
    elif output_format == "csv":
        return _conv_to_csv(results)
    else:
        lines = []
        for conv in results:
            date = conv.updated_at.strftime("%Y-%m-%d") if conv.updated_at else "unknown"
            title = conv.display_title[:50] if conv.display_title else conv.id[:20]
            msg_count = len(conv.messages)
            lines.append(f"{conv.id[:24]:24s}  {date:10s}  [{conv.provider:12s}]  {title} ({msg_count} msgs)")
        return "\n".join(lines)


def _output_summary_list(
    env: AppEnv,
    summaries: list[ConversationSummary],
    params: dict[str, Any],
) -> None:
    """Output a list of conversation summaries (memory-efficient).

    This is the fast path for --list mode that doesn't load messages.
    """

    output_format = params.get("output_format", "text")

    if output_format == "json":
        data = [
            {
                "id": str(s.id),
                "provider": s.provider,
                "title": s.display_title,
                "date": s.updated_at.isoformat() if s.updated_at else None,
                "tags": s.tags,
                "summary": s.summary,
            }
            for s in summaries
        ]
        env.ui.console.print(json.dumps(data, indent=2))
    elif output_format == "yaml":
        import yaml

        data = [
            {
                "id": str(s.id),
                "provider": s.provider,
                "title": s.display_title,
                "date": s.updated_at.isoformat() if s.updated_at else None,
                "tags": s.tags,
            }
            for s in summaries
        ]
        env.ui.console.print(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    else:
        # Plain text format (default)
        lines = []
        for s in summaries:
            date = s.updated_at.strftime("%Y-%m-%d") if s.updated_at else "unknown"
            title = s.display_title[:50] if s.display_title else str(s.id)[:20]
            # Note: message count not available in summary - show "?" or skip
            lines.append(f"{str(s.id)[:24]:24s}  {date:10s}  [{s.provider:12s}]  {title}")
        env.ui.console.print("\n".join(lines))


def _conv_to_csv(results: list[Conversation]) -> str:
    """Convert conversations to CSV string."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "provider", "title", "messages", "words"])

    for conv in results:
        date = conv.updated_at.strftime("%Y-%m-%d") if conv.updated_at else ""
        writer.writerow(
            [
                str(conv.id),
                date,
                conv.provider,
                conv.display_title or "",
                len(conv.messages),
                sum(m.word_count for m in conv.messages),
            ]
        )

    return output.getvalue()


def _conv_to_dict(conv: Conversation, fields: str | None) -> dict[str, Any]:
    """Convert conversation to dict, optionally selecting fields."""
    full = {
        "id": str(conv.id),
        "provider": conv.provider,
        "title": conv.display_title,
        "date": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": len(conv.messages),
        "words": sum(m.word_count for m in conv.messages),
        "tags": conv.tags,
        "summary": conv.summary,
    }
    if not fields:
        return full
    selected = [f.strip() for f in fields.split(",")]
    return {k: v for k, v in full.items() if k in selected}


def _conv_to_markdown(conv: Conversation) -> str:
    """Convert conversation to markdown."""
    lines = [f"# {conv.display_title or conv.id}", ""]
    if conv.updated_at:
        lines.append(f"**Date**: {conv.updated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Provider**: {conv.provider}")
    lines.append("")

    for msg in conv.messages:
        role_label = msg.role.capitalize()
        lines.append(f"## {role_label}")
        lines.append("")
        if msg.text:
            lines.append(msg.text)
        lines.append("")

    return "\n".join(lines)


def _conv_to_html(conv: Conversation) -> str:
    """Convert conversation to HTML."""
    # Simple HTML template
    title = conv.display_title or conv.id
    messages_html = []
    for msg in conv.messages:
        role_class = f"message-{msg.role}"
        text = (msg.text or "").replace("<", "&lt;").replace(">", "&gt;")
        messages_html.append(f'<div class="{role_class}"><strong>{msg.role}:</strong><p>{text}</p></div>')

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .message-user {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 8px; }}
        .message-assistant {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 8px; }}
        .message-system {{ background: #fff3e0; padding: 10px; margin: 10px 0; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {"".join(messages_html)}
</body>
</html>"""


def _conv_to_obsidian(conv: Conversation) -> str:
    """Convert conversation to Obsidian-compatible markdown with YAML frontmatter."""
    frontmatter = [
        "---",
        f"id: {conv.id}",
        f"provider: {conv.provider}",
        f"date: {conv.updated_at.isoformat() if conv.updated_at else 'unknown'}",
        f"tags: [{', '.join(conv.tags)}]" if conv.tags else "tags: []",
        "---",
        "",
    ]
    content = _conv_to_markdown(conv)
    return "\n".join(frontmatter) + content


def _conv_to_org(conv: Conversation) -> str:
    """Convert conversation to Org-mode format."""
    lines = [
        f"#+TITLE: {conv.display_title or conv.id}",
        f"#+DATE: {conv.updated_at.strftime('%Y-%m-%d') if conv.updated_at else 'unknown'}",
        f"#+PROPERTY: provider {conv.provider}",
        "",
    ]

    for msg in conv.messages:
        role_label = msg.role.upper()
        lines.append(f"* {role_label}")
        if msg.text:
            lines.append(msg.text)
        lines.append("")

    return "\n".join(lines)


def _conv_to_yaml(conv: Conversation, fields: str | None) -> str:
    """Convert conversation to YAML format.

    Args:
        conv: Conversation to format
        fields: Optional comma-separated field selector

    Returns:
        YAML-formatted string
    """
    import yaml

    data = _conv_to_dict(conv, fields)
    # For single conversation, also include full message content
    if fields is None or "messages" in fields:
        data["messages"] = [
            {
                "id": str(msg.id),
                "role": msg.role,
                "text": msg.text,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            for msg in conv.messages
        ]

    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _conv_to_plaintext(conv: Conversation) -> str:
    """Convert conversation to plain text (no markdown formatting).

    Strips all formatting, returning just the raw message content.
    Useful for piping to grep, wc, or other text processing tools.

    Args:
        conv: Conversation to format

    Returns:
        Plain text with messages separated by blank lines
    """
    lines = []

    for msg in conv.messages:
        if msg.text:
            # Just the raw text, no role labels or formatting
            lines.append(msg.text)
            lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# Streaming Output (Memory-Efficient)
# =============================================================================


def stream_conversation(
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
    conv_record = repo.backend.get_conversation(conversation_id)
    if not conv_record:
        env.ui.console.print(f"[red]Conversation not found: {conversation_id}[/red]")
        raise SystemExit(1)

    # Get stats for progress indication
    stats = repo.get_conversation_stats(conversation_id)

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
    for msg in repo.iter_messages(
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
        # Simple format: [role] text
        role_label = msg.role.upper() if msg.role else "UNKNOWN"
        if msg.text:
            sys.stdout.write(f"[{role_label}]\n{msg.text}\n\n")
        sys.stdout.flush()

    elif output_format == "markdown":
        # Markdown format with headers
        role_label = msg.role.capitalize() if msg.role else "Unknown"
        sys.stdout.write(f"## {role_label}\n\n")
        if msg.text:
            sys.stdout.write(f"{msg.text}\n\n")
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
            env.ui.console.print(content)
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
            content = _conv_to_html(conv)
        else:
            content = f"<html><body><pre>{content}</pre></body></html>"

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

    env.ui.console.print("[yellow]Could not copy to clipboard (no clipboard tool found).[/yellow]")


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
    except Exception:
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
        env.ui.console.print("[red]No rendered outputs found.[/red]")
        env.ui.console.print("Run 'polylogue sync' first to render conversations.")
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
        env.ui.console.print("[red]No rendered output found for this conversation.[/red]")
        env.ui.console.print("Run 'polylogue sync' to render conversations.")
        raise SystemExit(1)

    # Open in browser
    import webbrowser

    webbrowser.open(f"file://{render_file}")
    env.ui.console.print(f"Opened: {render_file}")
