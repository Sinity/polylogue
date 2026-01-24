"""Query execution for CLI query mode.

This module handles the execution of query-mode operations including:
- Filtering conversations via the filter chain API
- Formatting and outputting results
- Aggregation operations (--by-month, --by-provider, --by-tag)
- Modifier operations (--set, --add-tag, --delete)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation


def execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Execute a query-mode command.

    Args:
        env: Application environment with UI and config
        params: Parsed CLI parameters
    """
    from polylogue.cli.container import create_repository
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError
    from polylogue.lib.repository import ConversationRepository

    # Load config
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    # Build filter chain - create_repository returns StorageBackend
    backend = create_repository(config)
    conv_repo = ConversationRepository(backend)
    filter_chain = conv_repo.filter()

    # Apply query terms (positional args)
    query_terms = params.get("query", ())
    for term in query_terms:
        filter_chain = filter_chain.contains(term)

    # Apply --contains
    for term in params.get("contains", ()):
        filter_chain = filter_chain.contains(term)

    # Apply --no-contains
    for term in params.get("no_contains", ()):
        filter_chain = filter_chain.no_contains(term)

    # Apply --regex (in-memory predicate)
    for pattern in params.get("regex", ()):
        compiled = re.compile(pattern)

        def _matches_regex(c: Conversation, pat: re.Pattern[str] = compiled) -> bool:
            return any(pat.search(m.text) for m in c.messages if m.text)

        filter_chain = filter_chain.where(_matches_regex)

    # Apply --no-regex (in-memory predicate)
    for pattern in params.get("no_regex", ()):
        compiled = re.compile(pattern)

        def _not_matches_regex(c: Conversation, pat: re.Pattern[str] = compiled) -> bool:
            return not any(pat.search(m.text) for m in c.messages if m.text)

        filter_chain = filter_chain.where(_not_matches_regex)

    # Apply --provider (comma-separated)
    if params.get("provider"):
        providers = [p.strip() for p in params["provider"].split(",")]
        filter_chain = filter_chain.provider(*providers)

    # Apply --no-provider (comma-separated)
    if params.get("no_provider"):
        excluded = [p.strip() for p in params["no_provider"].split(",")]
        filter_chain = filter_chain.no_provider(*excluded)

    # Apply --tag (comma-separated)
    if params.get("tag"):
        tags = [t.strip() for t in params["tag"].split(",")]
        filter_chain = filter_chain.tag(*tags)

    # Apply --no-tag (comma-separated)
    if params.get("no_tag"):
        excluded = [t.strip() for t in params["no_tag"].split(",")]
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

    # Apply --id
    if params.get("id_prefix"):
        filter_chain = filter_chain.id(params["id_prefix"])

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

    # Apply --similar (future: vector search)
    if params.get("similar"):
        filter_chain = filter_chain.similar(params["similar"])

    # Check for delete with no filters (safety)
    if params.get("delete_matched"):
        has_filters = any([
            query_terms,
            params.get("contains"),
            params.get("provider"),
            params.get("tag"),
            params.get("since"),
            params.get("until"),
            params.get("id_prefix"),
            params.get("has_type"),
        ])
        if not has_filters:
            fail("query", "--delete requires at least one filter flag for safety")

    # Execute query
    if params.get("pick"):
        # Interactive picker
        result = filter_chain.pick()
        if result:
            results = [result]
        else:
            env.ui.console.print("No selection made.")
            return
    else:
        results = filter_chain.list()

    # Handle modifiers (write operations)
    if params.get("set_meta") or params.get("unset") or params.get("add_tag") or params.get("rm_tag"):
        _apply_modifiers(env, results, params)
        return

    if params.get("delete_matched"):
        _delete_conversations(env, results, params)
        return

    if params.get("annotate"):
        _annotate_conversations(env, results, params)
        return

    # Handle aggregation outputs
    if params.get("by_month"):
        _output_by_month(env, results)
        return

    if params.get("by_provider"):
        _output_by_provider(env, results)
        return

    if params.get("by_tag"):
        _output_by_tag(env, results)
        return

    # Handle stats-only output
    if params.get("stats_only"):
        _output_stats(env, results)
        return

    # Handle --csv output
    if params.get("csv_path"):
        _output_csv(env, results, params["csv_path"])
        return

    # Handle --open (open in browser/editor)
    if params.get("open_result"):
        _open_result(env, results, params)
        return

    # Regular output
    _output_results(env, results, params)


def _apply_modifiers(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Apply metadata modifiers to matched conversations.

    Note: Metadata modification requires StorageBackend protocol extension.
    Currently shows matched conversations without modifying.
    """
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    count = len(results)
    env.ui.console.print("[yellow]Metadata modification not yet implemented.[/yellow]")
    env.ui.console.print(f"Would modify {count} conversation(s):")
    for conv in results[:5]:
        env.ui.console.print(f"  {conv.id}: {conv.display_title}")
    if count > 5:
        env.ui.console.print(f"  ... and {count - 5} more")


def _delete_conversations(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Delete matched conversations.

    Note: Deletion requires StorageBackend.delete_conversation() protocol method.
    Currently shows what would be deleted without actual deletion.
    """
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    count = len(results)
    env.ui.console.print("[yellow]Deletion not yet implemented.[/yellow]")
    env.ui.console.print(f"Would delete {count} conversation(s):")
    for conv in results[:5]:
        env.ui.console.print(f"  {conv.id}: {conv.display_title}")
    if count > 5:
        env.ui.console.print(f"  ... and {count - 5} more")


def _annotate_conversations(
    env: AppEnv,
    results: list[Conversation],
    params: dict[str, Any],
) -> None:
    """Run LLM annotation on matched conversations.

    Note: Annotation requires LLM integration and metadata storage.
    Currently shows what would be annotated without actual annotation.
    """
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    count = len(results)
    prompt = params.get("annotate", "")
    env.ui.console.print("[yellow]LLM annotation not yet implemented.[/yellow]")
    env.ui.console.print(f"Would annotate {count} conversation(s) with prompt: {prompt}")


def _output_by_month(env: AppEnv, results: list[Conversation]) -> None:
    """Output aggregation by month."""
    by_month: Counter[str] = Counter()
    for conv in results:
        if conv.updated_at:
            month_key = conv.updated_at.strftime("%Y-%m")
            by_month[month_key] += 1
        else:
            by_month["unknown"] += 1

    env.ui.console.print(f"\nConversations by month ({len(results)} total):\n")
    for month, count in sorted(by_month.items()):
        bar = "#" * min(count, 50)
        env.ui.console.print(f"  {month}: {count:4d} {bar}")


def _output_by_provider(env: AppEnv, results: list[Conversation]) -> None:
    """Output aggregation by provider."""
    by_provider: Counter[str] = Counter()
    for conv in results:
        by_provider[conv.provider] += 1

    env.ui.console.print(f"\nConversations by provider ({len(results)} total):\n")
    for provider, count in sorted(by_provider.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results) if results else 0
        bar = "#" * int(pct / 2)
        env.ui.console.print(f"  {provider:15s}: {count:4d} ({pct:5.1f}%) {bar}")


def _output_by_tag(env: AppEnv, results: list[Conversation]) -> None:
    """Output aggregation by tag."""
    by_tag: Counter[str] = Counter()
    untagged = 0
    for conv in results:
        if conv.tags:
            for tag in conv.tags:
                by_tag[tag] += 1
        else:
            untagged += 1

    env.ui.console.print(f"\nConversations by tag ({len(results)} total, {untagged} untagged):\n")
    for tag, count in sorted(by_tag.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 50)
        env.ui.console.print(f"  {tag:20s}: {count:4d} {bar}")


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
    elif output_format == "html":
        return _conv_to_html(conv)
    elif output_format == "obsidian":
        return _conv_to_obsidian(conv)
    elif output_format == "org":
        return _conv_to_org(conv)
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
    else:
        lines = []
        for conv in results:
            date = conv.updated_at.strftime("%Y-%m-%d") if conv.updated_at else "unknown"
            title = conv.display_title[:50] if conv.display_title else conv.id[:20]
            msg_count = len(conv.messages)
            lines.append(f"{conv.id[:24]:24s}  {date:10s}  [{conv.provider:12s}]  {title} ({msg_count} msgs)")
        return "\n".join(lines)


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
    {''.join(messages_html)}
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
        if conv:
            content = _conv_to_html(conv)
        else:
            # Wrap plain content in HTML
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


def _output_csv(env: AppEnv, results: list[Conversation], csv_path: Path) -> None:
    """Write results to CSV file."""
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header - match old search command format for compatibility
        writer.writerow([
            "source", "provider", "conversation_id", "message_id",
            "role", "timestamp", "word_count", "text_preview"
        ])

        # Write rows for each message in each conversation
        for conv in results:
            source = conv.provider_meta.get("source", "") if conv.provider_meta else ""
            for msg in conv.messages:
                preview = (msg.text or "")[:100].replace("\n", " ")
                timestamp = msg.timestamp.isoformat() if msg.timestamp else ""
                writer.writerow([
                    source,
                    conv.provider,
                    str(conv.id),
                    str(msg.id),
                    msg.role,
                    timestamp,
                    msg.word_count,
                    preview,
                ])

    env.ui.console.print(f"Wrote {len(results)} conversations to {csv_path}")


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
