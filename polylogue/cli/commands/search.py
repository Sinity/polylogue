"""Search command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.cli.editor import open_in_browser, open_in_editor
from polylogue.cli.formatting import format_source_label
from polylogue.cli.helpers import fail, latest_render_path, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.index import rebuild_index
from polylogue.search import search_messages


@click.command("search")
@click.argument("query", required=False)
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--source", help="Filter by source/provider name")
@click.option("--since", help="Filter by date (ISO format, e.g. 2023-01-01)")
@click.option("--latest", is_flag=True, help="Open the latest render instead of searching")
@click.option("--list", "list_mode", is_flag=True, help="Print all hits (skip interactive picker)")
@click.option("--verbose", is_flag=True, help="Show snippets alongside hits")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines")
@click.option("--csv", type=click.Path(path_type=Path), help="Write CSV to file")
@click.option("--open", "open_result", is_flag=True, help="Open result path after selection")
@click.pass_obj
def search_command(
    env: AppEnv,
    query: str | None,
    limit: int,
    source: str | None,
    since: str | None,
    latest: bool,
    list_mode: bool,
    verbose: bool,
    json_output: bool,
    json_lines: bool,
    csv: Path | None,
    open_result: bool,
) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("search", str(exc))
    if query is None and not latest:
        latest = True
    if latest:
        if query:
            fail("search", "--latest cannot be combined with a query")
        if json_output or json_lines or csv:
            fail("search", "--latest cannot be combined with JSON/CSV output")
        target = latest_render_path(config.render_root)
        if not target:
            fail("search", "no rendered outputs found")
        if not open_result:
            env.ui.console.print(str(target))
            return
        if target.suffix.lower() == ".html" and open_in_browser(target):
            env.ui.console.print(f"Opened {target} in browser")
            return
        if open_in_editor(target):
            env.ui.console.print(f"Opened {target} in editor")
        else:
            env.ui.console.print(f"[yellow]Could not open {target}[/yellow]")
        return
    if not query:
        fail("search", "Query required.")
    try:
        result = search_messages(
            query,
            archive_root=config.archive_root,
            render_root_path=config.render_root,
            limit=limit,
            source=source,
            since=since,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Search index not built" in message:
            env.ui.console.print("Index missing; rebuilding...")
            try:
                rebuild_index()
            except Exception as build_exc:
                fail("search", f"Index rebuild failed: {build_exc}")
            result = search_messages(
                query,
                archive_root=config.archive_root,
                render_root_path=config.render_root,
                limit=limit,
                source=source,
                since=since,
            )
        else:
            fail("search", message)
    hits = result.hits

    if json_output:
        payload = []
        for hit in hits:
            row = {**hit.__dict__, "conversation_path": str(hit.conversation_path)}
            row["source"] = row.pop("source_name")
            payload.append(row)
        env.ui.console.print(json.dumps(payload, indent=2))
        return
    if json_lines:
        for hit in hits:
            row = {**hit.__dict__, "conversation_path": str(hit.conversation_path)}
            row["source"] = row.pop("source_name")
            env.ui.console.print(json.dumps(row))
        return
    if csv:
        rows = [
            {
                "source": hit.source_name or "",
                "provider": hit.provider_name,
                "conversation_id": hit.conversation_id,
                "message_id": hit.message_id,
                "title": hit.title or "",
                "timestamp": hit.timestamp or "",
                "snippet": hit.snippet,
                "path": str(hit.conversation_path),
            }
            for hit in hits
        ]
        csv.parent.mkdir(parents=True, exist_ok=True)
        with csv.open("w", encoding="utf-8", newline="") as handle:
            import csv as csv_module

            fieldnames = (
                list(rows[0].keys())
                if rows
                else [
                    "source",
                    "provider",
                    "conversation_id",
                    "message_id",
                    "title",
                    "timestamp",
                    "snippet",
                    "path",
                ]
            )
            writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            if rows:
                writer.writerows(rows)
        env.ui.console.print(f"Wrote {len(rows)} rows to {csv}")
        return

    env.ui.summary("Search", [f"Results: {len(hits)}", f"Query: {query}"])
    if not hits:
        return

    selected = hits
    show_snippet = list_mode or verbose
    interactive_pick = not env.ui.plain and not list_mode and len(hits) > 1
    if interactive_pick:
        options = [f"{idx}: {hit.title or hit.conversation_id}" for idx, hit in enumerate(hits, start=1)]
        choice = env.ui.choose("Select result", options)
        if not choice:
            env.ui.console.print("Search cancelled.")
            return
        try:
            index = int(choice.split(":", 1)[0]) - 1
            selected = [hits[index]]
        except Exception:
            selected = [hits[0]]

    if env.ui.plain or list_mode:
        for idx, hit in enumerate(hits, start=1):
            title = hit.title or hit.conversation_id
            source_label = format_source_label(hit.source_name, hit.provider_name)
            env.ui.console.print(f"{idx}. {title} ({source_label})")
            if show_snippet:
                env.ui.console.print(f"   {hit.snippet}")
            env.ui.console.print(f"   {hit.conversation_path}")

    if open_result:
        if len(selected) != 1:
            env.ui.console.print("[yellow]--open requires a single result. Use --limit 1 or --list.[/yellow]")
            return
        target = selected[0].conversation_path
        html_target = target.with_suffix(".html")
        if html_target.exists():
            target = html_target
        if target.suffix.lower() == ".html" and open_in_browser(target):
            env.ui.console.print(f"Opened {target} in browser")
            return
        if open_in_editor(target):
            env.ui.console.print(f"Opened {target} in editor")
        else:
            env.ui.console.print(f"[yellow]Could not open {target}[/yellow]")
