"""CLI entrypoint (clean surface, adaptive UI)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import click

from ..ui import create_ui
from ..paths import STATE_HOME
from ..config import (
    Config,
    ConfigError,
    Source,
    DEFAULT_INBOX_ROOT,
    default_config,
    load_config,
    update_config,
    write_config,
)
from ..export import export_jsonl
from ..health import cached_health_summary, get_health
from ..drive_client import DriveError
from ..index import rebuild_index
from ..run import latest_run, plan_sources, run_sources
from ..search import search_messages
from .editor import open_in_browser, open_in_editor


@dataclass
class AppEnv:
    ui: object
    config_path: Optional[Path] = None


def _should_use_plain(*, plain: bool, interactive: bool) -> bool:
    if plain:
        return True
    if interactive:
        return False
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    if env_force and env_force.lower() not in {"0", "false", "no"}:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())


def _announce_plain_mode() -> None:
    sys.stderr.write("Plain output active (non-TTY). Use --interactive from a TTY to re-enable prompts.\n")


def _format_timestamp(ts: int) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _format_cursors(cursors: dict) -> Optional[str]:
    if not cursors:
        return None
    parts: List[str] = []
    for name, cursor in cursors.items():
        detail_bits: List[str] = []
        file_count = cursor.get("file_count")
        if isinstance(file_count, int):
            detail_bits.append(f"{file_count} files")
        latest_mtime = cursor.get("latest_mtime")
        latest_label = None
        if isinstance(latest_mtime, (int, float)):
            latest_label = _format_timestamp(int(latest_mtime))
        else:
            latest_name = cursor.get("latest_file_name")
            latest_path = cursor.get("latest_path")
            if isinstance(latest_name, str):
                latest_label = latest_name
            elif isinstance(latest_path, str):
                latest_label = Path(latest_path).name
        if latest_label:
            detail_bits.append(f"latest {latest_label}")
        detail = ", ".join(detail_bits) if detail_bits else "unknown"
        parts.append(f"{name} ({detail})")
    return "; ".join(parts)


def _format_drift(drift: dict) -> List[str]:
    def _counts(label: str, payload: dict) -> str:
        return (
            f"{label}: {payload.get('conversations', 0)} conv, "
            f"{payload.get('messages', 0)} msg, "
            f"{payload.get('attachments', 0)} att"
        )

    return [
        _counts("New", drift.get("new", {})),
        _counts("Removed", drift.get("removed", {})),
        _counts("Changed", drift.get("changed", {})),
    ]


def _format_counts(counts: dict) -> str:
    return (
        f"{counts.get('conversations', 0)} conv, "
        f"{counts.get('messages', 0)} msg"
    )


def _format_attachments(counts: dict) -> str:
    return f"{counts.get('attachments', 0)} att"


def _format_skipped(counts: dict) -> str:
    return (
        f"{counts.get('skipped_conversations', 0)} conv, "
        f"{counts.get('skipped_messages', 0)} msg, "
        f"{counts.get('skipped_attachments', 0)} att"
    )


def _format_index_status(stage: str, indexed: bool, index_error: Optional[str]) -> str:
    if stage in {"ingest", "render"}:
        return "Index: skipped"
    if index_error:
        return "Index: error"
    if indexed:
        return "Index: ok"
    return "Index: up-to-date"


def _format_source_label(source_name: Optional[str], provider_name: str) -> str:
    if source_name and source_name != provider_name:
        return f"{source_name}/{provider_name}"
    return source_name or provider_name



def _fail(command: str, message: str) -> None:
    raise SystemExit(f"{command}: {message}")


def _is_declarative() -> bool:
    value = os.environ.get("POLYLOGUE_DECLARATIVE")
    if not value:
        return False
    return value.lower() not in {"0", "false", "no"}


def _source_state_path() -> Path:
    return STATE_HOME / "last-source.json"


def _load_last_source() -> Optional[str]:
    path = _source_state_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("source"), str):
        return payload["source"]
    return None


def _save_last_source(source_name: str) -> None:
    path = _source_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source_name}), encoding="utf-8")


def _maybe_prompt_sources(
    env: AppEnv,
    config: Config,
    selected_sources: Optional[List[str]],
    command: str,
) -> Optional[List[str]]:
    if selected_sources is not None or env.ui.plain:
        return selected_sources
    names = [source.name for source in config.sources]
    if len(names) <= 1:
        return selected_sources
    options = ["all"] + names
    last_choice = _load_last_source()
    if last_choice in options:
        options.remove(last_choice)
        options.insert(0, last_choice)
    choice = env.ui.choose(f"Select source for {command}", options)
    if not choice:
        return selected_sources
    _save_last_source(choice)
    if choice == "all":
        return None
    return [choice]


def _load_effective_config(env: AppEnv) -> Config:
    return load_config(env.config_path)


def _resolve_sources(config: Config, sources: Tuple[str, ...], command: str) -> Optional[List[str]]:
    if not sources:
        return None
    requested = list(dict.fromkeys(sources))
    if "last" in requested:
        if len(requested) > 1:
            _fail(command, "--source last cannot be combined with other sources")
        last = _load_last_source()
        if not last:
            _fail(command, "No previously selected source found for --source last")
        requested = [last]
    defined = {source.name for source in config.sources}
    missing = sorted(set(requested) - defined)
    if missing:
        _fail(command, f"Unknown source(s): {', '.join(missing)}")
    return requested


def _print_summary(env: AppEnv) -> None:
    ui = env.ui
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        ui.console.print(f"[yellow]{exc}[/yellow]")
        ui.console.print("Run `polylogue config init` to create a config.")
        return
    last_run = latest_run()
    last_line = "Last run: none"
    if last_run:
        last_line = f"Last run: {last_run.get('run_id')} ({last_run.get('timestamp')})"
    health_line = f"Health: {cached_health_summary(config.archive_root)}"
    ui.summary(
        "Polylogue",
        [
            f"Config: {config.path}",
            f"Archive root: {config.archive_root}",
            last_line,
            health_line,
        ],
    )


def _latest_render_path(archive_root: Path) -> Optional[Path]:
    render_root = archive_root / "render"
    if not render_root.exists():
        return None
    candidates = list(render_root.rglob("conversation.md")) + list(render_root.rglob("conversation.html"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--plain", is_flag=True, help="Force non-interactive plain output")
@click.option("--interactive", is_flag=True, help="Force interactive output")
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config.json")
@click.pass_context
def cli(ctx: click.Context, plain: bool, interactive: bool, config_path: Optional[Path]) -> None:
    """Polylogue CLI."""
    use_plain = _should_use_plain(plain=plain, interactive=interactive)
    env = AppEnv(ui=create_ui(use_plain), config_path=config_path)
    ctx.obj = env
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    forced_plain = bool(env_force and env_force.lower() not in {"0", "false", "no"})
    if use_plain and not plain and not interactive and not forced_plain:
        _announce_plain_mode()
    if ctx.invoked_subcommand is None:
        _print_summary(env)


@cli.command()
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option("--verbose", is_flag=True, help="Show detailed counts and drift")
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all", show_default=True)
@click.option("--source", "sources", multiple=True, help="Limit to source name (repeatable, or 'last')")
@click.pass_obj
def run(
    env: AppEnv,
    preview: bool,
    verbose: bool,
    stage: str,
    sources: Tuple[str, ...],
) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("run", str(exc))
    selected_sources = _resolve_sources(config, sources, "run")
    selected_sources = _maybe_prompt_sources(env, config, selected_sources, "run")
    if preview:
        try:
            plan_result = plan_sources(config, ui=env.ui, source_names=selected_sources)
        except DriveError as exc:
            _fail("run", str(exc))
        plan_lines = []
        if selected_sources:
            plan_lines.append(f"Sources: {', '.join(selected_sources)}")
        plan_lines.append(f"Counts: {_format_counts(plan_result.counts)}")
        cursor_line = _format_cursors(plan_result.cursors)
        if cursor_line:
            plan_lines.append(f"Cursors: {cursor_line}")
        if verbose:
            plan_lines.append(f"Attachments: {_format_attachments(plan_result.counts)}")
            plan_lines.append(f"Snapshot: {_format_timestamp(plan_result.timestamp)}")
        env.ui.summary("Preview", plan_lines)
        return
    if not env.ui.plain:
        if not env.ui.confirm("Proceed with run?", default=True):
            env.ui.console.print("Run cancelled.")
            return
    try:
        result = run_sources(
            config=config,
            stage=stage,
            plan=None,
            ui=env.ui,
            source_names=selected_sources,
        )
    except DriveError as exc:
        _fail("run", str(exc))
    run_lines = [
        f"Counts: {_format_counts(result.counts)}",
        _format_index_status(stage, result.indexed, result.index_error),
        f"Duration: {result.duration_ms}ms",
    ]
    if verbose:
        if selected_sources:
            run_lines.insert(0, f"Sources: {', '.join(selected_sources)}")
        run_lines.append(f"Attachments: {_format_attachments(result.counts)}")
        run_lines.append(f"Skipped: {_format_skipped(result.counts)}")
        run_lines.append(f"Run ID: {result.run_id}")
        run_lines.extend(_format_drift(result.drift))
    env.ui.summary(
        f"Run ({stage})" if stage != "all" else "Run",
        run_lines,
    )
    if result.index_error:
        error_line = f"Index error: {result.index_error}"
        hint_line = "Hint: run `polylogue run --stage index` to rebuild the index."
        if env.ui.plain:
            env.ui.console.print(error_line)
            env.ui.console.print(hint_line)
        else:
            env.ui.console.print(f"[yellow]{error_line}[/yellow]")
            env.ui.console.print(f"[yellow]{hint_line}[/yellow]")


@cli.command()
@click.pass_obj
def index(env: AppEnv) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("index", str(exc))
    try:
        result = run_sources(
            config=config,
            stage="index",
            plan=None,
            ui=env.ui,
            source_names=None,
        )
    except DriveError as exc:
        _fail("index", str(exc))
    env.ui.summary(
        "Index",
        [
            _format_index_status("index", result.indexed, result.index_error),
            f"Duration: {result.duration_ms}ms",
        ],
    )
    if result.index_error:
        error_line = f"Index error: {result.index_error}"
        hint_line = "Hint: run `polylogue index` to rebuild the index."
        if env.ui.plain:
            env.ui.console.print(error_line)
            env.ui.console.print(hint_line)
        else:
            env.ui.console.print(f"[yellow]{error_line}[/yellow]")
            env.ui.console.print(f"[yellow]{hint_line}[/yellow]")


@cli.command()
@click.argument("query", required=False)
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--latest", is_flag=True, help="Open the latest render instead of searching")
@click.option("--list", "list_mode", is_flag=True, help="Print all hits (skip interactive picker)")
@click.option("--verbose", is_flag=True, help="Show snippets alongside hits")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines")
@click.option("--csv", type=click.Path(path_type=Path), help="Write CSV to file")
@click.option("--open", "open_result", is_flag=True, help="Open result path after selection")
@click.pass_obj
def search(
    env: AppEnv,
    query: Optional[str],
    limit: int,
    latest: bool,
    list_mode: bool,
    verbose: bool,
    json_output: bool,
    json_lines: bool,
    csv: Optional[Path],
    open_result: bool,
) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("search", str(exc))
    if query is None and not latest:
        latest = True
    if latest:
        if query:
            _fail("search", "--latest cannot be combined with a query")
        if json_output or json_lines or csv:
            _fail("search", "--latest cannot be combined with JSON/CSV output")
        target = _latest_render_path(config.archive_root)
        if not target:
            _fail("search", "no rendered outputs found")
        if not open_result:
            env.ui.console.print(str(target))
            return
        if target.suffix.lower() == ".html":
            if open_in_browser(target):
                env.ui.console.print(f"Opened {target} in browser")
                return
        if open_in_editor(target):
            env.ui.console.print(f"Opened {target} in editor")
        else:
            env.ui.console.print(f"[yellow]Could not open {target}[/yellow]")
        return
    if not query:
        _fail("search", "Query required.")
    try:
        result = search_messages(query, archive_root=config.archive_root, limit=limit)
    except RuntimeError as exc:
        message = str(exc)
        if "Search index not built" in message:
            env.ui.console.print("Index missing; rebuilding...")
            try:
                rebuild_index()
            except Exception as build_exc:
                _fail("search", f"Index rebuild failed: {build_exc}")
            result = search_messages(query, archive_root=config.archive_root, limit=limit)
        else:
            _fail("search", message)
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

            fieldnames = list(rows[0].keys()) if rows else [
                "source",
                "provider",
                "conversation_id",
                "message_id",
                "title",
                "timestamp",
                "snippet",
                "path",
            ]
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
        options = [
            f"{idx}: {hit.title or hit.conversation_id}"
            for idx, hit in enumerate(hits, start=1)
        ]
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
            source_label = _format_source_label(hit.source_name, hit.provider_name)
            env.ui.console.print(f"{idx}. {title} ({source_label})")
            if show_snippet:
                env.ui.console.print(f"   {hit.snippet}")
            env.ui.console.print(f"   {hit.conversation_path}")
    elif selected:
        hit = selected[0]
        title = hit.title or hit.conversation_id
        source_label = _format_source_label(hit.source_name, hit.provider_name)
        env.ui.console.print(f"1. {title} ({source_label})")
        if show_snippet:
            env.ui.console.print(f"   {hit.snippet}")
        env.ui.console.print(f"   {hit.conversation_path}")

    if not env.ui.plain and not list_mode:
        open_result = True

    if open_result:
        if len(selected) != 1:
            env.ui.console.print("[yellow]--open requires a single result. Use --limit 1 or --list.[/yellow]")
            return
        target = selected[0].conversation_path
        html_target = target.with_suffix(".html")
        if html_target.exists():
            target = html_target
        if target.suffix.lower() == ".html":
            if open_in_browser(target):
                env.ui.console.print(f"Opened {target} in browser")
                return
        if open_in_editor(target):
            env.ui.console.print(f"Opened {target} in editor")
        else:
            env.ui.console.print(f"[yellow]Could not open {target}[/yellow]")




@cli.command()
@click.pass_obj
def health(env: AppEnv) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("health", str(exc))
    payload = get_health(config)
    cached = payload.get("cached")
    age = payload.get("age_seconds")
    header = f"Health (cached={cached}, age={age}s)" if cached is not None else "Health"
    checks = payload.get("checks", [])
    lines = []
    for check in checks:
        name = check.get("name")
        status = check.get("status")
        detail = check.get("detail")
        lines.append(f"{name}: {status} - {detail}")
    env.ui.summary(header, lines)


@cli.command()
@click.option("--out", type=click.Path(path_type=Path), help="Write JSONL export to path")
@click.pass_obj
def export(env: AppEnv, out: Optional[Path]) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("export", str(exc))
    target = export_jsonl(archive_root=config.archive_root, output_path=out)
    env.ui.console.print(f"Exported {target}")


@cli.group()
def config() -> None:
    """Configuration commands."""


@config.command("init")
@click.option("--interactive", "interactive", is_flag=True, help="Run interactive config init")
@click.option("--with-drive", is_flag=True, help="Include a Drive source without prompting")
@click.option("--drive-name", default="gemini", show_default=True, help="Drive source name")
@click.option("--drive-folder", default="Google AI Studio", show_default=True, help="Drive folder name")
@click.pass_obj
def config_init(
    env: AppEnv,
    interactive: bool,
    with_drive: bool,
    drive_name: str,
    drive_folder: str,
) -> None:
    target = env.config_path
    config = default_config(target)
    if config.path.exists():
        _fail("config init", f"config already exists at {config.path}")
    add_drive = with_drive
    if interactive and not env.ui.plain:
        prompt = (
            f"Add Drive source '{drive_name}' (folder '{drive_folder}') "
            f"when writing {config.path}?"
        )
        add_drive = env.ui.confirm(prompt, default=True)
    if add_drive:
        source_name = drive_name.strip() or "gemini"
        folder_name = drive_folder.strip() or "Google AI Studio"
        if any(source.name == source_name for source in config.sources):
            env.ui.console.print(f"Source '{source_name}' already exists; skipping.")
        else:
            config.sources.append(Source(name=source_name, folder=folder_name))
    write_config(config)
    env.ui.console.print(f"Config written to {config.path}")


@config.command("show")
@click.option("--pretty", is_flag=True, help="Show a human-readable summary (interactive only)")
@click.pass_obj
def config_show(env: AppEnv, pretty: bool) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("config show", str(exc))
    if pretty:
        if env.ui.plain:
            _fail("config show", "pretty output requires interactive mode")
        sources = []
        for source in config.sources:
            if source.folder:
                sources.append(f"{source.name}: drive folder '{source.folder}'")
            elif source.path:
                sources.append(f"{source.name}: {source.path}")
            else:
                sources.append(f"{source.name}: (missing path)")
        env.ui.summary(
            "Config",
            [
                f"Path: {config.path}",
                f"Archive root: {config.archive_root}",
                f"Sources: {', '.join(sources) if sources else 'none'}",
            ],
        )
        return
    payload = config.as_dict()
    raw_root = None
    try:
        raw = json.loads(config.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_root = raw.get("archive_root")
    except Exception:
        raw_root = None
    payload["resolved_archive_root"] = str(config.archive_root)
    payload["configured_archive_root"] = raw_root
    payload["env_overrides"] = {
        "POLYLOGUE_CONFIG": os.environ.get("POLYLOGUE_CONFIG"),
        "POLYLOGUE_ARCHIVE_ROOT": os.environ.get("POLYLOGUE_ARCHIVE_ROOT"),
    }
    env.ui.console.print(json.dumps(payload, indent=2))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_obj
def config_set(env: AppEnv, key: str, value: str) -> None:
    try:
        config = load_config(env.config_path)
    except ConfigError as exc:
        _fail("config set", str(exc))
    if _is_declarative():
        _fail("config set", "config set is disabled in declarative mode")
    try:
        if key == "archive_root":
            config = update_config(config, archive_root=Path(value))
        elif key.startswith("source."):
            raise ConfigError("Use `polylogue config edit` to manage sources.")
        else:
            raise ConfigError(f"Unknown config key '{key}'")
    except ConfigError as exc:
        _fail("config set", str(exc))
    write_config(config)
    env.ui.console.print(f"Config updated: {config.path}")


@config.command("edit")
@click.pass_obj
def config_edit(env: AppEnv) -> None:
    if env.ui.plain:
        _fail("config edit", "interactive mode required")
    if _is_declarative():
        _fail("config edit", "config edit is disabled in declarative mode")
    try:
        config = load_config(env.config_path)
    except ConfigError as exc:
        _fail("config edit", str(exc))

    changed = False
    while True:
        choice = env.ui.choose(
            "Config edit",
            ["Add source", "Edit source", "Remove source", "Set archive root", "Done"],
        )
        if not choice or choice == "Done":
            break
        if choice == "Add source":
            name = env.ui.input("Source name", default="inbox")
            if not name:
                continue
            name = name.strip()
            existing = [source for source in config.sources if source.name == name]
            if existing:
                if not env.ui.confirm(f"Replace existing source '{name}'?", default=False):
                    continue
                config.sources = [source for source in config.sources if source.name != name]
            kind = env.ui.choose("Source type", ["local path", "drive folder"])
            if not kind:
                continue
            if kind == "drive folder":
                folder = env.ui.input("Drive folder name", default="Google AI Studio")
                if not folder:
                    continue
                config.sources.append(Source(name=name, folder=folder.strip()))
            else:
                default_path = None
                if name == "inbox":
                    default_path = str(DEFAULT_INBOX_ROOT)
                path_value = env.ui.input("Local path", default=default_path)
                if not path_value:
                    continue
                config.sources.append(Source(name=name, path=Path(path_value).expanduser()))
            changed = True
        elif choice == "Edit source":
            names = [source.name for source in config.sources]
            if not names:
                env.ui.console.print("[yellow]No sources configured.[/yellow]")
                continue
            selected = env.ui.choose("Select source", names)
            if not selected:
                continue
            source = next((s for s in config.sources if s.name == selected), None)
            if source is None:
                continue
            kind = env.ui.choose("Source type", ["local path", "drive folder"])
            if not kind:
                continue
            if kind == "drive folder":
                folder = env.ui.input("Drive folder name", default=source.folder or "Google AI Studio")
                if not folder:
                    continue
                source.folder = folder.strip()
                source.path = None
            else:
                path_value = env.ui.input("Local path", default=str(source.path) if source.path else "")
                if not path_value:
                    continue
                source.path = Path(path_value).expanduser()
                source.folder = None
            changed = True
        elif choice == "Remove source":
            names = [source.name for source in config.sources]
            if not names:
                env.ui.console.print("[yellow]No sources configured.[/yellow]")
                continue
            selected = env.ui.choose("Remove which source?", names)
            if not selected:
                continue
            if env.ui.confirm(f"Remove source '{selected}'?", default=False):
                config.sources = [source for source in config.sources if source.name != selected]
                changed = True
        elif choice == "Set archive root":
            new_root = env.ui.input("Archive root", default=str(config.archive_root))
            if not new_root:
                continue
            config.archive_root = Path(new_root).expanduser()
            changed = True

    if changed:
        write_config(config)
        env.ui.console.print(f"Config updated: {config.path}")
    else:
        env.ui.console.print("No changes made.")


def main() -> None:
    cli()


__all__ = ["cli", "main"]
