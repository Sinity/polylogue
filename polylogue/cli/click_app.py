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
from ..config import (
    Config,
    ConfigError,
    Source,
    default_config,
    load_config,
    update_config,
    update_source,
    write_config,
)
from ..export import export_jsonl
from ..health import cached_health_summary, get_health
from ..drive_client import DriveError
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


def _fail(command: str, message: str) -> None:
    raise SystemExit(f"{command}: {message}")


def _load_effective_config(env: AppEnv) -> Config:
    return load_config(env.config_path)


def _resolve_sources(config: Config, sources: Tuple[str, ...], command: str) -> Optional[List[str]]:
    if not sources:
        return None
    defined = {source.name for source in config.sources}
    requested = list(dict.fromkeys(sources))
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
@click.option("--source", "sources", multiple=True, help="Limit to source name (repeatable)")
@click.pass_obj
def plan(env: AppEnv, sources: Tuple[str, ...]) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("plan", str(exc))
    selected_sources = _resolve_sources(config, sources, "plan")
    try:
        plan_result = plan_sources(config, ui=env.ui, source_names=selected_sources)
    except DriveError as exc:
        _fail("plan", str(exc))
    lines = [
        f"Snapshot: {_format_timestamp(plan_result.timestamp)}",
        f"Conversations: {plan_result.counts['conversations']}",
        f"Messages: {plan_result.counts['messages']}",
        f"Attachments: {plan_result.counts['attachments']}",
    ]
    if selected_sources:
        lines.insert(1, f"Sources: {', '.join(selected_sources)}")
    cursor_line = _format_cursors(plan_result.cursors)
    if cursor_line:
        lines.append(f"Cursors: {cursor_line}")
    env.ui.summary(
        "Plan",
        lines,
    )


@cli.command()
@click.option("--no-plan", is_flag=True, help="Skip plan preview")
@click.option("--strict-plan", is_flag=True, help="Fail if plan drift is detected")
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all")
@click.option("--source", "sources", multiple=True, help="Limit to source name (repeatable)")
@click.pass_obj
def run(
    env: AppEnv,
    no_plan: bool,
    strict_plan: bool,
    stage: str,
    sources: Tuple[str, ...],
) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("run", str(exc))
    selected_sources = _resolve_sources(config, sources, "run")
    plan_result = None
    if not no_plan:
        try:
            plan_result = plan_sources(config, ui=env.ui, source_names=selected_sources)
        except DriveError as exc:
            _fail("run", str(exc))
        plan_lines = [
            f"Snapshot: {_format_timestamp(plan_result.timestamp)}",
            f"Conversations: {plan_result.counts['conversations']}",
            f"Messages: {plan_result.counts['messages']}",
            f"Attachments: {plan_result.counts['attachments']}",
        ]
        if selected_sources:
            plan_lines.insert(1, f"Sources: {', '.join(selected_sources)}")
        cursor_line = _format_cursors(plan_result.cursors)
        if cursor_line:
            plan_lines.append(f"Cursors: {cursor_line}")
        env.ui.summary("Plan", plan_lines)
    if not env.ui.plain:
        if not env.ui.confirm("Proceed with run?", default=True):
            env.ui.console.print("Run cancelled.")
            return
    try:
        result = run_sources(
            config=config,
            stage=stage,
            plan=plan_result,
            ui=env.ui,
            source_names=selected_sources,
        )
    except DriveError as exc:
        _fail("run", str(exc))
    drift_total = 0
    if plan_result is not None:
        drift_total = sum(
            abs(value)
            for bucket in result.drift.values()
            for value in bucket.values()
        )
    env.ui.summary(
        "Run",
        [
            f"Run ID: {result.run_id}",
            f"Conversations: {result.counts['conversations']} (skipped {result.counts['skipped_conversations']})",
            f"Messages: {result.counts['messages']} (skipped {result.counts['skipped_messages']})",
            f"Attachments: {result.counts['attachments']} (skipped {result.counts['skipped_attachments']})",
            f"Indexed: {result.indexed}",
            f"Duration: {result.duration_ms}ms",
            *_format_drift(result.drift),
        ],
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
    if strict_plan and drift_total != 0:
        _fail("run", f"plan drift detected: {result.drift}")


@cli.command()
@click.option("--source", "sources", multiple=True, help="Limit to source name (repeatable)")
@click.pass_obj
def ingest(env: AppEnv, sources: Tuple[str, ...]) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("ingest", str(exc))
    selected_sources = _resolve_sources(config, sources, "ingest")
    try:
        result = run_sources(
            config=config,
            stage="ingest",
            plan=None,
            ui=env.ui,
            source_names=selected_sources,
        )
    except DriveError as exc:
        _fail("ingest", str(exc))
    env.ui.summary(
        "Ingest",
        [
            f"Run ID: {result.run_id}",
            f"Conversations: {result.counts['conversations']} (skipped {result.counts['skipped_conversations']})",
            f"Messages: {result.counts['messages']} (skipped {result.counts['skipped_messages']})",
            f"Attachments: {result.counts['attachments']} (skipped {result.counts['skipped_attachments']})",
            f"Duration: {result.duration_ms}ms",
        ],
    )


@cli.command()
@click.option("--source", "sources", multiple=True, help="Limit to source name (repeatable)")
@click.pass_obj
def render(env: AppEnv, sources: Tuple[str, ...]) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("render", str(exc))
    selected_sources = _resolve_sources(config, sources, "render")
    result = run_sources(
        config=config,
        stage="render",
        plan=None,
        ui=env.ui,
        source_names=selected_sources,
    )
    env.ui.summary(
        "Render",
        [
            f"Run ID: {result.run_id}",
            f"Duration: {result.duration_ms}ms",
        ],
    )


@cli.command()
@click.argument("query")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines")
@click.option("--csv", type=click.Path(path_type=Path), help="Write CSV to file")
@click.option("--pick", is_flag=True, help="Interactive picker for results")
@click.option("--open", "open_result", is_flag=True, help="Open result path after selection")
@click.pass_obj
def search(
    env: AppEnv,
    query: str,
    limit: int,
    json_output: bool,
    json_lines: bool,
    csv: Optional[Path],
    pick: bool,
    open_result: bool,
) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("search", str(exc))
    try:
        result = search_messages(query, archive_root=config.archive_root, limit=limit)
    except RuntimeError as exc:
        _fail("search", str(exc))
    hits = result.hits

    if json_output:
        payload = [
            {
                **hit.__dict__,
                "conversation_path": str(hit.conversation_path),
            }
            for hit in hits
        ]
        env.ui.console.print(json.dumps(payload, indent=2))
        return
    if json_lines:
        for hit in hits:
            payload = {**hit.__dict__, "conversation_path": str(hit.conversation_path)}
            env.ui.console.print(json.dumps(payload))
        return
    if csv:
        rows = [
            {
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
    for idx, hit in enumerate(hits, start=1):
        title = hit.title or hit.conversation_id
        env.ui.console.print(f"{idx}. {title} ({hit.provider_name})")
        env.ui.console.print(f"   {hit.snippet}")
        env.ui.console.print(f"   {hit.conversation_path}")

    selected = hits
    if pick and not env.ui.plain and len(hits) > 1:
        options = [f"{idx}: {hit.title or hit.conversation_id}" for idx, hit in enumerate(hits, start=1)]
        choice = env.ui.choose("Select result", options)
        if choice:
            try:
                index = int(choice.split(":", 1)[0]) - 1
                selected = [hits[index]]
            except Exception:
                selected = hits

    if open_result:
        if len(selected) != 1:
            env.ui.console.print("[yellow]--open requires a single result. Use --limit 1 or --pick.[/yellow]")
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
@click.option("--open", "open_result", is_flag=True, help="Open path in browser/editor")
@click.pass_obj
def open(env: AppEnv, open_result: bool) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("open", str(exc))
    target = _latest_render_path(config.archive_root)
    if not target:
        _fail("open", "no rendered outputs found")
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
@click.pass_obj
def config_show(env: AppEnv) -> None:
    try:
        config = _load_effective_config(env)
    except ConfigError as exc:
        _fail("config show", str(exc))
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
    try:
        if key == "archive_root":
            config = update_config(config, archive_root=Path(value))
        elif key.startswith("source."):
            parts = key.split(".", 2)
            if len(parts) != 3:
                raise ConfigError("Source updates require 'source.<name>.<field>'")
            _, source_name, field = parts
            config = update_source(config, source_name, field, value)
        else:
            raise ConfigError(f"Unknown config key '{key}'")
    except ConfigError as exc:
        _fail("config set", str(exc))
    write_config(config)
    env.ui.console.print(f"Config updated: {config.path}")


def main() -> None:
    cli()


__all__ = ["cli", "main"]
