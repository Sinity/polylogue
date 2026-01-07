"""Config commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from polylogue.cli.helpers import fail, format_sources_summary, is_declarative, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import (
    DEFAULT_INBOX_ROOT,
    ConfigError,
    Source,
    default_config,
    load_config,
    update_config,
    write_config,
)


@click.group("config")
def config_command() -> None:
    """Configuration commands."""


@config_command.command("init")
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
        fail("config init", f"config already exists at {config.path}")
    add_drive = with_drive
    if interactive and not env.ui.plain:
        prompt = f"Add Drive source '{drive_name}' (folder '{drive_folder}') when writing {config.path}?"
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


@config_command.command("show")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.pass_obj
def config_show(env: AppEnv, json_output: bool) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("config show", str(exc))
    if json_output:
        payload = config.as_dict()
        raw_root = None
        raw_render_root = None
        try:
            raw = json.loads(config.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw_root = raw.get("archive_root")
                raw_render_root = raw.get("render_root")
        except Exception:
            raw_root = None
            raw_render_root = None
        payload["resolved_archive_root"] = str(config.archive_root)
        payload["configured_archive_root"] = raw_root
        payload["resolved_render_root"] = str(config.render_root)
        payload["configured_render_root"] = raw_render_root
        payload["env_overrides"] = {
            "POLYLOGUE_CONFIG": os.environ.get("POLYLOGUE_CONFIG"),
            "POLYLOGUE_ARCHIVE_ROOT": os.environ.get("POLYLOGUE_ARCHIVE_ROOT"),
            "POLYLOGUE_RENDER_ROOT": os.environ.get("POLYLOGUE_RENDER_ROOT"),
        }
        env.ui.console.print(json.dumps(payload, indent=2))
        return
    env.ui.summary(
        "Config",
        [
            f"Path: {config.path}",
            f"Archive root: {config.archive_root}",
            f"Render root: {config.render_root}",
            f"Sources: {format_sources_summary(config.sources)}",
        ],
    )


@config_command.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_obj
def config_set(env: AppEnv, key: str, value: str) -> None:
    try:
        config = load_config(env.config_path)
    except ConfigError as exc:
        fail("config set", str(exc))
    if is_declarative():
        fail("config set", "config set is disabled in declarative mode")
    try:
        if key == "archive_root":
            config = update_config(config, archive_root=Path(value))
        elif key == "render_root":
            config = update_config(config, render_root=Path(value))
        elif key.startswith("source."):
            raise ConfigError("Use `polylogue config edit` to manage sources.")
        else:
            raise ConfigError(f"Unknown config key '{key}'")
    except ConfigError as exc:
        fail("config set", str(exc))
    write_config(config)
    env.ui.console.print(f"Config updated: {config.path}")


@config_command.command("edit")
@click.pass_obj
def config_edit(env: AppEnv) -> None:
    if env.ui.plain:
        fail("config edit", "interactive mode required")
    if is_declarative():
        fail("config edit", "config edit is disabled in declarative mode")
    try:
        config = load_config(env.config_path)
    except ConfigError as exc:
        fail("config edit", str(exc))

    changed = False
    while True:
        choice = env.ui.choose(
            "Config edit",
            [
                "Add source",
                "Edit source",
                "Remove source",
                "Set archive root",
                "Set render root",
                "Done",
            ],
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
        elif choice == "Set render root":
            new_root = env.ui.input("Render root", default=str(config.render_root))
            if not new_root:
                continue
            config.render_root = Path(new_root).expanduser()
            changed = True

    if changed:
        write_config(config)
        env.ui.console.print(f"Config updated: {config.path}")
    else:
        env.ui.console.print("No changes made.")
