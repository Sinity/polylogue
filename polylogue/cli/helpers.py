"""CLI helper functions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from polylogue.cli.formatting import format_sources_summary
from polylogue.cli.types import AppEnv
from polylogue.config import Config, ConfigError, load_config
from polylogue.health import cached_health_summary
from polylogue.pipeline.runner import latest_run


def fail(command: str, message: str) -> None:
    raise SystemExit(f"{command}: {message}")


def is_declarative() -> bool:
    value = os.environ.get("POLYLOGUE_DECLARATIVE")
    if not value:
        return False
    return value.lower() not in {"0", "false", "no"}


def source_state_path() -> Path:
    raw_state_root = os.environ.get("XDG_STATE_HOME")
    state_root = Path(raw_state_root).expanduser() if raw_state_root else Path.home() / ".local/state"
    return state_root / "polylogue" / "last-source.json"


def load_last_source() -> str | None:
    path = source_state_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("source"), str):
        return payload["source"]
    return None


def save_last_source(source_name: str) -> None:
    path = source_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source_name}), encoding="utf-8")


def maybe_prompt_sources(
    env: AppEnv,
    config: Config,
    selected_sources: list[str] | None,
    command: str,
) -> list[str] | None:
    if selected_sources is not None or env.ui.plain:
        return selected_sources
    names = [source.name for source in config.sources]
    if len(names) <= 1:
        return selected_sources
    options = ["all"] + names
    last_choice = load_last_source()
    if last_choice in options:
        options.remove(last_choice)
        options.insert(0, last_choice)
    choice = env.ui.choose(f"Select source for {command}", options)
    if not choice:
        return selected_sources
    save_last_source(choice)
    if choice == "all":
        return None
    return [choice]


def load_effective_config(env: AppEnv) -> Config:
    return load_config(env.config_path)


def resolve_sources(config: Config, sources: tuple[str, ...], command: str) -> list[str] | None:
    if not sources:
        return None
    requested = list(dict.fromkeys(sources))
    if "last" in requested:
        if len(requested) > 1:
            fail(command, "--source last cannot be combined with other sources")
        last = load_last_source()
        if not last:
            fail(command, "No previously selected source found for --source last")
        requested = [last]
    defined = {source.name for source in config.sources}
    missing = sorted(set(requested) - defined)
    if missing:
        known = ", ".join(sorted(defined)) or "none"
        fail(command, f"Unknown source(s): {', '.join(missing)}. Known sources: {known}")
    return requested


def print_summary(env: AppEnv) -> None:
    ui = env.ui
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        ui.console.print(f"[yellow]{exc}[/yellow]")
        ui.console.print("Run `polylogue config init` to create a config.")
        return
    last_run_data = latest_run()
    last_line = "Last run: none"
    if last_run_data:
        last_line = f"Last run: {last_run_data.get('run_id')} ({last_run_data.get('timestamp')})"
    health_line = f"Health: {cached_health_summary(config.archive_root)}"
    ui.summary(
        "Polylogue",
        [
            f"Config: {config.path}",
            f"Archive root: {config.archive_root}",
            f"Render root: {config.render_root}",
            f"Sources: {format_sources_summary(config.sources)}",
            last_line,
            health_line,
        ],
    )


def latest_render_path(render_root: Path) -> Path | None:
    if not render_root.exists():
        return None
    candidates = list(render_root.rglob("conversation.md")) + list(render_root.rglob("conversation.html"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
