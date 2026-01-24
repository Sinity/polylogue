"""CLI helper functions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn

from polylogue.cli.formatting import format_sources_summary
from polylogue.cli.types import AppEnv
from polylogue.config import Config, load_config
from polylogue.health import cached_health_summary, get_health
from polylogue.pipeline.runner import latest_run


def fail(command: str, message: str) -> NoReturn:
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
    if isinstance(payload, dict):
        source = payload.get("source")
        if isinstance(source, str):
            return source
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
    if last_choice and last_choice in options:
        options.remove(last_choice)
        options.insert(0, last_choice)
    choice = env.ui.choose(f"Select source for {command}", options)
    if not choice:
        fail(command, "No source selected.")
    # At this point choice is guaranteed to be str (not None) due to the above check
    save_last_source(choice)
    if choice == "all":
        return None
    return [choice]


def load_effective_config(env: AppEnv) -> Config:
    """Return the hardcoded configuration (zero-config)."""
    return load_config()


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
        requested = [last]  # last is guaranteed str here due to the if not last check above
    defined = {source.name for source in config.sources}
    missing = sorted(set(requested) - defined)
    if missing:
        known = ", ".join(sorted(defined)) or "none"
        fail(command, f"Unknown source(s): {', '.join(missing)}. Known sources: {known}")
    return requested


def print_summary(env: AppEnv, *, verbose: bool = False) -> None:
    ui = env.ui
    config = load_effective_config(env)
    last_run_data = latest_run()
    last_line = "Last run: none"
    if last_run_data:
        last_line = f"Last run: {last_run_data.run_id} ({last_run_data.timestamp})"

    lines = [
        f"Archive: {config.archive_root}",
        f"Render: {config.render_root}",
        f"Sources: {format_sources_summary(config.sources)}",
        last_line,
    ]

    if verbose:
        # Show detailed health checks
        payload = get_health(config)
        cached = payload.get("cached")
        age = payload.get("age_seconds")
        health_header = f"Health (cached={cached}, age={age}s)" if cached is not None else "Health"
        lines.append(health_header)
        checks = payload.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict):
                    name = check.get("name")
                    status = check.get("status")
                    detail = check.get("detail")
                    status_str = str(status) if status else "?"
                    icon = {"ok": "[green]✓[/green]", "warning": "[yellow]![/yellow]", "error": "[red]✗[/red]"}.get(status_str, "?")
                    if ui.plain:
                        icon = {"ok": "OK", "warning": "WARN", "error": "ERR"}.get(status_str, "?")
                    lines.append(f"  {icon} {name}: {detail}")
    else:
        lines.append(f"Health: {cached_health_summary(config.archive_root)}")

    ui.summary("Polylogue", lines)


def latest_render_path(render_root: Path) -> Path | None:
    if not render_root.exists():
        return None
    candidates = list(render_root.rglob("conversation.md")) + list(render_root.rglob("conversation.html"))
    if not candidates:
        return None
    # Handle race condition where files may be deleted between listing and stat
    latest: Path | None = None
    latest_mtime: float = 0.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = path
        except OSError:
            # File was deleted between listing and stat, skip it
            continue
    return latest
