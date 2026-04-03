"""CLI helpers for resolving and prompting source selections."""

from __future__ import annotations

from polylogue.cli.helper_source_state import load_last_source, save_last_source
from polylogue.cli.helper_support import fail
from polylogue.cli.types import AppEnv
from polylogue.config import Config


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
    save_last_source(choice)
    if choice == "all":
        return None
    return [choice]


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


__all__ = ["maybe_prompt_sources", "resolve_sources"]
