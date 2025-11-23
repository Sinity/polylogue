from __future__ import annotations

import argparse
import json

from ..commands import CommandEnv
from ..settings import (
    SETTINGS_PATH,
    clear_persisted_settings,
    ensure_settings_defaults,
    persist_settings,
)


def _settings_snapshot(settings) -> dict:
    return {
        "html": "on" if settings.html_previews else "off",
        "theme": settings.html_theme,
        "collapse_threshold": settings.collapse_threshold,
        "store": str(SETTINGS_PATH),
    }


def run_settings_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    settings = env.settings
    html_mode = getattr(args, "html", None)
    theme = getattr(args, "theme", None)
    collapse = getattr(args, "collapse_threshold", None)
    reset_flag = getattr(args, "reset", False)

    if reset_flag:
        clear_persisted_settings()
        ensure_settings_defaults(settings)
    else:
        ensure_settings_defaults(settings)

    changed = False
    if html_mode is not None:
        settings.html_previews = html_mode == "on"
        changed = True
    if theme is not None:
        settings.html_theme = theme
        changed = True
    if collapse is not None:
        settings.collapse_threshold = collapse
        changed = True

    if changed:
        persist_settings(settings)

    payload = _settings_snapshot(settings)
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return

    summary_lines = [
        f"HTML previews: {payload['html']}",
        f"HTML theme: {payload['theme']}",
    ]
    if payload["collapse_threshold"] is not None:
        summary_lines.append(f"Collapse threshold: {payload['collapse_threshold']}")
    summary_lines.append(f"Store: {payload['store']}")

    env.ui.summary("Settings", summary_lines)


__all__ = ["run_settings_cli"]
