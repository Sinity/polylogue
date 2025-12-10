from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..commands import CommandEnv
from ..config import OutputDirs
from ..settings import (
    SETTINGS_PATH,
    clear_persisted_settings,
    ensure_settings_defaults,
    persist_settings,
)
from ..config import CONFIG, CONFIG_PATH, persist_config


def _settings_snapshot(settings) -> dict:
    return {
        "html": "on" if settings.html_previews else "off",
        "theme": settings.html_theme,
        "collapse_threshold": settings.collapse_threshold,
        "store": str(SETTINGS_PATH),
        "config": str(CONFIG_PATH or (SETTINGS_PATH.parent / "config.json")),
    }


def run_settings_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    settings = env.settings
    config_obj = env.config
    html_mode = getattr(args, "html", None)
    theme = getattr(args, "theme", None)
    collapse = getattr(args, "collapse_threshold", None)
    reset_flag = getattr(args, "reset", False)
    input_root = getattr(args, "input_root", None)
    output_root = getattr(args, "output_root", None)

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

    config_changed = bool(changed or input_root or output_root or reset_flag)
    if config_changed:
        effective_input = Path(input_root).expanduser() if input_root else config_obj.exports.chatgpt
        effective_output = Path(output_root).expanduser() if output_root else config_obj.defaults.output_dirs.render.parent
        if output_root:
            current_parents = {
                config_obj.defaults.output_dirs.render.parent,
                config_obj.defaults.output_dirs.sync_drive.parent,
                config_obj.defaults.output_dirs.sync_codex.parent,
                config_obj.defaults.output_dirs.sync_claude_code.parent,
                config_obj.defaults.output_dirs.import_chatgpt.parent,
                config_obj.defaults.output_dirs.import_claude.parent,
            }
            if len(current_parents) > 1:
                env.ui.console.print(
                    "[yellow]Detected mixed output roots; aligning all providers under the new root."
                )
            new_root = effective_output
            config_obj.defaults.output_dirs = OutputDirs(
                render=new_root / "render",
                sync_drive=new_root / "gemini",
                sync_codex=new_root / "codex",
                sync_claude_code=new_root / "claude-code",
                import_chatgpt=new_root / "chatgpt",
                import_claude=new_root / "claude",
            )
        persist_config(
            input_root=effective_input,
            output_root=effective_output,
            collapse_threshold=settings.collapse_threshold,
            html_previews=settings.html_previews,
            html_theme=settings.html_theme,
            index=config_obj.index,
            path=CONFIG_PATH,
            roots=config_obj.defaults.roots if getattr(config_obj.defaults, "roots", None) else None,
        )

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
    summary_lines.append(f"Config: {payload['config']}")
    if input_root:
        summary_lines.append(f"Input root: {Path(input_root).expanduser()}")
    if output_root:
        summary_lines.append(f"Output root: {Path(output_root).expanduser()}")

    env.ui.summary("Settings", summary_lines)


__all__ = ["run_settings_cli"]
