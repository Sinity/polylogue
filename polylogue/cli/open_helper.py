from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ..commands import CommandEnv
from ..util import latest_run
from ..schema import stamp_payload
from .editor import open_in_editor, get_editor, open_in_browser


def run_open_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    provider = getattr(args, "provider", None)
    cmd = getattr(args, "command", None)
    entry = latest_run(provider=provider, cmd=cmd)
    fallback = getattr(args, "fallback", None)
    if not entry and fallback:
        path = Path(fallback)
        if getattr(args, "json", False):
            print(json.dumps(stamp_payload({"path": str(path), "source": "fallback"}), sort_keys=True))
            return
        ui.console.print(str(path))
        return
    if not entry:
        ui.console.print("[yellow]No recent runs found.")
        return

    path_value = entry.get("out") or entry.get("markdown_path") or entry.get("path")
    payload = {
        "provider": entry.get("provider"),
        "cmd": entry.get("cmd"),
        "timestamp": entry.get("timestamp"),
        "path": str(path_value) if path_value else None,
    }
    if getattr(args, "json", False):
        print(json.dumps(stamp_payload(payload), indent=2, sort_keys=True))
        return
    if not path_value:
        ui.console.print("[yellow]No path recorded for last run.")
        return
    target = Path(path_value)
    if getattr(args, "print_only", False):
        ui.console.print(str(target))
        return
    if target.suffix.lower() == ".html":
        if open_in_browser(target):
            ui.console.print(f"[dim]Opened {target} in browser[/dim]")
            return
    if open_in_editor(target):
        ui.console.print(f"[dim]Opened {target} in editor[/dim]")
        return
    editor = get_editor()
    if editor:
        ui.console.print(f"[yellow]Could not open with {editor}; path: {target}")
    else:
        ui.console.print(f"[yellow]Set $EDITOR to open files automatically. Path: {target}")


__all__ = ["run_open_cli"]
