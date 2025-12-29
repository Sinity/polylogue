from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from .. import paths as paths_module
from ..commands import CommandEnv


def prefs_path() -> Path:
    return paths_module.STATE_HOME / "prefs.json"


def load_prefs() -> Dict[str, Dict[str, str]]:
    path = prefs_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_prefs(prefs: Dict[str, Dict[str, str]]) -> None:
    path = prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, indent=2, sort_keys=True), encoding="utf-8")


def run_prefs_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    prefs = load_prefs()
    sub = getattr(args, "prefs_cmd", None)

    if sub == "list":
        if getattr(args, "json", False):
            print(json.dumps(prefs, indent=2, sort_keys=True))
            return
        lines = []
        for cmd, values in sorted(prefs.items()):
            lines.append(f"{cmd}:")
            for flag, val in values.items():
                lines.append(f"  {flag} = {val}")
        if not lines:
            lines.append("No preferences set.")
        ui.summary("Preferences", lines)
        return

    if sub == "set":
        command = args.command
        flag = args.flag
        value = args.value
        cmd_prefs = prefs.setdefault(command, {})
        cmd_prefs[flag] = value
        _save_prefs(prefs)
        if getattr(args, "json", False):
            print(json.dumps({"command": command, "flag": flag, "value": value}, sort_keys=True))
            return
        ui.console.print(f"[green]Saved preference: {command} {flag}={value}")
        return

    if sub == "clear":
        target = getattr(args, "command", None)
        if target:
            prefs.pop(target, None)
        else:
            prefs = {}
        _save_prefs(prefs)
        if getattr(args, "json", False):
            print(json.dumps({"cleared": target or "all"}, sort_keys=True))
            return
        ui.console.print(f"[green]Cleared preferences for {target or 'all commands'}")
        return

    ui.console.print("[red]Unknown prefs subcommand")
    raise SystemExit(1)


__all__ = ["run_prefs_cli", "prefs_path", "load_prefs"]
