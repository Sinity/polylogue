from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from ..paths import STATE_HOME
from ..commands import CommandEnv

PREFS_PATH = STATE_HOME / "prefs.json"


def _load_prefs() -> Dict[str, Dict[str, str]]:
    if not PREFS_PATH.exists():
        return {}
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_prefs(prefs: Dict[str, Dict[str, str]]) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def run_prefs_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    prefs = _load_prefs()
    sub = getattr(args, "prefs_cmd", None)

    if sub == "list":
        if getattr(args, "json", False):
            print(json.dumps(prefs, indent=2))
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
            print(json.dumps({"command": command, "flag": flag, "value": value}))
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
            print(json.dumps({"cleared": target or "all"}))
            return
        ui.console.print(f"[green]Cleared preferences for {target or 'all commands'}")
        return

    ui.console.print("[red]Unknown prefs subcommand")
    raise SystemExit(1)


__all__ = ["run_prefs_cli", "PREFS_PATH"]
