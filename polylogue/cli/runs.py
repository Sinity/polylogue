from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import List, Optional

from rich.table import Table

from ..commands import CommandEnv
from ..schema import stamp_payload
from ..util import load_runs, parse_input_time_to_epoch


def run_runs_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    limit = max(1, getattr(args, "limit", 50))
    provider_filter = _normalize_filter(getattr(args, "providers", None))
    cmd_filter = _normalize_filter(getattr(args, "commands", None))
    since_epoch = parse_input_time_to_epoch(getattr(args, "since", None))
    until_epoch = parse_input_time_to_epoch(getattr(args, "until", None))
    runs = load_runs(limit=limit)
    if provider_filter:
        runs = [run for run in runs if (run.get("provider") or "").lower() in provider_filter]
    if cmd_filter:
        runs = [run for run in runs if (run.get("cmd") or "").lower() in cmd_filter]
    if since_epoch is not None or until_epoch is not None:
        runs = [run for run in runs if _timestamp_in_range(run.get("timestamp"), since_epoch, until_epoch)]

    if getattr(args, "json", False):
        payload = stamp_payload({"runs": runs}) if not getattr(args, "json_verbose", False) else stamp_payload({"runs": runs, "count": len(runs)})
        print(json.dumps(payload, indent=2))
        return

    if env.ui.plain:
        _print_plain(env, runs)
        return

    table = Table(title=f"Recent Runs (n={len(runs)})")
    table.add_column("Timestamp")
    table.add_column("Command")
    table.add_column("Provider")
    table.add_column("Count", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Retries", justify="right")
    table.add_column("Failures", justify="right")
    for entry in runs:
        table.add_row(
            entry.get("timestamp", "-"),
            entry.get("cmd", "-"),
            entry.get("provider", "-"),
            str(entry.get("count", 0)),
            f"{entry.get('duration', 0) or 0:.1f}s",
            str(entry.get("driveRetries", entry.get("retries", 0) or 0)),
            str(entry.get("driveFailures", entry.get("failures", 0) or 0)),
        )
    env.ui.console.print(table)


def _normalize_filter(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    return values or None


def _print_plain(env: CommandEnv, runs: List[dict]) -> None:
    console = env.ui.console
    if not runs:
        console.print("No runs recorded.")
        return
    for entry in runs:
        console.print(
            f"{entry.get('timestamp','-')} :: {entry.get('cmd','-')} provider={entry.get('provider','-')} count={entry.get('count',0)} duration={entry.get('duration',0)}"
        )


def _timestamp_in_range(timestamp: Optional[str], since: Optional[float], until: Optional[float]) -> bool:
    if timestamp is None:
        return True
    try:
        if timestamp.endswith("Z"):
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(timestamp)
        epoch = dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return True
    if since is not None and epoch < since:
        return False
    if until is not None and epoch > until:
        return False
    return True
