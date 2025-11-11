from __future__ import annotations

import argparse
from pathlib import Path

from ..migration import LegacyMigrationReport, perform_legacy_migration
from ..paths import STATE_HOME


def _summarize(report: LegacyMigrationReport) -> list[str]:
    lines = []
    if report.state_source:
        lines.append(f"State source: {report.state_source}")
    else:
        lines.append("State source: missing")
    if report.runs_source:
        lines.append(f"Runs source: {report.runs_source}")
    else:
        lines.append("Runs source: missing")
    lines.append(f"Conversations processed: {report.conversations_migrated}")
    if report.runs_skipped:
        lines.append("Runs skipped: existing entries present (rerun with --force to replace)")
    else:
        lines.append(f"Runs processed: {report.runs_migrated}")
    if report.dry_run:
        lines.append("Dry run: no database changes were made.")
    return lines


def run_migrate_cli(args: argparse.Namespace, env) -> None:
    ui = env.ui
    state_path = Path(args.state_path).expanduser() if args.state_path else (STATE_HOME / "state.json")
    runs_path = Path(args.runs_path).expanduser() if args.runs_path else (STATE_HOME / "runs.json")
    report = perform_legacy_migration(
        state_path=state_path,
        runs_path=runs_path,
        db_path=env.conversations.database.resolve_path(),
        dry_run=args.dry_run,
        force_runs=args.force,
    )
    ui.summary("Legacy Migration", _summarize(report))
    for error in report.errors:
        ui.console.print(f"[red]{error}")
