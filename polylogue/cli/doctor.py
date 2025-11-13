from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from rich.table import Table

from ..commands import CommandEnv
from ..config import CONFIG_ENV, CONFIG_PATH, DEFAULT_PATHS
from ..doctor import run_doctor as doctor_run
from ..util import CLAUDE_CODE_PROJECT_ROOT, CODEX_SESSIONS_ROOT


def run_doctor_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    console = cast(Any, ui.console)
    codex_dir = Path(args.codex_dir).expanduser() if args.codex_dir else CODEX_SESSIONS_ROOT
    claude_dir = Path(args.claude_code_dir).expanduser() if args.claude_code_dir else CLAUDE_CODE_PROJECT_ROOT
    report = doctor_run(
        codex_dir=codex_dir,
        claude_code_dir=claude_dir,
        limit=args.limit,
        service=env.conversations,
        archive=env.archive,
    )

    sample_config = Path(__file__).resolve().parent.parent / "docs" / "polylogue.config.sample.jsonc"
    config_hint = {
        "cmd": "doctor",
        "checked": {k: int(v) for k, v in report.checked.items()},
        "issues": [
            {
                "provider": issue.provider,
                "path": str(issue.path),
                "message": issue.message,
                "severity": issue.severity,
                "hint": issue.hint,
            }
            for issue in report.issues
        ],
        "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
        "configEnv": CONFIG_ENV,
        "configCandidates": [str(p) for p in DEFAULT_PATHS],
        "configSample": str(sample_config),
    }

    if getattr(args, "json", False):
        print(json.dumps(config_hint, indent=2))
        return

    lines = [
        f"Codex sessions checked: {report.checked.get('codex', 0)}",
        f"Claude Code sessions checked: {report.checked.get('claude-code', 0)}",
    ]
    if CONFIG_PATH is None:
        candidates = ", ".join(str(p) for p in DEFAULT_PATHS)
        lines.append(
            f"No Polylogue config detected. Copy {sample_config} to one of [{candidates}] or set ${CONFIG_ENV}."
        )
    if not report.issues:
        lines.append("No issues detected.")
        ui.summary("Doctor", lines)
        return

    if not ui.plain:
        table = Table(title="Doctor Issues", show_lines=False)
        table.add_column("Provider")
        table.add_column("Severity")
        table.add_column("Path")
        table.add_column("Message")
        table.add_column("Hint")
        severity_styles = {"error": "bold red", "warning": "yellow", "info": "cyan"}
        for issue in report.issues:
            style = severity_styles.get(issue.severity.lower(), "")
            table.add_row(
                issue.provider,
                issue.severity.upper(),
                str(issue.path),
                issue.message,
                issue.hint or "",
                style=style,
            )
        console.print(table)
    lines.append(f"Found {len(report.issues)} issue(s):")
    for issue in report.issues:
        hint_text = f" Hint: {issue.hint}" if issue.hint else ""
        lines.append(f"- [{issue.severity}] {issue.provider}: {issue.path} — {issue.message}{hint_text}")
    ui.summary("Doctor", lines)


__all__ = ["run_doctor_cli"]
