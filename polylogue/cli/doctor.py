from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from rich.table import Table
from rich.text import Text

from ..commands import CommandEnv
from ..config import CONFIG_ENV, CONFIG_PATH, DEFAULT_PATHS, is_config_declarative
from ..doctor import run_doctor as doctor_run
from ..schema import stamp_payload
from ..util import CLAUDE_CODE_PROJECT_ROOT, CODEX_SESSIONS_ROOT


def run_doctor_cli(args: SimpleNamespace, env: CommandEnv) -> None:
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
    declarative, decl_reason, decl_target = is_config_declarative(CONFIG_PATH)

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
        "configDeclarative": declarative,
        "configDeclarativeReason": decl_reason or None,
        "configDeclarativeTarget": str(decl_target) if decl_target else None,
        "configEnv": CONFIG_ENV,
        "configCandidates": [str(p) for p in DEFAULT_PATHS],
        "configSample": str(sample_config),
        "credentialPath": str(report.credential_path),
        "tokenPath": str(report.token_path),
        "credentialEnv": report.credential_env,
        "tokenEnv": report.token_env,
        "credentialsPresent": report.credentials_present,
        "tokenPresent": report.token_present,
    }

    if getattr(args, "json", False):
        payload = stamp_payload(config_hint)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    lines = [
        f"Codex sessions checked: {report.checked.get('codex', 0)}",
        f"Claude Code sessions checked: {report.checked.get('claude-code', 0)}",
        f"Credentials: {'present' if report.credentials_present else 'missing'} ({report.credential_path})",
        f"Token: {'present' if report.token_present else 'missing'} ({report.token_path})",
    ]
    if declarative:
        reason = f" ({decl_reason})" if decl_reason else ""
        lines.append(f"Config is declarative{reason}; edit your Nix/flake module for changes.")
    if report.credential_env or report.token_env:
        lines.append(f"Env overrides: cred={report.credential_env or '-'} token={report.token_env or '-'}")
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
                Text(str(issue.path)),
                Text(issue.message),
                Text(issue.hint or ""),
                style=style,
            )
        console.print(table)
    lines.append(f"Found {len(report.issues)} issue(s):")
    for issue in report.issues:
        hint_text = f" Hint: {issue.hint}" if issue.hint else ""
        lines.append(f"- [{issue.severity}] {issue.provider}: {issue.path} — {issue.message}{hint_text}")
    ui.summary("Doctor", lines)


__all__ = ["run_doctor_cli"]
