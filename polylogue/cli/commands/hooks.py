"""Install and inspect Claude Code/Codex capture hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click

from polylogue.hooks import (
    HookChangePlan,
    HookHarness,
    HookHarnessStatus,
    HookSettingsError,
    hook_statuses,
    normalize_harness,
    plan_hook_change,
    resolve_events,
)


def _harness(value: str) -> HookHarness:
    try:
        return normalize_harness(value)
    except HookSettingsError as exc:
        raise click.BadParameter(str(exc), param_hint="--harness") from exc


def _resolved_events(harness: HookHarness, value: str) -> tuple[str, ...]:
    try:
        return resolve_events(harness, value)
    except HookSettingsError as exc:
        raise click.BadParameter(str(exc), param_hint="--events") from exc


def _emit_change(plan: HookChangePlan, *, json_output: bool) -> None:
    payload = cast(dict[str, object], plan.to_dict())
    if json_output:
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    action = str(payload["action"])
    harness = str(payload["harness"])
    changed = bool(payload["changed"])
    written = bool(payload["written"])
    if changed:
        mode = "updated" if written else "would update"
        click.echo(f"{harness}: {mode} {payload['settings_path']}")
        click.echo(str(payload["diff"]), nl=not str(payload["diff"]).endswith("\n"))
    else:
        click.echo(f"{harness}: no change ({action} is already satisfied)")


def _render_status(status: HookHarnessStatus, *, coverage: bool) -> list[str]:
    wired = ", ".join(status.wired_events) or "none"
    missing = ", ".join(status.missing_recommended_events) or "none"
    observed = ", ".join(status.observed_last_7d) or "none"
    lines = [
        f"{status.harness}: {status.flow_state}",
        f"  settings: {status.settings_path}",
        f"  feature enabled: {str(status.feature_enabled).lower()}",
        f"  polylogue-hook on PATH: {str(status.executable_available).lower()}",
        f"  wired: {wired}",
        f"  missing recommended: {missing}",
        f"  observed last 7d: {observed}",
    ]
    if status.coverage_checked:
        lines.append(
            "  sessions: "
            f"eligible={status.eligible_session_count} "
            f"with-events={status.sessions_with_hook_events} "
            f"without-events={status.sessions_without_hook_events}"
        )
    if coverage and status.coverage:
        lines.append("  event coverage:")
        lines.append("    event                  wired  observed  expected  role")
        for row in status.coverage:
            expected = "n/a" if row.expected_session_count is None else str(row.expected_session_count)
            lines.append(
                f"    {row.event:<22} "
                f"{('yes' if row.wired else 'no'):<6} "
                f"{row.observed_session_count:<8} "
                f"{expected:<8} "
                f"{row.enrichment}"
            )
    if status.evidence_note:
        lines.append(f"  evidence: {status.evidence_note}")
    return lines


@click.group("hooks")
def hooks_command() -> None:
    """Install and monitor Claude Code/Codex capture hooks."""


@hooks_command.command("install")
@click.option(
    "--harness",
    required=True,
    type=click.Choice(["claude-code", "codex"]),
    help="Harness settings to update.",
)
@click.option(
    "--events",
    default="recommended",
    show_default=True,
    help="recommended, all Polylogue-supported events, or a comma-separated list.",
)
@click.option("--dry-run", is_flag=True, help="Show the structured settings diff without writing it.")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON.")
def install_command(harness: str, events: str, dry_run: bool, json_output: bool) -> None:
    """Idempotently add Polylogue capture handlers to one harness."""

    normalized = _harness(harness)
    try:
        plan = plan_hook_change(
            "install",
            normalized,
            _resolved_events(normalized, events),
            dry_run=dry_run,
        )
    except HookSettingsError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_change(plan, json_output=json_output)


@hooks_command.command("uninstall")
@click.option(
    "--harness",
    required=True,
    type=click.Choice(["claude-code", "codex"]),
    help="Harness settings to update.",
)
@click.option(
    "--events",
    default="all",
    show_default=True,
    help="all Polylogue-supported events or a comma-separated list.",
)
@click.option("--dry-run", is_flag=True, help="Show the structured settings diff without writing it.")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON.")
def uninstall_command(harness: str, events: str, dry_run: bool, json_output: bool) -> None:
    """Remove only Polylogue-owned handlers from one harness."""

    normalized = _harness(harness)
    try:
        plan = plan_hook_change(
            "uninstall",
            normalized,
            _resolved_events(normalized, events),
            dry_run=dry_run,
        )
    except HookSettingsError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_change(plan, json_output=json_output)


@hooks_command.command("status")
@click.option(
    "--harness",
    type=click.Choice(["claude-code", "codex"]),
    default=None,
    help="Limit status to one harness.",
)
@click.option("--coverage", is_flag=True, help="Show the per-event trailing-seven-day evidence table.")
@click.option(
    "--archive-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    hidden=True,
)
@click.option("--json", "json_output", is_flag=True, help="Emit JSON.")
def status_command(
    harness: str | None,
    coverage: bool,
    archive_root: Path | None,
    json_output: bool,
) -> None:
    """Show wired, recommended, observed, and hook-flow liveness state."""

    normalized = _harness(harness) if harness else None
    try:
        statuses = hook_statuses(harness=normalized, coverage=True, archive_root_path=archive_root)
    except HookSettingsError as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps({"harnesses": [status.to_dict() for status in statuses]}, indent=2, default=str))
        return
    lines: list[str] = []
    for index, status in enumerate(statuses):
        if index:
            lines.append("")
        lines.extend(_render_status(status, coverage=coverage))
    click.echo("\n".join(lines))


__all__ = ["hooks_command"]
