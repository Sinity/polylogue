"""``maintenance verify-archive``: read-only archive-wide coherence gate.

Turns the manual post-rebuild / post-restore verification checklist into a
repeatable command: tier presence + schema versions, active-pointer
coherence (polylogue-k8kj class), source-vs-index materialization coverage,
FTS parity, lineage sanity, planner-stats presence (polylogue-l3tk class),
and an archive-wide counts summary. See
:mod:`polylogue.maintenance.archive_verification` for the check registry.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.core.outcomes import OutcomeCheck
    from polylogue.maintenance.archive_verification import ArchiveVerificationReport

_STATUS_LABELS = {"ok": "OK", "warning": "WARN", "error": "FAIL", "skip": "SKIP"}
_STATUS_ICONS = {"ok": "✓", "warning": "⚠", "error": "✗", "skip": "◌"}


@click.command("verify-archive")
@click.option(
    "--check",
    "selected_checks",
    multiple=True,
    help="Run only the named check(s) (repeatable). Omit to run every registered check.",
)
@click.option(
    "--sample-limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of representative evidence samples (worst sessions, offending ids, ...) per check.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Also fail (non-zero exit) when any check reports a warning, not only on errors.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def verify_archive_command(
    selected_checks: tuple[str, ...],
    sample_limit: int,
    strict: bool,
    output_format: str,
) -> None:
    """Prove the archive is coherent after a rebuild, restore, or promotion.

    Read-only: every check opens its tier database(s) ``mode=ro`` and never
    mutates the archive. A single check's failure -- including a tier being
    temporarily busy under a concurrent rebuild -- never aborts the others;
    each independently reports ok/warning/error/skip plus evidence numbers.

    Exit code is non-zero when any check reports an error (or, with
    ``--strict``, a warning).
    """
    from polylogue.maintenance.archive_verification import verify_archive

    try:
        report = verify_archive(
            archive_root(),
            checks=selected_checks or None,
            sample_limit=sample_limit,
        )
    except ValueError as exc:
        raise click.BadParameter(
            f"{exc}",
            param_hint="--check",
        ) from exc

    if output_format == "json":
        click.echo(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        _render_plain(report)

    if report.blocking or (strict and report.warning_count > 0):
        raise SystemExit(1)


def _render_plain(report: ArchiveVerificationReport) -> None:
    counts = report.summary_counts(include_skip=True)
    click.echo(f"Archive verification: {report.archive_root}")
    click.echo(
        f"  {counts['ok']} ok, {counts['warning']} warning, {counts['error']} error, {counts.get('skip', 0)} skipped"
    )
    click.echo("")
    for check in report.checks:
        click.echo(f"  {_format_check_line(check)}")
        for detail in check.details[:5]:
            click.echo(f"      {detail}")
    click.echo("")
    click.echo("BLOCKING" if report.blocking else "clear")


def _format_check_line(check: OutcomeCheck) -> str:
    status = check.status.value
    icon = _STATUS_ICONS.get(status, "")
    label = _STATUS_LABELS.get(status, status.upper())
    prefix = f"{icon} " if icon else ""
    return f"{prefix}[{label}] {check.name}: {check.summary}"
