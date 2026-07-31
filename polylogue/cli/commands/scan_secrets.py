"""Scan-secrets command: production wiring for the candidate-only secret
detector (polylogue-27m fix round, bulk mode polylogue-layg.1).

``polylogue/security/secret_scan.py`` defines the regex/entropy rules and
the non-injectable ``SECRET_CANDIDATE`` assertion write path. This command
is the CLI caller: ``--session <id>`` scans one known session's captured
block text/tool-input from ``index.db``; ``--all`` sweeps every session the
archive hasn't covered yet at the current scanner version, so an operator
with no prior signal (no session id in hand) can still discover candidates
archive-wide. Either way, findings land in ``user.db`` for
``polylogue ops excise`` triage.
"""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root


@click.command("scan-secrets")
@click.option("--session", "session_id", default=None, help="Session id to scan.")
@click.option(
    "--all",
    "scan_all",
    is_flag=True,
    default=False,
    help="Scan every session not yet covered by the current scanner version, instead of one --session.",
)
@click.option(
    "--limit",
    "page_limit",
    type=int,
    default=None,
    help="With --all, bound this invocation to one page of at most N pending sessions "
    "(default: drain every pending session across as many pages as needed).",
)
@click.option("--origin", default=None, help="With --all, restrict the sweep to one origin token.")
@click.option(
    "--status",
    "status_only",
    is_flag=True,
    default=False,
    help="Report scan coverage (pending session count) without scanning.",
)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format.",
)
@click.pass_obj
def scan_secrets_command(
    env: AppEnv,
    session_id: str | None,
    scan_all: bool,
    page_limit: int | None,
    origin: str | None,
    status_only: bool,
    output_format: str | None,
) -> None:
    """Scan captured content for credential-shaped spans.

    Records any findings as non-injectable ``SECRET_CANDIDATE`` assertions
    (a SHA-256 fingerprint, byte length, pattern id, and span offsets only
    -- the matched literal is never stored or printed). Review candidates
    via the assertion surfaces, then ``polylogue ops excise`` to remove
    confirmed secrets from the archive.

    Exactly one of ``--session <id>``, ``--all``, or ``--status`` is
    required: ``--session`` scans one known session; ``--all`` sweeps every
    session not yet covered at the current scanner version (bounded pages,
    resumable -- interrupting and re-running covers the same ground exactly
    once); ``--status`` reports how many sessions are still pending without
    scanning anything.
    """
    root = archive_root()

    if status_only:
        _report_status(env, root, origin=origin, output_format=output_format)
        return

    if scan_all:
        if session_id is not None:
            raise click.UsageError("--session and --all are mutually exclusive.")
        _scan_all(env, root, page_limit=page_limit, origin=origin, output_format=output_format)
        return

    if session_id is None:
        raise click.UsageError("Provide --session <id>, --all, or --status.")

    from polylogue.security.secret_scan import scan_session_for_secret_candidates

    result = scan_session_for_secret_candidates(root, session_id)

    if output_format == "json":
        import json as json_module

        click.echo(json_module.dumps({"status": "ok" if result.found else "not_found", **result.as_dict()}))
        return

    if not result.found:
        env.ui.console.print(f"No session found for {session_id!r}.")
        return

    env.ui.summary(
        f"Scanned session {session_id}",
        [
            f"  blocks scanned: {result.blocks_scanned}",
            f"  secret candidates found: {result.candidates_found}",
            *(
                ["  review candidates, then `polylogue ops excise` to remove confirmed secrets."]
                if result.candidates_found
                else []
            ),
        ],
    )


def _scan_all(
    env: AppEnv,
    root: object,
    *,
    page_limit: int | None,
    origin: str | None,
    output_format: str | None,
) -> None:
    from pathlib import Path

    from polylogue.security.secret_scan import DEFAULT_SECRET_SCAN_PAGE_SIZE, scan_archive_for_secret_candidates

    assert isinstance(root, Path)

    single_page = page_limit is not None
    page_size = page_limit if page_limit is not None else DEFAULT_SECRET_SCAN_PAGE_SIZE

    sessions_scanned = 0
    blocks_scanned = 0
    candidates_found = 0
    errors = 0
    pages = 0
    remaining_pending = 0
    while True:
        result = scan_archive_for_secret_candidates(root, max_sessions=page_size, origin=origin)
        pages += 1
        sessions_scanned += result.sessions_scanned
        blocks_scanned += result.blocks_scanned
        candidates_found += result.candidates_found
        errors += result.errors
        remaining_pending = result.remaining_pending
        if single_page or not result.more_pending:
            break
        if result.sessions_scanned == 0:
            # No forward progress this page (e.g. every candidate errored);
            # stop instead of looping forever against the same backlog.
            break

    if output_format == "json":
        import json as json_module

        click.echo(
            json_module.dumps(
                {
                    "status": "ok",
                    "pages": pages,
                    "sessions_scanned": sessions_scanned,
                    "blocks_scanned": blocks_scanned,
                    "candidates_found": candidates_found,
                    "errors": errors,
                    "remaining_pending": remaining_pending,
                    "more_pending": remaining_pending > 0,
                }
            )
        )
        return

    env.ui.summary(
        "Archive-wide secret scan" + (f" (origin={origin})" if origin else ""),
        [
            f"  pages: {pages}",
            f"  sessions scanned: {sessions_scanned}",
            f"  blocks scanned: {blocks_scanned}",
            f"  secret candidates found: {candidates_found}",
            *([f"  errors: {errors}"] if errors else []),
            f"  remaining pending: {remaining_pending}",
            *(
                ["  review candidates, then `polylogue ops excise` to remove confirmed secrets."]
                if candidates_found
                else []
            ),
        ],
    )


def _report_status(
    env: AppEnv,
    root: object,
    *,
    origin: str | None,
    output_format: str | None,
) -> None:
    from pathlib import Path

    from polylogue.security.secret_scan import SECRET_SCAN_VERSION, count_pending_secret_scan_sessions

    assert isinstance(root, Path)
    index_db = root / "index.db"
    ops_db = root / "ops.db"
    remaining = count_pending_secret_scan_sessions(index_db, ops_db, origin=origin)

    if output_format == "json":
        import json as json_module

        click.echo(
            json_module.dumps(
                {
                    "status": "ok",
                    "scanner_version": SECRET_SCAN_VERSION,
                    "remaining_pending": remaining,
                }
            )
        )
        return

    env.ui.summary(
        "Secret scan coverage",
        [
            f"  scanner version: {SECRET_SCAN_VERSION}",
            f"  sessions pending: {remaining}",
        ],
    )


__all__ = ["scan_secrets_command"]
