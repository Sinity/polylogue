"""Scan-secrets command: production wiring for the candidate-only secret
detector (polylogue-27m fix round).

``polylogue/security/secret_scan.py`` defines the regex/entropy rules and
the non-injectable ``SECRET_CANDIDATE`` assertion write path, but had zero
production callers -- an operator running the archive would never actually
get a candidate finding out of it, only a test invoking the functions
directly would. This command is that caller: it reads a session's captured
block text/tool-input from ``index.db``, scans it, and records any
candidates in ``user.db`` for ``polylogue ops excise`` triage.
"""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root


@click.command("scan-secrets")
@click.option("--session", "session_id", required=True, help="Session id to scan.")
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
def scan_secrets_command(env: AppEnv, session_id: str, output_format: str | None) -> None:
    """Scan a session's captured content for credential-shaped spans.

    Records any findings as non-injectable ``SECRET_CANDIDATE`` assertions
    (a SHA-256 fingerprint, byte length, pattern id, and span offsets only
    -- the matched literal is never stored or printed). Review candidates
    via the assertion surfaces, then ``polylogue ops excise`` to remove
    confirmed secrets from the archive.
    """
    from polylogue.security.secret_scan import scan_session_for_secret_candidates

    root = archive_root()
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


__all__ = ["scan_secrets_command"]
